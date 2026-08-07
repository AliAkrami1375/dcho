#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pyarrow>=15.0",
#   "numpy>=1.26",
#   "soundfile>=0.12",
#   "huggingface_hub>=0.28",
# ]
# ///
"""Pack a training tier into a compact, self-contained dataset.

Training on Hugging Face Jobs mounts the corpus repository directly and
copies nothing. Anywhere else - Colab, a rented box, a laptop - there is no
such mount, and pulling 47 GB at the start of every session is not
workable.

This produces a much smaller artefact holding only what training actually
reads: the clips of one tier, their transcripts, their speaker cluster and
their speaker embedding. Audio is re-encoded to FLAC, which is lossless and
roughly halves the size; decoding costs a few percent of a dataloader
worker and nothing on the GPU.

    tier_a_train   65.3 h   ~7.5 GB as raw PCM   ~4 GB as FLAC
    tier_b_train  180.8 h  ~20.8 GB as raw PCM  ~11 GB as FLAC

Embedding the speaker vector in the same row matters more than it looks: a
training session then needs exactly one artefact, with no second download
to align by index and no way for the two to drift apart.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf


def resolve(path: str, repo_hint: str | None = None) -> str:
    """Accept a local path or `repo_id:path/in/repo` and return a local path."""
    if Path(path).exists():
        return path
    if ":" in path:
        from huggingface_hub import hf_hub_download

        repo, inner = path.split(":", 1)
        return hf_hub_download(repo, inner, repo_type="dataset",
                               token=os.environ.get("HF_TOKEN"))
    raise SystemExit(f"cannot resolve {path!r}")


def load_selection(manifest_path: Path) -> dict[int, dict]:
    keep: dict[int, dict] = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                keep[int(row["idx"])] = row
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/corpus")
    ap.add_argument("--manifest", required=True, help="tier jsonl from build_manifest.py")
    ap.add_argument("--embeddings", default=None, help="speakers/embeddings.npy, aligned by idx")
    ap.add_argument("--out-dir", default="/tmp/dcho-pack")
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--name", default="tier_a")
    ap.add_argument("--shard-clips", type=int, default=4000)
    ap.add_argument("--codec", default="FLAC", choices=["FLAC", "WAV"])
    ap.add_argument("--max-minutes", type=float, default=0.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    keep = load_selection(Path(resolve(args.manifest)))
    print(f"[pack] {len(keep)} clips selected", flush=True)

    embeddings = None
    if args.embeddings:
        embeddings = np.load(resolve(args.embeddings), mmap_mode="r")
        print(f"[pack] embeddings {embeddings.shape}", flush=True)

    files = sorted(Path(args.data_dir).rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {args.data_dir}")

    started = time.time()
    rows: list[dict] = []
    shard_index = 0
    n_written = n_seen = 0
    seconds_written = 0.0
    bytes_written = 0

    def flush_shard():
        nonlocal rows, shard_index, bytes_written
        if not rows:
            return
        path = out_dir / f"{args.name}-{shard_index:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        bytes_written += path.stat().st_size
        print(f"[pack] shard {shard_index}: {len(rows)} clips, "
              f"{path.stat().st_size/1e6:.0f} MB", flush=True)
        rows = []
        shard_index += 1

    stop = False
    for path in files:
        if stop:
            break
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=64, columns=["sentence", "audio"]):
            d = batch.to_pydict()
            for text, blob in zip(d["sentence"], d["audio"]):
                idx = n_seen
                n_seen += 1
                meta = keep.get(idx)
                if meta is None:
                    continue
                if isinstance(blob, dict):
                    blob = blob.get("bytes")
                if not blob:
                    continue
                try:
                    audio, sr = sf.read(io.BytesIO(blob), dtype="int16", always_2d=True)
                except Exception:
                    continue
                if sr != 16000:
                    continue
                mono = audio[:, 0] if audio.shape[1] == 1 else audio.mean(axis=1).astype(np.int16)

                buf = io.BytesIO()
                sf.write(buf, mono, sr, format=args.codec, subtype="PCM_16")

                row = {
                    "idx": idx,
                    "text": text,
                    "audio": buf.getvalue(),
                    "sample_rate": sr,
                    "duration": float(len(mono)) / sr,
                    "speaker": int(meta["speaker"]),
                    "mos_ovr": float(meta.get("mos_ovr", 0.0)),
                }
                if embeddings is not None:
                    row["speaker_vector"] = np.asarray(embeddings[idx], dtype=np.float16).tobytes()
                rows.append(row)
                n_written += 1
                seconds_written += row["duration"]

                if len(rows) >= args.shard_clips:
                    flush_shard()
                if n_written % 5000 == 0:
                    print(f"[pack] {n_written}/{len(keep)}  {seconds_written/3600:.1f} h  "
                          f"{time.time()-started:.0f}s", flush=True)
                if args.max_minutes and (time.time() - started) / 60 > args.max_minutes:
                    print("[pack] time budget reached", flush=True)
                    stop = True
                    break
            if stop:
                break
    flush_shard()

    summary = {
        "name": args.name,
        "codec": args.codec,
        "clips_requested": len(keep),
        "clips_written": n_written,
        "hours": round(seconds_written / 3600, 2),
        "shards": shard_index,
        "size_mb": round(bytes_written / 1e6, 1),
        "has_speaker_vectors": embeddings is not None,
        "speaker_vector_dim": int(embeddings.shape[1]) if embeddings is not None else 0,
        "elapsed_minutes": round((time.time() - started) / 60, 2),
    }
    (out_dir / "pack_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if n_written < len(keep):
        print(f"[pack] WARNING: {len(keep)-n_written} selected clips were not written", flush=True)

    if args.repo_id:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(folder_path=str(out_dir), repo_id=args.repo_id, repo_type="dataset",
                          path_in_repo=args.name,
                          commit_message=f"pack {args.name}: {n_written} clips, {summary['hours']} h")
        print(f"[pack] pushed to {args.repo_id}/{args.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
