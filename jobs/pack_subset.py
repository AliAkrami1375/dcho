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
"""Pack a training split into a compact, self-contained dataset.

Training on Hugging Face Jobs can mount the 47 GB corpus repository
directly and read it in place. Colab cannot: there is no volume mount, so
the data has to arrive over the network, and re-streaming 47 GB every
session is not viable.

This job is what makes Colab practical. It reads the corpus once, keeps
only the clips a given manifest selects, re-encodes them as FLAC, and
attaches everything training needs - transcript and speaker vector - to
each row. The result is one self-contained dataset a notebook can pull in
a few minutes.

FLAC rather than raw PCM: lossless, so nothing is given up in the training
target, and it roughly halves the download. Decoding costs a few percent of
a dataloader worker, which is free next to the network time it saves.

Expected sizes, from the measured manifests:

    tier_a_train    65.3 h    ~7.5 GB as PCM    ~4 GB as FLAC
    tier_b_train   180.8 h   ~20.8 GB as PCM   ~11 GB as FLAC
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

TARGET_SR = 16000


def resolve(spec: str, default_name: str | None = None) -> str:
    """Accept a local path or a 'repo_id:path/in/repo' reference."""
    if Path(spec).exists():
        return spec
    repo, sep, name = spec.partition(":")
    if not sep:
        name = default_name or spec
        repo = spec
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, name, repo_type="dataset",
                           token=os.environ.get("HF_TOKEN"))


def load_manifest(spec: str) -> dict[int, dict]:
    """Manifest rows keyed by global corpus index."""
    rows = {}
    with open(resolve(spec), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[int(r["idx"])] = r
    return rows


def to_flac(blob: bytes) -> tuple[bytes, float] | None:
    """Re-encode to FLAC. Returns (bytes, seconds), or None if unreadable."""
    try:
        audio, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    except Exception:
        return None
    if sr != TARGET_SR:
        return None
    mono = audio.mean(axis=1)
    out = io.BytesIO()
    sf.write(out, mono, sr, format="FLAC", subtype="PCM_16")
    return out.getvalue(), len(mono) / sr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/corpus")
    ap.add_argument("--manifest", required=True,
                    help="local jsonl, or 'repo_id:path/in/repo.jsonl'")
    ap.add_argument("--embeddings", default="DibaAi/dcho-manifest:speakers/embeddings.npy")
    ap.add_argument("--out-dir", default="/tmp/dcho-pack")
    ap.add_argument("--repo-id", required=True, help="dataset repo to publish to")
    ap.add_argument("--rows-per-shard", type=int, default=2000)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    keep = load_manifest(args.manifest)
    print(f"[pack] manifest selects {len(keep)} clips", flush=True)

    embeddings = np.load(resolve(args.embeddings))
    print(f"[pack] speaker vectors {embeddings.shape}", flush=True)

    files = sorted(Path(args.data_dir).rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {args.data_dir}")
    print(f"[pack] {len(files)} source shards", flush=True)

    buffer: list[dict] = []
    shard_no = n_written = n_bad = 0
    seconds = 0.0
    written_bytes = 0

    def flush_shard():
        nonlocal buffer, shard_no, written_bytes
        if not buffer:
            return
        path = out_dir / f"train-{shard_no:04d}.parquet"
        # The audio is already FLAC, so a parquet codec has nothing left to
        # find; compressing again would only cost time.
        pq.write_table(pa.Table.from_pylist(buffer), path, compression="none")
        written_bytes += path.stat().st_size
        print(f"[pack] {path.name}  {len(buffer)} rows  "
              f"{path.stat().st_size/1e6:.0f} MB  running total {written_bytes/1e9:.2f} GB",
              flush=True)
        shard_no += 1
        buffer = []

    # Shards are walked in sorted order and rows counted as we go, which
    # reproduces exactly the global indices the audit and speaker jobs used.
    global_idx = 0
    stop = False

    for path in files:
        if stop:
            break
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=64, columns=["sentence", "audio"]):
            d = batch.to_pydict()
            for text, blob in zip(d["sentence"], d["audio"]):
                i = global_idx
                global_idx += 1

                row = keep.get(i)
                if row is None:
                    continue
                if isinstance(blob, dict):
                    blob = blob.get("bytes")
                encoded = to_flac(blob) if blob else None
                if encoded is None:
                    n_bad += 1
                    continue

                flac, duration = encoded
                buffer.append({
                    "idx": i,
                    "text": text,
                    "audio": flac,
                    "duration": round(duration, 4),
                    "speaker": int(row["speaker"]),
                    "speaker_vector": embeddings[i].astype(np.float16).tobytes(),
                    "mos_ovr": float(row["mos_ovr"]),
                })
                n_written += 1
                seconds += duration

                if len(buffer) >= args.rows_per_shard:
                    flush_shard()
                if n_written % 4000 == 0:
                    print(f"[pack] {n_written}/{len(keep)}  {seconds/3600:.1f} h  "
                          f"{(time.time()-started)/60:.1f} min", flush=True)
                if args.max_rows and n_written >= args.max_rows:
                    stop = True
                    break
                if args.max_minutes and (time.time() - started) / 60 > args.max_minutes:
                    print("[pack] time budget reached", flush=True)
                    stop = True
                    break
            if stop:
                break
    flush_shard()

    summary = {
        "source_manifest": args.manifest,
        "clips_selected": len(keep),
        "clips_written": n_written,
        "clips_unreadable": n_bad,
        "hours": round(seconds / 3600, 2),
        "shards": shard_no,
        "gb": round(written_bytes / 1e9, 3),
        "sample_rate": TARGET_SR,
        "audio_codec": "flac",
        "speaker_vector_dim": int(embeddings.shape[1]),
        "speaker_vector_dtype": "float16",
        "elapsed_minutes": round((time.time() - started) / 60, 2),
    }
    (out_dir / "pack_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(out_dir), repo_id=args.repo_id, repo_type="dataset",
        commit_message=f"pack: {n_written} clips, {seconds/3600:.1f} h, {written_bytes/1e9:.2f} GB",
    )
    print(f"[pack] pushed to {args.repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
