#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.44",
#   "pyarrow>=15.0",
#   "numpy>=1.26",
#   "soundfile>=0.12",
#   "huggingface_hub>=0.28",
# ]
# ///
"""Transcript verification by ASR round-trip.

A corpus stitched together from several sources always contains some
utterances whose text does not match their audio. Those rows are the
single worst thing that can be in a text-to-speech training set: the model
learns that text sometimes does not determine what is said, and the
symptom at inference time is a system that drops or invents words.

The check is to recognise every clip with an ASR model and compare the
result against the stored transcript. High character error rate means the
pair disagrees. Which of the two is wrong does not matter - the pair is
unusable either way.

Run this in `--sample` mode first. A few thousand clips is enough to see
the shape of the error distribution and to place the cut, and it costs
under twenty cents; committing to the full pass before knowing that shape
is how a cheap job turns into an expensive one.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

TARGET_SR = 16000
DEFAULT_MODEL = "openai/whisper-large-v3-turbo"

# Comparison-time normalisation. Deliberately aggressive: the goal is to
# detect utterance-level disagreement, so any difference that is merely
# orthographic must be erased or it inflates every score equally and
# washes out the signal.
_CHAR_MAP = {"ي": "ی", "ى": "ی", "ئ": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه",
             "أ": "ا", "إ": "ا", "آ": "ا", "ؤ": "و"}
_DIACRITICS = re.compile("[ً-ْٓ-ٰـ]")
_NON_PERSIAN = re.compile(r"[^ء-ی ]")
_SPACES = re.compile(r"\s+")


def canon(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = "".join(_CHAR_MAP.get(c, c) for c in text)
    text = _DIACRITICS.sub("", text)
    text = text.replace("‌", " ")
    text = _NON_PERSIAN.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance with a rolling row."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    r, h = canon(reference), canon(hypothesis)
    if not r:
        return 1.0
    return edit_distance(r, h) / len(r)


def read_audio(blob: bytes, max_seconds: float) -> np.ndarray | None:
    try:
        data, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    except Exception:
        return None
    if sr != TARGET_SR:
        return None
    audio = data.mean(axis=1)
    limit = int(max_seconds * sr)
    return audio[:limit] if len(audio) > limit else audio


def iter_rows(files, sample_every: int):
    n = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=64, columns=["sentence", "audio"]):
            d = batch.to_pydict()
            for text, blob in zip(d["sentence"], d["audio"]):
                if sample_every <= 1 or n % sample_every == 0:
                    if isinstance(blob, dict):
                        blob = blob.get("bytes")
                    yield n, path.name, text, blob
                n += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/corpus")
    ap.add_argument("--out-dir", default="/tmp/dcho-verify")
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sample", type=int, default=0,
                    help="verify roughly this many clips spread across the corpus; 0 = all")
    ap.add_argument("--total-rows", type=int, default=109401,
                    help="corpus size, used to space the sample evenly")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[verify] device {device}  model {args.model}", flush=True)

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()

    # Newer transformers releases dropped `forced_decoder_ids` from Whisper's
    # generate() in favour of explicit language/task keywords. Probe once so
    # the job works on either, rather than discovering it mid-run.
    import inspect

    gen_params = inspect.signature(model.generate).parameters
    use_kwargs = "language" in gen_params or "kwargs" in gen_params
    forced = None
    if not use_kwargs:
        forced = processor.get_decoder_prompt_ids(language="persian", task="transcribe")
    print(f"[verify] language selection via {'kwargs' if use_kwargs else 'forced ids'}", flush=True)

    files = sorted(Path(args.data_dir).rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {args.data_dir}")

    sample_every = max(1, args.total_rows // args.sample) if args.sample else 1
    print(f"[verify] {len(files)} shards, taking every {sample_every} row(s)", flush=True)

    started = time.time()
    results: list[dict] = []
    batch_audio: list[np.ndarray] = []
    batch_meta: list[tuple] = []
    audio_seconds = 0.0

    def flush():
        nonlocal batch_audio, batch_meta
        if not batch_audio:
            return
        inputs = processor(
            batch_audio, sampling_rate=TARGET_SR, return_tensors="pt",
            return_attention_mask=True,
        )
        feats = inputs.input_features.to(device, dtype=dtype)
        kwargs = {"max_new_tokens": 200}
        if use_kwargs:
            kwargs.update(language="fa", task="transcribe")
        else:
            kwargs["forced_decoder_ids"] = forced
        if "attention_mask" in inputs:
            kwargs["attention_mask"] = inputs.attention_mask.to(device)
        with torch.no_grad():
            ids = model.generate(feats, **kwargs)
        hyps = processor.batch_decode(ids, skip_special_tokens=True)
        for (idx, shard, ref), hyp in zip(batch_meta, hyps):
            results.append({
                "idx": idx, "shard": shard,
                "cer": round(cer(ref, hyp), 4),
                "ref_len": len(canon(ref)), "hyp_len": len(canon(hyp)),
            })
        batch_audio, batch_meta = [], []

    for idx, shard, text, blob in iter_rows(files, sample_every):
        audio = read_audio(blob, args.max_seconds) if blob else None
        if audio is None or len(audio) < TARGET_SR // 4:
            results.append({"idx": idx, "shard": shard, "cer": 1.0, "ref_len": 0, "hyp_len": 0})
            continue
        audio_seconds += len(audio) / TARGET_SR
        batch_audio.append(audio)
        batch_meta.append((idx, shard, text))
        if len(batch_audio) >= args.batch_size:
            flush()
            if len(results) % args.log_every < args.batch_size:
                el = time.time() - started
                print(f"[verify] {len(results)} clips  {el/60:.1f} min  "
                      f"{audio_seconds/max(el,1e-6):.1f}x realtime", flush=True)
        if args.max_minutes and (time.time() - started) / 60 > args.max_minutes:
            print("[verify] time budget reached", flush=True)
            break
    flush()

    scores = np.array([r["cer"] for r in results])
    elapsed = time.time() - started
    report = {
        "model": args.model,
        "n_verified": len(results),
        "sample_every": sample_every,
        "elapsed_minutes": round(elapsed / 60, 2),
        "audio_hours_processed": round(audio_seconds / 3600, 3),
        "realtime_factor": round(audio_seconds / max(elapsed, 1e-6), 1),
        "cer_percentiles": {
            f"p{p}": round(float(np.percentile(scores, p)), 4)
            for p in (5, 10, 25, 50, 75, 90, 95, 99)
        },
        "cer_mean": round(float(scores.mean()), 4),
        "keep_rate_at_threshold": {
            str(t): round(float((scores <= t).mean()), 4)
            for t in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
        },
    }
    if args.sample:
        # Extrapolating the full pass from the sample is the whole point of
        # running in sample mode, so state the number explicitly.
        full_hours = audio_seconds / 3600 * sample_every
        report["projected_full_run"] = {
            "audio_hours": round(full_hours, 1),
            "gpu_hours": round(full_hours / max(report["realtime_factor"], 1e-6) * 3600 / 3600, 2),
            "usd_at_l4x1": round(full_hours / max(report["realtime_factor"], 1e-6) * 0.80, 2),
        }

    (out_dir / "verify_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / "cer.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    if args.repo_id:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(folder_path=str(out_dir), repo_id=args.repo_id,
                          repo_type="dataset", path_in_repo="verify",
                          commit_message=f"verify: {len(results)} clips, median CER {report['cer_percentiles']['p50']}")
        print(f"[verify] pushed to {args.repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
