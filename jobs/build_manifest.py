#!/usr/bin/env python3
"""Assemble the training manifests.

Combines three separately produced sources - the corpus audit, the speaker
embeddings, and the speech-rate window derived from the audit - into the
tier files that training reads. Runs locally on a CPU in seconds and costs
nothing.

Two decisions are worth stating because they are not obvious from the code.

The evaluation split is **speaker-disjoint**. Held-out clips come from
speaker clusters that appear nowhere in training. A random split would let
the model be scored on voices it has already memorised, and the resulting
numbers would flatter it in exactly the dimension a multi-speaker model
most needs measured.

Filtering uses the speech-rate window rather than an ASR round trip. That
is a budget decision with a measured basis: verifying transcripts with
Whisper benchmarked at 2.3x real time, which projects to roughly $194 for
this corpus - more than the entire project budget. The speech-rate check
costs nothing and catches the same tail of gross mismatches, if less
precisely. See docs/DESIGN.fa.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_inputs(base: Path):
    import pyarrow.parquet as pq

    manifest = pq.read_table(base / "manifest.parquet").to_pandas()
    labels = np.load(base / "speakers" / "labels.npy")
    confidence = np.load(base / "speakers" / "confidence.npy")
    embeddings = np.load(base / "speakers" / "embeddings.npy", mmap_mode="r")

    n = len(manifest)
    if not (len(labels) == len(confidence) == len(embeddings) == n):
        raise SystemExit(
            f"inputs disagree on row count: manifest={n} labels={len(labels)} "
            f"confidence={len(confidence)} embeddings={len(embeddings)}"
        )
    return manifest, labels, confidence


def speech_rate_window(lps: np.ndarray, low_pct: float, high_pct: float):
    valid = lps[(lps > 0) & np.isfinite(lps)]
    return float(np.percentile(valid, low_pct)), float(np.percentile(valid, high_pct))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True,
                    help="directory holding manifest.parquet and speakers/")
    ap.add_argument("--out-dir", default="manifest")
    ap.add_argument("--eval-clips", type=int, default=500)
    ap.add_argument("--eval-speakers", type=int, default=40)
    ap.add_argument("--min-cluster-size", type=int, default=10)
    ap.add_argument("--max-eval-cluster-size", type=int, default=200,
                    help="reserving a very large cluster would cost real training data")
    ap.add_argument("--rate-low-pct", type=float, default=2.0)
    ap.add_argument("--rate-high-pct", type=float, default=98.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    base = Path(args.inputs)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    m, labels, confidence = load_inputs(base)
    rng = np.random.default_rng(args.seed)

    duration = m["duration"].to_numpy()
    mos = m["mos_ovr"].to_numpy()
    lps = m["letters_per_sec"].to_numpy()
    lo, hi = speech_rate_window(lps, args.rate_low_pct, args.rate_high_pct)
    rate_ok = (lps >= lo) & (lps <= hi)

    tier_a = (mos >= 3.6) & (duration >= 1.0) & (duration <= 12.0) & rate_ok
    tier_b = (mos >= 3.0) & (duration >= 0.8) & (duration <= 13.0) & rate_ok

    sizes = np.bincount(labels, minlength=int(labels.max()) + 1)
    eligible = [
        c for c in range(len(sizes))
        if args.min_cluster_size <= sizes[c] <= args.max_eval_cluster_size
        and tier_a[labels == c].sum() >= 5
    ]
    rng.shuffle(eligible)

    eval_speakers: list[int] = []
    eval_mask = np.zeros(len(m), dtype=bool)
    for c in eligible:
        if eval_mask.sum() >= args.eval_clips or len(eval_speakers) >= args.eval_speakers:
            break
        take = (labels == c) & tier_a
        if take.sum() == 0:
            continue
        eval_speakers.append(int(c))
        eval_mask |= take

    held_out = np.isin(labels, eval_speakers)
    tier_a_train = tier_a & ~held_out
    tier_b_train = tier_b & ~held_out

    def write(name: str, mask: np.ndarray) -> dict:
        idx = np.where(mask)[0]
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i in idx:
                f.write(json.dumps({
                    "idx": int(m["idx"].iloc[i]),
                    "shard": str(m["shard"].iloc[i]),
                    "duration": round(float(duration[i]), 3),
                    "speaker": int(labels[i]),
                    "speaker_confidence": round(float(confidence[i]), 4),
                    "mos_ovr": round(float(mos[i]), 3),
                    "letters_per_sec": round(float(lps[i]), 3),
                }) + "\n")
        return {
            "file": path.name,
            "clips": int(mask.sum()),
            "hours": round(float(duration[mask].sum()) / 3600, 2),
            "speakers": int(len(np.unique(labels[mask]))),
            "median_mos": round(float(np.median(mos[mask])), 3),
            "median_duration": round(float(np.median(duration[mask])), 2),
        }

    summary = {
        "speech_rate_window": [round(lo, 3), round(hi, 3)],
        "n_eval_speakers": len(eval_speakers),
        "eval_speakers": sorted(eval_speakers),
        "splits": {
            "tier_a_train": write("tier_a_train", tier_a_train),
            "tier_b_train": write("tier_b_train", tier_b_train),
            "eval": write("eval", eval_mask),
        },
    }

    # The guard against the exact mistake this split exists to prevent.
    overlap = set(np.unique(labels[tier_b_train]).tolist()) & set(
        np.unique(labels[eval_mask]).tolist()
    )
    summary["speaker_overlap_train_eval"] = len(overlap)

    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "eval_speakers"},
                     ensure_ascii=False, indent=2))

    if overlap:
        raise SystemExit(f"eval split leaks {len(overlap)} speakers into training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
