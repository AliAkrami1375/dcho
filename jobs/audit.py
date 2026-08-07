#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pyarrow>=15.0",
#   "numpy>=1.26",
#   "soundfile>=0.12",
#   "huggingface_hub>=0.28",
#   "tqdm>=4.66",
# ]
# ///
"""Corpus audit for dcho.

Runs first, costs almost nothing, and answers the questions every later
decision depends on: how many hours are actually here, how long are the
utterances, how is quality distributed, how many distinct texts are there,
and which characters does the frontend have to handle.

Two things worth knowing about how this is built.

It never decodes audio. `soundfile.info` reads only the container header,
which is enough for duration and sample rate and is roughly two orders of
magnitude cheaper than decoding. That is what lets the whole pass run on a
CPU flavour costing three cents an hour.

It computes a speech-rate estimate - Persian letters per second - for every
utterance. That single number is a remarkably good proxy for transcript
mismatch: a clip whose text is far too short or far too long for its
duration is almost always misaligned. It costs nothing here, and it lets
the expensive ASR verification pass later be pointed at the suspicious tail
instead of the whole corpus.

Output is a manifest (one row per utterance, no audio) plus a JSON report,
both pushed to a dataset repo on the Hub.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

PERSIAN_LETTER = re.compile(r"[ء-ی]")
LATIN = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"[0-9۰-۹٠-٩]")
ZWNJ = "‌"

# Percentiles reported for every continuous quantity.
PCT = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def percentiles(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {}
    qs = np.percentile(values, PCT)
    out = {f"p{p}": round(float(q), 4) for p, q in zip(PCT, qs)}
    out["mean"] = round(float(values.mean()), 4)
    out["min"] = round(float(values.min()), 4)
    out["max"] = round(float(values.max()), 4)
    return out


def audio_header(blob: bytes) -> tuple[float, int, str] | None:
    """(duration_seconds, sample_rate, format) without decoding samples."""
    try:
        info = sf.info(io.BytesIO(blob))
        return float(info.frames) / info.samplerate, int(info.samplerate), str(info.format)
    except Exception:
        return None


def find_parquet_files(root: Path) -> list[Path]:
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files found under {root}")
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/corpus", help="mounted dataset repo")
    ap.add_argument("--out-dir", default="/tmp/dcho-audit")
    ap.add_argument("--repo-id", default=None, help="dataset repo to push results to")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all; use a small value to smoke-test")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="stop early and still write output")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    started = time.time()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_parquet_files(data_dir)
    print(f"[audit] {len(files)} parquet files under {data_dir}", flush=True)

    # Per-utterance accumulators.
    durations: list[float] = []
    n_chars: list[int] = []
    n_words: list[int] = []
    n_letters: list[int] = []
    mos_cols: dict[str, list[float]] = {}

    char_counter: Counter = Counter()
    sr_counter: Counter = Counter()
    fmt_counter: Counter = Counter()
    text_counter: Counter = Counter()

    n_rows = 0
    n_unreadable = 0
    n_empty_text = 0
    n_latin = 0
    n_digit = 0
    n_zwnj = 0

    manifest_rows: list[dict] = []
    stop = False

    for path in files:
        if stop:
            break
        pf = pq.ParquetFile(path)
        schema_names = set(pf.schema_arrow.names)
        mos_names = [c for c in schema_names if c.startswith("mos_")]
        cols = [c for c in ("sentence", "audio", *mos_names) if c in schema_names]

        for batch in pf.iter_batches(batch_size=args.batch_size, columns=cols):
            d = batch.to_pydict()
            sentences = d.get("sentence", [])
            audios = d.get("audio", [])

            for i in range(len(sentences)):
                text = sentences[i] or ""
                blob = audios[i]
                if isinstance(blob, dict):
                    blob = blob.get("bytes")

                hdr = audio_header(blob) if blob else None
                if hdr is None:
                    n_unreadable += 1
                    continue
                dur, sr, fmt = hdr

                letters = len(PERSIAN_LETTER.findall(text))
                words = len(text.split())
                if letters == 0:
                    n_empty_text += 1
                if LATIN.search(text):
                    n_latin += 1
                if DIGIT.search(text):
                    n_digit += 1
                if ZWNJ in text:
                    n_zwnj += 1

                durations.append(dur)
                n_chars.append(len(text))
                n_words.append(words)
                n_letters.append(letters)
                sr_counter[sr] += 1
                fmt_counter[fmt] += 1
                char_counter.update(text)
                text_counter[text.strip()] += 1

                row = {
                    "idx": n_rows,
                    "shard": path.name,
                    "duration": round(dur, 4),
                    "sample_rate": sr,
                    "n_chars": len(text),
                    "n_words": words,
                    "n_letters": letters,
                    "letters_per_sec": round(letters / dur, 4) if dur > 0 else 0.0,
                }
                for c in mos_names:
                    v = d[c][i]
                    row[c] = None if v is None else round(float(v), 4)
                    mos_cols.setdefault(c, []).append(float(v) if v is not None else np.nan)
                manifest_rows.append(row)

                n_rows += 1
                if n_rows % 5000 == 0:
                    el = time.time() - started
                    print(f"[audit] {n_rows} rows  {sum(durations)/3600:.1f} h  {el:.0f}s", flush=True)

                if args.max_rows and n_rows >= args.max_rows:
                    stop = True
                    break
                if args.max_minutes and (time.time() - started) / 60 > args.max_minutes:
                    print("[audit] time budget reached, writing partial results", flush=True)
                    stop = True
                    break
            if stop:
                break

    dur = np.array(durations)
    lps = np.array([r["letters_per_sec"] for r in manifest_rows])
    total_hours = float(dur.sum()) / 3600

    # Percentiles hide multi-modality, and a corpus assembled from several
    # source datasets is almost guaranteed to be multi-modal in duration.
    # An explicit histogram shows the structure that p50/p90 flattens out.
    edges = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60, 1e9]
    hist, _ = np.histogram(dur, bins=edges)
    duration_histogram = {
        f"{edges[i]:g}-{edges[i+1]:g}s": int(hist[i])
        for i in range(len(hist)) if hist[i] > 0
    }

    # Per-shard aggregates. Shards usually follow source-corpus order, so
    # a shard whose statistics differ sharply from its neighbours is a
    # different underlying dataset and probably a different speaker pool.
    by_shard: dict[str, dict] = {}
    for r in manifest_rows:
        s = by_shard.setdefault(r["shard"], {"n": 0, "seconds": 0.0, "lps": [], "mos": []})
        s["n"] += 1
        s["seconds"] += r["duration"]
        s["lps"].append(r["letters_per_sec"])
        if r.get("mos_ovr") is not None:
            s["mos"].append(r["mos_ovr"])
    shard_summary = {
        k: {
            "n": v["n"],
            "hours": round(v["seconds"] / 3600, 3),
            "mean_duration": round(v["seconds"] / max(v["n"], 1), 2),
            "median_letters_per_sec": round(float(np.median(v["lps"])), 2) if v["lps"] else None,
            "median_mos_ovr": round(float(np.median(v["mos"])), 3) if v["mos"] else None,
        }
        for k, v in sorted(by_shard.items())
    }

    # Speech rate outliers. The window is set from the data rather than
    # hardcoded, so it adapts to whatever register this corpus turns out to
    # be in; anything outside it is a transcript-mismatch candidate.
    lps_valid = lps[(lps > 0) & np.isfinite(lps)]
    lo, hi = (np.percentile(lps_valid, [2, 98]) if len(lps_valid) else (0.0, 0.0))
    n_rate_outliers = int(((lps < lo) | (lps > hi)).sum())

    def tier(mos_min: float, dmin: float, dmax: float) -> dict:
        m = np.array(mos_cols.get("mos_ovr", []), dtype=float)
        if len(m) != len(dur):
            m = np.full(len(dur), np.nan)
        keep = (dur >= dmin) & (dur <= dmax) & (np.nan_to_num(m, nan=-1) >= mos_min)
        keep &= (lps >= lo) & (lps <= hi)
        return {
            "n": int(keep.sum()),
            "hours": round(float(dur[keep].sum()) / 3600, 2),
            "pct_of_corpus": round(100.0 * keep.sum() / max(len(dur), 1), 2),
        }

    report = {
        "dataset": "Thomcles/Persian-Farsi-Speech",
        "rows_scanned": n_rows,
        "elapsed_seconds": round(time.time() - started, 1),
        "total_hours": round(total_hours, 2),
        "unreadable_audio": n_unreadable,
        "empty_text": n_empty_text,
        "text_with_latin": n_latin,
        "text_with_digits": n_digit,
        "text_with_zwnj": n_zwnj,
        "unique_texts": len(text_counter),
        "duplicate_text_rows": n_rows - len(text_counter),
        "most_common_texts": text_counter.most_common(15),
        "sample_rates": dict(sr_counter),
        "audio_formats": dict(fmt_counter),
        "duration_seconds": percentiles(dur),
        "duration_histogram": duration_histogram,
        "duration_over_15s": int((dur > 15).sum()),
        "duration_over_20s": int((dur > 20).sum()),
        "duration_over_30s": int((dur > 30).sum()),
        "duration_under_2s": int((dur < 2).sum()),
        "by_shard": shard_summary,
        "n_chars": percentiles(np.array(n_chars)),
        "n_words": percentiles(np.array(n_words)),
        "letters_per_second": percentiles(lps_valid),
        "speech_rate_window": [round(float(lo), 3), round(float(hi), 3)],
        "speech_rate_outliers": n_rate_outliers,
        "mos": {c: percentiles(np.array(v, dtype=float)) for c, v in mos_cols.items()},
        "character_inventory": {
            "n_distinct": len(char_counter),
            "counts": dict(char_counter.most_common()),
        },
        "proposed_tiers": {
            "tier_A_mos3.6_1to12s": tier(3.6, 1.0, 12.0),
            "tier_B_mos3.0_0.8to13s": tier(3.0, 0.8, 13.0),
            "all_no_filter": tier(-1.0, 0.0, 1e9),
        },
    }

    (out_dir / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    table = pa.Table.from_pylist(manifest_rows)
    pq.write_table(table, out_dir / "manifest.parquet", compression="zstd")

    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("character_inventory", "most_common_texts")},
                     ensure_ascii=False, indent=2), flush=True)

    if args.repo_id:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"audit: {n_rows} rows, {total_hours:.1f} h",
        )
        print(f"[audit] pushed to {args.repo_id}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
