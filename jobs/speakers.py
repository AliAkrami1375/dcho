#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.44",
#   "speechbrain>=1.0",
#   "pyarrow>=15.0",
#   "numpy>=1.26",
#   "scikit-learn>=1.4",
#   "soundfile>=0.12",
#   "huggingface_hub>=0.28",
# ]
# ///
"""Speaker discovery.

The corpus carries no speaker labels - it was assembled by concatenating
several Farsi datasets - so they have to be recovered before anything
speaker-conditioned can be trained. Without them there is no way to build a
multi-speaker model, and no way to pick which voice the finished product
should actually ship.

Method: an x-vector per utterance, then clustering.

Two choices here are about cost rather than accuracy. Only a short centre
crop of each clip is embedded, because a speaker embedding saturates after
a few seconds of speech and the corpus median is over nine seconds -
cropping cuts the GPU time by roughly a factor of three for no measurable
loss. And clustering is fitted on a subsample rather than on all 109k
vectors, because agglomerative clustering is quadratic in memory; the full
set is then assigned to the resulting centroids, which is linear.
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
import pyarrow.parquet as pq
import soundfile as sf
import torch

DEFAULT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
TARGET_SR = 16000


def separation_report(vectors: np.ndarray, manifest_lps: np.ndarray | None, seed: int = 0) -> dict:
    """Measure whether an embedding actually separates speakers here.

    A speaker embedding can look healthy - finite, unsaturated, normalised -
    and still be useless on a given corpus, because it was trained on a
    different language and channel. These four numbers say whether it is
    carrying speaker identity or just recording conditions:

      centroid_cosine     how far the cloud spreads. Near 1.0 means every
                          embedding points the same way and there is nothing
                          to cluster.
      dims_for_90pct_var  effective dimensionality. A 512-d embedding using
                          14 of them has collapsed.
      random_pair_median  similarity between two arbitrary clips. On a
                          many-speaker corpus this should be well below the
                          model's same-speaker threshold.
      core_rate_tightening  the decisive one. Take the densest neighbourhood
                          and compare its speech-rate spread against the
                          whole corpus. Speech rate is a strong speaker
                          trait that the embedding never saw, so a genuine
                          speaker group must be markedly tighter. If it is
                          not, the neighbourhood is not a speaker.
    """
    rng = np.random.default_rng(seed)
    n = min(6000, len(vectors))
    idx = rng.choice(len(vectors), n, replace=False)
    probe = vectors[idx]

    centroid = vectors.mean(0)
    centroid /= np.linalg.norm(centroid) + 1e-9

    sv = np.linalg.svd(probe - probe.mean(0), compute_uv=False)
    ev = sv**2 / (sv**2).sum()

    sim = probe @ probe.T
    np.fill_diagonal(sim, -1.0)
    pairs = sim[np.triu_indices(n, 1)]

    out = {
        "centroid_cosine": round(float((vectors @ centroid).mean()), 4),
        "dims_for_90pct_var": int((np.cumsum(ev) < 0.9).sum()) + 1,
        "total_dims": int(vectors.shape[1]),
        "random_pair_similarity": {
            f"p{p}": round(float(np.percentile(pairs, p)), 4) for p in (1, 5, 25, 50, 75, 95)
        },
    }

    if manifest_lps is not None and len(manifest_lps) == len(vectors):
        radius = float(np.percentile(pairs, 99.5))
        counts = (sim > radius).sum(1)
        seed_i = int(counts.argmax())
        core = idx[np.where(sim[seed_i] > radius)[0]]
        if len(core) >= 30:
            def iqr(a):
                return float(np.percentile(a, 75) - np.percentile(a, 25))

            core_iqr = iqr(manifest_lps[core])
            all_iqr = iqr(manifest_lps)
            out["densest_core"] = {
                "radius": round(radius, 4),
                "n": int(len(core)),
                "speech_rate_iqr": round(core_iqr, 4),
                "corpus_speech_rate_iqr": round(all_iqr, 4),
                "core_rate_tightening_pct": round(100 * (1 - core_iqr / all_iqr), 1),
            }
    return out


def load_crop(blob: bytes, seconds: float) -> np.ndarray | None:
    """Centre crop of `seconds`, mono float32. None if unreadable."""
    try:
        data, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    except Exception:
        return None
    audio = data.mean(axis=1)
    if sr != TARGET_SR:
        # The corpus is uniformly 16 kHz; anything else is an anomaly worth
        # dropping rather than silently resampling.
        return None
    want = int(seconds * sr)
    if len(audio) <= want:
        return audio
    start = (len(audio) - want) // 2
    return audio[start : start + want]


@torch.no_grad()
def embed_all(files, model, device, crop_seconds, batch_size, max_rows, max_minutes, log_every):
    started = time.time()
    embeddings: list[np.ndarray] = []
    index: list[dict] = []
    buffer: list[np.ndarray] = []
    meta: list[dict] = []
    n_seen = n_bad = 0

    # The model may be running in half precision; inputs have to follow it.
    param_dtype = next(model.parameters()).dtype

    def flush():
        nonlocal buffer, meta
        if not buffer:
            return
        width = max(len(b) for b in buffer)
        batch = np.zeros((len(buffer), width), dtype=np.float32)
        mask = np.zeros((len(buffer), width), dtype=np.int64)
        for i, b in enumerate(buffer):
            batch[i, : len(b)] = b
            mask[i, : len(b)] = 1
        out = model(
            input_values=torch.from_numpy(batch).to(device=device, dtype=param_dtype),
            attention_mask=torch.from_numpy(mask).to(device),
        ).embeddings
        out = torch.nn.functional.normalize(out, dim=-1).float().cpu().numpy()
        embeddings.append(out)
        index.extend(meta)
        buffer, meta = [], []

    stop = False
    for path in files:
        if stop:
            break
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=64, columns=["audio"]):
            for blob in batch.to_pydict()["audio"]:
                if isinstance(blob, dict):
                    blob = blob.get("bytes")
                audio = load_crop(blob, crop_seconds) if blob else None
                if audio is None or len(audio) < TARGET_SR // 2:
                    n_bad += 1
                    n_seen += 1
                    continue
                buffer.append(audio)
                meta.append({"idx": n_seen, "shard": path.name})
                n_seen += 1
                if len(buffer) >= batch_size:
                    flush()
                if n_seen % log_every == 0:
                    el = time.time() - started
                    rate = n_seen / max(el, 1e-6)
                    print(f"[spk] {n_seen} clips  {el:.0f}s  {rate:.1f}/s", flush=True)
                if max_rows and n_seen >= max_rows:
                    stop = True
                    break
                if max_minutes and (time.time() - started) / 60 > max_minutes:
                    print("[spk] time budget reached", flush=True)
                    stop = True
                    break
            if stop:
                break
    flush()
    return (np.concatenate(embeddings) if embeddings else np.zeros((0, 512), np.float32)), index, n_seen, n_bad


def cluster(vectors: np.ndarray, threshold: float, fit_size: int, seed: int):
    """Agglomerative fit on a subsample, then nearest-centroid assignment."""
    from sklearn.cluster import AgglomerativeClustering

    rng = np.random.default_rng(seed)
    n = len(vectors)
    fit_idx = rng.choice(n, size=min(fit_size, n), replace=False)
    fit = vectors[fit_idx]

    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, metric="cosine", linkage="average"
    )
    labels_fit = model.fit_predict(fit)

    n_clusters = int(labels_fit.max()) + 1
    centroids = np.zeros((n_clusters, vectors.shape[1]), dtype=np.float32)
    for c in range(n_clusters):
        m = fit[labels_fit == c]
        v = m.mean(axis=0)
        centroids[c] = v / (np.linalg.norm(v) + 1e-9)

    # Cosine similarity to every centroid, in blocks so memory stays flat.
    labels = np.zeros(n, dtype=np.int32)
    confidence = np.zeros(n, dtype=np.float32)
    for s in range(0, n, 8192):
        block = vectors[s : s + 8192]
        sim = block @ centroids.T
        labels[s : s + 8192] = sim.argmax(axis=1)
        confidence[s : s + 8192] = sim.max(axis=1)
    return labels, confidence, centroids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/corpus")
    ap.add_argument("--out-dir", default="/tmp/dcho-speakers")
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--model-type", default="auto", choices=["auto", "ecapa", "wavlm"])
    ap.add_argument("--diagnose-only", action="store_true",
                    help="report separation quality and stop, without clustering")
    ap.add_argument("--manifest", default=None,
                    help="manifest parquet, enables the speech-rate check")
    ap.add_argument("--crop-seconds", type=float, default=4.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--cluster-threshold", type=float, default=None,
                    help="cosine distance; default is derived from the data")
    ap.add_argument("--sweep", type=float, nargs="*", default=None,
                    help="thresholds to report; default derives them from the data")
    ap.add_argument("--threshold-percentile", type=float, default=97.0,
                    help="pair-similarity percentile that defines same-speaker")
    ap.add_argument("--fit-size", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log-every", type=int, default=2000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[spk] device {device}", flush=True)
    if device == "cuda":
        print(f"[spk] gpu {torch.cuda.get_device_name()}", flush=True)

    # ECAPA-TDNN is the standard choice for speaker clustering and separates
    # far better than a transformer x-vector head. It is kept in float32:
    # this stage is short, and half precision is not worth any risk to the
    # numbers every later decision rests on.
    if "ecapa" in args.model.lower() or args.model_type == "ecapa":
        from speechbrain.inference.speaker import EncoderClassifier

        encoder = EncoderClassifier.from_hparams(
            source=args.model, savedir="/tmp/spk-model", run_opts={"device": device}
        )

        class _Ecapa:
            def __call__(self, input_values, attention_mask=None):
                lengths = (
                    attention_mask.sum(1).float() / input_values.shape[1]
                    if attention_mask is not None
                    else torch.ones(input_values.shape[0], device=input_values.device)
                )
                emb = encoder.encode_batch(input_values, lengths).squeeze(1)
                return type("O", (), {"embeddings": emb})()

            def parameters(self):
                return encoder.mods.parameters()

        model = _Ecapa()
    else:
        from transformers import WavLMForXVector

        model = WavLMForXVector.from_pretrained(args.model).to(device).eval()

    files = sorted(Path(args.data_dir).rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {args.data_dir}")
    print(f"[spk] {len(files)} shards", flush=True)

    started = time.time()
    vectors, index, n_seen, n_bad = embed_all(
        files, model, device, args.crop_seconds, args.batch_size,
        args.max_rows, args.max_minutes, args.log_every,
    )
    print(f"[spk] embedded {len(vectors)} of {n_seen} ({n_bad} unusable) "
          f"in {(time.time()-started)/60:.1f} min", flush=True)

    # Embedding is the expensive part and clustering is free, so the
    # threshold is measured rather than guessed: sweep it once over the
    # vectors we already have and report what each choice produces. A
    # threshold picked from a paper's default is a threshold picked for a
    # different corpus.
    rng = np.random.default_rng(args.seed)
    probe = vectors[rng.choice(len(vectors), size=min(4000, len(vectors)), replace=False)]
    sims = (probe @ probe.T)[np.triu_indices(len(probe), k=1)]
    similarity_profile = {
        f"p{p}": round(float(np.percentile(sims, p)), 4)
        for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)
    }
    print(f"[spk] pairwise cosine similarity: {similarity_profile}", flush=True)

    # Thresholds are derived from the measured similarity distribution, not
    # hardcoded. Different encoders live on completely different scales -
    # WavLM's x-vector head puts random pairs near 0.87 while ECAPA puts
    # them near 0.19 - so a fixed sweep range is meaningful for one encoder
    # and pure noise for the other. Percentiles of the observed pair
    # distribution are comparable across both.
    if args.sweep:
        thresholds = list(args.sweep)
    else:
        thresholds = sorted({
            round(1.0 - float(np.percentile(sims, q)), 3)
            for q in (85, 90, 93, 95, 97, 98, 99, 99.5)
        })
        print(f"[spk] sweep derived from pair distribution: {thresholds}", flush=True)

    sweep = {}
    for t in thresholds:
        lab, conf, _ = cluster(vectors, t, args.fit_size, args.seed)
        counts = np.bincount(lab)
        counts = counts[counts > 0]
        sweep[str(t)] = {
            "n_clusters": int(len(counts)),
            "largest_pct": round(100.0 * float(counts.max()) / len(lab), 2),
            "clusters_over_1pct": int((counts / len(lab) > 0.01).sum()),
            "mean_confidence": round(float(conf.mean()), 4),
        }
        print(f"[spk] threshold {t:.2f} -> {sweep[str(t)]}", flush=True)

    lps = None
    if args.manifest:
        import pyarrow.parquet as _pq

        src = args.manifest
        if not Path(src).exists():
            # Accept a hub repo id so the job does not need the manifest
            # staged into its image.
            from huggingface_hub import hf_hub_download

            src = hf_hub_download(src, "manifest.parquet", repo_type="dataset",
                                  token=os.environ.get("HF_TOKEN"))
        col = _pq.read_table(src, columns=["letters_per_sec"]).column(0).to_numpy()
        lps = col[: len(vectors)]

    sep = separation_report(vectors, lps, args.seed)
    print("[spk] separation report:", json.dumps(sep, indent=2), flush=True)

    if args.diagnose_only:
        (out_dir / "separation.json").write_text(
            json.dumps({"model": args.model, **sep}, indent=2), encoding="utf-8")
        np.save(out_dir / "embeddings.npy", vectors.astype(np.float16))
        return 0

    chosen = args.cluster_threshold
    if chosen is None:
        chosen = round(1.0 - float(np.percentile(sims, args.threshold_percentile)), 3)
        print(f"[spk] chosen threshold {chosen} "
              f"(p{args.threshold_percentile} of pair similarity)", flush=True)

    labels, confidence, centroids = cluster(vectors, chosen, args.fit_size, args.seed)

    sizes = np.bincount(labels)
    order = np.argsort(-sizes)
    summary = {
        "model": args.model,
        "crop_seconds": args.crop_seconds,
        "cluster_threshold": chosen,
        "threshold_sweep": sweep,
        "n_embedded": int(len(vectors)),
        "n_seen": int(n_seen),
        "n_unusable": int(n_bad),
        "separation": sep,
        "n_clusters": int(len(sizes)),
        "mean_assignment_confidence": round(float(confidence.mean()), 4),
        "low_confidence_below_0.5": int((confidence < 0.5).sum()),
        "clusters": [
            {"id": int(c), "n": int(sizes[c]),
             "pct": round(100.0 * sizes[c] / len(labels), 2),
             "mean_confidence": round(float(confidence[labels == c].mean()), 4)}
            for c in order[:60]
        ],
    }
    (out_dir / "speakers_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(out_dir / "embeddings.npy", vectors.astype(np.float16))
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "confidence.npy", confidence)
    np.save(out_dir / "centroids.npy", centroids)
    (out_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k != "clusters"}, indent=2), flush=True)
    print("[spk] top clusters:", flush=True)
    for c in summary["clusters"][:15]:
        print(f"    cluster {c['id']:3d}  n={c['n']:6d}  {c['pct']:5.2f}%  conf={c['mean_confidence']:.3f}", flush=True)

    if args.repo_id:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(folder_path=str(out_dir), repo_id=args.repo_id,
                          repo_type="dataset", path_in_repo="speakers",
                          commit_message=f"speakers: {summary['n_clusters']} clusters over {len(vectors)} clips")
        print(f"[spk] pushed to {args.repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
