"""Inference runtime.

Depends on onnxruntime and numpy only - deliberately not on torch. That is
what makes an install about 50 MB instead of two gigabytes, and it is the
difference between something that deploys onto a small VPS and something
that does not.

Two behaviours here matter more to how fast the system *feels* than the
model's raw throughput does.

Long text is split at punctuation and synthesised chunk by chunk, with each
chunk yielded as soon as it is ready. Time to first audio then stops
growing with the length of the input: a paragraph starts speaking as
quickly as a sentence does. Total synthesis time is unchanged, but the
perceived latency is completely different.

Thread count is capped rather than maximised. Beyond about four threads the
synchronisation overhead on a single short request exceeds what the extra
parallelism buys back, so more threads make one request slower. A server
handling concurrent requests should give each request one or two threads and
find its parallelism across requests instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Split points, strongest first. Splitting mid-clause would break prosody,
# so only real boundaries are used and a long clause is left intact.
_SPLIT_STRONG = re.compile(r"(?<=[.؟!…])\s+")
_SPLIT_WEAK = re.compile(r"(?<=[،؛:])\s+")


@dataclass
class SynthesisOptions:
    speaker: int = 0
    speed: float = 1.0
    noise_scale: float = 0.667
    noise_scale_w: float = 0.8
    max_chunk_chars: int = 220


class TTS:
    """ONNX-backed Persian speech synthesiser."""

    def __init__(
        self,
        model_path: str | Path,
        sample_rate: int = 16000,
        num_threads: int = 2,
        frontend=None,
    ):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(num_threads, 4))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self.sample_rate = sample_rate
        self._frontend = frontend
        self._cache: dict[str, list[int]] = {}

    @property
    def frontend(self):
        if self._frontend is None:
            from ..text.frontend import Frontend

            self._frontend = Frontend()
        return self._frontend

    def _encode(self, text: str) -> list[int]:
        """Phonemise with a cache; repeated phrases are common in real use."""
        ids = self._cache.get(text)
        if ids is None:
            ids = self.frontend(text).ids
            if len(self._cache) < 8192:
                self._cache[text] = ids
        return ids

    def _run(self, ids: list[int], opt: SynthesisOptions) -> np.ndarray:
        if len(ids) < 3:
            return np.zeros(0, dtype=np.float32)
        audio = self.session.run(
            None,
            {
                "text": np.asarray([ids], dtype=np.int64),
                "text_lengths": np.asarray([len(ids)], dtype=np.int64),
                "sid": np.asarray([opt.speaker], dtype=np.int64),
                "noise_scale": np.float32(opt.noise_scale),
                # Speed is the reciprocal of the duration multiplier: a
                # larger length_scale stretches the utterance out.
                "length_scale": np.float32(1.0 / max(opt.speed, 1e-3)),
                "noise_scale_w": np.float32(opt.noise_scale_w),
            },
        )[0]
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def split(self, text: str, max_chars: int) -> list[str]:
        """Break text into synthesis chunks at the strongest available point."""
        parts = [p.strip() for p in _SPLIT_STRONG.split(text) if p.strip()]
        out: list[str] = []
        for part in parts:
            if len(part) <= max_chars:
                out.append(part)
                continue
            buf = ""
            for clause in _SPLIT_WEAK.split(part):
                if buf and len(buf) + len(clause) + 1 > max_chars:
                    out.append(buf.strip())
                    buf = clause
                else:
                    buf = f"{buf} {clause}".strip()
            if buf:
                out.append(buf.strip())
        return out or ([text.strip()] if text.strip() else [])

    def stream(self, text: str, options: SynthesisOptions | None = None):
        """Yield audio chunk by chunk as each is synthesised."""
        opt = options or SynthesisOptions()
        for chunk in self.split(text, opt.max_chunk_chars):
            audio = self._run(self._encode(chunk), opt)
            if len(audio):
                yield audio

    def __call__(self, text: str, options: SynthesisOptions | None = None, **kwargs) -> np.ndarray:
        opt = options or SynthesisOptions(**kwargs)
        chunks = list(self.stream(text, opt))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        # A short silence between chunks restores the pause the split
        # consumed; without it the joins sound clipped.
        gap = np.zeros(int(0.08 * self.sample_rate), dtype=np.float32)
        joined: list[np.ndarray] = []
        for i, c in enumerate(chunks):
            if i:
                joined.append(gap)
            joined.append(c)
        return np.concatenate(joined)

    def say(self, text: str, out: str | Path, **kwargs) -> Path:
        audio = self(text, **kwargs)
        return write_wav(out, audio, self.sample_rate)


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    """Write 16-bit PCM without pulling in soundfile."""
    import wave

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak > 1.0:
        audio = audio / peak
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return path
