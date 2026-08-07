"""Training data pipeline.

The corpus is 47 GB of parquet holding uncompressed 16 kHz WAV. Two facts
about that shape the design.

WAV means decoding is essentially free - it is a header parse and a memory
copy, not a codec. That removes the usual reason for a precomputed feature
cache, and with it the cost of building one and the 30-odd GB of storage it
would occupy. Spectrograms are computed on the GPU, where they are also
close to free.

47 GB also means random access is the wrong access pattern. Rather than
index into arbitrary rows, this reads whole parquet row groups in a shuffled
order and shuffles again within a buffer. The result is close enough to
random for training while remaining a sequential read.
"""

from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..model.mel_processing import spectrogram_torch
from ..text.frontend import Frontend

TARGET_SR = 16000


@dataclass
class DataConfig:
    sampling_rate: int = TARGET_SR
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    min_audio_seconds: float = 0.8
    max_audio_seconds: float = 13.0
    add_blank: bool = True
    shuffle_buffer: int = 2048
    max_text_length: int = 400


def _read_audio(blob: bytes) -> np.ndarray | None:
    """Decode an audio blob to mono float32.

    Two containers reach this function. The raw corpus is 16-bit PCM WAV,
    which the standard library handles with no dependency at all. Packed
    subsets are FLAC, which halves their size for repeated downloads and
    needs soundfile. Trying stdlib first keeps the common path dependency
    free and the FLAC path is only reached when it is actually FLAC.
    """
    audio = _read_wav(blob)
    if audio is not None:
        return audio
    try:
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    except Exception:
        return None
    if sr != TARGET_SR:
        return None
    return data[:, 0] if data.shape[1] == 1 else data.mean(axis=1)


def _read_wav(blob: bytes) -> np.ndarray | None:
    """Decode 16-bit PCM WAV using only the standard library."""
    import wave

    try:
        with wave.open(io.BytesIO(blob), "rb") as w:
            if w.getsampwidth() != 2:
                return None
            sr = w.getframerate()
            n = w.getnframes()
            raw = w.readframes(n)
            channels = w.getnchannels()
    except Exception:
        return None

    if sr != TARGET_SR:
        return None
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio


class ParquetSpeechDataset(IterableDataset):
    """Streams (phoneme ids, spectrogram, waveform, speaker id) tuples.

    `speaker_map` maps a global row index to a speaker cluster id, as
    produced by the speaker discovery job. Rows absent from it are dropped:
    a multi-speaker model conditioned on a wrong speaker id learns to
    average voices, which is worse than having less data.
    """

    def __init__(
        self,
        data_dir: str | Path,
        config: DataConfig,
        speaker_map: dict[int, int] | None = None,
        keep_indices: set[int] | None = None,
        frontend: Frontend | None = None,
        seed: int = 1234,
        infinite: bool = True,
    ):
        super().__init__()
        self.files = sorted(Path(data_dir).rglob("*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"no parquet files under {data_dir}")
        self.config = config
        self.speaker_map = speaker_map
        self.keep_indices = keep_indices
        self.frontend = frontend or Frontend(add_blank=config.add_blank)
        self.seed = seed
        self.infinite = infinite

    def _shard_plan(self) -> list[tuple[Path, int]]:
        """(file, row_group) pairs this worker is responsible for."""
        import pyarrow.parquet as pq

        plan: list[tuple[Path, int]] = []
        for path in self.files:
            n_groups = pq.ParquetFile(path).num_row_groups
            plan.extend((path, g) for g in range(n_groups))

        info = get_worker_info()
        if info is not None:
            plan = plan[info.id :: info.num_workers]
        return plan

    def _row_offset(self, path: Path) -> int:
        """Global index of a shard's first row.

        Shards are read in sorted order and the corpus is a single split, so
        a cumulative row count reproduces the same global indices the audit
        and speaker jobs assigned.
        """
        if not hasattr(self, "_offsets"):
            import pyarrow.parquet as pq

            offsets, total = {}, 0
            for p in self.files:
                offsets[p] = total
                total += pq.ParquetFile(p).metadata.num_rows
            self._offsets = offsets
        return self._offsets[path]

    def _examples(self, epoch: int):
        import pyarrow.parquet as pq

        cfg = self.config
        rng = random.Random(self.seed + epoch * 7919)
        plan = self._shard_plan()
        rng.shuffle(plan)

        readers: dict[Path, "pq.ParquetFile"] = {}
        buffer: list = []

        for path, group in plan:
            pf = readers.get(path)
            if pf is None:
                pf = readers[path] = pq.ParquetFile(path)
            base = self._row_offset(path)

            names = set(pf.schema_arrow.names)
            cols = [c for c in ("sentence", "audio") if c in names]
            table = pf.read_row_group(group, columns=cols)

            # Row index within the shard for this row group.
            start = sum(pf.metadata.row_group(g).num_rows for g in range(group))
            d = table.to_pydict()

            for i, (text, blob) in enumerate(zip(d["sentence"], d["audio"])):
                idx = base + start + i
                if self.keep_indices is not None and idx not in self.keep_indices:
                    continue
                sid = 0
                if self.speaker_map is not None:
                    if idx not in self.speaker_map:
                        continue
                    sid = self.speaker_map[idx]

                if isinstance(blob, dict):
                    blob = blob.get("bytes")
                if not blob or not text:
                    continue
                if len(text) > cfg.max_text_length:
                    continue

                audio = _read_audio(blob)
                if audio is None:
                    continue
                seconds = len(audio) / cfg.sampling_rate
                if not (cfg.min_audio_seconds <= seconds <= cfg.max_audio_seconds):
                    continue

                try:
                    ids = self.frontend(text).ids
                except Exception:
                    continue
                if len(ids) < 4:
                    continue

                buffer.append((ids, audio, sid, idx))
                if len(buffer) >= cfg.shuffle_buffer:
                    j = rng.randrange(len(buffer))
                    buffer[j], buffer[-1] = buffer[-1], buffer[j]
                    yield buffer.pop()

        rng.shuffle(buffer)
        yield from buffer

    def __iter__(self):
        epoch = 0
        while True:
            yield from self._examples(epoch)
            epoch += 1
            if not self.infinite:
                return


def collate(batch, config: DataConfig):
    """Pad a list of examples into training tensors.

    Waveforms are trimmed to a whole number of hop lengths so that the
    spectrogram frame count and the waveform sample count stay in exact
    correspondence; an off-by-one here shows up much later as a persistent
    half-frame misalignment that is very hard to attribute.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    hop = config.hop_length
    max_text = max(len(b[0]) for b in batch)
    max_wav = max(len(b[1]) // hop * hop for b in batch)
    max_frames = max_wav // hop

    n = len(batch)
    text = torch.zeros(n, max_text, dtype=torch.long)
    text_lengths = torch.zeros(n, dtype=torch.long)
    wav = torch.zeros(n, 1, max_wav)
    wav_lengths = torch.zeros(n, dtype=torch.long)

    # A speaker is either a cluster id or a continuous vector, depending on
    # which conditioning path the model was built with. The batch carries
    # whichever the dataset produced and the model validates the shape.
    first = batch[0][2]
    vectors = isinstance(first, np.ndarray)
    sid = (torch.zeros(n, len(first)) if vectors else torch.zeros(n, dtype=torch.long))

    for i, (ids, audio, speaker, _) in enumerate(batch):
        text[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        text_lengths[i] = len(ids)
        usable = len(audio) // hop * hop
        wav[i, 0, :usable] = torch.from_numpy(audio[:usable])
        wav_lengths[i] = usable
        if vectors:
            sid[i] = torch.from_numpy(np.asarray(speaker, dtype=np.float32))
        else:
            sid[i] = speaker

    spec = spectrogram_torch(
        wav.squeeze(1), config.filter_length, config.hop_length, config.win_length
    )
    spec_lengths = (wav_lengths // hop).clamp(max=spec.shape[-1])

    return {
        "text": text,
        "text_lengths": text_lengths,
        "spec": spec,
        "spec_lengths": spec_lengths,
        "wav": wav,
        "wav_lengths": wav_lengths,
        "sid": sid,
    }


class PackedSpeechDataset(IterableDataset):
    """Reads the compact dataset produced by `jobs/pack_subset.py`.

    That format carries everything a training step needs in one row - FLAC
    audio, transcript, and the speaker vector - so nothing has to be joined
    against a second source at load time. It exists for environments that
    cannot mount the corpus repository, Colab in particular, where the whole
    split has to arrive over the network before training can start.
    """

    def __init__(
        self,
        data_dir: str | Path,
        config: DataConfig,
        frontend: Frontend | None = None,
        seed: int = 1234,
        infinite: bool = True,
    ):
        super().__init__()
        self.files = sorted(Path(data_dir).rglob("*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"no packed shards under {data_dir}")
        self.config = config
        self.frontend = frontend or Frontend(add_blank=config.add_blank)
        self.seed = seed
        self.infinite = infinite

    def _decode(self, blob: bytes) -> np.ndarray | None:
        """Decode a packed audio blob to mono float32.

        Prefers soundfile, which handles the FLAC the packer writes, and
        falls back to the stdlib WAV reader. The fallback is not just for
        convenience: it means a training environment without libsndfile can
        still read a PCM-packed dataset instead of failing silently with an
        empty loader.
        """
        try:
            import soundfile as sf

            audio, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
            if sr != self.config.sampling_rate:
                return None
            return audio.mean(axis=1)
        except ImportError:
            pass
        except Exception:
            return None
        return _read_wav(blob)

    def _examples(self, epoch: int):
        import pyarrow.parquet as pq

        cfg = self.config
        rng = random.Random(self.seed + epoch * 7919)

        plan: list[tuple[Path, int]] = []
        for path in self.files:
            plan.extend((path, g) for g in range(pq.ParquetFile(path).num_row_groups))

        info = get_worker_info()
        if info is not None:
            plan = plan[info.id :: info.num_workers]
        rng.shuffle(plan)

        buffer: list = []
        readers: dict[Path, object] = {}

        for path, group in plan:
            pf = readers.get(path)
            if pf is None:
                pf = readers[path] = pq.ParquetFile(path)
            d = pf.read_row_group(group).to_pydict()

            for i in range(len(d["idx"])):
                audio = self._decode(d["audio"][i])
                if audio is None:
                    continue
                seconds = len(audio) / cfg.sampling_rate
                if not (cfg.min_audio_seconds <= seconds <= cfg.max_audio_seconds):
                    continue

                text = d["text"][i]
                if not text or len(text) > cfg.max_text_length:
                    continue
                try:
                    ids = self.frontend(text).ids
                except Exception:
                    continue
                if len(ids) < 4:
                    continue

                vector = np.frombuffer(d["speaker_vector"][i], dtype=np.float16).astype(np.float32)
                buffer.append((ids, audio, vector, int(d["idx"][i])))

                if len(buffer) >= cfg.shuffle_buffer:
                    j = rng.randrange(len(buffer))
                    buffer[j], buffer[-1] = buffer[-1], buffer[j]
                    yield buffer.pop()

        rng.shuffle(buffer)
        yield from buffer

    def __iter__(self):
        epoch = 0
        while True:
            yield from self._examples(epoch)
            epoch += 1
            if not self.infinite:
                return


def load_speaker_map(path: str | Path) -> dict[int, int]:
    """Read the speaker assignment produced by the discovery job."""
    path = Path(path)
    if path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {int(k): int(v) for k, v in raw.items()}
    labels = np.load(path)
    return {int(i): int(v) for i, v in enumerate(labels)}
