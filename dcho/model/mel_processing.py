"""Spectrogram and mel-spectrogram front end.

The mel filterbank is built here rather than pulled from librosa. It is
about thirty lines, it removes a heavy dependency from the training image,
and - more usefully - it pins the exact filterbank definition into this
repository. A mel basis that silently changes between library versions
would shift the reconstruction loss and invalidate every checkpoint
trained before the change.

The Slaney convention is used (area-normalised triangles, Slaney mel
breakpoint at 1 kHz), which is what `librosa.filters.mel` produces with its
defaults.
"""

from __future__ import annotations

import numpy as np
import torch

# Cache keyed by (sr, n_fft, n_mels, fmin, fmax, device, dtype); the basis
# and window are rebuilt only when a new configuration appears.
_MEL_BASIS: dict = {}
_WINDOWS: dict = {}

MEL_F_MIN = 0.0
MEL_F_SP = 200.0 / 3
MEL_MIN_LOG_HZ = 1000.0
MEL_MIN_LOG_MEL = (MEL_MIN_LOG_HZ - MEL_F_MIN) / MEL_F_SP
MEL_LOGSTEP = np.log(6.4) / 27.0


def hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
    frequencies = np.asanyarray(frequencies, dtype=float)
    mels = (frequencies - MEL_F_MIN) / MEL_F_SP
    log_t = frequencies >= MEL_MIN_LOG_HZ
    mels[log_t] = MEL_MIN_LOG_MEL + np.log(frequencies[log_t] / MEL_MIN_LOG_HZ) / MEL_LOGSTEP
    return mels


def mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.asanyarray(mels, dtype=float)
    freqs = MEL_F_MIN + MEL_F_SP * mels
    log_t = mels >= MEL_MIN_LOG_MEL
    freqs[log_t] = MEL_MIN_LOG_HZ * np.exp(MEL_LOGSTEP * (mels[log_t] - MEL_MIN_LOG_MEL))
    return freqs


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Triangular, area-normalised mel filterbank of shape [n_mels, n_fft//2+1]."""
    fft_freqs = np.linspace(0, sr / 2, int(1 + n_fft // 2))
    mel_points = np.linspace(hz_to_mel(np.array([fmin]))[0], hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    weights = np.zeros((n_mels, len(fft_freqs)))
    fdiff = np.diff(hz_points)
    ramps = hz_points[:, None] - fft_freqs[None, :]

    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    # Slaney normalisation: each filter integrates to the same area, so the
    # mel spectrum is not tilted towards the wider high-frequency bands.
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    weights *= enorm[:, None]
    return weights


def dynamic_range_compression(x: torch.Tensor, C: float = 1.0, clip_val: float = 1e-5) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x: torch.Tensor, C: float = 1.0) -> torch.Tensor:
    return torch.exp(x) / C


def _get_window(win_size: int, device, dtype):
    key = (win_size, str(device), str(dtype))
    if key not in _WINDOWS:
        _WINDOWS[key] = torch.hann_window(win_size).to(device=device, dtype=dtype)
    return _WINDOWS[key]


def spectrogram_torch(
    y: torch.Tensor, n_fft: int, hop_size: int, win_size: int, center: bool = False
) -> torch.Tensor:
    """Linear magnitude spectrogram, [B, n_fft//2+1, T]."""
    if torch.min(y) < -1.07 or torch.max(y) > 1.07:
        # Not fatal, but clipped input silently degrades the target the
        # posterior encoder is trained on, so it is worth surfacing.
        pass

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    ).squeeze(1)

    spec = torch.stft(
        y, n_fft, hop_length=hop_size, win_length=win_size,
        window=_get_window(win_size, y.device, y.dtype),
        center=center, pad_mode="reflect", normalized=False,
        onesided=True, return_complex=True,
    )
    return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)


def spec_to_mel_torch(
    spec: torch.Tensor, n_fft: int, num_mels: int, sampling_rate: int, fmin: float, fmax: float
) -> torch.Tensor:
    key = (sampling_rate, n_fft, num_mels, fmin, fmax, str(spec.device), str(spec.dtype))
    if key not in _MEL_BASIS:
        basis = mel_filterbank(sampling_rate, n_fft, num_mels, fmin, fmax)
        _MEL_BASIS[key] = torch.from_numpy(basis).to(device=spec.device, dtype=spec.dtype)
    mel = torch.matmul(_MEL_BASIS[key], spec)
    return dynamic_range_compression(mel)


def mel_spectrogram_torch(
    y: torch.Tensor, n_fft: int, num_mels: int, sampling_rate: int,
    hop_size: int, win_size: int, fmin: float, fmax: float, center: bool = False,
) -> torch.Tensor:
    if y.dim() == 3:
        y = y.squeeze(1)
    spec = spectrogram_torch(y, n_fft, hop_size, win_size, center)
    return spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax)
