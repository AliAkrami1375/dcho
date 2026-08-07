"""Inverse STFT implemented as a matmul plus overlap-add.

`torch.istft` does not export to ONNX cleanly, and the FFT operator that
would replace it is awkward across runtimes. For the window sizes this
model uses (n_fft = 16) an explicit inverse-DFT matrix is both exportable
everywhere and faster than a general FFT, because the transform collapses
to a single small dense matmul that BLAS handles well.

The matrix is a registered buffer, so it is frozen into the exported graph
as a constant and costs nothing at load time.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class InverseSTFT(nn.Module):
    """Reconstruct a waveform from magnitude and phase.

    Input is magnitude/phase rather than real/imaginary because the decoder
    predicts a log-magnitude, which keeps the dynamic range of the network
    output bounded and makes training noticeably more stable.
    """

    def __init__(self, n_fft: int = 16, hop_length: int = 4, win_length: int | None = None):
        super().__init__()
        if win_length is None:
            win_length = n_fft
        if n_fft % 2 != 0:
            raise ValueError("n_fft must be even")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_freq = n_fft // 2 + 1

        # Inverse DFT restricted to Hermitian-symmetric input.
        # x[t] = (1/N) * sum_k w_k * (Re[k]*cos(2*pi*k*t/N) - Im[k]*sin(2*pi*k*t/N))
        # with w_k = 1 for the DC and Nyquist bins and 2 elsewhere, which is
        # what folds the discarded negative frequencies back in.
        k = np.arange(self.n_freq)[:, None]
        t = np.arange(n_fft)[None, :]
        angle = 2.0 * np.pi * k * t / n_fft

        weight = np.full((self.n_freq, 1), 2.0)
        weight[0, 0] = 1.0
        weight[-1, 0] = 1.0

        cos_basis = (weight * np.cos(angle)) / n_fft
        sin_basis = (weight * np.sin(angle)) / n_fft

        self.register_buffer("cos_basis", torch.from_numpy(cos_basis).float())
        self.register_buffer("sin_basis", torch.from_numpy(sin_basis).float())

        window = torch.hann_window(win_length)
        if win_length < n_fft:
            pad = n_fft - win_length
            window = F.pad(window, (pad // 2, pad - pad // 2))
        self.register_buffer("window", window)

        # Sum of squared analysis windows at each output sample, used to
        # undo the overlap-add weighting. Precomputed for the longest run we
        # ever need and sliced at call time.
        self.register_buffer("_wsq_cache", torch.zeros(0), persistent=False)

    def _window_envelope(self, n_frames: int, device, dtype) -> torch.Tensor:
        """Overlap-add envelope of the squared window, shape [1, 1, T_out]."""
        win_sq = (self.window**2).view(1, self.n_fft, 1).expand(1, self.n_fft, n_frames)
        out_len = (n_frames - 1) * self.hop_length + self.n_fft
        env = F.fold(
            win_sq.to(device=device, dtype=dtype),
            output_size=(1, out_len),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        ).view(1, 1, out_len)
        return env

    def forward(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """magnitude, phase: [B, n_freq, T]  ->  waveform [B, 1, T_out]"""
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)

        # [B, n_freq, T] -> [B, n_fft, T]
        frames = torch.einsum("bft,fn->bnt", real, self.cos_basis) - torch.einsum(
            "bft,fn->bnt", imag, self.sin_basis
        )
        frames = frames * self.window.view(1, -1, 1)

        n_frames = frames.shape[-1]
        out_len = (n_frames - 1) * self.hop_length + self.n_fft
        signal = F.fold(
            frames,
            output_size=(1, out_len),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        ).view(frames.shape[0], 1, out_len)

        envelope = self._window_envelope(n_frames, frames.device, frames.dtype)
        signal = signal / torch.clamp_min(envelope, 1e-8)

        # Drop the leading half-window so the result lines up with a
        # `center=True` forward transform, and trim to exactly
        # n_frames * hop_length output samples.
        offset = self.n_fft // 2
        return signal[..., offset : offset + n_frames * self.hop_length]


class ForwardSTFT(nn.Module):
    """Matching forward transform, used only to build training losses."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int | None = None):
        super().__init__()
        if win_length is None:
            win_length = n_fft
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.squeeze(1)
        spec = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        magnitude = torch.abs(spec)
        phase = torch.angle(spec)
        return magnitude, phase
