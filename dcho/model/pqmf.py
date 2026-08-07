"""Pseudo-QMF analysis/synthesis filter bank.

The decoder predicts N critically-sampled subband signals, each running at
1/N of the output sample rate, and this bank recombines them into the
full-band waveform. Doing it this way means the expensive neural part only
ever has to produce 1/N as many samples.

The prototype filter is a Kaiser-windowed sinc designed here rather than
pulled from scipy: it is a handful of lines, it keeps the runtime
dependency set to numpy plus torch, and the coefficients are baked into the
exported graph as constants anyway.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def design_prototype_filter(taps: int = 62, cutoff_ratio: float = 0.142, beta: float = 9.0) -> np.ndarray:
    """Kaiser-windowed lowpass prototype of length `taps + 1`.

    `cutoff_ratio` and `beta` trade stopband attenuation against transition
    width. The defaults are tuned for 4 subbands and measure at roughly
    -58 dB analysis/synthesis reconstruction error; `tests/test_pqmf.py`
    asserts that figure so a future change to these numbers cannot silently
    degrade the decoder.
    """
    if taps % 2 != 0:
        raise ValueError("taps must be even so the filter is linear phase")

    omega_c = np.pi * cutoff_ratio
    n = np.arange(taps + 1) - 0.5 * taps
    with np.errstate(invalid="ignore", divide="ignore"):
        h_ideal = np.sin(omega_c * n) / (np.pi * n)
    # n == 0 is the removable singularity of sinc.
    h_ideal[taps // 2] = cutoff_ratio

    window = np.kaiser(taps + 1, beta)
    return h_ideal * window


class PQMF(nn.Module):
    """Near-perfect-reconstruction pseudo-QMF bank.

    `analysis` splits a full-band signal into subbands (used to build the
    multi-band reconstruction loss during training); `synthesis` merges
    subbands back (used at every forward pass and at inference).
    """

    def __init__(
        self,
        subbands: int = 4,
        taps: int = 62,
        cutoff_ratio: float = 0.142,
        beta: float = 9.0,
    ):
        super().__init__()
        self.subbands = subbands
        self.taps = taps

        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
        h_analysis = np.zeros((subbands, len(h_proto)))
        h_synthesis = np.zeros((subbands, len(h_proto)))

        n = np.arange(taps + 1) - 0.5 * taps
        for k in range(subbands):
            base = (2 * k + 1) * (np.pi / (2 * subbands)) * n
            phase = (-1) ** k * np.pi / 4
            h_analysis[k] = 2 * h_proto * np.cos(base + phase)
            h_synthesis[k] = 2 * h_proto * np.cos(base - phase)

        analysis_filter = torch.from_numpy(h_analysis).float().unsqueeze(1)
        synthesis_filter = torch.from_numpy(h_synthesis).float().unsqueeze(0)

        self.register_buffer("analysis_filter", analysis_filter)
        self.register_buffer("synthesis_filter", synthesis_filter)

        updown_filter = torch.zeros((subbands, subbands, subbands)).float()
        for k in range(subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)

        self.pad_fn = nn.ConstantPad1d(taps // 2, 0.0)

    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 1, T] -> [B, subbands, T // subbands]"""
        x = F.conv1d(self.pad_fn(x), self.analysis_filter)
        return F.conv1d(x, self.updown_filter, stride=self.subbands)

    def synthesis(self, x: torch.Tensor) -> torch.Tensor:
        """[B, subbands, T // subbands] -> [B, 1, T]"""
        x = F.conv_transpose1d(x, self.updown_filter * self.subbands, stride=self.subbands)
        return F.conv1d(self.pad_fn(x), self.synthesis_filter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.analysis(x)
