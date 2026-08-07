"""Adversarial critics. Training only - none of this is exported.

Two complementary families are used:

`MultiPeriodDiscriminator` reshapes the waveform into 2-D by a prime
period and convolves over it. Speech is quasi-periodic at the pitch rate,
so viewing it at periods 2/3/5/7/11 exposes phase and periodicity errors
that a purely time-domain critic averages away.

`MultiResolutionDiscriminator` works on STFT magnitudes at three window
sizes. It catches spectral artefacts - the metallic ring and the smeared
high band - that a time-domain critic is comparatively blind to. It
matters more here than in a standard HiFi-GAN setup, because this decoder
predicts spectra directly.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import Conv1d, Conv2d
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

from .commons import get_padding
from .modules import LRELU_SLOPE


class DiscriminatorP(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3, use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class DiscriminatorS(nn.Module):
    """Plain time-domain critic on the raw waveform."""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm: bool = False, periods: tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs += [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
    """Critic over the STFT magnitude at one resolution."""

    def __init__(self, resolution: tuple[int, int, int], use_spectral_norm: bool = False):
        super().__init__()
        self.n_fft, self.hop_length, self.win_length = resolution
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (3, 9), padding=(1, 4))),
            norm_f(Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(Conv2d(32, 1, (3, 3), padding=(1, 1)))
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

    def _spectrogram(self, x):
        x = x.squeeze(1)
        x = F.pad(
            x.unsqueeze(1),
            (int((self.n_fft - self.hop_length) / 2), int((self.n_fft - self.hop_length) / 2)),
            mode="reflect",
        ).squeeze(1)
        spec = torch.stft(
            x, self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, center=False, return_complex=True,
        )
        return torch.abs(spec)

    def forward(self, x):
        fmap = []
        x = self._spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        resolutions: tuple[tuple[int, int, int], ...] = ((512, 128, 512), (256, 64, 256), (128, 32, 128)),
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(r, use_spectral_norm=use_spectral_norm) for r in resolutions]
        )

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class CombinedDiscriminator(nn.Module):
    """Runs the period and resolution critics and concatenates their outputs."""

    def __init__(
        self,
        use_spectral_norm: bool = False,
        periods: tuple[int, ...] = (2, 3, 5, 7, 11),
        resolutions: tuple[tuple[int, int, int], ...] = ((512, 128, 512), (256, 64, 256), (128, 32, 128)),
    ):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(use_spectral_norm, periods)
        self.mrd = MultiResolutionDiscriminator(resolutions, use_spectral_norm)

    def forward(self, y, y_hat):
        a = self.mpd(y, y_hat)
        b = self.mrd(y, y_hat)
        return tuple(x + y_ for x, y_ in zip(a, b))
