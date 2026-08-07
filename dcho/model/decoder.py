"""Multi-band iSTFT decoder.

This is the component that makes the model viable on a CPU. A conventional
HiFi-GAN decoder reaches the output sample rate purely through transposed
convolutions, and in a 16 kHz / hop 256 configuration that stack accounts
for the large majority of inference time.

Here the network only upsamples by 16x. It then predicts, for each of
`subbands` frequency bands, a short-time magnitude and phase; an inverse
STFT turns those into subband waveforms, and a PQMF bank merges the
subbands into the full-band signal. The remaining 16x of upsampling is
therefore done by two parameter-free linear operations instead of by
convolution.

Rate budget for the shipped configuration:

    network upsampling      4 x 4 = 16
    inverse STFT hop                4
    PQMF subbands                   4
    ------------------------------------
    total                          256  = hop_length
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from .commons import init_weights
from .istft import InverseSTFT
from .modules import LRELU_SLOPE, ResBlock1, ResBlock2
from .pqmf import PQMF


class MultiBandiSTFTGenerator(nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        gen_istft_n_fft: int,
        gen_istft_hop_size: int,
        subbands: int,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.subbands = subbands
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size
        self.n_freq = gen_istft_n_fft // 2 + 1

        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
        resblock_cls = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(resblock_cls(ch, k, d))

        # Two output planes per subband: log-magnitude and phase.
        self.post_channels = subbands * self.n_freq * 2
        self.conv_post = weight_norm(Conv1d(ch, self.post_channels, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        self.istft = InverseSTFT(n_fft=gen_istft_n_fft, hop_length=gen_istft_hop_size)
        self.pqmf = PQMF(subbands=subbands)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None):
        """x: [B, initial_channel, T]  ->  (waveform [B, 1, T*hop], subbands [B, S, T*hop/S])"""
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                out = self.resblocks[i * self.num_kernels + j](x)
                xs = out if xs is None else xs + out
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)

        b, _, t = x.shape
        x = x.view(b, self.subbands, 2 * self.n_freq, t)

        log_magnitude = x[:, :, : self.n_freq, :]
        phase = x[:, :, self.n_freq :, :]

        # Clamping the magnitude keeps a diverging generator from producing
        # infinities that would poison the discriminator in the same step.
        magnitude = torch.exp(log_magnitude).clamp(max=1e2)

        magnitude = magnitude.reshape(b * self.subbands, self.n_freq, t)
        phase = phase.reshape(b * self.subbands, self.n_freq, t)

        subband_audio = self.istft(magnitude, phase)
        subband_audio = subband_audio.view(b, self.subbands, -1)

        audio = self.pqmf.synthesis(subband_audio)
        return audio, subband_audio

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_post)

    @property
    def total_upsample(self) -> int:
        """Samples of output produced per input frame."""
        rate = 1
        for m in self.ups:
            rate *= m.stride[0]
        return rate * self.gen_istft_hop_size * self.subbands
