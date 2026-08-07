"""Decoder cost: multi-band iSTFT against a conventional HiFi-GAN stack.

The central architectural claim of this project is that replacing the
transposed-convolution upsampling stack with a short network prefix plus an
inverse STFT and a PQMF bank removes most of the decoder's inference cost
without changing what the decoder has to represent.

That claim is worth measuring rather than asserting, so this builds both
decoders at the same input width and output rate and times them on a CPU.
The HiFi-GAN reference is the V1 configuration adapted to 16 kHz with hop
256: upsampling [8, 8, 2, 2], initial width 512, the standard three
multi-receptive-field blocks.

Run directly to print the table; under unittest it asserts the ordering
that the architecture depends on.
"""

from __future__ import annotations

import time
import unittest

import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from dcho.model.commons import init_weights
from dcho.model.decoder import MultiBandiSTFTGenerator
from dcho.model.modules import LRELU_SLOPE, ResBlock1


class HiFiGANGenerator(nn.Module):
    """Reference decoder: upsampling entirely by transposed convolution."""

    def __init__(
        self,
        initial_channel: int = 192,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        upsample_rates=(8, 8, 2, 2),
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes=(16, 16, 4, 4),
        gin_channels: int = 256,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, 3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k, u, padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, 3, bias=False))
        self.ups.apply(init_weights)
        if gin_channels:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
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
        return torch.tanh(self.conv_post(x))

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_post)


def build_pair(initial_channel: int = 192, gin_channels: int = 256):
    mb = MultiBandiSTFTGenerator(
        initial_channel=initial_channel, resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[4, 4], upsample_initial_channel=256,
        upsample_kernel_sizes=[16, 16],
        gen_istft_n_fft=16, gen_istft_hop_size=4, subbands=4,
        gin_channels=gin_channels,
    ).eval()
    hifi = HiFiGANGenerator(initial_channel=initial_channel, gin_channels=gin_channels).eval()
    mb.remove_weight_norm()
    hifi.remove_weight_norm()
    return mb, hifi


def benchmark(frames: int = 625, runs: int = 5, threads: int = 1):
    """625 frames at hop 256 and 16 kHz is 10 seconds of speech."""
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    mb, hifi = build_pair()
    z = torch.randn(1, 192, frames)
    g = torch.randn(1, 256, 1)

    out = {}
    for name, model, call in (
        ("mb_istft", mb, lambda: mb(z, g)[0]),
        ("hifigan", hifi, lambda: hifi(z, g)),
    ):
        with torch.no_grad():
            call()
            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                y = call()
                times.append(time.perf_counter() - t0)
        times.sort()
        out[name] = {
            "params": sum(p.numel() for p in model.parameters()),
            "seconds": times[len(times) // 2],
            "samples": int(y.shape[-1]),
        }
    return out


class TestDecoderCost(unittest.TestCase):
    def test_multiband_decoder_is_smaller_and_faster(self):
        r = benchmark(frames=200, runs=3, threads=1)
        mb, hifi = r["mb_istft"], r["hifigan"]
        self.assertEqual(mb["samples"], hifi["samples"], "both must produce the same rate")
        self.assertLess(mb["params"], hifi["params"])
        self.assertLess(mb["seconds"], hifi["seconds"])

    def test_speedup_is_substantial(self):
        """A marginal win would not justify the extra machinery."""
        r = benchmark(frames=200, runs=3, threads=1)
        speedup = r["hifigan"]["seconds"] / r["mb_istft"]["seconds"]
        self.assertGreater(speedup, 2.0, f"speedup was only {speedup:.2f}x")


if __name__ == "__main__":
    for threads in (1, 4):
        r = benchmark(threads=threads)
        mb, hifi = r["mb_istft"], r["hifigan"]
        audio_s = mb["samples"] / 16000
        print(f"\n=== {threads} thread(s), {audio_s:.1f}s of 16 kHz audio ===")
        print(f"{'decoder':<12} {'params':>10} {'seconds':>9} {'RTF':>8}")
        for name, v in (("MB-iSTFT", mb), ("HiFi-GAN", hifi)):
            print(f"{name:<12} {v['params']/1e6:>9.2f}M {v['seconds']:>9.3f} {v['seconds']/audio_s:>8.4f}")
        print(f"{'ratio':<12} {hifi['params']/mb['params']:>9.2f}x {hifi['seconds']/mb['seconds']:>8.2f}x")
