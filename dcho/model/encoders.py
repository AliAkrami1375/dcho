"""Text encoder, posterior encoder and the normalising flow between them."""

from __future__ import annotations

import math

import torch
from torch import nn

from . import attentions, commons, modules


class TextEncoder(nn.Module):
    """Phonemes -> prior distribution over the latent, plus a context vector.

    Returns both the raw encoder output `x` (which conditions the duration
    predictor) and the projected prior statistics `m_p`, `logs_p`.
    """

    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor):
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class PosteriorEncoder(nn.Module):
    """Linear spectrogram -> latent. Training only; dropped at export."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class ResidualCouplingTransformersLayer(nn.Module):
    """Coupling layer with transformer blocks around the WaveNet transform.

    This is the VITS2 change to the flow. The WaveNet stack has a wide but
    still local receptive field; the two attention blocks let the coupling
    condition on the whole utterance. In practice it shows up as steadier
    intonation across long sentences, which is exactly the failure mode a
    purely convolutional flow has.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0.0,
        gin_channels: int = 0,
        mean_only: bool = False,
    ):
        assert channels % 2 == 0, "coupling layer needs an even channel count"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre_transformer = attentions.Encoder(
            self.half_channels, self.half_channels, n_heads=2, n_layers=1,
            kernel_size=kernel_size, p_dropout=p_dropout, window_size=None,
        )
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels, kernel_size, dilation_rate, n_layers,
            p_dropout=p_dropout, gin_channels=gin_channels,
        )
        self.post_transformer = attentions.Encoder(
            hidden_channels, hidden_channels, n_heads=2, n_layers=1,
            kernel_size=kernel_size, p_dropout=p_dropout, window_size=None,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        x0_ = self.pre_transformer(x0 * x_mask, x_mask)
        x0_ = x0_ + x0

        h = self.pre(x0_) * x_mask
        h = self.enc(h, x_mask, g=g)
        h = h + self.post_transformer(h, x_mask)

        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        x1 = (x1 - m) * torch.exp(-logs) * x_mask
        return torch.cat([x0, x1], 1)


class ResidualCouplingBlock(nn.Module):
    """Stack of coupling layers with channel flips in between.

    Only the first layer carries the transformer blocks. Putting them in
    every layer costs noticeably more for a benefit that did not survive
    scrutiny in the VITS2 ablations.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
        use_transformer_in_first: bool = True,
    ):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            if i == 0 and use_transformer_in_first:
                layer = ResidualCouplingTransformersLayer(
                    channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                    gin_channels=gin_channels, mean_only=True,
                )
            else:
                layer = modules.ResidualCouplingLayer(
                    channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                    gin_channels=gin_channels, mean_only=True,
                )
            self.flows.append(layer)
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x
