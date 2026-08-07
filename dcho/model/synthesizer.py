"""The dcho acoustic model.

A single-stage, non-autoregressive text-to-waveform network: a conditional
VAE whose prior is conditioned on phonemes through a normalising flow, with
a multi-band iSTFT decoder and adversarial training.

Single-stage matters for quality as much as for speed. A separate acoustic
model and vocoder are trained against different objectives and meet at an
intermediate representation neither of them owns, and the mismatch is
audible as the metallic ring that two-stage systems are known for. Here
the decoder is trained on the exact latents it will be given at inference.

At export time `enc_q` and every discriminator are dropped: the inference
graph is the text encoder, the duration predictor, the inverse flow and
the decoder.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from . import commons
from .decoder import MultiBandiSTFTGenerator
from .duration import StochasticDurationPredictor
from .encoders import PosteriorEncoder, ResidualCouplingBlock, TextEncoder
from .monotonic_align import maximum_path


class DeterministicDurationPredictor(nn.Module):
    """Fallback duration head, used only by the micro config for speed."""

    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()
        from .modules import LayerNorm

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            x = x + self.cond(torch.detach(g))
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        return self.proj(x * x_mask) * x_mask


class Synthesizer(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        gen_istft_n_fft: int,
        gen_istft_hop_size: int,
        subbands: int,
        n_flow_layers: int = 4,
        n_speakers: int = 0,
        gin_channels: int = 0,
        use_sdp: bool = True,
        speaker_embed_dim: int = 0,
        **kwargs,
    ):
        super().__init__()

        # Values the training loop reads back for its health checks. Kept
        # here rather than threaded through the return tuple, so adding a
        # diagnostic does not change the forward signature.
        self.diagnostics: dict[str, float] = {}

        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.use_sdp = use_sdp
        self.inter_channels = inter_channels

        self.enc_p = TextEncoder(
            n_vocab, inter_channels, hidden_channels, filter_channels,
            n_heads, n_layers, kernel_size, p_dropout,
        )
        self.dec = MultiBandiSTFTGenerator(
            inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
            upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
            gen_istft_n_fft, gen_istft_hop_size, subbands, gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4,
            n_flows=n_flow_layers, gin_channels=gin_channels,
        )

        if use_sdp:
            self.dp = StochasticDurationPredictor(
                hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
            )
        else:
            self.dp = DeterministicDurationPredictor(
                hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
            )

        # Two ways to condition on a voice.
        #
        # A lookup table over discovered speaker ids is the usual choice and
        # is what ships when clustering produces clean, well-populated
        # groups. It gives the model a free parameter per voice to refine.
        #
        # A projection from an external speaker embedding is the fallback
        # when clustering is unreliable - and whether it is reliable is a
        # property of the corpus and the encoder, not something to assume in
        # advance. It only requires the embedding to carry some voice
        # information, not to be cleanly separable, and it generalises to a
        # voice never seen in training.
        self.speaker_embed_dim = speaker_embed_dim
        if speaker_embed_dim:
            self.proj_g = nn.Sequential(
                nn.Linear(speaker_embed_dim, gin_channels),
                nn.LayerNorm(gin_channels),
            )
        elif n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    # -- helpers ----------------------------------------------------------

    def _speaker_embedding(self, sid):
        if self.speaker_embed_dim:
            if sid is None:
                raise ValueError("this model conditions on a speaker vector; sid is required")
            if sid.dim() != 2 or sid.shape[-1] != self.speaker_embed_dim:
                raise ValueError(
                    f"expected a speaker vector of shape [B, {self.speaker_embed_dim}], "
                    f"got {tuple(sid.shape)}"
                )
            return self.proj_g(sid).unsqueeze(-1)
        if self.n_speakers > 1:
            if sid is None:
                raise ValueError("this is a multi-speaker model; sid is required")
            return self.emb_g(sid).unsqueeze(-1)
        return None

    def _align(self, z_p, m_p, logs_p, x_mask, y_mask, mas_noise_scale: float = 0.0):
        """Monotonic alignment between phonemes and frames.

        `mas_noise_scale` implements the VITS2 noise-scaled search. Early in
        training the likelihood surface is nearly flat and the search can
        latch onto a degenerate path it never escapes; a little noise, decayed
        to zero over the first epochs, keeps it exploring long enough to find
        the real alignment.
        """
        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            if mas_noise_scale > 0.0:
                epsilon = torch.std(neg_cent) * torch.randn_like(neg_cent) * mas_noise_scale
                neg_cent = neg_cent + epsilon

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)

            # Alignment confidence, recorded before the search hardens it.
            #
            # The search returns a one-hot path by construction, so its own
            # entropy is identically zero and says nothing. What is
            # diagnostic is the *soft* posterior it chooses from: for each
            # frame, how sharply the model prefers one phoneme over the
            # others. Near 1 means it cannot tell them apart; falling
            # towards 0 is the earliest reliable sign a run will work.
            valid = attn_mask.squeeze(1).bool()
            logp = neg_cent.masked_fill(~valid, -1e4).log_softmax(dim=-1)
            probs = logp.exp()
            frame_entropy = -(probs * logp).sum(dim=-1)
            n_text = valid.any(dim=1).sum(dim=-1).clamp(min=2).float()
            norm = torch.log(n_text).unsqueeze(-1)
            frame_valid = valid.any(dim=-1).float()
            entropy = ((frame_entropy / norm) * frame_valid).sum() / frame_valid.sum().clamp(min=1)
            self.diagnostics["alignment_entropy"] = float(entropy)

            # neg_cent is [B, T_spec, T_text]; the search wants text first.
            path = maximum_path(neg_cent.transpose(1, 2), attn_mask.squeeze(1).transpose(1, 2))
            return path.transpose(1, 2).unsqueeze(1).detach()

    # -- training ---------------------------------------------------------

    def forward(self, x, x_lengths, y, y_lengths, sid=None, mas_noise_scale: float = 0.0):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        g = self._speaker_embedding(sid)

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        attn = self._align(z_p, m_p, logs_p, x_mask, y_mask, mas_noise_scale)

        w = attn.sum(2)
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
            logw = torch.log(w + 1e-6) * x_mask
            logw_hat = logw  # the SDP has no closed-form prediction to expose
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)
            logw_hat = logw
            logw = logw_

        # Expand the prior along the discovered alignment.
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o, o_subbands = self.dec(z_slice, g=g)

        return (
            o, o_subbands, l_length, attn, ids_slice, x_mask, y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            (logw, logw_hat),
        )

    # -- inference --------------------------------------------------------

    @torch.no_grad()
    def infer(
        self,
        x,
        x_lengths,
        sid=None,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_scale_w: float = 0.8,
        max_len: int | None = None,
    ):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        g = self._speaker_embedding(sid)

        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)

        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o, _ = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    # -- deployment -------------------------------------------------------

    def prepare_for_inference(self):
        """Strip training-only weights and fold away weight normalisation."""
        self.dec.remove_weight_norm()
        if hasattr(self, "enc_q"):
            del self.enc_q
        self.eval()
        return self

    def n_parameters(self, inference_only: bool = False) -> int:
        skip = ("enc_q.",) if inference_only else ()
        return sum(
            p.numel()
            for name, p in self.named_parameters()
            if not any(name.startswith(s) for s in skip)
        )
