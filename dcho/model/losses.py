"""Training objectives.

The generator is trained against five terms:

  mel         L1 between the mel of the generated and target waveform. This
              is what actually drives intelligibility; the adversarial terms
              only sharpen what it has already established.
  kl          keeps the flow-transformed posterior close to the phoneme
              prior, which is what forces the text to control the latent.
  duration    negative log-likelihood under the duration flow.
  adversarial least-squares GAN loss against every critic.
  feature     L1 between critic activations on real and generated audio.
  subband     multi-resolution STFT loss on the decoder's subband outputs.

The subband term is specific to this decoder. Without it the PQMF bands
are only ever supervised through their sum, and the network is free to put
compensating errors in neighbouring bands that cancel in the full-band
signal but produce aliasing the moment the bands are recombined at a
different phase.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def feature_loss(fmap_r, fmap_g) -> torch.Tensor:
    loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """Least-squares GAN: real -> 1, generated -> 0."""
    loss = 0.0
    r_losses, g_losses = [], []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0.0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask) -> torch.Tensor:
    """KL(q(z|spec) || p(z|text)) evaluated at the sampled z."""
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    return kl / torch.sum(z_mask)


class STFTLoss(torch.nn.Module):
    """Spectral convergence plus log-magnitude L1 at one resolution."""

    def __init__(self, n_fft: int = 1024, shift: int = 120, win_length: int = 600):
        super().__init__()
        self.n_fft = n_fft
        self.shift = shift
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def _magnitude(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x, self.n_fft, self.shift, self.win_length,
            window=self.window.to(x.device), return_complex=True,
        )
        return torch.sqrt(torch.clamp(spec.real**2 + spec.imag**2, min=1e-7))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x_mag = self._magnitude(x)
        y_mag = self._magnitude(y)
        sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
        mag_loss = F.l1_loss(torch.log(x_mag), torch.log(y_mag))
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Sum of `STFTLoss` at several resolutions.

    Three resolutions is the usual choice: a short window resolves
    transients, a long one resolves harmonic structure, and neither alone
    is sufficient.
    """

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (384, 683, 171),
        hop_sizes: tuple[int, ...] = (30, 60, 10),
        win_lengths: tuple[int, ...] = (150, 300, 60),
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList(
            [STFTLoss(f, s, w) for f, s, w in zip(fft_sizes, hop_sizes, win_lengths)]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """x, y: [B, S, T] subband signals, or [B, T] full-band."""
        if x.dim() == 3:
            b, s, t = x.shape
            x = x.reshape(b * s, t)
            y = y.reshape(b * s, t)

        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc, mag = f(x, y)
            sc_loss = sc_loss + sc
            mag_loss = mag_loss + mag
        n = len(self.stft_losses)
        return sc_loss / n, mag_loss / n


def duration_discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0.0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        loss += torch.mean((1 - dr) ** 2) + torch.mean(dg**2)
    return loss


def duration_generator_loss(disc_outputs):
    loss = 0.0
    for dg in disc_outputs:
        loss += torch.mean((1 - dg.float()) ** 2)
    return loss
