"""End-to-end shape and behaviour checks for the acoustic model.

These run on CPU with tiny tensors. They are not a quality test - they
exist so that a structural mistake is caught in seconds rather than after
an hour of GPU time.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from dcho.model.discriminators import CombinedDiscriminator
from dcho.model.duration import DurationDiscriminator
from dcho.model.losses import MultiResolutionSTFTLoss, kl_loss
from dcho.model.mel_processing import mel_spectrogram_torch, spectrogram_torch
from dcho.model.synthesizer import Synthesizer
from dcho.text.phonemes import N_SYMBOLS

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def build(name: str, n_speakers: int = 4) -> tuple[Synthesizer, dict]:
    cfg = load_config(name)
    data, model = cfg["data"], cfg["model"]
    net = Synthesizer(
        n_vocab=N_SYMBOLS,
        spec_channels=data["filter_length"] // 2 + 1,
        segment_size=cfg["train"]["segment_size"] // data["hop_length"],
        n_speakers=n_speakers,
        **{k: v for k, v in model.items() if not k.startswith("_")},
    )
    return net, cfg


class TestConfigs(unittest.TestCase):
    def test_upsample_budget_matches_hop_length(self):
        """The product of every upsampling stage must equal hop_length.

        Getting this wrong produces audio at the wrong rate with no error,
        so it is asserted rather than trusted.
        """
        for name in ("base", "nano", "micro"):
            cfg = load_config(name)
            m, d = cfg["model"], cfg["data"]
            total = 1
            for r in m["upsample_rates"]:
                total *= r
            total *= m["gen_istft_hop_size"] * m["subbands"]
            self.assertEqual(
                total, d["hop_length"],
                f"{name}: upsample budget {total} != hop_length {d['hop_length']}",
            )

    def test_segment_size_divides_hop(self):
        for name in ("base", "nano", "micro"):
            cfg = load_config(name)
            self.assertEqual(cfg["train"]["segment_size"] % cfg["data"]["hop_length"], 0)

    def test_mel_fmax_is_nyquist_or_below(self):
        for name in ("base", "nano", "micro"):
            d = load_config(name)["data"]
            self.assertLessEqual(d["mel_fmax"], d["sampling_rate"] / 2)


class TestSynthesizer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net, self.cfg = build("micro", n_speakers=4)
        self.hop = self.cfg["data"]["hop_length"]
        self.seg_frames = self.cfg["train"]["segment_size"] // self.hop

    def _batch(self, b=2, t_text=17, t_frames=60):
        x = torch.randint(1, N_SYMBOLS, (b, t_text))
        x_len = torch.tensor([t_text, t_text - 4])
        spec_ch = self.cfg["data"]["filter_length"] // 2 + 1
        y = torch.randn(b, spec_ch, t_frames).abs()
        y_len = torch.tensor([t_frames, t_frames - 11])
        sid = torch.tensor([0, 2])
        return x, x_len, y, y_len, sid

    def test_forward_shapes(self):
        x, x_len, y, y_len, sid = self._batch()
        out = self.net(x, x_len, y, y_len, sid, mas_noise_scale=0.01)
        o, o_sub, l_len, attn, ids, x_mask, y_mask, latents, logws = out

        self.assertEqual(o.shape, (2, 1, self.seg_frames * self.hop))
        self.assertEqual(o_sub.shape[1], self.cfg["model"]["subbands"])
        self.assertEqual(o_sub.shape[2], self.seg_frames * self.hop // self.cfg["model"]["subbands"])
        self.assertEqual(attn.shape, (2, 1, 60, 17))
        self.assertTrue(torch.isfinite(o).all())

    def test_alignment_is_a_valid_monotonic_surjection(self):
        x, x_len, y, y_len, sid = self._batch()
        _, _, _, attn, *_ = self.net(x, x_len, y, y_len, sid)
        a = attn.squeeze(1)  # [B, T_spec, T_text]
        for i, (tl, yl) in enumerate(zip(x_len, y_len)):
            sub = a[i, :yl, :tl]
            self.assertTrue(torch.all(sub.sum(1) == 1), "each frame takes exactly one phoneme")
            self.assertTrue(torch.all(sub.sum(0) >= 1), "no phoneme is skipped")
            idx = sub.argmax(1)
            self.assertTrue(torch.all(idx[1:] - idx[:-1] >= 0), "alignment must be monotone")

    def test_durations_sum_to_frame_count(self):
        x, x_len, y, y_len, sid = self._batch()
        _, _, _, attn, *_ = self.net(x, x_len, y, y_len, sid)
        w = attn.sum(2).squeeze(1)
        for i, yl in enumerate(y_len):
            self.assertEqual(int(w[i].sum()), int(yl))

    def test_backward_reaches_every_trainable_parameter(self):
        """A parameter with no gradient is dead weight and usually a bug."""
        x, x_len, y, y_len, sid = self._batch()
        out = self.net(x, x_len, y, y_len, sid)
        o, o_sub, l_len, attn, ids, x_mask, y_mask, latents, _ = out
        z, z_p, m_p, logs_p, m_q, logs_q = latents

        loss = o.pow(2).mean() + l_len.sum() + kl_loss(z_p, logs_q, m_p, logs_p, y_mask)
        loss.backward()

        missing = [
            n for n, p in self.net.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        # emb_g rows for unused speakers legitimately get no gradient only if
        # the embedding itself is untouched; here every branch should be hit.
        self.assertEqual(missing, [], f"parameters received no gradient: {missing[:8]}")

    def test_inference_length_follows_length_scale(self):
        self.net.eval()
        x = torch.randint(1, N_SYMBOLS, (1, 20))
        x_len = torch.tensor([20])
        sid = torch.tensor([1])
        with torch.no_grad():
            slow, *_ = self.net.infer(x, x_len, sid, length_scale=1.5)
            fast, *_ = self.net.infer(x, x_len, sid, length_scale=0.75)
        self.assertGreater(slow.shape[-1], fast.shape[-1])

    def test_inference_output_length_is_a_multiple_of_hop(self):
        self.net.eval()
        x = torch.randint(1, N_SYMBOLS, (1, 14))
        with torch.no_grad():
            o, *_ = self.net.infer(x, torch.tensor([14]), torch.tensor([0]))
        self.assertEqual(o.shape[-1] % self.hop, 0)
        self.assertTrue(torch.isfinite(o).all())

    def test_prepare_for_inference_drops_posterior_encoder(self):
        net, _ = build("micro", n_speakers=4)
        before = net.n_parameters()
        after = net.prepare_for_inference().n_parameters()
        self.assertLess(after, before)
        self.assertFalse(hasattr(net, "enc_q"))


class TestDiscriminators(unittest.TestCase):
    def test_combined_discriminator_runs(self):
        torch.manual_seed(0)
        d = CombinedDiscriminator()
        y = torch.randn(2, 1, 8192) * 0.2
        y_hat = torch.randn(2, 1, 8192) * 0.2
        dr, dg, fr, fg = d(y, y_hat)
        # 1 scale + 5 periods + 3 resolutions
        self.assertEqual(len(dr), 9)
        self.assertEqual(len(dr), len(dg))
        self.assertEqual(len(fr), len(fg))

    def test_duration_discriminator_runs(self):
        dd = DurationDiscriminator(96, 96, 3, 0.1, gin_channels=128)
        x = torch.randn(2, 96, 17)
        x_mask = torch.ones(2, 1, 17)
        dur_r = torch.rand(2, 1, 17) * 5
        dur_hat = torch.rand(2, 1, 17) * 5
        g = torch.randn(2, 128, 1)
        out = dd(x, x_mask, dur_r, dur_hat, g)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (2, 17, 1))


class TestSignalPath(unittest.TestCase):
    def test_mel_matches_expected_frame_count(self):
        y = torch.randn(2, 1, 16000) * 0.2
        mel = mel_spectrogram_torch(y, 1024, 80, 16000, 256, 1024, 0.0, 8000.0)
        self.assertEqual(mel.shape[:2], (2, 80))
        self.assertEqual(mel.shape[2], 16000 // 256)

    def test_spectrogram_frame_count(self):
        y = torch.randn(2, 16000) * 0.2
        spec = spectrogram_torch(y, 1024, 256, 1024)
        self.assertEqual(spec.shape, (2, 513, 16000 // 256))

    def test_mel_basis_rows_are_normalised(self):
        from dcho.model.mel_processing import mel_filterbank

        fb = mel_filterbank(16000, 1024, 80, 0.0, 8000.0)
        self.assertEqual(fb.shape, (80, 513))
        self.assertTrue((fb >= 0).all(), "mel weights must be non-negative")
        self.assertTrue((fb.sum(axis=1) > 0).all(), "no empty mel band")

    def test_multiresolution_stft_loss_is_zero_for_identical_signals(self):
        loss = MultiResolutionSTFTLoss()
        x = torch.randn(2, 4, 2048) * 0.2
        sc, mag = loss(x, x.clone())
        self.assertLess(float(sc), 1e-5)
        self.assertLess(abs(float(mag)), 1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestSpeakerConditioning(unittest.TestCase):
    """Both conditioning paths must work, because which one ships depends on
    a property of the corpus that is only known after measurement."""

    def test_lookup_table_path(self):
        net, _ = build("micro", n_speakers=4)
        self.assertTrue(hasattr(net, "emb_g"))
        self.assertFalse(hasattr(net, "proj_g"))

    def _build_vector_model(self, dim: int = 192):
        cfg = load_config("micro")
        data, model = cfg["data"], cfg["model"]
        return Synthesizer(
            n_vocab=N_SYMBOLS,
            spec_channels=data["filter_length"] // 2 + 1,
            segment_size=cfg["train"]["segment_size"] // data["hop_length"],
            n_speakers=0,
            speaker_embed_dim=dim,
            **{k: v for k, v in model.items() if not k.startswith("_")},
        ), cfg

    def test_vector_path_forward_and_infer(self):
        torch.manual_seed(0)
        net, cfg = self._build_vector_model()
        b, t_text, t_frames = 2, 15, 55
        x = torch.randint(1, N_SYMBOLS, (b, t_text))
        x_len = torch.tensor([t_text, t_text - 3])
        y = torch.randn(b, cfg["data"]["filter_length"] // 2 + 1, t_frames).abs()
        y_len = torch.tensor([t_frames, t_frames - 7])
        spk = torch.nn.functional.normalize(torch.randn(b, 192), dim=-1)

        o, *_ = net(x, x_len, y, y_len, spk)
        self.assertTrue(torch.isfinite(o).all())

        net.eval()
        with torch.no_grad():
            audio, *_ = net.infer(x[:1], x_len[:1], sid=spk[:1])
        self.assertTrue(torch.isfinite(audio).all())

    def test_vector_path_rejects_wrong_shape(self):
        net, _ = self._build_vector_model(dim=192)
        with self.assertRaises(ValueError):
            net._speaker_embedding(torch.randn(2, 512))
        with self.assertRaises(ValueError):
            net._speaker_embedding(torch.tensor([0, 1]))

    def test_different_vectors_give_different_audio(self):
        """If conditioning had no effect the model would average all voices,
        which is the exact failure this path exists to avoid."""
        torch.manual_seed(0)
        net, _ = self._build_vector_model()
        net.eval()
        x = torch.randint(1, N_SYMBOLS, (1, 12))
        x_len = torch.tensor([12])
        a = torch.nn.functional.normalize(torch.randn(1, 192), dim=-1)
        b = torch.nn.functional.normalize(torch.randn(1, 192), dim=-1)
        with torch.no_grad():
            ga = net._speaker_embedding(a)
            gb = net._speaker_embedding(b)
        self.assertGreater(float((ga - gb).abs().mean()), 1e-4)
