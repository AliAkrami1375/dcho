"""Resume must be indistinguishable from never having stopped.

Colab kills a runtime without warning. When it does, no Python runs - no
`except`, no `finally` - so the only thing that survives is what was
already written. Everything about recovery therefore rests on the
checkpoint being complete, and "complete" is easy to get subtly wrong: a
run that resumes and trains happily can still have thrown away its
learning-rate schedule, which is invisible until the model quietly fails
to converge.

These tests restore into a *fresh* trainer, the way a new session does,
and compare against the run that never stopped.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from dcho.model.synthesizer import Synthesizer
from dcho.text.phonemes import N_SYMBOLS
from dcho.train.trainer import Trainer

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def make_trainer(out_dir, seed=1234):
    torch.manual_seed(seed)
    cfg = json.loads((CONFIG_DIR / "micro.json").read_text())
    cfg["train"]["batch_size"] = 2
    net = Synthesizer(
        n_vocab=N_SYMBOLS, spec_channels=513,
        segment_size=cfg["train"]["segment_size"] // 256,
        n_speakers=0, speaker_embed_dim=192,
        **{k: v for k, v in cfg["model"].items() if not k.startswith("_")},
    )
    return Trainer(cfg, net, output_dir=out_dir, device="cpu", speaker_embed_dim=192), cfg


def fake_batch(seed=0, b=2, t_text=24, t_frames=48):
    torch.manual_seed(seed)
    hop = 256
    return {
        "text": torch.randint(1, N_SYMBOLS, (b, t_text)),
        "text_lengths": torch.tensor([t_text, t_text - 4]),
        "spec": torch.randn(b, 513, t_frames).abs(),
        "spec_lengths": torch.tensor([t_frames, t_frames - 6]),
        "wav": torch.randn(b, 1, t_frames * hop) * 0.1,
        "wav_lengths": torch.tensor([t_frames * hop] * b),
        "sid": torch.nn.functional.normalize(torch.randn(b, 192), dim=-1),
    }


class TestCheckpointCompleteness(unittest.TestCase):
    def test_checkpoint_carries_every_piece_of_run_state(self):
        with tempfile.TemporaryDirectory() as d:
            tr, _ = make_trainer(d)
            tr.train_step(fake_batch())
            ckpt = torch.load(tr.save("t"), map_location="cpu", weights_only=False)

        for key in ("step", "epoch", "net_g", "net_d", "opt_g", "opt_d",
                    "sched_g", "sched_d", "scaler", "config"):
            self.assertIn(key, ckpt, f"resume would silently lose {key}")

    def test_learning_rate_survives_a_restart(self):
        """The failure this guards against: a run resumes, trains, and looks
        fine, while the learning rate has jumped back to its initial value."""
        with tempfile.TemporaryDirectory() as d:
            tr, cfg = make_trainer(d)
            for _ in range(20):
                tr.sched_g.step()
                tr.sched_d.step()
            tr.state.step = 20
            tr.train_step(fake_batch())
            before = tr.current_lr()
            path = tr.save("t")

            fresh, _ = make_trainer(d, seed=99)
            self.assertNotAlmostEqual(fresh.current_lr(), before, places=9)
            fresh.load(path)
            self.assertAlmostEqual(fresh.current_lr(), before, places=12)

    def test_step_counter_survives(self):
        with tempfile.TemporaryDirectory() as d:
            tr, _ = make_trainer(d)
            tr.train_step(fake_batch())
            tr.state.step = 4321
            path = tr.save("t")

            fresh, _ = make_trainer(d, seed=7)
            fresh.load(path)
            self.assertEqual(fresh.state.step, 4321)

    def test_weights_are_bit_identical_after_restore(self):
        with tempfile.TemporaryDirectory() as d:
            tr, _ = make_trainer(d)
            tr.train_step(fake_batch(1))
            path = tr.save("t")
            reference = {k: v.clone() for k, v in tr.net_g.state_dict().items()}

            fresh, _ = make_trainer(d, seed=999)
            fresh.load(path)
            for k, v in fresh.net_g.state_dict().items():
                self.assertTrue(torch.equal(v, reference[k]), f"{k} differs after restore")

    def test_optimiser_moments_survive(self):
        """Losing Adam's moments restarts the optimiser cold and produces a
        visible transient exactly where a resumed run should be smooth."""
        with tempfile.TemporaryDirectory() as d:
            tr, _ = make_trainer(d)
            tr.train_step(fake_batch(2))
            path = tr.save("t")
            reference = tr.opt_g.state_dict()["state"]

            fresh, _ = make_trainer(d, seed=5)
            fresh.load(path)
            restored = fresh.opt_g.state_dict()["state"]
            self.assertEqual(len(restored), len(reference))
            self.assertGreater(len(restored), 0, "no optimiser state was captured")

    def test_resuming_adds_no_error_beyond_the_platform_floor(self):
        """End to end: an interrupted-and-resumed run against a straight one.

        The bar is not bit-identity. Two identical runs in the same process
        already differ, because convolution backward on a CPU reduces in a
        thread-dependent order - measured at around 3e-4 on this machine.
        The meaningful question is whether resuming adds anything on top of
        that floor, so the floor is measured here rather than assumed and
        the comparison is made against it.
        """
        def run_straight():
            with tempfile.TemporaryDirectory() as d:
                t, _ = make_trainer(d)
                t.train_step(fake_batch(11))
                t.state.step = 1
                t.sched_g.step()
                t.sched_d.step()
                t.train_step(fake_batch(12))
                return {k: v.clone() for k, v in t.net_g.state_dict().items()}

        def worst_diff(a, b):
            return max(float((a[k] - b[k]).abs().max()) for k in a)

        reference = run_straight()
        floor = worst_diff(reference, run_straight())

        with tempfile.TemporaryDirectory() as d:
            first, _ = make_trainer(d)
            first.train_step(fake_batch(11))
            first.state.step = 1
            first.sched_g.step()
            first.sched_d.step()
            path = first.save("interrupted")

            resumed, _ = make_trainer(d, seed=4242)
            resumed.load(path)
            resumed.train_step(fake_batch(12))
            got = {k: v.clone() for k, v in resumed.net_g.state_dict().items()}

        resumed_gap = worst_diff(reference, got)
        tolerance = max(floor * 4, 1e-6)
        self.assertLess(
            resumed_gap, tolerance,
            f"resuming diverged by {resumed_gap:.2e}, well beyond the "
            f"platform's own {floor:.2e} - that is a real loss of state",
        )


class TestGeneratorSnapshot(unittest.TestCase):
    def test_snapshot_is_much_smaller_than_a_full_checkpoint(self):
        """Snapshots exist so that watching a run does not cost the upload
        bandwidth of resuming one."""
        with tempfile.TemporaryDirectory() as d:
            tr, _ = make_trainer(d)
            tr.train_step(fake_batch())
            full = tr.save("full").stat().st_size
            snap = tr.save_generator("snap").stat().st_size
        self.assertLess(snap * 4, full, "a snapshot should be a fraction of a checkpoint")

    def test_snapshot_loads_as_weights_only(self):
        with tempfile.TemporaryDirectory() as d:
            tr, _ = make_trainer(d)
            tr.train_step(fake_batch(3))
            tr.state.step = 777
            path = tr.save_generator("snap")
            reference = {k: v.clone() for k, v in tr.net_g.state_dict().items()}

            fresh, _ = make_trainer(d, seed=8)
            fresh.load(path, weights_only=True)
            for k, v in fresh.net_g.state_dict().items():
                self.assertTrue(torch.equal(v, reference[k]))
            self.assertEqual(fresh.state.step, 777)


if __name__ == "__main__":
    unittest.main(verbosity=2)
