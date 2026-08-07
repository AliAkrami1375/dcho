"""Tests for the automatic stop conditions.

These matter more than their size suggests: a guard that fails to fire
costs GPU hours, and a guard that fires spuriously costs a whole run.
"""

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path

import torch

from dcho.train.guards import (
    BudgetGuard,
    HealthGuard,
    TrainingHalted,
    alignment_entropy,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class TestBudgetGuard(unittest.TestCase):
    def test_unbounded_budget_never_trips(self):
        g = BudgetGuard(hourly_rate_usd=0.80, max_cost_usd=None)
        g.check()
        self.assertEqual(g.remaining_usd(), float("inf"))

    def test_trips_once_spend_exceeds_ceiling(self):
        # Backdate the start so that an hour of spend has notionally passed.
        g = BudgetGuard(hourly_rate_usd=0.80, max_cost_usd=0.50,
                        started_at=time.time() - 3600)
        self.assertGreater(g.spent_usd(), 0.5)
        with self.assertRaises(TrainingHalted) as ctx:
            g.check()
        self.assertEqual(ctx.exception.reason, "budget_exhausted")

    def test_does_not_trip_below_ceiling(self):
        g = BudgetGuard(hourly_rate_usd=0.80, max_cost_usd=10.0,
                        started_at=time.time() - 3600)
        g.check()
        self.assertAlmostEqual(g.remaining_usd(), 10.0 - 0.80, places=2)

    def test_report_is_serialisable(self):
        g = BudgetGuard(hourly_rate_usd=0.80, max_cost_usd=2.0)
        json.dumps(g.report())


class TestHealthGuard(unittest.TestCase):
    def setUp(self):
        with open(CONFIG_DIR / "base.json", encoding="utf-8") as f:
            self.cfg = json.load(f)["train"]["guards"]

    def test_silent_before_the_deadline_step(self):
        g = HealthGuard(self.cfg)
        # Terrible metrics, but the deadline has not arrived yet.
        g.check(1000, {"alignment_entropy": 0.99, "loss_mel": 99.0})

    def test_trips_on_unconverged_alignment_after_deadline(self):
        g = HealthGuard(self.cfg)
        with self.assertRaises(TrainingHalted) as ctx:
            g.check(20000, {"alignment_entropy": 0.90})
        self.assertIn("alignment_entropy", ctx.exception.reason)

    def test_passes_when_alignment_has_converged(self):
        g = HealthGuard(self.cfg)
        g.check(20000, {"alignment_entropy": 0.10, "loss_mel": 12.0})

    def test_trips_on_collapsed_discriminator(self):
        g = HealthGuard(self.cfg)
        with self.assertRaises(TrainingHalted) as ctx:
            g.check(500, {"loss_disc": 0.001})
        self.assertIn("loss_disc", ctx.exception.reason)

    def test_a_single_non_finite_gradient_norm_does_not_trip(self):
        """This test previously asserted the opposite, and that assertion
        was the bug: with mixed precision the loss scaler produces exactly
        this on the first steps by design. See
        TestGuardsSurviveNormalStartup for the behaviour that replaced it."""
        g = HealthGuard(self.cfg)
        g.check(10, {"grad_norm": float("nan")})

    def test_missing_metric_is_not_an_error(self):
        """Metrics computed only every few thousand steps must not trip."""
        g = HealthGuard(self.cfg)
        g.check(60000, {})

    def test_fires_only_once(self):
        g = HealthGuard(self.cfg)
        with self.assertRaises(TrainingHalted):
            g.check(20000, {"alignment_entropy": 0.9})
        # After tripping, the loop is unwinding; further calls stay quiet so
        # the checkpoint write is not interrupted by a second exception.
        g.check(20001, {"alignment_entropy": 0.9})


class TestAlignmentEntropy(unittest.TestCase):
    def test_hard_alignment_scores_near_zero(self):
        b, t_spec, t_text = 1, 40, 8
        attn = torch.zeros(b, 1, t_spec, t_text)
        for s in range(t_spec):
            attn[0, 0, s, s * t_text // t_spec] = 1.0
        self.assertLess(alignment_entropy(attn), 0.01)

    def test_uniform_alignment_scores_near_one(self):
        attn = torch.ones(1, 1, 40, 8)
        self.assertGreater(alignment_entropy(attn), 0.95)

    def test_hard_alignment_beats_smeared_alignment(self):
        hard = torch.zeros(1, 1, 40, 8)
        for s in range(40):
            hard[0, 0, s, s * 8 // 40] = 1.0
        smeared = hard + 0.3
        self.assertLess(alignment_entropy(hard), alignment_entropy(smeared))

    def test_accepts_three_dimensional_input(self):
        attn = torch.ones(1, 40, 8)
        self.assertGreater(alignment_entropy(attn), 0.9)


class TestGuardsSurviveNormalStartup(unittest.TestCase):
    """The first Colab run halted at step 1. These pin the reasons."""

    def setUp(self):
        with open(CONFIG_DIR / "micro.json", encoding="utf-8") as f:
            self.cfg = json.load(f)["train"]["guards"]

    def test_mixed_precision_overflow_is_not_a_failure(self):
        """`GradScaler` starts at 2**16 and is designed to overflow, detect
        it, skip the step and back off. Treating that as a dead run ends
        training on step one of every mixed-precision job."""
        g = HealthGuard(self.cfg)
        for step in range(1, 15):
            g.check(step, {"grad_norm": float("inf"), "loss_disc": 9.0})

    def test_sustained_non_finite_gradients_do_halt(self):
        """A transient is the scaler working; a streak is divergence."""
        g = HealthGuard(self.cfg, nonfinite_patience=5)
        with self.assertRaises(TrainingHalted) as ctx:
            for step in range(1, 30):
                g.check(step, {"grad_norm": float("nan")})
        self.assertIn("grad_norm", ctx.exception.reason)

    def test_a_recovered_run_clears_the_streak(self):
        g = HealthGuard(self.cfg, nonfinite_patience=3)
        for step in range(1, 3):
            g.check(step, {"grad_norm": float("inf")})
        g.check(3, {"grad_norm": 400.0})
        for step in range(4, 6):
            g.check(step, {"grad_norm": float("inf")})   # must not trip

    def test_nothing_is_judged_during_warmup(self):
        """Initial loss and gradient values look catastrophic and are just
        what initialisation looks like."""
        g = HealthGuard(self.cfg, warmup_steps=500)
        g.check(1, {"loss_disc": 0.0001, "grad_norm": 90000.0, "loss_mel": 500.0})
        g.check(499, {"loss_disc": 0.0001, "grad_norm": 90000.0})

    def test_guards_apply_once_warmup_is_over(self):
        g = HealthGuard(self.cfg, warmup_steps=500)
        with self.assertRaises(TrainingHalted):
            g.check(501, {"loss_disc": 0.0001})

    def test_measured_step_one_metrics_pass(self):
        """Values actually observed on the packed corpus at step 1."""
        g = HealthGuard(self.cfg)
        g.check(1, {"loss_mel": 76.72, "loss_disc": 8.9833,
                    "grad_norm": 862.5, "alignment_entropy": 0.99})


class TestAlignmentEntropyIsInformative(unittest.TestCase):
    def test_hard_path_entropy_is_degenerate(self):
        """The search returns one-hot, so its entropy is always zero. This
        is why the reported figure comes from the soft posterior instead."""
        hard = torch.zeros(1, 1, 40, 8)
        for s in range(40):
            hard[0, 0, s, s * 8 // 40] = 1.0
        self.assertAlmostEqual(alignment_entropy(hard), 0.0, places=5)

    def test_model_reports_a_soft_entropy_that_moves(self):
        import json as _json

        from dcho.model.synthesizer import Synthesizer
        from dcho.text.phonemes import N_SYMBOLS

        torch.manual_seed(0)
        cfg = _json.loads((CONFIG_DIR / "micro.json").read_text())
        net = Synthesizer(
            n_vocab=N_SYMBOLS, spec_channels=513,
            segment_size=cfg["train"]["segment_size"] // 256,
            n_speakers=4,
            **{k: v for k, v in cfg["model"].items() if not k.startswith("_")},
        )
        x = torch.randint(1, N_SYMBOLS, (2, 15))
        y = torch.randn(2, 513, 55).abs()
        net(x, torch.tensor([15, 12]), y, torch.tensor([55, 40]), torch.tensor([0, 1]))

        h = net.diagnostics.get("alignment_entropy")
        self.assertIsNotNone(h, "the model must report an alignment entropy")
        self.assertGreater(h, 0.0, "an untrained model cannot be certain of its alignment")
        self.assertLessEqual(h, 1.0, "entropy is normalised to [0, 1]")


if __name__ == "__main__":
    unittest.main(verbosity=2)
