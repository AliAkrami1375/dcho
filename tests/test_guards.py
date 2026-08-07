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

    def test_trips_on_non_finite_gradient_norm(self):
        g = HealthGuard(self.cfg)
        with self.assertRaises(TrainingHalted):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
