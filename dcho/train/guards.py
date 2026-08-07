"""Automatic stop conditions for training runs.

The expensive failure mode in a project like this is not a crash. A crash
is free - it stops. The expensive failure is a run that trains for four
days and produces something unusable, because nobody was watching the
curves at hour three when it was already clear.

These guards encode "already clear" as assertions the training loop
evaluates on itself, and give it permission to stop. Two families:

  health   metrics that must have reached a value by a given step. Missing
           one means the run is not going to recover, so it ends.
  budget   a hard ceiling on spend, computed from wall clock and the
           hardware's published hourly rate.

The budget ceiling is deliberately a mechanism rather than a discipline.
Intending to keep an eye on cost is not a control; a process that exits on
its own is.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


class TrainingHalted(Exception):
    """Raised to end a run cleanly, after a checkpoint has been written."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass
class BudgetGuard:
    """Stops the run once it has spent `max_cost_usd`."""

    hourly_rate_usd: float
    max_cost_usd: float | None = None
    started_at: float = field(default_factory=time.time)

    def elapsed_hours(self) -> float:
        return (time.time() - self.started_at) / 3600

    def spent_usd(self) -> float:
        return self.elapsed_hours() * self.hourly_rate_usd

    def remaining_usd(self) -> float:
        if self.max_cost_usd is None:
            return math.inf
        return max(0.0, self.max_cost_usd - self.spent_usd())

    def check(self) -> None:
        if self.max_cost_usd is None:
            return
        if self.spent_usd() >= self.max_cost_usd:
            raise TrainingHalted(
                "budget_exhausted",
                f"spent ${self.spent_usd():.2f} of ${self.max_cost_usd:.2f} "
                f"after {self.elapsed_hours():.2f} h",
            )

    def report(self) -> dict:
        return {
            "elapsed_hours": round(self.elapsed_hours(), 3),
            "spent_usd": round(self.spent_usd(), 3),
            "remaining_usd": (None if self.max_cost_usd is None
                              else round(self.remaining_usd(), 3)),
        }


@dataclass
class HealthGuard:
    """Checks that named metrics have reached their targets by given steps.

    Configured from the `guards` block of a config file. Each entry is
    either a `{step, value}` pair meaning "by this step, this metric must
    be at or below this value", or a bare float meaning "at every step".
    """

    config: dict
    _tripped: bool = False

    # Maps a config key onto the metric name the training loop reports.
    CHECKS_AT_STEP = {
        "alignment_entropy_max_at_step": "alignment_entropy",
        "loss_mel_max_at_step": "loss_mel",
        "cer_max_at_step": "cer",
    }
    FLOOR_CHECKS = {"disc_loss_min": "loss_disc"}
    CEILING_CHECKS = {"grad_norm_max": "grad_norm"}

    def check(self, step: int, metrics: dict) -> None:
        if self._tripped:
            return

        for key, metric in self.CHECKS_AT_STEP.items():
            rule = self.config.get(key)
            if not isinstance(rule, dict):
                continue
            # Evaluated in a window just after the deadline: metrics like CER
            # are only computed every few thousand steps, so an exact-step
            # test would usually never fire.
            if step < rule["step"]:
                continue
            value = metrics.get(metric)
            if value is None or not math.isfinite(value):
                continue
            if value > rule["value"]:
                self._tripped = True
                raise TrainingHalted(
                    f"health_{metric}",
                    f"{metric}={value:.4f} at step {step}, "
                    f"required <= {rule['value']} by step {rule['step']}",
                )

        for key, metric in self.FLOOR_CHECKS.items():
            floor = self.config.get(key)
            value = metrics.get(metric)
            if floor is None or value is None or not math.isfinite(value):
                continue
            if value < floor:
                self._tripped = True
                raise TrainingHalted(
                    f"health_{metric}",
                    f"{metric}={value:.5f} fell below {floor}; "
                    "the critic has collapsed and the generator is no longer "
                    "receiving a useful signal",
                )

        for key, metric in self.CEILING_CHECKS.items():
            ceiling = self.config.get(key)
            value = metrics.get(metric)
            if ceiling is None or value is None:
                continue
            if not math.isfinite(value) or value > ceiling:
                self._tripped = True
                raise TrainingHalted(
                    f"health_{metric}", f"{metric}={value} exceeded {ceiling}"
                )


def alignment_entropy(attn) -> float:
    """How concentrated the alignment is, normalised to [0, 1].

    Computed over the hard alignment's implied duration distribution. A run
    whose alignment has locked in produces a low value; one still smearing
    probability across the text axis stays high. This is the earliest
    reliable signal that a text-to-speech run is going to work, which is why
    it is the first guard to fire.
    """
    import torch

    with torch.no_grad():
        a = attn.squeeze(1) if attn.dim() == 4 else attn      # [B, T_spec, T_text]
        p = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
        ent = -(p * torch.log(p + 1e-8)).sum(dim=-1)
        n_text = (a.sum(dim=1) > 0).sum(dim=-1).clamp(min=2).float()
        return float((ent / torch.log(n_text).unsqueeze(-1)).mean())
