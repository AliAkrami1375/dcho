"""Training loop.

Structured around one idea: a run should end by itself when it is no
longer worth continuing. Adversarial text-to-speech training fails in ways
that are obvious from the curves within an hour and invisible from the
final checkpoint four days later, so the loop evaluates its own health and
has permission to stop - on a blown budget, on an alignment that never
locked in, on a collapsed critic.

Every phase resumes from the previous one's checkpoint. Nothing here ever
starts from random weights except the first phase, which means the cost of
a mistake is the cost of one phase rather than of the whole schedule.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F

from ..model import commons
from ..model.discriminators import CombinedDiscriminator
from ..model.duration import DurationDiscriminator
from ..model.losses import (
    MultiResolutionSTFTLoss,
    discriminator_loss,
    duration_discriminator_loss,
    duration_generator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from ..model.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from ..model.synthesizer import Synthesizer
from .guards import BudgetGuard, HealthGuard, TrainingHalted


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_mel: float = math.inf
    history: list[dict] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        config: dict,
        model: Synthesizer,
        output_dir: str | Path,
        device: str = "cuda",
        speaker_embed_dim: int = 0,
    ):
        self.cfg = config
        self.train_cfg = config["train"]
        self.data_cfg = config["data"]
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.net_g = model.to(device)
        self.net_d = CombinedDiscriminator(
            use_spectral_norm=config["model"].get("use_spectral_norm", False)
        ).to(device)

        self.use_dur_disc = config["model"].get("use_duration_discriminator", False)
        self.net_dur_d = None
        if self.use_dur_disc:
            self.net_dur_d = DurationDiscriminator(
                config["model"]["hidden_channels"],
                config["model"]["hidden_channels"],
                3,
                0.1,
                gin_channels=config["model"].get("gin_channels", 0),
            ).to(device)

        lr = self.train_cfg["learning_rate"]
        betas = tuple(self.train_cfg["betas"])
        eps = self.train_cfg["eps"]
        self.opt_g = torch.optim.AdamW(self.net_g.parameters(), lr, betas=betas, eps=eps)
        self.opt_d = torch.optim.AdamW(self.net_d.parameters(), lr, betas=betas, eps=eps)
        self.opt_dur = (
            torch.optim.AdamW(self.net_dur_d.parameters(), lr, betas=betas, eps=eps)
            if self.net_dur_d
            else None
        )

        decay = self.train_cfg["lr_decay"]
        self.sched_g = torch.optim.lr_scheduler.ExponentialLR(self.opt_g, gamma=decay)
        self.sched_d = torch.optim.lr_scheduler.ExponentialLR(self.opt_d, gamma=decay)

        # The config carries a clip value; passing None here silently
        # disabled it, which is how an unclipped run reached the guards.
        self.grad_clip = self.train_cfg.get("grad_clip")
        self.use_amp = bool(self.train_cfg.get("fp16_run", False)) and device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.subband_loss = MultiResolutionSTFTLoss().to(device)

        self.budget = BudgetGuard(
            hourly_rate_usd=self.train_cfg.get("hourly_rate_usd", 0.80),
            max_cost_usd=self.train_cfg.get("max_cost_usd"),
        )
        guards = dict(self.train_cfg.get("guards", {}))
        self.health = HealthGuard(
            guards,
            warmup_steps=guards.pop("warmup_steps", 500),
            nonfinite_patience=guards.pop("nonfinite_patience", 20),
        )
        self.state = TrainState()

    # -- schedules --------------------------------------------------------

    def mas_noise_scale(self) -> float:
        """Noise added to the alignment search, decayed to zero.

        Early on the likelihood surface is nearly flat and the search can
        settle into a degenerate path it never leaves. A little noise keeps
        it exploring until there is real signal to follow, and then gets out
        of the way.
        """
        initial = self.train_cfg.get("mas_noise_scale_initial", 0.0)
        decay = self.train_cfg.get("mas_noise_scale_decay", 0.0)
        return max(0.0, initial - decay * self.state.step)

    # -- one step ---------------------------------------------------------

    def train_step(self, batch) -> dict:
        cfg, d = self.train_cfg, self.data_cfg
        dev = self.device

        x = batch["text"].to(dev, non_blocking=True)
        x_lengths = batch["text_lengths"].to(dev, non_blocking=True)
        spec = batch["spec"].to(dev, non_blocking=True)
        spec_lengths = batch["spec_lengths"].to(dev, non_blocking=True)
        y = batch["wav"].to(dev, non_blocking=True)
        sid = batch["sid"].to(dev, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            out = self.net_g(x, x_lengths, spec, spec_lengths, sid,
                             mas_noise_scale=self.mas_noise_scale())
            (y_hat, y_hat_sub, l_length, attn, ids_slice,
             x_mask, z_mask, latents, logws) = out
            z, z_p, m_p, logs_p, m_q, logs_q = latents

            segment = self.net_g.segment_size
            mel = spec_to_mel_torch(spec.float(), d["filter_length"], d["n_mel_channels"],
                                    d["sampling_rate"], d["mel_fmin"], d["mel_fmax"])
            y_mel = commons.slice_segments(mel, ids_slice, segment)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(), d["filter_length"], d["n_mel_channels"],
                d["sampling_rate"], d["hop_length"], d["win_length"],
                d["mel_fmin"], d["mel_fmax"],
            )
            y_slice = commons.slice_segments(y, ids_slice * d["hop_length"],
                                             segment * d["hop_length"])

        # -- critics ------------------------------------------------------
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            y_d_r, y_d_g, _, _ = self.net_d(y_slice, y_hat.detach())
        with torch.amp.autocast("cuda", enabled=False):
            loss_disc, _, _ = discriminator_loss(y_d_r, y_d_g)

        self.opt_d.zero_grad(set_to_none=True)
        self.scaler.scale(loss_disc).backward()
        self.scaler.unscale_(self.opt_d)
        grad_norm_d = commons.clip_grad_value_(self.net_d.parameters(), self.grad_clip)
        self.scaler.step(self.opt_d)

        loss_dur_disc = torch.zeros((), device=dev)
        if self.net_dur_d is not None:
            logw, logw_hat = logws
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                dr, dg = self.net_dur_d(latents_x := self._dur_context(x, x_lengths),
                                        x_mask, logw.detach(), logw_hat.detach())
            with torch.amp.autocast("cuda", enabled=False):
                loss_dur_disc = duration_discriminator_loss([dr], [dg])
            self.opt_dur.zero_grad(set_to_none=True)
            self.scaler.scale(loss_dur_disc).backward()
            self.scaler.step(self.opt_dur)

        # -- generator ----------------------------------------------------
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            _, y_d_g, fmap_r, fmap_g = self.net_d(y_slice, y_hat)
        with torch.amp.autocast("cuda", enabled=False):
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * cfg["c_mel"]
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * cfg["c_kl"]
            loss_dur = torch.sum(l_length.float())
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _ = generator_loss(y_d_g)

            # Supervise the subbands directly. Without this the decoder can
            # place errors in neighbouring bands that cancel in the sum but
            # alias apart once the bank recombines them.
            y_sub_real = self.net_g.dec.pqmf.analysis(y_slice.float())
            n = min(y_sub_real.shape[-1], y_hat_sub.shape[-1])
            sc, mag = self.subband_loss(y_hat_sub[..., :n].float(), y_sub_real[..., :n])
            loss_sub = sc * cfg["c_subband_sc"] + mag * cfg["c_subband_mag"]

            loss_g = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + loss_sub

            if self.net_dur_d is not None:
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    _, dg2 = self.net_dur_d(latents_x, x_mask, logws[0].detach(), logws[1])
                loss_g = loss_g + duration_generator_loss([dg2.float()]) * cfg["c_dur_adv"]

        self.opt_g.zero_grad(set_to_none=True)
        self.scaler.scale(loss_g).backward()
        self.scaler.unscale_(self.opt_g)
        grad_norm_g = commons.clip_grad_value_(self.net_g.parameters(), self.grad_clip)
        self.scaler.step(self.opt_g)
        self.scaler.update()

        return {
            "loss_g": float(loss_g),
            "loss_disc": float(loss_disc),
            "loss_mel": float(loss_mel),
            "loss_kl": float(loss_kl),
            "loss_dur": float(loss_dur),
            "loss_fm": float(loss_fm),
            "loss_gen": float(loss_gen),
            "loss_subband": float(loss_sub),
            "loss_dur_disc": float(loss_dur_disc),
            "grad_norm": float(grad_norm_g),
            "grad_norm_d": float(grad_norm_d),
            "alignment_entropy": self.net_g.diagnostics.get("alignment_entropy", float("nan")),
            "mas_noise": self.mas_noise_scale(),
            "lr": self.sched_g.get_last_lr()[0],
        }

    def _dur_context(self, x, x_lengths):
        """Text encoder output, as the duration critic's conditioning."""
        with torch.no_grad():
            h, _, _, _ = self.net_g.enc_p(x, x_lengths)
        return h.detach()

    # -- loop -------------------------------------------------------------

    def current_lr(self) -> float:
        return self.sched_g.get_last_lr()[0]

    def train(self, loader, max_steps: int, log_every: int | None = None,
              checkpoint_every: int | None = None, on_log=None) -> TrainState:
        log_every = log_every or self.train_cfg.get("log_interval", 100)
        checkpoint_every = checkpoint_every or self.train_cfg.get("checkpoint_interval", 5000)

        self.net_g.train()
        self.net_d.train()
        running: dict[str, float] = {}
        started = time.time()

        try:
            for batch in loader:
                if batch is None:
                    continue
                metrics = self.train_step(batch)
                self.state.step += 1
                for k, v in metrics.items():
                    running[k] = running.get(k, 0.0) + v

                self.budget.check()
                self.health.check(self.state.step, metrics)

                if self.state.step % log_every == 0:
                    avg = {k: v / log_every for k, v in running.items()}
                    avg["step"] = self.state.step
                    avg["elapsed_hours"] = round(self.budget.elapsed_hours(), 3)
                    avg["spent_usd"] = round(self.budget.spent_usd(), 3)
                    avg["steps_per_sec"] = round(
                        self.state.step / max(time.time() - started, 1e-9), 3
                    )
                    self.state.history.append(avg)
                    self._print(avg)
                    if on_log:
                        on_log(avg)
                    running = {}

                if self.state.step % checkpoint_every == 0:
                    self.save(f"step_{self.state.step}")

                self.sched_g.step()
                self.sched_d.step()

                if self.state.step >= max_steps:
                    break

        except TrainingHalted as halt:
            # A halted run still gets a checkpoint: the weights up to the
            # stop are often worth inspecting, and throwing them away would
            # mean paying for the same steps twice.
            print(f"\n[halt] {halt.reason}: {halt.detail}", flush=True)
            self.save("halted")
            self._write_summary(halted=halt)
            raise

        self.save("final")
        self._write_summary()
        return self.state

    def _print(self, m: dict) -> None:
        print(
            f"step {m['step']:>7d} | mel {m['loss_mel']:7.3f} | kl {m['loss_kl']:6.3f} "
            f"| dur {m['loss_dur']:7.3f} | disc {m['loss_disc']:6.3f} "
            f"| sub {m['loss_subband']:6.3f} | align_H {m['alignment_entropy']:.4f} "
            f"| {m['steps_per_sec']:.2f} it/s | ${m['spent_usd']:.2f}",
            flush=True,
        )

    # -- checkpoints ------------------------------------------------------

    def save_generator(self, tag: str) -> Path:
        """Write the generator alone - about a tenth of a full checkpoint.

        A full checkpoint is dominated by the discriminators and their
        optimiser moments, none of which is needed to listen to the model or
        to watch a curve. Pushing the whole thing on every logging interval
        spends a large share of a Colab session on upload rather than on
        training.
        """
        path = self.output_dir / f"{tag}.G.pth"
        torch.save(
            {"step": self.state.step, "config": self.cfg, "net_g": self.net_g.state_dict()},
            path,
        )
        return path

    def save(self, tag: str) -> Path:
        """Write everything needed to continue as though nothing happened.

        The schedulers and the loss scaler are part of that. Leaving them
        out looks harmless because the run still starts - but an
        ExponentialLR restarted at step zero puts the learning rate back to
        its initial value, which after twenty thousand steps is a jump of
        more than fifty times and undoes the run.
        """
        path = self.output_dir / f"{tag}.pth"
        payload = {
            "step": self.state.step,
            "epoch": self.state.epoch,
            "config": self.cfg,
            "history": self.state.history[-500:],
            "net_g": self.net_g.state_dict(),
            "net_d": self.net_d.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "sched_g": self.sched_g.state_dict(),
            "sched_d": self.sched_d.state_dict(),
            "scaler": self.scaler.state_dict(),
            # The model samples during training - the posterior encoder, the
            # stochastic duration predictor, dropout, the random segment
            # slice. Without the generator state a resumed run continues
            # from a different point in the random stream, which is not
            # wrong but is not reproducible either.
            "rng": torch.get_rng_state(),
            "rng_cuda": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
        }
        if self.net_dur_d is not None:
            payload["net_dur_d"] = self.net_dur_d.state_dict()
            payload["opt_dur"] = self.opt_dur.state_dict()
        torch.save(payload, path)
        return path

    def load(self, path: str | Path, weights_only: bool = False) -> None:
        """Restore a run. `weights_only` loads the generator alone, which is
        what a generator snapshot contains."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net_g.load_state_dict(ckpt["net_g"])
        if weights_only or "net_d" not in ckpt:
            self.state.step = ckpt.get("step", self.state.step)
            return
        self.net_d.load_state_dict(ckpt["net_d"])
        self.opt_g.load_state_dict(ckpt["opt_g"])
        self.opt_d.load_state_dict(ckpt["opt_d"])
        if self.net_dur_d is not None and "net_dur_d" in ckpt:
            self.net_dur_d.load_state_dict(ckpt["net_dur_d"])
            self.opt_dur.load_state_dict(ckpt["opt_dur"])

        self.state.step = ckpt.get("step", 0)
        self.state.epoch = ckpt.get("epoch", 0)
        self.state.history = list(ckpt.get("history", []))

        if "sched_g" in ckpt:
            self.sched_g.load_state_dict(ckpt["sched_g"])
            self.sched_d.load_state_dict(ckpt["sched_d"])
        else:
            # An older checkpoint predates scheduler state. Fast-forward the
            # decay to where this step should be rather than silently
            # resuming at the initial learning rate.
            for _ in range(self.state.step):
                self.sched_g.step()
                self.sched_d.step()
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("rng") is not None:
            torch.set_rng_state(ckpt["rng"].cpu().to(torch.uint8))
        if ckpt.get("rng_cuda") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
            except Exception:
                # A checkpoint made on a machine with a different GPU count
                # cannot restore per-device state; the run continues from a
                # fresh stream rather than failing.
                pass

    def _write_summary(self, halted: TrainingHalted | None = None) -> None:
        summary = {
            "steps": self.state.step,
            "budget": self.budget.report(),
            "halted": None if halted is None else
                      {"reason": halted.reason, "detail": halted.detail},
            "history": self.state.history[-200:],
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
