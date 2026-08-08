"""One training session, start to finish, for a runtime that can vanish.

This lives in the repository rather than in the notebook, and that is the
point. A notebook carries its own cells: someone who opened it last week is
still running last week's logic, even though the first thing it does is
pull a fresh copy of this package. Keeping the loop here means a fix
reaches every session on the next pull, whatever notebook opened it.

The design assumes the runtime dies without warning, because on Colab it
does. When the runtime is reclaimed the process is killed - no exception,
no `finally`, nothing gets a chance to save. Only state already written
survives. So:

  Google Drive holds the working checkpoint. It is a mounted filesystem, so
  a write is a write - no upload to wait for - and it outlives the runtime,
  which /content does not. This is what makes a disconnect cost minutes.

  The Hub holds the durable copy, pushed less often. Drive can be
  unmounted, filled, or wiped; the Hub is what survives that, and it is how
  a run is inspected from somewhere else.

  Both are written on a wall-clock timer rather than a step count, because
  wall clock is what actually bounds the loss and it bounds it the same way
  whichever GPU the session happened to get.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

CHECKPOINT_NAME = "latest.pth"


def mount_drive(path: str = "/content/drive") -> Path | None:
    """Mount Google Drive and return the dcho folder inside it.

    Returns None when not running under Colab, so the same call is safe
    anywhere and the caller does not have to know where it is.
    """
    try:
        from google.colab import drive
    except ImportError:
        return None

    if not Path(path).exists() or not any(Path(path).iterdir()):
        drive.mount(path)
    folder = Path(path) / "MyDrive" / "dcho"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _resolve_resume(store: Path | None, api, run_repo: str, token: str | None):
    """Find the newest checkpoint, preferring the one that is already local.

    Drive first because reading it costs nothing; the Hub only if Drive has
    nothing to offer, which is the case on a fresh machine or after Drive
    was cleared.
    """
    if store is not None:
        local = store / CHECKPOINT_NAME
        if local.exists():
            return str(local), "drive"

    if api is None:
        return None, None
    try:
        files = api.list_repo_files(run_repo)
    except Exception:
        return None, None

    from huggingface_hub import hf_hub_download

    if CHECKPOINT_NAME in files:
        return hf_hub_download(run_repo, CHECKPOINT_NAME, token=token), "hub"

    numbered = [f for f in files if f.startswith("step_") and f.endswith(".pth")]
    if numbered:
        newest = max(numbered, key=lambda f: int(f[5:-4]))
        return hf_hub_download(run_repo, newest, token=token), "hub"
    return None, None


def run_session(
    config_name: str = "micro",
    dataset: str = "DibaAi/dcho-tier-a",
    run_repo: str = "DibaAi/dcho-run-micro",
    max_steps: int = 30_000,
    speaker_embed_dim: int = 192,
    drive_minutes: float = 10.0,
    hub_minutes: float = 60.0,
    snapshot_every: int = 2_000,
    data_dir: str | None = None,
    drive_dir: str | Path | None = None,
    repo_root: str | Path = ".",
    token: str | None = None,
    device: str = "cuda",
    num_workers: int = 2,
) -> dict:
    """Run until `max_steps`, the runtime dies, or a guard stops it."""
    import torch
    from torch.utils.data import DataLoader

    from ..data.dataset import DataConfig, PackedSpeechDataset, collate
    from ..model.synthesizer import Synthesizer
    from ..text.phonemes import N_SYMBOLS
    from .guards import TrainingHalted
    from .trainer import Trainer

    import os

    token = token or os.environ.get("HF_TOKEN")
    repo_root = Path(repo_root)
    cfg = json.loads((repo_root / "configs" / f"{config_name}.json").read_text())

    api = None
    if token:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(run_repo, repo_type="model", private=True, exist_ok=True)

    store = Path(drive_dir) / run_repo.split("/")[-1] if drive_dir else None
    if store is not None:
        store.mkdir(parents=True, exist_ok=True)
        print(f"[session] checkpoints -> {store}")
    else:
        print("[session] no Drive; checkpoints go to the Hub only, and anything "
              "written under /content dies with the runtime")

    # -- data ------------------------------------------------------------
    if data_dir is None:
        from huggingface_hub import snapshot_download

        started = time.time()
        data_dir = snapshot_download(
            dataset, repo_type="dataset", local_dir="/content/data",
            allow_patterns=["*.parquet", "*.json"], token=token,
        )
        print(f"[session] dataset ready in {(time.time()-started)/60:.1f} min")

    data_cfg = DataConfig(**{k: v for k, v in cfg["data"].items()
                             if k in DataConfig.__dataclass_fields__})
    loader = DataLoader(
        PackedSpeechDataset(data_dir, data_cfg),
        batch_size=cfg["train"]["batch_size"], num_workers=num_workers,
        collate_fn=lambda b: collate(b, data_cfg),
        pin_memory=(device == "cuda"), persistent_workers=num_workers > 0,
    )

    # -- model -----------------------------------------------------------
    net = Synthesizer(
        n_vocab=N_SYMBOLS,
        spec_channels=cfg["data"]["filter_length"] // 2 + 1,
        segment_size=cfg["train"]["segment_size"] // cfg["data"]["hop_length"],
        n_speakers=0, speaker_embed_dim=speaker_embed_dim,
        **{k: v for k, v in cfg["model"].items() if not k.startswith("_")},
    )
    cfg["train"]["max_cost_usd"] = None       # Colab bills by the hour, not per job
    out_dir = store if store is not None else Path("/content/out")
    trainer = Trainer(cfg, net, output_dir=out_dir, device=device,
                      speaker_embed_dim=speaker_embed_dim)

    resume_path, source = _resolve_resume(store, api, run_repo, token)
    if resume_path:
        trainer.load(resume_path)
        print(f"[session] resumed from {source} at step {trainer.state.step}, "
              f"lr {trainer.current_lr():.3e} — {max_steps - trainer.state.step} to go")
    else:
        print(f"[session] fresh run, lr {trainer.current_lr():.3e}")

    # -- checkpointing ---------------------------------------------------
    clock = {"drive": time.time(), "hub": time.time()}

    def write_checkpoint(reason: str = "") -> None:
        path = trainer.save("latest")
        clock["drive"] = time.time()
        note = f" ({reason})" if reason else ""
        print(f"[session] checkpoint at step {trainer.state.step}{note}", flush=True)

        if api is None:
            return
        if reason or time.time() - clock["hub"] >= hub_minutes * 60:
            try:
                api.upload_file(
                    path_or_fileobj=str(path), path_in_repo=CHECKPOINT_NAME,
                    repo_id=run_repo, repo_type="model",
                    commit_message=f"step {trainer.state.step}",
                )
                clock["hub"] = time.time()
                print("[session]   mirrored to the Hub", flush=True)
            except Exception as exc:
                # A failed upload must not end a run that is otherwise fine;
                # the Drive copy is still there.
                print(f"[session]   Hub upload failed, continuing: {exc}", flush=True)

    def on_log(metrics: dict) -> None:
        if time.time() - clock["drive"] >= drive_minutes * 60:
            write_checkpoint()
        if snapshot_every and metrics["step"] % snapshot_every == 0 and api is not None:
            try:
                snap = trainer.save_generator(f"snap_{metrics['step']}")
                api.upload_file(path_or_fileobj=str(snap),
                                path_in_repo=f"snap_{metrics['step']}.G.pth",
                                repo_id=run_repo, repo_type="model",
                                commit_message=f"snapshot {metrics['step']}")
            except Exception as exc:
                print(f"[session]   snapshot failed, continuing: {exc}", flush=True)

    # -- run -------------------------------------------------------------
    outcome = "completed"
    detail = ""
    try:
        trainer.train(loader, max_steps=max_steps, checkpoint_every=10**9, on_log=on_log)
        write_checkpoint("finished")
    except TrainingHalted as halt:
        outcome, detail = "halted", f"{halt.reason}: {halt.detail}"
        print(f"[session] HALTED — {detail}", flush=True)
        write_checkpoint("halted")
    except KeyboardInterrupt:
        outcome = "interrupted"
        print("[session] interrupted", flush=True)
        write_checkpoint("interrupted")
    except Exception as exc:
        outcome, detail = "error", repr(exc)
        traceback.print_exc()
        write_checkpoint("error")
    finally:
        summary = out_dir / "summary.json"
        if api is not None and summary.exists():
            try:
                api.upload_file(path_or_fileobj=str(summary), path_in_repo="summary.json",
                                repo_id=run_repo, repo_type="model",
                                commit_message=f"summary at step {trainer.state.step}")
            except Exception:
                pass

    return {
        "outcome": outcome,
        "detail": detail,
        "step": trainer.state.step,
        "lr": trainer.current_lr(),
        "checkpoint": str(out_dir / CHECKPOINT_NAME),
        "history": trainer.state.history[-20:],
        "trainer": trainer,
    }
