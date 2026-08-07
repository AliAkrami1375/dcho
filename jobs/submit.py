#!/usr/bin/env python3
"""Submit dcho jobs to Hugging Face and follow them.

Every submission goes through here rather than through ad-hoc CLI calls so
that the flavour, the timeout and the estimated cost of a run are recorded
in one place and printed before anything starts spending.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, Volume, fetch_job_logs, inspect_job, run_uv_job

DATASET = "Thomcles/Persian-Farsi-Speech"

# `/data` is reserved by the Jobs runtime for its own artifacts whenever a
# local script is uploaded, so the corpus goes somewhere else.
MOUNT_PATH = "/corpus"

# From the published Jobs hardware table. Used only to print an estimate
# before launch and a settled figure afterwards.
HOURLY_USD = {
    "cpu-basic": 0.01,
    "cpu-upgrade": 0.03,
    "cpu-xl": 1.00,
    "cpu-performance": 1.90,
    "t4-small": 0.40,
    "t4-medium": 0.60,
    "l4x1": 0.80,
    "a10g-small": 1.00,
    "a10g-large": 1.50,
    "l40sx1": 1.80,
    "a100-large": 2.50,
}


def read_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    for candidate in (Path(__file__).resolve().parent.parent / ".hugging", Path.home() / ".hugging"):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8").strip()
    raise SystemExit("no token: set HF_TOKEN or place one in .hugging")


def parse_timeout_minutes(text: str) -> float:
    unit = text[-1]
    value = float(text[:-1]) if unit in "smhd" else float(text)
    return {"s": value / 60, "m": value, "h": value * 60, "d": value * 1440}.get(unit, value)


def billed_minutes(info) -> float | None:
    """Minutes the job actually ran, excluding time queued for hardware."""
    d = info.__dict__
    start = d.get("started_at") or d.get("created_at")
    end = d.get("ended_at") or d.get("finished_at")
    if not start or not end:
        return None
    return (end - start).total_seconds() / 60


def follow(job_id: str, token: str, flavor: str, poll: int = 20) -> str:
    started = time.time()
    seen = 0
    rate = HOURLY_USD.get(flavor, 0.0)
    while True:
        info = inspect_job(job_id=job_id, token=token)
        stage = info.status.stage
        try:
            logs = list(fetch_job_logs(job_id=job_id, token=token))
            for line in logs[seen:]:
                print(line, flush=True)
            seen = len(logs)
        except Exception:
            pass
        if stage in ("COMPLETED", "ERROR", "CANCELED", "DELETED"):
            wall = time.time() - started
            # Wall clock from submission includes time spent queued waiting
            # for hardware, which is not billed. Report the actual run window
            # and fall back to wall clock only when the API omits it.
            billed = billed_minutes(info)
            if billed is not None:
                print(
                    f"\n[submit] {stage} - billed {billed:.1f} min "
                    f"= ${billed/60*rate:.4f} on {flavor} "
                    f"({wall/60:.1f} min wall, the rest was queueing)",
                    flush=True,
                )
            else:
                print(
                    f"\n[submit] {stage} after {wall/60:.1f} min wall clock; "
                    f"billed duration unavailable (upper bound "
                    f"${wall/3600*rate:.4f} on {flavor})",
                    flush=True,
                )
            if info.status.message:
                print(f"[submit] {info.status.message}", flush=True)
            return stage
        time.sleep(poll)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("script", help="path to the job script")
    ap.add_argument("--flavor", default="cpu-upgrade")
    ap.add_argument("--timeout", default="30m")
    ap.add_argument("--name", default=None)
    ap.add_argument("--mount-dataset", default=DATASET,
                    help=f"dataset repo mounted read-only at {MOUNT_PATH}; empty to skip")
    ap.add_argument("--detach", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")

    # Everything after a standalone `--` belongs to the job script. argparse
    # cannot express this without REMAINDER swallowing our own options too,
    # so the split is done by hand before parsing.
    argv = sys.argv[1:]
    if "--" in argv:
        cut = argv.index("--")
        own, script_args = argv[:cut], argv[cut + 1 :]
    else:
        own, script_args = argv, []
    opts = ap.parse_args(own)

    token = read_token()

    minutes = parse_timeout_minutes(opts.timeout)
    rate = HOURLY_USD.get(opts.flavor)
    if rate is None:
        raise SystemExit(f"unknown flavor {opts.flavor}; see `hf jobs hardware`")
    worst_case = minutes / 60 * rate

    api = HfApi(token=token)
    who = api.whoami()

    print(f"[submit] user     : {who['name']}")
    print(f"[submit] script   : {opts.script}")
    print(f"[submit] flavor   : {opts.flavor}  (${rate:.2f}/h)")
    print(f"[submit] timeout  : {opts.timeout}")
    print(f"[submit] worst case cost if it runs to timeout: ${worst_case:.3f}")
    if script_args:
        print(f"[submit] args     : {' '.join(script_args)}")

    if not opts.yes and worst_case > 1.0:
        reply = input(f"[submit] proceed? this can cost up to ${worst_case:.2f} [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("[submit] aborted")
            return 1

    volumes = []
    if opts.mount_dataset:
        volumes.append(Volume(type="dataset", source=opts.mount_dataset,
                              mount_path=MOUNT_PATH, read_only=True))

    job = run_uv_job(
        str(opts.script),
        script_args=script_args,
        flavor=opts.flavor,
        timeout=opts.timeout,
        volumes=volumes or None,
        secrets={"HF_TOKEN": token},
        env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
        name=opts.name or Path(opts.script).stem,
        token=token,
    )
    print(f"[submit] job id   : {job.id}")
    print(f"[submit] url      : {job.url}", flush=True)

    if opts.detach:
        return 0
    return 0 if follow(job.id, token, opts.flavor) == "COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
