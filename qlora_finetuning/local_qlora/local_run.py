"""The optional full training run on this machine: before it starts, and while it runs
(req_003 v3.06).

The run itself is ``python -m scripts.qlora_train``, unchanged. A five-hour run on a laptop
fails for reasons a cloud job does not have, so two things stand around it here:

``preflight`` checks, and starts nothing:

====  ===============  =====================================================================
P1    The plan         the dry run succeeds; hours estimated from the last local run's speed
P2    A clean tree     no uncommitted change outside the ledger
P3    GPU memory       enough of the card is free; names the programs that hold memory
P4    Disk             room for the checkpoints
P5    Power            on AC power, and not set to sleep (skipped where it cannot be read)
P6    Nothing running  no training run is in progress, and none is waiting to be resumed
====  ===============  =====================================================================

``run_status`` says how far a run is: the step, the hours left at its speed so far, the last
loss, and, for a run that stopped, the command that continues it.

Usage::

    from qlora_finetuning.local_qlora.local_run import preflight, run_status

    outcome = preflight()
    outcome["passed"], outcome["checks"], outcome["estimate"]
    status = run_status()                       # the newest full local run
    status["step"], status["total"], status["hours_left"], status["state"]

A local run is a side note: it is never arm A1 and enters no comparison. This module reads
the machine through ``nvidia-smi``, ``powercfg``, and the Windows power status; each probe
answers "unknown" where it cannot read, and never raises. It imports no part of the
training stack; the plan loads the tokenizer when it is asked for.

From the command line: ``python -m scripts.qlora_local preflight`` and ``... status``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PureWindowsPath
from typing import Any

from qlora_finetuning.local_qlora.adapters import (
    CHECKPOINTS,
    TRAINING_NAME,
    checkpoint_steps,
    read_json,
    training_chain,
)
from qlora_finetuning.local_qlora.training import (
    LOG_NAME,
    MEASUREMENTS_NAME,
    TrainingContext,
    prepare,
)
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking import list_runs, load_manifest
from qlora_finetuning.tracking.config import TrackingError, load_tracking_config, utc_now
from qlora_finetuning.tracking.registry import read_git_info

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
LOCAL = "local"
LABEL = "local run: not an arm of the comparison"
RUNNING_MINUTES = 15  # a run whose log was written this recently is taken to be running
TRAINING_KINDS = ("train", "smoke")
RUNNING, INTERRUPTED, COMPLETED, FAILED = "running", "interrupted", "completed", "failed"
RESUME = "python -m scripts.qlora_train --resume"
_PROBE_SECONDS = 20
_NOT_A_PROGRAM = re.compile(r"^\[.*\]$")  # nvidia-smi's "[N/A]", "[Insufficient Permissions]"
_AC_SLEEP = re.compile(r"Current AC Power Setting Index:\s*0x([0-9a-fA-F]+)")


# --------------------------------------------------------------------------- reading the machine


@dataclass(frozen=True)
class GpuState:
    """What the first GPU reports. Amounts in GiB."""

    name: str
    total_gib: float
    used_gib: float
    free_gib: float
    programs: tuple[str, ...] = ()  # programs that hold memory on it, by file name


@dataclass(frozen=True)
class PowerState:
    """``None`` means the machine did not say."""

    on_ac: bool | None = None
    sleep_on_ac_seconds: int | None = None  # 0 is "never"


def _run(command: list[str]) -> str | None:
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=_PROBE_SECONDS, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout


def probe_gpu() -> GpuState | None:
    """The GPU's memory, from ``nvidia-smi``. ``None`` when it does not answer."""
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    first = (out or "").strip().splitlines()[:1]
    parts = [part.strip() for part in first[0].split(",")] if first else []
    try:
        total, used, free = (float(part) / 1024 for part in parts[1:4])
    except ValueError:
        return None
    if len(parts) != 4:
        return None
    apps = _run(["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"])
    names = {
        PureWindowsPath(line.strip()).name
        for line in (apps or "").splitlines()
        if line.strip() and not _NOT_A_PROGRAM.match(line.strip())
    }
    return GpuState(parts[0], total, used, free, tuple(sorted(names, key=str.lower)))


def probe_power() -> PowerState:
    """Whether the machine is on AC power, and when it sleeps there. Windows only."""
    if sys.platform != "win32":
        return PowerState()
    on_ac = None
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_ubyte),
                ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte),
                ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_ulong),
                ("BatteryFullLifeTime", ctypes.c_ulong),
            ]

        status = Status()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            on_ac = {0: False, 1: True}.get(status.ACLineStatus)
    except (OSError, AttributeError):
        on_ac = None
    out = _run(["powercfg", "/query", "SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE"])
    found = _AC_SLEEP.search(out or "")
    return PowerState(on_ac, int(found.group(1), 16) if found else None)


# --------------------------------------------------------------------------- the preflight


def _result(check_id: str, name: str, status: str, detail: str) -> dict[str, str]:
    return {"id": check_id, "name": name, "status": status, "detail": detail}


def _training_records(runs_dir: Path) -> dict[str, dict[str, Any]]:
    """Every run folder that holds a training record, by run id."""
    found: dict[str, dict[str, Any]] = {}
    if not runs_dir.is_dir():
        return found
    for folder in sorted(runs_dir.iterdir()):
        path = folder / TRAINING_NAME
        if path.is_file():
            try:
                found[folder.name] = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
    return found


def _measured_rate(runs_dir: Path, tracking: Any) -> tuple[float, str] | None:
    """Seconds per training sequence of the newest local run that measured it."""
    for run_id in sorted(_training_records(runs_dir), reverse=True):
        try:
            measured = read_json(runs_dir / run_id / MEASUREMENTS_NAME, "measurements")
            if load_manifest(run_id, tracking).get("where") != LOCAL:
                continue
        except ConfigError:
            continue
        rate = (measured.get("trainer") or {}).get("seconds_per_sequence")
        if isinstance(rate, (int, float)) and rate > 0:
            return float(rate), run_id
    return None


def _fresh(run_dir: Path, now: datetime) -> bool:
    """Whether the run's log, or its folder, was written in the last few minutes."""
    newest = 0.0
    for path in (run_dir / LOG_NAME, run_dir / CHECKPOINTS, run_dir):
        if path.exists():
            newest = max(newest, path.stat().st_mtime)
    return now.timestamp() - newest <= RUNNING_MINUTES * 60


def _check_idle(ctx: TrainingContext, now: datetime) -> dict[str, str]:
    name = "Nothing running"
    tracking, runs_dir = ctx.tracking_cfg, ctx.tracking_cfg.runs_dir
    try:
        runs = [
            entry
            for entry in list_runs(tracking)
            if entry.get("kind") in TRAINING_KINDS and entry.get("where") == LOCAL
        ]
    except TrackingError as error:
        return _result("P6", name, FAIL, f"the run index cannot be read: {error}")
    for entry in runs:
        if entry["status"] == "started" and _fresh(runs_dir / entry["run_id"], now):
            return _result(
                "P6",
                name,
                FAIL,
                f"run {entry['run_id']} is in progress (its files were written in the last "
                f"{RUNNING_MINUTES} minutes): one training run at a time",
            )
    # A full run that stopped is continued, never started again (D3).
    records = _training_records(runs_dir)
    continued = {record.get("continues") for record in records.values()}
    wanted = ctx.training_cfg.settings_sha256(False)
    for entry in runs:  # newest first
        run_id, record = entry["run_id"], records.get(entry["run_id"])
        if entry.get("kind") != "train" or record is None or record.get("smoke"):
            continue
        if entry["status"] == COMPLETED or run_id in continued:
            continue
        steps = checkpoint_steps(runs_dir / run_id / CHECKPOINTS)
        if steps and (record.get("key") or {}).get("settings_sha256") == wanted:
            total = (record.get("plan") or {}).get("total_steps")
            return _result(
                "P6",
                name,
                FAIL,
                f"run {run_id} stopped with a checkpoint at step {steps[-1]:,} of {total:,}: "
                f"continue it, do not start again: {RESUME} {run_id}",
            )
    return _result("P6", name, PASS, "no training run is in progress or waiting to be resumed")


def preflight(
    *,
    context: TrainingContext | None = None,
    gpu: Callable[[], GpuState | None] = probe_gpu,
    power: Callable[[], PowerState] = probe_power,
    disk: Callable[[Any], Any] = shutil.disk_usage,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Check that a full local run can be started. Starts nothing and writes nothing.

    ``gpu``, ``power``, and ``disk`` read the machine; tests pass stand-ins. Returns the
    checks, whether none failed, the plan, and the estimate.
    """
    ctx = (context or TrainingContext()).resolved()
    cfg, tracking = ctx.training_cfg, ctx.tracking_cfg
    root = Path(tracking.root)
    now = now or utc_now()
    checks: list[dict[str, str]] = []
    plan: dict[str, Any] | None = None
    estimate: dict[str, Any] | None = None

    # ---- P1: the plan
    try:
        prepared = prepare(ctx, smoke=False)
    except ConfigError as error:
        checks.append(_result("P1", "The plan", FAIL, str(error)))
    else:
        plan = prepared.plan.to_dict()
        sequences = plan["records"] * plan["epochs"]
        detail = (
            f"{plan['records']:,} records, {plan['epochs']} epochs, {plan['total_steps']:,} "
            f"optimizer steps, {len(plan['checkpoint_steps'])} checkpoints"
        )
        rate = _measured_rate(tracking.runs_dir, tracking)
        if rate is None:
            detail += "; no local run has measured a speed yet, so there is no estimate"
        else:
            hours = sequences * rate[0] / 3600
            estimate = {
                "sequences": sequences,
                "seconds_per_sequence": rate[0],
                "measured_by": rate[1],
                "training_hours": round(hours, 2),
            }
            detail += (
                f"; about {hours:.1f} hours of training at {rate[0]:.2f} s per sequence "
                f"(measured by {rate[1]}), plus the loss on dev at each checkpoint"
            )
        checks.append(_result("P1", "The plan", PASS, detail))

    # ---- P2: a clean tree
    git = ctx.git or read_git_info(root, tracking.git_commit_env, (tracking.ledger_prefix,))
    if git.dirty is None:
        checks.append(
            _result(
                "P2", "A clean tree", FAIL, "git did not answer: the run cannot be tied to a commit"
            )
        )
    elif git.dirty:
        shown = ", ".join(git.changed[:3]) + (" ..." if len(git.changed) > 3 else "")
        checks.append(
            _result(
                "P2",
                "A clean tree",
                FAIL,
                f"uncommitted changes ({shown}): commit first, so the run is a commit's",
            )
        )
    else:
        checks.append(
            _result("P2", "A clean tree", PASS, f"commit {git.commit[:12]}, working tree clean")
        )

    # ---- P3: GPU memory
    state = gpu()
    if state is None:
        checks.append(
            _result("P3", "GPU memory", FAIL, "no NVIDIA GPU answered (nvidia-smi was not found)")
        )
    else:
        held = ", ".join(state.programs[:6]) + (" ..." if len(state.programs) > 6 else "")
        figures = (
            f"{state.free_gib:.1f} GiB free of {state.total_gib:.1f} on {state.name} "
            f"({state.used_gib:.1f} in use)"
        )
        if state.free_gib < cfg.min_free_gpu_gib:
            checks.append(
                _result(
                    "P3",
                    "GPU memory",
                    FAIL,
                    f"{figures}; {cfg.min_free_gpu_gib:g} are asked. Close what holds it"
                    + (f": {held}" if held else "")
                    + ". With less, training spills into shared memory and slows severalfold",
                )
            )
        else:
            checks.append(_result("P3", "GPU memory", PASS, figures))

    # ---- P4: disk
    usage = disk(root)
    free_gb = usage.free / 1e9
    checks.append(
        _result(
            "P4",
            "Disk",
            PASS if free_gb >= cfg.min_free_disk_gb else FAIL,
            f"{free_gb:,.0f} GB free; {cfg.min_free_disk_gb:g} are asked",
        )
    )

    # ---- P5: power
    supply = power()
    if supply.on_ac is None:
        checks.append(_result("P5", "Power", SKIP, "the power state cannot be read on this system"))
    elif not supply.on_ac:
        checks.append(_result("P5", "Power", FAIL, "on battery: plug in before a run of hours"))
    elif supply.sleep_on_ac_seconds is None:
        checks.append(
            _result("P5", "Power", SKIP, "on AC power; the sleep setting could not be read")
        )
    elif supply.sleep_on_ac_seconds > 0:
        minutes = supply.sleep_on_ac_seconds // 60
        checks.append(
            _result(
                "P5",
                "Power",
                FAIL,
                f"on AC power, but the machine sleeps after {minutes:,} minutes: set sleep to "
                "never for the run",
            )
        )
    else:
        checks.append(_result("P5", "Power", PASS, "on AC power; sleep is set to never"))

    checks.append(_check_idle(ctx, now))
    return {
        "checks": checks,
        "passed": all(item["status"] != FAIL for item in checks),
        "plan": plan,
        "estimate": estimate,
    }


# --------------------------------------------------------------------------- following a run


def find_local_run(tracking: Any) -> str | None:
    """The newest full training run made on this machine, or ``None``.

    A run that another run continues is not it: the run that continues it is. (Two
    runs can start in the same second, and their ids then do not say which is later.)
    """
    records = _training_records(tracking.runs_dir)
    continued = {record.get("continues") for record in records.values()}
    for run_id, record in sorted(records.items(), reverse=True):
        if record.get("smoke") or run_id in continued:
            continue
        try:
            if load_manifest(run_id, tracking).get("where") == LOCAL:
                return run_id
        except TrackingError:
            continue
    return None


def _moment(text: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None


def run_status(
    run: str | None = None, *, tracking_cfg: Any = None, now: datetime | None = None
) -> dict[str, Any]:
    """How far a training run is. Without ``run``: the newest full local run.

    Raises ``ConfigError`` when there is no such run. A training log carries no time, so the
    speed of a run in progress is the age of its log over the steps it holds.
    """
    tracking = tracking_cfg or load_tracking_config()
    runs_dir, now = tracking.runs_dir, now or utc_now()
    if run is None:
        run = find_local_run(tracking)
        if run is None:
            raise ConfigError(
                "no full training run has been made on this machine yet: "
                "python -m scripts.qlora_local preflight, then python -m scripts.qlora_train"
            )
    training = read_json(runs_dir / run / TRAINING_NAME, "the training run's record")
    try:
        manifest = load_manifest(run, tracking)
    except TrackingError as error:
        raise ConfigError(f"run {run} is not in the registry here: {error}") from error
    plan = training.get("plan") or {}
    total, start = int(plan.get("total_steps") or 0), int(training.get("start_step") or 0)

    step, last_loss, dev_loss = start, None, None
    log = runs_dir / run / LOG_NAME
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if not isinstance(entry, dict) or "step" not in entry:
                continue
            step = max(step, int(entry["step"]))
            if "loss" in entry:
                last_loss = float(entry["loss"])
            if "eval_loss" in entry:
                dev_loss = {"step": int(entry["step"]), "loss": float(entry["eval_loss"])}

    try:
        measured = read_json(runs_dir / run / MEASUREMENTS_NAME, "measurements")
    except ConfigError:
        measured = {}
    per_step = (measured.get("trainer") or {}).get("seconds_per_step")
    started = _moment(manifest.get("started_at"))
    done_here = step - start
    if per_step is None and done_here > 0 and started is not None and log.is_file():
        elapsed = log.stat().st_mtime - started.timestamp()
        per_step = elapsed / done_here if elapsed > 0 else None

    status = manifest.get("status")
    if status == "started":
        state = RUNNING if _fresh(runs_dir / run, now) else INTERRUPTED
    else:
        state = COMPLETED if status == COMPLETED else FAILED
    continued_by = next(
        (
            other
            for other, record in sorted(_training_records(runs_dir).items())
            if record.get("continues") == run
        ),
        None,
    )
    left = max(total - step, 0)
    resumable = state in (INTERRUPTED, FAILED) and continued_by is None and left > 0
    return {
        "run": run,
        "smoke": bool(training.get("smoke")),
        "where": manifest.get("where"),
        "chain": training_chain(run, runs_dir),
        "continued_by": continued_by,
        "state": state,
        "step": step,
        "start_step": start,
        "total": total,
        "share": step / total if total else 0.0,
        "seconds_per_step": per_step,
        "hours_left": left * per_step / 3600 if per_step and state == RUNNING else None,
        "hours_to_finish_if_resumed": left * per_step / 3600 if per_step and resumable else None,
        "last_loss": last_loss,
        "last_dev_loss": dev_loss,
        "checkpoints": checkpoint_steps(runs_dir / run / CHECKPOINTS),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "resume_command": f"{RESUME} {run}" if resumable else None,
        "running_for": (
            str(timedelta(seconds=int(now.timestamp() - started.timestamp())))
            if state == RUNNING and started is not None
            else None
        ),
    }
