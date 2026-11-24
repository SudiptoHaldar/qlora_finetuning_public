"""The optional full training run on this machine: check, follow, and record it (req_003 v3.06).

A local full run is the owner's learning exercise. **It is a side note: not an arm of the
comparison, never arm A1, and without a price.** The run itself is
``python -m scripts.qlora_train``; this script stands around it.

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.qlora_local preflight
    venv\\Scripts\\python -m scripts.qlora_local status
    venv\\Scripts\\python -m scripts.qlora_local status --run <run_id>
    venv\\Scripts\\python -m scripts.qlora_local note --run <run_id>
    venv\\Scripts\\python -m scripts.qlora_local note --run <run_id> --check

``preflight`` starts nothing. It checks the plan (P1), a clean working tree (P2), free GPU
memory (P3: it names the programs that hold memory, so they can be closed), disk (P4),
power (P5), and that no training run is in progress or waiting to be resumed (P6).

``status`` says how far a run is: the step, the hours left, the last loss, and, for a run
that stopped, the command that continues it. Without ``--run`` it takes the newest full run
made here. Run it from a second terminal while training.

``note`` writes the committed record of the run, ``reports/local_qlora/local_run.json`` and
``.md``, when checks N1 to N5 pass: every checkpoint scored, every run local, a full run,
the selected checkpoint scored on all of dev, every score current. ``--split dev`` is for a
run whose checkpoints were all scored on dev. ``--check`` writes nothing.

The whole exercise, about five hours::

    venv\\Scripts\\python -m scripts.qlora_local preflight
    venv\\Scripts\\python -m scripts.qlora_train
    venv\\Scripts\\python -m scripts.job_run score --run <run_id> --split quick_dev
    venv\\Scripts\\python -m scripts.qlora_select --run <run_id> --split quick_dev --check
    venv\\Scripts\\python -m scripts.evaluate --predictor qwen3-4b-tuned \\
        --adapter <run_id>/checkpoint-<N> --split dev
    venv\\Scripts\\python -m scripts.qlora_local note --run <run_id>

Nothing here is billed, and nothing reaches AWS.

Exit codes: 0; 1 a check failed; 2 a config problem or an unknown run.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

try:
    from qlora_finetuning.local_qlora.local_note import DEV, QUICK_DEV, local_note
    from qlora_finetuning.local_qlora.local_run import (
        FAIL,
        LABEL,
        PASS,
        SKIP,
        preflight,
        run_status,
    )
    from qlora_finetuning.logging_setup import setup_logging
    from qlora_finetuning.settings import ConfigError
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.qlora_local"
    ) from None


def _only(stand_ins: dict[str, Any], *names: str) -> dict[str, Any]:
    return {name: stand_ins[name] for name in names if name in stand_ins}


def _preflight(stand_ins: dict[str, Any]) -> int:
    renamed = dict(stand_ins)
    if "training_context" in renamed:
        renamed["context"] = renamed.pop("training_context")
    outcome = preflight(**_only(renamed, "context", "gpu", "power", "disk", "now"))
    print(f"preflight for the full run on this machine  ({LABEL})")
    for check in outcome["checks"]:
        print(f"{check['status']}  {check['id']}  {check['name']}: {check['detail']}")
    counts = {
        name: sum(c["status"] == name for c in outcome["checks"]) for name in (PASS, FAIL, SKIP)
    }
    print(f"{counts[PASS]} passed, {counts[FAIL]} failed, {counts[SKIP]} skipped")
    if outcome["passed"]:
        print("READY: nothing was started. Start the run: python -m scripts.qlora_train")
        print("follow it from another terminal: python -m scripts.qlora_local status")
    else:
        print("NOT READY: a check failed; nothing was started")
    return 0 if outcome["passed"] else 1


def _status(args: argparse.Namespace, stand_ins: dict[str, Any]) -> int:
    status = run_status(args.run, **_only(stand_ins, "tracking_cfg", "now"))
    kind = "smoke run" if status["smoke"] else "full run"
    print(f"training run {status['run']}  ({kind}; {status['where']}; {LABEL})")
    if len(status["chain"]) > 1:
        print(f"continues: {' -> '.join(status['chain'][:-1])}")
    print(
        f"step {status['step']:,} of {status['total']:,}  ({100 * status['share']:.1f}%); "
        f"state: {status['state']}"
    )
    if status["seconds_per_step"]:
        print(f"about {status['seconds_per_step']:.1f} s per optimizer step")
    if status["hours_left"] is not None:
        print(f"about {status['hours_left']:.1f} hours left; running for {status['running_for']}")
    if status["last_loss"] is not None:
        print(f"last training loss: {status['last_loss']:.4f}")
    if status["last_dev_loss"]:
        dev = status["last_dev_loss"]
        print(f"loss on dev at step {dev['step']:,}: {dev['loss']:.4f}")
    if status["checkpoints"]:
        steps = ", ".join(f"{step:,}" for step in status["checkpoints"])
        print(f"checkpoints at steps: {steps}")
    if status["continued_by"]:
        print(f"continued by run {status['continued_by']}")
    if status["resume_command"]:
        left = status["hours_to_finish_if_resumed"]
        more = f" (about {left:.1f} hours more)" if left is not None else ""
        print(f"it stopped; continue it{more}: {status['resume_command']}")
    return 0


def _note(args: argparse.Namespace, stand_ins: dict[str, Any]) -> int:
    allowed = _only(stand_ins, "context", "training_cfg", "sagemaker_cfg", "budget")
    outcome = local_note(args.run, split=args.split, check=args.check, **allowed)
    print(f"training run {outcome['run']}  (checkpoints scored on {outcome['split']}; {LABEL})")
    for check in outcome["checks"]:
        status = PASS if check["passed"] else FAIL
        print(f"{status}  {check['id']}  {check['name']}: {check['detail']}")
    failed = sum(not check["passed"] for check in outcome["checks"])
    print(f"{len(outcome['checks']) - failed} passed, {failed} failed")
    note = outcome["note"]
    if note is not None:
        selected, dev = note["selected"], note["selected_on_dev"]
        print(
            f"selected here: checkpoint {selected['step']:,}, {selected['execution_accuracy']:,} "
            f"of {selected['examples']:,} right on {note['selection_split']}"
        )
        print(
            f"on all of dev: {dev['execution_accuracy']:,} of {dev['examples']:,} right on "
            f"execution, {dev['format_validity']:,} in the format"
        )
        if note["hours"] is not None:
            print(f"training took {note['hours']:.2f} hours")
        cloud = note["against_cloud"]
        if cloud is None:
            print("local against cloud: not yet (no cloud checkpoint is selected)")
        elif cloud["execution_gap_examples"] is not None:
            noise = "within noise" if cloud["within_noise"] else "more than noise"
            print(
                f"local against cloud on dev: {cloud['execution_gap_examples']:+,} examples "
                f"({noise})"
            )
    if outcome["written"]:
        print("WRITTEN: the local note; commit it: git add reports/local_qlora ledger")
    elif outcome["passed"]:
        print("CHECK ONLY: all checks pass; nothing was written")
    else:
        print("NOT WRITTEN: a check failed; nothing was written")
    return 0 if outcome["passed"] else 1


def main(argv: list[str] | None = None, **stand_ins: Any) -> int:
    parser = argparse.ArgumentParser(
        description="The optional full training run on this machine. A side note, not an arm."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="check that the run can be started; starts nothing")
    following = commands.add_parser("status", help="how far a run is")
    following.add_argument("--run", metavar="RUN_ID", help="default: the newest full local run")
    noting = commands.add_parser("note", help="write the committed record of the run")
    noting.add_argument("--run", required=True, metavar="RUN_ID", help="the training run")
    noting.add_argument(
        "--split",
        default=QUICK_DEV,
        choices=(QUICK_DEV, DEV),
        help="the split the checkpoints were scored on (default: quick_dev)",
    )
    noting.add_argument("--check", action="store_true", help="report only; write nothing")
    args = parser.parse_args(argv)

    setup_logging()
    try:
        if args.command == "preflight":
            return _preflight(stand_ins)
        if args.command == "status":
            return _status(args, stand_ins)
        return _note(args, stand_ins)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
