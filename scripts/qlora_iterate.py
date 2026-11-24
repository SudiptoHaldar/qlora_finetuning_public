"""Register the one corrective change of Path A, and see where it stands (req_003 v3.05).

If ``scripts.qlora_select`` gave the verdict ``missed``, one training setting may be changed
and the run made once more. The change is written down, with its reason, **before** the
second run is paid for::

    reports/local_qlora/iteration.json

Usage (from the repo root; the script imports the package, so it runs as a module)::

    # first: python -m scripts.qlora_analyze --run <first run>, and read its finding
    # then edit ONE value in config/training.yaml, and:
    venv\\Scripts\\python -m scripts.qlora_iterate propose --finding T2 --reason "peaks at step 300"
    venv\\Scripts\\python -m scripts.qlora_iterate propose --finding T2 --reason "..." --check
    venv\\Scripts\\python -m scripts.qlora_iterate status

It costs nothing and uses no GPU and no AWS. **It starts nothing**: the second training job
and its scoring job are launched by the owner with ``scripts.sagemaker_job``, on the first
training job's workspace. The launcher's check J10 refuses a second full training run that
is not registered here, and any run after it.

``--finding`` is the row of the table in the v3.05 spec that the change answers: T1 (more
epochs), T2 (a lower learning rate), T3 (a higher LoRA rank), T4 (a format miss, read as T1
or T2), or ``other`` for a change of your own within the allowed variables: ``epochs``,
``optimizer.learning_rate``, ``lora.r`` (with ``lora.alpha`` at twice it), ``lora.dropout``,
``schedule.warmup_fraction``. Nothing else may differ from the first run.

Checks I1 to I5: arm A1 is selected and missed; the error analysis exists; exactly one
allowed variable differs; no second run has been made; the experiment is otherwise the
first run's. The record is written only if all five pass.

``status`` says where the iteration stands and, once it is over, checks that
``config/training.yaml`` holds the settings of the run that was selected.

Exit codes: 0; 1 a check failed; 2 a config problem.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

try:
    from qlora_finetuning.local_qlora.iteration import (
        FINDINGS,
        NOTHING_SELECTED,
        iteration_status,
        propose,
    )
    from qlora_finetuning.logging_setup import setup_logging
    from qlora_finetuning.settings import ConfigError
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.qlora_iterate"
    ) from None

PASS, FAIL = "PASS", "FAIL"


def _print_checks(checks: list[dict[str, Any]]) -> None:
    for check in checks:
        status = PASS if check["passed"] else FAIL
        print(f"{status}  {check['id']}  {check['name']}: {check['detail']}")
    failed = sum(not check["passed"] for check in checks)
    print(f"{len(checks) - failed} passed, {failed} failed")


def _propose(args: argparse.Namespace, stand_ins: dict[str, Any]) -> int:
    outcome = propose(args.finding, args.reason or "", check=args.check, **stand_ins)
    _print_checks(outcome["checks"])
    change = outcome["change"]
    if change:
        print(f"change: {change['variable']} from {change['from']} to {change['to']}")
    if outcome["written"]:
        record = outcome["record"]
        if not record["in_table"]:
            print(f"note: this is not the change the table names for {record['finding']}")
        if record["analysis_finding"] not in (None, record["finding"]):
            print(f"note: the analysis read {record['analysis_finding']}")
        print("REGISTERED: commit it (config/training.yaml and reports/local_qlora), then plan:")
        print(
            "python -m scripts.sagemaker_job plan train --scope measured "
            "--workspace <first training job>"
        )
    elif outcome["passed"]:
        print("CHECK ONLY: all checks pass; nothing was written")
    else:
        print("NOT REGISTERED: a check failed; nothing was written")
    return 0 if outcome["passed"] else 1


def _status(stand_ins: dict[str, Any]) -> int:
    allowed = {name: stand_ins[name] for name in ("context", "training_cfg", "sagemaker_cfg")
               if name in stand_ins}  # fmt: skip
    status = iteration_status(**allowed)
    print(f"state: {status['state']}")
    if status["state"] == NOTHING_SELECTED:
        print("no checkpoint is selected yet: there is nothing to iterate on")
        return 0
    selection, iteration = status["selection"], status["iteration"]
    print(f"selected: {selection['selected']['adapter']}  (verdict {selection['verdict']})")
    if iteration is not None:
        print(
            f"change: {iteration['variable']} from {iteration['from']} to {iteration['to']}  "
            f"(finding {iteration['finding']}; registered {iteration.get('proposed_on')})"
        )
        print(f"reason: {iteration['reason'].encode('ascii', 'backslashreplace').decode()}")
        outcome = iteration.get("outcome")
        if outcome:
            noise = "within noise" if outcome["within_noise"] else "more than noise"
            print(
                f"second run: {iteration['second_run']}; the {outcome['winner']} run's checkpoint "
                f"is arm A1 ({outcome['gap_examples']:+,} examples, {noise})"
            )
    _print_checks(status["checks"])
    return 0 if status["passed"] else 1


def main(argv: list[str] | None = None, **stand_ins: Any) -> int:
    parser = argparse.ArgumentParser(
        description="Register the one corrective change of Path A. Starts nothing."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    proposing = commands.add_parser("propose", help="check and register the change")
    proposing.add_argument("--finding", required=True, choices=FINDINGS)
    proposing.add_argument("--reason", required=True, help="why this change, in your own words")
    proposing.add_argument("--check", action="store_true", help="check only; write nothing")
    commands.add_parser("status", help="where the iteration stands")
    args = parser.parse_args(argv)

    setup_logging()
    try:
        if args.command == "propose":
            return _propose(args, stand_ins)
        return _status(stand_ins)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
