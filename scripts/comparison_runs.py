"""The six test runs: check after the unlock, see where they stand, record them (req_005 v5.01).

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.comparison_runs preflight
    venv\\Scripts\\python -m scripts.comparison_runs status
    venv\\Scripts\\python -m scripts.comparison_runs record
    venv\\Scripts\\python -m scripts.comparison_runs void --run <run id> --reason "<why>" --yes

``preflight`` is the free check after the unlock: the ``gold`` predictor answers every test
example with its own gold reply, which must score all of them right and pass R1 to R5, and
the test prompts are counted in Qwen tokens (they could not be counted while the split was
locked). No model is asked and nothing is billed. While the split is locked it answers
``LOCKED``.

``status`` says, per arm, whether its test run is done, unfinished (with the ``--resume``
command), or missing. It reads only.

``record`` runs checks C1 to C8 over the six runs and, when all pass, writes the committed
six-arm record ``reports/comparison/six_arm_test_runs.json`` and ``.md``: run ids, counts,
tokens, billable seconds. With an arm missing or a check failed it writes nothing. Later
versions take the six run ids from this record.

**``void`` is the owner's.** It voids a test run that a fault in the pipeline or the service
made worthless, never one whose answers disappoint, so that the arm may answer the test
split once more. The reason is committed (``splits/test_run_voids.json``) and shown in the
six-arm record as a deviation.

None of the four calls AWS or needs a sign-in. ``record`` needs the six runs' folders on
this machine: a SageMaker job's runs arrive with ``sagemaker_job pull``.

Exit codes: 0 done (``status``: all six arms done); 1 a check failed, an arm is not done,
or the gold check did not score every example; 2 a config problem, a locked split, a
refusal, or ``--yes`` missing.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

try:
    from qlora_finetuning.comparison.config import ComparisonConfig, load_comparison_config
    from qlora_finetuning.comparison.freeze import current_arms
    from qlora_finetuning.comparison.gate import TestRunRefused
    from qlora_finetuning.comparison.runs import (
        DONE,
        arm_status,
        build_record,
        preflight,
        void_run,
        write_record,
    )
    from qlora_finetuning.data import TestLockedError
    from qlora_finetuning.evaluation.runner import RunnerContext
    from qlora_finetuning.local_qlora.job_record import read_job_record
    from qlora_finetuning.local_qlora.sagemaker_config import load_sagemaker_config
    from qlora_finetuning.logging_setup import setup_logging
    from qlora_finetuning.settings import ConfigError
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.comparison_runs"
    ) from None

SELF = "python -m scripts.comparison_runs"
PASS, FAIL = "PASS", "FAIL"


def _ascii(text: Any) -> str:
    return str(text).encode("ascii", "replace").decode()


def _relative(path: Any, cfg: ComparisonConfig) -> str:
    try:
        return path.relative_to(cfg.root).as_posix()
    except ValueError:
        return str(path)


def print_status(status: list[dict[str, Any]]) -> int:
    done = 0
    for entry in status:
        run = entry["results"][0] if entry["results"] else (entry["resume"] or "")
        print(f"{entry['arm']:<8} {entry['predictor']:<16} {entry['state']:<10} {run}")
        if entry["resume"]:
            print(f"         continue it with --resume {entry['resume']}")
        done += entry["state"] == DONE
    print(f"{done} of {len(status)} arms have their test result")
    return done


def print_record(record: dict[str, Any]) -> None:
    for check in record["checks"]:
        status = PASS if check["passed"] else FAIL
        print(f"{status}  {check['id']}  {check['name']}: {_ascii(check['detail'])}")
    failed = sum(not check["passed"] for check in record["checks"])
    print(f"{len(record['checks']) - failed} passed, {failed} failed")
    for arm, entry in record["arms"].items():
        every = entry["all"]
        right = every["execution_accuracy"]
        print(
            f"{arm:<8} execution accuracy {right['count']:,} of {every['examples']:,} "
            f"({right['percent']}%); in the format {every['format_validity']['count']:,}  "
            f"(run {entry['run_id']})"
        )
    for void in record["deviations"]:
        print(
            f"DEVIATION  {void['arm']}: {len(void['runs'])} run(s) voided: {_ascii(void['reason'])}"
        )


def main(
    argv: list[str] | None = None,
    context: RunnerContext | None = None,
    cfg: ComparisonConfig | None = None,
    **stand_ins: Any,
) -> int:
    parser = argparse.ArgumentParser(description="The six test runs of the comparison.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="after the unlock: gold on test, prompt lengths; free")
    commands.add_parser("status", help="which arms have their test result")
    commands.add_parser("record", help="checks C1 to C8; write the six-arm record")
    voiding = commands.add_parser("void", help="THE OWNER: void a run that a fault spoiled")
    voiding.add_argument("--run", required=True, metavar="RUN_ID", help="the test run to void")
    voiding.add_argument("--reason", required=True, help="what went wrong, in a sentence")
    voiding.add_argument("--yes", action="store_true", help="required: the arm may run again")
    args = parser.parse_args(argv)

    setup_logging()
    try:
        context = (context or RunnerContext()).resolved()
        cfg = cfg or load_comparison_config(root=context.tracking_cfg.root)

        if args.command == "preflight":
            outcome = preflight(context, stand_ins.get("counter"))
            gold, lengths = outcome["gold"], outcome["lengths"]
            every = gold["all"]
            for check in gold["checks"]:
                status = PASS if check["passed"] else FAIL
                print(f"{status}  {check['id']}  {check['name']}: {_ascii(check['detail'])}")
            right = every["execution_accuracy"]["count"]
            print(f"gold on test: {right:,} of {every['examples']:,} right on execution")
            if lengths["counted"]:
                print(
                    f"test sequences in Qwen tokens: {lengths['records']:,} counted, the "
                    f"longest {lengths['longest']:,}; {lengths['over_max_length']:,} over the "
                    f"training length of {lengths['max_length']:,}"
                )
                if lengths["over_max_length"]:
                    print(
                        "NOTE: nothing is cut at inference, but the adapter never saw a "
                        "sequence that long: a caveat for the report"
                    )
            else:
                print(f"test prompts not counted: {_ascii(lengths['reason'])}")
            ok = gold["result"] and right == every["examples"]
            print("PREFLIGHT PASSED: the runner works on the test split" if ok else
                  "PREFLIGHT FAILED: do not start a billed run")  # fmt: skip
            return 0 if ok else 1

        if args.command == "status":
            status = arm_status(cfg, context)
            return 0 if print_status(status) == len(status) else 1

        if args.command == "void":
            entry = void_run(cfg, context, args.run, args.reason, yes=args.yes)
            print(f"VOIDED: {', '.join(entry['runs'])}  (arm {entry['arm']})")
            print(f"reason: {_ascii(entry['reason'])}")
            print(f"arm {entry['arm']} may answer the test split once more")
            print(f"commit it: git add {_relative(cfg.voids_file, cfg)}")
            return 0

        arms_now = stand_ins.get("arms_now")
        if arms_now is None:
            arms_now, _ = current_arms(cfg, context)
        job_record = stand_ins.get("job_record")
        if job_record is None:
            job_record = read_job_record(load_sagemaker_config(cfg.root).record_file)
        record = build_record(
            cfg, context, job_record=job_record, arms_now=arms_now, git=stand_ins.get("git")
        )
    except TestLockedError as exc:
        print(f"LOCKED: {_ascii(exc)}")
        return 2
    except TestRunRefused as exc:
        print(f"REFUSED: {_ascii(exc)}")
        return 2
    except ConfigError as exc:  # bad config, no freeze record, an unknown run, --yes missing
        print(f"CONFIG ERROR: {_ascii(exc)}")
        return 2

    print_record(record)
    if not record["passed"]:
        print("NOT WRITTEN: the six-arm record is written when all six arms pass C1 to C8")
        print(f"see: {SELF} status")
        return 1
    markdown_path, json_path = write_record(record, cfg)
    print(f"record: {_relative(json_path, cfg)}")
    print(f"record: {_relative(markdown_path, cfg)}")
    print("commit both, with the ledger: milestone M7 is reached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
