"""Freeze the configuration of the six arms, and unlock the test split (req_005 v5.01).

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.comparison_freeze check
    venv\\Scripts\\python -m scripts.comparison_freeze freeze
    venv\\Scripts\\python -m scripts.comparison_freeze status
    venv\\Scripts\\python -m scripts.comparison_freeze unlock --yes --freeze <12 characters>

``check`` runs checks F1 to F8 and prints the estimate of the six test passes. It writes
nothing. ``freeze`` does the same and, when every check passes, writes the freeze record
``splits/config_freeze.json`` and its Markdown copy ``reports/comparison/config_freeze.md``,
and prints the record's SHA-256. Commit both files. The freeze can be written again as long
as the test split is locked; after the unlock it is never rewritten.

``status`` says whether the configuration is frozen, whether the test split is unlocked,
and whether anything frozen has changed since.

**``unlock`` is run by the owner, once.** ``--freeze`` takes the first twelve characters (or
more) of the freeze record's SHA-256: typing them is the sign-off that this record was read
and accepted. It needs the freeze record committed and a clean working tree, writes
``splits/test_unlock.json``, and that file is then committed too: a SageMaker job runs a
commit, so the unlock record must be in it. There is no command to lock again.

None of the four calls AWS or needs a sign-in. The dev result runs of the six arms must be
on this machine (``artifacts/runs/``), with the snapshot.

Exit codes: 0 done (for ``check``: every check passes; for ``status``: nothing frozen has
changed); 1 a check failed or the configuration changed since the freeze; 2 a config
problem, a refusal, or ``--yes`` or ``--freeze`` missing.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

try:
    from qlora_finetuning.comparison.config import ComparisonConfig, load_comparison_config
    from qlora_finetuning.comparison.freeze import (
        MIN_PREFIX,
        REPORT_NAME,
        Sources,
        build_freeze,
        freeze_digest,
        freeze_status,
        load_sources,
        unlock,
        write_freeze,
    )
    from qlora_finetuning.data.split import load_split_config
    from qlora_finetuning.evaluation.runner import RunnerContext
    from qlora_finetuning.logging_setup import setup_logging
    from qlora_finetuning.settings import ConfigError
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.comparison_freeze"
    ) from None

SELF = "python -m scripts.comparison_freeze"
PASS, FAIL = "PASS", "FAIL"


def _ascii(text: Any) -> str:
    return str(text).encode("ascii", "replace").decode()


def _relative(path: Any, cfg: ComparisonConfig) -> str:
    try:
        return path.relative_to(cfg.root).as_posix()
    except ValueError:
        return str(path)


def print_record(record: dict[str, Any]) -> None:
    for check in record["checks"]:
        status = PASS if check["passed"] else FAIL
        print(f"{status}  {check['id']}  {check['name']}: {_ascii(check['detail'])}")
    failed = sum(not check["passed"] for check in record["checks"])
    print(f"{len(record['checks']) - failed} passed, {failed} failed")
    for arm, entry in record["arms"].items():
        dev = entry.get("dev")
        score = (
            f"{dev['execution_accuracy']:,} of {dev['examples']:,} right on dev"
            if dev
            else "dev run not read"
        )
        print(f"{arm:<8} {entry['predictor']['name']:<16} {score}  (run {entry.get('dev_run')})")
    estimate = record.get("estimate") or {}
    for arm, entry in (estimate.get("arms") or {}).items():
        print(f"estimate  {arm:<8} {entry['usd']:>7}  at most {entry['at_most_usd']:>7}")
    if estimate.get("total_usd"):
        print(
            f"estimate  all six  {estimate['total_usd']:>7}  at most {estimate['at_most_usd']:>7}"
            "  (test prompts assumed as long as dev's)"
        )
    for name, entry in (estimate.get("paths") or {}).items():
        print(
            f"Path {name}: {entry['spent_usd']} spent, at most {entry['at_most_usd']} more, "
            f"cap {entry['cap_usd']}"
        )


def main(
    argv: list[str] | None = None,
    context: RunnerContext | None = None,
    cfg: ComparisonConfig | None = None,
    sources: Sources | None = None,
    **stand_ins: Any,
) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the six arms' configuration; unlock the test split."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="checks F1 to F8 and the estimate; writes nothing")
    commands.add_parser("freeze", help="write the freeze record, if every check passes")
    commands.add_parser("status", help="frozen? unlocked? does the freeze still hold?")
    opening = commands.add_parser("unlock", help="THE OWNER: unlock the test split, once")
    opening.add_argument("--yes", action="store_true", help="required: this cannot be undone")
    opening.add_argument(
        "--freeze",
        metavar="SHA256",
        help=f"the first {MIN_PREFIX} characters or more of the freeze record's SHA-256",
    )
    args = parser.parse_args(argv)

    setup_logging()
    try:
        context = (context or RunnerContext()).resolved()
        cfg = cfg or load_comparison_config(root=context.tracking_cfg.root)
        sources = sources or load_sources(cfg, context)

        if args.command == "status":
            status = freeze_status(cfg, context, sources)
            digest = status["freeze_sha256"]
            print(
                f"frozen:    {'yes, SHA-256 ' + digest if status['frozen'] else 'no'}"
                + ("" if status["frozen"] else f"  (run {SELF} freeze)")
            )
            print(f"test split: {'UNLOCKED' if status['unlocked'] else 'locked'}: {status['lock']}")
            if status["unlocked"]:
                names = (
                    "names this freeze record"
                    if status["unlock_names_freeze"]
                    else ("DOES NOT name the freeze record on disk")
                )
                print(
                    f"unlock:    {status['unlocked_on']}, freeze commit "
                    f"{str(status['freeze_commit'])[:12]}; {names}"
                )
            for difference in status["differences"]:
                print(f"CHANGED    {_ascii(difference)}")
            if status["frozen"] and not status["differences"]:
                print("nothing frozen has changed")
            wrong = status["differences"] or (
                status["unlocked"] and not status["unlock_names_freeze"]
            )
            return 1 if wrong else 0

        if args.command == "unlock":
            record = unlock(cfg, context, sources, yes=args.yes, freeze=args.freeze, **stand_ins)
            print(f"UNLOCKED: the test split, on {record['unlocked_on']}")
            print(f"freeze record {record['freeze_sha256']}")
            print(f"freeze commit {record['freeze_commit']}")
            unlock_file = (context.split_cfg or load_split_config()).unlock_file
            print(f"written: {_relative(unlock_file, cfg)}")
            print(f"commit it now: git add {_relative(unlock_file, cfg)}, then git commit")
            print("then: python -m scripts.comparison_runs preflight")
            return 0

        record = build_freeze(cfg, context, sources)
    except ConfigError as exc:  # bad config, no snapshot, a refusal, --yes missing
        print(f"CONFIG ERROR: {_ascii(exc)}")
        return 2

    print_record(record)
    if args.command == "check":
        print(f"would be SHA-256 {freeze_digest(record)}")
        print("CHECK ONLY: nothing was written")
        return 0 if record["passed"] else 1
    if not record["passed"]:
        print("NOT WRITTEN: a check failed, so no freeze record was written")
        return 1
    try:
        digest = write_freeze(record, cfg, context.split_cfg)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {_ascii(exc)}")
        return 2
    print(f"freeze record: {_relative(cfg.freeze_file, cfg)}")
    print(f"for reading:   {_relative(cfg.report_dir / REPORT_NAME, cfg)}")
    print(f"SHA-256: {digest}")
    print("read it, then commit both files. To unlock (the owner, once):")
    print(f"  {SELF} unlock --yes --freeze {digest[:MIN_PREFIX]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
