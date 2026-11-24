"""The comparison report: intervals, paired tests, error categories, the report itself
(req_005 v5.03).

Run from the repository root::

    venv\\Scripts\\python -m scripts.comparison_report check
    venv\\Scripts\\python -m scripts.comparison_report build
    venv\\Scripts\\python -m scripts.comparison_report examples

``check`` reads the six test runs the six-arm record names, rescored from their stored
replies, runs checks R1 to R8, and prints the main figures. It writes nothing. ``build``
does the same and writes ``reports/comparison/comparison_statistics.json`` and ``.md``,
``reports/comparison/accuracy_cost.svg``, ``reports/comparison_report.md``, and the block
between the markers in ``README.md``; it writes only when every check passes. A check may
pass **with a reservation** (the cost record is provisional): the report is then written
with "provisional" in its title.

``examples`` writes the question, the gold reply, and the arm's reply of the examples the
statistics record names to ``artifacts/comparison/error_examples.md``. That file holds
dataset text: it is git-ignored and is never committed.

None of the three calls AWS, needs a sign-in, or asks a model. They read test rows on this
machine, which the execution plan's principle 6 allows for this version.

Exit codes: 0 done (a reservation is not a failure); 1 a check failed, so nothing was
written; 2 a config problem or a missing record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from qlora_finetuning.comparison.config import load_comparison_config
    from qlora_finetuning.comparison.gate import file_sha256, read_freeze, shared_state
    from qlora_finetuning.comparison.outcomes import PRIMARY, read_arms
    from qlora_finetuning.comparison.report_build import build_outputs, write_outputs
    from qlora_finetuning.comparison.report_config import (
        ReportConfig,
        load_deviations,
        load_report_config,
    )
    from qlora_finetuning.comparison.report_examples import write_examples
    from qlora_finetuning.comparison.report_record import ReportInputs, build_statistics
    from qlora_finetuning.comparison.report_tables import shown_p, signed, span
    from qlora_finetuning.data import split_ids
    from qlora_finetuning.evaluation.runner import RunnerContext
    from qlora_finetuning.logging_setup import setup_logging
    from qlora_finetuning.settings import REPO_ROOT, ConfigError
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: "
        "python -m scripts.comparison_report"
    ) from None

SELF = "python -m scripts.comparison_report"
PASS, FAIL = "PASS", "FAIL"


def _ascii(text: Any) -> str:
    return str(text).encode("ascii", "replace").decode()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _read_json(path: Path, what: str) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ConfigError(f"{what} cannot be read: {exc}") from None


def gather(cfg: ReportConfig, root: Path, context: Any = None) -> ReportInputs:
    """Read the committed records and each arm's stored replies."""
    ctx = (context or RunnerContext()).resolved()
    comparison = ctx.comparison_cfg or load_comparison_config(root=root)
    six_arm = _read_json(cfg.six_arm_file, "the six-arm record")
    runs = {
        arm: str(entry.get("run_id"))
        for arm, entry in ((six_arm or {}).get("arms") or {}).items()
    }
    arms, problems = read_arms(runs, ctx) if runs else ({}, [])
    return ReportInputs(
        six_arm=six_arm,
        cost=_read_json(cfg.cost_file, "the cost record"),
        freeze=read_freeze(comparison),
        freeze_sha256=file_sha256(comparison.freeze_file),
        shared_now=shared_state(comparison, ctx.config_dir, ctx.config.flags_file),
        arms=arms,
        read_problems=problems,
        tuned=tuple(arm for arm, entry in comparison.arms.items() if entry.tuned),
        test_ids=tuple(split_ids("test", ctx.split_cfg)),
    )


def print_record(record: dict[str, Any], cfg: ReportConfig) -> None:
    for check in record["checks"]:
        status = PASS if check["passed"] else FAIL
        print(f"{status}  {check['id']}  {check['name']}: {_ascii(check['detail'])}")
    failed = sum(not check["passed"] for check in record["checks"])
    reserved = sum(check["reservation"] for check in record["checks"])
    print(
        f"{len(record['checks']) - failed} passed, {failed} failed, {reserved} with a reservation"
    )
    for arm, entry in (record.get("arms") or {}).items():
        metric = entry["metrics"][PRIMARY]
        print(
            f"arm  {arm:<8} execution accuracy {metric['percent']:>6}%  "
            f"({span(metric['interval'])}; by company "
            f"{span(entry['execution_accuracy_by_company'])})"
        )
    for pair in record.get("pairs") or []:
        verdict = "settled" if pair["settled"] else "NOT SETTLED"
        print(
            f"pair {pair['pair'][0]} minus {pair['pair'][1]:<8} "
            f"{signed(pair['difference_points']):>7} points  "
            f"({span(pair['interval'])}; by company {span(pair['interval_by_company'])})  "
            f"p adjusted {shown_p(pair['p_adjusted'], cfg.smallest_p)}  {verdict}"
        )
    overlap = record.get("overlap")
    if overlap:
        print(
            f"tuned arms: {overlap['all_tuned_right']:,} right in all, "
            f"{overlap['any_tuned_right']:,} in at least one; "
            f"{overlap['no_arm_right']:,} right in no arm"
        )


def main(argv: list[str] | None = None, root: Path | str | None = None, **stand_ins: Any) -> int:
    parser = argparse.ArgumentParser(description="The comparison report.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="R1 to R8 and the main figures; writes nothing")
    commands.add_parser("build", help="R1 to R8; write the record, the chart, the report")
    commands.add_parser("examples", help="write example text to the git-ignored artifacts/")
    args = parser.parse_args(argv)
    setup_logging()
    root = Path(root or REPO_ROOT)
    context = stand_ins.get("context")
    try:
        cfg = stand_ins.get("cfg") or load_report_config(root)
        report_dir = Path(
            stand_ins.get("report_dir") or load_comparison_config(root=root).report_dir
        )
        if args.command == "examples":
            stored = _read_json(
                report_dir / f"{cfg.statistics_stem}.json", "the statistics record"
            )
            if stored is None:
                raise ConfigError(f"there is no statistics record yet: run {SELF} build")
            path = write_examples(stored, cfg, context)
            print(f"written: {_relative(path, root)}")
            print("it holds dataset text: it is git-ignored; do not commit or copy it")
            return 0
        inputs = stand_ins.get("inputs") or gather(cfg, root, context)
        deviations = stand_ins.get("deviations")
        if deviations is None:
            deviations = load_deviations()
        record, outputs = build_outputs(
            build_statistics(inputs, cfg),
            inputs.cost or {},
            deviations,
            cfg,
            report_dir,
            inputs.test_ids,
        )
    except ConfigError as exc:
        print(f"CONFIG ERROR: {_ascii(exc)}")
        return 2

    print_record(record, cfg)
    if not record["passed"]:
        print("NOT WRITTEN: the report is written when R1 to R8 pass")
        return 1
    if record["provisional"]:
        print("PROVISIONAL: a check passed with a reservation; see the RESERVATION lines above")
    if args.command == "check":
        print("CHECK ONLY: nothing was written")
        return 0
    for path in write_outputs(record, outputs):
        print(f"written: {_relative(path, root)}")
    print("read the report, then commit these files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
