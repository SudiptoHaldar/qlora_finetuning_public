"""Compare the fine-tuned Nova models with their untuned baselines (req_004 v4.04).

Give it the run ids of the tuned runs (arms B1) and of the baseline runs they are compared
with (arms B0): for each model, quick-dev and dev, in any order. It computes each run's
summary again from its predictions file, pairs each tuned run with the baseline run of the
same model and split, runs checks V1 to V6, and, when they all pass, writes the committed
report ``reports/frontier/nova_tuned_report.md`` and ``.json``: both scores and their
difference, the questions tuning fixed and broke, the cost per example, and a verdict per
model.

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.nova_tuned --runs <B1 run_id> ... --baselines <B0 run_id> ...

The verdict comes from dev only. A model has **met** its threshold when its execution
accuracy is at least 10 points above its own baseline's and its format validity is at
least 98% (``config/frontier.yaml``, ``tuned_evaluation``). A model that **missed** makes
v4.05, one corrective iteration, eligible for it. This script starts nothing.

It calls no AWS service, needs no sign-in, and costs nothing: the runs were made, and paid
for, by ``python -m scripts.evaluate --predictor nova-lite-tuned|nova-pro-tuned``.

Exit codes: 0 the report was written and every model in it met its threshold; 1 a check
failed (nothing is written), or a model missed (the report is written); 2 a config problem
or an unknown run.

Needs the snapshot, the split manifest, the run folders under ``artifacts/runs/``, the
tuning record, and the gate record.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

try:
    from qlora_finetuning.data import manifest_hash
    from qlora_finetuning.evaluation.runner import RunnerContext, run_predictions, summarize_run
    from qlora_finetuning.frontier.config import FrontierConfig, load_frontier_config
    from qlora_finetuning.frontier.pricing import Pricing, load_pricing
    from qlora_finetuning.frontier.training_estimate import read_gate_record
    from qlora_finetuning.frontier.tuned_report import MET, build_report, write_report
    from qlora_finetuning.frontier.tuning_record import read_tuning_record
    from qlora_finetuning.logging_setup import setup_logging
    from qlora_finetuning.settings import ConfigError
    from qlora_finetuning.tracking import Ledger
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.nova_tuned"
    ) from None

PASS, FAIL = "PASS", "FAIL"


def _relative(path: Any, context: RunnerContext) -> str:
    try:
        return path.relative_to(context.tracking_cfg.root).as_posix()
    except ValueError:
        return str(path)


def _print_model(model: dict[str, Any], split: str) -> None:
    pair = next(p for p in model["pairs"] if p["split"] == split)
    result, total = model["verdict"], pair["examples"]
    both = pair["paired"]
    print(
        f"{model['model']:<10} {split}: execution accuracy "
        f"{pair['baseline']['execution_accuracy']:,} -> {pair['tuned']['execution_accuracy']:,} "
        f"of {total:,} ({result['execution_gain_points']} points; "
        f"{result['needed_execution_count']:,} needed); format validity "
        f"{pair['tuned']['format_validity']:,} of {total:,} "
        f"({result['needed_format_count']:,} needed); fixed {both['fixed']:,}, broken "
        f"{both['broken']:,}"
    )
    print(f"{result['verdict'].upper()}: {model['model']} ({model['tuned_arm']})")
    for reason in result["reasons"]:
        print(f"  {reason}")
    if result["corrective_iteration_eligible"]:
        print(f"  v4.05 is eligible for {model['model']}; nothing starts by itself")


def main(
    argv: list[str] | None = None,
    context: RunnerContext | None = None,
    cfg: FrontierConfig | None = None,
    pricing: Pricing | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Compare the fine-tuned Nova models with their untuned baselines."
    )
    parser.add_argument(
        "--runs", nargs="+", required=True, metavar="RUN_ID", help="the tuned runs (arms B1)"
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        required=True,
        metavar="RUN_ID",
        help="the untuned runs they are compared with (arms B0)",
    )
    args = parser.parse_args(argv)

    setup_logging()
    try:
        context = (context or RunnerContext()).resolved()
        cfg = cfg or load_frontier_config()
        pricing = pricing or load_pricing()
        every = [*args.runs, *args.baselines]
        if len(set(every)) != len(every):
            raise ConfigError("a run id was given twice")
        tuned = [summarize_run(run_id, context) for run_id in args.runs]
        baselines = [summarize_run(run_id, context) for run_id in args.baselines]
        lines = {run_id: run_predictions(run_id, context) for run_id in every}
        try:
            gate = read_gate_record(cfg)
        except ConfigError:
            gate = None  # reported by check V4
        report = build_report(
            tuned,
            baselines,
            lines,
            read_tuning_record(cfg),
            gate,
            Ledger(context.tracking_cfg).effective(),
            cfg,
            pricing,
            prompt_id=context.prompt.prompt_id,
            manifest_hash=manifest_hash(context.split_cfg),
        )
    except ConfigError as exc:  # bad config, no snapshot, an unknown or damaged run
        print(f"CONFIG ERROR: {exc}")
        return 2

    for item in report["checks"]:
        status = PASS if item["passed"] else FAIL
        print(f"{status}  {item['id']}  {item['name']}: {item['detail']}")
    failed = sum(not item["passed"] for item in report["checks"])
    print(f"{len(report['checks']) - failed} passed, {failed} failed")
    if failed:
        print("NOT WRITTEN: a check failed, so no report was written")
        return 1

    for model in report["models"]:
        _print_model(model, cfg.verdict_split)
    if report["partial"]:
        print(
            "PARTIAL: no tuned run was given for "
            f"{', '.join(report['models_without_runs'])}; milestone M6 needs every model"
        )
    markdown_path, json_path = write_report(report, cfg.report_dir)
    print(f"report: {_relative(markdown_path, context)}")
    print(f"report: {_relative(json_path, context)}")
    met = sum(model["verdict"]["verdict"] == MET for model in report["models"])
    print(f"{met} of {len(report['models'])} models met their threshold")
    return 0 if report["all_met"] else 1


if __name__ == "__main__":
    sys.exit(main())
