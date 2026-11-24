"""Write the report of the untuned Nova baselines from their runs (req_004 v4.01).

Give it the run ids of the baseline runs: each model of ``config/frontier.yaml`` on
``quick_dev`` and on ``dev``, in any order. It computes each run's summary again from its
predictions file, runs checks B1 to B4, and, when they all pass, writes the committed report
``reports/frontier/nova_baseline_report.md`` and ``.json``: scores, Bedrock's token counts,
their ratio to the Qwen counts, cost at list price, and latency.

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.nova_baseline --runs <run_id> <run_id> <run_id> <run_id>

It calls no AWS service and costs nothing: the runs were made, and paid for, by
``python -m scripts.evaluate --predictor nova-pro|nova-lite``.

Exit codes: 0 every check passed and the report was written; 1 a check failed (nothing is
written); 2 a config problem or an unknown run.

Needs the snapshot, the split manifest, the run folders under ``artifacts/runs/``, and the
committed token budget report.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from typing import Any

try:
    from qlora_finetuning.data import manifest_hash
    from qlora_finetuning.data.token_budget import load_token_budget_config
    from qlora_finetuning.data.token_budget_report import report_paths as token_report_paths
    from qlora_finetuning.evaluation.runner import RunnerContext, summarize_run
    from qlora_finetuning.frontier.baseline_report import (
        build_report,
        load_qwen_prompt_tokens,
        write_report,
    )
    from qlora_finetuning.frontier.config import FrontierConfig, load_frontier_config
    from qlora_finetuning.frontier.pricing import Pricing, load_pricing
    from qlora_finetuning.logging_setup import setup_logging
    from qlora_finetuning.settings import ConfigError
    from qlora_finetuning.tracking import Ledger
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.nova_baseline"
    ) from None

PASS, FAIL = "PASS", "FAIL"


def _relative(path: Any, context: RunnerContext) -> str:
    try:
        return path.relative_to(context.tracking_cfg.root).as_posix()
    except ValueError:
        return str(path)


def main(
    argv: list[str] | None = None,
    context: RunnerContext | None = None,
    cfg: FrontierConfig | None = None,
    pricing: Pricing | None = None,
    qwen_prompt_tokens: Mapping[str, int] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Report the untuned Nova baselines.")
    parser.add_argument(
        "--runs", nargs="+", required=True, metavar="RUN_ID", help="the baseline runs"
    )
    args = parser.parse_args(argv)

    setup_logging()
    try:
        context = (context or RunnerContext()).resolved()
        cfg = cfg or load_frontier_config()
        pricing = pricing or load_pricing()
        if qwen_prompt_tokens is None:
            _, token_json = token_report_paths(load_token_budget_config().report_dir)
            qwen_prompt_tokens = load_qwen_prompt_tokens(token_json)
        if len(set(args.runs)) != len(args.runs):
            raise ConfigError("a run id was given twice")
        summaries = [summarize_run(run_id, context) for run_id in args.runs]
        report = build_report(
            summaries,
            Ledger(context.tracking_cfg).effective(),
            cfg,
            pricing,
            qwen_prompt_tokens,
            prompt_id=context.prompt.prompt_id,
            manifest_hash=manifest_hash(context.split_cfg),
        )
    except ConfigError as exc:  # bad config, no snapshot, an unknown or damaged run
        print(f"CONFIG ERROR: {exc}")
        return 2

    for check in report["checks"]:
        status = PASS if check["passed"] else FAIL
        print(f"{status}  {check['id']}  {check['name']}: {check['detail']}")
    failed = sum(not check["passed"] for check in report["checks"])
    print(f"{len(report['checks']) - failed} passed, {failed} failed")
    if failed:
        print("NOT WRITTEN: a check failed, so no report was written")
        return 1

    for run in report["runs"]:
        accuracy = run["execution_accuracy"]
        rate = "-" if accuracy["rate"] is None else f"{100 * accuracy['rate']:.2f}%"
        print(
            f"{run['arm']:<8} {run['split']:<10} execution accuracy {accuracy['count']:,} of "
            f"{run['examples']:,}  {rate}; tokens in {run['input_tokens']:,}, out "
            f"{run['output_tokens']:,}; Nova/Qwen {run['nova_to_qwen_input_ratio']}; "
            f"cost ${run['cost_usd']}"
        )
    markdown_path, json_path = write_report(report, cfg.report_dir)
    print(f"report: {_relative(markdown_path, context)}")
    print(f"report: {_relative(json_path, context)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
