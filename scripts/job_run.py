"""Run one task of Path A: what a SageMaker job executes, and the same here (req_003 v3.03).

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.job_run smoke
    venv\\Scripts\\python -m scripts.job_run train
    venv\\Scripts\\python -m scripts.job_run train --resume <run_id>
    venv\\Scripts\\python -m scripts.job_run evaluate --predictor qwen3-4b --split dev
    venv\\Scripts\\python -m scripts.job_run evaluate --predictor qwen3-4b-tuned \\
        --adapter <run_id>/checkpoint-300 --split dev
    venv\\Scripts\\python -m scripts.job_run score --run <training run id> --split dev
    venv\\Scripts\\python -m scripts.job_run fetch-model

``smoke`` is the smoke training run, then a few quick-dev examples with its adapter and
without. ``train`` is the full training run: hours of GPU time. ``evaluate`` is one
evaluation run. ``score`` (req_003 v3.04) scores the untuned model and every checkpoint of
a training run on one split; run again, it skips what is already scored and resumes what
was cut short. Each calls the same functions as ``scripts.qlora_train`` and
``scripts.evaluate``; nothing is written for the cloud alone.

``fetch-model`` downloads the pinned model weights from the Hugging Face Hub into the local
cache (8 GB, once). A job runs it before its task; a task itself never uses the network.

A task first checks that the installed training stack is the pinned one. It always writes
``artifacts/jobs/<job name>/job_result.json``: the runs it made and what came of them.

``--where`` is ``local`` unless given; a job's entry point passes ``sagemaker``. In a job,
the examples come from a data bundle named by ``QLORA_DATA_BUNDLE`` (``fit`` and ``dev``
only), and the commit from ``QLORA_GIT_COMMIT``.

Exit codes: 0 the task completed; 1 a run or a check failed; 2 nothing could be started
(a config problem, another library version, a refused resume or adapter).
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

try:
    from qlora_finetuning.data.token_budget import load_token_budget_config
    from qlora_finetuning.local_qlora.job import (
        COMPLETED,
        CONFIG_ERROR,
        TASKS,
        result_path,
        run_task,
    )
    from qlora_finetuning.logging_setup import setup_logging
    from qlora_finetuning.settings import ConfigError
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.job_run"
    ) from None

FETCH = "fetch-model"
NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3", "filelock")


def print_result(result: dict[str, Any]) -> None:
    print(f"task {result['task']}  (job {result['job_name']}; where {result['where']})")
    for run in result["runs"]:
        parts = [f"{run['role']}: run {run.get('run_id')}", run["status"]]
        if run["role"] == "train" and run["status"] == COMPLETED:
            failed = ", ".join(run["checks_failed"]) or "none"
            parts.append(f"{run['steps']:,} steps; checks failed: {failed}")
        elif run["status"] == COMPLETED:
            if run.get("step") is not None:
                parts.insert(1, f"step {run['step']:,}")
            if run.get("reused"):
                parts.append("already scored")
            parts.append(
                f"{run['predictor']} on {run['split']}: {run['execution_accuracy']:,} of "
                f"{run['examples_predicted']:,} right on execution, "
                f"{run['format_validity']:,} in the format"
            )
        print("  " + "; ".join(parts))
    if result["error"]:
        print(f"ERROR: {result['error']}".encode("ascii", "replace").decode())
    print(f"STATUS: {result['status']}")


def main(argv: list[str] | None = None, **stand_ins: Any) -> int:
    parser = argparse.ArgumentParser(description="Run one task of Path A.")
    parser.add_argument("task", choices=(*TASKS, FETCH))
    parser.add_argument("--where", default="local", help="local or sagemaker")
    parser.add_argument("--scope", default="dev", choices=("dev", "measured"))
    parser.add_argument("--resume", metavar="RUN_ID", help="train or evaluate: continue a run")
    parser.add_argument("--predictor", help="evaluate: qwen3-4b or qwen3-4b-tuned")
    parser.add_argument("--adapter", metavar="RUN_ID[/checkpoint-N]", help="evaluate: the adapter")
    parser.add_argument("--split", help="evaluate: quick_dev or dev")
    parser.add_argument("--limit", type=int, help="evaluate or score: the first N examples only")
    parser.add_argument("--run", metavar="RUN_ID", help="score: the training run to score")
    args = parser.parse_args(argv)

    setup_logging()
    for noisy in NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.ERROR)

    if args.task == FETCH:
        try:
            from qlora_finetuning.local_qlora.backend import fetch_weights

            budget = load_token_budget_config()
            fetch_weights(budget)
        except ConfigError as exc:
            print(f"CONFIG ERROR: {exc}")
            return 2
        print(f"the weights of {budget.model_id} at {budget.revision[:12]}... are in the cache")
        return 0

    result = run_task(
        args.task,
        where=args.where,
        scope=args.scope,
        resume=args.resume,
        predictor=args.predictor,
        adapter=args.adapter,
        split=args.split,
        limit=args.limit,
        run=args.run,
        **stand_ins,
    )
    print_result(result)
    root = stand_ins.get("root")
    path = result_path(result["job_name"], root) if root else result_path(result["job_name"])
    print(f"result: {path.parent.name}/{path.name}")
    if result["status"] == COMPLETED:
        return 0
    return 2 if result["status"] == CONFIG_ERROR else 1


if __name__ == "__main__":
    sys.exit(main())
