"""Train a LoRA adapter on Qwen3-4B, or rehearse it with a smoke run (req_003 v3.02).

One command for Path A's training. It encodes the ``fit`` records as the sequences the token
budget counted, probes the worst batch, trains with the settings of ``config/training.yaml``,
saves a checkpoint every ``checkpoint.save_steps`` optimizer steps, and leaves the final
adapter under ``artifacts/adapters/<run_id>/``.

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.qlora_train --dry-run
    venv\\Scripts\\python -m scripts.qlora_train --smoke
    venv\\Scripts\\python -m scripts.qlora_train
    venv\\Scripts\\python -m scripts.qlora_train --resume <run_id>
    venv\\Scripts\\python -m scripts.qlora_train --smoke --stop-after-steps 6

``--dry-run`` needs the tokenizer and no GPU: it prints the plan (records, tokens, optimizer
steps, warmup, checkpoints, the worst batch) and starts no run. With ``--smoke`` it prints
the smoke run's plan.

``--smoke`` trains on a few records (the longest ones first), then loads the adapter through
the inference backend and asks it a few quick-dev prompts. It prints checks K1 to K7 and
takes a few minutes on a GPU. It is free, and it is not a result: a smoke adapter is never
arm A1.

Without a flag it is the full run on this machine: hours of GPU time. ``--resume`` continues
an interrupted run from its last checkpoint, in a new run. ``--stop-after-steps N`` fails a
run after N optimizer steps, to rehearse an interruption.

An adapter is scored afterwards, through the evaluation runner::

    venv\\Scripts\\python -m scripts.evaluate --predictor qwen3-4b-tuned \\
        --adapter <run_id> --split quick_dev

Exit codes: 0 the run completed and its checks pass (or the dry run printed its plan);
1 training failed or was stopped, or a check failed; 2 a config problem, a record over the
training length, or a resume that is refused. Nothing is ever truncated.

Needs the snapshot, the split manifest, and the training stack; a run also needs the model
weights (runbook section 5) and a GPU.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

try:
    from qlora_finetuning.local_qlora.training import (
        TrainingContext,
        TrainingError,
        prepare,
        run_training,
    )
    from qlora_finetuning.logging_setup import setup_logging
    from qlora_finetuning.settings import ConfigError
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.qlora_train"
    ) from None

PASS, FAIL = "PASS", "FAIL"
# Libraries that log every HTTP request or warn about a missing token; held quiet during a run.
NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3", "filelock")
EVALUATE = "python -m scripts.evaluate --predictor qwen3-4b-tuned"


def _steps(steps: list[int]) -> str:
    if len(steps) <= 6:
        return ", ".join(f"{step:,}" for step in steps)
    return f"{steps[0]:,}, {steps[1]:,}, ... {steps[-1]:,} ({len(steps)} checkpoints)"


def print_plan(plan: dict[str, Any], smoke: bool, dev_records: int) -> None:
    kind = "smoke run" if smoke else "full run"
    print(f"plan for the {kind}:")
    print(f"records               {plan['records']:,}")
    print(f"tokens in one pass    {plan['tokens_per_pass']:,}")
    print(
        f"batch                 {plan['batch_size']} x {plan['accumulation']} "
        f"({plan['batch_size'] * plan['accumulation']} sequences per optimizer step)"
    )
    print(f"epochs                {plan['epochs']}")
    print(f"optimizer steps       {plan['total_steps']:,} ({plan['steps_per_epoch']:,} per epoch)")
    print(f"warmup steps          {plan['warmup_steps']:,}")
    print(f"checkpoints at steps  {_steps(plan['checkpoint_steps'])}")
    print(
        f"worst batch           {len(plan['worst_batch'])} sequences, "
        f"{plan['worst_batch_width']:,} tokens wide"
    )
    print(f"dev records for loss  {dev_records:,}")


def print_result(result: dict[str, Any]) -> None:
    continues = f"; continues {result['continues']}" if result["continues"] else ""
    print(f"run {result['run_id']}  ({result['kind']}{continues})")
    for check in result["checks"]:
        status = PASS if check["passed"] else FAIL
        print(f"{status}  {check['id']}  {check['name']}: {check['detail']}")
    failed = sum(not check["passed"] for check in result["checks"])
    print(f"{len(result['checks']) - failed} passed, {failed} failed")
    trainer = result["measurements"].get("trainer") or {}
    print(
        f"steps: {result['steps']:,} of {result['plan']['total_steps']:,}"
        + (f" (from step {result['start_step']:,})" if result["start_step"] else "")
    )
    if trainer:
        print(
            f"training: {trainer['train_seconds']:g} s, {trainer['seconds_per_step']:g} s per "
            f"step, {trainer['seconds_per_sequence']:g} s per sequence; peak GPU memory "
            f"{trainer['peak_vram_allocated_gib']:g} GiB allocated, "
            f"{trainer['peak_vram_reserved_gib']:g} reserved"
        )
    print(f"adapter: {result['adapter']}  (sha256 {result['adapter_sha256'][:12]}...)")
    print(f"score it with: {EVALUATE} --adapter {result['adapter']} --split quick_dev")


def main(
    argv: list[str] | None = None,
    context: TrainingContext | None = None,
    engine: Any = None,
    reloader: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on Qwen3-4B.")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; start no run")
    parser.add_argument("--smoke", action="store_true", help="a few records, then checks K1-K7")
    parser.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted run")
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        metavar="N",
        help="fail the run after N optimizer steps (rehearses an interruption)",
    )
    parser.add_argument("--where", default="local", help="local or sagemaker (default: local)")
    parser.add_argument("--scope", default="dev", choices=("dev", "measured"))
    parser.add_argument("--label", help="the run's label")
    args = parser.parse_args(argv)
    if args.dry_run and (args.resume or args.stop_after_steps is not None):
        parser.error("--dry-run goes with --smoke or with nothing")

    setup_logging()
    for noisy in NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.ERROR)
    try:
        if args.dry_run:
            resolved = (context or TrainingContext()).resolved()
            prepared = prepare(resolved, args.smoke)
            print_plan(prepared.plan.to_dict(), args.smoke, len(prepared.dev_rows))
            print("DRY RUN: nothing was trained and no run was started")
            return 0
        result = run_training(
            smoke=args.smoke,
            resume=args.resume,
            stop_after_steps=args.stop_after_steps,
            where=args.where,
            scope=args.scope,
            label=args.label,
            context=context,
            engine=engine,
            reloader=reloader,
        )
    except TrainingError as exc:
        print(f"TRAINING FAILED: {exc}")
        if exc.run_id:
            flag = "--smoke " if args.smoke else ""
            print(f"checkpoints are kept in run {exc.run_id}")
            print(f"resume with: python -m scripts.qlora_train {flag}--resume {exc.run_id}")
        return 1
    except ConfigError as exc:  # config, a record over the length, a refused resume, no weights
        print(f"CONFIG ERROR: {exc}")
        return 2

    print_result(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
