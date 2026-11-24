"""Evaluate a predictor on a split, resume a run, or print a run's summary (req_002 v2.03).

One command for every arm. It takes the examples of a split in dataset order, asks the
predictor for each reply, writes ``predictions.jsonl`` in the run's folder as it goes,
records the run and what it consumed, and prints the totals with checks R1 to R5.

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.evaluate --predictor gold --split quick_dev
    venv\\Scripts\\python -m scripts.evaluate --predictor gold --split dev --limit 20
    venv\\Scripts\\python -m scripts.evaluate --predictor gold --split dev --resume <run_id>
    venv\\Scripts\\python -m scripts.evaluate --summary <run_id>

``--predictor`` names an entry of ``PREDICTORS`` below. ``gold`` is the fake predictor that
gives each example its own gold reply; it must score every example right. A model backend
adds its own entry. ``qwen3-4b`` is untuned Qwen3-4B in 4-bit on the local GPU (req_003
v3.01): free, no network, and it needs the training stack and the model weights, which are
downloaded once (runbook section 5); without them it exits 2 before any run exists.
``qwen3-4b-tuned`` is the same model with a trained LoRA adapter (v3.02), named with
``--adapter RUN_ID`` or ``--adapter RUN_ID/checkpoint-N``, the id of a training run; an
adapter trained for another prompt, split, or model commit is refused. ``--adapter selected``
(req_003 v3.04) is the checkpoint that ``scripts.qlora_select`` chose as arm A1.
``nova-pro`` and ``nova-lite`` are the untuned Amazon Nova models on
Bedrock (req_004 v4.01): **every call is billed, so those runs are started by the owner**,
after ``aws sso login --profile qlora``. ``nova-pro-tuned`` and ``nova-lite-tuned`` are the
fine-tuned models (v4.04), asked through the deployment the tuning record names. They refuse
to start until that deployment is Active, and they are billed and the owner's too.

``--split test`` (req_005 v5.01) is possible only after the owner's unlock, and then only
for the six arms at their frozen configuration, with ``--scope measured``, without
``--limit``, and once per arm: an interrupted run is continued with ``--resume``, a second
run is ``REFUSED``. The two Qwen arms answer test questions inside a SageMaker job only
(``scripts.sagemaker_job``). ``gold`` may run on test as a free check of the runner.

``--limit N`` predicts the first N examples: a partial run, never a result. ``--resume``
continues an interrupted run in a new one. ``--summary`` predicts nothing: it computes a
run's totals again from its predictions file.

Exit codes: 0 a run whose checks pass (complete, or the partial run that was asked for);
1 a check failed or the predictor failed; 2 a config problem, a locked split, or a resume
or a test run that is refused.

Needs the snapshot and the split manifest. The ``gold`` predictor needs nothing else; a
Nova predictor needs a valid AWS sign-in, and without one exits 2 before any run exists.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:
    from qlora_finetuning.comparison.gate import TEST, TestRunRefused, check_test_request
    from qlora_finetuning.data import TestLockedError, build_record, load_split
    from qlora_finetuning.evaluation.fake import gold_predictor
    from qlora_finetuning.evaluation.metrics import METRICS
    from qlora_finetuning.evaluation.predictions import PREDICTIONS_NAME
    from qlora_finetuning.evaluation.predictor import PredictorError
    from qlora_finetuning.evaluation.runner import (
        ResumeError,
        RunnerContext,
        run_evaluation,
        summarize_run,
    )
    from qlora_finetuning.logging_setup import setup_logging
    from qlora_finetuning.settings import ConfigError
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.evaluate"
    ) from None

PASS, FAIL = "PASS", "FAIL"


@dataclass(frozen=True)
class PredictorEntry:
    """A predictor the script can be asked for by name.

    ``factory`` builds it from the split name and the runner's context. It is a function, or
    ``"package.module:function"``, which is imported only when the entry is named, so a
    backend's libraries are loaded only when that backend is used.

    ``where`` is the place recorded for the run when ``--where`` is not given (``local`` if
    neither says). ``configs`` names config files the run snapshots beside the runner's own:
    the ones that decide this backend's replies. ``takes_adapter`` marks a predictor that
    is built for the adapter named with ``--adapter``; its factory takes it as a third
    argument.
    """

    factory: Callable[[str, RunnerContext], Any] | str
    kind: str
    path: str
    arm: str
    where: str | None = None
    configs: tuple[str, ...] = ()
    takes_adapter: bool = False

    def build(self, split: str, context: RunnerContext, adapter: str | None = None) -> Any:
        if adapter is not None and not self.takes_adapter:
            raise ConfigError("--adapter is for a tuned Qwen predictor; this one takes none")
        factory = self.factory
        if isinstance(factory, str):
            module, _, name = factory.partition(":")
            factory = getattr(importlib.import_module(module), name)
        if self.takes_adapter:
            return factory(split, context, adapter)
        return factory(split, context)


def _gold(split: str, context: RunnerContext) -> Any:
    """Each example's own gold reply. A locked test split raises here, before any run."""
    examples = load_split(split, context.split_cfg, context.data_cfg)
    return gold_predictor(build_record(example, context.prompt) for example in examples)


# qwen3-4b is the untuned Qwen model in 4-bit on the local GPU (req_003 v3.01): free, and it
# needs the weights in the Hugging Face cache; qwen3-4b-tuned adds a trained adapter (v3.02),
# named with --adapter. The Nova entries are the untuned models
# (req_004 v4.01) and the fine-tuned ones (v4.04); their calls are billed.
_NOVA = "qlora_finetuning.frontier.predictor"
_NOVA_CONFIGS = ("frontier", "aws")
_PACKAGING = "qlora_finetuning.packaging.predictors"
_PACKAGING_CONFIGS = ("local_qlora", "packaging")
PREDICTORS: dict[str, PredictorEntry] = {
    "gold": PredictorEntry(factory=_gold, kind="smoke", path="none", arm="none"),
    "qwen3-4b": PredictorEntry(
        factory="qlora_finetuning.local_qlora.predictor:qwen3_4b",
        kind="eval",
        path="A",
        arm="A0",
        where="local",
        configs=("local_qlora",),
    ),
    "qwen3-4b-tuned": PredictorEntry(
        factory="qlora_finetuning.local_qlora.predictor:qwen3_4b_tuned",
        kind="eval",
        path="A",
        arm="A1",
        where="local",
        configs=("local_qlora", "training"),
        takes_adapter=True,
    ),
    # The two packages of req_006 v6.02, loaded from their folders under artifacts/release/.
    # None is an arm: they are scored on dev, locally, and are refused on test.
    "qwen3-4b-package": PredictorEntry(
        factory=f"{_PACKAGING}:qwen3_4b_package",
        kind="eval",
        path="A",
        arm="none",
        where="local",
        configs=_PACKAGING_CONFIGS,
    ),
    "qwen3-4b-merged": PredictorEntry(
        factory=f"{_PACKAGING}:qwen3_4b_merged",
        kind="eval",
        path="A",
        arm="none",
        where="local",
        configs=_PACKAGING_CONFIGS,
    ),
    "qwen3-4b-merged-full": PredictorEntry(
        factory=f"{_PACKAGING}:qwen3_4b_merged_full",
        kind="eval",
        path="A",
        arm="none",
        where="local",
        configs=_PACKAGING_CONFIGS,
    ),
    "nova-pro": PredictorEntry(
        factory=f"{_NOVA}:nova_pro",
        kind="eval",
        path="B",
        arm="B0-Pro",
        where="bedrock",
        configs=_NOVA_CONFIGS,
    ),
    "nova-lite": PredictorEntry(
        factory=f"{_NOVA}:nova_lite",
        kind="eval",
        path="B",
        arm="B0-Lite",
        where="bedrock",
        configs=_NOVA_CONFIGS,
    ),
    "nova-pro-tuned": PredictorEntry(
        factory=f"{_NOVA}:nova_pro_tuned",
        kind="eval",
        path="B",
        arm="B1-Pro",
        where="bedrock",
        configs=_NOVA_CONFIGS,
    ),
    "nova-lite-tuned": PredictorEntry(
        factory=f"{_NOVA}:nova_lite_tuned",
        kind="eval",
        path="B",
        arm="B1-Lite",
        where="bedrock",
        configs=_NOVA_CONFIGS,
    ),
}


def _relative(path: Any, context: RunnerContext) -> str:
    try:
        return path.relative_to(context.tracking_cfg.root).as_posix()
    except ValueError:
        return str(path)


def _share(rate: dict[str, Any], total: int) -> str:
    percent = "-" if rate["rate"] is None else f"{100 * rate['rate']:.2f}%"
    return f"{rate['count']:,} of {total:,}  {percent}"


def print_summary(summary: dict[str, Any], context: RunnerContext) -> None:
    predictor = summary["predictor"]
    print(
        f"run {summary['run_id']}  ({summary['kind']}; arm {summary['arm']}; "
        f"split {summary['split']}; predictor {predictor['name']})"
    )
    for check in summary["checks"]:
        status = PASS if check["passed"] else FAIL
        print(f"{status}  {check['id']}  {check['name']}: {check['detail']}")
    failed = sum(not check["passed"] for check in summary["checks"])
    print(f"{len(summary['checks']) - failed} passed, {failed} failed")

    block, total = summary["all"], summary["all"]["examples"]
    for name in METRICS:
        print(f"{name.replace('_', ' '):<20} {_share(block[name], total)}")
    tight = summary["diagnostic"]["execution_accuracy_tight"]
    print(f"{'execution, tight rule':<20} {_share(tight, total)}  (diagnostic)")
    if summary["flagged_examples"]:
        unflagged = summary["unflagged"]
        print(
            f"without the {summary['flagged_examples']:,} flagged examples: execution accuracy "
            f"{_share(unflagged['execution_accuracy'], unflagged['examples'])}"
        )
    usage, latency = summary["usage"], summary["latency_seconds"]
    if usage["input_tokens"] is not None and usage["output_tokens"] is not None:
        print(f"tokens: input {usage['input_tokens']:,}, output {usage['output_tokens']:,}")
    if latency["mean"] is not None:
        print(
            f"latency in seconds: mean {latency['mean']:g}, median {latency['median']:g}, "
            f"95th percentile {latency['p95']:g}"
        )
    if summary["stop_reasons"]:
        reasons = ", ".join(f"{name} {count:,}" for name, count in summary["stop_reasons"].items())
        print(f"stop reasons: {reasons}")
    backend = summary.get("backend")
    if backend:
        shown = ", ".join(
            f"{name.replace('_', ' ')} {value}"
            for name, value in backend.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        )
        print(f"measurements: {shown}".encode("ascii", "replace").decode())
    if summary["result"]:
        print("RESULT: complete; this run counts as a result")
    elif summary["partial"] and summary["complete"]:
        print(
            f"PARTIAL: the first {summary['examples_asked']:,} of "
            f"{summary['split_examples']:,} examples; not a result"
        )
    else:
        print("NOT A RESULT: a check failed or the run is unfinished")
    folder = context.tracking_cfg.runs_dir / summary["run_id"]
    print(f"predictions: {_relative(folder / PREDICTIONS_NAME, context)}")


def main(
    argv: list[str] | None = None,
    context: RunnerContext | None = None,
    predictors: dict[str, PredictorEntry] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a predictor on a FinQA split.")
    parser.add_argument("--predictor", help="a name from the PREDICTORS table, for example gold")
    parser.add_argument("--split", help="quick_dev, dev, fit, or (once unlocked) test")
    parser.add_argument("--limit", type=int, help="predict the first N examples only")
    parser.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted run")
    parser.add_argument("--summary", metavar="RUN_ID", help="print a run's summary again")
    parser.add_argument("--label", help="the run's label (default: predictor and split)")
    parser.add_argument(
        "--adapter",
        metavar="RUN_ID[/checkpoint-N]",
        help=(
            "for qwen3-4b-tuned: the training run whose adapter, or checkpoint, is asked; "
            "'selected' is the checkpoint of the selection record"
        ),
    )
    parser.add_argument("--scope", default="dev", choices=("dev", "measured"))
    parser.add_argument(
        "--where", help="local, sagemaker, or bedrock (default: the predictor's own, else local)"
    )
    args = parser.parse_args(argv)

    if args.summary:
        if (
            args.predictor
            or args.split
            or args.resume
            or args.adapter
            or args.limit is not None
        ):
            parser.error("--summary takes a run id and nothing else")
    elif not (args.predictor and args.split):
        parser.error("give --predictor and --split, or --summary RUN_ID")

    setup_logging()
    table = PREDICTORS if predictors is None else predictors
    try:
        context = (context or RunnerContext()).resolved()
        if args.summary:
            summary = summarize_run(args.summary, context)
        else:
            entry = table.get(args.predictor)
            if entry is None:
                known = ", ".join(sorted(table)) or "none"
                print(f"UNKNOWN PREDICTOR: {args.predictor}. Known: {known}")
                return 2
            extra = tuple(name for name in entry.configs if name not in context.configs)
            if extra:
                context = dataclasses.replace(context, configs=(*context.configs, *extra))
            where = args.where or entry.where or "local"
            if args.split == TEST:
                # The gate of req_005 v5.01, asked before a model is loaded or a row is
                # read. The runner asks it again with the predictor that was built.
                check_test_request(
                    arm=entry.arm,
                    name=args.predictor,
                    where=where,
                    scope=args.scope,
                    limit=args.limit,
                    resume=args.resume,
                    ctx=context,
                )
            summary = run_evaluation(
                entry.build(args.split, context, args.adapter),
                args.split,
                label=args.label or f"{args.predictor} {args.split}",
                path=entry.path,
                arm=entry.arm,
                where=where,
                scope=args.scope,
                kind=entry.kind,
                limit=args.limit,
                resume=args.resume,
                context=context,
            )
    except PredictorError as exc:
        print(f"PREDICTOR FAILED: {exc}")
        if exc.run_id:
            try:
                kept = summarize_run(exc.run_id, context)["examples_predicted"]
            except ConfigError:
                kept = None
            if kept is not None:
                print(f"{kept:,} predictions are kept in run {exc.run_id}")
            print(
                "resume with: python -m scripts.evaluate "
                f"--predictor {args.predictor} --split {args.split} --resume {exc.run_id}"
            )
        return 1
    except TestLockedError as exc:  # reported, never worked around: nothing was read or run
        print(f"LOCKED: {exc}")
        return 2
    except (ResumeError, TestRunRefused) as exc:  # nothing was read, asked, or run
        print(f"REFUSED: {exc}")
        return 2
    except ConfigError as exc:  # bad config, no snapshot, no manifest, an unknown run id
        print(f"CONFIG ERROR: {exc}")
        return 2

    print_summary(summary, context)
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
