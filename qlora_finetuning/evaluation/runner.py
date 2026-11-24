"""The evaluation runner: one split, one predictor, a scored predictions file (req_002 v2.03).

Every arm is evaluated by this loop. It takes the examples of a split in dataset order, asks
the predictor for each reply in batches, and after each batch writes one line per example to
``predictions.jsonl`` in the run's folder. One call is one run in the registry, with one
ledger record for what it consumed, written whether the run completes or fails.

A predictor may also have a ``measurements()`` method (peak GPU memory, load time): what it
returns is written to ``measurements.json`` in the run folder and appears in the summary
under ``backend`` (req_003 v3.01).

A run cannot be reopened, so resuming is a new run: it copies the valid lines of the earlier
file and continues after them. A predictor failure stops the run; an example is never scored
as wrong because the infrastructure failed.

The summary is always computed from the file: stored outcomes are scored again under the
current ``config/evaluation.yaml``. Only a run that passes checks R1 to R5 and is not partial
is a result.

Usage::

    from qlora_finetuning.data import serialize_split
    from qlora_finetuning.evaluation.fake import gold_predictor
    from qlora_finetuning.evaluation.runner import run_evaluation, summarize_run

    predictor = gold_predictor(serialize_split("quick_dev"))
    summary = run_evaluation(predictor, "quick_dev", label="gold quick-dev",
                             path="none", arm="none", where="local", scope="dev", kind="smoke")
    summary["result"]                                     # True: complete, and not partial
    summary["all"]["execution_accuracy"]                  # {"count": 200, "rate": 1.0}
    summary["diagnostic"]["execution_accuracy_tight"]     # the same under the tight rule

    summarize_run(summary["run_id"])                      # the same totals, from the file

    # After a PredictorError, continue where the run stopped:
    run_evaluation(predictor, "quick_dev", ..., resume=error.run_id)

Import this module by its own name. It is not exported from ``qlora_finetuning.evaluation``,
because it imports the data package, whose gold self-check imports the metrics.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qlora_finetuning.data import (
    SerializationError,
    build_record,
    load_prompt,
    load_split,
    manifest_hash,
)
from qlora_finetuning.data.gold_check import load_gold_flags
from qlora_finetuning.data.token_budget import load_token_budget_config
from qlora_finetuning.dsl.config import load_dsl_config
from qlora_finetuning.dsl.outcome import run_reply
from qlora_finetuning.dsl.reply import ReplyFormat
from qlora_finetuning.evaluation.metrics import (
    EvaluationConfig,
    EvaluationError,
    ExampleScore,
    Gold,
    load_evaluation_config,
    score_reply,
    summarize,
)
from qlora_finetuning.evaluation.predictions import (
    PREDICTIONS_NAME,
    PredictionRecord,
    PredictionsError,
    append_records,
    outcome_from_dict,
    read_predictions,
)
from qlora_finetuning.evaluation.predictor import (
    PredictionRequest,
    PredictorError,
    check_predictions,
    describe,
)
from qlora_finetuning.logging_setup import get_logger
from qlora_finetuning.tracking import (
    Ledger,
    LedgerRecord,
    TrackingError,
    load_manifest,
    load_tracking_config,
    start_run,
)
from qlora_finetuning.tracking.config import check_no_secrets, timestamp
from qlora_finetuning.tracking.jsonl import append_line

SCHEMA = 1
EVALUATION_NAME = "evaluation.json"
SUMMARY_NAME = "summary.json"
MEASUREMENTS_NAME = "measurements.json"
PHASE = "eval"
TEST_SPLIT = "test"  # a run on it passes the gate of req_005 v5.01 first
# Every config file a run reads; each is copied into the run folder and hashed.
RUN_CONFIGS = ("evaluation", "prompt", "dsl", "token_budget", "split", "data")
# What decides a reply. A run can only be continued under the same values.
KEY_FIELDS = ("split", "manifest_hash", "prompt_id", "max_new_tokens", "predictor")

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
logger = get_logger(__name__)


class ResumeError(EvaluationError):
    """An earlier run cannot be continued: other settings, or a file that cannot be trusted."""


@dataclass
class RunnerContext:
    """Everything the runner reads from the project. ``None`` means "load the real one".

    A test fills it with a temporary split, a temporary tracking root, a fixed git state,
    and an empty environment, so nothing is written to the repository.
    """

    config: EvaluationConfig | None = None
    split_cfg: Any = None
    data_cfg: Any = None
    prompt: Any = None
    tracking_cfg: Any = None
    dsl_config: Any = None
    max_new_tokens: int | None = None
    git: Any = None
    environment: dict[str, Any] | None = None
    config_dir: Path | str | None = None
    configs: tuple[str, ...] = RUN_CONFIGS
    # config/comparison.yaml, for the gate on a test run (req_005 v5.01); None: the real one.
    comparison_cfg: Any = None

    def resolved(self) -> RunnerContext:
        """A copy with every setting that is still ``None`` loaded from ``config/``."""
        return dataclasses.replace(
            self,
            config=self.config or load_evaluation_config(),
            prompt=self.prompt or load_prompt(),
            tracking_cfg=self.tracking_cfg or load_tracking_config(),
            dsl_config=self.dsl_config or load_dsl_config(),
            max_new_tokens=self.max_new_tokens or load_token_budget_config().max_new_tokens,
        )


# --------------------------------------------------------------------------- small helpers


def _write_json(path: Path, data: dict[str, Any]) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _run_folder(run_id: str, ctx: RunnerContext) -> Path:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id) or ".." in run_id:
        raise EvaluationError(f"{run_id!r} is not a run id")
    return ctx.tracking_cfg.runs_dir / run_id


def _read_evaluation(run_id: str, ctx: RunnerContext) -> dict[str, Any]:
    path = _run_folder(run_id, ctx) / EVALUATION_NAME
    try:
        evaluation = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise EvaluationError(
            f"run {run_id} has no {EVALUATION_NAME}: it is not an evaluation run, or its "
            "folder is not on this machine"
        ) from None
    except ValueError:
        raise EvaluationError(f"the {EVALUATION_NAME} of run {run_id} is damaged") from None
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("key"), dict):
        raise EvaluationError(f"the {EVALUATION_NAME} of run {run_id} is damaged")
    return evaluation


def _check_limit(limit: Any) -> None:
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise EvaluationError(f"limit must be a positive whole number, got {limit!r}")


def _load_gold(split: str, limit: int | None, ctx: RunnerContext) -> tuple[Any, ...]:
    """The examples asked for, their canonical records and gold replies, and the split's size.

    ``load_split`` raises ``TestLockedError`` for a locked test split; it is not caught.
    """
    examples = load_split(split, ctx.split_cfg, ctx.data_cfg)
    total = len(examples)
    if limit is not None:
        examples = examples[:limit]
    reply_format = ReplyFormat.from_prompt(ctx.prompt)
    records, golds = [], []
    for example in examples:
        try:
            record = build_record(example, ctx.prompt)
            gold = Gold.from_target(record.target, reply_format, ctx.dsl_config)
        except (SerializationError, EvaluationError) as error:
            # No such example exists today: the gold self-check covers all 8,281.
            raise EvaluationError(
                f"example {example.id} has no usable gold reply ({error}); run "
                "python -m scripts.finqa_gold_check"
            ) from error
        records.append(record)
        golds.append(gold)
    return examples, records, golds, total, reply_format


def latency_stats(values: Sequence[float]) -> dict[str, Any]:
    """Mean, median, 95th percentile (nearest rank), maximum, and total, in seconds."""
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None, "total": None}
    ordered = sorted(values)
    p95 = ordered[max(math.ceil(0.95 * len(ordered)), 1) - 1]
    return {
        "mean": round(statistics.fmean(ordered), 6),
        "median": round(statistics.median(ordered), 6),
        "p95": round(p95, 6),
        "max": round(ordered[-1], 6),
        "total": round(math.fsum(ordered), 6),
    }


def _token_total(lines: Sequence[PredictionRecord], name: str) -> int | None:
    known = [getattr(line, name) for line in lines if getattr(line, name) is not None]
    return sum(known) if known else None


def _ledger_examples(ctx: RunnerContext) -> Counter:
    """Examples per run id, over the evaluation records of the ledger that count."""
    recorded: Counter = Counter()
    for record in Ledger(ctx.tracking_cfg).effective():
        if record.phase == PHASE:
            recorded[record.run_id] += record.examples or 0
    return recorded


def _measurements(predictor: Any) -> dict[str, Any] | None:
    """What a backend measured about its own run, as plain values, or ``None``.

    ``measurements()`` is optional on a predictor: peak memory and the like are known only
    inside its process. A method that raises, or returns something that is not plain values,
    is logged and skipped: a lost measurement must never cost a run its predictions.
    """
    method = getattr(predictor, "measurements", None)
    if not callable(method):
        return None
    try:
        value = method()
        if not isinstance(value, dict):
            raise TypeError("measurements() must return a mapping")
        plain = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
        check_no_secrets(plain, "the predictor's measurements")
    except Exception as error:  # whatever a backend raises here is not worth a run
        logger.warning(
            "the predictor's measurements were not stored: %s: %s", type(error).__name__, error
        )
        return None
    return plain


def _read_measurements(run_id: str, ctx: RunnerContext) -> dict[str, Any] | None:
    try:
        value = json.loads((_run_folder(run_id, ctx) / MEASUREMENTS_NAME).read_text("utf-8"))
    except (FileNotFoundError, ValueError):
        return None
    return value if isinstance(value, dict) else None


# --------------------------------------------------------------------------- resuming


def _resumable(
    resume: str, key: dict[str, Any], ids: Sequence[str], ctx: RunnerContext
) -> list[PredictionRecord]:
    """The lines of an earlier run that the new run starts from. Raises ``ResumeError``."""
    try:
        earlier = _read_evaluation(resume, ctx)
    except EvaluationError as error:
        raise ResumeError(f"run {resume} cannot be continued: {error}") from error
    differs = [name for name in KEY_FIELDS if earlier["key"].get(name) != key[name]]
    if differs:
        raise ResumeError(
            f"run {resume} cannot be continued: it was made with another "
            f"{', '.join(name.replace('_', ' ') for name in differs)}. Start a new run instead"
        )
    try:
        records, _ = read_predictions(_run_folder(resume, ctx) / PREDICTIONS_NAME, ids)
    except PredictionsError as error:
        raise ResumeError(f"run {resume} cannot be continued: {error}") from error
    if len(records) == len(ids):
        raise ResumeError(
            f"run {resume} already has a prediction for every example asked for; print it "
            "with --summary"
        )
    return records


def _recover_usage(
    done: Sequence[PredictionRecord], model_id: str | None, by_run: str, ctx: RunnerContext
) -> None:
    """Record what an earlier run consumed, if it ended without recording it.

    A run that was killed writes no ledger record. Its lines are in the file with their
    token counts, so the record is rebuilt from them and marked ``computed``.
    """
    recorded = _ledger_examples(ctx)
    for run_id, count in Counter(line.run_id for line in done).items():
        missing = count - recorded[run_id]
        if missing <= 0:
            continue
        lines = [line for line in done if line.run_id == run_id]
        try:
            manifest = load_manifest(run_id, ctx.tracking_cfg)
        except TrackingError:
            manifest = load_manifest(by_run, ctx.tracking_cfg)
        fields: dict[str, Any] = {}
        whole = missing == count  # nothing was recorded, so the token sums are the run's
        counts_known = all(
            line.input_tokens is not None and line.output_tokens is not None for line in lines
        )
        if whole and model_id and counts_known:
            fields.update(
                model_id=model_id,
                input_tokens=sum(line.input_tokens for line in lines),
                output_tokens=sum(line.output_tokens for line in lines),
            )
        Ledger(ctx.tracking_cfg).append(
            LedgerRecord(
                run_id=run_id,
                phase=PHASE,
                path=manifest["path"],
                arm=manifest["arm"],
                where=manifest["where"],
                examples=missing,
                source="computed",
                source_detail=(
                    f"rebuilt from the predictions file by run {by_run}: the run ended "
                    "without recording its usage"
                ),
                **fields,
            )
        )


# --------------------------------------------------------------------------- the run


def run_evaluation(
    predictor: Any,
    split: str,
    *,
    label: str,
    path: str,
    arm: str,
    where: str,
    scope: str,
    kind: str = "eval",
    limit: int | None = None,
    resume: str | None = None,
    description: str = "",
    context: RunnerContext | None = None,
) -> dict[str, Any]:
    """Predict and score a split with ``predictor``; return the run's summary.

    ``limit=N`` predicts the first N examples only: a partial run, which is never a result.
    ``resume=<run_id>`` continues an earlier run in a new one; it raises ``ResumeError``
    unless the split, the prompt, the reply limit, and the predictor are the same.

    A ``PredictorError`` stops the run and is raised again with ``run_id`` set. The run is
    marked failed, its lines are kept, and its usage is recorded.

    A run on the ``test`` split first passes ``check_test_run`` (req_005 v5.01): it raises
    ``TestLockedError`` while the split is locked and ``TestRunRefused`` when the run is not
    one the freeze allows. Both are raised before any example is read or any run exists.
    """
    ctx = (context or RunnerContext()).resolved()
    _check_limit(limit)
    described = describe(predictor)
    check_no_secrets(described, "the predictor's description")
    if split == TEST_SPLIT:
        # After the unlock nothing but this gate protects the test split (req_005 v5.01).
        # Imported here: the gate reads the data package, and only a test run needs it.
        from qlora_finetuning.comparison.gate import check_test_run

        check_test_run(
            described, arm=arm, where=where, scope=scope, limit=limit, resume=resume, ctx=ctx
        )

    examples, records, golds, total, reply_format = _load_gold(split, limit, ctx)
    if limit is not None and limit >= total:
        limit = None
    ids = [record.example_id for record in records]
    split_hash = manifest_hash(ctx.split_cfg)
    key = {
        "split": split,
        "manifest_hash": split_hash,
        "prompt_id": ctx.prompt.prompt_id,
        "max_new_tokens": ctx.max_new_tokens,
        "predictor": {name: described[name] for name in ("name", "model_id", "settings")},
    }
    done = [] if resume is None else _resumable(resume, key, ids, ctx)  # before any run exists

    with start_run(
        kind,
        label,
        path=path,
        arm=arm,
        where=where,
        scope=scope,
        configs=ctx.configs,
        split_manifest_hash=split_hash,
        description=description,
        cfg=ctx.tracking_cfg,
        config_dir=ctx.config_dir,
        git=ctx.git,
        environment=ctx.environment,
    ) as run:
        predictions_path = run.run_dir / PREDICTIONS_NAME
        _write_json(
            run.run_dir / EVALUATION_NAME,
            {
                "schema": SCHEMA,
                "run_id": run.run_id,
                "key": key,
                "limit": limit,
                "split_examples": total,
                "examples_asked": len(ids),
                "continues": resume,
                "predictor": described,
            },
        )
        run.outputs.update(evaluation=EVALUATION_NAME, predictions=PREDICTIONS_NAME)
        append_records(predictions_path, done)
        # The id usage is recorded under: the predictor's own, unless it is billed as another.
        usage_model_id = described.get("usage_model_id") or described["model_id"]
        _recover_usage(done, usage_model_id, run.run_id, ctx)

        batch_size = described["batch_size"]
        every = ctx.config.progress_every
        position = len(done)
        predicted = 0
        tokens = {"input_tokens": 0, "output_tokens": 0}
        tokens_known = True
        started = time.perf_counter()
        try:
            while position < len(examples):
                chunk = range(position, min(position + batch_size, len(examples)))
                requests = [
                    PredictionRequest(
                        example_id=records[i].example_id,
                        system=records[i].system,
                        user=records[i].user,
                        max_new_tokens=ctx.max_new_tokens,
                    )
                    for i in chunk
                ]
                predictions = check_predictions(requests, predictor.predict(requests))
                lines = []
                for i, prediction in zip(chunk, predictions, strict=True):
                    outcome = run_reply(
                        prediction.text, examples[i].table, reply_format, ctx.dsl_config
                    )
                    score = score_reply(
                        outcome, golds[i], ctx.config, records[i].example_id, ctx.dsl_config
                    )
                    lines.append(
                        PredictionRecord(
                            example_id=records[i].example_id,
                            index=i,
                            prompt_id=records[i].prompt_id,
                            run_id=run.run_id,
                            reply=prediction.text,
                            outcome=outcome.to_dict(),
                            score=score.to_dict(),
                            input_tokens=prediction.input_tokens,
                            output_tokens=prediction.output_tokens,
                            latency_seconds=float(prediction.latency_seconds),
                            stop_reason=prediction.stop_reason,
                        )
                    )
                append_records(predictions_path, lines)
                # Counted only now, with the lines on disk.
                for line in lines:
                    if line.input_tokens is None or line.output_tokens is None:
                        tokens_known = False
                    else:
                        tokens["input_tokens"] += line.input_tokens
                        tokens["output_tokens"] += line.output_tokens
                before = position
                position += len(lines)
                predicted += len(lines)
                if position // every > before // every or position == len(examples):
                    logger.info("%s: %d of %d examples predicted", split, position, len(examples))
        except PredictorError as error:
            error.run_id = run.run_id
            logger.error(
                "%s: the predictor failed after %d of %d examples: %s",
                split,
                position,
                len(examples),
                error,
            )
            raise
        finally:
            if predicted:
                usage: dict[str, Any] = {
                    "phase": PHASE,
                    "examples": predicted,
                    "wall_clock_seconds": round(time.perf_counter() - started, 3),
                    "source": "measured",
                    "source_detail": "time.perf_counter around the prediction loop",
                }
                if usage_model_id and tokens_known:
                    usage.update(
                        tokens,
                        model_id=usage_model_id,
                        source=described["usage_source"],
                        source_detail=(
                            "token counts as the predictor reported them, summed over the "
                            "run; wall clock by time.perf_counter around the prediction loop"
                        ),
                    )
                run.record(**usage)
            notes = _measurements(predictor)
            if notes is not None:
                _write_json(run.run_dir / MEASUREMENTS_NAME, notes)
                run.outputs["measurements"] = MEASUREMENTS_NAME

        summary = summarize_run(run.run_id, ctx)
        _write_json(run.run_dir / SUMMARY_NAME, summary)
        run.outputs["summary"] = SUMMARY_NAME
        if summary["result"]:
            append_line(ctx.config.results_file, _results_line(summary, run.manifest))
    return summary


# --------------------------------------------------------------------------- the summary


def _check(check_id: str, name: str, problems: list[str], ok_detail: str) -> dict[str, Any]:
    return {
        "id": check_id,
        "name": name,
        "passed": not problems,
        "detail": "; ".join(problems) if problems else ok_detail,
    }


def _some(ids: Sequence[str], what: str) -> list[str]:
    if not ids:
        return []
    shown = ", ".join(ids[:3]) + (", ..." if len(ids) > 3 else "")
    return [f"{len(ids):,} lines {what}: {shown}"]


def _rate(count: int, total: int) -> dict[str, Any]:
    return {"count": count, "rate": count / total if total else None}


def summarize_run(run_id: str, context: RunnerContext | None = None) -> dict[str, Any]:
    """The summary of a run, finished or not, computed from its predictions file.

    Every stored outcome is scored again under the current ``config/evaluation.yaml``, so
    the totals are right even if an allowance changed after the run. No predictor is used.
    The result holds no time stamp: asking twice gives the same answer.
    """
    ctx = (context or RunnerContext()).resolved()
    evaluation = _read_evaluation(run_id, ctx)
    key = evaluation["key"]
    split, limit = key["split"], evaluation.get("limit")
    manifest = load_manifest(run_id, ctx.tracking_cfg)

    _, records, golds, total, _ = _load_gold(split, limit, ctx)
    ids = [record.example_id for record in records]
    lines, dropped = read_predictions(_run_folder(run_id, ctx) / PREDICTIONS_NAME, ids)

    config = ctx.config
    tight_config = dataclasses.replace(config, execution=config.tight_execution)
    scores, tight, stale, unscored = [], [], [], []
    for line, gold in zip(lines, golds, strict=False):
        outcome = outcome_from_dict(line.outcome)
        if (outcome.value is None) == (outcome.error_code is None):
            # Neither a value nor an error, or both: not something run_reply wrote. The line
            # counts as wrong on everything and fails R5; it is never scored as it stands.
            unscored.append(line.example_id)
            scores.append(ExampleScore(line.example_id, False, False, False, False, False))
            tight.append(False)
            continue
        score = score_reply(outcome, gold, config, line.example_id, ctx.dsl_config)
        scores.append(score)
        tight.append(
            score_reply(outcome, gold, tight_config, line.example_id, ctx.dsl_config)
            .execution_correct
        )
        if line.score != score.to_dict():
            stale.append(line.example_id)
        if line.score.get("example_id") != line.example_id:
            unscored.append(line.example_id)

    flagged = load_gold_flags(config)
    totals = summarize(scores, flagged)
    unflagged_tight = [
        right for line, right in zip(lines, tight, strict=True) if line.example_id not in flagged
    ]

    current_hash = manifest_hash(ctx.split_cfg)
    other_prompt = [line.example_id for line in lines if line.prompt_id != ctx.prompt.prompt_id]
    question_problems = _some(other_prompt, "were predicted from another prompt wording")
    if key.get("prompt_id") != ctx.prompt.prompt_id:
        question_problems.append("the run was made with another prompt wording")
    if not (key.get("manifest_hash") == current_hash == manifest.get("split_manifest_hash")):
        question_problems.append("the run was made with another split manifest")

    recorded = _ledger_examples(ctx)
    per_run = Counter(line.run_id for line in lines)
    # More in the ledger than in the file is possible and truthful: a prediction that was
    # paid for and then lost with a cut-off line. Fewer means usage went unrecorded.
    usage_problems = [
        f"run {name} has {count:,} lines and only {recorded[name]:,} examples in the ledger"
        for name, count in sorted(per_run.items())
        if recorded[name] < count
    ]

    asked = len(ids)
    partial = limit is not None and asked < total
    complete_problems = []
    if len(lines) != asked:
        complete_problems.append(f"{len(lines):,} of {asked:,} examples have a prediction")
    if dropped:
        complete_problems.append("the last line is cut off")
    checks = [
        _check(
            "R1",
            "Complete",
            complete_problems,
            f"{asked:,} of {asked:,} examples, in dataset order"
            + (f" (the first {asked:,} of {total:,}: a partial run)" if partial else ""),
        ),
        _check(
            "R2",
            "Same question",
            question_problems,
            f"prompt {ctx.prompt.prompt_id[:12]}..., split manifest {current_hash[:12]}...",
        ),
        _check(
            "R3",
            "Scores current",
            _some(stale, "hold a score that differs from the one computed now"),
            "every stored score equals the score computed now",
        ),
        _check(
            "R4",
            "Usage adds up",
            usage_problems,
            f"the ledger records of {len(per_run):,} run(s) cover {len(lines):,} predictions",
        ),
        _check(
            "R5",
            "Nothing unscored",
            _some(unscored, "have no outcome or a score for another example"),
            "every line has an outcome and a score",
        ),
    ]
    complete = all(check["passed"] for check in checks)
    backend = _read_measurements(run_id, ctx)
    summary = {
        "schema": SCHEMA,
        "run_id": run_id,
        "kind": manifest["kind"],
        "label": manifest["label"],
        "path": manifest["path"],
        "arm": manifest["arm"],
        "where": manifest["where"],
        "scope": manifest["scope"],
        "git_commit": manifest["git"]["commit"],
        "continues": evaluation.get("continues"),
        "split": split,
        "manifest_hash": key.get("manifest_hash"),
        "prompt_id": key.get("prompt_id"),
        "max_new_tokens": key.get("max_new_tokens"),
        "predictor": evaluation.get("predictor"),
        "limit": limit,
        "split_examples": total,
        "examples_asked": asked,
        "examples_predicted": len(lines),
        "partial": partial,
        "complete": complete,
        "result": complete and not partial,
        "settings": {
            "execution": dataclasses.asdict(config.execution),
            "answer": dataclasses.asdict(config.answer),
        },
        "all": totals["all"],
        "unflagged": totals["unflagged"],
        "flagged_examples": totals["flagged_examples"],
        "diagnostic": {
            "rule": dataclasses.asdict(config.tight_execution),
            "execution_accuracy_tight": _rate(sum(tight), len(tight)),
            "execution_accuracy_tight_unflagged": _rate(sum(unflagged_tight), len(unflagged_tight)),
        },
        "usage": {
            "input_tokens": _token_total(lines, "input_tokens"),
            "output_tokens": _token_total(lines, "output_tokens"),
            "predictions_without_token_counts": sum(
                line.input_tokens is None or line.output_tokens is None for line in lines
            ),
        },
        "latency_seconds": latency_stats([line.latency_seconds for line in lines]),
        "stop_reasons": dict(sorted(Counter(line.stop_reason for line in lines).items())),
        "checks": checks,
    }
    if backend is not None:  # only a predictor with measurements() leaves any
        summary["backend"] = backend
    return summary


def run_predictions(run_id: str, context: RunnerContext | None = None) -> list[PredictionRecord]:
    """The prediction lines of a run, in dataset order, for a report that compares runs.

    The file is checked as ``summarize_run`` checks it: a clean beginning of the examples the
    run was asked for. The lines hold replies, so a caller that writes a committed file
    takes ids, scores, and counts from them and nothing else.
    """
    ctx = (context or RunnerContext()).resolved()
    evaluation = _read_evaluation(run_id, ctx)
    _, records, _, _, _ = _load_gold(evaluation["key"]["split"], evaluation.get("limit"), ctx)
    ids = [record.example_id for record in records]
    lines, _ = read_predictions(_run_folder(run_id, ctx) / PREDICTIONS_NAME, ids)
    return lines


def run_gold(run_id: str, context: RunnerContext | None = None) -> list[Gold]:
    """The gold replies of the examples a run was asked, in dataset order (req_003 v3.05).

    For a report that says why replies were wrong. A gold reply is dataset text: a caller
    compares against it and writes none of it to a committed file.
    """
    ctx = (context or RunnerContext()).resolved()
    evaluation = _read_evaluation(run_id, ctx)
    _, _, golds, _, _ = _load_gold(evaluation["key"]["split"], evaluation.get("limit"), ctx)
    return golds


def _results_line(summary: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """One line of counts for the committed results file. No example text, no reply."""
    every, unflagged = summary["all"], summary["unflagged"]
    line = {
        "schema": SCHEMA,
        "at": timestamp(),
        "run_id": summary["run_id"],
        "kind": manifest["kind"],
        "label": manifest["label"],
        "path": manifest["path"],
        "arm": manifest["arm"],
        "where": manifest["where"],
        "scope": manifest["scope"],
        "split": summary["split"],
        "examples": every["examples"],
        "format_validity": every["format_validity"]["count"],
        "execution_accuracy": every["execution_accuracy"]["count"],
        "answer_accuracy": every["answer_accuracy"]["count"],
        "program_match": every["program_match"]["count"],
        "execution_accuracy_tight": summary["diagnostic"]["execution_accuracy_tight"]["count"],
        "unflagged_examples": unflagged["examples"],
        "unflagged_execution_accuracy": unflagged["execution_accuracy"]["count"],
        "input_tokens": summary["usage"]["input_tokens"],
        "output_tokens": summary["usage"]["output_tokens"],
        "manifest_hash": summary["manifest_hash"],
        "prompt_id": summary["prompt_id"],
        "predictor": summary["predictor"]["name"],
        "model_id": summary["predictor"]["model_id"],
        "git_commit": manifest["git"]["commit"],
    }
    check_no_secrets(line, "the results line")
    return line
