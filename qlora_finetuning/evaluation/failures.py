"""Why a reply was wrong, in one computed category (req_003 v3.05).

Every wrong reply of an evaluation run gets exactly one category, from its stored
predictions line and the gold program. No model is asked and nothing is judged by reading,
so the same run always gives the same counts, and the categories are the same for every
arm. The first rule that fits decides:

=================  ============================================================================
``cut_off``        the reply hit the token limit (its stop reason is not ``end``)
``no_program``     no program line was read
``bad_program``    the program does not parse (the error code says how)
``not_executable`` it parses and cannot be run (a row that is not there, a division by zero)
``kind``           a number where yes or no was due, or the reverse
``steps_more``     more steps than the gold program
``steps_fewer``    fewer steps
``operation``      the same number of steps, another operation in at least one
``evidence``       the same operations; an argument that is neither gold's nor gold's reversed
``order``          the same operations and numbers; the two arguments of a step reversed
``scale``          the gold program itself, and a result off by a configured factor
``sign``           the gold program itself, and the negative of the gold result
``other``          anything left
=================  ============================================================================

A reversed division gives the reciprocal, not the negative: that is why programs are
compared, and why the value's own label (``mismatch``) is kept beside the category and
never decides it.

Usage::

    from qlora_finetuning.evaluation.failures import count_failures, run_failures

    failures = run_failures(run_id)            # one entry per example, right ones included
    counts = count_failures(failures)
    counts["right"], counts["categories"]["order"], counts["error_codes"]

**Import this module by its own name**: it reads runs through the runner, which the package
does not export (see ``qlora_finetuning.evaluation``). An entry holds ids, labels, and
counts. It holds no reply and no gold text, so it may be written to a committed report.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from qlora_finetuning.dsl import DslConfig
from qlora_finetuning.dsl.errors import EXECUTION, PROGRAM, REPLY, ProgramError
from qlora_finetuning.dsl.language import TABLE_OPERATIONS
from qlora_finetuning.dsl.outcome import ReplyOutcome
from qlora_finetuning.dsl.program import Program, parse_program
from qlora_finetuning.evaluation.metrics import (
    KIND,
    SCALE,
    SIGN,
    EvaluationConfig,
    Gold,
    _same_argument,
    score_reply,
)
from qlora_finetuning.evaluation.predictor import END

RIGHT = "right"
CUT_OFF, NO_PROGRAM, BAD_PROGRAM, NOT_EXECUTABLE = (
    "cut_off",
    "no_program",
    "bad_program",
    "not_executable",
)
STEPS_MORE, STEPS_FEWER, OPERATION, EVIDENCE, ORDER, OTHER = (
    "steps_more",
    "steps_fewer",
    "operation",
    "evidence",
    "order",
    "other",
)
# In the order the rules are tried, which is also the order of every table.
CATEGORIES = (
    CUT_OFF,
    NO_PROGRAM,
    BAD_PROGRAM,
    NOT_EXECUTABLE,
    KIND,
    STEPS_MORE,
    STEPS_FEWER,
    OPERATION,
    EVIDENCE,
    ORDER,
    SCALE,
    SIGN,
    OTHER,
)
_STAGES = {REPLY: NO_PROGRAM, PROGRAM: BAD_PROGRAM, EXECUTION: NOT_EXECUTABLE}
UNKNOWN_COMPANY = "(unknown)"


@dataclass(frozen=True)
class Failure:
    """One example of a run: its category, and what a report groups by. No text."""

    example_id: str
    category: str  # ``right``, or one of ``CATEGORIES``
    error_code: str | None
    mismatch: str | None  # the value's own label: sign, scale, kind, other
    format_valid: bool
    deviations: tuple[str, ...]
    answer_correct: bool
    stop_reason: str
    gold_steps: int  # 0 when the gold program does not parse
    gold_operations: tuple[str, ...]
    input_tokens: int | None

    @property
    def right(self) -> bool:
        return self.category == RIGHT


def _compare_programs(predicted: Program, gold: Program) -> str:
    """The category of a program that ran and gave a wrong result, from its shape."""
    if len(predicted.steps) != len(gold.steps):
        return STEPS_MORE if len(predicted.steps) > len(gold.steps) else STEPS_FEWER
    if any(
        step.operation != other.operation
        for step, other in zip(predicted.steps, gold.steps, strict=True)
    ):
        return OPERATION
    reversed_somewhere = False
    for step, other in zip(predicted.steps, gold.steps, strict=True):
        first, second = step.arguments
        gold_first, gold_second = other.arguments
        if step.operation in TABLE_OPERATIONS:  # its second argument is a placeholder
            if not _same_argument(first, gold_first):
                return EVIDENCE
            continue
        if _same_argument(first, gold_first) and _same_argument(second, gold_second):
            continue
        if _same_argument(first, gold_second) and _same_argument(second, gold_first):
            reversed_somewhere = True
            continue
        return EVIDENCE
    return ORDER if reversed_somewhere else ""


def classify(
    outcome: ReplyOutcome,
    stop_reason: str,
    gold: Gold,
    config: EvaluationConfig,
    dsl_config: DslConfig | None = None,
) -> str:
    """The category of one reply: ``right``, or the first rule of ``CATEGORIES`` that fits."""
    if (outcome.value is None) == (outcome.error_code is None):
        return OTHER  # neither a value nor an error: not something run_reply wrote
    score = score_reply(outcome, gold, config, dsl_config=dsl_config)
    if score.execution_correct:
        return RIGHT
    if stop_reason != END:
        return CUT_OFF
    if not score.executed:
        return _STAGES.get(str(outcome.error_stage), OTHER)
    if score.mismatch == KIND:
        return KIND
    if gold.program is None or outcome.program is None:
        return OTHER
    try:
        predicted = parse_program(outcome.program, dsl_config or _dsl_config())
    except ProgramError:
        return OTHER
    shape = _compare_programs(predicted, gold.program)
    if shape:
        return shape
    # The gold program itself, and still a wrong result: only the value can say more.
    return score.mismatch if score.mismatch in (SCALE, SIGN) else OTHER


def _dsl_config() -> DslConfig:
    from qlora_finetuning.dsl import load_dsl_config

    return load_dsl_config()


def company_of(example_id: str) -> str:
    """The company an example is from: FinQA ids start with its ticker and a slash."""
    head, slash, _ = example_id.partition("/")
    return head if slash and head else UNKNOWN_COMPANY


def run_failures(run_id: str, context: Any = None) -> list[Failure]:
    """One entry per predicted example of an evaluation run, in dataset order.

    Scores are computed now, from the stored outcomes, under the current allowances.
    Raises ``EvaluationError`` for a run that is not here or whose file cannot be trusted.
    """
    # The runner imports the data package, which imports this package's metrics.
    from qlora_finetuning.evaluation.predictions import outcome_from_dict
    from qlora_finetuning.evaluation.runner import RunnerContext, run_gold, run_predictions

    ctx = (context or RunnerContext()).resolved()
    lines = run_predictions(run_id, ctx)
    golds = run_gold(run_id, ctx)
    failures = []
    for line, gold in zip(lines, golds, strict=False):
        outcome = outcome_from_dict(line.outcome)
        unreadable = (outcome.value is None) == (outcome.error_code is None)
        score = None if unreadable else score_reply(outcome, gold, ctx.config, None, ctx.dsl_config)
        program = gold.program
        failures.append(
            Failure(
                example_id=line.example_id,
                category=classify(outcome, line.stop_reason, gold, ctx.config, ctx.dsl_config),
                error_code=outcome.error_code,
                mismatch=score.mismatch if score else None,
                format_valid=bool(score and score.format_valid),
                deviations=tuple(outcome.deviations),
                answer_correct=bool(score and score.answer_correct),
                stop_reason=line.stop_reason,
                gold_steps=len(program.steps) if program else 0,
                gold_operations=tuple(step.operation for step in program.steps) if program else (),
                input_tokens=line.input_tokens,
            )
        )
    return failures


def _sorted(counter: Counter) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def count_failures(failures: Iterable[Failure]) -> dict[str, Any]:
    """The counts of a run: right, each category, and what explains them further."""
    failures = list(failures)
    wrong = [item for item in failures if not item.right]
    by_category = Counter(item.category for item in wrong)
    not_run = (BAD_PROGRAM, NOT_EXECUTABLE, NO_PROGRAM)
    return {
        "examples": len(failures),
        "right": len(failures) - len(wrong),
        "categories": {name: by_category.get(name, 0) for name in CATEGORIES},
        "error_codes": _sorted(
            Counter(
                str(item.error_code)
                for item in wrong
                if item.category in not_run and item.error_code
            )
        ),
        "value_labels": _sorted(
            Counter(item.mismatch for item in wrong if item.mismatch in (SIGN, SCALE))
        ),
        "wrong_in_the_format": sum(item.format_valid for item in wrong),
        "answer_right_program_wrong": sum(item.answer_correct for item in wrong),
    }


def failure_counts(run_id: str, context: Any = None) -> dict[str, Any]:
    """``count_failures`` of a stored run."""
    return count_failures(run_failures(run_id, context))


def format_table(failures: Sequence[Failure]) -> dict[str, Any]:
    """Replies that are not in the strict format, right or wrong, by deviation."""
    outside = [item for item in failures if not item.format_valid]
    return {
        "in_the_format": len(failures) - len(outside),
        "not_in_the_format": len(outside),
        "not_in_the_format_and_right": sum(item.right for item in outside),
        "deviations": _sorted(
            Counter(name for item in outside for name in (item.deviations or ("(none recorded)",)))
        ),
        "stop_reasons": _sorted(Counter(item.stop_reason for item in failures)),
    }
