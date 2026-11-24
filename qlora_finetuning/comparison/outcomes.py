"""What each arm did on each test example, without any text (req_005 v5.03).

For a stored evaluation run this module gives, per example and in dataset order: whether
the reply was right under each of the four metrics, its error category, the number of steps
in the gold program, and how the reply stopped. Scores come from ``score_reply`` and
categories from ``classify``, as the runner and the error analysis compute them; the totals
come from ``summarize``. Nothing is judged here.

**No reply, question, or gold text leaves this module**: an ``ArmOutcomes`` holds example
ids, true and false, labels, and counts, so it may feed a committed record. Reading a run
on the test split loads test rows on this machine (the execution plan's principle 6, as
amended for this version); no model is asked and nothing is uploaded.

Usage::

    from qlora_finetuning.comparison.outcomes import companies, read_arms

    arms, problems = read_arms({"A1": "20261123T...-test-074f", ...}, ctx)
    arms["A1"].right["execution_accuracy"]      # one bool per test example
    arms["A1"].categories, arms["A1"].gold_steps, arms["A1"].counts
    companies(arms["A1"].example_ids)

This module imports neither boto3 nor the training stack.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qlora_finetuning.evaluation.failures import classify, company_of
from qlora_finetuning.evaluation.metrics import score_reply, summarize
from qlora_finetuning.settings import ConfigError

# The four metrics, primary first, and the field of a score that answers each.
METRICS = {
    "execution_accuracy": "execution_correct",
    "answer_accuracy": "answer_correct",
    "program_match": "program_match",
    "format_validity": "format_valid",
}
PRIMARY = "execution_accuracy"


@dataclass(frozen=True)
class ArmOutcomes:
    """One arm's test run, example by example. Ids, labels, and counts only."""

    arm: str
    run_id: str
    example_ids: tuple[str, ...]
    right: dict[str, tuple[bool, ...]]  # metric -> one answer per example
    categories: tuple[str, ...]  # ``right``, or one of ``failures.CATEGORIES``
    gold_steps: tuple[int, ...]  # 0 when the gold program does not parse
    stop_reasons: tuple[str, ...]
    counts: dict[str, int]  # ``examples`` and each metric, as ``summarize`` totals them


def read_arm(arm: str, run_id: str, context: Any = None, golds: Sequence[Any] | None = None):
    """The outcomes of one stored run. ``golds`` may be passed when another run of the same
    examples has already loaded them.

    Raises ``ConfigError`` (``EvaluationError``) for a run that is not here or whose file
    cannot be trusted.
    """
    # Imported here: the runner imports the data package, which a module may not need.
    from qlora_finetuning.evaluation.predictions import outcome_from_dict
    from qlora_finetuning.evaluation.runner import RunnerContext, run_gold, run_predictions

    ctx = (context or RunnerContext()).resolved()
    lines = run_predictions(run_id, ctx)
    if golds is None:
        golds = run_gold(run_id, ctx)
    if len(lines) > len(golds):
        raise ConfigError(f"run {run_id} has more predictions than there are gold replies")

    scores, categories, steps = [], [], []
    for line, gold in zip(lines, golds, strict=False):
        outcome = outcome_from_dict(line.outcome)
        scores.append(score_reply(outcome, gold, ctx.config, line.example_id, ctx.dsl_config))
        categories.append(classify(outcome, line.stop_reason, gold, ctx.config, ctx.dsl_config))
        steps.append(len(gold.program.steps) if gold.program else 0)

    block = summarize(scores)["all"]
    return ArmOutcomes(
        arm=arm,
        run_id=run_id,
        example_ids=tuple(line.example_id for line in lines),
        right={
            metric: tuple(bool(getattr(score, field)) for score in scores)
            for metric, field in METRICS.items()
        },
        categories=tuple(categories),
        gold_steps=tuple(steps),
        stop_reasons=tuple(str(line.stop_reason) for line in lines),
        counts={"examples": block["examples"]}
        | {metric: block[metric]["count"] for metric in METRICS},
    )


def read_arms(
    runs: Mapping[str, str], context: Any = None
) -> tuple[dict[str, ArmOutcomes], list[str]]:
    """The outcomes of each arm's run, and what could not be read.

    ``runs`` maps an arm to its run id, as the six-arm record names them. The gold replies
    are loaded once, with the first run that can be read.
    """
    from qlora_finetuning.evaluation.runner import RunnerContext, run_gold

    ctx = (context or RunnerContext()).resolved()
    arms: dict[str, ArmOutcomes] = {}
    problems: list[str] = []
    golds = None
    for arm, run_id in runs.items():
        try:
            if golds is None:
                golds = run_gold(run_id, ctx)
            arms[arm] = read_arm(arm, run_id, ctx, golds)
        except ConfigError as error:
            problems.append(f"arm {arm}: run {run_id} cannot be read here ({error})")
    return arms, problems


def companies(example_ids: Sequence[str]) -> list[str]:
    """The company of each example, from its id: the unit the split was made by."""
    return [company_of(example_id) for example_id in example_ids]
