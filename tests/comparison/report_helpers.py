"""Invented inputs for the comparison report's tests.

A thousand invented examples from twenty invented companies, fifty each, and the six arms of
``tests/comparison/cost_helpers.py`` with the same counts of right answers, so the invented
cost record of that module is about the same runs. Nothing here is a real example, a real
run, or a real result.

Which examples each arm gets right (execution accuracy), by position:

- A0 0 to 9, B0-Lite 0 to 199, B0-Pro 0 to 399
- A1 0 to 699, B1-Lite 100 to 749, B1-Pro 0 to 749

So A1 and B1-Lite share 600, A1 alone has 100, and B1-Lite alone has 50.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from qlora_finetuning.comparison.cost import build_record as build_cost_record
from qlora_finetuning.comparison.cost_config import load_cost_config
from qlora_finetuning.comparison.outcomes import METRICS, ArmOutcomes
from qlora_finetuning.comparison.report_config import ReportConfig, load_report_config
from qlora_finetuning.comparison.report_record import ReportInputs
from tests.comparison import cost_helpers as h

EXAMPLES = h.QUESTIONS
IDS = tuple(f"C{index // 50:02d}/2015/page_{index}.pdf-1" for index in range(EXAMPLES))
FREEZE = "f" * 64
SHARED = {"configs": {"prompt": "a" * 64, "evaluation": "b" * 64}, "gold_flags_sha256": "c" * 64}
TUNED = ("A1", "B1-Lite", "B1-Pro")
SPANS = {
    "A0": range(0, 10),
    "A1": range(0, 700),
    "B0-Lite": range(0, 200),
    "B1-Lite": range(100, 750),
    "B0-Pro": range(0, 400),
    "B1-Pro": range(0, 750),
}
WRONG = ("steps_fewer", "evidence", "order")
DEVIATIONS = [
    {"date": "2026-01-05", "what": "An invented deviation.", "recorded_in": "a/file.md"}
]


def outcomes(arm: str, right: Any = None, **changes: Any) -> ArmOutcomes:
    """One arm's invented outcomes. ``right`` replaces the positions it gets right."""
    chosen = set(SPANS[arm] if right is None else right)
    execution = tuple(index in chosen for index in range(EXAMPLES))
    in_format = tuple(arm in TUNED or ok or index % 2 == 0 for index, ok in enumerate(execution))
    by_metric = {metric: execution for metric in METRICS} | {"format_validity": in_format}
    values: dict[str, Any] = {
        "arm": arm,
        "run_id": h.TEST_RUNS[arm],
        "example_ids": IDS,
        "right": by_metric,
        "categories": tuple(
            "right" if ok else ("bad_program" if arm == "A0" else WRONG[index % 3])
            for index, ok in enumerate(execution)
        ),
        "gold_steps": tuple(index % 3 + 1 for index in range(EXAMPLES)),
        "stop_reasons": ("end",) * EXAMPLES,
        "counts": {"examples": EXAMPLES}
        | {metric: sum(values) for metric, values in by_metric.items()},
    }
    return ArmOutcomes(**(values | changes))


def all_outcomes(**replaced: ArmOutcomes) -> dict[str, ArmOutcomes]:
    return {arm: replaced.get(arm) or outcomes(arm) for arm in SPANS}


def six_arm(arms: dict[str, ArmOutcomes] | None = None) -> dict[str, Any]:
    """The invented six-arm record, with the counts of ``arms`` (the default outcomes)."""
    arms = arms or all_outcomes()
    record = h.six_arm() | {"freeze_sha256": FREEZE}
    for arm, entry in record["arms"].items():
        entry["all"] = {"examples": EXAMPLES} | {
            metric: {"count": arms[arm].counts[metric]} for metric in METRICS
        }
        entry["latency_seconds"] = {"mean": 0.5, "median": 0.4, "p95": 0.91234}
    return record


def cost_record(final: bool = False) -> dict[str, Any]:
    """The invented cost record: provisional, or reconciled with invented bill amounts."""
    inputs = h.inputs(pro_billed=True, bills=h.covering_bills()) if final else h.inputs()
    inputs.six_arm = inputs.six_arm | {"freeze_sha256": FREEZE}
    return build_cost_record(inputs, load_cost_config())


def freeze() -> dict[str, Any]:
    return {
        "shared": SHARED,
        "arms": {
            arm: {"dev": {"examples": 100, "execution_accuracy": h.CORRECT[arm] // 10}}
            for arm in SPANS
        },
    }


def inputs(**changes: Any) -> ReportInputs:
    arms = changes.pop("arms", None) or all_outcomes()
    values: dict[str, Any] = {
        "six_arm": six_arm(),
        "cost": cost_record(),
        "freeze": freeze(),
        "freeze_sha256": FREEZE,
        "shared_now": SHARED,
        "arms": arms,
        "read_problems": [],
        "tuned": TUNED,
        "test_ids": IDS,
    }
    return ReportInputs(**(values | changes))


def report_cfg(root: Any, **changes: Any) -> ReportConfig:
    """The real settings under a temporary root, with few resamples so tests are quick."""
    return dataclasses.replace(load_report_config(root=root), resamples=300, **changes)


def check(record: dict[str, Any], check_id: str) -> dict[str, Any]:
    return next(item for item in record["checks"] if item["id"] == check_id)


def failed(record: dict[str, Any]) -> list[str]:
    return [item["id"] for item in record["checks"] if not item["passed"]]


def pair(record: dict[str, Any], first: str, second: str) -> dict[str, Any]:
    return next(item for item in record["pairs"] if item["pair"] == [first, second])

