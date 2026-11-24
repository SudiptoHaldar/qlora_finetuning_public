"""The statistics record of the comparison, with checks R1 to R7 (req_005 v5.03).

``build_statistics`` takes the six-arm record, the cost record, the freeze record, and each
arm's outcomes on the test examples, and works out:

* per arm, the four metrics with their confidence intervals (execution accuracy also with
  the interval by company), the error categories, accuracy by gold steps;
* per tested pair, the four counts, the difference with both intervals, McNemar's exact
  p-value, its Holm-adjusted value, and whether the difference is **settled**;
* how the tuned arms overlap, and which examples no arm got right.

A difference is settled when its adjusted p-value is below one minus the confidence and
neither interval includes zero. Otherwise it is "not settled", whatever its p-value (the
owner's decision 2: questions about one company are not independent).

The record holds counts, shares, ids, and labels. **It holds no reply, no question, and no
gold text**, and every dollar figure in it is copied from the cost record.

Usage::

    from qlora_finetuning.comparison.report_record import ReportInputs, build_statistics

    record = build_statistics(inputs, cfg)
    record["passed"], record["provisional"]
    record["arms"]["A1"]["metrics"]["execution_accuracy"]     # count, percent, interval
    record["pairs"][3]["settled"], record["pairs"][3]["p_adjusted"]

Check R8 (the built report) is added by ``report_build``. This module imports neither
boto3 nor the training stack.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from itertools import combinations
from typing import Any

from qlora_finetuning.comparison.gate import shared_differences
from qlora_finetuning.comparison.outcomes import METRICS, PRIMARY, ArmOutcomes, companies
from qlora_finetuning.comparison.report_config import ReportConfig
from qlora_finetuning.comparison.runs import percent
from qlora_finetuning.comparison.statistics import (
    bootstrap_rates,
    holm,
    interval,
    mcnemar_exact,
)
from qlora_finetuning.evaluation.failures import CATEGORIES, RIGHT

SCHEMA = 1
COMMAND = "python -m scripts.comparison_report"
CHECK_NAMES = {
    "R1": "Records in place",
    "R2": "Replies found",
    "R3": "Counts agree",
    "R4": "Nothing frozen has changed",
    "R5": "Intervals hold their figures",
    "R6": "Pairs add up",
    "R7": "Costs taken, not computed",
    "R8": "No text, nothing by hand",
}
SLACK = 1e-12  # for comparing a point with the ends of its own interval


@dataclass
class ReportInputs:
    """Everything the statistics are built from. A test passes invented values."""

    six_arm: dict[str, Any] | None
    cost: dict[str, Any] | None
    freeze: dict[str, Any] | None
    freeze_sha256: str | None  # of the freeze file as it is now
    shared_now: dict[str, Any]  # ``gate.shared_state`` today
    arms: dict[str, ArmOutcomes]
    read_problems: list[str] = field(default_factory=list)
    tuned: tuple[str, ...] = ()
    test_ids: tuple[str, ...] | None = None  # the split's own test ids, when known


def make_check(
    check_id: str, problems: list[str], ok_detail: str, reservations: list[str] | None = None
) -> dict[str, Any]:
    reservations = reservations or []
    if problems:
        detail = "; ".join(problems)
    elif reservations:
        detail = f"{ok_detail}; RESERVATION: {'; '.join(reservations)}"
    else:
        detail = ok_detail
    return {
        "id": check_id,
        "name": CHECK_NAMES[check_id],
        "passed": not problems,
        "reservation": bool(reservations) and not problems,
        "detail": detail,
    }


def p_text(p: Fraction) -> str:
    """A p-value as text in scientific form with three digits: ``3.57e-3``."""
    return f"{Decimal(p.numerator) / Decimal(p.denominator):.2e}"


def _points(share: float) -> str:
    """A share as percentage points with two decimals; never ``-0.00``."""
    text = f"{share * 100:.2f}"
    return "0.00" if text == "-0.00" else text


def _span(values: Any, confidence: int) -> tuple[dict[str, str], tuple[float, float]]:
    low, high = interval(values, confidence)
    return {"low": _points(low), "high": _points(high)}, (low, high)


def _four(first: tuple[bool, ...], second: tuple[bool, ...]) -> dict[str, int]:
    both = sum(a and b for a, b in zip(first, second, strict=True))
    only_first = sum(a and not b for a, b in zip(first, second, strict=True))
    only_second = sum(b and not a for a, b in zip(first, second, strict=True))
    return {
        "both_right": both,
        "only_first": only_first,
        "only_second": only_second,
        "both_wrong": len(first) - both - only_first - only_second,
    }


def _by_steps(outcomes: ArmOutcomes, groups: tuple[int, ...]) -> list[dict[str, Any]]:
    right = outcomes.right[PRIMARY]
    rows = []
    counted = 0
    for position, steps in enumerate(groups):
        last = position == len(groups) - 1
        chosen = [
            index
            for index, value in enumerate(outcomes.gold_steps)
            if (value >= steps if last else value == steps)
        ]
        counted += len(chosen)
        rows.append(
            {
                "steps": f"{steps} or more" if last else str(steps),
                "examples": len(chosen),
                "right": sum(right[index] for index in chosen),
            }
        )
    left = [i for i, value in enumerate(outcomes.gold_steps) if value < groups[0]]
    if left:
        rows.append(
            {"steps": "other", "examples": len(left), "right": sum(right[i] for i in left)}
        )
    for row in rows:
        row["percent"] = percent(row["right"], row["examples"])
    return rows


def _records_problems(inputs: ReportInputs, cfg: ReportConfig) -> tuple[list[str], list[str]]:
    six, cost = inputs.six_arm, inputs.cost
    problems, reservations = [], []
    if six is None:
        problems.append("there is no six-arm record: run scripts.comparison_runs record")
    elif not six.get("passed"):
        problems.append("the six-arm record did not pass C1 to C8")
    if cost is None:
        problems.append("there is no cost record: run scripts.comparison_cost report")
    elif not cost.get("passed"):
        problems.append("the cost record did not pass K1 to K8")
    elif six is not None and cost.get("six_arm_freeze_sha256") != six.get("freeze_sha256"):
        problems.append("the cost record was built under another freeze record")
    elif cost.get("provisional"):
        reservations.append(
            "the cost record is provisional (its bill entries are missing); the report is too"
        )
    arms = list((six or {}).get("arms") or {})
    for pair in cfg.pairs:
        missing = [arm for arm in pair if arm not in arms]
        if six is not None and missing:
            problems.append(f"config report.yaml tests {', '.join(missing)}: not an arm")
    return problems, reservations


def build_statistics(inputs: ReportInputs, cfg: ReportConfig) -> dict[str, Any]:
    """The statistics record with checks R1 to R7. Writes nothing."""
    six = inputs.six_arm or {}
    cost = inputs.cost or {}
    order = list(six.get("arms") or {})
    examples = six.get("test_examples")
    problems: dict[str, list[str]] = {f"R{n}": [] for n in range(1, 8)}

    problems["R1"], reservations = _records_problems(inputs, cfg)

    # R2: each arm's replies are here, complete, and about the same examples in one order.
    problems["R2"] += list(inputs.read_problems)
    present = [arm for arm in order if arm in inputs.arms]
    for arm in order:
        if arm not in inputs.arms:
            if not any(f"arm {arm}:" in text for text in inputs.read_problems):
                problems["R2"].append(f"arm {arm}: its replies were not read")
            continue
        ids = inputs.arms[arm].example_ids
        if len(ids) != examples:
            problems["R2"].append(f"arm {arm}: {len(ids):,} of {examples:,} replies are here")
        elif ids != inputs.arms[present[0]].example_ids:
            problems["R2"].append(f"arm {arm}: its examples are not those of {present[0]}")
        elif inputs.test_ids is not None and ids != tuple(inputs.test_ids):
            problems["R2"].append(f"arm {arm}: its examples are not the test split's")

    # R3: the rescored counts are the six-arm record's.
    for arm in present:
        recorded = six["arms"][arm].get("all") or {}
        counts = inputs.arms[arm].counts
        for metric in METRICS:
            was = (recorded.get(metric) or {}).get("count")
            if counts[metric] != was:
                problems["R3"].append(
                    f"arm {arm}: {metric.replace('_', ' ')} is {counts[metric]:,} when "
                    f"rescored and {was} in the six-arm record"
                )

    # R4: the files every arm shares are the frozen ones.
    if inputs.freeze is None:
        problems["R4"].append("there is no freeze record")
    else:
        differs = shared_differences(inputs.freeze.get("shared") or {}, inputs.shared_now)
        if differs:
            problems["R4"].append(f"changed since the freeze: {', '.join(differs)}")
        if inputs.six_arm is not None and inputs.freeze_sha256 != six.get("freeze_sha256"):
            problems["R4"].append(
                "the freeze record is not the one the six test runs were made under"
            )

    ready = bool(order) and not any(problems[name] for name in ("R1", "R2", "R3", "R4"))
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_by": f"{COMMAND} build",
        "freeze_sha256": six.get("freeze_sha256"),
        "test_examples": examples,
        "cost_provisional": bool(cost.get("provisional")),
        "settings": {
            "resamples": cfg.resamples,
            "seed": cfg.seed,
            "company_seed": cfg.seed + 1,
            "confidence_percent": cfg.confidence_percent,
            "smallest_p": str(cfg.smallest_p),
            "pairs": [list(pair) for pair in cfg.pairs],
            "step_groups": list(cfg.step_groups),
        },
    }
    if ready:
        record |= _figures(inputs, cfg, order, problems)
    else:
        for name in ("R5", "R6", "R7"):
            problems[name].append("not computed: an earlier check failed")

    details = {
        "R1": "the six-arm record passed C1 to C8; the cost record passed K1 to K8",
        "R2": f"{len(order)} arms, {examples:,} replies each, the same examples in one order"
        if examples
        else "",
        "R3": "the four metrics of every arm equal the six-arm record when rescored",
        "R4": "the shared config files and the gold flags have the frozen hashes",
        "R5": "every interval contains its figure",
        "R6": "the four counts of every pair sum to the examples and match the arms' counts",
        "R7": "every dollar figure is the cost record's, for the same six runs",
    }
    checks = [
        make_check(name, problems[name], details[name], reservations if name == "R1" else None)
        for name in problems
    ]
    record["checks"] = checks
    record["passed"] = all(check["passed"] for check in checks)
    record["provisional"] = any(check["reservation"] for check in checks)
    return record


def _figures(
    inputs: ReportInputs, cfg: ReportConfig, order: list[str], problems: dict[str, list[str]]
) -> dict[str, Any]:
    six, cost = inputs.six_arm or {}, inputs.cost or {}
    reference = inputs.arms[order[0]]
    total = len(reference.example_ids)
    confidence = cfg.confidence_percent
    groups = companies(reference.example_ids)

    columns = {
        (arm, metric): inputs.arms[arm].right[metric] for arm in order for metric in METRICS
    }
    names = {key: f"{key[0]}|{key[1]}" for key in columns}
    by_example = bootstrap_rates(
        {names[key]: values for key, values in columns.items()},
        resamples=cfg.resamples,
        seed=cfg.seed,
    )
    by_company = bootstrap_rates(
        {arm: inputs.arms[arm].right[PRIMARY] for arm in order},
        resamples=cfg.resamples,
        seed=cfg.seed + 1,
        groups=groups,
    )
    held: list[tuple[str, float, tuple[float, float]]] = []  # for R5

    arms: dict[str, Any] = {}
    for arm in order:
        outcomes = inputs.arms[arm]
        recorded = six["arms"][arm]
        metrics = {}
        for metric in METRICS:
            count = outcomes.counts[metric]
            span, ends = _span(by_example[names[(arm, metric)]], confidence)
            held.append((f"{arm} {metric}", count / total, ends))
            metrics[metric] = {"count": count, "percent": percent(count, total), "interval": span}
        span, ends = _span(by_company[arm], confidence)
        held.append((f"{arm} by company", outcomes.counts[PRIMARY] / total, ends))
        tally = Counter(outcomes.categories)
        wrong_ids: dict[str, list[str]] = {}
        for example_id, category in zip(outcomes.example_ids, outcomes.categories, strict=True):
            if category != RIGHT and len(wrong_ids.setdefault(category, [])) >= (
                cfg.examples_per_category
            ):
                continue
            if category != RIGHT:
                wrong_ids[category].append(example_id)
        latency = recorded.get("latency_seconds") or {}
        dev = (((inputs.freeze or {}).get("arms") or {}).get(arm) or {}).get("dev") or {}
        arms[arm] = {
            "label": cfg.label(arm),
            "run_id": outcomes.run_id,
            "path": recorded.get("path"),
            "tuned": arm in inputs.tuned,
            "examples": total,
            "metrics": metrics,
            "execution_accuracy_by_company": span,
            "right": tally.get(RIGHT, 0),
            "categories": {name: tally.get(name, 0) for name in CATEGORIES},
            "not_in_the_format": total - outcomes.counts["format_validity"],
            "stop_reasons": dict(sorted(Counter(outcomes.stop_reasons).items())),
            "by_steps": _by_steps(outcomes, cfg.step_groups),
            "latency_seconds": {
                name: f"{float(latency[name]):.3f}"
                for name in ("mean", "median", "p95")
                if name in latency
            },
            "dev": {
                "examples": dev.get("examples"),
                "execution_accuracy": dev.get(PRIMARY),
                "percent": percent(dev.get(PRIMARY) or 0, dev.get("examples") or 0),
            }
            if dev
            else None,
            "example_ids": {name: wrong_ids[name] for name in CATEGORIES if name in wrong_ids},
        }

    # The tested pairs: exact test, Holm over the family, both intervals of the difference.
    alpha = Fraction(100 - confidence, 100)
    fours = [
        _four(inputs.arms[first].right[PRIMARY], inputs.arms[second].right[PRIMARY])
        for first, second in cfg.pairs
    ]
    raw = [mcnemar_exact(four["only_first"], four["only_second"]) for four in fours]
    pairs = []
    for (first, second), four, p, adjusted in zip(cfg.pairs, fours, raw, holm(raw), strict=True):
        difference = (four["only_first"] - four["only_second"]) / total
        example_span, example_ends = _span(
            by_example[names[(first, PRIMARY)]] - by_example[names[(second, PRIMARY)]],
            confidence,
        )
        company_span, company_ends = _span(by_company[first] - by_company[second], confidence)
        held.append((f"{first} minus {second}", difference, example_ends))
        held.append((f"{first} minus {second} by company", difference, company_ends))
        apart = all(low > 0 or high < 0 for low, high in (example_ends, company_ends))
        pairs.append(
            {"pair": [first, second]}
            | four
            | {
                "difference_points": _points(difference),
                "interval": example_span,
                "interval_by_company": company_span,
                "p_value": p_text(p),
                "p_adjusted": p_text(adjusted),
                "settled": bool(adjusted < alpha and apart),
                "favours": first if difference > 0 else second if difference < 0 else None,
            }
        )

    tested = {frozenset(pair) for pair in cfg.pairs}
    other_pairs = [
        {"pair": [first, second]}
        | _four(inputs.arms[first].right[PRIMARY], inputs.arms[second].right[PRIMARY])
        for first, second in combinations(order, 2)
        if frozenset((first, second)) not in tested
    ]

    # R5 and R6.
    for what, point, (low, high) in held:
        if not low - SLACK <= point <= high + SLACK:
            problems["R5"].append(f"{what}: {point:.4f} is outside {low:.4f} to {high:.4f}")
    for entry in pairs + other_pairs:
        first, second = entry["pair"]
        parts = [entry[k] for k in ("both_right", "only_first", "only_second", "both_wrong")]
        if (
            sum(parts) != total
            or entry["both_right"] + entry["only_first"] != inputs.arms[first].counts[PRIMARY]
            or entry["both_right"] + entry["only_second"] != inputs.arms[second].counts[PRIMARY]
        ):
            problems["R6"].append(f"{first} and {second}: the four counts do not add up")

    # How the tuned arms overlap, and what no arm answered.
    tuned = [arm for arm in order if arm in inputs.tuned]
    rows = list(zip(*(inputs.arms[arm].right[PRIMARY] for arm in order), strict=True))
    tuned_rows = (
        list(zip(*(inputs.arms[arm].right[PRIMARY] for arm in tuned), strict=True))
        if tuned
        else []
    )
    unanswered = [
        example_id
        for example_id, row in zip(reference.example_ids, rows, strict=True)
        if not any(row)
    ]

    # R7: the cost record is about these six runs; its figures are copied as text.
    costs: dict[str, Any] = {}
    for arm in order:
        priced = (cost.get("test_pass") or {}).get(arm)
        if not priced:
            problems["R7"].append(f"arm {arm}: the cost record has no test pass for it")
            continue
        if priced.get("run_id") != inputs.arms[arm].run_id:
            problems["R7"].append(f"arm {arm}: the cost record priced another run")
        if priced.get("correct") != inputs.arms[arm].counts[PRIMARY] or (
            priced.get("questions") != total
        ):
            problems["R7"].append(f"arm {arm}: the cost record has other counts")
        costs[arm] = {
            name: priced[name]
            for name in ("usd", "per_1000_questions", "per_1000_correct")
            if priced.get(name) is not None
        }

    return {
        "companies": len(set(groups)),
        "arms": arms,
        "pairs": pairs,
        "other_pairs": other_pairs,
        "overlap": {
            "tuned_arms": tuned,
            "all_tuned_right": sum(all(row) for row in tuned_rows),
            "any_tuned_right": sum(any(row) for row in tuned_rows),
            "no_arm_right": len(unanswered),
        },
        "unanswered_ids": unanswered,
        "costs": costs,
    }


def add_check(record: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    """The record with one more check (R8), and ``passed`` worked out again."""
    checks = [item for item in record["checks"] if item["id"] != check["id"]] + [check]
    return record | {
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
        "provisional": any(item["reservation"] for item in checks),
    }
