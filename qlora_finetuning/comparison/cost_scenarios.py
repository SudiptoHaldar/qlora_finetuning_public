"""Serving Qwen, and the break-evens against Nova (req_005 v5.02).

A Qwen arm is billed for the seconds an instance runs; a Nova arm for the tokens of each
question. This module puts both on one scale, from the figures ``cost.arm_figures`` took
from the test runs:

- ``serving``: a batch job at several sizes (its start-up, then the steady time per
  question), and an always-on endpoint at several shares of the measured throughput.
- ``crossing``: where two costs of the form ``fixed + volume * slope`` are equal.
- ``break_evens``: each Qwen arm against each Nova arm of its kind, per scenario, each
  marked reachable or not against the instance's capacity and the account's limits.

Usage::

    from qlora_finetuning.comparison.cost_scenarios import break_evens, crossing, serving

    scenarios = serving(figures, instances, cfg)
    crossing(Decimal("10"), Decimal("0"), Decimal("0"), Decimal("0.5"))["volume"]   # 20

Amounts are ``Decimal``. This module imports neither boto3 nor the training stack.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal
from typing import Any

from qlora_finetuning.comparison.cost_config import CostConfig, InstancePrices

ALWAYS_ON, BATCH, MAKING = "always_on", "batch", "making"
SECONDS_PER_HOUR = Decimal(3600)
THOUSAND = Decimal(1000)
ZERO = Decimal(0)


def serving(
    figures: dict[str, dict[str, Any]], instances: InstancePrices, cfg: CostConfig
) -> dict[str, dict[str, Any]]:
    """Per Qwen arm: a batch job at each size, an always-on endpoint at each utilization."""
    scenarios: dict[str, dict[str, Any]] = {}
    for arm, figure in figures.items():
        per_question = figure.get("seconds_per_question")
        if not per_question:
            continue
        instance_type = figure["instance_type"]
        training = instances.training[instance_type]
        hosting = instances.hosting.get(instance_type)
        startup = figure["startup_seconds"]
        per_hour = SECONDS_PER_HOUR / per_question
        per_month = per_hour * cfg.hours_per_month
        scenario: dict[str, Any] = {
            "instance_type": instance_type,
            "seconds_per_question": per_question,
            "startup_seconds": startup,
            "startup_usd": startup * training / SECONDS_PER_HOUR,
            "steady_usd_per_question": per_question * training / SECONDS_PER_HOUR,
            "questions_per_hour": per_hour,
            "questions_per_month": per_month,
            "batch": [],
            "always_on": None,
        }
        for size in cfg.batch_sizes:
            cost = (startup + size * per_question) * training / SECONDS_PER_HOUR
            scenario["batch"].append(
                {"questions": size, "usd": cost, "per_1000_questions": cost / size * THOUSAND}
            )
        if hosting is not None:
            monthly = hosting * cfg.hours_per_month
            scenario["always_on"] = {
                "hourly_usd": hosting,
                "monthly_usd": monthly,
                "at": [
                    {
                        "utilization": share,
                        "questions_per_month": per_month * share,
                        "per_1000_questions": monthly / (per_month * share) * THOUSAND,
                    }
                    for share in cfg.utilizations
                ],
            }
        scenarios[arm] = scenario
    return scenarios


def crossing(
    fixed_a: Decimal, slope_a: Decimal, fixed_b: Decimal, slope_b: Decimal
) -> dict[str, Any]:
    """Where two costs of the form ``fixed + volume * slope`` are equal.

    Returns ``{"kind": "volume", "volume": v, "cheaper_below": "a"|"b", "cheaper_above": ...}``
    when they cross at a volume above zero, ``{"kind": "always", "cheaper": "a"|"b"}`` when
    one is cheaper at every volume above zero, and ``{"kind": "equal"}`` when they are the
    same line.
    """
    if slope_a == slope_b:
        if fixed_a == fixed_b:
            return {"kind": "equal"}
        return {"kind": "always", "cheaper": "a" if fixed_a < fixed_b else "b"}
    volume = (fixed_b - fixed_a) / (slope_a - slope_b)
    if volume <= 0:
        return {"kind": "always", "cheaper": "a" if slope_a < slope_b else "b"}
    below = "a" if fixed_a < fixed_b else "b"
    return {
        "kind": "volume",
        "volume": volume,
        "cheaper_below": below,
        "cheaper_above": "b" if below == "a" else "a",
    }


def _break_even(
    scenario: str,
    arm_a: str,
    arm_b: str,
    lines: tuple[Decimal, Decimal, Decimal, Decimal],
    limits: dict[str, Decimal | None],
    lifetime: bool = False,
) -> dict[str, Any]:
    names = {"a": arm_a, "b": arm_b}
    found = crossing(*lines)
    entry: dict[str, Any] = {
        "scenario": scenario,
        "pair": [arm_a, arm_b],
        "kind": found["kind"],
        "questions": None,
        "cheaper_below": None,
        "cheaper_above": None,
        "cheaper_always": names.get(found.get("cheaper", "")),
        "reachable": True,
        "limit_note": "no volume to reach",
    }
    if found["kind"] != "volume":
        return entry
    volume = found["volume"].to_integral_value(rounding=ROUND_CEILING)
    entry |= {
        "questions": int(volume),
        "cheaper_below": names[found["cheaper_below"]],
        "cheaper_above": names[found["cheaper_above"]],
    }
    known = {arm: limit for arm, limit in limits.items() if limit is not None}
    if lifetime:
        entry["limit_note"] = "a volume over the models' whole use, not a month's"
        if known:
            months = volume / min(known.values())
            entry["limit_note"] += (
                ": under a month at the smaller monthly limit"
                if months < 1
                else f": about {months:.1f} months at the smaller monthly limit"
            )
        return entry
    over = [arm for arm, limit in known.items() if volume > limit]
    entry["reachable"] = not over
    entry["limit_note"] = (
        "within what both sides can serve in a month"
        if not over
        else "above what "
        + " and ".join(
            f"{arm} can serve in a month ({int(known[arm].to_integral_value()):,})" for arm in over
        )
    )
    return entry


def break_evens(
    figures: dict[str, dict[str, Any]],
    scenarios: dict[str, dict[str, Any]],
    making: dict[str, Decimal],
    storage: dict[str, dict[str, Any]],
    cfg: CostConfig,
) -> list[dict[str, Any]]:
    """Each Qwen arm against each Nova arm of its kind (tuned with tuned), per scenario."""
    tuned = set(making) | set(storage)
    entries: list[dict[str, Any]] = []
    for arm_a, scenario in scenarios.items():
        for arm_b, figure in figures.items():
            if figure.get("path") != "B" or (arm_a in tuned) != (arm_b in tuned):
                continue
            if not figure["questions"]:
                continue
            per_question_b = figure["usd"] / figure["questions"]
            monthly_b = (storage.get(arm_b) or {}).get("per_month", ZERO)
            requests = cfg.requests_per_minute.get(arm_b)
            limits = {
                arm_a: scenario["questions_per_month"],
                arm_b: Decimal(requests * 60 * cfg.hours_per_month) if requests else None,
            }
            if scenario["always_on"]:
                entries.append(
                    _break_even(
                        ALWAYS_ON,
                        arm_a,
                        arm_b,
                        (scenario["always_on"]["monthly_usd"], ZERO, monthly_b, per_question_b),
                        limits,
                    )
                )
            entries.append(
                _break_even(
                    BATCH,
                    arm_a,
                    arm_b,
                    (
                        scenario["startup_usd"] * cfg.batch_jobs_per_month,
                        scenario["steady_usd_per_question"],
                        monthly_b,
                        per_question_b,
                    ),
                    limits,
                )
            )
            if arm_a in tuned:
                entries.append(
                    _break_even(
                        MAKING,
                        arm_a,
                        arm_b,
                        (
                            making.get(arm_a, ZERO),
                            scenario["steady_usd_per_question"],
                            making.get(arm_b, ZERO),
                            per_question_b,
                        ),
                        limits,
                        lifetime=True,
                    )
                )
    return entries
