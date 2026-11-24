"""The tables and generated sentences of the comparison report (req_005 v5.03).

Everything a reader sees as a figure is made here, from the statistics record and the cost
record: Markdown tables, and sentences whose wording follows the data (who is ahead,
whether a difference is settled, which break-even can be reached). The report template
holds prose and placeholders only, so no figure is typed by hand and none goes stale.

``placeholders`` returns every value the template may name. ``render_statistics`` is the
Markdown copy of the statistics record.

Usage::

    from qlora_finetuning.comparison.report_tables import placeholders, render_statistics

    values = placeholders(record, cost, deviations, cfg, chart_path="comparison/chart.svg")
    values["summary_table"], values["findings"]
    text = render_statistics(record, cfg)

Dollar figures are the cost record's own text, shown with ``cost_report.usd``; this module
multiplies nothing by a price. It imports neither boto3 nor the training stack.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from qlora_finetuning.comparison.cost_report import SCENARIO_NAMES, break_even_text, usd
from qlora_finetuning.comparison.cost_scenarios import ALWAYS_ON
from qlora_finetuning.comparison.outcomes import METRICS, PRIMARY
from qlora_finetuning.comparison.report_config import ReportConfig
from qlora_finetuning.comparison.report_record import COMMAND
from qlora_finetuning.evaluation.failures import CATEGORIES

METRIC_NAMES = {
    "execution_accuracy": "Execution accuracy",
    "answer_accuracy": "Answer accuracy",
    "program_match": "Program match",
    "format_validity": "In the format",
}
CATEGORY_NAMES = {
    "cut_off": "Cut off at the token limit",
    "no_program": "No program line",
    "bad_program": "Program does not parse",
    "not_executable": "Program cannot be run",
    "kind": "Number where yes or no was due, or the reverse",
    "steps_more": "More steps than the gold program",
    "steps_fewer": "Fewer steps than the gold program",
    "operation": "Another operation",
    "evidence": "Other numbers, same operations",
    "order": "Arguments reversed",
    "scale": "Gold program, result off by a factor",
    "sign": "Gold program, result negated",
    "other": "Other",
}


# --------------------------------------------------------------------------- small pieces


def table(headers: list[str], rows: list[list[Any]], right: tuple[int, ...] = ()) -> str:
    """A Markdown table. ``right`` lists the columns (from 0) that hold figures."""
    rule = ["---:" if index in right else "---" for index in range(len(headers))]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(rule) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def span(interval: dict[str, str] | None) -> str:
    return f"{interval['low']} to {interval['high']}" if interval else "—"


def signed(points: str) -> str:
    return points if points.startswith("-") or Decimal(points) == 0 else f"+{points}"


def shown_p(text: str, smallest: Decimal) -> str:
    """A p-value for reading: four decimals, or "below" the smallest one shown."""
    value = Decimal(text)
    return f"below {smallest}" if value < smallest else f"{value:.4f}"


def _turn(interval: dict[str, str], turned: bool) -> dict[str, str]:
    """An interval of a difference, seen from the other arm when ``turned``."""
    if not turned:
        return interval

    def negate(text: str) -> str:
        value = -Decimal(text)
        return "0.00" if value == 0 else f"{value:.2f}"

    return {"low": negate(interval["high"]), "high": negate(interval["low"])}


def _count(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) else "—"


def _join(items: list[str]) -> str:
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _why_open(pair: dict[str, Any], record: dict[str, Any]) -> str:
    """Why a difference is not settled, in the order a reader should hear it."""
    reasons = []
    alpha = Decimal(100 - record["settings"]["confidence_percent"]) / 100
    if Decimal(pair["p_adjusted"]) >= alpha:
        reasons.append(f"its adjusted p-value is not below {alpha.normalize()}")
    for key, name in (("interval", "by example"), ("interval_by_company", "by company")):
        low, high = Decimal(pair[key]["low"]), Decimal(pair[key]["high"])
        if low <= 0 <= high:
            reasons.append(f"the interval {name} includes zero")
    return _join(reasons) or "an interval touches zero"


# --------------------------------------------------------------------------- accuracy


def summary_table(record: dict[str, Any]) -> str:
    rows = []
    for arm, entry in record["arms"].items():
        metric = entry["metrics"][PRIMARY]
        cost = record["costs"].get(arm, {})
        rows.append(
            [
                arm,
                entry["label"],
                f"{metric['percent']}%",
                span(metric["interval"]),
                usd(cost.get("per_1000_questions")),
                usd(cost.get("per_1000_correct")),
            ]
        )
    return table(
        ["Arm", "Model", "Execution accuracy", "Interval", "Per 1,000 questions",
         "Per 1,000 correct answers"],
        rows,
        right=(2, 3, 4, 5),
    )


def accuracy_table(record: dict[str, Any]) -> str:
    rows = [
        [arm]
        + [
            f"{entry['metrics'][metric]['percent']}% ({span(entry['metrics'][metric]['interval'])})"
            for metric in METRICS
        ]
        for arm, entry in record["arms"].items()
    ]
    return table(["Arm"] + [METRIC_NAMES[m] for m in METRICS], rows, right=(1, 2, 3, 4))


def execution_table(record: dict[str, Any]) -> str:
    rows = [
        [
            arm,
            _count(entry["metrics"][PRIMARY]["count"]),
            f"{entry['metrics'][PRIMARY]['percent']}%",
            span(entry["metrics"][PRIMARY]["interval"]),
            span(entry["execution_accuracy_by_company"]),
        ]
        for arm, entry in record["arms"].items()
    ]
    return table(
        ["Arm", "Right", "Execution accuracy", "Interval by example", "Interval by company"],
        rows,
        right=(1, 2, 3, 4),
    )


def dev_test_table(record: dict[str, Any]) -> str:
    rows = []
    for arm, entry in record["arms"].items():
        dev = entry["dev"] or {}
        test = entry["metrics"][PRIMARY]
        gap = (
            signed(f"{Decimal(test['percent']) - Decimal(dev['percent']):.2f}")
            if dev.get("percent")
            else "—"
        )
        rows.append(
            [
                arm,
                f"{_count(dev.get('execution_accuracy'))} of {_count(dev.get('examples'))}",
                f"{dev['percent']}%" if dev.get("percent") else "—",
                f"{_count(test['count'])} of {_count(entry['examples'])}",
                f"{test['percent']}%",
                gap,
            ]
        )
    return table(
        ["Arm", "Dev, right", "Dev", "Test, right", "Test", "Test minus dev, points"],
        rows,
        right=(1, 2, 3, 4, 5),
    )


# --------------------------------------------------------------------------- between arms


def pairs_table(record: dict[str, Any], cfg: ReportConfig) -> str:
    rows = [
        [
            f"{pair['pair'][0]} minus {pair['pair'][1]}",
            _count(pair["both_right"]),
            _count(pair["only_first"]),
            _count(pair["only_second"]),
            _count(pair["both_wrong"]),
            signed(pair["difference_points"]),
            span(pair["interval"]),
            span(pair["interval_by_company"]),
            shown_p(pair["p_value"], cfg.smallest_p),
            shown_p(pair["p_adjusted"], cfg.smallest_p),
            "settled" if pair["settled"] else "**not settled**",
        ]
        for pair in record["pairs"]
    ]
    return table(
        ["Pair", "Both right", "Only the first", "Only the second", "Both wrong",
         "Difference, points", "Interval by example", "Interval by company", "p",
         "p, adjusted", "Verdict"],
        rows,
        right=tuple(range(1, 10)),
    )


def other_pairs_table(record: dict[str, Any]) -> str:
    rows = [
        [
            f"{pair['pair'][0]} and {pair['pair'][1]}",
            _count(pair["both_right"]),
            _count(pair["only_first"]),
            _count(pair["only_second"]),
            _count(pair["both_wrong"]),
        ]
        for pair in record["other_pairs"]
    ]
    return table(
        ["Pair", "Both right", "Only the first", "Only the second", "Both wrong"],
        rows,
        right=(1, 2, 3, 4),
    )


def pair_sentence(pair: dict[str, Any], record: dict[str, Any], cfg: ReportConfig) -> str:
    """One tested pair in words: who is ahead, by how much, and whether it is settled."""
    first, second = pair["pair"]
    size = pair["difference_points"].lstrip("-")
    # The intervals are of first minus second; turn them when the second arm is ahead, so
    # that the sentence and its figures point the same way.
    turned = pair["favours"] == second
    figures = (
        f"interval by example {span(_turn(pair['interval'], turned))}, by company "
        f"{span(_turn(pair['interval_by_company'], turned))}; adjusted p "
        f"{shown_p(pair['p_adjusted'], cfg.smallest_p)}"
    )
    if pair["favours"] is None:
        return f"{first} and {second} have the same execution accuracy ({figures})."
    other = second if pair["favours"] == first else first
    lead = f"{pair['favours']} is ahead of {other} by {size} points ({figures})"
    if pair["settled"]:
        return f"{lead}. The difference is settled: it holds under both intervals."
    return f"{lead}. The difference is **not settled**: {_why_open(pair, record)}."


def overlap_sentence(record: dict[str, Any]) -> str:
    overlap, total = record["overlap"], record["test_examples"]
    tuned = _join(overlap["tuned_arms"])
    return (
        f"Of the {total:,} test questions, {overlap['all_tuned_right']:,} are right in all "
        f"the tuned arms ({tuned}) and {overlap['any_tuned_right']:,} in at least one of "
        f"them. {overlap['no_arm_right']:,} are right in none of the "
        f"{len(record['arms'])} arms."
    )


# --------------------------------------------------------------------------- cost


def making_table(cost: dict[str, Any]) -> str:
    rows = []
    for arm, entry in cost["making"].items():
        train, select, storage = entry["train"], entry["select"], entry["storage"]
        rows.append(
            [
                arm,
                f"{usd(train['usd'])} ({train['basis']})" if train else "—",
                usd(select["usd"]) if select else "—",
                usd(entry["making_usd"]),
                usd(storage["per_month"]) if storage else "—",
            ]
        )
    return table(
        ["Arm", "Training run", "Selection", "Together", "Storage a month"],
        rows,
        right=(1, 2, 3, 4),
    )


def test_pass_table(record: dict[str, Any], cost: dict[str, Any]) -> str:
    rows = []
    for arm in record["arms"]:
        figure = cost["test_pass"][arm]
        rows.append(
            [
                arm,
                usd(figure.get("usd")),
                usd(figure["per_1000_questions"]),
                usd(figure.get("per_1000_questions_steady")),
                usd(figure["per_1000_correct"]),
            ]
        )
    return table(
        ["Arm", "Test pass", "Per 1,000 questions", "Steady", "Per 1,000 correct answers"],
        rows,
        right=(1, 2, 3, 4),
    )


def serving_text(record: dict[str, Any], cost: dict[str, Any]) -> str:
    """For each tuned arm the cost record has a serving scenario for: batch and always-on."""
    parts = []
    for arm, scenario in cost.get("serving", {}).items():
        if not record["arms"].get(arm, {}).get("tuned"):
            continue
        per_hour = Decimal(scenario["questions_per_hour"]).to_integral_value()
        parts.append(
            f"**{arm}** answered about {per_hour:,} questions an hour on one "
            f"`{scenario['instance_type']}`. A job's start-up costs "
            f"{usd(scenario['startup_usd'])} before the first answer."
        )
        parts.append(
            table(
                ["Batch job of", "Cost", "Per 1,000 questions"],
                [
                    [f"{row['questions']:,} questions", usd(row["usd"]),
                     usd(row["per_1000_questions"])]
                    for row in scenario["batch"]
                ],
                right=(0, 1, 2),
            )
        )
        always = scenario.get("always_on")
        if always:
            parts.append(
                f"An always-on endpoint costs {usd(always['monthly_usd'])} a month, whatever "
                "the load:"
            )
            parts.append(
                table(
                    ["Utilization", "Questions a month", "Per 1,000 questions"],
                    [
                        [
                            f"{Decimal(row['utilization']) * 100:.0f}%",
                            f"{Decimal(row['questions_per_month']).to_integral_value():,}",
                            usd(row["per_1000_questions"]),
                        ]
                        for row in always["at"]
                    ],
                    right=(0, 1, 2),
                )
            )
    return "\n\n".join(parts) or "The cost record has no serving scenario for a tuned arm."


def break_even_table(cost: dict[str, Any]) -> str:
    rows = [
        [
            f"{entry['pair'][0]} and {entry['pair'][1]}",
            SCENARIO_NAMES[entry["scenario"]],
            break_even_text(entry),
            "yes" if entry["reachable"] else "**no**",
        ]
        for entry in cost["break_evens"]
    ]
    return table(["Pair", "Scenario", "Break-even", "Reachable here"], rows)


def project_sentence(cost: dict[str, Any]) -> str:
    parts = [
        f"Path {path} {usd(row['total'])} of its {usd(row['cap'])} cap"
        for path, row in cost["project"]["paths"].items()
        if row.get("cap")
    ]
    stored = cost["project"].get("storage_so_far") or {}
    text = f"The project so far: {_join(parts)}."
    if stored:
        total = sum(Decimal(value) for value in stored.values())
        text += f" Storage of the tuned Nova models so far adds {usd(total)}."
    return text


def prices_sentence(cost: dict[str, Any]) -> str:
    prices = cost["prices"]
    return (
        f"Prices: the Bedrock price list published {prices['bedrock']['published']}; the "
        f"SageMaker training price list of {prices['sagemaker']['training_published']} and "
        f"hosting price list of {prices['sagemaker']['hosting_published']}. Runs are priced "
        f"through {cost['through']}."
    )


def storage_per_month(cost: dict[str, Any]) -> str:
    amounts = sorted(
        {entry["storage"]["per_month"] for entry in cost["making"].values() if entry["storage"]}
    )
    return _join([usd(amount) for amount in amounts]) or "nothing"


# --------------------------------------------------------------------------- errors, speed


def categories_table(record: dict[str, Any]) -> str:
    arms = list(record["arms"])
    rows = [
        [f"{CATEGORY_NAMES[name]} (`{name}`)"]
        + [_count(record["arms"][arm]["categories"][name]) for arm in arms]
        for name in CATEGORIES
    ]
    rows.append(["**Right**"] + [f"**{record['arms'][arm]['right']:,}**" for arm in arms])
    return table(["Category"] + arms, rows, right=tuple(range(1, len(arms) + 1)))


def steps_table(record: dict[str, Any]) -> str:
    arms = list(record["arms"])
    groups = record["arms"][arms[0]]["by_steps"]
    rows = []
    for position, group in enumerate(groups):
        cells = [
            f"{record['arms'][arm]['by_steps'][position]['percent'] or '—'}%" for arm in arms
        ]
        rows.append([group["steps"], _count(group["examples"])] + cells)
    return table(
        ["Steps in the gold program", "Questions"] + arms,
        rows,
        right=tuple(range(1, len(arms) + 2)),
    )


def format_sentence(record: dict[str, Any]) -> str:
    outside = {arm: e["not_in_the_format"] for arm, e in record["arms"].items()}
    cut = {arm: e["categories"]["cut_off"] for arm, e in record["arms"].items()}
    clean = [arm for arm, count in outside.items() if count == 0]
    text = "Replies outside the strict two-line format: " + _join(
        [f"{arm} {count:,}" for arm, count in outside.items() if count]
    ) if any(outside.values()) else "Every reply of every arm is in the strict two-line format"
    text += "."
    if clean and any(outside.values()):
        text += f" Every reply of {_join(clean)} is in the format."
    if any(cut.values()):
        text += " Wrong replies that hit the reply limit: " + _join(
            [f"{arm} {count:,}" for arm, count in cut.items() if count]
        ) + "."
    return text


def latency_table(record: dict[str, Any]) -> str:
    rows = [
        [arm] + [entry["latency_seconds"].get(name, "—") for name in ("mean", "median", "p95")]
        for arm, entry in record["arms"].items()
    ]
    return table(["Arm", "Mean, seconds", "Median", "95th percentile"], rows, right=(1, 2, 3))


def example_ids_table(record: dict[str, Any]) -> str:
    rows = [
        [arm, f"`{name}`", ", ".join(f"`{example_id}`" for example_id in ids)]
        for arm, entry in record["arms"].items()
        for name, ids in entry["example_ids"].items()
    ]
    return table(["Arm", "Category", "Example ids"], rows)


# --------------------------------------------------------------------------- findings


def findings(record: dict[str, Any], cost: dict[str, Any], cfg: ReportConfig) -> list[str]:
    """The report's findings, each a sentence whose wording follows the figures."""
    arms = record["arms"]
    lines: list[str] = []

    gains = [
        pair
        for pair in record["pairs"]
        if arms[pair["pair"][0]]["tuned"] != arms[pair["pair"][1]]["tuned"]
    ]
    up = [p for p in gains if p["settled"] and arms.get(p["favours"], {}).get("tuned")]
    if gains and len(up) == len(gains):
        sizes = sorted(Decimal(p["difference_points"].lstrip("-")) for p in up)
        lines.append(
            f"**Tuning is the large effect.** It raised execution accuracy in all "
            f"{len(up)} models, by {sizes[0]} to {sizes[-1]} points, and every gain is settled."
        )
    else:
        lines += [f"**Tuning.** {pair_sentence(p, record, cfg)}" for p in gains]

    tuned = sorted(
        (arm for arm, entry in arms.items() if entry["tuned"]),
        key=lambda arm: -arms[arm]["metrics"][PRIMARY]["count"],
    )
    if tuned:
        lines.append(
            "**The tuned arms, by execution accuracy:** "
            + _join([f"{arm} {arms[arm]['metrics'][PRIMARY]['percent']}%" for arm in tuned])
            + "."
        )
    for pair in record["pairs"]:
        if all(arms[arm]["tuned"] for arm in pair["pair"]):
            lines.append(pair_sentence(pair, record, cfg))

    priced = sorted(
        (arm for arm in tuned if arm in record["costs"]),
        key=lambda arm: Decimal(record["costs"][arm]["per_1000_questions"]),
    )
    if priced:
        lines.append(
            "**Cost per 1,000 questions, tuned arms:** "
            + _join([f"{arm} {usd(record['costs'][arm]['per_1000_questions'])}" for arm in priced])
            + ". A Qwen arm's figure is the batch job that was paid for."
        )
    for entry in cost.get("break_evens", []):
        if entry["scenario"] == ALWAYS_ON or not all(
            arms.get(arm, {}).get("tuned") for arm in entry["pair"]
        ):
            continue
        reach = "" if entry["reachable"] else " (not reachable in this account)"
        lines.append(
            f"**{entry['pair'][0]} against {entry['pair'][1]}, "
            f"{SCENARIO_NAMES[entry['scenario']]}:** {break_even_text(entry)}{reach}."
        )
    always = [e for e in cost.get("break_evens", []) if e["scenario"] == ALWAYS_ON]
    if always and not any(entry["reachable"] for entry in always):
        lines.append(
            "**An always-on endpoint does not pay off in this account:** every break-even "
            "needs more questions a month than the account may send to the Nova model, or "
            "than one instance can answer."
        )
    elif always:
        lines += [
            f"**{e['pair'][0]} against {e['pair'][1]}, always-on endpoint:** "
            f"{break_even_text(e)}."
            for e in always
            if e["reachable"]
        ]
    return lines


# --------------------------------------------------------------------------- the two renderings


def placeholders(
    record: dict[str, Any],
    cost: dict[str, Any],
    deviations: list[dict[str, str]],
    cfg: ReportConfig,
    chart_path: str,
) -> dict[str, str]:
    """Every value the report template may name."""
    settings = record["settings"]
    provisional = record["provisional"]
    dev_examples = sorted(
        {entry["dev"]["examples"] for entry in record["arms"].values() if entry["dev"]}
    )
    return {
        "title": "The comparison" + (" (provisional)" if provisional else ""),
        "command": COMMAND,
        "provisional_note": (
            "**Provisional.** The cost record this report reads has no bill entry yet for "
            "the days of the last runs. No accuracy figure depends on it; the report is "
            "built again once the bill is entered."
            if provisional
            else "The cost record this report reads is reconciled with the AWS bill."
        ),
        "summary_table": summary_table(record),
        "findings": "\n".join(f"- {line}" for line in findings(record, cost, cfg)),
        "arms_table": table(
            ["Arm", "Model", "Path", "Tuned"],
            [
                [arm, e["label"], e["path"], "yes" if e["tuned"] else "no"]
                for arm, e in record["arms"].items()
            ],
        ),
        "test_examples": f"{record['test_examples']:,}",
        "companies": f"{record['companies']:,}",
        "dev_examples": _join([f"{count:,}" for count in dev_examples]) or "—",
        "freeze_sha256": str(record["freeze_sha256"]),
        "accuracy_table": accuracy_table(record),
        "execution_table": execution_table(record),
        "dev_test_table": dev_test_table(record),
        "resamples": f"{settings['resamples']:,}",
        "confidence": str(settings["confidence_percent"]),
        "seed": str(settings["seed"]),
        "pair_count": str(len(record["pairs"])),
        "pairs_table": pairs_table(record, cfg),
        "pair_sentences": "\n".join(
            f"- {pair_sentence(pair, record, cfg)}" for pair in record["pairs"]
        ),
        "other_pairs_table": other_pairs_table(record),
        "overlap_sentence": overlap_sentence(record),
        "making_table": making_table(cost),
        "test_pass_table": test_pass_table(record, cost),
        "serving_text": serving_text(record, cost),
        "break_even_table": break_even_table(cost),
        "project_sentence": project_sentence(cost),
        "prices_sentence": prices_sentence(cost),
        "storage_per_month": storage_per_month(cost),
        "chart_path": chart_path,
        "categories_table": categories_table(record),
        "steps_table": steps_table(record),
        "format_sentence": format_sentence(record),
        "latency_table": latency_table(record),
        "deviations_table": table(
            ["Date", "Deviation", "Recorded in"],
            [[d["date"], d["what"], f"`{d['recorded_in']}`"] for d in deviations],
        )
        if deviations
        else "None is listed.",
    }


def render_statistics(record: dict[str, Any], cfg: ReportConfig) -> str:
    """The Markdown copy of the statistics record."""
    title = "# The comparison's statistics" + (" (provisional)" if record["provisional"] else "")
    settings = record["settings"]
    lines = [
        title,
        "",
        f"Generated by `{COMMAND} build`. Do not edit by hand.",
        "",
        "Confidence intervals, paired tests, and error categories for the six test runs",
        "(req_005 v5.03). Every figure is computed from the stored replies of the runs the",
        "six-arm record names. This file holds counts and example ids: no reply, no question,",
        "no gold text.",
        "",
        f"- Intervals: paired percentile bootstrap, {settings['resamples']:,} resamples, "
        f"{settings['confidence_percent']}%, seed {settings['seed']} (by company: "
        f"{settings['company_seed']})",
        "- Tests: McNemar's exact test; p-values adjusted with Holm's method over the "
        f"{len(settings['pairs'])} tested pairs",
        "",
        "## Checks",
        "",
        "| Id | Check | Result | Detail |",
        "|---|---|---|---|",
    ]
    for check in record["checks"]:
        result = "PASS" if check["passed"] else "FAIL"
        if check["reservation"]:
            result = "PASS, with a reservation"
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| {check['id']} | {check['name']} | {result} | {detail} |")
    if not record.get("arms"):
        return "\n".join(lines) + "\n"
    sections = [
        ("Execution accuracy", execution_table(record)),
        ("The four metrics, with intervals by example", accuracy_table(record)),
        ("Dev beside test", dev_test_table(record)),
        ("The tested pairs", pairs_table(record, cfg)),
        ("The other pairs, as counts", other_pairs_table(record)),
        ("How the arms overlap", overlap_sentence(record)),
        ("Why replies are wrong", categories_table(record)),
        ("Execution accuracy by gold steps", steps_table(record)),
        ("The format", format_sentence(record)),
        ("Speed", latency_table(record)),
        ("Example ids per arm and category", example_ids_table(record)),
    ]
    for heading, body in sections:
        lines += ["", f"## {heading}", "", body]
    lines += [
        "",
        f"The ids of the {record['overlap']['no_arm_right']:,} questions no arm got right "
        "are in the JSON file (`unanswered_ids`).",
    ]
    return "\n".join(lines) + "\n"
