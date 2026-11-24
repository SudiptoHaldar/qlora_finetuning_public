"""Building the comparison report's files (req_005 v5.03).

``build_outputs`` takes the statistics record and the cost record and makes, as text and
without writing anything: the statistics record and its Markdown copy, the chart, the
report (the template with every placeholder filled), and the README block. It then runs
check **R8** over all of it and returns the record with that check added:

* every placeholder of the template is filled, and every value is used;
* no text holds a deployment name or an account number;
* every example id in the record is a test example's id;
* nothing looks like a credential or an address.

``write_outputs`` writes the files, only for a record that passed. The README is changed
only between two marker comments; its other lines, its byte order mark, and its line ends
are kept.

Usage::

    from qlora_finetuning.comparison.report_build import build_outputs, write_outputs

    record, outputs = build_outputs(record, cost, deviations, cfg, report_dir, test_ids)
    if record["passed"]:
        written = write_outputs(outputs, cfg)

No reply and no dataset text passes through this module: it sees the statistics record,
which holds none. The local examples file is ``report_examples``. This module imports
neither boto3 nor the training stack.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from qlora_finetuning.comparison.chart import ChartPoint, render_chart
from qlora_finetuning.comparison.gate import canonical_json
from qlora_finetuning.comparison.outcomes import PRIMARY
from qlora_finetuning.comparison.report_config import ReportConfig
from qlora_finetuning.comparison.report_record import COMMAND, add_check, make_check
from qlora_finetuning.comparison.report_tables import (
    findings,
    placeholders,
    render_statistics,
    summary_table,
)
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking.config import TrackingError, check_no_secrets

TEMPLATE = Path(__file__).with_name("report_template.md")
PLACEHOLDER = re.compile(r"\{\{([a-z0-9_]+)\}\}")
START, END = "<!-- comparison:start -->", "<!-- comparison:end -->"
BOM = "﻿"
# What must never reach a written file: a deployment or job name, an account number.
FORBIDDEN = (
    (re.compile(r"arn:aws", re.IGNORECASE), "an AWS resource name"),
    (re.compile(r"(?<![0-9A-Za-z])\d{12}(?![0-9A-Za-z])"), "what looks like an account number"),
)
README_FINDINGS = ("**Tuning", "**The tuned arms", "**Cost per")


@dataclass(frozen=True)
class Outputs:
    """The text of every file a build writes, by path. The README is the block alone."""

    statistics_json: tuple[Path, str]
    statistics_markdown: tuple[Path, str]
    chart: tuple[Path, str]
    report: tuple[Path, str]
    readme_block: tuple[Path, str]


def fill(template: str, values: dict[str, str]) -> tuple[str, list[str]]:
    """The template with its placeholders filled, and what is wrong with the filling."""
    named = set(PLACEHOLDER.findall(template))
    problems = [f"the template names {{{{{name}}}}}, which has no value"
                for name in sorted(named - set(values))]
    problems += [f"the value {name} is used nowhere in the template"
                 for name in sorted(set(values) - named)]
    text = PLACEHOLDER.sub(lambda match: values.get(match.group(1), match.group(0)), template)
    return text, problems


def chart_points(record: dict[str, Any], cfg: ReportConfig) -> list[ChartPoint]:
    colors = {
        arm: family["color"] for family in cfg.chart_families.values() for arm in family["arms"]
    }
    points = []
    for arm, entry in record["arms"].items():
        cost = (record["costs"].get(arm) or {}).get("per_1000_questions")
        if cost is None or arm not in colors:
            raise ConfigError(f"arm {arm} has no cost or no colour for the chart")
        metric = entry["metrics"][PRIMARY]
        points.append(
            ChartPoint(
                arm=arm,
                cost=Decimal(cost),
                percent=float(metric["percent"]),
                low=float(metric["interval"]["low"]),
                high=float(metric["interval"]["high"]),
                color=colors[arm],
                tuned=bool(entry["tuned"]),
            )
        )
    return points


def _relative(target: Path, start: Path) -> str:
    return Path(os.path.relpath(target, start)).as_posix()


def readme_block(record: dict[str, Any], cost: dict[str, Any], cfg: ReportConfig) -> str:
    """The lines that go between the README's markers, markers included."""
    chosen = [line for line in findings(record, cost, cfg) if line.startswith(README_FINDINGS)]
    link = _relative(cfg.report_file, cfg.readme_file.parent)
    lines = [
        START,
        "## Results" + (" (provisional)" if record["provisional"] else ""),
        "",
        f"Six arms on the {record['test_examples']:,} test questions, each run once. "
        f"Intervals are {record['settings']['confidence_percent']}% confidence intervals, in "
        "percent; costs are at list price.",
        "",
        summary_table(record),
        "",
        *[f"- {line}" for line in chosen],
        "",
        f"The full report, with the tests between arms, the cost model, the error "
        f"categories, and the caveats: [{link}]({link}).",
        "",
        f"This section is written by `{COMMAND} build`; do not edit between the markers.",
        END,
    ]
    return "\n".join(lines) + "\n"


def merge_readme(existing: str, block: str) -> str:
    """``existing`` with the block between its markers replaced, or appended once."""
    start, end = existing.find(START), existing.find(END)
    if start != -1 and end > start:
        tail = existing[end + len(END):]
        return existing[:start] + block.rstrip("\n") + tail
    if start != -1 or end != -1:
        raise ConfigError(
            "the README has one comparison marker without the other: restore both or remove it"
        )
    body = existing if existing.endswith("\n") or not existing else existing + "\n"
    return body + ("\n" if body else "") + block


def _ids_problems(record: dict[str, Any], test_ids: Any) -> list[str]:
    if test_ids is None:
        return []
    known = set(test_ids)
    listed = set(record.get("unanswered_ids") or [])
    for entry in (record.get("arms") or {}).values():
        for ids in entry["example_ids"].values():
            listed.update(ids)
    strangers = sorted(listed - known)
    return [f"{len(strangers):,} example ids are not test examples"] if strangers else []


def build_outputs(
    record: dict[str, Any],
    cost: dict[str, Any],
    deviations: list[dict[str, str]],
    cfg: ReportConfig,
    report_dir: Path,
    test_ids: Any = None,
    template: str | None = None,
) -> tuple[dict[str, Any], Outputs | None]:
    """The record with R8 added, and the text of every file. Writes nothing.

    For a record that already failed, R8 is marked "not built" and there are no outputs.
    """
    if not record.get("passed") or not record.get("arms"):
        failed = make_check("R8", ["not built: an earlier check failed"], "")
        return add_check(record, failed), None

    report_dir = Path(report_dir)
    chart_path = report_dir / cfg.chart_name
    template = template if template is not None else TEMPLATE.read_text(encoding="utf-8")
    values = placeholders(
        record, cost, deviations, cfg, _relative(chart_path, cfg.report_file.parent)
    )
    report, problems = fill(template, values)
    total = record["test_examples"]
    families = {name: family["color"] for name, family in cfg.chart_families.items()}
    chart = render_chart(
        chart_points(record, cfg),
        width=cfg.chart_width,
        height=cfg.chart_height,
        families=families,
        title="Execution accuracy against cost per 1,000 questions",
        note=(
            f"{total:,} test questions; whiskers are "
            f"{record['settings']['confidence_percent']}% confidence intervals"
        ),
    )
    block = readme_block(record, cost, cfg)

    problems += _ids_problems(record, test_ids)
    try:
        check_no_secrets(record, cfg.statistics_stem)
    except TrackingError as error:
        problems.append(str(error))

    def texts(final: dict[str, Any]) -> dict[str, str]:
        return {
            "the statistics record": canonical_json(final),
            "the statistics in Markdown": render_statistics(final, cfg),
            "the chart": chart,
            "the report": report,
            "the README block": block,
        }

    for what, text in texts(record).items():
        for pattern, name in FORBIDDEN:
            if pattern.search(text):
                problems.append(f"{what} holds {name}")
    check = make_check(
        "R8",
        problems,
        f"{len(values)} placeholders filled; ids are test ids; no resource name, no text",
    )
    final = add_check(record, check)
    written = texts(final)
    stem = report_dir / cfg.statistics_stem
    return final, Outputs(
        statistics_json=(stem.with_suffix(".json"), written["the statistics record"]),
        statistics_markdown=(stem.with_suffix(".md"), written["the statistics in Markdown"]),
        chart=(chart_path, chart),
        report=(cfg.report_file, report),
        readme_block=(cfg.readme_file, block),
    )


def write_readme(path: Path, block: str) -> None:
    """Put the block into the README, keeping its byte order mark and its line ends."""
    raw = path.read_bytes().decode("utf-8") if path.is_file() else ""
    marked = raw.startswith(BOM)
    crlf = "\r\n" in raw
    merged = merge_readme(raw.removeprefix(BOM).replace("\r\n", "\n"), block)
    text = (BOM if marked else "") + (merged.replace("\n", "\r\n") if crlf else merged)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def write_outputs(record: dict[str, Any], outputs: Outputs | None) -> list[Path]:
    """Write every file of a build. Refused unless R1 to R8 passed."""
    if outputs is None or not record.get("passed"):
        raise ConfigError("the comparison report is written only when R1 to R8 pass")
    written = []
    for path, text in (
        outputs.statistics_json,
        outputs.statistics_markdown,
        outputs.chart,
        outputs.report,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        written.append(path)
    write_readme(*outputs.readme_block)
    written.append(outputs.readme_block[0])
    return written
