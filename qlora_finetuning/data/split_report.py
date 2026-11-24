"""The split report: how fit, dev, and test compare, and whether the split is sound (v1.02).

Counts only, like the schema report: no question, passage, table, or answer text, and no
timestamp. This module and the builder are the only code that reads test rows before the
unlock, and they read them to count.

Usage::

    from qlora_finetuning.data.split_report import build_report, write_report

    report = build_report(manifest, rebuilt, examples, cfg, data_cfg)
    write_report(report, cfg.report_dir)
    report["columns"]["test"]["size"]["examples"]         # 2898
    report["leakage"]["test_against_train"]

``manifest`` is the committed one, ``rebuilt`` what the settings produce now; check P9
compares them.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qlora_finetuning.data.schema_report import _label, _number, example_facts, profile, render_json
from qlora_finetuning.data.snapshot import DataConfig, RawExample
from qlora_finetuning.data.split import (
    GROUP_KEYS,
    SplitConfig,
    content_hash,
    render_manifest,
)
from qlora_finetuning.data.split_loader import unlock_state

REPORT_STEM = "finqa_split_report"
REPORT_COMMAND = "python -m scripts.finqa_split"
COLUMNS = ("fit", "dev", "quick_dev", "train", "test")
_TABLE_KEY = re.compile(r"table_(\d+)")
_NOT_DIGIT = re.compile(r"[^\d]")

__all__ = [
    "COLUMNS",
    "build_report",
    "leakage",
    "render_json",
    "render_markdown",
    "run_checks",
    "stale_files",
    "write_report",
]


# --------------------------------------------------------------------------- leakage


def _row_pairs(row: Any, min_digits: int) -> set[tuple[str, str]]:
    """(row name, digits) for each cell after the first that holds enough digits.

    A row with no name gives nothing.
    """
    if not isinstance(row, list) or not row:
        return set()
    name = str(row[0]).strip()
    if not name:  # a header row: its "numbers" are years, which every table shares
        return set()
    pairs = set()
    for cell in row[1:]:
        digits = _NOT_DIGIT.sub("", str(cell))
        if len(digits) >= min_digits:
            pairs.add((name, digits))
    return pairs


def table_number_pairs(example: RawExample, min_digits: int) -> set[tuple[str, str]]:
    """Every (row name, number) pair in the example's table."""
    pairs: set[tuple[str, str]] = set()
    for row in example.table or []:
        pairs |= _row_pairs(row, min_digits)
    return pairs


def gold_number_pairs(example: RawExample, min_digits: int) -> set[tuple[str, str]]:
    """The (row name, number) pairs of the table rows the example's evidence points at."""
    table = example.table or []
    pairs: set[tuple[str, str]] = set()
    for key in example.gold_inds or {}:
        match = _TABLE_KEY.fullmatch(str(key))
        if match and int(match.group(1)) < len(table):
            pairs |= _row_pairs(table[int(match.group(1))], min_digits)
    return pairs


def leakage(
    held_out: Sequence[RawExample], seen: Sequence[RawExample], min_digits: int
) -> dict[str, int]:
    """How much of ``held_out`` could have been met in ``seen``.

    "Shares a number" means a (row name, number) pair of the example's evidence rows also
    occurs in a table on the seen side: for the same company, or for any company (which
    includes chance matches).
    """
    companies = {example.ticker for example in seen}
    reports = {example.report for example in seen}
    pages = {example.filename for example in seen}
    questions = {example.question for example in seen}
    by_company: dict[Any, set[tuple[str, str]]] = {}
    for example in seen:
        by_company.setdefault(example.ticker, set()).update(table_number_pairs(example, min_digits))
    everything = set().union(*by_company.values()) if by_company else set()

    checkable = same_company = any_company = 0
    for example in held_out:
        pairs = gold_number_pairs(example, min_digits)
        if not pairs:
            continue
        checkable += 1
        same_company += bool(pairs & by_company.get(example.ticker, set()))
        any_company += bool(pairs & everything)
    return {
        "examples": len(held_out),
        "company_also_on_the_seen_side": sum(e.ticker in companies for e in held_out),
        "report_also_on_the_seen_side": sum(e.report in reports for e in held_out),
        "page_also_on_the_seen_side": sum(e.filename in pages for e in held_out),
        "question_wording_also_on_the_seen_side": sum(e.question in questions for e in held_out),
        "examples_with_a_number_to_check": checkable,
        "sharing_a_number_with_the_same_company": same_company,
        "sharing_a_number_with_any_company": any_company,
    }


# --------------------------------------------------------------------------- statistics


def _percent(part: int, whole: int, places: int) -> float:
    return round(100 * part / whole, places) if whole else 0.0


def split_stats(
    pairs: Sequence[tuple[RawExample, dict[str, Any]]],
    group_by: str,
    total: int,
    official_splits: Sequence[str],
) -> dict[str, Any]:
    """Counts for one column of the report."""
    examples = [example for example, _ in pairs]
    counted = profile(pairs)
    programs, evidence, answers = counted["programs"], counted["evidence"], counted["answers"]
    groups = Counter(GROUP_KEYS[group_by](example) for example in examples)
    origin = Counter(example.official_split for example in examples)
    steps = programs["steps_in_total"]

    size = {
        "examples": len(examples),
        "share_of_all_examples_percent": _percent(len(examples), total, 2),
        "groups": len(groups),
        "largest_group": max(groups.values(), default=0),
        "companies": counted["identity"]["companies"],
        "reports": counted["identity"]["reports"],
        "pages": counted["identity"]["pages"],
    }
    for name in official_splits:
        size[f"from_the_official_{name}_file"] = origin[name]
    return {
        "size": size,
        "program_length": dict(programs["steps_per_program"]),
        "operations": dict(programs["operations"]),
        "operation_share_percent": {
            name: _percent(count, steps, 1) for name, count in programs["operations"].items()
        },
        "argument_kinds": dict(programs["argument_kinds"]),
        "other": {
            "yes_or_no_examples": answers["exe_ans_yes"] + answers["exe_ans_no"],
            "examples_with_a_table_operation": programs["examples_with_a_table_operation"],
            "examples_with_table_evidence_only": evidence["examples_with_table_evidence_only"],
            "examples_with_text_evidence_only": evidence["examples_with_text_evidence_only"],
            "examples_with_both": evidence["examples_with_both"],
            "examples_with_an_empty_answer_field": answers["answer_empty"],
        },
    }


# --------------------------------------------------------------------------- checks


def _check(check_id: str, name: str, problems: list[str], ok_detail: str) -> dict[str, Any]:
    return {
        "id": check_id,
        "name": name,
        "passed": not problems,
        "detail": "; ".join(problems) if problems else ok_detail,
    }


def _divided(examples: Sequence[RawExample], assignments: dict[str, str], key: Any) -> list[str]:
    """Keys whose examples are in more than one split."""
    found: dict[Any, set[str]] = {}
    for example in examples:
        if example.id in assignments:
            found.setdefault(key(example), set()).add(assignments[example.id])
    return sorted(str(item) for item, splits in found.items() if len(splits) > 1)


def run_checks(
    manifest: dict[str, Any],
    rebuilt: dict[str, Any],
    examples: Sequence[RawExample],
    cfg: SplitConfig,
    data_cfg: DataConfig,
) -> list[dict[str, Any]]:
    """Checks P1 to P10 on the committed manifest against the data and the settings."""
    assignments: dict[str, str] = manifest["assignments"]
    quick_dev: list[str] = manifest["quick_dev"]
    present = [example.id for example in examples]
    counts = Counter(assignments.values())
    train = counts["fit"] + counts["dev"]

    complete = []
    missing = set(present) - set(assignments)
    extra = set(assignments) - set(present)
    if missing:
        complete.append(f"{len(missing):,} examples have no assignment")
    if extra:
        complete.append(f"{len(extra):,} assigned ids are not in the data")
    if not complete and list(assignments) != present:
        complete.append("the ids are in a different order from the data")

    dataset = manifest["dataset"]
    same_data = []
    if (dataset.get("repo_id"), dataset.get("revision")) != (data_cfg.repo_id, data_cfg.revision):
        same_data.append("the manifest names another dataset revision")
    if dataset.get("files") != {item.official_split: item.sha256 for item in data_cfg.files}:
        same_data.append("the manifest names other file checksums")

    share = train / len(assignments) if assignments else 0.0
    ratio = []
    if abs(share - cfg.train_fraction) > cfg.train_fraction_tolerance:
        ratio.append(
            f"train share is {100 * share:.2f}%, outside {100 * cfg.train_fraction:.1f}% "
            f"plus or minus {100 * cfg.train_fraction_tolerance:.1f}"
        )

    group_of = GROUP_KEYS[cfg.group_by]
    whole = []
    divided = _divided(examples, assignments, group_of)
    if divided:
        whole.append(f"{len(divided)} groups are in more than one split: {', '.join(divided[:5])}")
    derived = {str(group_of(e)): assignments[e.id] for e in examples if e.id in assignments}
    if not divided and derived != manifest["groups"]:
        whole.append("the manifest's group table does not match its assignments")

    dev_size = []
    if counts["dev"] > cfg.dev_size:
        dev_size.append(f"dev has {counts['dev']:,} examples, more than {cfg.dev_size:,}")
    if counts["dev"] < cfg.dev_size - cfg.dev_size_tolerance:
        dev_size.append(
            f"dev has {counts['dev']:,} examples, fewer than "
            f"{cfg.dev_size - cfg.dev_size_tolerance:,}"
        )

    quick = []
    if len(quick_dev) != cfg.quick_dev_size:
        quick.append(f"quick-dev has {len(quick_dev):,} ids, expected {cfg.quick_dev_size:,}")
    outside = sum(assignments.get(item) != "dev" for item in quick_dev)
    if outside:
        quick.append(f"{outside:,} quick-dev ids are not in dev")

    pages = []
    divided_pages = _divided(examples, assignments, lambda e: e.filename)
    if divided_pages:
        pages.append(f"{len(divided_pages)} pages are in more than one split")
    divided_pairs = _divided(examples, assignments, lambda e: (e.filename, e.question))
    if divided_pairs:
        pages.append(f"{len(divided_pairs)} page-and-question groups are in more than one split")

    hashed = []
    if content_hash(assignments, quick_dev) != manifest["content_hash"]:
        hashed.append("the content hash does not match the assignments")

    reproducible = []
    if render_manifest(rebuilt) != render_manifest(manifest):
        reproducible.append(
            "building from the current settings and data gives a different manifest "
            f"(hash {str(rebuilt['content_hash'])[:12]}...)"
        )

    unlocked, reason = unlock_state(cfg, manifest)
    lock_problem = [] if unlocked or not cfg.unlock_file.exists() else [reason]

    return [
        _check("P1", "Complete", complete, f"{len(assignments):,} examples, each assigned once"),
        _check("P2", "Same data", same_data, f"{data_cfg.repo_id} at {data_cfg.revision[:12]}..."),
        _check(
            "P3",
            "Ratio",
            ratio,
            f"train {train:,} ({100 * share:.2f}%), test {counts['test']:,} "
            f"({100 - 100 * share:.2f}%)",
        ),
        _check(
            "P4",
            "Groups whole",
            whole,
            f"{len(manifest['groups']):,} groups by {cfg.group_by}, none divided",
        ),
        _check("P5", "Dev size", dev_size, f"dev has {counts['dev']:,} examples"),
        _check("P6", "Quick-dev", quick, f"{len(quick_dev):,} ids, all in dev"),
        _check("P7", "Pages and duplicates whole", pages, "no page or duplicate pair is divided"),
        _check("P8", "Hash", hashed, f"{str(manifest['content_hash'])[:12]}..."),
        _check("P9", "Reproducible", reproducible, "rebuilding gives the same manifest"),
        _check(
            "P10",
            "Test locked",
            lock_problem,
            f"unlocked: {reason}" if unlocked else f"locked: {reason}",
        ),
    ]


# --------------------------------------------------------------------------- report


def build_report(
    manifest: dict[str, Any],
    rebuilt: dict[str, Any],
    examples: Sequence[RawExample],
    cfg: SplitConfig,
    data_cfg: DataConfig,
) -> dict[str, Any]:
    """The whole split report as a dict."""
    assignments: dict[str, str] = manifest["assignments"]
    quick = set(manifest["quick_dev"])
    pairs = [(example, example_facts(example)) for example in examples]

    def members(column: str) -> list[tuple[RawExample, dict[str, Any]]]:
        if column == "quick_dev":
            return [pair for pair in pairs if pair[0].id in quick]
        wanted = ("fit", "dev") if column == "train" else (column,)
        return [pair for pair in pairs if assignments.get(pair[0].id) in wanted]

    by_column = {column: members(column) for column in COLUMNS}

    def only_examples(column: str) -> list[RawExample]:
        return [example for example, _ in by_column[column]]

    return {
        "schema": 1,
        "generated_by": REPORT_COMMAND,
        "manifest": {
            "file": cfg.manifest_file.name,
            "content_hash": manifest["content_hash"],
            "settings": manifest["settings"],
            "dataset": {
                "repo_id": manifest["dataset"].get("repo_id"),
                "revision": manifest["dataset"].get("revision"),
            },
        },
        "columns": {
            column: split_stats(by_column[column], cfg.group_by, len(examples), data_cfg.splits)
            for column in COLUMNS
        },
        "leakage": {
            "min_digits": cfg.leakage_min_digits,
            "test_against_train": leakage(
                only_examples("test"), only_examples("train"), cfg.leakage_min_digits
            ),
            "dev_against_fit": leakage(
                only_examples("dev"), only_examples("fit"), cfg.leakage_min_digits
            ),
        },
        "checks": run_checks(manifest, rebuilt, examples, cfg, data_cfg),
    }


def _table(report: dict[str, Any], section: str, code_keys: bool) -> list[str]:
    """One section as a Markdown table: a row per measure, a column per split."""
    columns = report["columns"]
    keys: list[str] = []
    for column in ("train", "test", "fit", "dev", "quick_dev"):
        keys += [key for key in columns[column][section] if key not in keys]
    lines = [
        "| Measure | " + " | ".join(COLUMNS).replace("quick_dev", "quick-dev") + " |",
        "|---|" + "---:|" * len(COLUMNS),
    ]
    places = 2 if section == "size" else 1  # shares of examples to 0.01, of steps to 0.1

    def cell(value: Any) -> str:
        return f"{value:.{places}f}" if isinstance(value, float) else _number(value)

    for key in keys:
        label = f"`{key}`" if code_keys else _label(key)
        missing = 0.0 if section.endswith("percent") else 0
        cells = [cell(columns[column][section].get(key, missing)) for column in COLUMNS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


SECTIONS = (
    ("size", "Size", False),
    ("program_length", "Programs by number of steps", True),
    ("operations", "Operations (steps)", True),
    ("operation_share_percent", "Operations (percent of the split's steps)", True),
    ("argument_kinds", "Argument kinds", True),
    ("other", "Other counts", False),
)


def render_markdown(report: dict[str, Any]) -> str:
    manifest = report["manifest"]
    settings = manifest["settings"]
    failed = [check for check in report["checks"] if not check["passed"]]
    lines = [
        "# FinQA split report",
        "",
        f"Generated by `{report['generated_by']}` from the committed split manifest and the",
        "pinned snapshot. Do not edit by hand.",
        "",
        "This file holds counts only. It contains no question, passage, table, or answer text.",
        "`train` is fit plus dev. `quick-dev` is part of dev.",
        "",
        "## Manifest",
        "",
        f"- File: `splits/{manifest['file']}`",
        f"- Content hash: `{manifest['content_hash']}`",
        f"- Grouped by: {settings['group_by']}",
        f"- Seed: {settings['seed']}",
        f"- Rule: `{settings['rule']}`",
        f"- Dataset: `{manifest['dataset']['repo_id']}` at `{manifest['dataset']['revision']}`",
        "",
        "## Checks",
        "",
        f"{len(report['checks']) - len(failed)} of {len(report['checks'])} passed.",
        "",
        "| Id | Check | Result | Detail |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {check['id']} | {check['name']} | {'PASS' if check['passed'] else 'FAIL'} "
        f"| {check['detail']} |"
        for check in report["checks"]
    ]
    for section, title, code_keys in SECTIONS:
        lines += ["", f"## {title}", ""]
        lines += _table(report, section, code_keys)

    leak = report["leakage"]
    lines += [
        "",
        "## What the held-out side could have met",
        "",
        "A number is shared when a row name together with a number of at least "
        f"{leak['min_digits']} digits, taken",
        "from the table rows an example's evidence points at, also occurs in a table on the",
        "other side. Matches with any company include chance matches.",
        "",
        "| Measure | test against train | dev against fit |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {_label(key)} | {_number(value)} | {_number(leak['dev_against_fit'][key])} |"
        for key, value in leak["test_against_train"].items()
    ]
    return "\n".join(lines) + "\n"


def report_paths(report_dir: Path | str) -> tuple[Path, Path]:
    report_dir = Path(report_dir)
    return report_dir / f"{REPORT_STEM}.md", report_dir / f"{REPORT_STEM}.json"


def write_report(report: dict[str, Any], report_dir: Path | str) -> tuple[Path, Path]:
    """Write the Markdown and JSON files and return their paths."""
    markdown_path, json_path = report_paths(report_dir)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    json_path.write_text(render_json(report), encoding="utf-8", newline="\n")
    return markdown_path, json_path


def stale_files(report: dict[str, Any], report_dir: Path | str) -> list[str]:
    """Names of the report files that are missing or differ from a fresh rendering (as text)."""
    stale = []
    markdown_path, json_path = report_paths(report_dir)
    for path, fresh in ((markdown_path, render_markdown(report)), (json_path, render_json(report))):
        if not path.is_file() or path.read_text(encoding="utf-8") != fresh:
            stale.append(path.name)
    return stale
