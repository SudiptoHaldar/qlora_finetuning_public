"""The serialization report: how long the records are, what was skipped, and whether the
records are sound (req_001 v1.03).

Counts, character lengths, and digests only. It holds no question, passage, table, or answer
text and no timestamp. The digests pin the real records: if one character of any fit or dev
record changes, a digest changes, and check C9 says so.

Usage::

    from qlora_finetuning.data.serialization_report import build_report, write_report

    by_split = {"fit": (fit_examples, fit_records), "dev": (dev_examples, dev_records)}
    report = build_report(by_split, prompt, previous, allowed_ids, test_ids)
    write_report(report, prompt.report_dir)
    report["splits"]["fit"]["user_characters"]["p99"]
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from qlora_finetuning.data.schema_report import _label, _number, render_json
from qlora_finetuning.data.serialization import (
    YES_NO,
    CanonicalRecord,
    PromptDefinition,
    _one_line,
    build_record,
    records_digest,
    text_evidence,
)
from qlora_finetuning.data.snapshot import RawExample

REPORT_STEM = "finqa_serialization_report"
REPORT_COMMAND = "python -m scripts.finqa_serialize"
_ANSWER = re.compile(r"-?\d+(\.\d+)?")
_BAR = re.compile(r"(?<!\\)\|")

SplitRecords = Mapping[str, tuple[Sequence[RawExample], Sequence[CanonicalRecord]]]

__all__ = [
    "build_report",
    "length_stats",
    "read_committed_digests",
    "render_json",
    "render_markdown",
    "run_checks",
    "stale_files",
    "write_report",
]


# --------------------------------------------------------------------------- statistics


def length_stats(values: Sequence[int]) -> dict[str, Any]:
    """Minimum, median, 90th and 99th percentile (nearest rank), and maximum."""
    if not values:
        return {"min": None, "median": None, "p90": None, "p99": None, "max": None}
    ordered = sorted(values)

    def rank(share: float) -> int:
        return ordered[max(math.ceil(share * len(ordered)), 1) - 1]

    median = statistics.median(ordered)
    return {
        "min": ordered[0],
        "median": int(median) if float(median).is_integer() else float(median),
        "p90": rank(0.90),
        "p99": rank(0.99),
        "max": ordered[-1],
    }


def split_stats(
    examples: Sequence[RawExample], records: Sequence[CanonicalRecord]
) -> dict[str, Any]:
    """Counts and lengths for one split."""
    evidence = [text_evidence(example) for example in examples]
    passages = Counter(min(len(found), 3) for found, _ in evidence)
    return {
        "records": len(records),
        "system_characters": len(records[0].system) if records else 0,
        "user_characters": length_stats([len(record.user) for record in records]),
        "target_characters": length_stats([len(record.target) for record in records]),
        "records_without_text_evidence": passages[0],
        "records_by_number_of_passages": {
            "0": passages[0],
            "1": passages[1],
            "2": passages[2],
            "3_or_more": passages[3],
        },
        "most_table_rows": max((len(example.table or []) for example in examples), default=0),
        "most_table_columns": max(
            (len(example.table[0]) for example in examples if example.table), default=0
        ),
        "answers_written_out_from_scientific_notation": sum(
            str(example.exe_ans) not in YES_NO and "e" in str(example.exe_ans).lower()
            for example in examples
        ),
        "yes_or_no_answers": sum(str(example.exe_ans) in YES_NO for example in examples),
        "skipped_evidence_keys": [
            {"example_id": example.id, "key": key}
            for example, (_, skipped) in zip(examples, evidence, strict=True)
            for key in skipped
        ],
        "digest": records_digest(records),
    }


# --------------------------------------------------------------------------- checks


def _check(check_id: str, name: str, problems: list[str], ok_detail: str) -> dict[str, Any]:
    return {
        "id": check_id,
        "name": name,
        "passed": not problems,
        "detail": "; ".join(problems) if problems else ok_detail,
    }


def _failures(ids: list[str], what: str) -> list[str]:
    if not ids:
        return []
    shown = ", ".join(ids[:3]) + (", ..." if len(ids) > 3 else "")
    return [f"{len(ids):,} records {what}: {shown}"]


def _table_block(user: str, prompt: PromptDefinition) -> list[str] | None:
    start = user.find(f"\n\n{prompt.table_label}\n")
    end = user.rfind(f"\n\n{prompt.question_label} ")
    if start < 0 or end < start:
        return None
    return user[start + len(prompt.table_label) + 3 : end].split("\n")


def _context_lines(user: str, prompt: PromptDefinition) -> list[str]:
    end = user.find(f"\n\n{prompt.table_label}\n")
    head = user[:end] if end >= 0 else user
    return head.split("\n")[1:]


def run_checks(
    by_split: SplitRecords,
    prompt: PromptDefinition,
    previous: Mapping[str, str] | None,
    allowed_ids: Collection[str],
    test_ids: Collection[str],
) -> list[dict[str, Any]]:
    """Checks C1 to C10 over every record of every split given."""
    pairs = [
        (example, record)
        for examples, records in by_split.values()
        for example, record in zip(examples, records, strict=True)
    ]
    shape = re.compile(
        re.escape(prompt.program_label) + r" ([^\n]+)\n" + re.escape(prompt.answer_label)
        + r" ([^\n]+)"
    )  # fmt: skip
    bad_shape, bad_program, bad_answer, bad_table = [], [], [], []
    bad_evidence, bad_question, bad_system, changed = [], [], [], []
    for example, record in pairs:
        match = shape.fullmatch(record.target)
        if match is None:
            bad_shape.append(record.example_id)
        else:
            if match.group(1) != example.program:
                bad_program.append(record.example_id)
            answer = match.group(2)
            if answer not in YES_NO and not _ANSWER.fullmatch(answer):
                bad_answer.append(record.example_id)

        block = _table_block(record.user, prompt)
        table = example.table or []
        columns = len(table[0]) if table else 0
        if (
            block is None
            or len(block) != len(table) + 1
            or any(len(_BAR.findall(line)) != columns + 1 for line in block)
        ):
            bad_table.append(record.example_id)

        lines = _context_lines(record.user, prompt)
        wanted = [f"{prompt.list_marker}{passage}" for passage in text_evidence(example)[0]]
        if lines != (wanted or [prompt.no_context]):
            bad_evidence.append(record.example_id)
        if not record.user.endswith(f"{prompt.question_label} {_one_line(example.question)}"):
            bad_question.append(record.example_id)
        if record.system != prompt.system or record.prompt_id != prompt.prompt_id:
            bad_system.append(record.example_id)
        if build_record(example, prompt) != record:
            changed.append(record.example_id)

    digests = {name: records_digest(records) for name, (_, records) in by_split.items()}
    moved = []
    if previous is not None:
        if previous.get("prompt_id") != prompt.prompt_id:
            moved.append(
                f"the prompt id changed from {str(previous.get('prompt_id'))[:12]}... to "
                f"{prompt.prompt_id[:12]}..."
            )
        moved += [
            f"the {name} records changed (digest {str(previous[name])[:12]}... is now "
            f"{digest[:12]}...)"
            for name, digest in digests.items()
            if name in previous and previous[name] != digest
        ]
        if moved:
            moved.append("if this was intended, review and commit the rewritten report")

    ids = [record.example_id for _, record in pairs]
    outside = sorted({i for i in ids if i in test_ids or i not in allowed_ids})

    total = len(pairs)
    return [
        _check(
            "C1", "Target shape", _failures(bad_shape, "are not two lines"), f"{total:,} targets"
        ),
        _check(
            "C2",
            "Program unchanged",
            _failures(bad_program, "differ from the dataset's program"),
            "every program is the dataset's, character for character",
        ),
        _check(
            "C3",
            "Answer form",
            _failures(bad_answer, "have an answer that is not a plain number or yes/no"),
            "every answer is a plain number or yes/no",
        ),
        _check(
            "C4",
            "Table shape",
            _failures(bad_table, "have a malformed table"),
            "every table has its header, separator, and rows",
        ),
        _check(
            "C5",
            "Evidence complete",
            _failures(bad_evidence, "do not show exactly their text evidence"),
            "every record shows exactly its text evidence",
        ),
        _check(
            "C6",
            "Question present",
            _failures(bad_question, "do not end with their question"),
            "every user text ends with its question",
        ),
        _check(
            "C7",
            "One system text",
            _failures(bad_system, "carry another system text or prompt id"),
            f"prompt id {prompt.prompt_id[:12]}...",
        ),
        _check(
            "C8",
            "Deterministic",
            _failures(changed, "came out differently the second time"),
            "serializing again gives identical records",
        ),
        _check(
            "C9",
            "Digests",
            moved,
            ", ".join(f"{name} {digest[:12]}..." for name, digest in digests.items()),
        ),
        _check(
            "C10",
            "No test record",
            _failures(outside, "are not in fit or dev"),
            "every record is a fit or dev example",
        ),
    ]


# --------------------------------------------------------------------------- report


def build_report(
    by_split: SplitRecords,
    prompt: PromptDefinition,
    previous: Mapping[str, str] | None,
    allowed_ids: Collection[str],
    test_ids: Collection[str],
    checked: Collection[str] = ("fit", "dev"),
) -> dict[str, Any]:
    """The whole report. ``previous`` is ``read_committed_digests`` before this build.

    Statistics are given for every split passed in; the checks run over the ``checked`` ones,
    so quick-dev, which is part of dev, is not checked twice.
    """
    to_check = {name: pair for name, pair in by_split.items() if name in checked}
    return {
        "schema": 1,
        "generated_by": REPORT_COMMAND,
        "prompt": {
            "prompt_id": prompt.prompt_id,
            "version": prompt.version,
            "system_characters": len(prompt.system),
        },
        "splits": {
            name: split_stats(examples, records) for name, (examples, records) in by_split.items()
        },
        "checks": run_checks(to_check, prompt, previous, allowed_ids, test_ids),
    }


def read_committed_digests(report_dir: Path | str) -> dict[str, str] | None:
    """The prompt id and per-split digests of the report on disk, or ``None`` if there is none."""
    path = report_paths(report_dir)[1]
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        digests = {name: stats["digest"] for name, stats in report["splits"].items()}
        digests["prompt_id"] = report["prompt"]["prompt_id"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return digests


def render_markdown(report: dict[str, Any]) -> str:
    prompt, splits = report["prompt"], report["splits"]
    names = list(splits)
    failed = [check for check in report["checks"] if not check["passed"]]
    lines = [
        "# FinQA serialization report",
        "",
        f"Generated by `{report['generated_by']}` from `config/prompt.yaml`, the frozen split,",
        "and the pinned snapshot. Do not edit by hand.",
        "",
        "This file holds counts, lengths in characters, and digests only. It contains no",
        "question, passage, table, or answer text. `quick_dev` is part of dev.",
        "",
        "## Prompt",
        "",
        f"- Prompt id: `{prompt['prompt_id']}`",
        f"- Version: {prompt['version']}",
        f"- System text: {prompt['system_characters']:,} characters, the same in every record",
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

    lines += [
        "",
        "## Records",
        "",
        "| Measure | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    rows: dict[str, list[Any]] = {}
    for name in names:
        for key, value in splits[name].items():
            if key in ("skipped_evidence_keys", "digest", "system_characters"):
                continue
            if isinstance(value, dict):
                for inner, number in value.items():
                    rows.setdefault(f"{_label(key)}: `{inner}`", []).append(number)
            else:
                rows.setdefault(_label(key), []).append(value)
    lines += [
        f"| {label} | " + " | ".join(_number(value) for value in values) + " |"
        for label, values in rows.items()
    ]

    lines += ["", "## Evidence keys that were skipped", ""]
    skipped = [(name, item) for name in names for item in splits[name]["skipped_evidence_keys"]]
    if skipped:
        lines += ["| Split | Example | Key |", "|---|---|---|"]
        lines += [
            f"| {name} | `{item['example_id']}` | `{item['key']}` |" for name, item in skipped
        ]
    else:
        lines.append("None.")

    lines += [
        "",
        "## Digests",
        "",
        "SHA-256 over every record of a split, in order. A changed digest means at least one",
        "character of one record changed.",
        "",
        "| Split | Digest |",
        "|---|---|",
    ]
    lines += [f"| {name} | `{splits[name]['digest']}` |" for name in names]
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
