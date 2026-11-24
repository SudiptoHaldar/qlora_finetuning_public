"""The gold self-check: every gold reply, scored by the harness itself (req_002 v2.02).

Before any model is scored, the scorer is tried on the replies whose score is known: the
dataset's own. Each example's gold reply is built by the serializer, run through
``run_reply``, and scored against itself. All four metrics must come out true.

An example whose gold program does not execute, or whose value does not match its gold
answer, is **flagged**. A flagged example is never dropped: it stays in its split and in
every total. Its id and the reason go into ``splits/finqa_gold_flags.json``, a file of its
own, so the split manifest and its hash never change.

This module lives in the data package because the check covers all 8,281 examples, test rows
included. It reads a test row's gold program and gold answer only, involves no model, and
reports counts and example ids: no question, passage, table, or answer text.

Usage::

    from qlora_finetuning.data.gold_check import load_gold_flags
    from qlora_finetuning.evaluation import summarize

    totals = summarize(scores, flagged=load_gold_flags())
    totals["all"]["execution_accuracy"]        # the headline, over every example
    totals["unflagged"]["execution_accuracy"]  # beside it, without the flagged examples

    # Rebuild the report and the flags (the script does this):
    report = build_report(examples, manifest, prompt, config, previous_flags=read_flags(config))
    write_report(report, config)
    write_flags(build_flags(report), config)
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qlora_finetuning.data.schema_report import _label, _number, render_json
from qlora_finetuning.data.serialization import (
    YES_NO,
    PromptDefinition,
    SerializationError,
    build_record,
)
from qlora_finetuning.data.snapshot import RawExample
from qlora_finetuning.dsl.config import DslConfig, load_dsl_config
from qlora_finetuning.dsl.executor import DUPLICATE_ROW
from qlora_finetuning.dsl.outcome import run_reply
from qlora_finetuning.dsl.reply import ReplyFormat
from qlora_finetuning.evaluation.metrics import (
    EvaluationConfig,
    EvaluationError,
    ExampleScore,
    Gold,
    allowance_used,
    load_evaluation_config,
    score_reply,
    summarize,
)

REPORT_STEM = "finqa_gold_check_report"
REPORT_COMMAND = "python -m scripts.finqa_gold_check"
SPLITS = ("fit", "dev", "test")
REPORTED = ("fit", "dev", "quick_dev", "test")
TRAINING_SPLITS = ("fit", "dev")

# Reasons for a flag, beside the error codes of qlora_finetuning.dsl.
VALUE_MISMATCH = "value_mismatch"
ANSWER_UNREADABLE = "answer_unreadable"
UNSERIALIZABLE = "unserializable"

_IDS_SHOWN = 50


@dataclass(frozen=True)
class GoldCheck:
    """What the self-check found for one example. Holds no text from the example."""

    example_id: str
    reason: str | None
    strict: bool
    canonical: bool
    score: ExampleScore
    allowance_used: float | None = None
    yes_no: bool = False
    duplicate_row: bool = False

    @property
    def flagged(self) -> bool:
        return self.reason is not None


def _failed(example_id: str, reason: str) -> GoldCheck:
    """An example that could not be scored at all: wrong on everything, and flagged."""
    score = ExampleScore(
        example_id=example_id,
        format_valid=False,
        executed=False,
        execution_correct=False,
        answer_correct=False,
        program_match=False,
        error_code=reason,
    )
    return GoldCheck(example_id, reason, strict=False, canonical=False, score=score)


def check_example(
    example: RawExample,
    prompt: PromptDefinition,
    config: EvaluationConfig,
    reply_format: ReplyFormat | None = None,
    dsl_config: DslConfig | None = None,
) -> GoldCheck:
    """Build an example's gold reply, run it, and score it against itself.

    The example is flagged when its gold program fails (the reason is the error code) or its
    value does not match the gold answer (``value_mismatch``), and when no gold reply can be
    built or read at all (``unserializable``, ``answer_unreadable``).
    """
    reply_format = reply_format or ReplyFormat.from_prompt(prompt)
    dsl_config = dsl_config or load_dsl_config()
    try:
        record = build_record(example, prompt)
    except SerializationError:
        return _failed(example.id, UNSERIALIZABLE)
    try:
        gold = Gold.from_target(record.target, reply_format, dsl_config)
    except EvaluationError:
        return _failed(example.id, ANSWER_UNREADABLE)

    outcome = run_reply(record.target, example.table, reply_format, dsl_config)
    score = score_reply(outcome, gold, config, example.id, dsl_config)
    if not outcome.ok:
        reason = outcome.error_code
    elif not score.execution_correct:
        reason = VALUE_MISMATCH
    else:
        reason = None

    used = None
    if outcome.ok and not isinstance(outcome.value, str) and not isinstance(gold.value, str):
        used = allowance_used(outcome.value, gold.value, config.execution)
    return GoldCheck(
        example_id=example.id,
        reason=reason,
        strict=outcome.strict,
        canonical=outcome.program == str(example.program).strip(),
        score=score,
        allowance_used=used,
        yes_no=str(example.exe_ans) in YES_NO,
        duplicate_row=DUPLICATE_ROW in outcome.notes,
    )


# --------------------------------------------------------------------------- report


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
    return [f"{len(ids):,} examples {what}: {shown}"]


def _stats(checks: Sequence[GoldCheck]) -> dict[str, Any]:
    """Counts for one split. Every count is over all the examples of the split."""
    totals = summarize(check.score for check in checks)["all"]
    flagged = [check for check in checks if check.flagged]
    used = [check.allowance_used for check in checks if check.allowance_used is not None]
    within = [share for share in used if share <= 1.0]
    return {
        "examples": len(checks),
        "gold_programs_that_reproduce_the_gold_answer": len(checks) - len(flagged),
        "flagged_examples": len(flagged),
        "flagged_by_reason": {
            str(reason): count
            for reason, count in sorted(
                Counter(check.reason for check in flagged).items(),
                key=lambda item: (-item[1], str(item[0])),
            )
        },
        "format_validity": totals["format_validity"]["count"],
        "execution_accuracy": totals["execution_accuracy"]["count"],
        "answer_accuracy": totals["answer_accuracy"]["count"],
        "program_match": totals["program_match"]["count"],
        "largest_share_of_the_allowance_used": round(max(within), 6) if within else None,
        "yes_or_no_examples": sum(check.yes_no for check in checks),
        "programs_naming_a_repeated_row": sum(check.duplicate_row for check in checks),
    }


def flags_content_hash(flagged: Sequence[dict[str, str]]) -> str:
    """SHA-256 over one line per flagged example: id, split, reason."""
    digest = hashlib.sha256()
    for item in flagged:
        digest.update(f"{item['example_id']}\t{item['split']}\t{item['reason']}\n".encode())
    return digest.hexdigest()


def build_report(
    examples: Sequence[RawExample],
    manifest: dict[str, Any],
    prompt: PromptDefinition,
    config: EvaluationConfig,
    previous_flags: dict[str, Any] | None = None,
    data_problem: str | None = None,
    dsl_config: DslConfig | None = None,
) -> dict[str, Any]:
    """Check every example and return the report as a dict.

    ``manifest`` is the split manifest. ``previous_flags`` is the flags file as it was on
    disk before this build (``read_flags``), or ``None`` if there was none. ``data_problem``
    is a sentence saying why the manifest does not describe ``examples``, or ``None`` when it
    does; the caller finds that out, so this function reads no file.
    """
    reply_format = ReplyFormat.from_prompt(prompt)
    dsl_config = dsl_config or load_dsl_config()

    def run() -> list[GoldCheck]:
        return [
            check_example(example, prompt, config, reply_format, dsl_config)
            for example in examples
        ]

    checks = run()
    deterministic = run() == checks

    assignments: dict[str, str] = manifest["assignments"]
    quick_dev = set(manifest["quick_dev"])
    by_split: dict[str, list[GoldCheck]] = {name: [] for name in REPORTED}
    unassigned: list[str] = []
    for check in checks:
        split = assignments.get(check.example_id)
        if split not in SPLITS:
            unassigned.append(check.example_id)
            continue
        by_split[split].append(check)
        if check.example_id in quick_dev:
            by_split["quick_dev"].append(check)

    flagged = [
        {
            "example_id": check.example_id,
            "split": str(assignments.get(check.example_id, "unassigned")),
            "reason": str(check.reason),
        }
        for check in checks
        if check.flagged
    ]
    settings = {
        "manifest_hash": str(manifest["content_hash"]),
        "prompt_id": prompt.prompt_id,
        "execution": {
            "absolute": config.execution.absolute,
            "relative": config.execution.relative,
        },
        "answer": {"absolute": config.answer.absolute, "relative": config.answer.relative},
    }

    seen = Counter(check.example_id for check in checks)
    repeated = sorted(example_id for example_id, count in seen.items() if count > 1)
    missing = sorted(set(assignments) - set(seen))
    malformed = [check.example_id for check in checks if not (check.strict and check.canonical)]
    training_flags = [item["example_id"] for item in flagged if item["split"] in TRAINING_SPLITS]
    inconsistent = [
        check.example_id for check in checks if not check.flagged and not check.score.all_correct
    ]

    moved: list[str] = []
    if previous_flags is not None:
        if previous_flags.get("settings") != settings:
            moved.append("the settings changed since the flags file was written")
        if previous_flags.get("flagged") != flagged:
            before = len(previous_flags.get("flagged") or [])
            moved.append(f"the flagged examples changed ({before:,} before, {len(flagged):,} now)")
        if moved:
            moved.append("if this was intended, review and commit the rewritten files")

    report_checks = [
        _check(
            "K1",
            "Complete",
            _some(unassigned, "have no split in the manifest")
            + _some(missing, "of the manifest were not checked")
            + _some(repeated, "were checked more than once"),
            f"{len(checks):,} examples, each checked once",
        ),
        _check(
            "K2",
            "Gold replies are well formed",
            _some(malformed, "have a gold reply that is not strict or not in canonical form"),
            "every gold reply is strict and its program is in canonical form",
        ),
        _check(
            "K3",
            "Fit and dev reproduce",
            _some(training_flags, "of fit or dev are flagged"),
            "no fit or dev example is flagged",
        ),
        _check(
            "K4",
            "Self-consistent",
            _some(inconsistent, "are not flagged yet do not score true on all four metrics"),
            f"{len(checks) - len(flagged):,} unflagged examples score true on all four metrics",
        ),
        _check(
            "K5",
            "Flags current",
            moved,
            # The same words with or without an earlier file, so a second run changes nothing.
            f"{len(flagged):,} flagged examples, as the flags file lists",
        ),
        _check(
            "K6",
            "Same data",
            [data_problem] if data_problem else [],
            f"the manifest {settings['manifest_hash'][:12]}... describes this data",
        ),
        _check(
            "K7",
            "Deterministic",
            [] if deterministic else ["checking a second time gave another result"],
            "checking a second time gives the same result",
        ),
    ]
    return {
        "schema": 1,
        "generated_by": REPORT_COMMAND,
        "settings": settings,
        "splits": {name: _stats(by_split[name]) for name in REPORTED},
        "total": _stats(checks),
        "flagged": flagged,
        "checks": report_checks,
    }


def _cell(value: Any) -> str:
    """A count with thousands separators; a share of the allowance with its digits kept."""
    return f"{value:g}" if isinstance(value, float) else _number(value)


def render_markdown(report: dict[str, Any]) -> str:
    settings, splits = report["settings"], report["splits"]
    names = list(splits)
    failed = [check for check in report["checks"] if not check["passed"]]
    lines = [
        "# FinQA gold self-check report",
        "",
        f"Generated by `{report['generated_by']}` from the pinned snapshot, the frozen split,",
        "`config/prompt.yaml`, and `config/evaluation.yaml`. Do not edit by hand.",
        "",
        "Every example's gold reply was run and scored by the evaluation harness itself. An",
        "example whose gold program does not reproduce its gold answer is flagged; it is never",
        "dropped. This file holds counts and example ids only: no question, passage, table,",
        "program, or answer text. `quick_dev` is part of dev. The test split is included: the",
        "check reads its gold programs and answers, and no model is involved.",
        "",
        "## Settings",
        "",
        f"- Split manifest: `{settings['manifest_hash']}`",
        f"- Prompt id: `{settings['prompt_id']}`",
        f"- Execution accuracy: absolute {settings['execution']['absolute']:g}, relative "
        f"{settings['execution']['relative']:g}",
        f"- Answer accuracy: absolute {settings['answer']['absolute']:g}, relative "
        f"{settings['answer']['relative']:g}",
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

    columns = names + ["all"]
    blocks = [splits[name] for name in names] + [report["total"]]
    lines += [
        "",
        "## Results",
        "",
        "Each count is over all the examples of its split.",
        "",
        "| Measure | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for key in blocks[0]:
        if key == "flagged_by_reason":
            continue
        lines.append(
            f"| {_label(key)} | " + " | ".join(_cell(block[key]) for block in blocks) + " |"
        )
    reasons = sorted({reason for block in blocks for reason in block["flagged_by_reason"]})
    lines += [
        f"| Flagged: `{reason}` | "
        + " | ".join(_number(block["flagged_by_reason"].get(reason, 0)) for block in blocks)
        + " |"
        for reason in reasons
    ]

    flagged = report["flagged"]
    lines += ["", "## Flagged examples", ""]
    if not flagged:
        lines.append("None.")
    else:
        lines += [
            f"{len(flagged):,} examples. The same list, in full, is in the JSON file and in the",
            "flags file that scoring reads.",
            "",
            "| Split | Example | Reason |",
            "|---|---|---|",
        ]
        lines += [
            f"| {item['split']} | `{item['example_id']}` | `{item['reason']}` |"
            for item in flagged[:_IDS_SHOWN]
        ]
        if len(flagged) > _IDS_SHOWN:
            lines.append(f"| ... | {len(flagged) - _IDS_SHOWN:,} more in the JSON file | |")
    return "\n".join(lines) + "\n"


def report_paths(report_dir: Path | str) -> tuple[Path, Path]:
    report_dir = Path(report_dir)
    return report_dir / f"{REPORT_STEM}.md", report_dir / f"{REPORT_STEM}.json"


def write_report(report: dict[str, Any], config: EvaluationConfig) -> tuple[Path, Path]:
    """Write the Markdown and JSON report files and return their paths."""
    markdown_path, json_path = report_paths(config.report_dir)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    json_path.write_text(render_json(report), encoding="utf-8", newline="\n")
    return markdown_path, json_path


# --------------------------------------------------------------------------- flags file


def build_flags(report: dict[str, Any]) -> dict[str, Any]:
    """The flags file: which examples are flagged and why, and the settings that decided it."""
    flagged = list(report["flagged"])
    counts = Counter(item["split"] for item in flagged)
    return {
        "schema": 1,
        "generated_by": REPORT_COMMAND,
        "settings": report["settings"],
        "counts": {**{name: counts.get(name, 0) for name in SPLITS}, "total": len(flagged)},
        "content_hash": flags_content_hash(flagged),
        "flagged": flagged,
    }


def write_flags(flags: dict[str, Any], config: EvaluationConfig) -> Path:
    config.flags_file.parent.mkdir(parents=True, exist_ok=True)
    config.flags_file.write_text(render_json(flags), encoding="utf-8", newline="\n")
    return config.flags_file


def read_flags(config: EvaluationConfig) -> dict[str, Any] | None:
    """The flags file as a dict, or ``None`` if it does not exist. Raises if it is damaged."""
    path = config.flags_file
    if not path.is_file():
        return None
    try:
        flags = json.loads(path.read_text(encoding="utf-8"))
        flagged = flags["flagged"]
        valid = isinstance(flagged, list) and flags["content_hash"] == flags_content_hash(flagged)
    except (ValueError, KeyError, TypeError):
        valid = False
    if not valid:
        raise EvaluationError(
            f"the gold flags file {path.name} is damaged or was edited by hand; restore it "
            f"from git or rebuild it with: {REPORT_COMMAND}"
        )
    return flags


def load_gold_flags(config: EvaluationConfig | None = None) -> frozenset[str]:
    """The ids of the flagged examples, for ``summarize(scores, flagged=...)``.

    Reads the committed flags file, never the dataset, so any code may call it. Raises
    ``EvaluationError`` when the file is missing: the self-check comes before any scoring.
    """
    config = config or load_evaluation_config()
    flags = read_flags(config)
    if flags is None:
        raise EvaluationError(
            f"no gold flags file at {config.flags_file.name}; build it with: {REPORT_COMMAND}"
        )
    return frozenset(str(item["example_id"]) for item in flags["flagged"])


def stale_files(
    report: dict[str, Any], config: EvaluationConfig, names: Collection[str] | None = None
) -> list[str]:
    """Names of the report and flags files that are missing or differ from a fresh rendering."""
    markdown_path, json_path = report_paths(config.report_dir)
    fresh = (
        (markdown_path, render_markdown(report)),
        (json_path, render_json(report)),
        (config.flags_file, render_json(build_flags(report))),
    )
    return [
        path.name
        for path, text in fresh
        if (names is None or path.name in names)
        and (not path.is_file() or path.read_text(encoding="utf-8") != text)
    ]
