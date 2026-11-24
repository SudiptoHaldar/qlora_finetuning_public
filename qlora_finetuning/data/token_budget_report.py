"""The token budget report: how long the records are in tokens, whether they fit, and how
many tokens one pass over the training data holds (req_001 v1.04).

Token counts and example ids only: no example text, no dollar figure, no timestamp. Prices are
applied elsewhere (req_004 v4.02, req_005 v5.02).

Usage::

    from qlora_finetuning.data.token_budget_report import build_report, write_report

    by_split = {"fit": (fit_records, fit_counts), "dev": (dev_records, dev_counts)}
    report = build_report(by_split, budget, inputs, committed_digests)
    write_report(report, budget.report_dir)
    report["splits"]["fit"]["full_sequence_tokens"]["max"]       # 1862
    report["splits"]["fit"]["tokens_in_one_pass"]                # 3254181
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from qlora_finetuning.data.schema_report import _label, _number, render_json
from qlora_finetuning.data.serialization import CanonicalRecord, records_digest
from qlora_finetuning.data.token_budget import TokenBudgetConfig, TokenCounts

REPORT_STEM = "finqa_token_budget_report"
REPORT_COMMAND = "python -m scripts.finqa_tokens"
SERIALIZE_COMMAND = "python -m scripts.finqa_serialize"

SplitCounts = Mapping[str, tuple[Sequence[CanonicalRecord], Sequence[TokenCounts]]]

__all__ = [
    "build_report",
    "render_json",
    "render_markdown",
    "run_checks",
    "split_stats",
    "stale_files",
    "token_stats",
    "write_report",
]


# --------------------------------------------------------------------------- statistics


def token_stats(values: Sequence[int]) -> dict[str, Any]:
    """Minimum, 50th, 90th, 98th, 99th percentile (nearest rank), and maximum."""
    if not values:
        return {"min": None, "p50": None, "p90": None, "p98": None, "p99": None, "max": None}
    ordered = sorted(values)

    def rank(share: float) -> int:
        return ordered[max(math.ceil(share * len(ordered)), 1) - 1]

    return {
        "min": ordered[0],
        "p50": rank(0.50),
        "p90": rank(0.90),
        "p98": rank(0.98),
        "p99": rank(0.99),
        "max": ordered[-1],
    }


def split_stats(
    records: Sequence[CanonicalRecord], counts: Sequence[TokenCounts], cfg: TokenBudgetConfig
) -> dict[str, Any]:
    """Token statistics for one split."""
    totals = [item.total for item in counts]
    in_one_pass = sum(totals)
    system_tokens = sum(item.system for item in counts)
    characters = sum(len(r.system) + len(r.user) + len(r.target) for r in records)
    longest = sorted(counts, key=lambda item: (-item.total, item.example_id))
    return {
        "records": len(counts),
        "system_text_tokens": counts[0].system if counts else 0,
        "template_tokens": counts[0].template if counts else 0,
        "user_text_tokens": token_stats([item.user for item in counts]),
        "target_tokens_with_end_token": token_stats([item.target for item in counts]),
        "prompt_tokens": token_stats([item.prompt for item in counts]),
        "full_sequence_tokens": token_stats(totals),
        "full_sequences_longer_than": {
            str(limit): sum(total > limit for total in totals) for limit in cfg.thresholds
        },
        "tokens_in_one_pass": in_one_pass,
        "prompt_tokens_in_one_pass": sum(item.prompt for item in counts),
        "system_text_tokens_in_one_pass": system_tokens,
        "system_text_share_percent": round(100 * system_tokens / in_one_pass, 1)
        if in_one_pass
        else 0.0,
        "characters_per_token": round(characters / in_one_pass, 2) if in_one_pass else 0.0,
        "longest": [
            {"example_id": item.example_id, "tokens": item.total}
            for item in longest[: cfg.longest_listed]
        ],
    }


# --------------------------------------------------------------------------- checks


def _check(check_id: str, name: str, problems: list[str], ok_detail: str) -> dict[str, Any]:
    return {
        "id": check_id,
        "name": name,
        "passed": not problems,
        "detail": "; ".join(problems) if problems else ok_detail,
    }


def _named(items: list[TokenCounts], what: str) -> list[str]:
    if not items:
        return []
    shown = ", ".join(item.example_id for item in items[:3]) + (", ..." if len(items) > 3 else "")
    return [f"{len(items):,} records {what}: {shown}"]


def run_checks(
    by_split: SplitCounts,
    cfg: TokenBudgetConfig,
    fingerprint: str,
    prompt_id: str,
    committed: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    """Checks T1 to T6 over the splits given."""
    counts = [item for _, split_counts in by_split.values() for item in split_counts]
    longest = max((item.total for item in counts), default=0)
    longest_reply = max((item.target for item in counts), default=0)
    too_long = sorted(
        (item for item in counts if item.total > cfg.max_length), key=lambda item: -item.total
    )
    reply_too_long = [item for item in counts if item.target > cfg.max_new_tokens]
    thinking = [item for item in counts if not item.thinking_off]
    overheads = sorted({item.template for item in counts})

    tokenizer = []
    if fingerprint != cfg.fingerprint:
        tokenizer.append(
            f"the tokenizer in use has fingerprint {fingerprint[:12]}..., not the pinned "
            f"{cfg.fingerprint[:12]}... of {cfg.model_id}"
        )

    same = []
    if committed is None:
        same.append(f"there is no serialization report to compare with; run {SERIALIZE_COMMAND}")
    else:
        if committed.get("prompt_id") != prompt_id:
            same.append("the serialization report was built with another prompt id")
        same += [
            f"the {name} records differ from the ones the serialization report pins"
            for name, (records, _) in by_split.items()
            if committed.get(name) != records_digest(records)
        ]
        if same:
            same.append(f"run {SERIALIZE_COMMAND} first")

    return [
        _check(
            "T1",
            "Fits",
            _named(too_long, f"are longer than {cfg.max_length:,} tokens"),
            f"longest full sequence is {longest:,} of {cfg.max_length:,} tokens",
        ),
        _check(
            "T2",
            "Reply limit",
            _named(reply_too_long, f"have a target longer than {cfg.max_new_tokens:,} tokens"),
            f"longest target is {longest_reply:,} of {cfg.max_new_tokens:,} tokens",
        ),
        _check("T3", "Tokenizer", tokenizer, f"{cfg.model_id} at {cfg.revision[:12]}..."),
        _check(
            "T4",
            "Thinking off",
            _named(thinking, "do not end with the empty think block, or have one in the target"),
            "every prompt ends with the empty think block",
        ),
        _check(
            "T5",
            "Template constant",
            [f"the template adds different numbers of tokens: {overheads}"]
            if len(overheads) > 1
            else [],
            f"the template adds {overheads[0] if overheads else 0} tokens to every record",
        ),
        _check("T6", "Same records", same, f"prompt id {prompt_id[:12]}..., digests as pinned"),
    ]


# --------------------------------------------------------------------------- report


def build_report(
    by_split: SplitCounts,
    cfg: TokenBudgetConfig,
    inputs: Mapping[str, str],
    committed: Mapping[str, str] | None,
    checked: Collection[str] = ("fit", "dev"),
) -> dict[str, Any]:
    """The whole report.

    ``inputs`` names what was counted: ``tokenizer_fingerprint``, ``prompt_id``, and
    ``split_content_hash``. ``committed`` is the serialization report's digests. Statistics
    are given for every split passed in; the checks run over the ``checked`` ones.
    """
    to_check = {name: pair for name, pair in by_split.items() if name in checked}
    return {
        "schema": 1,
        "generated_by": REPORT_COMMAND,
        "inputs": {
            "tokenizer": cfg.model_id,
            "tokenizer_revision": cfg.revision,
            "tokenizer_fingerprint": inputs["tokenizer_fingerprint"],
            "prompt_id": inputs["prompt_id"],
            "split_content_hash": inputs["split_content_hash"],
        },
        "limits": {"max_length": cfg.max_length, "max_new_tokens": cfg.max_new_tokens},
        "splits": {
            name: split_stats(records, counts, cfg) for name, (records, counts) in by_split.items()
        },
        "checks": run_checks(
            to_check, cfg, inputs["tokenizer_fingerprint"], inputs["prompt_id"], committed
        ),
    }


def _cell(value: Any) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else _number(value)


def render_markdown(report: dict[str, Any]) -> str:
    inputs, limits, splits = report["inputs"], report["limits"], report["splits"]
    names = list(splits)
    failed = [check for check in report["checks"] if not check["passed"]]
    lines = [
        "# FinQA token budget report",
        "",
        f"Generated by `{report['generated_by']}` from the canonical records of the fit and dev",
        "splits. Do not edit by hand.",
        "",
        "Token counts and example ids only: no example text and no prices. A full sequence is",
        "what Path A trains on: the chat-templated system and user texts, the target, and the",
        "end token. `quick_dev` is part of dev. Percentiles are nearest rank.",
        "",
        "## Inputs and limits",
        "",
        f"- Tokenizer: `{inputs['tokenizer']}` at `{inputs['tokenizer_revision']}`",
        f"- Tokenizer fingerprint: `{inputs['tokenizer_fingerprint']}`",
        f"- Prompt id: `{inputs['prompt_id']}`",
        f"- Split content hash: `{inputs['split_content_hash']}`",
        f"- Longest training sequence allowed (`max_length`): {limits['max_length']:,} tokens",
        f"- Longest reply allowed (`max_new_tokens`): {limits['max_new_tokens']:,} tokens",
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
        "## Tokens",
        "",
        "| Measure | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    rows: dict[str, list[Any]] = {}
    for name in names:
        for key, value in splits[name].items():
            if key == "longest":
                continue
            if isinstance(value, dict):
                for inner, number in value.items():
                    rows.setdefault(f"{_label(key)}: `{inner}`", []).append(number)
            else:
                rows.setdefault(_label(key), []).append(value)
    lines += [
        f"| {label} | " + " | ".join(_cell(value) for value in values) + " |"
        for label, values in rows.items()
    ]

    for name in names:
        lines += [
            "",
            f"## Longest records in {name}",
            "",
            "| Example | Full sequence tokens |",
            "|---|---:|",
        ]
        lines += [
            f"| `{item['example_id']}` | {item['tokens']:,} |" for item in splits[name]["longest"]
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
