"""Checks X1 to X7 on the exported training files, read back from disk (req_004 v4.02).

What is uploaded is the file, so the validator reads the bytes that were written and never
the records they were made from. It compares them with the canonical records rendered
again, with the test split's ids, and with AWS's rules for Nova fine-tuning data, written
out here as checks (AWS's own script is not run).

A failure names example ids, line numbers, and words. It never shows a text.

Usage::

    from qlora_finetuning.frontier.export_validation import load_token_volumes, validate_export

    volumes = load_token_volumes("reports/data/finqa_token_budget_report.json")
    checks = validate_export(export, cfg, context, volumes, ratio=Decimal("1.5"))
    all(check["passed"] for check in checks)

| Id | Check | Passes when |
|---|---|---|
| X1 | Same records | One line per record of the split, in order, with the record's texts |
| X2 | No test rows | No exported id is a test id; the two files share no id |
| X3 | Format | Bedrock's conversation format, UTF-8, line breaks only, allowed file names |
| X4 | Reserved words | No text holds one |
| X5 | Limits | Record counts and the longest record are inside AWS's limits |
| X6 | Same question | The prompt and split are the current ones, and the token report's |
| X7 | Reproducible | Rendering again gives the bytes on disk |
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

from qlora_finetuning.data import manifest_hash, split_ids
from qlora_finetuning.frontier.config import FrontierConfig
from qlora_finetuning.frontier.export import (
    RESERVED_WORDS,
    SPLITS,
    TRAINING,
    Export,
    ExportFile,
    render_split,
    sha256_of,
)
from qlora_finetuning.settings import ConfigError

FILE_NAME = re.compile(r"[A-Za-z0-9_.-]+")
# AWS: file names "can consist of only alphanumeric characters, underscores, hyphens,
# slashes, and dots".
S3_KEY = re.compile(r"[A-Za-z0-9_./-]+")
LINE_KEYS = {"schemaVersion", "system", "messages"}
SHOWN = 3


def check(check_id: str, name: str, problems: Sequence[str], ok_detail: str) -> dict[str, Any]:
    """One check as plain values: id, name, passed, and a one-line detail."""
    return {
        "id": check_id,
        "name": name,
        "passed": not problems,
        "detail": "; ".join(problems) if problems else ok_detail,
    }


def some(items: Sequence[Any], what: str) -> list[str]:
    """``3 lines ...: a, b, c, ...`` or nothing. Never more than three are shown."""
    if not items:
        return []
    shown = ", ".join(str(item) for item in items[:SHOWN])
    return [f"{len(items):,} {what}: {shown}{', ...' if len(items) > SHOWN else ''}"]


# --------------------------------------------------------------------------- token volumes


@dataclass(frozen=True)
class SplitVolume:
    """Qwen token counts of one split, from the committed token budget report."""

    records: int
    tokens_in_one_pass: int  # full sequences: prompt, target, end token
    prompt_tokens_in_one_pass: int
    longest: int  # the longest full sequence


@dataclass(frozen=True)
class TokenVolumes:
    prompt_id: str
    split_hash: str
    splits: dict[str, SplitVolume]


def load_token_volumes(token_budget_json: Path | str) -> TokenVolumes:
    """The token volumes of each split and what they were counted from."""
    path = Path(token_budget_json)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return TokenVolumes(
            prompt_id=str(data["inputs"]["prompt_id"]),
            split_hash=str(data["inputs"]["split_content_hash"]),
            splits={
                name: SplitVolume(
                    records=int(block["records"]),
                    tokens_in_one_pass=int(block["tokens_in_one_pass"]),
                    prompt_tokens_in_one_pass=int(block["prompt_tokens_in_one_pass"]),
                    longest=int(block["full_sequence_tokens"]["max"]),
                )
                for name, block in data["splits"].items()
            },
        )
    except FileNotFoundError:
        raise ConfigError(
            f"token budget report not found: {path}. Run python -m scripts.finqa_tokens"
        ) from None
    except (ValueError, KeyError, TypeError, AttributeError):
        raise ConfigError(f"token budget report cannot be read: {path}") from None


# --------------------------------------------------------------------------- reading a file


@dataclass
class _Read:
    """One exported file as found on disk, and the split's records rendered again."""

    file: ExportFile
    raw: bytes | None = None
    lines: list[str] = field(default_factory=list)
    objects: list[Any] = field(default_factory=list)  # one per line; None if not JSON
    ids: list[str] = field(default_factory=list)
    file_problems: list[str] = field(default_factory=list)
    fresh: bytes = b""
    records: list[Any] = field(default_factory=list)

    def name_of(self, index: int) -> str:
        """The example id of a line, for a message; the line number if there is none."""
        return self.ids[index] if index < len(self.ids) else f"line {index + 1}"


def _texts(obj: Any) -> tuple[str, str, str] | None:
    """The system, user, and target texts of a well-formed line, else ``None``."""
    try:
        system, messages = obj["system"], obj["messages"]
        if len(system) != 1 or set(system[0]) != {"text"} or len(messages) != 2:
            return None
        parts = [system[0]["text"]]
        for message, role in zip(messages, ("user", "assistant"), strict=True):
            content = message["content"]
            if set(message) != {"role", "content"} or message["role"] != role:
                return None
            if len(content) != 1 or set(content[0]) != {"text"}:
                return None
            parts.append(content[0]["text"])
    except (KeyError, TypeError, IndexError):
        return None
    if not all(isinstance(part, str) for part in parts):
        return None
    return parts[0], parts[1], parts[2]


def _read(file: ExportFile, role: str, cfg: FrontierConfig, context: Any) -> _Read:
    read = _Read(file=file)
    read.fresh, _, read.records = render_split(role, cfg, context)
    what = f"the {role} file"
    try:
        read.raw = file.path.read_bytes()
        read.ids = file.ids_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        read.file_problems.append(f"{what} or its id list is missing")
        return read
    try:
        text = read.raw.decode("utf-8")
    except UnicodeDecodeError:
        read.file_problems.append(f"{what} is not UTF-8")
        return read
    if "\r" in text:
        read.file_problems.append(f"{what} holds a carriage return")
    if text and not text.endswith("\n"):
        read.file_problems.append(f"{what} does not end with a line break")
    read.lines = text.split("\n")
    if read.lines and read.lines[-1] == "":
        read.lines.pop()
    for line in read.lines:
        try:
            read.objects.append(json.loads(line))
        except ValueError:
            read.objects.append(None)
    return read


# --------------------------------------------------------------------------- the checks


def _same_records(reads: dict[str, _Read]) -> dict[str, Any]:
    problems = []
    for role, read in reads.items():
        expected = len(read.records)
        if len(read.lines) != expected:
            problems.append(
                f"the {role} file has {len(read.lines):,} lines for {expected:,} records"
            )
        if read.ids != [record.example_id for record in read.records]:
            problems.append(f"the {role} id list is not the {SPLITS[role]} split in order")
        differing = [
            record.example_id
            for record, obj in zip(read.records, read.objects, strict=False)
            if _texts(obj) != (record.system, record.user, record.target)
        ]
        problems += some(differing, f"{role} lines differ from their canonical record")
    counts = ", ".join(f"{SPLITS[role]} {len(read.records):,}" for role, read in reads.items())
    return check("X1", "Same records", problems, f"one line per record, in order: {counts}")


def _no_test_rows(reads: dict[str, _Read], test_ids: frozenset[str]) -> dict[str, Any]:
    problems = []
    seen: dict[str, str] = {}
    shared = []
    for role, read in reads.items():
        ids = set(read.ids) | {record.example_id for record in read.records}
        problems += some(sorted(ids & test_ids), f"{role} ids are test ids")
        for example_id in read.ids:
            if seen.setdefault(example_id, role) != role:
                shared.append(example_id)
    problems += some(sorted(set(shared)), "ids are in both files")
    return check(
        "X2",
        "No test rows",
        problems,
        f"none of {len(seen):,} exported ids is among the {len(test_ids):,} test ids",
    )


def _format(reads: dict[str, _Read], export: Export, cfg: FrontierConfig) -> dict[str, Any]:
    problems = []
    for role, read in reads.items():
        problems += read.file_problems
        for path in (read.file.path, read.file.ids_path):
            if not FILE_NAME.fullmatch(path.name):
                problems.append(f"file name {path.name!r} has a character AWS does not allow")
        key = f"{cfg.s3_prefix}{export.export_id}/{read.file.path.name}"
        if not S3_KEY.fullmatch(key):
            problems.append(f"the S3 key of the {role} file has a character AWS does not allow")
        bad = []
        for index, obj in enumerate(read.objects):
            texts = _texts(obj)
            well_formed = (
                isinstance(obj, dict)
                and set(obj) == LINE_KEYS
                and obj["schemaVersion"] == cfg.schema_version
                and texts is not None
                and all(text.strip() for text in texts)
            )
            if not well_formed:
                bad.append(f"line {index + 1}")
        problems += some(bad, f"{role} lines are not a conversation of the expected shape")
    return check(
        "X3",
        "Format",
        problems,
        f"{export.schema_version}: system, one user turn, one assistant turn; UTF-8, "
        "line breaks only",
    )


def _reserved_words(reads: dict[str, _Read]) -> dict[str, Any]:
    found = []
    for read in reads.values():
        for index, obj in enumerate(read.objects):
            texts = _texts(obj)
            lowered = (
                " ".join(texts).lower() if texts else str(read.lines[index]).lower()
            )
            words = [word for word in RESERVED_WORDS if word in lowered]
            if words:
                found.append(f"{read.name_of(index)} ({', '.join(words)})")
    return check(
        "X4",
        "Reserved words",
        some(found, "records hold a reserved word"),
        f"none of the {len(RESERVED_WORDS)} reserved words, in any case, in any text",
    )


def _limits(
    reads: dict[str, _Read], cfg: FrontierConfig, volumes: TokenVolumes, ratio: Decimal
) -> dict[str, Any]:
    problems = []
    training = len(reads[TRAINING].lines)
    total = sum(len(read.lines) for read in reads.values())
    if not cfg.min_records <= training <= cfg.max_records:
        problems.append(
            f"{training:,} training records; AWS takes {cfg.min_records:,} to {cfg.max_records:,}"
        )
    if total > cfg.max_records:
        problems.append(
            f"{total:,} training and validation records; AWS takes at most {cfg.max_records:,}"
        )
    factor = ratio * (1 + cfg.margin_percent / 100)
    longest = 0
    for role in reads:
        volume = volumes.splits.get(SPLITS[role])
        if volume is None:
            problems.append(f"the token budget report has no {SPLITS[role]} split")
            continue
        estimated = int((volume.longest * factor).to_integral_value(rounding=ROUND_CEILING))
        longest = max(longest, estimated)
        if estimated > cfg.max_sample_tokens:
            problems.append(
                f"the longest {SPLITS[role]} record is about {estimated:,} tokens; "
                f"AWS takes at most {cfg.max_sample_tokens:,}"
            )
    return check(
        "X5",
        "Limits",
        problems,
        f"{training:,} training and {total - training:,} validation records; the longest "
        f"record is about {longest:,} of {cfg.max_sample_tokens:,} tokens",
    )


def _same_question(export: Export, context: Any, volumes: TokenVolumes) -> dict[str, Any]:
    prompt_id, split_hash = context.prompt.prompt_id, manifest_hash(context.split_cfg)
    problems = []
    if export.prompt_id != prompt_id:
        problems.append("the files were made from another prompt wording")
    if export.split_hash != split_hash:
        problems.append("the files were made from another split manifest")
    if volumes.prompt_id != prompt_id:
        problems.append(
            "the token budget report was made from another prompt wording; run "
            "python -m scripts.finqa_tokens"
        )
    if volumes.split_hash != split_hash:
        problems.append("the token budget report was made from another split manifest")
    return check(
        "X6",
        "Same question",
        problems,
        f"prompt {prompt_id[:12]}..., split manifest {split_hash[:12]}..., as in the token "
        "budget report",
    )


def _reproducible(reads: dict[str, _Read]) -> dict[str, Any]:
    problems = []
    for role, read in reads.items():
        fresh = sha256_of(read.fresh)
        on_disk = sha256_of(read.raw) if read.raw is not None else None
        if not fresh == on_disk == read.file.sha256:
            problems.append(f"the {role} file on disk is not what rendering gives now")
    digests = ", ".join(f"{role} {read.file.sha256[:12]}..." for role, read in reads.items())
    return check("X7", "Reproducible", problems, f"rendering again gives the same bytes: {digests}")


def validate_export(
    export: Export,
    cfg: FrontierConfig,
    context: Any,
    volumes: TokenVolumes,
    ratio: Decimal,
) -> list[dict[str, Any]]:
    """Run X1 to X7 on the files of ``export`` as they are on disk.

    ``context`` is the resolved ``RunnerContext`` the export was made with. ``ratio`` is
    the Nova to Qwen token ratio (measured or assumed), used for the per-sample limit.
    Only ``fit`` and ``dev`` are loaded; of the test split, only its ids are read.
    """
    reads = {role: _read(file, role, cfg, context) for role, file in export.files.items()}
    test_ids = frozenset(split_ids("test", context.split_cfg))
    return [
        _same_records(reads),
        _no_test_rows(reads, test_ids),
        _format(reads, export, cfg),
        _reserved_words(reads),
        _limits(reads, cfg, volumes, ratio),
        _same_question(export, context, volumes),
        _reproducible(reads),
    ]
