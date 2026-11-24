"""Canonical serialization: the one prompt and target every arm uses (req_001 v1.03).

An example becomes a ``CanonicalRecord`` of three texts:

- **system**: the same for every example; describes the whole output language;
- **user**: the text evidence (or ``(none)``), the whole table as Markdown, the question;
- **target**: two lines, ``PROGRAM: ...`` and ``ANSWER: ...``.

The wording comes from ``config/prompt.yaml``. Its hash, the **prompt id**, is on every
record. Path A wraps a record in the Qwen chat template and Path B in Bedrock's conversation
format; neither edits it.

Usage::

    from qlora_finetuning.data import serialize_split, load_prompt

    records = serialize_split("fit")          # also "dev", "quick_dev"; "test" is locked
    record = records[0]
    record.system, record.user, record.target, record.prompt_id

    # A prompt for input that is not a dataset example (the demo, a smoke test):
    from qlora_finetuning.data.serialization import format_user
    prompt = load_prompt()
    user = format_user(["Revenue rose."], [["", "2024"], ["Revenue", "10"]], "What was it?", prompt)

Serialization is a pure function of the example and the prompt definition. Nothing is
truncated here; length is the token budget's subject (v1.04).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from qlora_finetuning.data.snapshot import DataConfig, RawExample
from qlora_finetuning.data.split import SplitConfig
from qlora_finetuning.data.split_loader import TestLockedError, load_split
from qlora_finetuning.settings import REPO_ROOT, ConfigError, load_config

REQUIRED_KEYS = (
    "system",
    "user.context_label",
    "user.list_marker",
    "user.no_context",
    "user.table_label",
    "user.question_label",
    "target.program_label",
    "target.answer_label",
    "report_dir",
    "dump_dir",
)
YES_NO = ("yes", "no")
_TEXT_KEY = re.compile(r"text_(\d+)")
_TABLE_KEY = re.compile(r"table_(\d+)")
_LINE_BREAK = re.compile(r"\s*[\r\n]+\s*")


class SerializationError(ConfigError):
    """An example cannot be turned into a canonical record."""


@dataclass(frozen=True)
class PromptDefinition:
    """``config/prompt.yaml``: the system text, the layout, and the id that names them."""

    root: Path
    version: int
    system: str
    context_label: str
    list_marker: str
    no_context: str
    table_label: str
    question_label: str
    program_label: str
    answer_label: str
    report_dir: Path
    dump_dir: Path
    prompt_id: str


@dataclass(frozen=True)
class CanonicalRecord:
    """What a model is shown and what it is expected to reply, for one example."""

    example_id: str
    prompt_id: str
    system: str
    user: str
    target: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


# --------------------------------------------------------------------------- prompt definition


def prompt_id_of(version: int, system: str, user: dict[str, str], target: dict[str, str]) -> str:
    """The SHA-256 that names a prompt definition.

    Computed from the parsed values, so a comment or the order of keys in the file does not
    change it, and any character of the wording does.
    """
    canonical = json.dumps(
        {"version": version, "system": system, "user": user, "target": target},
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_prompt(
    root: Path | str = REPO_ROOT, config_dir: Path | str | None = None
) -> PromptDefinition:
    """Read ``config/prompt.yaml``. Relative paths in it resolve under ``root``."""
    raw = load_config("prompt", required=REQUIRED_KEYS, config_dir=config_dir)
    if raw["version"] < 1:
        raise ConfigError("config prompt.yaml: version must be 1 or more")
    system = str(raw["system"]).strip()
    if not system:
        raise ConfigError("config prompt.yaml: system must not be empty")
    user = {key: str(raw["user"][key]) for key in sorted(raw["user"])}
    target = {key: str(raw["target"][key]) for key in sorted(raw["target"])}
    for section, values in (("user", user), ("target", target)):
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise ConfigError(f"config prompt.yaml: {section}.{key} must be one line")
    root = Path(root)
    return PromptDefinition(
        root=root,
        version=int(raw["version"]),
        system=system,
        context_label=user["context_label"],
        list_marker=user["list_marker"],
        no_context=user["no_context"],
        table_label=user["table_label"],
        question_label=user["question_label"],
        program_label=target["program_label"],
        answer_label=target["answer_label"],
        report_dir=root / str(raw["report_dir"]),
        dump_dir=root / str(raw["dump_dir"]),
        prompt_id=prompt_id_of(int(raw["version"]), system, user, target),
    )


# --------------------------------------------------------------------------- the parts


def _one_line(text: Any) -> str:
    """Text on a single line: a line break and the spaces around it become one space."""
    return _LINE_BREAK.sub(" ", str(text)).strip()


def text_evidence(example: RawExample) -> tuple[list[str], list[str]]:
    """The passages an example's evidence names, and the keys that were skipped.

    Passages come from ``pre_text`` followed by ``post_text`` by index, in document order,
    each text once. Table evidence is not marked in the prompt, so ``table_N`` keys are
    passed over. A key that is neither ``table_N`` nor ``text_N``, or a text index outside
    the passages, is skipped and reported; the example is kept.
    """
    passages = list(example.pre_text or []) + list(example.post_text or [])
    if example.gold_inds is None:
        return [], ["gold_inds"]
    indices: list[int] = []
    skipped: list[str] = []
    for key in example.gold_inds:
        key = str(key)
        match = _TEXT_KEY.fullmatch(key)
        if match and int(match.group(1)) < len(passages):
            indices.append(int(match.group(1)))
        elif match or not _TABLE_KEY.fullmatch(key):  # out of range, or not a known key
            skipped.append(key)
    chosen: list[str] = []
    for index in sorted(set(indices)):
        passage = _one_line(passages[index])
        if passage not in chosen:
            chosen.append(passage)
    return chosen, skipped


def _cell(value: Any) -> str:
    """A table cell, word for word. Only what would break the table is changed."""
    return _one_line(value).replace("|", "\\|")


def format_table(table: Sequence[Sequence[Any]] | None, where: str = "") -> str:
    """The table as GitHub-Flavored Markdown: header row, separator, then the other rows."""
    rows = [list(row) for row in table or []]
    if not rows or not rows[0]:
        raise SerializationError(f"{where or 'an example'} has an empty table")
    lines = ["| " + " | ".join(_cell(cell) for cell in rows[0]) + " |"]
    lines.append("|" + "---|" * len(rows[0]))
    lines += ["| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows[1:]]
    return "\n".join(lines)


def format_answer(exe_ans: Any, where: str = "") -> str:
    """The executed result as it goes on the answer line.

    ``yes`` and ``no`` are copied. A number is copied as it is, except that scientific
    notation is written out as a plain number with the same digits: ``4.39e+07`` becomes
    ``43900000``. Decimal arithmetic is used, so no digit changes.
    """
    text = str(exe_ans).strip() if exe_ans is not None else ""
    if text in YES_NO:
        return text
    try:
        number = Decimal(text)
    except InvalidOperation:
        number = None
    if number is None or not number.is_finite():
        raise SerializationError(
            f"{where or 'an example'} has a result that is neither a number nor yes/no: {text!r}"
        )
    if "e" in text.lower():
        return format(number, "f")
    return text


def format_user(
    passages: Iterable[str],
    table: Sequence[Sequence[Any]] | None,
    question: str,
    prompt: PromptDefinition,
    where: str = "",
) -> str:
    """The user text: the context list (or the "no context" text), the table, the question."""
    lines = [f"{prompt.list_marker}{_one_line(passage)}" for passage in passages]
    context = "\n".join(lines) if lines else prompt.no_context
    return (
        f"{prompt.context_label}\n{context}\n\n"
        f"{prompt.table_label}\n{format_table(table, where)}\n\n"
        f"{prompt.question_label} {_one_line(question)}"
    )


def format_target(program: str, answer: str, prompt: PromptDefinition) -> str:
    """The target text: the program line and the answer line."""
    return f"{prompt.program_label} {program}\n{prompt.answer_label} {answer}"


# --------------------------------------------------------------------------- records


def build_record(example: RawExample, prompt: PromptDefinition) -> CanonicalRecord:
    """Turn one example into its canonical record. The program is copied unchanged."""
    where = f"example {example.id}"
    program = str(example.program or "").strip()
    if not program:
        raise SerializationError(f"{where} has no program")
    passages, _ = text_evidence(example)
    return CanonicalRecord(
        example_id=example.id,
        prompt_id=prompt.prompt_id,
        system=prompt.system,
        user=format_user(passages, example.table, example.question, prompt, where),
        target=format_target(program, format_answer(example.exe_ans, where), prompt),
    )


def serialize_split(
    name: str,
    prompt: PromptDefinition | None = None,
    cfg: SplitConfig | None = None,
    data_cfg: DataConfig | None = None,
) -> list[CanonicalRecord]:
    """The canonical records of ``fit``, ``dev``, or ``quick_dev``, in dataset order.

    Asking for ``test`` raises ``TestLockedError``, before and after the unlock of req_005
    v5.01: this is how training files and token counts are made, and none of them ever
    holds a test row. The evaluation runner reads test examples itself, through its gate.
    """
    if name == "test":
        raise TestLockedError(
            "serialize_split never serves the test split, locked or unlocked: training "
            "files and token counts are made from fit and dev. A test run goes through "
            "run_evaluation and its gate (req_005 v5.01)"
        )
    prompt = prompt or load_prompt()
    return [build_record(example, prompt) for example in load_split(name, cfg, data_cfg)]


def records_digest(records: Iterable[CanonicalRecord]) -> str:
    """SHA-256 over the records in order: one value that changes if any character does."""
    digest = hashlib.sha256()
    for record in records:
        line = json.dumps(
            [record.example_id, record.prompt_id, record.system, record.user, record.target],
            ensure_ascii=True,
        )
        digest.update(line.encode() + b"\n")
    return digest.hexdigest()
