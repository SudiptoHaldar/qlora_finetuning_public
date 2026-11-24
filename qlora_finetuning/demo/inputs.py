"""What the demo is asked: a question, a table, and optional context, from a JSON file.

An input file is one JSON object::

    {
      "context": ["One passage of the report.", "Another."],
      "table": [["", "2024", "2023"], ["revenue", "$ 1,150", "$ 920"]],
      "question": "what was the percentage change in revenue from 2023 to 2024?"
    }

``table`` is a list of rows, the first row the header; a cell is text or a number.
``context`` may be empty or left out. ``expected`` may hold the result a correct program
gives; the demo does not use it (the sample carries one for the acceptance check).

Everything else is refused before a model is loaded, with the file and the field named:
a missing or empty table, a missing question, a value of the wrong type, an unknown field.
The demo never reads the dataset: an input is a file somebody wrote.

Usage::

    from qlora_finetuning.demo.inputs import load_input, sample_path

    demo_input = load_input(sample_path())
    demo_input.question, demo_input.table, demo_input.passages
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qlora_finetuning.settings import ConfigError

SAMPLE_NAME = "sample_input.json"
FIELDS = ("question", "table", "context", "expected")
YES_NO = ("yes", "no")


class DemoInputError(ConfigError):
    """An input file cannot be used."""


@dataclass(frozen=True)
class DemoInput:
    """One question for the model. ``table`` holds text cells, as the prompt shows them."""

    source: str  # the file's name, for messages and for the output
    question: str
    table: tuple[tuple[str, ...], ...]
    passages: tuple[str, ...] = ()
    expected: float | str | None = None  # not used by the demo itself


def sample_path() -> Path:
    """The bundled sample: an invented company and invented figures, not a dataset row."""
    return Path(__file__).with_name(SAMPLE_NAME)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    )


def _table(raw: Any, where: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(raw, list) or not raw:
        raise DemoInputError(f"{where}: table must be a list of rows, the first one the header")
    rows = []
    for number, row in enumerate(raw):
        if not isinstance(row, list) or not row:
            raise DemoInputError(f"{where}: table row {number} must be a list of cells")
        for cell in row:
            if not isinstance(cell, str) and not _is_number(cell):
                raise DemoInputError(
                    f"{where}: table row {number} has a cell that is neither text nor a number"
                )
        rows.append(tuple(str(cell) for cell in row))
    return tuple(rows)


def _passages(raw: Any, where: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise DemoInputError(f"{where}: context must be a list of texts")
    return tuple(item for item in raw if item.strip())


def _expected(raw: Any, where: str) -> float | str | None:
    if raw is None:
        return None
    if isinstance(raw, str) and raw in YES_NO:
        return raw
    if _is_number(raw):
        return float(raw)
    raise DemoInputError(f"{where}: expected must be a number, or yes or no")


def read_input(raw: Any, source: str) -> DemoInput:
    """Check one decoded JSON value and return it as a ``DemoInput``."""
    where = f"input {source}"
    if not isinstance(raw, dict):
        raise DemoInputError(f"{where}: the file must hold one JSON object")
    unknown = sorted(str(key) for key in raw if key not in FIELDS)
    if unknown:
        raise DemoInputError(
            f"{where}: unknown field {', '.join(unknown)} (the fields are {', '.join(FIELDS)})"
        )
    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        raise DemoInputError(f"{where}: question must be a text that is not empty")
    if "table" not in raw:
        raise DemoInputError(f"{where}: table is missing; the model is always given a table")
    return DemoInput(
        source=source,
        question=question,
        table=_table(raw["table"], where),
        passages=_passages(raw.get("context"), where),
        expected=_expected(raw.get("expected"), where),
    )


def load_input(path: Path | str) -> DemoInput:
    """Read and check an input file. Raises ``DemoInputError``, naming the file."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise DemoInputError(
            f"input {path.name}: the file cannot be read ({type(error).__name__})"
        ) from None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise DemoInputError(
            f"input {path.name}: not JSON (line {error.lineno}, column {error.colno})"
        ) from None
    return read_input(raw, path.name)
