"""The predictions file: one JSON line per example, written as a run goes (req_002 v2.03).

A line holds everything about one reply: the text as the predictor returned it, the outcome
of running it (with the unrounded value), the score, the token counts, and the latency. It
holds no prompt and no gold text: the prompt is the prompt id plus the example id, and the
gold reply comes from the serializer again. So a changed allowance can be applied to a
stored run without asking a model anything.

The file lives in a run's folder under git-ignored ``artifacts/``. It is never committed: a
reply can repeat dataset text.

Usage::

    from qlora_finetuning.evaluation.predictions import append_records, read_predictions

    append_records(path, records)                      # after each batch; forced to disk
    records, dropped = read_predictions(path, ids)     # the valid beginning of the split
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from qlora_finetuning.dsl.outcome import ReplyOutcome
from qlora_finetuning.evaluation.metrics import EvaluationError, ExampleScore
from qlora_finetuning.evaluation.predictor import STOP_REASONS

SCHEMA = 1
PREDICTIONS_NAME = "predictions.jsonl"

_OUTCOME_FIELDS = tuple(field.name for field in fields(ReplyOutcome))
_OUTCOME_TUPLES = ("deviations", "step_values", "notes")
_SCORE_FIELDS = tuple(field.name for field in fields(ExampleScore))


class PredictionsError(EvaluationError):
    """A predictions file is damaged, or is not the beginning of the split it claims."""


def outcome_from_dict(data: dict[str, Any]) -> ReplyOutcome:
    """The ``ReplyOutcome`` that ``to_dict`` wrote."""
    values = dict(data)
    for name in _OUTCOME_TUPLES:
        values[name] = tuple(values.get(name) or ())
    return ReplyOutcome(**values)


def score_from_dict(data: dict[str, Any]) -> ExampleScore:
    """The ``ExampleScore`` that ``to_dict`` wrote."""
    values = dict(data)
    values["deviations"] = tuple(values.get("deviations") or ())
    return ExampleScore(**values)


@dataclass(frozen=True)
class PredictionRecord:
    """One line of the predictions file."""

    example_id: str
    index: int
    prompt_id: str
    run_id: str
    reply: str
    outcome: dict[str, Any]
    score: dict[str, Any]
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_seconds: float = 0.0
    stop_reason: str = "end"
    schema: int = SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> PredictionRecord:
        """Build a record from a parsed line. Raises ``ValueError`` saying what is wrong."""
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
        if data.get("schema") != SCHEMA:
            raise ValueError(f"unknown schema {data.get('schema')!r}")
        known = {field.name for field in fields(cls)}
        if set(data) != known:
            odd = sorted(set(data) ^ known)
            raise ValueError(f"missing or unknown field(s): {', '.join(odd)}")
        for name in ("example_id", "prompt_id", "run_id", "reply"):
            if not isinstance(data[name], str):
                raise ValueError(f"{name} is not text")
        if isinstance(data["index"], bool) or not isinstance(data["index"], int):
            raise ValueError("index is not a whole number")
        if data["stop_reason"] not in STOP_REASONS:
            raise ValueError(f"unknown stop reason {data['stop_reason']!r}")
        for name, expected in (("outcome", _OUTCOME_FIELDS), ("score", _SCORE_FIELDS)):
            part = data[name]
            if not isinstance(part, dict) or set(part) != set(expected):
                raise ValueError(f"{name} does not hold the expected fields")
        return cls(**data)


def render_record(record: PredictionRecord) -> str:
    return json.dumps(
        record.to_dict(), ensure_ascii=True, separators=(",", ":"), allow_nan=False
    )


def append_records(path: Path | str, records: Iterable[PredictionRecord]) -> int:
    """Append one line per record in a single write and force it to disk. Returns the count."""
    lines = [render_record(record) + "\n" for record in records]
    if not lines:
        return 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(lines))
        handle.flush()
        os.fsync(handle.fileno())
    return len(lines)


def read_predictions(
    path: Path | str, expected_ids: Sequence[str]
) -> tuple[list[PredictionRecord], bool]:
    """The records of a predictions file, and whether a cut-off last line was dropped.

    The file must be a clean beginning of ``expected_ids``: its lines name the first ids, in
    order, each once. A last piece that does not end in a newline is a line the writer never
    finished; it is dropped and reported. Anything else that is wrong raises
    ``PredictionsError`` naming the line. A missing file is an empty list.
    """
    path = Path(path)
    if not path.is_file():
        return [], False
    pieces = path.read_text(encoding="utf-8").split("\n")
    dropped = bool(pieces[-1].strip())  # text after the last newline: a cut-off line
    records: list[PredictionRecord] = []
    for number, raw in enumerate(pieces[:-1], start=1):
        where = f"{path.name} line {number}"
        try:
            record = PredictionRecord.from_dict(json.loads(raw))
        except ValueError as error:  # includes a line that is not JSON
            raise PredictionsError(f"{where} is damaged: {error}") from error
        position = len(records)
        if position >= len(expected_ids):
            raise PredictionsError(
                f"{where}: the file has more lines than the {len(expected_ids)} examples asked for"
            )
        if record.example_id != expected_ids[position] or record.index != position:
            raise PredictionsError(
                f"{where} holds {record.example_id} at position {record.index}, where "
                f"{expected_ids[position]} at position {position} was expected; the file is not "
                "a clean beginning of the split"
            )
        records.append(record)
    return records, dropped
