"""Stand-ins for the demo tests: a predictor without a model, and invented input.

Nothing here is a dataset row: the company and every figure are made up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qlora_finetuning.evaluation.predictor import Prediction

TABLE = [
    ["", "2024", "2023"],
    ["revenue", "$ 1,150", "$ 920"],
    ["operating income", "$ 138", "$ 101"],
]
QUESTION = "what was the percentage change in revenue from 2023 to 2024?"
PASSAGES = ["Harbour Mills is an invented company.", "Revenue rose in 2024."]
RIGHT_REPLY = "PROGRAM: subtract(1150, 920), divide(#0, 920)\nANSWER: 0.25"
RUN = "20261119T101500Z-train-qwen3-4b-fit-1a2b"


def raw_input(**changes: Any) -> dict[str, Any]:
    """An input file's content; a change of ``None`` removes the field."""
    raw: dict[str, Any] = {"context": list(PASSAGES), "table": TABLE, "question": QUESTION}
    raw.update(changes)
    return {key: value for key, value in raw.items() if value is not None}


def write_input(folder: Path, name: str = "question.json", **changes: Any) -> Path:
    path = folder / name
    path.write_text(json.dumps(raw_input(**changes)), encoding="utf-8")
    return path


class StubPredictor:
    """Answers every request with ``reply``; ``requests`` lists the batches it was given."""

    model_id = "stub/model"
    batch_size = 8
    usage_source = "measured"

    def __init__(
        self,
        reply: str = RIGHT_REPLY,
        *,
        name: str = "qwen3-4b-tuned",
        details: dict[str, Any] | None = None,
        input_tokens: int | None = 400,
        output_tokens: int | None = 20,
        stop_reason: str = "end",
        seconds: float = 1.25,
        error: Exception | None = None,
    ) -> None:
        self.reply = reply
        self.name = name
        self.details = details
        self.settings: dict[str, Any] = {"batch_size": 8}
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.stop_reason = stop_reason
        self.seconds = seconds
        self.error = error
        self.requests: list[list[Any]] = []

    def predict(self, requests: Any) -> list[Prediction]:
        self.requests.append(list(requests))
        if self.error is not None:
            raise self.error
        return [
            Prediction(
                text=self.reply,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                latency_seconds=self.seconds,
                stop_reason=self.stop_reason,
            )
            for _ in requests
        ]


def tuned_stub(**options: Any) -> StubPredictor:
    """A stub that says which adapter it carries, as the tuned predictor does."""
    details = {"training_run": RUN, "checkpoint": 500, "steps": 500, "smoke": False}
    return StubPredictor(details=details, **options)
