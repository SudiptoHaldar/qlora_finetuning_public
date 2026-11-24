"""Stand-ins for the Path A tests: a tokenizer that decodes, and a backend without a GPU."""

from __future__ import annotations

from typing import Any

import pytest

from qlora_finetuning.local_qlora.backend import Generated
from tests.evaluation.conftest import GOLD_REPLY, make_context

MODEL_ID = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
IM_END, END_OF_TEXT, IM_START = 90, 91, 92
END_IDS = frozenset({IM_END, END_OF_TEXT})


@pytest.fixture
def context(tmp_path):
    """The invented split and a runner context, all under the temporary folder."""
    return make_context(tmp_path)


class PieceTokenizer:
    """Decodes ids to fixed pieces of text. Three ids are special tokens."""

    PIECES = {
        0: "PROGRAM: add(1, 2)",
        1: "\n",
        2: "ANSWER: 3",
        3: "<think>",  # an ordinary token for Qwen3: decoding keeps it
        4: "</think>",
        5: "  café € ",
        6: "thoughts",
        IM_END: "<|im_end|>",
        END_OF_TEXT: "<|endoftext|>",
        IM_START: "<|im_start|>",
    }
    SPECIAL = {IM_END, END_OF_TEXT, IM_START}

    def decode(self, token_ids: Any, skip_special_tokens: bool = False) -> str:
        return "".join(
            self.PIECES[token]
            for token in token_ids
            if not (skip_special_tokens and token in self.SPECIAL)
        )

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return next((key for key, piece in self.PIECES.items() if piece == token), None)

    def convert_ids_to_tokens(self, token_id: int) -> str | None:
        return self.PIECES.get(token_id)


class OutOfMemoryError(RuntimeError):
    """Named like torch's, which the predictor recognizes by name."""


class StubBackend:
    """Stands in for ``QwenBackend``: answers every prompt with ``reply``.

    ``fail_after=N`` answers N batches and then raises ``error``. ``calls`` lists the
    batches of ``(system, user)`` pairs it was given.
    """

    model_id = MODEL_ID

    def __init__(
        self,
        reply: str = GOLD_REPLY,
        *,
        settings: dict[str, Any] | None = None,
        fail_after: int | None = None,
        error: Exception | None = None,
        batch_seconds: float = 2.0,
        notes: Any = None,
    ) -> None:
        self.reply = reply
        self.settings = settings or {}
        self.fail_after = fail_after
        self.error = error or RuntimeError("CUDA error: device-side assert")
        self.batch_seconds = batch_seconds
        self.notes = notes
        self.calls: list[list[tuple[str, str]]] = []
        self.limits: list[int] = []

    def generate(self, prompts, max_new_tokens):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise self.error
        self.calls.append(list(prompts))
        self.limits.append(max_new_tokens)
        return [
            Generated(
                text=self.reply,
                input_tokens=len(system.split()) + len(user.split()),
                output_tokens=len(self.reply.split()) + 1,
                stop_reason="end",
                batch_seconds=self.batch_seconds,
                batch_size=len(prompts),
            )
            for system, user in prompts
        ]

    def measurements(self):
        if self.notes is not None:
            return self.notes
        return {"peak_vram_allocated_gib": 3.5, "batches": len(self.calls), "gpu_name": "stub"}
