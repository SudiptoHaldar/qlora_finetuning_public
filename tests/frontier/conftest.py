"""Shared pieces for the Path B tests: a stub provider and baseline runs on an invented split.

Nothing here creates an AWS client or needs a sign-in. The runs go through the real
evaluation runner on the invented split of ``tests/evaluation/conftest.py`` (dev 8,
quick-dev 3), with everything written under ``tmp_path``.
"""

from __future__ import annotations

import pytest

from qlora_finetuning.evaluation.predictor import PredictorError
from qlora_finetuning.frontier.config import load_frontier_config
from qlora_finetuning.frontier.predictor import FrontierPredictor
from qlora_finetuning.frontier.provider import ProviderReply
from tests.evaluation.conftest import GOLD_REPLY, evaluate, make_context

INPUT_TOKENS, OUTPUT_TOKENS = 40, 12
QWEN_PROMPT_TOKENS = {"quick_dev": 30, "dev": 80}  # invented, for the token ratio


class StubProvider:
    """Answers every request with the invented split's gold reply and fixed token counts."""

    def __init__(self, text: str = GOLD_REPLY, fail_after: int | None = None) -> None:
        self.text = text
        self.fail_after = fail_after
        self.calls: list[tuple[str, str, str, int]] = []

    def converse(self, model_id, system, user, max_tokens):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise PredictorError("Bedrock answered ThrottlingException for the stub")
        self.calls.append((model_id, system, user, max_tokens))
        return ProviderReply(
            text=self.text,
            input_tokens=INPUT_TOKENS,
            output_tokens=OUTPUT_TOKENS,
            stop_reason="end",
            latency_seconds=0.25,
        )


def make_predictor(cfg, name: str, provider=None, **changes) -> FrontierPredictor:
    """The predictor of a configured model over a stub provider."""
    model = cfg.model(name)
    return FrontierPredictor(
        provider or StubProvider(),
        changes.get("model_id", model.model_id),
        name,
        changes.get("settings", cfg.settings),
    )


def baseline_run(context, cfg, name: str, split: str, **options) -> dict:
    """One baseline run as the script records it: path B, the model's untuned arm, Bedrock."""
    predictor = options.pop("predictor", None) or make_predictor(cfg, name)
    settings = {
        "label": f"{name} {split}",
        "path": "B",
        "arm": cfg.model(name).untuned_arm,
        "where": "bedrock",
        "kind": "eval",
    }
    settings.update(options)
    return evaluate(predictor, split, context, **settings)


@pytest.fixture
def context(tmp_path):
    return make_context(tmp_path)


@pytest.fixture
def cfg(tmp_path):
    """The committed frontier config, with its report folder under ``tmp_path``."""
    return load_frontier_config(root=tmp_path)


@pytest.fixture
def four_runs(context, cfg) -> list[dict]:
    """Each configured model on quick-dev and on dev: what the report expects."""
    return [
        baseline_run(context, cfg, name, split)
        for split in ("quick_dev", "dev")
        for name in cfg.models
    ]
