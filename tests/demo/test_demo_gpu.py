"""The demo with the real model on a GPU (req_006 v6.01). Deselected by default: ``pytest -m gpu``.

Loads ``Qwen/Qwen3-4B`` in 4-bit with the selected adapter and asks the invented sample.
Skipped when the selected checkpoint or the base model is not on this machine. Nothing is
downloaded, nothing is billed, and no AWS call is made.
"""

from __future__ import annotations

import json

import pytest

from qlora_finetuning.data.token_budget import load_token_budget_config
from qlora_finetuning.demo.cli import demo_context, main, require_selected_here
from qlora_finetuning.demo.inputs import load_input, sample_path
from qlora_finetuning.evaluation.metrics import load_evaluation_config, values_match
from qlora_finetuning.settings import ConfigError

pytestmark = pytest.mark.gpu


def test_the_selected_adapter_answers_the_sample(capsys):
    from qlora_finetuning.local_qlora.backend import weights_missing

    budget = load_token_budget_config()
    try:
        require_selected_here(demo_context())
    except ConfigError as error:
        pytest.skip(str(error))
    if weights_missing(budget.model_id, budget.revision):
        pytest.skip("the base model is not in the local cache")

    code = main(["--json"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert code == 0, lines
    data = json.loads(lines[-1])
    sample = load_input(sample_path())
    assert data["input"] == "sample_input.json" and data["predictor"] == "qwen3-4b-tuned"
    assert data["outcome"]["error_code"] is None and data["outcome"]["strict"] is True
    assert values_match(
        data["outcome"]["value"], sample.expected, load_evaluation_config().execution
    )
    assert data["agrees"] is True
    assert data["stop_reason"] == "end" and data["warnings"] == []
