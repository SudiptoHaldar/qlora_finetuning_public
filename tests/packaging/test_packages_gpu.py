"""A card's code with the real model on a GPU (req_006 v6.02). Deselected by default.

Run with ``pytest -m gpu tests/packaging``. Executes the adapter card's usage code in a new
process, in an empty directory, against the adapter package under ``artifacts/release/``.
Skipped when the package or the base model is not on this machine. Nothing is downloaded,
nothing is billed, and no AWS call is made.
"""

from __future__ import annotations

import pytest

from qlora_finetuning.data.serialization import load_prompt
from qlora_finetuning.data.token_budget import load_token_budget_config
from qlora_finetuning.demo.inputs import load_input, sample_path
from qlora_finetuning.evaluation.metrics import load_evaluation_config
from qlora_finetuning.local_qlora.adapters import ADAPTER_WEIGHTS
from qlora_finetuning.local_qlora.config import load_local_qlora_config
from qlora_finetuning.packaging.cards import ADAPTER, usage_code
from qlora_finetuning.packaging.config import load_packaging_config
from qlora_finetuning.packaging.verify import check_card_run, run_card_code

pytestmark = pytest.mark.gpu


def test_the_adapter_cards_code_answers_the_sample():
    from qlora_finetuning.local_qlora.backend import load_tokenizer_from, weights_missing

    cfg, budget = load_packaging_config(), load_token_budget_config()
    if not (cfg.adapter_dir / ADAPTER_WEIGHTS).is_file():
        pytest.skip("there is no adapter package: python -m scripts.hf_package adapter")
    if weights_missing(budget.model_id, budget.revision):
        pytest.skip("the base model is not in the local cache")

    code, problems = usage_code(ADAPTER, cfg, budget, load_local_qlora_config())
    assert problems == []
    result = run_card_code(
        code, {"ADAPTER": str(cfg.adapter_dir)}, timeout=cfg.card_code_timeout_seconds
    )
    found, facts = check_card_run(
        result,
        sample=load_input(sample_path()),
        prompt=load_prompt(),
        tokenizer=load_tokenizer_from(cfg.adapter_dir),
        budget=budget,
        evaluation_config=load_evaluation_config(),
    )
    assert found == [] and facts["program_ran"] and facts["in_the_format"]
