"""The real backend on a GPU (req_003 v3.01). Deselected by default: ``pytest -m gpu``.

Most tests load the small smoke model of ``config/local_env.yaml`` (same tokenizer and chat
template as the 4B model, a few seconds to load). The last one loads ``Qwen/Qwen3-4B`` and is
skipped when its weights are not in the cache. Prompts are invented; nothing is downloaded.
"""

from __future__ import annotations

import gc

import pytest

from qlora_finetuning.data.token_budget import load_token_budget_config
from qlora_finetuning.local_qlora.backend import (
    adapter_sha256,
    load_backend,
    weights_missing,
)
from qlora_finetuning.local_qlora.config import load_local_qlora_config
from qlora_finetuning.local_qlora.render import render_chat
from qlora_finetuning.settings import load_config

pytestmark = pytest.mark.gpu

SYSTEM = "Answer in one short line."
PROMPTS = [
    (SYSTEM, "What is 2 plus 2?"),
    (SYSTEM, "Name the capital of France. " + "Take your time. " * 40),
    (SYSTEM, "Spell the word cat, then the word dog, then the word bird."),
]
LIMIT = 24


def smoke_model() -> tuple[str, str]:
    smoke = load_config("local_env", required=["smoke.model_id", "smoke.revision"])["smoke"]
    return smoke["model_id"], smoke["revision"]


def release(backend) -> None:
    import torch

    backend.model = None
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def settings():
    return load_local_qlora_config(), load_token_budget_config()


@pytest.fixture(scope="module")
def small(settings):
    cfg, budget = settings
    model_id, revision = smoke_model()
    backend = load_backend(cfg, budget, model_id=model_id, revision=revision)
    yield backend
    release(backend)


def test_a_batch_of_prompts_of_different_lengths_generates(small, settings):
    _, budget = settings
    replies = small.generate(PROMPTS, LIMIT)
    assert len(replies) == 3
    for (system, user), reply in zip(PROMPTS, replies, strict=True):
        text = render_chat(system, user, small.tokenizer, budget.thinking_off_suffix)
        own_length = len(small.tokenizer(text, add_special_tokens=False)["input_ids"])
        assert reply.input_tokens == own_length  # not the padded width of the batch
        assert 1 <= reply.output_tokens <= LIMIT
        assert reply.stop_reason in ("end", "length")
        assert (reply.stop_reason == "length") == (reply.output_tokens == LIMIT) or (
            reply.stop_reason == "end"
        )
        assert "<think>" not in reply.text and "<|im_end|>" not in reply.text
        assert reply.batch_size == 3 and reply.batch_seconds > 0
    assert len({reply.input_tokens for reply in replies}) == 3


def test_decoding_is_greedy_and_repeats(small):
    config = small.model.generation_config
    assert config.do_sample is False and config.num_beams == 1
    assert config.temperature in (None, 1.0) and config.top_p in (None, 1.0)
    first = [reply.text for reply in small.generate(PROMPTS, LIMIT)]
    again = [reply.text for reply in small.generate(PROMPTS, LIMIT)]
    assert first == again


def test_the_tokenizer_pads_on_the_left_and_the_measurements_are_filled(small):
    assert small.tokenizer.padding_side == "left"
    small.generate(PROMPTS[:1], LIMIT)
    notes = small.measurements()
    assert notes["batches"] >= 1 and notes["prompts"] >= 1
    assert 0 < notes["weights_footprint_gib"] < 2
    assert notes["peak_vram_allocated_gib"] >= notes["weights_footprint_gib"]
    assert notes["peak_vram_reserved_gib"] <= notes["gpu_total_gib"]
    assert notes["generated_tokens"] > 0 and notes["generated_tokens_per_second"] > 0
    assert notes["think_blocks_removed"] == 0 and notes["load_seconds"] > 0
    assert all(not isinstance(value, str) or value == notes["gpu_name"] for value in notes.values())


def make_adapter(folder, base_model, trained: bool):
    """Save a LoRA adapter for the smoke model: untrained (zero), or with random weights."""
    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    wrapped = get_peft_model(
        base_model,
        LoraConfig(
            r=8,
            lora_alpha=64,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            init_lora_weights=not trained,  # False fills both matrices with random values
        ),
    )
    wrapped.save_pretrained(str(folder))
    return wrapped.unload()  # the base model again, without the adapter layers


def test_an_adapter_is_attached_and_named_by_its_hash(small, settings, tmp_path):
    cfg, budget = settings
    model_id, revision = smoke_model()
    base = [reply.text for reply in small.generate(PROMPTS, LIMIT)]
    assert small.settings["adapter"] is None

    small.model = make_adapter(tmp_path / "zero", small.model, trained=False)
    small.model = make_adapter(tmp_path / "random", small.model, trained=True)

    zero = load_backend(cfg, budget, model_id, revision, adapter_dir=tmp_path / "zero")
    try:
        assert zero.settings["adapter"] == adapter_sha256(tmp_path / "zero")
        # An untrained adapter adds nothing, so the replies are the base model's.
        assert [reply.text for reply in zero.generate(PROMPTS, LIMIT)] == base
    finally:
        release(zero)

    random = load_backend(cfg, budget, model_id, revision, adapter_dir=tmp_path / "random")
    try:
        assert random.settings["adapter"] == adapter_sha256(tmp_path / "random")
        assert random.settings["adapter"] != zero.settings["adapter"]
        # Random adapter weights change what the model writes: the adapter is in use.
        assert [reply.text for reply in random.generate(PROMPTS, LIMIT)] != base
    finally:
        release(random)


def test_the_4b_model_loads_and_answers(settings):
    cfg, budget = settings
    if weights_missing(budget.model_id, budget.revision):
        pytest.skip("the Qwen3-4B weights are not in the cache (runbook section 5)")
    backend = load_backend(cfg, budget)
    try:
        assert backend.model_id == "Qwen/Qwen3-4B"
        assert backend.settings["revision"] == budget.revision
        (reply,) = backend.generate(PROMPTS[:1], budget.max_new_tokens)
        assert reply.stop_reason in ("end", "length") and "<think>" not in reply.text
        notes = backend.measurements()
        assert 1 < notes["weights_footprint_gib"] < 8
        assert notes["peak_vram_allocated_gib"] < notes["gpu_total_gib"]
    finally:
        release(backend)
