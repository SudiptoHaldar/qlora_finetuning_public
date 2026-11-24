"""The merge, on a tiny invented model on the CPU (req_006 v6.02).

Needs the training stack and no GPU, no network, and no cached model: the model is built
from a config with a few thousand random weights. Where the stack is not installed, as in
CI, the tests are skipped.
"""

from __future__ import annotations

import json

import pytest

from qlora_finetuning.settings import ConfigError

RANK, ALPHA = 4, 8


@pytest.fixture
def stack():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    pytest.importorskip("safetensors")
    return torch, transformers, peft


@pytest.fixture
def tiny(stack, tmp_path):
    """A tiny Qwen3-shaped model saved as bfloat16, and a LoRA adapter with non-zero weights."""
    torch, transformers, peft = stack
    torch.manual_seed(0)
    config = transformers.AutoConfig.for_model(
        "qwen3", vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64, head_dim=8,
    )  # fmt: skip
    base_dir, adapter_dir = tmp_path / "base", tmp_path / "adapter"
    transformers.AutoModelForCausalLM.from_config(config).to(torch.bfloat16).save_pretrained(
        base_dir, safe_serialization=True
    )
    tuned = peft.get_peft_model(
        transformers.AutoModelForCausalLM.from_pretrained(base_dir, local_files_only=True),
        peft.LoraConfig(
            r=RANK, lora_alpha=ALPHA, lora_dropout=0.0, target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),  # fmt: skip
    )
    for name, parameter in tuned.named_parameters():
        if "lora_B" in name:  # PEFT starts B at zero: an adapter that changes nothing
            torch.nn.init.normal_(parameter, std=0.05)
    tuned.save_pretrained(adapter_dir)
    return base_dir, adapter_dir


def test_the_merged_weights_are_base_plus_the_adapters_product(stack, tiny, tmp_path):
    from safetensors.torch import load_file

    from qlora_finetuning.packaging.merge import merge_adapter

    torch, transformers, _ = stack
    base_dir, adapter_dir = tiny
    out = tmp_path / "merged"
    result = merge_adapter(base_dir, adapter_dir, out, end_ids=[2, 1], pad_token_id=0)

    assert result.saved_dtype == "bfloat16" and result.compute_dtype == "float32"
    assert result.adapted_tensors == 14  # 2 layers x 7 linear layers
    assert result.changed_tensor.endswith(".weight") and set(result.libraries) == {
        "torch", "transformers", "peft",
    }  # fmt: skip

    base = load_file(base_dir / "model.safetensors")
    merged = load_file(out / "model.safetensors")
    adapter = load_file(adapter_dir / "adapter_model.safetensors")
    assert set(merged) == set(base) and all(v.dtype == torch.bfloat16 for v in merged.values())
    changed = 0
    for key, weight in adapter.items():
        if ".lora_A." not in key:
            continue
        target = key.split(".lora_A.")[0].removeprefix("base_model.model.") + ".weight"
        product = adapter[key.replace(".lora_A.", ".lora_B.")].float() @ weight.float()
        expected = (base[target].float() + (ALPHA / RANK) * product).to(torch.bfloat16)
        assert torch.equal(merged[target], expected), target  # the sum, then rounded once
        assert not torch.equal(merged[target], base[target])
        changed += 1
    assert changed == 14
    untouched = [key for key in base if key.split(".")[-2] in ("embed_tokens", "norm")]
    assert untouched and all(torch.equal(merged[key], base[key]) for key in untouched)

    generation = json.loads((out / "generation_config.json").read_text(encoding="utf-8"))
    assert generation["do_sample"] is False and generation["eos_token_id"] == [2, 1]
    assert generation["pad_token_id"] == 0 and "temperature" not in generation
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["dtype"] == "bfloat16" and str(tmp_path) not in json.dumps(config)

    again = transformers.AutoModelForCausalLM.from_pretrained(out, local_files_only=True)
    assert not [name for name, _ in again.named_parameters() if "lora" in name]
    assert again.generation_config.do_sample is False


def test_a_folder_without_lora_weights_or_a_missing_base_is_a_config_error(stack, tiny, tmp_path):
    from safetensors.torch import save_file

    from qlora_finetuning.packaging.merge import merge_adapter

    torch, _, _ = stack
    base_dir, adapter_dir = tiny
    out = tmp_path / "out"
    with pytest.raises(ConfigError, match="could not be loaded for the merge"):
        merge_adapter(tmp_path / "absent", adapter_dir, out, end_ids=[1], pad_token_id=0)
    save_file({"x": torch.zeros(1)}, adapter_dir / "adapter_model.safetensors")
    with pytest.raises(ConfigError, match="holds no LoRA weights"):
        merge_adapter(base_dir, adapter_dir, tmp_path / "out", end_ids=[1], pad_token_id=0)
    assert not (tmp_path / "out").exists()  # nothing is written by a merge that was refused
