"""Tests for qlora_finetuning.local_qlora.config."""

from __future__ import annotations

import pytest
import yaml

from qlora_finetuning.local_qlora.config import load_local_qlora_config
from qlora_finetuning.settings import CONFIG_DIR, ConfigError


def write_config(tmp_path, change=None):
    raw = yaml.safe_load((CONFIG_DIR / "local_qlora.yaml").read_text(encoding="utf-8"))
    if change:
        change(raw)
    (tmp_path / "local_qlora.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    return tmp_path


def test_the_committed_file_describes_greedy_decoding_in_batches():
    cfg = load_local_qlora_config()
    assert cfg.predictor_name == "qwen3-4b"
    assert (cfg.quantization_type, cfg.double_quant, cfg.compute_dtype) == (
        "nf4",
        True,
        "bfloat16",
    )
    assert cfg.batch_size >= 1
    assert cfg.end_tokens == ("<|im_end|>", "<|endoftext|>")


def test_settings_hold_everything_that_can_change_a_reply():
    cfg = load_local_qlora_config()
    assert cfg.settings("abc") == {
        "revision": "abc",
        "quantization": {"type": "nf4", "double_quant": True, "compute_dtype": "bfloat16"},
        "enable_thinking": False,
        "do_sample": False,
        "num_beams": 1,
        "stop": "end tokens",
        "end_tokens": ["<|im_end|>", "<|endoftext|>"],
        "batch_size": cfg.batch_size,
        "adapter": None,
    }
    assert cfg.settings("abc", adapter="f" * 64)["adapter"] == "f" * 64


def test_the_model_is_not_repeated_here():
    raw = yaml.safe_load((CONFIG_DIR / "local_qlora.yaml").read_text(encoding="utf-8"))
    assert "model_id" not in raw and "revision" not in raw  # token_budget.yaml pins them


@pytest.mark.parametrize(
    "key",
    [
        "predictor_name",
        "quantization.type",
        "quantization.double_quant",
        "quantization.compute_dtype",
        "inference.batch_size",
        "inference.do_sample",
        "inference.num_beams",
        "inference.end_tokens",
    ],
)
def test_a_missing_key_is_named(tmp_path, key):
    def drop(raw):
        *parents, last = key.split(".")
        for parent in parents:
            raw = raw[parent]
        del raw[last]

    with pytest.raises(ConfigError, match=key.replace(".", r"\.")):
        load_local_qlora_config(write_config(tmp_path, drop))


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("inference", "batch_size", 0, "batch_size"),
        ("inference", "batch_size", 2.5, "batch_size"),
        ("inference", "batch_size", True, "batch_size"),
        ("inference", "do_sample", True, "do_sample must be false"),
        ("inference", "num_beams", 2, "num_beams must be 1"),
        ("inference", "end_tokens", "<|im_end|>", "end_tokens"),
        ("inference", "end_tokens", [7], "end_tokens"),
        ("quantization", "type", "int8", "quantization.type"),
        ("quantization", "compute_dtype", "float64", "compute_dtype"),
        ("quantization", "double_quant", "yes", "double_quant"),
    ],
)
def test_a_value_that_is_not_greedy_4_bit_decoding_is_refused(
    tmp_path, section, key, value, message
):
    def change(raw):
        raw[section][key] = value

    with pytest.raises(ConfigError, match=message):
        load_local_qlora_config(write_config(tmp_path, change))
