"""Tests for qlora_finetuning.local_qlora.backend that need no GPU and no training stack.

What must hold before a model is loaded, with every outside piece replaced by a stand-in.
The model itself is loaded only in ``test_backend_gpu.py``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from qlora_finetuning.local_qlora import backend
from qlora_finetuning.local_qlora.backend import (
    adapter_sha256,
    check_ready,
    download_command,
    weights_missing,
)
from qlora_finetuning.settings import ConfigError
from tests.data.synthetic import FakeTokenizer, fake_budget

from .conftest import PieceTokenizer


def stack(cuda=True):
    return SimpleNamespace(
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda)),
        transformers=None,
        peft=None,
    )


@pytest.fixture
def ready(monkeypatch, tmp_path):
    """Everything in place: the stack, the weights, the pinned tokenizer, a GPU."""
    monkeypatch.setattr(backend, "_stack", stack)
    monkeypatch.setattr(backend, "weights_missing", lambda model_id, revision: [])
    monkeypatch.setattr(backend, "_load_tokenizer", lambda *arguments: FakeTokenizer())
    return fake_budget(tmp_path)


def test_ready_returns_the_stack_and_the_tokenizer(ready):
    found, tokenizer = check_ready(ready)
    assert found.torch.cuda.is_available() and isinstance(tokenizer, FakeTokenizer)


def test_without_the_training_stack(ready, monkeypatch):
    monkeypatch.undo()
    for name in ("bitsandbytes", "peft", "torch", "transformers"):
        monkeypatch.setitem(sys.modules, name, None)  # importing it now fails
    with pytest.raises(ConfigError, match="training stack is not installed .*runbook"):
        check_ready(ready)


def test_a_training_stack_that_breaks_while_loading(ready, monkeypatch):
    """The first SageMaker smoke job: a library was there and failed to load, with no name."""
    monkeypatch.undo()
    for name in ("bitsandbytes", "peft", "torch", "transformers"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    class Broken:
        def find_spec(self, name, path=None, target=None):
            if name == "bitsandbytes":
                try:
                    raise RuntimeError("operator torchvision::nms does not exist")
                except RuntimeError as cause:
                    message = "Could not import module 'PreTrainedModel'"
                    raise ModuleNotFoundError(message) from cause
            return None

    monkeypatch.setattr(sys, "meta_path", [Broken(), *sys.meta_path])
    with pytest.raises(ConfigError) as caught:
        check_ready(ready)
    message = str(caught.value)
    assert "installed and cannot be loaded" in message and "None is missing" not in message
    assert "Could not import module 'PreTrainedModel'" in message
    assert "caused by RuntimeError: operator torchvision::nms does not exist" in message


def test_weights_that_were_never_downloaded(ready, monkeypatch):
    shards = [f"model-0000{n}-of-00004.safetensors" for n in (1, 2, 3, 4)]
    monkeypatch.setattr(backend, "weights_missing", lambda model_id, revision: shards)
    with pytest.raises(ConfigError) as caught:
        check_ready(ready)
    message = str(caught.value)
    assert "are not in the local Hugging Face cache" in message
    assert "model-00003-of-00004.safetensors, ..." in message  # three named, then the rest
    assert download_command(ready.model_id, ready.revision) in message
    assert message.endswith(f"hf download Qwen/Qwen3-4B --revision {ready.revision}")


def test_a_tokenizer_that_cannot_be_loaded(ready, monkeypatch):
    def broken(*arguments):
        raise OSError("tokenizer.json is not there\nsecond line")

    monkeypatch.setattr(backend, "_load_tokenizer", broken)
    with pytest.raises(ConfigError, match="OSError: tokenizer.json is not there.; download"):
        check_ready(ready)


def test_a_tokenizer_that_is_not_the_pinned_one(ready, monkeypatch):
    other = FakeTokenizer(vocabulary={"gamma": 0})
    monkeypatch.setattr(backend, "_load_tokenizer", lambda *arguments: other)
    with pytest.raises(ConfigError, match="is not the one pinned in config/token_budget.yaml"):
        check_ready(ready)


def test_without_a_gpu(ready, monkeypatch):
    monkeypatch.setattr(backend, "_stack", lambda: stack(cuda=False))
    with pytest.raises(ConfigError, match="no CUDA GPU"):
        check_ready(ready)


def test_another_model_and_commit_can_be_checked(ready, monkeypatch):
    seen = []
    monkeypatch.setattr(backend, "weights_missing", lambda *arguments: seen.append(arguments) or [])
    check_ready(ready, "Qwen/Qwen3-0.6B", "c" * 40)
    assert seen == [("Qwen/Qwen3-0.6B", "c" * 40)]


# --------------------------------------------------------------------------- the cache check


def fake_hub(monkeypatch, tmp_path, cached):
    """A stand-in ``huggingface_hub`` whose cache holds the files named in ``cached``."""

    def try_to_load_from_cache(repo_id, filename, revision=None):
        if filename not in cached:
            return None
        path = tmp_path / filename
        path.write_text(cached[filename] or "", encoding="utf-8")
        return str(path)

    module = SimpleNamespace(try_to_load_from_cache=try_to_load_from_cache)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


INDEX = json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors",
                                   "c": "model-1.safetensors"}})  # fmt: skip


@pytest.mark.parametrize(
    ("cached", "missing"),
    [
        ({}, ["config.json"]),
        ({"config.json": ""}, ["model.safetensors"]),
        ({"config.json": "", "model.safetensors": ""}, []),
        ({"config.json": "", "model.safetensors.index.json": INDEX},
         ["model-1.safetensors", "model-2.safetensors"]),
        ({"config.json": "", "model.safetensors.index.json": INDEX, "model-1.safetensors": ""},
         ["model-2.safetensors"]),
        ({"config.json": "", "model.safetensors.index.json": INDEX, "model-1.safetensors": "",
          "model-2.safetensors": ""}, []),
    ],
)  # fmt: skip
def test_which_weight_files_are_missing_from_the_cache(monkeypatch, tmp_path, cached, missing):
    fake_hub(monkeypatch, tmp_path, cached)
    assert weights_missing("Qwen/Qwen3-4B", "a" * 40) == missing


# --------------------------------------------------------------------------- adapters, end tokens


def test_an_adapter_is_named_by_the_hash_of_its_weights(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    assert adapter_sha256(tmp_path) == hashlib.sha256(b"weights").hexdigest()
    with pytest.raises(ConfigError, match="no adapter weights"):
        adapter_sha256(tmp_path / "elsewhere")


def test_end_tokens_must_be_tokens_of_the_tokenizer():
    tokenizer = PieceTokenizer()
    assert backend._end_ids(tokenizer, ["<|im_end|>", "<|endoftext|>"]) == [90, 91]
    with pytest.raises(ConfigError, match="'<stop>', which is not a token"):
        backend._end_ids(tokenizer, ["<stop>"])
