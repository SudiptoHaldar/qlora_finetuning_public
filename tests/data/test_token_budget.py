"""Tests for the token counter, with a stand-in tokenizer and invented records."""

from __future__ import annotations

import dataclasses
import sys

import pytest
import yaml

from qlora_finetuning.data import load_prompt
from qlora_finetuning.data.serialization import build_record
from qlora_finetuning.data.token_budget import (
    TokenBudgetError,
    TokenCounts,
    count_record,
    count_split,
    load_token_budget_config,
    load_tokenizer,
    render_prompt,
    tokenizer_fingerprint,
)
from qlora_finetuning.settings import CONFIG_DIR, REPO_ROOT, ConfigError
from tests.data.synthetic import GOLDEN_ROWS, FakeTokenizer, to_examples

SUFFIX = FakeTokenizer.THINKING_OFF_SUFFIX


@pytest.fixture(scope="module")
def prompt():
    return load_prompt()


@pytest.fixture
def records(prompt):
    return [build_record(item, prompt) for item in to_examples(list(GOLDEN_ROWS.values()))]


def write_config(tmp_path, change):
    raw = yaml.safe_load((CONFIG_DIR / "token_budget.yaml").read_text(encoding="utf-8"))
    change(raw)
    (tmp_path / "token_budget.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_token_budget_config(root=tmp_path, config_dir=tmp_path)


# --------------------------------------------------------------------------- config


def test_real_config_loads():
    cfg = load_token_budget_config()
    assert cfg.model_id == "Qwen/Qwen3-4B"
    assert len(cfg.revision) == 40 and len(cfg.fingerprint) == 64
    assert (cfg.max_length, cfg.max_new_tokens) == (2048, 192)
    assert cfg.thresholds == (768, 1024, 1280, 1536, 2048)
    assert cfg.max_length in cfg.thresholds
    assert cfg.thinking_off_suffix == "<think>\n\n</think>\n\n"
    assert cfg.longest_listed == 10
    assert cfg.report_dir == REPO_ROOT / "reports" / "data"


def test_thresholds_are_sorted_and_distinct(tmp_path):
    cfg = write_config(tmp_path, lambda raw: raw.update(thresholds=[2048, 768, 768, 1024]))
    assert cfg.thresholds == (768, 1024, 2048)
    assert cfg.report_dir == tmp_path / "reports" / "data"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: raw["tokenizer"].update(revision="main"), "full 40-character commit hash"),
        (lambda raw: raw["tokenizer"].update(fingerprint="abc"), "fingerprint must be 64"),
        (lambda raw: raw.update(max_length=0), "max_length must be a positive whole number"),
        (lambda raw: raw.update(max_new_tokens="192"), "max_new_tokens must be a positive whole"),
        (lambda raw: raw.update(max_length=True), "max_length must be a positive whole number"),
        (lambda raw: raw.update(thresholds=[768, -1]), "thresholds must be a list of positive"),
        (lambda raw: raw.update(thresholds=1536), "thresholds must be a list of positive"),
        (lambda raw: raw["tokenizer"].pop("model_id"), "is missing: tokenizer.model_id"),
    ],
)
def test_malformed_config_is_rejected(tmp_path, change, message):
    with pytest.raises(ConfigError, match=message):
        write_config(tmp_path, change)


# --------------------------------------------------------------------------- counting


def test_prompt_is_the_chat_template_with_thinking_off(records):
    prompt_text = render_prompt(records[0], FakeTokenizer())
    assert prompt_text.startswith("<start> system You are a financial calculation assistant.")
    assert f"<stop> <start> user {records[0].user} <stop> <start> assistant " in prompt_text
    assert prompt_text.endswith(SUFFIX)


def test_counts_for_the_golden_records(records, prompt):
    counts = count_split(records, FakeTokenizer(), SUFFIX)
    system = len(prompt.system.split())
    assert [item.example_id for item in counts] == [record.example_id for record in records]
    assert counts[0] == TokenCounts(
        example_id="ACME/2015/page_10.pdf-1",
        system=system,
        user=50,
        target=8,
        prompt=system + 50 + 10,
        thinking_off=True,
    )
    assert [(item.user, item.target) for item in counts] == [(50, 8), (35, 6), (35, 9), (40, 8)]
    assert all(item.system == system and item.template == 10 for item in counts)
    assert all(item.total == item.prompt + item.target for item in counts)


def test_target_is_counted_with_its_end_token(records):
    tokenizer = FakeTokenizer()
    words = len(records[1].target.split())
    assert count_record(records[1], tokenizer, SUFFIX).target == words + 1


def test_one_record_counts_the_same_alone_as_in_a_batch(records):
    tokenizer = FakeTokenizer()
    batch = count_split(records, tokenizer, SUFFIX)
    assert [count_record(record, tokenizer, SUFFIX) for record in records] == batch
    assert count_split([], tokenizer, SUFFIX) == []


def test_thinking_flag_is_false_for_another_suffix_or_a_think_block_in_the_target(records):
    tokenizer = FakeTokenizer()
    assert count_record(records[0], tokenizer, "<think>\n\n</think>\n\n").thinking_off is False
    thinking = dataclasses.replace(records[0], target="<think> hmm </think> PROGRAM: x\nANSWER: 1")
    assert count_record(thinking, tokenizer, SUFFIX).thinking_off is False


# --------------------------------------------------------------------------- tokenizer


def test_fingerprint_is_stable_and_sees_vocabulary_template_and_end_token():
    base = tokenizer_fingerprint(FakeTokenizer())
    assert len(base) == 64
    assert tokenizer_fingerprint(FakeTokenizer({"beta": 1, "alpha": 0})) == base  # order-free
    assert tokenizer_fingerprint(FakeTokenizer({"alpha": 0, "beta": 2})) != base

    other_template = FakeTokenizer()
    other_template.chat_template = "fake template v2"
    assert tokenizer_fingerprint(other_template) != base

    other_end = FakeTokenizer()
    other_end.eos_token = " <eos>"
    assert tokenizer_fingerprint(other_end) != base


def test_load_tokenizer_explains_a_missing_library(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)  # makes the import fail
    with pytest.raises(TokenBudgetError, match="transformers is not installed"):
        load_tokenizer(load_token_budget_config())


def test_load_tokenizer_explains_missing_files(monkeypatch):
    """A tokenizer that is not in the local cache names the command that fetches it."""
    calls = []

    class Loader:
        @staticmethod
        def from_pretrained(model_id, revision, local_files_only):
            calls.append((model_id, revision, local_files_only))
            raise OSError("not found in the cache\nsecond line that must not be shown")

    monkeypatch.setitem(sys.modules, "transformers", type("M", (), {"AutoTokenizer": Loader}))
    cfg = load_token_budget_config()
    with pytest.raises(TokenBudgetError) as caught:
        load_tokenizer(cfg)
    message = str(caught.value)
    assert "python -m scripts.finqa_tokens --download" in message
    assert "OSError: not found in the cache" in message and "second line" not in message
    assert calls == [(cfg.model_id, cfg.revision, True)]

    with pytest.raises(TokenBudgetError, match="check the network connection"):
        load_tokenizer(cfg, download=True)
    assert calls[-1][2] is False


def test_token_budget_error_is_a_config_error():
    assert issubclass(TokenBudgetError, ConfigError)
