"""Tests for the training encoder of qlora_finetuning.local_qlora.render (req_003 v3.02)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from qlora_finetuning.data import build_record, load_prompt, load_split
from qlora_finetuning.data.token_budget import (
    TokenBudgetError,
    count_split,
    load_token_budget_config,
    load_tokenizer,
)
from qlora_finetuning.local_qlora.render import (
    IGNORE,
    LengthError,
    encode_example,
    encode_records,
    render_chat,
)
from tests.data.synthetic import FakeTokenizer, build_frozen_split


class WordTokenizer(FakeTokenizer):
    """One id per distinct word, so that ids can be told apart."""

    def __init__(self):
        super().__init__()
        self.words: dict[str, int] = {}

    def _ids(self, text: str) -> list[int]:
        return [self.words.setdefault(word, len(self.words) + 1) for word in text.split()]

    def __call__(self, texts, add_special_tokens=False):
        if isinstance(texts, str):
            return {"input_ids": self._ids(texts)}
        return {"input_ids": [self._ids(text) for text in texts]}


SUFFIX = FakeTokenizer.THINKING_OFF_SUFFIX


def record(example_id="ACME/2015/page_10.pdf-1", user="what is alpha plus beta ?"):
    return SimpleNamespace(
        example_id=example_id,
        system="reply in two lines",
        user=user,
        target="PROGRAM: add(1, 2)\nANSWER: 3",
    )


def invented_records(tmp_path):
    split_cfg, data_cfg, _ = build_frozen_split(tmp_path)
    prompt = load_prompt(root=tmp_path)
    return [build_record(example, prompt) for example in load_split("fit", split_cfg, data_cfg)]


def test_a_sequence_is_the_prompt_then_the_reply_then_the_end_token():
    tokenizer = WordTokenizer()
    example = encode_example(record(), tokenizer, SUFFIX)
    prompt = tokenizer(
        render_chat("reply in two lines", "what is alpha plus beta ?", tokenizer, SUFFIX)
    )
    reply = tokenizer("PROGRAM: add(1, 2)\nANSWER: 3 <end>")["input_ids"]
    assert example.input_ids == prompt["input_ids"] + reply
    assert example.prompt_tokens == len(prompt["input_ids"])
    assert example.input_ids[-1] == tokenizer.words["<end>"]  # the model is taught to stop
    assert (example.length, example.target_tokens) == (len(example.input_ids), len(reply))
    assert example.example_id == "ACME/2015/page_10.pdf-1"


def test_only_the_reply_carries_a_label():
    example = encode_example(record(), WordTokenizer(), SUFFIX)
    cut = example.prompt_tokens
    assert example.labels[:cut] == [IGNORE] * cut
    assert example.labels[cut:] == example.input_ids[cut:]
    assert len(example.labels) == example.length
    assert sum(label != IGNORE for label in example.labels) == example.target_tokens == 6


def test_records_are_encoded_in_order_and_none_is_cut(tmp_path):
    tokenizer = FakeTokenizer()
    records = invented_records(tmp_path)
    rows = encode_records(records, tokenizer, SUFFIX, max_length=10_000)
    assert [row.example_id for row in rows] == [item.example_id for item in records]
    counts = count_split(records, tokenizer, SUFFIX)
    assert [row.length for row in rows] == [count.total for count in counts]
    assert [row.target_tokens for row in rows] == [count.target for count in counts]


def test_a_record_over_the_length_is_named_and_nothing_is_truncated():
    tokenizer = WordTokenizer()
    short = record("ACME/2015/page_10.pdf-1")
    long = [record(f"BOLT/2016/page_3.pdf-{n}", user="word " * 40) for n in range(1, 8)]
    limit = encode_example(short, tokenizer, SUFFIX).length
    assert len(encode_records([short], tokenizer, SUFFIX, limit)) == 1  # at the limit: allowed
    with pytest.raises(LengthError) as caught:
        encode_records([short, *long], tokenizer, SUFFIX, limit)
    message = str(caught.value)
    assert message.startswith(f"7 training sequences are longer than max_length {limit}")
    assert "BOLT/2016/page_3.pdf-1 (" in message and "BOLT/2016/page_3.pdf-5 (" in message
    assert "and 2 more" in message and "Nothing is truncated" in message
    assert "ACME" not in message and "word word" not in message  # ids and counts only


# --------------------------------------------------------------------------- the real tokenizer


@pytest.fixture(scope="module")
def real():
    budget = load_token_budget_config()
    try:
        return budget, load_tokenizer(budget)
    except TokenBudgetError as exc:
        pytest.skip(f"no tokenizer here: {exc}")


def test_with_the_real_tokenizer_a_sequence_has_the_budgets_length(real, tmp_path):
    budget, tokenizer = real
    records = invented_records(tmp_path)
    rows = encode_records(records, tokenizer, budget.thinking_off_suffix, budget.max_length)
    counts = count_split(records, tokenizer, budget.thinking_off_suffix)
    for row, count in zip(rows, counts, strict=True):
        assert (row.length, row.prompt_tokens, row.target_tokens) == (
            count.total,
            count.prompt,
            count.target,
        )


def test_with_the_real_tokenizer_the_labelled_part_is_the_reply_and_the_rest_the_prompt(
    real, tmp_path
):
    budget, tokenizer = real
    for item in invented_records(tmp_path)[:5]:
        row = encode_example(item, tokenizer, budget.thinking_off_suffix)
        labelled = [label for label in row.labels if label != IGNORE]
        assert tokenizer.decode(labelled) == item.target + "<|im_end|>"
        prompt = render_chat(item.system, item.user, tokenizer, budget.thinking_off_suffix)
        assert tokenizer.decode(row.input_ids[: row.prompt_tokens]) == prompt
        assert prompt.endswith(
            budget.thinking_off_suffix
        )  # the reply follows the empty think block
        assert row.input_ids[-1] == tokenizer.eos_token_id
