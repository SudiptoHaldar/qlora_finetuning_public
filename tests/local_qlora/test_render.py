"""Tests for qlora_finetuning.local_qlora.render: the prompt renderer and the reply reader."""

from __future__ import annotations

import pytest

from qlora_finetuning.data import (
    build_record,
    load_data_config,
    load_prompt,
    load_split,
    serialize_split,
    verify_snapshot,
)
from qlora_finetuning.data.token_budget import (
    TokenBudgetError,
    count_split,
    load_token_budget_config,
    load_tokenizer,
    render_prompt,
)
from qlora_finetuning.local_qlora.render import RenderError, read_generation, render_chat
from tests.data.synthetic import FakeTokenizer, build_frozen_split

from .conftest import END_IDS, END_OF_TEXT, IM_END, IM_START, PieceTokenizer

SUFFIX = FakeTokenizer.THINKING_OFF_SUFFIX
SYSTEM = "Reply in two lines.\n  Second line, with a trailing space "
USER = "Table:\n| café | 1,234 € |\n\nQuestion: what is ½ of it?\n"


def invented_records(tmp_path):
    split_cfg, data_cfg, _ = build_frozen_split(tmp_path)
    prompt = load_prompt(root=tmp_path)
    return [build_record(example, prompt) for example in load_split("dev", split_cfg, data_cfg)]


# --------------------------------------------------------------------------- the prompt


def test_the_texts_arrive_character_for_character():
    text = render_chat(SYSTEM, USER, FakeTokenizer(), SUFFIX)
    assert SYSTEM in text and USER in text
    assert text.index(SYSTEM) < text.index(USER)
    assert text.endswith(SUFFIX)


def test_the_renderer_gives_what_the_token_budget_counted(tmp_path):
    tokenizer = FakeTokenizer()
    for record in invented_records(tmp_path):
        rendered = render_chat(record.system, record.user, tokenizer, SUFFIX)
        assert rendered == render_prompt(record, tokenizer)
        assert record.system in rendered and record.user in rendered
        assert record.target not in rendered  # the prompt stops where the reply starts


class ThinkingTokenizer(FakeTokenizer):
    """A template that ignores the thinking switch, as a changed template might."""

    def apply_chat_template(self, messages, **options):
        options["enable_thinking"] = True
        return super().apply_chat_template(messages, **options)


def test_a_template_that_leaves_thinking_on_is_refused():
    with pytest.raises(RenderError, match="thinking-off"):
        render_chat(SYSTEM, USER, ThinkingTokenizer(), SUFFIX)
    with pytest.raises(RenderError):
        render_chat(SYSTEM, USER, FakeTokenizer(), "")


# --------------------------------------------------------------------------- the reply


def read(ids, limit=192):
    return read_generation(ids, PieceTokenizer(), END_IDS, limit)


@pytest.mark.parametrize("end", [IM_END, END_OF_TEXT])
def test_a_reply_ends_at_its_first_end_token(end):
    reply = read([0, 1, 2, end, END_OF_TEXT, END_OF_TEXT, 0])  # padded after it, in a batch
    assert reply.text == "PROGRAM: add(1, 2)\nANSWER: 3"
    assert reply.output_tokens == 4  # three tokens and the end token
    assert reply.stop_reason == "end" and not reply.think_removed


def test_a_reply_without_an_end_token_ran_to_the_limit():
    reply = read([0, 1, 2, 1, 0], limit=5)
    assert (reply.stop_reason, reply.output_tokens) == ("length", 5)
    assert reply.text == "PROGRAM: add(1, 2)\nANSWER: 3\nPROGRAM: add(1, 2)"


def test_nothing_past_the_limit_is_read():
    reply = read([0, 1, 2, IM_END], limit=2)
    assert (reply.text, reply.output_tokens, reply.stop_reason) == (
        "PROGRAM: add(1, 2)\n",
        2,
        "length",
    )


def test_an_empty_reply_is_a_reply():
    reply = read([IM_END])
    assert (reply.text, reply.output_tokens, reply.stop_reason) == ("", 1, "end")


def test_special_tokens_are_removed_and_nothing_else_is_tidied():
    reply = read([IM_START, 5, 0, 1, 5, IM_END])
    assert reply.text == "  café € PROGRAM: add(1, 2)\n  café € "


def test_a_thinking_block_at_the_start_is_removed_and_counted():
    reply = read([3, 1, 6, 1, 4, 1, 1, 0, 1, 2, IM_END])
    assert reply.text == "PROGRAM: add(1, 2)\nANSWER: 3"
    assert reply.think_removed
    assert reply.output_tokens == 11  # the block was generated, so it is counted


def test_an_unclosed_or_later_thinking_marker_is_left_for_the_parser():
    unclosed = read([3, 1, 6, 1, 0, IM_END])
    assert unclosed.text == "<think>\nthoughts\nPROGRAM: add(1, 2)" and not unclosed.think_removed
    later = read([0, 1, 3, 6, 4, 2, IM_END])
    assert later.text == "PROGRAM: add(1, 2)\n<think>thoughts</think>ANSWER: 3"
    assert not later.think_removed


# --------------------------------------------------------------------------- the real tokenizer


@pytest.fixture(scope="module")
def real():
    budget = load_token_budget_config()
    try:
        return budget, load_tokenizer(budget)
    except TokenBudgetError as exc:
        pytest.skip(f"no tokenizer here: {exc}")


def test_with_the_real_tokenizer_the_prompt_has_the_budgets_token_count(real, tmp_path):
    budget, tokenizer = real
    records = invented_records(tmp_path)
    counts = count_split(records, tokenizer, budget.thinking_off_suffix)
    for record, count in zip(records, counts, strict=True):
        text = render_chat(record.system, record.user, tokenizer, budget.thinking_off_suffix)
        assert len(tokenizer(text, add_special_tokens=False)["input_ids"]) == count.prompt
        assert record.system in text and record.user in text


def test_with_the_real_tokenizer_decoding_changes_nothing_and_keeps_think_markers(real):
    _, tokenizer = real
    end_ids = {tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
    written = "PROGRAM: add( 1 , 2 ) , divide(#0, 3)\nANSWER:  3 .  café isn't 5 %\n"
    ids = tokenizer(written + "<|im_end|>", add_special_tokens=False)["input_ids"]
    reply = read_generation(ids, tokenizer, end_ids, 192)
    assert reply.text == written and reply.stop_reason == "end"
    assert reply.output_tokens == len(ids)

    thinking = "<think>\nlet me see\n</think>\n\n" + written
    ids = tokenizer(thinking + "<|im_end|>", add_special_tokens=False)["input_ids"]
    assert "<think>" in tokenizer.decode(ids, skip_special_tokens=True)  # not special tokens
    reply = read_generation(ids, tokenizer, end_ids, 192)
    assert reply.text == written and reply.think_removed


def test_with_the_real_tokenizer_quick_dev_prompts_total_the_budgets_figure(real):
    """Counts only. 128,495 is the committed token budget report's figure for quick-dev."""
    if not all(item.ok for item in verify_snapshot(load_data_config())):
        pytest.skip("no FinQA snapshot here; fetch it with: python -m scripts.finqa_fetch")
    budget, tokenizer = real
    texts = [
        render_chat(record.system, record.user, tokenizer, budget.thinking_off_suffix)
        for record in serialize_split("quick_dev")
    ]
    lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
    assert (len(lengths), sum(lengths)) == (200, 128_495)
