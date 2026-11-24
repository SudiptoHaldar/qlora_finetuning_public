"""The reply parser (req_002 v2.01): strict replies, recovered replies, and reply errors."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from qlora_finetuning.dsl import DEVIATIONS, ReplyError, ReplyFormat, parse_reply

STRICT = "PROGRAM: subtract(920, 95), divide(#0, 95)\nANSWER: 8.68421"


def test_a_strict_reply_gives_both_fields(reply_format, config):
    parsed = parse_reply(STRICT, reply_format, config)
    assert parsed.program_text == "subtract(920, 95), divide(#0, 95)"
    assert parsed.answer_text == "8.68421"
    assert parsed.strict is True
    assert parsed.deviations == ()


@pytest.mark.parametrize(
    "text",
    [
        "\n\n  " + STRICT + "  \n\n",  # whitespace around the reply
        STRICT.replace("\n", "\r\n"),  # Windows line ends
        STRICT.replace("\n", "\r"),  # old Mac line ends
        "PROGRAM: subtract(920, 95), divide(#0, 95)   \nANSWER: 8.68421\t",  # at line ends
        "PROGRAM:    subtract(920, 95), divide(#0, 95)\nANSWER:\t8.68421",  # after a label
        "PROGRAM:subtract(920, 95), divide(#0, 95)\nANSWER:8.68421",  # none after a label
        "PROGRAM: subtract(920, 95), divide(#0, 95)\n\n  \nANSWER: 8.68421",  # a blank line
        "PROGRAM: subtract(920, 95), divide(#0, 95)\n  ANSWER: 8.68421",  # an indented line
    ],
)
def test_harmless_whitespace_keeps_a_reply_strict(text, reply_format, config):
    parsed = parse_reply(text, reply_format, config)
    assert parsed.strict is True
    assert parsed.program_text == "subtract(920, 95), divide(#0, 95)"
    assert parsed.answer_text == "8.68421"


@pytest.mark.parametrize(
    ("text", "deviations"),
    [
        ("Here is the calculation.\n" + STRICT, ("text_before",)),
        (STRICT.replace("\n", "\nthat gives\n"), ("text_between",)),
        (STRICT + "\nI hope this helps.", ("text_after",)),
        (STRICT.replace("PROGRAM:", "Program:"), ("label_case",)),
        (STRICT.replace("ANSWER:", "answer:"), ("label_case",)),
        ("PROGRAM: add(1, 2)", ("answer_missing",)),
        ("PROGRAM: add(1, 2)\nANSWER:", ("answer_missing",)),
        ("PROGRAM: add(1, 2)\nso the result is 3", ("text_after", "answer_missing")),
        ("Sure!\nprogram: add(1, 2)", ("text_before", "label_case", "answer_missing")),
        (
            "Sure!\nprogram: add(1, 2)\nwhich is\nanswer: 3\nDone.",
            ("text_before", "text_between", "text_after", "label_case"),
        ),
    ],
)
def test_each_deviation_is_named_and_the_reply_is_still_read(
    text, deviations, reply_format, config
):
    parsed = parse_reply(text, reply_format, config)
    assert parsed.strict is False
    assert parsed.deviations == deviations
    assert parsed.program_text in ("add(1, 2)", "subtract(920, 95), divide(#0, 95)")


def test_deviations_are_reported_in_a_fixed_order(reply_format, config):
    parsed = parse_reply("x\nprogram: add(1, 2)\ny\nanswer: 3\nz", reply_format, config)
    assert list(parsed.deviations) == [name for name in DEVIATIONS if name in parsed.deviations]


def test_a_fenced_reply_is_recovered(reply_format, config):
    parsed = parse_reply("```\n" + STRICT + "\n```", reply_format, config)
    assert parsed.program_text == "subtract(920, 95), divide(#0, 95)"
    assert parsed.answer_text == "8.68421"
    assert parsed.deviations == ("text_before", "text_after")


def test_the_first_program_line_wins(reply_format, config):
    text = "PROGRAM: add(1, 2)\nANSWER: 3\nPROGRAM: add(5, 5)\nANSWER: 10"
    parsed = parse_reply(text, reply_format, config)
    assert (parsed.program_text, parsed.answer_text) == ("add(1, 2)", "3")
    assert parsed.deviations == ("text_after",)


def test_an_answer_line_before_the_program_is_not_the_answer(reply_format, config):
    parsed = parse_reply("ANSWER: 3\nPROGRAM: add(1, 2)", reply_format, config)
    assert parsed.answer_text == ""
    assert parsed.deviations == ("text_before", "answer_missing")


def test_an_empty_program_field_is_left_to_the_program_parser(reply_format, config):
    parsed = parse_reply("PROGRAM:\nANSWER: 3", reply_format, config)
    assert parsed.program_text == ""
    assert parsed.strict is True


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t\n"])
def test_an_empty_reply_is_an_error(text, reply_format, config):
    with pytest.raises(ReplyError) as caught:
        parse_reply(text, reply_format, config)
    assert caught.value.code == "reply_empty"
    assert caught.value.stage == "reply"


@pytest.mark.parametrize(
    "text", ["The answer is 3.", "ANSWER: 3", "**PROGRAM:** add(1, 2)", "PROGRAM add(1, 2)"]
)
def test_a_reply_without_a_program_line_is_an_error(text, reply_format, config):
    with pytest.raises(ReplyError) as caught:
        parse_reply(text, reply_format, config)
    assert caught.value.code == "no_program"


def test_a_reply_over_the_limit_is_an_error_and_is_not_cut(reply_format, config):
    text = STRICT + " " * config.max_reply_chars
    with pytest.raises(ReplyError) as caught:
        parse_reply(text, reply_format, config)
    assert caught.value.code == "reply_too_long"
    exactly = STRICT + " " * (config.max_reply_chars - len(STRICT))
    assert parse_reply(exactly, reply_format, config).strict is True


def test_a_reply_that_is_not_text_is_the_callers_mistake(reply_format, config):
    with pytest.raises(TypeError):
        parse_reply(None, reply_format, config)


def test_labels_come_from_the_prompt_definition(config):
    prompt = SimpleNamespace(program_label="CALC:", answer_label="RESULT:")
    reply_format = ReplyFormat.from_prompt(prompt)
    parsed = parse_reply("CALC: add(1, 2)\nRESULT: 3", reply_format, config)
    assert (parsed.program_text, parsed.answer_text, parsed.strict) == ("add(1, 2)", "3", True)
    with pytest.raises(ReplyError):
        parse_reply("PROGRAM: add(1, 2)\nANSWER: 3", reply_format, config)


@pytest.mark.parametrize("label", ["", "  ", "TWO\nLINES:"])
def test_a_label_must_be_one_line_of_text(label):
    with pytest.raises(ValueError):
        ReplyFormat(program_label=label, answer_label="ANSWER:")
