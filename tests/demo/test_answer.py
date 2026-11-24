"""Tests for qlora_finetuning.demo.answer and .render, with a predictor that has no model."""

from __future__ import annotations

import dataclasses
import json

import pytest

from qlora_finetuning.data.serialization import format_user, load_prompt
from qlora_finetuning.data.token_budget import load_token_budget_config
from qlora_finetuning.demo.answer import adapter_name, answer
from qlora_finetuning.demo.inputs import read_input
from qlora_finetuning.demo.render import as_dict, as_json, as_text, ascii_text, warnings
from qlora_finetuning.dsl import CODES
from qlora_finetuning.evaluation.metrics import load_evaluation_config
from qlora_finetuning.evaluation.predictor import PredictorError
from qlora_finetuning.settings import ConfigError

from .helpers import (
    PASSAGES,
    QUESTION,
    RIGHT_REPLY,
    RUN,
    TABLE,
    StubPredictor,
    raw_input,
    tuned_stub,
)

PROGRAM = "PROGRAM: subtract(1150, 920), divide(#0, 920)"


def asked(reply=RIGHT_REPLY, predictor=None, **changes):
    predictor = predictor or tuned_stub(reply=reply)
    return answer(predictor, read_input(raw_input(**changes), "question.json")), predictor


# --------------------------------------------------------------------------- the prompt


def test_the_prompt_is_the_serializers_and_the_request_is_one_alone():
    result, predictor = asked()
    prompt, budget = load_prompt(), load_token_budget_config()
    assert len(predictor.requests) == 1 and len(predictor.requests[0]) == 1  # a batch of one
    request = predictor.requests[0][0]
    assert request.system == prompt.system
    assert request.user == format_user(PASSAGES, TABLE, QUESTION, prompt)
    assert request.max_new_tokens == budget.max_new_tokens == result.max_new_tokens
    assert request.example_id == "question.json"


def test_no_context_uses_the_prompts_own_wording():
    _, predictor = asked(context=None)
    assert load_prompt().no_context in predictor.requests[0][0].user


# --------------------------------------------------------------------------- the reply


def test_a_reply_that_runs_gives_the_value_the_steps_and_agreement():
    result, _ = asked()
    assert result.outcome.ok and result.outcome.strict
    assert result.outcome.program == "subtract(1150, 920), divide(#0, 920)"
    assert result.outcome.step_values == (230.0, 0.25) and result.outcome.value == 0.25
    assert result.agrees is True
    assert result.predictor == "qwen3-4b-tuned" and result.adapter == f"{RUN}/checkpoint-500"
    assert (result.input_tokens, result.output_tokens, result.seconds) == (400, 20, 1.25)


@pytest.mark.parametrize(
    ("reply", "agrees"),
    [
        (f"{PROGRAM}\nANSWER: 25%", True),  # a percentage on either scale
        (f"{PROGRAM}\nANSWER: 0.2501", True),  # inside the answer allowance
        (f"{PROGRAM}\nANSWER: 0.3", False),
        ("PROGRAM: greater(1150, 920)\nANSWER: yes", True),
        ("PROGRAM: greater(1150, 920)\nANSWER: no", False),
        (PROGRAM, None),  # no answer line
        (f"{PROGRAM}\nANSWER: a quarter", None),
        ("PROGRAM: divide(138, 0)\nANSWER: 0.12", None),  # the program did not run
    ],
)
def test_the_models_answer_is_compared_with_its_own_programs_result(reply, agrees):
    result, _ = asked(reply)
    assert result.agrees is agrees


def test_the_table_of_the_input_is_what_the_program_runs_on():
    result, _ = asked("PROGRAM: table_sum(operating income, none)\nANSWER: 239")
    assert result.outcome.value == 239.0 and result.agrees is True
    missing, _ = asked("PROGRAM: table_sum(net income, none)\nANSWER: 1")
    assert missing.outcome.error_code == "row_not_found"


def test_the_allowance_is_the_one_it_is_given():
    predictor = tuned_stub(reply=f"{PROGRAM}\nANSWER: 0.2501")
    config = load_evaluation_config()
    exact = dataclasses.replace(
        config, answer=dataclasses.replace(config.answer, absolute=0.0, relative=0.0000001)
    )
    demo_input = read_input(raw_input(), "question.json")
    assert answer(predictor, demo_input, evaluation_config=exact).agrees is False


def test_an_adapter_is_named_as_on_the_command_line():
    assert adapter_name(StubPredictor()) is None  # the untuned model
    final = StubPredictor(details={"training_run": RUN, "checkpoint": None})
    assert adapter_name(final) == RUN


def test_a_model_failure_and_a_wrong_number_of_replies_are_predictor_errors():
    with pytest.raises(PredictorError, match="out of memory"):
        asked(predictor=StubPredictor(error=PredictorError("out of memory")))

    class Twice(StubPredictor):
        def predict(self, requests):
            return super().predict(requests) * 2

    with pytest.raises(PredictorError, match="2 replies to one question"):
        asked(predictor=Twice())


def test_a_table_the_prompt_cannot_show_is_a_config_error():
    demo_input = dataclasses.replace(read_input(raw_input(), "odd.json"), table=((),))
    with pytest.raises(ConfigError, match="input odd.json has an empty table"):
        answer(tuned_stub(), demo_input)


# --------------------------------------------------------------------------- the length


def test_an_over_length_prompt_is_answered_with_a_warning_and_never_cut():
    budget = load_token_budget_config()
    edge = budget.max_length - budget.max_new_tokens
    fits, _ = asked(predictor=tuned_stub(input_tokens=edge))
    assert not fits.over_length and warnings(fits) == []
    long, predictor = asked(predictor=tuned_stub(input_tokens=edge + 1))
    assert long.over_length and long.outcome.ok  # answered all the same
    assert predictor.requests[0][0].user == format_user(PASSAGES, TABLE, QUESTION, load_prompt())
    assert "over the training length" in as_text(long)
    unknown, _ = asked(predictor=tuned_stub(input_tokens=None))
    assert not unknown.over_length and "prompt unknown" in as_text(unknown)


def test_a_reply_cut_off_at_the_limit_is_said():
    result, _ = asked(predictor=tuned_stub(stop_reason="length"))
    assert "cut off at the limit" in warnings(result)[0]
    assert "stopped at the reply limit" in as_text(result)


# --------------------------------------------------------------------------- the output


def test_the_text_shows_every_part_unrounded():
    result, _ = asked("PROGRAM: divide(138, 1150)\nANSWER: 0.12")
    lines = {line[:12].strip(): line[12:] for line in as_text(result).splitlines()}
    assert lines["Input"] == "question.json"
    assert lines["Model"] == f"qwen3-4b-tuned  {RUN}/checkpoint-500"
    assert lines["Question"] == QUESTION
    assert lines["Program"] == "divide(138, 1150)"
    assert lines["Steps"] == "#0 = 0.12"
    assert lines["Executed"] == "0.12"
    assert lines["Format"] == "exact"
    assert lines["Tokens"] == "prompt 400, reply 20, stopped at the end token"
    assert lines["Time"] in ("1.2 s", "1.3 s")
    assert lines["Model said"] == "0.12   agrees with the executed result"
    assert "Reply" not in lines and "Warning" not in lines

    third, _ = asked("PROGRAM: divide(1, 3)\nANSWER: 0.3333")
    assert repr(1 / 3) in as_text(third)  # every digit, no percent sign
    assert "Model said  0.3333   agrees with the executed result" in as_text(third)


def test_the_context_and_table_are_shown_only_when_asked_for():
    result, _ = asked()
    plain, shown = as_text(result), as_text(result, show_input=True)
    assert "Context" not in plain and "Table" not in plain
    assert shown.splitlines() == [
        *plain.splitlines()[:2],  # Input, Model
        f"Context     {PASSAGES[0]}",
        f"            {PASSAGES[1]}",
        "Table        | 2024 | 2023",
        "            revenue | $ 1,150 | $ 920",
        "            operating income | $ 138 | $ 101",
        *plain.splitlines()[2:],  # the question and everything after it, unchanged
    ]

    assert "context" not in as_dict(result) and "table" not in as_dict(result)
    with_input = as_dict(result, show_input=True)
    assert with_input["context"] == PASSAGES and with_input["table"] == TABLE
    del with_input["context"], with_input["table"]
    assert with_input == as_dict(result)
    assert json.loads(as_json(result, show_input=True))["table"] == TABLE


def test_shown_input_without_context_says_none_and_stays_ascii():
    result, _ = asked(context=None, table=[["", "2024"], ["café sales", 12]])
    shown = as_text(result, show_input=True)
    assert "Context     (none)" in shown
    assert "            caf\\xe9 sales | 12" in shown and shown.isascii()
    assert as_json(result, show_input=True).isascii()


def test_the_text_says_what_did_not_run_and_shows_the_reply():
    result, _ = asked("PROGRAM: divide(138, 0)\nANSWER: 0.12")
    text = as_text(result)
    assert "Executed    nothing: division_by_zero (execution) - " in text
    assert "Model said  0.12   not checked: the program did not run" in text
    assert "Reply       PROGRAM: divide(138, 0)\\nANSWER: 0.12" in text

    nothing, _ = asked("I cannot answer that.")
    text = as_text(nothing)
    assert "Program     none found in the reply" in text
    assert "no_program (reply)" in text and "Model said  nothing" in text
    untuned, _ = asked(predictor=StubPredictor(name="qwen3-4b"))
    assert "Model       qwen3-4b  (untuned)" in as_text(untuned)


def test_a_reply_outside_the_format_is_still_run_and_the_deviation_named():
    result, _ = asked(f"Here is the calculation.\n{PROGRAM}\nANSWER: 0.25")
    text = as_text(result)
    assert result.outcome.ok and not result.outcome.strict
    assert "Format      not exact: text_before" in text and "Reply       Here is" in text
    disagrees, _ = asked(f"{PROGRAM}\nANSWER: 0.3")
    assert "does NOT agree with the executed result" in as_text(disagrees)
    words, _ = asked(f"{PROGRAM}\nANSWER: a quarter")
    assert "not comparable" in as_text(words)


@pytest.mark.parametrize("code", sorted(CODES))
def test_every_error_code_renders(code):
    replies = {
        "reply_too_long": "PROGRAM: add(1, 2)" + " " * 5000,
        "reply_empty": "",
        "no_program": "nothing useful",
        "empty_program": "PROGRAM:\nANSWER: 1",
        "too_many_steps": "PROGRAM: " + ", ".join(["add(1, 2)"] * 40) + "\nANSWER: 3",
        "syntax": "PROGRAM: add(1, 2\nANSWER: 3",
        "unknown_operation": "PROGRAM: modulo(1, 2)\nANSWER: 3",
        "nested_call": "PROGRAM: add(add(1, 2), 3)\nANSWER: 6",
        "wrong_arity": "PROGRAM: add(1)\nANSWER: 1",
        "bad_argument": "PROGRAM: add(one, 2)\nANSWER: 3",
        "unknown_constant": "PROGRAM: add(const_77, 2)\nANSWER: 3",
        "bad_reference": "PROGRAM: add(#3, 2)\nANSWER: 3",
        "bad_row_name": "PROGRAM: table_sum(, none)\nANSWER: 3",
        "division_by_zero": "PROGRAM: divide(1, 0)\nANSWER: 3",
        "numeric_error": "PROGRAM: exp(1000000, 1000000)\nANSWER: 3",
        "yes_no_operand": "PROGRAM: greater(2, 1), add(#0, 1)\nANSWER: 3",
        "row_not_found": "PROGRAM: table_sum(net income, none)\nANSWER: 3",
        "row_not_numeric": "PROGRAM: table_sum(, none)\nANSWER: 3",
    }
    if code not in replies:  # table_missing: the demo always has a table
        pytest.skip("cannot happen in the demo")
    result, _ = asked(replies[code])
    assert not result.outcome.ok
    text = as_text(result)
    assert text.isascii() and "Executed    nothing: " in text
    json.loads(as_json(result))
    if result.outcome.error_code != code:  # the parser may name a neighbouring code
        assert result.outcome.error_code in CODES


def test_text_and_json_are_ascii_whatever_the_input_holds():
    table = [["", "2024"], ["café revenue €", "1150"]]
    result, _ = asked(
        "PROGRAM: table_sum(café revenue €, none)\nANSWER: 1150",
        table=table,
        question="what was the café's revenue — in €?",
    )
    assert result.outcome.value == 1150.0  # the row is found by its real name
    for output in (as_text(result), as_json(result)):
        assert output.isascii()
    assert "caf\\xe9 revenue \\u20ac" in as_text(result)
    assert json.loads(as_json(result))["question"] == "what was the café's revenue — in €?"
    assert ascii_text("two\r\nlines") == "two\\nlines"


def test_json_holds_the_same_values_as_the_text_and_the_same_bytes_each_time():
    result, _ = asked("PROGRAM: divide(1, 3)\nANSWER: 0.3333")
    line = as_json(result)
    assert "\n" not in line and line == as_json(result)
    data = json.loads(line)
    assert data == as_dict(result)
    assert data["outcome"]["value"] == 1 / 3 and data["outcome"]["step_values"] == [1 / 3]
    assert data["outcome"]["program"] == "divide(1, 3)" and data["agrees"] is True
    assert data["adapter"] == f"{RUN}/checkpoint-500" and data["warnings"] == []
    assert data["reply"] == "PROGRAM: divide(1, 3)\nANSWER: 0.3333"
    assert set(data["outcome"]) >= {"error_code", "deviations", "notes", "strict"}
