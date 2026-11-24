"""``run_reply`` (req_002 v2.01): one call, an outcome for every string, never an exception."""

from __future__ import annotations

import json
import random

import pytest

from qlora_finetuning.dsl import CODES, DEVIATIONS, ReplyOutcome, run_reply

STRICT = "PROGRAM: subtract(920, 95), divide(#0, 95)\nANSWER: 8.68421"


def test_the_usage_example(table):
    outcome = run_reply(STRICT, table)
    assert outcome.ok and outcome.strict
    assert outcome.value == pytest.approx(8.68421052631579)
    assert outcome.program == "subtract(920, 95), divide(#0, 95)"
    assert outcome.program_text == "subtract(920, 95), divide(#0, 95)"
    assert outcome.answer_text == "8.68421"
    assert outcome.step_values == (825.0, outcome.value)
    assert (outcome.error_code, outcome.error_stage, outcome.error_message) == (None, None, None)

    assert run_reply("PROGRAM: table_average(margin, none)\nANSWER: 0.1", table).value == (
        pytest.approx(0.1)
    )
    assert run_reply("PROGRAM: divide(5, 0)\nANSWER: 0", table).error_code == "division_by_zero"

    recovered = run_reply("Sure!\nprogram: add(1, 2)", table)
    assert recovered.deviations == ("text_before", "label_case", "answer_missing")
    assert (recovered.value, recovered.strict, recovered.answer_text) == (3.0, False, "")


def test_the_program_comes_back_in_canonical_form(table):
    outcome = run_reply("PROGRAM:add( 1,2 ),multiply(#0 , 3)\nANSWER: 9", table)
    assert outcome.strict
    assert outcome.program == "add(1, 2), multiply(#0, 3)"
    assert outcome.program_text == "add( 1,2 ),multiply(#0 , 3)"


@pytest.mark.parametrize(
    ("text", "code", "stage", "has_text", "has_program"),
    [
        ("", "reply_empty", "reply", False, False),
        ("The result is 3.", "no_program", "reply", False, False),
        ("x" * 5000, "reply_too_long", "reply", False, False),
        ("PROGRAM: sum(1, 2)\nANSWER: 3", "unknown_operation", "program", True, False),
        ("PROGRAM:\nANSWER: 3", "empty_program", "program", True, False),
        ("PROGRAM: divide(1, 0)\nANSWER: 3", "division_by_zero", "execution", True, True),
        ("PROGRAM: table_sum(profit, none)\nANSWER: 3", "row_not_found", "execution", True, True),
    ],
)
def test_an_error_at_each_stage_is_reported_not_raised(
    text, code, stage, has_text, has_program, table
):
    outcome = run_reply(text, table)
    assert not outcome.ok
    assert (outcome.error_code, outcome.error_stage) == (code, stage)
    assert CODES[code] == stage
    assert outcome.error_message
    assert (outcome.program_text is not None) == has_text
    assert (outcome.program is not None) == has_program
    assert outcome.value is None and outcome.step_values == ()


def test_format_and_program_are_judged_separately(table):
    """A strict reply can hold a bad program, and a loose reply a good one."""
    assert run_reply("PROGRAM: sum(1, 2)\nANSWER: 3", table).strict is True
    assert run_reply("ok\nPROGRAM: add(1, 2)\nANSWER: 3", table).ok is True


def test_a_table_is_only_needed_by_table_operations():
    assert run_reply("PROGRAM: add(1, 2)\nANSWER: 3").value == 3.0
    assert run_reply("PROGRAM: table_sum(revenue, none)\nANSWER: 3").error_code == "table_missing"


def test_the_note_for_a_repeated_row_reaches_the_outcome(table):
    assert run_reply("PROGRAM: table_sum(total, none)\nANSWER: 60", table).notes == (
        "duplicate_row",
    )


def test_defaults_come_from_the_config_files(table, reply_format, config):
    """With nothing passed, the labels of prompt.yaml and the limits of dsl.yaml are used."""
    assert run_reply(STRICT, table) == run_reply(STRICT, table, reply_format, config)


def test_given_labels_and_limits_are_used(table, config):
    from qlora_finetuning.dsl import DslConfig, ReplyFormat

    other = ReplyFormat(program_label="CALC:", answer_label="RESULT:")
    assert run_reply("CALC: add(1, 2)\nRESULT: 3", table, other, config).value == 3.0
    assert run_reply(STRICT, table, other, config).error_code == "no_program"
    tight = DslConfig(max_reply_chars=10, max_steps=1)
    assert run_reply(STRICT, table, config=tight).error_code == "reply_too_long"


def test_an_outcome_becomes_a_json_record(table):
    record = run_reply("x\n" + STRICT + "\nPROGRAM: again", table).to_dict()
    assert json.loads(json.dumps(record)) == record
    assert record["deviations"] == ["text_before", "text_after"]
    assert record["step_values"] == [825.0, record["value"]]
    failed = run_reply("PROGRAM: greater(2, 1), add(#0, 1)\nANSWER: 3", table).to_dict()
    assert failed["error_code"] == "yes_no_operand" and failed["value"] is None


def test_a_reply_that_is_not_text_is_the_callers_mistake(table):
    with pytest.raises(TypeError):
        run_reply(None, table)
    with pytest.raises(TypeError):
        run_reply(b"PROGRAM: add(1, 2)", table)


# --------------------------------------------------------------------------- no leaks

_SEEDS = [
    STRICT,
    "PROGRAM: table_average(net income ( loss ), none), divide(#0, const_100)\nANSWER: 0",
    "PROGRAM: add(6.9%, const_1), multiply(#0, 200), greater(#1, -3.5)\nANSWER: yes",
    "Sure!\nprogram: exp(2, .5), table_max(cash, end of year, none)\nanswer: 3\nDone.",
    "PROGRAM: table_sum(total, none), subtract(#0, 1,234)\nANSWER: 5",
]
_ALPHABET = "()#,%._- \t\n:0123456789abcdefghijklmnopqrstuvwxyzPROGAMNSWE$+é٥"


def _mutate(text: str, rng: random.Random) -> str:
    characters = list(text)
    for _ in range(rng.randint(1, 6)):
        at = rng.randrange(len(characters) + 1)
        action = rng.randrange(5)
        if action == 0 and characters:
            del characters[min(at, len(characters) - 1)]
        elif action == 1 and characters:
            source = characters[min(at, len(characters) - 1)]
            characters.insert(at, source * rng.randint(1, 4))
        elif action == 2 and len(characters) > 1:
            other = rng.randrange(len(characters))
            here = min(at, len(characters) - 1)
            characters[here], characters[other] = characters[other], characters[here]
        elif action == 3:
            characters.insert(at, rng.choice(_ALPHABET))
        else:
            start = rng.randrange(len(characters) + 1)
            characters[at:at] = characters[start : start + rng.randint(1, 20)]
    return "".join(characters)


def test_two_thousand_damaged_replies_all_come_back_as_outcomes(table):
    rng = random.Random(20261115)
    seen_codes: set[str] = set()
    succeeded = 0
    for _ in range(2000):
        text = _mutate(rng.choice(_SEEDS), rng)
        outcome = run_reply(text, table)  # any exception here fails the test
        assert isinstance(outcome, ReplyOutcome)
        assert set(outcome.deviations) <= set(DEVIATIONS)
        if outcome.ok:
            succeeded += 1
            assert isinstance(outcome.value, float) or outcome.value in ("yes", "no")
            assert outcome.program is not None
        else:
            assert CODES[outcome.error_code] == outcome.error_stage
            assert outcome.error_message.isascii()
            assert outcome.value is None
            seen_codes.add(outcome.error_code)
    # The generator must really exercise both paths, or the test proves nothing.
    assert succeeded > 100
    assert len(seen_codes) >= 8
