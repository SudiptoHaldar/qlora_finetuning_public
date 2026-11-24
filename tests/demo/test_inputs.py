"""Tests for qlora_finetuning.demo.inputs: the input file and the bundled sample."""

from __future__ import annotations

import json

import pytest

from qlora_finetuning.demo.inputs import (
    DemoInputError,
    load_input,
    read_input,
    sample_path,
)
from qlora_finetuning.dsl import run_reply
from qlora_finetuning.evaluation.metrics import load_evaluation_config, values_match
from qlora_finetuning.settings import ConfigError

from .helpers import PASSAGES, QUESTION, TABLE, raw_input, write_input


def test_an_input_file_is_read_as_it_was_written(tmp_path):
    demo_input = load_input(write_input(tmp_path))
    assert demo_input.source == "question.json"
    assert demo_input.question == QUESTION
    assert demo_input.table == tuple(tuple(row) for row in TABLE)
    assert demo_input.passages == tuple(PASSAGES)
    assert demo_input.expected is None


def test_context_may_be_absent_or_empty_and_a_number_cell_becomes_text(tmp_path):
    assert load_input(write_input(tmp_path, context=None)).passages == ()
    assert load_input(write_input(tmp_path, context=[])).passages == ()
    assert load_input(write_input(tmp_path, context=["  ", "kept"])).passages == ("kept",)
    table = [["", "2024"], ["revenue", 1150], ["margin", 12.5]]
    assert load_input(write_input(tmp_path, table=table)).table[1:] == (
        ("revenue", "1150"),
        ("margin", "12.5"),
    )


def test_expected_is_a_number_or_yes_or_no(tmp_path):
    assert load_input(write_input(tmp_path, expected=0.25)).expected == 0.25
    assert load_input(write_input(tmp_path, expected=2)).expected == 2.0
    assert load_input(write_input(tmp_path, expected="yes")).expected == "yes"
    for bad in ("maybe", True, [0.25]):
        with pytest.raises(DemoInputError, match="expected must be a number, or yes or no"):
            load_input(write_input(tmp_path, expected=bad))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"table": None}, "table is missing"),
        ({"table": []}, "table must be a list of rows"),
        ({"table": "revenue 1150"}, "table must be a list of rows"),
        ({"table": [["", "2024"], []]}, "table row 1 must be a list of cells"),
        ({"table": [["", "2024"], "revenue"]}, "table row 1 must be a list of cells"),
        ({"table": [["", "2024"], ["revenue", ["1150"]]]}, "table row 1 has a cell"),
        ({"table": [["", "2024"], ["revenue", None]]}, "table row 1 has a cell"),
        ({"table": [["", "2024"], ["revenue", True]]}, "table row 1 has a cell"),
        ({"question": None}, "question must be a text"),
        ({"question": "   "}, "question must be a text"),
        ({"question": 7}, "question must be a text"),
        ({"context": "one passage"}, "context must be a list of texts"),
        ({"context": ["fine", 3]}, "context must be a list of texts"),
        ({"qestion": "a misspelt field"}, "unknown field qestion"),
    ],
)
def test_a_file_that_cannot_be_used_is_refused_with_file_and_field_named(
    tmp_path, changes, message
):
    path = write_input(tmp_path, "asked.json", **changes)
    with pytest.raises(DemoInputError, match=message) as caught:
        load_input(path)
    assert "input asked.json" in str(caught.value)
    assert isinstance(caught.value, ConfigError)  # so every entry point prints one line


def test_a_file_that_is_not_json_or_not_an_object_or_not_there(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"question": "half a file"', encoding="utf-8")
    with pytest.raises(DemoInputError, match=r"input broken.json: not JSON \(line 1"):
        load_input(path)
    path.write_text('["a", "list"]', encoding="utf-8")
    with pytest.raises(DemoInputError, match="must hold one JSON object"):
        load_input(path)
    with pytest.raises(DemoInputError, match="input absent.json: the file cannot be read"):
        load_input(tmp_path / "absent.json")
    with pytest.raises(DemoInputError, match="must hold one JSON object"):
        read_input("text", "typed")


def test_a_file_saved_with_a_byte_order_mark_is_read(tmp_path):
    path = tmp_path / "notepad.json"
    path.write_text(json.dumps(raw_input()), encoding="utf-8-sig")
    assert load_input(path).question == QUESTION


def test_the_sample_is_valid_and_a_correct_program_gives_its_expected_result():
    sample = load_input(sample_path())
    assert sample.source == "sample_input.json" and sample.passages
    assert isinstance(sample.expected, float)
    outcome = run_reply(
        "PROGRAM: subtract(4812, 4275), divide(#0, 4275)\nANSWER: 0.12561",
        [list(row) for row in sample.table],
    )
    assert outcome.ok and outcome.strict
    assert values_match(outcome.value, sample.expected, load_evaluation_config().execution)


def test_the_sample_says_that_it_is_invented():
    """It is committed, so it must not be a dataset row; it says so itself."""
    text = sample_path().read_text(encoding="utf-8")
    assert "invented" in text and text.isascii()
