"""Tests for the predictions file: writing, reading back, and what counts as a clean beginning."""

from __future__ import annotations

import json

import pytest

from qlora_finetuning.dsl import DslConfig, ReplyFormat, run_reply
from qlora_finetuning.evaluation.metrics import EvaluationError, ExampleScore
from qlora_finetuning.evaluation.predictions import (
    PREDICTIONS_NAME,
    PredictionRecord,
    PredictionsError,
    append_records,
    outcome_from_dict,
    read_predictions,
    score_from_dict,
)

DSL = DslConfig(max_reply_chars=4000, max_steps=32)
FORMAT = ReplyFormat(program_label="PROGRAM:", answer_label="ANSWER:")
IDS = [f"AAA/2015/page_1.pdf-{number}" for number in range(1, 6)]
REPLY = "Sure!\nPROGRAM: subtract(920, 95), divide(#0, 95)\nANSWER: 8.68421"


def outcome():
    return run_reply(REPLY, None, FORMAT, DSL)


def score(example_id):
    return ExampleScore(
        example_id, False, True, True, True, True, deviations=("text_before",), relative_gap=6e-08
    )


def record(position, example_id=None, **changes):
    example_id = example_id or IDS[position]
    values = {
        "example_id": example_id,
        "index": position,
        "prompt_id": "p" * 64,
        "run_id": "20261116T000000Z-eval-x-0000",
        "reply": REPLY,
        "outcome": outcome().to_dict(),
        "score": score(example_id).to_dict(),
        "input_tokens": 100,
        "output_tokens": 12,
        "latency_seconds": 0.5,
        "stop_reason": "end",
    }
    values.update(changes)
    return PredictionRecord(**values)


def write(path, lines):
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")


@pytest.fixture
def path(tmp_path):
    return tmp_path / "run" / PREDICTIONS_NAME


# --------------------------------------------------------------------------- writing


def test_round_trip(path):
    records = [record(0), record(1, input_tokens=None, output_tokens=None), record(2)]
    assert append_records(path, records[:2]) == 2
    assert append_records(path, records[2:]) == 1
    data = path.read_bytes()
    assert data.endswith(b"\n") and b"\r" not in data and data.count(b"\n") == 3
    assert data.isascii()
    assert read_predictions(path, IDS) == (records, False)
    assert read_predictions(path, IDS[:3]) == (records, False)


def test_nothing_to_append_writes_nothing(path):
    assert append_records(path, []) == 0
    assert not path.exists()
    assert read_predictions(path, IDS) == ([], False)


def test_outcome_and_score_come_back_as_they_were():
    assert outcome_from_dict(json.loads(json.dumps(outcome().to_dict()))) == outcome()
    assert outcome().value == pytest.approx(8.684210526)
    assert score_from_dict(json.loads(json.dumps(score("A").to_dict()))) == score("A")
    failed = run_reply("PROGRAM: divide(5, 0)\nANSWER: 1", None, FORMAT, DSL)
    assert outcome_from_dict(failed.to_dict()) == failed


def test_a_reply_with_odd_characters_survives(path):
    odd = record(0, reply="PROGRAM: add(1, 2)\r\nANSWER: 3 é \U0001f600 \ud800 \x00")
    append_records(path, [odd])
    assert read_predictions(path, IDS)[0] == [odd]


def test_a_value_that_is_not_finite_cannot_be_written(path):
    broken = record(0, latency_seconds=float("nan"))
    with pytest.raises(ValueError):
        append_records(path, [broken])


# --------------------------------------------------------------------------- reading


def test_a_cut_off_last_line_is_dropped_and_reported(path):
    append_records(path, [record(0), record(1), record(2)])
    text = path.read_text(encoding="utf-8")
    path.write_text(text[:-40], encoding="utf-8", newline="\n")  # the writer died mid-line
    records, dropped = read_predictions(path, IDS)
    assert [item.example_id for item in records] == IDS[:2] and dropped

    path.write_text(text[:-1], encoding="utf-8", newline="\n")  # whole, but never ended
    records, dropped = read_predictions(path, IDS)
    assert len(records) == 2 and dropped


def test_a_damaged_line_in_the_middle_is_an_error(path):
    append_records(path, [record(0), record(1), record(2)])
    lines = path.read_text(encoding="utf-8").splitlines()
    for damage in ("{not json", "", "[1, 2]", '"text"'):
        write(path, [lines[0], damage, lines[2]])
        with pytest.raises(PredictionsError, match=f"{PREDICTIONS_NAME} line 2 is damaged"):
            read_predictions(path, IDS)
    assert issubclass(PredictionsError, EvaluationError)


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        ([0, 0], "line 2 holds .*-1 at position 0, where .*-2 at position 1 was expected"),
        ([1, 0], "line 1 holds .*-2 at position 1, where .*-1 at position 0"),
        ([0, 2], "line 2 holds .*-3 at position 2, where .*-2 at position 1"),
    ],
)
def test_a_repeated_or_misplaced_example_is_an_error(path, positions, message):
    append_records(path, [record(position) for position in positions])
    with pytest.raises(PredictionsError, match=message):
        read_predictions(path, IDS)


def test_an_unknown_id_a_wrong_position_and_too_many_lines(path):
    append_records(path, [record(0, example_id="ZZZ/2015/page_1.pdf-1")])
    with pytest.raises(PredictionsError, match="not a clean beginning of the split"):
        read_predictions(path, IDS)

    path.unlink()
    append_records(path, [record(0, index=3)])
    with pytest.raises(PredictionsError, match="at position 3"):
        read_predictions(path, IDS)

    path.unlink()
    append_records(path, [record(0), record(1), record(2)])
    with pytest.raises(PredictionsError, match="line 3: the file has more lines than the 2"):
        read_predictions(path, IDS[:2])


def changed(**changes):
    data = record(0).to_dict()
    for name, value in changes.items():
        if value is ...:
            del data[name]
        else:
            data[name] = value
    return json.dumps(data)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (changed(schema=2), "unknown schema 2"),
        (changed(reply=...), "missing or unknown field"),
        (changed(extra=1), "missing or unknown field"),
        (changed(reply=5), "reply is not text"),
        (changed(example_id=None), "example_id is not text"),
        (changed(index="0"), "index is not a whole number"),
        (changed(index=True), "index is not a whole number"),
        (changed(stop_reason="max_tokens"), "unknown stop reason"),
        (changed(outcome={"value": 1}), "outcome does not hold the expected fields"),
        (changed(score=None), "score does not hold the expected fields"),
    ],
)
def test_a_line_that_is_not_a_prediction_record(path, line, message):
    path.parent.mkdir(parents=True)
    write(path, [line])
    with pytest.raises(PredictionsError, match=message):
        read_predictions(path, IDS)
