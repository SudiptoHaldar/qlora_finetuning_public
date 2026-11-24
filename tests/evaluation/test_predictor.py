"""Tests for the ``Predictor`` protocol, its checks, and the fake predictor (req_002 v2.03)."""

from __future__ import annotations

import pytest

from qlora_finetuning.evaluation.fake import FakePredictor, gold_predictor
from qlora_finetuning.evaluation.predictor import (
    STOP_REASONS,
    USAGE_SOURCES,
    Prediction,
    PredictionRequest,
    Predictor,
    PredictorError,
    check_predictions,
    describe,
)


def request(example_id="A-1", system="you are careful", user="what is one plus two ?"):
    return PredictionRequest(example_id=example_id, system=system, user=user, max_new_tokens=192)


REQUESTS = [request("A-1"), request("A-2")]
GOOD = [Prediction("PROGRAM: add(1, 2)\nANSWER: 3", 10, 5, 0.25), Prediction("")]


# --------------------------------------------------------------------------- checking


def test_good_predictions_come_back_as_a_list():
    assert check_predictions(REQUESTS, tuple(GOOD)) == GOOD
    assert check_predictions([], []) == []
    assert Prediction("x").stop_reason == "end" and Prediction("x").input_tokens is None
    assert STOP_REASONS == ("end", "length", "other")
    assert USAGE_SOURCES == ("measured", "aws_api", "computed")


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        (GOOD[:1], "returned 1 predictions for 2 requests"),
        (GOOD + GOOD, "returned 4 predictions for 2 requests"),
        (None, "did not return a list"),
        ("PROGRAM: add(1, 2)", "did not return a list"),
        ((item for item in GOOD), "did not return a list"),
        ([GOOD[0], {"text": "x"}], "the prediction for A-2 is not a Prediction"),
        ([GOOD[0], "x"], "the prediction for A-2 is not a Prediction"),
        ([Prediction(None), GOOD[1]], "the prediction for A-1 has a text that is not a string"),
        ([Prediction(b"x"), GOOD[1]], "text that is not a string"),
        ([Prediction("x", input_tokens=-1), GOOD[1]], "input_tokens that is not a whole number"),
        ([Prediction("x", input_tokens=2.5), GOOD[1]], "input_tokens that is not a whole number"),
        ([Prediction("x", output_tokens=True), GOOD[1]], "output_tokens that is not a whole"),
        ([Prediction("x", output_tokens="5"), GOOD[1]], "output_tokens that is not a whole"),
        ([Prediction("x", latency_seconds=-0.1), GOOD[1]], "latency that is not a number"),
        ([Prediction("x", latency_seconds=float("nan")), GOOD[1]], "latency that is not a number"),
        ([Prediction("x", latency_seconds="1"), GOOD[1]], "latency that is not a number"),
        ([Prediction("x", stop_reason="max_tokens"), GOOD[1]], "the stop reason 'max_tokens'"),
    ],
)
def test_a_malformed_answer_is_a_predictor_failure(predictions, message):
    with pytest.raises(PredictorError, match=message):
        check_predictions(REQUESTS, predictions)


# --------------------------------------------------------------------------- describing


def test_describe_gives_plain_values():
    predictor = FakePredictor(
        {},
        name="qwen greedy",
        model_id="vendor/model",
        settings={"temperature": 0.0, "stop": ("a", "b"), "nested": {"k": 1}},
        batch_size=4,
        usage_source="measured",
    )
    assert describe(predictor) == {
        "name": "qwen greedy",
        "model_id": "vendor/model",
        "settings": {"nested": {"k": 1}, "stop": ["a", "b"], "temperature": 0.0},
        "batch_size": 4,
        "usage_source": "measured",
    }
    assert isinstance(predictor, Predictor)
    assert describe(FakePredictor({}))["model_id"] is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"name": ""}, "needs a name"),
        ({"name": None}, "needs a name"),
        ({"model_id": ""}, "model_id must be text or None"),
        ({"model_id": 5}, "model_id must be text or None"),
        ({"batch_size": 0}, "batch_size must be a positive whole number"),
        ({"batch_size": 2.0}, "batch_size must be a positive whole number"),
        ({"batch_size": True}, "batch_size must be a positive whole number"),
        ({"usage_source": "bill"}, "usage_source must be one of"),
        ({"settings": ["greedy"]}, "settings must be a mapping"),
        ({"settings": {"when": object()}}, "settings must hold plain values"),
        ({"settings": {"t": float("nan")}}, "settings must hold plain values"),
        ({"predict": None}, "has no predict method"),
    ],
)
def test_a_predictor_that_does_not_describe_itself(changes, message):
    predictor = FakePredictor({}, name="fake")
    for name, value in changes.items():
        setattr(predictor, name, value)
    with pytest.raises(PredictorError, match=message):
        describe(predictor)


def test_an_object_without_the_protocol_is_refused():
    assert not isinstance(object(), Predictor)
    with pytest.raises(PredictorError, match="needs a name"):
        describe(object())


# --------------------------------------------------------------------------- the fake


def test_fake_answers_from_a_table_and_counts_a_token_per_word():
    predictor = FakePredictor({"A-1": "PROGRAM: add(1, 2)\nANSWER: 3", "A-2": ""})
    first, second = predictor.predict(REQUESTS)
    assert first.text == "PROGRAM: add(1, 2)\nANSWER: 3"
    assert first.input_tokens == 3 + 6 and first.output_tokens == 5
    assert second.text == "" and second.output_tokens == 0
    assert first.latency_seconds == 0.0 and first.stop_reason == "end"
    assert predictor.asked == ["A-1", "A-2"]
    assert check_predictions(REQUESTS, [first, second])


def test_fake_answers_from_a_function_of_the_request():
    predictor = FakePredictor(
        lambda item: f"seen {item.example_id}", latency_seconds=0.5, stop_reason="length"
    )
    prediction = predictor.predict([request("B-9")])[0]
    assert prediction.text == "seen B-9"
    assert prediction.latency_seconds == 0.5 and prediction.stop_reason == "length"


def test_fake_without_a_reply_for_an_example_fails():
    predictor = FakePredictor({"A-1": "x"})
    with pytest.raises(PredictorError, match="no reply for A-2"):
        predictor.predict(REQUESTS)
    assert predictor.asked == []


def test_fake_fails_after_a_number_of_requests_and_a_batch_is_all_or_nothing():
    replies = {f"A-{n}": "x" for n in range(1, 7)}
    predictor = FakePredictor(replies, fail_after=3)
    assert len(predictor.predict([request("A-1"), request("A-2")])) == 2
    with pytest.raises(PredictorError, match="fails after 3 requests"):
        predictor.predict([request("A-3"), request("A-4")])  # would be the fourth
    assert predictor.asked == ["A-1", "A-2"]
    assert len(predictor.predict([request("A-3")])) == 1
    with pytest.raises(PredictorError):
        predictor.predict([request("A-4")])

    with pytest.raises(PredictorError):
        FakePredictor(replies, fail_after=0).predict([request("A-1")])


def test_gold_predictor_gives_each_example_its_own_gold_reply():
    class Record:
        def __init__(self, example_id, target):
            self.example_id, self.target = example_id, target

    predictor = gold_predictor([Record("A-1", "PROGRAM: add(1, 2)\nANSWER: 3")], batch_size=4)
    assert predictor.name == "gold" and predictor.batch_size == 4
    assert predictor.model_id is None and predictor.usage_source == "computed"
    assert predictor.predict([request("A-1")])[0].text == "PROGRAM: add(1, 2)\nANSWER: 3"
    assert gold_predictor([], name="other").name == "other"


def test_the_two_optional_fields_appear_only_when_the_predictor_has_them():
    predictor = FakePredictor({}, name="tuned", model_id="arn:deployment/x")
    assert set(describe(predictor)) == {
        "name", "model_id", "settings", "batch_size", "usage_source",
    }  # fmt: skip
    predictor.usage_model_id = "vendor/base"
    predictor.details = {"attempt": 1, "hyperparameters": {"epochCount": "2"}, "ids": ("a",)}
    described = describe(predictor)
    assert described["model_id"] == "arn:deployment/x"
    assert described["usage_model_id"] == "vendor/base"
    assert described["details"] == {
        "attempt": 1,
        "hyperparameters": {"epochCount": "2"},
        "ids": ["a"],
    }


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"usage_model_id": ""}, "usage_model_id must be text or None"),
        ({"usage_model_id": 7}, "usage_model_id must be text or None"),
        ({"details": ["tuned"]}, "details must be a mapping or None"),
        ({"details": {"when": object()}}, "details must hold plain values"),
    ],
)
def test_an_optional_field_that_is_not_plain_is_refused(changes, message):
    predictor = FakePredictor({}, name="fake")
    for name, value in changes.items():
        setattr(predictor, name, value)
    with pytest.raises(PredictorError, match=message):
        describe(predictor)
