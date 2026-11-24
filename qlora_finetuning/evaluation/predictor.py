"""The ``Predictor`` protocol: what a model backend gives the evaluation runner (req_002 v2.03).

A predictor takes prompts and returns replies. It knows its model; the runner knows the
split, the scoring, the predictions file, and the run registry. A backend implements this
protocol and nothing of the loop.

Usage, for a backend::

    from qlora_finetuning.evaluation.predictor import Prediction, PredictorError

    class MyPredictor:
        name = "my-model greedy"
        model_id = "vendor/my-model"
        settings = {"temperature": 0.0}      # everything that can change a reply
        batch_size = 8
        usage_source = "measured"            # where the token counts come from

        def predict(self, requests):
            # request.system and request.user are the canonical record's texts: wrap them in
            # the model's format and add no word. Return one prediction per request, in order.
            try:
                texts = self._generate(requests)
            except TimeoutError as error:
                raise PredictorError("the endpoint did not answer") from error
            return [Prediction(text=text, input_tokens=..., output_tokens=...) for text in texts]

Two optional attributes, for a model that is called under one id and billed under another
(a tuned Nova model is called by its deployment's ARN and billed as its base model):
``usage_model_id`` is the id the ledger records the run's tokens under (default:
``model_id``), and ``details`` is a mapping of plain values that says what else identifies
the model. ``model_id`` and ``settings`` decide whether a run can be resumed; ``details``
does not.

Raise ``PredictorError`` when the model could not be asked (after the backend's own retries).
The runner then stops, keeps what was predicted, and can be resumed. Never return an empty
text in place of a failure: an empty reply is scored as the model's answer.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

END = "end"
LENGTH = "length"
OTHER = "other"
STOP_REASONS = (END, LENGTH, OTHER)
USAGE_SOURCES = ("measured", "aws_api", "computed")


class PredictorError(Exception):
    """A predictor could not answer, or answered with something that is not a prediction.

    ``run_id`` is set by the runner to the run that was interrupted, so a caller can print
    the command that resumes it.
    """

    run_id: str | None = None


@dataclass(frozen=True)
class PredictionRequest:
    """One prompt. The two texts are a canonical record's, unchanged.

    ``example_id`` is for logs and for the fake predictor; a real predictor must not let it
    change the reply.
    """

    example_id: str
    system: str
    user: str
    max_new_tokens: int


@dataclass(frozen=True)
class Prediction:
    """One reply. ``text`` has whatever is specific to the model already removed.

    A token count is ``None`` when the backend does not know it. ``stop_reason`` is ``end``
    (the model finished), ``length`` (cut off at the reply limit), or ``other``.
    """

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_seconds: float = 0.0
    stop_reason: str = END


@runtime_checkable
class Predictor(Protocol):
    """What the runner needs from a model backend."""

    name: str
    model_id: str | None
    settings: Mapping[str, Any]
    batch_size: int
    usage_source: str

    def predict(self, requests: Sequence[PredictionRequest]) -> Sequence[Prediction]:
        """One prediction per request, in the same order. Raises ``PredictorError``."""
        ...


def _whole_or_unknown(value: Any) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def check_predictions(
    requests: Sequence[PredictionRequest], predictions: Any
) -> list[Prediction]:
    """Return ``predictions`` as a list, or raise ``PredictorError`` saying what is wrong.

    A backend that returns the wrong number of replies, or something that is not a reply,
    must not produce scores.
    """
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise PredictorError("the predictor did not return a list of predictions")
    if len(predictions) != len(requests):
        raise PredictorError(
            f"the predictor returned {len(predictions)} predictions for {len(requests)} requests"
        )
    for request, prediction in zip(requests, predictions, strict=True):
        where = f"the prediction for {request.example_id}"
        if not isinstance(prediction, Prediction):
            raise PredictorError(f"{where} is not a Prediction")
        if not isinstance(prediction.text, str):
            raise PredictorError(f"{where} has a text that is not a string")
        for name in ("input_tokens", "output_tokens"):
            if not _whole_or_unknown(getattr(prediction, name)):
                raise PredictorError(f"{where} has a {name} that is not a whole number or None")
        seconds = prediction.latency_seconds
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise PredictorError(f"{where} has a latency that is not a number of zero or more")
        if prediction.stop_reason not in STOP_REASONS:
            raise PredictorError(
                f"{where} has the stop reason {prediction.stop_reason!r}; "
                f"use one of: {', '.join(STOP_REASONS)}"
            )
    return list(predictions)


def describe(predictor: Any) -> dict[str, Any]:
    """A predictor's own description, as plain values, for the run and for the resume check.

    Raises ``PredictorError`` when the predictor does not describe itself properly.
    """
    name = getattr(predictor, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise PredictorError("a predictor needs a name")
    model_id = getattr(predictor, "model_id", None)
    if model_id is not None and (not isinstance(model_id, str) or not model_id.strip()):
        raise PredictorError(f"predictor {name}: model_id must be text or None")
    batch_size = getattr(predictor, "batch_size", None)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise PredictorError(f"predictor {name}: batch_size must be a positive whole number")
    usage_source = getattr(predictor, "usage_source", None)
    if usage_source not in USAGE_SOURCES:
        raise PredictorError(
            f"predictor {name}: usage_source must be one of: {', '.join(USAGE_SOURCES)}"
        )
    settings = getattr(predictor, "settings", None)
    if not isinstance(settings, Mapping):
        raise PredictorError(f"predictor {name}: settings must be a mapping")
    try:
        plain = json.loads(json.dumps(dict(settings), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise PredictorError(
            f"predictor {name}: settings must hold plain values (text, numbers, lists)"
        ) from error
    if not callable(getattr(predictor, "predict", None)):
        raise PredictorError(f"predictor {name} has no predict method")
    described = {
        "name": name,
        "model_id": model_id,
        "settings": plain,
        "batch_size": batch_size,
        "usage_source": usage_source,
    }
    # The two optional fields appear only when the predictor has them, so a predictor
    # without them describes itself exactly as it always did.
    usage_model_id = getattr(predictor, "usage_model_id", None)
    if usage_model_id is not None:
        if not isinstance(usage_model_id, str) or not usage_model_id.strip():
            raise PredictorError(f"predictor {name}: usage_model_id must be text or None")
        described["usage_model_id"] = usage_model_id
    details = getattr(predictor, "details", None)
    if details is not None:
        if not isinstance(details, Mapping):
            raise PredictorError(f"predictor {name}: details must be a mapping or None")
        try:
            described["details"] = json.loads(
                json.dumps(dict(details), sort_keys=True, allow_nan=False)
            )
        except (TypeError, ValueError) as error:
            raise PredictorError(
                f"predictor {name}: details must hold plain values (text, numbers, lists)"
            ) from error
    return described
