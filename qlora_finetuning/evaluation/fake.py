"""A predictor without a model, for tests and for proving the runner (req_002 v2.03).

``FakePredictor`` answers from a table of replies by example id, or from a function of the
request. It can be told to fail after a number of requests, which is how an interruption is
tested. ``gold_predictor`` answers every example with its own gold reply, so a run with it
must score every example right.

Usage::

    from qlora_finetuning.data import serialize_split
    from qlora_finetuning.evaluation.fake import FakePredictor, gold_predictor

    predictor = gold_predictor(serialize_split("quick_dev"))

    scripted = FakePredictor({"ACME/2015/page_10.pdf-1": "PROGRAM: add(1, 2)\\nANSWER: 3"})
    chatty = FakePredictor(lambda request: "Sure!\\nPROGRAM: add(1, 2)\\nANSWER: 3")
    failing = FakePredictor(replies, batch_size=4, fail_after=10)   # an interruption
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from qlora_finetuning.evaluation.predictor import (
    END,
    Prediction,
    PredictionRequest,
    PredictorError,
)

Replies = Mapping[str, str] | Callable[[PredictionRequest], str]


class FakePredictor:
    """Answers from ``replies``. Counts one token per word, so usage can be tested.

    ``fail_after=N`` answers N requests in total and then raises ``PredictorError``. A batch
    is all or nothing: one that would go past N fails whole. ``asked`` lists the example ids
    that were answered, in order.
    """

    def __init__(
        self,
        replies: Replies,
        *,
        name: str = "fake",
        model_id: str | None = None,
        settings: Mapping[str, Any] | None = None,
        batch_size: int = 1,
        fail_after: int | None = None,
        latency_seconds: float = 0.0,
        stop_reason: str = END,
        usage_source: str = "computed",
    ) -> None:
        self._replies = replies
        self.name = name
        self.model_id = model_id
        self.settings = dict(settings or {})
        self.batch_size = batch_size
        self.fail_after = fail_after
        self.latency_seconds = latency_seconds
        self.stop_reason = stop_reason
        self.usage_source = usage_source
        self.asked: list[str] = []

    def _reply(self, request: PredictionRequest) -> str:
        if callable(self._replies):
            return self._replies(request)
        try:
            return self._replies[request.example_id]
        except KeyError:
            raise PredictorError(
                f"the fake predictor has no reply for {request.example_id}"
            ) from None

    def predict(self, requests: Sequence[PredictionRequest]) -> list[Prediction]:
        if self.fail_after is not None and len(self.asked) + len(requests) > self.fail_after:
            raise PredictorError(f"the fake predictor fails after {self.fail_after} requests")
        predictions = []
        for request in requests:
            text = self._reply(request)
            predictions.append(
                Prediction(
                    text=text,
                    input_tokens=len(request.system.split()) + len(request.user.split()),
                    output_tokens=len(text.split()) if isinstance(text, str) else 0,
                    latency_seconds=self.latency_seconds,
                    stop_reason=self.stop_reason,
                )
            )
        self.asked.extend(request.example_id for request in requests)
        return predictions


def gold_predictor(records: Iterable[Any], **options: Any) -> FakePredictor:
    """A predictor that gives each example its own gold reply.

    ``records`` are canonical records (``serialize_split``). Options go to ``FakePredictor``.
    """
    options.setdefault("name", "gold")
    return FakePredictor({record.example_id: record.target for record in records}, **options)
