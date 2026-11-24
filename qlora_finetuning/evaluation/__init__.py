"""Metrics, Predictor protocol, and evaluation runner (req_002).

- ``metrics`` (v2.02): the comparison rule and the four metrics.
- ``predictor`` (v2.03): the protocol a model backend implements.
- ``fake`` (v2.03): a predictor without a model, for tests and for proving the runner.
- ``predictions`` (v2.03): the predictions file, one line per example.
- ``runner`` (v2.03): ``run_evaluation`` and ``summarize_run``. **Import it by its own
  name**, ``qlora_finetuning.evaluation.runner``: it is not exported here, because it imports
  the data package, whose gold self-check (``qlora_finetuning.data.gold_check``) imports the
  metrics from this package.

Usage::

    from qlora_finetuning.evaluation import Gold, load_evaluation_config, score_reply, summarize
    from qlora_finetuning.evaluation import Prediction, PredictorError     # for a backend
    from qlora_finetuning.evaluation.runner import run_evaluation, summarize_run
"""

from qlora_finetuning.evaluation.fake import FakePredictor, gold_predictor
from qlora_finetuning.evaluation.metrics import (
    METRICS,
    MISMATCHES,
    Allowance,
    EvaluationConfig,
    EvaluationError,
    ExampleScore,
    Gold,
    answer_agrees,
    load_evaluation_config,
    mismatch_label,
    numbers_match,
    programs_match,
    read_answer,
    score_reply,
    summarize,
    values_match,
)
from qlora_finetuning.evaluation.predictions import (
    PredictionRecord,
    PredictionsError,
    append_records,
    read_predictions,
)
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

__all__ = [
    "METRICS",
    "MISMATCHES",
    "STOP_REASONS",
    "USAGE_SOURCES",
    "Allowance",
    "EvaluationConfig",
    "EvaluationError",
    "ExampleScore",
    "FakePredictor",
    "Gold",
    "Prediction",
    "PredictionRecord",
    "PredictionRequest",
    "PredictionsError",
    "Predictor",
    "PredictorError",
    "answer_agrees",
    "append_records",
    "check_predictions",
    "describe",
    "gold_predictor",
    "load_evaluation_config",
    "mismatch_label",
    "numbers_match",
    "programs_match",
    "read_answer",
    "read_predictions",
    "score_reply",
    "summarize",
    "values_match",
]
