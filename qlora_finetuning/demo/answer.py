"""Ask a model one question and run its reply: the demo's one step.

``answer`` joins four things that exist elsewhere and adds none of its own:

- the prompt: ``load_prompt().system`` and ``format_user(...)``, unchanged;
- the model: any ``Predictor``, asked with one request, so in a batch of one;
- the reply: ``run_reply(text, table)``, the only place reply text is read or executed;
- the comparison: ``answer_agrees``, in the metrics.

It has no gold answer and computes no metric; it starts no run and writes nothing.

Usage::

    from qlora_finetuning.demo.answer import answer
    from qlora_finetuning.demo.inputs import load_input, sample_path

    result = answer(predictor, load_input(sample_path()))
    result.outcome.program      # the program the model proposed, in canonical form
    result.outcome.value        # what it gives when executed, not rounded
    result.agrees               # True, False, or None when there is nothing to compare
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qlora_finetuning.data.serialization import PromptDefinition, format_user, load_prompt
from qlora_finetuning.data.token_budget import TokenBudgetConfig, load_token_budget_config
from qlora_finetuning.demo.inputs import DemoInput
from qlora_finetuning.dsl import ReplyFormat, ReplyOutcome, run_reply
from qlora_finetuning.evaluation.metrics import (
    EvaluationConfig,
    answer_agrees,
    load_evaluation_config,
)
from qlora_finetuning.evaluation.predictor import PredictionRequest, PredictorError


@dataclass(frozen=True)
class DemoResult:
    """One input, the model's reply to it, and what running the reply gave."""

    demo_input: DemoInput
    predictor: str  # the predictor's name, e.g. qwen3-4b-tuned
    adapter: str | None  # RUN_ID or RUN_ID/checkpoint-N; None for the untuned model
    reply: str
    outcome: ReplyOutcome
    agrees: bool | None  # the model's answer against its own program's result
    input_tokens: int | None
    output_tokens: int | None
    stop_reason: str
    seconds: float
    max_new_tokens: int
    max_length: int

    @property
    def over_length(self) -> bool:
        """True when prompt and reply limit together exceed the training length."""
        if self.input_tokens is None:
            return False
        return self.input_tokens + self.max_new_tokens > self.max_length


def adapter_name(predictor: Any) -> str | None:
    """The adapter a tuned predictor carries, as it is named on the command line."""
    details = getattr(predictor, "details", None) or {}
    run = details.get("training_run")
    if not run:
        return None
    step = details.get("checkpoint")
    return str(run) if step is None else f"{run}/checkpoint-{step}"


def answer(
    predictor: Any,
    demo_input: DemoInput,
    *,
    prompt: PromptDefinition | None = None,
    budget: TokenBudgetConfig | None = None,
    evaluation_config: EvaluationConfig | None = None,
) -> DemoResult:
    """Ask ``predictor`` the input's question and run the reply on the input's table.

    The settings left as ``None`` are read from ``config/``. Raises ``PredictorError``
    when the model cannot answer, and a ``ConfigError`` when the input cannot be written
    as a prompt. A program that does not run is not an error here: it is in the outcome.
    """
    prompt = prompt or load_prompt()
    budget = budget or load_token_budget_config()
    evaluation_config = evaluation_config or load_evaluation_config()
    table = [list(row) for row in demo_input.table]

    request = PredictionRequest(
        example_id=demo_input.source,
        system=prompt.system,
        user=format_user(
            demo_input.passages, table, demo_input.question, prompt, f"input {demo_input.source}"
        ),
        max_new_tokens=budget.max_new_tokens,
    )
    predictions = predictor.predict([request])
    if len(predictions) != 1:
        raise PredictorError(f"the model gave {len(predictions)} replies to one question")
    prediction = predictions[0]

    outcome = run_reply(prediction.text, table, ReplyFormat.from_prompt(prompt))
    return DemoResult(
        demo_input=demo_input,
        predictor=str(predictor.name),
        adapter=adapter_name(predictor),
        reply=prediction.text,
        outcome=outcome,
        agrees=answer_agrees(outcome, evaluation_config),
        input_tokens=prediction.input_tokens,
        output_tokens=prediction.output_tokens,
        stop_reason=prediction.stop_reason,
        seconds=prediction.latency_seconds,
        max_new_tokens=budget.max_new_tokens,
        max_length=budget.max_length,
    )
