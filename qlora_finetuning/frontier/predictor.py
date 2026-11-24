"""The Path B predictor: a hosted model behind the evaluation runner (req_004 v4.01).

``FrontierPredictor`` implements the ``Predictor`` protocol over a ``FrontierProvider``. It
has no loop over a split and writes nothing to the ledger: the runner does both, and takes
the token counts from each prediction.

Usage (every real call is billed, so the owner runs it)::

    venv\\Scripts\\python -m scripts.evaluate --predictor nova-lite --split quick_dev

or, in code::

    from qlora_finetuning.evaluation.runner import run_evaluation
    from qlora_finetuning.frontier.predictor import build_predictor

    predictor = build_predictor("nova-lite")      # checks the sign-in; no billed call
    summary = run_evaluation(predictor, "quick_dev", label="nova-lite quick_dev",
                             path="B", arm="B0-Lite", where="bedrock", scope="dev")

A tuned model (req_004 v4.04) is the same predictor with another ``modelId``: the ARN of
the on-demand deployment that the tuning record names. Nothing else about the request
changes. Its usage is recorded under the base model's id, which is what Bedrock bills it
as::

    venv\\Scripts\\python -m scripts.evaluate --predictor nova-lite-tuned --split quick_dev

    predictor = build_tuned_predictor("nova-lite", context=context)
    predictor.model_id          # the deployment's ARN: what is called
    predictor.usage_model_id    # "amazon.nova-lite-v1:0": what it is billed as

``build_tuned_predictor`` refuses unless the deployment is Active.

A model with ``requests_per_minute`` in the config is paced: the predictor leaves at least
60 / that many seconds between the starts of two calls, tuned or untuned, so a run stays
under the account's rate limit instead of retrying against it. Pacing changes when a
question is asked, never the reply, and is not part of a run's settings.

A test gives ``FrontierPredictor`` a stub provider and needs no client.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from qlora_finetuning.data import manifest_hash
from qlora_finetuning.evaluation.predictor import Prediction, PredictionRequest
from qlora_finetuning.frontier.config import (
    FrontierConfig,
    load_aws_settings,
    load_frontier_config,
)
from qlora_finetuning.frontier.provider import FrontierProvider
from qlora_finetuning.frontier.tuned import TUNE, TUNED_SUFFIX, tuned_attempt, tuned_details
from qlora_finetuning.frontier.tuning_record import ACTIVE
from qlora_finetuning.settings import ConfigError


class FrontierPredictor:
    """One hosted model. Calls are sequential: a batch is one request.

    ``batch_size`` is 1 because the runner treats a batch as all or nothing, and with paid
    calls a batch that fails half-way would be paid for twice.

    ``model_id`` is what is called. For a tuned model that is a deployment's ARN, and
    ``usage_model_id`` names the base model it is billed as; ``details`` says which tuned
    model it is. Both are ``None`` for an untuned model.

    With ``requests_per_minute``, the starts of two calls are at least 60 / that many
    seconds apart. ``clock`` and ``sleep`` are replaceable so a test need not wait.
    """

    batch_size = 1
    usage_source = "aws_api"

    def __init__(
        self,
        provider: FrontierProvider,
        model_id: str,
        name: str,
        settings: Mapping[str, Any],
        *,
        usage_model_id: str | None = None,
        details: Mapping[str, Any] | None = None,
        requests_per_minute: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider = provider
        self.model_id = model_id
        self.name = name
        self.settings = dict(settings)
        self.usage_model_id = usage_model_id
        self.details = None if details is None else dict(details)
        self.requests_per_minute = requests_per_minute
        self._interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._clock, self._sleep = clock, sleep
        self._last_start: float | None = None

    def _pace(self) -> None:
        """Wait until the next call may start, then note that it starts."""
        if self._interval and self._last_start is not None:
            wait = self._last_start + self._interval - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last_start = self._clock()

    def predict(self, requests: Sequence[PredictionRequest]) -> list[Prediction]:
        """One prediction per request, in order. A provider failure raises ``PredictorError``."""
        predictions = []
        for request in requests:
            self._pace()
            reply = self.provider.converse(
                self.model_id, request.system, request.user, request.max_new_tokens
            )
            predictions.append(
                Prediction(
                    text=reply.text,
                    input_tokens=reply.input_tokens,
                    output_tokens=reply.output_tokens,
                    latency_seconds=reply.latency_seconds,
                    stop_reason=reply.stop_reason,
                )
            )
        return predictions


def build_predictor(
    model_name: str,
    *,
    cfg: FrontierConfig | None = None,
    aws: dict[str, str] | None = None,
    session: Any = None,
) -> FrontierPredictor:
    """The predictor for a model of ``config/frontier.yaml``, over the real Bedrock client.

    Before returning, it confirms the sign-in and the account with one free call, so that a
    missing sign-in is a ``ConfigError`` here and not a failed run later.
    """
    # boto3 is loaded only when a hosted model is asked for.
    from qlora_finetuning.frontier.bedrock import (
        BedrockProvider,
        aws_session,
        bedrock_runtime_client,
        check_identity,
        client_config,
    )

    cfg = cfg or load_frontier_config()
    aws = aws or load_aws_settings()
    model = cfg.model(model_name)
    session = session or aws_session(aws)
    check_identity(session.client("sts", config=client_config(cfg)), aws)
    client = bedrock_runtime_client(aws, cfg, session=session)
    provider = BedrockProvider(client, cfg, profile=aws["profile"])
    return FrontierPredictor(
        provider,
        model.model_id,
        model.name,
        cfg.settings,
        requests_per_minute=model.requests_per_minute,
    )


def build_tuned_predictor(
    model_name: str,
    *,
    context: Any,
    cfg: FrontierConfig | None = None,
    aws: dict[str, str] | None = None,
    session: Any = None,
) -> FrontierPredictor:
    """The predictor for the tuned model of ``model_name``, over the real Bedrock client.

    ``context`` is the runner's context: it carries the prompt and the split. The tuning
    record and the gate record are checked first, which needs no sign-in. Then the sign-in
    is confirmed and Bedrock is asked once whether the deployment is Active. No billed call
    is made. Every refusal is a ``ConfigError`` that names the command to run.
    """
    cfg = cfg or load_frontier_config()
    attempt = tuned_attempt(
        model_name, cfg, context.prompt.prompt_id, manifest_hash(context.split_cfg)
    )

    # boto3 is loaded only when a hosted model is asked for.
    from qlora_finetuning.frontier.bedrock import (
        BedrockProvider,
        aws_session,
        bedrock_control_client,
        bedrock_runtime_client,
        check_identity,
        client_config,
        deployment_status,
    )

    aws = aws or load_aws_settings()
    session = session or aws_session(aws)
    check_identity(session.client("sts", config=client_config(cfg)), aws)
    control = bedrock_control_client(aws, cfg, session=session)
    status = deployment_status(control, attempt.deployment_arn, aws["profile"])
    if status != ACTIVE:
        raise ConfigError(
            f"the deployment of {model_name} (attempt {attempt.attempt}) is "
            f"{status or 'not there'} according to Bedrock, not {ACTIVE}; run {TUNE} status"
        )
    client = bedrock_runtime_client(aws, cfg, session=session)
    provider = BedrockProvider(client, cfg, profile=aws["profile"])
    return FrontierPredictor(
        provider,
        attempt.deployment_arn,
        f"{model_name}{TUNED_SUFFIX}",
        cfg.settings,
        usage_model_id=attempt.base_model_id,
        details=tuned_details(attempt),
        requests_per_minute=cfg.model(model_name).requests_per_minute,
    )


def nova_pro(split: str, context: Any) -> FrontierPredictor:
    """Untuned Amazon Nova Pro, for ``scripts.evaluate --predictor nova-pro``."""
    return build_predictor("nova-pro")


def nova_lite(split: str, context: Any) -> FrontierPredictor:
    """Untuned Amazon Nova Lite, for ``scripts.evaluate --predictor nova-lite``."""
    return build_predictor("nova-lite")


def nova_pro_tuned(split: str, context: Any) -> FrontierPredictor:
    """Fine-tuned Amazon Nova Pro, for ``scripts.evaluate --predictor nova-pro-tuned``."""
    return build_tuned_predictor("nova-pro", context=context)


def nova_lite_tuned(split: str, context: Any) -> FrontierPredictor:
    """Fine-tuned Amazon Nova Lite, for ``scripts.evaluate --predictor nova-lite-tuned``."""
    return build_tuned_predictor("nova-lite", context=context)
