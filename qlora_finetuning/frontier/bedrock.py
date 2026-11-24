"""Amazon Bedrock behind the ``FrontierProvider`` interface: the Converse API (req_004 v4.01).

One call is one reply. The request holds the system text and the user text unchanged, the
reply limit, and greedy decoding; nothing else is sent. Retrying is botocore's own. When it
gives up, the provider raises ``PredictorError``, the evaluation run stops, and it can be
resumed.

Usage::

    from qlora_finetuning.frontier.bedrock import BedrockProvider, bedrock_runtime_client
    from qlora_finetuning.frontier.config import load_aws_settings, load_frontier_config

    cfg, aws = load_frontier_config(), load_aws_settings()
    provider = BedrockProvider(bedrock_runtime_client(aws, cfg), cfg, profile=aws["profile"])
    reply = provider.converse("amazon.nova-lite-v1:0", system, user, max_tokens=192)
    reply.text, reply.input_tokens, reply.output_tokens, reply.stop_reason

Every call to ``converse`` is billed, so it is made by the owner's runs only. This is the
only module of the package that imports boto3 or botocore; tests hand the provider a stub
client.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
)

from qlora_finetuning.aws_common import (  # noqa: F401  (re-exported: Path B's callers)
    AUTH_ERROR_CODES,
    AUTH_EXCEPTIONS,
    aws_session,
    check_identity,
    sign_in_message,
)
from qlora_finetuning.evaluation.predictor import END, LENGTH, OTHER, PredictorError
from qlora_finetuning.frontier.config import FrontierConfig
from qlora_finetuning.frontier.provider import ProviderReply
from qlora_finetuning.logging_setup import get_logger
from qlora_finetuning.settings import ConfigError

# Converse stop reasons. Anything not named here is "other": the service's own answer to
# that prompt (a content filter, a malformed output), which a retry would not change.
STOP_REASONS = {"end_turn": END, "stop_sequence": END, "max_tokens": LENGTH}
CACHE_COUNTS = ("cacheReadInputTokens", "cacheWriteInputTokens")

logger = get_logger(__name__)


def client_config(cfg: FrontierConfig) -> Config:
    """The botocore settings of ``config/frontier.yaml``. ``max_attempts`` counts the first try."""
    return Config(
        retries={"total_max_attempts": cfg.max_attempts, "mode": cfg.retry_mode},
        connect_timeout=cfg.connect_timeout,
        read_timeout=cfg.read_timeout,
    )


def bedrock_runtime_client(aws: dict[str, str], cfg: FrontierConfig, session: Any = None) -> Any:
    """The real ``bedrock-runtime`` client. The only place that creates one."""
    session = session or aws_session(aws)
    return session.client("bedrock-runtime", config=client_config(cfg))


def bedrock_control_client(aws: dict[str, str], cfg: FrontierConfig, session: Any = None) -> Any:
    """The real ``bedrock`` control-plane client: jobs, tuned models, deployments (v4.03)."""
    session = session or aws_session(aws)
    return session.client("bedrock", config=client_config(cfg))


def deployment_status(control: Any, deployment_arn: str, profile: str = "qlora") -> str | None:
    """What Bedrock says the status of an on-demand deployment is; ``None`` if it has none.

    One read on the control-plane client, free. It is asked before a tuned model's run
    starts (v4.04), so a deployment that was deleted or is still being created is a
    ``ConfigError`` there and not a failed, paid run.
    """
    try:
        response = control.get_custom_model_deployment(
            customModelDeploymentIdentifier=deployment_arn
        )
    except AUTH_EXCEPTIONS:
        raise ConfigError(sign_in_message(profile)) from None
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code", "ClientError")
        if code in AUTH_ERROR_CODES:
            raise ConfigError(sign_in_message(profile)) from None
        if code == "ResourceNotFoundException":
            return None
        raise ConfigError(f"AWS answered {code} when asked about the deployment") from None
    except BotoCoreError as error:
        raise ConfigError(
            f"AWS could not be reached to read the deployment: {type(error).__name__}"
        ) from None
    status = response.get("status") if isinstance(response, dict) else None
    return status if isinstance(status, str) else None


def build_request(
    model_id: str, system: str, user: str, max_tokens: int, cfg: FrontierConfig
) -> dict[str, Any]:
    """The Converse request: the two texts unchanged, the reply limit, greedy decoding."""
    return {
        "modelId": model_id,
        "system": [{"text": system}],
        "messages": [{"role": "user", "content": [{"text": user}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": cfg.temperature},
        "additionalModelRequestFields": {"inferenceConfig": {"topK": cfg.top_k}},
    }


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def read_response(response: Any, latency_seconds: float) -> ProviderReply:
    """Turn a Converse response into a reply, or raise ``PredictorError`` if it is not one.

    An empty reply is a reply. A response without a message, a stop reason, or whole token
    counts is a failure, and so is one that reports cached tokens: this project sends no
    cache point, so ``inputTokens`` must be every input token.
    """
    if not isinstance(response, dict):
        raise PredictorError("Bedrock returned something that is not a response")
    message = (response.get("output") or {}).get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise PredictorError("the Bedrock response has no assistant message")
    content = message.get("content")
    if not isinstance(content, list):
        raise PredictorError("the Bedrock response has no content list")
    blocks = [block for block in content if isinstance(block, dict)]
    text = "".join(block["text"] for block in blocks if isinstance(block.get("text"), str))
    reasoning = sum("reasoningContent" in block for block in blocks)
    if reasoning:
        logger.warning("%d reasoning block(s) were left out of a reply", reasoning)

    stop = response.get("stopReason")
    if not isinstance(stop, str) or not stop:
        raise PredictorError("the Bedrock response has no stop reason")
    usage = response.get("usage")
    if not isinstance(usage, dict) or not all(
        _whole(usage.get(name)) for name in ("inputTokens", "outputTokens")
    ):
        raise PredictorError("the Bedrock response has no whole token counts")
    cached = [name for name in CACHE_COUNTS if usage.get(name)]
    if cached:
        raise PredictorError(
            f"the Bedrock response reports cached tokens ({', '.join(cached)}), so its "
            "input token count is not the whole prompt"
        )
    return ProviderReply(
        text=text,
        input_tokens=usage["inputTokens"],
        output_tokens=usage["outputTokens"],
        stop_reason=STOP_REASONS.get(stop, OTHER),
        latency_seconds=latency_seconds,
    )


class BedrockProvider:
    """``FrontierProvider`` over a ``bedrock-runtime`` client."""

    def __init__(
        self,
        client: Any,
        cfg: FrontierConfig,
        profile: str = "qlora",
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.client = client
        self.cfg = cfg
        self.profile = profile
        self._clock = clock

    def converse(self, model_id: str, system: str, user: str, max_tokens: int) -> ProviderReply:
        """One billed call. Raises ``PredictorError`` with the error code, never with text.

        The latency is our own clock around the call, including botocore's retries and waits.
        """
        request = build_request(model_id, system, user, max_tokens, self.cfg)
        started = self._clock()
        try:
            response = self.client.converse(**request)
        except AUTH_EXCEPTIONS:
            raise PredictorError(sign_in_message(self.profile)) from None
        except ClientError as error:
            details = error.response.get("Error", {})
            code = details.get("Code", "ClientError")
            if code in AUTH_ERROR_CODES:
                raise PredictorError(sign_in_message(self.profile)) from None
            # The service's message goes to the log only: an error is stored with the run.
            logger.error("Bedrock %s for %s: %s", code, model_id, details.get("Message", ""))
            raise PredictorError(
                f"Bedrock answered {code} for {model_id} (after botocore's retries)"
            ) from None
        except BotoCoreError as error:
            raise PredictorError(
                f"Bedrock could not be reached for {model_id}: {type(error).__name__}"
            ) from None
        return read_response(response, self._clock() - started)
