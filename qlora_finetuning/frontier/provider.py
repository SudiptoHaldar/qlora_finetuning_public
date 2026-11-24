"""The ``FrontierProvider`` interface: what Path B needs from a hosted-model service (req_004).

It has one method in v4.01, the one that exists: ask a model for a reply. The other steps of
the hosted path (stage a training file, start a job, poll it, deploy the tuned model) are
added by the versions that build them.

Usage::

    from qlora_finetuning.frontier.provider import ProviderReply

    class StubProvider:
        def converse(self, model_id, system, user, max_tokens):
            return ProviderReply(text="PROGRAM: add(1, 2)\\nANSWER: 3", input_tokens=40,
                                 output_tokens=12, stop_reason="end", latency_seconds=0.2)

The Bedrock implementation is ``qlora_finetuning.frontier.bedrock.BedrockProvider``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ProviderReply:
    """One reply from a hosted model.

    ``text`` is the reply as the model wrote it. The token counts are the service's own.
    ``stop_reason`` is already one of the runner's three: ``end``, ``length``, ``other``.
    """

    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    latency_seconds: float


@runtime_checkable
class FrontierProvider(Protocol):
    """A hosted-model service. A failure raises ``PredictorError``."""

    def converse(self, model_id: str, system: str, user: str, max_tokens: int) -> ProviderReply:
        """Send the two texts unchanged and return the model's reply."""
        ...
