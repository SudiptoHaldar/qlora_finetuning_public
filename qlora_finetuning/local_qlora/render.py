"""What Qwen reads and what it wrote: the prompt renderer and the reply reader (req_003 v3.01).

``render_chat`` wraps the canonical system and user texts in the model's chat template with
thinking switched off. It adds structure and no word. ``read_generation`` turns one row of
generated token ids into the reply text, its token count, and why it stopped.

Usage::

    from qlora_finetuning.local_qlora.render import read_generation, render_chat

    prompt = render_chat(record.system, record.user, tokenizer, budget.thinking_off_suffix)
    reply = read_generation(new_token_ids, tokenizer, end_ids, limit=192)
    reply.text, reply.output_tokens, reply.stop_reason      # "end" or "length"

For training (req_003 v3.02), ``encode_example`` adds the reply to that same prompt::

    example = encode_example(record, tokenizer, budget.thinking_off_suffix)
    example.input_ids, example.labels      # labels are -100 over the prompt
    rows = encode_records(records, tokenizer, budget.thinking_off_suffix, budget.max_length)

Everything takes the tokenizer as an argument, so this module imports no part of the
training stack and a test can hand over a stand-in.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

from qlora_finetuning.evaluation.predictor import END, LENGTH
from qlora_finetuning.settings import ConfigError

IGNORE = -100  # the label of a token that carries no loss

# A complete thinking block at the very start of a reply, with the blank lines after it.
# The two markers are ordinary tokens for Qwen3, so decoding does not remove them.
_THINK_BLOCK = re.compile(r"\s*<think>.*?</think>\s*", re.DOTALL)


class RenderError(ValueError):
    """The chat template did not produce a prompt with thinking switched off."""


@dataclass(frozen=True)
class Generation:
    """One reply, read from its token ids."""

    text: str
    output_tokens: int  # up to and including the end token
    stop_reason: str  # "end": the model finished; "length": cut at the reply limit
    think_removed: bool = False  # a thinking block was taken off the start


def render_chat(system: str, user: str, tokenizer: Any, thinking_off_suffix: str) -> str:
    """The prompt as the model receives it: chat template, reply marker, thinking off.

    The two texts go in unchanged. ``thinking_off_suffix`` is what the template must end a
    prompt with when thinking is off (``config/token_budget.yaml``); a prompt that ends
    otherwise is refused, because a changed template would silently turn thinking back on.
    """
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not thinking_off_suffix or not text.endswith(thinking_off_suffix):
        raise RenderError(
            "the chat template did not end the prompt with the thinking-off block of "
            "config/token_budget.yaml (tokenizer.thinking_off_suffix)"
        )
    return text


def read_generation(
    token_ids: Sequence[int], tokenizer: Any, end_ids: Collection[int], limit: int
) -> Generation:
    """Read one row of newly generated ids. ``limit`` is the reply limit in tokens.

    The reply ends at the first end token: in a batch, a row that finished early is padded
    after it. A row without one ran to the limit. The text is the tokens before the end
    token, decoded with special tokens removed and otherwise as the model wrote them, except
    that one complete ``<think>...</think>`` block at the very start is taken off. An
    unclosed ``<think>`` stays, for the reply parser to fault.
    """
    ids = list(token_ids)[:limit]
    cut = next((index for index, token in enumerate(ids) if token in end_ids), None)
    body = ids if cut is None else ids[:cut]
    text = tokenizer.decode(body, skip_special_tokens=True)
    match = _THINK_BLOCK.match(text)
    if match:
        text = text[match.end() :]
    return Generation(
        text=text,
        output_tokens=len(body) + (0 if cut is None else 1),
        stop_reason=LENGTH if cut is None else END,
        think_removed=match is not None,
    )


# --------------------------------------------------------------------------- training sequences


class LengthError(ConfigError):
    """A training sequence is longer than the training length. Nothing is ever cut to fit."""


@dataclass(frozen=True)
class EncodedExample:
    """One training sequence: the prompt, the reply, the end token, and where the loss is."""

    example_id: str
    input_ids: list[int]
    labels: list[int]  # -100 over the prompt; the token ids over the reply and the end token
    prompt_tokens: int

    @property
    def length(self) -> int:
        return len(self.input_ids)

    @property
    def target_tokens(self) -> int:
        """The reply with its end token: the part that carries the loss."""
        return len(self.input_ids) - self.prompt_tokens


def encode_example(record: Any, tokenizer: Any, thinking_off_suffix: str) -> EncodedExample:
    """A canonical record as the sequence Path A trains on (req_003 v3.02).

    The ids of the prompt of ``render_chat``, then the ids of the target followed by the end
    token. The two parts are tokenized separately, so no token can merge across the boundary
    and the length is the one the token budget counted. Only the reply is labelled: the
    model is not trained to repeat its prompt, and it is trained to stop.
    """
    prompt_text = render_chat(record.system, record.user, tokenizer, thinking_off_suffix)
    prompt = list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    reply = list(
        tokenizer(record.target + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    )
    return EncodedExample(
        example_id=record.example_id,
        input_ids=prompt + reply,
        labels=[IGNORE] * len(prompt) + reply,
        prompt_tokens=len(prompt),
    )


def encode_records(
    records: Sequence[Any], tokenizer: Any, thinking_off_suffix: str, max_length: int
) -> list[EncodedExample]:
    """Every record encoded, in order. Raises ``LengthError`` naming each one that is too long."""
    encoded = [encode_example(record, tokenizer, thinking_off_suffix) for record in records]
    too_long = [example for example in encoded if example.length > max_length]
    if too_long:
        shown = ", ".join(f"{e.example_id} ({e.length:,})" for e in too_long[:5])
        more = f", and {len(too_long) - 5:,} more" if len(too_long) > 5 else ""
        raise LengthError(
            f"{len(too_long):,} training sequences are longer than max_length {max_length:,} "
            f"of config/token_budget.yaml: {shown}{more}. Nothing is truncated: raise the "
            "limit or decide, for both paths, to leave the records out"
        )
    return encoded
