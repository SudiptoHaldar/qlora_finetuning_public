"""Token counts of the canonical records (req_001 v1.04).

A record is counted as the **full sequence** Path A trains on: the tokenizer's own chat
template over the system and user texts, with the generation prompt and thinking switched
off, then the target, then the end token. Nothing is truncated anywhere: a record longer than
``max_length`` in ``config/token_budget.yaml`` is an error, not something to cut.

Only the tokenizer is loaded: no model weights and no GPU.

Usage::

    from qlora_finetuning.data import serialize_split
    from qlora_finetuning.data.token_budget import (
        count_split, load_token_budget_config, load_tokenizer,
    )

    budget = load_token_budget_config()
    tokenizer = load_tokenizer(budget)             # from the local Hugging Face cache
    counts = count_split(serialize_split("fit"), tokenizer, budget.thinking_off_suffix)
    counts[0].total, counts[0].prompt, counts[0].target
    budget.max_length, budget.max_new_tokens       # the limits both paths read

The renderer used for models belongs to req_003 v3.01. For the same record it must produce
the token counts given here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qlora_finetuning.data.serialization import CanonicalRecord
from qlora_finetuning.settings import REPO_ROOT, ConfigError, load_config

REQUIRED_KEYS = (
    "tokenizer.model_id",
    "tokenizer.revision",
    "tokenizer.fingerprint",
    "tokenizer.thinking_off_suffix",
    "max_length",
    "max_new_tokens",
    "thresholds",
    "longest_listed",
    "report_dir",
)
TOKENS_COMMAND = "python -m scripts.finqa_tokens"
INSTALL_HINT = "install the training stack: docs/runbook.md section 3.1"
THINK_OPEN = "<think>"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class TokenBudgetError(ConfigError):
    """The tokenizer cannot be loaded, or is not the pinned one."""


@dataclass(frozen=True)
class TokenBudgetConfig:
    """``config/token_budget.yaml`` with its path resolved."""

    root: Path
    model_id: str
    revision: str
    fingerprint: str
    thinking_off_suffix: str
    max_length: int
    max_new_tokens: int
    thresholds: tuple[int, ...]
    longest_listed: int
    report_dir: Path


@dataclass(frozen=True)
class TokenCounts:
    """Token counts of one record."""

    example_id: str
    system: int  # the system text alone
    user: int  # the user text alone
    target: int  # the target with the end token
    prompt: int  # the chat-templated system and user texts, ready for the reply
    thinking_off: bool  # the prompt ends with the empty think block and the target has none

    @property
    def total(self) -> int:
        """The full training sequence: prompt, target, end token."""
        return self.prompt + self.target

    @property
    def template(self) -> int:
        """What the chat template adds around the two texts."""
        return self.prompt - self.system - self.user


# --------------------------------------------------------------------------- config


def _positive(raw: dict[str, Any], key: str) -> int:
    value = raw[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError(
            f"config token_budget.yaml: {key} must be a positive whole number, got {value!r}"
        )
    return value


def load_token_budget_config(
    root: Path | str = REPO_ROOT, config_dir: Path | str | None = None
) -> TokenBudgetConfig:
    """Read ``config/token_budget.yaml``. Relative paths in it resolve under ``root``."""
    raw = load_config("token_budget", required=REQUIRED_KEYS, config_dir=config_dir)
    tokenizer = raw["tokenizer"]
    revision, fingerprint = str(tokenizer["revision"]), str(tokenizer["fingerprint"])
    if not _COMMIT.fullmatch(revision):
        raise ConfigError(
            "config token_budget.yaml: tokenizer.revision must be a full 40-character commit "
            f"hash, not a branch name or a short hash: {revision!r}"
        )
    if not _SHA256.fullmatch(fingerprint):
        raise ConfigError(
            "config token_budget.yaml: tokenizer.fingerprint must be 64 lower-case hex characters"
        )
    thresholds = raw["thresholds"]
    if not isinstance(thresholds, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in thresholds
    ):
        raise ConfigError(
            "config token_budget.yaml: thresholds must be a list of positive whole numbers"
        )
    root = Path(root)
    return TokenBudgetConfig(
        root=root,
        model_id=str(tokenizer["model_id"]),
        revision=revision,
        fingerprint=fingerprint,
        thinking_off_suffix=str(tokenizer["thinking_off_suffix"]),
        max_length=_positive(raw, "max_length"),
        max_new_tokens=_positive(raw, "max_new_tokens"),
        thresholds=tuple(sorted(set(thresholds))),
        longest_listed=_positive(raw, "longest_listed"),
        report_dir=root / str(raw["report_dir"]),
    )


# --------------------------------------------------------------------------- tokenizer


def load_tokenizer(cfg: TokenBudgetConfig, download: bool = False) -> Any:
    """Load the pinned tokenizer from the local Hugging Face cache.

    ``transformers`` is part of the training stack, so it is imported here. With ``download``
    the files may be fetched (a few megabytes, once). No model weights are loaded.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise TokenBudgetError(f"transformers is not installed; {INSTALL_HINT}") from exc
    try:
        return AutoTokenizer.from_pretrained(
            cfg.model_id, revision=cfg.revision, local_files_only=not download
        )
    except Exception as exc:  # missing files, no network: the library raises several kinds
        first_line = (str(exc).strip().splitlines() or ["(no message)"])[0]
        hint = (
            "check the network connection and the tokenizer pin"
            if download
            else f"fetch it once with: {TOKENS_COMMAND} --download"
        )
        raise TokenBudgetError(
            f"the tokenizer of {cfg.model_id} at {cfg.revision[:12]}... could not be loaded "
            f"({type(exc).__name__}: {first_line}); {hint}"
        ) from exc


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """SHA-256 over the vocabulary, the chat template, and the end token.

    Two tokenizers with the same fingerprint split every text the same way and wrap a chat
    the same way, which is what the budget depends on.
    """
    payload = json.dumps(
        [sorted(tokenizer.get_vocab().items()), tokenizer.chat_template, tokenizer.eos_token],
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# --------------------------------------------------------------------------- counting


def render_prompt(record: CanonicalRecord, tokenizer: Any) -> str:
    """The prompt as the model receives it: chat template, reply marker, thinking off."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": record.system},
            {"role": "user", "content": record.user},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _lengths(tokenizer: Any, texts: Sequence[str]) -> list[int]:
    if not texts:
        return []
    return [len(ids) for ids in tokenizer(list(texts), add_special_tokens=False)["input_ids"]]


def count_split(
    records: Sequence[CanonicalRecord], tokenizer: Any, thinking_off_suffix: str
) -> list[TokenCounts]:
    """Token counts for every record, in order. Texts are tokenized in batches."""
    prompts = [render_prompt(record, tokenizer) for record in records]
    systems = _lengths(tokenizer, [record.system for record in records])
    users = _lengths(tokenizer, [record.user for record in records])
    targets = _lengths(tokenizer, [record.target + tokenizer.eos_token for record in records])
    prompt_lengths = _lengths(tokenizer, prompts)
    return [
        TokenCounts(
            example_id=record.example_id,
            system=systems[i],
            user=users[i],
            target=targets[i],
            prompt=prompt_lengths[i],
            thinking_off=prompts[i].endswith(thinking_off_suffix)
            and THINK_OPEN not in record.target,
        )
        for i, record in enumerate(records)
    ]


def count_record(record: CanonicalRecord, tokenizer: Any, thinking_off_suffix: str) -> TokenCounts:
    """Token counts of one record."""
    return count_split([record], tokenizer, thinking_off_suffix)[0]
