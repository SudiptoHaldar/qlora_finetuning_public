"""The safety limits of the DSL, from ``config/dsl.yaml`` (req_002 v2.01).

Usage::

    from qlora_finetuning.dsl.config import load_dsl_config

    config = load_dsl_config()
    config.max_reply_chars   # 4000
    config.max_steps         # 32
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qlora_finetuning.settings import ConfigError, load_config

REQUIRED_KEYS = ("max_reply_chars", "max_steps")


@dataclass(frozen=True)
class DslConfig:
    """Limits checked before any other work on a reply."""

    max_reply_chars: int
    max_steps: int

    def __post_init__(self) -> None:
        for key in REQUIRED_KEYS:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(f"config dsl.yaml: {key} must be a positive whole number")


def load_dsl_config(config_dir: Path | str | None = None) -> DslConfig:
    """Read and validate ``config/dsl.yaml``."""
    raw = load_config("dsl", required=REQUIRED_KEYS, config_dir=config_dir)
    return DslConfig(max_reply_chars=raw["max_reply_chars"], max_steps=raw["max_steps"])
