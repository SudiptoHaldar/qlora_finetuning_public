"""Path A settings: ``config/local_qlora.yaml`` (req_003 v3.01).

How the Qwen model is loaded (4-bit quantization) and how it generates (greedy, in batches,
stopping at its own end tokens). The model and its commit are not here: they are pinned once,
in ``config/token_budget.yaml``.

Usage::

    from qlora_finetuning.local_qlora.config import load_local_qlora_config

    cfg = load_local_qlora_config()
    cfg.batch_size, cfg.end_tokens
    cfg.settings(revision="1cfa9a72...")     # everything that can change a reply

This module imports no part of the training stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qlora_finetuning.settings import ConfigError, load_config

REQUIRED_KEYS = (
    "predictor_name",
    "quantization.type",
    "quantization.double_quant",
    "quantization.compute_dtype",
    "inference.batch_size",
    "inference.do_sample",
    "inference.num_beams",
    "inference.end_tokens",
)
QUANTIZATION_TYPES = ("nf4", "fp4")  # what bitsandbytes offers for 4-bit
COMPUTE_DTYPES = ("bfloat16", "float16")
FILE = "config local_qlora.yaml"


@dataclass(frozen=True)
class LocalQloraConfig:
    """The typed content of ``config/local_qlora.yaml``."""

    predictor_name: str
    quantization_type: str
    double_quant: bool
    compute_dtype: str
    batch_size: int
    end_tokens: tuple[str, ...]

    def settings(self, revision: str, adapter: str | None = None) -> dict[str, Any]:
        """Everything that can change a reply. A run is resumed only under equal settings.

        ``revision`` is the model's commit. ``adapter`` is the SHA-256 of an adapter's
        weights file, or ``None`` for the untuned model: never a path, which differs between
        this machine and a cloud job.
        """
        return {
            "revision": revision,
            "quantization": {
                "type": self.quantization_type,
                "double_quant": self.double_quant,
                "compute_dtype": self.compute_dtype,
            },
            "enable_thinking": False,
            "do_sample": False,
            "num_beams": 1,
            "stop": "end tokens",
            "end_tokens": list(self.end_tokens),
            "batch_size": self.batch_size,
            "adapter": adapter,
        }


def _is_whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_local_qlora_config(config_dir: Path | str | None = None) -> LocalQloraConfig:
    """Read and check ``config/local_qlora.yaml``. A problem is a ``ConfigError`` naming it."""
    raw = load_config("local_qlora", required=REQUIRED_KEYS, config_dir=config_dir)
    quantization, inference = raw["quantization"], raw["inference"]

    name = raw["predictor_name"]
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{FILE}: predictor_name must be text")
    if quantization["type"] not in QUANTIZATION_TYPES:
        raise ConfigError(
            f"{FILE}: quantization.type must be one of {', '.join(QUANTIZATION_TYPES)}, "
            f"got {quantization['type']!r}"
        )
    if not isinstance(quantization["double_quant"], bool):
        raise ConfigError(f"{FILE}: quantization.double_quant must be true or false")
    if quantization["compute_dtype"] not in COMPUTE_DTYPES:
        raise ConfigError(
            f"{FILE}: quantization.compute_dtype must be one of {', '.join(COMPUTE_DTYPES)}, "
            f"got {quantization['compute_dtype']!r}"
        )

    batch_size = inference["batch_size"]
    if not _is_whole(batch_size) or batch_size < 1:
        raise ConfigError(
            f"{FILE}: inference.batch_size must be a positive whole number, got {batch_size!r}"
        )
    # The file describes greedy decoding and nothing else: every arm must repeat.
    if inference["do_sample"] is not False:
        raise ConfigError(
            f"{FILE}: inference.do_sample must be false: every arm decodes greedily (plan 4.6)"
        )
    if not _is_whole(inference["num_beams"]) or inference["num_beams"] != 1:
        raise ConfigError(f"{FILE}: inference.num_beams must be 1: there is no beam search")
    end_tokens = inference["end_tokens"]
    if (
        not isinstance(end_tokens, list)
        or not end_tokens
        or any(not isinstance(token, str) or not token for token in end_tokens)
    ):
        raise ConfigError(f"{FILE}: inference.end_tokens must be a list of tokens, as text")

    return LocalQloraConfig(
        predictor_name=name.strip(),
        quantization_type=quantization["type"],
        double_quant=quantization["double_quant"],
        compute_dtype=quantization["compute_dtype"],
        batch_size=batch_size,
        end_tokens=tuple(end_tokens),
    )
