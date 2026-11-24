"""Merging the adapter into the base model's weights (req_006 v6.02, spec D7 to D9).

The adapter is added to the **full-precision** base model, never to the 4-bit one. The sum
is formed in 32-bit floats and saved in the type the base model is stored in, so the copy
has the base model's size. The copy is saved with greedy generation settings: the base
model's own settings sample, and a plain ``generate()`` on the copy would then give
replies nobody measured.

Usage (on the CPU; about 16 GB of memory for Qwen3-4B, some minutes)::

    from qlora_finetuning.packaging.merge import merge_adapter

    result = merge_adapter("Qwen/Qwen3-4B", adapter_dir, out_dir, revision=commit,
                           end_ids=[151645, 151643], pad_token_id=151643)
    result.saved_dtype, result.changed_tensor, result.libraries

This is the only module of ``packaging/`` that imports the training stack, and it does so
inside the function. The merged copy is **not arm A1**: loaded in 4-bit, it quantizes
other weights than the ones the adapter was trained against.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qlora_finetuning.data.token_budget import INSTALL_HINT
from qlora_finetuning.local_qlora.adapters import ADAPTER_WEIGHTS
from qlora_finetuning.logging_setup import get_logger
from qlora_finetuning.settings import ConfigError

logger = get_logger(__name__)

LORA_A, LORA_B = ".lora_A.", ".lora_B."
PEFT_PREFIX = "base_model.model."
# How far the merged weight that is checked may be from base + scale * B @ A.
CHECK_TOLERANCE = 1e-5


@dataclass(frozen=True)
class MergeResult:
    """What a merge did."""

    saved_dtype: str
    compute_dtype: str
    parameters: int  # of the merged model
    adapted_tensors: int  # weights the adapter changes
    changed_tensor: str  # the one that was checked against base + scale * B @ A
    libraries: dict[str, str]


def _stack() -> Any:
    try:
        import peft
        import safetensors.torch
        import torch
        import transformers
    except ImportError as exc:
        raise ConfigError(
            f"the training stack is not installed ({exc.name} is missing); {INSTALL_HINT}"
        ) from exc
    return torch, transformers, peft, safetensors.torch


def _stored_dtype(torch: Any, config: Any) -> Any:
    """The type the base model's weights are stored in, as its config says."""
    stored = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if isinstance(stored, str):
        stored = getattr(torch, stored, None)
    if stored is None:
        raise ConfigError("the base model's config does not say which type its weights have")
    return stored


def merge_adapter(
    base: str | Path,
    adapter_dir: Path | str,
    out_dir: Path | str,
    *,
    end_ids: Sequence[int],
    pad_token_id: int,
    revision: str | None = None,
    compute_dtype: str = "float32",
    max_shard_size: str = "4GB",
) -> MergeResult:
    """Write the base model with the adapter merged in to ``out_dir``. Runs on the CPU.

    ``base`` is a model id in the local cache (with ``revision``) or a folder. Nothing is
    downloaded. One adapted weight is compared with ``base + scale * B @ A`` before
    anything is saved, and a merge that leaves an adapter layer behind is refused. Raises
    ``ConfigError``.
    """
    torch, transformers, peft, safetensors_torch = _stack()
    adapter_dir, out_dir = Path(adapter_dir), Path(out_dir)
    compute = getattr(torch, compute_dtype)

    adapter_config = peft.PeftConfig.from_pretrained(str(adapter_dir))
    weights = safetensors_torch.load_file(str(adapter_dir / ADAPTER_WEIGHTS))
    a_keys = sorted(key for key in weights if LORA_A in key)
    if not a_keys:
        raise ConfigError(f"{adapter_dir} holds no LoRA weights: it is not a LoRA adapter")
    checked = a_keys[0]
    target = checked.split(LORA_A)[0].removeprefix(PEFT_PREFIX) + ".weight"

    logger.info("loading the base model in %s on the CPU", compute_dtype)
    try:
        config = transformers.AutoConfig.from_pretrained(
            str(base), revision=revision, local_files_only=True
        )
        saved = _stored_dtype(torch, config)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            str(base), revision=revision, local_files_only=True, dtype=compute
        )
    except ConfigError:
        raise
    except Exception as exc:  # a missing file, too little memory, a library mismatch
        raise ConfigError(
            f"the base model could not be loaded for the merge "
            f"({type(exc).__name__}: {(str(exc).strip().splitlines() or [''])[0]})"
        ) from exc

    parameters = dict(model.named_parameters())
    if target not in parameters:
        raise ConfigError(f"the adapter changes {target}, which the base model does not have")
    scale = adapter_config.lora_alpha / adapter_config.r
    product = weights[checked.replace(LORA_A, LORA_B)].to(compute) @ weights[checked].to(compute)
    expected = parameters[target].detach().clone() + scale * product
    del parameters

    logger.info("merging %d adapted weights", len(a_keys))
    merged = peft.PeftModel.from_pretrained(
        model, str(adapter_dir), is_trainable=False
    ).merge_and_unload(safe_merge=True)
    left = [name for name, _ in merged.named_parameters() if "lora_" in name]
    if left:
        raise ConfigError(f"the merge left adapter layers behind ({left[0]}, ...)")
    got = dict(merged.named_parameters())[target].detach()
    gap = float((got - expected).abs().max())
    if not gap <= CHECK_TOLERANCE * max(1.0, float(expected.abs().max())):
        raise ConfigError(
            f"the merged weight {target} is not base + scale * B @ A (largest gap {gap:g})"
        )

    merged.to(saved)
    merged.generation_config = transformers.GenerationConfig(
        do_sample=False, num_beams=1, eos_token_id=list(end_ids), pad_token_id=pad_token_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("saving the merged copy as %s", str(saved).removeprefix("torch."))
    merged.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size=max_shard_size)
    return MergeResult(
        saved_dtype=str(saved).removeprefix("torch."),
        compute_dtype=compute_dtype,
        parameters=sum(parameter.numel() for parameter in merged.parameters()),
        adapted_tensors=len(a_keys),
        changed_tensor=target,
        libraries={
            "torch": torch.__version__.split("+")[0],
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    )
