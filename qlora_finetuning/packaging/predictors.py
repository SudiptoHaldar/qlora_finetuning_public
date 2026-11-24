"""Predictors that load a package from its folder (req_006 v6.02, spec D13 and D18).

Three names for ``scripts.evaluate``, each a ``QwenPredictor`` over the ordinary backend:

- ``qwen3-4b-package``: the adapter package on the 4-bit base model. Same weights as
  ``qwen3-4b-tuned --adapter selected``, so its replies must be the same.
- ``qwen3-4b-merged``: the merged copy, loaded in 4-bit.
- ``qwen3-4b-merged-full``: the merged copy as it is stored, not quantized.

None is an arm. Each is refused on the test split before anything is loaded, and the gate
refuses it there as well. Runs are local, free, and development evidence only::

    venv\\Scripts\\python -m scripts.evaluate --predictor qwen3-4b-package --split dev

The settings a run is recorded with are known without loading a model
(``package_settings``, ``merged_settings``), so a run can be found again by them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qlora_finetuning.data.token_budget import TokenBudgetConfig, load_token_budget_config
from qlora_finetuning.local_qlora.adapters import adapter_sha256
from qlora_finetuning.local_qlora.config import LocalQloraConfig, load_local_qlora_config
from qlora_finetuning.local_qlora.predictor import QwenPredictor
from qlora_finetuning.local_qlora.scoring import expected_settings
from qlora_finetuning.packaging.config import PackagingConfig, load_packaging_config
from qlora_finetuning.packaging.merged_package import merged_weights, read_merge_metadata
from qlora_finetuning.packaging.sources import Source, packaging_context, selected_source
from qlora_finetuning.settings import ConfigError

PACKAGE = "qwen3-4b-package"
MERGED = "qwen3-4b-merged"
MERGED_FULL = "qwen3-4b-merged-full"
TEST = "test"
BUILD = "python -m scripts.hf_package"


def refuse_test(split: str, name: str) -> None:
    if split == TEST:
        raise ConfigError(
            f"{name} is not one of the six arms and never answers the test split: a package "
            "is scored on dev only"
        )


def package_settings(
    local_cfg: LocalQloraConfig, budget: TokenBudgetConfig, adapter: str
) -> dict[str, Any]:
    """The settings of a run of the adapter package: those of the adapter by name."""
    return expected_settings(local_cfg, budget.revision, adapter)


def merged_settings(
    local_cfg: LocalQloraConfig,
    budget: TokenBudgetConfig,
    *,
    weights_sha256: str,
    adapter: str,
    quantized: bool,
) -> dict[str, Any]:
    """The settings of a run of the merged copy. They name the weights and the precision."""
    settings = expected_settings(local_cfg, budget.revision, None)
    settings["merged"] = {"weights_sha256": weights_sha256, "adapter": adapter}
    if not quantized:
        settings["quantization"] = None
        settings["dtype"] = local_cfg.compute_dtype
    return settings


def _source(context: Any) -> Source:
    return selected_source(
        packaging_context(context.tracking_cfg, context.prompt, context.split_cfg)
    )


def build_package_predictor(
    cfg: PackagingConfig, source: Source, load: Any = None
) -> QwenPredictor:
    """The adapter package on the 4-bit base model. Refused unless it holds arm A1's weights."""
    folder = cfg.adapter_dir
    if not folder.is_dir():
        raise ConfigError(f"there is no adapter package yet: run {BUILD} adapter")
    if adapter_sha256(folder) != source.sha256:
        raise ConfigError(
            f"the adapter package does not hold the selected checkpoint's weights: "
            f"run {BUILD} adapter again"
        )
    if load is None:
        # The training stack is loaded only when the model is asked for.
        from qlora_finetuning.local_qlora.backend import load_backend as load

    backend = load(load_local_qlora_config(), load_token_budget_config(), adapter_dir=folder)
    return QwenPredictor(
        backend,
        PACKAGE,
        backend.settings,
        details={
            "package": "adapter",
            "training_run": source.run_id,
            "checkpoint": source.step,
            "steps": source.step,
            "smoke": False,
        },
    )


def build_merged_predictor(
    cfg: PackagingConfig, source: Source, *, quantized: bool, load: Any = None
) -> QwenPredictor:
    """The merged copy from its folder. Refused unless it was merged from arm A1's weights."""
    folder: Path = cfg.merged_dir
    if not folder.is_dir():
        raise ConfigError(f"there is no merged copy yet: run {BUILD} merge")
    merged_from = read_merge_metadata(folder)["adapter"].get("weights_sha256")
    if merged_from != source.sha256:
        raise ConfigError(
            f"the merged copy was not made from the selected checkpoint: run {BUILD} merge again"
        )
    local_cfg, budget = load_local_qlora_config(), load_token_budget_config()
    settings = merged_settings(
        local_cfg,
        budget,
        weights_sha256=merged_weights(folder)["sha256"],
        adapter=source.sha256,
        quantized=quantized,
    )
    if load is None:
        from qlora_finetuning.local_qlora.backend import load_folder_backend as load

    backend = load(
        local_cfg, budget, folder, model_id=cfg.merged_repo, settings=settings, quantized=quantized
    )
    return QwenPredictor(
        backend,
        MERGED if quantized else MERGED_FULL,
        backend.settings,
        details={
            "package": "merged",
            "precision": "4-bit" if quantized else local_cfg.compute_dtype,
            "training_run": source.run_id,
            "checkpoint": source.step,
        },
    )


def qwen3_4b_package(split: str, context: Any) -> QwenPredictor:
    """For ``scripts.evaluate --predictor qwen3-4b-package``."""
    refuse_test(split, PACKAGE)
    return build_package_predictor(load_packaging_config(), _source(context))


def qwen3_4b_merged(split: str, context: Any) -> QwenPredictor:
    """For ``scripts.evaluate --predictor qwen3-4b-merged``."""
    refuse_test(split, MERGED)
    return build_merged_predictor(load_packaging_config(), _source(context), quantized=True)


def qwen3_4b_merged_full(split: str, context: Any) -> QwenPredictor:
    """For ``scripts.evaluate --predictor qwen3-4b-merged-full``."""
    refuse_test(split, MERGED_FULL)
    return build_merged_predictor(load_packaging_config(), _source(context), quantized=False)
