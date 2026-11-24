"""The merged copy as a package: the base model with the adapter in its weights.

It loads with ``transformers`` alone. It carries Qwen's weights, so Qwen's licence file and
a notice of what was changed go with it. It is **not arm A1**: it is scored on dev only,
and its card labels every test figure as the adapter's.

Usage::

    from qlora_finetuning.packaging.merged_package import build_merged_package, merged_names

    build_merged_package(cfg, source, prompt=prompt, budget=budget, local_cfg=local_cfg,
                         tokenizer_paths=paths, licence_path=qwen_licence)
    merged_names(cfg)                 # every file of the finished package
    merged_weights(cfg.merged_dir)    # the weight files' sizes and hashes, and one hash over them

The merge itself is ``merge.py``. This module imports no part of the training stack.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from qlora_finetuning.data.serialization import PromptDefinition
from qlora_finetuning.data.token_budget import TokenBudgetConfig
from qlora_finetuning.demo.inputs import sample_path
from qlora_finetuning.local_qlora.config import LocalQloraConfig
from qlora_finetuning.packaging.adapter_package import (
    CARD_NAME,
    LICENSE_NAME,
    PROMPT_NAME,
    SAMPLE_NAME,
    copy_tokenizer,
)
from qlora_finetuning.packaging.config import PackagingConfig
from qlora_finetuning.packaging.files import (
    combined_sha256,
    copy_text,
    read_json,
    sha256_file,
    write_json_lf,
    write_text_lf,
)
from qlora_finetuning.packaging.sources import SCHEMA, Source, prompt_file
from qlora_finetuning.settings import ConfigError

NOTICE_NAME = "NOTICE"
MERGE_NAME = "merge_metadata.json"
CONFIG_NAME = "config.json"
GENERATION_NAME = "generation_config.json"
WEIGHTS_INDEX = "model.safetensors.index.json"
WEIGHTS_SINGLE = "model.safetensors"
QWEN_COPYRIGHT = "Copyright 2024 Alibaba Cloud"  # as Qwen's LICENSE file states it


def weight_names(folder: Path) -> list[str]:
    """The weight files a saved model's index names, in order; one file when not split."""
    index = folder / WEIGHTS_INDEX
    if index.is_file():
        mapping = read_json(index, "the merged copy's weight index")["weight_map"]
        return sorted(set(mapping.values()))
    return [WEIGHTS_SINGLE]


def merged_names(cfg: PackagingConfig) -> tuple[str, ...]:
    """Every file of the finished merged package, as it is on disk now."""
    folder = cfg.merged_dir
    weights = weight_names(folder)
    index = [WEIGHTS_INDEX] if (folder / WEIGHTS_INDEX).is_file() else []
    return (
        CONFIG_NAME,
        GENERATION_NAME,
        *weights,
        *index,
        *cfg.tokenizer_files,
        PROMPT_NAME,
        SAMPLE_NAME,
        MERGE_NAME,
        LICENSE_NAME,
        NOTICE_NAME,
        CARD_NAME,
    )


def merged_weights(folder: Path) -> dict[str, Any]:
    """The weight files' sizes and hashes, and the one hash that names the copy in a run."""
    entries = []
    for name in weight_names(folder):
        path = folder / name
        if not path.is_file():
            raise ConfigError(
                f"the merged copy in {folder.name}/ is incomplete ({name} is missing): "
                "run python -m scripts.hf_package merge"
            )
        entries.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"files": entries, "sha256": combined_sha256(entries)}


def read_merge_metadata(folder: Path) -> dict[str, Any]:
    data = read_json(folder / MERGE_NAME, "the merged copy's record of its merge")
    if not isinstance(data, dict) or not isinstance(data.get("adapter"), dict):
        raise ConfigError(f"the merged copy's {MERGE_NAME} is damaged")
    return data


def notice_text(cfg: PackagingConfig, source: Source, budget: TokenBudgetConfig) -> str:
    name = cfg.merged_repo.split("/", 1)[1]
    return (
        f"{name}\n\n"
        f"This model is a modified copy of {budget.model_id} (commit {budget.revision}),\n"
        f"{QWEN_COPYRIGHT}, licensed under the Apache License, Version 2.0.\n\n"
        "Changes made to the original:\n"
        f"- A LoRA adapter ({cfg.adapter_repo}, weights SHA-256\n"
        f"  {source.sha256}) was merged into the weights of\n"
        "  the linear layers of the attention and feed-forward blocks.\n"
        "- The generation settings were replaced by greedy decoding.\n"
        "The tokenizer files are unchanged.\n\n"
        "The licence text is in the file LICENSE.\n"
    )


def build_merged_package(
    cfg: PackagingConfig,
    source: Source,
    *,
    prompt: PromptDefinition,
    budget: TokenBudgetConfig,
    local_cfg: LocalQloraConfig,
    tokenizer_paths: Mapping[str, Path | str],
    licence_path: Path | str,
    end_ids: list[int],
    pad_token_id: int,
    merge: Callable[..., Any],
    base: str | Path | None = None,
) -> Path:
    """Merge and write the merged package, all but its card. Returns the folder.

    ``merge`` is ``merge.merge_adapter`` (a test passes a stand-in); ``base`` is the base
    model's id in the cache, or a folder. ``end_ids`` and ``pad_token_id`` are the pinned
    tokenizer's, for the greedy generation settings the copy is saved with.
    """
    folder = cfg.merged_dir
    folder.mkdir(parents=True, exist_ok=True)
    result = merge(
        base or budget.model_id,
        source.path,
        folder,
        revision=None if base else budget.revision,
        end_ids=end_ids,
        pad_token_id=pad_token_id,
        compute_dtype=cfg.compute_dtype,
        max_shard_size=cfg.max_shard_size,
    )
    copy_tokenizer(cfg.tokenizer_files, tokenizer_paths, folder)
    write_json_lf(folder / PROMPT_NAME, prompt_file(prompt, budget, local_cfg))
    copy_text(sample_path(), folder / SAMPLE_NAME)
    copy_text(licence_path, folder / LICENSE_NAME)
    write_text_lf(folder / NOTICE_NAME, notice_text(cfg, source, budget))
    write_json_lf(
        folder / MERGE_NAME,
        {
            "schema": SCHEMA,
            "base_model": {"id": budget.model_id, "commit": budget.revision},
            "adapter": {
                "repository": cfg.adapter_repo,
                "weights_sha256": source.sha256,
                "training_run": source.run_id,
                "selected_step": source.step,
            },
            "merge": {
                "method": "W + (alpha / r) * B @ A, added to the full-precision base weights",
                "compute_dtype": result.compute_dtype,
                "saved_dtype": result.saved_dtype,
                "adapted_tensors": result.adapted_tensors,
                "parameters": result.parameters,
                "checked_tensor": result.changed_tensor,
            },
            "generation": {"do_sample": False, "num_beams": 1, "stop": "the end tokens"},
            "libraries": result.libraries,
        },
    )
    return folder
