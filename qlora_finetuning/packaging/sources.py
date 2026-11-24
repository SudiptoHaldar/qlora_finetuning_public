"""What a package is built from: the selected checkpoint and the committed records.

Arm A1 is the checkpoint the selection record names, and nothing else. The folder
``artifacts/adapters/<run>/`` holds a run's *last* step, which is another adapter, so the
weights are always taken from ``artifacts/runs/<run>/checkpoints/checkpoint-N/`` after
their hash is compared with the selection record's.

Usage::

    from qlora_finetuning.packaging.sources import packaging_context, read_records, selected_source

    source = selected_source(packaging_context())      # ConfigError unless it is arm A1
    source.path, source.sha256, source.step
    records = read_records(cfg)
    records_problems(records)                          # why a card may not quote them

This module imports no part of the training stack and reads no split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from qlora_finetuning.data.serialization import PromptDefinition, load_prompt
from qlora_finetuning.data.token_budget import TokenBudgetConfig
from qlora_finetuning.local_qlora.adapters import (
    METADATA_NAME,
    SELECTED,
    TRAINING_NAME,
    adapter_sha256,
    read_selection,
)
from qlora_finetuning.local_qlora.config import LocalQloraConfig
from qlora_finetuning.local_qlora.predictor import resolve_tuned_adapter
from qlora_finetuning.local_qlora.training_config import load_training_config
from qlora_finetuning.packaging.config import PackagingConfig
from qlora_finetuning.packaging.files import read_json
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking.config import load_tracking_config

PUBLISHED_ARM = "A1"
SCHEMA = 1


@dataclass(frozen=True)
class Source:
    """The selected checkpoint on this machine, checked."""

    name: str  # RUN_ID/checkpoint-N
    path: Path
    run_id: str
    step: int
    sha256: str  # of the weights file; equal to the selection record's
    selection: dict[str, Any]
    training: dict[str, Any]  # the training run's own record: settings and plan
    extras: dict[str, Any]  # library versions and trainable parameters, when known


@dataclass(frozen=True)
class Records:
    """The committed records a card quotes."""

    six_arm: dict[str, Any]
    statistics: dict[str, Any]
    cost: dict[str, Any]


def packaging_context(
    tracking_cfg: Any = None, prompt: PromptDefinition | None = None, split_cfg: Any = None
) -> SimpleNamespace:
    """What the adapter checks read: the prompt, the split, and where runs are kept."""
    return SimpleNamespace(
        prompt=prompt or load_prompt(),
        split_cfg=split_cfg,
        tracking_cfg=tracking_cfg or load_tracking_config(),
    )


def selected_source(context: Any) -> Source:
    """The checkpoint that is arm A1, found and checked. Raises ``ConfigError``.

    Every check of ``--adapter selected``: the selection was made for today's model commit,
    prompt wording, and split; the checkpoint is on this machine; its weights have the hash
    the selection was made on.
    """
    ref = resolve_tuned_adapter(SELECTED, context)
    tracking = context.tracking_cfg
    training_cfg = load_training_config(root=tracking.root)
    selection = read_selection(training_cfg.selection_record)
    if ref.checkpoint is None:
        raise ConfigError(
            f"the selection record names {ref.name}, which is not a checkpoint: a package is "
            "built from RUN_ID/checkpoint-N only"
        )
    training = read_json(
        tracking.runs_dir / ref.run_id / TRAINING_NAME, "the training run's record"
    )
    extras: dict[str, Any] = {}
    final = training_cfg.adapters_dir / ref.run_id / METADATA_NAME
    if final.is_file():
        # The run's last adapter is another adapter; only what is true of the whole run
        # is read from its metadata.
        metadata = read_json(final, "the run's adapter metadata")
        extras = {
            "libraries": metadata.get("libraries"),
            "trainable_parameters": metadata.get("trainable_parameters"),
        }
    return Source(
        name=ref.name,
        path=ref.path,
        run_id=ref.run_id,
        step=int(ref.checkpoint),
        sha256=adapter_sha256(ref.path),
        selection=selection,
        training=training,
        extras=extras,
    )


def read_records(cfg: PackagingConfig) -> Records:
    """The three comparison records. Raises ``ConfigError`` if one is missing or damaged."""
    return Records(
        six_arm=read_json(cfg.six_arm_file, "the six-arm record"),
        statistics=read_json(cfg.statistics_file, "the comparison's record"),
        cost=read_json(cfg.cost_file, "the cost record"),
    )


def records_problems(records: Records) -> list[str]:
    """Why the records may not be quoted on a card; empty when they may (check P1)."""
    problems = []
    if not records.six_arm.get("passed"):
        problems.append("the six-arm record did not pass its checks C1 to C8")
    if not records.cost.get("passed"):
        problems.append("the cost record did not pass its checks K1 to K8")
    elif records.cost.get("provisional"):
        problems.append("the cost record is provisional: the bill is not entered")
    if not records.statistics.get("passed"):
        problems.append("the comparison's record did not pass its checks R1 to R8")
    elif records.statistics.get("provisional"):
        problems.append("the comparison's record is provisional")
    if PUBLISHED_ARM not in (records.statistics.get("arms") or {}):
        problems.append(f"the comparison's record has no arm {PUBLISHED_ARM}")
    freezes = {
        records.six_arm.get("freeze_sha256"),
        records.statistics.get("freeze_sha256"),
        records.cost.get("six_arm_freeze_sha256"),
    }
    if len(freezes) != 1 or None in freezes:
        problems.append("the three records do not name the same freeze record")
    return problems


def prompt_file(
    prompt: PromptDefinition, budget: TokenBudgetConfig, local_cfg: LocalQloraConfig
) -> dict[str, Any]:
    """``prompt.json`` of a package: the wording, word for word, and how it is used."""
    return {
        "schema": SCHEMA,
        "prompt_id": prompt.prompt_id,
        "version": prompt.version,
        "system": prompt.system,
        "user": {
            "context_label": prompt.context_label,
            "list_marker": prompt.list_marker,
            "no_context": prompt.no_context,
            "table_label": prompt.table_label,
            "question_label": prompt.question_label,
        },
        "target": {
            "program_label": prompt.program_label,
            "answer_label": prompt.answer_label,
        },
        "thinking": False,
        "end_tokens": list(local_cfg.end_tokens),
        "max_new_tokens": budget.max_new_tokens,
        "max_length": budget.max_length,
        "notes": [
            "The user text is the context label, the passages one per line after the list "
            "marker (or the no-context text), a blank line, the table label, the table as "
            "a Markdown table with a separator row after its first row, a blank line, "
            "then the question label, a space, and the question.",
            "The system and user texts are wrapped by the tokenizer's chat template with "
            "enable_thinking=False and add_generation_prompt=True.",
            "The reply is two lines: the program label and a program, then the answer "
            "label and the result. Run the program; do not trust the answer line.",
        ],
    }


def training_metadata(
    source: Source,
    records: Records,
    *,
    prompt: PromptDefinition,
    budget: TokenBudgetConfig,
    local_cfg: LocalQloraConfig,
    manifest_hash: str,
) -> dict[str, Any]:
    """``training_metadata.json``: how the adapter was trained and chosen.

    Written for a reader outside the project: no job name, no bucket, no machine path.
    """
    settings = dict(source.training.get("settings") or {})
    plan = dict(source.training.get("plan") or {})
    selected = dict(source.selection.get("selected") or {})
    train = ((records.cost.get("making") or {}).get(PUBLISHED_ARM) or {}).get("train") or {}
    return {
        "schema": SCHEMA,
        "base_model": {"id": budget.model_id, "commit": budget.revision},
        "adapter": {
            "weights_sha256": source.sha256,
            "training_run": source.run_id,
            "selected_step": source.step,
            "trainable_parameters": source.extras.get("trainable_parameters"),
        },
        "prompt_id": prompt.prompt_id,
        "split_manifest_hash": manifest_hash,
        "training": {
            "lora": settings.get("lora"),
            "optimizer": settings.get("optimizer"),
            "schedule": settings.get("schedule"),
            "batch": settings.get("batch"),
            "epochs": settings.get("epochs"),
            "seed": settings.get("seed"),
            "precision": settings.get("precision"),
            "gradient_checkpointing": settings.get("gradient_checkpointing"),
            "records": plan.get("records"),
            "tokens_per_pass": plan.get("tokens_per_pass"),
            "steps_per_epoch": plan.get("steps_per_epoch"),
            "steps_planned": plan.get("total_steps"),
            "warmup_steps": plan.get("warmup_steps"),
            "checkpoint_steps": plan.get("checkpoint_steps"),
            "max_length": budget.max_length,
            "loss_on": "the reply only",
        },
        "quantization": {
            "load_in_4bit": True,
            "type": local_cfg.quantization_type,
            "double_quant": local_cfg.double_quant,
            "compute_dtype": local_cfg.compute_dtype,
        },
        "selection": {
            "rule": "the checkpoint with the most dev examples right by execution accuracy",
            "split": source.selection.get("split"),
            "examples": selected.get("examples"),
            "execution_accuracy": selected.get("execution_accuracy"),
            "answer_accuracy": selected.get("answer_accuracy"),
            "program_match": selected.get("program_match"),
            "format_validity": selected.get("format_validity"),
            "dev_loss": selected.get("dev_loss"),
        },
        "hardware": {
            "instance_type": train.get("instance_type"),
            "billable_seconds": train.get("billable_seconds"),
        },
        "libraries": source.extras.get("libraries"),
    }
