"""Path A training settings: ``config/training.yaml`` (req_003 v3.02).

Every hyperparameter of a QLoRA training run. Nothing here is left to a library default.

Usage::

    from qlora_finetuning.local_qlora.training_config import load_training_config

    cfg = load_training_config()
    cfg.learning_rate, cfg.batch_size, cfg.accumulation
    cfg.epochs_for(smoke=False), cfg.save_steps_for(smoke=True)
    cfg.settings(smoke=False)          # what a resume compares
    cfg.settings_sha256(smoke=False)   # what an adapter's metadata carries

This module imports no part of the training stack.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qlora_finetuning.settings import REPO_ROOT, ConfigError, load_config

REQUIRED_KEYS = (
    "lora.r",
    "lora.alpha",
    "lora.dropout",
    "lora.target_modules",
    "optimizer.name",
    "optimizer.learning_rate",
    "optimizer.weight_decay",
    "optimizer.max_grad_norm",
    "schedule.type",
    "schedule.warmup_fraction",
    "batch.size",
    "batch.accumulation",
    "epochs",
    "seed",
    "precision",
    "gradient_checkpointing",
    "checkpoint.save_steps",
    "checkpoint.dev_loss",
    "logging_steps",
    "adapters_dir",
    "smoke.examples",
    "smoke.epochs",
    "smoke.save_steps",
    "smoke.logging_steps",
    "smoke.dev_examples",
    "smoke.generate_examples",
    "smoke.generate_batch_size",
    "smoke.max_final_loss_ratio",
    "selection.split",
    "selection.min_execution_accuracy_percent",
    "selection.min_format_validity_percent",
    "selection.record_file",
    "selection.noise_examples",
    "iteration.record_file",
    "iteration.analysis_file",
    "iteration.changes",
    "local_run.min_free_gpu_gib",
    "local_run.min_free_disk_gb",
    "local_run.note_file",
)
# The one variable a corrective iteration may change (req_003 v3.05, D8), as dotted keys
# of ``settings()``. ``lora.alpha`` follows ``lora.r`` at twice its value and is not one.
ITERATION_VARIABLES = (
    "epochs",
    "optimizer.learning_rate",
    "lora.r",
    "lora.dropout",
    "schedule.warmup_fraction",
)
OPTIMIZERS = ("paged_adamw_8bit", "adamw_torch")
SCHEDULES = ("cosine", "linear", "constant")
PRECISIONS = ("bf16", "fp16")
SELECTION_SPLITS = ("dev", "quick_dev")
FILE = "config training.yaml"


@dataclass(frozen=True)
class SmokeConfig:
    """The smoke run's own sizes. Everything else is the full run's."""

    examples: int
    epochs: int
    save_steps: int
    logging_steps: int
    dev_examples: int
    generate_examples: int
    generate_batch_size: int
    max_final_loss_ratio: float


@dataclass(frozen=True)
class TrainingConfig:
    """The typed content of ``config/training.yaml``, with its path resolved."""

    root: Path
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: str | tuple[str, ...]
    optimizer: str
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    schedule: str
    warmup_fraction: float
    batch_size: int
    accumulation: int
    epochs: int
    seed: int
    precision: str
    gradient_checkpointing: bool
    save_steps: int
    dev_loss: bool
    logging_steps: int
    adapters_dir: Path
    smoke: SmokeConfig
    # Which checkpoint becomes arm A1 (req_003 v3.04). Not part of ``settings``: none of
    # it decides an adapter's weights.
    selection_split: str
    min_execution_accuracy_percent: float
    min_format_validity_percent: float
    selection_record: Path
    # The corrective iteration (req_003 v3.05). Not part of ``settings`` either.
    noise_examples: int = 10
    iteration_record: Path | None = None
    analysis_record: Path | None = None
    iteration_changes: tuple[tuple[str, str, float], ...] = ()  # finding, variable, value
    # The optional full run on this machine (req_003 v3.06). Not part of ``settings``.
    min_free_gpu_gib: float = 14.0
    min_free_disk_gb: float = 10.0
    local_note: Path | None = None

    def epochs_for(self, smoke: bool) -> int:
        return self.smoke.epochs if smoke else self.epochs

    def save_steps_for(self, smoke: bool) -> int:
        return self.smoke.save_steps if smoke else self.save_steps

    def logging_steps_for(self, smoke: bool) -> int:
        return self.smoke.logging_steps if smoke else self.logging_steps

    def settings(self, smoke: bool) -> dict[str, Any]:
        """Everything that decides the weights a run produces.

        A run is resumed, and an adapter recognized, only under equal settings.
        """
        modules = self.target_modules
        return {
            "lora": {
                "r": self.lora_r,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
                "target_modules": modules if isinstance(modules, str) else list(modules),
            },
            "optimizer": {
                "name": self.optimizer,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "max_grad_norm": self.max_grad_norm,
            },
            "schedule": {"type": self.schedule, "warmup_fraction": self.warmup_fraction},
            "batch": {"size": self.batch_size, "accumulation": self.accumulation},
            "epochs": self.epochs_for(smoke),
            "seed": self.seed,
            "precision": self.precision,
            "gradient_checkpointing": self.gradient_checkpointing,
            "save_steps": self.save_steps_for(smoke),
            "dev_loss": self.dev_loss,
            "smoke_examples": self.smoke.examples if smoke else None,
        }

    def settings_sha256(self, smoke: bool) -> str:
        return sha256_of(self.settings(smoke))


def sha256_of(value: Any) -> str:
    """SHA-256 of plain values, written the same way every time."""
    text = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _whole(raw: Any, key: str, minimum: int = 1) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < minimum:
        raise ConfigError(
            f"{FILE}: {key} must be a whole number of at least {minimum}, got {raw!r}"
        )
    return raw


def _number(raw: Any, key: str, low: float, high: float, low_open: bool = False) -> float:
    ok = not isinstance(raw, bool) and isinstance(raw, (int, float))
    if ok:
        ok = (low < raw if low_open else low <= raw) and raw <= high
    if not ok:
        bound = f"above {low}" if low_open else f"at least {low}"
        raise ConfigError(f"{FILE}: {key} must be a number {bound} and at most {high}, got {raw!r}")
    return float(raw)


def _flag(raw: Any, key: str) -> bool:
    if not isinstance(raw, bool):
        raise ConfigError(f"{FILE}: {key} must be true or false, got {raw!r}")
    return raw


def _choice(raw: Any, key: str, allowed: tuple[str, ...]) -> str:
    if raw not in allowed:
        raise ConfigError(f"{FILE}: {key} must be one of {', '.join(allowed)}, got {raw!r}")
    return raw


def load_training_config(
    root: Path | str = REPO_ROOT, config_dir: Path | str | None = None
) -> TrainingConfig:
    """Read and check ``config/training.yaml``. Relative paths in it resolve under ``root``."""
    raw = load_config("training", required=REQUIRED_KEYS, config_dir=config_dir)
    lora, optimizer, schedule = raw["lora"], raw["optimizer"], raw["schedule"]
    batch, checkpoint, smoke = raw["batch"], raw["checkpoint"], raw["smoke"]
    selection, iteration = raw["selection"], raw["iteration"]
    changes = iteration["changes"]
    if not isinstance(changes, dict) or not changes:
        raise ConfigError(f"{FILE}: iteration.changes must map a finding to a variable and value")
    table = []
    for finding, change in changes.items():
        variable = change.get("variable") if isinstance(change, dict) else None
        value = change.get("value") if isinstance(change, dict) else None
        if variable not in ITERATION_VARIABLES:
            raise ConfigError(
                f"{FILE}: iteration.changes.{finding}.variable must be one of "
                f"{', '.join(ITERATION_VARIABLES)}, got {variable!r}"
            )
        table.append(
            (str(finding), variable, _number(value, f"iteration.changes.{finding}.value", 0, 1000))
        )

    modules = lora["target_modules"]
    if isinstance(modules, list) and modules and all(isinstance(m, str) and m for m in modules):
        modules = tuple(modules)
    elif not isinstance(modules, str) or not modules:
        raise ConfigError(
            f"{FILE}: lora.target_modules must be text (all-linear) or a list of module names"
        )

    smoke_cfg = SmokeConfig(
        examples=_whole(smoke["examples"], "smoke.examples"),
        epochs=_whole(smoke["epochs"], "smoke.epochs"),
        save_steps=_whole(smoke["save_steps"], "smoke.save_steps"),
        logging_steps=_whole(smoke["logging_steps"], "smoke.logging_steps"),
        dev_examples=_whole(smoke["dev_examples"], "smoke.dev_examples"),
        generate_examples=_whole(smoke["generate_examples"], "smoke.generate_examples"),
        generate_batch_size=_whole(smoke["generate_batch_size"], "smoke.generate_batch_size", 2),
        max_final_loss_ratio=_number(
            smoke["max_final_loss_ratio"], "smoke.max_final_loss_ratio", 0, 1, low_open=True
        ),
    )
    batch_size = _whole(batch["size"], "batch.size")
    if smoke_cfg.examples < batch_size:
        raise ConfigError(f"{FILE}: smoke.examples must be at least batch.size")

    root = Path(root)
    return TrainingConfig(
        root=root,
        lora_r=_whole(lora["r"], "lora.r"),
        lora_alpha=_whole(lora["alpha"], "lora.alpha"),
        lora_dropout=_number(lora["dropout"], "lora.dropout", 0, 0.5),
        target_modules=modules,
        optimizer=_choice(optimizer["name"], "optimizer.name", OPTIMIZERS),
        learning_rate=_number(
            optimizer["learning_rate"], "optimizer.learning_rate", 0, 0.01, low_open=True
        ),
        weight_decay=_number(optimizer["weight_decay"], "optimizer.weight_decay", 0, 1),
        max_grad_norm=_number(
            optimizer["max_grad_norm"], "optimizer.max_grad_norm", 0, 100, low_open=True
        ),
        schedule=_choice(schedule["type"], "schedule.type", SCHEDULES),
        warmup_fraction=_number(schedule["warmup_fraction"], "schedule.warmup_fraction", 0, 0.5),
        batch_size=batch_size,
        accumulation=_whole(batch["accumulation"], "batch.accumulation"),
        epochs=_whole(raw["epochs"], "epochs"),
        seed=_whole(raw["seed"], "seed", 0),
        precision=_choice(raw["precision"], "precision", PRECISIONS),
        gradient_checkpointing=_flag(raw["gradient_checkpointing"], "gradient_checkpointing"),
        save_steps=_whole(checkpoint["save_steps"], "checkpoint.save_steps"),
        dev_loss=_flag(checkpoint["dev_loss"], "checkpoint.dev_loss"),
        logging_steps=_whole(raw["logging_steps"], "logging_steps"),
        adapters_dir=root / str(raw["adapters_dir"]),
        smoke=smoke_cfg,
        selection_split=_choice(selection["split"], "selection.split", SELECTION_SPLITS),
        min_execution_accuracy_percent=_number(
            selection["min_execution_accuracy_percent"],
            "selection.min_execution_accuracy_percent",
            0,
            100,
        ),
        min_format_validity_percent=_number(
            selection["min_format_validity_percent"],
            "selection.min_format_validity_percent",
            0,
            100,
        ),
        selection_record=root / str(selection["record_file"]),
        noise_examples=_whole(selection["noise_examples"], "selection.noise_examples", 0),
        iteration_record=root / str(iteration["record_file"]),
        analysis_record=root / str(iteration["analysis_file"]),
        iteration_changes=tuple(table),
        min_free_gpu_gib=_number(
            raw["local_run"]["min_free_gpu_gib"], "local_run.min_free_gpu_gib", 0, 1024
        ),
        min_free_disk_gb=_number(
            raw["local_run"]["min_free_disk_gb"], "local_run.min_free_disk_gb", 0, 1_000_000
        ),
        local_note=root / str(raw["local_run"]["note_file"]),
    )
