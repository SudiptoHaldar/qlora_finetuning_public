"""Saved LoRA adapters: where they are, what they were trained for (req_003 v3.02).

A completed training run leaves its adapter in ``artifacts/adapters/<run_id>/`` beside
``adapter_metadata.json``. Each checkpoint of a run is an adapter too, under the run's
``checkpoints/checkpoint-N/``; what it was trained for is in the run's ``training.json``.

An adapter is named by its training run, never by a path::

    from qlora_finetuning.local_qlora.adapters import check_adapter, resolve_adapter

    ref = resolve_adapter("20261119T101500Z-train-qwen3-4b-fit-1a2b", adapters_dir, runs_dir)
    ref = resolve_adapter("20261119T101500Z-train-qwen3-4b-fit-1a2b/checkpoint-300", ...)
    ref.path, ref.metadata["steps"], ref.metadata["smoke"]
    check_adapter(ref, model_id=..., revision=..., prompt_id=..., manifest_hash=...)

``check_adapter`` refuses an adapter trained for another model commit, prompt wording, or
split: a tuned model is asked only about the experiment it was tuned for.

This module imports no part of the training stack. Nothing it writes holds example text.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking.config import check_no_secrets

METADATA_NAME = "adapter_metadata.json"
TRAINING_NAME = "training.json"  # in a training run's folder: what was asked
CHECKPOINTS = "checkpoints"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_FILES = (ADAPTER_WEIGHTS, ADAPTER_CONFIG)
# What an adapter must share with the experiment it is asked about.
IDENTITY = ("model_id", "revision", "prompt_id", "manifest_hash")

_NAME = re.compile(r"(?P<run>[A-Za-z0-9][A-Za-z0-9._-]*)(?:/checkpoint-(?P<step>[0-9]{1,9}))?")
_CHECKPOINT = re.compile(r"checkpoint-([0-9]{1,9})")


@dataclass(frozen=True)
class AdapterRef:
    """An adapter on disk and what is known about it."""

    name: str  # as given: RUN_ID or RUN_ID/checkpoint-N
    path: Path
    run_id: str
    checkpoint: int | None  # None for a run's final adapter
    metadata: dict[str, Any]


def adapter_sha256(adapter_dir: Path | str) -> str:
    """SHA-256 of an adapter's weights file: what names the adapter in a run's settings."""
    path = Path(adapter_dir) / ADAPTER_WEIGHTS
    if not path.is_file():
        raise ConfigError(f"no adapter weights at {path}: expected a folder saved by PEFT")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    check_no_secrets(data, path.name)
    text = json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"{what} is missing: {path}") from None
    except ValueError:
        raise ConfigError(f"{what} is damaged: {path}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{what} is damaged: {path}")
    return data


def checkpoint_steps(checkpoints_dir: Path) -> list[int]:
    """The steps that have a checkpoint with adapter weights, in order."""
    if not checkpoints_dir.is_dir():
        return []
    steps = []
    for folder in checkpoints_dir.iterdir():
        match = _CHECKPOINT.fullmatch(folder.name)
        if match and (folder / ADAPTER_WEIGHTS).is_file():
            steps.append(int(match.group(1)))
    return sorted(steps)


def resolve_adapter(name: str, adapters_dir: Path, runs_dir: Path) -> AdapterRef:
    """Find the adapter called ``name``: a training run's id, or ``RUN_ID/checkpoint-N``.

    Raises ``ConfigError`` when the name is not one, or nothing is there.
    """
    match = _NAME.fullmatch(name or "") if isinstance(name, str) else None
    if match is None or ".." in name:
        raise ConfigError(
            f"{name!r} is not an adapter: give a training run's id, or RUN_ID/checkpoint-N"
        )
    run_id, step = match.group("run"), match.group("step")
    if step is None:
        folder = adapters_dir / run_id
        if not (folder / METADATA_NAME).is_file():
            steps = checkpoint_steps(runs_dir / run_id / CHECKPOINTS)
            hint = (
                f"; it has checkpoints, name one: {run_id}/checkpoint-{steps[-1]}" if steps else ""
            )
            raise ConfigError(
                f"run {run_id} has no finished adapter under {adapters_dir.name}/: it is not "
                f"a training run that completed on this machine{hint}"
            )
        metadata = read_json(folder / METADATA_NAME, "the adapter's metadata")
        if not (folder / ADAPTER_WEIGHTS).is_file():
            raise ConfigError(f"the adapter of run {run_id} has no weights file: {folder}")
        return AdapterRef(name, folder, run_id, None, metadata)

    number = int(step)
    folder = runs_dir / run_id / CHECKPOINTS / f"checkpoint-{number}"
    if not (folder / ADAPTER_WEIGHTS).is_file():
        steps = checkpoint_steps(runs_dir / run_id / CHECKPOINTS)
        known = ", ".join(str(item) for item in steps) or "none"
        raise ConfigError(
            f"run {run_id} has no checkpoint at step {number} on this machine (it has: {known})"
        )
    training = read_json(runs_dir / run_id / TRAINING_NAME, "the training run's record")
    metadata = dict(training.get("key") or {})
    metadata.update(run_id=run_id, steps=number, checkpoint=number)
    return AdapterRef(name, folder, run_id, number, metadata)


def check_adapter(
    ref: AdapterRef, *, model_id: str, revision: str, prompt_id: str, manifest_hash: str
) -> None:
    """Refuse an adapter that was trained for another experiment. Raises ``ConfigError``."""
    current = {
        "model_id": model_id,
        "revision": revision,
        "prompt_id": prompt_id,
        "manifest_hash": manifest_hash,
    }
    words = {
        "model_id": "base model",
        "revision": "commit of the base model",
        "prompt_id": "prompt wording",
        "manifest_hash": "split",
    }
    missing = [name for name in IDENTITY if not ref.metadata.get(name)]
    if missing:
        raise ConfigError(
            f"adapter {ref.name} does not say what it was trained for "
            f"({', '.join(missing)} missing); it cannot be used"
        )
    differs = [words[name] for name in IDENTITY if ref.metadata[name] != current[name]]
    if differs:
        raise ConfigError(
            f"adapter {ref.name} was trained for another {' and another '.join(differs)}; "
            "it answers a different experiment. Train again under the current settings"
        )


# --------------------------------------------------------------------------- a resumed training


def training_chain(run_id: str, runs_dir: Path) -> list[str]:
    """The training run and the runs it continues, the earliest first (req_003 v3.04).

    A resumed run holds only the checkpoints made after the resume; the earlier ones are in
    the run it continues. Raises ``ConfigError`` if a run of the chain has no record here.
    """
    chain: list[str] = []
    current: str | None = run_id
    while current:
        if current in chain or not _NAME.fullmatch(current) or "/" in current:
            raise ConfigError(f"{current!r} is not a training run, or its chain loops")
        training = read_json(runs_dir / current / TRAINING_NAME, "the training run's record")
        chain.append(current)
        current = training.get("continues")
    return chain[::-1]


def chain_checkpoints(run_id: str, runs_dir: Path) -> dict[int, str]:
    """Every checkpoint of a training, as ``{step: "RUN_ID/checkpoint-N"}``, along its chain.

    A step that two runs of the chain both hold is taken from the later run.
    """
    found: dict[int, str] = {}
    for member in training_chain(run_id, runs_dir):
        for step in checkpoint_steps(runs_dir / member / CHECKPOINTS):
            found[step] = f"{member}/checkpoint-{step}"
    return dict(sorted(found.items()))


# --------------------------------------------------------------------------- the selection record

SELECTED = "selected"  # what --adapter may say to mean the checkpoint of the selection record
SELECTION_SCHEMA = 1
# Schema 2 (req_003 v3.05) is the record after a corrective iteration: the same fields
# for the run that won, and both runs under ``iterations``.
ITERATED_SCHEMA = 2
SELECTION_SCHEMAS = (SELECTION_SCHEMA, ITERATED_SCHEMA)


def read_selection(path: Path | str) -> dict[str, Any]:
    """The committed selection record: which checkpoint is arm A1 (req_003 v3.04).

    Written by ``python -m scripts.qlora_select``. Raises ``ConfigError`` if there is none.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(
            f"no checkpoint has been selected yet ({path.name} does not exist): score the "
            "checkpoints and run python -m scripts.qlora_select"
        )
    record = read_json(path, "the selection record")
    selected = record.get("selected")
    if (
        record.get("schema") not in SELECTION_SCHEMAS
        or not isinstance(selected, dict)
        or not isinstance(selected.get("adapter"), str)
        or not isinstance(record.get("key"), dict)
    ):
        raise ConfigError(f"the selection record {path.name} is damaged")
    return record


def selected_adapter(
    path: Path | str, *, model_id: str, revision: str, prompt_id: str, manifest_hash: str
) -> str:
    """The name of the selected checkpoint, ``RUN_ID/checkpoint-N``.

    Refused if the selection was made for another model commit, prompt wording, or split:
    arm A1 of another experiment is not arm A1 of this one.
    """
    record = read_selection(path)
    current = {
        "model_id": model_id,
        "revision": revision,
        "prompt_id": prompt_id,
        "manifest_hash": manifest_hash,
    }
    differs = [name for name in IDENTITY if record["key"].get(name) != current[name]]
    if differs:
        raise ConfigError(
            "the selection record was made for another experiment "
            f"({', '.join(differs)} differ); select again under the current settings"
        )
    return str(record["selected"]["adapter"])
