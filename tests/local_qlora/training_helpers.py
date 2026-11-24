"""Stand-ins for the training tests: a context under ``tmp_path`` and an engine without a model.

The invented split has 30 fit records and 8 dev records. With the committed batch (2 x 8) a
smoke run of them is 2 optimizer steps an epoch, 6 in all, with checkpoints at 4 and 6.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from qlora_finetuning.local_qlora.adapters import (
    ADAPTER_CONFIG,
    ADAPTER_WEIGHTS,
    adapter_sha256,
)
from qlora_finetuning.local_qlora.config import load_local_qlora_config
from qlora_finetuning.local_qlora.training import (
    TRAINER_STATE,
    TrainingContext,
    TrainingStopped,
)
from qlora_finetuning.local_qlora.training_config import load_training_config
from tests.data.synthetic import SECRET_WORDS, FakeTokenizer, fake_budget
from tests.evaluation.conftest import GIT, make_context

CONFIGS = ("training", "local_qlora", "token_budget", "prompt")


def make_training_context(root: Path, **changes: Any) -> TrainingContext:
    """Everything a training run reads, all under ``root``, with the stand-in tokenizer."""
    runner = make_context(root)
    tokenizer = FakeTokenizer()
    context = TrainingContext(
        training_cfg=load_training_config(root=root),
        local_cfg=load_local_qlora_config(),
        budget=fake_budget(root, tokenizer),
        prompt=runner.prompt,
        split_cfg=runner.split_cfg,
        data_cfg=runner.data_cfg,
        tracking_cfg=runner.tracking_cfg,
        tokenizer=tokenizer,
        git=GIT,
        environment={},
        configs=CONFIGS,
    )
    return dataclasses.replace(context, **changes)


def with_training(context: TrainingContext, **changes: Any) -> TrainingContext:
    """The context with some training settings changed."""
    return dataclasses.replace(
        context, training_cfg=dataclasses.replace(context.training_cfg, **changes)
    )


def write_checkpoint(folder: Path, step: int, resumable: bool = True) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ADAPTER_WEIGHTS).write_bytes(f"weights after step {step}".encode())
    (folder / ADAPTER_CONFIG).write_text('{"r": 16}', encoding="utf-8")
    if resumable:
        (folder / TRAINER_STATE).write_text(json.dumps({"global_step": step}), encoding="utf-8")


class StubEngine:
    """Stands in for ``TrlEngine``: walks the plan's steps and writes checkpoint folders.

    ``fail_at=N`` raises ``error`` in step N. ``losses`` gives the loss of each step.
    ``end_early`` stops one step short without raising, as a broken trainer might.
    """

    def __init__(
        self,
        *,
        fail_at: int | None = None,
        error: Exception | None = None,
        losses: Any = None,
        end_early: bool = False,
        unchanged: bool = True,
    ) -> None:
        self.fail_at = fail_at
        self.error = error or RuntimeError("CUDA error: zebrafish in the message")
        self.losses = losses or (lambda step: 2.0 / step)
        self.end_early = end_early
        self.unchanged = unchanged
        self.jobs: list[Any] = []

    def fit(self, job) -> None:
        self.jobs.append(job)
        plan = job.plan
        job.progress.update(
            trainable_parameters=1234,
            only_adapter_trains=True,
            dataset_unchanged=self.unchanged,
            probe={
                "sequences": plan.batch_size,
                "width": plan.worst_batch_width,
                "loss": 2.5,
                "seconds": 0.1,
                "peak_vram_allocated_gib": 1.5,
                "peak_vram_reserved_gib": 1.75,
            },
        )
        last = plan.total_steps - (1 if self.end_early else 0)
        for step in range(job.start_step + 1, last + 1):
            if job.stop_after_steps is not None and step - 1 >= job.stop_after_steps:
                raise TrainingStopped(f"stopped on request after {step - 1} steps")
            if self.fail_at == step:
                raise self.error
            job.log({"step": step, "epoch": step / plan.steps_per_epoch, "loss": self.losses(step)})
            job.progress["steps_done"] = step
            if step in plan.checkpoint_steps:
                write_checkpoint(job.checkpoints_dir / f"checkpoint-{step}", step)
                job.log({"step": step, "eval_loss": 1.0 / step})
        job.progress["final_checkpoint"] = str(job.checkpoints_dir / f"checkpoint-{last}")
        job.progress["trainer"] = {
            "train_seconds": 1.0,
            "seconds_per_step": 0.5,
            "seconds_per_sequence": 0.03,
            "peak_vram_allocated_gib": 1.25,
            "peak_vram_reserved_gib": 1.5,
        }


class StubReloader:
    """Stands in for ``reload_adapter``: the adapter 'answers' three prompts."""

    def __init__(self, error: Exception | None = None, **changes: Any) -> None:
        self.error = error
        self.changes = changes
        self.calls: list[Path] = []

    def __call__(self, context, adapter_dir: Path) -> dict[str, Any]:
        self.calls.append(Path(adapter_dir))
        if self.error is not None:
            raise self.error
        result = {
            "adapter_sha256": adapter_sha256(adapter_dir),
            "asked": 3,
            "replies": 3,
            "think_markers": 0,
            "programs_that_run": 3,
            "identical_replies": 2,
            "batch_size": 8,
        }
        result.update(self.changes)
        return result


def files_under(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file())


def holds_example_text(folder: Path) -> list[str]:
    """Files under ``folder`` that hold a word of the invented dataset."""
    hits = []
    for path in files_under(folder):
        text = path.read_bytes().decode("utf-8", "replace").lower()
        if any(word in text for word in SECRET_WORDS):
            hits.append(path.name)
    return hits
