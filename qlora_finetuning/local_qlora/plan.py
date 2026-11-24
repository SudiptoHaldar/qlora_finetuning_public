"""The arithmetic of a training run: steps, warmup, checkpoints, the worst batch (req_003 v3.02).

Pure functions on sequence lengths. The dry run prints the plan; the run trains by it and
checks the trainer's own step count against it.

Usage::

    from qlora_finetuning.local_qlora.plan import select_smoke, training_plan

    plan = training_plan(lengths, cfg, smoke=False)
    plan.total_steps, plan.warmup_steps, plan.checkpoint_steps
    plan.worst_batch, plan.worst_batch_width      # the longest sequences, probed first

    positions = select_smoke(lengths, examples=64, batch_size=2)   # which records a smoke run uses
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from qlora_finetuning.local_qlora.training_config import TrainingConfig


@dataclass(frozen=True)
class TrainingPlan:
    """What a run will do, from the lengths of its sequences and the config."""

    records: int
    tokens_per_pass: int  # every sequence once
    batch_size: int
    accumulation: int
    epochs: int
    steps_per_epoch: int  # optimizer steps
    total_steps: int
    warmup_steps: int
    save_steps: int
    checkpoint_steps: tuple[int, ...]  # every save_steps, and the last step
    worst_batch: tuple[int, ...]  # positions of the batch_size longest sequences
    worst_batch_width: int  # the longest of them, in tokens

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["checkpoint_steps"] = list(self.checkpoint_steps)
        data["worst_batch"] = list(self.worst_batch)
        return data

    def tokens_for_steps(self, steps: int) -> int:
        """Tokens processed in ``steps`` optimizer steps, as a share of the whole run's."""
        if self.total_steps <= 0:
            return 0
        return round(
            self.tokens_per_pass * self.epochs * min(steps, self.total_steps) / self.total_steps
        )


def longest(lengths: Sequence[int], count: int) -> tuple[int, ...]:
    """Positions of the ``count`` longest sequences, longest first; ties go to the earlier."""
    order = sorted(range(len(lengths)), key=lambda position: (-lengths[position], position))
    return tuple(order[:count])


def select_smoke(lengths: Sequence[int], examples: int, batch_size: int) -> list[int]:
    """Which records a smoke run trains on, as positions in dataset order.

    The longest records that fill one batch come first, so the smoke run meets the worst
    batch of the full run. The rest are the first records of the split.
    """
    chosen = list(longest(lengths, batch_size))
    for position in range(len(lengths)):
        if len(chosen) >= examples:
            break
        if position not in chosen:
            chosen.append(position)
    return chosen


def training_plan(lengths: Sequence[int], cfg: TrainingConfig, smoke: bool) -> TrainingPlan:
    """The plan for sequences of these lengths.

    The step count follows ``transformers.Trainer``: a last, smaller group of batches is a
    step of its own, and the total is the epochs times the steps of one epoch, rounded up.
    """
    records = len(lengths)
    if records < cfg.batch_size:
        raise ValueError(f"{records} records cannot fill a batch of {cfg.batch_size}")
    epochs, save_steps = cfg.epochs_for(smoke), cfg.save_steps_for(smoke)
    batches = math.ceil(records / cfg.batch_size)
    steps_per_epoch = max(math.ceil(batches / cfg.accumulation), 1)
    total_steps = math.ceil(epochs * steps_per_epoch)
    checkpoints = list(range(save_steps, total_steps + 1, save_steps))
    if not checkpoints or checkpoints[-1] != total_steps:
        checkpoints.append(total_steps)
    worst = longest(lengths, cfg.batch_size)
    return TrainingPlan(
        records=records,
        tokens_per_pass=sum(lengths),
        batch_size=cfg.batch_size,
        accumulation=cfg.accumulation,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        warmup_steps=math.ceil(cfg.warmup_fraction * total_steps),
        save_steps=save_steps,
        checkpoint_steps=tuple(checkpoints),
        worst_batch=worst,
        worst_batch_width=max(lengths[position] for position in worst),
    )
