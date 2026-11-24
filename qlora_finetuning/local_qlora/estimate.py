"""How long a job will take, from what earlier jobs measured (req_003 v3.04).

The owner sees hours and dollars before launching a job. The figures are not guesses: they
are the work the job has to do (training sequences, replies) times the seconds that the
last pulled job of this project measured on the same instance type, plus that job's
overhead (installing the libraries, downloading and loading the model).

Usage::

    from qlora_finetuning.local_qlora.estimate import estimate_job, measured_rates

    rates = measured_rates(job_record, root, "ml.g5.xlarge")      # from pulled jobs' result files
    estimate = estimate_job("score", cfg, rates, examples=499, checkpoints=10)
    estimate.seconds, estimate.cost, estimate.proposed_max_runtime, estimate.detail

``None`` in place of an estimate means there is nothing to measure from yet: the smoke job
has not been run and pulled. Amounts are ``Decimal``. This module imports neither boto3 nor
the training stack.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from qlora_finetuning.local_qlora.job import JOBS_DIR, RESULT_NAME
from qlora_finetuning.local_qlora.sagemaker_config import SageMakerConfig
from qlora_finetuning.tracking import Ledger

TRAIN, TUNED, UNTUNED, OVERHEAD = "train", "tuned", "untuned", "overhead"
TUNED_ROLES = ("eval-tuned", "score-checkpoint")
UNTUNED_ROLES = ("eval-untuned", "score-untuned")


@dataclass
class Rates:
    """Seconds per unit of work, each from the newest pulled job that measured it."""

    seconds: dict[str, Decimal] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)  # which job each figure is from

    def get(self, name: str) -> Decimal | None:
        return self.seconds.get(name)

    def missing(self, names: tuple[str, ...]) -> list[str]:
        return [name for name in names if name not in self.seconds]


@dataclass(frozen=True)
class Estimate:
    """What a job is expected to take and cost, and the limit proposed for it."""

    task: str
    seconds: int
    cost: Decimal
    proposed_max_runtime: int
    detail: str
    sources: tuple[str, ...]


def _number(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return Decimal(str(value))


def rates_of(result: dict[str, Any], billable_seconds: int | None) -> dict[str, Decimal]:
    """The seconds per unit of work that one job's result file shows.

    A training run gives seconds per training sequence. Evaluation runs give seconds per
    reply, with an adapter and without. The overhead is the job's billable time less the
    training and generating it did: setup, model loads, and everything between.
    """
    work = Decimal(0)
    generated: dict[str, list[Decimal]] = {
        TUNED: [Decimal(0), Decimal(0)],
        UNTUNED: [Decimal(0), Decimal(0)],
    }
    rates: dict[str, Decimal] = {}
    for run in result.get("runs", []):
        if run.get("status") != "completed" or run.get("reused"):
            continue
        role = run.get("role")
        if role == TRAIN:
            trainer = run.get("trainer") or {}
            per_sequence = _number(trainer.get("seconds_per_sequence"))
            if per_sequence is not None:
                rates[TRAIN] = per_sequence
            work += _number(trainer.get("train_seconds")) or Decimal(0)
            continue
        backend = run.get("backend") or {}
        seconds, prompts = _number(backend.get("generate_seconds")), _number(backend.get("prompts"))
        if seconds is None or prompts is None:
            continue
        work += seconds
        tuned = role in TUNED_ROLES or (role == "eval" and bool(run.get("adapter")))
        kind = TUNED if tuned else UNTUNED
        generated[kind][0] += seconds
        generated[kind][1] += prompts
    for kind, (seconds, prompts) in generated.items():
        if prompts > 0:
            rates[kind] = seconds / prompts
    if billable_seconds is not None and billable_seconds > work:
        rates[OVERHEAD] = Decimal(billable_seconds) - work
    return rates


def measured_rates(record: dict[str, Any], root: Path | str, instance_type: str) -> Rates:
    """The newest figure for each kind of work, from pulled jobs on ``instance_type``."""
    rates = Rates()
    for job in reversed(record.get("jobs", [])):
        if not job.get("pulled") or job.get("instance_type") != instance_type:
            continue
        path = Path(root) / JOBS_DIR / str(job.get("name")) / RESULT_NAME
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for name, value in rates_of(result, job.get("billable_seconds")).items():
            if name not in rates.seconds:
                rates.seconds[name] = value
                rates.sources[name] = str(job["name"])
    return rates


def committed_path_a(
    cfg: SageMakerConfig, tracking_cfg: Any, record: dict[str, Any], excluding: str | None = None
) -> Decimal:
    """What Path A has cost so far, and may yet cost.

    Billable seconds in the ledger at the price of their instance type, plus the maximum of
    every job in the job record that was launched and not yet pulled. ``excluding`` leaves
    one job's maximum out: the job that is being planned.
    """
    total = Decimal(0)
    for line in Ledger(tracking_cfg).effective():
        if line.path == "A" and line.billable_seconds:
            count = line.instance_count or 1
            total += cfg.cost_of(line.billable_seconds, line.instance_type, count)
    for job in record.get("jobs", []):
        if not job.get("pulled") and job.get("name") != excluding:
            total += Decimal(str(job.get("max_cost_usd", "0")))
    return total


def _hours(seconds: Decimal | int) -> str:
    return f"{Decimal(seconds) / 3600:.1f}"


def estimate_job(
    task: str,
    cfg: SageMakerConfig,
    rates: Rates,
    *,
    sequences: int = 0,
    examples: int = 0,
    checkpoints: int = 0,
    tuned: bool = False,
) -> Estimate | None:
    """The estimate for a job, or ``None`` when a figure it needs was never measured.

    ``train``: ``sequences`` training sequences. ``score``: ``examples`` replies from the
    untuned model and from each of ``checkpoints`` checkpoints. ``evaluate``: ``examples``
    replies, from a tuned model or the untuned one.
    """
    if task == "train":
        needs = (TRAIN, OVERHEAD)
    elif task == "score":
        needs = (TUNED, UNTUNED, OVERHEAD)
    else:
        needs = (TUNED if tuned else UNTUNED, OVERHEAD)
    if rates.missing(needs):
        return None

    overhead = rates.seconds[OVERHEAD]
    if task == "train":
        work = sequences * rates.seconds[TRAIN]
        detail = f"{sequences:,} training sequences at {rates.seconds[TRAIN]:.2f} s"
    elif task == "score":
        work = examples * (rates.seconds[UNTUNED] + checkpoints * rates.seconds[TUNED])
        detail = (
            f"{examples:,} replies from the untuned model at {rates.seconds[UNTUNED]:.2f} s and "
            f"from each of {checkpoints} checkpoints at {rates.seconds[TUNED]:.2f} s"
        )
    else:
        rate = rates.seconds[needs[0]]
        work = examples * rate
        detail = f"{examples:,} replies at {rate:.2f} s"
    seconds = math.ceil(work + overhead)
    with_margin = math.ceil(Decimal(seconds) * cfg.runtime_margin / 60) * 60
    return Estimate(
        task=task,
        seconds=seconds,
        cost=cfg.cost_of(seconds, count=cfg.instance_count),
        proposed_max_runtime=max(cfg.minimum_runtime_seconds, with_margin),
        detail=(
            f"{detail}, plus {_hours(overhead)} hours of overhead: about {_hours(seconds)} hours"
        ),
        sources=tuple(sorted({rates.sources[name] for name in needs})),
    )
