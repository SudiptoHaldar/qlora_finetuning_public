"""The tuning record: what was launched on Bedrock, and what became of it (req_004 v4.03).

One entry per model and attempt: the job's name and request token, its hyperparameters and
S3 locations, the ARN Bedrock returned, how the job ended, the tuned model, its losses, its
on-demand deployment, and the amount billed once the owner has read it. The record is the
only memory of a job across sessions, so it is written the moment a job ARN is known, and
it is committed: ``reports/frontier/nova_tuning.json`` with a Markdown twin.

It holds names, ARNs, numbers, and dates. It never holds a line of a training file.

Usage::

    from qlora_finetuning.frontier.tuning_record import read_tuning_record

    attempt = read_tuning_record().latest("nova-lite")
    attempt.job_status            # "Completed"
    attempt.deployment_arn        # the modelId to ask the tuned model with (v4.04)
    attempt.deployment_status     # "Active"

This module calls no AWS service. ``qlora_finetuning.frontier.tuning`` fills the record.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qlora_finetuning.frontier.config import FrontierConfig, load_frontier_config
from qlora_finetuning.settings import ConfigError

SCHEMA = 1
REPORT_STEM = "nova_tuning"
REPORT_COMMAND = "python -m scripts.nova_tune"

IN_PROGRESS, COMPLETED, FAILED, STOPPING, STOPPED = (
    "InProgress",
    "Completed",
    "Failed",
    "Stopping",
    "Stopped",
)
ENDED = (COMPLETED, FAILED, STOPPED)  # a job in one of these will not change again
CREATING, ACTIVE = "Creating", "Active"


@dataclass
class Attempt:
    """One launch of one model. Everything after ``job_arn`` is filled as it becomes known."""

    model: str  # the name in config/frontier.yaml, for example nova-lite
    attempt: int
    arm: str  # the tuned arm, for example B1-Lite
    base_model_id: str  # the inference id of the base model: what prices are keyed by
    customization_model_id: str
    job_name: str
    custom_model_name: str
    deployment_name: str
    client_request_token: str
    hyperparameters: dict[str, str]
    export_id: str
    training_s3_uri: str
    validation_s3_uri: str
    output_s3_uri: str
    estimated_training_tokens: int
    estimated_usd: str
    run_id: str | None = None  # the launch, in the run registry
    job_arn: str | None = None
    job_status: str | None = None
    failed_in: str | None = None  # validation | data processing | training
    failure_message: str | None = None
    created_at: str | None = None
    ended_at: str | None = None
    model_arn: str | None = None
    training_loss: float | None = None
    validation_loss: float | None = None
    metrics: dict[str, Any] | None = None  # the loss curves, summarized
    deployment_arn: str | None = None
    deployment_status: str | None = None
    deployment_failure: str | None = None
    billed_usd: str | None = None  # read from the AWS bill by the owner
    deployment_deleted: bool = False
    model_deleted: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, int]:
        return self.model, self.attempt

    @property
    def ended(self) -> bool:
        return self.job_status in ENDED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Attempt:
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"the tuning record has unknown field(s): {', '.join(unknown)}")
        try:
            return cls(**data)
        except TypeError as error:
            raise ConfigError(f"the tuning record has an incomplete entry: {error}") from None


@dataclass
class TuningRecord:
    """Every attempt, in the order they were launched."""

    attempts: list[Attempt] = field(default_factory=list)

    def find(self, model: str, attempt: int) -> Attempt | None:
        return next((item for item in self.attempts if item.key == (model, attempt)), None)

    def of(self, model: str) -> list[Attempt]:
        return sorted(
            (item for item in self.attempts if item.model == model), key=lambda a: a.attempt
        )

    def latest(self, model: str) -> Attempt | None:
        """The model's highest-numbered attempt, or ``None`` if it was never launched."""
        attempts = self.of(model)
        return attempts[-1] if attempts else None

    def put(self, attempt: Attempt) -> None:
        """Add an attempt, or replace the one with the same model and number."""
        for index, existing in enumerate(self.attempts):
            if existing.key == attempt.key:
                self.attempts[index] = attempt
                return
        self.attempts.append(attempt)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "generated_by": REPORT_COMMAND,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


def as_text(moment: Any) -> str | None:
    """A time from boto3 as UTC text (``2026-11-17T04:05:06Z``); ``None`` stays ``None``."""
    if moment is None:
        return None
    if isinstance(moment, datetime):
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(moment)


def report_paths(report_dir: Path | str) -> tuple[Path, Path]:
    report_dir = Path(report_dir)
    return report_dir / f"{REPORT_STEM}.md", report_dir / f"{REPORT_STEM}.json"


def read_tuning_record(cfg: FrontierConfig | None = None) -> TuningRecord:
    """The committed tuning record; an empty one if nothing was ever launched."""
    _, path = report_paths((cfg or load_frontier_config()).report_dir)
    if not path.is_file():
        return TuningRecord()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["schema"] != SCHEMA:
            raise ConfigError(f"the tuning record {path.name} has an unknown schema")
        return TuningRecord([Attempt.from_dict(item) for item in data["attempts"]])
    except (ValueError, KeyError, TypeError):
        raise ConfigError(f"the tuning record {path.name} is damaged") from None


def _cell(value: Any) -> str:
    return "-" if value in (None, "") else str(value)


def _loss(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def render_markdown(record: TuningRecord) -> str:
    lines = [
        "# Nova tuning record",
        "",
        f"Written by `{REPORT_COMMAND}`. Do not edit by hand.",
        "",
        "What was launched on Bedrock for Path B and what became of it: jobs, tuned models,",
        "and on-demand deployments. Names, ARNs, numbers, and dates only. Dollar amounts are",
        "estimates at list price until the owner has entered what the AWS bill says.",
        "",
    ]
    if not record.attempts:
        return "\n".join([*lines, "No job has been launched.", ""])
    lines += [
        "## Jobs",
        "",
        "| Model | Attempt | Arm | Job status | Failed in | Created | Ended | Estimate | Billed |",
        "|---|---:|---|---|---|---|---|---:|---:|",
    ]
    lines += [
        f"| {a.model} | {a.attempt} | {a.arm} | {_cell(a.job_status)} | {_cell(a.failed_in)} "
        f"| {_cell(a.created_at)} | {_cell(a.ended_at)} | ${a.estimated_usd} "
        f"| {'$' + a.billed_usd if a.billed_usd else '-'} |"
        for a in record.attempts
    ]
    lines += [
        "",
        "## Tuned models and deployments",
        "",
        "| Model | Attempt | Training loss | Validation loss | Deployment | Deleted |",
        "|---|---:|---:|---:|---|---|",
    ]
    for a in record.attempts:
        deleted = ", ".join(
            name
            for name, gone in (("deployment", a.deployment_deleted), ("model", a.model_deleted))
            if gone
        )
        lines.append(
            f"| {a.model} | {a.attempt} | {_loss(a.training_loss)} | {_loss(a.validation_loss)} "
            f"| {_cell(a.deployment_status)} | {deleted or '-'} |"
        )
    for a in record.attempts:
        lines += [
            "",
            f"## {a.model}, attempt {a.attempt}",
            "",
            f"- Base model for the job: `{a.customization_model_id}`",
            f"- Hyperparameters: `{json.dumps(a.hyperparameters, sort_keys=True)}`",
            f"- Export id: `{a.export_id}`",
            f"- Training file: `{a.training_s3_uri}`",
            f"- Validation file: `{a.validation_s3_uri}`",
            f"- Output: `{a.output_s3_uri}`",
            f"- Estimated training tokens: {a.estimated_training_tokens:,}",
            f"- Job: `{_cell(a.job_arn)}` (name `{a.job_name}`)",
            f"- Tuned model: `{_cell(a.model_arn)}`",
            f"- Deployment: `{_cell(a.deployment_arn)}`",
            f"- Launch run: `{_cell(a.run_id)}`",
        ]
        if a.failure_message:
            lines.append(f"- Failure message from Bedrock: {a.failure_message}")
        if a.deployment_failure:
            lines.append(f"- Deployment failure from Bedrock: {a.deployment_failure}")
        for name, curve in sorted((a.metrics or {}).items()):
            lines.append(
                f"- {name.capitalize()} loss over {curve['steps']:,} steps: first "
                f"{_loss(curve['first'])}, last {_loss(curve['last'])}, lowest "
                f"{_loss(curve['lowest'])} at step {curve['lowest_step']}"
            )
        lines += [f"- Note: {note}" for note in a.notes]
    return "\n".join(lines) + "\n"


def render_json(record: TuningRecord) -> str:
    return json.dumps(record.to_dict(), indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def write_tuning_record(record: TuningRecord, cfg: FrontierConfig) -> tuple[Path, Path]:
    """Write the Markdown and JSON files and return their paths."""
    markdown_path, json_path = report_paths(cfg.report_dir)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(record), encoding="utf-8", newline="\n")
    json_path.write_text(render_json(record), encoding="utf-8", newline="\n")
    return markdown_path, json_path
