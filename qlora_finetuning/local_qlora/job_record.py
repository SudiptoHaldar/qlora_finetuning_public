"""The committed record of every SageMaker job (req_003 v3.03).

``reports/local_qlora/sagemaker_jobs.json`` is written the moment a job's ARN is known, and
updated by ``status`` and ``pull``. It is where later code and the owner find a job: its
name, what it ran, on what, where its files are, how it ended, and what it was billed for.
It holds names, ids, counts, and amounts: no example text.

Usage::

    from qlora_finetuning.local_qlora.job_record import latest_job, read_job_record

    record = read_job_record(cfg.record_file)
    job = latest_job(record)                 # or latest_job(record, "qlora-smoke-1a2b3c4d-a1")
    job["status"], job["billable_seconds"], job["workspace_uri"], job["runs"]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking.config import check_no_secrets

SCHEMA = 1
FINAL_STATUSES = ("Completed", "Failed", "Stopped")


def read_job_record(path: Path | str) -> dict[str, Any]:
    """The record, or an empty one if no job was ever launched."""
    path = Path(path)
    if not path.is_file():
        return {"schema": SCHEMA, "jobs": []}
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        raise ConfigError(f"the job record {path.name} is not valid JSON") from None
    if (
        not isinstance(record, dict)
        or record.get("schema") != SCHEMA
        or not isinstance(record.get("jobs"), list)
    ):
        raise ConfigError(f"the job record {path.name} is damaged")
    return record


def write_job_record(path: Path | str, record: dict[str, Any]) -> None:
    check_no_secrets(record, "the job record")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def find_job(record: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((job for job in record["jobs"] if job.get("name") == name), None)


def latest_job(record: dict[str, Any], name: str | None = None) -> dict[str, Any]:
    """The job called ``name``, or the last one launched. Raises ``ConfigError`` if none."""
    if name is not None:
        job = find_job(record, name)
        if job is None:
            known = ", ".join(job["name"] for job in record["jobs"]) or "none"
            raise ConfigError(f"no job called {name} in the job record (known: {known})")
        return job
    if not record["jobs"]:
        raise ConfigError("no job has been launched yet: the job record is empty")
    return record["jobs"][-1]


def upsert_job(record: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    """Put ``job`` into the record, replacing the one with its name; returns the record."""
    for position, existing in enumerate(record["jobs"]):
        if existing.get("name") == job["name"]:
            record["jobs"][position] = job
            return record
    record["jobs"].append(job)
    return record


def is_final(job: dict[str, Any]) -> bool:
    return job.get("status") in FINAL_STATUSES
