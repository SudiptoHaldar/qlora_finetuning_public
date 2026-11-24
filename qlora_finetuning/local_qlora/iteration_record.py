"""The record of the one corrective iteration (req_003 v3.05).

``reports/local_qlora/iteration.json`` is committed. It is written by
``python -m scripts.qlora_iterate propose`` **before** the second training run is paid for:
which run missed, what the analysis found, the one setting that changes, and why. The
selection adds how it ended.

Usage::

    from qlora_finetuning.local_qlora.iteration_record import read_iteration

    record = read_iteration(training_cfg.iteration_record)     # None: no change is registered
    record["variable"], record["from"], record["to"], record["settings_sha256"]
    record["second_run"], record["outcome"]                    # None until the second selection

This module holds only the file's reading and writing, so that the selection, the launcher,
and the iteration's own checks can all use it without importing each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qlora_finetuning.local_qlora.adapters import read_json, write_json
from qlora_finetuning.settings import ConfigError

ITERATION_SCHEMA = 1
_REQUIRED = ("first_run", "variable", "from", "to", "settings_sha256", "first_settings_sha256")


def read_iteration(path: Path | str | None) -> dict[str, Any] | None:
    """The iteration record, or ``None`` if no change is registered. Raises if damaged."""
    if path is None or not Path(path).is_file():
        return None
    record = read_json(Path(path), "the iteration record")
    if record.get("schema") != ITERATION_SCHEMA or any(name not in record for name in _REQUIRED):
        raise ConfigError(f"the iteration record {Path(path).name} is damaged")
    return record


def write_iteration(path: Path | str, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, record)


def record_outcome(path: Path | str | None, second_run: str, outcome: dict[str, Any]) -> None:
    """Note in the record which run was the second, and how the comparison ended."""
    record = read_iteration(path)
    if record is None:
        raise ConfigError("there is no iteration record to complete")
    record["second_run"] = second_run
    record["outcome"] = {
        name: outcome.get(name)
        for name in ("winner", "gap_examples", "within_noise", "verdict", "selected")
    }
    write_iteration(Path(path), record)


def second_run_made(iteration: dict[str, Any] | None, job_record: dict[str, Any]) -> str | None:
    """The second run or job, if the one corrective iteration has already been made.

    Either the selection has compared a second run, or a job launched as the iteration's
    training has completed. A job that failed or was stopped has not made it: it is
    resumed, or launched again.
    """
    if iteration is not None and iteration.get("second_run"):
        return str(iteration["second_run"])
    for job in job_record.get("jobs", []):
        if job.get("iteration") and job.get("status") == "Completed":
            return str(job.get("name"))
    return None
