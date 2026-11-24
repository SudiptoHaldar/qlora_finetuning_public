"""Tests for SageMakerJobs.pull: a finished job's runs, ledger lines, and billable seconds.

The workspace is a stand-in S3 bucket filled with what a job would have written.
"""

from __future__ import annotations

import json

import pytest

from qlora_finetuning.evaluation.metrics import load_evaluation_config
from qlora_finetuning.local_qlora.job_record import read_job_record
from qlora_finetuning.local_qlora.sagemaker import JobOptions
from qlora_finetuning.tracking import Ledger
from qlora_finetuning.tracking.jsonl import read_lines

from .sagemaker_helpers import make_jobs, put_json_lines, workspace_key

TRAIN = "20261120T100000Z-smoke-qwen3-4b-smoke-aaaa"
TUNED = "20261120T101000Z-eval-qwen3-4b-tuned-quick-dev-bbbb"
OTHER = "20261119T000000Z-eval-someone-elses-cccc"


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    return make_jobs(tmp_path, monkeypatch)


def ledger_line(run_id, record_id, **more):
    line = {
        "schema": 1, "run_id": run_id, "phase": "train", "path": "A", "arm": "none",
        "where": "sagemaker", "source": "computed", "source_detail": "", "record_id": record_id,
        "recorded_at": "2026-11-20T10:05:00Z", "wall_clock_seconds": 170.0, "examples": 64,
        "note": "",
    }  # fmt: skip
    line.update(more)
    return line


def fill_workspace(jobs, job, result=True):
    """What the smoke job leaves in its workspace."""
    s3 = jobs.s3

    def put(relative, content):
        s3.objects[workspace_key(jobs, job, relative)] = content

    put(f"artifacts/runs/{TRAIN}/manifest.json", b'{"run_id": "train"}')
    put(f"artifacts/runs/{TRAIN}/training_log.jsonl", b'{"step": 1, "loss": 2.27}\n')
    put(f"artifacts/runs/{TRAIN}/checkpoints/checkpoint-4/adapter_model.safetensors", b"w" * 50)
    put(f"artifacts/runs/{TUNED}/predictions.jsonl", b'{"example_id": "x"}\n')
    put(f"artifacts/runs/{OTHER}/manifest.json", b"an earlier job's run on the same workspace")
    put(f"artifacts/adapters/{TRAIN}/adapter_model.safetensors", b"final weights")
    put(f"artifacts/adapters/{TRAIN}/adapter_metadata.json", b'{"steps": 12}')
    if result:
        runs = [
            {"role": "train", "run_id": TRAIN, "status": "completed"},
            {"role": "eval-tuned", "run_id": TUNED, "status": "completed"},
        ]
        content = {"status": "completed", "task": "smoke", "runs": runs}
        put(f"artifacts/jobs/{job['name']}/job_result.json", json.dumps(content).encode())
    put_json_lines(
        s3,
        workspace_key(jobs, job, "ledger/runs.jsonl"),
        [
            {"schema": 1, "event": "started", "run_id": TRAIN, "at": "2026-11-20T10:00:00Z"},
            {"schema": 1, "event": "finished", "run_id": TRAIN, "at": "2026-11-20T10:05:00Z"},
        ],
    )
    put_json_lines(
        s3,
        workspace_key(jobs, job, "ledger/ledger.jsonl"),
        [
            ledger_line(TRAIN, "1" * 32, model_id="Qwen/Qwen3-4B", training_tokens=136_191),
            ledger_line(TUNED, "2" * 32, phase="eval", arm="A1", source="measured", examples=24),
        ],
    )
    put_json_lines(
        s3,
        workspace_key(jobs, job, "ledger/evaluations.jsonl"),
        [{"schema": 1, "run_id": TUNED, "arm": "A1", "split": "quick_dev", "examples": 200}],
    )


def launched(jobs, finish=True, **finish_options):
    _, job = jobs.launch(JobOptions("smoke"), yes=True)
    if finish:
        jobs.sagemaker.finish(job["name"], **finish_options)
    return job


def results_file(jobs):
    return load_evaluation_config(root=jobs.tracking.root).results_file


def test_pull_brings_the_runs_the_ledger_lines_and_the_billable_seconds(jobs, tmp_path):
    job = launched(jobs, billable=1500)
    fill_workspace(jobs, job)
    outcome = jobs.pull()
    assert outcome["pulled"] and outcome["result"]["status"] == "completed"

    runs = tmp_path / "artifacts" / "runs"
    assert (runs / TRAIN / "manifest.json").read_bytes() == b'{"run_id": "train"}'
    assert (runs / TRAIN / "training_log.jsonl").is_file()
    assert (runs / TUNED / "predictions.jsonl").is_file()
    assert (tmp_path / "artifacts" / "adapters" / TRAIN / "adapter_model.safetensors").is_file()
    assert (tmp_path / "artifacts" / "jobs" / job["name"] / "job_result.json").is_file()
    # Checkpoints stay in S3 unless asked for, and another job's run is not this job's.
    assert not (runs / TRAIN / "checkpoints").exists() and not (runs / OTHER).exists()
    assert outcome["files"] == 6

    assert outcome["ledger_lines"] == {"runs": 2, "ledger": 2, "results": 1}
    index = read_lines(jobs.tracking.runs_index_file)
    assert [(line["run_id"], line["event"]) for line in index if line["run_id"] == TRAIN] == [
        (TRAIN, "started"),
        (TRAIN, "finished"),
    ]
    assert [line["run_id"] for line in read_lines(results_file(jobs))] == [TUNED]

    records = Ledger(jobs.tracking).read()
    assert [record.record_id for record in records[:2]] == ["1" * 32, "2" * 32]
    assert records[0].training_tokens == 136_191 and records[1].arm == "A1"
    billed = records[2]
    assert outcome["billable_recorded"] is True
    assert (billed.run_id, billed.external_id) == (job["launch_run_id"], job["name"])
    assert (billed.instance_type, billed.instance_count, billed.billable_seconds) == (
        "ml.g5.xlarge",
        1,
        1500,
    )
    assert (billed.source, billed.phase, billed.path, billed.arm, billed.where) == (
        "aws_api",
        "other",
        "A",
        "none",
        "sagemaker",
    )
    assert billed.billed_amount_usd is None  # seconds, never dollars

    stored = read_job_record(jobs.cfg.record_file)["jobs"][0]
    assert stored["pulled"] is True and stored["status"] == "Completed"
    assert stored["billable_seconds"] == 1500 and stored["result_status"] == "completed"
    assert [run["run_id"] for run in stored["runs"]] == [TRAIN, TUNED]


def test_pulling_again_adds_nothing(jobs):
    job = launched(jobs)
    fill_workspace(jobs, job)
    jobs.pull()
    before = (
        Ledger(jobs.tracking).read(),
        read_lines(jobs.tracking.runs_index_file),
        read_lines(results_file(jobs)),
    )
    downloads = len(jobs.s3.gets)
    again = jobs.pull()
    assert again["ledger_lines"] == {"runs": 0, "ledger": 0, "results": 0}
    assert again["billable_recorded"] is False and again["files"] == 0
    assert (
        Ledger(jobs.tracking).read(),
        read_lines(jobs.tracking.runs_index_file),
        read_lines(results_file(jobs)),
    ) == before
    # Only the result file and the three ledger files were read again; no run file.
    assert len(jobs.s3.gets) - downloads == 4


def test_checkpoints_come_only_when_asked_for(jobs, tmp_path):
    job = launched(jobs)
    fill_workspace(jobs, job)
    jobs.pull()
    weights = (
        tmp_path / "artifacts" / "runs" / TRAIN / "checkpoints" / "checkpoint-4"
        / "adapter_model.safetensors"
    )  # fmt: skip
    assert not weights.exists()
    assert jobs.pull(checkpoints=True)["files"] == 1 and weights.stat().st_size == 50


def test_a_job_that_is_still_running_is_not_pulled(jobs, tmp_path):
    job = launched(jobs, finish=False)
    fill_workspace(jobs, job)
    outcome = jobs.pull()
    assert outcome["pulled"] is False and outcome["reason"] == "the job is InProgress"
    assert (
        Ledger(jobs.tracking).read() == []
        and not (tmp_path / "artifacts" / "runs" / TRAIN).exists()
    )
    assert read_job_record(jobs.cfg.record_file)["jobs"][0]["pulled"] is False


def test_a_failed_job_is_billed_and_its_runs_are_found_without_a_result_file(jobs, tmp_path):
    job = launched(jobs, status="Failed", billable=420, FailureReason="AlgorithmError: exit 1")
    fill_workspace(jobs, job, result=False)
    outcome = jobs.pull()
    assert outcome["pulled"] and outcome["result"] == {}
    # No result file: every run the workspace holds is taken.
    for run_id in (TRAIN, TUNED, OTHER):
        assert (tmp_path / "artifacts" / "runs" / run_id).is_dir()
    billed = Ledger(jobs.tracking).read()[-1]
    assert (billed.billable_seconds, billed.note) == (420, "job status Failed")
    stored = read_job_record(jobs.cfg.record_file)["jobs"][0]
    assert stored["result_status"] is None and stored["failure_reason"] == "AlgorithmError: exit 1"
    assert sorted(run["run_id"] for run in stored["runs"]) == sorted([TRAIN, TUNED, OTHER])


def test_a_stopped_job_with_an_empty_workspace_still_records_its_seconds(jobs):
    job = launched(jobs, finish=False)
    jobs.stop(yes=True)
    outcome = jobs.pull()
    assert outcome["pulled"] and outcome["files"] == 0
    (billed,) = Ledger(jobs.tracking).read()
    assert (billed.billable_seconds, billed.external_id) == (120, job["name"])


def test_a_key_that_would_leave_the_folder_is_not_written(jobs, tmp_path):
    job = launched(jobs)
    fill_workspace(jobs, job)
    jobs.s3.objects[workspace_key(jobs, job, f"artifacts/runs/{TRAIN}/../../../escape.txt")] = b"x"
    jobs.pull()
    assert not (tmp_path / "escape.txt").exists() and not (tmp_path.parent / "escape.txt").exists()


def test_nothing_pulled_into_a_committed_file_holds_example_text(jobs):
    job = launched(jobs)
    fill_workspace(jobs, job)
    jobs.s3.objects[workspace_key(jobs, job, f"artifacts/runs/{TUNED}/predictions.jsonl")] = (
        b'{"reply": "zebrafish revenue rose"}\n'
    )
    jobs.pull()
    committed = [
        jobs.cfg.record_file,
        jobs.tracking.ledger_file,
        jobs.tracking.runs_index_file,
        results_file(jobs),
    ]
    for path in committed:
        assert "zebrafish" not in path.read_text(encoding="utf-8"), path.name
