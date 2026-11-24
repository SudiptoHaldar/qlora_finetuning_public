"""Tests for qlora_finetuning.frontier.tuning_record: the committed record of jobs."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from qlora_finetuning.frontier.tuning_record import (
    Attempt,
    TuningRecord,
    as_text,
    read_tuning_record,
    render_markdown,
    report_paths,
    write_tuning_record,
)
from qlora_finetuning.settings import ConfigError

ARN = "arn:aws:bedrock:us-east-1:123456789012"


def attempt(model="nova-lite", number=1, **changes) -> Attempt:
    values = {
        "model": model,
        "attempt": number,
        "arm": "B1-Lite" if model == "nova-lite" else "B1-Pro",
        "base_model_id": f"amazon.{model}-v1:0",
        "customization_model_id": f"amazon.{model}-v1:0:300k",
        "job_name": f"qlora-finqa-{model}-abc123-a{number}",
        "custom_model_name": f"qlora-finqa-{model}-abc123-a{number}",
        "deployment_name": f"qlora-finqa-{model}-abc123-a{number}",
        "client_request_token": "qlora-0123",
        "hyperparameters": {"epochCount": "2", "learningRate": "0.00001"},
        "export_id": "abc123",
        "training_s3_uri": "s3://bucket/data/bedrock/abc123/finqa_fit.jsonl",
        "validation_s3_uri": "s3://bucket/data/bedrock/abc123/finqa_dev.jsonl",
        "output_s3_uri": f"s3://bucket/bedrock/abc123/{model}-a{number}/",
        "estimated_training_tokens": 75_000,
        "estimated_usd": "0.15",
        "job_arn": f"{ARN}:model-customization-job/x/{number}",
        "job_status": "InProgress",
    }
    values.update(changes)
    return Attempt(**values)


def test_nothing_launched_reads_as_an_empty_record(cfg):
    record = read_tuning_record(cfg)
    assert record.attempts == [] and record.latest("nova-lite") is None
    assert "No job has been launched." in render_markdown(record)


def test_find_latest_and_put(cfg):
    record = TuningRecord()
    record.put(attempt(number=1, job_status="Failed"))
    record.put(attempt("nova-pro", 1))
    record.put(attempt(number=2))
    assert [a.key for a in record.attempts] == [("nova-lite", 1), ("nova-pro", 1), ("nova-lite", 2)]
    assert record.find("nova-lite", 1).job_status == "Failed"
    assert record.latest("nova-lite").attempt == 2 and record.latest("nova-pro").attempt == 1
    assert [a.attempt for a in record.of("nova-lite")] == [1, 2]
    assert record.find("nova-lite", 3) is None and record.latest("nova-micro") is None

    record.put(attempt(number=1, job_status="Stopped"))  # the same key replaces, in place
    assert len(record.attempts) == 3 and record.attempts[0].job_status == "Stopped"


def test_an_attempt_knows_when_its_job_has_ended():
    assert not attempt(job_status="InProgress").ended
    assert not attempt(job_status="Stopping").ended
    for status in ("Completed", "Failed", "Stopped"):
        assert attempt(job_status=status).ended, status


def test_the_record_round_trips_through_its_files(cfg, tmp_path):
    record = TuningRecord()
    record.put(
        attempt(
            job_status="Completed",
            model_arn=f"{ARN}:custom-model/x",
            training_loss=0.6,
            validation_loss=0.7,
            metrics={"training": {"steps": 4, "first": 2.5, "last": 0.6, "lowest": 0.4,
                                  "lowest_step": 2}},
            deployment_arn=f"{ARN}:custom-model-deployment/x",
            deployment_status="Active",
            billed_usd="0.14",
            created_at="2026-11-17T04:05:06Z",
            notes=["a note"],
        )
    )  # fmt: skip
    markdown_path, json_path = write_tuning_record(record, cfg)
    assert (markdown_path, json_path) == report_paths(cfg.report_dir)
    assert json_path == tmp_path / "reports" / "frontier" / "nova_tuning.json"
    assert read_tuning_record(cfg) == record
    stored = json.loads(json_path.read_text(encoding="utf-8"))
    assert stored["schema"] == 1 and stored["attempts"][0]["deployment_status"] == "Active"

    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.isascii() and "\r" not in markdown and str(tmp_path) not in markdown
    row = "| nova-lite | 1 | B1-Lite | Completed | - | 2026-11-17T04:05:06Z | - | $0.15 | $0.14 |"
    assert row in markdown
    assert "| nova-lite | 1 | 0.6000 | 0.7000 | Active | - |" in markdown
    assert "- Training loss over 4 steps: first 2.5000, last 0.6000, lowest 0.4000 at step 2" in (
        markdown
    )
    assert "- Note: a note" in markdown and "`amazon.nova-lite-v1:0:300k`" in markdown


def test_failures_and_deletions_are_shown(cfg):
    record = TuningRecord(
        [
            attempt(
                job_status="Failed",
                failed_in="data processing",
                failure_message="a reserved keyword was found",
            ),
            attempt("nova-pro", deployment_deleted=True, model_deleted=True,
                    deployment_failure="quota"),
        ]
    )  # fmt: skip
    markdown = render_markdown(record)
    assert "| Failed | data processing |" in markdown
    assert "- Failure message from Bedrock: a reserved keyword was found" in markdown
    assert "| deployment, model |" in markdown
    assert "- Deployment failure from Bedrock: quota" in markdown


@pytest.mark.parametrize(
    "content",
    ["{not json", "[]", '{"schema": 1}', '{"schema": 1, "attempts": [{"model": "x"}]}',
     '{"schema": 1, "attempts": [{"surprise": 1}]}', '{"schema": 9, "attempts": []}'],
)  # fmt: skip
def test_a_damaged_record_is_a_config_error(cfg, content):
    path = report_paths(cfg.report_dir)[1]
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match="tuning record"):
        read_tuning_record(cfg)


def test_times_from_boto3_become_utc_text():
    assert as_text(datetime(2026, 11, 17, 4, 5, 6, tzinfo=UTC)) == "2026-11-17T04:05:06Z"
    eastern = timezone(timedelta(hours=-5))
    assert as_text(datetime(2026, 11, 16, 23, 5, 6, tzinfo=eastern)) == "2026-11-17T04:05:06Z"
    assert as_text(datetime(2026, 11, 17, 4, 5, 6)) == "2026-11-17T04:05:06Z"  # taken as UTC
    assert as_text(None) is None and as_text("2026-11-17T04:05:06Z") == "2026-11-17T04:05:06Z"
