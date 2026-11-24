"""Tests for the cost ledger. Everything is written under a temporary folder."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from qlora_finetuning.tracking import (
    Ledger,
    LedgerRecord,
    TrackingError,
    record_from_converse_usage,
    record_from_sagemaker_job,
    sum_converse_usage,
)

RUN = "20261115T061500Z-eval-b0-pro-dev-3f9a"


def tokens(**overrides) -> LedgerRecord:
    """A valid record of Bedrock inference tokens."""
    values = {
        "run_id": RUN,
        "phase": "eval",
        "path": "B",
        "arm": "B0-Pro",
        "where": "bedrock",
        "source": "aws_api",
        "source_detail": "bedrock-runtime:Converse usage, summed over the run",
        "model_id": "amazon.nova-pro-v1:0",
        "input_tokens": 812_000,
        "output_tokens": 41_000,
        "examples": 500,
        "wall_clock_seconds": 930.5,
    }
    values.update(overrides)
    return LedgerRecord(**values)


def compute(**overrides) -> LedgerRecord:
    """A valid record of SageMaker instance time."""
    values = {
        "run_id": "20261115T070000Z-train-a1-0c2d",
        "phase": "train",
        "path": "A",
        "arm": "A1",
        "where": "sagemaker",
        "source": "aws_api",
        "instance_type": "ml.g5.xlarge",
        "instance_count": 1,
        "billable_seconds": 5400,
        "wall_clock_seconds": 5390.0,
        "external_id": "qlora-a1-20261115",
    }
    values.update(overrides)
    return LedgerRecord(**values)


def ledger_text(cfg) -> str:
    return cfg.ledger_file.read_text(encoding="utf-8") if cfg.ledger_file.exists() else ""


# --------------------------------------------------------------------------- round trip


def test_append_and_read_round_trip(cfg):
    ledger = Ledger(cfg)
    written = ledger.append(tokens())
    assert ledger.read() == [written]
    assert written.schema == 1
    assert len(written.record_id) == 32
    assert written.recorded_at.endswith("Z")


def test_one_json_object_per_line_without_empty_fields(cfg):
    Ledger(cfg).append(tokens())
    Ledger(cfg).append(compute())
    lines = ledger_text(cfg).splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["input_tokens"] == 812_000
    assert "instance_type" not in first
    assert "note" not in first
    assert "\r" not in ledger_text(cfg)


def test_reading_a_missing_ledger_is_empty(cfg):
    assert Ledger(cfg).read() == []
    assert Ledger(cfg).summarize() == []


# --------------------------------------------------------------------------- rules


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (
            tokens(input_tokens=None, output_tokens=None, examples=None, wall_clock_seconds=None),
            "at least one quantity",
        ),
        (tokens(input_tokens=-1), "cannot be negative"),
        (tokens(wall_clock_seconds=-0.5), "cannot be negative"),
        (compute(instance_type=None), "needs instance_type and instance_count"),
        (compute(instance_count=None), "needs instance_type and instance_count"),
        (compute(instance_count=0), "at least 1"),
        (tokens(model_id=None), "needs model_id"),
        (tokens(billed_amount_usd="12.34"), "only allowed with source 'bill'"),
        (tokens(source="bill", billed_amount_usd=12.34), "not a float"),
        (tokens(source="bill", billed_amount_usd="twelve"), "not a number"),
        (tokens(source="bill", billed_amount_usd="-1.00"), "cannot be negative"),
        (tokens(source="guess"), "is not a source"),
        (tokens(phase="inference"), "allowed value for phases"),
        (tokens(arm="B2"), "allowed value for arms"),
        (tokens(where="azure"), "allowed value for places"),
        (tokens(path="C"), "allowed value for paths"),
        (tokens(examples=True), "whole number"),
        (tokens(input_tokens=1.5), "whole number"),
        (tokens(wall_clock_seconds="long"), "must be a number"),
        (tokens(run_id=""), "needs a run_id"),
        (tokens(replaces="feedfacefeedfacefeedfacefeedface"), "not in the ledger"),
        (tokens(note="ask owner@example.com"), "email address"),
        (tokens(source_detail="key AKIAIOSFODNN7EXAMPLE"), "access key"),
    ],
)  # fmt: skip
def test_invalid_record_is_rejected_and_nothing_is_written(cfg, record, message):
    ledger = Ledger(cfg)
    ledger.append(compute())
    before = ledger_text(cfg)
    with pytest.raises(TrackingError, match=message):
        ledger.append(record)
    assert ledger_text(cfg) == before


def test_duplicate_record_id_is_rejected(cfg):
    ledger = Ledger(cfg)
    first = ledger.append(tokens())
    with pytest.raises(TrackingError, match="already in the ledger"):
        ledger.append(tokens(record_id=first.record_id))


def test_billed_amount_is_stored_as_text(cfg):
    ledger = Ledger(cfg)
    ledger.append(tokens(source="bill", billed_amount_usd=Decimal("12.30")))
    ledger.append(tokens(source="bill", billed_amount_usd=7))
    assert [record.billed_amount_usd for record in ledger.read()] == ["12.30", "7"]


def test_zero_is_a_valid_quantity(cfg):
    Ledger(cfg).append(tokens(output_tokens=0, examples=0))


# --------------------------------------------------------------------------- corrections


def test_replacing_record_hides_the_old_one(cfg):
    ledger = Ledger(cfg)
    estimate = ledger.append(
        tokens(phase="train", arm="B1-Pro", source="computed", input_tokens=None,
               output_tokens=None, examples=None, wall_clock_seconds=None,
               training_tokens=9_000_000, source_detail="tokens in the training file x 2 epochs")
    )  # fmt: skip
    billed = ledger.append(
        tokens(phase="train", arm="B1-Pro", source="bill", input_tokens=None, output_tokens=None,
               examples=None, wall_clock_seconds=None, training_tokens=9_120_000,
               billed_amount_usd="72.96", replaces=estimate.record_id)
    )  # fmt: skip
    assert len(ledger.read()) == 2
    assert ledger.effective() == [billed]
    (row,) = ledger.summarize()
    assert row["training_tokens"] == 9_120_000
    assert row["billed_amount_usd"] == "72.96"
    assert row["sources"] == {"bill": 1}


# --------------------------------------------------------------------------- summary


def test_summarize_totals_per_arm_and_phase(cfg):
    ledger = Ledger(cfg)
    ledger.append(tokens())
    ledger.append(
        tokens(input_tokens=188_000, output_tokens=9_000, examples=100, wall_clock_seconds=69.5)
    )
    ledger.append(compute(instance_count=2))
    rows = ledger.summarize()
    assert [(row["arm"], row["phase"]) for row in rows] == [("A1", "train"), ("B0-Pro", "eval")]
    a1, b0 = rows
    assert a1["billable_instance_seconds"] == 10_800  # 5400 seconds x 2 instances
    assert a1["records"] == 1
    assert b0["records"] == 2
    assert b0["input_tokens"] == 1_000_000
    assert b0["output_tokens"] == 50_000
    assert b0["examples"] == 600
    assert b0["wall_clock_seconds"] == pytest.approx(1000.0)
    assert b0["billed_amount_usd"] == "0"
    assert b0["sources"] == {"aws_api": 2}


def test_summarize_adds_dollar_amounts_exactly(cfg):
    ledger = Ledger(cfg)
    for amount in ("0.10", "0.20"):
        ledger.append(tokens(source="bill", billed_amount_usd=amount))
    assert ledger.summarize()[0]["billed_amount_usd"] == "0.30"


def test_summarize_by_other_fields(cfg):
    ledger = Ledger(cfg)
    ledger.append(tokens())
    ledger.append(compute())
    rows = ledger.summarize(by=("where",))
    assert [row["where"] for row in rows] == ["bedrock", "sagemaker"]
    assert ledger.summarize(by=("instance_type",))[0]["instance_type"] == "-"


@pytest.mark.parametrize("by", [("colour",), ()])
def test_summarize_rejects_unknown_grouping(cfg, by):
    with pytest.raises(TrackingError, match="cannot group by"):
        Ledger(cfg).summarize(by=by)


# --------------------------------------------------------------------------- damaged files


def test_corrupt_line_names_the_file_and_line(cfg):
    Ledger(cfg).append(tokens())
    with cfg.ledger_file.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    with pytest.raises(TrackingError, match="ledger.jsonl line 2"):
        Ledger(cfg).read()


def test_unknown_schema_and_unknown_field(cfg):
    cfg.ledger_file.parent.mkdir(parents=True)
    cfg.ledger_file.write_text('{"schema": 2, "run_id": "x"}\n', encoding="utf-8")
    with pytest.raises(TrackingError, match="record 1: unknown ledger schema"):
        Ledger(cfg).read()
    line = tokens().to_dict() | {"colour": "red"}
    cfg.ledger_file.write_text(json.dumps(line) + "\n", encoding="utf-8")
    with pytest.raises(TrackingError, match="unknown ledger field"):
        Ledger(cfg).read()


def test_line_that_is_not_an_object(cfg):
    cfg.ledger_file.parent.mkdir(parents=True)
    cfg.ledger_file.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(TrackingError, match="not a JSON object"):
        Ledger(cfg).read()


# --------------------------------------------------------------------------- AWS responses


def test_sum_converse_usage():
    usages = [
        {"inputTokens": 1500, "outputTokens": 80, "totalTokens": 1580},
        {"inputTokens": 1700, "outputTokens": 95, "totalTokens": 1795},
    ]
    assert sum_converse_usage(usages) == {"inputTokens": 3200, "outputTokens": 175}
    assert sum_converse_usage([]) == {"inputTokens": 0, "outputTokens": 0}


@pytest.mark.parametrize(
    "usage",
    [
        {"inputTokens": 5},
        {"inputTokens": "5", "outputTokens": 1},
        {"inputTokens": -1, "outputTokens": 1},
    ],
)
def test_sum_converse_usage_rejects_bad_blocks(usage):
    with pytest.raises(TrackingError, match="whole number"):
        sum_converse_usage([usage])


def test_record_from_converse_usage(make_run, cfg):
    run = make_run()
    record = record_from_converse_usage(
        {"inputTokens": 812_000, "outputTokens": 41_000, "totalTokens": 853_000},
        run,
        model_id="amazon.nova-pro-v1:0",
        examples=500,
        wall_clock_seconds=930.5,
    )
    assert (record.run_id, record.arm, record.where, record.phase) == (
        run.run_id,
        "B0-Pro",
        "bedrock",
        "eval",
    )
    assert (record.input_tokens, record.output_tokens, record.examples) == (812_000, 41_000, 500)
    assert record.source == "aws_api"
    assert Ledger(cfg).read() == []  # built, not appended
    run.append(record)
    assert Ledger(cfg).read() == [record]


JOB = {
    "TrainingJobName": "qlora-a1-20261115",
    "TrainingJobStatus": "Completed",
    "ResourceConfig": {"InstanceType": "ml.g5.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 50},
    "TrainingTimeInSeconds": 5390,
    "BillableTimeInSeconds": 5400,
}


def test_record_from_sagemaker_job(make_run):
    run = make_run("train", "a1", path="A", arm="A1", where="sagemaker")
    record = record_from_sagemaker_job(JOB, run, phase="train")
    assert record.billable_seconds == 5400
    assert record.wall_clock_seconds == 5390.0
    assert (record.instance_type, record.instance_count) == ("ml.g5.xlarge", 1)
    assert record.external_id == "qlora-a1-20261115"
    assert record.source == "aws_api"
    assert "per instance" in record.source_detail
    run.append(record)


@pytest.mark.parametrize("status", ["Stopped", "Failed"])
def test_stopped_and_failed_jobs_are_still_billed(make_run, status):
    run = make_run("train", "a1", path="A", arm="A1", where="sagemaker")
    record = record_from_sagemaker_job(JOB | {"TrainingJobStatus": status}, run, phase="train")
    assert record.billable_seconds == 5400
    assert status in record.note


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"TrainingJobStatus": "InProgress"}, "has not finished"),
        ({"TrainingJobStatus": "Stopping"}, "has not finished"),
        ({"BillableTimeInSeconds": None}, "no BillableTimeInSeconds"),
        ({"ResourceConfig": {}}, "no instance type or count"),
    ],
)
def test_record_from_sagemaker_job_rejects_incomplete_jobs(make_run, change, message):
    run = make_run("train", "a1", path="A", arm="A1", where="sagemaker")
    with pytest.raises(TrackingError, match=message):
        record_from_sagemaker_job(JOB | change, run, phase="train")


def test_job_without_training_time_still_records_billable_time(make_run):
    run = make_run("train", "a1", path="A", arm="A1", where="sagemaker")
    job = {key: value for key, value in JOB.items() if key != "TrainingTimeInSeconds"}
    assert record_from_sagemaker_job(job, run, phase="train").wall_clock_seconds is None
