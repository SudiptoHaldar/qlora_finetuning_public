"""Tests for launching a fine-tuning job: once, recorded, and only when every check passes.

Bedrock and S3 are in-memory stand-ins. No test creates anything in AWS.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal

import pytest

from qlora_finetuning.frontier.tuning import Refused, TuningError
from qlora_finetuning.frontier.tuning_record import read_tuning_record, report_paths
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking import GitInfo, load_manifest
from tests.evaluation.conftest import ledger_records, run_folders
from tests.frontier.export_helpers import FakeS3
from tests.frontier.tuning_helpers import FakeBedrock, client_error, make_tuning, open_gate


@pytest.fixture
def s3():
    return FakeS3()


@pytest.fixture
def bedrock():
    return FakeBedrock()


@pytest.fixture
def gate(cfg, context, s3):
    return open_gate(cfg, context, s3)


@pytest.fixture
def tuning(cfg, context, bedrock, s3, gate):
    return make_tuning(cfg, context, bedrock, s3)


def test_a_launch_creates_the_job_and_records_it(tuning, bedrock, cfg, context, gate):
    plan = tuning.plan("nova-lite")
    attempt = tuning.launch(plan)

    assert bedrock.count("create_model_customization_job") == 1
    job = bedrock.jobs[plan.job_name]
    assert job["request"] == plan.request()
    assert attempt.job_arn == job["jobArn"] and attempt.job_status == "InProgress"

    # The tuning record: the only memory of the job across sessions.
    stored = read_tuning_record(cfg).latest("nova-lite")
    assert stored == attempt
    assert (stored.model, stored.attempt, stored.arm) == ("nova-lite", 1, "B1-Lite")
    assert stored.base_model_id == "amazon.nova-lite-v1:0"
    assert stored.customization_model_id == "amazon.nova-lite-v1:0:300k"
    assert stored.client_request_token == plan.token
    assert stored.hyperparameters == {
        "epochCount": "2",
        "learningRate": "0.00001",
        "learningRateWarmupSteps": "10",
    }
    assert stored.export_id == gate["export_id"]
    assert (stored.estimated_training_tokens, stored.estimated_usd) == (75_000, "0.15")
    assert all(path.is_file() for path in report_paths(cfg.report_dir))

    # The ledger: the estimated tokens, marked as computed, tied to the job.
    (record,) = ledger_records(context)
    assert (record.phase, record.path, record.arm, record.where) == (
        "train",
        "B",
        "B1-Lite",
        "bedrock",
    )
    assert (record.training_tokens, record.source) == (75_000, "computed")
    assert record.model_id == "amazon.nova-lite-v1:0" and record.external_id == attempt.job_arn
    assert record.billed_amount_usd is None and record.input_tokens is None

    # The run: a measured training run that names the job.
    (run_id,) = run_folders(context)
    assert stored.run_id == run_id == record.run_id
    manifest = load_manifest(run_id, context.tracking_cfg)
    assert (manifest["kind"], manifest["scope"], manifest["status"]) == (
        "train",
        "measured",
        "completed",
    )
    assert manifest["outputs"] == {"job_arn": attempt.job_arn, "job_name": plan.job_name}
    assert {"frontier", "pricing", "aws"} <= set(manifest["configs"])


def test_a_launch_with_a_failing_check_creates_and_writes_nothing(
    cfg, context, bedrock, s3, gate
):
    bedrock.provisioned = [{"provisionedModelName": "left-over"}]
    tuning = make_tuning(cfg, context, bedrock, s3)
    with pytest.raises(Refused, match="a check failed, so nothing was launched") as caught:
        tuning.launch(tuning.plan("nova-lite"))
    assert [c["id"] for c in caught.value.checks if c["passed"] is False] == ["T5"]
    assert bedrock.changes() == []
    assert not any(path.exists() for path in report_paths(cfg.report_dir))
    assert run_folders(context) == [] and ledger_records(context) == []


def test_a_launch_from_uncommitted_changes_is_refused(cfg, context, bedrock, s3, gate):
    dirty = GitInfo(commit="a" * 40, branch="main", dirty=True, changed=("scripts/x.py",))
    tuning = make_tuning(cfg, dataclasses.replace(context, git=dirty), bedrock, s3)
    with pytest.raises(Refused) as caught:
        tuning.launch(tuning.plan("nova-lite"))
    assert [c["id"] for c in caught.value.checks if c["passed"] is False] == ["T8"]
    assert bedrock.changes() == []


def test_a_launch_without_clients_asks_for_the_sign_in(cfg, context, gate):
    tuning = make_tuning(cfg, context)
    with pytest.raises(ConfigError, match="aws sso login --profile qlora"):
        tuning.launch(tuning.plan("nova-lite"))
    assert run_folders(context) == []


def test_a_lost_reply_followed_by_a_repeat_ends_with_one_job(tuning, bedrock, cfg, context):
    plan = tuning.plan("nova-lite")
    bedrock.lose_reply = True  # Bedrock creates the job; the caller only sees an error
    with pytest.raises(TuningError, match="InternalServerException when asked to create"):
        tuning.launch(plan)
    assert len(bedrock.jobs) == 1
    assert read_tuning_record(cfg).attempts == []  # we never learned the ARN
    assert ledger_records(context) == []
    (failed_run,) = run_folders(context)
    assert load_manifest(failed_run, context.tracking_cfg)["status"] == "failed"

    attempt = tuning.launch(plan)  # the same command again
    assert len(bedrock.jobs) == 1  # adopted, not created a second time
    assert bedrock.count("create_model_customization_job") == 1
    assert attempt.job_arn == bedrock.jobs[plan.job_name]["jobArn"]
    assert attempt.created_at == "2026-11-17T04:05:06Z"
    assert read_tuning_record(cfg).latest("nova-lite") == attempt
    (record,) = ledger_records(context)
    assert record.external_id == attempt.job_arn and record.training_tokens == 75_000


@pytest.mark.parametrize(
    "code", ["ConflictException", "ServiceQuotaExceededException", "ValidationException",
             "ThrottlingException", "AccessDeniedException"]
)  # fmt: skip
def test_a_refusal_by_bedrock_is_reported_with_its_code_and_records_no_job(
    tuning, bedrock, cfg, context, code
):
    bedrock.fail_create = client_error(code, "CreateModelCustomizationJob")
    with pytest.raises(TuningError, match=f"AWS answered {code} when asked to create the job"):
        tuning.launch(tuning.plan("nova-lite"))
    assert bedrock.jobs == {} and bedrock.count("create_model_customization_job") == 1
    assert read_tuning_record(cfg).attempts == [] and ledger_records(context) == []


def test_a_second_attempt_is_another_job_and_must_be_asked_for(tuning, bedrock, cfg):
    first = tuning.launch(tuning.plan("nova-lite"))
    bedrock.finish(first.job_name, status="Failed", failed_in="validationDetails")
    tuning.refresh()

    with pytest.raises(Refused) as caught:  # the same attempt again
        tuning.launch(tuning.plan("nova-lite"))
    assert [c["id"] for c in caught.value.checks if c["passed"] is False] == ["T6"]

    second = tuning.launch(tuning.plan("nova-lite", attempt=2))
    assert second.job_name.endswith("-a2") and second.job_name != first.job_name
    assert second.client_request_token != first.client_request_token
    assert second.output_s3_uri.endswith("/nova-lite-a2/")
    assert len(bedrock.jobs) == 2
    record = read_tuning_record(cfg)
    assert [a.attempt for a in record.of("nova-lite")] == [1, 2]
    assert record.find("nova-lite", 1).job_status == "Failed"  # kept, with how it ended


def test_a_second_attempt_that_would_pass_the_cap_is_refused(cfg, context, bedrock, s3, gate):
    tuning = make_tuning(cfg, context, bedrock, s3)
    first = tuning.launch(tuning.plan("nova-lite"))
    bedrock.finish(first.job_name, status="Stopped")
    tuning.refresh()
    # 0.15 is committed; a second attempt needs 0.165 more. The gate saw one attempt only.
    tight = dataclasses.replace(cfg, path_b_cap_usd=Decimal("0.30"))
    tuning = make_tuning(tight, context, bedrock, s3)
    with pytest.raises(Refused) as caught:
        tuning.launch(tuning.plan("nova-lite", attempt=2))
    refused = {c["id"] for c in caught.value.checks if c["passed"] is False}
    assert "T7" in refused and len(bedrock.jobs) == 1


def test_nova_pro_is_launched_only_after_nova_lite_is_completed_and_billed(
    tuning, bedrock, cfg, context
):
    with pytest.raises(Refused) as caught:
        tuning.launch(tuning.plan("nova-pro"))
    assert [c["id"] for c in caught.value.checks if c["passed"] is False] == ["T9"]

    lite = tuning.launch(tuning.plan("nova-lite"))
    bedrock.finish(lite.job_name)
    tuning.refresh()
    with pytest.raises(Refused):  # completed, but the bill has not been read
        tuning.launch(tuning.plan("nova-pro"))
    assert len(bedrock.jobs) == 1

    tuning.bill("nova-lite", "0.16")
    pro = tuning.launch(tuning.plan("nova-pro"))
    assert pro.arm == "B1-Pro" and pro.customization_model_id == "amazon.nova-pro-v1:0:300k"
    assert bedrock.jobs[pro.job_name]["request"]["baseModelIdentifier"] == (
        "amazon.nova-pro-v1:0:300k"
    )
    assert len(bedrock.jobs) == 2
    train = [r for r in ledger_records(context) if r.source == "computed"]
    assert {r.arm for r in train} == {"B1-Lite", "B1-Pro"}


def test_no_launch_ever_starts_a_second_model(tuning, bedrock):
    tuning.launch(tuning.plan("nova-lite"))
    assert list(bedrock.jobs) == [tuning.plan("nova-lite").job_name]
    assert bedrock.count("create_custom_model_deployment") == 0  # nor deploys by itself
