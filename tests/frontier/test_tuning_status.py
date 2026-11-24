"""Tests for following a job, deploying, stopping, billing, and tearing down.

Bedrock and S3 are in-memory stand-ins. No test changes anything in AWS.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from qlora_finetuning.frontier.pricing import load_pricing
from qlora_finetuning.frontier.tuning import (
    BedrockTuning,
    Refused,
    TuningError,
    committed_path_b,
    is_active,
    summarize_metrics,
)
from qlora_finetuning.frontier.tuning_record import read_tuning_record, report_paths
from qlora_finetuning.settings import ConfigError
from tests.evaluation.conftest import ledger_records
from tests.frontier.export_helpers import FakeS3
from tests.frontier.tuning_helpers import (
    TRAINING_CSV,
    FakeBedrock,
    client_error,
    make_tuning,
    open_gate,
    put_metrics,
)


@pytest.fixture
def s3():
    return FakeS3()


@pytest.fixture
def bedrock():
    return FakeBedrock()


@pytest.fixture
def tuning(cfg, context, bedrock, s3):
    open_gate(cfg, context, s3)
    return make_tuning(cfg, context, bedrock, s3)


@pytest.fixture
def lite(tuning):
    """Nova Lite, launched and still running."""
    return tuning.launch(tuning.plan("nova-lite"))


def completed(tuning, bedrock, s3, lite):
    bedrock.finish(lite.job_name)
    put_metrics(s3, lite)
    return tuning.refresh().latest("nova-lite")


# --------------------------------------------------------------------------- status


def test_refresh_only_reads_and_keeps_a_running_job_running(tuning, bedrock, lite, cfg):
    before = bedrock.changes()
    attempt = tuning.refresh().latest("nova-lite")
    assert attempt.job_status == "InProgress" and not attempt.ended
    assert attempt.created_at == "2026-11-17T04:05:06Z" and attempt.ended_at is None
    assert attempt.model_arn is None and attempt.metrics is None
    assert bedrock.changes() == before  # nothing created, stopped, or deleted
    assert BedrockTuning.pending(read_tuning_record(cfg)) == [attempt]


def test_a_completed_job_gives_the_model_its_losses_and_the_loss_curves(
    tuning, bedrock, s3, lite, cfg
):
    attempt = completed(tuning, bedrock, s3, lite)
    assert attempt.job_status == "Completed" and attempt.ended
    assert attempt.ended_at == "2026-11-17T07:30:00Z"
    assert attempt.model_arn.endswith(f":custom-model/{lite.job_name}")
    assert (attempt.training_loss, attempt.validation_loss) == (0.6, 0.7)
    assert attempt.metrics == {
        "training": {"steps": 4, "first": 2.5, "last": 0.6, "lowest": 0.4, "lowest_step": 2},
        "validation": {"steps": 2, "first": 1.1, "last": 0.7, "lowest": 0.7, "lowest_step": 3},
    }
    folder = cfg.export_dir / attempt.export_id / "nova-lite-a1"
    assert (folder / "step_wise_training_metrics.csv").read_bytes() == TRAINING_CSV
    assert (folder / "validation_metrics.csv").is_file()
    assert attempt.notes == [] and BedrockTuning.pending(read_tuning_record(cfg)) == []


def test_losses_bedrock_reports_as_minus_one_come_from_the_loss_curves(
    tuning, bedrock, s3, lite
):
    """Seen on the first Nova Lite job (2026-11-18): -1 for both losses."""
    bedrock.finish(
        lite.job_name,
        trainingMetrics={"trainingLoss": -1.0},
        validationMetrics=[{"validationLoss": -1.0}],
    )
    attempt = tuning.refresh().latest("nova-lite")  # the curves are not there yet
    assert (attempt.training_loss, attempt.validation_loss) == (None, None)
    put_metrics(s3, lite)
    attempt = tuning.refresh().latest("nova-lite")
    assert (attempt.training_loss, attempt.validation_loss) == (0.6, 0.7)


def test_a_record_that_already_holds_minus_one_is_repaired(tuning, bedrock, s3, lite, cfg):
    completed(tuning, bedrock, s3, lite)
    _, json_path = report_paths(cfg.report_dir)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    data["attempts"][0].update(training_loss=-1.0, validation_loss=-1.0)
    json_path.write_text(json.dumps(data), encoding="utf-8")
    reads = bedrock.count("get_model_customization_job")
    attempt = tuning.refresh().latest("nova-lite")
    assert (attempt.training_loss, attempt.validation_loss) == (0.6, 0.7)
    assert bedrock.count("get_model_customization_job") == reads


def test_the_loss_curves_are_fetched_once(tuning, bedrock, s3, lite):
    completed(tuning, bedrock, s3, lite)
    downloads, reads = s3.count("get_object"), bedrock.count("get_model_customization_job")
    tuning.refresh()
    tuning.refresh()
    assert s3.count("get_object") == downloads == 2
    assert bedrock.count("get_model_customization_job") == reads  # an ended job is not re-read


@pytest.mark.parametrize(
    ("phase", "name"),
    [
        ("validationDetails", "validation"),
        ("dataProcessingDetails", "data processing"),
        ("trainingDetails", "training"),
    ],
)
def test_a_failed_job_says_where_it_failed(tuning, bedrock, lite, phase, name):
    bedrock.finish(lite.job_name, status="Failed", failed_in=phase)
    attempt = tuning.refresh().latest("nova-lite")
    assert (attempt.job_status, attempt.failed_in) == ("Failed", name)
    assert attempt.failure_message == "stubbed failure" and attempt.model_arn is None
    assert attempt.ended and attempt.metrics is None


def test_missing_or_damaged_metrics_are_noted_and_the_status_is_kept(
    tuning, bedrock, s3, lite
):
    bedrock.finish(lite.job_name)
    attempt = tuning.refresh().latest("nova-lite")  # no metrics files were written
    assert attempt.job_status == "Completed" and attempt.model_arn and attempt.metrics is None
    assert attempt.notes == [
        "the loss curves could not be read: step_wise_training_metrics.csv is not under "
        "the job's output location"
    ]
    assert tuning.refresh().latest("nova-lite").notes == attempt.notes  # not noted twice

    put_metrics(s3, lite, training=b"step,loss\n0,1\n")
    attempt = tuning.refresh().latest("nova-lite")
    assert attempt.metrics is None
    assert "the loss curves could not be read: no usable training_loss column" in attempt.notes

    put_metrics(s3, lite)  # the real files appear
    assert tuning.refresh().latest("nova-lite").metrics["training"]["steps"] == 4


def test_summarize_metrics_reads_first_last_and_lowest():
    assert summarize_metrics(TRAINING_CSV, "training_loss") == {
        "steps": 4, "first": 2.5, "last": 0.6, "lowest": 0.4, "lowest_step": 2,
    }  # fmt: skip
    for content in (b"", b"step_number,training_loss\n", b"step_number,training_loss\n0,high\n"):
        with pytest.raises(ValueError, match="training_loss column|no rows"):
            summarize_metrics(content, "training_loss")


def test_wait_looks_again_until_every_job_has_ended(cfg, context, bedrock, s3):
    open_gate(cfg, context, s3)
    naps: list[float] = []

    def sleep(seconds):
        naps.append(seconds)
        if len(naps) == 2:
            bedrock.finish(job_name)

    tuning = make_tuning(cfg, context, bedrock, s3, sleep=sleep)
    job_name = tuning.launch(tuning.plan("nova-lite")).job_name
    looks: list[str] = []
    record = tuning.wait(on_look=lambda r: looks.append(r.latest("nova-lite").job_status))
    assert naps == [60, 60]  # the configured interval
    assert looks == ["InProgress", "InProgress", "Completed"]
    assert record.latest("nova-lite").job_status == "Completed"


def test_refresh_without_a_sign_in_is_a_config_error(cfg, context, s3):
    open_gate(cfg, context, s3)
    with pytest.raises(ConfigError, match="aws sso login"):
        make_tuning(cfg, context).refresh()


def test_refresh_with_nothing_launched_writes_no_record(tuning, cfg):
    assert tuning.refresh().attempts == []
    assert not any(path.exists() for path in report_paths(cfg.report_dir))


# --------------------------------------------------------------------------- deploy


def test_deploy_creates_one_on_demand_deployment_for_a_completed_model(
    tuning, bedrock, s3, lite, cfg
):
    done = completed(tuning, bedrock, s3, lite)
    attempt = tuning.deploy("nova-lite")
    deployment = bedrock.deployments[lite.deployment_name]
    assert deployment["request"] == {
        "modelDeploymentName": lite.deployment_name,
        "modelArn": done.model_arn,
        "description": f"On-demand deployment of {lite.custom_model_name}",
        "clientRequestToken": f"{lite.client_request_token}-deploy",
        "tags": [{"key": "project", "value": "qlora-finetuning"}],
    }
    assert attempt.deployment_arn == deployment["customModelDeploymentArn"]
    assert attempt.deployment_status == "Creating" and not is_active(attempt)
    assert BedrockTuning.pending(read_tuning_record(cfg)) == [attempt]

    deployment["status"] = "Active"
    attempt = tuning.refresh().latest("nova-lite")
    assert attempt.deployment_status == "Active" and is_active(attempt)
    assert BedrockTuning.pending(read_tuning_record(cfg)) == []


def test_deploy_is_refused_before_the_job_has_completed_and_when_already_deployed(
    tuning, bedrock, s3, lite
):
    with pytest.raises(Refused, match="no completed, existing tuned model"):
        tuning.deploy("nova-lite")
    with pytest.raises(Refused, match="nova-pro has no launched job"):
        tuning.deploy("nova-pro")
    assert bedrock.count("create_custom_model_deployment") == 0

    completed(tuning, bedrock, s3, lite)
    tuning.deploy("nova-lite")
    with pytest.raises(Refused, match="already deployed"):
        tuning.deploy("nova-lite")
    assert bedrock.count("create_custom_model_deployment") == 1


def test_deploy_adopts_a_deployment_that_already_exists_under_the_name(
    tuning, bedrock, s3, lite
):
    done = completed(tuning, bedrock, s3, lite)
    bedrock.create_custom_model_deployment(
        modelDeploymentName=lite.deployment_name, modelArn=done.model_arn
    )
    attempt = tuning.deploy("nova-lite")
    assert bedrock.count("create_custom_model_deployment") == 1  # only the earlier one
    assert attempt.deployment_arn.endswith(f"/{lite.deployment_name}")


def test_deploy_is_refused_while_provisioned_throughput_exists(tuning, bedrock, s3, lite):
    completed(tuning, bedrock, s3, lite)
    bedrock.provisioned = [{"provisionedModelName": "left-over"}]
    with pytest.raises(Refused, match="Provisioned Throughput \\(left-over\\)"):
        tuning.deploy("nova-lite")
    assert bedrock.deployments == {}


def test_a_failed_deployment_is_shown_with_its_message(tuning, bedrock, s3, lite):
    completed(tuning, bedrock, s3, lite)
    tuning.deploy("nova-lite")
    bedrock.deployments[lite.deployment_name].update(status="Failed", failureMessage="no capacity")
    attempt = tuning.refresh().latest("nova-lite")
    assert (attempt.deployment_status, attempt.deployment_failure) == ("Failed", "no capacity")
    assert not is_active(attempt)


# --------------------------------------------------------------------------- stop


def test_stop_stops_a_running_job_and_only_a_running_one(tuning, bedrock, s3, lite):
    attempt = tuning.stop("nova-lite")
    assert attempt.job_status == "Stopping" and bedrock.jobs[lite.job_name]["status"] == "Stopping"
    with pytest.raises(Refused, match="is Stopping, not running"):
        tuning.stop("nova-lite")
    bedrock.finish(lite.job_name, status="Stopped")
    stopped = tuning.refresh().latest("nova-lite")
    assert stopped.job_status == "Stopped" and stopped.model_arn is None  # no model is made
    assert bedrock.count("stop_model_customization_job") == 1


def test_stop_sees_that_the_job_finished_meanwhile(tuning, bedrock, lite, cfg):
    bedrock.finish(lite.job_name)
    with pytest.raises(Refused, match="is Completed, not running"):
        tuning.stop("nova-lite")
    assert bedrock.count("stop_model_customization_job") == 0
    assert read_tuning_record(cfg).latest("nova-lite").job_status == "Completed"


# --------------------------------------------------------------------------- bill


def test_bill_replaces_the_estimate_in_the_ledger(tuning, bedrock, s3, lite, context, cfg):
    completed(tuning, bedrock, s3, lite)
    pricing = load_pricing()
    assert committed_path_b(ledger_records(context), pricing) == Decimal("0.15")

    attempt = tuning.bill("nova-lite", "0.1740")
    assert attempt.billed_usd == "0.174"
    records = ledger_records(context)
    estimate, bill = records[0], records[-1]
    assert (bill.source, bill.billed_amount_usd, bill.replaces) == (
        "bill",
        "0.174",
        estimate.record_id,
    )
    assert bill.external_id == lite.job_arn and bill.run_id == estimate.run_id
    assert (bill.phase, bill.arm, bill.model_id) == ("train", "B1-Lite", "amazon.nova-lite-v1:0")
    assert bill.training_tokens is None  # the bill gives dollars; the token count stays unknown
    # The estimate no longer counts: Path B has cost what the bill says.
    from qlora_finetuning.tracking import Ledger

    effective = Ledger(context.tracking_cfg).effective()
    assert committed_path_b(effective, pricing) == Decimal("0.174")
    assert read_tuning_record(cfg).latest("nova-lite").billed_usd == "0.174"

    tuning.bill("nova-lite", "0.18")  # a corrected reading replaces the first
    effective = Ledger(context.tracking_cfg).effective()
    assert committed_path_b(effective, pricing) == Decimal("0.18")
    assert [r.source for r in effective if r.external_id == lite.job_arn] == ["bill"]


@pytest.mark.parametrize("amount", ["a lot", "-1", "NaN", "Infinity", ""])
def test_a_billed_amount_must_be_a_number_of_zero_or_more(tuning, lite, context, amount):
    with pytest.raises(ConfigError, match="the billed amount"):
        tuning.bill("nova-lite", amount)
    assert len(ledger_records(context)) == 1


def test_bill_needs_a_launched_job_and_works_without_a_sign_in(cfg, context, bedrock, s3, lite):
    offline = make_tuning(cfg, context)  # no clients: entering a number asks AWS nothing
    assert offline.bill("nova-lite", "0").billed_usd == "0"
    with pytest.raises(Refused, match="nova-pro has no launched job"):
        offline.bill("nova-pro", "1")
    with pytest.raises(Refused, match="no launched job"):
        offline.bill("nova-lite", "1", attempt=2)


# --------------------------------------------------------------------------- teardown


def test_teardown_of_the_deployment_keeps_the_tuned_model(tuning, bedrock, s3, lite, cfg):
    completed(tuning, bedrock, s3, lite)
    tuning.deploy("nova-lite")
    attempt = tuning.teardown("nova-lite", everything=False)
    assert bedrock.deployments == {} and bedrock.deleted_models == []
    assert attempt.deployment_deleted and not attempt.model_deleted
    assert attempt.deployment_status == "Deleted" and not is_active(attempt)
    assert tuning.refresh().latest("nova-lite").deployment_status == "Deleted"  # not asked again

    redeployed = tuning.deploy("nova-lite")  # the model is still there
    assert redeployed.deployment_status == "Creating" and not redeployed.deployment_deleted


def test_teardown_of_everything_deletes_the_deployment_first_then_the_model(
    tuning, bedrock, s3, lite
):
    done = completed(tuning, bedrock, s3, lite)
    tuning.deploy("nova-lite")
    attempt = tuning.teardown("nova-lite", everything=True)
    assert bedrock.changes()[-2:] == ["delete_custom_model_deployment", "delete_custom_model"]
    assert bedrock.deleted_models == [done.model_arn]
    assert attempt.deployment_deleted and attempt.model_deleted
    assert s3.count("delete_object") == 0 and len(s3.objects) == 4  # S3 is never touched
    with pytest.raises(Refused, match="no completed, existing tuned model"):
        tuning.deploy("nova-lite")


def test_teardown_touches_only_what_the_record_names(tuning, bedrock, s3, lite):
    with pytest.raises(Refused, match="nothing left to delete"):
        tuning.teardown("nova-lite", everything=True)  # running: no model, no deployment
    completed(tuning, bedrock, s3, lite)
    with pytest.raises(Refused, match="nothing left to delete"):
        tuning.teardown("nova-lite", everything=False)  # not deployed
    with pytest.raises(Refused, match="nova-pro has no launched job"):
        tuning.teardown("nova-pro", everything=True)
    bedrock.deployments["a-strangers-deployment"] = {
        "customModelDeploymentArn": "arn:x", "modelDeploymentName": "a-strangers-deployment",
    }  # fmt: skip
    tuning.teardown("nova-lite", everything=True)
    assert "a-strangers-deployment" in bedrock.deployments


def test_deleting_something_already_gone_counts_as_deleted(tuning, bedrock, s3, lite):
    done = completed(tuning, bedrock, s3, lite)
    tuning.deploy("nova-lite")
    del bedrock.deployments[lite.deployment_name]  # deleted by hand in the console
    bedrock.deleted_models.append(done.model_arn)
    attempt = tuning.teardown("nova-lite", everything=True)
    assert attempt.deployment_deleted and attempt.model_deleted


def test_another_aws_error_during_teardown_is_reported_and_half_done_work_is_kept(
    tuning, bedrock, s3, lite, cfg
):
    completed(tuning, bedrock, s3, lite)
    tuning.deploy("nova-lite")

    def refuse(modelIdentifier):  # noqa: N803
        raise client_error("ConflictException", "DeleteCustomModel")

    bedrock.delete_custom_model = refuse
    with pytest.raises(TuningError, match="ConflictException when asked to delete the tuned"):
        tuning.teardown("nova-lite", everything=True)
    stored = read_tuning_record(cfg).latest("nova-lite")
    assert stored.deployment_deleted and not stored.model_deleted  # what happened, no more


# --------------------------------------------------------------------------- the record


def test_the_record_holds_names_numbers_and_arns_and_no_example_text(
    tuning, bedrock, s3, lite, cfg
):
    completed(tuning, bedrock, s3, lite)
    tuning.deploy("nova-lite")
    tuning.bill("nova-lite", "0.16")
    for path in report_paths(cfg.report_dir):
        text = path.read_text(encoding="utf-8")
        assert text.isascii() and "\r" not in text
        for word in ("zebrafish", "xylophone", "subtract", "PROGRAM"):
            assert word not in text, word
    stored = json.loads(report_paths(cfg.report_dir)[1].read_text(encoding="utf-8"))
    assert stored["attempts"][0]["billed_usd"] == "0.16"
