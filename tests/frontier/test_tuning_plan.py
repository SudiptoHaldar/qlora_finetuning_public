"""Tests for the plan and checks T1 to T9 of qlora_finetuning.frontier.tuning.

Nothing here reaches AWS: Bedrock and S3 are the in-memory stand-ins.
"""

from __future__ import annotations

import dataclasses
import json
import re
from decimal import Decimal

import pytest

from qlora_finetuning.frontier.pricing import load_pricing
from qlora_finetuning.frontier.training_estimate import report_paths as gate_paths
from qlora_finetuning.frontier.tuning import (
    TuningError,
    build_plan,
    committed_path_b,
    request_token,
)
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking import GitInfo
from tests.evaluation.conftest import ledger_records
from tests.frontier.conftest import baseline_run
from tests.frontier.export_helpers import FakeS3
from tests.frontier.tuning_helpers import BUCKET, ROLE, FakeBedrock, make_tuning, open_gate

# The name patterns of the installed boto3's CreateModelCustomizationJob.
JOB_NAME = re.compile(r"[a-zA-Z0-9](-*[a-zA-Z0-9\+\-\.])*")
MODEL_NAME = re.compile(r"([0-9a-zA-Z][_-]?){1,63}")
TOKEN = re.compile(r"[a-zA-Z0-9]([-a-zA-Z0-9]{0,254}[a-zA-Z0-9])?")


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


def result(checks, check_id):
    return next(item for item in checks if item["id"] == check_id)


def failed(checks):
    return [item["id"] for item in checks if item["passed"] is False]


def skipped(checks):
    return [item["id"] for item in checks if item["passed"] is None]


def rewrite_gate(cfg, change) -> None:
    path = gate_paths(cfg.report_dir)[1]
    record = json.loads(path.read_text(encoding="utf-8"))
    change(record)
    path.write_text(json.dumps(record), encoding="utf-8")


# --------------------------------------------------------------------------- the plan


def test_the_plan_names_the_files_the_model_and_the_cost(tuning, gate):
    plan = tuning.plan("nova-lite")
    name = f"qlora-finqa-nova-lite-{gate['export_id']}-a1"
    assert plan.job_name == plan.custom_model_name == plan.deployment_name == name
    assert plan.model.customization_model_id == "amazon.nova-lite-v1:0:300k"
    assert plan.training_s3_uri == gate["staged"]["training"]["s3_uri"]
    assert plan.validation_s3_uri == gate["staged"]["validation"]["s3_uri"]
    assert plan.output_s3_uri == f"s3://{BUCKET}/bedrock/{gate['export_id']}/nova-lite-a1/"
    # 37,500 Nova tokens an epoch (30,000 Qwen tokens * 1.25), 2 epochs, $2 per 1M.
    assert plan.estimated_training_tokens == 75_000
    assert plan.estimated_usd == Decimal("0.15")
    assert plan.estimated_with_margin_usd == Decimal("0.165")
    pro = tuning.plan("nova-pro", attempt=2)
    assert pro.job_name.endswith("-a2") and pro.estimated_usd == Decimal("0.6")


def test_the_request_is_exactly_what_bedrock_is_sent(tuning, gate):
    plan = tuning.plan("nova-lite")
    tags = [{"key": "project", "value": "qlora-finetuning"}]
    assert plan.request() == {
        "jobName": plan.job_name,
        "customModelName": plan.job_name,
        "roleArn": ROLE,
        "clientRequestToken": plan.token,
        "baseModelIdentifier": "amazon.nova-lite-v1:0:300k",
        "customizationType": "FINE_TUNING",
        "trainingDataConfig": {"s3Uri": gate["staged"]["training"]["s3_uri"]},
        "validationDataConfig": {
            "validators": [{"s3Uri": gate["staged"]["validation"]["s3_uri"]}]
        },
        "outputDataConfig": {"s3Uri": plan.output_s3_uri},
        # Bedrock's defaults, sent explicitly and as text (the owner's decision).
        "hyperParameters": {
            "epochCount": "2",
            "learningRate": "0.00001",
            "learningRateWarmupSteps": "10",
        },
        "jobTags": tags,
        "customModelTags": tags,
    }
    assert "provisioned" not in json.dumps(plan.request()).lower()


def test_names_and_the_token_fit_awss_patterns(tuning):
    for model in ("nova-lite", "nova-pro"):
        plan = tuning.plan(model, attempt=12)
        assert JOB_NAME.fullmatch(plan.job_name) and len(plan.job_name) <= 63
        assert MODEL_NAME.fullmatch(plan.custom_model_name)
        assert TOKEN.fullmatch(plan.token) and len(plan.token) <= 256
        assert TOKEN.fullmatch(plan.token + "-deploy")


def test_the_same_launch_gives_the_same_token_and_any_difference_another():
    settings = {"epochCount": "2", "learningRate": "0.00001"}
    token = request_token("nova-lite", "abc", settings, 1)
    assert token == request_token("nova-lite", "abc", dict(reversed(settings.items())), 1)
    others = {
        request_token("nova-pro", "abc", settings, 1),
        request_token("nova-lite", "abd", settings, 1),
        request_token("nova-lite", "abc", settings, 2),
        request_token("nova-lite", "abc", dict(settings, epochCount="3"), 1),
    }
    assert token not in others and len(others) == 4


def test_a_plan_for_files_that_are_not_staged_has_no_request(cfg, context):
    open_gate(cfg, context, staged=False)
    plan = make_tuning(cfg, context).plan("nova-lite")
    assert plan.training_s3_uri is None
    with pytest.raises(TuningError, match="not staged"):
        plan.request()


def test_planning_needs_a_gate_record_a_known_model_and_a_real_attempt(cfg, context, tuning):
    with pytest.raises(ConfigError, match="no model named 'nova-max'"):
        tuning.plan("nova-max")
    for attempt in (0, -1, True, 1.0):
        with pytest.raises(ConfigError, match="an attempt is a whole number of 1 or more"):
            tuning.plan("nova-lite", attempt)
    with pytest.raises(ConfigError, match="the gate record is damaged"):
        build_plan("nova-lite", 1, cfg, {"export_id": "x"}, load_pricing(), BUCKET, ROLE)
    gate_paths(cfg.report_dir)[1].unlink()
    with pytest.raises(ConfigError, match="no gate record"):
        tuning.plan("nova-lite")


# --------------------------------------------------------------------------- all pass


def test_every_check_passes_for_the_first_model_and_nothing_is_changed(tuning, bedrock, s3):
    checks = tuning.preflight(tuning.plan("nova-lite"))
    assert [item["id"] for item in checks] == [f"T{n}" for n in range(1, 10)]
    assert all(item["passed"] is True for item in checks), failed(checks)
    assert result(checks, "T9")["detail"] == "nova-lite is first in the launch order"
    assert "is $0.17 of $200.00" in result(checks, "T7")["detail"]  # $0.165, half up
    assert bedrock.changes() == [] and s3.count("put_object") == 2  # only the staging


def test_without_clients_the_aws_checks_are_skipped_not_passed(cfg, context, gate):
    checks = make_tuning(cfg, context).preflight(make_tuning(cfg, context).plan("nova-lite"))
    assert skipped(checks) == ["T3", "T4", "T5", "T6"] and failed(checks) == []
    assert "needs a sign-in" in result(checks, "T4")["detail"]
    assert not all(item["passed"] is True for item in checks)  # a skip never counts as a pass


# --------------------------------------------------------------------------- T1, T2, T3


def test_t1_fails_for_a_closed_gate_unstaged_files_and_no_record(cfg, context, bedrock, s3):
    open_gate(cfg, context, s3)

    def close(record):
        record["passed"] = False
        record["checks"][-2]["passed"] = False  # G1

    rewrite_gate(cfg, close)
    tuning = make_tuning(cfg, context, bedrock, s3)
    checks = tuning.preflight(tuning.plan("nova-lite"))
    assert "T1" in failed(checks)
    assert "the gate is closed (G1 failed)" in result(checks, "T1")["detail"]

    open_gate(cfg, context, s3, staged=False)
    checks = tuning.preflight(tuning.plan("nova-lite"))
    assert failed(checks) == ["T1", "T3"]
    assert "run python -m scripts.nova_export --upload" in result(checks, "T1")["detail"]


def test_t2_fails_when_what_the_gate_priced_has_changed(cfg, context, bedrock, s3, gate):
    more_epochs = make_tuning(dataclasses.replace(cfg, epoch_count=3), context, bedrock, s3)
    checks = more_epochs.preflight(more_epochs.plan("nova-lite"))
    assert failed(checks) == ["T2"]
    assert "the epoch count changed since the gate record was written" in (
        result(checks, "T2")["detail"]
    )

    higher_cap = make_tuning(
        dataclasses.replace(cfg, path_b_cap_usd=Decimal("250")), context, bedrock, s3
    )
    assert "the cap changed" in result(
        higher_cap.preflight(higher_cap.plan("nova-lite")), "T2"
    )["detail"]

    rewrite_gate(cfg, lambda record: record["inputs"].update(prompt_id="f" * 64))
    tuning = make_tuning(cfg, context, bedrock, s3)
    assert "the prompt wording changed" in result(
        tuning.preflight(tuning.plan("nova-lite")), "T2"
    )["detail"]

    open_gate(cfg, context, s3)
    rewrite_gate(cfg, lambda record: record["inputs"]["price_list"].update(published="2027-01-05"))
    assert "the price list changed" in result(
        tuning.preflight(tuning.plan("nova-lite")), "T2"
    )["detail"]


def test_t3_fails_when_an_object_is_missing_or_is_another_file(tuning, s3, gate):
    key = gate["staged"]["training"]["s3_uri"].removeprefix(f"s3://{BUCKET}/")
    original = s3.objects[(BUCKET, key)]

    s3.objects[(BUCKET, key)] = {"Body": original["Body"] + b"{}\n", "ChecksumSHA256": "AAAA"}
    checks = tuning.preflight(tuning.plan("nova-lite"))
    assert failed(checks) == ["T3"]
    assert "is not the file the gate record names" in result(checks, "T3")["detail"]

    del s3.objects[(BUCKET, key)]
    checks = tuning.preflight(tuning.plan("nova-lite"))
    assert failed(checks) == ["T3"] and "is not there" in result(checks, "T3")["detail"]


# --------------------------------------------------------------------------- T4, T5, T6


def test_t4_fails_for_a_model_that_is_not_fine_tunable_or_is_legacy(tuning, bedrock):
    bedrock.fine_tunable = ["amazon.nova-pro-v1:0:300k"]
    checks = tuning.preflight(tuning.plan("nova-lite"))
    assert failed(checks) == ["T4"]
    assert "amazon.nova-lite-v1:0:300k is not listed as fine-tunable" in (
        result(checks, "T4")["detail"]
    )
    bedrock.fine_tunable.append("amazon.nova-lite-v1:0:300k")
    bedrock.lifecycle = "LEGACY"
    checks = tuning.preflight(tuning.plan("nova-lite"))
    assert failed(checks) == ["T4"]
    assert "amazon.nova-lite-v1:0 is LEGACY; a Legacy model cannot be fine-tuned" in (
        result(checks, "T4")["detail"]
    )


def test_t5_fails_when_the_account_has_provisioned_throughput(tuning, bedrock):
    bedrock.provisioned = [{"provisionedModelName": "left-over", "status": "InService"}]
    checks = tuning.preflight(tuning.plan("nova-lite"))
    assert failed(checks) == ["T5"]
    assert "Provisioned Throughput (left-over), which bills by the hour" in (
        result(checks, "T5")["detail"]
    )


def test_t6_fails_for_an_attempt_that_was_launched_and_names_the_next(tuning):
    plan = tuning.plan("nova-lite")
    tuning.launch(plan)
    checks = tuning.preflight(plan)
    assert "T6" in failed(checks)
    assert result(checks, "T6")["detail"] == (
        "attempt 1 of nova-lite was already launched (job status InProgress); another "
        "launch needs --attempt 2"
    )
    assert result(tuning.preflight(tuning.plan("nova-lite", 2)), "T6")["passed"] is True


def test_t6_adopts_our_own_job_and_refuses_a_strangers(tuning, bedrock):
    plan = tuning.plan("nova-lite")
    bedrock.create_model_customization_job(**plan.request())  # created, then the reply lost
    check = result(tuning.preflight(plan), "T6")
    assert check["passed"] is True and "adopts it and creates nothing" in check["detail"]

    bedrock.jobs[plan.job_name]["clientRequestToken"] = "someone-else"
    check = result(tuning.preflight(plan), "T6")
    assert check["passed"] is False
    assert f"Bedrock has a job named {plan.job_name} that this record does not know" in (
        check["detail"]
    )


# --------------------------------------------------------------------------- T7, T8, T9


def test_t7_passes_at_the_cap_and_fails_one_cent_above(cfg, context, bedrock, s3, gate):
    exact = dataclasses.replace(cfg, path_b_cap_usd=Decimal("0.165"))
    tuning = make_tuning(exact, context, bedrock, s3)
    assert result(tuning.preflight(tuning.plan("nova-lite")), "T7")["passed"] is True

    tight = dataclasses.replace(cfg, path_b_cap_usd=Decimal("0.155"))
    tuning = make_tuning(tight, context, bedrock, s3)
    check = result(tuning.preflight(tuning.plan("nova-lite")), "T7")
    assert check["passed"] is False
    assert check["detail"] == (
        "Path B so far is $0.00; with this job ($0.17 with the margin) it would be $0.17, "
        "above the cap of $0.16"
    )


def test_t7_counts_what_path_b_has_already_committed(cfg, context, bedrock, s3, gate):
    tuning = make_tuning(cfg, context, bedrock, s3)
    tuning.launch(tuning.plan("nova-lite"))  # $0.15 of training tokens in the ledger
    # A second attempt: 0.15 committed + 0.165 for this one = 0.315.
    roomy = make_tuning(dataclasses.replace(cfg, path_b_cap_usd=Decimal("0.315")), context,
                        bedrock, s3)  # fmt: skip
    assert result(roomy.preflight(roomy.plan("nova-lite", 2)), "T7")["passed"] is True
    tight = make_tuning(dataclasses.replace(cfg, path_b_cap_usd=Decimal("0.31")), context,
                        bedrock, s3)  # fmt: skip
    check = result(tight.preflight(tight.plan("nova-lite", 2)), "T7")
    assert check["passed"] is False and "Path B so far is $0.15" in check["detail"]


def test_committed_counts_inference_training_and_bills_once(four_runs, cfg, context):
    pricing = load_pricing()
    inference = committed_path_b(ledger_records(context), pricing)
    assert inference == Decimal("0.00083248")  # the four baseline runs, at list price
    # A run of a model without a price, and a run of another path, add nothing.
    from tests.frontier.conftest import make_predictor

    foreign = make_predictor(cfg, "nova-pro", model_id="arn:aws:bedrock:us-east-1:1:x/y")
    baseline_run(context, cfg, "nova-pro", "fit", predictor=foreign)
    assert committed_path_b(ledger_records(context), pricing) == inference


def test_t8_fails_for_uncommitted_changes_and_an_unknown_commit(cfg, context, bedrock, s3, gate):
    dirty = GitInfo(commit="a" * 40, branch="main", dirty=True,
                    changed=("config/frontier.yaml", "a.py", "b.py", "c.py"))  # fmt: skip
    tuning = make_tuning(cfg, dataclasses.replace(context, git=dirty), bedrock, s3)
    checks = tuning.preflight(tuning.plan("nova-lite"))
    assert failed(checks) == ["T8"]
    assert result(checks, "T8")["detail"] == (
        "there are uncommitted changes (config/frontier.yaml, a.py, b.py, ...); commit them first"
    )
    unknown = GitInfo(commit="unknown")
    tuning = make_tuning(cfg, dataclasses.replace(context, git=unknown), bedrock, s3)
    assert "the commit of this code is not known" in result(
        tuning.preflight(tuning.plan("nova-lite")), "T8"
    )["detail"]


def test_t9_holds_nova_pro_until_nova_lite_has_completed_and_been_billed(tuning, bedrock):
    pro = tuning.plan("nova-pro")
    assert failed(tuning.preflight(pro)) == ["T9"]
    assert result(tuning.preflight(pro), "T9")["detail"] == "nova-lite has not been launched"

    lite = tuning.launch(tuning.plan("nova-lite"))
    assert "the job of nova-lite is InProgress, not Completed" in (
        result(tuning.preflight(pro), "T9")["detail"]
    )

    bedrock.finish(lite.job_name)
    tuning.refresh()
    detail = result(tuning.preflight(pro), "T9")["detail"]
    assert "the amount AWS billed for nova-lite has not been entered" in detail
    assert "nova_tune bill --model nova-lite --usd N" in detail

    tuning.bill("nova-lite", "0.18")
    check = result(tuning.preflight(pro), "T9")
    assert check["passed"] is True
    # Billed 0.18 against 0.15 is 1.20 times; Pro's 0.60 would then be 0.72.
    assert check["detail"] == (
        "nova-lite was billed $0.18 against an estimate of $0.15 (1.20 times); at that "
        "proportion this job is about $0.72"
    )
    assert failed(tuning.preflight(pro)) == []


def test_t9_fails_when_nova_lites_job_failed(tuning, bedrock):
    lite = tuning.launch(tuning.plan("nova-lite"))
    bedrock.finish(lite.job_name, status="Failed", failed_in="dataProcessingDetails")
    tuning.refresh()
    assert "the job of nova-lite is Failed, not Completed" in (
        result(tuning.preflight(tuning.plan("nova-pro")), "T9")["detail"]
    )


def test_nova_lite_never_waits_on_nova_pro(tuning):
    assert result(tuning.preflight(tuning.plan("nova-lite")), "T9")["passed"] is True
