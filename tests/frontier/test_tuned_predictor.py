"""Tests for the tuned Nova predictors: the factory, the request, and runs through the runner."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from botocore.exceptions import EndpointConnectionError, NoCredentialsError

from qlora_finetuning.data import TestLockedError
from qlora_finetuning.evaluation.predictor import PredictionRequest, PredictorError, describe
from qlora_finetuning.evaluation.runner import ResumeError
from qlora_finetuning.frontier import predictor as predictor_module
from qlora_finetuning.frontier.bedrock import BedrockProvider, deployment_status
from qlora_finetuning.frontier.predictor import build_predictor, build_tuned_predictor
from qlora_finetuning.frontier.pricing import inference_cost, load_pricing
from qlora_finetuning.frontier.tuned import tuned_details
from qlora_finetuning.frontier.tuning import committed_path_b
from qlora_finetuning.frontier.tuning_record import read_tuning_record
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking import Ledger
from tests.evaluation.conftest import (
    ledger_records,
    read_lines,
    records_for,
    results_lines,
    run_folders,
)
from tests.frontier.conftest import INPUT_TOKENS, OUTPUT_TOKENS, StubProvider
from tests.frontier.tuned_helpers import (
    ScriptedProvider,
    deploy_models,
    right_for,
    tuned_predictor,
    tuned_run,
)
from tests.frontier.tuning_helpers import FakeBedrock, client_error

AWS = {"profile": "qlora", "account_id": "123456789012", "region": "us-east-1"}


def converse_response(text="PROGRAM: add(1, 2)\nANSWER: 3") -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 40, "outputTokens": 12},
    }


class FakeSession:
    """Hands out an STS stub, the in-memory Bedrock, and a runtime stub, and notes the order."""

    def __init__(self, bedrock: FakeBedrock, account="123456789012"):
        self.sts = MagicMock()
        self.sts.get_caller_identity.return_value = {"Account": account}
        self.runtime = MagicMock()
        self.bedrock = bedrock
        self.asked: list[str] = []

    def client(self, service, config=None):
        self.asked.append(service)
        return {"sts": self.sts, "bedrock": self.bedrock, "bedrock-runtime": self.runtime}[service]


def deployed(cfg, context, status="Active", models=("nova-pro", "nova-lite")) -> FakeBedrock:
    """Deployed attempts in the record, and the same deployments in an in-memory Bedrock."""
    record = deploy_models(cfg, context, models=models)
    bedrock = FakeBedrock()
    for attempt in record.attempts:
        bedrock.deployments[attempt.deployment_name] = {
            "customModelDeploymentArn": attempt.deployment_arn,
            "modelDeploymentName": attempt.deployment_name,
            "modelArn": attempt.model_arn,
            "status": status,
        }
    return bedrock


# --------------------------------------------------------------------------- the one read


def test_deployment_status_is_what_bedrock_says_and_none_when_it_is_not_there(cfg, context):
    bedrock = deployed(cfg, context)
    arn = read_tuning_record(cfg).latest("nova-lite").deployment_arn
    assert deployment_status(bedrock, arn) == "Active"
    assert deployment_status(bedrock, arn + "-gone") is None
    assert bedrock.changes() == []  # a read


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (NoCredentialsError(), "aws sso login --profile qlora"),
        (client_error("ExpiredTokenException"), "aws sso login --profile qlora"),
        (client_error("AccessDeniedException"), "AWS answered AccessDeniedException when asked"),
        (
            EndpointConnectionError(endpoint_url="https://bedrock.example"),
            "AWS could not be reached to read the deployment: EndpointConnectionError",
        ),
    ],
)
def test_deployment_status_turns_aws_errors_into_config_errors(error, message):
    control = MagicMock()
    control.get_custom_model_deployment.side_effect = error
    with pytest.raises(ConfigError, match=message):
        deployment_status(control, "arn:aws:bedrock:us-east-1:1:custom-model-deployment/x")


def test_a_response_without_a_status_counts_as_not_there():
    control = MagicMock()
    control.get_custom_model_deployment.return_value = {"modelDeploymentName": "x"}
    assert deployment_status(control, "arn") is None


# --------------------------------------------------------------------------- the factory


def test_the_factory_checks_the_record_the_sign_in_and_the_deployment_and_bills_nothing(
    cfg, context
):
    bedrock = deployed(cfg, context)
    session = FakeSession(bedrock)
    predictor = build_tuned_predictor(
        "nova-lite", context=context, cfg=cfg, aws=AWS, session=session
    )
    attempt = read_tuning_record(cfg).latest("nova-lite")
    assert session.asked == ["sts", "bedrock", "bedrock-runtime"]
    assert bedrock.calls == ["get_custom_model_deployment"] and bedrock.changes() == []
    assert session.runtime.mock_calls == []  # building a predictor costs nothing
    assert describe(predictor) == {
        "name": "nova-lite-tuned",
        "model_id": attempt.deployment_arn,
        "settings": cfg.settings,
        "batch_size": 1,
        "usage_source": "aws_api",
        "usage_model_id": "amazon.nova-lite-v1:0",
        "details": tuned_details(attempt),
    }
    assert predictor.provider.client is session.runtime


def test_the_factory_refuses_from_the_record_alone_before_any_sign_in(cfg, context):
    session = FakeSession(FakeBedrock())
    with pytest.raises(ConfigError, match="nova-pro has not been fine-tuned"):
        build_tuned_predictor("nova-pro", context=context, cfg=cfg, aws=AWS, session=session)
    assert session.asked == []


@pytest.mark.parametrize(
    ("status", "said"), [("Creating", "is Creating"), ("Failed", "is Failed"), (None, "not there")]
)
def test_the_factory_refuses_when_bedrock_does_not_say_active(cfg, context, status, said):
    bedrock = deployed(cfg, context, status=status or "Active")
    if status is None:
        bedrock.deployments.clear()  # recorded Active, deleted in the console
    session = FakeSession(bedrock)
    with pytest.raises(ConfigError) as caught:
        build_tuned_predictor("nova-lite", context=context, cfg=cfg, aws=AWS, session=session)
    message = str(caught.value)
    assert said in message and "according to Bedrock, not Active" in message
    assert "run python -m scripts.nova_tune status" in message
    assert session.asked == ["sts", "bedrock"]  # no runtime client: nothing could be asked


def test_the_factory_refuses_another_account(cfg, context):
    session = FakeSession(deployed(cfg, context), account="111122223333")
    with pytest.raises(ConfigError, match="expected 123456789012"):
        build_tuned_predictor("nova-lite", context=context, cfg=cfg, aws=AWS, session=session)
    assert session.asked == ["sts"]


@pytest.mark.parametrize(
    ("factory", "name"),
    [
        (predictor_module.nova_pro_tuned, "nova-pro"),
        (predictor_module.nova_lite_tuned, "nova-lite"),
    ],
)
def test_the_script_factories_name_their_model_and_pass_the_context(monkeypatch, factory, name):
    seen = []
    monkeypatch.setattr(
        predictor_module,
        "build_tuned_predictor",
        lambda model, context: seen.append((model, context)),
    )
    factory("quick_dev", "the context")
    assert seen == [(name, "the context")]


# --------------------------------------------------------------------------- the request


def test_the_request_differs_from_the_untuned_one_in_the_model_id_only(cfg, context):
    bedrock = deployed(cfg, context)
    session = FakeSession(bedrock)
    session.runtime.converse.return_value = converse_response()
    tuned = build_tuned_predictor("nova-lite", context=context, cfg=cfg, aws=AWS, session=session)
    untuned = build_predictor("nova-lite", cfg=cfg, aws=AWS, session=session)
    request = [PredictionRequest("ex-0", "system text \n", "user text£ ", 192)]
    tuned.predict(request)
    untuned.predict(request)
    sent_tuned, sent_untuned = (call.kwargs for call in session.runtime.converse.call_args_list)
    assert sent_tuned["modelId"] == tuned.model_id != sent_untuned["modelId"]
    assert sent_untuned["modelId"] == "amazon.nova-lite-v1:0"
    assert {k: v for k, v in sent_tuned.items() if k != "modelId"} == {
        k: v for k, v in sent_untuned.items() if k != "modelId"
    }
    assert isinstance(tuned.provider, BedrockProvider)


# --------------------------------------------------------------------------- through the runner


def test_a_tuned_run_is_recorded_on_the_tuned_arm_and_billed_as_the_base_model(cfg, context):
    deploy_models(cfg, context)
    provider = ScriptedProvider(right_for("dev", 8))
    predictor = tuned_predictor(cfg, "nova-lite", provider)
    summary = tuned_run(context, cfg, "nova-lite", "dev", [], predictor=predictor)
    attempt = read_tuning_record(cfg).latest("nova-lite")

    assert summary["result"] and summary["all"]["execution_accuracy"]["count"] == 8
    assert (summary["path"], summary["arm"], summary["where"]) == ("B", "B1-Lite", "bedrock")
    assert summary["predictor"]["name"] == "nova-lite-tuned"
    assert summary["predictor"]["model_id"] == attempt.deployment_arn
    assert summary["predictor"]["settings"] == cfg.settings  # equal to the baseline's
    assert summary["predictor"]["details"]["job_arn"] == attempt.job_arn

    # Every call went to the deployment, with the canonical texts, in dataset order.
    records = records_for("dev", context)
    assert {call[0] for call in provider.calls} == {attempt.deployment_arn}
    assert [(call[1], call[2]) for call in provider.calls] == [(r.system, r.user) for r in records]

    record = next(r for r in ledger_records(context) if r.run_id == summary["run_id"])
    assert (record.model_id, record.arm, record.source) == (
        "amazon.nova-lite-v1:0", "B1-Lite", "aws_api",
    )  # fmt: skip
    assert (record.input_tokens, record.output_tokens) == (8 * INPUT_TOKENS, 8 * OUTPUT_TOKENS)
    line = next(row for row in results_lines(context) if row["run_id"] == summary["run_id"])
    assert (line["arm"], line["model_id"]) == ("B1-Lite", attempt.deployment_arn)


def test_the_cap_check_counts_tuned_inference_because_it_is_priced_as_the_base_model(cfg, context):
    deploy_models(cfg, context)
    pricing = load_pricing()
    before = committed_path_b(Ledger(context.tracking_cfg).effective(), pricing)
    tuned_run(context, cfg, "nova-pro", "dev", right_for("dev", 8))
    after = committed_path_b(Ledger(context.tracking_cfg).effective(), pricing)
    expected = inference_cost("amazon.nova-pro-v1:0", 8 * INPUT_TOKENS, 8 * OUTPUT_TOKENS, pricing)
    assert after - before == expected > 0


def test_a_failed_tuned_run_resumes_against_the_same_deployment_only(cfg, context):
    deploy_models(cfg, context)
    failing = tuned_predictor(cfg, "nova-lite", StubProvider(fail_after=3))
    with pytest.raises(PredictorError) as caught:
        tuned_run(context, cfg, "nova-lite", "dev", [], predictor=failing)
    failed_id = caught.value.run_id
    assert len(read_lines(failed_id, context)) == 3

    other = tuned_predictor(cfg, "nova-lite", StubProvider(), model_id="arn:another-deployment")
    with pytest.raises(ResumeError, match="another predictor"):
        tuned_run(context, cfg, "nova-lite", "dev", [], predictor=other, resume=failed_id)
    # Nor can the untuned model continue a tuned run, or the other way round.
    untuned = tuned_predictor(cfg, "nova-lite", StubProvider(), model_id="amazon.nova-lite-v1:0")
    with pytest.raises(ResumeError, match="another predictor"):
        tuned_run(context, cfg, "nova-lite", "dev", [], predictor=untuned, resume=failed_id)

    same = tuned_predictor(cfg, "nova-lite", StubProvider())
    summary = tuned_run(context, cfg, "nova-lite", "dev", [], predictor=same, resume=failed_id)
    assert summary["result"] and summary["continues"] == failed_id
    assert len(same.provider.calls) == 5  # the three answered before are not asked again


def test_the_locked_test_split_is_refused_and_the_tuned_model_is_asked_nothing(cfg, context):
    deploy_models(cfg, context)
    provider = StubProvider()
    before = run_folders(context)
    with pytest.raises(TestLockedError):
        tuned_run(
            context,
            cfg,
            "nova-lite",
            "test",
            [],
            predictor=tuned_predictor(cfg, "nova-lite", provider),
        )
    assert provider.calls == [] and run_folders(context) == before
