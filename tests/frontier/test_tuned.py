"""Tests for qlora_finetuning.frontier.tuned: which attempt is this experiment's B1."""

from __future__ import annotations

import json

import pytest

from qlora_finetuning.data import manifest_hash
from qlora_finetuning.frontier.training_estimate import report_paths as gate_paths
from qlora_finetuning.frontier.tuned import (
    attempt_problems,
    export_problems,
    tuned_attempt,
    tuned_details,
)
from qlora_finetuning.frontier.tuning_record import (
    TuningRecord,
    read_tuning_record,
    write_tuning_record,
)
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking.config import check_no_secrets
from tests.evaluation.conftest import GOLD_REPLY
from tests.frontier.tuned_helpers import deploy_models, deployed_attempt


def ask(cfg, context, name="nova-lite", **overrides):
    prompt_id = overrides.pop("prompt_id", context.prompt.prompt_id)
    split_hash = overrides.pop("split_hash", manifest_hash(context.split_cfg))
    return tuned_attempt(name, cfg, prompt_id, split_hash, **overrides)


def change(cfg, name="nova-lite", **changes) -> None:
    """Change fields of the recorded latest attempt of ``name``."""
    record = read_tuning_record(cfg)
    attempt = record.latest(name)
    for field, value in changes.items():
        setattr(attempt, field, value)
    write_tuning_record(record, cfg)


def test_a_completed_job_with_an_active_deployment_can_be_asked(cfg, context):
    deploy_models(cfg, context)
    attempt = ask(cfg, context)
    assert (attempt.model, attempt.attempt, attempt.arm) == ("nova-lite", 1, "B1-Lite")
    assert attempt.deployment_arn.endswith(f"custom-model-deployment/{attempt.deployment_name}")
    assert attempt.base_model_id == "amazon.nova-lite-v1:0"  # what the calls are billed as
    assert ask(cfg, context, "nova-pro").arm == "B1-Pro"


def test_nothing_launched_names_the_launch_command_and_the_owner(cfg, context):
    with pytest.raises(ConfigError) as caught:
        ask(cfg, context)
    message = str(caught.value)
    assert "nova-lite has not been fine-tuned" in message
    assert "python -m scripts.nova_tune launch --model nova-lite --yes" in message
    assert "owner" in message and "billed" in message


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"job_status": "InProgress"}, "is InProgress, not Completed; run python -m scripts.nova"),
        ({"job_status": "Failed"}, "is Failed, not Completed"),
        ({"job_status": None}, "is not known, not Completed"),
        ({"job_arn": None}, "has not been fine-tuned"),
        ({"model_deleted": True}, "was deleted; a new attempt is needed"),
        ({"deployment_arn": None}, "has no deployment; the owner runs python -m scripts.nova_tune"),
        ({"deployment_deleted": True}, "nova_tune deploy --model nova-lite --yes"),
        ({"deployment_status": "Creating"}, "is Creating in the tuning record, not Active"),
        ({"deployment_status": "Failed"}, "is Failed in the tuning record, not Active"),
        ({"export_id": "0123456789ab"}, "was tuned on export 0123456789ab, and the gate record"),
    ],
)
def test_each_reason_an_attempt_cannot_be_asked_names_what_to_do(cfg, context, changes, message):
    deploy_models(cfg, context)
    change(cfg, **changes)
    with pytest.raises(ConfigError, match=message):
        ask(cfg, context)
    assert ask(cfg, context, "nova-pro")  # the other model is not affected


def test_a_changed_prompt_or_split_means_the_model_answers_another_question(cfg, context):
    deploy_models(cfg, context)
    with pytest.raises(ConfigError, match="the prompt wording changed since the training files"):
        ask(cfg, context, prompt_id="another-prompt")
    with pytest.raises(ConfigError, match="the split manifest changed since the training files"):
        ask(cfg, context, split_hash="another-split")


def test_without_a_gate_record_nothing_says_what_the_model_was_tuned_on(cfg, context):
    deploy_models(cfg, context)
    for path in gate_paths(cfg.report_dir):
        path.unlink()
    with pytest.raises(ConfigError, match="there is no gate record; run python -m scripts.nova_e"):
        ask(cfg, context)


def test_the_latest_attempt_decides_even_when_an_earlier_one_is_deployed(cfg, context):
    record = deploy_models(cfg, context)
    export_id = record.latest("nova-lite").export_id
    record.put(
        deployed_attempt(
            cfg, "nova-lite", export_id, attempt=2, job_status="InProgress", deployment_arn=None
        )
    )
    write_tuning_record(record, cfg)
    with pytest.raises(ConfigError, match=r"nova-lite \(attempt 2\) is InProgress"):
        ask(cfg, context)


def test_an_unknown_model_is_a_config_error(cfg, context):
    with pytest.raises(ConfigError, match="no model named 'nova-max'"):
        ask(cfg, context, "nova-max")


def test_a_record_and_a_gate_can_be_passed_in(cfg, context):
    attempt = deployed_attempt(cfg, "nova-lite", "abcabcabcabc")
    gate = {
        "export_id": "abcabcabcabc",
        "inputs": {"prompt_id": "p", "split_content_hash": "s"},
    }
    found = tuned_attempt("nova-lite", cfg, "p", "s", record=TuningRecord([attempt]), gate=gate)
    assert found is attempt
    assert export_problems(attempt, gate, "p", "s") == []
    assert attempt_problems(attempt, "nova-lite") == []
    assert len(export_problems(attempt, gate, "q", "t")) == 2
    assert export_problems(attempt, None, "p", "s") == [
        "there is no gate record; run python -m scripts.nova_export"
    ]


def test_the_details_are_names_arns_and_numbers(cfg, context):
    deploy_models(cfg, context)
    attempt = ask(cfg, context)
    details = tuned_details(attempt)
    assert details == {
        "tuned": True,
        "model": "nova-lite",
        "attempt": 1,
        "base_model_id": "amazon.nova-lite-v1:0",
        "deployment_arn": attempt.deployment_arn,
        "model_arn": attempt.model_arn,
        "job_arn": attempt.job_arn,
        "export_id": attempt.export_id,
        "hyperparameters": {
            "epochCount": "2",
            "learningRate": "0.00001",
            "learningRateWarmupSteps": "10",
        },
    }
    text = json.dumps(details)
    assert json.loads(text) == details  # plain values
    check_no_secrets(details, "the details")
    assert GOLD_REPLY not in text and "subtract" not in text
    details["hyperparameters"]["epochCount"] = "5"
    assert attempt.hyperparameters["epochCount"] == "2"  # a copy
