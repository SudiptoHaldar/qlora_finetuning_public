"""Shared pieces for the v4.04 tests: deployed attempts, scripted replies, tuned runs.

Nothing here reaches AWS. A deployed attempt is written straight into the tuning record,
as v4.03 would leave it after ``launch``, ``status``, and ``deploy``; the gate record is the
real one of v4.02 on the invented split (dev 8, quick-dev 3). Runs go through the real
evaluation runner over a provider that answers from a script.
"""

from __future__ import annotations

from qlora_finetuning.frontier.predictor import FrontierPredictor
from qlora_finetuning.frontier.provider import ProviderReply
from qlora_finetuning.frontier.tuned import tuned_details
from qlora_finetuning.frontier.tuning_record import (
    Attempt,
    TuningRecord,
    read_tuning_record,
    write_tuning_record,
)
from tests.evaluation.conftest import GOLD_REPLY, evaluate
from tests.frontier.conftest import INPUT_TOKENS, OUTPUT_TOKENS, baseline_run, make_predictor
from tests.frontier.tuning_helpers import ACCOUNT, BUCKET, open_gate

WRONG_REPLY = "PROGRAM: add(1, 2)\nANSWER: 3"  # well-formed, and not the gold answer
MALFORMED_REPLY = "the answer is probably 825"  # no program: fails format validity
SPLITS = ("quick_dev", "dev")


class ScriptedProvider:
    """Answers the n-th request with the n-th reply; ``right`` says which are the gold one."""

    def __init__(self, right: list[bool], wrong: str = WRONG_REPLY) -> None:
        self.replies = [GOLD_REPLY if ok else wrong for ok in right]
        self.calls: list[tuple[str, str, str, int]] = []

    def converse(self, model_id, system, user, max_tokens):
        text = self.replies[len(self.calls)]
        self.calls.append((model_id, system, user, max_tokens))
        return ProviderReply(
            text=text,
            input_tokens=INPUT_TOKENS,
            output_tokens=OUTPUT_TOKENS,
            stop_reason="end",
            latency_seconds=0.25,
        )


def deployed_attempt(cfg, model_name: str, export_id: str, attempt: int = 1, **changes) -> Attempt:
    """An attempt whose job completed and whose deployment is Active."""
    model = cfg.model(model_name)
    name = f"{cfg.name_prefix}-{model_name}-{export_id}-a{attempt}"
    values = {
        "model": model_name,
        "attempt": attempt,
        "arm": model.tuned_arm,
        "base_model_id": model.model_id,
        "customization_model_id": model.customization_model_id,
        "job_name": name,
        "custom_model_name": name,
        "deployment_name": name,
        "client_request_token": f"token-{model_name}-{attempt}",
        "hyperparameters": cfg.hyperparameters,
        "export_id": export_id,
        "training_s3_uri": f"s3://{BUCKET}/data/bedrock/{export_id}/finqa_fit.jsonl",
        "validation_s3_uri": f"s3://{BUCKET}/data/bedrock/{export_id}/finqa_dev.jsonl",
        "output_s3_uri": f"s3://{BUCKET}/bedrock/{export_id}/{model_name}-a{attempt}/",
        "estimated_training_tokens": 1_000_000,
        "estimated_usd": "2.00",
        "job_arn": f"{ACCOUNT}:model-customization-job/{name}",
        "job_status": "Completed",
        "model_arn": f"{ACCOUNT}:custom-model/{name}",
        "training_loss": 0.6,
        "validation_loss": 0.7,
        "deployment_arn": f"{ACCOUNT}:custom-model-deployment/{name}",
        "deployment_status": "Active",
    }
    values.update(changes)
    return Attempt(**values)


def deploy_models(cfg, context, models=("nova-pro", "nova-lite"), **changes) -> TuningRecord:
    """An open gate for the invented split, and a deployed attempt per model in the record."""
    gate = open_gate(cfg, context)
    record = read_tuning_record(cfg)
    for name in models:
        record.put(deployed_attempt(cfg, name, gate["export_id"], **changes))
    write_tuning_record(record, cfg)
    return record


def tuned_predictor(cfg, name: str, provider, record=None, **changes) -> FrontierPredictor:
    """The tuned predictor of ``name`` over a stub provider, as the factory builds it."""
    attempt = (record or read_tuning_record(cfg)).latest(name)
    return FrontierPredictor(
        provider,
        changes.get("model_id", attempt.deployment_arn),
        f"{name}-tuned",
        changes.get("settings", cfg.settings),
        usage_model_id=changes.get("usage_model_id", attempt.base_model_id),
        details=tuned_details(attempt),
    )


def tuned_run(context, cfg, name: str, split: str, right: list[bool], **options) -> dict:
    """One tuned run as the script records it: path B, the model's tuned arm, Bedrock."""
    wrong = options.pop("wrong", WRONG_REPLY)
    predictor = options.pop("predictor", None) or tuned_predictor(
        cfg, name, ScriptedProvider(right, wrong)
    )
    settings = {
        "label": f"{name}-tuned {split}",
        "path": "B",
        "arm": cfg.model(name).tuned_arm,
        "where": "bedrock",
        "kind": "eval",
    }
    settings.update(options)
    return evaluate(predictor, split, context, **settings)


def scripted_baseline(context, cfg, name: str, split: str, right: list[bool], **options) -> dict:
    """A baseline run whose replies are right or wrong as ``right`` says."""
    predictor = make_predictor(cfg, name, ScriptedProvider(right))
    return baseline_run(context, cfg, name, split, predictor=predictor, **options)


def sizes() -> dict[str, int]:
    return {"quick_dev": 3, "dev": 8}


def right_for(split: str, correct: int) -> list[bool]:
    """The first ``correct`` examples right, the rest wrong."""
    total = sizes()[split]
    return [index < correct for index in range(total)]
