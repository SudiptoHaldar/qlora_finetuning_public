"""Which tuned model is this experiment's B1 (req_004 v4.04).

A tuned Nova model is asked through the on-demand deployment that the tuning record of
v4.03 names. Before a question is sent, three things must be true of the model's latest
attempt: its job completed, its deployment is recorded as Active, and it was tuned on the
training files of the current export, made for today's prompt and split. Anything else is
a ``ConfigError`` that names the command to run.

Usage::

    from qlora_finetuning.frontier.tuned import tuned_attempt, tuned_details

    attempt = tuned_attempt("nova-lite", cfg, prompt.prompt_id, split_hash)
    attempt.deployment_arn        # the modelId to call
    attempt.base_model_id         # what the calls are billed as
    tuned_details(attempt)        # what a run stores about the model that answered

This module reads two committed records and calls no AWS service. Whether Bedrock agrees
that the deployment is Active is asked by the predictor's factory, once, before a run.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qlora_finetuning.frontier.config import FrontierConfig
from qlora_finetuning.frontier.training_estimate import read_gate_record
from qlora_finetuning.frontier.tuning_record import (
    ACTIVE,
    COMPLETED,
    Attempt,
    TuningRecord,
    read_tuning_record,
)
from qlora_finetuning.settings import ConfigError

TUNE = "python -m scripts.nova_tune"
TUNED_SUFFIX = "-tuned"  # the predictor of model M is named M-tuned


def attempt_problems(attempt: Attempt | None, model_name: str) -> list[str]:
    """Why the recorded attempt cannot be asked questions; empty when it can."""
    if attempt is None or not attempt.job_arn:
        return [
            f"{model_name} has not been fine-tuned: the tuning record has no job for it. "
            f"The owner runs {TUNE} launch --model {model_name} --yes (billed)"
        ]
    which = f"{model_name} (attempt {attempt.attempt})"
    if attempt.job_status != COMPLETED:
        return [
            f"the fine-tuning job of {which} is {attempt.job_status or 'not known'}, not "
            f"{COMPLETED}; run {TUNE} status"
        ]
    if attempt.model_deleted:
        return [f"the tuned model of {which} was deleted; a new attempt is needed"]
    if not attempt.deployment_arn or attempt.deployment_deleted:
        return [
            f"the tuned model of {which} has no deployment; the owner runs "
            f"{TUNE} deploy --model {model_name} --yes"
        ]
    if attempt.deployment_status != ACTIVE:
        return [
            f"the deployment of {which} is {attempt.deployment_status or 'not known'} in the "
            f"tuning record, not {ACTIVE}; run {TUNE} status"
        ]
    return []


def export_problems(
    attempt: Attempt, gate: Mapping[str, Any] | None, prompt_id: str, split_hash: str
) -> list[str]:
    """Why the attempt is not the tuned model of today's experiment; empty when it is."""
    if not gate:
        return ["there is no gate record; run python -m scripts.nova_export"]
    problems = []
    if attempt.export_id != gate.get("export_id"):
        problems.append(
            f"{attempt.model} (attempt {attempt.attempt}) was tuned on export "
            f"{attempt.export_id}, and the gate record names export {gate.get('export_id')}"
        )
    inputs = gate.get("inputs") or {}
    if inputs.get("prompt_id") != prompt_id:
        problems.append(
            "the prompt wording changed since the training files were exported, so the "
            "tuned model was trained for another question"
        )
    if inputs.get("split_content_hash") != split_hash:
        problems.append("the split manifest changed since the training files were exported")
    return problems


def tuned_attempt(
    model_name: str,
    cfg: FrontierConfig,
    prompt_id: str,
    split_hash: str,
    record: TuningRecord | None = None,
    gate: Mapping[str, Any] | None = None,
) -> Attempt:
    """The latest attempt of ``model_name``, if it can be asked and is today's experiment.

    ``record`` and ``gate`` default to the committed files. Raises ``ConfigError`` with the
    first thing that is wrong.
    """
    cfg.model(model_name)  # an unknown name is a config error
    record = read_tuning_record(cfg) if record is None else record
    attempt = record.latest(model_name)
    problems = attempt_problems(attempt, model_name)
    if not problems:
        if gate is None:
            try:
                gate = read_gate_record(cfg)
            except ConfigError:
                gate = None
        problems = export_problems(attempt, gate, prompt_id, split_hash)
    if problems:
        raise ConfigError(problems[0])
    return attempt


def tuned_details(attempt: Attempt) -> dict[str, Any]:
    """What a run stores about the tuned model that answered. Names, ARNs, numbers only."""
    return {
        "tuned": True,
        "model": attempt.model,
        "attempt": attempt.attempt,
        "base_model_id": attempt.base_model_id,
        "deployment_arn": attempt.deployment_arn,
        "model_arn": attempt.model_arn,
        "job_arn": attempt.job_arn,
        "export_id": attempt.export_id,
        "hyperparameters": dict(attempt.hyperparameters),
    }
