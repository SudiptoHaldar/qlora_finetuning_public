"""Stand-ins for the comparison tests: a frozen, unlocked invented split under ``tmp_path``.

The invented split of ``tests/evaluation/conftest.py`` has 22 test examples. Nothing here
touches the repository's own lock: the freeze record and the unlock record are written
under the temporary root.
"""

from __future__ import annotations

from typing import Any

from qlora_finetuning.comparison.config import ComparisonConfig, load_comparison_config
from qlora_finetuning.comparison.gate import (
    SCHEMA,
    contract_now,
    shared_state,
    write_canonical,
)
from qlora_finetuning.evaluation.fake import gold_predictor
from qlora_finetuning.evaluation.runner import RunnerContext, run_evaluation
from tests.evaluation.conftest import records_for

DEPLOYMENT = "arn:aws:bedrock:us-east-1:000000000000:custom-model-deployment/{}"
QWEN = {"adapter": None, "batch_size": 4, "revision": "abc"}
NOVA = {"api": "converse", "temperature": 0, "top_k": 1}
# What the invented freeze says each arm's predictor is.
PREDICTORS: dict[str, dict[str, Any]] = {
    "A0": {"name": "qwen3-4b", "model_id": "Qwen/Qwen3-4B", "settings": QWEN},
    "A1": {
        "name": "qwen3-4b-tuned",
        "model_id": "Qwen/Qwen3-4B",
        "settings": QWEN | {"adapter": "f" * 64},
    },
    "B0-Lite": {"name": "nova-lite", "model_id": "amazon.nova-lite-v1:0", "settings": NOVA},
    "B1-Lite": {
        "name": "nova-lite-tuned",
        "model_id": DEPLOYMENT.format("lite"),
        "settings": NOVA,
    },
    "B0-Pro": {"name": "nova-pro", "model_id": "amazon.nova-pro-v1:0", "settings": NOVA},
    "B1-Pro": {"name": "nova-pro-tuned", "model_id": DEPLOYMENT.format("pro"), "settings": NOVA},
}
USAGE = {"A": "measured", "B": "aws_api"}


def comparison_cfg(context: RunnerContext) -> ComparisonConfig:
    return load_comparison_config(root=context.tracking_cfg.root)


def freeze_record(context: RunnerContext, arms: dict[str, dict[str, Any]] | None = None) -> dict:
    """A freeze record for the invented split, as ``build_freeze`` would shape it."""
    cfg = comparison_cfg(context)
    arms = PREDICTORS if arms is None else arms
    return {
        "schema": SCHEMA,
        "contract": contract_now(context),
        "shared": shared_state(cfg, context.config_dir, context.config.flags_file),
        "arms": {
            arm: {
                "path": cfg.arms[arm].path,
                "where": cfg.arms[arm].where,
                "predictor": predictor,
                "identity": {},
            }
            for arm, predictor in arms.items()
        },
        "checks": [],
        "passed": True,
    }


def write_freeze_record(context: RunnerContext, arms: dict | None = None) -> str:
    return write_canonical(comparison_cfg(context).freeze_file, freeze_record(context, arms))


def write_unlock(context: RunnerContext, digest: str | None, **more: Any) -> None:
    record = {"schema": SCHEMA, "manifest_hash": context.manifest["content_hash"]} | more
    if digest is not None:
        record["freeze_sha256"] = digest
    write_canonical(context.split_cfg.unlock_file, record)


def freeze_and_unlock(context: RunnerContext, arms: dict | None = None) -> str:
    """Freeze the invented arms and unlock the invented split; returns the freeze's hash."""
    digest = write_freeze_record(context, arms)
    write_unlock(context, digest, freeze_commit="0" * 40, unlocked_on="2026-11-22")
    return digest


def arm_predictor(arm: str, context: RunnerContext, **options: Any) -> Any:
    """A predictor that describes itself as the frozen one of ``arm`` and answers with gold."""
    cfg = comparison_cfg(context)
    frozen = PREDICTORS[arm]
    settings = {
        "name": frozen["name"],
        "model_id": frozen["model_id"],
        "settings": frozen["settings"],
        "usage_source": USAGE[cfg.arms[arm].path],
    }
    settings.update(options)
    return gold_predictor(records_for("test", context), **settings)


def run_arm(arm: str, context: RunnerContext, predictor: Any = None, **options: Any) -> dict:
    """One test run of ``arm`` as the real entry points would start it."""
    cfg = comparison_cfg(context)
    settings: dict[str, Any] = {
        "label": f"{cfg.arms[arm].predictor} test",
        "path": cfg.arms[arm].path,
        "arm": arm,
        "where": cfg.arms[arm].where,
        "scope": "measured",
        "kind": "eval",
    }
    settings.update(options)
    predictor = predictor if predictor is not None else arm_predictor(arm, context)
    return run_evaluation(predictor, "test", context=context, **settings)


# --------------------------------------------------------------------------- freeze and record


class FakeGit:
    """Stands in for ``GitQueries``: the unlock record is committed, and history is linear."""

    def __init__(self, unlock_commit: str | None = "c" * 40, ancestor: bool = True) -> None:
        self.unlock_commit, self.ancestor = unlock_commit, ancestor

    def added_in(self, path: Any) -> str | None:
        return self.unlock_commit

    def is_ancestor(self, earlier: str, later: str) -> bool:
        return self.ancestor


def dev_run(arm: str, context: RunnerContext, **options: Any) -> dict:
    """A complete dev result of ``arm``, made by a predictor that is the frozen one."""
    cfg = comparison_cfg(context)
    frozen = PREDICTORS[arm]
    predictor = gold_predictor(
        records_for("dev", context),
        name=frozen["name"],
        model_id=frozen["model_id"],
        settings=frozen["settings"],
        usage_source=USAGE[cfg.arms[arm].path],
    )
    settings: dict[str, Any] = {
        "label": f"{frozen['name']} dev",
        "path": cfg.arms[arm].path,
        "arm": arm,
        "where": cfg.arms[arm].where,
        "scope": "dev",
        "kind": "eval",
    }
    settings.update(options)
    return run_evaluation(predictor, "dev", context=context, **settings)


def six_dev_runs(context: RunnerContext) -> dict[str, str]:
    return {arm: dev_run(arm, context)["run_id"] for arm in PREDICTORS}


def arms_now(**changes: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """What ``current_arms`` would say: each arm's predictor, and what identifies its model."""
    arms = {
        arm: {"predictor": dict(predictor), "identity": {"attempt": 1} if "B1" in arm else {}}
        for arm, predictor in PREDICTORS.items()
    }
    for arm, change in changes.items():
        arms[arm]["predictor"].update(change)
    return arms


def job_record_for(runs: dict[str, str], **changes: Any) -> dict[str, Any]:
    """A job record in which each Qwen run was made by its own pulled, billed job."""
    jobs = []
    for arm in ("A0", "A1"):
        if arm in runs:
            job = {
                "name": f"qlora-evaluate-{arm.lower()}",
                "arm": arm,
                "instance_type": "ml.g5.xlarge",
                "pulled": True,
                "billable_seconds": 3900,
                "runs": [{"run_id": runs[arm]}],
            }
            job.update(changes)
            jobs.append(job)
    return {"schema": 1, "jobs": jobs}
