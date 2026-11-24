"""Tests for what req_005 v5.01 adds to the launcher: the test bundle, check J11, and the
refusal to rehearse on the test split.

Nothing here reaches AWS, and the split that is unlocked is the invented one under
``tmp_path``.
"""

from __future__ import annotations

import json

import pytest

from qlora_finetuning.data.bundle import export_bundle, read_record
from qlora_finetuning.data.split import read_manifest
from qlora_finetuning.data.token_budget import load_token_budget_config
from qlora_finetuning.local_qlora.config import load_local_qlora_config
from qlora_finetuning.local_qlora.sagemaker import JobOptions, is_test_job, rehearse
from qlora_finetuning.local_qlora.scoring import expected_settings
from qlora_finetuning.settings import ConfigError
from scripts import sagemaker_job
from tests.comparison.helpers import write_freeze_record, write_unlock

from .sagemaker_helpers import GIT, make_jobs, with_cfg, workspace_key

A0 = {"task": "evaluate", "predictor": "qwen3-4b", "split": "test", "scope": "measured"}


def check(plan, check_id):
    return next(item for item in plan.checks if item["id"] == check_id)


def context_of(jobs):
    """The launcher's own context, with the invented split's manifest for the helpers."""
    context = jobs._runner_context()
    context.manifest = read_manifest(jobs._split_cfg)
    return context


def frozen_a0(**changes):
    budget = load_token_budget_config()
    settings = expected_settings(load_local_qlora_config(), budget.revision, None)
    return {"name": "qwen3-4b", "model_id": budget.model_id, "settings": settings | changes}


def unlock(jobs, arms=None) -> None:
    """Freeze arm A0 as today's configuration defines it, unlock, and write the test bundle."""
    context = context_of(jobs)
    digest = write_freeze_record(context, arms or {"A0": frozen_a0()})
    write_unlock(context, digest)
    export_bundle(jobs.cfg.test_bundle_dir, ("test",), cfg=jobs._split_cfg, data_cfg=jobs._data_cfg)


def measure(jobs) -> None:
    """Run and pull a smoke job that measured, so that an evaluate job has an estimate."""
    _, job = jobs.launch(JobOptions("smoke"), yes=True)
    jobs.sagemaker.finish(job["name"], billable=1500)
    runs = [
        {"role": "eval-tuned", "run_id": "a", "status": "completed",
         "backend": {"generate_seconds": 30, "prompts": 24}},
        {"role": "eval-untuned", "run_id": "b", "status": "completed",
         "backend": {"generate_seconds": 36, "prompts": 24}},
    ]  # fmt: skip
    content = {"status": "completed", "task": "smoke", "runs": runs}
    key = workspace_key(jobs, job, f"artifacts/jobs/{job['name']}/job_result.json")
    jobs.s3.objects[key] = json.dumps(content).encode()
    assert jobs.pull()["pulled"]


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    return make_jobs(tmp_path, monkeypatch)


@pytest.fixture
def ready(jobs):
    measure(jobs)
    unlock(jobs)
    return jobs


# --------------------------------------------------------------------------- J11


def test_a_job_that_is_not_on_test_passes_j11(jobs):
    plan = jobs.plan(JobOptions("smoke"))
    assert check(plan, "J11") == {
        "id": "J11", "name": "Test run", "status": "PASS",
        "detail": "not a job on the test split",
    }  # fmt: skip
    assert not is_test_job(JobOptions("evaluate", predictor="qwen3-4b", split="dev"))


def test_a_test_job_is_refused_while_the_split_is_locked(jobs):
    plan = jobs.plan(JobOptions(**A0))
    assert not plan.passed
    assert check(plan, "J11")["status"] == "FAIL"
    assert "the test split is locked" in check(plan, "J11")["detail"]
    assert check(plan, "J3")["status"] == "FAIL"
    assert "bundle --split test" in check(plan, "J3")["detail"]
    _, job = jobs.launch(JobOptions(**A0), yes=True)
    assert job is None and jobs.sagemaker.created == []


def test_a_test_job_at_the_frozen_configuration_may_be_planned(ready):
    plan = ready.plan(JobOptions(**A0))
    assert plan.passed, [item for item in plan.checks if item["status"] != "PASS"]
    assert check(plan, "J11")["detail"] == "arm A0 at the frozen configuration; no test result yet"
    assert "test 22; 22 examples, test rows only" in check(plan, "J3")["detail"]
    assert plan.arm == "A0" and "22 replies" in plan.estimate.detail

    record = read_record(ready.cfg.test_bundle_dir)
    bundle = plan.definition["InputDataConfig"][1]["DataSource"]["S3DataSource"]["S3Uri"]
    assert bundle.endswith(ready.bundle_key_prefix(record))
    other = ready.plan(JobOptions("evaluate", predictor="qwen3-4b", split="dev"))
    dev_bundle = other.definition["InputDataConfig"][1]["DataSource"]["S3DataSource"]["S3Uri"]
    assert dev_bundle != bundle  # a dev job is still given fit and dev
    arguments = plan.definition["AlgorithmSpecification"]["ContainerArguments"]
    assert arguments[arguments.index("--split") + 1] == "test"
    assert arguments[arguments.index("--scope") + 1] == "measured"


def test_the_launch_uploads_the_test_bundle_and_a_second_job_waits_for_the_pull(ready):
    record = read_record(ready.cfg.test_bundle_dir)
    prefix = ready.bundle_key_prefix(record)
    before = set(ready.s3.objects)
    _, job = ready.launch(JobOptions(**A0), yes=True)
    assert job is not None and job["arm"] == "A0" and job["bundle_sha256"] == record["sha256"]
    uploaded = {key for key in set(ready.s3.objects) - before if key.startswith(prefix)}
    assert len(uploaded) == 2  # the examples and the bundle's record: test rows, once

    ready.sagemaker.finish(job["name"])
    again = ready.plan(JobOptions(**A0, attempt=2))
    assert check(again, "J11")["status"] == "FAIL"
    assert f"has not been pulled ({job['name']})" in check(again, "J11")["detail"]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"scope": "dev"}, "add --scope measured"),
        ({"limit": 5}, "--limit is refused on the test split"),
        # arm A1 is not in this invented freeze record: nothing says what it should be
        (
            {"predictor": "qwen3-4b-tuned", "adapter": "run/checkpoint-4", "workspace": "w"},
            "only the six arms",
        ),
    ],
)
def test_how_a_test_job_is_asked_is_checked_at_plan_time(ready, options, message):
    plan = ready.plan(JobOptions(**(A0 | options)))
    assert check(plan, "J11")["status"] == "FAIL" and not plan.passed
    assert message in check(plan, "J11")["detail"]


def test_only_an_evaluate_job_may_read_test(ready):
    plan = ready.plan(JobOptions("score", run="some-run", workspace="w", split="test"))
    assert check(plan, "J11")["status"] == "FAIL"
    assert "only an evaluate job may read the test split" in check(plan, "J11")["detail"]
    # and it is given fit and dev, where the loader refuses test whatever the unlock says
    assert "fit 30, dev 8" in check(plan, "J3")["detail"]


def test_a_changed_setting_fails_j11(jobs):
    measure(jobs)
    unlock(jobs, {"A0": frozen_a0(batch_size=1)})
    plan = jobs.plan(JobOptions(**A0))
    detail = check(plan, "J11")["detail"]
    assert check(plan, "J11")["status"] == "FAIL"
    assert "is not the frozen one" in detail and "settings (batch_size)" in detail


def test_a_fit_and_dev_bundle_in_the_test_folder_fails_j3(ready):
    with_cfg(ready, test_bundle_dir=ready.cfg.bundle_dir)
    plan = ready.plan(JobOptions(**A0))
    assert check(plan, "J3")["status"] == "FAIL"
    assert "this job is given test" in check(plan, "J3")["detail"]


def test_a_rehearsal_never_reads_test(ready, tmp_path):
    with pytest.raises(ConfigError, match="a rehearsal never reads the test split"):
        rehearse(
            "evaluate", ready.cfg, tmp_path, GIT,
            arguments=("--predictor", "qwen3-4b", "--split", "test"),
        )  # fmt: skip


# --------------------------------------------------------------------------- the command line


def run(jobs, *arguments):
    return sagemaker_job.main(list(arguments), jobs=jobs)


def test_the_test_bundle_command_follows_the_lock(jobs, capsys):
    assert run(jobs, "bundle", "--split", "test") == 2
    assert capsys.readouterr().out.startswith("LOCKED: the test split is locked")
    assert not jobs.cfg.test_bundle_dir.exists()

    write_unlock(context_of(jobs), None)  # unlocked by a record that names no freeze
    assert run(jobs, "bundle", "--split", "test") == 2
    assert "there is no freeze record" in capsys.readouterr().out
    assert not jobs.cfg.test_bundle_dir.exists()

    context = context_of(jobs)
    write_unlock(context, write_freeze_record(context, {"A0": frozen_a0()}))
    assert run(jobs, "bundle", "--split", "test") == 0
    out = capsys.readouterr().out
    assert "bundle: test 22; 22 examples" in out and "artifacts/bundle_test/" in out
    assert read_record(jobs.cfg.test_bundle_dir)["splits"] == {"test": 22}
    assert read_record(jobs.cfg.bundle_dir)["splits"] == {"fit": 30, "dev": 8}  # untouched


def test_the_plan_prints_j11(ready, capsys):
    assert run(ready, "plan", "evaluate", "--predictor", "qwen3-4b", "--split", "test",
               "--scope", "measured") == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "PASS  J11  Test run: arm A0 at the frozen configuration" in out
    assert run(ready, "rehearse", "evaluate", "--predictor", "qwen3-4b", "--split", "test") == 2
    assert "a rehearsal never reads the test split" in capsys.readouterr().out
