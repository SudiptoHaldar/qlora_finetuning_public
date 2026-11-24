"""Tests for what req_003 v3.04 adds to the launcher: the estimate and check J9, the score
job, ``--adapter selected``, ``--max-runtime``, ``pull --checkpoint``, and ``keep``.

Nothing here reaches AWS: the clients are stand-ins.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from qlora_finetuning.local_qlora.adapters import ADAPTER_WEIGHTS
from qlora_finetuning.local_qlora.job_record import read_job_record
from qlora_finetuning.local_qlora.sagemaker import JobOptions, task_arguments
from qlora_finetuning.settings import ConfigError
from scripts import sagemaker_job

from .sagemaker_helpers import make_jobs, workspace_key
from .selection_helpers import TRAIN_JOB, scored_world

SMOKE = "qlora-smoke-01234567-a1"
RUN = "20261120T100000Z-train-qwen3-4b-fit-aaaa"
RIGHT = {2: 3, 4: 5, 6: 5}


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    return make_jobs(tmp_path, monkeypatch)


def check(plan, check_id):
    return next(item for item in plan.checks if item["id"] == check_id)


def measure(jobs, per_sequence=0.9):
    """Run and pull a smoke job that measured: 1,500 s billed, 236 s of them work."""
    _, job = jobs.launch(JobOptions("smoke"), yes=True)
    jobs.sagemaker.finish(job["name"], billable=1500)
    runs = [
        {"role": "train", "run_id": "t", "status": "completed",
         "trainer": {"seconds_per_sequence": per_sequence, "train_seconds": 170}},
        {"role": "eval-tuned", "run_id": "a", "status": "completed",
         "backend": {"generate_seconds": 30, "prompts": 24}},
        {"role": "eval-untuned", "run_id": "b", "status": "completed",
         "backend": {"generate_seconds": 36, "prompts": 24}},
    ]  # fmt: skip
    content = {"status": "completed", "task": "smoke", "runs": runs}
    key = workspace_key(jobs, job, f"artifacts/jobs/{job['name']}/job_result.json")
    jobs.s3.objects[key] = json.dumps(content).encode()
    assert jobs.pull()["pulled"]
    return job


# --------------------------------------------------------------------------- the estimate, J9


def test_the_smoke_job_needs_no_estimate_and_the_others_need_a_measurement(jobs):
    plan = jobs.plan(JobOptions("smoke"))
    assert plan.estimate is None and check(plan, "J9")["status"] == "PASS"
    assert "the smoke job is what measures" in check(plan, "J9")["detail"]

    plan = jobs.plan(JobOptions("train"))
    assert plan.estimate is None and not plan.passed
    assert check(plan, "J9")["status"] == "FAIL"
    assert "run and pull the smoke job first" in check(plan, "J9")["detail"]
    _, job = jobs.launch(JobOptions("train"), yes=True)
    assert job is None and jobs.sagemaker.created == []  # an unestimated job is not launched


def test_after_a_pulled_smoke_job_the_plan_says_hours_and_dollars(jobs):
    measure(jobs)
    plan = jobs.plan(JobOptions("train"))
    assert plan.passed and check(plan, "J9")["status"] == "PASS"
    estimate = plan.estimate
    assert estimate.seconds == 1345  # the bundle's 30 fit records x 3 epochs x 0.9 s + 1,264 s
    assert estimate.cost == jobs.cfg.cost_of(1345) and estimate.proposed_max_runtime == 7200
    detail = check(plan, "J9")["detail"]
    assert "90 training sequences at 0.90 s" in detail and f"measured by {SMOKE}" in detail
    assert "about $0.53" in detail

    score = jobs.plan(JobOptions("score", run=RUN, workspace=TRAIN_JOB))
    assert score.estimate.task == "score" and "8 replies from the untuned model" in (
        score.estimate.detail
    )
    tuned = JobOptions(
        "evaluate", predictor="qwen3-4b-tuned", adapter="r/checkpoint-3", split="dev", limit=4,
        workspace=TRAIN_JOB,
    )  # fmt: skip
    assert jobs.plan(tuned).estimate.seconds == 1269  # 4 replies at 1.25 s + 1,264 s
    untuned = JobOptions("evaluate", predictor="qwen3-4b", split="dev")
    assert jobs.plan(untuned).estimate.seconds == 1276  # 8 replies at 1.5 s + 1,264 s


def test_an_estimate_over_the_limit_fails_j9_and_names_the_limit_to_ask_for(jobs):
    measure(jobs, per_sequence=500)
    plan = jobs.plan(JobOptions("train"))
    assert plan.estimate.seconds == 46264 and plan.max_runtime == 43200
    assert check(plan, "J9")["status"] == "FAIL" and not plan.passed
    assert "over this job's limit of 43,200 s: launch with --max-runtime 69420" in (
        check(plan, "J9")["detail"]
    )

    plan = jobs.plan(JobOptions("train", max_runtime=69420))
    assert plan.passed and plan.max_runtime == 69420
    assert plan.max_cost == jobs.cfg.max_cost("train", 69420) == Decimal("1.408") * 69420 / 3600
    assert plan.definition["StoppingCondition"] == {"MaxRuntimeInSeconds": 69420}
    assert f"at most {'$27.15'} for this job" in check(plan, "J7")["detail"]

    _, job = jobs.launch(JobOptions("train", max_runtime=69420), yes=True)
    request = jobs.sagemaker.jobs[job["name"]]["request"]
    assert request["StoppingCondition"] == {"MaxRuntimeInSeconds": 69420}


@pytest.mark.parametrize("seconds", [0, 599, -1])
def test_a_max_runtime_under_ten_minutes_is_refused(jobs, seconds):
    with pytest.raises(ConfigError, match="--max-runtime is in seconds and must be at least 600"):
        jobs.plan(JobOptions("smoke", max_runtime=seconds or -5))


# --------------------------------------------------------------------------- the score job


def test_a_score_job_names_the_run_and_the_workspace_of_the_job_that_trained_it(jobs):
    options = JobOptions("score", run=RUN, workspace=TRAIN_JOB, scope="measured")
    assert task_arguments(options) == ["--scope", "measured", "--run", RUN]
    plan = jobs.plan(options)
    assert (plan.task, plan.arm, plan.workspace) == ("score", "A1", TRAIN_JOB)
    assert plan.max_runtime == 21600 and plan.max_cost == Decimal("1.408") * 6
    arguments = plan.definition["AlgorithmSpecification"]["ContainerArguments"]
    assert arguments[2:] == ["score", "--where", "sagemaker", "--scope", "measured", "--run", RUN]
    assert plan.definition["CheckpointConfig"]["S3Uri"].endswith(f"/workspaces/{TRAIN_JOB}/")


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (JobOptions("score", workspace=TRAIN_JOB), "a score job needs --run"),
        (JobOptions("score", run=RUN), "add --workspace"),
        (JobOptions("train", resume="r1"), "add --workspace"),
        (
            JobOptions("evaluate", predictor="qwen3-4b-tuned", adapter="r", split="dev"),
            "add --workspace",
        ),
    ],
)
def test_a_job_that_continues_anothers_work_must_say_where_it_is(jobs, options, message):
    with pytest.raises(ConfigError, match=message):
        jobs.plan(options)
    with pytest.raises(ConfigError, match=message):
        jobs.launch(options, yes=True)
    assert jobs.sagemaker.created == [] and jobs.s3.puts == []


# --------------------------------------------------------------------------- --adapter selected


def test_selected_becomes_the_checkpoint_of_the_selection_record(tmp_path, monkeypatch):
    world = scored_world(tmp_path, right=RIGHT)
    jobs = make_jobs(tmp_path, monkeypatch)
    options = JobOptions(
        "evaluate", predictor="qwen3-4b-tuned", adapter="selected", split="dev",
        workspace=TRAIN_JOB,
    )  # fmt: skip
    with pytest.raises(ConfigError, match="no checkpoint has been selected yet"):
        jobs.plan(options)
    world.select()
    plan = jobs.plan(options)
    assert plan.options.adapter == f"{world.run}/checkpoint-4"
    arguments = plan.definition["AlgorithmSpecification"]["ContainerArguments"]
    assert arguments[arguments.index("--adapter") + 1] == f"{world.run}/checkpoint-4"
    assert "selected" not in arguments


# --------------------------------------------------------------------------- pull --checkpoint


def fill_checkpoints(jobs, job):
    for step in (4, 8):
        folder = f"artifacts/runs/{RUN}/checkpoints/checkpoint-{step}"
        for name, content in (
            ("adapter_model.safetensors", f"weights {step}".encode()),
            ("adapter_config.json", b'{"r": 16}'),
            ("optimizer.pt", b"o" * 500),  # for resuming there, not for use here
        ):
            jobs.s3.objects[workspace_key(jobs, job, f"{folder}/{name}")] = content
    jobs.s3.objects[workspace_key(jobs, job, f"artifacts/runs/{RUN}/training.json")] = b"{}"


def test_pull_checkpoint_downloads_one_adapter_and_changes_nothing(jobs, tmp_path):
    _, job = jobs.launch(JobOptions("smoke"), yes=True)
    fill_checkpoints(jobs, job)
    before = (dict(jobs.s3.objects), list(jobs.s3.puts), read_job_record(jobs.cfg.record_file))
    fetched = jobs.pull_checkpoint(8)
    assert fetched == {"job": SMOKE, "files": 3, "adapters": [f"{RUN}/checkpoint-8"]}
    folder = tmp_path / "artifacts" / "runs" / RUN
    assert (folder / "checkpoints" / "checkpoint-8" / ADAPTER_WEIGHTS).read_bytes() == b"weights 8"
    assert (folder / "training.json").is_file()
    assert not (folder / "checkpoints" / "checkpoint-8" / "optimizer.pt").exists()
    assert not (folder / "checkpoints" / "checkpoint-4").exists()
    assert before == (
        dict(jobs.s3.objects), list(jobs.s3.puts), read_job_record(jobs.cfg.record_file)
    )  # fmt: skip
    assert jobs.pull_checkpoint(8, SMOKE)["files"] == 0  # already here, and the same
    with pytest.raises(ConfigError, match="has no checkpoint at step 12"):
        jobs.pull_checkpoint(12)


# --------------------------------------------------------------------------- keep


def test_keep_is_the_owners_and_copies_the_selected_adapter_and_its_record(tmp_path, monkeypatch):
    world = scored_world(tmp_path, right=RIGHT)
    jobs = make_jobs(tmp_path, monkeypatch)
    with pytest.raises(ConfigError, match="run by the owner, with --yes"):
        jobs.keep()
    with pytest.raises(ConfigError, match="no checkpoint has been selected yet"):
        jobs.keep(yes=True)
    record = world.select()["record"]
    assert jobs.s3.puts == []

    kept = jobs.keep(yes=True)
    prefix = f"{jobs.names.adapters_prefix}{world.run}/checkpoint-4/"
    assert sorted(jobs.s3.puts) == sorted(
        prefix + name
        for name in ("adapter_model.safetensors", "adapter_config.json", "adapter_selection.json")
    )
    assert jobs.s3.objects[prefix + ADAPTER_WEIGHTS] == (
        world.checkpoint(4) / ADAPTER_WEIGHTS
    ).read_bytes()
    assert kept["adapter"] == f"{world.run}/checkpoint-4"
    assert kept["adapter_sha256"] == record["selected"]["adapter_sha256"]
    assert kept["uri"] == f"s3://{jobs.names.bucket}/{prefix}"
    stored = read_job_record(jobs.cfg.record_file)
    assert [item["adapter"] for item in stored["kept"]] == [kept["adapter"]]
    assert len(stored["jobs"]) == 2  # the jobs are as they were

    jobs.keep(yes=True)  # again: the same objects, one entry
    assert len(read_job_record(jobs.cfg.record_file)["kept"]) == 1
    assert not any(key.startswith(prefix) and "predictions" in key for key in jobs.s3.objects)


def test_keep_refuses_weights_that_are_not_the_selected_ones_or_not_here(tmp_path, monkeypatch):
    world = scored_world(tmp_path, right=RIGHT)
    jobs = make_jobs(tmp_path, monkeypatch)
    world.select()
    weights = world.checkpoint(4) / ADAPTER_WEIGHTS
    weights.write_bytes(b"other weights")
    with pytest.raises(ConfigError, match="are not the ones the selection record names"):
        jobs.keep(yes=True)
    weights.unlink()
    with pytest.raises(ConfigError, match=r"pull it first: pull --checkpoint 4"):
        jobs.keep(yes=True)
    assert jobs.s3.puts == []


# --------------------------------------------------------------------------- the script


def run(jobs, *arguments):
    return sagemaker_job.main(list(arguments), jobs=jobs)


def test_the_plan_prints_the_estimate_and_the_limit_to_ask_for(jobs, capsys):
    assert run(jobs, "plan", "train") == 1
    out = capsys.readouterr().out
    assert "FAIL  J9  Estimated: no pulled job on ml.g5.xlarge has measured" in out
    assert "estimate  " not in out

    measure(jobs)
    capsys.readouterr()
    assert run(jobs, "plan", "train") == 0
    out = capsys.readouterr().out
    assert "estimate    90 training sequences at 0.90 s" in out and "about $0.53" in out
    assert "proposed    --max-runtime 7200" in out and "PASS  J9  Estimated" in out
    assert run(jobs, "plan", "train", "--max-runtime", "7200") == 0
    assert "runtime     at most 120 minutes: at most $2.82" in capsys.readouterr().out


def test_score_pull_checkpoint_and_keep_from_the_command_line(tmp_path, monkeypatch, capsys):
    world = scored_world(tmp_path, right=RIGHT)
    jobs = make_jobs(tmp_path, monkeypatch)
    assert run(jobs, "plan", "score", "--run", world.run) == 2
    assert "CONFIG ERROR: this job continues an earlier job's work" in capsys.readouterr().out
    run(jobs, "plan", "score", "--run", world.run, "--workspace", TRAIN_JOB)
    assert f"job_run score --where sagemaker --run {world.run}" in (
        capsys.readouterr().out
    )

    assert run(jobs, "keep") == 2
    assert "CONFIG ERROR: keep writes the selected adapter to S3" in capsys.readouterr().out
    world.select()
    assert run(jobs, "keep", "--yes") == 0
    out = capsys.readouterr().out
    assert f"kept: {world.run}/checkpoint-4  (sha256 " in out and "at:   s3://" in out

    assert run(jobs, "pull", "--checkpoint", "4") == 2  # this workspace holds no checkpoint
    assert "has no checkpoint at step 4" in capsys.readouterr().out
