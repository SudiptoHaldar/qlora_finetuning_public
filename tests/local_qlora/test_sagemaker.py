"""Tests for qlora_finetuning.local_qlora.sagemaker: plan, launch, status, logs, stop.

Every client is a stand-in; nothing reaches AWS. ``pull`` is in ``test_sagemaker_pull.py``.
"""

from __future__ import annotations

import ast
import io
import json
import tarfile
from decimal import Decimal
from pathlib import Path

import pytest

from qlora_finetuning.local_qlora import sagemaker as sagemaker_module
from qlora_finetuning.local_qlora.job_record import read_job_record
from qlora_finetuning.local_qlora.sagemaker import (
    BOOTSTRAP,
    ENTRY_LINE,
    MAX_ARGUMENT,
    SOURCE_PATHS,
    JobError,
    JobOptions,
    build_source,
    job_definition,
    job_name,
    task_arguments,
    usd,
    working_tree_files,
)
from qlora_finetuning.settings import REPO_ROOT, ConfigError
from qlora_finetuning.tracking import Ledger, LedgerRecord, load_manifest
from tests.evaluation.conftest import ledger_records, run_folders

from .sagemaker_helpers import (
    COMMIT,
    DIRTY,
    SOURCE,
    FakeIam,
    FakeLogs,
    FakeQuotas,
    FakeS3,
    FakeSageMaker,
    FakeSts,
    client_error,
    make_jobs,
    with_cfg,
)

SMOKE = "qlora-smoke-01234567-a1"
PACKAGE = Path(sagemaker_module.__file__).parent


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    return make_jobs(tmp_path, monkeypatch)


def check(plan, check_id):
    return next(item for item in plan.checks if item["id"] == check_id)


def context_of(jobs):
    """What the tracking helpers of the evaluation tests take."""
    from types import SimpleNamespace

    return SimpleNamespace(tracking_cfg=jobs.tracking)


# --------------------------------------------------------------------------- names and arguments


def test_a_job_name_says_the_task_the_commit_and_the_attempt(jobs):
    assert job_name(jobs.cfg, "smoke", COMMIT) == SMOKE
    assert job_name(jobs.cfg, "evaluate", COMMIT, 3) == "qlora-evaluate-01234567-a3"
    with pytest.raises(ConfigError, match="is not a job name"):
        job_name(with_cfg(jobs, name_prefix="q" * 60).cfg, "smoke", COMMIT)
    with pytest.raises(ConfigError):
        job_name(jobs.cfg, "smoke", COMMIT, 0)


def test_task_arguments_are_what_job_run_takes():
    assert task_arguments(JobOptions("smoke")) == []
    options = JobOptions(
        "evaluate", predictor="qwen3-4b-tuned", adapter="run/checkpoint-300", split="dev",
        limit=24, resume="r1", scope="measured",
    )  # fmt: skip
    assert task_arguments(options) == [
        "--scope", "measured", "--resume", "r1", "--predictor", "qwen3-4b-tuned",
        "--adapter", "run/checkpoint-300", "--split", "dev", "--limit", "24",
    ]  # fmt: skip


def test_amounts_are_shown_in_cents_half_up():
    assert usd(Decimal("1.408")) == "$1.41" and usd(Decimal("0.825")) == "$0.83"
    assert usd(Decimal("0")) == "$0.00"


# --------------------------------------------------------------------------- the job definition


def definition(jobs, options=None, **changes):
    values = {
        "name": SMOKE,
        "workspace": SMOKE,
        "commit": COMMIT,
        "code_uri": "s3://bucket/sagemaker/code/c/",
        "bundle_uri": "s3://bucket/data/bundles/b/",
    }
    values.update(changes)
    return job_definition(jobs.cfg, jobs.names, options or JobOptions("smoke"), **values)


def test_the_job_definition(jobs):
    request = definition(jobs)
    assert request["TrainingJobName"] == SMOKE and request["RoleArn"] == jobs.names.role_arn
    algorithm = request["AlgorithmSpecification"]
    assert algorithm["TrainingImage"] == jobs.cfg.image and algorithm["TrainingInputMode"] == "File"
    assert algorithm["ContainerEntrypoint"] == ["bash", "-c"]
    assert algorithm["ContainerArguments"] == [ENTRY_LINE, "job", "smoke", "--where", "sagemaker"]
    assert request["ResourceConfig"] == {
        "InstanceType": "ml.g5.xlarge",
        "InstanceCount": 1,
        "VolumeSizeInGB": 60,
    }
    assert request["StoppingCondition"] == {"MaxRuntimeInSeconds": 3600}
    channels = {
        item["ChannelName"]: item["DataSource"]["S3DataSource"]
        for item in request["InputDataConfig"]
    }
    assert channels["code"]["S3Uri"] == "s3://bucket/sagemaker/code/c/"
    assert channels["bundle"]["S3Uri"] == "s3://bucket/data/bundles/b/"
    assert all(source["S3DataType"] == "S3Prefix" for source in channels.values())
    bucket = f"s3://{jobs.names.bucket}"
    assert request["CheckpointConfig"] == {
        "S3Uri": f"{bucket}/sagemaker/workspaces/{SMOKE}/",
        "LocalPath": "/opt/ml/checkpoints",
    }
    assert request["OutputDataConfig"] == {"S3OutputPath": f"{bucket}/sagemaker/output/"}
    assert {"Key": "project", "Value": "qlora-finetuning"} in request["Tags"]
    assert {"Key": "task", "Value": "smoke"} in request["Tags"]


def test_the_environment_names_the_commit_and_the_bundle_and_holds_no_secret(jobs):
    environment = definition(jobs)["Environment"]
    assert environment["QLORA_GIT_COMMIT"] == COMMIT and environment["QLORA_JOB_NAME"] == SMOKE
    assert environment["QLORA_DATA_BUNDLE"] == "/opt/ml/input/data/bundle"
    text = json.dumps(environment).lower()
    assert not any(word in text for word in ("secret", 'token"', "password", "akia"))
    assert all(len(value) <= 512 for value in environment.values()) and len(environment) <= 100


def test_entry_point_strings_fit_the_api(jobs):
    options = JobOptions(
        "evaluate", predictor="qwen3-4b-tuned", split="dev",
        adapter="20261120T101500Z-train-qwen3-4b-fit-1a2b/checkpoint-900",
    )  # fmt: skip
    arguments = definition(jobs, options)["AlgorithmSpecification"]["ContainerArguments"]
    assert all(len(item) <= MAX_ARGUMENT for item in arguments) and len(arguments) <= 100
    assert len(ENTRY_LINE) <= MAX_ARGUMENT and BOOTSTRAP in ENTRY_LINE
    assert "source.tar.gz" in ENTRY_LINE and ENTRY_LINE.endswith('"$@"')
    with pytest.raises(ConfigError, match="over 256 characters"):
        definition(jobs, JobOptions("evaluate", predictor="qwen3-4b", split="x" * 300))


def test_a_task_changes_the_runtime_and_a_workspace_can_be_shared(jobs):
    request = definition(jobs, JobOptions("train", resume="r1"), workspace="qlora-train-a-a1")
    assert request["StoppingCondition"] == {"MaxRuntimeInSeconds": 43200}
    assert request["CheckpointConfig"]["S3Uri"].endswith("/sagemaker/workspaces/qlora-train-a-a1/")
    assert request["AlgorithmSpecification"]["ContainerArguments"][-2:] == ["--resume", "r1"]


# --------------------------------------------------------------------------- the code


def test_the_bootstrap_script_has_unix_line_endings_and_keeps_the_cache_out():
    content = (REPO_ROOT / BOOTSTRAP).read_bytes()
    assert b"\r" not in content and content.startswith(b"#!/bin/bash\n")
    text = content.decode("ascii")
    assert "/opt/ml/checkpoints" in text and "requirements-train.txt" in text
    assert "scripts.job_run fetch-model" in text and 'scripts.job_run "$@"' in text
    # Only artifacts/ and ledger/ are linked into the workspace: not the model cache.
    assert text.count("ln -s") == 2 and ".cache" not in text


def test_the_source_is_the_commits_code_and_nothing_else():
    archive = tarfile.open(fileobj=io.BytesIO(build_source(REPO_ROOT, "HEAD")), mode="r:gz")
    names = [member.name for member in archive.getmembers() if member.isfile()]
    tops = {name.split("/")[0] for name in names}
    assert tops <= set(SOURCE_PATHS) and {"qlora_finetuning", "scripts", "config", "splits"} <= tops
    assert "requirements-train.txt" in names and "config/token_budget.yaml" in names
    assert not tops & {"artifacts", "ledger", "reqspecs", "reports", "tests", ".env", "venv"}
    for name in names:  # a shell script with Windows line endings does not run in the job
        if name.endswith(".sh"):
            assert b"\r" not in archive.extractfile(name).read(), name


def test_the_working_tree_files_are_the_same_paths_without_ignored_ones():
    names = working_tree_files(REPO_ROOT)
    assert "scripts/job_run.py" in names and "qlora_finetuning/local_qlora/job.py" in names
    assert all(name.split("/")[0] in SOURCE_PATHS for name in names)
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)


def test_an_unknown_commit_is_a_config_error():
    with pytest.raises(ConfigError, match="git archive failed"):
        build_source(REPO_ROOT, "0" * 40)


# --------------------------------------------------------------------------- the checks


def test_a_plan_with_everything_in_place_passes_and_creates_nothing(jobs):
    plan = jobs.plan(JobOptions("smoke"))
    assert plan.passed and [item["id"] for item in plan.checks] == [f"J{n}" for n in range(1, 12)]
    assert (plan.name, plan.workspace, plan.arm, plan.commit) == (SMOKE, SMOKE, "none", COMMIT)
    assert plan.max_cost == Decimal("1.408") and plan.cap == Decimal("100")
    assert check(plan, "J3")["detail"].startswith("fit 30, dev 8; 38 examples, no test id")
    assert (
        "$0.00 + at most $1.41 for this job = $1.41 of the $100.00 cap"
        in check(plan, "J7")["detail"]
    )
    assert "ml.g5.xlarge at $1.408 an hour" in check(plan, "J8")["detail"]
    assert plan.definition["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"].endswith(
        f"/sagemaker/code/{COMMIT}/"
    )
    assert jobs.sagemaker.created == [] and jobs.s3.puts == []
    assert run_folders(context_of(jobs)) == [] and not jobs.cfg.record_file.exists()


def test_signed_out_the_plan_fails_j1_and_skips_what_needs_aws(tmp_path, monkeypatch):
    jobs = make_jobs(tmp_path, monkeypatch, sts=FakeSts(signed_in=False))
    plan = jobs.plan(JobOptions("smoke"))
    assert not plan.passed
    assert check(plan, "J1")["status"] == "FAIL"
    assert "aws sso login --profile qlora" in check(plan, "J1")["detail"]
    assert [check(plan, name)["status"] for name in ("J4", "J5", "J6")] == ["SKIP"] * 3
    assert [check(plan, name)["status"] for name in ("J2", "J3", "J7", "J8")] == ["PASS"] * 4
    assert jobs.sagemaker.calls == []


@pytest.mark.parametrize(
    ("clients", "check_id", "message"),
    [
        ({"sts": FakeSts(account="111122223333")}, "J1", "signs in to account 111122223333"),
        ({"git": DIRTY}, "J2", "uncommitted changes (config/training.yaml)"),
        ({"iam": FakeIam(exists=False)}, "J4", "does not exist"),
        ({"iam": FakeIam(service="ec2.amazonaws.com")}, "J4", "does not trust sagemaker"),
        ({"quotas": FakeQuotas(value=0.0)}, "J5", "ml.g5.xlarge for training job usage: 0"),
        ({"quotas": FakeQuotas(name="other")}, "J5", "no quota called"),
    ],
)
def test_each_check_can_fail(tmp_path, monkeypatch, clients, check_id, message):
    jobs = make_jobs(tmp_path, monkeypatch, **clients)
    plan = jobs.plan(JobOptions("smoke"))
    failed = [item["id"] for item in plan.checks if item["status"] == "FAIL"]
    assert failed == [check_id] and message in check(plan, check_id)["detail"]


def test_a_role_that_trusts_several_services_passes(tmp_path, monkeypatch):
    iam = FakeIam(service=["ec2.amazonaws.com", "sagemaker.amazonaws.com"])
    assert make_jobs(tmp_path, monkeypatch, iam=iam).plan(JobOptions("smoke")).passed


def test_j3_fails_without_a_bundle_or_with_one_for_other_data(jobs):
    record = jobs.cfg.bundle_dir / "bundle.json"
    data = json.loads(record.read_text(encoding="utf-8"))
    record.write_text(json.dumps(data | {"manifest_hash": "f" * 64}), encoding="utf-8")
    plan = jobs.plan(JobOptions("smoke"))
    assert "another split manifest" in check(plan, "J3")["detail"] and not plan.passed
    record.unlink()
    assert "no data bundle" in check(jobs.plan(JobOptions("smoke")), "J3")["detail"]


def test_j6_fails_while_another_job_of_the_project_runs(jobs):
    jobs.sagemaker.jobs["qlora-train-aaaa-a1"] = {"TrainingJobStatus": "InProgress"}
    jobs.sagemaker.jobs["other-team-job"] = {"TrainingJobStatus": "InProgress"}
    plan = jobs.plan(JobOptions("smoke"))
    assert check(plan, "J6")["detail"] == "in progress: qlora-train-aaaa-a1"
    jobs.sagemaker.jobs["qlora-train-aaaa-a1"]["TrainingJobStatus"] = "Completed"
    assert jobs.plan(JobOptions("smoke")).passed


def billable(jobs, seconds, instance_type="ml.g5.xlarge"):
    Ledger(jobs.tracking).append(
        LedgerRecord(
            run_id="20261120T000000Z-other-x-0000",
            phase="train",
            path="A",
            arm="A1",
            where="sagemaker",
            source="aws_api",
            instance_type=instance_type,
            instance_count=1,
            billable_seconds=seconds,
        )  # fmt: skip
    )


def test_j7_counts_what_path_a_has_cost_and_what_is_still_out(jobs):
    billable(jobs, 36_000)  # ten hours: $14.08
    plan = jobs.plan(JobOptions("smoke"))
    assert plan.committed == Decimal("14.08")
    assert "$14.08 + at most $1.41 for this job = $15.49" in check(plan, "J7")["detail"]

    billable(jobs, 216_000)  # sixty hours more: $98.56 so far, and the cap is $100
    plan = jobs.plan(JobOptions("smoke"))
    assert plan.passed and "= $99.97 of the $100.00 cap" in check(plan, "J7")["detail"]
    # A training job may cost $16.90: that would go over.
    plan = jobs.plan(JobOptions("train"))
    assert not plan.passed and check(plan, "J7")["status"] == "FAIL"
    assert "= $115.46 of the $100.00 cap" in check(plan, "J7")["detail"]


def test_j7_counts_a_job_that_was_launched_and_not_pulled_at_its_maximum(jobs):
    _, job = jobs.launch(JobOptions("smoke"), yes=True)
    jobs.sagemaker.finish(job["name"])
    plan = jobs.plan(JobOptions("train"))
    assert plan.committed == Decimal("1.408")  # the smoke job's maximum: it is not pulled yet
    assert jobs.plan(JobOptions("smoke")).committed == Decimal("0")  # the same job, not twice


def test_j7_fails_on_seconds_it_cannot_price(jobs):
    billable(jobs, 100, instance_type="ml.p4d.24xlarge")
    plan = jobs.plan(JobOptions("smoke"))
    assert "no price for ml.p4d.24xlarge" in check(plan, "J7")["detail"] and not plan.passed


def test_the_plan_of_an_evaluate_job_takes_its_arm_from_the_predictor(jobs):
    options = JobOptions(
        "evaluate",
        predictor="qwen3-4b-tuned",
        adapter="r/checkpoint-3",
        split="dev",
        workspace="qlora-train-01234567-a1",
    )
    plan = jobs.plan(options)
    assert plan.arm == "A1" and plan.max_cost == Decimal("1.408") * 4
    assert jobs.plan(JobOptions("evaluate", predictor="qwen3-4b", split="dev")).arm == "A0"
    with pytest.raises(ConfigError, match="needs --predictor"):
        jobs.plan(JobOptions("evaluate", split="dev"))
    with pytest.raises(ConfigError, match="unknown task"):
        jobs.plan(JobOptions("deploy"))


# --------------------------------------------------------------------------- launching


def test_launch_is_the_owners_and_needs_yes(jobs):
    with pytest.raises(ConfigError, match="run by the owner, with --yes"):
        jobs.launch(JobOptions("smoke"))
    assert jobs.sagemaker.calls == [] and jobs.s3.puts == []


def test_launch_uploads_creates_one_job_and_writes_the_record(jobs):
    plan, job = jobs.launch(JobOptions("smoke"), yes=True)
    assert plan.passed and job["name"] == SMOKE
    bucket_keys = jobs.s3.puts
    assert bucket_keys[0] == f"sagemaker/code/{COMMIT}/source.tar.gz"
    assert jobs.s3.objects[bucket_keys[0]] == SOURCE
    assert sorted(key.rsplit("/", 1)[-1] for key in bucket_keys[1:]) == [
        "bundle.json",
        "examples.jsonl",
    ]
    assert all(key.startswith("data/bundles/") for key in bucket_keys[1:])
    assert jobs.sagemaker.created == [SMOKE]
    request = jobs.sagemaker.jobs[SMOKE]["request"]
    assert request == plan.definition  # what was planned is what was created

    (stored,) = read_job_record(jobs.cfg.record_file)["jobs"]
    assert stored == job
    assert (stored["arn"], stored["status"], stored["adopted"]) == (
        f"arn:aws:sagemaker:us-east-1:123456789012:training-job/{SMOKE}",
        "InProgress",
        False,
    )
    assert (stored["task"], stored["arm"], stored["phase"], stored["commit"]) == (
        "smoke", "none", "other", COMMIT,
    )  # fmt: skip
    assert (stored["instance_type"], stored["max_runtime_seconds"]) == ("ml.g5.xlarge", 3600)
    assert stored["max_cost_usd"] == "1.408" and stored["pulled"] is False
    assert stored["workspace_uri"].endswith(f"/sagemaker/workspaces/{SMOKE}/")
    assert stored["bundle_uri"].startswith(f"s3://{jobs.names.bucket}/data/bundles/")

    # The launch is a run in the registry, with no ledger record yet: nothing is known to bill.
    manifest = load_manifest(stored["launch_run_id"], jobs.tracking)
    assert (manifest["kind"], manifest["where"], manifest["path"], manifest["arm"]) == (
        "other", "sagemaker", "A", "none",
    )  # fmt: skip
    assert manifest["outputs"]["job"] == SMOKE and manifest["status"] == "completed"
    assert ledger_records(context_of(jobs)) == []
    assert "zebrafish" not in jobs.cfg.record_file.read_text(encoding="utf-8")


def test_a_failed_check_launches_nothing(tmp_path, monkeypatch):
    jobs = make_jobs(tmp_path, monkeypatch, git=DIRTY)
    plan, job = jobs.launch(JobOptions("smoke"), yes=True)
    assert job is None and not plan.passed
    assert jobs.sagemaker.created == [] and jobs.s3.puts == []
    assert not jobs.cfg.record_file.exists() and run_folders(context_of(jobs)) == []


def test_a_job_that_already_exists_is_adopted_not_created_twice(jobs):
    """The reply to "create the job" was lost: launching again must not start a second job."""
    first_plan, first = jobs.launch(JobOptions("smoke"), yes=True)
    jobs.cfg.record_file.unlink()  # as if the first launch died before it wrote the record
    _, again = jobs.launch(JobOptions("smoke"), yes=True)
    assert jobs.sagemaker.created == [SMOKE]  # once
    assert again["adopted"] is True and again["arn"] == first["arn"]
    assert [job["name"] for job in read_job_record(jobs.cfg.record_file)["jobs"]] == [SMOKE]
    assert jobs.s3.puts.count(f"sagemaker/code/{COMMIT}/source.tar.gz") == 1  # equal: not re-sent


def test_a_new_attempt_is_a_new_job(jobs):
    _, first = jobs.launch(JobOptions("smoke"), yes=True)
    jobs.sagemaker.finish(first["name"], "Failed")
    _, second = jobs.launch(JobOptions("smoke", attempt=2), yes=True)
    assert second["name"] == "qlora-smoke-01234567-a2" and second["adopted"] is False
    assert len(read_job_record(jobs.cfg.record_file)["jobs"]) == 2


def test_other_content_under_the_same_key_stops_the_launch(jobs):
    jobs.s3.objects[f"sagemaker/code/{COMMIT}/source.tar.gz"] = b"something else"
    with pytest.raises(ConfigError, match="already exists with other content"):
        jobs.launch(JobOptions("smoke"), yes=True)
    assert jobs.sagemaker.created == []


def test_aws_refusing_the_job_is_reported_with_its_code(jobs, monkeypatch):
    def refuse(**request):
        raise client_error("ResourceLimitExceeded", "CreateTrainingJob")

    monkeypatch.setattr(jobs.sagemaker, "create_training_job", refuse)
    with pytest.raises(JobError, match="AWS answered ResourceLimitExceeded when asked to create"):
        jobs.launch(JobOptions("smoke"), yes=True)
    assert not jobs.cfg.record_file.exists()


def test_an_expired_sign_in_during_a_call_gives_the_sign_in_message(jobs, monkeypatch):
    def expired(**request):
        raise client_error("ExpiredTokenException", "DescribeTrainingJob")

    jobs.launch(JobOptions("smoke"), yes=True)
    monkeypatch.setattr(jobs.sagemaker, "describe_training_job", expired)
    with pytest.raises(ConfigError, match="aws sso login --profile qlora"):
        jobs.status()


# --------------------------------------------------------------------------- following a job


def test_status_reads_the_job_and_updates_the_record(jobs):
    jobs.launch(JobOptions("smoke"), yes=True)
    jobs.sagemaker.jobs[SMOKE].update(SecondaryStatus="Training")
    assert jobs.status()["secondary_status"] == "Training"
    jobs.sagemaker.finish(
        SMOKE, "Failed", billable=600, FailureReason="AlgorithmError: exit 1\nmore"
    )
    job = jobs.status(SMOKE)
    assert (job["status"], job["billable_seconds"], job["training_seconds"]) == ("Failed", 600, 600)
    assert job["failure_reason"] == "AlgorithmError: exit 1"  # the first line only
    assert job["training_started_at"].startswith("2026-11-20T10:00")
    assert read_job_record(jobs.cfg.record_file)["jobs"][0]["status"] == "Failed"
    with pytest.raises(ConfigError, match="no job called nope"):
        jobs.status("nope")


def test_status_before_any_launch(jobs):
    with pytest.raises(ConfigError, match="no job has been launched yet"):
        jobs.status()


def test_logs_returns_the_last_lines(tmp_path, monkeypatch):
    jobs = make_jobs(tmp_path, monkeypatch, logs=FakeLogs([f"line {n}" for n in range(10)]))
    jobs.launch(JobOptions("smoke"), yes=True)
    assert jobs.logs(tail=3) == ["line 7", "line 8", "line 9"]
    quiet = make_jobs(tmp_path / "quiet", monkeypatch, logs=FakeLogs(None))
    quiet.launch(JobOptions("smoke"), yes=True)
    assert quiet.logs() == []  # the container has not started


def test_stop_is_the_owners_and_needs_yes(jobs):
    jobs.launch(JobOptions("smoke"), yes=True)
    with pytest.raises(ConfigError, match="run by the owner, with --yes"):
        jobs.stop()
    assert jobs.sagemaker.stopped == []
    job = jobs.stop(yes=True)
    assert jobs.sagemaker.stopped == [SMOKE] and job["status"] == "Stopped"


# --------------------------------------------------------------------------- guards


def imports_of(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_only_the_launcher_and_staging_use_botocore_and_nothing_names_boto3():
    for path in sorted(PACKAGE.glob("*.py")):
        roots = {name.split(".")[0] for name in imports_of(path)}
        assert "boto3" not in roots, path.name  # sessions come from qlora_finetuning.aws_common
        if path.name not in ("sagemaker.py", "staging.py"):
            assert "botocore" not in roots, path.name
        assert not any(name.startswith("qlora_finetuning.frontier") for name in imports_of(path))


def test_nothing_uploads_anything_but_the_code_and_the_bundle():
    source = (PACKAGE / "sagemaker.py").read_text(encoding="utf-8")
    # launch: the archive, and the bundle's two files. keep: the adapter's files, the record.
    assert source.count("staging.put_checked(") == 4
    assert not any(item in SOURCE_PATHS for item in ("artifacts", "ledger", "tests", "reqspecs"))


def test_faked_s3_pages(tmp_path):
    from qlora_finetuning.local_qlora.staging import list_keys

    s3 = FakeS3(page_size=2)
    for number in range(5):
        s3.objects[f"p/{number}"] = b"x" * number
    s3.objects["q/other"] = b"y"
    assert list_keys(s3, "bucket", "p/", profile="qlora") == {f"p/{n}": n for n in range(5)}
    assert FakeSageMaker().calls == []
