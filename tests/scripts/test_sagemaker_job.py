"""Tests for scripts.sagemaker_job and scripts.job_run, with stand-in clients and tasks."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from qlora_finetuning.local_qlora.job_record import read_job_record
from qlora_finetuning.local_qlora.training import TrainingError
from scripts import job_run, sagemaker_job
from tests.local_qlora.sagemaker_helpers import DIRTY, FakeLogs, FakeSts, make_jobs

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE = "qlora-smoke-01234567-a1"


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    return make_jobs(tmp_path, monkeypatch)


def run(jobs, *arguments):
    return sagemaker_job.main(list(arguments), jobs=jobs)


# --------------------------------------------------------------------------- sagemaker_job


def test_bundle_writes_fit_and_dev_and_says_so(jobs, capsys):
    (jobs.cfg.bundle_dir / "examples.jsonl").unlink()
    assert run(jobs, "bundle") == 0
    out = capsys.readouterr().out
    assert "bundle: fit 30, dev 8; 38 examples" in out and re.search(
        r"^sha256: [0-9a-f]{64}$", out, re.M
    )
    assert "no test row is in it" in out and "written to artifacts/bundle/" in out
    assert (jobs.cfg.bundle_dir / "examples.jsonl").is_file()
    assert out.isascii() and "zebrafish" not in out and jobs.sagemaker.calls == []


def test_plan_prints_the_job_its_checks_and_its_most(jobs, capsys):
    assert run(jobs, "plan", "smoke") == 0
    out = capsys.readouterr().out
    assert f"job {SMOKE}  (task smoke; arm none; commit 0123456789ab)" in out
    assert "instance    1 x ml.g5.xlarge, 60 GB volume" in out
    assert "runs        python -m scripts.job_run smoke --where sagemaker" in out
    assert "runtime     at most 60 minutes: at most $1.41 ($1.408 an hour)" in out
    assert out.count("PASS  J") == 11 and "11 passed, 0 failed, 0 skipped" in out
    assert out.rstrip().endswith("PLAN ONLY: nothing was created")
    assert jobs.sagemaker.created == [] and jobs.s3.puts == [] and out.isascii()


def test_a_plan_with_a_failed_check_exits_1(tmp_path, monkeypatch, capsys):
    jobs = make_jobs(tmp_path, monkeypatch, sts=FakeSts(signed_in=False), git=DIRTY)
    assert run(jobs, "plan", "smoke") == 1
    out = capsys.readouterr().out
    assert "FAIL  J1  Signed in: the AWS sign-in is missing or has expired" in out
    assert "FAIL  J2  A commit: uncommitted changes" in out and "SKIP  J5  Quota" in out
    assert "6 passed, 2 failed, 3 skipped" in out


def test_launch_needs_yes_and_then_says_how_to_follow_the_job(jobs, capsys):
    assert run(jobs, "launch", "smoke") == 2
    assert "CONFIG ERROR: launch creates a billed SageMaker job" in capsys.readouterr().out
    assert jobs.sagemaker.created == []

    assert run(jobs, "launch", "smoke", "--yes") == 0
    out = capsys.readouterr().out
    assert f"LAUNCHED: {SMOKE}" in out and "arn: arn:aws:sagemaker:" in out
    assert "follow it with: python -m scripts.sagemaker_job status" in out
    assert "git add reports/local_qlora ledger" in out
    assert jobs.sagemaker.created == [SMOKE]

    assert run(jobs, "launch", "smoke", "--yes") == 0  # the same job again: adopted
    assert f"ADOPTED (it already existed): {SMOKE}" in capsys.readouterr().out
    assert jobs.sagemaker.created == [SMOKE]


def test_a_launch_with_a_failed_check_exits_1_and_creates_nothing(tmp_path, monkeypatch, capsys):
    jobs = make_jobs(tmp_path, monkeypatch, git=DIRTY)
    assert run(jobs, "launch", "smoke", "--yes") == 1
    assert "NOT LAUNCHED: a check failed; nothing was created" in capsys.readouterr().out
    assert jobs.sagemaker.created == []


def test_status_logs_and_stop(tmp_path, monkeypatch, capsys):
    jobs = make_jobs(tmp_path, monkeypatch, logs=FakeLogs(["step 1", "café done"]))
    assert run(jobs, "status") == 2  # nothing launched yet
    assert "no job has been launched yet" in capsys.readouterr().out
    run(jobs, "launch", "smoke", "--yes")
    capsys.readouterr()

    assert run(jobs, "status") == 0
    out = capsys.readouterr().out
    assert "status      InProgress / Starting" in out and "pulled      no" in out
    assert run(jobs, "logs", "--tail", "2") == 0
    out = capsys.readouterr().out
    assert "step 1" in out and "caf? done" in out and out.isascii()

    assert run(jobs, "stop") == 2 and jobs.sagemaker.stopped == []
    capsys.readouterr()
    assert run(jobs, "stop", "--yes") == 0
    out = capsys.readouterr().out
    assert (
        "status      Stopped" in out and "billable    120 seconds: $0.05 at the list price" in out
    )


def test_pull_reports_what_came_home(jobs, capsys):
    run(jobs, "launch", "smoke", "--yes")
    assert run(jobs, "pull") == 0
    assert "NOT PULLED: the job is InProgress" in capsys.readouterr().out

    jobs.sagemaker.finish(SMOKE, billable=1500)
    assert run(jobs, "pull") == 1  # the job ended, but it left no result: its task never finished
    out = capsys.readouterr().out
    assert "billable    1,500 seconds: $0.59 at the list price" in out
    assert "billable seconds recorded" in out and "no result file" in out
    assert read_job_record(jobs.cfg.record_file)["jobs"][0]["pulled"] is True


def test_the_help_names_who_runs_what():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.sagemaker_job", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "THE OWNER: upload and create one billed job" in result.stdout
    assert "Traceback" not in result.stderr


def test_running_by_file_path_gives_a_hint_not_a_traceback(tmp_path):
    for name in ("sagemaker_job", "job_run"):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / f"{name}.py")],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert f"python -m scripts.{name}" in result.stderr and "Traceback" not in result.stderr


# --------------------------------------------------------------------------- job_run


def trainer_that(passes=True, error=None):
    def trainer(**options):
        if error:
            raise error
        return {
            "run_id": "20261120T100000Z-train-x-aaaa", "kind": "train", "continues": None,
            "steps": 918, "adapter": "20261120T100000Z-train-x-aaaa", "adapter_sha256": "a" * 64,
            "passed": passes, "checks": [{"id": "K5", "passed": passes}], "measurements": {},
        }  # fmt: skip

    return trainer


def test_job_run_prints_the_runs_and_exits_by_the_result(tmp_path, capsys):
    options = {"root": tmp_path, "check_pins": False}
    assert job_run.main(["train", "--where", "sagemaker"], trainer=trainer_that(), **options) == 0
    out = capsys.readouterr().out
    assert "task train  (job local-train; where sagemaker)" in out
    assert (
        "train: run 20261120T100000Z-train-x-aaaa; completed; 918 steps; checks failed: none" in out
    )
    assert "STATUS: completed" in out and "result: local-train/job_result.json" in out

    assert job_run.main(["train"], trainer=trainer_that(passes=False), **options) == 1
    assert "STATUS: failed" in capsys.readouterr().out

    error = TrainingError("OutOfMemoryError after 0 of 918 steps")
    error.run_id = "20261120T100000Z-train-x-bbbb"
    assert job_run.main(["train"], trainer=trainer_that(error=error), **options) == 1
    out = capsys.readouterr().out
    assert "ERROR: TrainingError: OutOfMemoryError after 0 of 918 steps" in out

    assert job_run.main(["evaluate", "--split", "dev"], **options) == 2
    out = capsys.readouterr().out
    assert "needs --predictor and --split" in out and "STATUS: config_error" in out
    assert out.isascii()


def test_job_run_refuses_an_unknown_task():
    with pytest.raises(SystemExit) as caught:
        job_run.main(["deploy"])
    assert caught.value.code == 2
