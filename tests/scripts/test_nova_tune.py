"""Tests for scripts.nova_tune, with in-memory stand-ins for Bedrock and S3."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from qlora_finetuning.frontier.config import load_frontier_config
from qlora_finetuning.frontier.tuning_record import read_tuning_record, report_paths
from qlora_finetuning.settings import ConfigError
from scripts import nova_tune
from scripts.nova_tune import Clients
from tests.evaluation.conftest import ledger_records, make_context, run_folders
from tests.frontier.export_helpers import FakeS3
from tests.frontier.tuning_helpers import FakeBedrock, open_gate, put_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
CHANGING = [
    ("launch", "--model", "nova-lite"),
    ("deploy", "--model", "nova-lite"),
    ("stop", "--model", "nova-lite"),
    ("teardown", "--model", "nova-lite", "--all"),
    ("teardown", "--model", "nova-lite", "--deployment"),
]


@pytest.fixture
def context(tmp_path):
    return make_context(tmp_path)


@pytest.fixture
def cfg(tmp_path):
    return load_frontier_config(root=tmp_path)


@pytest.fixture
def s3():
    return FakeS3()


@pytest.fixture
def bedrock():
    return FakeBedrock()


@pytest.fixture
def gate(cfg, context, s3):
    return open_gate(cfg, context, s3)


def signed_out(cfg):
    raise ConfigError("the AWS sign-in is missing or has expired: run 'aws sso login'")


@pytest.fixture
def run(context, cfg, bedrock, s3):
    def call(*arguments, factory=None, sleep=None):
        return nova_tune.main(
            list(arguments),
            context=context,
            cfg=cfg,
            client_factory=factory or (lambda cfg: Clients(bedrock, s3)),
            sleep=sleep,
        )

    return call


# --------------------------------------------------------------------------- plan


def test_plan_shows_both_models_and_starts_nothing(run, gate, bedrock, tmp_path, capsys):
    assert run("plan") == 1  # nova-pro waits for nova-lite's bill (T9)
    out = capsys.readouterr().out
    assert out.index("nova-lite, attempt 1  (arm B1-Lite)") < out.index("nova-pro, attempt 1")
    assert f"job, tuned model, deployment: qlora-finqa-nova-lite-{gate['export_id']}-a1" in out
    assert "base model: amazon.nova-lite-v1:0:300k" in out
    assert "hyperparameters: epochCount 2, learningRate 0.00001, learningRateWarmupSteps 10" in out
    assert "estimated: 75,000 training tokens, $0.15 ($0.17 with the margin)" in out
    assert "9 passed, 0 failed, 0 skipped" in out  # nova-lite
    assert "FAIL  T9  Earlier bill seen: nova-lite has not been launched" in out
    assert out.rstrip().endswith("nothing was started")
    assert out.isascii() and str(tmp_path) not in out
    assert bedrock.changes() == []


def test_plan_for_one_model_exits_0_when_every_check_passes(run, gate, capsys):
    assert run("plan", "--model", "nova-lite") == 0
    out = capsys.readouterr().out
    assert "nova-pro" not in out and "FAIL" not in out and "SKIP" not in out


def test_plan_without_a_sign_in_skips_the_aws_checks(run, gate, capsys):
    assert run("plan", "--model", "nova-lite", factory=signed_out) == 1
    out = capsys.readouterr().out
    assert out.startswith("no sign-in: the AWS sign-in is missing or has expired")
    assert "the checks that need AWS are skipped" in out
    for check_id in ("T3", "T4", "T5", "T6"):
        assert re.search(rf"SKIP  {check_id}  .*needs a sign-in", out), check_id
    assert "5 passed, 0 failed, 4 skipped" in out


def test_plan_with_the_gate_closed_fails_t1(run, cfg, context, s3, capsys):
    open_gate(cfg, context, s3, staged=False)
    assert run("plan", "--model", "nova-lite") == 1
    out = capsys.readouterr().out
    assert "FAIL  T1  Gate open: the training files are not staged" in out
    assert "training file:   not staged" in out


def test_an_unknown_model_or_a_missing_gate_record_exits_2(run, gate, cfg, capsys):
    assert run("plan", "--model", "nova-max") == 2
    assert "CONFIG ERROR: config frontier.yaml: no model named 'nova-max'" in (
        capsys.readouterr().out
    )
    from qlora_finetuning.frontier.training_estimate import report_paths as gate_paths

    gate_paths(cfg.report_dir)[1].unlink()
    assert run("plan") == 2
    assert "CONFIG ERROR: no gate record" in capsys.readouterr().out


# --------------------------------------------------------------------------- --yes


@pytest.mark.parametrize("arguments", CHANGING, ids=lambda a: " ".join(a[:1] + a[3:]))
def test_no_command_changes_aws_without_yes(run, gate, bedrock, context, cfg, arguments, capsys):
    assert run(*arguments) == 1
    out = capsys.readouterr().out
    assert "NOT DONE: this would " in out and "with --yes" in out
    assert bedrock.changes() == []
    assert run_folders(context) == [] and ledger_records(context) == []
    assert not any(path.exists() for path in report_paths(cfg.report_dir))


def test_the_parser_gives_every_changing_command_a_yes_flag():
    parser = nova_tune._parser()
    for name in nova_tune.CHANGES_AWS:
        model = ["--model", "nova-lite"] + (["--all"] if name == "teardown" else [])
        assert parser.parse_args([name, *model]).yes is False
        assert parser.parse_args([name, *model, "--yes"]).yes is True
    for name in ("plan", "status"):
        assert not hasattr(parser.parse_args([name]), "yes")  # they cannot change anything


# --------------------------------------------------------------------------- the whole path


def test_the_owners_sequence_from_launch_to_two_active_deployments(
    run, gate, bedrock, s3, cfg, context, capsys
):
    lite = f"qlora-finqa-nova-lite-{gate['export_id']}-a1"
    pro = f"qlora-finqa-nova-pro-{gate['export_id']}-a1"

    assert run("launch", "--model", "nova-lite", "--yes") == 0
    out = capsys.readouterr().out
    assert f"LAUNCHED: {bedrock.jobs[lite]['jobArn']}" in out
    assert "tuning record: reports/frontier/nova_tuning.md  (commit it)" in out

    assert run("status") == 1  # still running
    assert "nova-lite a1: job InProgress; estimate $0.15" in capsys.readouterr().out

    # Nova Pro is held back until Nova Lite has completed and its bill has been read.
    assert run("launch", "--model", "nova-pro", "--yes") == 1
    out = capsys.readouterr().out
    assert out.count("REFUSED: a check failed, so nothing was launched") == 1
    assert "FAIL  T9  Earlier bill seen: the job of nova-lite is InProgress" in out
    assert list(bedrock.jobs) == [lite]

    bedrock.finish(lite)
    put_metrics(s3, read_tuning_record(cfg).latest("nova-lite"))
    assert run("status") == 0
    out = capsys.readouterr().out
    assert "nova-lite a1: job Completed; training loss 0.6000; validation loss 0.7000; " in out
    assert "not deployed; estimate $0.15" in out

    assert run("bill", "--model", "nova-lite", "--usd", "0.16", factory=signed_out) == 0
    assert "BILLED: nova-lite a1 is $0.16 in the ledger; the estimate was $0.15" in (
        capsys.readouterr().out
    )

    assert run("plan", "--model", "nova-pro") == 0
    assert "nova-lite was billed $0.16 against an estimate of $0.15 (1.07 times)" in (
        capsys.readouterr().out
    )
    assert run("launch", "--model", "nova-pro", "--yes") == 0
    capsys.readouterr()
    bedrock.finish(pro)
    assert run("status") == 0  # the loss curves of nova-pro are missing: noted, not fatal
    assert "note: the loss curves could not be read" in capsys.readouterr().out

    for model, name in (("nova-lite", lite), ("nova-pro", pro)):
        assert run("deploy", "--model", model, "--yes") == 0
        assert f"DEPLOYING: {bedrock.deployments[name]['customModelDeploymentArn']}" in (
            capsys.readouterr().out
        )
    assert run("status") == 1
    assert "deployment not Active yet: nova-lite, nova-pro" in capsys.readouterr().out
    for name in (lite, pro):
        bedrock.deployments[name]["status"] = "Active"
    assert run("status") == 0
    out = capsys.readouterr().out
    assert "deployment Active; billed $0.16" in out and out.isascii()

    record = read_tuning_record(cfg)
    assert [(a.model, a.job_status, a.deployment_status) for a in record.attempts] == [
        ("nova-lite", "Completed", "Active"),
        ("nova-pro", "Completed", "Active"),
    ]
    assert bedrock.changes() == [
        "create_model_customization_job",
        "create_model_customization_job",
        "create_custom_model_deployment",
        "create_custom_model_deployment",
    ]
    assert len(run_folders(context)) == 2  # one run per launch; status and deploy are none


def test_status_wait_prints_each_look_and_ends_when_the_job_does(run, gate, bedrock, capsys):
    run("launch", "--model", "nova-lite", "--yes")
    capsys.readouterr()
    name = next(iter(bedrock.jobs))
    naps = []

    def sleep(seconds):
        naps.append(seconds)
        bedrock.finish(name, status="Failed", failed_in="dataProcessingDetails")

    assert run("status", "--wait", sleep=sleep) == 1  # it ended, but it failed
    out = capsys.readouterr().out
    assert naps == [60] and out.count("nova-lite a1: job ") == 2
    assert "nova-lite a1: job Failed; failed in data processing; estimate $0.15" in out


def test_stop_and_teardown(run, gate, bedrock, s3, cfg, capsys):
    run("launch", "--model", "nova-lite", "--yes")
    name = next(iter(bedrock.jobs))
    assert run("stop", "--model", "nova-lite", "--yes") == 0
    assert f"STOPPING: {bedrock.jobs[name]['jobArn']}" in capsys.readouterr().out
    assert run("stop", "--model", "nova-lite", "--yes") == 1
    assert "REFUSED: the job of nova-lite is Stopping, not running" in capsys.readouterr().out

    bedrock.finish(name)  # pretend it completed after all, to have something to delete
    run("status")
    run("deploy", "--model", "nova-lite", "--yes")
    capsys.readouterr()
    assert run("teardown", "--model", "nova-lite", "--deployment", "--yes") == 0
    assert "DELETED: the deployment of nova-lite a1" in capsys.readouterr().out
    assert run("teardown", "--model", "nova-lite", "--all", "--yes") == 0
    assert "DELETED: the deployment and the tuned model of nova-lite a1" in capsys.readouterr().out
    assert run("teardown", "--model", "nova-lite", "--all", "--yes") == 1
    assert "REFUSED: nova-lite has nothing left to delete" in capsys.readouterr().out
    attempt = read_tuning_record(cfg).latest("nova-lite")
    assert attempt.deployment_deleted and attempt.model_deleted
    assert run("status") == 0
    assert "tuned model deleted" in capsys.readouterr().out


# --------------------------------------------------------------------------- sign-in


def test_status_with_nothing_launched_needs_no_sign_in(run, capsys):
    assert run("status", factory=signed_out) == 0
    assert capsys.readouterr().out.strip() == "no job has been launched"
    assert run("status") == 0
    assert capsys.readouterr().out.strip() == "no job has been launched"


@pytest.mark.parametrize(
    "arguments",
    [("launch", "--model", "nova-lite", "--yes"), ("deploy", "--model", "nova-lite", "--yes")],
)
def test_a_changing_command_without_a_sign_in_exits_2(run, gate, arguments, capsys):
    assert run(*arguments, factory=signed_out) == 2
    assert capsys.readouterr().out.startswith("CONFIG ERROR: the AWS sign-in is missing")


def test_status_of_a_launched_job_without_a_sign_in_exits_2(run, gate, capsys):
    run("launch", "--model", "nova-lite", "--yes")
    capsys.readouterr()
    assert run("status", factory=signed_out) == 2
    assert "CONFIG ERROR: the AWS sign-in" in capsys.readouterr().out


def test_bill_rejects_a_bad_amount_and_teardown_needs_to_be_told_what(run, gate, capsys):
    run("launch", "--model", "nova-lite", "--yes")
    capsys.readouterr()
    assert run("bill", "--model", "nova-lite", "--usd", "lots") == 2
    assert "CONFIG ERROR: the billed amount is not a number" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        run("teardown", "--model", "nova-lite", "--yes")  # neither --deployment nor --all
    with pytest.raises(SystemExit):
        run("launch", "--yes")  # no model named: nothing is ever launched by default


# --------------------------------------------------------------------------- guards


def test_nothing_in_the_project_can_create_provisioned_throughput():
    forbidden = "create_provisioned_model" + "_throughput"
    scanned = [
        *(REPO_ROOT / "qlora_finetuning").rglob("*.py"),
        *(REPO_ROOT / "scripts").glob("*.py"),
    ]
    assert len(scanned) > 40
    for path in scanned:
        text = path.read_text(encoding="utf-8").lower()
        assert forbidden not in text, path.name
        assert "createprovisionedmodelthroughput" not in text, path.name


def test_runs_as_a_module():
    """``bill`` asks AWS nothing, so this needs no network: an unknown model is exit 2."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.nova_tune", "bill", "--model", "nova-max", "--usd", "1"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "CONFIG ERROR: config frontier.yaml: no model named 'nova-max'" in result.stdout
    assert "Traceback" not in result.stderr


def test_running_by_file_path_gives_a_hint_not_a_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "nova_tune.py"), "plan"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "python -m scripts.nova_tune" in result.stderr
    assert "Traceback" not in result.stderr
