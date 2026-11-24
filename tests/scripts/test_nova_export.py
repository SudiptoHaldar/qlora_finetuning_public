"""Tests for scripts.nova_export, on the invented split with an in-memory stand-in for S3."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from qlora_finetuning.frontier.config import load_frontier_config
from qlora_finetuning.frontier.training_estimate import read_gate_record, report_paths
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking import load_manifest
from scripts import nova_export
from tests.evaluation.conftest import ledger_records, make_context, run_folders
from tests.frontier.export_helpers import FakeS3, volumes_for, write_baseline_report

REPO_ROOT = Path(__file__).resolve().parents[2]
BUCKET = "qlora-sagemaker-123456789012"


@pytest.fixture
def context(tmp_path):
    return make_context(tmp_path)


@pytest.fixture
def cfg(tmp_path):
    return load_frontier_config(root=tmp_path)


def no_client(cfg):
    raise AssertionError("an AWS client was asked for")


def run(context, cfg, *arguments, s3=None, volumes=None, factory=None):
    if factory is None:
        factory = no_client if s3 is None else (lambda cfg: (s3, BUCKET, "qlora"))
    return nova_export.main(
        list(arguments),
        context=context,
        cfg=cfg,
        volumes=volumes or volumes_for(context),
        s3_factory=factory,
    )


def test_without_a_measured_ratio_the_gate_is_closed_and_the_exit_code_is_1(
    context, cfg, tmp_path, capsys
):
    assert run(context, cfg) == 1
    out = capsys.readouterr().out
    assert "training: artifacts/bedrock/" in out and "30 records" in out
    for check_id in ("X1", "X2", "X3", "X4", "X5", "X6", "X7", "G2"):
        assert f"PASS  {check_id}  " in out, check_id
    assert "FAIL  G1  Ratio measured: not measured: no nova_baseline_report.json" in out
    assert "8 passed, 1 failed" in out
    assert "Nova tokens per Qwen token: 1.5 (NOT MEASURED, assumed)" in out
    assert "training, 2 epochs, both models: $0.90 ($0.99 with the 10% margin)" in out
    assert "training by epoch count: 1: $0.45, 2: $0.90, 3: $1.35, 4: $1.80, 5: $2.25" in out
    assert "of a cap of $200.00" in out
    assert "gate record: reports/frontier/nova_training_gate.md" in out
    assert out.rstrip().endswith("GATE CLOSED: a check failed; no fine-tuning job may be started")
    assert out.isascii() and str(tmp_path) not in out
    for word in ("zebrafish", "xylophone", "subtract", "PROGRAM"):
        assert word not in out, word

    record = read_gate_record(cfg)
    assert record["passed"] is False and record["staged"] is None
    assert record["files"]["training"]["records"] == 30


def test_with_a_baseline_report_the_gate_opens_and_the_exit_code_is_0(context, cfg, capsys):
    write_baseline_report(cfg, context)
    assert run(context, cfg) == 0
    out = capsys.readouterr().out
    assert "PASS  G1  Ratio measured: Nova counts 1.25 tokens per Qwen token" in out
    assert "9 passed, 0 failed" in out and "FAIL" not in out
    assert "Nova tokens per Qwen token: 1.25 (measured)" in out
    assert out.rstrip().endswith("GATE OPEN: every check passed")
    assert read_gate_record(cfg)["passed"] is True


def test_the_first_export_is_one_run_and_a_repeat_is_none(context, cfg, capsys):
    run(context, cfg)
    first = capsys.readouterr().out
    (run_id,) = run_folders(context)
    assert f"run {run_id}" in first and "-export-nova-export-" in run_id
    manifest = load_manifest(run_id, context.tracking_cfg)
    assert (manifest["kind"], manifest["path"], manifest["arm"]) == ("export", "B", "none")
    assert {"frontier", "pricing", "prompt"} <= set(manifest["configs"])
    assert set(manifest["outputs"]) == {"training_sha256", "validation_sha256"}
    (record,) = ledger_records(context)
    assert (record.phase, record.examples, record.source) == ("other", 38, "measured")
    assert record.input_tokens is None and record.billed_amount_usd is None

    run(context, cfg)  # the same files: nothing new happened
    assert "run " not in capsys.readouterr().out.replace("run the v4.01", "")
    assert run_folders(context) == [run_id] and len(ledger_records(context)) == 1


def test_check_writes_no_record_and_reports_a_stale_one(context, cfg, capsys):
    write_baseline_report(cfg, context)
    assert run(context, cfg, "--check") == 1
    out = capsys.readouterr().out
    assert "STALE: nova_training_gate.md, nova_training_gate.json" in out
    assert not any(path.exists() for path in report_paths(cfg.report_dir))
    assert run_folders(context) == []

    # The files were already written by the check; the first record is still one run.
    assert run(context, cfg) == 0
    assert "\nrun " in capsys.readouterr().out and len(run_folders(context)) == 1
    assert run(context, cfg, "--check") == 0
    assert "the gate record is current" in capsys.readouterr().out

    cheaper = dataclasses.replace(cfg, epoch_count=1)  # the config changed since
    assert run(context, cheaper, "--check") == 1
    assert "STALE" in capsys.readouterr().out


def test_upload_stages_both_files_and_writes_the_locations_into_the_record(
    context, cfg, capsys
):
    write_baseline_report(cfg, context)
    s3 = FakeS3()
    assert run(context, cfg, "--upload", s3=s3) == 0
    out = capsys.readouterr().out
    record = read_gate_record(cfg)
    prefix = f"s3://{BUCKET}/data/bedrock/{record['export_id']}/"
    assert f"staged training: {prefix}finqa_fit.jsonl" in out
    assert f"staged validation: {prefix}finqa_dev.jsonl" in out
    assert record["staged"]["training"] == {
        "s3_uri": prefix + "finqa_fit.jsonl",
        "bytes": record["files"]["training"]["bytes"],
        "sha256": record["files"]["training"]["sha256"],
    }
    assert s3.count("put_object") == 2
    (run_id,) = run_folders(context)
    outputs = load_manifest(run_id, context.tracking_cfg)["outputs"]
    assert outputs["training_s3_uri"] == prefix + "finqa_fit.jsonl"

    # A later plain run keeps the locations, asks AWS nothing, and the record stays current.
    assert run(context, cfg) == 0
    assert read_gate_record(cfg)["staged"] == record["staged"]
    assert run(context, cfg, "--check") == 0
    assert run_folders(context) == [run_id]


def test_upload_does_not_need_the_gate_to_be_open(context, cfg, capsys):
    s3 = FakeS3()
    assert run(context, cfg, "--upload", s3=s3) == 1  # G1 fails: nothing was measured
    assert "staged training: s3://" in capsys.readouterr().out
    record = read_gate_record(cfg)
    assert record["passed"] is False and record["staged"] is not None
    assert s3.count("put_object") == 2


def test_upload_is_refused_when_a_file_check_failed(context, cfg, capsys):
    stale_report = volumes_for(context, prompt_id="f" * 64)  # X6 fails
    assert run(context, cfg, "--upload", volumes=stale_report) == 1  # no_client would raise
    out = capsys.readouterr().out
    assert "FAIL  X6  Same question" in out
    assert "REFUSED: nothing was uploaded, because a check X1 to X7 failed" in out
    assert "staged" not in out and read_gate_record(cfg)["staged"] is None


def test_upload_without_a_sign_in_exits_2_and_writes_no_record(context, cfg, capsys):
    def signed_out(cfg):
        raise ConfigError("the AWS sign-in is missing or has expired: run 'aws sso login'")

    assert run(context, cfg, "--upload", factory=signed_out) == 2
    assert capsys.readouterr().out.startswith("CONFIG ERROR: the AWS sign-in is missing")
    assert not any(path.exists() for path in report_paths(cfg.report_dir))
    assert run_folders(context) == []


def test_an_estimate_above_the_cap_closes_the_gate(context, cfg, capsys):
    write_baseline_report(cfg, context)
    tight = dataclasses.replace(cfg, path_b_cap_usd=Decimal("4"))
    assert run(context, tight) == 1
    out = capsys.readouterr().out
    assert "FAIL  G2  Within the cap: the Path B estimate is $4.81, above the cap of $4.00" in out
    assert "GATE CLOSED" in out


def test_upload_and_check_do_not_go_together(context, cfg):
    with pytest.raises(SystemExit) as caught:
        run(context, cfg, "--upload", "--check")
    assert caught.value.code == 2
    assert run_folders(context) == []


def test_without_upload_boto3_is_not_loaded():
    code = (
        "import sys, scripts.nova_export; "
        "print(sorted(m for m in ('boto3', 'botocore') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_runs_as_a_module():
    """With the snapshot it reports on the real files; without it, a config error."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.nova_export", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1, 2), result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert ("X1  Same records" in result.stdout) or result.stdout.startswith("CONFIG ERROR")


def test_running_by_file_path_gives_a_hint_not_a_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "nova_export.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "python -m scripts.nova_export" in result.stderr
    assert "Traceback" not in result.stderr
