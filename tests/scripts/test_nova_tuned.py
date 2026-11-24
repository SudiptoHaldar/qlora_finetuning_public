"""Tests for scripts.nova_tuned, on runs made by the real runner over stub providers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from qlora_finetuning.frontier.config import load_frontier_config
from qlora_finetuning.frontier.training_estimate import report_paths as gate_paths
from qlora_finetuning.frontier.tuned_report import report_paths
from scripts import nova_tuned
from tests.evaluation.conftest import GOLD_REPLY, ledger_records, make_context, run_folders
from tests.frontier.tuned_helpers import (
    SPLITS,
    deploy_models,
    right_for,
    scripted_baseline,
    sizes,
    tuned_run,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = ("nova-pro", "nova-lite")


@pytest.fixture
def context(tmp_path):
    return make_context(tmp_path)


@pytest.fixture
def cfg(tmp_path):
    return load_frontier_config(root=tmp_path)


def make_runs(context, cfg, models=MODELS, tuned_right=7, baseline_right=2):
    """Run ids of the tuned runs and of their baselines, per model and split."""
    deploy_models(cfg, context, models=models)
    tuned, baselines = [], []
    for split in SPLITS:
        total = sizes()[split]
        for name in models:
            baselines.append(
                scripted_baseline(
                    context, cfg, name, split, right_for(split, min(baseline_right, total))
                )["run_id"]
            )
            tuned.append(
                tuned_run(context, cfg, name, split, right_for(split, min(tuned_right, total)))[
                    "run_id"
                ]
            )
    return tuned, baselines


def run(context, cfg, tuned, baselines):
    return nova_tuned.main(["--runs", *tuned, "--baselines", *baselines], context=context, cfg=cfg)


def test_both_models_meeting_the_threshold_give_a_report_and_exit_0(context, cfg, tmp_path, capsys):
    tuned, baselines = make_runs(context, cfg)
    before = (run_folders(context), len(ledger_records(context)))
    assert run(context, cfg, tuned, baselines) == 0
    out = capsys.readouterr().out
    assert out.count("PASS  ") == 6 and "FAIL" not in out
    assert "PASS  V2  Paired: 4 tuned runs, each with its baseline" in out
    assert "6 passed, 0 failed" in out
    assert (
        "nova-lite  dev: execution accuracy 2 -> 7 of 8 (62.50 points; 3 needed); format "
        "validity 8 of 8 (8 needed); fixed 5, broken 0"
    ) in out
    assert "MET: nova-lite (B1-Lite)" in out and "MET: nova-pro (B1-Pro)" in out
    assert "2 of 2 models met their threshold" in out
    assert "report: reports/frontier/nova_tuned_report.md" in out
    assert "report: reports/frontier/nova_tuned_report.json" in out
    assert "PARTIAL" not in out and "MISSED" not in out
    assert out.isascii() and str(tmp_path) not in out
    assert GOLD_REPLY not in out and "subtract" not in out

    markdown_path, json_path = report_paths(cfg.report_dir)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["passed"] and report["all_met"] and not report["partial"]
    assert markdown_path.read_text(encoding="utf-8").startswith("# Nova tuned-model report")
    # It reads runs; it starts none and records nothing.
    assert (run_folders(context), len(ledger_records(context))) == before


def test_a_model_that_missed_exits_1_and_the_report_is_still_written(context, cfg, capsys):
    tuned, baselines = make_runs(context, cfg, tuned_right=2)
    assert run(context, cfg, tuned, baselines) == 1
    out = capsys.readouterr().out
    assert "6 passed, 0 failed" in out
    assert "MISSED: nova-lite (B1-Lite)" in out and "MISSED: nova-pro (B1-Pro)" in out
    assert "execution accuracy rose by 0.00 points; 10 are needed (3 of 8 correct, 2 were)" in out
    assert "v4.05 is eligible for nova-pro; nothing starts by itself" in out
    assert "0 of 2 models met their threshold" in out
    _, json_path = report_paths(cfg.report_dir)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["passed"] and not report["all_met"]


def test_one_model_only_is_reported_as_partial(context, cfg, capsys):
    tuned, baselines = make_runs(context, cfg, models=("nova-lite",))
    assert run(context, cfg, tuned, baselines) == 0
    out = capsys.readouterr().out
    assert "PARTIAL: no tuned run was given for nova-pro; milestone M6 needs every model" in out
    assert "1 of 1 models met their threshold" in out


def test_a_failed_check_exits_1_and_writes_nothing(context, cfg, capsys):
    tuned, baselines = make_runs(context, cfg)
    assert run(context, cfg, tuned, baselines[1:]) == 1
    out = capsys.readouterr().out
    assert "FAIL  V2  Paired:" in out and "has 0 baseline runs, expected 1" in out
    assert "5 passed, 1 failed" in out
    assert "NOT WRITTEN: a check failed, so no report was written" in out
    assert "report:" not in out and "MET" not in out
    assert not any(path.exists() for path in report_paths(cfg.report_dir))


def test_without_a_gate_record_the_check_says_so_and_nothing_is_written(context, cfg, capsys):
    tuned, baselines = make_runs(context, cfg)
    for path in gate_paths(cfg.report_dir):
        path.unlink()
    assert run(context, cfg, tuned, baselines) == 1
    out = capsys.readouterr().out
    assert "FAIL  V4  The right files: there is no gate record" in out
    assert not any(path.exists() for path in report_paths(cfg.report_dir))


def test_an_unknown_run_or_a_repeated_run_exits_2(context, cfg, capsys):
    tuned, baselines = make_runs(context, cfg, models=("nova-lite",))
    assert run(context, cfg, [*tuned, "20261118T000000Z-eval-nothing-0000"], baselines) == 2
    assert capsys.readouterr().out.startswith("CONFIG ERROR: run 20261118T000000Z-eval-nothing")
    assert run(context, cfg, tuned, [*baselines, tuned[0]]) == 2
    assert "CONFIG ERROR: a run id was given twice" in capsys.readouterr().out
    assert not any(path.exists() for path in report_paths(cfg.report_dir))


def test_both_lists_of_runs_are_required(context, cfg):
    for arguments in ([], ["--runs", "x"], ["--baselines", "y"]):
        with pytest.raises(SystemExit) as caught:
            nova_tuned.main(arguments, context=context, cfg=cfg)
        assert caught.value.code == 2


def test_runs_as_a_module_and_needs_no_aws_library():
    code = (
        "import sys; from scripts import nova_tuned; "
        "print(sorted(m for m in ('boto3', 'botocore', 'torch') if m in sys.modules)); "
        "sys.exit(nova_tuned.main(['--runs', '20261118T000000Z-eval-no-0000', "
        "'--baselines', '20261118T000000Z-eval-no-0001']))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert result.stdout.startswith("[]")  # it cannot call AWS: boto3 is never loaded
    assert "CONFIG ERROR: run 20261118T000000Z-eval-no-0000 has no evaluation.json" in (
        result.stdout
    )
    assert "Traceback" not in result.stderr


def test_running_by_file_path_gives_a_hint_not_a_traceback(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "nova_tuned.py"),
            "--runs",
            "x",
            "--baselines",
            "y",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "python -m scripts.nova_tuned" in result.stderr
    assert "Traceback" not in result.stderr
