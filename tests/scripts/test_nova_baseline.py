"""Tests for scripts.nova_baseline, on runs made by the real runner over a stub provider."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from qlora_finetuning.frontier.baseline_report import report_paths
from qlora_finetuning.frontier.config import load_frontier_config
from scripts import nova_baseline
from tests.evaluation.conftest import ledger_records, make_context, run_folders
from tests.frontier.conftest import QWEN_PROMPT_TOKENS, baseline_run

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def context(tmp_path):
    return make_context(tmp_path)


@pytest.fixture
def cfg(tmp_path):
    return load_frontier_config(root=tmp_path)


@pytest.fixture
def run_ids(context, cfg):
    return [
        baseline_run(context, cfg, name, split)["run_id"]
        for split in ("quick_dev", "dev")
        for name in cfg.models
    ]


def run(context, cfg, *run_ids):
    return nova_baseline.main(
        ["--runs", *run_ids], context=context, cfg=cfg, qwen_prompt_tokens=QWEN_PROMPT_TOKENS
    )


def test_four_baseline_runs_give_a_report_and_exit_0(context, cfg, run_ids, tmp_path, capsys):
    before = (run_folders(context), len(ledger_records(context)))
    assert run(context, cfg, *run_ids) == 0
    out = capsys.readouterr().out
    assert out.count("PASS  ") == 4 and "FAIL" not in out
    assert "PASS  B1  4 results: one result per model (nova-pro, nova-lite)" in out
    assert "4 passed, 0 failed" in out
    assert "B0-Pro   dev        execution accuracy 8 of 8  100.00%; tokens in 320, out 96" in out
    assert "Nova/Qwen 4.0000; cost $0.000563" in out
    assert "report: reports/frontier/nova_baseline_report.md" in out
    assert "report: reports/frontier/nova_baseline_report.json" in out
    assert out.isascii() and str(tmp_path) not in out and "subtract" not in out

    markdown_path, json_path = report_paths(cfg.report_dir)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["passed"] and [r["run_id"] for r in report["runs"]] == run_ids
    assert markdown_path.read_text(encoding="utf-8").startswith("# Nova zero-shot baseline report")
    # It reads runs; it starts none and records nothing.
    assert (run_folders(context), len(ledger_records(context))) == before


def test_a_failed_check_exits_1_and_writes_nothing(context, cfg, run_ids, capsys):
    assert run(context, cfg, *run_ids[:3]) == 1
    out = capsys.readouterr().out
    assert "FAIL  B1  4 results: nova-lite on dev: 0 runs, expected 1" in out
    assert "3 passed, 1 failed" in out
    assert "NOT WRITTEN: a check failed, so no report was written" in out
    assert "report:" not in out
    assert not any(path.exists() for path in report_paths(cfg.report_dir))


def test_an_unknown_run_or_a_repeated_run_exits_2(context, cfg, run_ids, capsys):
    assert run(context, cfg, *run_ids[:3], "20261116T000000Z-eval-nothing-0000") == 2
    assert capsys.readouterr().out.startswith("CONFIG ERROR: run 20261116T000000Z-eval-nothing")
    assert run(context, cfg, *run_ids[:3], run_ids[0]) == 2
    assert "CONFIG ERROR: a run id was given twice" in capsys.readouterr().out
    assert not any(path.exists() for path in report_paths(cfg.report_dir))


def test_runs_are_required(context, cfg):
    with pytest.raises(SystemExit) as caught:
        nova_baseline.main([], context=context, cfg=cfg)
    assert caught.value.code == 2


def test_runs_as_a_module():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.nova_baseline", "--runs", "20261116T000000Z-eval-no-0000"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "CONFIG ERROR: run 20261116T000000Z-eval-no-0000 has no evaluation.json" in (
        result.stdout
    )
    assert "Traceback" not in result.stderr


def test_running_by_file_path_gives_a_hint_not_a_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "nova_baseline.py"), "--runs", "x"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "python -m scripts.nova_baseline" in result.stderr
    assert "Traceback" not in result.stderr
