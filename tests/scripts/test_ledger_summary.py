"""Tests for scripts.ledger_summary, including one dummy run from start to summary."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from qlora_finetuning.tracking import GitInfo, Ledger, load_tracking_config, start_run
from scripts import ledger_summary

REPO_ROOT = Path(__file__).resolve().parents[2]
CLEAN = GitInfo(commit="0123456789abcdef0123456789abcdef01234567", branch="main", dirty=False)


@pytest.fixture
def cfg(tmp_path):
    return load_tracking_config(root=tmp_path)


def dummy_run(cfg):
    """A run with two ledger records: a computed estimate and reported usage."""
    with start_run(
        "train", "b1 pro", path="B", arm="B1-Pro", where="bedrock", scope="dev",
        configs=["aws"], cfg=cfg, git=CLEAN, environment={},
    ) as run:  # fmt: skip
        run.record(
            phase="train",
            model_id="amazon.nova-pro-v1:0:300k",
            training_tokens=9_000_000,
            source="computed",
            source_detail="tokens in the training file x 2 epochs",
        )
        run.record(
            phase="eval",
            model_id="amazon.nova-pro-v1:0",
            input_tokens=812_000,
            output_tokens=41_000,
            examples=500,
            wall_clock_seconds=930.5,
            source="aws_api",
        )
    return run


def test_empty_ledger_prints_no_records(cfg, capsys):
    assert ledger_summary.main([], cfg=cfg) == 0
    assert capsys.readouterr().out.strip() == "no records"


def test_dummy_run_end_to_end(cfg, capsys):
    run = dummy_run(cfg)
    assert (run.run_dir / "manifest.json").is_file()
    assert (run.run_dir / "config" / "aws.yaml").is_file()
    assert len(cfg.runs_index_file.read_text(encoding="utf-8").splitlines()) == 2
    assert len(Ledger(cfg).read()) == 2

    assert ledger_summary.main([], cfg=cfg) == 0
    out = capsys.readouterr().out
    assert "Runs (1, newest first)" in out
    assert run.run_id in out
    assert "completed" in out
    assert "Ledger totals by arm, phase (2 records)" in out
    assert "9,000,000" in out
    assert "812,000" in out
    assert "930.5" in out
    assert "1 of 2 records are estimates (computed)" in out
    assert out.isascii()


def test_group_by_option(cfg, capsys):
    dummy_run(cfg)
    assert ledger_summary.main(["--by", "where, source"], cfg=cfg) == 0
    out = capsys.readouterr().out
    assert "Ledger totals by where, source" in out
    assert "computed" in out


def test_json_output(cfg, capsys):
    run = dummy_run(cfg)
    assert ledger_summary.main(["--json"], cfg=cfg) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["runs"][0]["run_id"] == run.run_id
    assert summary["records"] == 2
    assert summary["estimates"] == 1
    assert summary["replaced_records"] == 0
    assert {row["phase"] for row in summary["totals"]} == {"train", "eval"}


def test_unknown_grouping_is_a_config_error(cfg, capsys):
    assert ledger_summary.main(["--by", "colour"], cfg=cfg) == 2
    assert "CONFIG ERROR" in capsys.readouterr().out


def test_corrupt_ledger_is_a_config_error(cfg, capsys):
    dummy_run(cfg)
    with cfg.ledger_file.open("a", encoding="utf-8") as handle:
        handle.write("{broken\n")
    assert ledger_summary.main([], cfg=cfg) == 2
    assert "ledger.jsonl line 3" in capsys.readouterr().out


def test_format_table_aligns_numbers_right():
    table = ledger_summary.format_table(
        ("arm", "examples"), [{"arm": "A0", "examples": 5}, {"arm": "B0-Pro", "examples": 1200}]
    )
    assert table.splitlines() == [
        "arm     examples",
        "------  --------",
        "A0             5",
        "B0-Pro     1,200",
    ]


def test_runs_as_a_module_on_the_real_ledger():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.ledger_summary", "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "runs" in json.loads(result.stdout)


def test_running_by_file_path_gives_a_hint_not_a_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "ledger_summary.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "python -m scripts.ledger_summary" in result.stderr
    assert "Traceback" not in result.stderr
