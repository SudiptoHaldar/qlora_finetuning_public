"""Tests for scripts.qlora_train, on an invented split with a stand-in engine (no model)."""

from __future__ import annotations

import dataclasses
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import qlora_train
from tests.evaluation.conftest import ledger_records, run_folders
from tests.local_qlora.conftest import OutOfMemoryError
from tests.local_qlora.training_helpers import (
    StubEngine,
    StubReloader,
    make_training_context,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def context(tmp_path):
    return make_training_context(tmp_path)


def run(context, *arguments, engine=None, reloader=None):
    return qlora_train.main(
        list(arguments),
        context=context,
        engine=engine or StubEngine(),
        reloader=reloader or StubReloader(),
    )


def run_id_in(out):
    return re.search(r"^run (\S+)", out, flags=re.MULTILINE).group(1)


def test_the_dry_run_prints_the_plan_and_starts_no_run(context, capsys):
    engine = StubEngine()
    assert run(context, "--dry-run", engine=engine) == 0
    out = capsys.readouterr().out
    assert "plan for the full run:" in out
    assert re.search(r"^records +30$", out, flags=re.MULTILINE)
    assert re.search(r"^tokens in one pass +[\d,]+$", out, flags=re.MULTILINE)
    assert "batch                 2 x 8 (16 sequences per optimizer step)" in out
    assert "optimizer steps       6 (2 per epoch)" in out
    assert "warmup steps          1" in out and "checkpoints at steps  6" in out
    assert re.search(r"^worst batch +2 sequences, [\d,]+ tokens wide$", out, flags=re.MULTILINE)
    assert "dev records for loss  8" in out
    assert out.rstrip().endswith("DRY RUN: nothing was trained and no run was started")
    assert engine.jobs == [] and run_folders(context) == [] and ledger_records(context) == []
    assert out.isascii() and "zebrafish" not in out


def test_the_dry_run_of_a_smoke_run(context, capsys):
    assert run(context, "--dry-run", "--smoke") == 0
    out = capsys.readouterr().out
    assert "plan for the smoke run:" in out and "checkpoints at steps  4, 6" in out


def test_a_smoke_run_prints_its_checks_and_how_to_score_the_adapter(context, tmp_path, capsys):
    assert run(context, "--smoke") == 0
    out = capsys.readouterr().out
    run_id = run_id_in(out)
    assert f"run {run_id}  (smoke)" in out
    assert out.count("PASS  K") == 7 and "FAIL" not in out and "7 passed, 0 failed" in out
    assert "PASS  K2  Labels: only the reply and its end token carry a label" in out
    assert "PASS  K7  Adapter reloads: 3 prompts answered; 3 programs run without an error" in out
    assert "steps: 6 of 6" in out
    assert "training: 1 s, 0.5 s per step" in out
    assert f"adapter: {run_id}  (sha256 " in out
    assert f"--predictor qwen3-4b-tuned --adapter {run_id} --split quick_dev" in out
    assert out.isascii() and str(tmp_path) not in out and "zebrafish" not in out
    assert run_folders(context) == [run_id]


def test_a_failed_check_exits_1(context, capsys):
    assert run(context, "--smoke", engine=StubEngine(losses=lambda step: 2.0)) == 1
    out = capsys.readouterr().out
    assert "FAIL  K6  Loss falls" in out and "6 passed, 1 failed" in out


def test_a_training_failure_exits_1_and_says_how_to_resume(context, capsys):
    engine = StubEngine(fail_at=6, error=OutOfMemoryError("CUDA out of memory"))
    assert run(context, "--smoke", engine=engine) == 1
    out = capsys.readouterr().out
    (run_id,) = run_folders(context)
    assert out.startswith("TRAINING FAILED: OutOfMemoryError after 5 of 6 steps")
    assert f"checkpoints are kept in run {run_id}" in out
    assert f"resume with: python -m scripts.qlora_train --smoke --resume {run_id}" in out
    assert "Traceback" not in out


def test_stop_after_steps_then_resume(context, capsys):
    assert run(context, "--smoke", "--stop-after-steps", "5") == 1
    out = capsys.readouterr().out
    assert "TRAINING FAILED: stopped on request after 5 steps" in out
    (first,) = run_folders(context)
    assert run(context, "--smoke", "--resume", first) == 0
    out = capsys.readouterr().out
    assert f"(smoke; continues {first})" in out
    assert "steps: 6 of 6 (from step 4)" in out and "7 passed, 0 failed" in out


def test_config_problems_and_a_refused_resume_exit_2(context, capsys):
    short = dataclasses.replace(context, budget=dataclasses.replace(context.budget, max_length=40))
    assert run(short, "--smoke") == 2
    out = capsys.readouterr().out
    assert out.startswith("CONFIG ERROR: 30 training sequences are longer than max_length 40")
    assert "Nothing is truncated" in out

    assert run(context, "--resume", "20261119T000000Z-train-none-0000") == 2
    assert "CONFIG ERROR: run 20261119T000000Z-train-none-0000 cannot be continued" in (
        capsys.readouterr().out
    )
    assert run(context, "--smoke", "--stop-after-steps", "0") == 2
    assert run_folders(context) == []


@pytest.mark.parametrize(
    "arguments", [("--dry-run", "--resume", "x"), ("--dry-run", "--stop-after-steps", "3")]
)
def test_arguments_that_do_not_go_together(context, arguments):
    with pytest.raises(SystemExit) as caught:
        run(context, *arguments)
    assert caught.value.code == 2


def test_runs_as_a_module():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.qlora_train", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--dry-run" in result.stdout and "--stop-after-steps" in result.stdout
    assert "Traceback" not in result.stderr


def test_running_by_file_path_gives_a_hint_not_a_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "qlora_train.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "python -m scripts.qlora_train" in result.stderr
    assert "Traceback" not in result.stderr
