"""Tests for scripts.qlora_select: what it prints and how it exits."""

from __future__ import annotations

import subprocess
import sys

from qlora_finetuning.settings import REPO_ROOT
from scripts import qlora_select
from tests.local_qlora.selection_helpers import World, scored_world

RIGHT = {2: 3, 4: 5, 6: 5}


def run(world, *arguments):
    return qlora_select.main(["--run", world.run, *arguments], **world.stand_ins())


def test_a_selection_prints_the_checks_the_table_and_what_to_do_next(tmp_path, capsys):
    world = scored_world(tmp_path, right=RIGHT)
    assert run(world) == 0
    out = capsys.readouterr().out
    assert f"training run {world.run}  (on dev)" in out
    assert out.count("PASS  L") == 6 and "6 passed, 0 failed" in out
    assert "PASS  L1  Every checkpoint scored: 3 checkpoints and the untuned model" in out
    lines = out.splitlines()
    assert any(line.split() == ["untuned", "0", "0.00%", "0", "0.00%"] for line in lines)
    best = next(line for line in lines if line.endswith("<- best"))
    assert best.split()[:5] == ["4", "5", "62.50%", "8", "100.00%"]
    assert f"best: checkpoint 4  ({world.run}/checkpoint-4)" in out
    assert "above the untuned model by 62.50 points" in out and "VERDICT: met" in out
    assert "SELECTED: the selection record is written" in out
    assert "python -m scripts.sagemaker_job pull --checkpoint 4" in out
    assert world.training.training_cfg.selection_record.is_file()


def test_check_writes_nothing_and_says_so(tmp_path, capsys):
    world = scored_world(tmp_path, right=RIGHT)
    assert run(world, "--check") == 0
    assert "CHECK ONLY: all checks pass; nothing was written" in capsys.readouterr().out
    assert not world.training.training_cfg.selection_record.exists()


def test_a_missed_threshold_exits_1_with_the_record_written(tmp_path, capsys):
    world = scored_world(tmp_path, right={2: 1, 4: 2, 6: 3})
    assert run(world) == 1
    out = capsys.readouterr().out
    assert "VERDICT: missed" in out and "SELECTED: the selection record is written" in out
    assert "note: it is the last checkpoint; the model may still have been improving" in out


def test_a_failed_check_exits_1_and_selects_nothing(tmp_path, capsys):
    world = World(tmp_path)
    assert run(world) == 1
    out = capsys.readouterr().out
    assert "FAIL  L1  Every checkpoint scored: checkpoint 2 has 0 complete results" in out
    assert "0 passed, 1 failed" in out and "NOT SELECTED: a check failed" in out
    assert "VERDICT" not in out


def test_config_problems_exit_2(tmp_path, capsys):
    world = scored_world(tmp_path, right=RIGHT)
    assert run(world, "--limit", "3") == 2
    assert "CONFIG ERROR: a selection is made from complete runs" in capsys.readouterr().out
    stand_ins = world.stand_ins()
    assert qlora_select.main(["--run", "20261120T000000Z-train-x-0000"], **stand_ins) == 2
    assert "CONFIG ERROR" in capsys.readouterr().out


def test_a_rehearsal_says_it_is_one(tmp_path, capsys):
    world = World(tmp_path, right={2: 1, 4: 3, 6: 2})
    world.record_jobs(world.score(limit=3))
    assert run(world, "--limit", "3", "--check") == 0
    out = capsys.readouterr().out
    assert "(rehearsal on dev, the first 3 examples)" in out and "CHECK ONLY" in out


def test_the_help_and_running_by_file_path(tmp_path):
    helped = subprocess.run(
        [sys.executable, "-m", "scripts.qlora_select", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert helped.returncode == 0 and "--check" in helped.stdout and "--run RUN_ID" in helped.stdout
    by_path = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "qlora_select.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert by_path.returncode != 0 and "python -m scripts.qlora_select" in by_path.stderr
    assert "Traceback" not in by_path.stderr
