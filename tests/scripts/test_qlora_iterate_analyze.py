"""Tests for scripts.qlora_analyze and scripts.qlora_iterate: what they print, how they exit."""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import pytest

from qlora_finetuning.settings import REPO_ROOT
from scripts import qlora_analyze, qlora_iterate, qlora_select
from tests.local_qlora.selection_helpers import REVERSED, World, scored_world

MISSING = {2: 1, 4: 2, 6: 3}


@pytest.fixture
def world(tmp_path):
    world = scored_world(tmp_path, right=MISSING, wrong={6: REVERSED})
    world.select()
    return world


def analyze(world, *arguments):
    return qlora_analyze.main(["--run", world.run, *arguments], **world.stand_ins())


def iterate(world, *arguments, **changes):
    stand_ins = world.stand_ins()
    if changes:
        stand_ins["training_cfg"] = dataclasses.replace(world.first_cfg, **changes)
    return qlora_iterate.main(list(arguments), **stand_ins)


# --------------------------------------------------------------------------- qlora_analyze


def test_the_analysis_prints_the_categories_the_curve_and_the_finding(world, capsys):
    assert analyze(world) == 0
    out = capsys.readouterr().out
    assert f"training run {world.run}  (on dev)" in out and "6 passed, 0 failed" in out
    assert "checkpoint 6: 3 of 8 right on execution" in out
    lines = [line.split() for line in out.splitlines()]
    assert ["order", "5", "0"] in lines and ["no_program", "0", "8"] in lines
    assert ["right", "3", "0"] in lines
    assert "against the untuned model: 3 fixed, 0 broken" in out
    assert any(line[:2] == ["6", "3"] and line[-2:] == ["<-", "selected"] for line in lines)
    assert "the last three checkpoints are rising" in out
    assert "never right at any checkpoint: 5" in out
    assert "FINDING: T1 (" in out and "the table's change: epochs to 5" in out
    assert "ANALYSED: the report is written" in out
    assert world.training.training_cfg.analysis_record.is_file()
    assert "PROGRAM" not in out  # no reply is printed unless asked for


def test_check_writes_nothing_and_show_prints_replies_to_the_terminal_only(world, capsys):
    assert analyze(world, "--check", "--show", "order", "--n", "2") == 0
    out = capsys.readouterr().out
    assert "CHECK ONLY: all checks pass; nothing was written" in out
    assert "--- replies in category order (terminal only; do not commit) ---" in out
    assert out.count("  reply: PROGRAM: subtract(95, 920)") == 2
    assert out.count("  gold : subtract(920, 95)  ->  825") == 2
    assert not world.training.training_cfg.analysis_record.exists()
    folder = world.training.training_cfg.analysis_record.parent
    assert not any("subtract(95" in path.read_text(encoding="utf-8") for path in folder.iterdir())

    assert analyze(world, "--check", "--show", "cut_off") == 0
    assert "(none)" in capsys.readouterr().out


def test_analyze_exits_1_on_a_failed_check_and_2_on_a_config_problem(tmp_path, capsys):
    unscored = World(tmp_path)
    assert analyze(unscored, "--check") == 1
    out = capsys.readouterr().out
    assert "FAIL  L1" in out and "NOT WRITTEN: a check failed" in out and "FINDING" not in out

    unselected = scored_world(tmp_path / "other", right=MISSING)
    assert analyze(unselected) == 2
    assert "CONFIG ERROR: no checkpoint has been selected yet" in capsys.readouterr().out
    assert analyze(unselected, "--limit", "3") == 2
    assert "it is a rehearsal: add --check" in capsys.readouterr().out


# --------------------------------------------------------------------------- qlora_iterate


def test_propose_registers_one_change_and_says_what_to_do_next(world, capsys):
    analyze(world)
    capsys.readouterr()
    assert iterate(world, "propose", "--finding", "T2", "--reason", "x", learning_rate=1e-4) == 0
    out = capsys.readouterr().out
    assert out.count("PASS  I") == 5 and "5 passed, 0 failed" in out
    assert "change: optimizer.learning_rate from 0.0002 to 0.0001" in out
    assert "note: the analysis read T1" in out  # the owner chose another row than it read
    assert "REGISTERED: commit it" in out
    assert "sagemaker_job plan train --scope measured --workspace <first training job>" in out
    assert world.first_cfg.iteration_record.is_file()


def test_propose_reports_failed_checks_and_writes_nothing(world, capsys):
    assert iterate(world, "propose", "--finding", "T1", "--reason", "x", epochs=5) == 1
    out = capsys.readouterr().out
    assert "FAIL  I2  Analysed: there is no error analysis yet" in out
    assert "NOT REGISTERED: a check failed; nothing was written" in out
    assert not world.first_cfg.iteration_record.exists()

    analyze(world)
    capsys.readouterr()
    assert iterate(world, "propose", "--finding", "T1", "--reason", "x") == 1
    assert "FAIL  I3  One variable: config/training.yaml is the first run's" in (
        capsys.readouterr().out
    )
    assert iterate(world, "propose", "--finding", "T1", "--reason", "x", "--check", epochs=5) == 0
    assert "CHECK ONLY" in capsys.readouterr().out
    assert iterate(world, "propose", "--finding", "other", "--reason", "x", epochs=4) == 0
    assert "note: this is not the change the table names for other" in capsys.readouterr().out


def test_status_from_nothing_selected_to_done(tmp_path, capsys):
    fresh = World(tmp_path / "fresh")
    assert iterate(fresh, "status") == 0
    assert "state: nothing selected" in capsys.readouterr().out

    world = scored_world(tmp_path, right=MISSING)
    world.select()
    analyze(world)
    assert iterate(world, "status") == 0
    out = capsys.readouterr().out
    assert "state: no iteration" in out and "(verdict missed)" in out
    assert "PASS  S1  The config is the selected run's" in out

    iterate(world, "propose", "--finding", "T2", "--reason", "peaks early", learning_rate=1e-4)
    world.iterate({2: 3, 4: 5, 6: 4}, learning_rate=1e-4)
    capsys.readouterr()
    assert qlora_select.main(["--run", world.run], **world.stand_ins()) == 0
    out = capsys.readouterr().out
    assert "ITERATION: the second run's checkpoint is arm A1 (+2 examples" in out
    assert iterate(world, "status") == 0
    out = capsys.readouterr().out
    assert "state: done" in out and "reason: peaks early" in out
    assert f"second run: {world.run}; the second run's checkpoint is arm A1" in out


def test_select_says_when_the_first_run_stays_and_the_config_must_be_reverted(tmp_path, capsys):
    world = scored_world(tmp_path, right=MISSING)
    world.select()
    analyze(world)
    iterate(world, "propose", "--finding", "T2", "--reason", "x", learning_rate=1e-4)
    world.iterate({2: 1, 4: 1, 6: 2}, learning_rate=1e-4)
    capsys.readouterr()
    assert qlora_select.main(["--run", world.run], **world.stand_ins()) == 1  # still missed
    out = capsys.readouterr().out
    assert "ITERATION: the first run's checkpoint stays arm A1 (-1 examples" in out
    assert "revert the change in config/training.yaml" in out and "VERDICT: missed" in out
    assert iterate(world, "status") == 1
    assert "FAIL  S1" in capsys.readouterr().out


def test_config_problems_exit_2(world, capsys):
    with pytest.raises(SystemExit):
        iterate(world, "propose", "--finding", "T5", "--reason", "x")  # not a choice
    assert iterate(world, "propose", "--finding", "T1", "--reason", "  ") == 2
    assert "CONFIG ERROR: a change needs its reason" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["qlora_analyze", "qlora_iterate"])
def test_the_help_and_running_by_file_path(tmp_path, name):
    helped = subprocess.run(
        [sys.executable, "-m", f"scripts.{name}", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert helped.returncode == 0 and "usage:" in helped.stdout
    by_path = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / f"{name}.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert by_path.returncode != 0 and f"python -m scripts.{name}" in by_path.stderr
    assert "Traceback" not in by_path.stderr
