"""Tests for scripts.qlora_local: what preflight, status, and note print, and how they exit."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from qlora_finetuning.local_qlora.local_run import GpuState, PowerState
from qlora_finetuning.local_qlora.training import TrainingError, run_training
from qlora_finetuning.settings import REPO_ROOT
from scripts import qlora_local
from tests.local_qlora.selection_helpers import REVERSED, World
from tests.local_qlora.training_helpers import (
    StubEngine,
    StubReloader,
    make_training_context,
    with_training,
)

LABEL = "local run: not an arm of the comparison"


@pytest.fixture
def ctx(tmp_path):
    return with_training(make_training_context(tmp_path), save_steps=2)


def machine(free_gib=15.5, on_ac=True):
    return {
        "gpu": lambda: GpuState("Test GPU", 16.0, 16.0 - free_gib, free_gib, ("browser.exe",)),
        "power": lambda: PowerState(on_ac, 0),
        "disk": lambda path: SimpleNamespace(free=500e9),
    }


def test_preflight_prints_the_checks_and_starts_nothing(ctx, capsys):
    assert qlora_local.main(["preflight"], training_context=ctx, **machine()) == 0
    out = capsys.readouterr().out
    assert f"preflight for the full run on this machine  ({LABEL})" in out
    assert out.count("PASS  P") == 6 and "6 passed, 0 failed, 0 skipped" in out
    assert "READY: nothing was started. Start the run: python -m scripts.qlora_train" in out
    assert not ctx.tracking_cfg.runs_dir.exists()


def test_preflight_exits_1_when_the_machine_is_not_ready(ctx, capsys):
    crowded = machine(free_gib=13.6, on_ac=False)
    assert qlora_local.main(["preflight"], training_context=ctx, **crowded) == 1
    out = capsys.readouterr().out
    assert "FAIL  P3  GPU memory: 13.6 GiB free of 16.0" in out and "browser.exe" in out
    assert "FAIL  P5  Power: on battery" in out
    assert "4 passed, 2 failed, 0 skipped" in out and "NOT READY" in out


def test_status_of_a_stopped_and_of_a_finished_run(ctx, capsys):
    tracking = {"tracking_cfg": ctx.tracking_cfg}
    assert qlora_local.main(["status"], **tracking) == 2
    assert "CONFIG ERROR: no full training run has been made" in capsys.readouterr().out

    with pytest.raises(TrainingError) as caught:
        run_training(context=ctx, engine=StubEngine(fail_at=5), reloader=StubReloader())
    first = caught.value.run_id
    assert qlora_local.main(["status"], **tracking) == 0
    out = capsys.readouterr().out
    assert f"training run {first}  (full run; local; {LABEL})" in out
    assert "step 4 of 6  (66.7%); state: failed" in out and "checkpoints at steps: 2, 4" in out
    assert f"python -m scripts.qlora_train --resume {first}" in out

    run_training(context=ctx, resume=first, engine=StubEngine(), reloader=StubReloader())
    assert qlora_local.main(["status"], **tracking) == 0
    out = capsys.readouterr().out
    assert f"continues: {first}" in out and "step 6 of 6  (100.0%); state: completed" in out
    assert "loss on dev at step 6: 0.1667" in out and "--resume" not in out
    assert qlora_local.main(["status", "--run", first], **tracking) == 0
    assert "continued by run " in capsys.readouterr().out


def test_note_writes_the_record_and_says_what_it_found(tmp_path, capsys):
    world = World(tmp_path, right={2: 1, 4: 3, 6: 2}, where="local", scope="dev")
    world.wrong = {4: REVERSED}
    world.score(split="quick_dev")
    arguments = ["note", "--run", world.run]
    assert qlora_local.main(arguments, **world.stand_ins()) == 1
    out = capsys.readouterr().out
    assert "FAIL  N4  On dev: checkpoint 4 has no complete result on dev" in out
    assert "NOT WRITTEN: a check failed" in out

    world.evaluate_checkpoint(4)
    assert qlora_local.main([*arguments, "--check"], **world.stand_ins()) == 0
    assert "CHECK ONLY: all checks pass; nothing was written" in capsys.readouterr().out
    assert qlora_local.main(arguments, **world.stand_ins()) == 0
    out = capsys.readouterr().out
    assert f"(checkpoints scored on quick_dev; {LABEL})" in out and "5 passed, 0 failed" in out
    assert "selected here: checkpoint 4, 3 of 3 right on quick_dev" in out
    assert "on all of dev: 3 of 8 right on execution, 8 in the format" in out
    assert "local against cloud: not yet" in out and "WRITTEN: the local note" in out
    assert world.training.training_cfg.local_note.is_file()
    assert (
        qlora_local.main(["note", "--run", "20261120T000000Z-train-x-0000"], **world.stand_ins())
        == 2
    )


def test_the_help_and_running_by_file_path(tmp_path):
    helped = subprocess.run(
        [sys.executable, "-m", "scripts.qlora_local", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert helped.returncode == 0 and "preflight" in helped.stdout and "note" in helped.stdout
    by_path = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "qlora_local.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert by_path.returncode != 0 and "python -m scripts.qlora_local" in by_path.stderr
    assert "Traceback" not in by_path.stderr
