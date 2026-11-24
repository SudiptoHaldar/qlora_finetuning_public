"""Tests for qlora_finetuning.local_qlora.local_run: the preflight and the progress line of
the optional full run on this machine (req_003 v3.06). The GPU, the power state, and the
disk are stand-ins.
"""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from qlora_finetuning.local_qlora import local_run as local_run_module
from qlora_finetuning.local_qlora.local_run import (
    _AC_SLEEP,
    COMPLETED,
    FAILED,
    INTERRUPTED,
    RUNNING,
    GpuState,
    PowerState,
    find_local_run,
    preflight,
    probe_gpu,
    probe_power,
    run_status,
)
from qlora_finetuning.local_qlora.training import LOG_NAME, TrainingError, run_training
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking import GitInfo
from qlora_finetuning.tracking.config import utc_now
from qlora_finetuning.tracking.jsonl import read_lines

from .training_helpers import (
    StubEngine,
    StubReloader,
    files_under,
    make_training_context,
    with_training,
)

ROOMY = GpuState("Test GPU", 16.0, 0.5, 15.5, ("browser.exe",))
MAINS = PowerState(on_ac=True, sleep_on_ac_seconds=0)


@pytest.fixture
def ctx(tmp_path):
    """Checkpoints at steps 2, 4, and 6 of the invented split's 6."""
    return with_training(make_training_context(tmp_path), save_steps=2)


def check(outcome, check_id):
    return next(item for item in outcome["checks"] if item["id"] == check_id)


def flight(ctx, gpu=ROOMY, power=MAINS, free=500e9, **options):
    return preflight(
        context=ctx,
        gpu=lambda: gpu,
        power=lambda: power,
        disk=lambda path: SimpleNamespace(free=free),
        **options,
    )


def train(ctx, **options):
    engine = StubEngine(fail_at=options.pop("fail_at", None))
    return run_training(context=ctx, engine=engine, reloader=StubReloader(), **options)


def stopped(ctx, at=5):
    """A full run that failed after its checkpoint at step 4; returns its id."""
    with pytest.raises(TrainingError) as caught:
        train(ctx, fail_at=at)
    return caught.value.run_id


def unfinish(ctx, run_id):
    """Make the run look killed: no ``finished`` line, and a manifest still ``started``."""
    manifest = ctx.tracking_cfg.runs_dir / run_id / "manifest.json"
    content = json.loads(manifest.read_text(encoding="utf-8"))
    content.update(status="started", finished_at=None)
    manifest.write_text(json.dumps(content), encoding="utf-8")
    path = ctx.tracking_cfg.runs_index_file
    kept = [
        line
        for line in read_lines(path)
        if not (line.get("run_id") == run_id and line.get("event") == "finished")
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in kept), encoding="utf-8")


# --------------------------------------------------------------------------- the config


def test_the_committed_file_says_what_the_machine_must_have_free():
    from qlora_finetuning.local_qlora.training_config import load_training_config

    cfg = load_training_config()
    assert (cfg.min_free_gpu_gib, cfg.min_free_disk_gb) == (14.0, 10.0)
    assert cfg.local_note.name == "local_run.json"
    assert cfg.local_note.parent == cfg.selection_record.parent
    assert cfg.local_note != cfg.selection_record  # a file of its own: never the selection
    assert "local" not in json.dumps(cfg.settings(False))  # none of it changes a training run


# --------------------------------------------------------------------------- the preflight


def test_a_machine_that_is_ready_passes_and_nothing_is_started(ctx, tmp_path):
    before = files_under(tmp_path)
    outcome = flight(ctx)
    assert outcome["passed"] and [item["id"] for item in outcome["checks"]] == [
        f"P{n}" for n in range(1, 7)
    ]
    assert all(item["status"] == "PASS" for item in outcome["checks"])
    assert outcome["plan"]["total_steps"] == 6 and outcome["plan"]["checkpoint_steps"] == [2, 4, 6]
    assert (
        "30 records, 3 epochs, 6 optimizer steps, 3 checkpoints" in check(outcome, "P1")["detail"]
    )
    assert "no local run has measured a speed yet" in check(outcome, "P1")["detail"]
    assert outcome["estimate"] is None
    assert "15.5 GiB free of 16.0 on Test GPU" in check(outcome, "P3")["detail"]
    assert check(outcome, "P5")["detail"] == "on AC power; sleep is set to never"
    assert files_under(tmp_path) == before  # no run, no file


def test_p1_estimates_from_what_the_last_local_run_measured(ctx):
    result = train(ctx)
    outcome = flight(ctx)
    assert outcome["estimate"] == {
        "sequences": 90,
        "seconds_per_sequence": 0.03,
        "measured_by": result["run_id"],
        "training_hours": 0.0,
    }
    assert "at 0.03 s per sequence" in check(outcome, "P1")["detail"]


def test_p1_fails_when_the_plan_cannot_be_made(ctx, monkeypatch):
    def refuse(context, smoke):
        raise ConfigError("record ACME/1 is over the training length")

    monkeypatch.setattr(local_run_module, "prepare", refuse)
    outcome = flight(ctx)
    assert not outcome["passed"] and outcome["plan"] is None
    assert check(outcome, "P1")["detail"] == "record ACME/1 is over the training length"
    assert len(outcome["checks"]) == 6  # the other checks are still made


def test_p2_wants_a_clean_tree(ctx):
    dirty = GitInfo("0" * 40, "main", True, ("config/training.yaml", "a.py", "b.py", "c.py"))
    outcome = flight(dataclasses.replace(ctx, git=dirty))
    assert check(outcome, "P2")["status"] == "FAIL" and not outcome["passed"]
    assert (
        "uncommitted changes (config/training.yaml, a.py, b.py ...)"
        in (check(outcome, "P2")["detail"])
    )
    unknown = dataclasses.replace(ctx, git=GitInfo("unknown"))
    assert "git did not answer" in check(flight(unknown), "P2")["detail"]


def test_p3_wants_the_card_free_and_names_what_holds_it(ctx):
    crowded = GpuState("Test GPU", 16.0, 2.4, 13.6, tuple(f"p{n}.exe" for n in range(8)))
    detail = check(flight(ctx, gpu=crowded), "P3")
    assert detail["status"] == "FAIL"
    assert "13.6 GiB free of 16.0" in detail["detail"] and "14 are asked" in detail["detail"]
    assert (
        "Close what holds it: p0.exe, p1.exe, p2.exe, p3.exe, p4.exe, p5.exe ..."
        in (detail["detail"])
    )
    at_the_limit = GpuState("Test GPU", 16.0, 2.0, 14.0)
    assert check(flight(ctx, gpu=at_the_limit), "P3")["status"] == "PASS"
    lenient = with_training(ctx, min_free_gpu_gib=13.0)
    assert check(flight(lenient, gpu=crowded), "P3")["status"] == "PASS"
    missing = check(flight(ctx, gpu=None), "P3")
    assert missing["status"] == "FAIL" and "no NVIDIA GPU answered" in missing["detail"]


def test_p4_wants_room_for_the_checkpoints(ctx):
    assert check(flight(ctx, free=9e9), "P4")["status"] == "FAIL"
    assert check(flight(ctx, free=10e9), "P4")["detail"] == "10 GB free; 10 are asked"


@pytest.mark.parametrize(
    ("power", "status", "words"),
    [
        (PowerState(False, 0), "FAIL", "on battery"),
        (PowerState(True, 1800), "FAIL", "sleeps after 30 minutes"),
        (PowerState(True, None), "SKIP", "the sleep setting could not be read"),
        (PowerState(None, None), "SKIP", "cannot be read on this system"),
    ],
)
def test_p5_power(ctx, power, status, words):
    outcome = flight(ctx, power=power)
    assert check(outcome, "P5")["status"] == status and words in check(outcome, "P5")["detail"]
    assert outcome["passed"] == (status != "FAIL")  # what cannot be read does not fail


def test_p6_a_run_that_stopped_is_continued_not_started_again(ctx):
    run_id = stopped(ctx)
    outcome = flight(ctx)
    detail = check(outcome, "P6")["detail"]
    assert check(outcome, "P6")["status"] == "FAIL"
    assert f"run {run_id} stopped with a checkpoint at step 4 of 6" in detail
    assert f"python -m scripts.qlora_train --resume {run_id}" in detail

    train(ctx, resume=run_id)  # continued: nothing is left to resume
    assert check(flight(ctx), "P6")["status"] == "PASS"


def test_p6_ignores_what_is_not_this_run_to_resume(ctx):
    with pytest.raises(TrainingError):
        train(ctx, smoke=True, fail_at=3)  # a smoke run is not resumed into a full one
    assert check(flight(ctx), "P6")["status"] == "PASS"
    stopped(ctx)
    other = with_training(ctx, learning_rate=1e-4)  # made under other settings than today's
    assert check(flight(other), "P6")["status"] == "PASS"


def test_p6_a_run_in_progress(ctx):
    result = train(ctx)
    unfinish(ctx, result["run_id"])
    detail = check(flight(ctx), "P6")
    assert (
        detail["status"] == "FAIL"
        and f"run {result['run_id']} is in progress" in (detail["detail"])
    )
    later = utc_now() + timedelta(hours=2)  # its files are old: it was killed, not running
    assert "stopped with a checkpoint at step 6" in check(flight(ctx, now=later), "P6")["detail"]


# --------------------------------------------------------------------------- reading the machine


def test_the_gpu_is_read_from_nvidia_smi(monkeypatch):
    answers = {
        "--query-gpu=name,memory.total,memory.used,memory.free": (
            "NVIDIA GeForce RTX 4090 Laptop GPU, 16376, 2480, 13569\n"
        ),
        "--query-compute-apps=process_name": (
            "[Insufficient Permissions]\n"
            "C:\\Program Files\\Browser\\brave.exe\n"
            "C:\\Program Files\\Browser\\brave.exe\n"
            "C:\\Windows\\explorer.exe\n"
        ),
    }
    monkeypatch.setattr(local_run_module, "_run", lambda command: answers[command[1]])
    state = probe_gpu()
    assert state.name == "NVIDIA GeForce RTX 4090 Laptop GPU"
    assert (round(state.total_gib, 2), round(state.used_gib, 2)) == (15.99, 2.42)
    assert round(state.free_gib, 2) == 13.25 and state.programs == ("brave.exe", "explorer.exe")


@pytest.mark.parametrize("answer", [None, "", "not, a, gpu, line\n", "one, two\n"])
def test_a_gpu_that_does_not_answer_is_none_not_an_error(monkeypatch, answer):
    monkeypatch.setattr(local_run_module, "_run", lambda command: answer)
    assert probe_gpu() is None


def test_a_tool_that_is_missing_gives_no_answer():
    assert local_run_module._run(["a-tool-that-is-not-installed-anywhere"]) is None


def test_the_power_state_is_unknown_off_windows_and_the_sleep_setting_is_parsed(monkeypatch):
    monkeypatch.setattr(local_run_module.sys, "platform", "linux")
    assert probe_power() == PowerState()
    text = (
        "    Current AC Power Setting Index: 0x00000708\n    Current DC Power Setting Index: 0x0\n"
    )
    assert int(_AC_SLEEP.search(text).group(1), 16) == 1800


# --------------------------------------------------------------------------- the progress line


def test_a_completed_run(ctx):
    result = train(ctx)
    status = run_status(tracking_cfg=ctx.tracking_cfg)  # the newest full local run
    assert status["run"] == result["run_id"] and not status["smoke"]
    assert (status["step"], status["total"], status["share"]) == (6, 6, 1.0)
    assert status["state"] == COMPLETED and status["resume_command"] is None
    assert status["seconds_per_step"] == 0.5 and status["hours_left"] is None
    assert status["last_dev_loss"] == {"step": 6, "loss": pytest.approx(1 / 6)}
    assert status["last_loss"] == pytest.approx(2 / 6) and status["checkpoints"] == [2, 4, 6]
    assert status["chain"] == [result["run_id"]] and status["where"] == "local"


def test_a_run_that_stopped_says_how_to_continue_and_a_resumed_one_says_what_it_continues(ctx):
    first = stopped(ctx)
    status = run_status(first, tracking_cfg=ctx.tracking_cfg)
    assert (status["step"], status["state"], status["checkpoints"]) == (4, FAILED, [2, 4])
    assert status["resume_command"] == f"python -m scripts.qlora_train --resume {first}"

    second = train(ctx, resume=first)["run_id"]
    assert run_status(first, tracking_cfg=ctx.tracking_cfg)["continued_by"] == second
    assert run_status(first, tracking_cfg=ctx.tracking_cfg)["resume_command"] is None
    status = run_status(tracking_cfg=ctx.tracking_cfg)
    assert status["run"] == second and status["chain"] == [first, second]
    assert (status["start_step"], status["step"], status["state"]) == (4, 6, COMPLETED)


def test_a_run_in_progress_has_hours_left_from_its_log(ctx):
    run_id = stopped(ctx)
    unfinish(ctx, run_id)
    tracking = ctx.tracking_cfg
    started = run_status(run_id, tracking_cfg=tracking)["started_at"]
    from qlora_finetuning.local_qlora.local_run import _moment

    begun = _moment(started)
    log = tracking.runs_dir / run_id / LOG_NAME
    os.utime(log, (begun.timestamp() + 40, begun.timestamp() + 40))  # 4 steps in 40 seconds
    now = begun + timedelta(seconds=50)
    status = run_status(run_id, tracking_cfg=tracking, now=now)
    assert status["state"] == RUNNING and status["seconds_per_step"] == pytest.approx(10.0)
    assert status["hours_left"] == pytest.approx(2 * 10 / 3600)
    assert status["running_for"] == "0:00:50" and status["resume_command"] is None

    killed = run_status(run_id, tracking_cfg=tracking, now=begun + timedelta(hours=3))
    assert killed["state"] == INTERRUPTED and killed["hours_left"] is None
    assert killed["resume_command"] and killed["hours_to_finish_if_resumed"] is not None


def test_the_newest_full_local_run_is_found_and_a_smoke_run_is_not_it(ctx):
    assert find_local_run(ctx.tracking_cfg) is None
    with pytest.raises(ConfigError, match="no full training run has been made on this machine"):
        run_status(tracking_cfg=ctx.tracking_cfg)
    smoke = train(ctx, smoke=True)["run_id"]
    assert find_local_run(ctx.tracking_cfg) is None
    assert run_status(smoke, tracking_cfg=ctx.tracking_cfg)["smoke"]  # by name it can be read
    full = train(ctx)["run_id"]
    train(ctx, smoke=True)
    cloud = train(ctx, where="sagemaker")["run_id"]
    assert find_local_run(ctx.tracking_cfg) == full != cloud
    with pytest.raises(ConfigError, match="the training run's record"):
        run_status("20261120T000000Z-train-x-0000", tracking_cfg=ctx.tracking_cfg)


def test_the_module_does_not_load_the_training_stack_or_reach_aws():
    source = Path(local_run_module.__file__).read_text(encoding="utf-8")
    for name in ("import torch", "import transformers", "boto", "sagemaker"):
        assert name not in source
