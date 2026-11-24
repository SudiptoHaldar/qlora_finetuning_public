"""Tests for qlora_finetuning.local_qlora.selection: which checkpoint becomes arm A1 (v3.04).

A stand-in training run is scored by stand-in models through the real job task and runner;
the selection then reads the real run folders, registry, ledger, and job record.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from qlora_finetuning.evaluation.predictor import describe
from qlora_finetuning.local_qlora import backend as backend_module
from qlora_finetuning.local_qlora.adapters import (
    ADAPTER_WEIGHTS,
    SELECTED,
    adapter_sha256,
    read_selection,
    selected_adapter,
)
from qlora_finetuning.local_qlora.predictor import build_tuned_predictor
from qlora_finetuning.local_qlora.selection import MET, MISSED, render_report, select_checkpoint
from qlora_finetuning.local_qlora.training_config import load_training_config
from qlora_finetuning.settings import ConfigError

from .conftest import StubBackend
from .selection_helpers import NO_FORMAT, SCORE_JOB, TRAIN_JOB, World, scored_world
from .training_helpers import holds_example_text

RIGHT = {2: 3, 4: 5, 6: 5}  # of 8: checkpoints 4 and 6 tie on execution accuracy


def check(outcome, check_id):
    return next(item for item in outcome["checks"] if item["id"] == check_id)


def record_of(world):
    path = world.training.training_cfg.selection_record
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


# --------------------------------------------------------------------------- the config


def test_the_committed_file_says_what_arm_a1_must_reach():
    cfg = load_training_config()
    assert cfg.selection_split == "dev"
    assert (cfg.min_execution_accuracy_percent, cfg.min_format_validity_percent) == (40, 98)
    assert cfg.selection_record.name == "adapter_selection.json"
    assert cfg.selection_record.parent.name == "local_qlora"
    assert "selection" not in json.dumps(cfg.settings(False))  # it changes no training run


# --------------------------------------------------------------------------- the choice


def test_the_best_execution_accuracy_wins_and_a_tie_goes_to_the_earlier_step(tmp_path):
    world = scored_world(tmp_path, right=RIGHT)
    outcome = world.select()
    assert outcome["passed"] and [item["id"] for item in outcome["checks"]] == [
        f"L{n}" for n in range(1, 7)
    ]
    assert outcome["selected"]["step"] == 4 and outcome["verdict"] == MET
    assert outcome["selected"]["adapter"] == f"{world.run}/checkpoint-4"
    assert [row["execution_accuracy"] for row in outcome["rows"]] == [3, 5, 5]
    assert outcome["untuned"]["execution_accuracy"] == 0 and not outcome["rehearsal"]
    assert outcome["notes"] == {
        "selected_is_last_checkpoint": False,
        "lowest_dev_loss_step": 6,  # the loss is shown, and did not choose
        "execution_gain_points_over_untuned": "62.50",
    }


def test_a_tie_on_execution_goes_to_the_higher_format_validity(tmp_path):
    world = scored_world(tmp_path, right={2: 5, 4: 5, 6: 4}, wrong={2: NO_FORMAT})
    outcome = world.select()
    assert [row["format_validity"] for row in outcome["rows"]] == [5, 8, 8]
    assert outcome["selected"]["step"] == 4


def test_the_last_checkpoint_being_best_is_noted(tmp_path):
    outcome = scored_world(tmp_path, right={2: 3, 4: 4, 6: 6}).select()
    assert outcome["selected"]["step"] == 6
    assert outcome["notes"]["selected_is_last_checkpoint"]


# --------------------------------------------------------------------------- the record


def test_the_record_says_which_run_is_arm_a1(tmp_path):
    world = scored_world(tmp_path, right=RIGHT)
    outcome = world.select()
    path = world.training.training_cfg.selection_record
    assert outcome["written"] == [str(path), str(path.with_suffix(".md"))]
    record = read_selection(path)
    assert record == outcome["record"] and record["training_run"] == world.run
    assert record["chain"] == [world.run] and record["split"] == "dev"
    selected = record["selected"]
    assert selected["adapter_sha256"] == adapter_sha256(world.checkpoint(4))
    assert selected["examples"] == 8 and selected["dev_loss"] == 0.25
    assert record["verdict"] == MET
    assert record["selected_execution_accuracy_percent"] == "62.50"
    assert record["selected_format_validity_percent"] == "100.00"
    assert record["thresholds"] == {
        "min_execution_accuracy_percent": 40,
        "min_format_validity_percent": 98,
    }
    assert [row["step"] for row in record["checkpoints"]] == [2, 4, 6]
    assert len({row["eval_run_id"] for row in record["checkpoints"]}) == 3
    assert record["untuned"]["adapter"] is None
    assert record["jobs"] == [
        {"name": TRAIN_JOB, "task": "train", "instance_type": "ml.g5.xlarge",
         "billable_seconds": 9000, "cost_usd": "3.52"},
        {"name": SCORE_JOB, "task": "score", "instance_type": "ml.g5.xlarge",
         "billable_seconds": 12000, "cost_usd": "4.69"},
    ]  # fmt: skip
    assert record["key"]["prompt_id"] == world.runner.prompt.prompt_id
    assert set(record["key"]) == {
        "model_id", "revision", "prompt_id", "manifest_hash", "settings_sha256"
    }  # fmt: skip


def test_the_page_shows_every_checkpoint_and_marks_the_selected_one(tmp_path):
    world = scored_world(tmp_path, right=RIGHT)
    outcome = world.select()
    page = Path(outcome["written"][1]).read_text(encoding="utf-8")
    assert page == render_report(outcome["record"])
    assert f"**Selected**: checkpoint 4 (`{world.run}/checkpoint-4`)" in page
    assert "**Verdict**: met. Execution accuracy 62.50% (at least 40% asked)" in page
    assert "| untuned | 0 | 0 |" in page and "| 4 **(selected)** | 5 | 8 |" in page
    assert "The loss on dev is shown and is never used." in page
    assert "- The selected checkpoint is not the last one." in page
    assert "| `qlora-score-01234567-a1` | score | ml.g5.xlarge | 12,000 | $4.69 |" in page


def test_the_same_runs_give_the_same_files_and_no_example_text(tmp_path):
    world = scored_world(tmp_path, right=RIGHT)
    files = [Path(name) for name in world.select()["written"]]
    before = [path.read_bytes() for path in files]
    world.select()
    assert [path.read_bytes() for path in files] == before  # no time stamp, nothing random
    assert all(b"\r\n" not in content for content in before)
    assert holds_example_text(files[0].parent) == []
    text = files[0].read_text(encoding="utf-8")
    assert "PROGRAM" not in text and "ANSWER" not in text  # counts, never a reply


# --------------------------------------------------------------------------- the verdict


def test_under_the_execution_threshold_the_verdict_is_missed_and_is_still_recorded(tmp_path):
    world = scored_world(tmp_path, right={2: 1, 4: 2, 6: 3})  # 37.5% at best: 40% is asked
    outcome = world.select()
    assert outcome["passed"] and outcome["verdict"] == MISSED and outcome["written"]
    assert record_of(world)["verdict"] == MISSED
    assert "**Verdict**: missed." in Path(outcome["written"][1]).read_text(encoding="utf-8")


def test_under_the_format_threshold_the_verdict_is_missed(tmp_path):
    world = scored_world(tmp_path, right={2: 4, 4: 6, 6: 5}, wrong={4: NO_FORMAT})
    outcome = world.select()
    assert outcome["selected"]["step"] == 4 and outcome["selected"]["format_validity"] == 6
    assert outcome["verdict"] == MISSED  # 75% are in the format: 98% is asked


def test_the_thresholds_are_the_configured_ones(tmp_path):
    world = scored_world(tmp_path, right={2: 1, 4: 2, 6: 3})
    lenient = dataclasses.replace(world.training.training_cfg, min_execution_accuracy_percent=37.5)
    outcome = select_checkpoint(world.run, **(world.stand_ins() | {"training_cfg": lenient}))
    assert outcome["verdict"] == MET  # exactly at the threshold


# --------------------------------------------------------------------------- the checks


def test_l1_nothing_scored_yet(tmp_path):
    world = World(tmp_path)
    outcome = world.select()
    assert not outcome["passed"] and outcome["rows"] == [] and outcome["selected"] is None
    assert [item["id"] for item in outcome["checks"]] == ["L1"]
    assert "checkpoint 2 has 0 complete results on dev" in check(outcome, "L1")["detail"]
    assert "the untuned model has 0 complete results" in check(outcome, "L1")["detail"]
    assert outcome["written"] == [] and record_of(world) is None


def test_l1_one_checkpoint_is_missing(tmp_path):
    world = World(tmp_path, right=RIGHT)
    result = world.score()
    world.record_jobs(result)
    shutil.rmtree(world.runs_dir / result["runs"][2]["run_id"])
    outcome = world.select()
    assert check(outcome, "L1")["detail"] == "checkpoint 4 has 0 complete results on dev"
    assert outcome["written"] == []


def test_l3_weights_that_changed_after_the_scoring(tmp_path):
    world = scored_world(tmp_path, right=RIGHT)
    (world.checkpoint(2) / ADAPTER_WEIGHTS).write_bytes(b"other weights")
    outcome = world.select()
    assert not check(outcome, "L3")["passed"] and not outcome["passed"]
    assert "checkpoint 2 was scored with other weights" in check(outcome, "L3")["detail"]
    assert outcome["written"] == [] and record_of(world) is None


def test_l3_without_the_weights_here_the_jobs_recorded_hash_is_used(tmp_path):
    world = scored_world(tmp_path, right=RIGHT)
    shutil.rmtree(world.runs_dir / world.run / "checkpoints")  # as after a pull without them
    assert world.select()["passed"]
    result = world.root / "artifacts" / "jobs" / SCORE_JOB / "job_result.json"
    result.unlink()
    outcome = world.select()
    assert "no recorded hash for checkpoint 2" in check(outcome, "L3")["detail"]


def test_l4_a_run_made_under_other_settings_or_cut_short(tmp_path):
    world = scored_world(tmp_path, right=RIGHT)
    other = dataclasses.replace(world.training.training_cfg, learning_rate=0.5)
    outcome = select_checkpoint(world.run, **(world.stand_ins() | {"training_cfg": other}))
    assert "another experiment (settings_sha256 differ)" in check(outcome, "L4")["detail"]
    assert outcome["written"] == []

    measurements = world.runs_dir / world.run / "measurements.json"
    content = json.loads(measurements.read_text(encoding="utf-8"))
    measurements.write_text(json.dumps(content | {"steps_done": 5}), encoding="utf-8")
    outcome = world.select()
    assert "the run ended at step 5, of 6 planned" in check(outcome, "L4")["detail"]


def test_l5_runs_made_on_this_machine_are_not_arm_a1(tmp_path):
    world = scored_world(tmp_path, right=RIGHT, where="local", scope="dev")
    outcome = world.select()
    assert not check(outcome, "L5")["passed"] and outcome["written"] == []
    assert "is local and dev, not sagemaker and measured" in check(outcome, "L5")["detail"]
    assert all(check(outcome, f"L{n}")["passed"] for n in (1, 2, 3, 4, 6))
    assert outcome["selected"]["step"] == 4  # the table is still shown


@pytest.mark.parametrize(
    ("jobs", "message"),
    [
        ({"pulled": False}, f"job {TRAIN_JOB} is not pulled"),
        ({"billed": False}, "its billable seconds are not recorded"),
    ],
)
def test_l5_a_job_that_is_not_pulled_or_not_billed(tmp_path, jobs, message):
    world = World(tmp_path, right=RIGHT)
    world.record_jobs(world.score(), **jobs)
    outcome = world.select()
    assert message in check(outcome, "L5")["detail"] and outcome["written"] == []


def test_l5_a_run_that_no_job_made(tmp_path):
    world = World(tmp_path, right=RIGHT)
    world.score()
    outcome = world.select()  # no job record at all
    assert "belongs to no job in the job record" in check(outcome, "L5")["detail"]


# --------------------------------------------------------------------------- check, rehearsal


def test_check_reports_and_writes_nothing(tmp_path):
    world = scored_world(tmp_path, right=RIGHT)
    outcome = world.select(check=True)
    assert outcome["passed"] and outcome["selected"]["step"] == 4 and outcome["verdict"] == MET
    assert outcome["written"] == [] and record_of(world) is None


def test_a_rehearsal_on_partial_runs_is_allowed_only_as_a_check(tmp_path):
    world = World(tmp_path, right={2: 1, 4: 3, 6: 2})
    world.record_jobs(world.score(limit=3))
    for options in ({"limit": 3}, {"split": "quick_dev"}):
        with pytest.raises(ConfigError, match="it is a rehearsal: add --check"):
            world.select(**options)
    outcome = world.select(limit=3, check=True)
    assert outcome["rehearsal"] and outcome["limit"] == 3 and outcome["passed"]
    assert outcome["selected"]["step"] == 4 and outcome["selected"]["examples"] == 3
    assert outcome["written"] == []
    # The partial runs are not the complete ones a selection is made from.
    assert not world.select(check=True)["passed"]


def test_a_smoke_run_is_never_arm_a1(tmp_path):
    world = World(tmp_path, smoke=True)
    with pytest.raises(ConfigError, match="is a smoke run: its checkpoints are never arm A1"):
        world.select()
    assert world.select(check=True)["rehearsal"]


def test_an_unknown_run_is_a_config_error(tmp_path):
    world = World(tmp_path)
    with pytest.raises(ConfigError, match="the training run's record"):
        select_checkpoint("20261120T000000Z-train-x-0000", **world.stand_ins())


# --------------------------------------------------------------------------- --adapter selected


@pytest.fixture
def loaded(monkeypatch):
    seen = {}

    def fake_load(cfg, budget, adapter_dir=None):
        seen.update(adapter_dir=adapter_dir, calls=seen.get("calls", 0) + 1)
        return StubBackend(settings=cfg.settings(budget.revision, adapter="a" * 64))

    monkeypatch.setattr(backend_module, "load_backend", fake_load)
    return seen


def test_selected_names_the_checkpoint_of_the_record(tmp_path, loaded):
    world = scored_world(tmp_path, right=RIGHT)
    with pytest.raises(ConfigError, match="no checkpoint has been selected yet"):
        build_tuned_predictor(SELECTED, world.runner)
    world.select()
    predictor = build_tuned_predictor(SELECTED, world.runner)
    assert loaded == {"adapter_dir": world.checkpoint(4), "calls": 1}
    details = describe(predictor)["details"]
    assert (details["training_run"], details["checkpoint"]) == (world.run, 4)


def test_selected_is_refused_when_the_weights_or_the_experiment_changed(tmp_path, loaded):
    world = scored_world(tmp_path, right=RIGHT)
    world.select()
    path = world.training.training_cfg.selection_record
    record = json.loads(path.read_text(encoding="utf-8"))

    (world.checkpoint(4) / ADAPTER_WEIGHTS).write_bytes(b"other weights")
    with pytest.raises(ConfigError, match="not the ones the selection was made on"):
        build_tuned_predictor(SELECTED, world.runner)

    identity = {name: record["key"][name] for name in ("model_id", "revision", "manifest_hash")}
    assert selected_adapter(path, prompt_id=record["key"]["prompt_id"], **identity) == (
        f"{world.run}/checkpoint-4"
    )
    with pytest.raises(ConfigError, match=r"another experiment \(prompt_id differ\)"):
        selected_adapter(path, prompt_id="f" * 64, **identity)

    path.write_text(json.dumps(record | {"schema": 99}), encoding="utf-8")
    with pytest.raises(ConfigError, match="is damaged"):
        build_tuned_predictor(SELECTED, world.runner)
    assert "calls" not in loaded  # every refusal came before the model was loaded
