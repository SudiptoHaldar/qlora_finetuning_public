"""Tests for the one corrective iteration (req_003 v3.05): registering the change (I1 to I5),
selecting across two runs, the launch check J10, and where the iteration stands.

Two stand-in training runs are made and scored under ``tmp_path``. Nothing reaches AWS.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from qlora_finetuning.local_qlora import sagemaker as sagemaker_module
from qlora_finetuning.local_qlora.adapters import read_selection, selected_adapter
from qlora_finetuning.local_qlora.analysis import analyze_run
from qlora_finetuning.local_qlora.iteration import (
    DONE,
    NO_ITERATION,
    NOTHING_SELECTED,
    REGISTERED,
    SECOND_MADE,
    changed_variables,
    iteration_status,
    propose,
)
from qlora_finetuning.local_qlora.iteration_record import read_iteration
from qlora_finetuning.local_qlora.job_record import read_job_record
from qlora_finetuning.local_qlora.sagemaker import JobOptions
from qlora_finetuning.local_qlora.selection import MET, MISSED, select_checkpoint
from qlora_finetuning.settings import ConfigError

from .sagemaker_helpers import make_jobs
from .selection_helpers import TRAIN_JOB, TRAIN_JOB_2, World, scored_world
from .test_sagemaker_selection import measure

MISSING = {2: 1, 4: 2, 6: 3}  # 37.5% at best: under the 40% asked
BETTER = {2: 3, 4: 5, 6: 4}
LOWER_RATE = {"learning_rate": 1e-4}


def check(outcome, check_id):
    return next(item for item in outcome["checks"] if item["id"] == check_id)


def missed_world(root, right=None):
    """A first run that is selected, missed its threshold, and is analysed."""
    world = scored_world(root, right=right or MISSING)
    assert world.select()["verdict"] == MISSED
    assert analyze_run(world.run, **world.stand_ins())["written"]
    return world


def edited(world, **changes):
    """What the owner's edit of config/training.yaml gives."""
    return dataclasses.replace(world.first_cfg, **changes)


def register(world, finding="T2", reason="accuracy peaks early and falls", **changes):
    stand_ins = world.stand_ins() | {"training_cfg": edited(world, **(changes or LOWER_RATE))}
    return propose(finding, reason, **stand_ins)


def iterated(root, second=None, **changes):
    """Both runs made and scored; the change registered; the second not yet selected."""
    world = missed_world(root)
    assert register(world, **changes)["written"]
    world.iterate(second or BETTER, **(changes or LOWER_RATE))
    return world


# --------------------------------------------------------------------------- the config


def test_the_committed_file_names_the_noise_the_records_and_the_table():
    from qlora_finetuning.local_qlora.training_config import (
        ITERATION_VARIABLES,
        load_training_config,
    )

    cfg = load_training_config()
    assert cfg.noise_examples == 10
    assert cfg.iteration_record.name == "iteration.json"
    assert cfg.analysis_record.name == "error_analysis.json"
    assert cfg.iteration_record.parent == cfg.selection_record.parent
    assert cfg.iteration_changes == (
        ("T1", "epochs", 5.0),
        ("T2", "optimizer.learning_rate", 1e-4),
        ("T3", "lora.r", 32.0),
    )
    assert {variable for _, variable, _ in cfg.iteration_changes} <= set(ITERATION_VARIABLES)
    text = json.dumps(cfg.settings(False))
    assert "noise" not in text and "iteration" not in text  # none of it changes a training run
    assert "seed" not in ITERATION_VARIABLES and "batch.size" not in ITERATION_VARIABLES


# --------------------------------------------------------------------------- what changed


SETTINGS = {
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"},
    "optimizer": {"name": "paged_adamw_8bit", "learning_rate": 0.0002},
    "schedule": {"type": "cosine", "warmup_fraction": 0.05},
    "batch": {"size": 2, "accumulation": 8},
    "epochs": 3,
    "seed": 42,
}


def changed(**paths):
    new = json.loads(json.dumps(SETTINGS))
    for path, value in paths.items():
        *parents, last = path.split("__")
        target = new
        for parent in parents:
            target = target[parent]
        target[last] = value
    return changed_variables(SETTINGS, new)


def test_one_allowed_variable_is_named():
    assert changed() == ([], [])
    assert changed(epochs=5) == (["epochs"], [])
    assert changed(optimizer__learning_rate=1e-4) == (["optimizer.learning_rate"], [])
    assert changed(lora__dropout=0.1) == (["lora.dropout"], [])
    assert changed(schedule__warmup_fraction=0.1) == (["schedule.warmup_fraction"], [])
    assert changed(optimizer__learning_rate=2.0e-4) == ([], [])  # the same number, written anew


def test_alpha_follows_r_and_is_not_a_variable_of_its_own():
    assert changed(lora__r=32, lora__alpha=64) == (["lora.r"], [])
    variables, problems = changed(lora__r=32)
    assert variables == ["lora.r"] and "lora.alpha must stay at 2 times lora.r" in problems[0]
    variables, problems = changed(lora__alpha=64)
    assert variables == [] and "must stay at 2 times" in problems[0]


def test_two_variables_and_a_setting_that_may_not_change():
    assert changed(epochs=5, optimizer__learning_rate=1e-4)[0] == [
        "epochs",
        "optimizer.learning_rate",
    ]
    variables, problems = changed(seed=7, batch__size=4)
    assert variables == [] and problems == [
        "batch.size may not change in an iteration",
        "seed may not change in an iteration",
    ]


# --------------------------------------------------------------------------- I1 to I5


def test_a_registered_change_is_written_with_its_reason(tmp_path):
    world = missed_world(tmp_path)
    outcome = register(world, reason="  accuracy peaks at step 4  ")
    assert outcome["passed"] and [item["id"] for item in outcome["checks"]] == [
        f"I{n}" for n in range(1, 6)
    ]
    assert check(outcome, "I3")["detail"] == "optimizer.learning_rate from 0.0002 to 0.0001"
    path = world.first_cfg.iteration_record
    assert outcome["written"] == [str(path)]
    record = read_iteration(path)
    assert record == outcome["record"]
    assert (record["first_run"], record["first_selected"]) == (
        world.run,
        f"{world.run}/checkpoint-6",
    )
    assert (record["variable"], record["from"], record["to"]) == (
        "optimizer.learning_rate",
        0.0002,
        0.0001,
    )
    assert record["finding"] == "T2" and record["in_table"]
    assert record["analysis_finding"] == "T1"  # what the analysis read; the owner chose
    assert record["reason"] == "accuracy peaks at step 4"
    assert record["first_settings_sha256"] == world.first_cfg.settings_sha256(False)
    assert record["settings_sha256"] == edited(world, **LOWER_RATE).settings_sha256(False)
    assert record["second_run"] is None and record["outcome"] is None


def test_a_change_the_table_does_not_name_is_recorded_as_such(tmp_path):
    world = missed_world(tmp_path)
    assert not register(world, learning_rate=5e-5)["record"]["in_table"]
    assert not register(world, finding="other", lora_dropout=0.1)["record"]["in_table"]
    assert register(world, finding="T4", epochs=5)["record"]["in_table"]  # read as T1
    assert register(world, finding="T3", lora_r=32, lora_alpha=64)["record"]["in_table"]


def test_check_writes_nothing_and_a_finding_or_reason_must_be_given(tmp_path):
    world = missed_world(tmp_path)
    stand_ins = world.stand_ins() | {"training_cfg": edited(world, **LOWER_RATE)}
    assert propose("T2", "why", check=True, **stand_ins)["passed"]
    assert not world.first_cfg.iteration_record.exists()
    for finding, reason, message in (
        ("T5", "why", "T5 is a fault to look for, not a change"),
        ("T9", "why", "unknown finding"),
        ("T2", "   ", "a change needs its reason"),
    ):
        with pytest.raises(ConfigError, match=message):
            propose(finding, reason, **stand_ins)


def test_i1_nothing_selected_or_nothing_missed(tmp_path):
    world = scored_world(tmp_path, right=MISSING)
    outcome = register(world)
    assert [item["id"] for item in outcome["checks"]] == ["I1"] and not outcome["passed"]
    assert "no checkpoint has been selected yet" in check(outcome, "I1")["detail"]

    met = scored_world(tmp_path / "met", right={2: 3, 4: 5, 6: 5})
    assert met.select()["verdict"] == MET
    analyze_run(met.run, **met.stand_ins())
    outcome = register(met)
    assert "the verdict is met" in check(outcome, "I1")["detail"]
    assert all(check(outcome, f"I{n}")["passed"] for n in (2, 3, 4, 5))
    assert outcome["written"] == [] and not met.first_cfg.iteration_record.exists()


def test_i2_the_analysis_must_be_of_the_selected_checkpoint(tmp_path):
    world = scored_world(tmp_path, right=MISSING)
    world.select()
    assert "there is no error analysis yet" in check(register(world), "I2")["detail"]
    analyze_run(world.run, **world.stand_ins())
    path = world.first_cfg.analysis_record
    analysis = json.loads(path.read_text(encoding="utf-8"))
    analysis["selected"]["adapter"] = f"{world.run}/checkpoint-2"
    path.write_text(json.dumps(analysis), encoding="utf-8")
    assert "not of the selected" in check(register(world), "I2")["detail"]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"seed": 42}, "config/training.yaml is the first run's: edit one value"),
        ({"learning_rate": 1e-4, "epochs": 5}, "2 variables differ"),
        ({"seed": 7}, "seed may not change in an iteration"),
        ({"lora_r": 32}, "lora.alpha must stay at 2 times lora.r"),
        ({"learning_rate": 1e-4, "batch_size": 4}, "batch.size may not change"),
    ],
)
def test_i3_exactly_one_allowed_variable(tmp_path, changes, message):
    world = missed_world(tmp_path)
    outcome = register(world, **changes)
    assert message in check(outcome, "I3")["detail"] and outcome["written"] == []


def test_i4_the_iteration_is_made_once(tmp_path):
    world = iterated(tmp_path)
    outcome = register(world)
    assert f"has been made ({TRAIN_JOB_2})" in check(outcome, "I4")["detail"]
    world.select()
    assert f"has been made ({world.run})" in check(register(world), "I4")["detail"]


def test_i5_what_the_arms_share_may_not_have_changed(tmp_path):
    world = missed_world(tmp_path)
    other_limit = dataclasses.replace(world.runner, max_new_tokens=64)
    stand_ins = world.stand_ins() | {
        "training_cfg": edited(world, **LOWER_RATE),
        "context": other_limit,
    }
    outcome = propose("T2", "why", **stand_ins)
    assert "the reply limit has changed" in check(outcome, "I5")["detail"]
    assert outcome["written"] == []


# --------------------------------------------------------------------------- selecting again


def test_the_second_run_replaces_the_first_when_it_is_better(tmp_path):
    world = iterated(tmp_path)
    selection_path = world.first_cfg.selection_record
    first_record = json.loads(selection_path.read_text(encoding="utf-8"))

    outcome = world.select()
    change = outcome["iteration"]
    assert outcome["passed"] and outcome["written"]
    assert (change["winner"], change["gap_examples"], change["within_noise"]) == (
        "second",
        2,
        False,
    )
    assert (change["variable"], change["to"]) == ("optimizer.learning_rate", 0.0001)

    record = read_selection(selection_path)
    assert record["schema"] == 2 and record["training_run"] == world.run
    assert record["selected"]["adapter"] == f"{world.run}/checkpoint-4"
    assert record["verdict"] == MET and record["selected_execution_accuracy_percent"] == "62.50"
    assert record["key"]["settings_sha256"] == world.training.training_cfg.settings_sha256(False)
    first, second = record["iterations"]
    assert (first["training_run"], first["verdict"]) == (world.first_run, MISSED)
    assert first["selected"] == first_record["selected"]
    assert (second["training_run"], second["selected"]["step"]) == (world.run, 4)
    assert len(first["checkpoints"]) == len(second["checkpoints"]) == 3
    assert [job["name"] for job in record["jobs"]] == [
        TRAIN_JOB,
        "qlora-score-01234567-a1",
        TRAIN_JOB_2,
        "qlora-score-89abcdef-a1",
    ]  # what arm A1 cost is both runs

    kept = selection_path.with_name("adapter_selection_first.json")
    assert json.loads(kept.read_text(encoding="utf-8")) == first_record
    assert kept.with_suffix(".md").is_file()
    page = selection_path.with_suffix(".md").read_text(encoding="utf-8")
    assert "## The corrective iteration" in page and "more than noise (1 examples)" in page
    assert page.index("## The corrective iteration") < page.index("## What it cost")
    assert f"| second: `{world.run}` **(arm A1)** | 4 | 5 | 8 | met |" in page

    finished = read_iteration(world.first_cfg.iteration_record)
    assert finished["second_run"] == world.run
    assert finished["outcome"] == {
        "winner": "second",
        "gap_examples": 2,
        "within_noise": False,
        "verdict": MET,
        "selected": f"{world.run}/checkpoint-4",
    }
    identity = {name: record["key"][name] for name in ("model_id", "revision", "prompt_id")}
    identity["manifest_hash"] = record["key"]["manifest_hash"]
    assert selected_adapter(selection_path, **identity) == f"{world.run}/checkpoint-4"


def test_a_tie_keeps_the_first_runs_checkpoint(tmp_path):
    world = iterated(tmp_path, second={2: 1, 4: 3, 6: 2})  # 3 at best, as the first run
    outcome = world.select()
    change = outcome["iteration"]
    assert (change["winner"], change["gap_examples"], change["within_noise"]) == ("first", 0, True)
    assert outcome["selected"]["step"] == 4  # this run's own best is still shown

    record = read_selection(world.first_cfg.selection_record)
    assert record["schema"] == 2 and record["training_run"] == world.first_run
    assert record["selected"]["adapter"] == f"{world.first_run}/checkpoint-6"
    assert record["verdict"] == MISSED  # one iteration only: the path goes on as it stands
    assert record["key"]["settings_sha256"] == world.first_cfg.settings_sha256(False)
    assert record["iterations"][1]["training_run"] == world.run
    page = world.first_cfg.selection_record.with_suffix(".md").read_text(encoding="utf-8")
    assert f"| first: `{world.first_run}` **(arm A1)** |" in page
    assert "+0 examples" in page and "the change has not been shown to matter" in page


def test_a_worse_second_run_loses(tmp_path):
    world = iterated(tmp_path, second={2: 1, 4: 1, 6: 2})
    assert world.select()["iteration"]["winner"] == "first"
    assert read_iteration(world.first_cfg.iteration_record)["outcome"]["gap_examples"] == -1


def test_selecting_again_gives_the_same_files(tmp_path):
    world = iterated(tmp_path)
    world.select()
    folder = world.first_cfg.selection_record.parent
    before = {path.name: path.read_bytes() for path in sorted(folder.iterdir())}
    assert world.select()["iteration"]["winner"] == "second"
    assert {path.name: path.read_bytes() for path in sorted(folder.iterdir())} == before


def test_the_first_run_is_not_selected_again_after_an_iteration(tmp_path):
    world = iterated(tmp_path)
    world.select()
    first = world.stand_ins() | {"training_cfg": world.first_cfg}
    with pytest.raises(ConfigError, match="select with the second run's id"):
        select_checkpoint(world.first_run, **first)
    looked = select_checkpoint(world.first_run, check=True, **first)
    assert looked["passed"] and looked["iteration"] is None and looked["written"] == []


def test_a_second_run_that_is_not_registered_is_refused(tmp_path):
    world = missed_world(tmp_path)
    world.iterate(BETTER, **LOWER_RATE)  # trained and scored, and never proposed
    before = world.first_cfg.selection_record.read_bytes()
    with pytest.raises(ConfigError, match="only as the registered corrective iteration"):
        world.select()
    looked = world.select(check=True)
    assert looked["passed"] and looked["iteration"] is None
    assert world.first_cfg.selection_record.read_bytes() == before


def test_a_second_run_made_with_another_change_than_registered_is_refused(tmp_path):
    world = missed_world(tmp_path)
    register(world)  # the learning rate
    world.iterate(BETTER, epochs=5)  # and then more epochs were trained
    with pytest.raises(ConfigError, match="was not made with the registered change"):
        world.select()


# --------------------------------------------------------------------------- where it stands


def test_status_follows_the_iteration_and_checks_the_config_at_its_end(tmp_path):
    fresh = World(tmp_path / "fresh")
    assert iteration_status(**_status_ins(fresh))["state"] == NOTHING_SELECTED

    world = missed_world(tmp_path)
    status = iteration_status(**_status_ins(world))
    assert (status["state"], status["passed"]) == (NO_ITERATION, True)
    edit = _status_ins(world) | {"training_cfg": edited(world, **LOWER_RATE)}
    status = iteration_status(**edit)  # the file is edited, and nothing is registered yet
    assert (
        not status["passed"]
        and "register the change (propose), or revert it" in (status["checks"][0]["detail"])
    )

    register(world)
    status = iteration_status(**edit)
    assert (status["state"], status["passed"], status["checks"]) == (REGISTERED, True, [])
    world.iterate(BETTER, **LOWER_RATE)
    assert iteration_status(**_status_ins(world))["state"] == SECOND_MADE
    world.select()
    status = iteration_status(**_status_ins(world))
    assert (status["state"], status["passed"]) == (DONE, True)
    assert status["iteration"]["outcome"]["winner"] == "second"


def test_when_the_first_run_stays_the_change_must_be_reverted(tmp_path):
    world = iterated(tmp_path, second={2: 1, 4: 1, 6: 2})
    world.select()
    status = iteration_status(**_status_ins(world))  # the config still holds the change
    assert status["state"] == DONE and not status["passed"]
    assert "revert the change and commit" in status["checks"][0]["detail"]
    reverted = _status_ins(world) | {"training_cfg": world.first_cfg}
    assert iteration_status(**reverted)["passed"]


def _status_ins(world):
    stand_ins = world.stand_ins()
    return {name: stand_ins[name] for name in ("context", "training_cfg", "sagemaker_cfg")}


# --------------------------------------------------------------------------- J10


def plan_train(tmp_path, monkeypatch, cfg, **options):
    jobs = make_jobs(tmp_path, monkeypatch)
    monkeypatch.setattr(sagemaker_module, "load_training_config", lambda root=None: cfg)
    plan = jobs.plan(JobOptions("train", **options))
    return jobs, next(item for item in plan.checks if item["id"] == "J10")


def test_j10_passes_for_a_first_run_and_for_what_is_not_a_full_training(tmp_path, monkeypatch):
    world = World(tmp_path)
    jobs, result = plan_train(tmp_path, monkeypatch, world.first_cfg)
    assert result["status"] == "PASS" and "no run is selected yet" in result["detail"]

    world = missed_world(tmp_path / "selected")
    jobs, _ = plan_train(tmp_path / "selected", monkeypatch, world.first_cfg)
    for options in (
        JobOptions("smoke"),
        JobOptions("score", run=world.run, workspace=TRAIN_JOB),
        JobOptions("train", resume=world.run, workspace=TRAIN_JOB),
    ):
        result = next(item for item in jobs.plan(options).checks if item["id"] == "J10")
        assert result["status"] == "PASS" and "not a full training run" in result["detail"]


def test_j10_refuses_a_second_run_that_is_not_registered(tmp_path, monkeypatch):
    world = missed_world(tmp_path)
    _, result = plan_train(tmp_path, monkeypatch, world.first_cfg, workspace=TRAIN_JOB)
    assert result["status"] == "FAIL" and "is registered first" in result["detail"]

    register(world)  # the learning rate; but the file now holds more epochs
    other = edited(world, epochs=5)
    _, result = plan_train(tmp_path, monkeypatch, other, workspace=TRAIN_JOB)
    assert (
        result["status"] == "FAIL"
        and "python -m scripts.qlora_iterate propose" in (result["detail"])
    )


def test_j10_passes_the_registered_change_on_the_first_jobs_workspace(tmp_path, monkeypatch):
    world = missed_world(tmp_path)
    register(world)
    cfg = edited(world, **LOWER_RATE)
    _, result = plan_train(tmp_path, monkeypatch, cfg, workspace=TRAIN_JOB)
    assert result["status"] == "PASS"
    assert "optimizer.learning_rate from 0.0002 to 0.0001 (finding T2)" in result["detail"]
    for options in ({}, {"workspace": "qlora-train-other-a1"}):
        _, result = plan_train(tmp_path, monkeypatch, cfg, **options)
        assert result["status"] == "FAIL" and f"add --workspace {TRAIN_JOB}" in result["detail"]


def test_j10_refuses_a_third_run_and_allows_a_failed_second_to_be_made_again(tmp_path, monkeypatch):
    world = iterated(tmp_path)
    cfg = world.training.training_cfg
    _, result = plan_train(tmp_path, monkeypatch, cfg, workspace=TRAIN_JOB)
    assert result["status"] == "FAIL" and f"has been made ({TRAIN_JOB_2})" in result["detail"]

    record = read_job_record(world.sagemaker_cfg.record_file)
    for job in record["jobs"]:
        if job["name"] == TRAIN_JOB_2:
            job["status"] = "Failed"
    Path(world.sagemaker_cfg.record_file).write_text(json.dumps(record), encoding="utf-8")
    _, result = plan_train(tmp_path, monkeypatch, cfg, workspace=TRAIN_JOB)
    assert result["status"] == "PASS"


def test_a_launched_second_run_is_marked_in_the_job_record(tmp_path, monkeypatch):
    world = missed_world(tmp_path)
    register(world)
    jobs = make_jobs(tmp_path, monkeypatch)
    measure(jobs)  # the smoke job, so that the plan has an estimate
    assert "iteration" not in read_job_record(jobs.cfg.record_file)["jobs"][-1]
    cfg = edited(world, **LOWER_RATE)
    monkeypatch.setattr(sagemaker_module, "load_training_config", lambda root=None: cfg)
    plan, job = jobs.launch(JobOptions("train", workspace=TRAIN_JOB), yes=True)
    assert plan.passed and job["iteration"] is True and job["workspace"] == TRAIN_JOB
    # The stand-in commit gives the job the name the world's first training job has, so
    # the entry is replaced in place: look it up by name.
    stored = next(
        item
        for item in read_job_record(jobs.cfg.record_file)["jobs"]
        if item["name"] == job["name"]
    )
    assert stored["iteration"] is True and stored["arn"] == job["arn"]
