"""Tests for the score task: every checkpoint of a training run, each scored once (v3.04).

The models are stand-ins; the job task, the evaluation runner, and the registry are real.
"""

from __future__ import annotations

import json
import shutil

import pytest

from qlora_finetuning.local_qlora.adapters import (
    TRAINING_NAME,
    adapter_sha256,
    chain_checkpoints,
    training_chain,
)
from qlora_finetuning.local_qlora.job import run_task
from qlora_finetuning.local_qlora.scoring import (
    COMPLETE,
    INCOMPLETE,
    expected_settings,
    find_evaluation,
)
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking import load_manifest
from scripts import job_run

from .selection_helpers import SCORE_JOB, World
from .training_helpers import write_checkpoint

LATER = "20261121T000000Z-train-qwen3-4b-fit-bbbb"


@pytest.fixture
def world(tmp_path):
    return World(tmp_path, right={2: 3, 4: 5, 6: 4})


def test_the_untuned_model_then_every_checkpoint_in_step_order(world):
    result = world.score()
    assert result["status"] == "completed" and result["job_name"] == SCORE_JOB
    runs = result["runs"]
    assert [run["role"] for run in runs] == ["score-untuned"] + ["score-checkpoint"] * 3
    assert [run["step"] for run in runs] == [None, 2, 4, 6]
    assert [run["execution_accuracy"] for run in runs] == [0, 3, 5, 4]
    assert world.built == [None, *(f"{world.run}/checkpoint-{step}" for step in (2, 4, 6))]
    for run in runs:
        assert (run["complete"], run["result"], run["reused"]) == (True, True, False)
        assert run["training_run"] == world.run and run["examples_predicted"] == 8
        manifest = load_manifest(run["run_id"], world.runner.tracking_cfg)
        assert (manifest["where"], manifest["scope"]) == ("sagemaker", "measured")
    assert runs[0]["adapter"] is None and runs[0]["adapter_sha256"] is None
    for run in runs[1:]:
        assert run["adapter"] == f"{world.run}/checkpoint-{run['step']}"
        assert run["adapter_sha256"] == adapter_sha256(world.checkpoint(run["step"]))


def test_what_is_already_scored_is_not_scored_again(world):
    first = world.score()
    asked = len(world.built)
    again = world.score()
    assert len(world.built) == asked  # no model was loaded
    assert [run["reused"] for run in again["runs"]] == [True] * 4
    assert [run["run_id"] for run in again["runs"]] == [run["run_id"] for run in first["runs"]]
    assert [run["execution_accuracy"] for run in again["runs"]] == [0, 3, 5, 4]


def test_a_missing_score_is_made_and_the_others_are_kept(world):
    first = world.score()
    shutil.rmtree(world.runs_dir / first["runs"][2]["run_id"])  # checkpoint 4's run is lost
    again = world.score()
    assert [run["reused"] for run in again["runs"]] == [True, True, False, True]
    assert again["runs"][2]["run_id"] != first["runs"][2]["run_id"]


def test_another_limit_is_another_question(world):
    world.score()
    short = world.score(limit=3)
    assert [run["reused"] for run in short["runs"]] == [False] * 4
    assert all(run["examples_predicted"] == 3 and not run["result"] for run in short["runs"])
    covering = world.score(limit=99)  # a limit that covers the split is no limit
    assert [run["reused"] for run in covering["runs"]] == [True] * 4


def test_a_run_that_was_cut_short_is_resumed(world):
    seen = []

    def finder(ctx, *, split, limit, settings):
        return (INCOMPLETE, "an-earlier-run", None) if settings["adapter"] else (None, None, None)

    def evaluator(predictor, split, **options):
        seen.append(options["resume"])
        return {
            "run_id": f"20261120T10000{len(seen)}Z-eval-x",
            "predictor": {"name": "qwen3-4b"},
            "examples_predicted": 8,
            "complete": True,
            "result": True,
            "all": {"format_validity": {"count": 8}, "execution_accuracy": {"count": 1}},
        }

    result = world.score(finder=finder, evaluator=evaluator)
    assert seen == [None, "an-earlier-run", "an-earlier-run", "an-earlier-run"]
    assert result["status"] == "completed" and not any(run["reused"] for run in result["runs"])


def test_find_evaluation_tells_complete_from_cut_short_from_nothing(world):
    world.score(limit=3)
    ctx = world.runner
    from qlora_finetuning.data.token_budget import load_token_budget_config
    from qlora_finetuning.local_qlora.config import load_local_qlora_config

    cfg, revision = load_local_qlora_config(), load_token_budget_config().revision
    untuned = expected_settings(cfg, revision, None)
    state, run_id, summary = find_evaluation(ctx, split="dev", limit=3, settings=untuned)
    assert state == COMPLETE and summary["run_id"] == run_id
    assert find_evaluation(ctx, split="dev", limit=None, settings=untuned) == (None, None, None)
    other = expected_settings(cfg, revision, "f" * 64)
    assert find_evaluation(ctx, split="dev", limit=3, settings=other) == (None, None, None)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"run": None}, "a score task needs --run"),
        ({"run": "20261120T000000Z-train-x-0000"}, "the training run's record"),
    ],
)
def test_a_score_task_that_cannot_start(world, options, message):
    result = world.score(**options)
    assert result["status"] == "config_error" and message in result["error"]
    assert result["runs"] == [] and world.built == []


def test_a_run_whose_checkpoints_are_not_here_is_refused(world):
    shutil.rmtree(world.runs_dir / world.run / "checkpoints")
    result = world.score()
    assert result["status"] == "config_error" and "has no checkpoint here" in result["error"]


# --------------------------------------------------------------------------- a resumed training


def continue_run(world, steps):
    """A later run that continues the world's, holding checkpoints at ``steps``."""
    folder = world.runs_dir / LATER
    folder.mkdir()
    (folder / TRAINING_NAME).write_text(json.dumps({"continues": world.run}), encoding="utf-8")
    for step in steps:
        write_checkpoint(folder / "checkpoints" / f"checkpoint-{step}", step + 100)


def test_checkpoints_come_from_the_whole_chain_and_a_later_run_wins(world):
    continue_run(world, steps=(6, 8))
    assert training_chain(LATER, world.runs_dir) == [world.run, LATER]
    assert chain_checkpoints(LATER, world.runs_dir) == {
        2: f"{world.run}/checkpoint-2",
        4: f"{world.run}/checkpoint-4",
        6: f"{LATER}/checkpoint-6",
        8: f"{LATER}/checkpoint-8",
    }
    assert list(chain_checkpoints(world.run, world.runs_dir)) == [2, 4, 6]


def test_a_chain_that_loops_or_breaks_is_refused(world):
    record = world.runs_dir / world.run / TRAINING_NAME
    training = json.loads(record.read_text(encoding="utf-8"))
    record.write_text(json.dumps(training | {"continues": world.run}), encoding="utf-8")
    with pytest.raises(ConfigError, match="its chain loops"):
        training_chain(world.run, world.runs_dir)
    record.write_text(json.dumps(training | {"continues": "a-run-not-here"}), encoding="utf-8")
    with pytest.raises(ConfigError, match="the training run's record"):
        training_chain(world.run, world.runs_dir)


# --------------------------------------------------------------------------- the script


def test_job_run_prints_each_step_and_says_what_was_already_scored(world, capsys):
    arguments = ["score", "--run", world.run, "--split", "dev", "--where", "sagemaker"]
    options = {
        "root": world.root,
        "check_pins": False,
        "runner_context": world.runner,
        "factory": world.factory,
    }
    assert job_run.main(arguments, **options) == 0
    out = capsys.readouterr().out
    assert "task score" in out and "score-untuned: run " in out
    assert out.count("score-checkpoint: run ") == 3 and "; step 4; completed; " in out
    assert "already scored" not in out

    assert job_run.main(arguments, **options) == 0
    assert capsys.readouterr().out.count("already scored") == 4
    assert job_run.main(["score"], **options) == 2  # no --run: a config error


def test_run_task_knows_the_task():
    from qlora_finetuning.local_qlora.job import TASKS

    assert "score" in TASKS and run_task.__kwdefaults__["run"] is None
