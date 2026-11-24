"""Tests for qlora_finetuning.local_qlora.job, with stand-in training and evaluation."""

from __future__ import annotations

import json

import pytest

from qlora_finetuning.evaluation.predictor import PredictorError
from qlora_finetuning.local_qlora import job as job_module
from qlora_finetuning.local_qlora.job import (
    PREDICTORS,
    TUNED,
    UNTUNED,
    pin_differences,
    result_path,
    run_task,
)
from qlora_finetuning.local_qlora.training import TrainingError
from qlora_finetuning.settings import ConfigError
from tests.evaluation.conftest import make_context

from .training_helpers import make_training_context

TRAIN_RUN = "20261120T100000Z-smoke-qwen3-4b-smoke-aaaa"


class Recorder:
    """Stands in for run_training, run_evaluation, and the predictor factory."""

    def __init__(self, train_passes=True, train_error=None, eval_error=None):
        self.train_passes, self.train_error, self.eval_error = train_passes, train_error, eval_error
        self.trained: list[dict] = []
        self.evaluated: list[dict] = []
        self.built: list[tuple] = []

    def trainer(self, **options):
        self.trained.append(options)
        if self.train_error:
            raise self.train_error
        return {
            "run_id": TRAIN_RUN,
            "kind": "smoke" if options["smoke"] else "train",
            "continues": options.get("resume"),
            "steps": 12,
            "adapter": TRAIN_RUN,
            "adapter_sha256": "a" * 64,
            "passed": self.train_passes,
            "checks": [{"id": "K6", "passed": self.train_passes}, {"id": "K1", "passed": True}],
            "measurements": {"probe": {"width": 1862}, "trainer": {"seconds_per_step": 14.0}},
        }

    def factory(self, name, adapter, context):
        self.built.append((name, adapter, tuple(context.configs)))
        return object()

    def evaluator(self, predictor, split, **options):
        self.evaluated.append({"split": split, **options})
        if self.eval_error:
            raise self.eval_error
        number = len(self.evaluated)
        return {
            "run_id": f"20261120T10000{number}Z-eval-{number}",
            "predictor": {"name": options["label"].split()[0]},
            "examples_predicted": options["limit"] or 499,
            "complete": True,
            "result": options["limit"] is None,
            "all": {"format_validity": {"count": 20}, "execution_accuracy": {"count": 9}},
        }


@pytest.fixture
def contexts(tmp_path):
    return {
        "root": tmp_path,
        "check_pins": False,
        "runner_context": make_context(tmp_path),
        "training_context": make_training_context(tmp_path / "training"),
    }


def run(task, recorder, contexts, **options):
    return run_task(
        task,
        trainer=recorder.trainer,
        evaluator=recorder.evaluator,
        factory=recorder.factory,
        **contexts,
        **options,
    )


def stored(result, tmp_path):
    return json.loads(result_path(result["job_name"], tmp_path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- the tasks


def test_smoke_trains_then_scores_with_the_adapter_and_without(contexts, tmp_path):
    recorder = Recorder()
    result = run("smoke", recorder, contexts, where="sagemaker")
    assert result["status"] == "completed" and result["error"] is None
    (trained,) = recorder.trained
    assert (trained["smoke"], trained["where"], trained["scope"]) == (True, "sagemaker", "dev")

    assert recorder.built == [
        (TUNED, TRAIN_RUN, recorder.built[0][2]),
        (UNTUNED, None, recorder.built[1][2]),
    ]
    assert {"local_qlora", "training"} <= set(recorder.built[0][2])
    tuned, untuned = recorder.evaluated
    for call, arm in ((tuned, "A1"), (untuned, "A0")):
        assert (call["split"], call["limit"], call["where"]) == ("quick_dev", 24, "sagemaker")
        assert (call["path"], call["arm"], call["kind"], call["resume"]) == ("A", arm, "eval", None)

    assert [run_["role"] for run_ in result["runs"]] == ["train", "eval-tuned", "eval-untuned"]
    assert result["runs"][0]["steps"] == 12 and result["runs"][0]["checks_failed"] == []
    assert (
        result["runs"][1]["adapter"] == TRAIN_RUN and result["runs"][1]["execution_accuracy"] == 9
    )
    assert (result["task"], result["where"], result["job_name"]) == (
        "smoke",
        "sagemaker",
        "local-smoke",
    )
    assert stored(result, tmp_path) == json.loads(json.dumps(result))


def test_a_failed_smoke_check_fails_the_task_after_the_scoring(contexts):
    recorder = Recorder(train_passes=False)
    result = run("smoke", recorder, contexts)
    assert result["status"] == "failed" and result["error"] == "a smoke check failed: K6"
    assert len(recorder.evaluated) == 2  # the adapter exists, so it is still asked


def test_train_runs_the_full_run_or_continues_one(contexts):
    recorder = Recorder()
    result = run("train", recorder, contexts, where="sagemaker", scope="measured")
    (trained,) = recorder.trained
    assert (trained["smoke"], trained["resume"], trained["scope"]) == (False, None, "measured")
    assert recorder.evaluated == [] and result["runs"][0]["kind"] == "train"

    again = Recorder()
    run("train", again, contexts, resume="20261120T000000Z-train-x-0000")
    assert again.trained[0]["resume"] == "20261120T000000Z-train-x-0000"


def test_evaluate_runs_one_evaluation(contexts):
    recorder = Recorder()
    adapter = f"{TRAIN_RUN}/checkpoint-300"
    result = run(
        "evaluate", recorder, contexts, predictor=TUNED, adapter=adapter, split="dev", resume="r1"
    )
    assert recorder.trained == [] and recorder.built[0][:2] == (TUNED, adapter)
    (call,) = recorder.evaluated
    assert (call["split"], call["limit"], call["resume"], call["arm"]) == ("dev", None, "r1", "A1")
    (entry,) = result["runs"]
    assert (entry["role"], entry["split"], entry["result"], entry["adapter"]) == (
        "eval",
        "dev",
        True,
        adapter,
    )


# --------------------------------------------------------------------------- failures


def test_a_training_failure_is_recorded_with_its_run(contexts, tmp_path):
    error = TrainingError("OutOfMemoryError after 0 of 12 steps")
    error.run_id = TRAIN_RUN
    result = run("smoke", Recorder(train_error=error), contexts)
    assert result["status"] == "failed" and "OutOfMemoryError after 0 of 12" in result["error"]
    assert result["runs"] == [{"role": "train", "run_id": TRAIN_RUN, "status": "failed"}]
    assert stored(result, tmp_path)["status"] == "failed"  # the file is written all the same


def test_a_predictor_failure_is_recorded(contexts):
    error = PredictorError("RuntimeError while generating a batch of 8")
    error.run_id = "20261120T100009Z-eval-x"
    result = run("evaluate", Recorder(eval_error=error), contexts, predictor=UNTUNED, split="dev")
    assert result["status"] == "failed"
    assert result["runs"] == [{"role": "eval", "run_id": error.run_id, "status": "failed"}]


@pytest.mark.parametrize(
    ("task", "options", "message"),
    [
        ("deploy", {}, "unknown task 'deploy'"),
        ("evaluate", {"split": "dev"}, "needs --predictor and --split"),
        ("evaluate", {"predictor": UNTUNED}, "needs --predictor and --split"),
        ("evaluate", {"predictor": "nova-lite", "split": "dev"}, "evaluates qwen3-4b or"),
    ],
)
def test_a_task_that_cannot_start_is_a_config_error_and_starts_nothing(
    contexts, task, options, message
):
    recorder = Recorder()
    result = run(task, recorder, contexts, **options)
    assert result["status"] == "config_error" and message in result["error"]
    assert recorder.trained == [] and recorder.evaluated == [] and result["runs"] == []


def test_a_refused_adapter_is_a_config_error(contexts):
    recorder = Recorder()

    def refuse(name, adapter, context):
        raise ConfigError("adapter x was trained for another prompt wording")

    result = run_task(
        "evaluate",
        predictor=TUNED,
        adapter="x",
        split="dev",
        trainer=recorder.trainer,
        evaluator=recorder.evaluator,
        factory=refuse,
        **contexts,
    )
    assert result["status"] == "config_error" and "another prompt wording" in result["error"]
    assert recorder.evaluated == []


# --------------------------------------------------------------------------- the pins


def test_a_library_that_is_not_the_pinned_one_stops_the_task(contexts, monkeypatch):
    monkeypatch.setattr(job_module, "read_pins", lambda *a: {"torch": "2.14.1", "peft": "0.21.3"})
    monkeypatch.setattr(
        job_module, "installed_versions", lambda pins: {"torch": "2.10.0", "peft": "0.21.3"}
    )
    recorder = Recorder()
    result = run("smoke", recorder, contexts | {"check_pins": True})
    assert result["status"] == "config_error"
    assert "not the pinned one: torch is 2.10.0, pinned 2.14.1" in result["error"]
    assert result["libraries"] == {"torch": "2.10.0", "peft": "0.21.3"}
    assert recorder.trained == []


def test_pin_differences(monkeypatch):
    monkeypatch.setattr(job_module, "read_pins", lambda *a: {"torch": "2.14.1", "trl": "1.18.0"})
    for installed, expected in (
        ({"torch": "2.14.1+cu126", "trl": "1.18.0"}, []),  # a local build suffix is accepted
        ({"torch": "2.14.1", "trl": None}, ["trl is not installed (pinned 1.18.0)"]),
        ({"torch": "2.14.10", "trl": "1.18.0"}, ["torch is 2.14.10, pinned 2.14.1"]),
    ):
        monkeypatch.setattr(job_module, "installed_versions", lambda pins, found=installed: found)
        assert pin_differences() == expected


# --------------------------------------------------------------------------- environment, guards


def test_the_job_name_and_the_commit_come_from_the_environment(contexts, monkeypatch, tmp_path):
    monkeypatch.setenv("QLORA_JOB_NAME", "qlora-smoke-0123abcd-a1")
    monkeypatch.setenv("QLORA_GIT_COMMIT", "0123abcd" * 5)
    result = run("train", Recorder(), contexts)
    assert (result["job_name"], result["commit"]) == ("qlora-smoke-0123abcd-a1", "0123abcd" * 5)
    assert result_path("qlora-smoke-0123abcd-a1", tmp_path).is_file()


def test_the_predictors_are_recorded_as_the_evaluate_script_records_them():
    from scripts.evaluate import PREDICTORS as SCRIPT_TABLE

    assert set(PREDICTORS) == {UNTUNED, TUNED}
    for name, entry in PREDICTORS.items():
        script = SCRIPT_TABLE[name]
        assert (script.path, script.kind, script.where) == ("A", "eval", "local")
        assert (script.arm, script.configs) == (entry["arm"], entry["configs"])
