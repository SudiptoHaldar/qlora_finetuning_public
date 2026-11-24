"""Tests for qlora_finetuning.local_qlora.adapters and the tuned predictor (req_003 v3.02)."""

from __future__ import annotations

import json

import pytest

from qlora_finetuning.data import manifest_hash
from qlora_finetuning.evaluation.predictor import describe
from qlora_finetuning.local_qlora import backend as backend_module
from qlora_finetuning.local_qlora.adapters import (
    ADAPTER_WEIGHTS,
    CHECKPOINTS,
    METADATA_NAME,
    check_adapter,
    checkpoint_steps,
    read_json,
    resolve_adapter,
    write_json,
)
from qlora_finetuning.local_qlora.predictor import build_tuned_predictor, qwen3_4b_tuned
from qlora_finetuning.local_qlora.training import run_training
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking.config import TrackingError

from .conftest import MODEL_ID, REVISION, StubBackend
from .training_helpers import StubEngine, StubReloader, make_training_context, write_checkpoint

RUN = "20261119T101500Z-train-qwen3-4b-fit-1a2b"


@pytest.fixture
def training_context(tmp_path):
    return make_training_context(tmp_path)


def trained(training_context, smoke=False):
    """A completed training run with the stand-in engine; returns its result."""
    return run_training(
        smoke=smoke, context=training_context, engine=StubEngine(), reloader=StubReloader()
    )


def folders(training_context):
    return training_context.training_cfg.adapters_dir, training_context.tracking_cfg.runs_dir


# --------------------------------------------------------------------------- files


def test_json_round_trip_and_damage(tmp_path):
    path = tmp_path / METADATA_NAME
    write_json(path, {"steps": 6, "smoke": False})
    assert read_json(path, "the metadata") == {"steps": 6, "smoke": False}
    assert path.read_text(encoding="utf-8").endswith("}\n")
    with pytest.raises(ConfigError, match="the metadata is missing"):
        read_json(tmp_path / "absent.json", "the metadata")
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="the metadata is damaged"):
        read_json(path, "the metadata")
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ConfigError, match="damaged"):
        read_json(path, "the metadata")


def test_nothing_that_looks_like_a_credential_is_written(tmp_path):
    with pytest.raises(TrackingError, match="email address"):
        write_json(tmp_path / METADATA_NAME, {"note": "ask someone@example.com"})


def test_checkpoint_steps_lists_folders_that_hold_an_adapter(tmp_path):
    assert checkpoint_steps(tmp_path / "absent") == []
    for step in (100, 4, 20):
        write_checkpoint(tmp_path / f"checkpoint-{step}", step)
    (tmp_path / "checkpoint-7").mkdir()  # no weights in it
    (tmp_path / "notes").mkdir()
    assert checkpoint_steps(tmp_path) == [4, 20, 100]


# --------------------------------------------------------------------------- finding an adapter


def test_a_run_id_names_the_final_adapter(training_context):
    result = trained(training_context)
    ref = resolve_adapter(result["run_id"], *folders(training_context))
    assert (ref.run_id, ref.checkpoint, ref.name) == (result["run_id"], None, result["run_id"])
    assert ref.path == training_context.training_cfg.adapters_dir / result["run_id"]
    assert ref.metadata["adapter_sha256"] == result["adapter_sha256"]
    assert (ref.metadata["steps"], ref.metadata["smoke"]) == (6, False)


def test_a_checkpoint_is_named_by_its_run_and_step(training_context):
    result = trained(training_context, smoke=True)  # checkpoints at 4 and 6
    name = f"{result['run_id']}/checkpoint-4"
    ref = resolve_adapter(name, *folders(training_context))
    assert (ref.run_id, ref.checkpoint, ref.name) == (result["run_id"], 4, name)
    assert ref.path.name == "checkpoint-4" and ref.path.parent.name == CHECKPOINTS
    assert (ref.path / ADAPTER_WEIGHTS).read_bytes() == b"weights after step 4"
    # What it was trained for comes from the run's own record.
    assert ref.metadata["prompt_id"] == training_context.prompt.prompt_id
    assert (ref.metadata["steps"], ref.metadata["smoke"]) == (4, True)


@pytest.mark.parametrize(
    "name", ["", "../other", "run/../x", "run/checkpoint-x", "run/final", "a b", None, 7]
)
def test_a_name_that_is_not_an_adapter(training_context, name):
    with pytest.raises(ConfigError, match="is not an adapter"):
        resolve_adapter(name, *folders(training_context))


def test_a_run_without_a_finished_adapter_points_at_its_checkpoints(training_context):
    adapters_dir, runs_dir = folders(training_context)
    with pytest.raises(ConfigError, match="has no finished adapter under adapters/") as caught:
        resolve_adapter(RUN, adapters_dir, runs_dir)
    assert "name one" not in str(caught.value)
    write_checkpoint(runs_dir / RUN / CHECKPOINTS / "checkpoint-200", 200)
    with pytest.raises(ConfigError, match=f"name one: {RUN}/checkpoint-200"):
        resolve_adapter(RUN, adapters_dir, runs_dir)
    with pytest.raises(ConfigError, match=r"no checkpoint at step 300 .*\(it has: 200\)"):
        resolve_adapter(f"{RUN}/checkpoint-300", adapters_dir, runs_dir)
    with pytest.raises(ConfigError, match="the training run's record is missing"):
        resolve_adapter(f"{RUN}/checkpoint-200", adapters_dir, runs_dir)


# --------------------------------------------------------------------------- the experiment


def current(training_context):
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "prompt_id": training_context.prompt.prompt_id,
        "manifest_hash": manifest_hash(training_context.split_cfg),
    }


def test_an_adapter_for_the_current_experiment_is_accepted(training_context):
    ref = resolve_adapter(trained(training_context)["run_id"], *folders(training_context))
    check_adapter(ref, **current(training_context))


@pytest.mark.parametrize(
    ("field", "words"),
    [
        ("prompt_id", "another prompt wording"),
        ("manifest_hash", "another split"),
        ("revision", "another commit of the base model"),
        ("model_id", "another base model"),
    ],
)
def test_an_adapter_trained_for_another_experiment_is_refused(training_context, field, words):
    ref = resolve_adapter(trained(training_context)["run_id"], *folders(training_context))
    with pytest.raises(ConfigError, match=f"was trained for {words}; it answers a different"):
        check_adapter(ref, **(current(training_context) | {field: "something else"}))


def test_an_adapter_that_does_not_say_what_it_was_trained_for_is_refused(training_context):
    ref = resolve_adapter(trained(training_context)["run_id"], *folders(training_context))
    del ref.metadata["prompt_id"]
    with pytest.raises(ConfigError, match=r"does not say what it was trained for \(prompt_id"):
        check_adapter(ref, **current(training_context))


# --------------------------------------------------------------------------- the tuned predictor


@pytest.fixture
def loaded(monkeypatch):
    """Replace the model load: records what was asked for and returns a stub backend."""
    seen = {}

    def fake_load(cfg, budget, adapter_dir=None):
        seen.update(adapter_dir=adapter_dir, calls=seen.get("calls", 0) + 1)
        return StubBackend(settings=cfg.settings(budget.revision, adapter="a" * 64))

    monkeypatch.setattr(backend_module, "load_backend", fake_load)
    return seen


def runner_context(training_context):
    """What scripts.evaluate hands a factory: the prompt, the split, where runs are."""
    from types import SimpleNamespace

    return SimpleNamespace(
        prompt=training_context.prompt,
        split_cfg=training_context.split_cfg,
        tracking_cfg=training_context.tracking_cfg,
    )


def test_the_tuned_predictor_names_its_adapter_in_settings_and_details(training_context, loaded):
    result = trained(training_context)
    context = runner_context(training_context)
    predictor = build_tuned_predictor(result["run_id"], context)
    assert loaded == {"adapter_dir": training_context.training_cfg.adapters_dir / result["run_id"],
                      "calls": 1}  # fmt: skip
    described = describe(predictor)
    assert described["name"] == "qwen3-4b-tuned" and described["model_id"] == MODEL_ID
    assert described["settings"]["adapter"] == "a" * 64  # the hash decides a resume
    assert described["details"] == {
        "training_run": result["run_id"],
        "checkpoint": None,
        "steps": 6,
        "smoke": False,
    }


def test_a_smoke_adapter_and_a_checkpoint_say_so(training_context, loaded):
    result = trained(training_context, smoke=True)
    context = runner_context(training_context)
    predictor = qwen3_4b_tuned("quick_dev", context, f"{result['run_id']}/checkpoint-4")
    assert predictor.name == "qwen3-4b-tuned-smoke"  # never read as arm A1's result
    assert predictor.details == {
        "training_run": result["run_id"],
        "checkpoint": 4,
        "steps": 4,
        "smoke": True,
    }
    assert loaded["adapter_dir"].name == "checkpoint-4"


def test_every_refusal_comes_before_the_model_is_loaded(training_context, loaded):
    result = trained(training_context)
    context = runner_context(training_context)
    for adapter in (None, ""):
        with pytest.raises(ConfigError, match="needs an adapter: add --adapter RUN_ID"):
            qwen3_4b_tuned("quick_dev", context, adapter)
    with pytest.raises(ConfigError, match="has no finished adapter"):
        build_tuned_predictor(RUN, context)

    folder = training_context.training_cfg.adapters_dir / result["run_id"]
    metadata = json.loads((folder / METADATA_NAME).read_text(encoding="utf-8"))
    write_json(folder / METADATA_NAME, metadata | {"prompt_id": "f" * 64})
    with pytest.raises(ConfigError, match="trained for another prompt wording"):
        build_tuned_predictor(result["run_id"], context)

    write_json(folder / METADATA_NAME, metadata)
    (folder / ADAPTER_WEIGHTS).write_bytes(b"other weights")
    with pytest.raises(ConfigError, match="are not the ones its metadata records"):
        build_tuned_predictor(result["run_id"], context)
    assert loaded == {}
