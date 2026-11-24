"""Tests for the gate on a test run (req_005 v5.01), on an invented split."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from qlora_finetuning.comparison import gate
from qlora_finetuning.comparison.gate import (
    SCHEMA,
    TestRunRefused,
    check_test_request,
    file_sha256,
    predictor_differences,
    read_voids,
    results_on_test,
    run_chain,
    runs_on_test,
    write_canonical,
)
from qlora_finetuning.data import TestLockedError
from qlora_finetuning.data.bundle import BUNDLE_ENV, export_bundle
from qlora_finetuning.evaluation.fake import gold_predictor
from qlora_finetuning.evaluation.predictor import PredictorError
from tests.comparison.helpers import (
    PREDICTORS,
    arm_predictor,
    comparison_cfg,
    freeze_and_unlock,
    run_arm,
    write_freeze_record,
    write_unlock,
)
from tests.evaluation.conftest import evaluate, make_context, records_for, run_folders

ARM = "B0-Lite"


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.delenv(BUNDLE_ENV, raising=False)
    return make_context(tmp_path)


@pytest.fixture
def unlocked(context):
    freeze_and_unlock(context)
    return context


def in_a_job(context, monkeypatch, tmp_path) -> None:
    """Make the process look like a SageMaker job that was given the test bundle."""
    folder = tmp_path / "bundle_test"
    export_bundle(folder, ("test",), cfg=context.split_cfg, data_cfg=context.data_cfg)
    monkeypatch.setenv(BUNDLE_ENV, str(folder))


def void(context, *runs: str) -> None:
    cfg = comparison_cfg(context)
    voids = [*read_voids(cfg), {"arm": ARM, "runs": list(runs), "reason": "a fault"}]
    write_canonical(cfg.voids_file, {"schema": SCHEMA, "voids": voids})


# --------------------------------------------------------------------------- lock and unlock


def test_a_locked_split_is_still_locked(context):
    with pytest.raises(TestLockedError, match="the test split is locked"):
        check_test_request(
            arm=ARM, name="nova-lite", where="bedrock", scope="measured", limit=None,
            resume=None, ctx=context,
        )  # fmt: skip
    assert run_folders(context) == []


def test_an_unlock_without_a_freeze_record_allows_nothing(context):
    write_unlock(context, None)
    with pytest.raises(TestRunRefused, match="there is no freeze record"):
        evaluate(gold_predictor([]), "test", context)
    assert run_folders(context) == []


def test_the_unlock_must_name_the_freeze_record_on_disk(context):
    digest = write_freeze_record(context)
    write_unlock(context, None)
    with pytest.raises(TestRunRefused, match="does not name a freeze record"):
        run_arm(ARM, context, predictor=gold_predictor([], **PREDICTORS[ARM]))
    write_unlock(context, "0" * 64)
    with pytest.raises(TestRunRefused, match="the freeze record changed after the unlock"):
        run_arm(ARM, context, predictor=gold_predictor([], **PREDICTORS[ARM]))
    write_unlock(context, digest)
    assert run_arm(ARM, context)["result"]


def test_the_gold_check_may_run_on_test_after_the_unlock(unlocked):
    predictor = gold_predictor(records_for("test", unlocked))
    summary = evaluate(predictor, "test", unlocked)
    assert summary["result"] and summary["all"]["execution_accuracy"]["count"] == 22
    again = evaluate(gold_predictor(records_for("test", unlocked)), "test", unlocked)
    assert again["result"]  # not an arm: it may run again


def test_only_the_six_arms_and_gold_may_answer_test(unlocked):
    other = gold_predictor(records_for("test", unlocked), name="fake")
    with pytest.raises(TestRunRefused, match="only the six arms"):
        evaluate(other, "test", unlocked)
    with pytest.raises(TestRunRefused, match="arm B0-Lite is the predictor nova-lite"):
        run_arm(ARM, unlocked, predictor=arm_predictor("B0-Pro", unlocked))


# --------------------------------------------------------------------------- one run per arm


def test_an_arm_answers_test_once(unlocked):
    first = run_arm(ARM, unlocked)
    assert first["result"] and first["split"] == "test" and first["scope"] == "measured"
    assert [line["run_id"] for line in results_on_test(unlocked.config.results_file, ARM)] == [
        first["run_id"]
    ]
    with pytest.raises(TestRunRefused, match="already has its test result") as refused:
        run_arm(ARM, unlocked)
    assert first["run_id"] in str(refused.value) and "comparison_runs void" in str(refused.value)
    assert run_arm("B0-Pro", unlocked)["result"]  # another arm is not affected


def test_a_voided_run_allows_exactly_one_more(unlocked):
    first = run_arm(ARM, unlocked)
    void(unlocked, first["run_id"])
    second = run_arm(ARM, unlocked)
    assert second["result"] and second["run_id"] != first["run_id"]
    with pytest.raises(TestRunRefused, match="already has its test result"):
        run_arm(ARM, unlocked)


def test_an_unfinished_run_is_continued_not_restarted(unlocked):
    with pytest.raises(PredictorError) as stopped:
        run_arm(ARM, unlocked, predictor=arm_predictor(ARM, unlocked, fail_after=5))
    first = stopped.value.run_id
    assert [run["run_id"] for run in runs_on_test(unlocked.tracking_cfg, ARM)] == [first]
    with pytest.raises(TestRunRefused, match=f"--resume {first}"):
        run_arm(ARM, unlocked)
    summary = run_arm(ARM, unlocked, resume=first)
    assert summary["result"] and summary["continues"] == first
    assert run_chain(summary["run_id"], unlocked.tracking_cfg.runs_dir) == [
        summary["run_id"],
        first,
    ]


def test_a_voided_unfinished_run_may_be_started_again(unlocked):
    with pytest.raises(PredictorError) as stopped:
        run_arm(ARM, unlocked, predictor=arm_predictor(ARM, unlocked, fail_after=5))
    void(unlocked, stopped.value.run_id)
    assert run_arm(ARM, unlocked)["result"]


# --------------------------------------------------------------------------- how a run is asked


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"limit": 5}, "--limit is refused on the test split"),
        ({"scope": "dev"}, "add --scope measured"),
        ({"where": "local"}, "in bedrock only, not local"),
    ],
)
def test_a_test_run_is_whole_measured_and_in_the_cloud(unlocked, options, message):
    with pytest.raises(TestRunRefused, match=message):
        run_arm(ARM, unlocked, **options)
    assert run_folders(unlocked) == []


def test_a_qwen_arm_answers_test_only_inside_a_job(unlocked, monkeypatch, tmp_path):
    with pytest.raises(TestRunRefused, match="inside a SageMaker job only"):
        run_arm("A0", unlocked)
    with pytest.raises(TestRunRefused, match="in sagemaker only, not local"):
        run_arm("A0", unlocked, where="local")
    in_a_job(unlocked, monkeypatch, tmp_path)
    summary = run_arm("A0", unlocked)
    assert summary["result"] and summary["all"]["examples"] == 22
    assert run_arm("A1", unlocked)["result"]


def test_planning_asks_about_a_job_from_this_machine(unlocked):
    request = {
        "arm": "A0", "name": "qwen3-4b", "where": "sagemaker", "scope": "measured",
        "limit": None, "resume": None, "ctx": unlocked,
    }  # fmt: skip
    assert gate.request_problems(**request, planning=True) == []
    assert "inside a SageMaker job only" in gate.request_problems(**request)[0]


# --------------------------------------------------------------------------- what is frozen


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"settings": {"api": "converse", "temperature": 1, "top_k": 1}}, "settings .temperature."),
        ({"model_id": "amazon.nova-lite-v2:0"}, "its model id differ"),
    ],
)
def test_a_predictor_that_is_not_the_frozen_one_is_refused(unlocked, change, message):
    with pytest.raises(TestRunRefused, match=message):
        run_arm(ARM, unlocked, predictor=arm_predictor(ARM, unlocked, **change))
    assert run_folders(unlocked) == []


def test_a_changed_shared_file_is_refused(unlocked):
    flags = Path(unlocked.config.flags_file)
    flags.write_text(flags.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(TestRunRefused, match="changed since the freeze: the gold flags"):
        run_arm(ARM, unlocked)


def test_a_changed_reply_limit_is_refused(unlocked):
    changed = dataclasses.replace(unlocked, max_new_tokens=64)
    changed.manifest = unlocked.manifest
    with pytest.raises(TestRunRefused, match="the max new tokens"):
        run_arm(ARM, changed)


def test_a_freeze_record_edited_after_the_unlock_is_refused(unlocked):
    changed = {arm: dict(entry) for arm, entry in PREDICTORS.items()}
    changed[ARM]["model_id"] = "another-model"
    write_freeze_record(unlocked, changed)
    with pytest.raises(TestRunRefused, match="the freeze record changed after the unlock"):
        run_arm(ARM, unlocked, predictor=arm_predictor(ARM, unlocked, model_id="another-model"))


# --------------------------------------------------------------------------- small pieces


def test_a_hash_is_the_same_with_either_line_end(tmp_path):
    (tmp_path / "unix.yaml").write_bytes(b"version: 1\nkey: value\n")
    (tmp_path / "windows.yaml").write_bytes(b"version: 1\r\nkey: value\r\n")
    assert file_sha256(tmp_path / "unix.yaml") == file_sha256(tmp_path / "windows.yaml")
    assert file_sha256(tmp_path / "absent.yaml") is None


def test_predictor_differences_name_what_differs():
    frozen = {"name": "a", "model_id": "m", "settings": {"x": 1, "y": 2}}
    assert predictor_differences(frozen, dict(frozen)) == []
    other = {"name": "b", "model_id": "m", "settings": {"x": 1, "y": 3, "z": 0}}
    assert predictor_differences(frozen, other) == ["name", "settings (y, z)"]


def test_dev_runs_never_reach_the_gate(context):
    # No freeze, no unlock: dev is evaluated as it always was.
    assert evaluate(gold_predictor(records_for("dev", context)), "dev", context)["result"]


def test_the_gate_does_not_import_the_runner_or_aws():
    source = Path(gate.__file__).read_text(encoding="utf-8")
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    for banned in ("evaluation.runner", "boto", "local_qlora", "frontier"):
        assert not any(banned in line for line in imports), banned
