"""Tests for the six-arm record, voiding a run, and the check after the unlock (req_005 v5.01)."""

from __future__ import annotations

import dataclasses
import json
from datetime import date
from pathlib import Path

import pytest

from qlora_finetuning.comparison.config import load_comparison_config
from qlora_finetuning.comparison.gate import TestRunRefused, read_voids
from qlora_finetuning.comparison.runs import (
    DONE,
    MISSING,
    UNFINISHED,
    arm_status,
    build_record,
    percent,
    preflight,
    prompt_lengths,
    render_markdown,
    report_paths,
    void_run,
    write_record,
)
from qlora_finetuning.data import TestLockedError
from qlora_finetuning.data.bundle import BUNDLE_ENV, export_bundle
from qlora_finetuning.evaluation.predictor import PredictorError
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking.jsonl import append_line
from tests.comparison.helpers import (
    PREDICTORS,
    FakeGit,
    arm_predictor,
    arms_now,
    comparison_cfg,
    dev_run,
    freeze_and_unlock,
    job_record_for,
    run_arm,
)
from tests.data.synthetic import SECRET_WORDS
from tests.evaluation.conftest import make_context

REASON = "the deployment was deleted while the run was going"
TODAY = date(2026, 11, 23)


@pytest.fixture
def context(tmp_path, monkeypatch):
    """A frozen, unlocked invented split, in a process that looks like a test job."""
    monkeypatch.delenv(BUNDLE_ENV, raising=False)
    context = make_context(tmp_path)
    freeze_and_unlock(context)
    folder = tmp_path / "bundle_test"
    export_bundle(folder, ("test",), cfg=context.split_cfg, data_cfg=context.data_cfg)
    monkeypatch.setenv(BUNDLE_ENV, str(folder))
    return context


@pytest.fixture
def runs(context):
    return {arm: run_arm(arm, context)["run_id"] for arm in PREDICTORS}


def record_of(context, runs, **options):
    settings = {"job_record": job_record_for(runs), "arms_now": arms_now(), "git": FakeGit()}
    settings.update(options)
    return build_record(comparison_cfg(context), context, **settings)


def check(record, check_id):
    return next(item for item in record["checks"] if item["id"] == check_id)


def failed(record):
    return [item["id"] for item in record["checks"] if not item["passed"]]


# --------------------------------------------------------------------------- where the arms stand


def test_status_goes_from_missing_to_unfinished_to_done(context):
    cfg = comparison_cfg(context)
    assert {entry["state"] for entry in arm_status(cfg, context)} == {MISSING}
    with pytest.raises(PredictorError) as stopped:
        run_arm("B0-Pro", context, predictor=arm_predictor("B0-Pro", context, fail_after=4))
    done = run_arm("B0-Lite", context)
    status = {entry["arm"]: entry for entry in arm_status(cfg, context)}
    assert status["B0-Pro"]["state"] == UNFINISHED
    assert status["B0-Pro"]["resume"] == stopped.value.run_id
    assert status["B0-Lite"]["state"] == DONE and status["B0-Lite"]["results"] == [done["run_id"]]
    assert status["A0"]["state"] == MISSING and status["A0"]["where"] == "sagemaker"


# --------------------------------------------------------------------------- the record


def test_six_complete_runs_make_the_record(context, runs):
    cfg = comparison_cfg(context)
    record = record_of(context, runs)
    assert record["passed"] and failed(record) == []
    assert [item["id"] for item in record["checks"]] == [f"C{n}" for n in range(1, 9)]
    assert list(record["arms"]) == list(PREDICTORS) and record["test_examples"] == 22
    a1 = record["arms"]["A1"]
    assert a1["run_id"] == runs["A1"] and a1["model_id"] == "Qwen/Qwen3-4B"
    assert a1["all"]["execution_accuracy"] == {"count": 22, "percent": "100.00"}
    assert a1["jobs"] == [
        {"name": "qlora-evaluate-a1", "instance_type": "ml.g5.xlarge", "billable_seconds": 3900}
    ]
    assert record["arms"]["B0-Pro"]["jobs"] == [] and record["deviations"] == []
    assert record["unlock"]["commit"] == "c" * 40

    markdown_path, json_path = write_record(record, cfg)
    assert (markdown_path, json_path) == report_paths(cfg)
    assert json.loads(json_path.read_text(encoding="utf-8")) == record
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown == render_markdown(record)
    for arm, run_id in runs.items():
        assert f"| {arm} | 22 (100.00%)" in markdown and run_id in markdown
    for text in (markdown, json_path.read_text(encoding="utf-8")):
        assert not any(word in text for word in SECRET_WORDS)
        assert "$" not in text and "confidence" not in text.replace("no confidence", "")


def test_five_arms_write_nothing(context):
    cfg = comparison_cfg(context)
    runs = {arm: run_arm(arm, context)["run_id"] for arm in PREDICTORS if arm != "B1-Pro"}
    record = record_of(context, runs)
    assert not record["passed"] and failed(record) == ["C1"]
    assert "arm B1-Pro: missing" in check(record, "C1")["detail"]
    with pytest.raises(ConfigError, match="written only when all six arms pass"):
        write_record(record, cfg)
    assert not cfg.report_dir.exists()


def test_a_run_with_a_lost_line_is_not_a_result(context, runs):
    path = context.tracking_cfg.runs_dir / runs["B0-Lite"] / "predictions.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:-1]), encoding="utf-8", newline="")
    record = record_of(context, runs)
    assert {"C1", "C2"} <= set(failed(record))
    assert "21 of 22" in check(record, "C2")["detail"]


def test_arms_of_one_model_must_have_been_sent_the_same_tokens(context):
    class OneMore:
        """The frozen predictor of B1-Lite, reporting one input token too many."""

        def __init__(self, inner):
            self.inner = inner
            for name in ("name", "model_id", "settings", "batch_size", "usage_source"):
                setattr(self, name, getattr(inner, name))

        def predict(self, requests):
            return [
                dataclasses.replace(prediction, input_tokens=prediction.input_tokens + 1)
                for prediction in self.inner.predict(requests)
            ]

    runs = {arm: run_arm(arm, context)["run_id"] for arm in PREDICTORS if arm != "B1-Lite"}
    runs["B1-Lite"] = run_arm(
        "B1-Lite", context, predictor=OneMore(arm_predictor("B1-Lite", context))
    )["run_id"]
    record = record_of(context, runs)
    assert failed(record) == ["C3"]
    assert (
        "B0-Lite and B1-Lite were sent different numbers of input tokens"
        in check(record, "C3")["detail"]
    )


def test_a_shared_file_changed_after_the_runs_fails_c4(context, runs):
    flags = Path(context.config.flags_file)
    flags.write_text(flags.read_text(encoding="utf-8") + " ", encoding="utf-8")
    record = record_of(context, runs)
    assert failed(record) == ["C4"] and "the gold flags" in check(record, "C4")["detail"]


@pytest.mark.parametrize(
    ("git", "message"),
    [
        (FakeGit(ancestor=False), "does not contain the unlock record"),
        (FakeGit(unlock_commit=None), "the unlock record is not committed"),
    ],
)
def test_every_run_comes_after_the_committed_unlock(context, runs, git, message):
    record = record_of(context, runs, git=git)
    assert failed(record) == ["C5"] and message in check(record, "C5")["detail"]


def test_a_tuned_arm_must_have_asked_todays_tuned_model(context, runs):
    changed = arms_now(**{"B1-Pro": {"model_id": "another-deployment"}})
    del changed["A0"]
    record = record_of(context, runs, arms_now=changed)
    detail = check(record, "C6")["detail"]
    assert failed(record) == ["C6"]
    assert "arm B1-Pro: run" in detail and "tuning record's deployment" in detail
    assert "arm A0: its model can no longer be worked out" in detail


def test_usage_must_be_recorded_for_every_arm(context, runs):
    record = record_of(context, runs, job_record=job_record_for({"A0": runs["A0"]}))
    assert failed(record) == ["C7"]
    assert "arm A1: no SageMaker job in the job record made run" in check(record, "C7")["detail"]
    record = record_of(context, runs, job_record=job_record_for(runs, pulled=False))
    assert "is not pulled or has no billable seconds" in check(record, "C7")["detail"]


def test_a_nova_run_without_awss_token_counts_fails_c7(context):
    runs = {arm: run_arm(arm, context)["run_id"] for arm in PREDICTORS if arm != "B0-Pro"}
    computed = arm_predictor("B0-Pro", context, usage_source="computed")
    runs["B0-Pro"] = run_arm("B0-Pro", context, predictor=computed)["run_id"]
    record = record_of(context, runs)
    assert failed(record) == ["C7"]
    assert (
        "arm B0-Pro: not every prediction has AWS's token counts" in check(record, "C7")["detail"]
    )


def test_a_stray_test_result_fails_c8_and_gold_does_not(context, runs):
    from qlora_finetuning.evaluation.fake import gold_predictor
    from tests.evaluation.conftest import evaluate, records_for

    evaluate(gold_predictor(records_for("test", context)), "test", context)
    assert record_of(context, runs)["passed"]  # the gold check is not a stray run
    stray = {"run_id": "stray-run", "split": "test", "arm": "none", "predictor": "other"}
    append_line(context.config.results_file, stray)
    record = record_of(context, runs)
    assert failed(record) == ["C8"] and "stray-run" in check(record, "C8")["detail"]


def test_there_is_no_record_without_a_freeze(tmp_path, monkeypatch):
    monkeypatch.delenv(BUNDLE_ENV, raising=False)
    context = make_context(tmp_path)
    with pytest.raises(ConfigError, match="there is no freeze record"):
        record_of(context, {})


# --------------------------------------------------------------------------- voiding


def test_a_void_is_the_owners_and_needs_a_reason(context, runs, monkeypatch):
    cfg = comparison_cfg(context)
    with pytest.raises(ConfigError, match="the owner's decision, with --yes"):
        void_run(cfg, context, runs["B0-Lite"], REASON)
    with pytest.raises(ConfigError, match="a reason of at least 15 characters"):
        void_run(cfg, context, runs["B0-Lite"], "  bad   run ", yes=True)
    with pytest.raises(ConfigError, match="is not a test run on this machine"):
        void_run(cfg, context, "no-such-run", REASON, yes=True)
    monkeypatch.delenv(BUNDLE_ENV)  # a dev run is made here, from the snapshot
    on_dev = dev_run("B0-Lite", context)["run_id"]
    with pytest.raises(ConfigError, match="is not a test run on this machine"):
        void_run(cfg, context, on_dev, REASON, yes=True)
    assert read_voids(cfg) == [] and not cfg.voids_file.exists()


def test_a_voided_result_lets_the_arm_run_once_more_and_shows_as_a_deviation(context, runs):
    cfg = comparison_cfg(context)
    entry = void_run(cfg, context, runs["B0-Lite"], REASON, yes=True, today=TODAY)
    assert entry == {
        "arm": "B0-Lite", "runs": [runs["B0-Lite"]], "reason": REASON, "had_result": True,
        "voided_on": "2026-11-23",
    }  # fmt: skip
    with pytest.raises(ConfigError, match="is already voided"):
        void_run(cfg, context, runs["B0-Lite"], REASON, yes=True)
    status = {item["arm"]: item["state"] for item in arm_status(cfg, context)}
    assert status["B0-Lite"] == MISSING and status["B0-Pro"] == DONE
    assert not record_of(context, runs)["passed"]

    runs["B0-Lite"] = run_arm("B0-Lite", context)["run_id"]
    with pytest.raises(TestRunRefused, match="already has its test result"):
        run_arm("B0-Lite", context)
    record = record_of(context, runs)
    assert record["passed"] and record["deviations"] == [entry]
    assert REASON in render_markdown(record)


def test_a_chain_is_voided_from_its_newest_run(context):
    cfg = comparison_cfg(context)
    with pytest.raises(PredictorError) as stopped:
        run_arm("B0-Pro", context, predictor=arm_predictor("B0-Pro", context, fail_after=4))
    first = stopped.value.run_id
    second = run_arm("B0-Pro", context, resume=first)["run_id"]
    with pytest.raises(ConfigError, match=f"run {second} continues run {first}"):
        void_run(cfg, context, first, REASON, yes=True)
    entry = void_run(cfg, context, second, REASON, yes=True)
    assert entry["runs"] == [second, first] and entry["had_result"]
    assert run_arm("B0-Pro", context)["result"]  # a fresh run, not a resume


# --------------------------------------------------------------------------- after the unlock


def test_the_preflight_follows_the_lock(tmp_path, monkeypatch):
    monkeypatch.delenv(BUNDLE_ENV, raising=False)
    locked = make_context(tmp_path)
    with pytest.raises(TestLockedError, match="the test split is locked"):
        preflight(locked, counter=lambda record: 10)


def test_the_preflight_scores_gold_on_test_and_counts_the_prompts(context):
    outcome = preflight(context, counter=lambda record: 1500)
    gold = outcome["gold"]
    assert gold["result"] and gold["all"]["execution_accuracy"]["count"] == 22
    assert gold["arm"] == "none" and gold["kind"] == "smoke"
    assert outcome["lengths"] == {
        "counted": True, "records": 22, "longest": 1500, "max_length": 2048,
        "over_max_length": 0,
    }  # fmt: skip
    status = arm_status(comparison_cfg(context), context)
    assert {entry["state"] for entry in status} == {MISSING}  # gold is not an arm


def test_a_sequence_over_the_training_length_is_counted_not_refused():
    lengths = prompt_lengths([1, 2, 3], counter=lambda record: 2047 + record)
    assert lengths["longest"] == 2050 and lengths["over_max_length"] == 2


def test_percent_is_text_from_counts():
    assert percent(358, 499) == "71.74" and percent(0, 22) == "0.00" and percent(1, 0) is None


def test_the_real_config_names_the_six_arms():
    cfg = load_comparison_config()
    assert list(cfg.arms) == ["A0", "A1", "B0-Lite", "B1-Lite", "B0-Pro", "B1-Pro"]
