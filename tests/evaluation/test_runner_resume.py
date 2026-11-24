"""Tests for interrupting and resuming an evaluation (req_002 v2.03).

The rule under test: an example is never predicted twice and never skipped, and the ledger
records of the runs involved cover what was consumed.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
import yaml

from qlora_finetuning.data import load_prompt
from qlora_finetuning.evaluation.fake import FakePredictor, gold_predictor
from qlora_finetuning.evaluation.predictor import PredictorError
from qlora_finetuning.evaluation.runner import ResumeError, summarize_run
from qlora_finetuning.settings import CONFIG_DIR
from qlora_finetuning.tracking import load_manifest, start_run

from .conftest import (
    GIT,
    GOLD_REPLY,
    evaluate,
    ids_of,
    ledger_records,
    predictions_path,
    read_lines,
    records_for,
    results_lines,
    run_folders,
    write_lines,
)


def gold(context, split="dev", **options):
    return gold_predictor(records_for(split, context), **options)


def interrupted(context, fail_after, batch_size=1, split="dev", **options):
    """Run until the fake predictor fails; return the failed run's id and the predictor."""
    predictor = gold(context, split, batch_size=batch_size, fail_after=fail_after, **options)
    with pytest.raises(PredictorError) as caught:
        evaluate(predictor, split, context)
    assert caught.value.run_id is not None
    return caught.value.run_id, predictor


def without_run_ids(lines):
    return [{key: value for key, value in line.items() if key != "run_id"} for line in lines]


def check(summary, check_id):
    return next(item for item in summary["checks"] if item["id"] == check_id)


# --------------------------------------------------------------------------- interrupt, resume


@pytest.mark.parametrize(
    ("batch_size", "fail_after", "kept"),
    [(1, 0, 0), (1, 1, 1), (1, 5, 5), (1, 7, 7), (3, 4, 3), (3, 7, 6), (8, 5, 0), (5, 5, 5)],
)
def test_a_resumed_run_equals_an_uninterrupted_one(context, batch_size, fail_after, kept):
    whole = evaluate(gold(context, batch_size=batch_size), "dev", context)
    ids = ids_of("dev", context)

    failed_id, first = interrupted(context, fail_after, batch_size)
    assert first.asked == ids[:kept]
    manifest = load_manifest(failed_id, context.tracking_cfg)
    assert manifest["status"] == "failed"
    assert manifest["error"].startswith("PredictorError: the fake predictor fails after")
    partial = summarize_run(failed_id, context)
    assert partial["examples_predicted"] == kept and not partial["complete"]
    assert not check(partial, "R1")["passed"]
    recorded = {record.run_id: record.examples for record in ledger_records(context)}
    assert recorded.get(failed_id, 0) == kept  # usage recorded although the run failed

    second = gold(context, batch_size=batch_size)
    summary = evaluate(second, "dev", context, resume=failed_id)
    assert summary["result"] and summary["continues"] == failed_id
    assert second.asked == ids[kept:]  # never asked twice, none skipped
    assert first.asked + second.asked == ids

    lines = read_lines(summary["run_id"], context)
    assert without_run_ids(lines) == without_run_ids(read_lines(whole["run_id"], context))
    assert [line["run_id"] for line in lines] == [failed_id] * kept + [summary["run_id"]] * (
        8 - kept
    )
    assert summary["all"] == whole["all"]

    recorded = {record.run_id: record.examples for record in ledger_records(context)}
    assert recorded.get(failed_id, 0) + recorded[summary["run_id"]] == 8
    assert check(summary, "R4")["passed"]
    # The failed run is no result; the uninterrupted and the resumed runs are.
    assert [line["run_id"] for line in results_lines(context)] == [
        whole["run_id"],
        summary["run_id"],
    ]
    # The earlier run's own file is left as it was.
    assert len(read_lines(failed_id, context)) == kept


def test_a_run_can_be_resumed_more_than_once(context):
    ids = ids_of("dev", context)
    first_id, _ = interrupted(context, fail_after=2)
    second = gold(context, fail_after=3)
    with pytest.raises(PredictorError) as caught:
        evaluate(second, "dev", context, resume=first_id)
    second_id = caught.value.run_id
    assert second.asked == ids[2:5]
    assert len(read_lines(second_id, context)) == 5

    third = gold(context)
    summary = evaluate(third, "dev", context, resume=second_id)
    assert third.asked == ids[5:] and summary["result"]
    assert [line["run_id"] for line in read_lines(summary["run_id"], context)] == (
        [first_id] * 2 + [second_id] * 3 + [summary["run_id"]] * 3
    )
    assert sorted(record.examples for record in ledger_records(context)) == [2, 3, 3]
    assert "the ledger records of 3 run(s) cover 8 predictions" in check(summary, "R4")["detail"]


def test_a_different_batch_size_may_continue_a_run(context):
    failed_id, _ = interrupted(context, fail_after=3)
    second = gold(context, batch_size=4)
    assert evaluate(second, "dev", context, resume=failed_id)["result"]
    assert second.asked == ids_of("dev", context)[3:]


def test_a_partial_run_can_be_continued_to_the_whole_split(context):
    first = evaluate(gold(context), "dev", context, limit=3)
    assert first["partial"] and results_lines(context) == []
    second = gold(context)
    summary = evaluate(second, "dev", context, resume=first["run_id"])
    assert summary["result"] and not summary["partial"]
    assert second.asked == ids_of("dev", context)[3:]
    assert len(results_lines(context)) == 1


# --------------------------------------------------------------------------- a killed run


def test_a_killed_run_gets_its_usage_recorded_when_it_is_resumed(context):
    failed_id, _ = interrupted(context, fail_after=4, model_id="vendor/model")
    kept = read_lines(failed_id, context)
    # A killed process writes no ledger record: take the record away again.
    context.tracking_cfg.ledger_file.write_text("", encoding="utf-8")

    summary = evaluate(gold(context, model_id="vendor/model"), "dev", context, resume=failed_id)
    records = {record.run_id: record for record in ledger_records(context)}
    rebuilt = records[failed_id]
    assert rebuilt.examples == 4 and rebuilt.source == "computed"
    assert "rebuilt from the predictions file" in rebuilt.source_detail
    assert summary["run_id"] in rebuilt.source_detail
    assert rebuilt.model_id == "vendor/model"
    assert rebuilt.input_tokens == sum(line["input_tokens"] for line in kept)
    assert rebuilt.output_tokens == sum(line["output_tokens"] for line in kept)
    assert (rebuilt.path, rebuilt.arm, rebuilt.where) == ("none", "none", "local")
    assert rebuilt.wall_clock_seconds is None  # not known, so not invented
    assert records[summary["run_id"]].examples == 4
    assert check(summary, "R4")["passed"] and summary["result"]


def test_a_cut_off_last_line_is_dropped_and_that_example_predicted_again(context):
    failed_id, _ = interrupted(context, fail_after=4)
    path = predictions_path(failed_id, context)
    path.write_text(path.read_text(encoding="utf-8")[:-25], encoding="utf-8", newline="\n")
    context.tracking_cfg.ledger_file.write_text("", encoding="utf-8")  # killed mid-write

    second = gold(context)
    summary = evaluate(second, "dev", context, resume=failed_id)
    assert second.asked == ids_of("dev", context)[3:]
    assert summary["result"] and len(read_lines(summary["run_id"], context)) == 8
    assert sorted(record.examples for record in ledger_records(context)) == [3, 5]


def test_a_prediction_paid_for_twice_is_recorded_twice(context):
    """The line was lost after its usage was recorded: the ledger says more than the file."""
    failed_id, _ = interrupted(context, fail_after=4)
    path = predictions_path(failed_id, context)
    path.write_text(path.read_text(encoding="utf-8")[:-25], encoding="utf-8", newline="\n")

    summary = evaluate(gold(context), "dev", context, resume=failed_id)
    assert sorted(record.examples for record in ledger_records(context)) == [4, 5]
    assert check(summary, "R4")["passed"] and summary["result"]


# --------------------------------------------------------------------------- refusals


def reworded_prompt(tmp_path):
    raw = yaml.safe_load((CONFIG_DIR / "prompt.yaml").read_text(encoding="utf-8"))
    raw["system"] = raw["system"].replace("financial", "careful")
    folder = tmp_path / "reworded"
    folder.mkdir()
    (folder / "prompt.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_prompt(root=tmp_path, config_dir=folder)


def test_resume_is_refused_under_other_settings_and_starts_no_run(context, tmp_path):
    failed_id, _ = interrupted(context, fail_after=3)
    before = run_folders(context)
    cases = [
        (gold(context, "quick_dev"), "quick_dev", context, "another split"),
        (gold(context, name="other"), "dev", context, "another predictor"),
        (gold(context, settings={"temperature": 0.7}), "dev", context, "another predictor"),
        (gold(context, model_id="vendor/model"), "dev", context, "another predictor"),
        (
            gold(context),
            "dev",
            dataclasses.replace(context, max_new_tokens=64),
            "another max new tokens",
        ),
        (
            gold(context),
            "dev",
            dataclasses.replace(context, prompt=reworded_prompt(tmp_path)),
            "another prompt id",
        ),
    ]
    for predictor, split, other, message in cases:
        with pytest.raises(ResumeError, match=message):
            evaluate(predictor, split, other, resume=failed_id)
        assert predictor.asked == []
    assert run_folders(context) == before
    assert len(ledger_records(context)) == 1


@pytest.mark.parametrize("damage", ["repeated", "out of order", "not json", "other example"])
def test_resume_is_refused_when_the_file_cannot_be_trusted(context, damage):
    failed_id, _ = interrupted(context, fail_after=4)
    lines = read_lines(failed_id, context)
    if damage == "repeated":
        lines = [lines[0], lines[0], lines[2], lines[3]]
    elif damage == "out of order":
        lines = [lines[1], lines[0], lines[2], lines[3]]
    elif damage == "other example":
        lines[1]["example_id"] = ids_of("fit", context)[0]
    write_lines(failed_id, context, lines)
    if damage == "not json":
        path = predictions_path(failed_id, context)
        text = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join([text[0], "{broken", *text[2:]]) + "\n", encoding="utf-8")

    before = run_folders(context)
    predictor = gold(context)
    with pytest.raises(ResumeError, match=f"run {failed_id} cannot be continued: .* line "):
        evaluate(predictor, "dev", context, resume=failed_id)
    assert predictor.asked == [] and run_folders(context) == before


def test_resume_of_a_finished_an_unknown_or_another_kind_of_run(context):
    whole = evaluate(gold(context), "dev", context)
    with pytest.raises(ResumeError, match="already has a prediction for every example"):
        evaluate(gold(context), "dev", context, resume=whole["run_id"])

    with pytest.raises(ResumeError, match="has no evaluation.json"):
        evaluate(gold(context), "dev", context, resume="20261116T000000Z-eval-nothing-0000")
    with pytest.raises(ResumeError, match="is not a run id"):
        evaluate(gold(context), "dev", context, resume="../outside")

    with start_run(
        "train",
        "not an evaluation",
        path="A",
        arm="A1",
        where="local",
        scope="dev",
        cfg=context.tracking_cfg,
        git=GIT,
        environment={},
    ) as other:
        pass
    with pytest.raises(ResumeError, match="it is not an evaluation run"):
        evaluate(gold(context), "dev", context, resume=other.run_id)

    folder = context.tracking_cfg.runs_dir / whole["run_id"]
    (folder / "evaluation.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ResumeError, match="is damaged"):
        evaluate(gold(context), "dev", context, resume=whole["run_id"])
    assert len(run_folders(context)) == 2


def test_a_limit_smaller_than_what_was_predicted_is_refused(context):
    failed_id, _ = interrupted(context, fail_after=5)
    with pytest.raises(ResumeError, match="more lines than the 3 examples asked for"):
        evaluate(gold(context), "dev", context, resume=failed_id, limit=3)


def test_the_resumed_file_is_valid_json_lines(context):
    failed_id, _ = interrupted(context, fail_after=3, batch_size=2)
    predictor = FakePredictor(lambda request: GOLD_REPLY, name="gold", batch_size=2)
    summary = evaluate(predictor, "dev", context, resume=failed_id)
    data = predictions_path(summary["run_id"], context).read_bytes()
    assert data.endswith(b"\n") and data.count(b"\n") == 8 and b"\r" not in data
    assert all(json.loads(line) for line in data.splitlines())
