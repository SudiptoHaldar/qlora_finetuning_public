"""Tests for the evaluation runner and its summary, on an invented split (req_002 v2.03).

Interruptions and resuming are in ``test_runner_resume.py``.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

import pytest
import yaml

from qlora_finetuning.data import (
    TestLockedError,
    load_data_config,
    load_prompt,
    serialize_split,
    verify_snapshot,
)
from qlora_finetuning.evaluation.fake import FakePredictor, gold_predictor
from qlora_finetuning.evaluation.metrics import (
    Allowance,
    EvaluationError,
    load_evaluation_config,
    summarize,
)
from qlora_finetuning.evaluation.predictions import score_from_dict
from qlora_finetuning.evaluation.predictor import Prediction, PredictorError
from qlora_finetuning.evaluation.runner import (
    EVALUATION_NAME,
    MEASUREMENTS_NAME,
    SUMMARY_NAME,
    RunnerContext,
    latency_stats,
    summarize_run,
)
from qlora_finetuning.settings import CONFIG_DIR, REPO_ROOT
from qlora_finetuning.tracking import TrackingError, load_manifest, load_tracking_config

from .conftest import (
    CONFIGS,
    GIT,
    GOLD_REPLY,
    evaluate,
    ids_of,
    ledger_records,
    read_lines,
    records_for,
    results_lines,
    run_folders,
    write_lines,
)

METRIC_NAMES = ("format_validity", "execution_accuracy", "answer_accuracy", "program_match")


def counts(summary, block="all"):
    return [summary[block][name]["count"] for name in METRIC_NAMES]


def checks(summary):
    return {check["id"]: check for check in summary["checks"]}


def gold(split, context, **options):
    return gold_predictor(records_for(split, context), **options)


class Bare:
    """A predictor that reports no token counts."""

    name = "bare"
    model_id = "vendor/bare"
    settings: dict = {}
    batch_size = 2
    usage_source = "aws_api"

    def predict(self, requests):
        return [Prediction(GOLD_REPLY) for _ in requests]


# --------------------------------------------------------------------------- a clean run


def test_gold_predictor_scores_every_example_right(context):
    predictor = gold("dev", context)
    summary = evaluate(predictor, "dev", context)
    ids = ids_of("dev", context)

    assert summary["result"] and summary["complete"] and not summary["partial"]
    assert summary["examples_predicted"] == summary["examples_asked"] == 8
    assert summary["split_examples"] == 8 and summary["limit"] is None
    assert counts(summary) == [8, 8, 8, 8] and counts(summary, "unflagged") == [8, 8, 8, 8]
    assert summary["all"]["execution_accuracy"] == {"count": 8, "rate": 1.0}
    assert summary["diagnostic"]["execution_accuracy_tight"] == {"count": 8, "rate": 1.0}
    assert summary["diagnostic"]["rule"] == {"absolute": 0.00001, "relative": 0.00001}
    assert summary["flagged_examples"] == 0 and summary["continues"] is None
    assert [check["id"] for check in summary["checks"]] == ["R1", "R2", "R3", "R4", "R5"]
    assert all(check["passed"] for check in summary["checks"]), summary["checks"]
    assert predictor.asked == ids  # dataset order, each once


def test_the_predictions_file_holds_one_line_per_example_and_no_prompt_text(context):
    summary = evaluate(gold("dev", context), "dev", context)
    run_id = summary["run_id"]
    lines = read_lines(run_id, context)
    assert [line["example_id"] for line in lines] == ids_of("dev", context)
    assert [line["index"] for line in lines] == list(range(8))
    first = lines[0]
    assert first["reply"] == GOLD_REPLY and first["run_id"] == run_id
    assert first["prompt_id"] == context.prompt.prompt_id
    assert first["outcome"]["value"] == 825.0 and first["outcome"]["strict"] is True
    assert first["outcome"]["program"] == "subtract(920, 95)"
    assert first["score"]["execution_correct"] is True
    assert first["score"]["example_id"] == first["example_id"]
    assert first["stop_reason"] == "end" and first["latency_seconds"] == 0.0
    assert first["input_tokens"] > 0 and first["output_tokens"] == 5

    folder = context.tracking_cfg.runs_dir / run_id
    text = (folder / "predictions.jsonl").read_text(encoding="utf-8")
    assert "Question:" not in text and "zebrafish" not in text  # no prompt text (D16)
    assert context.prompt.system[:40] not in text
    assert sorted(path.name for path in folder.iterdir()) == [
        "config",
        "evaluation.json",
        "manifest.json",
        "predictions.jsonl",
        "summary.json",
    ]
    assert json.loads((folder / SUMMARY_NAME).read_text(encoding="utf-8")) == summary


def test_the_run_is_in_the_registry_with_one_ledger_record(context):
    summary = evaluate(gold("dev", context), "dev", context, label="gold dev")
    manifest = load_manifest(summary["run_id"], context.tracking_cfg)
    assert manifest["status"] == "completed" and manifest["kind"] == "smoke"
    assert (manifest["path"], manifest["arm"], manifest["where"]) == ("none", "none", "local")
    assert manifest["split_manifest_hash"] == context.manifest["content_hash"]
    assert sorted(manifest["configs"]) == sorted(CONFIGS)
    assert manifest["outputs"] == {
        "evaluation": "evaluation.json",
        "predictions": "predictions.jsonl",
        "summary": "summary.json",
    }

    (record,) = ledger_records(context)
    assert record.run_id == summary["run_id"] and record.phase == "eval"
    assert record.examples == 8 and record.wall_clock_seconds >= 0
    assert record.source == "measured"
    # A fake predictor names no model, so no token count reaches the ledger (D28).
    assert record.input_tokens is None and record.model_id is None

    evaluation = json.loads(
        (context.tracking_cfg.runs_dir / summary["run_id"] / EVALUATION_NAME).read_text("utf-8")
    )
    assert evaluation["key"] == {
        "split": "dev",
        "manifest_hash": context.manifest["content_hash"],
        "prompt_id": context.prompt.prompt_id,
        "max_new_tokens": 192,
        "predictor": {"name": "gold", "model_id": None, "settings": {}},
    }
    assert evaluation["examples_asked"] == 8 and evaluation["continues"] is None


def test_a_complete_run_adds_one_line_of_counts_to_the_results_file(context):
    summary = evaluate(gold("quick_dev", context), "quick_dev", context)
    (line,) = results_lines(context)
    assert line["run_id"] == summary["run_id"] and line["split"] == "quick_dev"
    assert [line[name] for name in METRIC_NAMES] == [3, 3, 3, 3]
    assert line["examples"] == 3 and line["execution_accuracy_tight"] == 3
    assert line["unflagged_examples"] == 3 and line["predictor"] == "gold"
    assert line["manifest_hash"] == context.manifest["content_hash"]
    assert line["git_commit"] == GIT.commit and line["kind"] == "smoke"
    text = context.config.results_file.read_text(encoding="utf-8")
    assert "PROGRAM" not in text and "subtract" not in text  # counts only


def test_requests_carry_the_canonical_texts_unchanged(context):
    seen = []

    def reply(request):
        seen.append(request)
        return GOLD_REPLY

    evaluate(FakePredictor(reply, batch_size=3), "dev", context)
    records = records_for("dev", context)
    assert [item.example_id for item in seen] == [record.example_id for record in records]
    for request, record in zip(seen, records, strict=True):
        assert request.system == record.system and request.user == record.user
        assert request.max_new_tokens == 192


def test_batches_follow_the_predictor_and_keep_the_order(context):
    sizes = []

    class Batched(FakePredictor):
        def predict(self, requests):
            sizes.append(len(requests))
            return super().predict(requests)

    predictor = Batched(lambda request: GOLD_REPLY, batch_size=3)
    summary = evaluate(predictor, "dev", context)
    assert sizes == [3, 3, 2] and summary["result"]
    assert predictor.asked == ids_of("dev", context)


# --------------------------------------------------------------------------- wrong replies


@pytest.mark.parametrize(
    ("change", "expected", "deviation", "code"),
    [
        (lambda text: text.replace("PROGRAM:", "program:"), [0, 8, 8, 8], "label_case", None),
        (lambda text: "Sure!\n" + text, [0, 8, 8, 8], "text_before", None),
        (lambda text: "", [0, 0, 0, 0], None, "reply_empty"),
        (lambda text: "I cannot tell.", [0, 0, 0, 0], None, "no_program"),
        (lambda text: "PROGRAM: divide(9, 0)\nANSWER: 825", [8, 0, 8, 0], None, "division_by_zero"),
        (lambda text: "PROGRAM: add(800, 25)\nANSWER: 900", [8, 8, 0, 0], None, None),
    ],
)
def test_a_reply_that_is_wrong_is_scored_and_stays_in_every_total(
    context, change, expected, deviation, code
):
    summary = evaluate(FakePredictor(lambda request: change(GOLD_REPLY)), "dev", context)
    assert counts(summary) == expected
    assert summary["all"]["examples"] == 8  # nothing leaves a denominator
    assert summary["result"]  # every example has a prediction, however bad
    assert summary["all"]["deviations"] == ({deviation: 8} if deviation else {})
    assert summary["all"]["error_codes"] == ({code: 8} if code else {})


def test_a_mixed_run_equals_summarize_over_its_scores(context):
    ids = ids_of("dev", context)
    replies = {name: GOLD_REPLY for name in ids[:5]}
    replies.update({name: "PROGRAM: add(1, 2)\nANSWER: 3" for name in ids[5:]})
    summary = evaluate(FakePredictor(replies), "dev", context)
    assert counts(summary) == [8, 5, 5, 5]
    assert summary["all"]["execution_accuracy"]["rate"] == 5 / 8
    assert summary["all"]["mismatches"] == {"other": 3}
    stored = [score_from_dict(line["score"]) for line in read_lines(summary["run_id"], context)]
    assert summary["all"] == summarize(stored)["all"]


def test_a_reply_cut_off_at_the_limit_is_still_scored(context):
    predictor = FakePredictor(lambda request: "PROGRAM: subtract(920,", stop_reason="length")
    summary = evaluate(predictor, "dev", context)
    assert summary["stop_reasons"] == {"length": 8}
    assert counts(summary) == [0, 0, 0, 0] and summary["result"]
    assert summary["all"]["executed"]["count"] == 0


# --------------------------------------------------------------------------- limit


def test_a_limited_run_is_partial_and_never_a_result(context):
    predictor = gold("dev", context)
    summary = evaluate(predictor, "dev", context, limit=3)
    assert summary["partial"] and summary["complete"] and not summary["result"]
    assert (summary["examples_asked"], summary["examples_predicted"]) == (3, 3)
    assert summary["split_examples"] == 8 and summary["limit"] == 3
    assert counts(summary) == [3, 3, 3, 3]
    assert "the first 3 of 8: a partial run" in checks(summary)["R1"]["detail"]
    assert predictor.asked == ids_of("dev", context)[:3]
    assert results_lines(context) == []  # not listed as a result (D12)
    assert ledger_records(context)[0].examples == 3


def test_a_limit_of_the_whole_split_is_not_partial(context):
    summary = evaluate(gold("dev", context), "dev", context, limit=50)
    assert not summary["partial"] and summary["result"] and summary["limit"] is None


@pytest.mark.parametrize("limit", [0, -1, 1.5, True, "3"])
def test_a_limit_that_is_not_a_positive_whole_number(context, limit):
    with pytest.raises(EvaluationError, match="limit must be a positive whole number"):
        evaluate(gold("dev", context), "dev", context, limit=limit)
    assert run_folders(context) == []


# --------------------------------------------------------------------------- usage


def test_token_counts_reach_the_ledger_when_the_predictor_names_a_model(context):
    predictor = gold("dev", context, model_id="vendor/model", latency_seconds=0.25)
    summary = evaluate(predictor, "dev", context, path="B", arm="B0-Lite", kind="eval")
    lines = read_lines(summary["run_id"], context)
    (record,) = ledger_records(context)
    assert record.model_id == "vendor/model" and record.source == "computed"
    assert record.input_tokens == sum(line["input_tokens"] for line in lines) > 0
    assert record.output_tokens == 8 * 5
    assert (record.path, record.arm, record.examples) == ("B", "B0-Lite", 8)
    assert summary["usage"] == {
        "input_tokens": record.input_tokens,
        "output_tokens": 40,
        "predictions_without_token_counts": 0,
    }
    assert summary["latency_seconds"] == {
        "mean": 0.25,
        "median": 0.25,
        "p95": 0.25,
        "max": 0.25,
        "total": 2.0,
    }
    assert results_lines(context)[0]["model_id"] == "vendor/model"


def test_unknown_token_counts_are_left_out_not_guessed(context):
    summary = evaluate(Bare(), "dev", context)
    (record,) = ledger_records(context)
    assert record.input_tokens is None and record.model_id is None
    assert record.source == "measured" and record.examples == 8
    assert summary["usage"] == {
        "input_tokens": None,
        "output_tokens": None,
        "predictions_without_token_counts": 8,
    }
    assert summary["result"]


def test_latency_statistics():
    assert latency_stats([4.0, 1.0, 3.0, 2.0]) == {
        "mean": 2.5,
        "median": 2.5,
        "p95": 4.0,
        "max": 4.0,
        "total": 10.0,
    }
    values = [float(number) for number in range(1, 101)]
    assert latency_stats(values)["p95"] == 95.0 and latency_stats(values)["median"] == 50.5
    assert latency_stats([]) == {
        "mean": None,
        "median": None,
        "p95": None,
        "max": None,
        "total": None,
    }


# --------------------------------------------------------------------------- the summary


def test_summarize_run_gives_the_same_totals_from_the_file(context):
    summary = evaluate(gold("dev", context), "dev", context)
    assert summarize_run(summary["run_id"], context) == summary
    assert summarize_run(summary["run_id"], context) == summary  # and again


def test_the_tight_diagnostic_and_a_changed_allowance_need_no_predictor(context):
    # 827 against a gold of 825 is within 0.5% and outside 0.001%.
    near = "PROGRAM: add(825, 2)\nANSWER: 827"
    summary = evaluate(FakePredictor(lambda request: near), "dev", context)
    assert summary["all"]["execution_accuracy"]["count"] == 8
    assert summary["diagnostic"]["execution_accuracy_tight"] == {"count": 0, "rate": 0.0}
    assert summary["diagnostic"]["execution_accuracy_tight_unflagged"]["count"] == 0
    assert results_lines(context)[0]["execution_accuracy_tight"] == 0

    tight = dataclasses.replace(
        context.config, execution=Allowance(absolute=0.00001, relative=0.00001)
    )
    again = summarize_run(summary["run_id"], dataclasses.replace(context, config=tight))
    assert again["all"]["execution_accuracy"]["count"] == 0  # rescored from stored values
    assert again["all"]["answer_accuracy"]["count"] == 8
    stale = checks(again)["R3"]
    assert not stale["passed"] and "8 lines hold a score that differs" in stale["detail"]
    assert not again["complete"] and not again["result"]


def test_flagged_examples_stay_in_the_headline(context):
    from qlora_finetuning.data.gold_check import build_flags, write_flags

    ids = ids_of("dev", context)
    flagged = [{"example_id": ids[0], "split": "dev", "reason": "value_mismatch"}]
    write_flags(build_flags({"flagged": flagged, "settings": {}}), context.config)
    summary = evaluate(gold("dev", context), "dev", context)
    assert summary["all"]["examples"] == 8 and summary["unflagged"]["examples"] == 7
    assert summary["flagged_examples"] == 1
    assert summary["diagnostic"]["execution_accuracy_tight_unflagged"]["count"] == 7
    assert results_lines(context)[0]["unflagged_examples"] == 7


def test_summary_of_an_unknown_or_malformed_run_id(context):
    with pytest.raises(EvaluationError, match="has no evaluation.json"):
        summarize_run("20261116T000000Z-eval-nothing-0000", context)
    for bad in ("../outside", "a/b", "", "..", None):
        with pytest.raises(EvaluationError, match="is not a run id"):
            summarize_run(bad, context)


# --------------------------------------------------------------------------- checks


def test_r1_fails_when_a_line_is_missing(context):
    summary = evaluate(gold("dev", context), "dev", context)
    lines = read_lines(summary["run_id"], context)
    write_lines(summary["run_id"], context, lines[:-1])
    again = summarize_run(summary["run_id"], context)
    check = checks(again)["R1"]
    assert not check["passed"] and check["detail"] == "7 of 8 examples have a prediction"
    assert not again["complete"] and again["all"]["examples"] == 7

    path = context.tracking_cfg.runs_dir / summary["run_id"] / "predictions.jsonl"
    path.write_text(path.read_text(encoding="utf-8")[:-30], encoding="utf-8", newline="\n")
    assert "the last line is cut off" in checks(summarize_run(summary["run_id"], context))["R1"][
        "detail"
    ]


def test_r2_fails_under_another_prompt_wording(context, tmp_path):
    summary = evaluate(gold("dev", context), "dev", context)
    raw = yaml.safe_load((CONFIG_DIR / "prompt.yaml").read_text(encoding="utf-8"))
    raw["system"] = raw["system"].replace("financial", "careful")
    folder = tmp_path / "reworded"
    folder.mkdir()
    (folder / "prompt.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    reworded = dataclasses.replace(context, prompt=load_prompt(root=tmp_path, config_dir=folder))
    check = checks(summarize_run(summary["run_id"], reworded))["R2"]
    assert not check["passed"]
    assert "8 lines were predicted from another prompt wording" in check["detail"]
    assert "the run was made with another prompt wording" in check["detail"]


def test_r4_fails_when_usage_went_unrecorded(context):
    summary = evaluate(gold("dev", context), "dev", context)
    context.tracking_cfg.ledger_file.write_text("", encoding="utf-8")
    check = checks(summarize_run(summary["run_id"], context))["R4"]
    assert not check["passed"]
    assert "has 8 lines and only 0 examples in the ledger" in check["detail"]


def test_r5_fails_on_a_line_without_an_outcome(context):
    summary = evaluate(gold("dev", context), "dev", context)
    lines = read_lines(summary["run_id"], context)
    lines[2]["outcome"]["value"] = None  # neither a value nor an error
    lines[4]["score"]["example_id"] = lines[0]["example_id"]
    write_lines(summary["run_id"], context, lines)
    check = checks(summarize_run(summary["run_id"], context))["R5"]
    assert not check["passed"] and check["detail"].startswith("2 lines have no outcome")


# --------------------------------------------------------------------------- failures


def test_a_malformed_answer_fails_the_run_and_scores_nothing(context):
    class Short(FakePredictor):
        def predict(self, requests):
            return super().predict(requests)[:-1] if len(self.asked) >= 4 else (
                super().predict(requests)
            )

    predictor = Short(lambda request: GOLD_REPLY, batch_size=2)
    with pytest.raises(PredictorError, match="returned 1 predictions for 2 requests") as caught:
        evaluate(predictor, "dev", context)
    run_id = caught.value.run_id
    manifest = load_manifest(run_id, context.tracking_cfg)
    assert manifest["status"] == "failed" and manifest["error"].startswith("PredictorError")
    assert len(read_lines(run_id, context)) == 4  # the bad batch left no line
    assert ledger_records(context)[0].examples == 4
    assert results_lines(context) == []


def test_any_other_error_in_a_predictor_also_keeps_lines_and_usage(context):
    def reply(request):
        if request.example_id == ids_of("dev", context)[5]:
            raise RuntimeError("out of memory")
        return GOLD_REPLY

    with pytest.raises(RuntimeError, match="out of memory"):
        evaluate(FakePredictor(reply), "dev", context)
    (run_id,) = run_folders(context)
    assert load_manifest(run_id, context.tracking_cfg)["status"] == "failed"
    assert len(read_lines(run_id, context)) == 5
    assert ledger_records(context)[0].examples == 5


def test_a_predictor_that_does_not_describe_itself_starts_no_run(context):
    predictor = gold("dev", context, batch_size=0)
    with pytest.raises(PredictorError, match="batch_size") as caught:
        evaluate(predictor, "dev", context)
    assert caught.value.run_id is None and run_folders(context) == []

    leaky = gold("dev", context, settings={"contact": "someone@example.com"})
    with pytest.raises(TrackingError, match="email address"):
        evaluate(leaky, "dev", context)
    assert run_folders(context) == []


# --------------------------------------------------------------------------- the test lock


def test_the_locked_test_split_is_refused_before_the_predictor_is_asked(context):
    predictor = FakePredictor(lambda request: GOLD_REPLY)
    with pytest.raises(TestLockedError, match="the test split is locked"):
        evaluate(predictor, "test", context)
    assert predictor.asked == [] and run_folders(context) == []
    assert ledger_records(context) == []


def test_an_unlocked_test_split_is_guarded_by_the_gate(context):
    # After the unlock the lock protects nothing; the gate of req_005 v5.01 takes over.
    # Its own tests are in tests/comparison/test_gate.py.
    from qlora_finetuning.comparison.gate import TestRunRefused
    from tests.comparison.helpers import freeze_and_unlock

    unlock = context.split_cfg.unlock_file
    unlock.parent.mkdir(parents=True, exist_ok=True)
    unlock.write_text(
        json.dumps({"manifest_hash": context.manifest["content_hash"]}), encoding="utf-8"
    )
    predictor = gold("test", context)  # the loader alone is satisfied by the unlock record
    with pytest.raises(TestRunRefused, match="there is no freeze record"):
        evaluate(predictor, "test", context)
    assert predictor.asked == [] and run_folders(context) == []

    freeze_and_unlock(context)
    summary = evaluate(gold("test", context), "test", context)
    assert summary["result"] and counts(summary) == [22, 22, 22, 22]


def test_fit_can_be_evaluated(context):
    assert counts(evaluate(gold("fit", context, batch_size=7), "fit", context)) == [30] * 4


# --------------------------------------------------------------------------- guards


@pytest.mark.parametrize(
    "first",
    [
        "qlora_finetuning.evaluation",
        "qlora_finetuning.data.gold_check",
        "qlora_finetuning.evaluation.runner",
        "scripts.evaluate",
    ],
)
def test_every_import_order_works(first):
    """The runner imports the data package, whose gold self-check imports the metrics."""
    code = (
        f"import sys, {first}\n"
        "import qlora_finetuning.evaluation.runner, qlora_finetuning.data.gold_check\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_the_package_itself_loads_without_the_data_package():
    code = (
        "import sys, qlora_finetuning.evaluation\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('qlora_finetuning.data'))\n"
        "sys.exit(1 if loaded or 'qlora_finetuning.evaluation.runner' in sys.modules else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_evaluation_package_imports_neither_training_path():
    folder = REPO_ROOT / "qlora_finetuning" / "evaluation"
    for path in sorted(folder.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "local_qlora" not in text and "frontier" not in text, path.name
    init = (folder / "__init__.py").read_text(encoding="utf-8")
    assert "from qlora_finetuning.evaluation.runner" not in init.split('"""')[2]


# --------------------------------------------------------------------------- the real data


@pytest.mark.parametrize(("split", "size"), [("quick_dev", 200), ("dev", 499)])
def test_the_gold_predictor_on_the_real_split(tmp_path, split, size):
    """Counts only. Everything is written under the temporary folder."""
    if not all(item.ok for item in verify_snapshot(load_data_config())):
        pytest.skip("no FinQA snapshot here; fetch it with: python -m scripts.finqa_fetch")
    config = dataclasses.replace(
        load_evaluation_config(), results_file=tmp_path / "evaluations.jsonl"
    )
    context = RunnerContext(
        config=config, tracking_cfg=load_tracking_config(root=tmp_path), git=GIT, environment={}
    )
    predictor = gold_predictor(serialize_split(split), batch_size=16)
    summary = evaluate(predictor, split, context)
    assert summary["result"] and counts(summary) == [size] * 4
    assert summary["diagnostic"]["execution_accuracy_tight"]["count"] == size
    assert summary["max_new_tokens"] == 192


# --------------------------------------------------------------------------- billed as another


def test_usage_is_recorded_under_the_id_the_model_is_billed_as(context):
    """A tuned model is called by its deployment and billed as its base model."""
    predictor = gold("dev", context, model_id="arn:aws:bedrock:deployment/tuned")
    predictor.usage_model_id = "vendor/base"
    predictor.details = {"attempt": 1}
    summary = evaluate(predictor, "dev", context, path="B", arm="B1-Lite", kind="eval")
    (record,) = ledger_records(context)
    assert record.model_id == "vendor/base" and record.arm == "B1-Lite"
    assert record.input_tokens == summary["usage"]["input_tokens"] > 0
    # The run itself says what was called, and what else identifies it.
    assert summary["predictor"]["model_id"] == "arn:aws:bedrock:deployment/tuned"
    assert summary["predictor"]["usage_model_id"] == "vendor/base"
    assert summary["predictor"]["details"] == {"attempt": 1}
    assert results_lines(context)[0]["model_id"] == "arn:aws:bedrock:deployment/tuned"


def test_run_predictions_gives_the_lines_of_a_run_in_dataset_order(context):
    from qlora_finetuning.evaluation.runner import run_predictions

    summary = evaluate(gold("dev", context), "dev", context)
    lines = run_predictions(summary["run_id"], context)
    assert [line.example_id for line in lines] == ids_of("dev", context)
    assert all(line.score["execution_correct"] for line in lines)
    limited = evaluate(gold("dev", context), "dev", context, limit=3)
    assert len(run_predictions(limited["run_id"], context)) == 3
    with pytest.raises(EvaluationError, match="has no evaluation.json"):
        run_predictions("20261116T000000Z-eval-nothing-0000", context)


# --------------------------------------------------------------------------- measurements


class Measuring(FakePredictor):
    """A predictor that measured something about its own run (req_003 v3.01)."""

    notes = {"peak_vram_allocated_gib": 3.25, "gpu_name": "stand-in", "batches": 8}

    def measurements(self):
        if isinstance(self.notes, Exception):
            raise self.notes
        return self.notes


def measuring(context, **options):
    replies = {record.example_id: record.target for record in records_for("dev", context)}
    return Measuring(replies, **options)


def test_a_predictors_measurements_are_stored_with_the_run_and_shown_in_the_summary(context):
    summary = evaluate(measuring(context), "dev", context)
    folder = context.tracking_cfg.runs_dir / summary["run_id"]
    assert json.loads((folder / MEASUREMENTS_NAME).read_text(encoding="utf-8")) == Measuring.notes
    assert summary["backend"] == Measuring.notes and summary["result"]
    assert summarize_run(summary["run_id"], context)["backend"] == Measuring.notes
    assert load_manifest(summary["run_id"], context.tracking_cfg)["outputs"]["measurements"] == (
        MEASUREMENTS_NAME
    )


def test_measurements_are_kept_when_the_run_fails(context):
    with pytest.raises(PredictorError) as caught:
        evaluate(measuring(context, fail_after=3), "dev", context)
    folder = context.tracking_cfg.runs_dir / caught.value.run_id
    assert json.loads((folder / MEASUREMENTS_NAME).read_text(encoding="utf-8")) == Measuring.notes


@pytest.mark.parametrize(
    "notes",
    [RuntimeError("the GPU went away"), ["not", "a", "mapping"], {"peak": float("nan")},
     {"model": object()}],
)  # fmt: skip
def test_measurements_that_fail_or_are_not_plain_values_cost_the_run_nothing(context, notes):
    predictor = measuring(context)
    predictor.notes = notes
    summary = evaluate(predictor, "dev", context)
    assert summary["result"] and counts(summary) == [8] * 4
    assert "backend" not in summary
    assert not (context.tracking_cfg.runs_dir / summary["run_id"] / MEASUREMENTS_NAME).exists()


def test_a_predictor_without_measurements_runs_exactly_as_before(context):
    summary = evaluate(gold_predictor(records_for("dev", context)), "dev", context)
    folder = context.tracking_cfg.runs_dir / summary["run_id"]
    assert "backend" not in summary and not (folder / MEASUREMENTS_NAME).exists()
    assert "measurements" not in load_manifest(summary["run_id"], context.tracking_cfg)["outputs"]
    assert "backend" not in json.loads((folder / SUMMARY_NAME).read_text(encoding="utf-8"))
