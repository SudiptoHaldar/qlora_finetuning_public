"""Tests for qlora_finetuning.frontier.tuned_report, on runs made by the real runner."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from qlora_finetuning.data import manifest_hash
from qlora_finetuning.evaluation.runner import run_predictions
from qlora_finetuning.frontier.pricing import inference_cost, load_pricing
from qlora_finetuning.frontier.training_estimate import read_gate_record
from qlora_finetuning.frontier.tuned_report import (
    build_report,
    model_of,
    paired,
    render_json,
    render_markdown,
    report_paths,
    sign_test,
    training_cost,
    verdict,
    write_report,
)
from qlora_finetuning.frontier.tuning_record import read_tuning_record, write_tuning_record
from qlora_finetuning.tracking import Ledger, LedgerRecord
from tests.evaluation.conftest import GOLD_REPLY, ids_of, records_for
from tests.frontier.conftest import INPUT_TOKENS, OUTPUT_TOKENS, StubProvider
from tests.frontier.export_helpers import detail, failed
from tests.frontier.tuned_helpers import (
    MALFORMED_REPLY,
    SPLITS,
    deploy_models,
    deployed_attempt,
    right_for,
    scripted_baseline,
    tuned_predictor,
    tuned_run,
)

MODELS = ("nova-pro", "nova-lite")
# Right or wrong per dev example. Tuning fixes 2, 3, and 5 and breaks 1.
B0_DEV = [True, True, False, False, True, False, False, False]
B1_DEV = [True, False, True, True, True, True, False, False]


def report_for(context, cfg, tuned, baselines, **changes):
    every = [*tuned, *baselines]
    arguments = {
        "record": read_tuning_record(cfg),
        "gate": read_gate_record(cfg),
        "ledger_records": Ledger(context.tracking_cfg).effective(),
        "prompt_id": context.prompt.prompt_id,
        "manifest_hash": manifest_hash(context.split_cfg),
    }
    arguments.update(changes)
    return build_report(
        tuned,
        baselines,
        {s["run_id"]: run_predictions(s["run_id"], context) for s in every},
        arguments["record"],
        arguments["gate"],
        arguments["ledger_records"],
        cfg,
        load_pricing(),
        prompt_id=arguments["prompt_id"],
        manifest_hash=arguments["manifest_hash"],
    )


def runs(context, cfg, models=MODELS, tuned_right=7, baseline_right=2, splits=SPLITS):
    """Baseline and tuned runs per model and split; ``*_right`` is the count right on dev."""
    baselines, tuned = [], []
    for split in splits:
        for name in models:
            total = len(ids_of(split, context))
            baselines.append(
                scripted_baseline(
                    context, cfg, name, split, right_for(split, min(baseline_right, total))
                )
            )
            tuned.append(
                tuned_run(context, cfg, name, split, right_for(split, min(tuned_right, total)))
            )
    return tuned, baselines


@pytest.fixture
def scored(context, cfg):
    deploy_models(cfg, context)
    return runs(context, cfg)


# --------------------------------------------------------------------------- the arithmetic


@pytest.mark.parametrize(
    ("fixed", "broken", "expected"),
    [
        (0, 0, None),  # no question changed: there is nothing to test
        (5, 0, 0.0625),  # 2 * (1/2)^5
        (0, 5, 0.0625),  # the test is two-sided
        (1, 1, 1.0),
        (10, 2, 0.0385742),  # 2 * (1 + 12 + 66) / 4096
        (3, 1, 0.625),  # 2 * (1 + 4) / 16
    ],
)
def test_the_sign_test_against_values_computed_by_hand(fixed, broken, expected):
    assert sign_test(fixed, broken) == expected


def test_a_lopsided_split_of_many_questions_is_far_from_chance():
    assert sign_test(120, 20) < 1e-15
    assert sign_test(60, 60) == 1.0


def test_pairs_are_counted_question_by_question(context, cfg):
    deploy_models(cfg, context, models=("nova-lite",))
    before = scripted_baseline(context, cfg, "nova-lite", "dev", B0_DEV)
    after = tuned_run(context, cfg, "nova-lite", "dev", B1_DEV)
    result = paired(
        run_predictions(before["run_id"], context), run_predictions(after["run_id"], context)
    )
    ids = ids_of("dev", context)
    assert (result["both_right"], result["both_wrong"]) == (2, 2)
    assert (result["fixed"], result["broken"]) == (3, 1)
    assert result["fixed_ids"] == [ids[2], ids[3], ids[5]] and result["broken_ids"] == [ids[1]]
    assert result["sign_test_p_value"] == 0.625
    assert result["broke_more_than_a_tenth_of_fixed"] is True  # 1 broken against 3 fixed


@pytest.mark.parametrize(
    ("fixed", "broken", "flag"), [(10, 1, False), (10, 2, True), (0, 0, False), (0, 1, True)]
)
def test_breaking_more_than_a_tenth_of_what_was_fixed_is_stated(fixed, broken, flag):
    class Line:
        def __init__(self, example_id, right):
            self.example_id, self.score = example_id, {"execution_correct": right}

    before = [Line(f"f{i}", False) for i in range(fixed)] + [
        Line(f"b{i}", True) for i in range(broken)
    ]
    after = [Line(f"f{i}", True) for i in range(fixed)] + [
        Line(f"b{i}", False) for i in range(broken)
    ]
    assert paired(before, after)["broke_more_than_a_tenth_of_fixed"] is flag


@pytest.mark.parametrize(
    ("b0", "b1", "well_formed", "expected", "gain_met", "format_met"),
    [
        (123, 172, 499, "missed", False, True),  # Nova Lite: 49 more is 9.82 points
        (123, 173, 499, "met", True, True),  # 50 more is 10.02 points
        (190, 239, 499, "missed", False, True),  # Nova Pro
        (190, 240, 499, "met", True, True),
        (123, 300, 489, "missed", True, False),  # 489 of 499 is 97.996%
        (123, 300, 490, "met", True, True),
        (123, 100, 499, "missed", False, True),  # tuning made it worse
        (123, 123, 480, "missed", False, False),
    ],
)
def test_the_verdict_turns_exactly_at_the_owners_two_thresholds(
    cfg, b0, b1, well_formed, expected, gain_met, format_met
):
    result = verdict(b0, b1, well_formed, 499, cfg)
    assert result["verdict"] == expected
    assert (result["execution_gain_met"], result["format_validity_met"]) == (gain_met, format_met)
    assert result["needed_execution_count"] == b0 + 50 and result["needed_format_count"] == 490
    assert result["corrective_iteration_eligible"] is (expected == "missed")
    assert len(result["reasons"]) == (not gain_met) + (not format_met)


def test_the_verdict_says_what_was_needed_and_what_was_reached(cfg):
    result = verdict(123, 172, 480, 499, cfg)
    assert (
        result["execution_gain_points"] == "9.82" and result["format_validity_percent"] == "96.19"
    )
    assert result["reasons"] == [
        "execution accuracy rose by 9.82 points; 10 are needed (173 of 499 correct, 172 were)",
        "format validity is 96.19%; 98% is needed (490 of 499 well-formed, 480 were)",
    ]


def test_a_run_without_examples_has_missed(cfg):
    result = verdict(0, 0, 0, 0, cfg)
    assert result["verdict"] == "missed" and result["reasons"] == ["the run has no examples"]
    assert result["execution_gain_points"] is None


# --------------------------------------------------------------------------- a good report


def test_two_models_scored_on_both_splits_give_a_passing_report(context, cfg, scored):
    tuned, baselines = scored
    report = report_for(context, cfg, tuned, baselines)
    assert report["passed"] and failed(report["checks"]) == []
    assert [c["id"] for c in report["checks"]] == ["V1", "V2", "V3", "V4", "V5", "V6"]
    assert report["all_met"] and not report["partial"] and report["models_without_runs"] == []
    assert [m["model"] for m in report["models"]] == ["nova-pro", "nova-lite"]  # config order

    lite = report["models"][1]
    assert (lite["untuned_arm"], lite["tuned_arm"]) == ("B0-Lite", "B1-Lite")
    assert [pair["split"] for pair in lite["pairs"]] == ["quick_dev", "dev"]
    dev = lite["pairs"][1]
    assert (dev["baseline"]["execution_accuracy"], dev["tuned"]["execution_accuracy"]) == (2, 7)
    assert dev["difference_points"]["execution_accuracy"] == "62.50"
    assert dev["difference_points"]["format_validity"] == "0.00"
    assert (dev["paired"]["fixed"], dev["paired"]["broken"]) == (5, 0)
    assert dev["paired"]["sign_test_p_value"] == 0.0625
    assert dev["tuned_format_failure_ids"] == []
    assert lite["verdict"]["verdict"] == "met" and lite["verdict"]["reasons"] == []
    assert lite["verdict"]["needed_execution_count"] == 3  # 2, and 10 points of 8 is one more


def test_cost_is_per_example_and_per_correct_answer_at_the_base_models_prices(context, cfg, scored):
    tuned, baselines = scored
    pro = report_for(context, cfg, tuned, baselines)["models"][0]["pairs"][1]
    cost = inference_cost(
        "amazon.nova-pro-v1:0", 8 * INPUT_TOKENS, 8 * OUTPUT_TOKENS, load_pricing()
    )
    for side, correct in (("baseline", 2), ("tuned", 7)):
        arm = pro[side]
        assert arm["priced_as"] == "amazon.nova-pro-v1:0"
        assert Decimal(arm["cost_usd"]) == cost.quantize(Decimal("0.000001"))
        assert Decimal(arm["cost_per_example_usd"]) == (cost / 8).quantize(Decimal("0.000001"))
        assert Decimal(arm["cost_per_correct_answer_usd"]) == (cost / correct).quantize(
            Decimal("0.000001")
        )
    assert pro["tuned"]["model_id"].startswith("arn:aws:bedrock:")  # what was called


def test_a_tuned_model_that_got_nothing_right_has_no_cost_per_correct_answer(context, cfg):
    deploy_models(cfg, context, models=("nova-lite",))
    tuned, baselines = runs(context, cfg, models=("nova-lite",), tuned_right=0, baseline_right=0)
    report = report_for(context, cfg, tuned, baselines)
    assert report["passed"] and not report["all_met"]
    assert report["models"][0]["pairs"][1]["tuned"]["cost_per_correct_answer_usd"] is None
    assert report["models"][0]["pairs"][1]["paired"]["sign_test_p_value"] is None
    assert "no question changed" in render_markdown(report)


def test_the_training_cost_is_the_estimate_until_the_bill_is_entered(context, cfg, scored):
    tuned, baselines = scored
    attempt = read_tuning_record(cfg).latest("nova-lite")
    ledger, pricing = Ledger(context.tracking_cfg), load_pricing()
    assert training_cost(attempt, ledger.effective(), pricing) is None  # nothing recorded
    estimate = LedgerRecord(
        run_id=tuned[0]["run_id"],
        phase="train",
        path="B",
        arm="B1-Lite",
        where="bedrock",
        source="computed",
        model_id=attempt.base_model_id,
        training_tokens=1_500_000,
        external_id=attempt.job_arn,
    )
    ledger.append(estimate)
    assert training_cost(attempt, ledger.effective(), pricing) == {
        "usd": "3.00",  # 1.5M tokens at $2.00 per million
        "basis": "estimate",
        "training_tokens": 1_500_000,
    }
    ledger.append(
        LedgerRecord(
            run_id=tuned[0]["run_id"],
            phase="train",
            path="B",
            arm="B1-Lite",
            where="bedrock",
            source="bill",
            model_id=attempt.base_model_id,
            billed_amount_usd="3.47",
            external_id=attempt.job_arn,
            replaces=estimate.record_id,
        )
    )
    assert training_cost(attempt, ledger.effective(), pricing) == {"usd": "3.47", "basis": "billed"}
    report = report_for(context, cfg, tuned, baselines)
    assert report["models"][1]["training_cost"] == {"usd": "3.47", "basis": "billed"}
    assert report["models"][0]["training_cost"] is None
    text = render_markdown(report)
    assert "| nova-lite | $3.47 | the AWS bill, entered by the owner |" in text
    assert "| nova-pro | - | not in the ledger |" in text


# --------------------------------------------------------------------------- the verdicts


def test_a_model_that_did_not_gain_enough_has_missed_and_v4_05_is_eligible(context, cfg):
    deploy_models(cfg, context)
    tuned, baselines = runs(context, cfg, tuned_right=2, baseline_right=2)
    report = report_for(context, cfg, tuned, baselines)
    assert report["passed"] and not report["all_met"]  # the checks pass; the models missed
    for model in report["models"]:
        assert model["verdict"]["verdict"] == "missed"
        assert model["verdict"]["corrective_iteration_eligible"] is True
        assert model["verdict"]["execution_gain_points"] == "0.00"
    text = render_markdown(report)
    assert text.count("**MISSED**") == 2
    assert "v4.05 (one corrective iteration) is eligible. Nothing starts by itself." in text


def test_a_model_that_gained_but_lost_the_format_has_missed(context, cfg):
    deploy_models(cfg, context, models=("nova-lite",))
    baseline = scripted_baseline(context, cfg, "nova-lite", "dev", right_for("dev", 2))
    tuned = tuned_run(context, cfg, "nova-lite", "dev", right_for("dev", 7), wrong=MALFORMED_REPLY)
    report = report_for(context, cfg, [tuned], [baseline])
    result = report["models"][0]["verdict"]
    assert (result["execution_gain_met"], result["format_validity_met"]) == (True, False)
    assert result["verdict"] == "missed"
    pair = report["models"][0]["pairs"][0]
    assert pair["tuned_format_failure_ids"] == [ids_of("dev", context)[7]]
    assert pair["tuned"]["format_validity"] == 7


def test_the_verdict_comes_from_dev_and_never_from_quick_dev(context, cfg):
    deploy_models(cfg, context, models=("nova-lite",))
    baselines = [
        scripted_baseline(context, cfg, "nova-lite", "quick_dev", right_for("quick_dev", 0)),
        scripted_baseline(context, cfg, "nova-lite", "dev", right_for("dev", 7)),
    ]
    tuned = [
        tuned_run(context, cfg, "nova-lite", "quick_dev", right_for("quick_dev", 3)),  # +100 points
        tuned_run(context, cfg, "nova-lite", "dev", right_for("dev", 7)),  # no gain
    ]
    report = report_for(context, cfg, tuned, baselines)
    assert report["passed"] and report["models"][0]["verdict"]["verdict"] == "missed"
    assert report["models"][0]["pairs"][0]["difference_points"]["execution_accuracy"] == "100.00"


def test_one_model_only_is_a_partial_report(context, cfg):
    deploy_models(cfg, context, models=("nova-lite",))
    tuned, baselines = runs(context, cfg, models=("nova-lite",))
    report = report_for(context, cfg, tuned, baselines)
    assert report["passed"] and report["partial"] and report["all_met"]
    assert report["models_without_runs"] == ["nova-pro"]
    assert [m["model"] for m in report["models"]] == ["nova-lite"]
    text = render_markdown(report)
    assert "**Partial report**: no tuned run was given for nova-pro" in text
    assert "Milestone M6 needs every model." in text


# --------------------------------------------------------------------------- the checks


def test_v1_fails_for_a_run_that_is_not_a_result(context, cfg):
    deploy_models(cfg, context, models=("nova-lite",))
    baseline = scripted_baseline(context, cfg, "nova-lite", "dev", right_for("dev", 2))
    partial = tuned_run(context, cfg, "nova-lite", "dev", right_for("dev", 7), limit=3)
    report = report_for(context, cfg, [partial], [baseline])
    assert not report["passed"] and report["models"] == []
    assert "V1" in failed(report["checks"]) and "V2" in failed(report["checks"])
    assert f"run {partial['run_id']} is not a result" in detail(report["checks"], "V1")
    assert "do not hold the same examples in the same order" in detail(report["checks"], "V2")
    assert not report["all_met"] and not report["partial"]


def test_v2_fails_without_exactly_one_baseline_per_tuned_run(context, cfg, scored):
    tuned, baselines = scored
    missing = report_for(context, cfg, tuned, baselines[1:])  # no quick-dev baseline for one
    assert failed(missing["checks"]) == ["V2"]
    assert "has 0 baseline runs, expected 1" in detail(missing["checks"], "V2")

    extra = scripted_baseline(context, cfg, "nova-lite", "dev", right_for("dev", 2))
    twice = report_for(context, cfg, tuned, [*baselines, extra])
    assert "(nova-lite on dev) has 2 baseline runs, expected 1" in detail(twice["checks"], "V2")

    unpaired = report_for(context, cfg, tuned[:3], baselines)
    assert "pairs with no tuned run" in detail(unpaired["checks"], "V2")

    again = tuned_run(context, cfg, "nova-lite", "dev", right_for("dev", 7))
    doubled = report_for(context, cfg, [*tuned, again], baselines)
    assert "more than one tuned run for nova-lite on dev" in detail(doubled["checks"], "V2")


def test_v2_fails_when_the_two_runs_of_a_pair_did_not_answer_the_same_question(context, cfg):
    deploy_models(cfg, context, models=("nova-lite",))
    baseline = scripted_baseline(context, cfg, "nova-lite", "dev", right_for("dev", 2))
    warmer = tuned_predictor(
        cfg, "nova-lite", StubProvider(), settings=dict(cfg.settings, temperature=0.7)
    )
    other_settings = tuned_run(context, cfg, "nova-lite", "dev", [], predictor=warmer)
    report = report_for(context, cfg, [other_settings], [baseline])
    assert failed(report["checks"]) == ["V2"]
    assert "made under different settings" in detail(report["checks"], "V2")

    good = tuned_run(context, cfg, "nova-lite", "dev", right_for("dev", 7))
    for change, message in (
        ({"prompt_id": "another"}, "today's prompt wording are not the same"),
        ({"manifest_hash": "another"}, "today's split manifest are not the same"),
    ):
        report = report_for(context, cfg, [good], [baseline], **change)
        assert message in detail(report["checks"], "V2")


def test_v2_fails_for_a_run_on_the_wrong_arm_or_of_an_unknown_model(context, cfg):
    deploy_models(cfg, context, models=("nova-lite",))
    baseline = scripted_baseline(context, cfg, "nova-lite", "dev", right_for("dev", 2))
    wrong_arm = tuned_run(context, cfg, "nova-lite", "dev", right_for("dev", 7), arm="B0-Lite")
    report = report_for(context, cfg, [wrong_arm], [baseline])
    assert "tuned run is arm B0-Lite, expected B1-Lite" in detail(report["checks"], "V2")

    swapped = report_for(context, cfg, [baseline], [wrong_arm])  # a baseline given as tuned
    assert not swapped["passed"]


def test_v3_fails_when_a_run_asked_another_deployment_than_the_latest_attempts(context, cfg):
    record = deploy_models(cfg, context, models=("nova-lite",))
    tuned, baselines = runs(context, cfg, models=("nova-lite",))
    assert report_for(context, cfg, tuned, baselines)["passed"]

    # A second attempt is launched afterwards: the runs are now of an older model.
    export_id = record.latest("nova-lite").export_id
    record.put(deployed_attempt(cfg, "nova-lite", export_id, attempt=2))
    write_tuning_record(record, cfg)
    report = report_for(context, cfg, tuned, baselines)
    assert failed(report["checks"]) == ["V3"]
    assert "did not ask the deployment of nova-lite's latest attempt (2)" in detail(
        report["checks"], "V3"
    )


def test_v3_fails_without_a_completed_attempt_in_the_record(context, cfg, scored):
    tuned, baselines = scored
    record = read_tuning_record(cfg)
    record.latest("nova-pro").job_status = "Failed"
    report = report_for(context, cfg, tuned, baselines, record=record)
    assert failed(report["checks"]) == ["V3"]
    assert "the latest job of nova-pro is Failed, not Completed" in detail(report["checks"], "V3")

    record.attempts = [a for a in record.attempts if a.model != "nova-pro"]
    report = report_for(context, cfg, tuned, baselines, record=record)
    assert "the tuning record has no attempt for nova-pro" in detail(report["checks"], "V3")


def test_v4_fails_when_the_model_was_tuned_on_other_files(context, cfg, scored):
    tuned, baselines = scored
    gate = dict(read_gate_record(cfg), export_id="0123456789ab")
    report = report_for(context, cfg, tuned, baselines, gate=gate)
    assert failed(report["checks"]) == ["V4"]
    assert "the gate record names export 0123456789ab" in detail(report["checks"], "V4")

    report = report_for(context, cfg, tuned, baselines, gate=None)
    assert failed(report["checks"]) == ["V4"]
    assert detail(report["checks"], "V4") == (
        "there is no gate record; run python -m scripts.nova_export"
    )


def test_v5_fails_when_a_tuned_run_is_not_recorded_under_the_base_model(context, cfg):
    deploy_models(cfg, context, models=("nova-lite",))
    baseline = scripted_baseline(context, cfg, "nova-lite", "dev", right_for("dev", 2))
    predictor = tuned_predictor(cfg, "nova-lite", StubProvider())
    predictor.usage_model_id = None  # the ledger then holds the deployment's ARN
    tuned = tuned_run(context, cfg, "nova-lite", "dev", [], predictor=predictor)
    report = report_for(context, cfg, [tuned], [baseline])
    assert failed(report["checks"]) == ["V5"]
    assert f"run {tuned['run_id']} is not recorded under amazon.nova-lite-v1:0" in detail(
        report["checks"], "V5"
    )

    report = report_for(context, cfg, [tuned], [baseline], ledger_records=[])
    assert f"run {baseline['run_id']} has no ledger record" in detail(report["checks"], "V5")


def test_v6_fails_for_a_model_scored_on_quick_dev_only(context, cfg):
    deploy_models(cfg, context)
    tuned, baselines = runs(context, cfg, splits=("quick_dev",))
    report = report_for(context, cfg, tuned, baselines)
    assert failed(report["checks"]) == ["V6"]
    assert "nova-pro has no tuned and baseline pair on dev" in detail(report["checks"], "V6")


def test_the_model_of_a_run_is_the_configured_name_for_both_arms(context, cfg, scored):
    tuned, baselines = scored
    assert {model_of(run) for run in tuned} == {model_of(run) for run in baselines} == set(MODELS)


# --------------------------------------------------------------------------- the files


def test_the_report_holds_counts_and_ids_and_no_example_text(context, cfg, scored, tmp_path):
    tuned, baselines = scored
    report = report_for(context, cfg, tuned, baselines)
    markdown_path, json_path = write_report(report, cfg.report_dir)
    assert (markdown_path, json_path) == report_paths(cfg.report_dir)
    assert markdown_path.name == "nova_tuned_report.md" and json_path.name.endswith(".json")
    markdown, text = (
        markdown_path.read_text(encoding="utf-8"),
        json_path.read_text(encoding="utf-8"),
    )
    assert json.loads(text) == json.loads(render_json(report)) == report
    assert markdown == render_markdown(report) and markdown.endswith("\n")
    assert markdown.startswith("# Nova tuned-model report\n")
    for heading in (
        "## Verdict", "## Inputs", "## Checks", "## Scores", "## Question by question",
        "## Cost of asking", "## Cost of tuning", "## Replies of the tuned models",
        "## The tuned models", "## Runs",
    ):  # fmt: skip
        assert heading in markdown
    assert markdown.count("**MET**") == 2 and "6 of 6 passed." in markdown
    assert "| nova-lite | dev | difference | | +0.00 | +62.50 |" in markdown
    records = [record for split in SPLITS for record in records_for(split, context)]
    for content in (markdown, text):
        assert content.isascii() and str(tmp_path) not in content
        assert GOLD_REPLY not in content and "subtract" not in content
        assert not any(record.user in content or record.system in content for record in records)
    # Example ids are allowed: they are what v4.05 would look up.
    assert report["models"][0]["pairs"][1]["paired"]["fixed_ids"][0] in text
