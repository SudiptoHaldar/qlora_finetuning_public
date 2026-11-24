"""The cost model: categories, prices, per-arm figures, scenarios, break-evens, checks.

Invented ledger lines and prices only (``cost_helpers``): no network, no credentials, no
GPU, no file of the repository's own ledger.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from qlora_finetuning.comparison import cost, cost_bill, cost_report, cost_scenarios
from qlora_finetuning.comparison.cost import (
    BILLED,
    DEVELOPMENT,
    ESTIMATE,
    NO_CHARGE,
    SELECT,
    TEST,
    THROUGH_JOB,
    TRAIN,
    build_record,
    model_total,
    place_all,
    run_date,
    storage_lines,
    storage_so_far,
)
from qlora_finetuning.comparison.cost_bill import add_bill, read_bills
from qlora_finetuning.comparison.cost_config import load_cost_config
from qlora_finetuning.comparison.cost_report import render_markdown, usd, write_record
from qlora_finetuning.comparison.cost_scenarios import ALWAYS_ON, BATCH, MAKING, crossing
from qlora_finetuning.comparison.gate import canonical_json
from qlora_finetuning.frontier.pricing import inference_cost
from qlora_finetuning.settings import ConfigError
from tests.comparison import cost_helpers as h

D = Decimal


@pytest.fixture
def cfg(tmp_path):
    return load_cost_config(root=tmp_path)


def by_run(placed):
    return {item.record.run_id: item for item in placed}


def check(record, check_id):
    return next(c for c in record["checks"] if c["id"] == check_id)


def break_even(record, arm_a, arm_b, scenario):
    return next(
        e
        for e in record["break_evens"]
        if e["pair"] == [arm_a, arm_b] and e["scenario"] == scenario
    )


# --------------------------------------------------------------------------- categories


def test_each_record_lands_in_its_category():
    placed, _ = place_all(h.inputs())
    found = by_run(placed)
    assert found[f"{h.DAY}-other-job-train"].category == TRAIN
    assert found[f"{h.DAY}-other-job-score"].category == SELECT
    assert found[f"{h.DAY}-other-job-smoke"].category == DEVELOPMENT
    assert found[f"{h.DAY}-other-job-a1"].category == TEST
    assert found[h.TEST_RUNS["B1-Pro"]].category == TEST
    assert found[f"{h.DAY}-eval-nova-lite-dev-0001"].category == DEVELOPMENT
    assert found[f"{h.DAY}-train-nova-lite-0001"].category == TRAIN
    assert found[f"{h.DAY}-eval-qwen-quick-dev-0001"].category == NO_CHARGE
    assert Counter(item.category for item in placed) == {
        TRAIN: 4,  # the training job, its run, and the two customization jobs
        SELECT: 2,  # the scoring job and its run
        TEST: 8,  # four Nova runs; two Qwen jobs and their runs
        DEVELOPMENT: 2,  # the smoke job and a Nova run on dev
        NO_CHARGE: 1,
    }
    assert not [item.problem for item in placed if item.problem]


def test_a_run_inside_a_job_is_priced_through_its_job():
    found = by_run(place_all(h.inputs())[0])
    inside = found[h.TEST_RUNS["A1"]]
    assert (inside.category, inside.amount, inside.basis) == (TEST, D(0), THROUGH_JOB)
    assert found[h.TRAIN_RUN].amount == 0
    assert found[f"{h.DAY}-other-job-a1"].amount == D("1.1")


def test_a_run_on_the_owners_machine_has_no_charge():
    local = by_run(place_all(h.inputs())[0])[f"{h.DAY}-eval-qwen-quick-dev-0001"]
    assert (local.category, local.amount) == (NO_CHARGE, D(0))


def test_which_job_trained_comes_from_the_selection_record_not_the_arm():
    # Without the selection record nothing says which job made A1: it is development.
    placed, _ = place_all(h.inputs(selection=None))
    found = by_run(placed)
    assert found[f"{h.DAY}-other-job-train"].category == DEVELOPMENT
    assert found[f"{h.DAY}-other-job-score"].category == DEVELOPMENT


def test_an_earlier_tuning_attempt_is_development():
    attempts = h.tuning()
    attempts.append(attempts[0] | {"attempt": 2, "job_arn": h.ARN.format("lite-2")})
    found = by_run(place_all(h.inputs(tuning=attempts))[0])
    assert found[f"{h.DAY}-train-nova-lite-0001"].category == DEVELOPMENT


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"where": "sagemaker", "external_id": "job-nobody-knows", "billable_seconds": 60,
          "instance_type": h.INSTANCE}, "is not in the job record"),
        ({"where": "sagemaker", "external_id": "job-smoke", "billable_seconds": 60,
          "instance_type": "ml.unpriced"}, "no training price"),
        ({"where": "bedrock", "model_id": "amazon.unknown-v1:0", "input_tokens": 5},
         "no price for model"),
        ({"where": "bedrock", "model_id": h.LITE}, "nothing to price"),
        ({"where": "the moon"}, "unknown place"),
    ],
)  # fmt: skip
def test_a_record_that_fits_no_category_stops_the_model(cfg, changes, message):
    stray = h.line(f"{h.DAY}-other-stray-0001", "none", changes.pop("where"), **changes)
    values = h.inputs()
    values.ledger.append(stray)
    record = build_record(values, cfg)
    assert not check(record, "K4")["passed"]
    assert message in check(record, "K4")["detail"]
    assert not record["passed"]


def test_a_replaced_record_is_not_counted_because_only_effective_records_arrive(cfg):
    # The script passes Ledger.effective(): the model never sees a replaced line. Adding the
    # replaced estimate beside the billed amount would count the job twice, and K5 says so.
    values = h.inputs(pro_billed=True)
    values.ledger.append(
        h.line(f"{h.DAY}-train-nova-pro-0001", "B1-Pro", "bedrock", phase="train",
               source="computed", model_id=h.PRO, training_tokens=2_000_000,
               external_id=h.ARN.format("pro"))
    )  # fmt: skip
    assert not check(build_record(values, cfg), "K5")["passed"]


# --------------------------------------------------------------------------- prices


def test_a_nova_run_costs_what_inference_cost_says():
    values = h.inputs()
    found = by_run(place_all(values)[0])
    for arm in ("B0-Lite", "B1-Lite", "B0-Pro", "B1-Pro"):
        model_id = h.LITE if "Lite" in arm else h.PRO
        expected = inference_cost(model_id, 1_000_000, h.OUTPUT[arm], values.pricing)
        assert found[h.TEST_RUNS[arm]].amount == expected
    assert found[h.TEST_RUNS["B1-Pro"]].amount == D("1.1")


def test_a_job_costs_its_seconds_times_the_hourly_rate():
    found = by_run(place_all(h.inputs())[0])
    assert found[f"{h.DAY}-other-job-train"].amount == D(10_000) * D("3.60") / 3600 == D(10)
    assert found[f"{h.DAY}-other-job-a0"].amount == D("2.2")


def test_a_billed_amount_is_used_as_it_stands_and_an_estimate_is_marked(cfg):
    found = by_run(place_all(h.inputs())[0])
    lite = found[f"{h.DAY}-train-nova-lite-0001"]
    pro = found[f"{h.DAY}-train-nova-pro-0001"]
    assert (lite.amount, lite.basis) == (D("5.00"), BILLED)
    assert (pro.amount, pro.basis) == (D(20), ESTIMATE)

    record = build_record(h.inputs(), cfg)
    assert record["making"]["B1-Lite"]["train"]["basis"] == BILLED
    assert record["making"]["B1-Pro"]["train"]["basis"] == ESTIMATE
    assert check(record, "K2")["reservation"]
    assert len(record["estimates"]) == 1 and "B1-Pro" in record["estimates"][0]


def test_money_is_decimal_and_never_a_float(cfg):
    placed, _ = place_all(h.inputs())
    assert all(type(item.amount) is Decimal for item in placed)
    text = canonical_json(build_record(h.inputs(), cfg))
    figures = json.loads(text)["test_pass"]["A1"]
    assert figures["usd"] == "1.100000"
    assert isinstance(figures["per_1000_questions"], str)


def test_the_day_of_a_run_is_read_from_its_id():
    assert run_date("20261123T181752Z-eval-nova-pro-tuned-test-79c0") == "2026-11-23"
    assert run_date("not-a-run-id") is None


# --------------------------------------------------------------------------- per arm


def test_what_it_cost_to_make_each_tuned_model(cfg):
    making = build_record(h.inputs(), cfg)["making"]
    assert making["A1"]["train"]["usd"] == "10.000000"
    assert making["A1"]["select"]["usd"] == "2.000000"
    assert making["A1"]["making_usd"] == "12.000000"
    assert making["A1"]["storage"] is None
    assert making["B1-Lite"]["select"] is None
    assert making["B1-Lite"]["storage"]["per_month"] == "2.000000"
    assert making["B1-Lite"]["storage"]["months"] == {"1": "2.000000", "12": "24.000000"}
    assert set(making) == {"A1", "B1-Lite", "B1-Pro"}


def test_as_run_and_steady_differ_by_the_start_up(cfg):
    figures = build_record(h.inputs(), cfg)["test_pass"]
    a1 = figures["A1"]
    assert a1["per_1000_questions"] == "1.100000"  # 1,100 billable seconds
    assert a1["per_1000_questions_steady"] == "1.000000"  # the run's own 1,000 seconds
    assert a1["startup_seconds"] == "100.000000"
    assert a1["seconds_per_question"] == "1.000000"
    assert figures["A0"]["per_1000_questions"] == "2.200000"
    assert "per_1000_questions_steady" not in figures["B1-Pro"]
    assert figures["B1-Pro"]["per_1000_questions"] == "1.100000"
    assert figures["B0-Lite"]["per_1000_questions"] == "0.120000"


def test_cost_per_1000_correct_answers_uses_the_six_arm_records_count(cfg):
    figures = build_record(h.inputs(), cfg)["test_pass"]
    assert D(figures["A1"]["per_1000_correct"]) == (D("1.1") / 700 * 1000).quantize(D("0.000001"))
    assert figures["B1-Lite"]["correct"] == 650


def test_an_arm_with_no_correct_answer_gets_no_figure_not_a_division_by_zero(cfg):
    values = h.inputs()
    values.six_arm["arms"]["A0"]["all"]["execution_accuracy"]["count"] = 0
    figures = build_record(values, cfg)["test_pass"]["A0"]
    assert figures["per_1000_correct"] is None
    assert figures["per_1000_correct_steady"] is None
    assert figures["per_1000_questions"] == "2.200000"


def test_storage_is_one_model_month_for_each_calendar_month_a_model_exists(cfg):
    storage = storage_lines(h.inputs(), cfg)
    assert storage["B1-Lite"]["since"] == "2026-11-02"
    assert storage_so_far(storage, "2026-11-01") == {}
    assert storage_so_far(storage, "2026-11-02") == {"B1-Lite": D(2), "B1-Pro": D(2)}
    assert storage_so_far(storage, "2026-11-30")["B1-Pro"] == D(2)
    assert storage_so_far(storage, "2026-12-01")["B1-Pro"] == D(4)
    assert storage_so_far(storage, "2027-01-15")["B1-Pro"] == D(6)


def test_a_deleted_model_has_no_storage_line(cfg):
    attempts = h.tuning()
    attempts[1]["model_deleted"] = True
    assert set(storage_lines(h.inputs(tuning=attempts), cfg)) == {"B1-Lite"}


def test_the_project_totals_per_path_and_category(cfg):
    paths = build_record(h.inputs(), cfg)["project"]["paths"]
    assert paths["A"] | {} == {
        TRAIN: "10.000000", SELECT: "2.000000", TEST: "3.300000", DEVELOPMENT: "0.300000",
        NO_CHARGE: "0.000000", "total": "15.600000", "cap": "100.000000",
    }  # fmt: skip
    assert paths["B"]["total"] == "27.540000"
    assert paths["B"][TRAIN] == "25.000000"


# --------------------------------------------------------------------------- scenarios


def test_the_always_on_cost_per_question_doubles_when_utilization_halves(cfg):
    always = build_record(h.inputs(), cfg)["serving"]["A1"]["always_on"]
    assert always["monthly_usd"] == "5256.000000"  # $7.20 an hour for 730 hours
    rows = {D(row["utilization"]): D(row["per_1000_questions"]) for row in always["at"]}
    assert rows[D("1")] == D("2")  # 3,600 questions an hour, 2,628,000 a month
    assert rows[D("0.5")] == rows[D("1")] * 2
    assert rows[D("0.01")] == rows[D("1")] * 100


def test_a_batch_jobs_cost_per_question_falls_toward_the_steady_figure(cfg):
    scenario = build_record(h.inputs(), cfg)["serving"]["A1"]
    assert scenario["startup_usd"] == "0.100000"
    rows = [D(row["per_1000_questions"]) for row in scenario["batch"]]
    assert rows == sorted(rows, reverse=True)
    assert all(row > 1 for row in rows)  # the steady figure is $1.00 per 1,000
    assert rows[-1] - 1 < D("0.01")
    first = scenario["batch"][0]
    assert (first["questions"], first["usd"]) == (100, "0.200000")  # 100 s start-up, 100 s work


def test_no_always_on_scenario_without_a_hosting_price(cfg):
    record = build_record(h.inputs(instances=h.instances(hosting=False)), cfg)
    assert record["serving"]["A1"]["always_on"] is None
    assert not check(record, "K3")["passed"]
    assert "no hosting price" in check(record, "K3")["detail"]
    assert not [e for e in record["break_evens"] if e["scenario"] == ALWAYS_ON]


# --------------------------------------------------------------------------- break-evens


def test_two_lines_that_cross_give_the_volume_at_which_both_cost_the_same():
    lines = (D("0.10"), D("0.001"), D("2"), D("0.00011"))
    found = crossing(*lines)
    assert found["kind"] == "volume"
    volume = found["volume"]
    assert lines[0] + volume * lines[1] == pytest.approx(lines[2] + volume * lines[3])
    assert (found["cheaper_below"], found["cheaper_above"]) == ("a", "b")


def test_two_lines_that_never_cross_name_the_one_that_is_always_cheaper():
    assert crossing(D(1), D("0.001"), D(2), D("0.002")) == {"kind": "always", "cheaper": "a"}
    assert crossing(D(2), D("0.002"), D(1), D("0.001")) == {"kind": "always", "cheaper": "b"}
    assert crossing(D(1), D("0.001"), D(2), D("0.001")) == {"kind": "always", "cheaper": "a"}
    assert crossing(D(1), D("0.001"), D(1), D("0.001")) == {"kind": "equal"}
    # Equal at volume zero: above it the smaller slope is cheaper.
    assert crossing(D(0), D("0.002"), D(0), D("0.001")) == {"kind": "always", "cheaper": "b"}


def test_tuned_arms_are_compared_with_tuned_and_untuned_with_untuned(cfg):
    pairs = {tuple(e["pair"]) for e in build_record(h.inputs(), cfg)["break_evens"]}
    assert pairs == {("A0", "B0-Lite"), ("A0", "B0-Pro"), ("A1", "B1-Lite"), ("A1", "B1-Pro")}


def test_the_batch_break_even_against_a_cheaper_per_question_model(cfg):
    entry = break_even(build_record(h.inputs(), cfg), "A1", "B1-Lite", BATCH)
    # A1: $0.10 a job and $0.001 a question. B1-Lite: $2.00 storage and $0.00011 a question.
    assert entry["questions"] == 2135
    assert (entry["cheaper_below"], entry["cheaper_above"]) == ("A1", "B1-Lite")
    assert entry["reachable"]


def test_a_pair_that_never_crosses_gives_no_volume(cfg):
    record = build_record(h.inputs(), cfg)
    entry = break_even(record, "A1", "B1-Pro", BATCH)
    assert (entry["kind"], entry["questions"], entry["cheaper_always"]) == ("always", None, "A1")
    assert break_even(record, "A0", "B0-Lite", BATCH)["cheaper_always"] == "B0-Lite"


def test_a_break_even_above_what_either_side_can_serve_is_not_reachable(cfg):
    record = build_record(h.inputs(), cfg)
    entry = break_even(record, "A1", "B1-Pro", ALWAYS_ON)
    # ($5,256 - $2 storage) / $0.0011 a question
    assert entry["questions"] == 4_776_364
    assert entry["cheaper_above"] == "A1"
    assert not entry["reachable"]
    assert "A1 can serve in a month (2,628,000)" in entry["limit_note"]
    assert "B1-Pro can serve in a month (876,000)" in entry["limit_note"]
    assert check(record, "K7")["passed"]
    assert all("reachable" in e and e["limit_note"] for e in record["break_evens"])


def test_the_break_even_with_the_cost_of_making_the_model(cfg):
    record = build_record(h.inputs(), cfg)
    # A1 costs $12 to make (training and selection) and $0.001 a question; B1-Lite $5 and
    # $0.00011: B1-Lite is cheaper to make and to run.
    assert break_even(record, "A1", "B1-Lite", MAKING)["cheaper_always"] == "B1-Lite"
    assert break_even(record, "A1", "B1-Pro", MAKING)["cheaper_always"] == "A1"
    assert not [
        e for e in record["break_evens"] if e["scenario"] == MAKING and e["pair"][0] == "A0"
    ]

    dearer = h.inputs()
    for index, item in enumerate(dearer.ledger):
        if item.run_id == f"{h.DAY}-train-nova-lite-0001":
            dearer.ledger[index] = replace(item, billed_amount_usd="30.00")
    entry = break_even(build_record(dearer, cfg), "A1", "B1-Lite", MAKING)
    assert entry["questions"] == 20_225  # (30 - 12) / (0.001 - 0.00011), rounded up
    assert (entry["cheaper_below"], entry["cheaper_above"]) == ("A1", "B1-Lite")
    assert entry["reachable"] and "whole use" in entry["limit_note"]


# --------------------------------------------------------------------------- the checks


def test_every_check_passes_and_a_reservation_makes_the_record_provisional(cfg):
    record = build_record(h.inputs(), cfg)
    assert [c["id"] for c in record["checks"]] == [f"K{n}" for n in range(1, 9)]
    assert all(c["passed"] for c in record["checks"])
    assert record["passed"] and record["provisional"]
    assert {c["id"] for c in record["checks"] if c["reservation"]} == {"K2", "K8"}


def test_with_every_amount_billed_and_entered_the_record_is_final(cfg):
    record = build_record(h.inputs(pro_billed=True, bills=h.covering_bills()), cfg)
    assert record["passed"] and not record["provisional"]
    assert not record["estimates"]
    assert not [c for c in record["checks"] if c["reservation"]]


def test_k1_needs_a_six_arm_record_that_passed(cfg):
    record = build_record(h.inputs(six_arm=None), cfg)
    assert not check(record, "K1")["passed"]
    assert "there is no six-arm record" in check(record, "K1")["detail"]
    failed = h.six_arm() | {"passed": False}
    assert not check(build_record(h.inputs(six_arm=failed), cfg), "K1")["passed"]


def test_k2_fails_when_the_ledger_and_the_six_arm_record_disagree(cfg):
    values = h.inputs()
    values.six_arm["arms"]["B1-Pro"]["output_tokens"] += 1
    assert "output_tokens" in check(build_record(values, cfg), "K2")["detail"]

    values = h.inputs()
    values.six_arm["arms"]["A1"]["jobs"][0]["billable_seconds"] += 1
    detail = check(build_record(values, cfg), "K2")
    assert not detail["passed"] and "job-a1" in detail["detail"]

    values = h.inputs()
    values.ledger = [r for r in values.ledger if r.run_id != h.TEST_RUNS["B0-Lite"]]
    assert "no ledger record" in check(build_record(values, cfg), "K2")["detail"]


def test_k5_fails_when_the_categories_do_not_sum_to_what_the_cap_checks_count(cfg):
    record = build_record(h.inputs(committed={"A": D("15.60"), "B": D("99")}), cfg)
    assert not check(record, "K5")["passed"]
    assert "Path B" in check(record, "K5")["detail"]
    assert "not computed" in check(build_record(h.inputs(committed={}), cfg), "K5")["detail"]


def test_k6_fails_when_the_arms_did_not_answer_the_same_questions(cfg):
    values = h.inputs()
    values.six_arm["arms"]["A0"]["all"]["examples"] = 999
    assert "999 examples" in check(build_record(values, cfg), "K6")["detail"]
    values = h.inputs()
    values.six_arm["arms"]["B0-Pro"]["input_tokens"] = 7
    values.ledger = [
        replace(r, input_tokens=7) if r.run_id == h.TEST_RUNS["B0-Pro"] else r
        for r in values.ledger
    ]
    assert "unequal input tokens" in check(build_record(values, cfg), "K6")["detail"]


def test_k8_compares_each_entered_amount_with_the_model_for_the_same_days(cfg):
    bills = [{"service": "bedrock", "usd": "27.54", "through": "2026-11-01"}]
    record = build_record(h.inputs(bills=bills), cfg)
    k8 = check(record, "K8")
    assert k8["passed"] and k8["reservation"]  # SageMaker has no entry yet
    assert "sagemaker" in k8["detail"] and "bedrock covers" not in k8["detail"]
    service = next(s for s in record["reconciliation"] if s["service"] == "bedrock")
    assert service["entries"][0]["difference_usd"] == "0.000000"
    assert service["entries"][0]["agrees"] and service["covered"]


def test_k8_fails_when_the_bill_and_the_model_are_apart(cfg):
    bills = [{"service": "sagemaker", "usd": "17.00", "through": "2026-11-01"}]
    k8 = check(build_record(h.inputs(bills=bills), cfg), "K8")
    assert not k8["passed"] and "billed $17.00, the model $15.60" in k8["detail"]
    near = [{"service": "sagemaker", "usd": "16.00", "through": "2026-11-01"}]
    assert check(build_record(h.inputs(bills=near), cfg), "K8")["passed"]  # within $0.50


def test_a_bill_that_stops_before_the_last_run_does_not_cover_it(cfg):
    bills = [{"service": "sagemaker", "usd": "0", "through": "2026-10-31"}]
    record = build_record(h.inputs(bills=bills), cfg)
    k8 = check(record, "K8")
    assert k8["passed"] and k8["reservation"]
    assert "no bill amount for sagemaker covers 2026-11-01" in k8["detail"]


def test_the_models_total_for_bedrock_includes_storage(cfg):
    values = h.inputs()
    placed, _ = place_all(values)
    storage = storage_lines(values, cfg)
    assert model_total(placed, "bedrock", "2026-11-01", storage) == D("27.54")
    assert model_total(placed, "bedrock", "2026-11-02", storage) == D("31.54")
    assert model_total(placed, "sagemaker", "2026-11-02", storage) == D("15.60")
    assert model_total(placed, "sagemaker", "2026-10-31", storage) == 0


# --------------------------------------------------------------------------- the record


def test_the_same_state_gives_the_same_bytes(cfg):
    assert canonical_json(build_record(h.inputs(), cfg)) == canonical_json(
        build_record(h.inputs(), cfg)
    )


def test_the_record_is_written_as_json_and_markdown_that_agree(cfg, tmp_path):
    record = build_record(h.inputs(pro_billed=True, bills=h.covering_bills()), cfg)
    markdown_path, json_path = write_record(record, cfg, tmp_path / "reports")
    assert (markdown_path.name, json_path.name) == ("cost_model.md", "cost_model.json")
    stored = json.loads(json_path.read_text(encoding="utf-8"))
    text = markdown_path.read_text(encoding="utf-8")
    assert text.startswith("# The cost model\n")
    for arm, figure in stored["test_pass"].items():
        assert f"| {arm} |" in text and usd(figure["usd"]) in text
    assert usd(stored["making"]["A1"]["train"]["usd"]) == "$10.00"
    assert "4,776,364 questions" in text and "**no**" in text
    assert "No priced figure is an estimate" in text
    assert "Left out of the model: S3 storage and requests" in text
    assert "\r" not in text and "\r" not in json_path.read_text(encoding="utf-8")


def test_a_provisional_record_says_so_in_its_title(cfg):
    text = render_markdown(build_record(h.inputs(), cfg))
    assert text.startswith("# The cost model (provisional)\n")
    assert "PASS, with a reservation" in text
    assert "- Estimate: B1-Pro train" in text
    assert "| sagemaker | 2026-11-01 | not entered | $15.60 |" in text


def test_a_record_that_failed_a_check_is_not_written(cfg, tmp_path):
    record = build_record(h.inputs(six_arm=None), cfg)
    with pytest.raises(ConfigError, match="K1 to K8"):
        write_record(record, cfg, tmp_path)
    assert not list(tmp_path.iterdir())


def test_amounts_are_shown_with_two_decimals_from_a_dollar_and_three_below():
    assert usd("1.628587") == "$1.63"
    assert usd("0.047202") == "$0.047"
    assert usd("5256") == "$5,256.00"
    assert usd("0") == "$0.00"
    assert usd(None) == "—"


def test_the_record_holds_no_name_address_or_example_text(cfg):
    text = canonical_json(build_record(h.inputs(), cfg)) + render_markdown(
        build_record(h.inputs(), cfg)
    )
    assert "@" not in text
    assert "question_text" not in text and "gold" not in text


# --------------------------------------------------------------------------- the bill file


def test_a_bill_entry_is_appended_and_earlier_entries_stay(cfg):
    today = date(2026, 11, 24)
    first = add_bill(cfg, "bedrock", "78.19", "2026-11-22", today)
    before = cfg.bill_file.read_text(encoding="utf-8")
    add_bill(cfg, "sagemaker", "8.24", "2026-11-22", today)
    entries = read_bills(cfg.bill_file)
    assert entries[0] == first == {
        "service": "bedrock", "usd": "78.19", "through": "2026-11-22", "entered_on": "2026-11-24",
    }  # fmt: skip
    assert [entry["service"] for entry in entries] == ["bedrock", "sagemaker"]
    assert json.loads(before)["entries"] == entries[:1]
    assert "\r" not in cfg.bill_file.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("service", "amount", "through", "message"),
    [
        ("ec2", "1.00", "2026-11-22", "unknown service"),
        ("bedrock", "-1", "2026-11-22", "zero or more"),
        ("bedrock", "a lot", "2026-11-22", "not an amount"),
        ("bedrock", "1.00", "2026-11-25", "in the future"),
        ("bedrock", "1.00", "yesterday", "must be a date"),
    ],
)
def test_a_bill_entry_that_makes_no_sense_is_refused(cfg, service, amount, through, message):
    with pytest.raises(ConfigError, match=message):
        add_bill(cfg, service, amount, through, date(2026, 11, 24))
    assert not cfg.bill_file.exists()


def test_no_bill_file_means_no_entries_and_a_broken_one_is_an_error(cfg):
    assert read_bills(cfg.bill_file) == []
    cfg.bill_file.parent.mkdir(parents=True)
    cfg.bill_file.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot be read"):
        read_bills(cfg.bill_file)


# --------------------------------------------------------------------------- guards


@pytest.mark.parametrize("module", [cost, cost_bill, cost_report, cost_scenarios])
def test_the_cost_modules_call_no_aws_service_and_read_no_split(module):
    source = Path(module.__file__).read_text(encoding="utf-8")
    code = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert not [line for line in code if "boto" in line or "transformers" in line]
    assert "load_split" not in source and "split_loader" not in source
