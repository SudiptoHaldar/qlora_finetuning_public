"""Tests for the statistics record and checks R1 to R7 (req_005 v5.03).

The inputs are the invented ones of ``tests/comparison/report_helpers.py``.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from fractions import Fraction

import pytest

from qlora_finetuning.comparison.gate import canonical_json
from qlora_finetuning.comparison.report_record import (
    add_check,
    build_statistics,
    make_check,
    p_text,
)
from qlora_finetuning.comparison.statistics import holm, mcnemar_exact
from tests.comparison import report_helpers as r


@pytest.fixture
def cfg(tmp_path):
    return r.report_cfg(tmp_path)


@pytest.fixture
def record(cfg):
    return build_statistics(r.inputs(), cfg)


# --------------------------------------------------------------------------- the record


def test_the_invented_inputs_pass_r1_to_r7_with_a_reservation(record):
    assert r.failed(record) == []
    assert [item["id"] for item in record["checks"]] == [f"R{n}" for n in range(1, 8)]
    assert record["passed"] and record["provisional"] and record["cost_provisional"]
    first = r.check(record, "R1")
    assert first["reservation"] and "RESERVATION: the cost record is provisional" in first["detail"]


def test_a_reconciled_cost_record_gives_a_record_that_is_not_provisional(cfg):
    record = build_statistics(r.inputs(cost=r.cost_record(final=True)), cfg)
    assert record["passed"] and not record["provisional"]
    assert not r.check(record, "R1")["reservation"]


def test_each_arm_has_counts_shares_and_both_intervals(record):
    arm = record["arms"]["A1"]
    metric = arm["metrics"]["execution_accuracy"]
    assert metric["count"] == 700 and metric["percent"] == "70.00"
    low, high = Decimal(metric["interval"]["low"]), Decimal(metric["interval"]["high"])
    assert low < 70 < high and high - low < 10
    by_company = arm["execution_accuracy_by_company"]
    assert Decimal(by_company["low"]) < 70 < Decimal(by_company["high"])
    assert arm["label"] == "Qwen3-4B with QLoRA" and arm["tuned"] and arm["path"] == "A"
    assert arm["latency_seconds"] == {"mean": "0.500", "median": "0.400", "p95": "0.912"}
    assert arm["dev"] == {"examples": 100, "execution_accuracy": 70, "percent": "70.00"}
    assert record["companies"] == 20 and record["test_examples"] == 1000


def test_right_or_wrong_together_makes_the_interval_by_company_the_wider(record):
    arm = record["arms"]["A1"]  # right on whole companies, wrong on whole companies
    narrow = arm["metrics"]["execution_accuracy"]["interval"]
    wide = arm["execution_accuracy_by_company"]
    assert Decimal(wide["high"]) - Decimal(wide["low"]) > Decimal(narrow["high"]) - Decimal(
        narrow["low"]
    )


def test_categories_steps_and_the_format(record):
    arm = record["arms"]["A1"]
    assert arm["right"] == 700 and sum(arm["categories"].values()) == 300
    assert arm["categories"]["evidence"] == 100 and arm["categories"]["cut_off"] == 0
    assert arm["not_in_the_format"] == 0 and arm["stop_reasons"] == {"end": 1000}
    assert record["arms"]["A0"]["categories"]["bad_program"] == 990
    assert record["arms"]["A0"]["not_in_the_format"] == 495
    steps = arm["by_steps"]
    assert [row["steps"] for row in steps] == ["1", "2", "3 or more"]
    assert sum(row["examples"] for row in steps) == 1000
    assert sum(row["right"] for row in steps) == 700
    assert steps[0]["percent"] == "70.06"  # 234 of 334


def test_only_a_few_example_ids_per_category_and_all_are_wrong_ones(record, cfg):
    ids = record["arms"]["A1"]["example_ids"]
    assert set(ids) == {"steps_fewer", "evidence", "order"}
    assert all(len(found) == cfg.examples_per_category for found in ids.values())
    wrong = set(r.IDS[700:])
    assert all(example_id in wrong for found in ids.values() for example_id in found)


def test_a_tested_pair_has_four_counts_a_difference_and_two_p_values(record, cfg):
    pair = r.pair(record, "A1", "B1-Lite")
    assert (pair["both_right"], pair["only_first"], pair["only_second"], pair["both_wrong"]) == (
        600,
        100,
        50,
        250,
    )
    assert pair["difference_points"] == "5.00" and pair["favours"] == "A1"
    raw = [mcnemar_exact(p["only_first"], p["only_second"]) for p in record["pairs"]]
    assert pair["p_value"] == p_text(mcnemar_exact(100, 50))
    assert [p["p_adjusted"] for p in record["pairs"]] == [p_text(value) for value in holm(raw)]
    assert Decimal(pair["p_adjusted"]) >= Decimal(pair["p_value"])
    assert [p["pair"] for p in record["pairs"]] == [list(names) for names in cfg.pairs]


def test_a_large_gain_is_settled_and_favours_the_tuned_arm(record):
    gain = r.pair(record, "A1", "A0")
    assert gain["settled"] and gain["favours"] == "A1" and gain["difference_points"] == "69.00"
    behind = r.pair(record, "A1", "B1-Pro")
    assert behind["favours"] == "B1-Pro" and behind["difference_points"] == "-5.00"
    assert Decimal(behind["interval"]["high"]) < 0


def test_a_small_difference_is_not_settled(cfg):
    close = r.outcomes("B1-Pro", right=list(range(0, 675)) + list(range(700, 775)))
    record = build_statistics(r.inputs(arms=r.all_outcomes(**{"B1-Pro": close})), cfg)
    pair = r.pair(record, "A1", "B1-Pro")  # 25 against 75: 50 more for B1-Pro, in few companies
    assert (pair["only_first"], pair["only_second"]) == (25, 75)
    assert Decimal(pair["p_adjusted"]) < Decimal("0.05")
    assert Decimal(pair["interval_by_company"]["high"]) >= 0
    assert not pair["settled"]


def test_a_difference_in_one_company_is_not_settled_whatever_its_p_value(cfg):
    """Decision 2: forty questions of one company decide nothing about other companies."""
    lite = r.outcomes("B1-Lite", right=range(0, 660))  # all of A1's but the last 40: one company
    arms = r.all_outcomes(**{"B1-Lite": lite})
    record = build_statistics(r.inputs(arms=arms, six_arm=r.six_arm(arms)), cfg)
    pair = r.pair(record, "A1", "B1-Lite")
    assert (pair["only_first"], pair["only_second"]) == (40, 0)
    assert Decimal(pair["p_adjusted"]) < Decimal("0.0001")
    assert Decimal(pair["interval"]["low"]) > 0
    assert Decimal(pair["interval_by_company"]["low"]) == 0
    assert not pair["settled"] and pair["favours"] == "A1"


def test_equal_arms_favour_nobody(cfg):
    same = r.outcomes("B1-Lite", right=range(0, 700))
    arms = r.all_outcomes(**{"B1-Lite": same})
    record = build_statistics(r.inputs(arms=arms, six_arm=r.six_arm(arms)), cfg)
    pair = r.pair(record, "A1", "B1-Lite")
    assert pair["favours"] is None and pair["difference_points"] == "0.00"
    assert pair["p_value"] == "1.00e+0" and not pair["settled"]


def test_the_other_pairs_are_counts_only_and_the_overlap_is_counted(record):
    assert len(record["other_pairs"]) == 9
    assert all("p_value" not in pair for pair in record["other_pairs"])
    assert record["overlap"] == {
        "tuned_arms": ["A1", "B1-Lite", "B1-Pro"],
        "all_tuned_right": 600,
        "any_tuned_right": 750,
        "no_arm_right": 250,
    }
    assert record["unanswered_ids"] == list(r.IDS[750:])


def test_costs_are_the_cost_records_own_text(record):
    priced = r.cost_record()["test_pass"]["A1"]
    assert record["costs"]["A1"] == {
        "usd": priced["usd"],
        "per_1000_questions": priced["per_1000_questions"],
        "per_1000_correct": priced["per_1000_correct"],
    }


def test_the_same_inputs_give_the_same_bytes(cfg):
    assert canonical_json(build_statistics(r.inputs(), cfg)) == canonical_json(
        build_statistics(r.inputs(), cfg)
    )


def test_the_settings_are_in_the_record(record, cfg):
    assert record["settings"] == {
        "resamples": 300,
        "seed": cfg.seed,
        "company_seed": cfg.seed + 1,
        "confidence_percent": 95,
        "smallest_p": "0.0001",
        "pairs": [list(pair) for pair in cfg.pairs],
        "step_groups": [1, 2, 3],
    }


# --------------------------------------------------------------------------- R1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"six_arm": None}, "there is no six-arm record"),
        ({"cost": None}, "there is no cost record"),
        ({"six_arm": r.six_arm() | {"passed": False}}, "did not pass C1 to C8"),
        ({"cost": r.cost_record() | {"passed": False}}, "did not pass K1 to K8"),
        (
            {"cost": r.cost_record() | {"six_arm_freeze_sha256": "0" * 64}},
            "built under another freeze record",
        ),
    ],
)
def test_r1_fails_without_its_records(cfg, changes, message):
    record = build_statistics(r.inputs(**changes), cfg)
    assert "R1" in r.failed(record) and message in r.check(record, "R1")["detail"]
    assert not record["passed"] and "arms" not in record
    assert "not computed" in r.check(record, "R5")["detail"]


def test_r1_fails_for_a_pair_that_names_no_arm(cfg):
    other = dataclasses.replace(cfg, pairs=(("A1", "C9"),))
    record = build_statistics(r.inputs(), other)
    assert "config report.yaml tests C9: not an arm" in r.check(record, "R1")["detail"]


# --------------------------------------------------------------------------- R2 to R4


def test_r2_fails_for_a_run_that_could_not_be_read(cfg):
    arms = r.all_outcomes()
    del arms["B0-Pro"]
    problem = "arm B0-Pro: run x cannot be read here (no such run)"
    record = build_statistics(r.inputs(arms=arms, read_problems=[problem]), cfg)
    assert r.failed(record)[0] == "R2" and r.check(record, "R2")["detail"] == problem
    silent = build_statistics(r.inputs(arms=arms), cfg)
    assert "arm B0-Pro: its replies were not read" in r.check(silent, "R2")["detail"]


def test_r2_fails_for_a_short_run_and_for_other_examples(cfg):
    short = dataclasses.replace(r.outcomes("A1"), example_ids=r.IDS[:900])
    record = build_statistics(r.inputs(arms=r.all_outcomes(A1=short)), cfg)
    assert "arm A1: 900 of 1,000 replies are here" in r.check(record, "R2")["detail"]

    turned = dataclasses.replace(r.outcomes("A1"), example_ids=tuple(reversed(r.IDS)))
    record = build_statistics(r.inputs(arms=r.all_outcomes(A1=turned)), cfg)
    assert "arm A1: its examples are not those of A0" in r.check(record, "R2")["detail"]

    record = build_statistics(r.inputs(test_ids=tuple(reversed(r.IDS))), cfg)
    assert "its examples are not the test split's" in r.check(record, "R2")["detail"]


def test_r3_fails_when_a_rescored_count_is_not_the_recorded_one(cfg):
    six = r.six_arm()
    six["arms"]["A1"]["all"]["answer_accuracy"]["count"] = 699
    record = build_statistics(r.inputs(six_arm=six), cfg)
    assert r.failed(record)[0] == "R3"
    assert "arm A1: answer accuracy is 700 when rescored and 699" in r.check(record, "R3")["detail"]


def test_r4_fails_for_a_changed_shared_file_or_another_freeze(cfg):
    changed = r.SHARED | {"configs": r.SHARED["configs"] | {"prompt": "9" * 64}}
    record = build_statistics(r.inputs(shared_now=changed), cfg)
    assert "changed since the freeze: config/prompt.yaml" in r.check(record, "R4")["detail"]

    record = build_statistics(r.inputs(freeze_sha256="1" * 64), cfg)
    assert "not the one the six test runs were made under" in r.check(record, "R4")["detail"]

    record = build_statistics(r.inputs(freeze=None), cfg)
    assert "there is no freeze record" in r.check(record, "R4")["detail"]


# --------------------------------------------------------------------------- R5 to R7


def test_r5_and_r6_pass_with_what_they_checked(record):
    assert r.check(record, "R5")["detail"] == "every interval contains its figure"
    assert "sum to the examples" in r.check(record, "R6")["detail"]


def test_r7_fails_when_the_cost_record_is_about_other_runs(cfg):
    cost = r.cost_record()
    cost["test_pass"]["A1"] = cost["test_pass"]["A1"] | {"run_id": "another-run"}
    del cost["test_pass"]["B0-Pro"]
    record = build_statistics(r.inputs(cost=cost), cfg)
    detail = r.check(record, "R7")["detail"]
    assert "arm A1: the cost record priced another run" in detail
    assert "arm B0-Pro: the cost record has no test pass for it" in detail
    assert not record["passed"] and "B0-Pro" not in record["costs"]


def test_r7_fails_when_the_counts_differ(cfg):
    fewer = r.outcomes("B1-Pro", right=range(0, 740))
    six = r.six_arm(r.all_outcomes(**{"B1-Pro": fewer}))
    record = build_statistics(r.inputs(arms=r.all_outcomes(**{"B1-Pro": fewer}), six_arm=six), cfg)
    assert r.failed(record) == ["R7"]
    assert "arm B1-Pro: the cost record has other counts" in r.check(record, "R7")["detail"]


# --------------------------------------------------------------------------- small pieces


def test_p_text_is_scientific_with_three_digits():
    assert p_text(Fraction(357, 100000)) == "3.57e-3"
    assert p_text(Fraction(1)) == "1.00e+0"
    assert p_text(mcnemar_exact(2, 2096)).endswith("e-625")


def test_a_check_says_what_failed_or_what_is_reserved():
    assert make_check("R1", ["broken"], "fine") == {
        "id": "R1",
        "name": "Records in place",
        "passed": False,
        "reservation": False,
        "detail": "broken",
    }
    reserved = make_check("R1", [], "fine", ["later"])
    assert reserved["passed"] and reserved["reservation"]
    assert reserved["detail"] == "fine; RESERVATION: later"


def test_adding_a_check_works_out_passed_again(record):
    worse = add_check(record, make_check("R8", ["a name"], ""))
    assert not worse["passed"] and worse["checks"][-1]["id"] == "R8"
    better = add_check(worse, make_check("R8", [], "fine"))
    assert better["passed"] and [c["id"] for c in better["checks"]].count("R8") == 1
