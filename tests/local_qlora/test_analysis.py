"""Tests for qlora_finetuning.local_qlora.analysis: where a trained adapter goes wrong (v3.05).

The stand-in world's dev examples all have the gold program ``subtract(920, 95)``, so a
crafted wrong reply lands in a known category.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qlora_finetuning.local_qlora.adapters import ADAPTER_WEIGHTS
from qlora_finetuning.local_qlora.analysis import (
    FALLING,
    FLAT,
    NONE,
    RISING,
    _trend,
    analyze_run,
    read_finding,
    render_analysis,
)
from qlora_finetuning.settings import ConfigError

from .selection_helpers import (
    NOT_A_PROGRAM,
    REVERSED,
    TOO_LONG,
    World,
    scored_world,
)
from .training_helpers import holds_example_text

RIGHT = {2: 1, 4: 2, 6: 3}  # of 8: rising to the end, and 37.5% at best: a missed verdict
WRONG = {4: TOO_LONG, 6: REVERSED}  # step 2 answers add(1, 2): another operation
CHANGES = (("T1", "epochs", 5.0), ("T2", "optimizer.learning_rate", 1e-4), ("T3", "lora.r", 32.0))


@pytest.fixture
def world(tmp_path):
    world = scored_world(tmp_path, right=RIGHT, wrong=WRONG)
    world.select()
    return world


def analyse(world, **options):
    return analyze_run(world.run, **world.stand_ins(), **options)


# --------------------------------------------------------------------------- the report


def test_the_selected_checkpoints_failures_by_category(world):
    outcome = analyse(world)
    analysis = outcome["analysis"]
    assert outcome["passed"] and analysis["training_run"] == world.run
    selected = analysis["selected"]
    assert (selected["step"], selected["adapter"]) == (6, f"{world.run}/checkpoint-6")
    counts = selected["counts"]
    assert (counts["examples"], counts["right"]) == (8, 3)
    assert counts["categories"]["order"] == 5 and sum(counts["categories"].values()) == 5
    assert counts["value_labels"] == {"sign": 5}  # the value's label; it did not decide
    assert selected["format"]["in_the_format"] == 8
    assert not selected["execution_threshold_met"] and selected["format_threshold_met"]
    assert list(analysis["wrong_example_ids"]) == ["order"]  # only categories that occur
    assert len(set(analysis["wrong_example_ids"]["order"])) == 5


def test_beside_the_untuned_model_and_over_the_checkpoints(world):
    analysis = analyse(world)["analysis"]
    untuned = analysis["untuned"]
    assert untuned["counts"]["categories"]["no_program"] == 8 and untuned["counts"]["right"] == 0
    assert (untuned["fixed"], untuned["broken"]) == (3, 0)
    assert (untuned["right_in_both"], untuned["wrong_in_both"]) == (0, 5)

    points = analysis["checkpoints"]
    assert [(point["step"], point["right"]) for point in points] == [(2, 1), (4, 2), (6, 3)]
    assert points[0]["categories"]["operation"] == 7
    assert points[1]["categories"]["steps_more"] == 6
    assert points[2]["categories"]["order"] == 5
    assert [point["dev_loss"] for point in points] == [0.5, 0.25, pytest.approx(1 / 6)]
    assert analysis["never_right"]["count"] == 5 == len(analysis["never_right"]["example_ids"])


def test_where_the_failures_sit(world):
    where = analyse(world)["analysis"]["where"]
    assert where["by_gold_steps"] == [{"group": "1", "examples": 8, "right": 3}]
    assert where["by_gold_operation"] == [{"group": "subtract", "examples": 8, "right": 3}]
    companies = where["by_company"]
    assert sum(group["examples"] for group in companies) == 8 and len(companies) >= 2
    assert sum(group["right"] for group in companies) == 3
    quarters = where["by_prompt_length"]
    assert [group["examples"] for group in quarters] == [2, 2, 2, 2]
    assert all(group["tokens_from"] <= group["tokens_to"] for group in quarters)


def test_the_finding_is_read_from_the_curve(world):
    analysis = analyse(world)["analysis"]
    assert analysis["trend"]["last_three"] == RISING and analysis["trend"]["best_step"] == 6
    assert analysis["finding"]["row"] == "T1"
    assert (analysis["finding"]["variable"], analysis["finding"]["value"]) == ("epochs", 5.0)


# --------------------------------------------------------------------------- the files


def test_the_report_is_written_for_the_selected_run_and_holds_no_reply(world):
    outcome = analyse(world)
    path = world.training.training_cfg.analysis_record
    assert outcome["written"] == [str(path), str(path.with_suffix(".md"))]
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == json.loads(json.dumps(outcome["analysis"]))
    page = path.with_suffix(".md").read_text(encoding="utf-8")
    assert page == render_analysis(stored)
    assert "| Arguments reversed (`order`) | 5 | 62.5% | 0 |" in page
    assert "| 6 **(selected)** | 3 |" in page and "**T1**" in page
    assert "It decides nothing" in page

    for text in (path.read_text(encoding="utf-8"), page):
        assert "PROGRAM" not in text and "ANSWER" not in text and "subtract(" not in text
        assert "\r\n" not in text
    assert holds_example_text(path.parent) == []


def test_the_same_runs_give_the_same_files(world):
    files = [Path(name) for name in analyse(world)["written"]]
    before = [path.read_bytes() for path in files]
    analyse(world)
    assert [path.read_bytes() for path in files] == before


def test_check_writes_nothing(world):
    outcome = analyse(world, check=True)
    assert outcome["analysis"]["selected"]["step"] == 6 and outcome["written"] == []
    assert not world.training.training_cfg.analysis_record.exists()


def test_without_a_selection_only_a_look_is_allowed(tmp_path):
    world = scored_world(tmp_path, right=RIGHT)
    with pytest.raises(ConfigError, match="no checkpoint has been selected yet"):
        analyse(world)
    outcome = analyse(world, check=True)
    assert outcome["passed"] and outcome["analysis"]["selected"]["step"] == 6


def test_a_run_whose_checks_fail_is_shown_and_not_written(world):
    (world.checkpoint(2) / ADAPTER_WEIGHTS).write_bytes(b"other weights")  # L3 fails now
    outcome = analyse(world)
    assert not outcome["passed"] and outcome["analysis"] is not None
    assert outcome["written"] == []


def test_nothing_scored_gives_no_analysis(tmp_path):
    world = World(tmp_path)
    outcome = analyse(world, check=True)
    assert outcome["analysis"] is None and not outcome["passed"]
    assert [item["id"] for item in outcome["checks"]] == ["L1"]


def test_a_rehearsal_needs_check(tmp_path):
    world = World(tmp_path, right=RIGHT)
    world.record_jobs(world.score(limit=3))
    with pytest.raises(ConfigError, match="it is a rehearsal: add --check"):
        analyse(world, limit=3)
    outcome = analyse(world, limit=3, check=True)
    assert outcome["rehearsal"] and outcome["analysis"]["examples"] == 3


def test_the_first_runs_analysis_is_kept_when_another_run_is_selected(world):
    analyse(world)
    path = world.training.training_cfg.analysis_record
    first = json.loads(path.read_text(encoding="utf-8"))
    later = dict(first, training_run="20261121T000000Z-train-qwen3-4b-fit-bbbb")
    from qlora_finetuning.local_qlora.analysis import _write

    _write(later, path)
    kept = path.with_name("error_analysis_first.json")
    assert json.loads(kept.read_text(encoding="utf-8")) == first
    assert kept.with_suffix(".md").is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["training_run"] == later["training_run"]
    _write(dict(later, training_run="a-third"), path)  # the first stays the first
    assert json.loads(kept.read_text(encoding="utf-8")) == first


# --------------------------------------------------------------------------- the trend


def rows(*rights):
    return [
        {"step": 100 * (index + 1), "execution_accuracy": right}
        for index, right in enumerate(rights)
    ]


@pytest.mark.parametrize(
    ("rights", "expected"),
    [
        ((100, 105, 111), RISING),  # 11 more than three checkpoints back: more than noise
        ((100, 105, 110), FLAT),  # exactly the noise: not told apart
        ((120, 115, 109), FALLING),
        ((120, 150, 115), FLAT),  # what lies between does not count
        ((100, 200), FLAT),  # fewer than three checkpoints say nothing
    ],
)
def test_the_last_three_checkpoints_rise_fall_or_stay_within_noise(rights, expected):
    assert _trend(rows(*rights), 100, 10)["last_three"] == expected


def test_the_trend_says_where_the_best_is_and_when_it_was_first_reached():
    trend = _trend(rows(100, 180, 195, 200, 190, 185), 400, 10)
    assert trend["best_step"] == 400 and trend["first_step_within_noise_of_best"] == 300
    assert not trend["best_is_among_the_last_two"] and not trend["best_is_in_the_first_half"]
    assert trend["drop_from_best_to_last"] == 15 and trend["range_from_the_middle_on"] == 15
    early = _trend(rows(100, 200, 150, 140, 130, 120), 200, 10)
    assert early["best_is_in_the_first_half"] and early["drop_from_best_to_last"] == 80
    assert _trend(rows(1, 2, 3), 300, 10)["best_is_among_the_last_two"]


# --------------------------------------------------------------------------- the finding


def counts(right=150, **categories):
    table = dict.fromkeys(
        ("cut_off", "no_program", "bad_program", "not_executable", "kind", "steps_more",
         "steps_fewer", "operation", "evidence", "order", "scale", "sign", "other"), 0
    )  # fmt: skip
    table.update(categories)
    return {"examples": 499, "right": right, "categories": table}


def finding(rights, selected, losses=None, met=(False, True), **categories):
    trend = _trend(rows(*rights), selected, 10)
    losses = losses or {}
    last = 100 * len(rights)
    wrong = counts(max(rights), **(categories or {"order": 499 - max(rights)}))
    return read_finding(trend, wrong, losses, last, met[0], met[1], CHANGES)


def test_t1_the_best_is_at_the_end_and_still_rising():
    found = finding((100, 120, 140, 160, 180), 500)
    assert (found["row"], found["variable"], found["value"]) == ("T1", "epochs", 5.0)
    assert finding((100, 120, 140, 180, 175), 400)["row"] == "T1"  # the one before the last
    assert finding((100, 170, 175, 178, 180), 500)["row"] == NONE  # at the end, and flat


def test_t2_an_early_peak_a_fall_and_a_rising_loss():
    rights, losses = (150, 190, 170, 160, 150, 150), {200: 0.30, 600: 0.45}
    found = finding(rights, 200, losses)
    assert (found["row"], found["variable"], found["value"]) == (
        "T2",
        "optimizer.learning_rate",
        1e-4,
    )
    assert finding(rights, 200, {200: 0.30, 600: 0.25})["row"] == NONE  # the loss kept falling
    assert finding(rights, 200)["row"] == NONE  # no loss recorded: not shown
    assert finding((150, 190, 185, 184, 182, 181), 200, losses)["row"] == NONE  # within noise


def test_t3_flat_from_the_middle_on_and_mostly_calculation():
    flat = (100, 150, 160, 165, 168, 166)
    found = finding(flat, 500, steps_more=200, evidence=100, order=31)
    assert (found["row"], found["variable"], found["value"]) == ("T3", "lora.r", 32.0)
    assert finding(flat, 500, steps_more=100, order=231)["row"] == NONE  # mostly reversed


def test_t4_a_format_miss_is_read_as_the_curve_says():
    rising = finding((100, 120, 140, 160, 180), 500, met=(True, False))
    assert (rising["row"], rising["reads_as"], rising["variable"]) == ("T4", "T1", "epochs")
    falling = finding((150, 190, 170, 160, 150, 150), 200, {200: 0.3, 600: 0.4}, met=(True, False))
    assert (falling["row"], falling["reads_as"]) == ("T4", "T2")
    assert finding((100, 170, 175, 178, 180), 500, met=(True, False))["row"] == NONE


def test_t5_failures_that_training_should_have_removed_point_at_a_fault():
    found = finding((100, 120, 140, 160, 180), 500, cut_off=200, bad_program=100, order=19)
    assert found["row"] == "T5" and found["variable"] is None  # before any other row
    assert "fault" in found["why"]
    assert finding((100, 120, 140, 160, 180), 500, cut_off=150, order=169)["row"] == "T1"


def test_the_table_values_come_from_the_config(world):
    assert world.training.training_cfg.iteration_changes == CHANGES
    assert world.training.training_cfg.noise_examples == 1  # the stand-in world's; 10 is committed


def test_replies_that_are_not_programs_are_counted_with_their_error(tmp_path):
    world = scored_world(tmp_path, right=RIGHT, wrong={6: NOT_A_PROGRAM})
    analysis = analyse(world, check=True)["analysis"]
    counts_ = analysis["selected"]["counts"]
    assert counts_["categories"]["bad_program"] == 5 and sum(counts_["error_codes"].values()) == 5
    assert analysis["finding"]["row"] == "T5"  # most failures are not programs: look for a fault
