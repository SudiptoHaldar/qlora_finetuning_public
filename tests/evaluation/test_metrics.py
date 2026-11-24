"""Tests for the comparison rule and the four metrics (req_002 v2.02).

Every program, table, and reply here is invented. Most tests use a config built by hand, with
a tight allowance for execution and a loose one for the answer line, so that they show which
allowance a metric uses whatever ``config/evaluation.yaml`` holds.
"""

from __future__ import annotations

import dataclasses
import math
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from qlora_finetuning.dsl import DslConfig, ReplyFormat, parse_program, run_reply
from qlora_finetuning.evaluation import metrics
from qlora_finetuning.evaluation.metrics import (
    METRICS,
    MISMATCHES,
    Allowance,
    EvaluationConfig,
    EvaluationError,
    ExampleScore,
    Gold,
    allowance_used,
    load_evaluation_config,
    mismatch_label,
    numbers_match,
    programs_match,
    read_answer,
    score_reply,
    summarize,
    values_match,
)
from qlora_finetuning.settings import CONFIG_DIR, REPO_ROOT, ConfigError

TIGHT = Allowance(absolute=0.00001, relative=0.00001)
LOOSE = Allowance(absolute=0.00001, relative=0.005)
CONFIG = EvaluationConfig(
    root=Path("."),
    execution=TIGHT,
    answer=LOOSE,
    scale_factors=(100.0, 1000.0, 1000000.0),
    flags_file=Path("flags.json"),
    report_dir=Path("reports"),
    results_file=Path("evaluations.jsonl"),
    diagnostic_relative=0.00001,
    progress_every=50,
)
DSL = DslConfig(max_reply_chars=4000, max_steps=32)
FORMAT = ReplyFormat(program_label="PROGRAM:", answer_label="ANSWER:")
TABLE = [["", "2024", "2023"], ["revenue", "$ 920", "$ 95"], ["margin", "12.5%", "10%"]]

GOLD_TARGET = "PROGRAM: subtract(920, 95), divide(#0, 95)\nANSWER: 8.68421"


def program(text):
    return parse_program(text, DSL)


def gold_for(target=GOLD_TARGET):
    return Gold.from_target(target, FORMAT, DSL)


def score(reply, target=GOLD_TARGET, config=CONFIG, example_id=None):
    outcome = run_reply(reply, TABLE, FORMAT, DSL)
    return score_reply(outcome, gold_for(target), config, example_id, DSL)


# --------------------------------------------------------------------------- comparing


def test_absolute_allowance_at_the_boundary_and_past_it():
    allowance = Allowance(absolute=0.5, relative=0.0)
    assert numbers_match(10.5, 10.0, allowance)
    assert numbers_match(9.5, 10.0, allowance)
    assert not numbers_match(10.5000001, 10.0, allowance)
    assert not numbers_match(9.4999999, 10.0, allowance)


def test_relative_allowance_at_the_boundary_and_past_it():
    allowance = Allowance(absolute=0.0, relative=0.25)
    assert numbers_match(10.0, 8.0, allowance)
    assert numbers_match(6.0, 8.0, allowance)
    assert not numbers_match(10.001, 8.0, allowance)
    assert not numbers_match(5.999, 8.0, allowance)


def test_the_larger_of_the_two_parts_applies():
    # Small result: the absolute part decides. Large result: the relative part does.
    assert numbers_match(0.000029, 0.00002, TIGHT)
    assert not numbers_match(0.000031, 0.00002, TIGHT)
    assert numbers_match(1000.009, 1000.0, TIGHT)
    assert not numbers_match(1000.02, 1000.0, TIGHT)
    assert not numbers_match(1000.02, 1000.0, Allowance(absolute=0.00001, relative=0.0))
    assert not numbers_match(0.000029, 0.00002, Allowance(absolute=0.0, relative=0.00001))


def test_a_gold_of_zero_negative_values_and_a_very_large_value():
    assert numbers_match(0.000005, 0.0, TIGHT)
    assert numbers_match(-0.000005, 0.0, TIGHT)
    assert not numbers_match(0.00002, 0.0, TIGHT)
    assert numbers_match(0.0, 0.0, Allowance(absolute=0.0, relative=0.01))
    assert not numbers_match(1e-12, 0.0, Allowance(absolute=0.0, relative=0.01))

    assert numbers_match(-1000.005, -1000.0, TIGHT)
    assert not numbers_match(1000.0, -1000.0, TIGHT)

    assert numbers_match(1e12 + 5e6, 1e12, TIGHT)
    assert not numbers_match(1e12 + 2e7, 1e12, TIGHT)


def test_the_loose_allowance_accepts_what_the_tight_one_refuses():
    assert numbers_match(8.68, 8.68421, LOOSE)
    assert not numbers_match(8.68, 8.68421, TIGHT)
    assert not numbers_match(8.6, 8.68421, LOOSE)


def test_yes_and_no_match_exactly_and_never_match_a_number():
    assert values_match("yes", "yes", TIGHT)
    assert values_match("no", "no", TIGHT)
    assert not values_match("yes", "no", TIGHT)
    assert not values_match(1.0, "yes", TIGHT)
    assert not values_match("yes", 1.0, TIGHT)
    assert not values_match(None, 1.0, TIGHT)
    assert not values_match(None, "yes", TIGHT)
    assert values_match(825.0, 825.0, TIGHT)


def test_allowance_used_is_the_gap_as_a_share_of_what_is_allowed():
    assert allowance_used(10.25, 10.0, Allowance(absolute=0.5, relative=0.0)) == 0.5
    assert allowance_used(10.0, 10.0, TIGHT) == 0.0
    assert allowance_used(12.0, 8.0, Allowance(absolute=0.0, relative=0.25)) == 2.0
    relative_only = Allowance(absolute=0.0, relative=0.01)
    assert allowance_used(0.0, 0.0, relative_only) == 0.0
    assert allowance_used(0.1, 0.0, relative_only) == math.inf


# --------------------------------------------------------------------------- the answer line


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("825", (825.0,)),
        ("-12.5", (-12.5,)),
        ("0.084", (0.084,)),
        (".35", (0.35,)),
        ("+5", (5.0,)),
        ("- 5", (-5.0,)),
        ("$1,234.50", (1234.5,)),
        ("$ 1,234.50", (1234.5,)),
        ("-$5", (-5.0,)),
        ("1,000", (1000.0,)),
        ("9.2e+08", (920000000.0,)),
        ("1E5", (100000.0,)),
        ("2.5e-3", (0.0025,)),
        ("8.68.", (8.68,)),
        ("5.", (5.0,)),
        ("  825  ", (825.0,)),
        ("yes", ("yes",)),
        ("YES", ("yes",)),
        (" No. ", ("no",)),
    ],
)
def test_read_answer_reads_a_number_or_yes_no(text, expected):
    assert read_answer(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("8.4%", (8.4, 0.084)), ("8.4 %", (8.4, 0.084)), ("-5%", (-5.0, -0.05)), ("$50%", (50, 0.5))],
)
def test_a_percent_answer_can_mean_either_scale(text, expected):
    assert read_answer(text) == pytest.approx(expected)
    assert len(read_answer(text)) == 2


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        ".",
        "%",
        "$",
        "3.8 million",
        "( 5 )",
        "12 34",
        "1, 2",
        "yes no",
        "maybe",
        "--5",
        "5-",
        "1e",
        "e5",
        "1e1000",
        "1e999",
        "nan",
        "inf",
        "1_000",
        "１２",
        "0x10",
        "8.4%%",
        "9" * 401,
        None,
        5,
    ],
)
def test_read_answer_refuses_what_is_not_an_answer(text):
    assert read_answer(text) == ()


# --------------------------------------------------------------------------- the gold reply


def test_gold_from_target_reads_the_program_and_the_value():
    gold = gold_for()
    assert gold.value == 8.68421
    assert gold.program_text == "subtract(920, 95), divide(#0, 95)"
    assert gold.answer_text == "8.68421"
    assert [step.operation for step in gold.program.steps] == ["subtract", "divide"]

    assert gold_for("PROGRAM: greater(920, 95)\nANSWER: yes").value == "yes"
    assert gold_for("PROGRAM: add(1, 2)\nANSWER: 920000000").value == 920000000.0


def test_gold_with_a_program_that_does_not_parse_keeps_its_value():
    gold = gold_for("PROGRAM: add(1)\nANSWER: 3")
    assert gold.program is None and gold.value == 3.0


@pytest.mark.parametrize(
    "target",
    [
        "ANSWER: 3",  # no program line
        "PROGRAM: add(1, 2)",  # no answer
        "PROGRAM: add(1, 2)\nANSWER: about three",
        "PROGRAM: add(1, 2)\nANSWER: 8.4%",  # two readings: not a gold value
        "",
    ],
)
def test_gold_that_cannot_be_read_raises(target):
    with pytest.raises(EvaluationError):
        gold_for(target)


def test_evaluation_error_is_a_config_error():
    assert issubclass(EvaluationError, ConfigError)


# --------------------------------------------------------------------------- program match


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("add(4850.0, 1)", "add(4850, 1)"),
        ("add(3.50, 1)", "add(3.5, 1)"),
        ("multiply(.35, 2)", "multiply(0.35, 2)"),
        ("divide(5, const_100)", "divide(5, 100)"),
        ("multiply(5, const_m1)", "multiply(5, -1)"),
        ("add(6.9%, 1)", "add(0.069, 1)"),
        ("exp(2, 3)", "exp(2.0, const_3)"),
        ("add( 1,2 ),add(#0 , 3)", "add(1, 2), add(#0, 3)"),
        ("table_sum(revenue, none)", "table_sum(revenue, 5)"),
        ("table_max( revenue , none)", "table_max(revenue, none)"),
    ],
)
def test_programs_match_compares_numbers_by_value(first, second):
    assert programs_match(program(first), program(second))
    assert programs_match(program(second), program(first))


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("add(1, 2)", "add(2, 1)"),  # D15: argument order matters
        ("add(1, 2), add(3, 4)", "add(3, 4), add(1, 2)"),  # D15: step order matters
        ("add(1, 2)", "subtract(1, 2)"),
        ("add(1, 2)", "add(1, 2), add(#0, 1)"),
        ("add(1, 2), add(3, 4), add(#0, #1)", "add(1, 2), add(3, 4), add(#1, #1)"),
        ("add(1, 2), add(#0, 1)", "add(1, 2), add(0, 1)"),  # a reference is not a number
        ("table_sum(revenue, none)", "table_sum(margin, none)"),
        ("table_sum(revenue, none)", "table_sum(Revenue, none)"),
        ("table_sum(revenue, none)", "table_max(revenue, none)"),
        ("add(1, 2)", "add(1, 2.0000001)"),
    ],
)
def test_programs_that_differ_do_not_match(first, second):
    assert not programs_match(program(first), program(second))


def test_a_missing_program_matches_nothing():
    assert not programs_match(None, program("add(1, 2)"))
    assert not programs_match(program("add(1, 2)"), None)
    assert not programs_match(None, None)


# --------------------------------------------------------------------------- mismatch labels


@pytest.mark.parametrize(
    ("value", "gold", "label"),
    [
        (868.421, 8.68421, "scale"),
        (0.0868421, 8.68421, "scale"),
        (8684.21, 8.68421, "scale"),
        (0.00868421, 8.68421, "scale"),
        (8684210.0, 8.68421, "scale"),
        (8.68421e-06, 8.68421, "scale"),
        (-825.0, 825.0, "sign"),
        (825.0, -825.0, "sign"),
        (825.0, "yes", "kind"),
        ("no", 825.0, "kind"),
        ("yes", "no", "other"),
        (826.0, 825.0, "other"),
        (86.8421, 8.68421, "other"),  # ten is not a configured factor
        (5.0, 0.0, "other"),
        (0.0, 5.0, "other"),
    ],
)
def test_mismatch_label_names_a_wrong_result(value, gold, label):
    assert mismatch_label(value, gold, CONFIG) == label
    assert label in MISMATCHES


@pytest.mark.parametrize(
    ("value", "gold"), [(825.0, 825.0), (8.684210526, 8.68421), ("yes", "yes"), (None, 825.0)]
)
def test_a_right_or_absent_result_gets_no_label(value, gold):
    assert mismatch_label(value, gold, CONFIG) is None


# --------------------------------------------------------------------------- scoring


def test_the_usage_example():
    config = load_evaluation_config()
    gold = Gold.from_target(GOLD_TARGET, FORMAT)
    reply = "PROGRAM: subtract(920.0, 95), divide(#0, 95)\nANSWER: 8.68"
    result = score_reply(run_reply(reply, table=None), gold, config)
    assert result.execution_correct and result.answer_correct
    assert result.program_match and result.format_valid
    assert summarize([result])["all"]["execution_accuracy"] == {"count": 1, "rate": 1.0}


def test_a_perfect_reply():
    result = score(GOLD_TARGET, example_id="A-1")
    assert result.all_correct and result.executed
    assert result.example_id == "A-1"
    assert (result.error_code, result.mismatch, result.deviations) == (None, None, ())
    assert result.relative_gap == pytest.approx(6.06e-08, rel=0.01)


def test_right_program_and_wrong_answer():
    result = score("PROGRAM: subtract(920, 95), divide(#0, 95)\nANSWER: 9.5")
    assert result.execution_correct and result.program_match and result.format_valid
    assert not result.answer_correct and not result.all_correct


def test_wrong_program_and_right_answer():
    result = score("PROGRAM: add(1, 2)\nANSWER: 8.68421")
    assert result.executed and not result.execution_correct and not result.program_match
    assert result.answer_correct
    assert result.mismatch == "other"


def test_the_answer_line_is_judged_more_loosely_than_the_program():
    # 8.68 on the answer line is within 0.5%; a program that gives 8.68 is not within 0.001%.
    result = score("PROGRAM: add(8.68, 0)\nANSWER: 8.68")
    assert result.answer_correct and not result.execution_correct
    loose = dataclasses.replace(CONFIG, execution=LOOSE)
    assert score("PROGRAM: add(8.68, 0)\nANSWER: 8.68", config=loose).execution_correct


def test_a_recovered_reply_is_right_on_execution_and_wrong_on_format():
    result = score("Sure!\n" + GOLD_TARGET)
    assert result.execution_correct and result.answer_correct and result.program_match
    assert not result.format_valid
    assert result.deviations == ("text_before",)


@pytest.mark.parametrize(
    ("reply", "code"),
    [
        ("", "reply_empty"),
        ("I cannot tell.", "no_program"),
        ("PROGRAM: add(1)\nANSWER: 8.68421", "wrong_arity"),
        ("PROGRAM: divide(5, 0)\nANSWER: 8.68421", "division_by_zero"),
        ("PROGRAM: table_sum(no such row, none)\nANSWER: 8.68421", "row_not_found"),
    ],
)
def test_an_error_at_each_stage_is_wrong_on_what_could_not_be_judged(reply, code):
    result = score(reply)
    assert result.error_code == code
    assert not result.executed and not result.execution_correct and not result.program_match
    assert result.mismatch is None and result.relative_gap is None
    # D13: the answer line is judged whether or not the program ran.
    assert result.answer_correct == ("ANSWER: 8.68421" in reply)


def test_an_equivalent_program_is_right_but_is_not_a_program_match():
    result = score("PROGRAM: divide(825, 95)\nANSWER: 8.68421")
    assert result.execution_correct and not result.program_match

    swapped = score("PROGRAM: add(2, 1)\nANSWER: 3", target="PROGRAM: add(1, 2)\nANSWER: 3")
    assert swapped.execution_correct and not swapped.program_match


def test_a_result_on_another_scale_is_wrong_and_labelled():
    reply = "PROGRAM: subtract(920, 95), divide(#0, 95), multiply(#1, const_100)\nANSWER: "
    with_sign = score(reply + "868.421%")
    assert not with_sign.execution_correct and with_sign.mismatch == "scale"
    assert with_sign.answer_correct  # D9: a percent sign allows either scale
    without_sign = score(reply + "868.421")
    assert not without_sign.answer_correct  # D10: no sign, no leniency


def test_yes_and_no_gold():
    target = "PROGRAM: greater(920, 95)\nANSWER: yes"
    assert score(target, target=target).all_correct
    assert score("PROGRAM: greater(920, 95)\nANSWER: Yes.", target=target).answer_correct

    number = score("PROGRAM: subtract(920, 95)\nANSWER: 825", target=target)
    assert not number.execution_correct and not number.answer_correct
    assert number.mismatch == "kind" and number.relative_gap is None

    other = score("PROGRAM: greater(95, 920)\nANSWER: no", target=target)
    assert not other.execution_correct and other.mismatch == "other"


def test_a_gold_of_zero_has_no_relative_gap():
    target = "PROGRAM: subtract(95, 95)\nANSWER: 0"
    result = score(target, target=target)
    assert result.all_correct and result.relative_gap is None


def test_no_reply_matches_a_gold_program_that_does_not_parse():
    target = "PROGRAM: add(1)\nANSWER: 3"
    result = score("PROGRAM: add(1, 2)\nANSWER: 3", target=target)
    assert result.execution_correct and not result.program_match


def test_score_to_dict_holds_plain_values():
    result = score("Sure!\n" + GOLD_TARGET, example_id="A-1").to_dict()
    assert result["deviations"] == ["text_before"]
    assert result["example_id"] == "A-1" and result["format_valid"] is False
    assert set(result) == {
        "example_id",
        "format_valid",
        "executed",
        "execution_correct",
        "answer_correct",
        "program_match",
        "error_code",
        "deviations",
        "mismatch",
        "relative_gap",
    }


# --------------------------------------------------------------------------- totals


def made(example_id, fmt=True, executed=True, execution=True, answer=True, match=True, **more):
    return ExampleScore(example_id, fmt, executed, execution, answer, match, **more)


def test_summarize_counts_over_every_example():
    scores = [
        made("a"),
        made("b", fmt=False, deviations=("text_before", "label_case")),
        made("c", execution=False, match=False, mismatch="scale"),
        made("d", executed=False, execution=False, match=False, error_code="syntax"),
        made("e", executed=False, execution=False, answer=False, match=False, error_code="syntax"),
    ]
    totals = summarize(scores)
    block = totals["all"]
    assert block["examples"] == 5
    assert block["format_validity"] == {"count": 4, "rate": 0.8}
    assert block["execution_accuracy"] == {"count": 2, "rate": 0.4}
    assert block["answer_accuracy"] == {"count": 4, "rate": 0.8}
    assert block["program_match"] == {"count": 2, "rate": 0.4}
    assert block["executed"] == {"count": 3, "rate": 0.6}
    assert block["error_codes"] == {"syntax": 2}
    assert block["deviations"] == {"label_case": 1, "text_before": 1}
    assert block["mismatches"] == {"scale": 1}
    assert totals["unflagged"] == block and totals["flagged_examples"] == 0
    assert all(name in block for name in METRICS)


def test_flagged_ids_change_only_the_unflagged_block():
    scores = [made("a"), made("b", execution=False), made("c", execution=False)]
    plain = summarize(scores)
    totals = summarize(scores, flagged={"b", "not in the scores"})
    assert totals["all"] == plain["all"]
    assert totals["all"]["execution_accuracy"] == {"count": 1, "rate": pytest.approx(1 / 3)}
    assert totals["unflagged"]["examples"] == 2
    assert totals["unflagged"]["execution_accuracy"] == {"count": 1, "rate": 0.5}
    assert totals["flagged_examples"] == 1


def test_summarize_all_wrong_empty_and_a_generator():
    wrong = made("a", fmt=False, executed=False, execution=False, answer=False, match=False)
    block = summarize([wrong, wrong])["all"]
    assert all(block[name] == {"count": 0, "rate": 0.0} for name in METRICS)

    empty = summarize([])
    assert empty["all"]["examples"] == 0
    assert all(empty["all"][name] == {"count": 0, "rate": None} for name in METRICS)
    assert empty["unflagged"] == empty["all"] and empty["flagged_examples"] == 0

    assert summarize(item for item in [made("a")])["all"]["examples"] == 1


# --------------------------------------------------------------------------- config


def changed_config(tmp_path, change):
    raw = yaml.safe_load((CONFIG_DIR / "evaluation.yaml").read_text(encoding="utf-8"))
    change(raw)
    folder = tmp_path / "config"
    folder.mkdir()
    (folder / "evaluation.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    return folder


def test_the_committed_config_holds_the_decided_allowances(tmp_path):
    config = load_evaluation_config()
    assert config.execution == Allowance(absolute=0.00001, relative=0.005)
    assert config.answer == Allowance(absolute=0.00001, relative=0.005)
    assert config.scale_factors == (100.0, 1000.0, 1000000.0)
    assert config.flags_file == REPO_ROOT / "splits" / "finqa_gold_flags.json"
    assert config.report_dir == REPO_ROOT / "reports" / "data"
    assert config.results_file == REPO_ROOT / "ledger" / "evaluations.jsonl"
    assert config.diagnostic_relative == 0.00001 and config.progress_every == 50
    assert config.tight_execution == Allowance(absolute=0.00001, relative=0.00001)

    moved = load_evaluation_config(root=tmp_path)
    assert moved.flags_file == tmp_path / "splits" / "finqa_gold_flags.json"
    assert moved.root == tmp_path


@pytest.mark.parametrize("value", [-0.1, "0.5", True, None, float("inf"), float("nan")])
def test_an_allowance_that_is_not_a_number_of_zero_or_more(tmp_path, value):
    folder = changed_config(tmp_path, lambda raw: raw["execution"].update(relative=value))
    with pytest.raises(ConfigError, match="execution.relative"):
        load_evaluation_config(config_dir=folder)


def test_both_parts_zero_is_refused_and_one_part_zero_is_allowed(tmp_path):
    folder = changed_config(tmp_path, lambda raw: raw["answer"].update(absolute=0, relative=0))
    with pytest.raises(ConfigError, match="cannot both be zero"):
        load_evaluation_config(config_dir=folder)

    (tmp_path / "other").mkdir()
    folder = changed_config(tmp_path / "other", lambda raw: raw["answer"].update(absolute=0))
    assert load_evaluation_config(config_dir=folder).answer == Allowance(0.0, 0.005)


def test_a_missing_key_is_named(tmp_path):
    folder = changed_config(tmp_path, lambda raw: raw["answer"].pop("relative"))
    with pytest.raises(ConfigError, match="answer.relative"):
        load_evaluation_config(config_dir=folder)


@pytest.mark.parametrize("value", [-1, "0.1", True, None, float("inf"), float("nan")])
def test_the_diagnostic_allowance_must_be_a_number_of_zero_or_more(tmp_path, value):
    change = {"execution_relative": value}
    folder = changed_config(tmp_path, lambda raw: raw["diagnostic"].update(change))
    with pytest.raises(ConfigError, match="diagnostic.execution_relative"):
        load_evaluation_config(config_dir=folder)


@pytest.mark.parametrize("value", [0, -5, 2.5, "50", True])
def test_progress_every_must_be_a_positive_whole_number(tmp_path, value):
    folder = changed_config(tmp_path, lambda raw: raw["runner"].update(progress_every=value))
    with pytest.raises(ConfigError, match="runner.progress_every"):
        load_evaluation_config(config_dir=folder)


@pytest.mark.parametrize("factors", [[100, 1], [100, "1000"], [True], 100, [0.5]])
def test_scale_factors_must_be_numbers_above_one(tmp_path, factors):
    folder = changed_config(tmp_path, lambda raw: raw.update(scale_factors=factors))
    with pytest.raises(ConfigError, match="scale_factors"):
        load_evaluation_config(config_dir=folder)


# --------------------------------------------------------------------------- guards


def test_the_metrics_do_not_load_the_data_package():
    """The data package's gold self-check imports the metrics, so the reverse must not hold."""
    code = (
        "import sys, qlora_finetuning.evaluation.metrics\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('qlora_finetuning.data'))\n"
        "print(loaded)\n"
        "sys.exit(1 if loaded else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_package_exports_the_metrics():
    import qlora_finetuning.evaluation as package

    for name in package.__all__:
        assert hasattr(package, name)
        if hasattr(metrics, name) and name in getattr(metrics, "__dict__", {}):
            assert getattr(package, name) is getattr(metrics, name)
    assert "score_reply" in package.__all__ and "summarize" in package.__all__
    assert "run_evaluation" not in package.__all__  # the runner is imported by its own name


# --------------------------------------------------------------------------- answer_agrees


def agrees(reply, config=CONFIG):
    return metrics.answer_agrees(run_reply(reply, TABLE, FORMAT, DSL), config)


def test_answer_agrees_compares_the_answer_line_with_the_programs_own_result():
    program_line = "PROGRAM: subtract(920, 95), divide(#0, 95)"
    assert agrees(f"{program_line}\nANSWER: 8.68421") is True
    assert agrees(f"{program_line}\nANSWER: 8.68") is True  # inside the answer allowance
    assert agrees(f"{program_line}\nANSWER: 8.5") is False
    assert agrees("PROGRAM: divide(95, 920)\nANSWER: 10.33%") is True  # either scale of a %
    assert agrees("PROGRAM: greater(920, 95)\nANSWER: yes") is True
    assert agrees("PROGRAM: greater(920, 95)\nANSWER: no") is False
    assert agrees("PROGRAM: greater(920, 95)\nANSWER: 1") is False  # a number is not a word


def test_answer_agrees_is_none_when_there_is_nothing_to_compare():
    assert agrees("PROGRAM: divide(5, 0)\nANSWER: 0") is None  # the program did not run
    assert agrees("PROGRAM: add(1, 2)\nANSWER: about three") is None
    assert agrees("PROGRAM: add(1, 2)") is None  # no answer line
    assert agrees("no reply in the format") is None


def test_answer_agrees_uses_the_answer_allowance_and_is_not_a_metric():
    reply = "PROGRAM: subtract(920, 95), divide(#0, 95)\nANSWER: 8.68"
    strict_answer = dataclasses.replace(CONFIG, execution=LOOSE, answer=TIGHT)
    assert agrees(reply) is True and agrees(reply, strict_answer) is False
    assert len(METRICS) == 4
    totals = summarize([score(reply)])
    assert not any("agree" in key for key in totals["all"])
