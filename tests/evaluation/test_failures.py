"""Tests for qlora_finetuning.evaluation.failures: one computed category per wrong reply."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from qlora_finetuning.dsl import DslConfig, ReplyFormat, run_reply
from qlora_finetuning.dsl.outcome import ReplyOutcome
from qlora_finetuning.evaluation import failures as failures_module
from qlora_finetuning.evaluation.failures import (
    CATEGORIES,
    RIGHT,
    Failure,
    classify,
    company_of,
    count_failures,
    failure_counts,
    format_table,
    run_failures,
)
from qlora_finetuning.evaluation.fake import FakePredictor
from qlora_finetuning.evaluation.metrics import Allowance, EvaluationConfig, Gold

from .conftest import GOLD_REPLY, evaluate, ids_of

CONFIG = EvaluationConfig(
    root=Path("."),
    execution=Allowance(absolute=0.00001, relative=0.00001),
    answer=Allowance(absolute=0.00001, relative=0.005),
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
TWO_STEPS = "PROGRAM: subtract(920, 95), divide(#0, 95)\nANSWER: 8.68421"


def category(reply, target=TWO_STEPS, stop_reason="end"):
    outcome = run_reply(reply, TABLE, FORMAT, DSL)
    return classify(outcome, stop_reason, Gold.from_target(target, FORMAT, DSL), CONFIG, DSL)


def reply(program, answer="1"):
    return f"PROGRAM: {program}\nANSWER: {answer}"


# --------------------------------------------------------------------------- each category


def test_a_right_reply_is_right_however_it_got_there():
    assert category(TWO_STEPS) == RIGHT
    assert category(reply("divide(825, 95)", "8.68421")) == RIGHT  # fewer steps, the same value
    assert category(TWO_STEPS, stop_reason="length") == RIGHT


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I do not know.", "no_program"),
        (reply("subtract(920 95"), "bad_program"),
        (reply("divide(subtract(920, 95), 95)"), "bad_program"),  # a nested call
        (reply("table_sum(profit, none)"), "not_executable"),  # no such row
        (reply("greater(920, 95)", "yes"), "kind"),
        (reply("subtract(920, 95), divide(#0, 95), multiply(#1, const_100)"), "steps_more"),
        (reply("subtract(920, 95)", "825"), "steps_fewer"),
        (reply("add(920, 95), divide(#0, 95)"), "operation"),
        (reply("subtract(920, 96), divide(#0, 95)"), "evidence"),
        (reply("subtract(95, 920), divide(#0, 95)"), "order"),  # the negative of gold
        (reply("subtract(920, 95), divide(95, #0)"), "order"),  # the reciprocal: no sign error
    ],
)
def test_a_wrong_reply_gets_the_first_category_that_fits(text, expected):
    assert category(text) == expected


def test_a_reply_cut_off_at_the_limit_is_cut_off_whatever_else_is_wrong():
    for text in ("I do not", reply("subtract(920, 96), divide(#0, 95)"), reply("subtract(920")):
        assert category(text, stop_reason="length") == "cut_off"
    assert category(reply("subtract(920, 95)"), stop_reason="other") == "cut_off"


def test_precedence_between_categories_that_both_fit():
    # Reversed in one step and another number in the next: the number decides.
    assert category(reply("subtract(95, 920), divide(#0, 96)")) == "evidence"
    # More steps, and another operation among them: the count of steps decides.
    assert category(reply("add(920, 95), divide(#0, 95), add(#1, 1)")) == "steps_more"
    # A yes where a number was due is a matter of kind before it is one of steps.
    assert category(
        reply("greater(920, 95)", "yes"), "PROGRAM: subtract(920, 95)\nANSWER: 825"
    ) == ("kind")


def test_reversed_arguments_of_a_comparison_and_a_table_row():
    yes = "PROGRAM: greater(920, 95)\nANSWER: yes"
    assert category(reply("greater(95, 920)", "no"), yes) == "order"
    total = "PROGRAM: table_sum(revenue, none)\nANSWER: 1015"
    assert category(total, total) == RIGHT
    assert category(reply("table_sum(margin, none)"), total) == "evidence"


@pytest.mark.parametrize(
    ("gold_answer", "expected"),
    [("8.25", "scale"), ("-825", "sign"), ("7", "other")],
)
def test_the_gold_program_with_a_wrong_result_is_labelled_by_its_value(gold_answer, expected):
    # Possible only where the gold answer is not what the gold program gives.
    target = f"PROGRAM: subtract(920, 95)\nANSWER: {gold_answer}"
    assert category(reply("subtract(920, 95)", "825"), target) == expected


def test_what_cannot_be_compared_is_other():
    no_gold_program = Gold(program=None, value=5.0, program_text="x", answer_text="5")
    outcome = run_reply(reply("subtract(920, 95)"), TABLE, FORMAT, DSL)
    assert classify(outcome, "end", no_gold_program, CONFIG, DSL) == "other"
    gold = Gold.from_target(TWO_STEPS, FORMAT, DSL)
    assert classify(ReplyOutcome(), "end", gold, CONFIG, DSL) == "other"  # no value, no error


def test_every_category_is_a_name_of_the_table():
    assert len(set(CATEGORIES)) == len(CATEGORIES) == 13 and RIGHT not in CATEGORIES
    assert CATEGORIES[:4] == ("cut_off", "no_program", "bad_program", "not_executable")


# --------------------------------------------------------------------------- counts


def failure(example_id, name, **more):
    values = {
        "example_id": example_id, "category": name, "error_code": None, "mismatch": None,
        "format_valid": True, "deviations": (), "answer_correct": False, "stop_reason": "end",
        "gold_steps": 1, "gold_operations": ("subtract",), "input_tokens": 100,
    }  # fmt: skip
    return Failure(**(values | more))


def test_counts_add_up_and_say_what_explains_them():
    found = [
        failure("A/1", RIGHT),
        failure("A/2", "order", mismatch="sign"),
        failure("B/1", "bad_program", error_code="nested_call", format_valid=False),
        failure("B/2", "bad_program", error_code="nested_call", answer_correct=True),
        failure("B/3", "not_executable", error_code="row_not_found"),
        failure("C/1", "evidence", mismatch="scale"),
    ]
    counts = count_failures(found)
    assert counts["examples"] == 6 and counts["right"] == 1
    assert sum(counts["categories"].values()) + counts["right"] == counts["examples"]
    assert list(counts["categories"]) == list(CATEGORIES)  # every category, in order
    assert counts["categories"]["bad_program"] == 2 and counts["categories"]["cut_off"] == 0
    assert counts["error_codes"] == {"nested_call": 2, "row_not_found": 1}
    assert counts["value_labels"] == {"scale": 1, "sign": 1}
    assert counts["wrong_in_the_format"] == 4 and counts["answer_right_program_wrong"] == 1
    assert count_failures([])["examples"] == 0


def test_the_format_table_counts_right_replies_too():
    found = [
        failure("A/1", RIGHT, format_valid=False, deviations=("text_after",)),
        failure("A/2", "cut_off", format_valid=False, deviations=("answer_missing",),
                stop_reason="length"),
        failure("A/3", "order"),
        failure("A/4", "other", format_valid=False),
    ]  # fmt: skip
    table = format_table(found)
    assert (table["in_the_format"], table["not_in_the_format"]) == (1, 3)
    assert table["not_in_the_format_and_right"] == 1
    assert table["deviations"] == {"(none recorded)": 1, "answer_missing": 1, "text_after": 1}
    assert table["stop_reasons"] == {"end": 3, "length": 1}


def test_the_company_is_the_start_of_the_id():
    assert company_of("ACME/2015/page_10.pdf-1") == "ACME"
    assert company_of("no-slash") == "(unknown)" and company_of("/x") == "(unknown)"


# --------------------------------------------------------------------------- a stored run


def test_a_stored_run_is_read_scored_now_and_holds_no_text(context):
    ids = ids_of("dev", context)
    wrong = {ids[1]: reply("subtract(95, 920)", "-825"), ids[2]: "zebrafish, I think"}
    predictor = FakePredictor(lambda request: wrong.get(request.example_id, GOLD_REPLY))
    summary = evaluate(predictor, "dev", context)

    found = run_failures(summary["run_id"], context)
    assert [item.example_id for item in found] == ids
    assert [item.category for item in found[:3]] == [RIGHT, "order", "no_program"]
    assert found[1].mismatch == "sign" and found[1].gold_operations == ("subtract",)
    assert found[2].error_code and not found[2].format_valid and found[0].input_tokens > 0
    counts = failure_counts(summary["run_id"], context)
    assert counts["right"] == summary["all"]["execution_accuracy"]["count"] == len(ids) - 2
    assert "zebrafish" not in repr(found) and "subtract(95" not in repr(found)


def test_the_module_imports_neither_path_and_no_model():
    tree = ast.parse(Path(failures_module.__file__).read_text(encoding="utf-8"))
    names = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [None])
    }
    assert not any(
        part in str(name) for name in names for part in ("local_qlora", "frontier", "torch")
    )
