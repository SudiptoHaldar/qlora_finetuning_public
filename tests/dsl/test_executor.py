"""The executor (req_002 v2.01). Values are computed by hand; the table is invented."""

from __future__ import annotations

import pytest

from qlora_finetuning.dsl import (
    DslError,
    ExecutionError,
    Program,
    ProgramError,
    Step,
    cell_number,
    execute,
    parse_program,
    row_numbers,
)
from qlora_finetuning.dsl.program import Argument


def run(text, config, table=None):
    return execute(parse_program(text, config), table)


def code_of(text, config, table=None) -> str:
    with pytest.raises(ExecutionError) as caught:
        run(text, config, table)
    assert caught.value.stage == "execution"
    return caught.value.code


# --------------------------------------------------------------------------- arithmetic


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("add(920, 95)", 1015.0),
        ("subtract(95, 920)", -825.0),
        ("multiply(-3.5, 4)", -14.0),
        ("divide(1, 8)", 0.125),
        ("exp(2, 10)", 1024.0),
        ("exp(4, .5)", 2.0),
        ("exp(2, -1)", 0.5),
        ("add(6.9%, const_1)", 1.069),
        ("divide(7%, const_100)", 0.0007),
        ("multiply(250, const_m1)", -250.0),
        ("multiply(3, const_1000000000)", 3e9),
        ("subtract(920, 95), divide(#0, 95)", 825 / 95),
        ("add(1, 2), add(3, 4), multiply(#0, #1), subtract(#2, #0)", 18.0),
        ("add(1, 2), add(10, 20)", 30.0),  # an unused step is allowed
    ],
)
def test_arithmetic_by_hand(text, value, config):
    result = run(text, config)
    assert result.value == pytest.approx(value, rel=1e-12)
    assert isinstance(result.value, float)


def test_the_value_is_not_rounded(config):
    assert run("divide(1, 3)", config).value == 1 / 3


def test_every_step_value_is_kept(config):
    result = run("subtract(920, 95), divide(#0, 95), greater(#1, 8)", config)
    assert result.step_values == (825.0, 825 / 95, "yes")
    assert result.value == "yes"
    assert result.notes == ()


@pytest.mark.parametrize(
    ("text", "value"),
    [("greater(2, 1)", "yes"), ("greater(1, 2)", "no"), ("greater(2, 2)", "no"),
     ("greater(-1, -2)", "yes"), ("greater(50%, .4)", "yes")],
)  # fmt: skip
def test_greater_is_strict(text, value, config):
    assert run(text, config).value == value


def test_greater_may_sit_in_the_middle_when_nothing_uses_it(config):
    assert run("greater(2, 1), add(1, 2)", config).value == 3.0


@pytest.mark.parametrize(
    "text",
    ["greater(2, 1), add(#0, 1)", "greater(2, 1), greater(1, #0)", "greater(1, 2), exp(#0, 2)"],
)
def test_a_yes_or_no_value_is_not_a_number(text, config):
    assert code_of(text, config) == "yes_no_operand"


@pytest.mark.parametrize(
    "text",
    ["divide(5, 0)", "divide(0, 0)", "divide(5, 0.0)", "subtract(3, 3), divide(1, #0)",
     "divide(1, 0%)"],
)  # fmt: skip
def test_division_by_zero(text, config):
    assert code_of(text, config) == "division_by_zero"


@pytest.mark.parametrize(
    "text",
    [
        "exp(10, 400)",  # overflow in a power
        "exp(-8, .5)",  # a negative number to a fractional power
        "exp(0, -1)",  # zero to a negative power
        "multiply(1000000000, 1000000000), multiply(#0, #0), multiply(#1, #1), "
        "multiply(#2, #2), multiply(#3, #3), multiply(#4, #4)",  # overflow by multiplying
    ],
)
def test_a_result_that_is_not_a_finite_number(text, config):
    assert code_of(text, config) == "numeric_error"


# --------------------------------------------------------------------------- table cells


@pytest.mark.parametrize(
    ("cell", "number"),
    [
        ("920", 920.0),
        ("$ 920", 920.0),
        ("$920", 920.0),
        ("$ 1,234,567.5", 1234567.5),
        ("-9 ( 9 )", -9.0),
        ("$ -9.5 ( 9.5 )", -9.5),
        ("12.5%", 0.125),
        ("7.5 %", 0.075),
        ("9.9% ( 9.9 % )", 0.099),
        ("-5%", -0.05),
        ("+5%", 0.05),
        ("+10 % ( +10 % )", 0.1),
        (".5", 0.5),
        (" 4 ", 4.0),
        (4, 4.0),
    ],
)
def test_cells_that_are_numbers(cell, number):
    assert cell_number(cell) == pytest.approx(number)


@pytest.mark.parametrize(
    "cell",
    ["", " ", "-", "$ -", "n/a", "( 9 )", "( 9 ) % (  % )", "9 a", "2024 vs 2023", "1e5",
     "inf", "nan", "1_000", "+-5", "++5", "5.", "%", "12.5%%", "٥", None],
)  # fmt: skip
def test_cells_that_are_not_numbers(cell):
    assert cell_number(cell) is None


# --------------------------------------------------------------------------- table operations


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("table_sum(revenue, none)", 920 + 95 + 1234),
        ("table_average(revenue, none)", (920 + 95 + 1234) / 3),
        ("table_max(revenue, none)", 1234.0),
        ("table_min(revenue, none)", 95.0),
        ("table_average(margin, none)", (0.125 + 0.10 + 0.075) / 3),
        ("table_sum(net income ( loss ), none)", 0.0),
        ("table_min(net income ( loss ), none)", -9.0),
        ("table_sum(cash, end of year, none)", 6.0),
        ("table_max(growth ( in ( thousands ) ), none)", 30.0),
        ("table_sum(change, none)", 0.10),
        ("table_sum(revenue, 5)", 920 + 95 + 1234),  # the second argument is ignored
        ("table_sum(revenue, none), divide(#0, const_1000)", 2.249),
    ],
)
def test_table_operations_by_hand(text, value, config, table):
    result = run(text, config, table)
    assert result.value == pytest.approx(value, abs=1e-12)
    assert isinstance(result.value, float)


def test_the_header_row_can_be_named(config):
    table = [["year", "2024", "2023"], ["revenue", "1", "2"]]
    assert run("table_max(year, none)", config, table).value == 2024.0


@pytest.mark.parametrize("name", ["notes", "blanks", "bracketed"])
def test_a_row_with_a_cell_that_is_not_a_number(name, config, table):
    assert code_of(f"table_sum({name}, none)", config, table) == "row_not_numeric"


def test_a_row_with_no_value_cells(config):
    assert code_of("table_sum(lonely, none)", config, [["lonely"]]) == "row_not_numeric"


@pytest.mark.parametrize("name", ["profit", "Revenue", "revenu", "revenue 2024", "#0"])
def test_a_row_that_is_not_in_the_table(name, config, table):
    assert code_of(f"table_sum({name}, none)", config, table) == "row_not_found"


def test_an_empty_table_has_no_rows(config):
    assert code_of("table_sum(revenue, none)", config, []) == "row_not_found"


def test_a_table_operation_without_a_table(config):
    assert code_of("table_sum(revenue, none)", config) == "table_missing"
    assert run("add(1, 2)", config).value == 3.0  # arithmetic needs none


def test_a_repeated_row_name_means_the_last_row(config, table):
    result = run("table_sum(total, none)", config, table)
    assert result.value == 60.0
    assert result.notes == ("duplicate_row",)
    twice = run("table_sum(total, none), table_max(total, none)", config, table)
    assert twice.notes == ("duplicate_row",)
    assert run("table_sum(revenue, none)", config, table).notes == ()


def test_row_numbers_reports_the_numbers_and_the_repeat(table):
    assert row_numbers(table, "total") == ([10.0, 20.0, 30.0], True)
    assert row_numbers(table, "cash, end of year") == ([1.0, 2.0, 3.0], False)


def test_row_names_are_compared_after_trimming_the_cell(config):
    table = [["", "a"], ["  revenue  ", "7"]]
    assert run("table_sum(revenue, none)", config, table).value == 7.0


# --------------------------------------------------------------------------- hand-built programs


def test_a_hand_built_program_is_checked_too():
    one = Argument("integer", "1", 1.0)
    forward = Program((Step("add", (Argument("reference", "#3", 3), one)),))
    with pytest.raises(ProgramError) as caught:
        execute(forward)
    assert caught.value.code == "bad_reference"

    unknown = Program((Step("launch", (Argument("integer", "1", 1.0),) * 2),))
    with pytest.raises(DslError) as caught:
        execute(unknown)
    assert caught.value.code == "unknown_operation"

    with pytest.raises(DslError) as caught:
        execute(Program(()))
    assert caught.value.code == "empty_program"

    no_value = Program((Step("add", (Argument("none", "none"), Argument("integer", "1", 1.0))),))
    with pytest.raises(DslError) as caught:
        execute(no_value)
    assert caught.value.code == "bad_argument"
