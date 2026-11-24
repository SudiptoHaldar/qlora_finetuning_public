"""The program parser (req_002 v2.01). Every program here is invented."""

from __future__ import annotations

import pytest

from qlora_finetuning.dsl import (
    CONSTANTS,
    OPERATIONS,
    DslConfig,
    ProgramError,
    format_program,
    parse_program,
)


def code_of(text: str, config: DslConfig) -> str:
    with pytest.raises(ProgramError) as caught:
        parse_program(text, config)
    assert caught.value.stage == "program"
    return caught.value.code


# --------------------------------------------------------------------------- valid programs


def test_a_two_step_program(config):
    program = parse_program("subtract(920, 95), divide(#0, 95)", config)
    assert [step.operation for step in program.steps] == ["subtract", "divide"]
    first, second = program.steps[1].arguments
    assert (first.kind, first.text, first.value) == ("reference", "#0", 0)
    assert (second.kind, second.text, second.value) == ("integer", "95", 95.0)


@pytest.mark.parametrize("name", OPERATIONS)
def test_each_operation_parses(name, config):
    second = "none" if name.startswith("table_") else "2"
    program = parse_program(f"{name}(revenue, {second})" if second == "none" else f"{name}(1, 2)",
                            config)  # fmt: skip
    assert program.steps[0].operation == name


@pytest.mark.parametrize(
    "text",
    [
        "add(1,2)",
        "  add ( 1 , 2 )  ",
        "add(\t1,\t2\t)",
        "add(1, 2),add(#0, 3)",
        "add(1, 2)  ,  add(#0, 3)",
    ],
)
def test_spacing_is_free_around_every_part(text, config):
    program = parse_program(text, config)
    assert format_program(program) in ("add(1, 2)", "add(1, 2), add(#0, 3)")


@pytest.mark.parametrize(
    ("text", "kind", "value"),
    [
        ("920", "integer", 920.0),
        ("-121", "integer", -121.0),
        ("0", "integer", 0.0),
        ("007", "integer", 7.0),
        ("4474.2", "decimal", 4474.2),
        ("4850.0", "decimal", 4850.0),
        ("-3.5", "decimal", -3.5),
        (".35", "decimal", 0.35),
        ("-.5", "decimal", -0.5),
        ("6.9%", "percent", 0.069),
        ("7%", "percent", 0.07),
        ("-2.5%", "percent", -0.025),
        ("const_100", "constant", 100.0),
        ("const_m1", "constant", -1.0),
    ],
)
def test_number_forms_that_are_accepted(text, kind, value, config):
    argument = parse_program(f"add({text}, 1)", config).steps[0].arguments[0]
    assert argument.kind == kind
    assert argument.text == text
    assert argument.value == pytest.approx(value)


@pytest.mark.parametrize(
    "text",
    [
        "+5", "1e5", "1E5", "5.", ".", "-", "--5", "1.2.3", "$5", "$ 5", "5 %", "5%%", "%",
        "1_000", "inf", "nan", "-inf", "0x10", "( 5 )", "five", "none", "", "1 2", "5-",
        "٥", "５", "²",
    ],
)  # fmt: skip
def test_number_forms_that_are_rejected(text, config):
    assert code_of(f"add(1, {text})", config) in ("bad_argument", "nested_call")


def test_a_thousands_separator_reads_as_a_third_argument(config):
    assert code_of("add(1,234, 5)", config) == "wrong_arity"


def test_a_number_too_large_to_be_finite_is_rejected(config):
    assert code_of(f"add(1, {'9' * 400})", config) == "bad_argument"


@pytest.mark.parametrize("name", CONSTANTS)
def test_each_constant_has_its_value(name, config):
    argument = parse_program(f"multiply(2, {name})", config).steps[0].arguments[1]
    assert argument.kind == "constant"
    expected = -1.0 if name == "const_m1" else float(name.removeprefix("const_"))
    assert argument.value == expected


@pytest.mark.parametrize("name", ["const_12", "const_0", "const_m2", "const_", "const_1.5"])
def test_a_constant_outside_the_list_is_an_error(name, config):
    assert code_of(f"multiply(2, {name})", config) == "unknown_constant"


# --------------------------------------------------------------------------- references


def test_references_reach_any_earlier_step(config):
    program = parse_program("add(1, 2), add(3, 4), subtract(#0, #1), divide(#2, #0)", config)
    assert [a.value for a in program.steps[3].arguments] == [2, 0]


@pytest.mark.parametrize(
    "text",
    [
        "add(#0, 1)",  # to itself
        "add(1, 2), add(#1, 3)",  # to itself, later
        "add(#1, 2), add(3, 4)",  # forward
        "add(1, 2), add(#5, 3)",  # to nothing
        "add(1, 2), add(#, 3)",
        "add(1, 2), add(#-1, 3)",
        "add(1, 2), add(# 0, 3)",
        "add(1, 2), add(#0.0, 3)",
        "add(1, 2), add(#a, 3)",
        "add(1, 2), add(#٠, 3)",
        "add(1, 2), add(#" + "9" * 3000 + ", 3)",
    ],
)
def test_a_reference_must_name_an_earlier_step(text, config):
    assert code_of(text, config) == "bad_reference"


def test_a_reference_with_leading_zeros_is_read(config):
    program = parse_program("add(1, 2), add(#" + "0" * 3000 + ", 3)", config)
    assert program.steps[1].arguments[0].value == 0


# --------------------------------------------------------------------------- malformed


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("", "empty_program"),
        ("   ", "empty_program"),
        ("add(1)", "wrong_arity"),
        ("add()", "wrong_arity"),
        ("add(1, 2, 3)", "wrong_arity"),
        ("table_sum(revenue)", "wrong_arity"),
        ("add(subtract(1, 2), 3)", "nested_call"),
        ("divide(add(1, 2), subtract(3, 4))", "nested_call"),
        ("add((1), 2)", "nested_call"),
        ("add(1, 2", "syntax"),
        ("add(1, (2", "nested_call"),
        ("add 1 2", "syntax"),
        ("42", "syntax"),
        ("add(1, 2),", "syntax"),
        ("add(1, 2), ", "syntax"),
        (", add(1, 2)", "syntax"),
        ("add(1, 2) add(3, 4)", "syntax"),
        ("add(1, 2).", "syntax"),
        ("add(1, 2); add(3, 4)", "syntax"),
        ("add(1, 2))", "syntax"),
        ("(1, 2)", "syntax"),
        ("table_sum(revenue, none", "syntax"),
        ("table_sum(net income ( loss, none)", "syntax"),
        ("sum(1, 2)", "unknown_operation"),
        ("Add(1, 2)", "unknown_operation"),
        ("ADD(1, 2)", "unknown_operation"),
        ("os.system(1, 2)", "unknown_operation"),
        ("add.__class__(1, 2)", "unknown_operation"),
        ("table_median(revenue, none)", "unknown_operation"),
        ("add (1, 2), minus(3, 4)", "unknown_operation"),
        ("add(1, 2),\nadd(3, 4)", "unknown_operation"),
        ("table_sum(, none)", "bad_row_name"),
        ("table_sum(   , none)", "bad_row_name"),
        ("table_sum(revenue, )", "bad_argument"),
        ("table_sum(revenue, nothing)", "bad_argument"),
        ("add(none, 1)", "bad_argument"),
    ],
)
def test_malformed_programs_get_a_named_error(text, code, config):
    assert code_of(text, config) == code


def test_a_closing_bracket_inside_a_row_name_ends_the_step_early(config):
    assert code_of("table_sum(net income ) loss, none)", config) == "wrong_arity"


def test_more_steps_than_the_limit_is_an_error(config):
    small = DslConfig(max_reply_chars=4000, max_steps=3)
    assert len(parse_program("add(1, 2), add(1, 2), add(1, 2)", small).steps) == 3
    assert code_of("add(1, 2), add(1, 2), add(1, 2), add(1, 2)", small) == "too_many_steps"


def test_an_error_says_where_and_quotes_little(config):
    with pytest.raises(ProgramError) as caught:
        parse_program("add(1, 2), " + "x" * 500 + "(1, 2)", config)
    assert caught.value.code == "unknown_operation"
    assert caught.value.position == 11
    assert "x" * 41 not in str(caught.value)


def test_an_error_message_is_ascii(config):
    with pytest.raises(ProgramError) as caught:
        parse_program("add(1, été)", config)
    assert str(caught.value).isascii()


def test_a_program_that_is_not_text_is_the_callers_mistake(config):
    with pytest.raises(TypeError):
        parse_program(None, config)


# --------------------------------------------------------------------------- row names


@pytest.mark.parametrize(
    ("text", "row_name"),
    [
        ("table_sum(revenue, none)", "revenue"),
        ("table_max(expected volatility, none)", "expected volatility"),
        ("table_sum(  revenue  ,  none  )", "revenue"),
        ("table_sum(net income ( loss ), none)", "net income ( loss )"),
        ("table_sum(growth ( in ( thousands ) ), none)", "growth ( in ( thousands ) )"),
        ("table_sum(cash, end of year, none)", "cash, end of year"),
        ("table_sum(a ( b, c ), d, none)", "a ( b, c ), d"),
        ("table_sum(r&d / sales - 2024, none)", "r&d / sales - 2024"),
        ("table_sum(#0, none)", "#0"),
        ("table_sum(add(1, 2), none)", "add(1, 2)"),
    ],
)
def test_a_row_name_runs_to_the_last_top_level_comma(text, row_name, config):
    first, second = parse_program(text, config).steps[0].arguments
    assert (first.kind, first.text, first.value) == ("row_name", row_name, None)
    assert (second.kind, second.text) == ("none", "none")


def test_a_table_step_may_carry_a_number_as_second_argument(config):
    second = parse_program("table_average(revenue, 3)", config).steps[0].arguments[1]
    assert (second.kind, second.value) == ("integer", 3.0)


def test_a_table_step_can_be_followed_by_other_steps(config):
    program = parse_program("table_sum(net income ( loss ), none), divide(#0, 3)", config)
    assert [step.operation for step in program.steps] == ["table_sum", "divide"]


# --------------------------------------------------------------------------- canonical form


@pytest.mark.parametrize(
    ("text", "canonical"),
    [
        ("subtract(920, 95), divide(#0, 95)", "subtract(920, 95), divide(#0, 95)"),
        ("add( 6.9%,const_1 ),multiply(#0,200)", "add(6.9%, const_1), multiply(#0, 200)"),
        ("  table_max( net income ( loss ) ,none )", "table_max(net income ( loss ), none)"),
        ("greater(4850.0, .35)", "greater(4850.0, .35)"),
    ],
)
def test_the_canonical_form_is_stable(text, canonical, config):
    program = parse_program(text, config)
    assert format_program(program) == canonical
    assert parse_program(canonical, config) == program
