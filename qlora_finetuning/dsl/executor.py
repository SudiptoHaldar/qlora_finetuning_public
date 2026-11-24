"""The executor: a parsed program and the example's table in, a value out (req_002 v2.01).

The model proposes the calculation; this code performs it. Numbers are Python floats and are
not rounded here: comparing a value with the gold answer is the metrics' subject (v2.02).

Each operation is one entry of a fixed table. Nothing is looked up by a name the model chose
except a table row, and a row is found by comparing text.

Usage::

    from qlora_finetuning.dsl.config import load_dsl_config
    from qlora_finetuning.dsl.executor import execute
    from qlora_finetuning.dsl.program import parse_program

    table = [["", "2024", "2023"], ["revenue", "$ 920", "$ 95"], ["margin", "12.5%", "10%"]]
    config = load_dsl_config()

    execute(parse_program("subtract(920, 95), divide(#0, 95)", config)).value   # 8.684...
    execute(parse_program("table_average(margin, none)", config), table).value  # 0.1125
    execute(parse_program("greater(920, 95)", config)).value                    # "yes"
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from qlora_finetuning.dsl.errors import ExecutionError, ProgramError, excerpt
from qlora_finetuning.dsl.language import (
    BLANKS,
    NO,
    REFERENCE,
    TABLE_OPERATIONS,
    YES,
    read_number,
)
from qlora_finetuning.dsl.program import Program, Step

DUPLICATE_ROW = "duplicate_row"

Value = float | str
Table = Sequence[Sequence[Any]]


@dataclass(frozen=True)
class Execution:
    """The result of a program: the value of its last step, and of every step."""

    value: Value
    step_values: tuple[Value, ...]
    notes: tuple[str, ...] = ()


# --------------------------------------------------------------------------- arithmetic


def _divide(first: float, second: float) -> float:
    if second == 0:
        raise ExecutionError("division_by_zero", "division by zero")
    return first / second


def _power(first: float, second: float) -> float:
    try:
        return math.pow(first, second)
    except (ValueError, OverflowError) as error:
        raise ExecutionError(
            "numeric_error", "the power has no finite real value"
        ) from error


_ARITHMETIC: dict[str, Callable[[float, float], Value]] = {
    "add": lambda first, second: first + second,
    "subtract": lambda first, second: first - second,
    "multiply": lambda first, second: first * second,
    "divide": _divide,
    "exp": _power,
    "greater": lambda first, second: YES if first > second else NO,
}

_TABLE: dict[str, Callable[[list[float]], float]] = {
    "table_sum": sum,
    "table_average": lambda numbers: sum(numbers) / len(numbers),
    "table_max": max,
    "table_min": min,
}


# --------------------------------------------------------------------------- tables


def cell_number(cell: Any) -> float | None:
    """The number in a table cell, or ``None`` when the cell is not a number.

    Dollar signs are removed, the text before the first ``(`` is kept, and thousands
    separators are removed. A number followed by ``%`` is divided by 100. So ``$ 1,234``
    is 1234, ``-9 ( 9 )`` is -9, and ``12.5%`` is 0.125. A bracketed negative such as
    ``( 9 )`` is not a number under this rule. A cell may carry a plus sign (``+5%``),
    which a program literal may not.
    """
    text = str(cell).replace("$", "").strip()
    text = text.split("(", 1)[0].strip().replace(",", "")
    if text.startswith("+") and not text.startswith("+-"):
        text = text[1:]
    if text.endswith("%"):
        number = read_number(text[:-1].strip())
        return None if number is None else number / 100.0
    return read_number(text)


def row_numbers(table: Table | None, row_name: str) -> tuple[list[float], bool]:
    """The numbers of the row called ``row_name``, and whether the name is repeated.

    The name is compared with the first cell of every row, the header row included. When
    two rows have the name, the last one is used.
    """
    if table is None:
        raise ExecutionError("table_missing", "a table operation needs a table")
    matches = [row for row in table if len(row) > 0 and str(row[0]).strip() == row_name]
    if not matches:
        raise ExecutionError("row_not_found", f"no table row is named {excerpt(row_name)!r}")
    numbers = [cell_number(cell) for cell in matches[-1][1:]]
    if not numbers or any(number is None for number in numbers):
        raise ExecutionError(
            "row_not_numeric", f"the row {excerpt(row_name)!r} has a cell that is not a number"
        )
    return [number for number in numbers if number is not None], len(matches) > 1


# --------------------------------------------------------------------------- execution


def _operands(step: Step, values: list[Value]) -> tuple[float, float]:
    found: list[float] = []
    for argument in step.arguments:
        if argument.kind == REFERENCE:
            index = argument.value
            if not isinstance(index, int) or not 0 <= index < len(values):
                raise ProgramError("bad_reference", f"{argument.text} is not an earlier step")
            value = values[index]
            if isinstance(value, str):
                raise ExecutionError(
                    "yes_no_operand", f"{argument.text} is {value}, which is not a number"
                )
        else:
            value = argument.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ProgramError(
                    "bad_argument", f"{step.operation} cannot use {excerpt(argument.text)!r}"
                )
        found.append(float(value))
    return found[0], found[1]


def execute(program: Program, table: Table | None = None) -> Execution:
    """Run a parsed program. Raises ``ExecutionError``; see ``errors.CODES`` for the codes."""
    values: list[Value] = []
    notes: list[str] = []
    for step in program.steps:
        if step.operation in TABLE_OPERATIONS:
            numbers, repeated = row_numbers(table, step.arguments[0].text.strip(BLANKS))
            if repeated and DUPLICATE_ROW not in notes:
                notes.append(DUPLICATE_ROW)
            result: Value = _TABLE[step.operation](numbers)
        elif step.operation in _ARITHMETIC:
            result = _ARITHMETIC[step.operation](*_operands(step, values))
        else:
            raise ProgramError("unknown_operation", f"unknown operation: {step.operation!r}")
        if not isinstance(result, str) and not math.isfinite(result):
            raise ExecutionError("numeric_error", f"{step.operation} has no finite result")
        values.append(result if isinstance(result, str) else float(result))
    if not values:
        raise ProgramError("empty_program", "the program has no steps")
    return Execution(value=values[-1], step_values=tuple(values), notes=tuple(notes))
