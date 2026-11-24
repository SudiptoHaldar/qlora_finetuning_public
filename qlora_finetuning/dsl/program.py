"""The program parser: program text in, a checked list of steps out (req_002 v2.01).

A program is steps separated by commas. A step is ``name(first, second)``. Spaces and tabs
are allowed around every name, bracket, comma, and argument, and nowhere inside one::

    subtract(920, 95), divide(#0, 95)
    table_average(net income ( loss ), none)

The text is read once, left to right, by hand. Nothing in it is evaluated: a name is looked
up among ten operations, an argument is matched against the few forms the language has, and
everything else is a ``ProgramError`` with a code.

Usage::

    from qlora_finetuning.dsl.config import load_dsl_config
    from qlora_finetuning.dsl.program import format_program, parse_program

    program = parse_program("add( 6.9%,const_1 ),multiply(#0, 200)", load_dsl_config())
    program.steps[0].arguments[0].kind     # "percent"
    program.steps[0].arguments[0].value    # 0.069
    format_program(program)                # "add(6.9%, const_1), multiply(#0, 200)"
"""

from __future__ import annotations

from dataclasses import dataclass

from qlora_finetuning.dsl.config import DslConfig
from qlora_finetuning.dsl.errors import ProgramError, excerpt
from qlora_finetuning.dsl.language import (
    BLANKS,
    CONSTANT,
    CONSTANTS,
    DECIMAL,
    INTEGER,
    NONE,
    OPERATIONS,
    PERCENT,
    REFERENCE,
    ROW_NAME,
    TABLE_OPERATIONS,
    is_digits,
    read_number,
)

# A reference with more digits than this cannot name a step and is not converted.
_REFERENCE_DIGITS = 6


@dataclass(frozen=True)
class Argument:
    """One argument of a step.

    ``text`` is the argument as written, without surrounding blanks. ``value`` is the number
    it stands for (a percent literal already divided by 100), the step number of a reference,
    or ``None`` for a row name and for ``none``.
    """

    kind: str
    text: str
    value: float | int | None = None


@dataclass(frozen=True)
class Step:
    operation: str
    arguments: tuple[Argument, Argument]


@dataclass(frozen=True)
class Program:
    steps: tuple[Step, ...]


def _argument(text: str, step_index: int, position: int) -> Argument:
    """Read one argument of an arithmetic step, or the second argument of a table step."""
    if text.startswith("#"):
        digits = text[1:]
        if not is_digits(digits) or len(digits.lstrip("0")) > _REFERENCE_DIGITS:
            raise ProgramError(
                "bad_reference", f"not a step reference: {excerpt(text)}", position
            )
        number = int(digits.lstrip("0") or "0")  # int() refuses very long digit strings
        if number >= step_index:
            raise ProgramError(
                "bad_reference",
                f"step {step_index} refers to #{number}, which is not an earlier step",
                position,
            )
        return Argument(REFERENCE, text, number)
    if text.startswith("const_"):
        if text not in CONSTANTS:
            raise ProgramError("unknown_constant", f"unknown constant: {excerpt(text)}", position)
        return Argument(CONSTANT, text, CONSTANTS[text])
    if text.endswith("%"):
        number = read_number(text[:-1])
        if number is not None:
            return Argument(PERCENT, text, number / 100.0)
    else:
        number = read_number(text)
        if number is not None:
            return Argument(DECIMAL if "." in text else INTEGER, text, number)
    raise ProgramError(
        "bad_argument", f"not a number, a reference, or a constant: {excerpt(text)!r}", position
    )


def _arithmetic_step(text: str, name: str, opened: int, index: int) -> tuple[Step, int]:
    """Read the arguments of an arithmetic step; return it and the position after ``)``."""
    closed = text.find(")", opened + 1)
    nested = text.find("(", opened + 1)
    if nested != -1 and (closed == -1 or nested < closed):
        raise ProgramError("nested_call", f"a bracket inside the arguments of {name}", nested)
    if closed == -1:
        raise ProgramError("syntax", f"{name} has no closing bracket", opened)
    parts = text[opened + 1 : closed].split(",")
    if len(parts) != 2:
        raise ProgramError(
            "wrong_arity", f"{name} takes two arguments, found {len(parts)}", opened
        )
    first, second = (_argument(part.strip(BLANKS), index, opened) for part in parts)
    return Step(name, (first, second)), closed + 1


def _table_step(text: str, name: str, opened: int, index: int) -> tuple[Step, int]:
    """Read a table step. The row name runs to the last comma outside inner brackets."""
    depth = 1
    last_comma = -1
    closed = -1
    for position in range(opened + 1, len(text)):
        character = text[position]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                closed = position
                break
        elif character == "," and depth == 1:
            last_comma = position
    if closed == -1:
        raise ProgramError("syntax", f"{name} has no closing bracket", opened)
    if last_comma == -1:
        raise ProgramError("wrong_arity", f"{name} takes two arguments, found 1", opened)
    row_name = text[opened + 1 : last_comma].strip(BLANKS)
    if not row_name:
        raise ProgramError("bad_row_name", f"{name} names no row", opened)
    second_text = text[last_comma + 1 : closed].strip(BLANKS)
    if second_text == NONE:
        second = Argument(NONE, NONE)
    else:
        second = _argument(second_text, index, last_comma)
    return Step(name, (Argument(ROW_NAME, row_name), second)), closed + 1


def _skip_blanks(text: str, position: int) -> int:
    while position < len(text) and text[position] in BLANKS:
        position += 1
    return position


def parse_program(text: str, config: DslConfig) -> Program:
    """Parse program text. Raises ``ProgramError``; see ``errors.CODES`` for the codes."""
    if not isinstance(text, str):
        raise TypeError("a program must be a string")
    if not text.strip():
        raise ProgramError("empty_program", "the program is empty")

    steps: list[Step] = []
    position = _skip_blanks(text, 0)
    while True:
        opened = text.find("(", position)
        if opened == -1:
            raise ProgramError(
                "syntax", f"expected a step, found: {excerpt(text, position)!r}", position
            )
        name = text[position:opened].strip(BLANKS)
        if name not in OPERATIONS:
            if not name or "," in name or ")" in name:
                raise ProgramError(
                    "syntax", f"expected a step, found: {excerpt(text, position)!r}", position
                )
            raise ProgramError(
                "unknown_operation", f"unknown operation: {excerpt(name)!r}", position
            )
        if len(steps) >= config.max_steps:
            raise ProgramError(
                "too_many_steps", f"the program has more than {config.max_steps} steps", position
            )
        read = _table_step if name in TABLE_OPERATIONS else _arithmetic_step
        step, position = read(text, name, opened, len(steps))
        steps.append(step)

        position = _skip_blanks(text, position)
        if position == len(text):
            return Program(tuple(steps))
        if text[position] != ",":
            raise ProgramError(
                "syntax", f"unexpected text after a step: {excerpt(text, position)!r}", position
            )
        position = _skip_blanks(text, position + 1)
        if position == len(text):
            raise ProgramError("syntax", "the program ends with a comma", position)


def format_program(program: Program) -> str:
    """The canonical form: ``op(a, b), op(a, b)``, with each argument as it was written."""
    return ", ".join(
        f"{step.operation}({step.arguments[0].text}, {step.arguments[1].text})"
        for step in program.steps
    )
