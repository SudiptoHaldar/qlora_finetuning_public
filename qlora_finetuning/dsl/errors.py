"""Named errors for everything that can be wrong with a model's reply (req_002 v2.01).

Every failure caused by reply text is a ``DslError`` with a stable ``code`` and the ``stage``
it belongs to. The evaluation runner stores the code per example, so codes are never renamed
once a model has been scored.

Usage::

    from qlora_finetuning.dsl.errors import DslError, ProgramError

    try:
        parse_program("add(1)", config)
    except DslError as error:
        error.code      # "wrong_arity"
        error.stage     # "program"
        error.position  # where in the text, when known
"""

from __future__ import annotations

REPLY = "reply"
PROGRAM = "program"
EXECUTION = "execution"

# Every code and the stage that raises it.
CODES: dict[str, str] = {
    "reply_too_long": REPLY,
    "reply_empty": REPLY,
    "no_program": REPLY,
    "empty_program": PROGRAM,
    "too_many_steps": PROGRAM,
    "syntax": PROGRAM,
    "unknown_operation": PROGRAM,
    "nested_call": PROGRAM,
    "wrong_arity": PROGRAM,
    "bad_argument": PROGRAM,
    "unknown_constant": PROGRAM,
    "bad_reference": PROGRAM,
    "bad_row_name": PROGRAM,
    "division_by_zero": EXECUTION,
    "numeric_error": EXECUTION,
    "yes_no_operand": EXECUTION,
    "table_missing": EXECUTION,
    "row_not_found": EXECUTION,
    "row_not_numeric": EXECUTION,
}

EXCERPT_CHARS = 40


def excerpt(text: str, position: int = 0) -> str:
    """At most 40 characters of ``text`` from ``position``, safe to print on any console."""
    piece = text[max(position, 0) : max(position, 0) + EXCERPT_CHARS]
    return piece.encode("ascii", "backslashreplace").decode("ascii")


class DslError(Exception):
    """A reply, a program, or its execution is not acceptable. ``code`` says why."""

    stage = ""

    def __init__(self, code: str, message: str, position: int | None = None) -> None:
        if CODES.get(code) != self.stage:
            raise ValueError(f"{code!r} is not a {self.stage or 'known'} error code")
        super().__init__(message)
        self.code = code
        self.position = position


class ReplyError(DslError):
    """The reply as a whole cannot be read."""

    stage = REPLY


class ProgramError(DslError):
    """The program text is not a valid program."""

    stage = PROGRAM


class ExecutionError(DslError):
    """A valid program cannot be executed."""

    stage = EXECUTION
