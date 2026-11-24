"""The fixed vocabulary of the program language (req_002 v2.01).

Ten operations, seventeen constants, seven argument kinds. This is not configuration:
changing an entry changes what "correct" means for every model. A test compares the table
with ``config/data.yaml`` and with the system text of ``config/prompt.yaml``.

Usage::

    from qlora_finetuning.dsl.language import CONSTANTS, OPERATIONS, read_number

    "table_sum" in OPERATIONS    # True
    CONSTANTS["const_m1"]        # -1.0
    read_number("-4474.2")       # -4474.2
    read_number("1e5")           # None: not a form the language allows
"""

from __future__ import annotations

import math

ARITHMETIC_OPERATIONS = ("add", "subtract", "multiply", "divide", "exp", "greater")
TABLE_OPERATIONS = ("table_sum", "table_average", "table_max", "table_min")
OPERATIONS = ARITHMETIC_OPERATIONS + TABLE_OPERATIONS

CONSTANTS: dict[str, float] = {
    "const_1": 1.0,
    "const_2": 2.0,
    "const_3": 3.0,
    "const_4": 4.0,
    "const_5": 5.0,
    "const_6": 6.0,
    "const_7": 7.0,
    "const_8": 8.0,
    "const_9": 9.0,
    "const_10": 10.0,
    "const_100": 100.0,
    "const_1000": 1000.0,
    "const_10000": 10000.0,
    "const_100000": 100000.0,
    "const_1000000": 1000000.0,
    "const_1000000000": 1000000000.0,
    "const_m1": -1.0,
}

INTEGER = "integer"
DECIMAL = "decimal"
PERCENT = "percent"
REFERENCE = "reference"
CONSTANT = "constant"
ROW_NAME = "row_name"
NONE = "none"
ARGUMENT_KINDS = (INTEGER, DECIMAL, PERCENT, REFERENCE, CONSTANT, ROW_NAME, NONE)

YES = "yes"
NO = "no"

DIGITS = "0123456789"
BLANKS = " \t"


def is_digits(text: str) -> bool:
    """One or more ASCII digits. ``str.isdigit`` would also accept other scripts' digits."""
    return bool(text) and all(character in DIGITS for character in text)


def read_number(text: str) -> float | None:
    """The value of a plain number, or ``None`` when ``text`` is not one.

    A plain number is an optional minus sign, then digits with an optional fraction
    (``920``, ``4474.2``) or a fraction that starts with the point (``.35``). No plus sign,
    exponent, thousands separator, or space. The text is checked character by character
    before ``float`` sees it, because ``float`` alone would also accept ``inf``, ``nan``,
    ``1e5``, and ``1_000``.
    """
    body = text[1:] if text.startswith("-") else text
    whole, point, fraction = body.partition(".")
    if point:
        if not is_digits(fraction) or (whole and not is_digits(whole)):
            return None
    elif not is_digits(whole):
        return None
    value = float(text)
    return value if math.isfinite(value) else None
