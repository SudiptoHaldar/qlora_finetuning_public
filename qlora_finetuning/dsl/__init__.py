"""Program parser and safe executor (req_002 v2.01).

A model replies with a short program and an answer. This package reads the reply and runs
the program; it is the one place where reply text is parsed or executed, for scoring and for
the demo alike. No reply text is ever evaluated as Python.

- ``reply``: the reply parser. Finds the program and the answer; says whether the two-line
  format was kept exactly.
- ``program``: the program parser. Ten operations, seventeen constants, a canonical form.
- ``executor``: runs the steps on floats, reading table rows by name. Does not round.
- ``outcome``: ``run_reply``, the three joined; never raises on any string.
- ``errors``: every failure has a code and a stage.

Usage::

    from qlora_finetuning.dsl import run_reply

    table = [["", "2024", "2023"], ["revenue", "$ 920", "$ 95"], ["margin", "12.5%", "10%"]]
    outcome = run_reply("PROGRAM: table_average(margin, none)\\nANSWER: 0.1125", table)
    outcome.value        # 0.1125
    outcome.error_code   # None
"""

from qlora_finetuning.dsl.config import DslConfig, load_dsl_config
from qlora_finetuning.dsl.errors import (
    CODES,
    DslError,
    ExecutionError,
    ProgramError,
    ReplyError,
)
from qlora_finetuning.dsl.executor import Execution, cell_number, execute, row_numbers
from qlora_finetuning.dsl.language import ARGUMENT_KINDS, CONSTANTS, OPERATIONS
from qlora_finetuning.dsl.outcome import ReplyOutcome, run_reply
from qlora_finetuning.dsl.program import Argument, Program, Step, format_program, parse_program
from qlora_finetuning.dsl.reply import DEVIATIONS, ParsedReply, ReplyFormat, parse_reply

__all__ = [
    "ARGUMENT_KINDS",
    "CODES",
    "CONSTANTS",
    "DEVIATIONS",
    "OPERATIONS",
    "Argument",
    "DslConfig",
    "DslError",
    "Execution",
    "ExecutionError",
    "ParsedReply",
    "Program",
    "ProgramError",
    "ReplyError",
    "ReplyFormat",
    "ReplyOutcome",
    "Step",
    "cell_number",
    "execute",
    "format_program",
    "load_dsl_config",
    "parse_program",
    "parse_reply",
    "row_numbers",
    "run_reply",
]
