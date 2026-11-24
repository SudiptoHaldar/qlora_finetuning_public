"""One call for scoring a reply: parse it, parse its program, execute it (req_002 v2.01).

``run_reply`` never raises on any string. Whatever is wrong with a reply comes back as an
error code in the outcome, so a runner can score thousands of replies without a ``try``.

Usage::

    from qlora_finetuning.dsl import run_reply

    table = [["", "2024", "2023"], ["revenue", "$ 920", "$ 95"]]
    outcome = run_reply("PROGRAM: subtract(920, 95), divide(#0, 95)\\nANSWER: 8.68421", table)
    outcome.ok           # True
    outcome.strict       # True: the reply kept the two-line format exactly
    outcome.value        # 8.684210526..., not rounded
    outcome.program      # "subtract(920, 95), divide(#0, 95)"

    run_reply("PROGRAM: divide(5, 0)\\nANSWER: 0", table).error_code   # "division_by_zero"
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from qlora_finetuning.dsl.config import DslConfig, load_dsl_config
from qlora_finetuning.dsl.errors import DslError
from qlora_finetuning.dsl.executor import Table, Value, execute
from qlora_finetuning.dsl.program import format_program, parse_program
from qlora_finetuning.dsl.reply import ReplyFormat, parse_reply


@dataclass(frozen=True)
class ReplyOutcome:
    """Everything known about one reply after trying to run it.

    ``program_text`` and ``answer_text`` are what the reply parser found. ``program`` is the
    canonical form, set once the program parsed. ``value`` is set once it executed. When
    something failed, the three ``error_`` fields say what and at which stage.
    """

    strict: bool = False
    deviations: tuple[str, ...] = ()
    program_text: str | None = None
    answer_text: str | None = None
    program: str | None = None
    value: Value | None = None
    step_values: tuple[Value, ...] = ()
    notes: tuple[str, ...] = ()
    error_code: str | None = None
    error_stage: str | None = None
    error_message: str | None = None

    @property
    def ok(self) -> bool:
        """True when the program executed."""
        return self.error_code is None

    def to_dict(self) -> dict[str, Any]:
        """Plain values, ready for a JSON Lines record."""
        result = asdict(self)
        for key in ("deviations", "step_values", "notes"):
            result[key] = list(result[key])
        return result


@lru_cache(maxsize=1)
def _default_config() -> DslConfig:
    return load_dsl_config()


@lru_cache(maxsize=1)
def _default_format() -> ReplyFormat:
    # Imported here: the package must load, and run on a given format, without the data package.
    from qlora_finetuning.data.serialization import load_prompt

    return ReplyFormat.from_prompt(load_prompt())


def run_reply(
    text: str,
    table: Table | None = None,
    reply_format: ReplyFormat | None = None,
    config: DslConfig | None = None,
) -> ReplyOutcome:
    """Parse a reply, parse its program, and execute it on ``table``.

    Without ``reply_format`` the labels of ``config/prompt.yaml`` are used; without
    ``config``, the limits of ``config/dsl.yaml``. Both are read once per process.
    """
    if not isinstance(text, str):
        raise TypeError("a reply must be a string")
    config = config or _default_config()
    reply_format = reply_format or _default_format()

    found: dict[str, Any] = {}
    try:
        parsed = parse_reply(text, reply_format, config)
        found.update(
            strict=parsed.strict,
            deviations=parsed.deviations,
            program_text=parsed.program_text,
            answer_text=parsed.answer_text,
        )
        program = parse_program(parsed.program_text, config)
        found["program"] = format_program(program)
        execution = execute(program, table)
        found.update(
            value=execution.value, step_values=execution.step_values, notes=execution.notes
        )
    except DslError as error:
        found.update(error_code=error.code, error_stage=error.stage, error_message=str(error))
    return ReplyOutcome(**found)
