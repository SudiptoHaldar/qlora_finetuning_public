"""A demo result as text for a person, or as JSON for a program.

Both hold the same values. Both are ASCII: any other character is written as a backslash
escape, because the Windows console cannot print everything a report may contain. A value
is shown as it is: no rounding, no percent sign, no unit.

Usage::

    from qlora_finetuning.demo.render import as_json, as_text

    print(as_text(result))
    print(as_text(result, show_input=True))   # with the context and the table
    print(as_json(result))      # one line; json.loads gives the outcome's fields
"""

from __future__ import annotations

import json
from typing import Any

from qlora_finetuning.demo.answer import DemoResult

WIDTH = 12  # the label column
STOPPED = {
    "end": "stopped at the end token",
    "length": "stopped at the reply limit",
    "other": "stopped for another reason",
}


def ascii_text(text: Any) -> str:
    """``text`` on one line, with every character outside ASCII written as an escape."""
    line = str(text).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    return line.encode("ascii", "backslashreplace").decode("ascii")


def show_value(value: Any) -> str:
    """A number exactly as Python holds it; ``yes`` and ``no`` as they are."""
    return value if isinstance(value, str) else repr(value)


def warnings(result: DemoResult) -> list[str]:
    """What a reader should know before trusting the reply."""
    found = []
    if result.stop_reason == "length":
        found.append(
            f"the reply was cut off at the limit of {result.max_new_tokens} tokens"
        )
    if result.over_length:
        found.append(
            f"the prompt has {result.input_tokens} tokens; with the reply limit of "
            f"{result.max_new_tokens} that is over the training length of "
            f"{result.max_length}, and the adapter saw no sequence that long"
        )
    return found


def _agreement(result: DemoResult) -> str:
    if not result.outcome.ok:
        return "not checked: the program did not run"
    if result.agrees is None:
        return "not comparable: the answer line is not a number, yes, or no"
    if result.agrees:
        return "agrees with the executed result"
    return "does NOT agree with the executed result"


def _input_rows(result: DemoResult) -> list[tuple[str, str]]:
    """The context and the table the model was given, a line each, labelled once."""
    passages = list(result.demo_input.passages) or ["(none)"]
    table = [" | ".join(str(cell) for cell in row) for row in result.demo_input.table]
    rows = [("Context" if not number else "", text) for number, text in enumerate(passages)]
    return rows + [("Table" if not number else "", text) for number, text in enumerate(table)]


def as_text(result: DemoResult, *, show_input: bool = False) -> str:
    """The result as labelled lines; with ``show_input``, the context and table too."""
    outcome = result.outcome
    rows: list[tuple[str, str]] = [
        ("Input", result.demo_input.source),
        ("Model", result.predictor + (f"  {result.adapter}" if result.adapter else "  (untuned)")),
    ]
    if show_input:
        rows += _input_rows(result)
    rows.append(("Question", result.demo_input.question))
    program = outcome.program if outcome.program is not None else outcome.program_text
    rows.append(("Program", program if program else "none found in the reply"))
    if outcome.ok:
        steps = "   ".join(
            f"#{number} = {show_value(value)}" for number, value in enumerate(outcome.step_values)
        )
        rows.append(("Steps", steps))
        rows.append(("Executed", show_value(outcome.value)))
    else:
        rows.append(
            (
                "Executed",
                f"nothing: {outcome.error_code} ({outcome.error_stage}) - {outcome.error_message}",
            )
        )
    said = outcome.answer_text if outcome.answer_text else "nothing"
    rows.append(("Model said", f"{said}   {_agreement(result)}"))
    if outcome.strict:
        rows.append(("Format", "exact"))
    else:
        broken = ", ".join(outcome.deviations) or "the two lines were not found"
        rows.append(("Format", f"not exact: {broken}"))
    if outcome.notes:
        rows.append(("Notes", ", ".join(outcome.notes)))
    if not outcome.ok or not outcome.strict:
        rows.append(("Reply", result.reply if result.reply else "(empty)"))
    prompt_tokens = "unknown" if result.input_tokens is None else str(result.input_tokens)
    reply_tokens = "unknown" if result.output_tokens is None else str(result.output_tokens)
    stopped = STOPPED.get(result.stop_reason, result.stop_reason)
    rows.append(("Tokens", f"prompt {prompt_tokens}, reply {reply_tokens}, {stopped}"))
    rows.append(("Time", f"{result.seconds:.1f} s"))
    rows += [("Warning", text) for text in warnings(result)]
    return "\n".join(f"{label:<{WIDTH}}{ascii_text(text)}" for label, text in rows)


def as_dict(result: DemoResult, *, show_input: bool = False) -> dict[str, Any]:
    """Plain values: the input's name and question, the reply, and the whole outcome.

    With ``show_input``, the context and the table are added under those two names.
    """
    shown: dict[str, Any] = {}
    if show_input:
        shown = {
            "context": list(result.demo_input.passages),
            "table": [list(row) for row in result.demo_input.table],
        }
    return {
        **shown,
        "input": result.demo_input.source,
        "question": result.demo_input.question,
        "predictor": result.predictor,
        "adapter": result.adapter,
        "reply": result.reply,
        "outcome": result.outcome.to_dict(),
        "agrees": result.agrees,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "stop_reason": result.stop_reason,
        "seconds": result.seconds,
        "warnings": warnings(result),
    }


def as_json(result: DemoResult, *, show_input: bool = False) -> str:
    """The result as one line of ASCII JSON."""
    return json.dumps(
        as_dict(result, show_input=show_input), ensure_ascii=True, allow_nan=False
    )
