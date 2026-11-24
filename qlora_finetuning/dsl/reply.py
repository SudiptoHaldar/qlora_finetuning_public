"""The reply parser: a model's reply in, the program text and the answer text out.

A reply is **strict** when it holds exactly the two fields the prompt asks for::

    PROGRAM: subtract(920, 95), divide(#0, 95)
    ANSWER: 8.68421

Whitespace around the reply, blank lines, spaces or tabs at either end of a line, and
``\\r\\n`` line ends do not break strictness. Anything else does, and is named in
``deviations``. A reply that is not strict is still read when it has a program line: the
program is taken from the first line that starts with the program label, the answer from the
first later line that starts with the answer label, and other lines are ignored.

The labels come from ``config/prompt.yaml``; none is written here. The parser knows nothing
about any model: a backend removes whatever is specific to its model before calling it.

Usage::

    from qlora_finetuning.data import load_prompt
    from qlora_finetuning.dsl.config import load_dsl_config
    from qlora_finetuning.dsl.reply import ReplyFormat, parse_reply

    reply_format = ReplyFormat.from_prompt(load_prompt())
    parsed = parse_reply("Sure!\\nprogram: add(1, 2)", reply_format, load_dsl_config())
    parsed.program_text   # "add(1, 2)"
    parsed.strict         # False
    parsed.deviations     # ("text_before", "label_case", "answer_missing")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qlora_finetuning.dsl.config import DslConfig
from qlora_finetuning.dsl.errors import ReplyError
from qlora_finetuning.dsl.language import BLANKS

TEXT_BEFORE = "text_before"
TEXT_BETWEEN = "text_between"
TEXT_AFTER = "text_after"
LABEL_CASE = "label_case"
ANSWER_MISSING = "answer_missing"
# In the order they are reported.
DEVIATIONS = (TEXT_BEFORE, TEXT_BETWEEN, TEXT_AFTER, LABEL_CASE, ANSWER_MISSING)


@dataclass(frozen=True)
class ReplyFormat:
    """The two labels a reply is expected to carry."""

    program_label: str
    answer_label: str

    def __post_init__(self) -> None:
        for label in (self.program_label, self.answer_label):
            if not isinstance(label, str) or not label.strip() or "\n" in label:
                raise ValueError("a reply label must be one line of text")

    @classmethod
    def from_prompt(cls, prompt: Any) -> ReplyFormat:
        """The labels of a ``PromptDefinition`` (``config/prompt.yaml``)."""
        return cls(program_label=prompt.program_label, answer_label=prompt.answer_label)


@dataclass(frozen=True)
class ParsedReply:
    """What was found in a reply. ``answer_text`` is empty when there is no answer."""

    program_text: str
    answer_text: str
    strict: bool
    deviations: tuple[str, ...]


def _after_label(line: str, label: str) -> tuple[str, bool] | None:
    """The text after ``label`` and whether the case matched, or ``None`` if it is not there."""
    head = line[: len(label)]
    if head == label:
        return line[len(label) :].strip(BLANKS), True
    if head.lower() == label.lower():
        return line[len(label) :].strip(BLANKS), False
    return None


def parse_reply(text: str, reply_format: ReplyFormat, config: DslConfig) -> ParsedReply:
    """Find the program and the answer in a reply.

    Raises ``ReplyError`` with code ``reply_too_long``, ``reply_empty``, or ``no_program``.
    """
    if not isinstance(text, str):
        raise TypeError("a reply must be a string")
    if len(text) > config.max_reply_chars:
        raise ReplyError(
            "reply_too_long",
            f"the reply has {len(text):,} characters; the limit is {config.max_reply_chars:,}",
        )
    lines = [
        line.strip(BLANKS) for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    lines = [line for line in lines if line.strip()]
    if not lines:
        raise ReplyError("reply_empty", "the reply is empty")

    program_at = answer_at = None
    program = answer = None
    for index, line in enumerate(lines):
        if program is None:
            program = _after_label(line, reply_format.program_label)
            if program is not None:
                program_at = index
        elif answer is None:
            answer = _after_label(line, reply_format.answer_label)
            if answer is not None:
                answer_at = index
                break
    if program is None or program_at is None:
        raise ReplyError(
            "no_program", f"no line of the reply starts with {reply_format.program_label}"
        )

    found = set()
    if program_at > 0:
        found.add(TEXT_BEFORE)
    if answer is None or answer_at is None:
        if program_at < len(lines) - 1:
            found.add(TEXT_AFTER)
        found.add(ANSWER_MISSING)
    else:
        if answer_at > program_at + 1:
            found.add(TEXT_BETWEEN)
        if answer_at < len(lines) - 1:
            found.add(TEXT_AFTER)
        if not answer[0]:
            found.add(ANSWER_MISSING)
    if not program[1] or (answer is not None and not answer[1]):
        found.add(LABEL_CASE)

    deviations = tuple(name for name in DEVIATIONS if name in found)
    return ParsedReply(
        program_text=program[0],
        answer_text=answer[0] if answer is not None else "",
        strict=not deviations,
        deviations=deviations,
    )
