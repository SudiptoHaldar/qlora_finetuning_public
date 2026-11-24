"""When a reply is right: the comparison rule and the four metrics (req_002 v2.02).

``qlora_finetuning.dsl`` turns a reply into a value or a named error. This module compares
that with the gold reply and gives four true-or-false results per example:

- **format validity**: the reply kept the two-line format exactly;
- **execution accuracy** (primary): the program ran and its value matches the gold result;
- **answer accuracy**: the number or ``yes`` / ``no`` on the answer line matches it;
- **program match**: the program is the gold program, numbers compared by value.

Two numbers match when they differ by at most the larger of an absolute allowance and a
relative allowance times the gold value. The allowances come from ``config/evaluation.yaml``
and were fixed before any model was scored.

Usage::

    from qlora_finetuning.dsl import ReplyFormat, run_reply
    from qlora_finetuning.evaluation.metrics import (
        Gold, load_evaluation_config, score_reply, summarize,
    )

    config = load_evaluation_config()
    labels = ReplyFormat(program_label="PROGRAM:", answer_label="ANSWER:")
    gold = Gold.from_target("PROGRAM: subtract(920, 95), divide(#0, 95)\\nANSWER: 8.68421", labels)

    reply = "PROGRAM: subtract(920.0, 95), divide(#0, 95)\\nANSWER: 8.68"
    score = score_reply(run_reply(reply), gold, config)
    score.execution_correct   # True: 8.684210... matches 8.68421
    score.answer_correct      # True: 8.68 is within 0.5%
    score.program_match       # True: 920.0 and 920 are the same number

    summarize([score])["all"]["execution_accuracy"]   # {"count": 1, "rate": 1.0}

This module does not import the data package, so the data package's gold self-check can use
it. Every rate is a count over all the examples given: a reply with an error is wrong on
what could not be judged, and nothing is left out of a denominator.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from qlora_finetuning.dsl.config import DslConfig, load_dsl_config
from qlora_finetuning.dsl.errors import DslError, ProgramError
from qlora_finetuning.dsl.language import (
    BLANKS,
    NO,
    NONE,
    REFERENCE,
    ROW_NAME,
    TABLE_OPERATIONS,
    YES,
    is_digits,
    read_number,
)
from qlora_finetuning.dsl.outcome import ReplyOutcome
from qlora_finetuning.dsl.program import Argument, Program, parse_program
from qlora_finetuning.dsl.reply import ReplyFormat, parse_reply
from qlora_finetuning.settings import REPO_ROOT, ConfigError, load_config

REQUIRED_KEYS = (
    "execution.absolute",
    "execution.relative",
    "answer.absolute",
    "answer.relative",
    "scale_factors",
    "flags_file",
    "report_dir",
    "results_file",
    "diagnostic.execution_relative",
    "runner.progress_every",
)
METRICS = ("format_validity", "execution_accuracy", "answer_accuracy", "program_match")

# Labels for a wrong numeric result, for error analysis. In the order they are tried.
KIND = "kind"
SIGN = "sign"
SCALE = "scale"
OTHER = "other"
MISMATCHES = (KIND, SIGN, SCALE, OTHER)

_ANSWER_CHARS = 400  # longer text after the answer label is not a number
_EXPONENT_DIGITS = 3
_SAME_NUMBER = 1e-12  # two ways of writing one number may differ in the last bit

Value = float | str


class EvaluationError(ConfigError):
    """A gold reply cannot be used, or an evaluation file is missing."""


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class Allowance:
    """How far a number may be from the gold one: the larger of the two parts applies."""

    absolute: float
    relative: float


@dataclass(frozen=True)
class EvaluationConfig:
    """``config/evaluation.yaml`` with its paths resolved."""

    root: Path
    execution: Allowance
    answer: Allowance
    scale_factors: tuple[float, ...]
    flags_file: Path
    report_dir: Path
    results_file: Path
    diagnostic_relative: float
    progress_every: int

    @property
    def tight_execution(self) -> Allowance:
        """The diagnostic rule for execution accuracy: the same absolute part, a tighter
        relative part. Shown beside the headline; it decides no score and no flag."""
        return Allowance(absolute=self.execution.absolute, relative=self.diagnostic_relative)


def _allowance(raw: dict[str, Any], name: str) -> Allowance:
    parts = []
    for key in ("absolute", "relative"):
        value = raw[name][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ConfigError(
                f"config evaluation.yaml: {name}.{key} must be a number of zero or more, "
                f"got {value!r}"
            )
        if not math.isfinite(value):
            raise ConfigError(f"config evaluation.yaml: {name}.{key} must be finite")
        parts.append(float(value))
    if parts[0] == 0 and parts[1] == 0:
        raise ConfigError(
            f"config evaluation.yaml: {name}.absolute and {name}.relative cannot both be zero"
        )
    return Allowance(absolute=parts[0], relative=parts[1])


def load_evaluation_config(
    root: Path | str = REPO_ROOT, config_dir: Path | str | None = None
) -> EvaluationConfig:
    """Read and validate ``config/evaluation.yaml``. Relative paths resolve under ``root``."""
    raw = load_config("evaluation", required=REQUIRED_KEYS, config_dir=config_dir)
    factors = raw["scale_factors"]
    if not isinstance(factors, list) or any(
        isinstance(f, bool) or not isinstance(f, (int, float)) or f <= 1 for f in factors
    ):
        raise ConfigError(
            "config evaluation.yaml: scale_factors must be a list of numbers above 1"
        )
    tight = raw["diagnostic"]["execution_relative"]
    if isinstance(tight, bool) or not isinstance(tight, (int, float)) or not 0 <= tight < math.inf:
        raise ConfigError(
            "config evaluation.yaml: diagnostic.execution_relative must be a number of zero "
            f"or more, got {tight!r}"
        )
    every = raw["runner"]["progress_every"]
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise ConfigError(
            "config evaluation.yaml: runner.progress_every must be a positive whole number, "
            f"got {every!r}"
        )
    root = Path(root)
    return EvaluationConfig(
        root=root,
        execution=_allowance(raw, "execution"),
        answer=_allowance(raw, "answer"),
        scale_factors=tuple(float(f) for f in factors),
        flags_file=root / str(raw["flags_file"]),
        report_dir=root / str(raw["report_dir"]),
        results_file=root / str(raw["results_file"]),
        diagnostic_relative=float(tight),
        progress_every=every,
    )


@lru_cache(maxsize=1)
def _default_dsl_config() -> DslConfig:
    return load_dsl_config()


# --------------------------------------------------------------------------- comparing


def numbers_match(value: float, gold: float, allowance: Allowance) -> bool:
    """True when ``value`` is within the allowance of ``gold``."""
    return abs(value - gold) <= max(allowance.absolute, allowance.relative * abs(gold))


def values_match(value: Value | None, gold: Value, allowance: Allowance) -> bool:
    """``yes`` / ``no`` match exactly; numbers by the allowance; one never matches the other."""
    if value is None:
        return False
    if isinstance(value, str) or isinstance(gold, str):
        return value == gold
    return numbers_match(value, gold, allowance)


def allowance_used(value: float, gold: float, allowance: Allowance) -> float:
    """The gap as a share of what is allowed: 1.0 is the edge, above it is a mismatch."""
    allowed = max(allowance.absolute, allowance.relative * abs(gold))
    if allowed == 0:  # a relative-only rule and a gold of zero: only zero matches
        return 0.0 if value == gold else math.inf
    return abs(value - gold) / allowed


# --------------------------------------------------------------------------- the answer line


def _unsigned_number(text: str) -> float | None:
    """A number without a sign: digits, an optional fraction, an optional exponent."""
    mantissa, marker, exponent = text.partition("e")
    if not mantissa or mantissa[0] in "+-" or read_number(mantissa) is None:
        return None
    if marker:
        digits = exponent[1:] if exponent[:1] in ("+", "-") else exponent
        if not is_digits(digits) or len(digits) > _EXPONENT_DIGITS:
            return None
    value = float(text)
    return value if math.isfinite(value) else None


def read_answer(text: Any) -> tuple[Value, ...]:
    """What the text after the answer label can mean.

    Returns ``("yes",)`` or ``("no",)``; or one number; or, for an answer ending in ``%``,
    two numbers: the one written and that number divided by 100, because gold results are
    written both ways. Returns ``()`` when the text is neither: words, brackets, two numbers.

    A final full stop and the case of ``yes`` / ``no`` are ignored. A number may carry a
    ``$``, thousands separators, a sign, and an exponent.
    """
    if not isinstance(text, str) or len(text) > _ANSWER_CHARS:
        return ()
    cleaned = text.strip()
    if len(cleaned) > 1 and cleaned.endswith("."):
        cleaned = cleaned[:-1].rstrip()
    lowered = cleaned.lower()
    if lowered in (YES, NO):
        return (lowered,)

    body = lowered.replace("$", "").replace(",", "").strip()
    percent = body.endswith("%")
    if percent:
        body = body[:-1].rstrip()
    sign = 1.0
    if body[:1] in ("+", "-"):
        sign = -1.0 if body[0] == "-" else 1.0
        body = body[1:].lstrip(BLANKS)
    number = _unsigned_number(body)
    if number is None:
        return ()
    number *= sign
    return (number, number / 100.0) if percent else (number,)


# --------------------------------------------------------------------------- the gold reply


@dataclass(frozen=True)
class Gold:
    """The reply an example is scored against.

    ``value`` is the gold result as it stands on the answer line. ``program`` is ``None``
    when the gold program itself does not parse; no reply can then be a program match.
    """

    program: Program | None
    value: Value
    program_text: str
    answer_text: str

    @classmethod
    def from_target(
        cls, target: str, reply_format: ReplyFormat, dsl_config: DslConfig | None = None
    ) -> Gold:
        """Read a canonical record's target text. Raises ``EvaluationError`` if it cannot be."""
        dsl_config = dsl_config or _default_dsl_config()
        try:
            parsed = parse_reply(target, reply_format, dsl_config)
        except DslError as error:
            raise EvaluationError(f"the gold reply cannot be read: {error}") from error
        readings = read_answer(parsed.answer_text)
        if len(readings) != 1:
            raise EvaluationError("the gold answer is neither a plain number nor yes/no")
        try:
            program = parse_program(parsed.program_text, dsl_config)
        except ProgramError:
            program = None
        return cls(
            program=program,
            value=readings[0],
            program_text=parsed.program_text,
            answer_text=parsed.answer_text,
        )


# --------------------------------------------------------------------------- program match


def _argument_key(argument: Argument) -> tuple[str, Any]:
    if argument.kind == REFERENCE:
        return REFERENCE, argument.value
    if argument.kind == ROW_NAME:
        return ROW_NAME, argument.text
    if argument.kind == NONE:
        return NONE, None
    return "number", float(argument.value)  # integer, decimal, percent, constant


def _same_argument(first: Argument, second: Argument) -> bool:
    kind, value = _argument_key(first)
    other_kind, other_value = _argument_key(second)
    if kind != other_kind:
        return False
    if kind == "number":
        return math.isclose(value, other_value, rel_tol=_SAME_NUMBER, abs_tol=0.0)
    return value == other_value


def programs_match(first: Program | None, second: Program | None) -> bool:
    """True when two programs have the same steps, numbers compared by value.

    ``4850.0`` and ``4850``, ``.35`` and ``0.35``, ``const_100`` and ``100``, ``6.9%`` and
    ``0.069`` are the same argument. The second argument of a table step is ignored. The
    order of arguments and of steps matters: ``add(a, b)`` is not ``add(b, a)`` here; an
    equivalent program is credited by execution accuracy instead.
    """
    if first is None or second is None or len(first.steps) != len(second.steps):
        return False
    for step, other in zip(first.steps, second.steps, strict=True):
        if step.operation != other.operation:
            return False
        compared = 1 if step.operation in TABLE_OPERATIONS else 2
        for index in range(compared):
            if not _same_argument(step.arguments[index], other.arguments[index]):
                return False
    return True


# --------------------------------------------------------------------------- scoring


def mismatch_label(value: Value | None, gold: Value, config: EvaluationConfig) -> str | None:
    """Why a result is wrong, in one word, or ``None`` when it is right or absent.

    ``kind``: a number where ``yes`` / ``no`` was due, or the reverse. ``sign``: the negative
    of the gold value. ``scale``: the gold value times or divided by a configured factor.
    ``other``: anything else. A label never makes a result right.
    """
    if value is None or values_match(value, gold, config.execution):
        return None
    if isinstance(value, str) != isinstance(gold, str):
        return KIND
    if isinstance(value, str) or isinstance(gold, str):
        return OTHER  # yes where no was due
    if value != 0 and gold != 0:
        if numbers_match(-value, gold, config.execution):
            return SIGN
        for factor in config.scale_factors:
            if numbers_match(value * factor, gold, config.execution) or numbers_match(
                value / factor, gold, config.execution
            ):
                return SCALE
    return OTHER


@dataclass(frozen=True)
class ExampleScore:
    """The four results for one reply, with what explains a wrong one."""

    example_id: str | None
    format_valid: bool
    executed: bool
    execution_correct: bool
    answer_correct: bool
    program_match: bool
    error_code: str | None = None
    deviations: tuple[str, ...] = ()
    mismatch: str | None = None
    relative_gap: float | None = None

    @property
    def all_correct(self) -> bool:
        return (
            self.format_valid
            and self.execution_correct
            and self.answer_correct
            and self.program_match
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain values, ready for a JSON Lines record."""
        result = asdict(self)
        result["deviations"] = list(self.deviations)
        return result


def score_reply(
    outcome: ReplyOutcome,
    gold: Gold,
    config: EvaluationConfig,
    example_id: str | None = None,
    dsl_config: DslConfig | None = None,
) -> ExampleScore:
    """Score one reply, already run by ``run_reply``, against its gold reply."""
    executed = outcome.ok
    execution_correct = executed and values_match(outcome.value, gold.value, config.execution)
    answer_correct = any(
        values_match(reading, gold.value, config.answer)
        for reading in read_answer(outcome.answer_text or "")
    )

    predicted = None
    if outcome.program is not None:
        try:
            predicted = parse_program(outcome.program, dsl_config or _default_dsl_config())
        except ProgramError:
            predicted = None

    gap = None
    if (
        executed
        and not isinstance(outcome.value, str)
        and not isinstance(gold.value, str)
        and gold.value != 0
    ):
        gap = abs(outcome.value - gold.value) / abs(gold.value)

    return ExampleScore(
        example_id=example_id,
        format_valid=outcome.strict,
        executed=executed,
        execution_correct=execution_correct,
        answer_correct=answer_correct,
        program_match=programs_match(predicted, gold.program),
        error_code=outcome.error_code,
        deviations=tuple(outcome.deviations),
        mismatch=mismatch_label(outcome.value, gold.value, config) if executed else None,
        relative_gap=gap,
    )


def answer_agrees(outcome: ReplyOutcome, config: EvaluationConfig) -> bool | None:
    """Whether a reply's answer line says what its own program gives (req_006 v6.01).

    The answer line is read by ``read_answer`` and compared with the executed value under
    the ``answer`` allowance, the executed value standing where the gold value stands in
    ``score_reply``. ``None`` means there is nothing to compare: the program did not run,
    or the answer line holds neither a number nor ``yes`` / ``no``.

    It is for showing one reply (the demo). It is not a metric: no gold value is involved
    and no run summary reports it.
    """
    if not outcome.ok or outcome.value is None:
        return None
    readings = read_answer(outcome.answer_text or "")
    if not readings:
        return None
    return any(values_match(reading, outcome.value, config.answer) for reading in readings)


# --------------------------------------------------------------------------- totals


def _rate(count: int, total: int) -> dict[str, Any]:
    return {"count": count, "rate": count / total if total else None}


def _sorted_counts(counter: Counter) -> dict[str, int]:
    return {str(k): v for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))}


def _block(scores: Sequence[ExampleScore]) -> dict[str, Any]:
    total = len(scores)
    return {
        "examples": total,
        "format_validity": _rate(sum(s.format_valid for s in scores), total),
        "execution_accuracy": _rate(sum(s.execution_correct for s in scores), total),
        "answer_accuracy": _rate(sum(s.answer_correct for s in scores), total),
        "program_match": _rate(sum(s.program_match for s in scores), total),
        "executed": _rate(sum(s.executed for s in scores), total),
        "error_codes": _sorted_counts(Counter(s.error_code for s in scores if s.error_code)),
        "deviations": _sorted_counts(Counter(d for s in scores for d in s.deviations)),
        "mismatches": _sorted_counts(Counter(s.mismatch for s in scores if s.mismatch)),
    }


def summarize(
    scores: Iterable[ExampleScore], flagged: Collection[str] = ()
) -> dict[str, Any]:
    """Totals over ``scores``: once over all of them, once over the unflagged ones.

    ``flagged`` holds the ids of examples whose gold program does not reproduce its gold
    answer (``load_gold_flags``). They stay in ``"all"``, which is the headline, and are left
    out of ``"unflagged"``, which is there to read beside it. A rate is its count divided by
    the number of examples in the block, and ``None`` for an empty block.
    """
    scores = list(scores)
    flagged = frozenset(flagged)
    unflagged = [score for score in scores if score.example_id not in flagged]
    return {
        "all": _block(scores),
        "unflagged": _block(unflagged),
        "flagged_examples": len(scores) - len(unflagged),
    }
