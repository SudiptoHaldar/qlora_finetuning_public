"""The language table and the limits (req_002 v2.01) against the project's config files."""

from __future__ import annotations

import pytest

from qlora_finetuning.data import load_prompt
from qlora_finetuning.dsl import (
    ARGUMENT_KINDS,
    CODES,
    CONSTANTS,
    OPERATIONS,
    DslConfig,
    DslError,
    ExecutionError,
    ProgramError,
    ReplyError,
    ReplyFormat,
    load_dsl_config,
)
from qlora_finetuning.dsl.errors import excerpt
from qlora_finetuning.dsl.language import read_number
from qlora_finetuning.settings import ConfigError, load_config


@pytest.fixture(scope="module")
def expected() -> dict:
    return load_config("data")["finqa"]["expected"]


def test_operations_are_the_ones_the_dataset_uses(expected):
    assert list(OPERATIONS) == list(expected["operations"])
    assert len(OPERATIONS) == 10


def test_constants_are_the_ones_the_dataset_uses(expected):
    assert list(CONSTANTS) == list(expected["constants"])
    assert len(CONSTANTS) == 17


def test_argument_kinds_are_the_ones_the_reports_count(expected):
    assert set(ARGUMENT_KINDS) == set(expected["argument_kinds"])


def test_each_constant_means_the_number_in_its_name():
    for name, value in CONSTANTS.items():
        suffix = name.removeprefix("const_")
        assert value == (-1.0 if suffix == "m1" else float(suffix))


def test_the_system_text_names_every_operation_and_constant():
    """What the executor accepts is what every model is told it may write."""
    system = load_prompt().system
    words = set(system.replace(",", " ").replace(":", " ").split())
    assert set(OPERATIONS) <= words
    assert set(CONSTANTS) <= words


def test_the_default_labels_are_the_ones_in_the_prompt_definition():
    prompt = load_prompt()
    reply_format = ReplyFormat.from_prompt(prompt)
    assert (reply_format.program_label, reply_format.answer_label) == (
        prompt.program_label,
        prompt.answer_label,
    )


# --------------------------------------------------------------------------- numbers


@pytest.mark.parametrize(
    ("text", "value"),
    [("0", 0.0), ("920", 920.0), ("-121", -121.0), ("4474.2", 4474.2), (".35", 0.35),
     ("-.5", -0.5), ("0.0", 0.0), ("-0", 0.0)],
)  # fmt: skip
def test_plain_numbers_are_read(text, value):
    assert read_number(text) == value


@pytest.mark.parametrize(
    "text",
    ["", "-", ".", "5.", "+5", "1e5", "1,000", "1_000", " 5", "5 ", "inf", "nan", "--5",
     "1.2.3", "0x1f", "٥", "9" * 400],
)  # fmt: skip
def test_everything_else_is_not_a_number(text):
    assert read_number(text) is None


# --------------------------------------------------------------------------- limits and errors


def test_the_committed_limits_load():
    config = load_dsl_config()
    assert config.max_reply_chars >= 1000  # far above the longest correct reply (145)
    assert config.max_steps >= 6  # the longest gold program


def _write(tmp_path, body: str):
    (tmp_path / "dsl.yaml").write_text(body, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    "body",
    [
        "version: 1\nmax_reply_chars: 4000\n",  # a key is missing
        "version: 1\nmax_reply_chars: 0\nmax_steps: 32\n",
        "version: 1\nmax_reply_chars: -5\nmax_steps: 32\n",
        "version: 1\nmax_reply_chars: 4000\nmax_steps: 3.5\n",
        "version: 1\nmax_reply_chars: 4000\nmax_steps: many\n",
        "version: 1\nmax_reply_chars: true\nmax_steps: 32\n",
        "max_reply_chars: 4000\nmax_steps: 32\n",  # no version
    ],
)
def test_a_bad_limits_file_is_refused(body, tmp_path):
    with pytest.raises(ConfigError):
        load_dsl_config(_write(tmp_path, body))


def test_a_good_limits_file_loads(tmp_path):
    config = load_dsl_config(_write(tmp_path, "version: 1\nmax_reply_chars: 50\nmax_steps: 2\n"))
    assert config == DslConfig(max_reply_chars=50, max_steps=2)


def test_every_code_belongs_to_one_stage():
    assert set(CODES.values()) == {"reply", "program", "execution"}
    assert len(CODES) == 19


@pytest.mark.parametrize(
    ("kind", "code"),
    [(ReplyError, "syntax"), (ProgramError, "no_program"), (ExecutionError, "syntax"),
     (ProgramError, "made_up"), (DslError, "syntax")],
)  # fmt: skip
def test_a_code_cannot_be_raised_at_the_wrong_stage(kind, code):
    with pytest.raises(ValueError):
        kind(code, "message")


def test_an_error_carries_its_code_stage_and_position():
    error = ProgramError("syntax", "something is off", 7)
    assert (error.code, error.stage, error.position, str(error)) == (
        "syntax",
        "program",
        7,
        "something is off",
    )
    assert isinstance(error, DslError)


def test_an_excerpt_is_short_and_ascii():
    assert excerpt("x" * 100) == "x" * 40
    assert excerpt("abcdef", 4) == "ef"
    assert excerpt("été ٥").isascii()
