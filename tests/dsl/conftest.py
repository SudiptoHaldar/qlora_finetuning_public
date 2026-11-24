"""Shared pieces for the DSL tests. Programs and tables here are invented."""

from __future__ import annotations

import pytest

from qlora_finetuning.dsl import DslConfig, ReplyFormat

# An invented table: a header row, plain rows, and one of each awkward kind.
TABLE = [
    ["", "2024", "2023", "2022"],
    ["revenue", "$ 920", "$ 95", "$ 1,234"],
    ["margin", "12.5%", "10%", "7.5 %"],
    ["net income ( loss )", "-9 ( 9 )", "4", "5"],
    ["cash, end of year", "1", "2", "3"],
    ["growth ( in ( thousands ) )", "10", "20", "30"],
    ["notes", "n/a", "4", "5"],
    ["blanks", "", "4", "5"],
    ["bracketed", "( 9 )", "4", "5"],
    ["total", "1", "2", "3"],
    ["total", "10", "20", "30"],
    ["change", "+5%", "+10 % ( +10 % )", "-5%"],
]


@pytest.fixture
def config() -> DslConfig:
    return DslConfig(max_reply_chars=4000, max_steps=32)


@pytest.fixture
def reply_format() -> ReplyFormat:
    return ReplyFormat(program_label="PROGRAM:", answer_label="ANSWER:")


@pytest.fixture
def table() -> list[list[str]]:
    return [list(row) for row in TABLE]
