"""Tests for qlora_finetuning.logging_setup."""

from __future__ import annotations

import logging

import pytest

from qlora_finetuning import settings
from qlora_finetuning.logging_setup import HANDLER_NAME, get_logger, setup_logging
from qlora_finetuning.settings import ConfigError


def our_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if h.get_name() == HANDLER_NAME]


@pytest.fixture(autouse=True)
def clean_logging(monkeypatch, tmp_path):
    """Start without our handler or LOG_LEVEL, and leave the root logger as it was found."""
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.setattr(settings, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(settings, "_dotenv", None)
    root = logging.getLogger()
    level = root.level
    for handler in our_handlers():  # another test may have run an entry point
        root.removeHandler(handler)
    yield
    for handler in our_handlers():
        root.removeHandler(handler)
    root.setLevel(level)


def test_level_from_argument():
    assert setup_logging("DEBUG") == logging.DEBUG
    assert logging.getLogger().level == logging.DEBUG


def test_level_from_environment(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    assert setup_logging() == logging.WARNING


def test_default_level_is_info():
    assert setup_logging() == logging.INFO


def test_lower_case_level_is_accepted():
    assert setup_logging("error") == logging.ERROR


def test_unknown_level_raises():
    with pytest.raises(ConfigError, match="LOUD"):
        setup_logging("loud")
    assert our_handlers() == []


def test_second_call_changes_level_without_adding_a_handler():
    setup_logging("INFO")
    setup_logging("DEBUG")
    assert len(our_handlers()) == 1
    assert logging.getLogger().level == logging.DEBUG


def test_log_lines_go_to_stderr_only(capsys):
    setup_logging("INFO")
    get_logger("qlora_finetuning.test").info("hello from the test")
    captured = capsys.readouterr()
    assert "hello from the test" in captured.err
    assert "INFO qlora_finetuning.test: hello from the test" in captured.err
    assert captured.out == ""


def test_lines_below_the_level_are_dropped(capsys):
    setup_logging("WARNING")
    get_logger("qlora_finetuning.test").info("quiet")
    assert "quiet" not in capsys.readouterr().err


def test_get_logger_returns_the_named_logger():
    assert get_logger("qlora_finetuning.data").name == "qlora_finetuning.data"
