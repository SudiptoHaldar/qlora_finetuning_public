"""Logging setup shared by every entry point.

Usage::

    from qlora_finetuning.logging_setup import get_logger, setup_logging

    setup_logging()                 # once, at the start of a script; level from LOG_LEVEL
    log = get_logger(__name__)
    log.info("split manifest loaded")

Log lines go to standard error, so whatever a script prints on standard output (for example
PASS and FAIL lines) stays clean and can be piped.
"""

from __future__ import annotations

import logging
import sys

from qlora_finetuning.settings import ConfigError, get_str

LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
HANDLER_NAME = "qlora_finetuning"


def setup_logging(level: str | None = None) -> int:
    """Configure the root logger and return the numeric level in effect.

    The level is ``level`` if given, else the ``LOG_LEVEL`` variable, else INFO. The handler is
    added once; calling again only changes the level.
    """
    name = (level or get_str("LOG_LEVEL", "INFO")).strip().upper()
    if name not in LEVELS:
        raise ConfigError(f"unknown log level {name!r}; use one of {', '.join(LEVELS)}")

    root = logging.getLogger()
    if not any(handler.get_name() == HANDLER_NAME for handler in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.set_name(HANDLER_NAME)
        handler.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(handler)
    root.setLevel(LEVELS[name])
    return LEVELS[name]


def get_logger(name: str) -> logging.Logger:
    """Return the logger for ``name``; pass ``__name__``."""
    return logging.getLogger(name)
