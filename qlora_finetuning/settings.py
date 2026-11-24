"""Environment accessors and the config loader.

Environment variables come from the process environment first, then from ``.env`` at the
repository root. Tunable values come from YAML files under ``config/``; nothing tunable is
hard-coded.

Usage::

    from qlora_finetuning.settings import get_str, get_int, load_config

    level = get_str("LOG_LEVEL", "INFO")
    workers = get_int("QLORA_WORKERS", 2)
    aws = load_config("aws", required=["bucket.name", "region"])
    bucket = aws["bucket"]["name"]

Every problem is reported as ``ConfigError`` with a message that names the variable, file, or
keys involved.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
ENV_FILE = REPO_ROOT / ".env"

# Parsed .env content, read once on first use. The values are never written to os.environ.
_dotenv: dict[str, str | None] | None = None


class ConfigError(Exception):
    """An environment variable or config file is missing or invalid."""


# --------------------------------------------------------------------------- environment


def _env(name: str) -> str | None:
    """Look ``name`` up in the environment, then in ``.env``. An empty value counts as unset."""
    global _dotenv
    value = os.environ.get(name)
    if value:
        return value
    if _dotenv is None:
        _dotenv = dict(dotenv_values(ENV_FILE)) if ENV_FILE.is_file() else {}
    return _dotenv.get(name) or None


def get_str(name: str, default: str | None = None) -> str:
    """Return the variable, or ``default``. With neither, raise ``ConfigError``."""
    value = _env(name)
    if value is not None:
        return value
    if default is not None:
        return default
    raise ConfigError(f"environment variable {name} is not set (see .env.example)")


def get_int(name: str, default: int | None = None) -> int:
    """Return the variable as an integer, or ``default``. With neither, raise ``ConfigError``."""
    value = _env(name)
    if value is None:
        if default is not None:
            return default
        raise ConfigError(f"environment variable {name} is not set (see .env.example)")
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"environment variable {name} must be an integer, got {value!r}") from exc


# --------------------------------------------------------------------------- config files


def _lookup(mapping: Any, dotted: str) -> Any:
    node = mapping
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _is_missing(value: Any) -> bool:
    """None, an empty string, and an empty list count as missing. 0 and False do not."""
    return value is None or (isinstance(value, (str, list)) and len(value) == 0)


def load_config(
    name: str,
    required: Iterable[str] = (),
    config_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Load ``<config_dir>/<name>.yaml`` and return it as a new dict.

    The file must be a mapping with an integer ``version`` key. ``required`` lists dotted keys
    (for example ``"bucket.name"``) that must be present and non-empty; every missing one is
    named in the error. Nothing is cached, so each call returns an independent dict.
    """
    path = (
        Path(config_dir) / f"{name}.yaml" if config_dir is not None else CONFIG_DIR / f"{name}.yaml"
    )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file is not valid YAML: {path}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config file must be a mapping: {path}")

    version = raw.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ConfigError(f"config file needs an integer 'version' key: {path}")

    missing = [key for key in required if _is_missing(_lookup(raw, key))]
    if missing:
        raise ConfigError(f"config {path.name} is missing: {', '.join(missing)}")
    return raw
