"""Shared pieces of the run registry and cost ledger: config, errors, time, the secret guard."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qlora_finetuning.settings import REPO_ROOT, ConfigError, load_config

REQUIRED_KEYS = (
    "runs_dir",
    "ledger_file",
    "runs_index_file",
    "git_commit_env",
    "allowed.kinds",
    "allowed.paths",
    "allowed.arms",
    "allowed.places",
    "allowed.scopes",
    "allowed.phases",
)

# Values that must never reach a manifest or a ledger line (working agreement 15).
_ACCESS_KEY = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


class TrackingError(ConfigError):
    """A run or ledger record is invalid, or a tracking file cannot be read."""


@dataclass(frozen=True)
class TrackingConfig:
    """Where tracking files live and which values a run or record may carry."""

    root: Path
    runs_dir: Path
    ledger_file: Path
    runs_index_file: Path
    git_commit_env: str
    kinds: frozenset[str]
    paths: frozenset[str]
    arms: frozenset[str]
    places: frozenset[str]
    scopes: frozenset[str]
    phases: frozenset[str]

    def check(self, field_name: str, value: str) -> None:
        """Raise ``TrackingError`` unless ``value`` is allowed for ``field_name``."""
        allowed: frozenset[str] = getattr(self, field_name)
        if value not in allowed:
            raise TrackingError(
                f"{value!r} is not an allowed value for {field_name}; "
                f"use one of: {', '.join(sorted(allowed))}"
            )

    @property
    def ledger_prefix(self) -> str:
        """The ledger folder relative to the root, as git prints it (``ledger/``)."""
        try:
            relative = self.ledger_file.parent.relative_to(self.root)
        except ValueError:
            return ""
        return relative.as_posix().rstrip("/") + "/"


def load_tracking_config(
    root: Path | str = REPO_ROOT, config_dir: Path | str | None = None
) -> TrackingConfig:
    """Read ``config/tracking.yaml``. Relative paths in it resolve under ``root``.

    Tests pass a temporary folder as ``root`` so nothing is written to the repository.
    """
    raw = load_config("tracking", required=REQUIRED_KEYS, config_dir=config_dir)
    root = Path(root)
    allowed = raw["allowed"]
    return TrackingConfig(
        root=root,
        runs_dir=root / raw["runs_dir"],
        ledger_file=root / raw["ledger_file"],
        runs_index_file=root / raw["runs_index_file"],
        git_commit_env=str(raw["git_commit_env"]),
        kinds=frozenset(map(str, allowed["kinds"])),
        paths=frozenset(map(str, allowed["paths"])),
        arms=frozenset(map(str, allowed["arms"])),
        places=frozenset(map(str, allowed["places"])),
        scopes=frozenset(map(str, allowed["scopes"])),
        phases=frozenset(map(str, allowed["phases"])),
    )


def utc_now() -> datetime:
    return datetime.now(UTC)


def timestamp(moment: datetime | None = None) -> str:
    """UTC, ISO 8601, ending in ``Z``: ``2026-11-15T06:15:00Z``."""
    moment = moment or utc_now()
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def check_no_secrets(value: Any, where: str) -> None:
    """Raise ``TrackingError`` if any text in ``value`` looks like a credential or an email.

    Looks at values, not key names: ``input_tokens`` is a legitimate key.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            check_no_secrets(key, where)
            check_no_secrets(item, where)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            check_no_secrets(item, where)
    elif isinstance(value, str):
        if _ACCESS_KEY.search(value):
            raise TrackingError(f"{where} contains what looks like an AWS access key id")
        if _EMAIL.search(value):
            raise TrackingError(f"{where} contains what looks like an email address")
