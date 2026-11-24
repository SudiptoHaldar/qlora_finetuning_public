"""Export settings: ``config/public_export.yaml`` (req_006 v6.04).

Usage::

    from qlora_finetuning.public_export.config import load_export_config

    cfg = load_export_config()
    cfg.left_out, cfg.moved            # what is not copied, and what is copied elsewhere
    cfg.rules()                        # the literal replacements and rewrites, in order
    cfg.destination(None)              # the folder of --to, or of QLORA_PUBLIC_EXPORT_DIR

The value of a ``from_config`` replacement (the AWS account number) is read from its own
config file when the rules are asked for, so it is written in one place only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qlora_finetuning.settings import REPO_ROOT, ConfigError, get_str, load_config

FILE = "config public_export.yaml"
DESTINATION_ENV = "QLORA_PUBLIC_EXPORT_DIR"
REQUIRED_KEYS = (
    "report_dir",
    "public_repository",
    "left_out",
    "moved",
    "replacements",
    "path_rewrites",
    "may_not_remain",
    "allowed_email_domains",
    "ignored_at_destination",
    "limits.max_file_mb",
    "limits.forbidden_suffixes",
    "suite_timeout_minutes",
)


@dataclass(frozen=True)
class Rule:
    """One literal replacement: every ``old`` becomes ``new``."""

    name: str
    old: str
    new: str


@dataclass(frozen=True)
class ExportConfig:
    """The typed content of ``config/public_export.yaml``."""

    root: Path
    config_dir: Path | None
    report_dir: Path
    public_repository: str
    left_out: tuple[str, ...]
    moved: tuple[tuple[str, str], ...]
    replacements: tuple[dict[str, str], ...]
    path_rewrites: tuple[Rule, ...]
    may_not_remain: tuple[tuple[str, re.Pattern[str]], ...]
    invented_values: tuple[str, ...]
    allowed_email_domains: tuple[str, ...]
    ignored_at_destination: tuple[str, ...]
    max_file_bytes: int
    forbidden_suffixes: tuple[str, ...]
    suite_timeout_seconds: int

    @property
    def record_json(self) -> Path:
        return self.report_dir / "export_record.json"

    @property
    def record_markdown(self) -> Path:
        return self.report_dir / "export_record.md"

    def replacement_rules(self) -> tuple[Rule, ...]:
        """The replacements with their values found. Raises ``ConfigError``."""
        rules = []
        for item in self.replacements:
            if "from" in item:
                old = item["from"]
            else:
                name, _, key = item["from_config"].partition(".")
                value = load_config(name, required=(key,), config_dir=self.config_dir)
                for part in key.split("."):
                    value = value[part]
                old = str(value)
            if not old or old == item["to"]:
                raise ConfigError(f"{FILE}: the replacement of {item['name']} changes nothing")
            rules.append(Rule(item["name"], old, item["to"]))
        return tuple(rules)

    def rules(self) -> tuple[Rule, ...]:
        """Every literal change made to a text file: the replacements, then the rewrites."""
        return (*self.replacement_rules(), *self.path_rewrites)

    def destination(self, given: str | None) -> Path:
        """The folder the copy goes to: ``given``, or the one named in ``.env``."""
        text = given or get_str(DESTINATION_ENV, "")
        if not text:
            raise ConfigError(
                f"no destination: give --to FOLDER, or set {DESTINATION_ENV} in .env"
            )
        folder = Path(text)
        if not folder.is_absolute():
            raise ConfigError(f"the destination must be a full path, got {text!r}")
        return folder


def _text(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{FILE}: {key} must be text")
    return value


def _path(value: Any, key: str) -> str:
    text = _text(value, key).replace("\\", "/")
    if text.startswith("/") or ".." in text.split("/") or ":" in text:
        raise ConfigError(f"{FILE}: {key} must be a path inside the repository, got {value!r}")
    return text


def _names(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ConfigError(f"{FILE}: {key} must be a list of names")
    return tuple(value)


def _pairs(value: Any, key: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ConfigError(f"{FILE}: {key} must be a list of entries")
    return value


def load_export_config(
    root: Path | str = REPO_ROOT, config_dir: Path | str | None = None
) -> ExportConfig:
    """Read and check ``config/public_export.yaml``. A problem is a ``ConfigError`` naming it."""
    raw = load_config("public_export", required=REQUIRED_KEYS, config_dir=config_dir)
    root = Path(root)

    left_out = tuple(_path(item, "left_out") for item in _names(raw["left_out"], "left_out"))
    moved = tuple(
        (_path(item.get("from"), "moved.from"), _path(item.get("to"), "moved.to"))
        for item in _pairs(raw["moved"], "moved")
    )
    replacements = []
    for number, item in enumerate(_pairs(raw["replacements"], "replacements")):
        where = f"replacements[{number}]"
        entry = {"name": _text(item.get("name"), f"{where}.name"), "to": str(item.get("to", ""))}
        if not entry["to"]:
            raise ConfigError(f"{FILE}: {where}.to must say what the value becomes")
        if ("from" in item) == ("from_config" in item):
            raise ConfigError(f"{FILE}: {where} needs exactly one of from and from_config")
        source = "from" if "from" in item else "from_config"
        entry[source] = _text(item[source], f"{where}.{source}")
        if source == "from_config" and "." not in entry[source]:
            raise ConfigError(f"{FILE}: {where}.from_config must be written as file.key")
        replacements.append(entry)
    rewrites = tuple(
        Rule(
            f"the path {_path(item.get('from'), 'path_rewrites.from')}",
            _path(item.get("from"), "path_rewrites.from"),
            _path(item.get("to"), "path_rewrites.to"),
        )
        for item in _pairs(raw["path_rewrites"], "path_rewrites")
    )
    patterns = []
    for number, item in enumerate(_pairs(raw["may_not_remain"], "may_not_remain")):
        where = f"may_not_remain[{number}]"
        try:
            compiled = re.compile(_text(item.get("pattern"), f"{where}.pattern"))
        except re.error as exc:
            raise ConfigError(f"{FILE}: {where}.pattern is not a pattern ({exc})") from None
        patterns.append((_text(item.get("name"), f"{where}.name"), compiled))

    megabytes, minutes = raw["limits"]["max_file_mb"], raw["suite_timeout_minutes"]
    for value, key in ((megabytes, "limits.max_file_mb"), (minutes, "suite_timeout_minutes")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(f"{FILE}: {key} must be a whole number of 1 or more")
    repository = _text(raw["public_repository"], "public_repository")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repository):
        raise ConfigError(f"{FILE}: public_repository must be written as owner/name")

    return ExportConfig(
        root=root,
        config_dir=None if config_dir is None else Path(config_dir),
        report_dir=root / _path(raw["report_dir"], "report_dir"),
        public_repository=repository,
        left_out=left_out,
        moved=moved,
        replacements=tuple(replacements),
        path_rewrites=rewrites,
        may_not_remain=tuple(patterns),
        invented_values=_names(raw.get("invented_values") or [], "invented_values"),
        allowed_email_domains=tuple(
            name.lower() for name in _names(raw["allowed_email_domains"], "allowed_email_domains")
        ),
        ignored_at_destination=_names(raw["ignored_at_destination"], "ignored_at_destination"),
        max_file_bytes=megabytes * 1024 * 1024,
        forbidden_suffixes=tuple(
            name.lower() for name in _names(raw["limits"]["forbidden_suffixes"], "limits")
        ),
        suite_timeout_seconds=minutes * 60,
    )
