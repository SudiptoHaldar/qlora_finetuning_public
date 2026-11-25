"""Packaging settings: ``config/packaging.yaml`` (req_006 v6.02).

Usage::

    from qlora_finetuning.packaging.config import load_packaging_config

    cfg = load_packaging_config()
    cfg.adapter_dir, cfg.merged_dir          # the two package folders, under artifacts/release/
    cfg.adapter_repo, cfg.merged_repo        # the names the cards print
    cfg.licence_problems(today)              # why a card may not be built yet

This module imports no part of the training stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from qlora_finetuning.settings import REPO_ROOT, ConfigError, load_config

FILE = "config packaging.yaml"
REQUIRED_KEYS = (
    "release_dir",
    "report_dir",
    "code_repository",
    "author",
    "license",
    "adapter.folder",
    "adapter.repo_id",
    "merged.folder",
    "merged.repo_id",
    "merged.compute_dtype",
    "merged.max_shard_size",
    "tokenizer_files",
    "records.six_arm",
    "records.statistics",
    "records.cost",
    "verification.split",
    "verification.card_code_timeout_seconds",
    "publish.token_env",
    "publish.hub_files",
    "publish.card_code_timeout_seconds",
    "licences.max_age_days",
    "licences.entries",
    "forbidden",
)
COMPUTE_DTYPES = ("float32", "float64")
_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_FOLDER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
LICENCE_FIELDS = ("name", "role", "identifier", "source", "found", "read_on")


@dataclass(frozen=True)
class LicenceEntry:
    """What one licence's source said, on which day, and whether the owner confirmed it."""

    name: str
    role: str
    identifier: str
    source: str
    found: str
    read_on: str
    confirmed_on: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "identifier": self.identifier,
            "source": self.source,
            "found": self.found,
            "read_on": self.read_on,
            "confirmed_on": self.confirmed_on,
        }


@dataclass(frozen=True)
class PackagingConfig:
    """The typed content of ``config/packaging.yaml``."""

    root: Path
    release_dir: Path
    report_dir: Path
    adapter_dir: Path
    merged_dir: Path
    adapter_repo: str
    merged_repo: str
    code_repository: str
    author: str
    license: str
    compute_dtype: str
    max_shard_size: str
    tokenizer_files: tuple[str, ...]
    six_arm_file: Path
    statistics_file: Path
    cost_file: Path
    split: str
    allowance_examples: int | None
    card_code_timeout_seconds: int
    licence_max_age_days: int
    licences: tuple[LicenceEntry, ...]
    forbidden: tuple[tuple[str, re.Pattern[str]], ...]
    token_env: str = "HF_TOKEN"
    hub_files: tuple[str, ...] = (".gitattributes",)
    publish_timeout_seconds: int = 7200

    @property
    def record_json(self) -> Path:
        return self.report_dir / "package_record.json"

    @property
    def record_markdown(self) -> Path:
        return self.report_dir / "package_record.md"

    @property
    def adapter_card(self) -> Path:
        return self.report_dir / "adapter_card.md"

    @property
    def merged_card(self) -> Path:
        return self.report_dir / "merged_card.md"

    @property
    def publish_json(self) -> Path:
        return self.report_dir / "publish_record.json"

    @property
    def publish_markdown(self) -> Path:
        return self.report_dir / "publish_record.md"

    @property
    def card_runs_file(self) -> Path:
        """What running the cards' code gave; written by ``hf_package verify``."""
        return self.release_dir / "card_runs.json"

    def licence_problems(self, today: date) -> list[str]:
        """Why the licence entries do not allow a card to be built; empty when they do."""
        problems = []
        for entry in self.licences:
            if entry.confirmed_on is None:
                problems.append(f"the licence of {entry.name} is not confirmed by the owner")
                continue
            age = (today - date.fromisoformat(entry.confirmed_on)).days
            if age > self.licence_max_age_days:
                problems.append(
                    f"the licence of {entry.name} was confirmed {age} days ago, more than "
                    f"{self.licence_max_age_days}: read it again at its source"
                )
        return problems


def _text(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{FILE}: {key} must be text")
    return value.strip()


def _day(value: Any, key: str) -> str:
    text = str(value) if value is not None else ""
    if not _DAY.fullmatch(text):
        raise ConfigError(f"{FILE}: {key} must be a day written as YYYY-MM-DD")
    try:
        date.fromisoformat(text)
    except ValueError:
        raise ConfigError(f"{FILE}: {key} is not a day of the calendar") from None
    return text


def _whole(value: Any, key: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{FILE}: {key} must be a whole number of {minimum} or more")
    return value


def _licences(raw: Any) -> tuple[LicenceEntry, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{FILE}: licences.entries must be a list of entries")
    entries = []
    for number, item in enumerate(raw):
        where = f"licences.entries[{number}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{FILE}: {where} must be a mapping")
        missing = [name for name in LICENCE_FIELDS if not item.get(name)]
        if missing:
            raise ConfigError(f"{FILE}: {where} is missing: {', '.join(missing)}")
        confirmed = item.get("confirmed_on")
        entries.append(
            LicenceEntry(
                name=_text(item["name"], f"{where}.name"),
                role=_text(item["role"], f"{where}.role"),
                identifier=_text(item["identifier"], f"{where}.identifier"),
                source=_text(item["source"], f"{where}.source"),
                found=" ".join(_text(item["found"], f"{where}.found").split()),
                read_on=_day(item["read_on"], f"{where}.read_on"),
                confirmed_on=(
                    None if confirmed is None else _day(confirmed, f"{where}.confirmed_on")
                ),
            )
        )
    names = [entry.name for entry in entries]
    if len(set(names)) != len(names):
        raise ConfigError(f"{FILE}: licences.entries names a licence twice")
    return tuple(entries)


def _forbidden(raw: Any) -> tuple[tuple[str, re.Pattern[str]], ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{FILE}: forbidden must be a list of patterns")
    patterns = []
    for number, item in enumerate(raw):
        where = f"forbidden[{number}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{FILE}: {where} must be a mapping with name and pattern")
        name = _text(item.get("name"), f"{where}.name")
        try:
            patterns.append((name, re.compile(_text(item.get("pattern"), f"{where}.pattern"))))
        except re.error as exc:
            raise ConfigError(f"{FILE}: {where}.pattern is not a pattern ({exc})") from None
    return tuple(patterns)


def load_packaging_config(
    root: Path | str = REPO_ROOT, config_dir: Path | str | None = None
) -> PackagingConfig:
    """Read and check ``config/packaging.yaml``. A problem is a ``ConfigError`` naming it."""
    raw = load_config("packaging", required=REQUIRED_KEYS, config_dir=config_dir)
    root = Path(root)
    adapter, merged, verification = raw["adapter"], raw["merged"], raw["verification"]

    folders = {}
    for key, section in (("adapter", adapter), ("merged", merged)):
        folder = _text(section["folder"], f"{key}.folder")
        if not _FOLDER.fullmatch(folder):
            raise ConfigError(f"{FILE}: {key}.folder must be one folder name, not a path")
        if not _REPO_ID.fullmatch(_text(section["repo_id"], f"{key}.repo_id")):
            raise ConfigError(f"{FILE}: {key}.repo_id must be written as account/name")
        folders[key] = folder
    if folders["adapter"] == folders["merged"]:
        raise ConfigError(f"{FILE}: the two packages must have different folders")
    if merged["compute_dtype"] not in COMPUTE_DTYPES:
        raise ConfigError(
            f"{FILE}: merged.compute_dtype must be one of {', '.join(COMPUTE_DTYPES)}"
        )
    names = raw["tokenizer_files"]
    if not isinstance(names, list) or any(
        not isinstance(name, str) or not _FOLDER.fullmatch(name) for name in names
    ):
        raise ConfigError(f"{FILE}: tokenizer_files must be a list of file names")
    allowance = verification.get("allowance_examples")
    if allowance is not None:
        allowance = _whole(allowance, "verification.allowance_examples", 1)
    code_repository = _text(raw["code_repository"], "code_repository")
    if not code_repository.startswith("https://"):
        raise ConfigError(f"{FILE}: code_repository must be an https address")
    author = _text(raw["author"], "author")
    if "@" in author:
        raise ConfigError(f"{FILE}: author is an account name, never an email address")

    release_dir = root / _text(raw["release_dir"], "release_dir")
    records = raw["records"]
    return PackagingConfig(
        root=root,
        release_dir=release_dir,
        report_dir=root / _text(raw["report_dir"], "report_dir"),
        adapter_dir=release_dir / folders["adapter"],
        merged_dir=release_dir / folders["merged"],
        adapter_repo=adapter["repo_id"].strip(),
        merged_repo=merged["repo_id"].strip(),
        code_repository=code_repository,
        author=author,
        license=_text(raw["license"], "license"),
        compute_dtype=merged["compute_dtype"],
        max_shard_size=_text(merged["max_shard_size"], "merged.max_shard_size"),
        tokenizer_files=tuple(names),
        six_arm_file=root / _text(records["six_arm"], "records.six_arm"),
        statistics_file=root / _text(records["statistics"], "records.statistics"),
        cost_file=root / _text(records["cost"], "records.cost"),
        split=_text(verification["split"], "verification.split"),
        allowance_examples=allowance,
        card_code_timeout_seconds=_whole(
            verification["card_code_timeout_seconds"], "verification.card_code_timeout_seconds", 1
        ),
        licence_max_age_days=_whole(raw["licences"]["max_age_days"], "licences.max_age_days", 1),
        licences=_licences(raw["licences"]["entries"]),
        token_env=_text(raw["publish"]["token_env"], "publish.token_env").strip(),
        hub_files=tuple(raw["publish"]["hub_files"]),
        publish_timeout_seconds=_whole(
            raw["publish"]["card_code_timeout_seconds"],
            "publish.card_code_timeout_seconds",
            1,
        ),
        forbidden=_forbidden(raw["forbidden"]),
    )
