"""Writing the public copy to its folder, and comparing a folder with what it should hold.

The folder is outside this repository and becomes the working tree of the public one. The
writer is careful with it: it writes into an empty folder or over a previous export, it
removes only files that the previous export wrote and this one no longer has, and it never
reads, writes, or removes the folder's ``.git``.

Usage::

    from qlora_finetuning.public_export.build import compare, write_copy

    result = write_copy(dest, expected, cfg, source_root=cfg.root, clean=True)
    result.written, result.removed
    compare(dest, expected, cfg)       # problems; [] when the folder is exactly the export
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qlora_finetuning.public_export.config import ExportConfig
from qlora_finetuning.public_export.snapshot import NOTICE_NAME
from qlora_finetuning.public_export.transform import Expected
from qlora_finetuning.settings import ConfigError

MANIFEST_NAME = ".public_export_manifest.json"  # git-ignored: what the last export wrote
GIT_DIR = ".git"
SCHEMA = 1
SHOWN = 5


@dataclass(frozen=True)
class BuildResult:
    written: int
    removed: tuple[str, ...]


def _inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
    except ValueError:
        return False
    return True


def _damaged(dest: Path) -> str:
    return f"{MANIFEST_NAME} in {dest} is damaged; it cannot be exported over"


def read_manifest(dest: Path) -> dict[str, Any] | None:
    """What the last export wrote into ``dest``, or ``None`` if there was none."""
    path = dest / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        raise ConfigError(_damaged(dest)) from None
    if not isinstance(data, dict) or not isinstance(data.get("files"), list):
        raise ConfigError(_damaged(dest))
    return data


def destination_problems(dest: Path, cfg: ExportConfig, source_root: Path) -> list[str]:
    """Why ``dest`` may not be written to; empty when it may (spec D5)."""
    if not dest.is_absolute():
        return [f"the destination must be a full path, got {dest}"]
    if _inside(dest, source_root) or _inside(source_root, dest):
        return ["the destination must be outside this repository, and not a folder above it"]
    if dest.exists() and not dest.is_dir():
        return [f"{dest} is a file, not a folder"]
    if not dest.exists():
        return [] if dest.parent.is_dir() else [f"the folder above {dest.name} does not exist"]
    present = [
        entry.name for entry in dest.iterdir() if entry.name not in cfg.ignored_at_destination
    ]
    if not present:
        return []
    if read_manifest(dest) is None or not (dest / NOTICE_NAME).is_file():
        shown = ", ".join(sorted(present)[:SHOWN])
        return [
            f"{dest} is not empty and is not a previous export (it holds {shown}): nothing "
            "was written"
        ]
    return []


def _target(dest: Path, relative: str) -> Path:
    parts = relative.split("/")
    if not relative or relative.startswith("/") or ".." in parts or ":" in relative:
        raise ConfigError(f"{relative!r} is not a path inside the copy")
    if parts[0] == GIT_DIR:
        raise ConfigError("an export never writes into .git")
    return dest.joinpath(*parts)


def write_copy(
    dest: Path, expected: Expected, cfg: ExportConfig, *, source_root: Path, clean: bool
) -> BuildResult:
    """Write the copy into ``dest``. Raises ``ConfigError`` and writes nothing if refused.

    Files of the previous export that this one no longer has are removed, and folders
    they leave empty. Nothing else in the folder is touched.
    """
    problems = destination_problems(dest, cfg, source_root)
    if problems:
        raise ConfigError(problems[0])
    targets = {relative: _target(dest, relative) for relative in expected.files}
    previous = read_manifest(dest) if dest.exists() else None

    removed = []
    for relative in sorted(set((previous or {}).get("files", [])) - set(targets)):
        path = _target(dest, relative)
        if path.is_file():
            path.unlink()
            removed.append(relative)
            folder = path.parent
            while folder != dest and not any(folder.iterdir()):
                folder.rmdir()
                folder = folder.parent

    dest.mkdir(exist_ok=True)
    for relative, path in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected.files[relative])
    manifest = {
        "schema": SCHEMA,
        "commit": expected.info.sha,
        "day": expected.info.day,
        "clean": bool(clean),
        "files": sorted(targets),
    }
    (dest / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return BuildResult(written=len(targets), removed=tuple(removed))


def files_at(dest: Path, cfg: ExportConfig) -> dict[str, Path]:
    """Every file under ``dest`` that could be part of the copy, by its forward-slash path."""
    skipped = set(cfg.ignored_at_destination)
    found: dict[str, Path] = {}
    if not dest.is_dir():
        return found
    stack = [dest]
    while stack:
        folder = stack.pop()
        for entry in folder.iterdir():
            if entry.name in skipped or entry.name == MANIFEST_NAME:
                continue
            if entry.is_dir():
                stack.append(entry)
            elif entry.is_file():
                found[entry.relative_to(dest).as_posix()] = entry
    return found


def compare(dest: Path, expected: Expected, cfg: ExportConfig) -> list[str]:
    """How the folder differs from the export of its commit; empty when it is exactly that."""
    present = files_at(dest, cfg)
    problems = []
    for kind, names in (
        ("missing", sorted(set(expected.files) - set(present))),
        ("not part of the export", sorted(set(present) - set(expected.files))),
        (
            "changed since the export",
            sorted(
                name
                for name in set(present) & set(expected.files)
                if present[name].read_bytes() != expected.files[name]
            ),
        ),
    ):
        if names:
            more = f", and {len(names) - SHOWN} more" if len(names) > SHOWN else ""
            problems.append(f"{len(names)} file(s) {kind}: {', '.join(names[:SHOWN])}{more}")
    return problems
