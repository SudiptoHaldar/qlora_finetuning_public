"""Small file helpers for the packages: hashes, listings, and copies (req_006 v6.02).

A package is a flat folder. A text file in it is UTF-8 with Unix line ends, whatever git
checked out on this machine, so the same package has the same bytes everywhere.

Usage::

    from qlora_finetuning.packaging.files import copy_text, listing, sha256_file

    copy_text(source, package / "prompt.json")          # line ends become LF
    listing(package)                                     # [{"name", "bytes", "sha256"}, ...]
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from qlora_finetuning.settings import ConfigError


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text_lf(path: Path | str) -> str:
    """A text file's content with Unix line ends."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"a file a package needs is missing: {path}") from None
    return text.replace("\r\n", "\n").replace("\r", "\n")


def write_text_lf(path: Path | str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def write_json_lf(path: Path | str, data: Any) -> None:
    """Plain JSON, indented, ASCII, with Unix line ends."""
    write_text_lf(path, json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def read_json(path: Path | str, what: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"{what} is missing: {path}") from None
    except ValueError:
        raise ConfigError(f"{what} is damaged: {path}") from None


def copy_bytes(source: Path | str, target: Path | str) -> None:
    """A copy with the same bytes: for weights and tokenizer files."""
    if not Path(source).is_file():
        raise ConfigError(f"a file a package needs is missing: {source}")
    shutil.copyfile(source, target)


def copy_text(source: Path | str, target: Path | str) -> None:
    """A copy of a text file with Unix line ends."""
    write_text_lf(target, read_text_lf(source))


def listing(folder: Path | str) -> list[dict[str, Any]]:
    """Every file of a flat folder, by name: its size and its SHA-256."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(folder.iterdir())
        if path.is_file()
    ]


def combined_sha256(entries: list[dict[str, Any]]) -> str:
    """One hash over several files: the SHA-256 of their names and hashes, in name order."""
    lines = "".join(f"{entry['name']}:{entry['sha256']}\n" for entry in sorted(
        entries, key=lambda item: item["name"]
    ))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()
