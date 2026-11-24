"""Looking through a package for what must not be published (req_006 v6.02, spec D28).

A package is public material: no AWS account number, resource name, S3 path, job name,
access key, token, email address, or path on someone's machine. The patterns are in
``config/packaging.yaml``.

Usage::

    from qlora_finetuning.packaging.scan import scan_folder, scan_text

    scan_folder(cfg.adapter_dir, cfg.forbidden, skip=cfg.tokenizer_files)   # [] when clean
    scan_text(card_text, cfg.forbidden, "the card")

Weight files are not text and are not read. The tokenizer files hold arbitrary strings, so
the caller passes them in ``skip``; they are checked by their bytes instead.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from pathlib import Path

BINARY_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")
Pattern = tuple[str, re.Pattern[str]]


def scan_text(text: str, patterns: Sequence[Pattern], where: str) -> list[str]:
    """One problem for each kind of forbidden string found in ``text``."""
    return [f"{where} holds {name}" for name, pattern in patterns if pattern.search(text)]


def scan_folder(
    folder: Path, patterns: Sequence[Pattern], skip: Collection[str] = ()
) -> list[str]:
    """Every text file of a flat folder, scanned. A file that is not UTF-8 is a problem."""
    problems: list[str] = []
    if not folder.is_dir():
        return problems
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.name in skip or path.suffix in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"{path.name} is not a text file and is not a weight file")
            continue
        problems += scan_text(text, patterns, path.name)
    return problems
