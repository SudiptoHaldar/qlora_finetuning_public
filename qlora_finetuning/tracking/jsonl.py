"""Append-only JSON Lines files: one JSON object per line."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from qlora_finetuning.tracking.config import TrackingError


def append_line(path: Path | str, obj: dict[str, Any]) -> None:
    """Append ``obj`` as one line and force it to disk. Creates the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_lines(path: Path | str) -> list[dict[str, Any]]:
    """Return every object in the file, in order. A missing file is an empty list.

    A line that is not a JSON object raises ``TrackingError`` naming the file and line.
    """
    path = Path(path)
    if not path.is_file():
        return []
    objects: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise TrackingError(f"{path.name} line {number} is not valid JSON") from exc
            if not isinstance(obj, dict):
                raise TrackingError(f"{path.name} line {number} is not a JSON object")
            objects.append(obj)
    return objects
