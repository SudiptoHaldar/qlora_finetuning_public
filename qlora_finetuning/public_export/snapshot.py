"""How a folder says it is a public snapshot, and what the test suite skips there.

A public copy holds redacted records: the freeze record names AWS resources, so its
content changed with the placeholder account number, and its hash is no longer the one the
unlock record names. Tests that verify those committed records carry the marker
``private_records``. They run in the private repository and are skipped in a snapshot,
with the reason in the test output. Nothing else is skipped for being public.

Usage (``tests/conftest.py``)::

    from qlora_finetuning.public_export.snapshot import REASON, skip_private_records

    def pytest_collection_modifyitems(config, items):
        skip_private_records(items, REPO_ROOT, pytest.mark.skip(reason=REASON))

Give the marker only to a test that fails in a snapshot because a freeze record was
redacted. It is not a way to silence a test.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

NOTICE_NAME = "PUBLIC_SNAPSHOT.md"
MARKER = "private_records"
REASON = (
    "redacted public snapshot: the committed freeze records cannot be verified here "
    f"(see {NOTICE_NAME})"
)


def is_snapshot(root: Path | str) -> bool:
    """Whether ``root`` is a public copy: it carries the snapshot notice."""
    return (Path(root) / NOTICE_NAME).is_file()


def skip_private_records(items: Iterable[Any], root: Path | str, skip: Any) -> int:
    """In a snapshot, give ``skip`` to every test marked ``private_records``.

    ``items`` are pytest's collected tests and ``skip`` is ``pytest.mark.skip(...)``. Returns
    how many were skipped; in the private repository none is.
    """
    if not is_snapshot(root):
        return 0
    skipped = 0
    for item in items:
        if item.get_closest_marker(MARKER) is not None:
            item.add_marker(skip)
            skipped += 1
    return skipped
