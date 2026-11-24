"""Settings for the whole test suite.

In a public snapshot of the repository (req_006 v6.04) the committed freeze records are
redacted, so the tests that verify them, marked ``private_records``, are skipped there
with the reason in the output. In the private repository they run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qlora_finetuning.public_export.snapshot import REASON, skip_private_records

REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(config, items):
    skip_private_records(items, REPO_ROOT, pytest.mark.skip(reason=REASON))
