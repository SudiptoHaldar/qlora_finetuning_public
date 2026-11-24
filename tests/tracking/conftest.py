"""Fixtures for the tracking tests: everything is written under a temporary folder."""

from __future__ import annotations

import pytest

from qlora_finetuning.tracking import GitInfo, load_tracking_config, start_run

CLEAN = GitInfo(commit="0123456789abcdef0123456789abcdef01234567", branch="main", dirty=False)
DIRTY = GitInfo(
    commit=CLEAN.commit,
    branch="main",
    dirty=True,
    changed=("scripts/train.py", "config/training.yaml"),
)
NO_GIT = GitInfo(commit="unknown")


@pytest.fixture
def cfg(tmp_path):
    """The real allowed values, with every path under the test's temporary folder."""
    return load_tracking_config(root=tmp_path)


@pytest.fixture
def make_run(cfg):
    """Start a run with safe defaults: fixed git state and no environment probing."""

    def factory(kind="eval", label="b0 pro dev", **overrides):
        options = {
            "path": "B",
            "arm": "B0-Pro",
            "where": "bedrock",
            "scope": "dev",
            "cfg": cfg,
            "git": CLEAN,
            "environment": {},
        }
        options.update(overrides)
        return start_run(kind, label, **options)

    return factory
