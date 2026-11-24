"""Tests for the package root: version and sub-package skeleton."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import qlora_finetuning

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBPACKAGES = (
    "tracking",
    "data",
    "dsl",
    "evaluation",
    "local_qlora",
    "frontier",
    "comparison",
    "demo",
    "packaging",
    "public_export",
)


def test_version_matches_release_number_file():
    expected = (REPO_ROOT / "release_number.md").read_text(encoding="utf-8").strip()
    assert expected
    assert qlora_finetuning.__version__ == expected


def test_version_falls_back_when_file_is_absent(tmp_path):
    assert qlora_finetuning._read_version(tmp_path / "absent.md") == "0+unknown"


def test_version_falls_back_when_file_is_empty(tmp_path):
    path = tmp_path / "release_number.md"
    path.write_text("  \n", encoding="utf-8")
    assert qlora_finetuning._read_version(path) == "0+unknown"


def test_version_is_stripped(tmp_path):
    path = tmp_path / "release_number.md"
    path.write_text("1.4\n", encoding="utf-8")
    assert qlora_finetuning._read_version(path) == "1.4"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports_and_names_its_spec_group(name):
    module = importlib.import_module(f"qlora_finetuning.{name}")
    assert module.__doc__ and "req_" in module.__doc__


def test_importing_the_package_stays_light():
    """A bare import must not pull in YAML or boto3 (or, later, torch)."""
    code = (
        "import sys, qlora_finetuning; "
        "print(sorted(m for m in ('yaml', 'boto3', 'torch') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
