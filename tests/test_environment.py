"""Tests for qlora_finetuning.environment. None of them needs the training stack."""

from __future__ import annotations

import json

import pytest

from qlora_finetuning import environment
from qlora_finetuning.environment import (
    environment_report,
    installed_versions,
    normalize,
    read_pins,
    version_matches,
)
from qlora_finetuning.settings import ConfigError


def write(tmp_path, text: str):
    path = tmp_path / "requirements-train.txt"
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- read_pins


def test_read_pins_parses_exact_pins(tmp_path):
    path = write(tmp_path, "torch==2.14.1\ntransformers==5.23.0\n")
    assert read_pins(path) == {"torch": "2.14.1", "transformers": "5.23.0"}


def test_read_pins_ignores_comments_blanks_and_options(tmp_path):
    text = (
        "# header\n\n--extra-index-url https://example.invalid/whl\n-r requirements.txt\n"
        "torch==2.14.1  # the CUDA build is installed first\n"
    )
    assert read_pins(write(tmp_path, text)) == {"torch": "2.14.1"}


def test_read_pins_normalizes_names(tmp_path):
    assert read_pins(write(tmp_path, "HuggingFace_Hub==2.4.0\n")) == {"huggingface-hub": "2.4.0"}


@pytest.mark.parametrize(
    "line", ["torch>=2.14", "torch", "torch==2.14.1 ; python_version>'3.10'", "torch==="]
)
def test_read_pins_rejects_anything_but_an_exact_pin(tmp_path, line):
    with pytest.raises(ConfigError, match="line 2"):
        read_pins(write(tmp_path, f"peft==0.21.3\n{line}\n"))


def test_read_pins_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        read_pins(tmp_path / "absent.txt")


def test_committed_pins_file_is_exact_and_complete():
    pins = read_pins()
    assert {
        "torch",
        "transformers",
        "datasets",
        "peft",
        "trl",
        "accelerate",
        "bitsandbytes",
    } <= set(pins)
    assert all("+" not in version for version in pins.values()), "pins carry no CUDA suffix"


# --------------------------------------------------------------------------- versions


def test_installed_versions_reports_present_and_absent():
    versions = installed_versions(["pytest", "surely-not-a-real-package-xyz"])
    assert versions["pytest"]
    assert versions["surely-not-a-real-package-xyz"] is None


@pytest.mark.parametrize(
    ("installed", "pinned", "expected"),
    [
        ("2.14.1", "2.14.1", True),
        ("2.14.1+cu126", "2.14.1", True),
        ("2.14.1+cu126", "2.14.1+cu126", True),
        ("2.14.1+cu130", "2.14.1+cu126", False),
        ("2.14.10", "2.14.1", False),
        ("2.14.0", "2.14.1", False),
        (None, "2.14.1", False),
    ],
)
def test_version_matches(installed, pinned, expected):
    assert version_matches(installed, pinned) is expected


def test_normalize():
    assert normalize("Hugging_Face.Hub") == "hugging-face-hub"


# --------------------------------------------------------------------------- report


def test_environment_report_without_torch(tmp_path, monkeypatch):
    monkeypatch.setattr(environment, "_torch_info", lambda: None)
    report = environment_report(
        write(tmp_path, "pytest==0.0.1\nsurely-not-a-real-package-xyz==1.0\n")
    )
    assert report["python"].count(".") == 2
    assert report["packages"]["pytest"]["pinned"] == "0.0.1"
    assert report["packages"]["pytest"]["installed"]
    assert report["packages"]["surely-not-a-real-package-xyz"]["installed"] is None
    assert report["torch"] is None
    json.dumps(report)  # must be plain data


def test_environment_report_survives_a_missing_pins_file(tmp_path, monkeypatch):
    monkeypatch.setattr(environment, "_torch_info", lambda: None)
    report = environment_report(tmp_path / "absent.txt")
    assert "not found" in report["pins_error"]
    assert report["packages"] == {}


def test_torch_info_never_raises():
    info = environment._torch_info()
    assert info is None or isinstance(info, dict)
