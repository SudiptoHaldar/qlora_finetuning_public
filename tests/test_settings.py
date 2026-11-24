"""Tests for qlora_finetuning.settings: environment accessors and load_config()."""

from __future__ import annotations

import os

import pytest

from qlora_finetuning import settings
from qlora_finetuning.settings import ConfigError, get_int, get_str, load_config

VAR = "QLORA_TEST_SETTING"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """No real .env, no leftover variable, and a fresh .env cache for every test."""
    monkeypatch.delenv(VAR, raising=False)
    monkeypatch.setattr(settings, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(settings, "_dotenv", None)


def write_env(tmp_path, text: str) -> None:
    (tmp_path / ".env").write_text(text, encoding="utf-8")


def write_config(tmp_path, text: str, name: str = "sample") -> None:
    (tmp_path / f"{name}.yaml").write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- get_str


def test_get_str_from_environment(monkeypatch):
    monkeypatch.setenv(VAR, "hello")
    assert get_str(VAR) == "hello"


def test_get_str_default():
    assert get_str(VAR, "fallback") == "fallback"


def test_get_str_missing_raises_naming_the_variable():
    with pytest.raises(ConfigError, match=VAR):
        get_str(VAR)


def test_get_str_empty_value_counts_as_unset(monkeypatch):
    monkeypatch.setenv(VAR, "")
    assert get_str(VAR, "fallback") == "fallback"


# --------------------------------------------------------------------------- get_int


def test_get_int_from_environment(monkeypatch):
    monkeypatch.setenv(VAR, "42")
    assert get_int(VAR) == 42


def test_get_int_default():
    assert get_int(VAR, 7) == 7


def test_get_int_zero_default_is_a_default():
    assert get_int(VAR, 0) == 0


def test_get_int_missing_raises():
    with pytest.raises(ConfigError, match=VAR):
        get_int(VAR)


def test_get_int_non_integer_raises_naming_variable_and_value(monkeypatch):
    monkeypatch.setenv(VAR, "many")
    with pytest.raises(ConfigError, match=f"{VAR}.*'many'"):
        get_int(VAR)


# --------------------------------------------------------------------------- .env


def test_value_read_from_dotenv_file(tmp_path):
    write_env(tmp_path, f"{VAR}=from-file\n")
    assert get_str(VAR) == "from-file"


def test_environment_wins_over_dotenv(monkeypatch, tmp_path):
    write_env(tmp_path, f"{VAR}=from-file\n")
    monkeypatch.setenv(VAR, "from-env")
    assert get_str(VAR) == "from-env"


def test_dotenv_values_are_not_written_to_the_environment(tmp_path):
    write_env(tmp_path, f"{VAR}=from-file\n")
    assert get_str(VAR) == "from-file"
    assert VAR not in os.environ


def test_missing_dotenv_file_is_fine():
    assert get_str(VAR, "fallback") == "fallback"


# --------------------------------------------------------------------------- load_config


def test_load_committed_aws_config():
    cfg = load_config("aws", required=["bucket.name", "region", "budget.cap_usd"])
    assert cfg["version"] == 1
    assert cfg["bucket"]["name"] == "qlora-sagemaker-123456789012"
    assert cfg["region"] == "us-east-1"
    assert cfg["budget"]["cap_usd"] == 300


def test_load_config_from_another_folder(tmp_path):
    write_config(tmp_path, "version: 3\nlimit: 5\n")
    assert load_config("sample", config_dir=tmp_path) == {"version": 3, "limit": 5}


def test_load_config_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config("absent", config_dir=tmp_path)


def test_load_config_invalid_yaml(tmp_path):
    write_config(tmp_path, "version: 1\nkey: [unclosed\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config("sample", config_dir=tmp_path)


def test_load_config_not_a_mapping(tmp_path):
    write_config(tmp_path, "- just\n- a list\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config("sample", config_dir=tmp_path)


@pytest.mark.parametrize(
    "text", ["limit: 5\n", "version: true\n", 'version: "1"\n', "version: 1.5\n"]
)
def test_load_config_needs_an_integer_version(tmp_path, text):
    write_config(tmp_path, text)
    with pytest.raises(ConfigError, match="integer 'version'"):
        load_config("sample", config_dir=tmp_path)


def test_load_config_lists_every_missing_key(tmp_path):
    write_config(tmp_path, "version: 1\nbucket:\n  name: b\n")
    with pytest.raises(ConfigError) as excinfo:
        load_config(
            "sample", required=["bucket.name", "bucket.region", "budget.cap"], config_dir=tmp_path
        )
    message = str(excinfo.value)
    assert "bucket.region" in message
    assert "budget.cap" in message
    assert "bucket.name" not in message


def test_load_config_empty_values_count_as_missing(tmp_path):
    write_config(tmp_path, "version: 1\nname: ''\nitems: []\nnothing:\n")
    with pytest.raises(ConfigError) as excinfo:
        load_config("sample", required=["name", "items", "nothing"], config_dir=tmp_path)
    assert all(key in str(excinfo.value) for key in ("name", "items", "nothing"))


def test_load_config_zero_and_false_are_values(tmp_path):
    write_config(tmp_path, "version: 1\ncount: 0\nenabled: false\n")
    cfg = load_config("sample", required=["count", "enabled"], config_dir=tmp_path)
    assert cfg["count"] == 0
    assert cfg["enabled"] is False


def test_load_config_returns_a_new_dict_each_call(tmp_path):
    write_config(tmp_path, "version: 1\nlimit: 5\n")
    first = load_config("sample", config_dir=tmp_path)
    first["limit"] = 99
    assert load_config("sample", config_dir=tmp_path)["limit"] == 5
