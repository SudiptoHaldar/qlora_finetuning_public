"""Tests for scripts.finqa_fetch, on an invented dataset with a stand-in downloader."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from qlora_finetuning.data import snapshot_path
from scripts import finqa_fetch
from tests.data.synthetic import FakeDownloader, build_snapshot

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_fetch_downloads_and_verifies(tmp_path, capsys):
    cfg = build_snapshot(tmp_path, place_files=False)
    download = FakeDownloader(tmp_path / "source")
    assert finqa_fetch.main([], cfg=cfg, download=download) == 0
    out = capsys.readouterr().out
    assert f"Dataset {cfg.repo_id} at commit {cfg.revision}" in out
    assert "Downloaded 3 of 3 files" in out
    assert out.count("PASS  ") == 3
    assert "PASS  train  data/train-00000-of-00001.parquet: " in out
    assert "3 verified, 0 failed" in out
    assert out.isascii()
    assert len(download.calls) == 3


def test_second_run_downloads_nothing(tmp_path, capsys):
    cfg = build_snapshot(tmp_path)
    download = FakeDownloader(tmp_path / "source")
    assert finqa_fetch.main([], cfg=cfg, download=download) == 0
    assert "Downloaded 0 of 3 files" in capsys.readouterr().out
    assert download.calls == []


def test_verify_only_never_downloads(tmp_path, capsys):
    cfg = build_snapshot(tmp_path, place_files=False)
    download = FakeDownloader(tmp_path / "source")
    assert finqa_fetch.main(["--verify-only"], cfg=cfg, download=download) == 1
    out = capsys.readouterr().out
    assert out.count("FAIL  ") == 3
    assert "file is missing" in out
    assert "0 verified, 3 failed" in out
    assert "Downloaded" not in out
    assert download.calls == []
    assert not snapshot_path(cfg).exists()


def test_verify_only_passes_on_a_good_snapshot(tmp_path, capsys):
    cfg = build_snapshot(tmp_path)
    assert finqa_fetch.main(["--verify-only"], cfg=cfg) == 0
    assert "3 verified, 0 failed" in capsys.readouterr().out


def test_corrupt_download_fails_with_the_reason(tmp_path, capsys):
    cfg = build_snapshot(tmp_path, place_files=False)
    bad = cfg.files[1].path
    download = FakeDownloader(tmp_path / "source", corrupt=(bad,))
    assert finqa_fetch.main([], cfg=cfg, download=download) == 1
    out = capsys.readouterr().out
    assert f"FAIL  validation  {bad}: size is" in out
    assert "2 verified, 1 failed" in out
    assert "SNAPSHOT ERROR:" in out
    assert "Traceback" not in out


def test_failed_download_exits_1(tmp_path, capsys):
    cfg = build_snapshot(tmp_path, place_files=False)
    download = FakeDownloader(tmp_path / "source", fail=True)
    assert finqa_fetch.main([], cfg=cfg, download=download) == 1
    assert "SNAPSHOT ERROR: download of" in capsys.readouterr().out


def test_force_downloads_everything_again(tmp_path, capsys):
    cfg = build_snapshot(tmp_path)
    download = FakeDownloader(tmp_path / "source")
    assert finqa_fetch.main(["--force"], cfg=cfg, download=download) == 0
    assert "Downloaded 3 of 3 files" in capsys.readouterr().out
    assert all(call[3] for call in download.calls)


def test_verify_only_and_force_cannot_be_combined(tmp_path):
    with pytest.raises(SystemExit) as caught:
        finqa_fetch.main(["--verify-only", "--force"], cfg=build_snapshot(tmp_path))
    assert caught.value.code == 2


def test_broken_config_exits_2(monkeypatch, capsys):
    def broken():
        raise finqa_fetch.ConfigError("config data.yaml is missing: finqa.revision")

    monkeypatch.setattr(finqa_fetch, "load_data_config", broken)
    assert finqa_fetch.main(["--verify-only"]) == 2
    assert capsys.readouterr().out.startswith("CONFIG ERROR: ")


def test_runs_as_a_module_without_touching_the_network():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.finqa_fetch", "--verify-only"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1), result.stderr  # 1 where the snapshot is not fetched
    assert "Dataset bevaya/FinQA at commit" in result.stdout
    assert "verified," in result.stdout


def test_running_by_file_path_gives_a_hint_not_a_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "finqa_fetch.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "python -m scripts.finqa_fetch" in result.stderr
    assert "Traceback" not in result.stderr
