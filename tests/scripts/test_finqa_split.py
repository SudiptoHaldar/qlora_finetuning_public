"""Tests for scripts.finqa_split, on an invented dataset in a temporary folder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from qlora_finetuning.data import snapshot_path
from qlora_finetuning.data.split import read_manifest
from qlora_finetuning.data.split_report import report_paths
from scripts import finqa_split
from tests.data.synthetic import build_frozen_split, build_split_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def fresh(tmp_path, **overrides):
    """A snapshot and a split config with no manifest written yet."""
    cfg, data_cfg, manifest = build_frozen_split(tmp_path, write=False, **overrides)
    return cfg, data_cfg, manifest


def test_first_run_writes_the_manifest_and_the_report(tmp_path, capsys):
    cfg, data_cfg, expected = fresh(tmp_path)
    assert finqa_split.main([], cfg=cfg, data_cfg=data_cfg) == 0
    out = capsys.readouterr().out
    assert "written splits/finqa_split_manifest.json" in out
    assert out.count("PASS  ") == 10
    assert "10 passed, 0 failed" in out
    assert "fit 30  dev 8  test 22  quick-dev 3  (groups by company: 5, 2, 4)" in out
    assert f"content hash {expected['content_hash']}" in out
    assert "written reports/data/finqa_split_report.md" in out
    assert out.isascii() and str(tmp_path) not in out

    assert read_manifest(cfg) == expected
    markdown_path, json_path = report_paths(cfg.report_dir)
    assert markdown_path.is_file()
    assert (
        json.loads(json_path.read_text(encoding="utf-8"))["columns"]["test"]["size"]["examples"]
        == 22
    )
    assert not cfg.unlock_file.exists()


def test_second_run_changes_nothing(tmp_path, capsys):
    cfg, data_cfg, _ = fresh(tmp_path)
    finqa_split.main([], cfg=cfg, data_cfg=data_cfg)
    files = [cfg.manifest_file, *report_paths(cfg.report_dir)]
    before = [path.read_bytes() for path in files]
    capsys.readouterr()
    assert finqa_split.main([], cfg=cfg, data_cfg=data_cfg) == 0
    assert "unchanged splits/finqa_split_manifest.json" in capsys.readouterr().out
    assert [path.read_bytes() for path in files] == before


def test_check_passes_after_a_build_and_fails_before_one(tmp_path, capsys):
    cfg, data_cfg, _ = fresh(tmp_path)
    assert finqa_split.main(["--check"], cfg=cfg, data_cfg=data_cfg) == 1
    out = capsys.readouterr().out
    assert "SPLIT ERROR: the split manifest does not exist" in out
    assert not cfg.manifest_file.exists() and not cfg.report_dir.exists()

    finqa_split.main([], cfg=cfg, data_cfg=data_cfg)
    capsys.readouterr()
    assert finqa_split.main(["--check"], cfg=cfg, data_cfg=data_cfg) == 0
    assert "manifest and report files are current" in capsys.readouterr().out


def test_check_notices_an_edited_report(tmp_path, capsys):
    cfg, data_cfg, _ = fresh(tmp_path)
    finqa_split.main([], cfg=cfg, data_cfg=data_cfg)
    markdown_path, _ = report_paths(cfg.report_dir)
    markdown_path.write_text("edited\n", encoding="utf-8")
    capsys.readouterr()
    assert finqa_split.main(["--check"], cfg=cfg, data_cfg=data_cfg) == 1
    out = capsys.readouterr().out
    assert "OUT OF DATE: finqa_split_report.md." in out
    assert markdown_path.read_text(encoding="utf-8") == "edited\n"


def test_changed_settings_never_replace_the_manifest(tmp_path, capsys):
    cfg, data_cfg, _ = fresh(tmp_path)
    finqa_split.main([], cfg=cfg, data_cfg=data_cfg)
    before = cfg.manifest_file.read_bytes()
    changed = build_split_config(tmp_path, seed=cfg.seed + 1)
    capsys.readouterr()

    assert finqa_split.main([], cfg=changed, data_cfg=data_cfg) == 1
    out = capsys.readouterr().out
    assert "SPLIT ERROR: finqa_split_manifest.json exists and differs" in out
    assert "PASS" not in out
    assert cfg.manifest_file.read_bytes() == before

    assert finqa_split.main(["--check"], cfg=changed, data_cfg=data_cfg) == 1
    out = capsys.readouterr().out
    assert "FAIL  P9  Reproducible" in out
    assert "OUT OF DATE: finqa_split_manifest.json" in out
    assert cfg.manifest_file.read_bytes() == before


def test_split_that_cannot_be_built_exits_1(tmp_path, capsys):
    cfg, data_cfg, _ = fresh(tmp_path)
    impossible = build_split_config(tmp_path, quick_dev_size=50, dev_size=8)
    assert finqa_split.main([], cfg=impossible, data_cfg=data_cfg) == 1
    assert "SPLIT ERROR: quick_dev_size 50 is larger than dev" in capsys.readouterr().out
    assert not cfg.manifest_file.exists()


def test_missing_snapshot_exits_2_with_the_fetch_command(tmp_path, capsys):
    cfg, data_cfg, _ = fresh(tmp_path)
    for item in data_cfg.files:
        (snapshot_path(data_cfg) / item.path).unlink()
    assert finqa_split.main([], cfg=cfg, data_cfg=data_cfg) == 2
    out = capsys.readouterr().out
    assert out.startswith("CONFIG ERROR: ")
    assert "python -m scripts.finqa_fetch" in out
    assert not cfg.manifest_file.exists()


def test_runs_as_a_module():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.finqa_split", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # 0 with a fetched snapshot and current files; 2 where there is no snapshot (CI).
    assert result.returncode in (0, 2), result.stdout + result.stderr
    assert "Traceback" not in result.stderr


def test_running_by_file_path_gives_a_hint_not_a_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "finqa_split.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "python -m scripts.finqa_split" in result.stderr
    assert "Traceback" not in result.stderr
