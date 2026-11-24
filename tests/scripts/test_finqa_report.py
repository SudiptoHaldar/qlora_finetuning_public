"""Tests for scripts.finqa_report, on an invented dataset."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from qlora_finetuning.data import snapshot_path
from qlora_finetuning.data.schema_report import report_paths
from scripts import finqa_report
from tests.data.synthetic import build_snapshot, clean_rows, row

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_report_is_written_and_every_check_passes(tmp_path, capsys):
    cfg = build_snapshot(tmp_path)
    assert finqa_report.main([], cfg=cfg) == 0
    out = capsys.readouterr().out
    assert out.count("PASS  ") == 10
    assert "PASS  S1  Rows per file: 5 examples in total" in out
    assert "10 passed, 0 failed" in out
    assert "written reports/data/finqa_schema_report.md" in out
    assert "written reports/data/finqa_schema_report.json" in out
    assert out.isascii()
    assert str(tmp_path) not in out

    markdown_path, json_path = report_paths(cfg.report_dir)
    assert markdown_path.read_text(encoding="utf-8").startswith("# FinQA schema report")
    assert json.loads(json_path.read_text(encoding="utf-8"))["total"]["examples"] == 5


def test_check_passes_after_a_build_and_writes_nothing(tmp_path, capsys):
    cfg = build_snapshot(tmp_path)
    assert finqa_report.main(["--check"], cfg=cfg) == 1
    out = capsys.readouterr().out
    assert "OUT OF DATE: finqa_schema_report.md, finqa_schema_report.json" in out
    assert "python -m scripts.finqa_report" in out
    assert not cfg.report_dir.exists()

    assert finqa_report.main([], cfg=cfg) == 0
    capsys.readouterr()
    assert finqa_report.main(["--check"], cfg=cfg) == 0
    assert "report files are current" in capsys.readouterr().out


def test_check_notices_an_edited_report(tmp_path, capsys):
    cfg = build_snapshot(tmp_path)
    finqa_report.main([], cfg=cfg)
    markdown_path, _ = report_paths(cfg.report_dir)
    markdown_path.write_text(
        markdown_path.read_text(encoding="utf-8") + "edited by hand\n", encoding="utf-8"
    )
    capsys.readouterr()
    assert finqa_report.main(["--check"], cfg=cfg) == 1
    out = capsys.readouterr().out
    assert "OUT OF DATE: finqa_schema_report.md in reports/data" in out
    assert "finqa_schema_report.json" not in out


def test_failed_check_exits_1_and_still_writes(tmp_path, capsys):
    rows = clean_rows()
    rows["train"].append(row("ACME/2015/page_20.pdf-1", program="modulo(7, 2)"))
    cfg = build_snapshot(tmp_path, rows)
    assert finqa_report.main([], cfg=cfg) == 1
    out = capsys.readouterr().out
    assert "FAIL  S7  Operations: unexpected operations: modulo" in out
    assert "9 passed, 1 failed" in out
    assert report_paths(cfg.report_dir)[0].is_file()


def test_missing_snapshot_exits_2_with_the_fetch_command(tmp_path, capsys):
    cfg = build_snapshot(tmp_path, place_files=False)
    assert finqa_report.main([], cfg=cfg) == 2
    out = capsys.readouterr().out
    assert out.startswith("CONFIG ERROR: ")
    assert "python -m scripts.finqa_fetch" in out
    assert not cfg.report_dir.exists()


def test_altered_snapshot_exits_2(tmp_path, capsys):
    cfg = build_snapshot(tmp_path)
    with (snapshot_path(cfg) / cfg.files[0].path).open("ab") as handle:
        handle.write(b"x")
    assert finqa_report.main([], cfg=cfg) == 2
    assert "size is" in capsys.readouterr().out


def test_runs_as_a_module():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.finqa_report", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # 0 with a fetched snapshot and a current report; 2 where there is no snapshot (CI).
    assert result.returncode in (0, 2), result.stdout + result.stderr
    assert "Traceback" not in result.stderr


def test_running_by_file_path_gives_a_hint_not_a_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "finqa_report.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "python -m scripts.finqa_report" in result.stderr
    assert "Traceback" not in result.stderr
