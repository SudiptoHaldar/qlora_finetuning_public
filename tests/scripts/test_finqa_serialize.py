"""Tests for scripts.finqa_serialize, on an invented dataset in a temporary folder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from qlora_finetuning.data import load_prompt, snapshot_path, split_ids
from qlora_finetuning.data.serialization_report import report_paths
from qlora_finetuning.settings import CONFIG_DIR
from scripts import finqa_serialize
from tests.data.synthetic import build_frozen_split

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def setup(tmp_path):
    cfg, data_cfg, _ = build_frozen_split(tmp_path)
    return cfg, data_cfg, load_prompt(root=tmp_path)


def run(setup, *arguments):
    cfg, data_cfg, prompt = setup
    return finqa_serialize.main(list(arguments), cfg=cfg, data_cfg=data_cfg, prompt=prompt)


def reworded(tmp_path):
    """The real prompt definition with one word of the system text changed."""
    raw = yaml.safe_load((CONFIG_DIR / "prompt.yaml").read_text(encoding="utf-8"))
    raw["system"] = raw["system"].replace("financial", "careful")
    folder = tmp_path / "reworded"
    folder.mkdir()
    (folder / "prompt.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_prompt(root=tmp_path, config_dir=folder)


def test_first_run_writes_the_report(setup, tmp_path, capsys):
    assert run(setup) == 0
    out = capsys.readouterr().out
    assert out.count("PASS  ") == 10
    assert "PASS  C1  Target shape: 38 targets" in out
    assert "10 passed, 0 failed" in out
    assert f"fit 30  dev 8  quick_dev 3  (prompt id {setup[2].prompt_id})" in out
    assert "written reports/data/finqa_serialization_report.md" in out
    assert out.isascii() and str(tmp_path) not in out
    assert "dumped" not in out

    markdown_path, json_path = report_paths(setup[2].report_dir)
    assert markdown_path.is_file()
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["splits"]["fit"]["records"] == 30
    assert not setup[2].dump_dir.exists()


def test_second_run_changes_nothing_and_check_passes(setup, capsys):
    run(setup)
    files = report_paths(setup[2].report_dir)
    before = [path.read_bytes() for path in files]
    assert run(setup) == 0
    assert [path.read_bytes() for path in files] == before
    capsys.readouterr()
    assert run(setup, "--check") == 0
    assert "report files are current" in capsys.readouterr().out


def test_check_before_any_build_and_after_an_edit(setup, capsys):
    assert run(setup, "--check") == 1
    out = capsys.readouterr().out
    assert "OUT OF DATE: finqa_serialization_report.md, finqa_serialization_report.json" in out
    assert not setup[2].report_dir.exists()

    run(setup)
    markdown_path, _ = report_paths(setup[2].report_dir)
    markdown_path.write_text("edited\n", encoding="utf-8")
    capsys.readouterr()
    assert run(setup, "--check") == 1
    assert "OUT OF DATE: finqa_serialization_report.md in reports/data" in capsys.readouterr().out
    assert markdown_path.read_text(encoding="utf-8") == "edited\n"


def test_changed_wording_fails_the_digest_check_once(setup, tmp_path, capsys):
    cfg, data_cfg, _ = setup
    run(setup)
    changed = (cfg, data_cfg, reworded(tmp_path))
    capsys.readouterr()

    assert run(changed, "--check") == 1
    assert "OUT OF DATE" in capsys.readouterr().out

    assert run(changed) == 1
    out = capsys.readouterr().out
    assert "FAIL  C9  Digests: the prompt id changed from" in out
    assert "the fit records changed" in out
    assert "written reports/data/finqa_serialization_report.json" in out

    assert run(changed) == 0  # the rewritten report is now the reference
    assert "PASS  C9  Digests" in capsys.readouterr().out
    assert run(changed, "--check") == 0


def test_dump_writes_readable_records_under_the_dump_folder_only(setup, capsys):
    cfg, _, prompt = setup
    assert run(setup, "--dump") == 0
    out = capsys.readouterr().out
    assert "dumped artifacts/serialized/fit.jsonl (not for committing)" in out
    assert sorted(path.name for path in prompt.dump_dir.iterdir()) == ["dev.jsonl", "fit.jsonl"]
    lines = (prompt.dump_dir / "fit.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["example_id"] for line in lines] == split_ids("fit", cfg)
    first = json.loads(lines[0])
    assert set(first) == {"example_id", "prompt_id", "system", "user", "target"}
    assert first["user"].startswith("Context:\n")


def test_check_and_dump_cannot_be_combined(setup):
    with pytest.raises(SystemExit) as caught:
        run(setup, "--check", "--dump")
    assert caught.value.code == 2


def test_show_prints_one_fit_or_dev_record_and_writes_nothing(setup, capsys):
    cfg, _, prompt = setup
    example_id = split_ids("dev", cfg)[0]
    assert run(setup, "--show", example_id) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"{example_id}  (dev; prompt id {prompt.prompt_id[:12]}...)")
    assert "\nSYSTEM\n" in out and "\nUSER\n" in out and "\nTARGET\n" in out
    assert "You are a financial calculation assistant." in out
    assert f"Question: what was the change in xylophone sales for {example_id}?" in out
    assert "PROGRAM: subtract(920, 95)\nANSWER: 825" in out
    assert "PASS" not in out
    assert not prompt.report_dir.exists() and not prompt.dump_dir.exists()


def test_show_refuses_a_test_id_without_reading_data(setup, capsys, monkeypatch):
    cfg, _, _ = setup

    def forbidden(*args, **kwargs):
        raise AssertionError("data was read for a test id")

    monkeypatch.setattr(finqa_serialize, "load_split", forbidden)
    test_id = split_ids("test", cfg)[0]
    assert run(setup, "--show", test_id) == 1
    out = capsys.readouterr().out
    assert out.startswith(f"REFUSED: {test_id} is in the test split")
    assert "xylophone" not in out


def test_show_with_an_unknown_id(setup, capsys):
    assert run(setup, "--show", "NOPE/2015/page_1.pdf-1") == 1
    assert capsys.readouterr().out.startswith("NOT FOUND: ")


def test_missing_manifest_or_snapshot_exits_2(tmp_path, capsys):
    cfg, data_cfg, _ = build_frozen_split(tmp_path, write=False)
    prompt = load_prompt(root=tmp_path)
    assert finqa_serialize.main([], cfg=cfg, data_cfg=data_cfg, prompt=prompt) == 2
    assert "python -m scripts.finqa_split" in capsys.readouterr().out

    cfg, data_cfg, _ = build_frozen_split(tmp_path / "other")
    for item in data_cfg.files:
        (snapshot_path(data_cfg) / item.path).unlink()
    assert finqa_serialize.main([], cfg=cfg, data_cfg=data_cfg, prompt=prompt) == 2
    assert "python -m scripts.finqa_fetch" in capsys.readouterr().out
    assert not prompt.report_dir.exists()


def test_runs_as_a_module():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.finqa_serialize", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # 0 with a fetched snapshot and a current report; 2 where there is no snapshot (CI).
    assert result.returncode in (0, 2), result.stdout + result.stderr
    assert "Traceback" not in result.stderr


def test_running_by_file_path_gives_a_hint_not_a_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "finqa_serialize.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "python -m scripts.finqa_serialize" in result.stderr
    assert "Traceback" not in result.stderr
