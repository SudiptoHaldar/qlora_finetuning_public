"""Tests for scripts.finqa_gold_check, on an invented dataset in a temporary folder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from qlora_finetuning.data import load_prompt, snapshot_path, split_ids
from qlora_finetuning.data.gold_check import load_gold_flags, report_paths
from qlora_finetuning.evaluation.metrics import load_evaluation_config
from scripts import finqa_gold_check
from tests.data.synthetic import SECRET_WORDS, build_frozen_split, split_rows

REPO_ROOT = Path(__file__).resolve().parents[2]


def build(root, broken_id=None):
    """A frozen synthetic split under ``root``; ``broken_id`` gets a gold answer that is wrong."""
    rows = split_rows()
    if broken_id is not None:
        for items in rows.values():
            for item in items:
                if item["id"] == broken_id:
                    item["exe_ans"] = "900"
    cfg, data_cfg, _ = build_frozen_split(root, rows=rows)
    return cfg, data_cfg, load_prompt(root=root), load_evaluation_config(root=root)


@pytest.fixture
def setup(tmp_path):
    return build(tmp_path)


def run(setup, *arguments):
    cfg, data_cfg, prompt, config = setup
    return finqa_gold_check.main(
        list(arguments), cfg=cfg, data_cfg=data_cfg, prompt=prompt, config=config
    )


def files(setup):
    config = setup[3]
    return [*report_paths(config.report_dir), config.flags_file]


def test_first_run_writes_the_report_and_the_flags(setup, tmp_path, capsys):
    assert run(setup) == 0
    out = capsys.readouterr().out
    assert out.count("PASS  ") == 7 and "FAIL" not in out
    assert "PASS  K1  Complete: 60 examples, each checked once" in out
    assert "7 passed, 0 failed" in out
    assert "fit: 30 of 30 gold programs reproduce the gold answer, 0 flagged" in out
    assert "dev: 8 of 8 gold programs reproduce the gold answer, 0 flagged" in out
    assert "quick_dev: 3 of 3 gold programs reproduce the gold answer, 0 flagged" in out
    assert "test: 22 of 22 gold programs reproduce the gold answer, 0 flagged" in out
    assert "written reports/data/finqa_gold_check_report.md" in out
    assert "written reports/data/finqa_gold_check_report.json" in out
    assert "written splits/finqa_gold_flags.json" in out
    assert out.isascii() and str(tmp_path) not in out
    assert not any(word in out for word in SECRET_WORDS)

    assert all(path.is_file() for path in files(setup))
    report = json.loads(files(setup)[1].read_text(encoding="utf-8"))
    assert report["total"]["examples"] == 60
    assert load_gold_flags(setup[3]) == frozenset()


def test_second_run_changes_nothing_and_check_passes(setup, capsys):
    run(setup)
    before = [path.read_bytes() for path in files(setup)]
    assert run(setup) == 0
    assert [path.read_bytes() for path in files(setup)] == before
    capsys.readouterr()
    assert run(setup, "--check") == 0
    assert "report and flags files are current" in capsys.readouterr().out
    assert [path.read_bytes() for path in files(setup)] == before


def test_check_before_any_build_and_after_an_edit(setup, capsys):
    assert run(setup, "--check") == 1
    out = capsys.readouterr().out
    assert (
        "OUT OF DATE: finqa_gold_check_report.md, finqa_gold_check_report.json, "
        "finqa_gold_flags.json" in out
    )
    assert "written" not in out
    assert not any(path.exists() for path in files(setup))

    run(setup)
    markdown_path = files(setup)[0]
    markdown_path.write_text("edited\n", encoding="utf-8")
    capsys.readouterr()
    assert run(setup, "--check") == 1
    assert "OUT OF DATE: finqa_gold_check_report.md." in capsys.readouterr().out
    assert markdown_path.read_text(encoding="utf-8") == "edited\n"


def test_a_flagged_test_example_is_reported_and_leaves_exit_0(tmp_path, capsys):
    probe = build(tmp_path / "probe")
    test_id = split_ids("test", probe[0])[0]
    setup = build(tmp_path / "flagged", broken_id=test_id)
    capsys.readouterr()

    assert run(setup) == 0
    out = capsys.readouterr().out
    assert "test: 21 of 22 gold programs reproduce the gold answer, 1 flagged" in out
    assert "PASS  K3  Fit and dev reproduce: no fit or dev example is flagged" in out
    assert load_gold_flags(setup[3]) == frozenset({test_id})
    flags = json.loads(setup[3].flags_file.read_text(encoding="utf-8"))
    assert flags["flagged"] == [
        {"example_id": test_id, "split": "test", "reason": "value_mismatch"}
    ]
    assert flags["counts"] == {"fit": 0, "dev": 0, "test": 1, "total": 1}
    assert run(setup, "--check") == 0


def test_a_flagged_fit_example_exits_1(tmp_path, capsys):
    probe = build(tmp_path / "probe")
    fit_id = split_ids("fit", probe[0])[0]
    setup = build(tmp_path / "flagged", broken_id=fit_id)
    capsys.readouterr()

    assert run(setup) == 1
    out = capsys.readouterr().out
    assert f"FAIL  K3  Fit and dev reproduce: 1 examples of fit or dev are flagged: {fit_id}" in out
    assert "6 passed, 1 failed" in out
    assert "fit: 29 of 30 gold programs reproduce the gold answer, 1 flagged" in out
    assert load_gold_flags(setup[3]) == frozenset({fit_id})  # still written, for review


def test_changed_settings_fail_the_flags_check_once(setup, capsys):
    run(setup)
    flags_file = setup[3].flags_file
    flags = json.loads(flags_file.read_text(encoding="utf-8"))
    flags["settings"]["execution"]["relative"] = 0.00001  # as if written under another rule
    flags_file.write_text(json.dumps(flags, indent=2) + "\n", encoding="utf-8")
    capsys.readouterr()

    assert run(setup, "--check") == 1
    assert "OUT OF DATE: finqa_gold_flags.json." in capsys.readouterr().out

    assert run(setup) == 1
    out = capsys.readouterr().out
    assert "FAIL  K5  Flags current: the settings changed since the flags file was written" in out
    assert "written splits/finqa_gold_flags.json" in out

    assert run(setup) == 0  # the rewritten flags file is now the reference
    assert "PASS  K5  Flags current" in capsys.readouterr().out
    assert run(setup, "--check") == 0


def test_a_damaged_flags_file_exits_2(setup, capsys):
    run(setup)
    setup[3].flags_file.write_text("{", encoding="utf-8")
    capsys.readouterr()
    assert run(setup) == 2
    out = capsys.readouterr().out
    assert out.startswith("CONFIG ERROR: ") and "python -m scripts.finqa_gold_check" in out


def test_missing_manifest_or_snapshot_exits_2(tmp_path, capsys):
    cfg, data_cfg, _ = build_frozen_split(tmp_path, write=False)
    prompt, config = load_prompt(root=tmp_path), load_evaluation_config(root=tmp_path)
    arguments = {"cfg": cfg, "data_cfg": data_cfg, "prompt": prompt, "config": config}
    assert finqa_gold_check.main([], **arguments) == 2
    assert "python -m scripts.finqa_split" in capsys.readouterr().out

    cfg, data_cfg, _ = build_frozen_split(tmp_path / "other")
    for item in data_cfg.files:
        (snapshot_path(data_cfg) / item.path).unlink()
    arguments.update(cfg=cfg, data_cfg=data_cfg)
    assert finqa_gold_check.main([], **arguments) == 2
    assert "python -m scripts.finqa_fetch" in capsys.readouterr().out
    assert not config.flags_file.exists() and not config.report_dir.exists()


def test_a_manifest_for_other_data_fails_the_data_check(tmp_path, capsys):
    cfg, _, prompt, config = build(tmp_path / "first")
    _, other_data, _, _ = build(tmp_path / "second", broken_id="AAA/2016/page_1.pdf-1")
    capsys.readouterr()
    result = finqa_gold_check.main([], cfg=cfg, data_cfg=other_data, prompt=prompt, config=config)
    assert result == 1
    assert "FAIL  K6  Same data: " in capsys.readouterr().out


def test_runs_as_a_module():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.finqa_gold_check", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # 0 with a fetched snapshot and current files; 2 where there is no snapshot (CI).
    assert result.returncode in (0, 2), result.stdout + result.stderr
    assert "Traceback" not in result.stderr


def test_running_by_file_path_gives_a_hint_not_a_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "finqa_gold_check.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "python -m scripts.finqa_gold_check" in result.stderr
    assert "Traceback" not in result.stderr
