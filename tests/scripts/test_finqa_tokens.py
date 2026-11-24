"""Tests for scripts.finqa_tokens, on an invented dataset with a stand-in tokenizer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from qlora_finetuning.data import load_prompt
from qlora_finetuning.data.token_budget import TokenBudgetError
from qlora_finetuning.data.token_budget_report import report_paths
from scripts import finqa_serialize, finqa_tokens
from tests.data.synthetic import FakeTokenizer, build_frozen_split, fake_budget

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def setup(tmp_path):
    """A frozen split, the serialization report it needs, and a budget for the stand-in."""
    cfg, data_cfg, _ = build_frozen_split(tmp_path)
    prompt = load_prompt(root=tmp_path)
    assert finqa_serialize.main([], cfg=cfg, data_cfg=data_cfg, prompt=prompt) == 0
    return cfg, data_cfg, prompt, fake_budget(tmp_path)


def run(setup, *arguments, budget=None, tokenizer=None):
    cfg, data_cfg, prompt, default_budget = setup
    return finqa_tokens.main(
        list(arguments),
        cfg=cfg,
        data_cfg=data_cfg,
        prompt=prompt,
        budget=budget or default_budget,
        tokenizer=tokenizer or FakeTokenizer(),
    )


def test_first_run_writes_the_report(setup, tmp_path, capsys):
    capsys.readouterr()
    assert run(setup) == 0
    out = capsys.readouterr().out
    assert out.count("PASS  ") == 6
    assert "PASS  T1  Fits: longest full sequence is 278 of 2,048 tokens" in out
    assert "6 passed, 0 failed" in out
    assert "fit: 30 records, full sequence p50 278 p99 278 max 278, 8,340 tokens in one pass" in out
    assert "quick_dev: 3 records" in out
    assert "written reports/data/finqa_token_budget_report.md" in out
    assert out.isascii() and str(tmp_path) not in out

    report = json.loads(report_paths(setup[3].report_dir)[1].read_text(encoding="utf-8"))
    assert report["splits"]["dev"]["records"] == 8
    assert report["inputs"]["prompt_id"] == setup[2].prompt_id


def test_second_run_changes_nothing_and_check_passes(setup, capsys):
    run(setup)
    files = report_paths(setup[3].report_dir)
    before = [path.read_bytes() for path in files]
    assert run(setup) == 0
    assert [path.read_bytes() for path in files] == before
    capsys.readouterr()
    assert run(setup, "--check") == 0
    assert "report files are current" in capsys.readouterr().out


def test_check_before_a_build_writes_nothing(setup, capsys):
    capsys.readouterr()
    assert run(setup, "--check") == 1
    out = capsys.readouterr().out
    assert "OUT OF DATE: finqa_token_budget_report.md, finqa_token_budget_report.json" in out
    assert not report_paths(setup[3].report_dir)[0].exists()


def test_record_over_the_length_fails_and_is_named(setup, tmp_path, capsys):
    capsys.readouterr()
    assert run(setup, budget=fake_budget(tmp_path, max_length=270)) == 1
    out = capsys.readouterr().out
    assert "FAIL  T1  Fits: 38 records are longer than 270 tokens: " in out
    assert "5 passed, 1 failed" in out
    assert "written reports/data/finqa_token_budget_report.json" in out


def test_wrong_tokenizer_fails_the_fingerprint_check(setup, capsys):
    capsys.readouterr()
    assert run(setup, tokenizer=FakeTokenizer({"other": 1})) == 1
    assert "FAIL  T3  Tokenizer: the tokenizer in use has fingerprint" in capsys.readouterr().out


def test_missing_serialization_report_fails_the_same_records_check(tmp_path, capsys):
    cfg, data_cfg, _ = build_frozen_split(tmp_path)
    code = finqa_tokens.main(
        [],
        cfg=cfg,
        data_cfg=data_cfg,
        prompt=load_prompt(root=tmp_path),
        budget=fake_budget(tmp_path),
        tokenizer=FakeTokenizer(),
    )
    assert code == 1
    assert "FAIL  T6  Same records: there is no serialization report" in capsys.readouterr().out


def test_missing_tokenizer_exits_2_with_the_fix(setup, monkeypatch, capsys):
    cfg, data_cfg, prompt, budget = setup
    seen = []

    def no_tokenizer(budget_cfg, download=False):
        seen.append(download)
        raise TokenBudgetError("the tokenizer could not be loaded; fetch it once with: ...")

    monkeypatch.setattr(finqa_tokens, "load_tokenizer", no_tokenizer)
    monkeypatch.setattr(finqa_tokens, "setup_logging", lambda: None)
    capsys.readouterr()
    for arguments, download in (([], False), (["--download"], True)):
        code = finqa_tokens.main(
            arguments, cfg=cfg, data_cfg=data_cfg, prompt=prompt, budget=budget
        )
        assert code == 2
        assert capsys.readouterr().out.startswith("CONFIG ERROR: the tokenizer could not be loaded")
        assert seen[-1] is download
    assert not report_paths(budget.report_dir)[0].exists()


def test_missing_manifest_exits_2(tmp_path, capsys):
    cfg, data_cfg, _ = build_frozen_split(tmp_path, write=False)
    code = finqa_tokens.main(
        [],
        cfg=cfg,
        data_cfg=data_cfg,
        prompt=load_prompt(root=tmp_path),
        budget=fake_budget(tmp_path),
        tokenizer=FakeTokenizer(),
    )
    assert code == 2
    assert "python -m scripts.finqa_split" in capsys.readouterr().out


def test_runs_as_a_module():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.finqa_tokens", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # 0 with the tokenizer, the snapshot, and a current report; 2 where one is missing (CI).
    assert result.returncode in (0, 2), result.stdout + result.stderr
    assert "Traceback" not in result.stderr


def test_running_by_file_path_gives_a_hint_not_a_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "finqa_tokens.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "python -m scripts.finqa_tokens" in result.stderr
    assert "Traceback" not in result.stderr
