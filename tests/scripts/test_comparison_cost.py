"""``scripts/comparison_cost.py``: check, report, and the owner's bill entry.

The inputs are the invented ones of ``tests/comparison/cost_helpers.py``; the script's own
gathering of the repository's records is not run here.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from qlora_finetuning.comparison.cost_bill import read_bills
from qlora_finetuning.comparison.cost_config import load_cost_config
from scripts import comparison_cost
from tests.comparison import cost_helpers as h


@pytest.fixture
def cfg(tmp_path):
    return load_cost_config(root=tmp_path)


def run(argv, cfg, tmp_path, **stand_ins):
    stand_ins.setdefault("inputs", h.inputs())
    return comparison_cost.main(
        argv, root=tmp_path, cfg=cfg, report_dir=tmp_path / "reports" / "comparison", **stand_ins
    )


def test_check_prints_the_checks_and_the_figures_and_writes_nothing(cfg, tmp_path, capsys):
    assert run(["check"], cfg, tmp_path) == 0
    out = capsys.readouterr().out
    assert "PASS  K1  Six test runs" in out and "PASS  K8  Agrees with the bill" in out
    assert "8 passed, 0 failed, 2 with a reservation" in out
    assert "making    A1       training run $10.00 (list price); selection $2.00" in out
    assert "making    B1-Lite  training run $5.00 (billed); storage $2.00 a month" in out
    assert "per 1,000 questions $1.10 (steady $1.00)" in out
    assert "NOT REACHABLE HERE" in out
    assert "Path A: $15.60 so far of $100.00" in out
    assert "PROVISIONAL" in out and "CHECK ONLY: nothing was written" in out
    assert not (tmp_path / "reports").exists()
    assert out.isascii()


def test_report_writes_the_record_and_says_it_is_provisional(cfg, tmp_path, capsys):
    assert run(["report"], cfg, tmp_path) == 0
    out = capsys.readouterr().out
    assert "record: reports/comparison/cost_model.json" in out
    assert "record: reports/comparison/cost_model.md" in out
    folder = tmp_path / "reports" / "comparison"
    stored = json.loads((folder / "cost_model.json").read_text(encoding="utf-8"))
    assert stored["passed"] and stored["provisional"]
    text = (folder / "cost_model.md").read_text(encoding="utf-8")
    assert text.startswith("# The cost model (provisional)")


def test_a_final_record_is_not_called_provisional(cfg, tmp_path, capsys):
    final = h.inputs(pro_billed=True, bills=h.covering_bills())
    assert run(["report"], cfg, tmp_path, inputs=final) == 0
    out = capsys.readouterr().out
    assert "PROVISIONAL" not in out and "8 passed, 0 failed, 0 with a reservation" in out
    assert "commit both, with the bill file" in out


def test_a_failed_check_writes_nothing_and_exits_1(cfg, tmp_path, capsys):
    assert run(["report"], cfg, tmp_path, inputs=h.inputs(six_arm=None)) == 1
    out = capsys.readouterr().out
    assert "FAIL  K1" in out and "NOT WRITTEN" in out
    assert not (tmp_path / "reports").exists()


def test_the_owners_bill_entry_is_appended(cfg, tmp_path, capsys):
    argv = ["bill", "--service", "bedrock", "--usd", "78.19", "--through", "2026-11-22"]
    assert run(argv, cfg, tmp_path, today=date(2026, 11, 24)) == 0
    out = capsys.readouterr().out
    assert "ENTERED: bedrock $78.19 through 2026-11-22" in out
    assert "written: reports/comparison/aws_bill.json" in out
    assert read_bills(cfg.bill_file)[0]["usd"] == "78.19"


def test_a_refused_bill_entry_exits_2(cfg, tmp_path, capsys):
    argv = ["bill", "--service", "ec2", "--usd", "1", "--through", "2026-11-22"]
    assert run(argv, cfg, tmp_path, today=date(2026, 11, 24)) == 2
    assert "CONFIG ERROR: unknown service" in capsys.readouterr().out
    assert not cfg.bill_file.exists()


def test_the_script_calls_no_aws_service():
    source = Path(comparison_cost.__file__).read_text(encoding="utf-8")
    assert "boto" not in source and "load_split" not in source
