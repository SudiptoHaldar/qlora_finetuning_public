"""``scripts/comparison_report.py``: check, build, and the local examples file.

Most tests give the script the invented inputs of ``tests/comparison/report_helpers.py``.
One makes six runs on the invented split and lets the script gather them itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qlora_finetuning.comparison.gate import write_canonical
from qlora_finetuning.comparison.runs import build_record as build_six_arm_record
from qlora_finetuning.data.bundle import BUNDLE_ENV, export_bundle
from scripts import comparison_report
from tests.comparison import report_helpers as r
from tests.comparison.helpers import (
    PREDICTORS,
    FakeGit,
    arms_now,
    comparison_cfg,
    freeze_and_unlock,
    job_record_for,
    run_arm,
)
from tests.evaluation.conftest import ids_of, make_context


@pytest.fixture
def cfg(tmp_path):
    return r.report_cfg(tmp_path)


def run(argv, cfg, tmp_path, **stand_ins):
    stand_ins.setdefault("inputs", r.inputs())
    stand_ins.setdefault("deviations", r.DEVIATIONS)
    return comparison_report.main(
        argv, root=tmp_path, cfg=cfg, report_dir=tmp_path / "reports" / "comparison", **stand_ins
    )


def test_check_prints_the_checks_and_the_figures_and_writes_nothing(cfg, tmp_path, capsys):
    assert run(["check"], cfg, tmp_path) == 0
    out = capsys.readouterr().out
    assert "PASS  R1  Records in place" in out and "PASS  R8  No text, nothing by hand" in out
    assert "8 passed, 0 failed, 1 with a reservation" in out
    assert "arm  A1       execution accuracy  70.00%" in out
    assert "pair A1 minus A0" in out and "+69.00 points" in out and "settled" in out
    assert "tuned arms: 600 right in all, 750 in at least one; 250 right in no arm" in out
    assert "PROVISIONAL" in out and "CHECK ONLY: nothing was written" in out
    assert not (tmp_path / "reports").exists() and not (tmp_path / "README.md").exists()
    assert out.isascii()


def test_build_writes_the_record_the_chart_the_report_and_the_readme(cfg, tmp_path, capsys):
    assert run(["build"], cfg, tmp_path) == 0
    out = capsys.readouterr().out
    for name in (
        "reports/comparison/comparison_statistics.json",
        "reports/comparison/comparison_statistics.md",
        "reports/comparison/accuracy_cost.svg",
        "reports/comparison_report.md",
        "README.md",
    ):
        assert f"written: {name}" in out and (tmp_path / name).is_file()
    stored = json.loads(
        (tmp_path / "reports/comparison/comparison_statistics.json").read_text(encoding="utf-8")
    )
    assert stored["passed"] and stored["provisional"] and len(stored["checks"]) == 8
    report = (tmp_path / "reports/comparison_report.md").read_text(encoding="utf-8")
    assert report.startswith("# The comparison (provisional)")
    assert "## Results (provisional)" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_a_final_report_is_not_called_provisional(cfg, tmp_path, capsys):
    final = r.inputs(cost=r.cost_record(final=True))
    assert run(["build"], cfg, tmp_path, inputs=final) == 0
    out = capsys.readouterr().out
    assert "PROVISIONAL" not in out and "8 passed, 0 failed, 0 with a reservation" in out


def test_a_failed_check_writes_nothing_and_exits_1(cfg, tmp_path, capsys):
    assert run(["build"], cfg, tmp_path, inputs=r.inputs(cost=None)) == 1
    out = capsys.readouterr().out
    assert "FAIL  R1" in out and "FAIL  R8" in out and "NOT WRITTEN" in out
    assert not (tmp_path / "reports").exists()


def test_a_pair_that_is_not_settled_is_printed_as_such(cfg, tmp_path, capsys):
    lite = r.outcomes("B1-Lite", right=range(0, 660))
    arms = r.all_outcomes(**{"B1-Lite": lite})
    assert run(["check"], cfg, tmp_path, inputs=r.inputs(arms=arms, six_arm=r.six_arm(arms))) == 1
    out = capsys.readouterr().out  # R7 fails: the invented cost record has the other counts
    assert "NOT SETTLED" in out and "FAIL  R7" in out


def test_examples_without_a_statistics_record_exits_2(cfg, tmp_path, capsys):
    assert run(["examples"], cfg, tmp_path) == 2
    assert "CONFIG ERROR: there is no statistics record yet" in capsys.readouterr().out
    assert not (tmp_path / "artifacts").exists()


# --------------------------------------------------------------------------- with real runs


@pytest.fixture
def context(tmp_path, monkeypatch):
    """A frozen, unlocked invented split with all six arms run on its 22 test examples."""
    monkeypatch.delenv(BUNDLE_ENV, raising=False)
    context = make_context(tmp_path)
    freeze_and_unlock(context)
    folder = tmp_path / "bundle_test"
    export_bundle(folder, ("test",), cfg=context.split_cfg, data_cfg=context.data_cfg)
    monkeypatch.setenv(BUNDLE_ENV, str(folder))
    return context


def test_the_script_gathers_the_runs_the_six_arm_record_names(context, cfg, tmp_path, capsys):
    runs = {arm: run_arm(arm, context)["run_id"] for arm in PREDICTORS}
    six_arm = build_six_arm_record(
        comparison_cfg(context),
        context,
        job_record=job_record_for(runs),
        arms_now=arms_now(),
        git=FakeGit(),
    )
    assert six_arm["passed"]
    write_canonical(cfg.six_arm_file, six_arm)
    context.comparison_cfg = comparison_cfg(context)

    inputs = comparison_report.gather(cfg, tmp_path, context)
    assert list(inputs.arms) == sorted(six_arm["arms"]) and inputs.read_problems == []
    assert inputs.test_ids == tuple(ids_of("test", context))
    assert inputs.tuned == ("A1", "B1-Lite", "B1-Pro") and inputs.cost is None
    assert inputs.freeze_sha256 == six_arm["freeze_sha256"]

    code = comparison_report.main(
        ["check"],
        root=tmp_path,
        cfg=cfg,
        report_dir=tmp_path / "reports" / "comparison",
        context=context,
        deviations=r.DEVIATIONS,
    )
    out = capsys.readouterr().out
    assert code == 1 and "FAIL  R1  Records in place: there is no cost record" in out
    for passed in ("R2  Replies found", "R3  Counts agree", "R4  Nothing frozen has changed"):
        assert f"PASS  {passed}" in out


def test_examples_writes_the_text_under_artifacts(context, cfg, tmp_path, capsys):
    run_id = run_arm("B0-Lite", context)["run_id"]
    first = ids_of("test", context)[0]
    stored = {
        "arms": {
            "B0-Lite": {
                "label": "Nova Lite, untuned",
                "run_id": run_id,
                "categories": {"order": 1},
                "example_ids": {"order": [first]},
            }
        }
    }
    folder = tmp_path / "reports" / "comparison"
    folder.mkdir(parents=True)
    (folder / "comparison_statistics.json").write_text(json.dumps(stored), encoding="utf-8")
    assert run(["examples"], cfg, tmp_path, context=context) == 0
    out = capsys.readouterr().out
    assert "written: artifacts/comparison/error_examples.md" in out and "do not commit" in out
    text = (tmp_path / "artifacts/comparison/error_examples.md").read_text(encoding="utf-8")
    assert f"**{first}**" in text and "subtract(920, 95)" in text


# --------------------------------------------------------------------------- guards


def test_the_script_calls_no_aws_service_and_asks_no_model():
    source = Path(comparison_report.__file__).read_text(encoding="utf-8")
    assert "boto" not in source and "run_evaluation" not in source and "Predictor" not in source
