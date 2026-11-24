"""Tests for the comparison report's settings (req_005 v5.03)."""

from __future__ import annotations

import copy
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from qlora_finetuning.comparison.config import load_comparison_config
from qlora_finetuning.comparison.report_config import load_deviations, load_report_config
from qlora_finetuning.settings import CONFIG_DIR, REPO_ROOT, ConfigError

REAL = yaml.safe_load((CONFIG_DIR / "report.yaml").read_text(encoding="utf-8"))


def changed(tmp_path, change) -> Path:
    """A config folder whose ``report.yaml`` is the real one after ``change(raw)``."""
    raw = copy.deepcopy(REAL)
    change(raw)
    folder = tmp_path / "config"
    folder.mkdir(exist_ok=True)
    (folder / "report.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    return folder


def test_the_committed_settings_load(tmp_path):
    cfg = load_report_config(root=tmp_path)
    assert (cfg.resamples, cfg.seed, cfg.confidence_percent) == (10000, 20261115, 95)
    assert cfg.six_arm_file == tmp_path / "reports" / "comparison" / "six_arm_test_runs.json"
    assert cfg.cost_file == tmp_path / "reports" / "comparison" / "cost_model.json"
    assert cfg.report_file == tmp_path / "reports" / "comparison_report.md"
    assert cfg.readme_file == tmp_path / "README.md"
    assert cfg.examples_file == tmp_path / "artifacts" / "comparison" / "error_examples.md"
    assert cfg.statistics_stem == "comparison_statistics" and cfg.chart_name == "accuracy_cost.svg"
    assert cfg.step_groups == (1, 2, 3) and cfg.examples_per_category == 3
    assert cfg.smallest_p == Decimal("0.0001")
    assert cfg.label("A1") == "Qwen3-4B with QLoRA" and cfg.label("Z9") == "Z9"


def test_the_pairs_and_the_chart_name_the_arms_of_the_comparison(tmp_path):
    cfg = load_report_config(root=tmp_path)
    arms = load_comparison_config().arms
    assert len(cfg.pairs) == 6 and cfg.pairs[0] == ("A1", "A0")
    assert {arm for pair in cfg.pairs for arm in pair} <= set(arms)
    charted = [arm for family in cfg.chart_families.values() for arm in family["arms"]]
    assert sorted(charted) == sorted(arms) and set(cfg.arm_labels) == set(arms)
    assert cfg.chart_families["Qwen3-4B"] == {"color": "#2a78d6", "arms": ("A0", "A1")}
    tuned = {arm for arm, entry in arms.items() if entry.tuned}
    assert all(first in tuned for first, _ in cfg.pairs)  # a difference is tuned minus the other


def test_the_seed_is_the_projects_seed():
    split = yaml.safe_load((CONFIG_DIR / "split.yaml").read_text(encoding="utf-8"))
    assert load_report_config().seed == split["seed"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: raw["outputs"].update(examples="reports/examples.md"), "under artifacts/"),
        (lambda raw: raw["outputs"].update(examples="artifacts/../x.md"), "under artifacts/"),
        (lambda raw: raw["bootstrap"].update(resamples=10), "100 or more"),
        (lambda raw: raw["bootstrap"].update(confidence_percent=100), "from 50 to 99"),
        (lambda raw: raw["bootstrap"].update(seed="x"), "whole number"),
        (lambda raw: raw.update(pairs="A1"), "must list pairs"),
        (lambda raw: raw.update(pairs=[["A1"]]), "must name two arms"),
        (lambda raw: raw.update(pairs=[["A1", "A1"]]), "names A1 twice"),
        (lambda raw: raw.update(pairs=[["A1", "A0"], ["A0", "A1"]]), "same two arms twice"),
        (lambda raw: raw.update(step_groups=[2, 1]), "must rise"),
        (lambda raw: raw.update(smallest_p=0.0001), "quoted text"),
        (lambda raw: raw.update(smallest_p="2"), "quoted text"),
        (lambda raw: raw.update(arm_labels={}), "arm_labels must map"),
        (lambda raw: raw["chart"].update(width=10), "300 or more"),
        (lambda raw: raw["chart"]["families"]["Nova Pro"].update(color="green"), "must be like"),
        (lambda raw: raw["chart"]["families"]["Nova Pro"].update(arms=["A0"]), "each once"),
        (lambda raw: raw.pop("pairs"), "pairs"),
    ],
)
def test_a_bad_setting_is_named(tmp_path, change, message):
    with pytest.raises(ConfigError, match=message):
        load_report_config(root=tmp_path, config_dir=changed(tmp_path, change))


def test_the_committed_deviations_load_and_point_at_files_that_exist():
    deviations = load_deviations()
    assert len(deviations) >= 10
    for entry in deviations:
        assert set(entry) == {"date", "what", "recorded_in"}
        assert entry["date"].startswith("2026-") and entry["what"].endswith(".")
        assert (REPO_ROOT / entry["recorded_in"]).is_file(), entry["recorded_in"]
    assert [entry["date"] for entry in deviations] == sorted(e["date"] for e in deviations)


@pytest.mark.parametrize(
    "content",
    [
        "version: 1\ndeviations: nothing\n",
        "version: 1\ndeviations:\n  - date: '2026-01-01'\n    what: 'x'\n",
        "version: 1\ndeviations:\n  - date: 2026-01-01\n    what: 'x'\n    recorded_in: 'y'\n",
    ],
)
def test_a_bad_deviations_file_is_refused(tmp_path, content):
    (tmp_path / "report_deviations.yaml").write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match="report_deviations.yaml"):
        load_deviations(config_dir=tmp_path)
