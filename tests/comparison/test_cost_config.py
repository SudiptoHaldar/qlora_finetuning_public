"""``config/cost_model.yaml`` and the SageMaker prices of ``config/pricing.yaml``."""

from __future__ import annotations

import shutil
from decimal import Decimal

import pytest

from qlora_finetuning.comparison.cost_config import load_cost_config, load_instance_prices
from qlora_finetuning.settings import REPO_ROOT, ConfigError


@pytest.fixture
def config_dir(tmp_path):
    """A copy of the two config files, to change."""
    folder = tmp_path / "config"
    folder.mkdir()
    for name in ("cost_model.yaml", "pricing.yaml"):
        shutil.copy(REPO_ROOT / "config" / name, folder / name)
    return folder


def change(config_dir, name, old, new):
    path = config_dir / name
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_the_repositorys_cost_config_loads(tmp_path):
    cfg = load_cost_config(root=tmp_path)
    assert cfg.report_stem == "cost_model"
    assert cfg.bill_file == tmp_path / "reports" / "comparison" / "aws_bill.json"
    assert set(cfg.record_files) == {"six_arm", "jobs", "selection", "tuning"}
    assert cfg.record_files["six_arm"].name == "six_arm_test_runs.json"
    assert cfg.hours_per_month == 730
    assert cfg.utilizations == (Decimal("1"), Decimal("0.5"), Decimal("0.1"), Decimal("0.01"))
    assert 2898 in cfg.batch_sizes and cfg.batch_jobs_per_month == 1
    assert cfg.storage_months == (1, 12)
    assert cfg.requests_per_minute == {"B0-Lite": 200, "B1-Lite": 200, "B0-Pro": 25, "B1-Pro": 20}
    assert cfg.services == {"bedrock": "bedrock", "sagemaker": "sagemaker"}
    assert (cfg.tolerance_usd, cfg.tolerance_percent) == (Decimal("0.50"), Decimal("2"))
    assert len(cfg.left_out) == 3


def test_the_cost_config_holds_no_price_and_no_run_id():
    text = (REPO_ROOT / "config" / "cost_model.yaml").read_text(encoding="utf-8")
    assert "per_million" not in text and "1.408" not in text
    assert "-eval-" not in text and "-train-" not in text


def test_the_hosting_price_is_in_the_price_file_with_its_own_date():
    prices = load_instance_prices()
    assert prices.training["ml.g5.xlarge"] == Decimal("1.408")
    assert prices.hosting["ml.g5.xlarge"] == Decimal("1.408")
    assert prices.training_published == "2026-11-14"
    assert prices.hosting_published == "2026-11-23"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('utilizations: ["1", "0.5", "0.1", "0.01"]', 'utilizations: ["1.5"]', "at most 1"),
        ('utilizations: ["1", "0.5", "0.1", "0.01"]', "utilizations: [1, 0.5]", "quoted text"),
        ("hours_per_month: 730", "hours_per_month: 0", "whole number of 1 or more"),
        ("batch_sizes: [100, 1000, 2898, 10000, 100000]", "batch_sizes: [100, 100]", "twice"),
        ('tolerance_usd: "0.50"', "tolerance_usd: 0.50", "quoted text"),
        ("    B1-Pro: 20", "    B1-Pro: many", "whole number"),
        ("  tuning: reports/frontier/nova_tuning.json", "  other: x.json", "must name exactly"),
    ],
)
def test_a_cost_config_that_makes_no_sense_is_refused(config_dir, tmp_path, old, new, message):
    change(config_dir, "cost_model.yaml", old, new)
    with pytest.raises(ConfigError, match=message):
        load_cost_config(root=tmp_path, config_dir=config_dir)


def test_a_price_file_without_a_hosting_price_is_refused(config_dir):
    change(config_dir, "pricing.yaml", "  hosting:\n", "  hosting_gone:\n")
    with pytest.raises(ConfigError, match="hosting"):
        load_instance_prices(config_dir=config_dir)


def test_a_hosting_price_must_be_quoted_text(config_dir):
    change(config_dir, "pricing.yaml", 'ml.g5.xlarge: "1.408"\n', "ml.g5.xlarge: 1.408\n")
    with pytest.raises(ConfigError, match="quoted text"):
        load_instance_prices(config_dir=config_dir)
