"""Tests for qlora_finetuning.frontier.config: the committed file and each bad value."""

from __future__ import annotations

import copy
import json
from decimal import Decimal

import pytest
import yaml

from qlora_finetuning.frontier.config import (
    load_aws_settings,
    load_bucket,
    load_frontier_config,
)
from qlora_finetuning.settings import CONFIG_DIR, ConfigError
from scripts.evaluate import PREDICTORS

GOOD = {
    "version": 4,
    "models": {
        "nova-pro": {
            "model_id": "amazon.nova-pro-v1:0",
            "untuned_arm": "B0-Pro",
            "customization_model_id": "amazon.nova-pro-v1:0:300k",
            "tuned_arm": "B1-Pro",
            "requests_per_minute": 18,
        },
        "nova-lite": {
            "model_id": "amazon.nova-lite-v1:0",
            "untuned_arm": "B0-Lite",
            "customization_model_id": "amazon.nova-lite-v1:0:300k",
            "tuned_arm": "B1-Lite",
        },
    },
    "inference": {"temperature": 0, "top_k": 1},
    "client": {
        "retry_mode": "adaptive",
        "max_attempts": 12,
        "connect_timeout": 10,
        "read_timeout": 120,
    },
    "report_dir": "reports/frontier",
    "customization": {
        "epoch_count": 2,
        "learning_rate": "0.00001",
        "learning_rate_warmup_steps": 10,
        "launch_order": ["nova-lite", "nova-pro"],
        "name_prefix": "qlora-finqa",
        "output_prefix": "bedrock/",
        "poll_seconds": 60,
        "tags": {"project": "qlora-finetuning"},
    },
    "export": {
        "schema_version": "bedrock-conversation-2024",
        "dir": "artifacts/bedrock",
        "s3_prefix": "data/bedrock/",
        "min_records": 8,
        "max_records": 20000,
        "max_sample_tokens": 32000,
    },
    "budget": {"path_b_cap_usd": 200},
    "estimate": {"assumed_token_ratio": 1.5, "margin_percent": 10, "storage_months": 1},
    "tuned_evaluation": {
        "splits": ["quick_dev", "dev"],
        "verdict_split": "dev",
        "min_execution_gain_points": 10,
        "min_format_validity_percent": 98,
    },
}


def write(tmp_path, content, name="frontier"):
    (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(content), encoding="utf-8")
    return tmp_path


def changed(path: str, value):
    content = copy.deepcopy(GOOD)
    node = content
    *parents, last = path.split(".")
    for part in parents:
        node = node[part]
    node[last] = value
    return content


def test_the_committed_config_loads_with_the_in_region_model_ids(tmp_path):
    cfg = load_frontier_config(root=tmp_path)
    assert list(cfg.models) == ["nova-pro", "nova-lite"]
    assert cfg.model("nova-pro").model_id == "amazon.nova-pro-v1:0"
    assert cfg.model("nova-lite").model_id == "amazon.nova-lite-v1:0"
    assert (cfg.temperature, cfg.top_k) == (0, 1)
    assert (cfg.retry_mode, cfg.max_attempts) == ("adaptive", 12)
    assert (cfg.connect_timeout, cfg.read_timeout) == (10, 120)
    assert cfg.report_dir == tmp_path / "reports" / "frontier"


def test_the_committed_config_holds_the_owners_decisions_for_the_gate(tmp_path):
    cfg = load_frontier_config(root=tmp_path)
    assert cfg.epoch_count == 2  # Bedrock's default for both models
    assert cfg.path_b_cap_usd == Decimal("200") and isinstance(cfg.path_b_cap_usd, Decimal)
    assert cfg.schema_version == "bedrock-conversation-2024"
    assert cfg.export_dir == tmp_path / "artifacts" / "bedrock"  # git-ignored
    assert cfg.s3_prefix == "data/bedrock/"  # under data/, which the Bedrock role may read
    assert (cfg.min_records, cfg.max_records, cfg.max_sample_tokens) == (8, 20000, 32000)
    assert cfg.assumed_token_ratio == Decimal("1.5") and cfg.margin_percent == Decimal("10")
    assert cfg.storage_months == 1


def test_the_committed_config_holds_the_owners_decisions_for_the_jobs():
    cfg = load_frontier_config()
    # Bedrock's defaults for both models, sent explicitly and as text.
    assert cfg.hyperparameters == {
        "epochCount": "2",
        "learningRate": "0.00001",
        "learningRateWarmupSteps": "10",
    }
    assert cfg.launch_order == ("nova-lite", "nova-pro")  # the cheaper job first
    assert (cfg.name_prefix, cfg.output_prefix, cfg.poll_seconds) == (
        "qlora-finqa",
        "bedrock/",
        60,
    )
    assert cfg.tags == {"project": "qlora-finetuning"}
    lite = cfg.model("nova-lite")
    assert (lite.customization_model_id, lite.tuned_arm) == (
        "amazon.nova-lite-v1:0:300k",
        "B1-Lite",
    )
    assert cfg.model("nova-pro").tuned_arm == "B1-Pro"


def test_the_customization_ids_are_the_ones_aws_check_asks_about():
    aws = yaml.safe_load((CONFIG_DIR / "aws.yaml").read_text(encoding="utf-8"))
    models = load_frontier_config().models.values()
    assert sorted(model.customization_model_id for model in models) == sorted(
        aws["bedrock"]["fine_tunable_models"]
    )
    for model in models:  # the customization id is the inference id plus a suffix
        assert model.customization_model_id.startswith(model.model_id + ":")


def test_the_tuned_arms_and_the_role_are_known(tmp_path):
    from qlora_finetuning.frontier.config import load_customization_role
    from qlora_finetuning.tracking import load_tracking_config

    arms = load_tracking_config().arms
    for model in load_frontier_config().models.values():
        assert {model.untuned_arm, model.tuned_arm} <= arms
    assert load_customization_role() == (
        "arn:aws:iam::123456789012:role/qlora-bedrock-customization"
    )
    with pytest.raises(ConfigError, match="missing: roles.bedrock_customization.arn"):
        load_customization_role(write(tmp_path, {"version": 1}, "aws"))


def test_the_committed_config_holds_the_owners_thresholds_for_the_tuned_models(tmp_path):
    cfg = load_frontier_config(root=tmp_path)
    assert cfg.tuned_splits == ("quick_dev", "dev") and cfg.verdict_split == "dev"
    assert cfg.min_execution_gain_points == Decimal("10")  # the owner, 2026-11-18
    assert cfg.min_format_validity_percent == Decimal("98")
    assert isinstance(cfg.min_execution_gain_points, Decimal)


def test_the_new_keys_do_not_change_what_decides_a_reply():
    """Runs made under version 1 of the file must still be resumable."""
    assert set(load_frontier_config().settings) == {
        "api", "temperature", "top_k", "top_p", "stop_sequences", "service_tier", "reasoning",
    }  # fmt: skip


def test_a_fractional_cap_is_exact(tmp_path):
    cfg = load_frontier_config(config_dir=write(tmp_path, changed("budget.path_b_cap_usd", 0.1)))
    assert cfg.path_b_cap_usd == Decimal("0.1")  # not 0.1000000000000000055...


def test_the_bucket_name_comes_from_the_aws_config(tmp_path):
    assert load_bucket() == "qlora-sagemaker-123456789012"
    with pytest.raises(ConfigError, match="missing: bucket.name"):
        load_bucket(write(tmp_path, {"version": 1, "profile": "p"}, "aws"))


def test_the_committed_file_equals_the_values_these_tests_use():
    committed = yaml.safe_load((CONFIG_DIR / "frontier.yaml").read_text(encoding="utf-8"))
    committed.pop("verified_on")
    assert committed == GOOD


def test_settings_hold_everything_that_can_change_a_reply_as_plain_values():
    settings = load_frontier_config().settings
    assert settings == {
        "api": "converse",
        "temperature": 0,
        "top_k": 1,
        "top_p": None,
        "stop_sequences": [],
        "service_tier": "default",
        "reasoning": "not sent",
    }
    assert json.loads(json.dumps(settings)) == settings


def test_the_script_entries_use_the_configured_arms():
    cfg = load_frontier_config()
    for name, model in cfg.models.items():
        entry = PREDICTORS[name]
        assert (entry.path, entry.arm, entry.where) == ("B", model.untuned_arm, "bedrock")
        assert entry.configs == ("frontier", "aws")
        tuned = PREDICTORS[f"{name}-tuned"]
        assert (tuned.path, tuned.arm, tuned.where) == ("B", model.tuned_arm, "bedrock")
        assert tuned.configs == ("frontier", "aws")


def test_an_unknown_model_names_the_known_ones():
    with pytest.raises(ConfigError, match="no model named 'nova-max'. Known: nova-lite, nova-pro"):
        load_frontier_config().model("nova-max")


def test_the_lowest_documented_temperature_is_accepted(tmp_path):
    cfg = load_frontier_config(config_dir=write(tmp_path, changed("inference.temperature", 1e-5)))
    assert cfg.temperature == 0.00001 and cfg.settings["temperature"] == 0.00001


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("inference.temperature", True, "inference.temperature must be a number from 0 to 1"),
        ("inference.temperature", 1.5, "inference.temperature must be a number from 0 to 1"),
        ("inference.temperature", -0.1, "inference.temperature must be a number from 0 to 1"),
        ("inference.temperature", "0", "inference.temperature must be a number from 0 to 1"),
        ("inference.top_k", 1.0, "inference.top_k must be a whole number of 0 or more"),
        ("inference.top_k", -1, "inference.top_k must be a whole number of 0 or more"),
        ("inference.top_k", 129, "inference.top_k must be at most 128"),
        ("client.retry_mode", "eager", "client.retry_mode must be one of: standard"),
        ("client.max_attempts", 0, "client.max_attempts must be a whole number of 1 or more"),
        ("client.max_attempts", True, "client.max_attempts must be a whole number of 1 or more"),
        ("client.read_timeout", 0, "client.read_timeout must be a whole number of 1 or more"),
        ("client.connect_timeout", "10", "client.connect_timeout must be a whole number"),
        ("models", ["nova-pro"], "models must be a mapping of name to model"),
        ("models.nova-pro", {"model_id": "x"}, "models.nova-pro needs text for: untuned_arm"),
        ("models.nova-lite", "amazon.nova-lite-v1:0", "models.nova-lite needs text for: model_id"),
        ("customization.epoch_count", 0, "epoch_count must be a whole number of 1 or more"),
        ("customization.epoch_count", 6, "customization.epoch_count must be at most 5"),
        ("customization.epoch_count", 2.0, "epoch_count must be a whole number of 1 or more"),
        ("customization.learning_rate", 0.00001, "learning_rate must be quoted text"),
        ("customization.learning_rate", "fast", "learning_rate is not a number"),
        ("customization.learning_rate", "0", "must be above 0 and at most 0.0001"),
        ("customization.learning_rate", "0.001", "must be above 0 and at most 0.0001"),
        ("customization.learning_rate_warmup_steps", 101, "must be at most 100"),
        ("customization.learning_rate_warmup_steps", -1, "whole number of 0 or more"),
        ("customization.launch_order", ["nova-lite"], "must name every model once"),
        ("customization.launch_order", "nova-lite", "must name every model once"),
        ("customization.launch_order", ["nova-lite", "nova-lite"], "every model once"),
        ("customization.name_prefix", "QLoRA_FinQA", "only lower-case letters"),
        ("customization.output_prefix", "bedrock", "output_prefix must end with"),
        ("customization.poll_seconds", 0, "poll_seconds must be a whole number"),
        ("customization.tags", {"project": 7}, "tags must map text to text"),
        ("customization.tags", ["project"], "tags must map text to text"),
        ("export.schema_version", 2024, "export.schema_version must be text"),
        ("export.dir", "  ", "export.dir must be text"),
        ("export.s3_prefix", "data/bedrock", "export.s3_prefix must end with '/'"),
        ("export.s3_prefix", "/data/bedrock/", "export.s3_prefix must end with '/'"),
        ("export.min_records", 0, "export.min_records must be a whole number of 1 or more"),
        ("export.max_records", 7, "export.max_records must be a whole number of 8 or more"),
        ("export.max_sample_tokens", 0, "export.max_sample_tokens must be a whole number"),
        ("budget.path_b_cap_usd", "200", "budget.path_b_cap_usd must be a number"),
        ("budget.path_b_cap_usd", -1, "budget.path_b_cap_usd must be a number of 0 or more"),
        ("budget.path_b_cap_usd", True, "budget.path_b_cap_usd must be a number"),
        ("estimate.assumed_token_ratio", 0, "assumed_token_ratio must be a number of 0.1 or more"),
        ("estimate.margin_percent", -5, "estimate.margin_percent must be a number of 0 or more"),
        ("estimate.storage_months", 1.5, "estimate.storage_months must be a whole number"),
        ("tuned_evaluation.splits", "dev", "splits must list split names, each once"),
        ("tuned_evaluation.splits", [7], "splits must list split names, each once"),
        ("tuned_evaluation.splits", ["dev", "dev"], "splits must list split names, each once"),
        ("tuned_evaluation.splits", ["dev", "test"], "must not name test"),
        ("tuned_evaluation.verdict_split", "fit", "verdict_split must be one of"),
        ("tuned_evaluation.min_execution_gain_points", -1, "must be a number of 0 or more"),
        ("tuned_evaluation.min_execution_gain_points", "10", "must be a number"),
        ("tuned_evaluation.min_format_validity_percent", 101, "must be at most 100"),
    ],
)
def test_a_bad_value_is_a_config_error_naming_the_key(tmp_path, path, value, message):
    with pytest.raises(ConfigError, match=message):
        load_frontier_config(config_dir=write(tmp_path, changed(path, value)))


def test_a_missing_key_is_named(tmp_path):
    content = copy.deepcopy(GOOD)
    del content["inference"]["top_k"]
    del content["report_dir"]
    with pytest.raises(ConfigError, match="missing: inference.top_k, report_dir"):
        load_frontier_config(config_dir=write(tmp_path, content))


def test_a_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="config file not found"):
        load_frontier_config(config_dir=tmp_path)


def test_aws_settings_are_text(tmp_path):
    assert load_aws_settings() == {
        "profile": "qlora",
        "account_id": "123456789012",
        "region": "us-east-1",
    }
    aws = {"version": 1, "profile": "p", "account_id": 123456789012, "region": "us-east-1"}
    assert load_aws_settings(write(tmp_path, aws, "aws"))["account_id"] == "123456789012"
    del aws["region"]
    with pytest.raises(ConfigError, match="missing: region"):
        load_aws_settings(write(tmp_path, aws, "aws"))


def test_a_model_may_be_paced_and_the_pace_must_be_a_whole_number(tmp_path):
    cfg = load_frontier_config(config_dir=write(tmp_path, GOOD))
    assert cfg.model("nova-pro").requests_per_minute == 18
    assert cfg.model("nova-lite").requests_per_minute is None
    assert "requests_per_minute" not in cfg.settings  # pacing is not part of a run's settings
    for bad in (0, -1, 2.5, True, "18"):
        content = copy.deepcopy(GOOD)
        content["models"]["nova-pro"]["requests_per_minute"] = bad
        with pytest.raises(ConfigError, match="requests_per_minute must be a whole number"):
            load_frontier_config(config_dir=write(tmp_path, content))
