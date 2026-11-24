"""Tests for qlora_finetuning.frontier.pricing: the committed prices and exact arithmetic."""

from __future__ import annotations

import copy
from decimal import Decimal

import pytest
import yaml

from qlora_finetuning.frontier.config import load_frontier_config
from qlora_finetuning.frontier.pricing import inference_cost, load_pricing
from qlora_finetuning.settings import ConfigError

PRO, LITE = "amazon.nova-pro-v1:0", "amazon.nova-lite-v1:0"
GOOD = {
    "version": 1,
    "price_list": {
        "offer": "AmazonBedrock",
        "region": "us-east-1",
        "published": "2026-11-11",
        "read_on": "2026-11-16",
    },
    "bedrock": {
        PRO: {
            "input_per_million": "0.80",
            "output_per_million": "3.20",
            "training_per_million": "8.00",
            "storage_per_model_month": "1.95",
        }
    },
}


def write(tmp_path, content):
    (tmp_path / "pricing.yaml").write_text(yaml.safe_dump(content), encoding="utf-8")
    return tmp_path


def test_the_committed_prices_are_those_of_the_price_list_of_2026_11_11():
    pricing = load_pricing()
    assert (pricing.offer, pricing.region) == ("AmazonBedrock", "us-east-1")
    assert (pricing.published, pricing.read_on) == ("2026-11-11", "2026-11-16")
    pro, lite = pricing.model(PRO), pricing.model(LITE)
    assert (pro.input_per_million, pro.output_per_million) == (Decimal("0.80"), Decimal("3.20"))
    assert (lite.input_per_million, lite.output_per_million) == (Decimal("0.06"), Decimal("0.24"))
    assert (pro.training_per_million, lite.training_per_million) == (Decimal("8"), Decimal("2"))
    assert pro.storage_per_model_month == lite.storage_per_model_month == Decimal("1.95")


def test_every_configured_model_has_a_price():
    pricing = load_pricing()
    for model in load_frontier_config().models.values():
        assert pricing.model(model.model_id).input_per_million > 0


def test_cost_is_exact_decimal_arithmetic():
    pricing = load_pricing()
    cost = inference_cost(PRO, 128_495, 6_000, pricing)
    assert isinstance(cost, Decimal) and cost == Decimal("0.121996")
    assert inference_cost(LITE, 1_000_000, 1_000_000, pricing) == Decimal("0.30")
    assert inference_cost(LITE, 1, 0, pricing) == Decimal("0.00000006")
    assert inference_cost(PRO, 0, 0, pricing) == 0
    assert inference_cost(PRO, 10, 10) == Decimal("0.00004")  # loads the committed file


def test_an_unknown_model_is_a_config_error_naming_the_known_ones():
    with pytest.raises(ConfigError, match="no prices for 'amazon.nova-premier-v1:0'. Known: "):
        inference_cost("amazon.nova-premier-v1:0", 1, 1, load_pricing())


@pytest.mark.parametrize("tokens", [-1, 1.5, True, None, "10"])
def test_a_token_count_must_be_a_whole_number(tokens):
    with pytest.raises(ConfigError, match="must be a whole number of zero or more"):
        inference_cost(PRO, tokens, 1, load_pricing())
    with pytest.raises(ConfigError, match="output_tokens must be a whole number"):
        inference_cost(PRO, 1, tokens, load_pricing())


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (0.8, "must be quoted text"),
        (8, "must be quoted text"),
        (None, "must be quoted text"),
        ("cheap", "is not a number"),
        ("-0.80", "must be zero or more"),
        ("NaN", "must be zero or more"),
        ("Infinity", "must be zero or more"),
    ],
)
def test_a_price_that_is_not_quoted_decimal_text_is_a_config_error(tmp_path, value, message):
    content = copy.deepcopy(GOOD)
    content["bedrock"][PRO]["input_per_million"] = value
    with pytest.raises(ConfigError, match=f"bedrock.{PRO}.input_per_million {message}"):
        load_pricing(write(tmp_path, content))


def test_a_missing_price_or_date_is_a_config_error(tmp_path):
    content = copy.deepcopy(GOOD)
    del content["bedrock"][PRO]["output_per_million"]
    with pytest.raises(ConfigError, match="output_per_million must be quoted text"):
        load_pricing(write(tmp_path, content))
    content = copy.deepcopy(GOOD)
    del content["price_list"]["published"]
    with pytest.raises(ConfigError, match="missing: price_list.published"):
        load_pricing(write(tmp_path, content))
    content = copy.deepcopy(GOOD)
    content["bedrock"] = ["0.80"]
    with pytest.raises(ConfigError, match="bedrock must be a mapping"):
        load_pricing(write(tmp_path, content))


def test_the_good_file_of_these_tests_loads(tmp_path):
    pricing = load_pricing(write(tmp_path, GOOD))
    assert list(pricing.models) == [PRO]
