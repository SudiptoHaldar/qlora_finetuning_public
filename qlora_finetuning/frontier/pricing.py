"""Prices from ``config/pricing.yaml`` and the cost of an inference run (req_004 v4.01).

The ledger holds quantities only. A dollar amount is computed here, as a ``Decimal``, from
a token count and a price that carries the date of its price list.

Usage::

    from qlora_finetuning.frontier.pricing import inference_cost, load_pricing

    pricing = load_pricing()
    pricing.published                                                    # "2026-11-11"
    inference_cost("amazon.nova-pro-v1:0", 128_495, 6_000, pricing)      # Decimal("0.121996")
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from qlora_finetuning.settings import ConfigError, load_config

REQUIRED_KEYS = (
    "price_list.offer",
    "price_list.region",
    "price_list.published",
    "price_list.read_on",
    "bedrock",
)
PRICE_KEYS = (
    "input_per_million",
    "output_per_million",
    "training_per_million",
    "storage_per_model_month",
)
MILLION = Decimal(1_000_000)
FILE = "config pricing.yaml"


@dataclass(frozen=True)
class ModelPrice:
    """US dollars: per 1M tokens, and per stored custom model per month."""

    input_per_million: Decimal
    output_per_million: Decimal
    training_per_million: Decimal
    storage_per_model_month: Decimal


@dataclass(frozen=True)
class Pricing:
    offer: str
    region: str
    published: str
    read_on: str
    models: dict[str, ModelPrice]

    def model(self, model_id: str) -> ModelPrice:
        try:
            return self.models[model_id]
        except KeyError:
            known = ", ".join(sorted(self.models))
            raise ConfigError(f"{FILE}: no prices for {model_id!r}. Known: {known}") from None


def _price(model_id: str, key: str, value: Any) -> Decimal:
    where = f"{FILE}: bedrock.{model_id}.{key}"
    if not isinstance(value, str):
        raise ConfigError(f"{where} must be quoted text such as \"0.80\", got {value!r}")
    try:
        price = Decimal(value)
    except InvalidOperation:
        raise ConfigError(f"{where} is not a number: {value!r}") from None
    if not price.is_finite() or price < 0:
        raise ConfigError(f"{where} must be zero or more, got {value!r}")
    return price


def load_pricing(config_dir: Path | str | None = None) -> Pricing:
    """Read and validate ``config/pricing.yaml``."""
    raw = load_config("pricing", required=REQUIRED_KEYS, config_dir=config_dir)
    entries = raw["bedrock"]
    if not isinstance(entries, dict):
        raise ConfigError(f"{FILE}: bedrock must be a mapping of model id to prices")
    models = {}
    for model_id, entry in entries.items():
        entry = entry if isinstance(entry, dict) else {}
        models[str(model_id)] = ModelPrice(
            **{key: _price(str(model_id), key, entry.get(key)) for key in PRICE_KEYS}
        )
    price_list = raw["price_list"]
    return Pricing(
        offer=str(price_list["offer"]),
        region=str(price_list["region"]),
        published=str(price_list["published"]),
        read_on=str(price_list["read_on"]),
        models=models,
    )


def inference_cost(
    model_id: str, input_tokens: int, output_tokens: int, pricing: Pricing | None = None
) -> Decimal:
    """Input tokens and output tokens, each times its price. Exact; round only to show it."""
    for name, count in (("input_tokens", input_tokens), ("output_tokens", output_tokens)):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ConfigError(f"{name} must be a whole number of zero or more, got {count!r}")
    price = (pricing or load_pricing()).model(model_id)
    return (
        input_tokens * price.input_per_million + output_tokens * price.output_per_million
    ) / MILLION
