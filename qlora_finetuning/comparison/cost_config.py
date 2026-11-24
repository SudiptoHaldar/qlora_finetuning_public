"""The cost model's settings and the SageMaker instance prices (req_005 v5.02).

``config/cost_model.yaml`` says where the committed records are, how Path A's serving cost
is stated, what the account may send to each Nova arm, and how the model is compared with
the AWS bill. It holds no price and no run id. The instance prices are read from
``config/pricing.yaml``: the training rate that every job was billed at, and the hosting
rate of a real-time endpoint that the always-on scenario uses.

Usage::

    from qlora_finetuning.comparison.cost_config import load_cost_config, load_instance_prices

    cfg = load_cost_config()
    cfg.record_files["six_arm"], cfg.bill_file, cfg.utilizations, cfg.batch_sizes
    prices = load_instance_prices()
    prices.training["ml.g5.xlarge"], prices.hosting["ml.g5.xlarge"]     # Decimal("1.408")

Amounts are ``Decimal``. This module imports neither boto3 nor the training stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from qlora_finetuning.settings import REPO_ROOT, ConfigError, load_config

FILE = "config cost_model.yaml"
PRICING_FILE = "config pricing.yaml"
REQUIRED_KEYS = (
    "report_stem",
    "bill_file",
    "records",
    "serving.hours_per_month",
    "serving.utilizations",
    "serving.batch_sizes",
    "serving.batch_jobs_per_month",
    "storage_months",
    "account_limits.read_on",
    "account_limits.requests_per_minute",
    "reconciliation.services",
    "reconciliation.tolerance_usd",
    "reconciliation.tolerance_percent",
)
RECORDS = ("six_arm", "jobs", "selection", "tuning")
PRICING_KEYS = (
    "sagemaker.price_list.published",
    "sagemaker.training",
    "sagemaker.hosting.price_list.published",
    "sagemaker.hosting.instances",
)


@dataclass(frozen=True)
class CostConfig:
    """The typed content of ``config/cost_model.yaml``, with its paths resolved."""

    root: Path
    report_stem: str
    bill_file: Path
    record_files: dict[str, Path]
    hours_per_month: int
    utilizations: tuple[Decimal, ...]
    batch_sizes: tuple[int, ...]
    batch_jobs_per_month: int
    storage_months: tuple[int, ...]
    limits_read_on: str
    requests_per_minute: dict[str, int]
    services: dict[str, str]  # the bill's service -> the ledger's ``where``
    tolerance_usd: Decimal
    tolerance_percent: Decimal
    left_out: tuple[str, ...]


@dataclass(frozen=True)
class InstancePrices:
    """US dollars per instance and hour, by instance type, with the price lists' dates."""

    training: dict[str, Decimal]
    training_published: str
    hosting: dict[str, Decimal]
    hosting_published: str


def _amount(value: Any, where: str) -> Decimal:
    if not isinstance(value, str):
        raise ConfigError(f'{where} must be quoted text such as "0.50", got {value!r}')
    try:
        amount = Decimal(value)
    except InvalidOperation:
        raise ConfigError(f"{where} is not a number: {value!r}") from None
    if not amount.is_finite() or amount < 0:
        raise ConfigError(f"{where} must be zero or more, got {value!r}")
    return amount


def _whole(value: Any, where: str, least: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        raise ConfigError(f"{where} must be a whole number of {least} or more, got {value!r}")
    return value


def _whole_list(value: Any, where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{where} must be a list of whole numbers")
    numbers = tuple(_whole(item, where) for item in value)
    if len(set(numbers)) != len(numbers):
        raise ConfigError(f"{where} names a number twice")
    return numbers


def load_cost_config(
    root: Path | str = REPO_ROOT, config_dir: Path | str | None = None
) -> CostConfig:
    """Read and check ``config/cost_model.yaml``. Relative paths resolve under ``root``."""
    raw = load_config("cost_model", required=REQUIRED_KEYS, config_dir=config_dir)
    root = Path(root)

    records = raw["records"]
    if not isinstance(records, dict) or sorted(records) != sorted(RECORDS):
        raise ConfigError(f"{FILE}: records must name exactly: {', '.join(RECORDS)}")

    serving = raw["serving"]
    shares = serving["utilizations"]
    if not isinstance(shares, list) or not shares:
        raise ConfigError(f"{FILE}: serving.utilizations must be a list of quoted shares")
    utilizations = tuple(_amount(share, f"{FILE}: serving.utilizations") for share in shares)
    if any(share <= 0 or share > 1 for share in utilizations):
        raise ConfigError(f"{FILE}: each of serving.utilizations must be above 0 and at most 1")

    limits = raw["account_limits"]["requests_per_minute"]
    if not isinstance(limits, dict) or not limits:
        raise ConfigError(f"{FILE}: account_limits.requests_per_minute must map arms to limits")

    reconciliation = raw["reconciliation"]
    services = reconciliation["services"]
    if not isinstance(services, dict) or not services:
        raise ConfigError(f"{FILE}: reconciliation.services must map a service to a place")

    left_out = raw.get("left_out") or []
    if not isinstance(left_out, list):
        raise ConfigError(f"{FILE}: left_out must be a list")

    return CostConfig(
        root=root,
        report_stem=str(raw["report_stem"]),
        bill_file=root / str(raw["bill_file"]),
        record_files={name: root / str(records[name]) for name in RECORDS},
        hours_per_month=_whole(serving["hours_per_month"], f"{FILE}: serving.hours_per_month"),
        utilizations=utilizations,
        batch_sizes=_whole_list(serving["batch_sizes"], f"{FILE}: serving.batch_sizes"),
        batch_jobs_per_month=_whole(
            serving["batch_jobs_per_month"], f"{FILE}: serving.batch_jobs_per_month"
        ),
        storage_months=_whole_list(raw["storage_months"], f"{FILE}: storage_months"),
        limits_read_on=str(raw["account_limits"]["read_on"]),
        requests_per_minute={
            str(arm): _whole(limit, f"{FILE}: account_limits.requests_per_minute.{arm}")
            for arm, limit in limits.items()
        },
        services={str(service): str(place) for service, place in services.items()},
        tolerance_usd=_amount(
            reconciliation["tolerance_usd"], f"{FILE}: reconciliation.tolerance_usd"
        ),
        tolerance_percent=_amount(
            reconciliation["tolerance_percent"], f"{FILE}: reconciliation.tolerance_percent"
        ),
        left_out=tuple(str(item) for item in left_out),
    )


def _prices(entries: Any, where: str) -> dict[str, Decimal]:
    if not isinstance(entries, dict) or not entries:
        raise ConfigError(f"{where} must map instance types to prices")
    return {str(kind): _amount(price, f"{where}.{kind}") for kind, price in entries.items()}


def load_instance_prices(config_dir: Path | str | None = None) -> InstancePrices:
    """The SageMaker training and hosting prices of ``config/pricing.yaml``."""
    raw = load_config("pricing", required=PRICING_KEYS, config_dir=config_dir)["sagemaker"]
    hosting = raw["hosting"]
    return InstancePrices(
        training=_prices(raw["training"], f"{PRICING_FILE}: sagemaker.training"),
        training_published=str(raw["price_list"]["published"]),
        hosting=_prices(hosting["instances"], f"{PRICING_FILE}: sagemaker.hosting.instances"),
        hosting_published=str(hosting["price_list"]["published"]),
    )
