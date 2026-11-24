"""Path A's AWS settings: ``config/sagemaker.yaml``, with names from ``aws.yaml`` and prices
from ``pricing.yaml`` (req_003 v3.03).

Usage::

    from qlora_finetuning.local_qlora.sagemaker_config import load_aws_names, load_sagemaker_config

    cfg = load_sagemaker_config()
    cfg.image, cfg.instance_type, cfg.task("smoke").max_runtime_seconds
    cfg.max_cost("smoke")              # Decimal: the most a smoke job can cost
    cfg.cost_of(1800, "ml.g5.xlarge")  # Decimal: seconds times the price

    names = load_aws_names()
    names.bucket, names.role_arn, names.profile

A price is quoted text read as ``Decimal``; no price is written in code. This module imports
neither boto3 nor the training stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from qlora_finetuning.settings import REPO_ROOT, ConfigError, load_config

REQUIRED_KEYS = (
    "image",
    "instance_type",
    "instance_count",
    "volume_gb",
    "workspace_prefix",
    "code_prefix",
    "bundle_prefix",
    "output_prefix",
    "name_prefix",
    "bundle_dir",
    "test_bundle_dir",
    "record_file",
    "tasks",
    "smoke.evaluate_examples",
    "budget.path_a_cap_usd",
    "tags",
    "estimate.runtime_margin",
    "estimate.minimum_runtime_seconds",
)
AWS_KEYS = (
    "profile",
    "account_id",
    "region",
    "bucket.name",
    "roles.sagemaker_execution.arn",
    "roles.sagemaker_execution.trusted_service",
    "quota.service_code",
    "quota.usage",
    "bucket.prefixes.adapters",
)
PRICING_KEYS = ("sagemaker.price_list.published", "sagemaker.training")
TASKS = ("smoke", "train", "evaluate", "score")
PHASES = ("train", "eval", "other")
ARMS = ("A0", "A1", "none")
SECONDS_PER_HOUR = Decimal(3600)
FILE = "config sagemaker.yaml"
_PREFIX = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]*[A-Za-z0-9]")
_NAME_PREFIX = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
_IMAGE = re.compile(
    r"[0-9]{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/[a-z0-9._/-]+:[A-Za-z0-9._-]+"
)


@dataclass(frozen=True)
class TaskConfig:
    """One kind of job: how long it may run, and how its billable time is recorded."""

    name: str
    max_runtime_seconds: int
    phase: str
    arm: str | None  # None: the arm is the predictor's (an evaluate job)


@dataclass(frozen=True)
class AwsNames:
    """The account names a job needs, from ``config/aws.yaml``."""

    profile: str
    account_id: str
    region: str
    bucket: str
    role_arn: str
    role_trusts: str
    quota_service: str
    quota_usage: str
    adapters_prefix: str  # where a selected adapter is kept, for example ``adapters/``

    @property
    def settings(self) -> dict[str, str]:
        """What ``aws_common`` takes."""
        return {"profile": self.profile, "account_id": self.account_id, "region": self.region}


@dataclass(frozen=True)
class SageMakerConfig:
    """The typed content of ``config/sagemaker.yaml``, with its paths resolved and prices."""

    root: Path
    image: str
    instance_type: str
    instance_count: int
    volume_gb: int
    workspace_prefix: str
    code_prefix: str
    bundle_prefix: str
    output_prefix: str
    name_prefix: str
    bundle_dir: Path
    test_bundle_dir: Path  # the test examples alone, for a test evaluation job (req_005 v5.01)
    record_file: Path
    tasks: dict[str, TaskConfig]
    smoke_examples: int
    path_a_cap_usd: Decimal
    tags: dict[str, str]
    hourly_prices: dict[str, Decimal]  # per instance type
    prices_published: str
    runtime_margin: Decimal  # a job's proposed limit is its estimate times this
    minimum_runtime_seconds: int

    def task(self, name: str) -> TaskConfig:
        if name not in self.tasks:
            raise ConfigError(f"unknown task {name!r}; use one of: {', '.join(self.tasks)}")
        return self.tasks[name]

    def hourly(self, instance_type: str | None = None) -> Decimal:
        instance_type = instance_type or self.instance_type
        if instance_type not in self.hourly_prices:
            raise ConfigError(
                f"config pricing.yaml has no price for {instance_type} (sagemaker.training)"
            )
        return self.hourly_prices[instance_type]

    def cost_of(self, seconds: int, instance_type: str | None = None, count: int = 1) -> Decimal:
        """Seconds on ``count`` instances at the hourly price. Exact; round only to show it."""
        return Decimal(seconds) * count * self.hourly(instance_type) / SECONDS_PER_HOUR

    def max_cost(self, task: str, max_runtime_seconds: int | None = None) -> Decimal:
        """The most a job of ``task`` can cost: its maximum runtime at the price."""
        if max_runtime_seconds is not None:
            return self.cost_of(max_runtime_seconds, count=self.instance_count)
        return self.cost_of(self.task(task).max_runtime_seconds, count=self.instance_count)


def _whole(raw: Any, key: str, minimum: int = 1) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < minimum:
        raise ConfigError(
            f"{FILE}: {key} must be a whole number of at least {minimum}, got {raw!r}"
        )
    return raw


def _prefix(raw: Any, key: str) -> str:
    if not isinstance(raw, str) or not _PREFIX.fullmatch(raw) or ".." in raw:
        raise ConfigError(f"{FILE}: {key} must be an S3 key prefix without a leading slash")
    return raw


def _amount(raw: Any, where: str) -> Decimal:
    try:
        value = Decimal(str(raw))
    except InvalidOperation:
        value = Decimal(-1)
    if isinstance(raw, (bool, float)) or not value.is_finite() or value <= 0:
        raise ConfigError(f"{where} must be a positive amount written as text, got {raw!r}")
    return value


def _margin(raw: Any) -> Decimal:
    value = _amount(raw, f"{FILE}: estimate.runtime_margin")
    if value < 1:
        raise ConfigError(f"{FILE}: estimate.runtime_margin must be at least 1, got {raw!r}")
    return value


def _tasks(raw: Any) -> dict[str, TaskConfig]:
    if not isinstance(raw, dict) or set(raw) != set(TASKS):
        raise ConfigError(f"{FILE}: tasks must name exactly: {', '.join(TASKS)}")
    tasks = {}
    for name in TASKS:
        entry = raw[name] if isinstance(raw[name], dict) else {}
        phase, arm = entry.get("phase"), entry.get("arm")
        if phase not in PHASES:
            raise ConfigError(f"{FILE}: tasks.{name}.phase must be one of {', '.join(PHASES)}")
        if arm is not None and arm not in ARMS:
            raise ConfigError(f"{FILE}: tasks.{name}.arm must be one of {', '.join(ARMS)}")
        tasks[name] = TaskConfig(
            name=name,
            max_runtime_seconds=_whole(
                entry.get("max_runtime_seconds"), f"tasks.{name}.max_runtime_seconds", 60
            ),
            phase=phase,
            arm=arm,
        )
    return tasks


def load_sagemaker_config(
    root: Path | str = REPO_ROOT, config_dir: Path | str | None = None
) -> SageMakerConfig:
    """Read and check ``config/sagemaker.yaml`` and the SageMaker prices of ``pricing.yaml``."""
    raw = load_config("sagemaker", required=REQUIRED_KEYS, config_dir=config_dir)
    pricing = load_config("pricing", required=PRICING_KEYS, config_dir=config_dir)["sagemaker"]

    image = raw["image"]
    if not isinstance(image, str) or not _IMAGE.fullmatch(image):
        raise ConfigError(
            f"{FILE}: image must be a full ECR image URI with a tag "
            "(<account>.dkr.ecr.<region>.amazonaws.com/<repository>:<tag>)"
        )
    name_prefix = raw["name_prefix"]
    if not isinstance(name_prefix, str) or not _NAME_PREFIX.fullmatch(name_prefix):
        raise ConfigError(f"{FILE}: name_prefix must be lower-case letters, digits, and hyphens")
    tags = raw["tags"]
    if not isinstance(tags, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in tags.items()
    ):
        raise ConfigError(f"{FILE}: tags must map text to text")
    entries = pricing["training"]
    if not isinstance(entries, dict) or not entries:
        raise ConfigError(
            "config pricing.yaml: sagemaker.training must map instance types to prices"
        )
    prices = {
        str(kind): _amount(price, f"config pricing.yaml: sagemaker.training.{kind}")
        for kind, price in entries.items()
    }
    instance_type = str(raw["instance_type"])
    if instance_type not in prices:
        raise ConfigError(
            f"{FILE}: instance_type {instance_type} has no price in config pricing.yaml "
            "(sagemaker.training)"
        )

    root = Path(root)
    return SageMakerConfig(
        root=root,
        image=image,
        instance_type=instance_type,
        instance_count=_whole(raw["instance_count"], "instance_count"),
        volume_gb=_whole(raw["volume_gb"], "volume_gb", 30),
        workspace_prefix=_prefix(raw["workspace_prefix"], "workspace_prefix"),
        code_prefix=_prefix(raw["code_prefix"], "code_prefix"),
        bundle_prefix=_prefix(raw["bundle_prefix"], "bundle_prefix"),
        output_prefix=_prefix(raw["output_prefix"], "output_prefix"),
        name_prefix=name_prefix,
        bundle_dir=root / str(raw["bundle_dir"]),
        test_bundle_dir=root / str(raw["test_bundle_dir"]),
        record_file=root / str(raw["record_file"]),
        tasks=_tasks(raw["tasks"]),
        smoke_examples=_whole(raw["smoke"]["evaluate_examples"], "smoke.evaluate_examples"),
        path_a_cap_usd=_amount(raw["budget"]["path_a_cap_usd"], f"{FILE}: budget.path_a_cap_usd"),
        tags=dict(tags),
        hourly_prices=prices,
        prices_published=str(pricing["price_list"]["published"]),
        runtime_margin=_margin(raw["estimate"]["runtime_margin"]),
        minimum_runtime_seconds=_whole(
            raw["estimate"]["minimum_runtime_seconds"], "estimate.minimum_runtime_seconds", 60
        ),
    )


def load_aws_names(config_dir: Path | str | None = None) -> AwsNames:
    """The profile, bucket, role, and quota names of ``config/aws.yaml``."""
    raw = load_config("aws", required=AWS_KEYS, config_dir=config_dir)
    role = raw["roles"]["sagemaker_execution"]
    return AwsNames(
        profile=str(raw["profile"]),
        account_id=str(raw["account_id"]),
        region=str(raw["region"]),
        bucket=str(raw["bucket"]["name"]),
        role_arn=str(role["arn"]),
        role_trusts=str(role["trusted_service"]),
        quota_service=str(raw["quota"]["service_code"]),
        quota_usage=str(raw["quota"]["usage"]),
        adapters_prefix=str(raw["bucket"]["prefixes"]["adapters"]),
    )
