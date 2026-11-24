"""Path B settings: ``config/frontier.yaml`` and the AWS names it needs (req_004).

Usage::

    from qlora_finetuning.frontier.config import load_frontier_config

    cfg = load_frontier_config()
    cfg.model("nova-pro").model_id      # "amazon.nova-pro-v1:0"
    cfg.settings                        # everything that can change a reply, as plain values
    cfg.epoch_count, cfg.path_b_cap_usd # what the budget gate prices, and against what

This module imports neither boto3 nor the training stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from qlora_finetuning.settings import REPO_ROOT, ConfigError, load_config

REQUIRED_KEYS = (
    "models",
    "inference.temperature",
    "inference.top_k",
    "client.retry_mode",
    "client.max_attempts",
    "client.connect_timeout",
    "client.read_timeout",
    "report_dir",
    "customization.epoch_count",
    "customization.learning_rate",
    "customization.learning_rate_warmup_steps",
    "customization.launch_order",
    "customization.name_prefix",
    "customization.output_prefix",
    "customization.poll_seconds",
    "customization.tags",
    "export.schema_version",
    "export.dir",
    "export.s3_prefix",
    "export.min_records",
    "export.max_records",
    "export.max_sample_tokens",
    "budget.path_b_cap_usd",
    "estimate.assumed_token_ratio",
    "estimate.margin_percent",
    "estimate.storage_months",
    "tuned_evaluation.splits",
    "tuned_evaluation.verdict_split",
    "tuned_evaluation.min_execution_gain_points",
    "tuned_evaluation.min_format_validity_percent",
)
AWS_REQUIRED_KEYS = ("profile", "account_id", "region")
RETRY_MODES = ("standard", "adaptive", "legacy")
MAX_TOP_K = 128  # the Nova request schema: 0 to 128
MAX_EPOCHS = 5  # Bedrock's range for Nova Pro and Nova Lite: 1 to 5
MAX_WARMUP_STEPS = 100  # Bedrock's range: 0 to 100
MAX_LEARNING_RATE = Decimal("0.0001")  # Bedrock's maximum for Nova Pro and Lite
MODEL_KEYS = ("model_id", "untuned_arm", "customization_model_id", "tuned_arm")
NAME_PART = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")  # safe inside a Bedrock job name
FILE = "config frontier.yaml"


@dataclass(frozen=True)
class FrontierModel:
    """One hosted model: the name the scripts use, its two Bedrock ids, and its arms."""

    name: str
    model_id: str  # answers questions
    untuned_arm: str
    customization_model_id: str  # only for starting a fine-tuning job
    tuned_arm: str
    # The most calls a minute the predictor makes to this model, tuned or untuned; None
    # means no pacing. It changes when a question is asked, never the reply.
    requests_per_minute: int | None = None


@dataclass(frozen=True)
class FrontierConfig:
    """The typed content of ``config/frontier.yaml``."""

    models: dict[str, FrontierModel]
    temperature: int | float
    top_k: int
    retry_mode: str
    max_attempts: int
    connect_timeout: int
    read_timeout: int
    report_dir: Path
    # v4.02: fine-tuning, the training files, the budget gate.
    epoch_count: int
    schema_version: str
    export_dir: Path
    s3_prefix: str
    min_records: int
    max_records: int
    max_sample_tokens: int
    path_b_cap_usd: Decimal
    assumed_token_ratio: Decimal
    margin_percent: Decimal
    storage_months: int
    # v4.03: the fine-tuning jobs.
    learning_rate: str  # text, sent to Bedrock as written
    learning_rate_warmup_steps: int
    launch_order: tuple[str, ...]
    name_prefix: str
    output_prefix: str
    poll_seconds: int
    tags: dict[str, str]
    # v4.04: the dev evaluation of the tuned models.
    tuned_splits: tuple[str, ...]
    verdict_split: str  # the one split the verdict is computed from
    min_execution_gain_points: Decimal  # tuned minus untuned execution accuracy, in points
    min_format_validity_percent: Decimal

    @property
    def hyperparameters(self) -> dict[str, str]:
        """What a job is launched with, as Bedrock takes it: text to text."""
        return {
            "epochCount": str(self.epoch_count),
            "learningRate": self.learning_rate,
            "learningRateWarmupSteps": str(self.learning_rate_warmup_steps),
        }

    @property
    def settings(self) -> dict[str, Any]:
        """Everything that can change a reply. A run is resumed only under equal settings.

        ``top_p``, stop sequences, the service tier, and a reasoning setting are not sent;
        they are named so that a later change to one of them shows up as a difference.
        """
        return {
            "api": "converse",
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": None,
            "stop_sequences": [],
            "service_tier": "default",
            "reasoning": "not sent",
        }

    def model(self, name: str) -> FrontierModel:
        try:
            return self.models[name]
        except KeyError:
            known = ", ".join(sorted(self.models))
            raise ConfigError(f"{FILE}: no model named {name!r}. Known: {known}") from None


def _whole(raw: dict[str, Any], section: str, key: str, minimum: int) -> int:
    value = raw[section][key]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(
            f"{FILE}: {section}.{key} must be a whole number of {minimum} or more, got {value!r}"
        )
    return value


def _number(raw: dict[str, Any], section: str, key: str, minimum: str) -> Decimal:
    """A YAML number as an exact decimal (through its text, never through a float's bits)."""
    value = raw[section][key]
    where = f"{FILE}: {section}.{key}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number, got {value!r}")
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise ConfigError(f"{where} must be a number, got {value!r}") from None
    if not number.is_finite() or number < Decimal(minimum):
        raise ConfigError(f"{where} must be a number of {minimum} or more, got {value!r}")
    return number


def _text(raw: dict[str, Any], section: str, key: str) -> str:
    value = raw[section][key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{FILE}: {section}.{key} must be text, got {value!r}")
    return value


def _learning_rate(value: Any) -> str:
    """The learning rate as the text Bedrock will be sent; a float is refused."""
    where = f"{FILE}: customization.learning_rate"
    if not isinstance(value, str):
        raise ConfigError(f'{where} must be quoted text such as "0.00001", got {value!r}')
    try:
        rate = Decimal(value)
    except InvalidOperation:
        raise ConfigError(f"{where} is not a number: {value!r}") from None
    if not rate.is_finite() or not 0 < rate <= MAX_LEARNING_RATE:
        raise ConfigError(
            f"{where} must be above 0 and at most {MAX_LEARNING_RATE}, got {value!r}"
        )
    return value


def _models(raw: Any) -> dict[str, FrontierModel]:
    if not isinstance(raw, dict) or not raw:
        raise ConfigError(f"{FILE}: models must be a mapping of name to model")
    models = {}
    for name, entry in raw.items():
        entry = entry if isinstance(entry, dict) else {}
        values = {key: entry.get(key) for key in MODEL_KEYS}
        bad = [key for key, value in values.items() if not isinstance(value, str) or not value]
        if not isinstance(name, str) or bad:
            raise ConfigError(f"{FILE}: models.{name} needs text for: {', '.join(bad) or 'name'}")
        pace = entry.get("requests_per_minute")
        if pace is not None and (isinstance(pace, bool) or not isinstance(pace, int) or pace < 1):
            raise ConfigError(
                f"{FILE}: models.{name}.requests_per_minute must be a whole number of 1 or "
                f"more, got {pace!r}"
            )
        models[name] = FrontierModel(name=name, **values, requests_per_minute=pace)
    return models


def load_frontier_config(
    root: Path | str = REPO_ROOT, config_dir: Path | str | None = None
) -> FrontierConfig:
    """Read and validate ``config/frontier.yaml``. Relative paths resolve under ``root``."""
    raw = load_config("frontier", required=REQUIRED_KEYS, config_dir=config_dir)
    temperature = raw["inference"]["temperature"]
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0 <= temperature <= 1
    ):
        raise ConfigError(
            f"{FILE}: inference.temperature must be a number from 0 to 1, got {temperature!r}"
        )
    top_k = _whole(raw, "inference", "top_k", 0)
    if top_k > MAX_TOP_K:
        raise ConfigError(f"{FILE}: inference.top_k must be at most {MAX_TOP_K}, got {top_k}")
    retry_mode = raw["client"]["retry_mode"]
    if retry_mode not in RETRY_MODES:
        raise ConfigError(
            f"{FILE}: client.retry_mode must be one of: {', '.join(RETRY_MODES)}; "
            f"got {retry_mode!r}"
        )
    epoch_count = _whole(raw, "customization", "epoch_count", 1)
    if epoch_count > MAX_EPOCHS:
        raise ConfigError(
            f"{FILE}: customization.epoch_count must be at most {MAX_EPOCHS}, got {epoch_count}"
        )
    s3_prefix = _text(raw, "export", "s3_prefix")
    if s3_prefix.startswith("/") or not s3_prefix.endswith("/"):
        raise ConfigError(
            f"{FILE}: export.s3_prefix must end with '/' and not start with one, "
            f"got {s3_prefix!r}"
        )
    min_records = _whole(raw, "export", "min_records", 1)
    models = _models(raw["models"])
    learning_rate = _learning_rate(raw["customization"]["learning_rate"])
    warmup = _whole(raw, "customization", "learning_rate_warmup_steps", 0)
    if warmup > MAX_WARMUP_STEPS:
        raise ConfigError(
            f"{FILE}: customization.learning_rate_warmup_steps must be at most "
            f"{MAX_WARMUP_STEPS}, got {warmup}"
        )
    order = raw["customization"]["launch_order"]
    if not isinstance(order, list) or sorted(map(str, order)) != sorted(models):
        raise ConfigError(
            f"{FILE}: customization.launch_order must name every model once: "
            f"{', '.join(models)}"
        )
    name_prefix = _text(raw, "customization", "name_prefix")
    bad_names = [name for name in (name_prefix, *models) if not NAME_PART.fullmatch(name)]
    if bad_names:
        raise ConfigError(
            f"{FILE}: a model name and customization.name_prefix may hold only lower-case "
            f"letters, digits, and inner hyphens; got {', '.join(bad_names)}"
        )
    output_prefix = _text(raw, "customization", "output_prefix")
    if output_prefix.startswith("/") or not output_prefix.endswith("/"):
        raise ConfigError(
            f"{FILE}: customization.output_prefix must end with '/' and not start with one, "
            f"got {output_prefix!r}"
        )
    tags = raw["customization"]["tags"]
    if not isinstance(tags, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in tags.items()
    ):
        raise ConfigError(f"{FILE}: customization.tags must map text to text")
    max_records = _whole(raw, "export", "max_records", min_records)
    tuned_splits = raw["tuned_evaluation"]["splits"]
    if (
        not isinstance(tuned_splits, list)
        or not tuned_splits
        or not all(isinstance(name, str) and name for name in tuned_splits)
        or len(set(tuned_splits)) != len(tuned_splits)
    ):
        raise ConfigError(f"{FILE}: tuned_evaluation.splits must list split names, each once")
    if "test" in tuned_splits:
        raise ConfigError(
            f"{FILE}: tuned_evaluation.splits must not name test: it is scored in req_005 only"
        )
    verdict_split = _text(raw, "tuned_evaluation", "verdict_split")
    if verdict_split not in tuned_splits:
        raise ConfigError(
            f"{FILE}: tuned_evaluation.verdict_split must be one of tuned_evaluation.splits: "
            f"{', '.join(tuned_splits)}"
        )
    min_format = _number(raw, "tuned_evaluation", "min_format_validity_percent", "0")
    if min_format > 100:
        raise ConfigError(
            f"{FILE}: tuned_evaluation.min_format_validity_percent must be at most 100, "
            f"got {min_format}"
        )
    root = Path(root)
    return FrontierConfig(
        models=models,
        temperature=temperature,
        top_k=top_k,
        retry_mode=retry_mode,
        max_attempts=_whole(raw, "client", "max_attempts", 1),
        connect_timeout=_whole(raw, "client", "connect_timeout", 1),
        read_timeout=_whole(raw, "client", "read_timeout", 1),
        report_dir=root / str(raw["report_dir"]),
        epoch_count=epoch_count,
        schema_version=_text(raw, "export", "schema_version"),
        export_dir=root / _text(raw, "export", "dir"),
        s3_prefix=s3_prefix,
        min_records=min_records,
        max_records=max_records,
        max_sample_tokens=_whole(raw, "export", "max_sample_tokens", 1),
        path_b_cap_usd=_number(raw, "budget", "path_b_cap_usd", "0"),
        assumed_token_ratio=_number(raw, "estimate", "assumed_token_ratio", "0.1"),
        margin_percent=_number(raw, "estimate", "margin_percent", "0"),
        storage_months=_whole(raw, "estimate", "storage_months", 0),
        learning_rate=learning_rate,
        learning_rate_warmup_steps=warmup,
        launch_order=tuple(map(str, order)),
        name_prefix=name_prefix,
        output_prefix=output_prefix,
        poll_seconds=_whole(raw, "customization", "poll_seconds", 1),
        tags=dict(tags),
        tuned_splits=tuple(tuned_splits),
        verdict_split=verdict_split,
        min_execution_gain_points=_number(
            raw, "tuned_evaluation", "min_execution_gain_points", "0"
        ),
        min_format_validity_percent=min_format,
    )


def load_aws_settings(config_dir: Path | str | None = None) -> dict[str, str]:
    """The profile, account id, and region of ``config/aws.yaml``, each as text."""
    raw = load_config("aws", required=AWS_REQUIRED_KEYS, config_dir=config_dir)
    return {key: str(raw[key]) for key in AWS_REQUIRED_KEYS}


def load_customization_role(config_dir: Path | str | None = None) -> str:
    """The ARN of the role Bedrock fine-tuning jobs run as, from ``config/aws.yaml``."""
    key = "roles.bedrock_customization.arn"
    raw = load_config("aws", required=(key,), config_dir=config_dir)
    return str(raw["roles"]["bedrock_customization"]["arn"])


def load_bucket(config_dir: Path | str | None = None) -> str:
    """The project's S3 bucket name, from ``config/aws.yaml``."""
    raw = load_config("aws", required=("bucket.name",), config_dir=config_dir)
    return str(raw["bucket"]["name"])
