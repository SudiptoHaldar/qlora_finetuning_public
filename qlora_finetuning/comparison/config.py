"""The comparison's settings: ``config/comparison.yaml`` (req_005 v5.01).

Usage::

    from qlora_finetuning.comparison.config import load_comparison_config

    cfg = load_comparison_config()
    cfg.freeze_file, cfg.voids_file, cfg.report_dir
    cfg.arms["A1"].predictor, cfg.arms["A1"].where        # "qwen3-4b-tuned", "sagemaker"
    cfg.arm("B1-Pro").model                                # "nova-pro"

This module imports neither boto3 nor the training stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qlora_finetuning.settings import REPO_ROOT, ConfigError, load_config

REQUIRED_KEYS = (
    "freeze_file",
    "voids_file",
    "report_dir",
    "shared_configs",
    "gold_predictor",
    "arms",
)
PATHS = ("A", "B")
PLACES = {"A": "sagemaker", "B": "bedrock"}
FILE = "config comparison.yaml"


@dataclass(frozen=True)
class ArmConfig:
    """One arm of the comparison: which predictor it is and where it may answer test."""

    arm: str
    path: str
    predictor: str
    where: str
    model: str | None  # Path B: the model's name in frontier.yaml
    tuned: bool


@dataclass(frozen=True)
class ComparisonConfig:
    """The typed content of ``config/comparison.yaml``, with its paths resolved."""

    root: Path
    freeze_file: Path
    voids_file: Path
    report_dir: Path
    shared_configs: tuple[str, ...]
    gold_predictor: str
    arms: dict[str, ArmConfig]

    def arm(self, name: str) -> ArmConfig:
        if name not in self.arms:
            raise ConfigError(
                f"{name!r} is not an arm of the comparison; the arms are: {', '.join(self.arms)}"
            )
        return self.arms[name]


def _arms(raw: Any) -> dict[str, ArmConfig]:
    if not isinstance(raw, dict) or not raw:
        raise ConfigError(f"{FILE}: arms must map each arm to its settings")
    arms: dict[str, ArmConfig] = {}
    for name, entry in raw.items():
        entry = entry if isinstance(entry, dict) else {}
        path, predictor, where = entry.get("path"), entry.get("predictor"), entry.get("where")
        if path not in PATHS:
            raise ConfigError(f"{FILE}: arms.{name}.path must be one of {', '.join(PATHS)}")
        if not isinstance(predictor, str) or not predictor.strip():
            raise ConfigError(f"{FILE}: arms.{name}.predictor must name a predictor")
        if where != PLACES[path]:
            raise ConfigError(
                f"{FILE}: arms.{name}.where must be {PLACES[path]} for an arm of Path {path}"
            )
        model = entry.get("model")
        if path == "B" and (not isinstance(model, str) or not model.strip()):
            raise ConfigError(f"{FILE}: arms.{name}.model must name a model of frontier.yaml")
        tuned = entry.get("tuned", False)
        if not isinstance(tuned, bool):
            raise ConfigError(f"{FILE}: arms.{name}.tuned must be true or false")
        arms[str(name)] = ArmConfig(
            arm=str(name),
            path=path,
            predictor=predictor,
            where=where,
            model=model if path == "B" else None,
            tuned=tuned,
        )
    predictors = [arm.predictor for arm in arms.values()]
    if len(set(predictors)) != len(predictors):
        raise ConfigError(f"{FILE}: two arms name the same predictor")
    return arms


def load_comparison_config(
    root: Path | str = REPO_ROOT, config_dir: Path | str | None = None
) -> ComparisonConfig:
    """Read and check ``config/comparison.yaml``. Relative paths resolve under ``root``."""
    raw = load_config("comparison", required=REQUIRED_KEYS, config_dir=config_dir)
    shared = raw["shared_configs"]
    if (
        not isinstance(shared, list)
        or not all(isinstance(name, str) and name.strip() for name in shared)
        or len(set(shared)) != len(shared)
    ):
        raise ConfigError(f"{FILE}: shared_configs must list config names, each once")
    gold = raw["gold_predictor"]
    if not isinstance(gold, str) or not gold.strip():
        raise ConfigError(f"{FILE}: gold_predictor must name a predictor")
    root = Path(root)
    return ComparisonConfig(
        root=root,
        freeze_file=root / str(raw["freeze_file"]),
        voids_file=root / str(raw["voids_file"]),
        report_dir=root / str(raw["report_dir"]),
        shared_configs=tuple(shared),
        gold_predictor=gold,
        arms=_arms(raw["arms"]),
    )
