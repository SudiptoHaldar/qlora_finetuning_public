"""The comparison report's settings (req_005 v5.03).

``config/report.yaml`` says which committed records the report is built from, what it
writes, how its confidence intervals are made, and which pairs of arms are tested.
``config/report_deviations.yaml`` is the list of deviations from the plan that the report
prints. Neither holds a result.

Usage::

    from qlora_finetuning.comparison.report_config import load_deviations, load_report_config

    cfg = load_report_config()
    cfg.resamples, cfg.seed, cfg.confidence_percent      # 10000, 20261115, 95
    cfg.pairs                                             # (("A1", "A0"), ...)
    cfg.report_file, cfg.examples_file
    for entry in load_deviations():
        entry["date"], entry["what"], entry["recorded_in"]

This module imports neither boto3 nor the training stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from qlora_finetuning.settings import REPO_ROOT, ConfigError, load_config

FILE = "config report.yaml"
DEVIATIONS_FILE = "config report_deviations.yaml"
REQUIRED_KEYS = (
    "records.six_arm",
    "records.cost",
    "outputs.statistics_stem",
    "outputs.chart",
    "outputs.report",
    "outputs.readme",
    "outputs.examples",
    "bootstrap.resamples",
    "bootstrap.seed",
    "bootstrap.confidence_percent",
    "pairs",
    "step_groups",
    "examples_per_category",
    "smallest_p",
    "arm_labels",
    "chart.width",
    "chart.height",
    "chart.families",
)
ARTIFACTS = "artifacts"  # git-ignored: the only place a file with dataset text may go
DEVIATION_KEYS = ("date", "what", "recorded_in")
_COLOR = re.compile(r"#[0-9a-fA-F]{6}")


@dataclass(frozen=True)
class ReportConfig:
    """The typed content of ``config/report.yaml``, with its paths resolved."""

    root: Path
    six_arm_file: Path
    cost_file: Path
    statistics_stem: str
    chart_name: str
    report_file: Path
    readme_file: Path
    examples_file: Path
    resamples: int
    seed: int
    confidence_percent: int
    pairs: tuple[tuple[str, str], ...]
    step_groups: tuple[int, ...]
    examples_per_category: int
    smallest_p: Decimal
    arm_labels: dict[str, str]
    chart_width: int
    chart_height: int
    chart_families: dict[str, dict[str, Any]]  # family -> {"color": "#rrggbb", "arms": (...)}

    def label(self, arm: str) -> str:
        """How an arm is named for a reader; the arm's own name when none is configured."""
        return self.arm_labels.get(arm, arm)


def _whole(value: Any, where: str, least: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        raise ConfigError(f"{where} must be a whole number of {least} or more, got {value!r}")
    return value


def _name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a name, got {value!r}")
    return value


def _pairs(raw: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{FILE}: pairs must list pairs of arms")
    pairs = []
    for entry in raw:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ConfigError(f"{FILE}: each of pairs must name two arms, got {entry!r}")
        first, second = (_name(arm, f"{FILE}: pairs") for arm in entry)
        if first == second:
            raise ConfigError(f"{FILE}: a pair names {first} twice")
        pairs.append((first, second))
    if len({frozenset(pair) for pair in pairs}) != len(pairs):
        raise ConfigError(f"{FILE}: pairs names the same two arms twice")
    return tuple(pairs)


def _step_groups(raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{FILE}: step_groups must be a list of whole numbers")
    groups = tuple(_whole(item, f"{FILE}: step_groups") for item in raw)
    if list(groups) != sorted(set(groups)):
        raise ConfigError(f"{FILE}: step_groups must rise, each number once")
    return groups


def load_report_config(
    root: Path | str = REPO_ROOT, config_dir: Path | str | None = None
) -> ReportConfig:
    """Read and check ``config/report.yaml``. Relative paths resolve under ``root``."""
    raw = load_config("report", required=REQUIRED_KEYS, config_dir=config_dir)
    root = Path(root)
    outputs, bootstrap, chart = raw["outputs"], raw["bootstrap"], raw["chart"]

    confidence = _whole(bootstrap["confidence_percent"], f"{FILE}: bootstrap.confidence_percent")
    if not 50 <= confidence <= 99:
        raise ConfigError(f"{FILE}: bootstrap.confidence_percent must be from 50 to 99")

    examples = Path(_name(outputs["examples"], f"{FILE}: outputs.examples"))
    if examples.is_absolute() or examples.parts[0] != ARTIFACTS or ".." in examples.parts:
        raise ConfigError(
            f"{FILE}: outputs.examples must be under {ARTIFACTS}/, which git ignores: the "
            "file holds dataset text"
        )

    smallest = raw["smallest_p"]
    try:
        smallest_p = Decimal(smallest) if isinstance(smallest, str) else None
    except InvalidOperation:
        smallest_p = None
    if smallest_p is None or not 0 < smallest_p < 1:
        raise ConfigError(f'{FILE}: smallest_p must be quoted text such as "0.0001"')

    labels = raw["arm_labels"]
    if not isinstance(labels, dict) or not labels:
        raise ConfigError(f"{FILE}: arm_labels must map each arm to a name for a reader")

    return ReportConfig(
        root=root,
        six_arm_file=root / _name(raw["records"]["six_arm"], f"{FILE}: records.six_arm"),
        cost_file=root / _name(raw["records"]["cost"], f"{FILE}: records.cost"),
        statistics_stem=_name(outputs["statistics_stem"], f"{FILE}: outputs.statistics_stem"),
        chart_name=_name(outputs["chart"], f"{FILE}: outputs.chart"),
        report_file=root / _name(outputs["report"], f"{FILE}: outputs.report"),
        readme_file=root / _name(outputs["readme"], f"{FILE}: outputs.readme"),
        examples_file=root / examples,
        resamples=_whole(bootstrap["resamples"], f"{FILE}: bootstrap.resamples", least=100),
        seed=_whole(bootstrap["seed"], f"{FILE}: bootstrap.seed", least=0),
        confidence_percent=confidence,
        pairs=_pairs(raw["pairs"]),
        step_groups=_step_groups(raw["step_groups"]),
        examples_per_category=_whole(
            raw["examples_per_category"], f"{FILE}: examples_per_category"
        ),
        smallest_p=smallest_p,
        arm_labels={str(arm): _name(text, f"{FILE}: arm_labels.{arm}") for arm, text in
                    labels.items()},
        chart_width=_whole(chart["width"], f"{FILE}: chart.width", least=300),
        chart_height=_whole(chart["height"], f"{FILE}: chart.height", least=200),
        chart_families=_families(chart["families"]),
    )


def _families(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict) or not raw:
        raise ConfigError(f"{FILE}: chart.families must map a base model to a colour and arms")
    families: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for name, entry in raw.items():
        entry = entry if isinstance(entry, dict) else {}
        color, arms = entry.get("color"), entry.get("arms")
        if not isinstance(color, str) or not _COLOR.fullmatch(color):
            raise ConfigError(f'{FILE}: chart.families.{name}.color must be like "#2a78d6"')
        if not isinstance(arms, list) or not arms or seen & set(map(str, arms)):
            raise ConfigError(f"{FILE}: chart.families.{name}.arms must list arms, each once")
        seen |= set(map(str, arms))
        families[str(name)] = {"color": color.lower(), "arms": tuple(str(arm) for arm in arms)}
    return families


def load_deviations(config_dir: Path | str | None = None) -> list[dict[str, str]]:
    """The deviations from the plan, in the order of ``config/report_deviations.yaml``."""
    raw = load_config("report_deviations", required=("deviations",), config_dir=config_dir)
    entries = raw["deviations"]
    if not isinstance(entries, list):
        raise ConfigError(f"{DEVIATIONS_FILE}: deviations must be a list")
    deviations = []
    for number, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(key), str) or not entry[key].strip()
            for key in DEVIATION_KEYS
        ):
            raise ConfigError(
                f"{DEVIATIONS_FILE}: entry {number} must have quoted text for "
                f"{', '.join(DEVIATION_KEYS)}"
            )
        deviations.append({key: entry[key].strip() for key in DEVIATION_KEYS})
    return deviations
