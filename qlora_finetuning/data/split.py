"""The frozen split: who goes to ``fit``, ``dev``, and ``test`` (req_001 v1.02).

Every example gets one label, by a rule with no randomness in it:

1. Examples are grouped (by company, by default). A group is never divided.
2. Groups are put in a fixed order: the SHA-256 of ``"<seed>:<salt>:<group key>"``.
3. Walking that order, a group joins the training side whenever it still fits under the
   target (65% of the examples). Everything else is ``test``.
4. ``dev`` is carved out of the training side the same way; the rest is ``fit``.
5. ``quick_dev`` marks a fixed number of dev examples, ordered by the digest of their id.

The result is a manifest, written **once** to ``splits/finqa_split_manifest.json``. After
that it changes only by a recorded decision, because every result depends on it.

Usage::

    from qlora_finetuning.data import load_data_config, load_raw_examples
    from qlora_finetuning.data.split import build_split, load_split_config, write_manifest

    cfg, data_cfg = load_split_config(), load_data_config()
    manifest = build_split(load_raw_examples(data_cfg), cfg, data_cfg)
    write_manifest(manifest, cfg)        # writes if absent; refuses a different one
    manifest["counts"]["fit"]["examples"], manifest["content_hash"]

To read examples by split, use ``qlora_finetuning.data.load_split``; this module builds.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qlora_finetuning.data.snapshot import DataConfig, RawExample
from qlora_finetuning.settings import REPO_ROOT, ConfigError, load_config

REQUIRED_KEYS = (
    "group_by",
    "seed",
    "train_fraction",
    "train_fraction_tolerance",
    "dev_size",
    "dev_size_tolerance",
    "quick_dev_size",
    "leakage_min_digits",
    "manifest_file",
    "unlock_file",
    "report_dir",
)
SPLITS = ("fit", "dev", "test")
RULE = "sha256-order-fill-v1"
BUILD_COMMAND = "python -m scripts.finqa_split"
MANIFEST_KEYS = (
    "schema",
    "dataset",
    "settings",
    "counts",
    "content_hash",
    "groups",
    "quick_dev",
    "assignments",
)

# What "the same group" means for each allowed value of ``group_by``.
GROUP_KEYS = {
    "company": lambda example: example.ticker,
    "report": lambda example: example.report,
    "page": lambda example: example.filename,
}


class SplitError(ConfigError):
    """The split cannot be built, or the manifest is missing, altered, or for other data."""


@dataclass(frozen=True)
class SplitConfig:
    """``config/split.yaml`` with its paths resolved."""

    root: Path
    group_by: str
    seed: int
    train_fraction: float
    train_fraction_tolerance: float
    dev_size: int
    dev_size_tolerance: int
    quick_dev_size: int
    leakage_min_digits: int
    manifest_file: Path
    unlock_file: Path
    report_dir: Path


# --------------------------------------------------------------------------- config


def _integer(raw: dict[str, Any], key: str, minimum: int) -> int:
    value = raw[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(
            f"config split.yaml: {key} must be a whole number of at least {minimum}, got {value!r}"
        )
    return value


def _fraction(raw: dict[str, Any], key: str) -> float:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
        raise ConfigError(f"config split.yaml: {key} must be between 0 and 1, got {value!r}")
    return float(value)


def load_split_config(
    root: Path | str = REPO_ROOT, config_dir: Path | str | None = None
) -> SplitConfig:
    """Read ``config/split.yaml``. Relative paths in it resolve under ``root``."""
    raw = load_config("split", required=REQUIRED_KEYS, config_dir=config_dir)
    group_by = str(raw["group_by"])
    if group_by not in GROUP_KEYS:
        raise ConfigError(
            f"config split.yaml: group_by must be one of {', '.join(GROUP_KEYS)}, got {group_by!r}"
        )
    root = Path(root)
    return SplitConfig(
        root=root,
        group_by=group_by,
        seed=_integer(raw, "seed", 0),
        train_fraction=_fraction(raw, "train_fraction"),
        train_fraction_tolerance=_fraction(raw, "train_fraction_tolerance"),
        dev_size=_integer(raw, "dev_size", 1),
        dev_size_tolerance=_integer(raw, "dev_size_tolerance", 0),
        quick_dev_size=_integer(raw, "quick_dev_size", 1),
        leakage_min_digits=_integer(raw, "leakage_min_digits", 1),
        manifest_file=root / str(raw["manifest_file"]),
        unlock_file=root / str(raw["unlock_file"]),
        report_dir=root / str(raw["report_dir"]),
    )


# --------------------------------------------------------------------------- the rule


def group_key(example: RawExample, group_by: str) -> str:
    """The key that keeps an example with its group. Raises if the example has none."""
    key = GROUP_KEYS[group_by](example)
    if not key:
        raise SplitError(
            f"example {example.id!r} has no {group_by} to group by (its id does not parse)"
        )
    return str(key)


def order_key(seed: int, salt: str, key: str) -> str:
    """Where ``key`` stands in the fixed order: the SHA-256 of ``"<seed>:<salt>:<key>"``.

    A hash, not a random generator, so the order is the same on every Python version and
    operating system.
    """
    return hashlib.sha256(f"{seed}:{salt}:{key}".encode()).hexdigest()


def fill(sizes: Mapping[str, int], seed: int, salt: str, target: int) -> list[str]:
    """Walk the groups in digest order and take each one that still fits under ``target``.

    The total never exceeds the target. A group that would overshoot is passed over, and a
    later, smaller one may still be taken, which is how the total lands on or just under it.
    """
    chosen: list[str] = []
    total = 0
    for key in sorted(sizes, key=lambda item: order_key(seed, salt, item)):
        if total + sizes[key] <= target:
            chosen.append(key)
            total += sizes[key]
    return chosen


def content_hash(assignments: Mapping[str, str], quick_dev: Collection[str]) -> str:
    """One value that identifies the split: SHA-256 over a line per example, in order.

    Each line is ``<id>\\t<split>\\t<1 if quick-dev else 0>\\n``.
    """
    quick = set(quick_dev)
    digest = hashlib.sha256()
    for example_id, split in assignments.items():
        digest.update(f"{example_id}\t{split}\t{int(example_id in quick)}\n".encode())
    return digest.hexdigest()


def build_split(
    examples: Sequence[RawExample], cfg: SplitConfig, data_cfg: DataConfig
) -> dict[str, Any]:
    """Assign every example and return the manifest. Pure: no file access and no clock."""
    ids = Counter(example.id for example in examples)
    repeated = sorted(example_id for example_id, count in ids.items() if count > 1)
    if repeated:
        raise SplitError(f"example ids are not unique: {', '.join(repeated[:5])}")

    keys = {example.id: group_key(example, cfg.group_by) for example in examples}
    sizes = Counter(keys.values())
    train_target = round(len(examples) * cfg.train_fraction)
    train_groups = set(fill(sizes, cfg.seed, "train", train_target))
    train_sizes = {key: size for key, size in sizes.items() if key in train_groups}
    train_total = sum(train_sizes.values())
    if cfg.dev_size >= train_total:
        raise SplitError(
            f"dev_size {cfg.dev_size} does not fit in a training side of {train_total} examples"
        )
    dev_groups = set(fill(train_sizes, cfg.seed, "dev", cfg.dev_size))

    def split_of(key: str) -> str:
        if key in dev_groups:
            return "dev"
        return "fit" if key in train_groups else "test"

    assignments = {example.id: split_of(keys[example.id]) for example in examples}
    dev_ids = [example_id for example_id, split in assignments.items() if split == "dev"]
    if cfg.quick_dev_size > len(dev_ids):
        raise SplitError(
            f"quick_dev_size {cfg.quick_dev_size} is larger than dev, which has {len(dev_ids)} "
            "examples"
        )
    chosen = set(
        sorted(dev_ids, key=lambda item: order_key(cfg.seed, "quick_dev", item))[
            : cfg.quick_dev_size
        ]
    )
    quick_dev = [example_id for example_id in dev_ids if example_id in chosen]

    groups = {key: split_of(key) for key in sorted(sizes)}
    counts: dict[str, Any] = {"examples": len(examples)}
    for split in SPLITS:
        counts[split] = {
            "examples": sum(value == split for value in assignments.values()),
            "groups": sum(value == split for value in groups.values()),
        }
    counts["train"] = {
        "examples": counts["fit"]["examples"] + counts["dev"]["examples"],
        "groups": counts["fit"]["groups"] + counts["dev"]["groups"],
    }
    counts["quick_dev"] = len(quick_dev)

    return {
        "schema": 1,
        "dataset": {
            "repo_id": data_cfg.repo_id,
            "revision": data_cfg.revision,
            "files": {item.official_split: item.sha256 for item in data_cfg.files},
        },
        "settings": {
            "group_by": cfg.group_by,
            "seed": cfg.seed,
            "train_fraction": cfg.train_fraction,
            "dev_size": cfg.dev_size,
            "quick_dev_size": cfg.quick_dev_size,
            "rule": RULE,
        },
        "counts": counts,
        "content_hash": content_hash(assignments, quick_dev),
        "groups": groups,
        "quick_dev": quick_dev,
        "assignments": assignments,
    }


# --------------------------------------------------------------------------- the manifest


def render_manifest(manifest: dict[str, Any]) -> str:
    """The manifest as text: one assignment per line, so a change is readable in a diff."""
    return json.dumps(manifest, indent=1, ensure_ascii=True) + "\n"


def validate_manifest(manifest: Any, where: str = "the split manifest") -> None:
    """Raise ``SplitError`` unless the manifest has the right shape and its hash is correct."""
    if not isinstance(manifest, dict):
        raise SplitError(f"{where} is not a JSON object")
    missing = [key for key in MANIFEST_KEYS if key not in manifest]
    if missing:
        raise SplitError(f"{where} is missing: {', '.join(missing)}")
    if manifest["schema"] != 1:
        raise SplitError(f"{where} has schema {manifest['schema']!r}; this code reads schema 1")
    assignments, quick_dev = manifest["assignments"], manifest["quick_dev"]
    if not isinstance(assignments, dict) or not isinstance(quick_dev, list):
        raise SplitError(f"{where}: assignments must be an object and quick_dev a list")
    unknown = sorted({str(value) for value in assignments.values()} - set(SPLITS))
    if unknown:
        raise SplitError(f"{where} uses unknown split names: {', '.join(unknown)}")
    outside = [item for item in quick_dev if assignments.get(item) != "dev"]
    if outside or len(set(quick_dev)) != len(quick_dev):
        raise SplitError(f"{where}: quick_dev must list distinct ids that are assigned to dev")
    if content_hash(assignments, quick_dev) != manifest["content_hash"]:
        raise SplitError(
            f"{where} does not match its content hash: it was edited after it was written. "
            "Restore it from git"
        )


def read_manifest(cfg: SplitConfig) -> dict[str, Any]:
    """Read and validate the committed manifest."""
    path = cfg.manifest_file
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SplitError(
            f"the split manifest does not exist: {path}. Build it with: {BUILD_COMMAND}"
        ) from exc
    try:
        manifest = json.loads(text)
    except ValueError as exc:
        raise SplitError(f"the split manifest is not valid JSON: {path}") from exc
    validate_manifest(manifest, f"the split manifest {path.name}")
    return manifest


def write_manifest(manifest: dict[str, Any], cfg: SplitConfig) -> bool:
    """Write the manifest if there is none. Return whether it was written.

    An identical manifest is left as it is. A different one is never replaced: the split is
    frozen, and changing it is a decision that is made by hand and recorded.
    """
    validate_manifest(manifest, "the manifest to write")
    text = render_manifest(manifest)
    path = cfg.manifest_file
    if path.is_file():
        if path.read_text(encoding="utf-8") == text:
            return False
        raise SplitError(
            f"{path.name} exists and differs from the split these settings produce. The split "
            "is frozen and nothing was written. If the settings or the data changed by "
            "mistake, restore them. To change the split on purpose, record the decision in "
            "the req_001 roadmap, delete the file by hand, and build again"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return True
