"""Read FinQA examples by split, with the test split locked (req_001 v1.02).

This is how every stage outside the data package gets examples.

Usage::

    from qlora_finetuning.data import load_split, manifest_hash

    fit = load_split("fit")              # 4,884 records, for training
    dev = load_split("dev")              # 499, for checkpoint selection and validation
    quick = load_split("quick_dev")      # 200 of the dev examples, for fast baselines
    run = start_run(..., split_manifest_hash=manifest_hash())

    load_split("test")                   # raises TestLockedError

The test split is read once, by req_005 v5.01, which creates the unlock record named in
``config/split.yaml``. Until that record exists and names the current manifest, asking for
``test`` raises before any data is read. The lock guards against accident; it is a
convention enforced by tests, not access control.

Before returning anything, a loader checks that the manifest describes exactly the data on
disk: same dataset commit, same file checksums, same ids in the same order.

In a cloud job there is no snapshot. The environment variable ``QLORA_DATA_BUNDLE`` names a
data bundle (``qlora_finetuning/data/bundle.py``, req_003 v3.03) holding ``fit`` and ``dev``
only, and the loader serves those from it after the same kind of checks. From such a bundle
``test`` is refused whatever the unlock record says: the rows are not there. The two
evaluation jobs of req_005 v5.01 are given a test bundle instead, and ``test`` is served
from it when the unlock record in the job's code names the manifest.

Unlocking does not make a test run free: ``run_evaluation`` puts every run on ``test``
through the gate of ``qlora_finetuning/comparison/gate.py``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qlora_finetuning.data.bundle import BUNDLE_ENV, bundle_split, read_record
from qlora_finetuning.data.snapshot import (
    DataConfig,
    RawExample,
    load_data_config,
    load_raw_examples,
)
from qlora_finetuning.data.split import SplitConfig, SplitError, load_split_config, read_manifest
from qlora_finetuning.settings import get_str

LOADABLE = ("fit", "dev", "quick_dev", "test")


class TestLockedError(SplitError):
    """The test split was asked for before req_005 v5.01 unlocked it."""

    __test__ = False  # the name starts with "Test"; this keeps pytest from collecting it


# --------------------------------------------------------------------------- checks


def check_manifest_against_data(
    manifest: dict[str, Any], examples: Sequence[RawExample], data_cfg: DataConfig
) -> None:
    """Raise ``SplitError`` unless the manifest was built from exactly this data."""
    dataset = manifest["dataset"]
    if dataset.get("repo_id") != data_cfg.repo_id or dataset.get("revision") != data_cfg.revision:
        raise SplitError(
            "the split manifest was built from another dataset revision "
            f"({dataset.get('repo_id')} at {dataset.get('revision')}); config/data.yaml pins "
            f"{data_cfg.repo_id} at {data_cfg.revision}"
        )
    checksums = {item.official_split: item.sha256 for item in data_cfg.files}
    if dataset.get("files") != checksums:
        raise SplitError("the split manifest was built from files with other checksums")

    assigned = list(manifest["assignments"])
    present = [example.id for example in examples]
    if assigned != present:
        missing = sorted(set(present) - set(assigned))
        extra = sorted(set(assigned) - set(present))
        if missing:
            detail = f"{len(missing)} examples have no assignment, for example {missing[0]}"
        elif extra:
            detail = f"{len(extra)} assigned ids are not in the data, for example {extra[0]}"
        else:
            detail = "the ids are in a different order"
        raise SplitError(f"the split manifest does not match the snapshot: {detail}")


def unlock_state(cfg: SplitConfig, manifest: dict[str, Any]) -> tuple[bool, str]:
    """Whether the test split is unlocked, and a sentence saying why or why not.

    Unlocked means: the unlock record exists, is a JSON object, and its ``manifest_hash``
    equals this manifest's content hash. Any other file at that path does not unlock.
    """
    path = cfg.unlock_file
    if not path.is_file():
        return False, "no unlock record exists"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return False, f"the unlock record {path.name} is not valid JSON"
    if not isinstance(record, dict) or "manifest_hash" not in record:
        return False, f"the unlock record {path.name} does not name a manifest hash"
    if record["manifest_hash"] != manifest["content_hash"]:
        return False, f"the unlock record {path.name} names another manifest"
    return True, f"the unlock record {path.name} names this manifest"


def is_test_unlocked(cfg: SplitConfig | None = None) -> bool:
    cfg = cfg or load_split_config()
    return unlock_state(cfg, read_manifest(cfg))[0]


def _require_unlocked(cfg: SplitConfig, manifest: dict[str, Any]) -> None:
    unlocked, reason = unlock_state(cfg, manifest)
    if not unlocked:
        raise TestLockedError(
            f"the test split is locked: {reason}. It is read once, by req_005 v5.01, after "
            "every configuration is frozen. Use fit, dev, or quick_dev"
        )


def _check_name(name: str) -> None:
    if name not in LOADABLE:
        raise SplitError(f"unknown split {name!r}; use one of: {', '.join(LOADABLE)}")


def _ids(manifest: dict[str, Any], name: str) -> list[str]:
    if name == "quick_dev":
        return list(manifest["quick_dev"])
    return [example_id for example_id, split in manifest["assignments"].items() if split == name]


# --------------------------------------------------------------------------- public


def split_ids(name: str, cfg: SplitConfig | None = None) -> list[str]:
    """The ids of a split, in dataset order, without reading the data.

    Works for ``test`` too: the ids are in the committed manifest and say nothing about the
    questions.
    """
    _check_name(name)
    return _ids(read_manifest(cfg or load_split_config()), name)


def manifest_hash(cfg: SplitConfig | None = None) -> str:
    """The content hash of the committed manifest, for ``start_run(split_manifest_hash=...)``."""
    return str(read_manifest(cfg or load_split_config())["content_hash"])


def load_split(
    name: str, cfg: SplitConfig | None = None, data_cfg: DataConfig | None = None
) -> list[RawExample]:
    """Return the examples of ``fit``, ``dev``, ``quick_dev``, or (once unlocked) ``test``.

    Records come back in dataset order. Raises ``TestLockedError`` for ``test`` while it is
    locked, before any data is read, and ``SplitError`` if the manifest is missing, altered,
    or does not describe the data on disk.
    """
    _check_name(name)
    cfg = cfg or load_split_config()
    manifest = read_manifest(cfg)
    bundle = get_str(BUNDLE_ENV, "")
    if bundle:
        # A cloud job (req_003 v3.03): there is no snapshot, only a bundle. It holds fit and
        # dev, or, for the two test jobs of req_005 v5.01, the test examples alone.
        if name == "test":
            if "test" not in read_record(bundle)["splits"]:
                raise TestLockedError(
                    f"the test split is not in this data bundle ({BUNDLE_ENV} is set): the "
                    "job was given fit and dev only. Only an evaluation job of req_005 v5.01 "
                    "is given the test bundle"
                )
            _require_unlocked(cfg, manifest)
        return bundle_split(name, Path(bundle), cfg, manifest, data_cfg, _ids(manifest, name))
    return load_split_from_snapshot(name, cfg, data_cfg)


def load_split_from_snapshot(
    name: str, cfg: SplitConfig | None = None, data_cfg: DataConfig | None = None
) -> list[RawExample]:
    """``load_split`` from the snapshot on disk, whatever ``QLORA_DATA_BUNDLE`` says.

    For the code that writes a bundle, which must read the real data. Everything else calls
    ``load_split``.
    """
    _check_name(name)
    cfg = cfg or load_split_config()
    manifest = read_manifest(cfg)
    if name == "test":
        _require_unlocked(cfg, manifest)
    data_cfg = data_cfg or load_data_config()
    examples = load_raw_examples(data_cfg)
    check_manifest_against_data(manifest, examples, data_cfg)
    wanted = set(_ids(manifest, name))
    return [example for example in examples if example.id in wanted]
