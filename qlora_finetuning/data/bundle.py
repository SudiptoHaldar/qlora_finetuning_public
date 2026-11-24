"""The data bundle: the ``fit`` and ``dev`` examples in one file, for a cloud job (req_003 v3.03).

The snapshot holds every FinQA row, the test rows among them, so it never leaves this
machine. A job is given a bundle instead: the raw examples of ``fit`` and ``dev`` and a
small record of what they are. ``load_split`` reads it when the environment variable
``QLORA_DATA_BUNDLE`` names its folder, and callers do not change.

From req_005 v5.01 there is a second kind, the **test bundle**: the test examples and
nothing else, for the two evaluation jobs that answer the test split. It can be written
only after the unlock, and a bundle is one kind or the other, never both: no job that
trains or selects is ever handed a test row.

Usage::

    from qlora_finetuning.data.bundle import export_bundle, read_bundle

    record = export_bundle("artifacts/bundle")        # from the snapshot; fit and dev only
    record["splits"], record["sha256"]                # {"fit": 4884, "dev": 499}, the file's hash

    record, examples = read_bundle("artifacts/bundle")   # checked against the current manifest

    # In a job: QLORA_DATA_BUNDLE=/opt/ml/input/data/bundle, then, as everywhere,
    from qlora_finetuning.data import load_split
    load_split("fit")          # from the bundle
    load_split("test")         # TestLockedError: this bundle holds no test row

    export_bundle("artifacts/bundle_test", ("test",))   # TestLockedError until the unlock

A ``fit`` and ``dev`` bundle cannot hold a test row: asking for one is refused, and every id
written or read is checked against the splits the bundle names. A bundle holds dataset text,
so it stays under git-ignored ``artifacts/`` and in the project's private bucket.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from qlora_finetuning.data.snapshot import DataConfig, RawExample, load_data_config
from qlora_finetuning.data.split import SplitConfig, SplitError, load_split_config, read_manifest

SCHEMA = 1
BUNDLE_ENV = "QLORA_DATA_BUNDLE"
EXAMPLES_NAME = "examples.jsonl"
RECORD_NAME = "bundle.json"
BUNDLE_SPLITS = ("fit", "dev")  # what a training, scoring, or dev evaluation job is given
TEST = "test"
TEST_SPLITS = (TEST,)  # the other kind of bundle (req_005 v5.01): test rows and nothing else


class BundleError(SplitError):
    """A bundle is missing, altered, built from other data, or would hold a test row."""


def _split_of(manifest: dict[str, Any]) -> dict[str, str]:
    return dict(manifest["assignments"])


def _line(example: RawExample) -> str:
    # Key order is kept (the evidence keys of gold_inds are in document order), and the
    # text is ASCII-escaped, so the same examples always give the same bytes.
    return json.dumps(dataclasses.asdict(example), ensure_ascii=True, separators=(",", ":"))


def export_bundle(
    folder: Path | str,
    splits: tuple[str, ...] = BUNDLE_SPLITS,
    cfg: SplitConfig | None = None,
    data_cfg: DataConfig | None = None,
) -> dict[str, Any]:
    """Write the bundle of ``splits`` into ``folder`` and return its record.

    Reads the snapshot through ``load_split``, so the pinned checksums and the manifest are
    checked first. ``splits`` is ``fit`` and ``dev`` (or one of them), or ``("test",)`` alone:
    anything else raises ``BundleError``, and so does a row that belongs to another split.
    The test bundle raises ``TestLockedError`` until req_005 v5.01 has unlocked the split.
    The same data gives the same bytes.
    """
    # Always the snapshot, never another bundle. (The loader imports this module.)
    from qlora_finetuning.data.split_loader import load_split_from_snapshot as load_split

    splits = tuple(splits)
    refused = [name for name in splits if name not in BUNDLE_SPLITS]
    if splits != TEST_SPLITS and (refused or not splits or len(set(splits)) != len(splits)):
        raise BundleError(
            f"a bundle holds {' and '.join(BUNDLE_SPLITS)}, or {TEST} alone; refused: "
            f"{', '.join(refused) or 'an empty or repeated list'}. Test rows are never mixed "
            "with training rows"
        )
    cfg = cfg or load_split_config()
    data_cfg = data_cfg or load_data_config()
    manifest = read_manifest(cfg)
    assigned = _split_of(manifest)

    lines: list[str] = []
    counts: dict[str, int] = {}
    for name in splits:
        examples = load_split(name, cfg, data_cfg)
        wrong = [e.id for e in examples if assigned.get(e.id) != name]
        if wrong:  # cannot happen with an honest loader; this is the last guard before a file
            raise BundleError(
                f"{len(wrong)} examples of {name} are assigned elsewhere, for example {wrong[0]}"
            )
        counts[name] = len(examples)
        lines.extend(_line(example) for example in examples)

    content = ("\n".join(lines) + "\n").encode("ascii")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / EXAMPLES_NAME).write_bytes(content)
    record = {
        "schema": SCHEMA,
        "dataset": {"repo_id": data_cfg.repo_id, "revision": data_cfg.revision},
        "manifest_hash": str(manifest["content_hash"]),
        "splits": counts,
        "examples": len(lines),
        "file": EXAMPLES_NAME,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    text = json.dumps(record, indent=2, ensure_ascii=True) + "\n"
    (folder / RECORD_NAME).write_text(text, encoding="utf-8", newline="\n")
    return record


def read_record(folder: Path | str) -> dict[str, Any]:
    """The bundle's record, with its shape checked. Reads no example."""
    path = Path(folder) / RECORD_NAME
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise BundleError(
            f"no data bundle at {Path(folder)}: write one with "
            "python -m scripts.sagemaker_job bundle"
        ) from None
    except ValueError:
        raise BundleError(f"the bundle record {path} is not valid JSON") from None
    if (
        not isinstance(record, dict)
        or record.get("schema") != SCHEMA
        or not isinstance(record.get("splits"), dict)
        or not isinstance(record.get("dataset"), dict)
    ):
        raise BundleError(f"the bundle record {path} is damaged")
    return record


def read_bundle(
    folder: Path | str,
    cfg: SplitConfig | None = None,
    data_cfg: DataConfig | None = None,
    manifest: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[RawExample]]:
    """The bundle's record and its examples, checked against the current project.

    Raises ``BundleError`` unless the file has the recorded hash, the bundle was built from
    the pinned dataset commit and the current split manifest, it names ``fit`` and ``dev``
    only or ``test`` only, and every example in it belongs to one of the splits it names.
    """
    folder = Path(folder)
    record = read_record(folder)
    cfg = cfg or load_split_config()
    data_cfg = data_cfg or load_data_config()
    manifest = manifest or read_manifest(cfg)

    extra = [name for name in record["splits"] if name not in BUNDLE_SPLITS]
    if extra and tuple(record["splits"]) != TEST_SPLITS:
        raise BundleError(f"the bundle names a split it may not hold: {', '.join(extra)}")
    dataset = record["dataset"]
    if dataset.get("repo_id") != data_cfg.repo_id or dataset.get("revision") != data_cfg.revision:
        raise BundleError(
            "the bundle was built from another dataset commit than config/data.yaml pins; "
            "write it again"
        )
    if record.get("manifest_hash") != manifest["content_hash"]:
        raise BundleError("the bundle was built from another split manifest; write it again")

    path = folder / EXAMPLES_NAME
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        raise BundleError(f"the bundle has no {EXAMPLES_NAME}: {folder}") from None
    if hashlib.sha256(content).hexdigest() != record.get("sha256"):
        raise BundleError(f"{path.name} is not the file the bundle's record describes")

    assigned = _split_of(manifest)
    examples: list[RawExample] = []
    for number, line in enumerate(content.decode("ascii").splitlines(), start=1):
        try:
            examples.append(RawExample(**json.loads(line)))
        except (ValueError, TypeError):
            raise BundleError(f"{path.name} line {number} is not an example") from None
    outside = [e.id for e in examples if assigned.get(e.id) not in record["splits"]]
    if outside:
        kind = "test rows" if any(assigned.get(i) == TEST for i in outside) else "rows"
        raise BundleError(
            f"the bundle holds {len(outside)} {kind} outside the splits it names, "
            f"for example {outside[0]}; do not use it"
        )
    return record, examples


def bundle_split(
    name: str,
    folder: Path | str,
    cfg: SplitConfig,
    manifest: dict[str, Any],
    data_cfg: DataConfig | None = None,
    ids: list[str] | None = None,
) -> list[RawExample]:
    """The examples of ``name`` from a bundle, in dataset order. ``ids`` are the manifest's.

    ``quick_dev`` is served from ``dev``. A split the bundle does not hold is an error, and
    the bundle's ids for the split must be exactly the manifest's, in order.
    """
    record, examples = read_bundle(folder, cfg, data_cfg, manifest)
    source = "dev" if name == "quick_dev" else name
    if source not in record["splits"]:
        held = " and ".join(record["splits"]) or "nothing"
        raise BundleError(f"the data bundle holds {held}, not {name}")
    wanted = list(ids if ids is not None else [])
    by_id = {example.id: example for example in examples}
    missing = [example_id for example_id in wanted if example_id not in by_id]
    if missing:
        raise BundleError(
            f"the bundle lacks {len(missing)} examples of {name}, for example {missing[0]}"
        )
    chosen = set(wanted)
    selected = [example for example in examples if example.id in chosen]
    if [example.id for example in selected] != wanted:
        raise BundleError(f"the bundle's examples of {name} are not in the manifest's order")
    return selected
