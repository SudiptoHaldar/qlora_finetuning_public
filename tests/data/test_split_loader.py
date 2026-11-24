"""Tests for the split loaders and the test lock, plus guards on the repository itself."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from qlora_finetuning.comparison.config import load_comparison_config
from qlora_finetuning.comparison.gate import unlock_problems
from qlora_finetuning.data import (
    SplitError,
    TestLockedError,
    is_test_unlocked,
    load_data_config,
    load_raw_examples,
    load_split,
    load_split_config,
    manifest_hash,
    split_ids,
    split_loader,
    verify_snapshot,
)
from qlora_finetuning.data.split import build_split, read_manifest, render_manifest
from qlora_finetuning.data.split_loader import check_manifest_against_data, unlock_state
from tests.data.synthetic import build_frozen_split, build_snapshot, row, split_rows

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def frozen(tmp_path):
    return build_frozen_split(tmp_path)


def unlock(cfg, record):
    cfg.unlock_file.parent.mkdir(parents=True, exist_ok=True)
    text = record if isinstance(record, str) else json.dumps(record)
    cfg.unlock_file.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- loading


@pytest.mark.parametrize("name", ["fit", "dev"])
def test_split_comes_back_in_dataset_order(frozen, name):
    cfg, data_cfg, manifest = frozen
    wanted = [i for i, split in manifest["assignments"].items() if split == name]
    examples = load_split(name, cfg, data_cfg)
    assert [example.id for example in examples] == wanted
    assert split_ids(name, cfg) == wanted
    assert all(manifest["groups"][example.ticker] == name for example in examples)


def test_quick_dev_returns_the_marked_dev_examples(frozen):
    cfg, data_cfg, manifest = frozen
    quick = load_split("quick_dev", cfg, data_cfg)
    assert [example.id for example in quick] == manifest["quick_dev"]
    assert {example.id for example in quick} <= set(split_ids("dev", cfg))
    assert len(quick) == cfg.quick_dev_size


def test_fit_dev_and_test_cover_everything_without_overlap(frozen):
    cfg, _, manifest = frozen
    parts = [split_ids(name, cfg) for name in ("fit", "dev", "test")]
    assert sum(len(part) for part in parts) == len(manifest["assignments"]) == 60
    assert len(set().union(*map(set, parts))) == 60


@pytest.mark.parametrize("name", ["train", "validation", "TEST", ""])
def test_unknown_split_name(frozen, name):
    cfg, data_cfg, _ = frozen
    with pytest.raises(SplitError, match="use one of: fit, dev, quick_dev, test"):
        load_split(name, cfg, data_cfg)
    with pytest.raises(SplitError, match="unknown split"):
        split_ids(name, cfg)


def test_manifest_hash_is_the_content_hash(frozen):
    cfg, _, manifest = frozen
    assert manifest_hash(cfg) == manifest["content_hash"]


def test_missing_manifest_stops_every_loader(tmp_path):
    cfg, data_cfg, _ = build_frozen_split(tmp_path, write=False)
    for call in (lambda: load_split("fit", cfg, data_cfg), lambda: manifest_hash(cfg)):
        with pytest.raises(SplitError, match="python -m scripts.finqa_split"):
            call()


# --------------------------------------------------------------------------- manifest vs data


def test_manifest_for_another_dataset_revision_is_refused(frozen):
    cfg, data_cfg, _ = frozen
    other = dataclasses.replace(data_cfg, revision="f" * 40)
    with pytest.raises(SplitError, match="built from another dataset revision"):
        check_manifest_against_data(read_manifest(cfg), load_raw_examples(data_cfg), other)


def test_manifest_for_other_file_checksums_is_refused(frozen):
    cfg, data_cfg, _ = frozen
    files = (dataclasses.replace(data_cfg.files[0], sha256="0" * 64), *data_cfg.files[1:])
    other = dataclasses.replace(data_cfg, files=files)
    with pytest.raises(SplitError, match="other checksums"):
        check_manifest_against_data(read_manifest(cfg), load_raw_examples(data_cfg), other)


def test_example_without_an_assignment_is_refused(frozen):
    cfg, data_cfg, manifest = frozen
    examples = load_raw_examples(data_cfg)
    extra = dataclasses.replace(examples[0], id="ZZZ/2015/page_1.pdf-1")
    with pytest.raises(SplitError, match="1 examples have no assignment, for example ZZZ/"):
        check_manifest_against_data(manifest, [*examples, extra], data_cfg)


def test_assignment_without_an_example_is_refused(frozen):
    cfg, data_cfg, manifest = frozen
    examples = load_raw_examples(data_cfg)
    with pytest.raises(SplitError, match="1 assigned ids are not in the data"):
        check_manifest_against_data(manifest, examples[1:], data_cfg)
    with pytest.raises(SplitError, match="different order"):
        check_manifest_against_data(manifest, list(reversed(examples)), data_cfg)


def test_loading_fails_when_the_data_changed_under_the_manifest(tmp_path):
    cfg, _, _ = build_frozen_split(tmp_path)
    rows = split_rows()
    rows["train"].append(row("NEW/2015/page_1.pdf-1"))
    changed = build_snapshot(tmp_path / "other", rows)
    with pytest.raises(SplitError, match="other checksums"):
        load_split("fit", cfg, changed)


# --------------------------------------------------------------------------- the lock


def test_test_split_is_locked_without_an_unlock_record(frozen):
    cfg, data_cfg, _ = frozen
    with pytest.raises(TestLockedError, match="no unlock record exists") as caught:
        load_split("test", cfg, data_cfg)
    assert "req_005 v5.01" in str(caught.value)
    assert is_test_unlocked(cfg) is False


def test_locked_call_reads_no_data(frozen, monkeypatch):
    cfg, data_cfg, _ = frozen

    def forbidden(*args, **kwargs):
        raise AssertionError("the data was read while the test split was locked")

    monkeypatch.setattr(split_loader, "load_raw_examples", forbidden)
    with pytest.raises(TestLockedError):
        load_split("test", cfg, data_cfg)


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        ({"manifest_hash": "0" * 64}, "names another manifest"),
        ({"unlocked_by": "someone"}, "does not name a manifest hash"),
        (["not", "an", "object"], "does not name a manifest hash"),
        ("{broken", "is not valid JSON"),
        ("", "is not valid JSON"),
    ],
)
def test_a_stray_or_stale_unlock_file_does_not_unlock(frozen, record, reason):
    cfg, data_cfg, manifest = frozen
    unlock(cfg, record)
    assert unlock_state(cfg, manifest) == (False, f"the unlock record test_unlock.json {reason}")
    with pytest.raises(TestLockedError, match=reason):
        load_split("test", cfg, data_cfg)
    assert is_test_unlocked(cfg) is False


def test_a_record_naming_this_manifest_unlocks(frozen):
    cfg, data_cfg, manifest = frozen
    unlock(cfg, {"manifest_hash": manifest["content_hash"], "unlocked_by": "req_005 v5.01"})
    assert is_test_unlocked(cfg) is True
    test = load_split("test", cfg, data_cfg)
    assert [example.id for example in test] == split_ids("test", cfg)
    assert all(manifest["groups"][example.ticker] == "test" for example in test)


def test_test_ids_are_available_while_locked(frozen):
    cfg, _, manifest = frozen
    ids = split_ids("test", cfg)
    assert ids and all(manifest["assignments"][example_id] == "test" for example_id in ids)


def test_other_splits_do_not_need_the_unlock(frozen):
    cfg, data_cfg, _ = frozen
    assert load_split("fit", cfg, data_cfg)
    assert not cfg.unlock_file.exists()


def test_test_locked_error_is_a_split_error_and_not_collected_by_pytest():
    assert issubclass(TestLockedError, SplitError)
    assert TestLockedError.__test__ is False


# --------------------------------------------------------------------------- repository guards

# The only files that may use the raw reader, which returns every row including test rows.
RAW_READER_ALLOWED = (
    "qlora_finetuning/data/",
    "scripts/finqa_gold_check.py",
    "scripts/finqa_report.py",
    "scripts/finqa_split.py",
    "tests/data/",
    "tests/scripts/test_finqa_report.py",
    "tests/scripts/test_finqa_split.py",
)
RAW_READER = re.compile(r"\bload_raw_examples\b|\bread_table\b")


def raw_reader_users() -> list[str]:
    users = []
    for folder in ("qlora_finetuning", "scripts", "tests"):
        for path in sorted((REPO_ROOT / folder).rglob("*.py")):
            relative = path.relative_to(REPO_ROOT).as_posix()
            if RAW_READER.search(path.read_text(encoding="utf-8")):
                users.append(relative)
    return users


def test_only_the_data_package_uses_the_raw_reader():
    outside = [path for path in raw_reader_users() if not path.startswith(RAW_READER_ALLOWED)]
    assert not outside, (
        "these files read FinQA rows directly; use qlora_finetuning.data.load_split instead:\n"
        + "\n".join(outside)
    )


def test_three_scripts_and_no_other_may_use_the_raw_reader():
    """v2.02 added the gold self-check, which covers the test rows. Nothing else was added."""
    scripts = [path for path in RAW_READER_ALLOWED if path.startswith("scripts/")]
    assert scripts == [
        "scripts/finqa_gold_check.py",
        "scripts/finqa_report.py",
        "scripts/finqa_split.py",
    ]
    assert len(RAW_READER_ALLOWED) == 7


def test_raw_reader_scan_sees_the_known_users():
    users = raw_reader_users()
    assert "qlora_finetuning/data/snapshot.py" in users
    assert "qlora_finetuning/data/split_loader.py" in users
    assert RAW_READER.search("examples = load_raw_examples(cfg)")
    assert not RAW_READER.search("examples = load_split('fit')")


@pytest.mark.private_records
def test_the_repository_is_locked_or_unlocked_against_its_freeze_record():
    """The owner unlocked the test split on 2026-11-23 (req_005 v5.01).

    Before that, no unlock record existed. Since then, the record must name this manifest
    and the freeze record on disk; anything else leaves the test split exposed or unusable.
    """
    cfg = load_split_config()
    if not cfg.unlock_file.exists():
        assert not is_test_unlocked(cfg)
        return
    assert is_test_unlocked(cfg)
    assert unlock_problems(load_comparison_config(), cfg) == []


def test_committed_manifest_is_valid_and_matches_the_settings():
    cfg = load_split_config()
    manifest = read_manifest(cfg)  # validates the shape and the content hash
    settings, counts = manifest["settings"], manifest["counts"]
    assert settings["group_by"] == cfg.group_by
    assert settings["seed"] == cfg.seed
    assert settings["train_fraction"] == cfg.train_fraction
    assert (settings["dev_size"], settings["quick_dev_size"]) == (cfg.dev_size, cfg.quick_dev_size)
    assert manifest["dataset"]["revision"] == load_data_config().revision
    assert counts["examples"] == len(manifest["assignments"]) == 8281
    assert counts["fit"]["examples"] + counts["dev"]["examples"] == 5383
    assert counts["test"]["examples"] == 2898
    assert len(manifest["quick_dev"]) == 200
    assert manifest_hash() == manifest["content_hash"]


def test_committed_manifest_is_what_the_settings_and_the_snapshot_produce():
    """Skips where the snapshot has not been fetched (CI); the data files are git-ignored."""
    data_cfg = load_data_config()
    if not all(item.ok for item in verify_snapshot(data_cfg)):
        pytest.skip("no FinQA snapshot here; fetch it with: python -m scripts.finqa_fetch")
    cfg = load_split_config()
    rebuilt = build_split(load_raw_examples(data_cfg), cfg, data_cfg)
    assert render_manifest(rebuilt) == render_manifest(read_manifest(cfg))
    assert len(load_split("fit")) == 4884
    assert len(load_split("dev")) == 499
    assert len(load_split("quick_dev")) == 200
    if is_test_unlocked(cfg):
        assert len(load_split("test")) == 2898
    else:
        with pytest.raises(TestLockedError):
            load_split("test")
