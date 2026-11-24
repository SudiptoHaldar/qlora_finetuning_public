"""Tests for qlora_finetuning.data.bundle and the bundle mode of load_split (req_003 v3.03)."""

from __future__ import annotations

import json

import pytest

from qlora_finetuning.data import TestLockedError, load_split, split_ids
from qlora_finetuning.data.bundle import (
    BUNDLE_ENV,
    EXAMPLES_NAME,
    RECORD_NAME,
    BundleError,
    export_bundle,
    read_bundle,
    read_record,
)
from qlora_finetuning.data.split import read_manifest
from tests.data.synthetic import SECRET_WORDS, build_frozen_split


@pytest.fixture
def frozen(tmp_path):
    return build_frozen_split(tmp_path)  # split_cfg, data_cfg, manifest: fit 30, dev 8, test 22


@pytest.fixture
def bundle(frozen, tmp_path):
    split_cfg, data_cfg, _ = frozen
    folder = tmp_path / "bundle"
    export_bundle(folder, cfg=split_cfg, data_cfg=data_cfg)
    return folder


def rewrite(folder, change):
    """Change the bundle's lines and make its record agree, as a careless edit would."""
    import hashlib

    path = folder / EXAMPLES_NAME
    lines = change(path.read_text(encoding="ascii").splitlines())
    content = ("\n".join(lines) + "\n").encode("ascii")
    path.write_bytes(content)
    record = json.loads((folder / RECORD_NAME).read_text(encoding="utf-8"))
    record.update(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    (folder / RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")


# --------------------------------------------------------------------------- writing


def test_a_bundle_holds_fit_and_dev_and_says_what_it_is(frozen, tmp_path):
    split_cfg, data_cfg, manifest = frozen
    record = export_bundle(tmp_path / "bundle", cfg=split_cfg, data_cfg=data_cfg)
    assert record["splits"] == {"fit": 30, "dev": 8} and record["examples"] == 38
    assert record["dataset"] == {"repo_id": data_cfg.repo_id, "revision": data_cfg.revision}
    assert record["manifest_hash"] == manifest["content_hash"]
    content = (tmp_path / "bundle" / EXAMPLES_NAME).read_bytes()
    assert record["bytes"] == len(content) and len(content.splitlines()) == 38
    assert read_record(tmp_path / "bundle") == record
    # It is the dataset's text: it belongs under artifacts/, never in a committed file.
    assert any(word in content.decode("ascii") for word in SECRET_WORDS)


def test_no_test_id_is_in_a_bundle(frozen, bundle):
    split_cfg, _, _ = frozen
    written = {json.loads(line)["id"] for line in (bundle / EXAMPLES_NAME).read_text().splitlines()}
    assert written == set(split_ids("fit", split_cfg)) | set(split_ids("dev", split_cfg))
    assert not written & set(split_ids("test", split_cfg))


def test_the_same_data_gives_the_same_bytes(frozen, tmp_path):
    split_cfg, data_cfg, _ = frozen
    first = export_bundle(tmp_path / "one", cfg=split_cfg, data_cfg=data_cfg)
    second = export_bundle(tmp_path / "two", cfg=split_cfg, data_cfg=data_cfg)
    assert first == second
    assert (tmp_path / "one" / EXAMPLES_NAME).read_bytes() == (
        tmp_path / "two" / EXAMPLES_NAME
    ).read_bytes()


@pytest.mark.parametrize(
    "splits",
    [
        ("fit", "test"),
        ("test", "dev"),
        ("test", "test"),
        ("quick_dev",),
        (),
        ("fit", "fit"),
        ("fit", "dev", "all"),
    ],
)
def test_only_fit_and_dev_or_test_alone_may_be_bundled(frozen, tmp_path, splits):
    split_cfg, data_cfg, _ = frozen
    with pytest.raises(BundleError, match="a bundle holds fit and dev, or test alone"):
        export_bundle(tmp_path / "bundle", splits, cfg=split_cfg, data_cfg=data_cfg)
    assert not (tmp_path / "bundle").exists()


def unlock(split_cfg, manifest) -> None:
    split_cfg.unlock_file.parent.mkdir(parents=True, exist_ok=True)
    record = {"manifest_hash": manifest["content_hash"]}
    split_cfg.unlock_file.write_text(json.dumps(record), encoding="utf-8")


def test_the_test_bundle_cannot_be_written_while_the_split_is_locked(frozen, tmp_path):
    split_cfg, data_cfg, _ = frozen
    with pytest.raises(TestLockedError, match="no unlock record exists"):
        export_bundle(tmp_path / "bundle_test", ("test",), cfg=split_cfg, data_cfg=data_cfg)
    assert not (tmp_path / "bundle_test").exists()


def test_the_test_bundle_holds_the_test_examples_and_nothing_else(frozen, tmp_path, monkeypatch):
    split_cfg, data_cfg, manifest = frozen
    unlock(split_cfg, manifest)
    folder = tmp_path / "bundle_test"
    record = export_bundle(folder, ("test",), cfg=split_cfg, data_cfg=data_cfg)
    assert record["splits"] == {"test": 22} and record["examples"] == 22
    _, examples = read_bundle(folder, split_cfg, data_cfg)
    assert [example.id for example in examples] == split_ids("test", split_cfg)

    monkeypatch.setenv(BUNDLE_ENV, str(folder))
    assert len(load_split("test", split_cfg, data_cfg)) == 22  # in a job, from the bundle
    for name in ("fit", "dev", "quick_dev"):
        with pytest.raises(BundleError, match=f"holds test, not {name}"):
            load_split(name, split_cfg, data_cfg)

    split_cfg.unlock_file.unlink()  # the job's code holds no unlock record: still locked
    with pytest.raises(TestLockedError, match="no unlock record exists"):
        load_split("test", split_cfg, data_cfg)


def test_a_bundle_that_mixes_test_with_training_rows_is_refused(frozen, bundle, tmp_path):
    split_cfg, data_cfg, _ = frozen
    path = bundle / RECORD_NAME
    record = json.loads(path.read_text(encoding="utf-8"))
    record["splits"]["test"] = 22
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(BundleError, match="names a split it may not hold: test"):
        read_bundle(bundle, split_cfg, data_cfg)


def test_a_bundle_of_one_split(frozen, tmp_path):
    split_cfg, data_cfg, _ = frozen
    record = export_bundle(tmp_path / "bundle", ("dev",), cfg=split_cfg, data_cfg=data_cfg)
    assert record["splits"] == {"dev": 8}


# --------------------------------------------------------------------------- reading


def test_reading_gives_the_examples_as_the_snapshot_has_them(frozen, bundle):
    split_cfg, data_cfg, _ = frozen
    record, examples = read_bundle(bundle, split_cfg, data_cfg)
    from_snapshot = load_split("fit", split_cfg, data_cfg) + load_split("dev", split_cfg, data_cfg)
    assert examples == from_snapshot  # every field of every example, in the same order
    assert record["examples"] == len(examples)


def test_a_changed_file_is_refused(frozen, bundle):
    split_cfg, data_cfg, _ = frozen
    path = bundle / EXAMPLES_NAME
    path.write_bytes(path.read_bytes().replace(b"zebrafish", b"zebrafosh", 1))
    with pytest.raises(BundleError, match="is not the file the bundle's record describes"):
        read_bundle(bundle, split_cfg, data_cfg)


def test_a_test_row_in_a_bundle_is_refused_on_reading(frozen, bundle):
    """However it got there: the record agrees with the file, and the row is still refused."""
    split_cfg, data_cfg, _ = frozen
    test_id = split_ids("test", split_cfg)[0]

    def smuggle(lines):
        row = json.loads(lines[0])
        row["id"] = test_id
        return [*lines, json.dumps(row)]

    rewrite(bundle, smuggle)
    with pytest.raises(BundleError, match="holds 1 test rows outside the splits it names"):
        read_bundle(bundle, split_cfg, data_cfg)


def test_a_bundle_for_other_data_is_refused(frozen, bundle):
    split_cfg, data_cfg, _ = frozen
    path = bundle / RECORD_NAME
    record = json.loads(path.read_text(encoding="utf-8"))
    for change, message in (
        ({"manifest_hash": "f" * 64}, "another split manifest"),
        ({"dataset": {"repo_id": "x/y", "revision": "a" * 40}}, "another dataset commit"),
        ({"splits": {"fit": 30, "test": 22}}, "a split it may not hold: test"),
        ({"schema": 99}, "is damaged"),
    ):
        path.write_text(json.dumps(record | change), encoding="utf-8")
        with pytest.raises(BundleError, match=message):
            read_bundle(bundle, split_cfg, data_cfg)
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(BundleError, match="not valid JSON"):
        read_record(bundle)
    with pytest.raises(BundleError, match="no data bundle at .* scripts.sagemaker_job bundle"):
        read_record(bundle / "absent")


# --------------------------------------------------------------------------- load_split in a job


def test_load_split_serves_the_same_examples_from_a_bundle(frozen, bundle, monkeypatch):
    split_cfg, data_cfg, _ = frozen
    expected = {name: load_split(name, split_cfg, data_cfg) for name in ("fit", "dev", "quick_dev")}
    monkeypatch.setenv(BUNDLE_ENV, str(bundle))
    for name, examples in expected.items():
        assert load_split(name, split_cfg, data_cfg) == examples
    assert [e.id for e in load_split("quick_dev", split_cfg, data_cfg)] == split_ids(
        "quick_dev", split_cfg
    )


def test_a_job_needs_no_snapshot(frozen, bundle, monkeypatch):
    split_cfg, data_cfg, _ = frozen
    import shutil

    from qlora_finetuning.data import snapshot_path
    from qlora_finetuning.data.snapshot import SnapshotError

    shutil.rmtree(snapshot_path(data_cfg))
    with pytest.raises(SnapshotError, match="file is missing"):
        load_split("fit", split_cfg, data_cfg)
    monkeypatch.setenv(BUNDLE_ENV, str(bundle))
    assert len(load_split("fit", split_cfg, data_cfg)) == 30


def test_a_fit_and_dev_bundle_never_serves_test_even_when_unlocked(frozen, bundle, monkeypatch):
    split_cfg, data_cfg, manifest = frozen
    unlock(split_cfg, manifest)
    assert len(load_split("test", split_cfg, data_cfg)) == 22  # unlocked, from the snapshot
    monkeypatch.setenv(BUNDLE_ENV, str(bundle))
    with pytest.raises(TestLockedError, match="not in this data bundle"):
        load_split("test", split_cfg, data_cfg)


def test_a_bundle_without_the_split_or_out_of_order_is_refused(frozen, tmp_path, monkeypatch):
    split_cfg, data_cfg, _ = frozen
    only_dev = tmp_path / "only_dev"
    export_bundle(only_dev, ("dev",), cfg=split_cfg, data_cfg=data_cfg)
    monkeypatch.setenv(BUNDLE_ENV, str(only_dev))
    with pytest.raises(BundleError, match="holds dev, not fit"):
        load_split("fit", split_cfg, data_cfg)

    shuffled = tmp_path / "shuffled"
    export_bundle(shuffled, cfg=split_cfg, data_cfg=data_cfg)
    rewrite(shuffled, lambda lines: [lines[1], lines[0], *lines[2:]])
    monkeypatch.setenv(BUNDLE_ENV, str(shuffled))
    with pytest.raises(BundleError, match="not in the manifest's order"):
        load_split("fit", split_cfg, data_cfg)

    short = tmp_path / "short"
    export_bundle(short, cfg=split_cfg, data_cfg=data_cfg)
    rewrite(short, lambda lines: lines[1:])
    monkeypatch.setenv(BUNDLE_ENV, str(short))
    with pytest.raises(BundleError, match="lacks 1 examples of fit"):
        load_split("fit", split_cfg, data_cfg)


def test_without_the_variable_nothing_changes(frozen, monkeypatch):
    split_cfg, data_cfg, _ = frozen
    monkeypatch.delenv(BUNDLE_ENV, raising=False)
    assert len(load_split("fit", split_cfg, data_cfg)) == 30
    with pytest.raises(TestLockedError, match="no unlock record exists"):
        load_split("test", split_cfg, data_cfg)
    assert read_manifest(split_cfg)["content_hash"]
