"""Tests for the pinned snapshot: config, download, verification, and the raw reader.

Everything runs on invented rows in a temporary folder with a stand-in downloader.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from datetime import UTC, datetime

import pytest
import yaml

from qlora_finetuning.data import (
    SnapshotError,
    fetch_snapshot,
    load_data_config,
    load_raw_examples,
    read_columns,
    snapshot_path,
    verify_snapshot,
)
from qlora_finetuning.data import snapshot as snapshot_module
from qlora_finetuning.settings import REPO_ROOT, ConfigError
from tests.data.synthetic import REVISION, FakeDownloader, build_snapshot, clean_rows, row

# --------------------------------------------------------------------------- config


def test_real_config_loads_and_pins_a_full_commit():
    cfg = load_data_config()
    assert cfg.repo_id == "bevaya/FinQA"
    assert len(cfg.revision) == 40
    assert cfg.splits == ("train", "validation", "test")
    assert sum(item.rows for item in cfg.files) == 8281
    assert snapshot_path(cfg) == REPO_ROOT / "artifacts" / "data" / "finqa" / cfg.revision
    assert cfg.report_dir == REPO_ROOT / "reports" / "data"
    assert len(cfg.expected["columns"]) == 16


def test_paths_resolve_under_the_given_root(tmp_path):
    cfg = load_data_config(root=tmp_path)
    assert snapshot_path(cfg).is_relative_to(tmp_path)
    assert cfg.report_dir.is_relative_to(tmp_path)


def edited_config(tmp_path, edit):
    """Write the synthetic config, apply ``edit`` to its ``finqa`` block, and load it."""
    build_snapshot(tmp_path)
    path = tmp_path / "config" / "data.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(raw["finqa"])
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_data_config(root=tmp_path, config_dir=tmp_path / "config")


@pytest.mark.parametrize("revision", ["main", "3d6a736", REVISION.upper(), REVISION + "0"])
def test_revision_must_be_a_full_commit_hash(tmp_path, revision):
    with pytest.raises(ConfigError, match="full 40-character commit hash"):
        edited_config(tmp_path, lambda finqa: finqa.update(revision=revision))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda f: f["files"][0].update(sha256="abc"), "sha256 must be 64"),
        (lambda f: f["files"][0].update(bytes=0), "bytes must be a positive integer"),
        (lambda f: f["files"][0].update(rows=True), "rows must be a positive integer"),
        (lambda f: f["files"][0].pop("rows"), "is missing: rows"),
        (lambda f: f["files"][0].update(path="../outside.parquet"), "must be a relative path"),
        (lambda f: f["files"][0].update(path="C:/data/x.parquet"), "must be a relative path"),
        (lambda f: f["files"][1].update(official_split="train"), "repeats official_split: train"),
        (lambda f: f.update(files="train.parquet"), "finqa.files must be a list"),
        (lambda f: f["expected"].update(id_pattern="(("), "id_pattern is not valid"),
        (lambda f: f.pop("revision"), "finqa.revision"),
    ],
)
def test_malformed_config_is_rejected(tmp_path, edit, message):
    with pytest.raises(ConfigError, match=message):
        edited_config(tmp_path, edit)


# --------------------------------------------------------------------------- fetch


def test_fetch_downloads_missing_files_and_records_the_snapshot(tmp_path):
    cfg = build_snapshot(tmp_path, place_files=False)
    download = FakeDownloader(tmp_path / "source")
    moment = datetime(2026, 11, 15, 6, 15, tzinfo=UTC)

    downloaded = fetch_snapshot(cfg, download=download, now=moment)

    assert downloaded == [item.path for item in cfg.files]
    assert download.calls[0] == (cfg.repo_id, cfg.files[0].path, REVISION, False)
    assert all(check.ok for check in verify_snapshot(cfg))
    record = json.loads((snapshot_path(cfg) / "snapshot.json").read_text(encoding="utf-8"))
    assert record["revision"] == REVISION
    assert record["fetched_at"] == "2026-11-15T06:15:00Z"
    assert record["downloaded"] == downloaded
    assert [item["sha256"] for item in record["files"]] == [item.sha256 for item in cfg.files]
    assert set(record["libraries"]) == {"huggingface_hub", "pyarrow"}


def test_second_fetch_downloads_nothing_and_keeps_the_record(tmp_path):
    cfg = build_snapshot(tmp_path, place_files=False)
    fetch_snapshot(cfg, download=FakeDownloader(tmp_path / "source"))
    record_path = snapshot_path(cfg) / "snapshot.json"
    before = record_path.read_text(encoding="utf-8")

    again = FakeDownloader(tmp_path / "source")
    assert fetch_snapshot(cfg, download=again, now=datetime(2030, 1, 1, tzinfo=UTC)) == []
    assert again.calls == []
    assert record_path.read_text(encoding="utf-8") == before


def test_fetch_downloads_only_the_missing_file(tmp_path):
    cfg = build_snapshot(tmp_path)
    missing = cfg.files[1].path
    (snapshot_path(cfg) / missing).unlink()
    download = FakeDownloader(tmp_path / "source")
    assert fetch_snapshot(cfg, download=download) == [missing]
    assert download.filenames == [missing]


def test_wrong_checksum_after_download_raises_and_deletes_nothing(tmp_path):
    cfg = build_snapshot(tmp_path, place_files=False)
    bad = cfg.files[0]
    download = FakeDownloader(tmp_path / "source", corrupt=(bad.path,))
    with pytest.raises(SnapshotError) as caught:
        fetch_snapshot(cfg, download=download)
    message = str(caught.value)
    assert bad.path in message
    assert f"expected {bad.size:,}" in message
    assert "Nothing was deleted" in message
    assert (snapshot_path(cfg) / bad.path).is_file()
    assert not (snapshot_path(cfg) / "snapshot.json").exists()


def test_present_but_wrong_file_is_left_alone_without_force(tmp_path):
    cfg = build_snapshot(tmp_path)
    target = snapshot_path(cfg) / cfg.files[2].path
    data = bytearray(target.read_bytes())
    data[-1] ^= 0xFF  # same size, different content
    target.write_bytes(bytes(data))
    download = FakeDownloader(tmp_path / "source")

    with pytest.raises(SnapshotError, match="--force") as caught:
        fetch_snapshot(cfg, download=download)
    assert cfg.files[2].sha256 in str(caught.value)
    assert download.calls == []
    assert target.read_bytes() == bytes(data)

    assert fetch_snapshot(cfg, download=download, force=True) == [i.path for i in cfg.files]
    assert all(call[3] is True for call in download.calls)
    assert all(check.ok for check in verify_snapshot(cfg))


def test_download_error_is_reported_in_one_line(tmp_path):
    cfg = build_snapshot(tmp_path, place_files=False)
    with pytest.raises(SnapshotError) as caught:
        fetch_snapshot(cfg, download=FakeDownloader(tmp_path / "source", fail=True))
    message = str(caught.value)
    assert "ConnectionError: no route to host" in message
    assert "second line" not in message


def test_default_downloader_explains_a_missing_library(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # makes the import fail
    cfg = build_snapshot(tmp_path, place_files=False)
    with pytest.raises(SnapshotError, match="huggingface_hub is not installed"):
        fetch_snapshot(cfg)


# --------------------------------------------------------------------------- verify


def test_verify_reports_each_kind_of_problem(tmp_path):
    cfg = build_snapshot(tmp_path)
    folder = snapshot_path(cfg)
    train, validation, test = cfg.files
    (folder / train.path).unlink()
    with (folder / validation.path).open("ab") as handle:
        handle.write(b"extra")
    data = bytearray((folder / test.path).read_bytes())
    data[-1] ^= 0xFF
    (folder / test.path).write_bytes(bytes(data))

    checks = {check.official_split: check for check in verify_snapshot(cfg)}
    assert not any(check.ok for check in checks.values())
    assert checks["train"].detail == "file is missing"
    assert "size is" in checks["validation"].detail
    assert checks["test"].detail.startswith("sha256 is ")
    assert test.sha256 in checks["test"].detail


def test_verify_checks_the_row_count(tmp_path):
    cfg = build_snapshot(tmp_path)
    wrong = snapshot_module.SnapshotFile(**{**cfg.files[0].__dict__, "rows": cfg.files[0].rows + 1})
    check = snapshot_module._check_file(snapshot_path(cfg), wrong)
    assert not check.ok
    assert check.detail == "has 3 rows, expected 4"


def test_verified_file_detail_shows_size_rows_and_checksum(tmp_path):
    cfg = build_snapshot(tmp_path)
    check = verify_snapshot(cfg)[0]
    assert check.ok
    assert "3 rows" in check.detail
    assert cfg.files[0].sha256[:12] in check.detail


# --------------------------------------------------------------------------- reader


def test_examples_come_back_in_order_as_plain_values(tmp_path):
    cfg = build_snapshot(tmp_path)
    examples = load_raw_examples(cfg)
    expected_ids = [r["id"] for rows in clean_rows().values() for r in rows]
    assert [example.id for example in examples] == expected_ids
    assert [e.official_split for e in examples] == ["train"] * 3 + ["validation", "test"]

    first = examples[0]
    assert type(first.pre_text) is list and type(first.table[0]) is list
    assert first.table[1] == ["xylophone sales", "920", "95"]
    assert first.ann_table_rows == [1] and first.ann_text_rows == []
    assert first.steps == {"op": ["minus1-1"], "arg1": ["920"], "arg2": ["95"], "res": ["825"]}
    assert first.program == "subtract(920, 95)"
    assert first.explanation == ""


def test_gold_inds_is_parsed_in_its_original_order(tmp_path):
    rows = {"train": [row(gold_inds={"text_3": "b", "table_1": "a", "text_0": "c"})]}
    example = load_raw_examples(build_snapshot(tmp_path, rows))[0]
    assert list(example.gold_inds) == ["text_3", "table_1", "text_0"]
    assert example.gold_inds["table_1"] == "a"


@pytest.mark.parametrize("stored", ["not json", "[1, 2]", '"text"'])
def test_gold_inds_that_is_not_a_json_object_becomes_none(tmp_path, stored):
    rows = {"train": [row(gold_inds=stored)]}
    assert load_raw_examples(build_snapshot(tmp_path, rows))[0].gold_inds is None


def test_id_parts_are_read_from_the_id(tmp_path):
    example = load_raw_examples(build_snapshot(tmp_path))[1]
    assert (example.ticker, example.year, example.page, example.index) == ("ACME", 2015, 10, 2)
    assert example.report == "ACME/2015"
    assert example.filename == "ACME/2015/page_10.pdf"


def test_id_that_does_not_match_leaves_the_parts_empty(tmp_path):
    rows = {"train": [row("not-an-id", filename="somewhere")]}
    example = load_raw_examples(build_snapshot(tmp_path, rows))[0]
    assert example.id == "not-an-id"
    assert (example.ticker, example.year, example.page, example.index) == (None,) * 4
    assert example.report is None


def test_missing_snapshot_names_the_fetch_command(tmp_path):
    cfg = build_snapshot(tmp_path, place_files=False)
    with pytest.raises(SnapshotError, match="python -m scripts.finqa_fetch"):
        load_raw_examples(cfg)


def test_altered_file_is_refused(tmp_path):
    cfg = build_snapshot(tmp_path)
    target = snapshot_path(cfg) / cfg.files[0].path
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0xFF
    target.write_bytes(bytes(data))
    with pytest.raises(SnapshotError, match="sha256 is"):
        load_raw_examples(cfg)


def test_snapshot_error_is_a_config_error():
    assert issubclass(SnapshotError, ConfigError)


# --------------------------------------------------------------------------- schema


def test_column_types_are_named_plainly(tmp_path):
    cfg = build_snapshot(tmp_path)
    columns = read_columns(cfg)
    assert list(columns) == ["train", "validation", "test"]
    assert columns["train"] == {str(k): str(v) for k, v in cfg.expected["columns"].items()}
    assert columns["train"]["table"] == "list[list[string]]"
    assert columns["train"]["steps"].startswith("struct{op: list[string], arg1:")


def test_type_name_falls_back_to_the_arrow_name():
    import pyarrow as pa

    assert snapshot_module.type_name(pa.int64()) == "int64"
    assert snapshot_module.type_name(pa.large_list(pa.large_string())) == "list[string]"


# --------------------------------------------------------------------------- live


@pytest.mark.network
def test_pinned_commit_lists_the_configured_files():
    """The Hub still serves the pinned commit, with the sizes and checksums in the config."""
    cfg = load_data_config()
    url = f"https://huggingface.co/api/datasets/{cfg.repo_id}/tree/{cfg.revision}/data"
    with urllib.request.urlopen(url, timeout=30) as response:
        listing = {item["path"]: item for item in json.load(response)}
    for item in cfg.files:
        assert listing[item.path]["size"] == item.size
        assert listing[item.path]["lfs"]["oid"] == item.sha256
