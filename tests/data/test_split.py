"""Tests for the split builder and the write-once manifest. All rows are invented."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter

import pytest
import yaml

from qlora_finetuning.data import SplitError, load_data_config, load_split_config
from qlora_finetuning.data.split import (
    MANIFEST_KEYS,
    RULE,
    build_split,
    content_hash,
    fill,
    group_key,
    order_key,
    read_manifest,
    render_manifest,
    validate_manifest,
    write_manifest,
)
from qlora_finetuning.settings import REPO_ROOT, ConfigError
from tests.data.synthetic import (
    COMPANY_SIZES,
    build_frozen_split,
    build_split_config,
    company_rows,
    row,
    to_examples,
)


@pytest.fixture
def examples():
    return to_examples(company_rows())


@pytest.fixture
def cfg(tmp_path):
    return build_split_config(tmp_path)


@pytest.fixture
def data_cfg():
    return load_data_config()  # only its pin is copied into the manifest


# --------------------------------------------------------------------------- config


def test_real_config_loads():
    cfg = load_split_config()
    assert (cfg.group_by, cfg.seed, cfg.train_fraction) == ("company", 20261115, 0.65)
    assert (cfg.dev_size, cfg.quick_dev_size) == (500, 200)
    assert cfg.manifest_file == REPO_ROOT / "splits" / "finqa_split_manifest.json"
    assert cfg.unlock_file == REPO_ROOT / "splits" / "test_unlock.json"
    assert cfg.report_dir == REPO_ROOT / "reports" / "data"


def test_paths_resolve_under_the_given_root(tmp_path):
    cfg = build_split_config(tmp_path)
    assert cfg.manifest_file.is_relative_to(tmp_path)
    assert cfg.unlock_file.is_relative_to(tmp_path)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"group_by": "sector"}, "group_by must be one of company, report, page"),
        ({"train_fraction": 1.5}, "train_fraction must be between 0 and 1"),
        ({"train_fraction": True}, "train_fraction must be between 0 and 1"),
        ({"seed": "abc"}, "seed must be a whole number"),
        ({"dev_size": 0}, "dev_size must be a whole number of at least 1"),
        ({"quick_dev_size": 2.5}, "quick_dev_size must be a whole number"),
        ({"manifest_file": ""}, "is missing: manifest_file"),
    ],
)
def test_malformed_config_is_rejected(tmp_path, override, message):
    with pytest.raises(ConfigError, match=message):
        build_split_config(tmp_path, **override)


# --------------------------------------------------------------------------- the rule


def test_order_key_is_a_fixed_function():
    """These values must never change: the committed split depends on them."""
    assert order_key(20261115, "train", "AAPL") == (
        "4d25a9e7c03c5da93c927e4b045fcb59a9f9f25eef20e3ae64b9e9a466379aa9"
    )
    assert order_key(7, "dev", "ACME") == (
        "571f92a4cdb7d363fe5ad674d3018884926c0523d8c5fe27418ae77c55ab3dda"
    )
    assert order_key(7, "dev", "ACME") != order_key(7, "train", "ACME")
    assert order_key(7, "dev", "ACME") != order_key(8, "dev", "ACME")


def test_fill_passes_over_a_group_that_would_overshoot_and_takes_later_ones():
    sizes = {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}
    # Digest order for seed 1, salt "train" is d, a, c, b, e.
    assert fill(sizes, 1, "train", 8) == ["d", "a", "e"]
    assert fill(sizes, 1, "train", 100) == ["d", "a", "c", "b", "e"]
    assert fill(sizes, 1, "train", 0) == []
    assert fill({}, 1, "train", 5) == []


@pytest.mark.parametrize("target", range(0, 61, 7))
def test_fill_never_exceeds_the_target(target):
    chosen = fill(COMPANY_SIZES, 20261115, "train", target)
    assert sum(COMPANY_SIZES[key] for key in chosen) <= target
    assert len(set(chosen)) == len(chosen)


def test_group_key_for_each_level(examples):
    first = examples[0]
    assert group_key(first, "company") == "AAA"
    assert group_key(first, "report") == "AAA/2016"
    assert group_key(first, "page") == "AAA/2016/page_1.pdf"


def test_example_without_a_group_key_is_rejected(cfg, data_cfg):
    examples = to_examples([row("no-such-format", filename="somewhere")])
    with pytest.raises(SplitError, match="'no-such-format' has no company to group by"):
        build_split(examples, cfg, data_cfg)


def test_repeated_id_is_rejected(cfg, data_cfg, examples):
    with pytest.raises(SplitError, match="example ids are not unique: AAA/2016/page_1.pdf-1"):
        build_split([*examples, examples[0]], cfg, data_cfg)


def test_train_target_for_the_real_dataset():
    assert round(8281 * 0.65) == 5383


# --------------------------------------------------------------------------- the build


def test_every_example_is_assigned_and_groups_stay_whole(examples, cfg, data_cfg):
    manifest = build_split(examples, cfg, data_cfg)
    assignments = manifest["assignments"]
    assert list(assignments) == [example.id for example in examples]
    assert set(assignments.values()) == {"fit", "dev", "test"}
    for example in examples:
        assert assignments[example.id] == manifest["groups"][example.ticker]
    assert list(manifest["groups"]) == sorted(COMPANY_SIZES)


def test_counts_and_sizes(examples, cfg, data_cfg):
    manifest = build_split(examples, cfg, data_cfg)
    counts = manifest["counts"]
    tally = Counter(manifest["assignments"].values())
    assert counts["examples"] == 60
    assert {name: counts[name]["examples"] for name in ("fit", "dev", "test")} == dict(tally)
    assert counts["train"]["examples"] == tally["fit"] + tally["dev"] <= 39  # 65% of 60
    assert counts["train"]["examples"] >= 37
    assert 0 < counts["dev"]["examples"] <= cfg.dev_size
    assert sum(counts[name]["groups"] for name in ("fit", "dev", "test")) == len(COMPANY_SIZES)
    assert counts["train"]["groups"] == counts["fit"]["groups"] + counts["dev"]["groups"]


def test_quick_dev_is_inside_dev_and_of_the_configured_size(examples, cfg, data_cfg):
    manifest = build_split(examples, cfg, data_cfg)
    quick = manifest["quick_dev"]
    assert len(quick) == len(set(quick)) == cfg.quick_dev_size == manifest["counts"]["quick_dev"]
    assert all(manifest["assignments"][example_id] == "dev" for example_id in quick)
    order = list(manifest["assignments"])
    assert quick == sorted(quick, key=order.index)  # dataset order


def test_manifest_records_the_data_and_the_settings(examples, cfg, data_cfg):
    manifest = build_split(examples, cfg, data_cfg)
    assert tuple(manifest) == MANIFEST_KEYS
    assert manifest["dataset"]["revision"] == data_cfg.revision
    assert manifest["dataset"]["files"] == {i.official_split: i.sha256 for i in data_cfg.files}
    assert manifest["settings"] == {
        "group_by": "company",
        "seed": 20261115,
        "train_fraction": 0.65,
        "dev_size": 8,
        "quick_dev_size": 3,
        "rule": RULE,
    }


@pytest.mark.parametrize("group_by", ["company", "report", "page"])
def test_each_grouping_keeps_its_groups_whole(examples, cfg, data_cfg, group_by):
    manifest = build_split(examples, dataclasses.replace(cfg, group_by=group_by), data_cfg)
    splits_per_group: dict[str, set[str]] = {}
    for example in examples:
        key = group_key(example, group_by)
        splits_per_group.setdefault(key, set()).add(manifest["assignments"][example.id])
    assert all(len(found) == 1 for found in splits_per_group.values())
    assert manifest["counts"]["train"]["examples"] <= 39


def test_same_input_gives_the_same_manifest_and_another_seed_a_different_one(
    examples, cfg, data_cfg
):
    first = build_split(examples, cfg, data_cfg)
    assert render_manifest(build_split(examples, cfg, data_cfg)) == render_manifest(first)
    other = build_split(examples, dataclasses.replace(cfg, seed=cfg.seed + 1), data_cfg)
    assert other["content_hash"] != first["content_hash"]


def test_input_order_does_not_change_who_goes_where(examples, cfg, data_cfg):
    forward = build_split(examples, cfg, data_cfg)
    backward = build_split(list(reversed(examples)), cfg, data_cfg)
    assert backward["groups"] == forward["groups"]
    assert set(backward["quick_dev"]) == set(forward["quick_dev"])


def test_dev_size_that_does_not_fit_is_rejected(examples, cfg, data_cfg):
    with pytest.raises(SplitError, match="dev_size 60 does not fit"):
        build_split(examples, dataclasses.replace(cfg, dev_size=60), data_cfg)


def test_quick_dev_larger_than_dev_is_rejected(examples, cfg, data_cfg):
    with pytest.raises(SplitError, match="quick_dev_size 9 is larger than dev"):
        build_split(examples, dataclasses.replace(cfg, quick_dev_size=9), data_cfg)


# --------------------------------------------------------------------------- hash


def test_content_hash_changes_with_one_assignment_and_with_one_quick_dev_flag():
    assignments = {"A/2015/page_1.pdf-1": "fit", "B/2015/page_1.pdf-1": "dev"}
    base = content_hash(assignments, ["B/2015/page_1.pdf-1"])
    assert len(base) == 64
    assert (
        content_hash({**assignments, "A/2015/page_1.pdf-1": "test"}, ["B/2015/page_1.pdf-1"])
        != base
    )
    assert content_hash(assignments, []) != base
    assert content_hash(dict(reversed(assignments.items())), ["B/2015/page_1.pdf-1"]) != base
    assert content_hash(assignments, ("B/2015/page_1.pdf-1",)) == base


# --------------------------------------------------------------------------- manifest file


def test_manifest_round_trips_with_one_assignment_per_line(tmp_path):
    cfg, _, manifest = build_frozen_split(tmp_path)
    assert read_manifest(cfg) == manifest
    text = cfg.manifest_file.read_text(encoding="utf-8")
    assert text == render_manifest(manifest)
    assert text.endswith("}\n") and b"\r\n" not in cfg.manifest_file.read_bytes()
    assert sum(line.strip().startswith('"AAA/') for line in text.splitlines()) == 12


def test_manifest_is_written_once(tmp_path):
    cfg, data_cfg, manifest = build_frozen_split(tmp_path, write=False)
    assert write_manifest(manifest, cfg) is True
    before = cfg.manifest_file.read_bytes()
    assert write_manifest(manifest, cfg) is False
    assert cfg.manifest_file.read_bytes() == before


def test_a_different_manifest_is_never_overwritten(tmp_path):
    cfg, data_cfg, manifest = build_frozen_split(tmp_path)
    before = cfg.manifest_file.read_bytes()
    from qlora_finetuning.data import load_raw_examples

    other = build_split(
        load_raw_examples(data_cfg), dataclasses.replace(cfg, seed=cfg.seed + 1), data_cfg
    )
    with pytest.raises(SplitError, match="The split is frozen and nothing was written"):
        write_manifest(other, cfg)
    assert cfg.manifest_file.read_bytes() == before


def test_identical_manifest_is_accepted_after_a_line_ending_conversion(tmp_path):
    cfg, _, manifest = build_frozen_split(tmp_path)
    cfg.manifest_file.write_bytes(cfg.manifest_file.read_bytes().replace(b"\n", b"\r\n"))
    assert write_manifest(manifest, cfg) is False
    assert read_manifest(cfg) == manifest


def test_missing_manifest_names_the_build_command(tmp_path):
    cfg = build_split_config(tmp_path)
    with pytest.raises(SplitError, match="python -m scripts.finqa_split"):
        read_manifest(cfg)


def edit_manifest(cfg, change):
    manifest = json.loads(cfg.manifest_file.read_text(encoding="utf-8"))
    change(manifest)
    cfg.manifest_file.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda m: m["assignments"].update({"AAA/2016/page_1.pdf-1": "test"}), "content hash"),
        (lambda m: m["assignments"].update({"AAA/2016/page_1.pdf-1": "holdout"}), "unknown split"),
        (lambda m: m["quick_dev"].append("AAA/2016/page_1.pdf-1"), "assigned to dev"),
        (lambda m: m["quick_dev"].append(m["quick_dev"][0]), "distinct ids"),
        (lambda m: m.pop("groups"), "is missing: groups"),
        (lambda m: m.update(schema=2), "schema 2"),
        (lambda m: m.update(assignments=[]), "assignments must be an object"),
    ],
)
def test_hand_edited_manifest_fails_validation(tmp_path, change, message):
    cfg, _, _ = build_frozen_split(tmp_path)
    edit_manifest(cfg, change)
    with pytest.raises(SplitError, match=message):
        read_manifest(cfg)


def test_manifest_that_is_not_json_or_not_an_object(tmp_path):
    cfg, _, _ = build_frozen_split(tmp_path)
    cfg.manifest_file.write_text("{broken", encoding="utf-8")
    with pytest.raises(SplitError, match="not valid JSON"):
        read_manifest(cfg)
    with pytest.raises(SplitError, match="is not a JSON object"):
        validate_manifest([1, 2])


def test_split_error_is_a_config_error():
    assert issubclass(SplitError, ConfigError)


def test_real_config_file_has_no_unknown_keys():
    raw = yaml.safe_load((REPO_ROOT / "config" / "split.yaml").read_text(encoding="utf-8"))
    assert set(raw) == {
        "version", "verified_on", "group_by", "seed", "train_fraction",
        "train_fraction_tolerance", "dev_size", "dev_size_tolerance", "quick_dev_size",
        "leakage_min_digits", "manifest_file", "unlock_file", "report_dir",
    }  # fmt: skip
