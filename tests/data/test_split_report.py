"""Tests for the split report: statistics per split, leakage, and checks P1 to P10."""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest

from qlora_finetuning.data import load_raw_examples
from qlora_finetuning.data.split import build_split, content_hash
from qlora_finetuning.data.split_report import (
    COLUMNS,
    build_report,
    gold_number_pairs,
    leakage,
    render_json,
    render_markdown,
    report_paths,
    run_checks,
    stale_files,
    table_number_pairs,
    write_report,
)
from tests.data.synthetic import SECRET_WORDS, build_frozen_split, row, to_examples


@pytest.fixture
def frozen(tmp_path):
    cfg, data_cfg, manifest = build_frozen_split(tmp_path)
    return cfg, data_cfg, manifest, load_raw_examples(data_cfg)


def checks_for(frozen, manifest=None, rebuilt=None, cfg=None, data_cfg=None, examples=None):
    base_cfg, base_data, base_manifest, base_examples = frozen
    manifest = base_manifest if manifest is None else manifest
    results = run_checks(
        manifest,
        base_manifest if rebuilt is None else rebuilt,
        base_examples if examples is None else examples,
        cfg or base_cfg,
        data_cfg or base_data,
    )
    return {item["id"]: item for item in results}


def rehash(manifest):
    manifest["content_hash"] = content_hash(manifest["assignments"], manifest["quick_dev"])
    return manifest


# --------------------------------------------------------------------------- leakage


def table_row(example_id, table, gold, question=None):
    question = question or f"question about {example_id}"
    return row(example_id, table=table, gold_inds=gold, question=question)


def test_number_pairs_keep_the_row_name_and_the_digits():
    example = to_examples(
        [
            table_row(
                "ACME/2015/page_1.pdf-1",
                [
                    ["", "2015", "2014"],
                    [" net sales ", "$ 12,345", "987"],
                    ["total", "5.5%", "40000"],
                ],
                {"table_1": "x", "table_9": "out of range", "text_0": "y"},
            )
        ]
    )[0]
    assert table_number_pairs(example, 4) == {("net sales", "12345"), ("total", "40000")}
    assert gold_number_pairs(example, 4) == {("net sales", "12345")}
    assert gold_number_pairs(example, 3) == {("net sales", "12345"), ("net sales", "987")}
    assert gold_number_pairs(example, 6) == set()


def test_leakage_counts_what_the_held_out_side_could_have_met():
    seen = to_examples(
        [
            table_row(
                "ACME/2014/page_1.pdf-1", [["", "2014"], ["net sales", "12345"]], {}, "shared?"
            ),
            table_row("BOLT/2014/page_1.pdf-1", [["", "2014"], ["revenue", "77777"]], {}),
        ]
    )
    held_out = to_examples(
        [
            # Same company, another year, evidence row repeats a number seen in training.
            table_row(
                "ACME/2015/page_2.pdf-1",
                [["", "2015", "2014"], ["net sales", "20000", "12345"]],
                {"table_1": "x"},
                "shared?",
            ),
            # Another company whose evidence happens to match a seen table: a chance match.
            table_row(
                "CRUX/2015/page_1.pdf-1", [["", "2015"], ["revenue", "77777"]], {"table_1": "x"}
            ),
            # Evidence in the text only: nothing to check.
            table_row(
                "CRUX/2015/page_2.pdf-1", [["", "2015"], ["revenue", "55555"]], {"text_0": "x"}
            ),
            # Same page as a seen example.
            table_row("BOLT/2014/page_1.pdf-2", [["", "2014"], ["costs", "11"]], {"table_1": "x"}),
        ]
    )
    assert leakage(held_out, seen, 4) == {
        "examples": 4,
        "company_also_on_the_seen_side": 2,
        "report_also_on_the_seen_side": 1,
        "page_also_on_the_seen_side": 1,
        "question_wording_also_on_the_seen_side": 1,
        "examples_with_a_number_to_check": 2,
        "sharing_a_number_with_the_same_company": 1,
        "sharing_a_number_with_any_company": 2,
    }
    assert leakage([], seen, 4)["examples"] == 0
    assert leakage(held_out, [], 4)["sharing_a_number_with_any_company"] == 0


# --------------------------------------------------------------------------- statistics


def test_columns_describe_each_split(frozen):
    cfg, data_cfg, manifest, examples = frozen
    report = build_report(manifest, manifest, examples, cfg, data_cfg)
    columns, counts = report["columns"], manifest["counts"]
    assert tuple(columns) == COLUMNS
    for name in ("fit", "dev", "test", "train"):
        assert columns[name]["size"]["examples"] == counts[name]["examples"]
        assert columns[name]["size"]["groups"] == counts[name]["groups"]
        assert columns[name]["size"]["companies"] == counts[name]["groups"]
    assert columns["quick_dev"]["size"]["examples"] == 3
    assert columns["train"]["size"]["share_of_all_examples_percent"] == round(
        100 * counts["train"]["examples"] / 60, 2
    )
    origin = [columns["train"]["size"][f"from_the_official_{n}_file"] for n in data_cfg.splits]
    assert sum(origin) == counts["train"]["examples"]
    fit = columns["fit"]
    assert fit["program_length"] == {"1": counts["fit"]["examples"]}
    assert fit["operations"] == {"subtract": counts["fit"]["examples"]}
    assert fit["operation_share_percent"] == {"subtract": 100.0}
    assert fit["argument_kinds"] == {"integer": 2 * counts["fit"]["examples"]}
    assert fit["other"]["examples_with_table_evidence_only"] == counts["fit"]["examples"]
    assert report["manifest"]["content_hash"] == manifest["content_hash"]
    assert report["leakage"]["test_against_train"]["company_also_on_the_seen_side"] == 0
    assert report["leakage"]["dev_against_fit"]["examples"] == counts["dev"]["examples"]


# --------------------------------------------------------------------------- checks


def test_a_clean_build_passes_every_check(frozen):
    checks = checks_for(frozen)
    assert list(checks) == [f"P{n}" for n in range(1, 11)]
    assert all(item["passed"] for item in checks.values()), checks
    assert checks["P10"]["detail"] == "locked: no unlock record exists"
    assert "none divided" in checks["P4"]["detail"]


def test_p1_missing_extra_and_reordered_ids(frozen):
    _, _, manifest, examples = frozen
    assert (
        "1 assigned ids are not in the data"
        in checks_for(frozen, examples=examples[1:])["P1"]["detail"]
    )
    extra = dataclasses.replace(examples[0], id="ZZZ/2015/page_1.pdf-1")
    result = checks_for(frozen, examples=[*examples, extra])["P1"]
    assert result["detail"] == "1 examples have no assignment"
    reordered = checks_for(frozen, examples=list(reversed(examples)))["P1"]
    assert reordered["detail"] == "the ids are in a different order from the data"


def test_p2_other_revision_or_checksums(frozen):
    _, data_cfg, _, _ = frozen
    other = dataclasses.replace(data_cfg, revision="f" * 40)
    assert "another dataset revision" in checks_for(frozen, data_cfg=other)["P2"]["detail"]
    files = (dataclasses.replace(data_cfg.files[0], sha256="0" * 64), *data_cfg.files[1:])
    other = dataclasses.replace(data_cfg, files=files)
    assert (
        checks_for(frozen, data_cfg=other)["P2"]["detail"]
        == "the manifest names other file checksums"
    )


def test_p3_ratio_outside_the_tolerance(frozen):
    cfg = dataclasses.replace(frozen[0], train_fraction_tolerance=0.001)
    result = checks_for(frozen, cfg=cfg)["P3"]
    assert not result["passed"]
    assert result["detail"].startswith("train share is 63.33%")
    assert checks_for(frozen)["P3"]["detail"].startswith("train 38 (63.33%), test 22 (36.67%)")


def test_p4_and_p7_a_divided_group(frozen):
    manifest = copy.deepcopy(frozen[2])
    victim = next(i for i, split in manifest["assignments"].items() if split == "fit")
    manifest["assignments"][victim] = "test"
    checks = checks_for(frozen, manifest=rehash(manifest))
    assert f"1 groups are in more than one split: {victim.split('/')[0]}" == checks["P4"]["detail"]
    assert "1 pages are in more than one split" in checks["P7"]["detail"]
    assert checks["P8"]["passed"] and not checks["P9"]["passed"]


def test_p4_group_table_that_disagrees_with_the_assignments(frozen):
    manifest = copy.deepcopy(frozen[2])
    manifest["groups"]["AAA"] = "test" if manifest["groups"]["AAA"] != "test" else "fit"
    result = checks_for(frozen, manifest=manifest)["P4"]
    assert result["detail"] == "the manifest's group table does not match its assignments"


def test_p5_dev_too_large_or_too_small(frozen):
    cfg, _, manifest, _ = frozen
    dev = manifest["counts"]["dev"]["examples"]
    large = checks_for(frozen, cfg=dataclasses.replace(cfg, dev_size=dev - 1))["P5"]
    assert large["detail"] == f"dev has {dev} examples, more than {dev - 1}"
    small = checks_for(
        frozen, cfg=dataclasses.replace(cfg, dev_size=dev + 5, dev_size_tolerance=2)
    )["P5"]
    assert small["detail"] == f"dev has {dev} examples, fewer than {dev + 3}"


def test_p6_quick_dev_wrong_size_or_outside_dev(frozen):
    manifest = copy.deepcopy(frozen[2])
    manifest["quick_dev"].pop()
    assert (
        checks_for(frozen, manifest=manifest)["P6"]["detail"] == "quick-dev has 2 ids, expected 3"
    )
    manifest = copy.deepcopy(frozen[2])
    manifest["quick_dev"][0] = next(i for i, s in manifest["assignments"].items() if s == "fit")
    assert "1 quick-dev ids are not in dev" in checks_for(frozen, manifest=manifest)["P6"]["detail"]


def test_p7_a_duplicate_pair_across_splits(frozen):
    cfg, data_cfg, manifest, examples = frozen
    fit_example = next(e for e in examples if manifest["assignments"][e.id] == "fit")
    test_example = next(e for e in examples if manifest["assignments"][e.id] == "test")
    twin = dataclasses.replace(
        test_example, filename=fit_example.filename, question=fit_example.question
    )
    changed = [twin if e.id == test_example.id else e for e in examples]
    detail = checks_for(frozen, examples=changed)["P7"]["detail"]
    assert "1 pages are in more than one split" in detail
    assert "1 page-and-question groups are in more than one split" in detail


def test_p8_wrong_hash(frozen):
    manifest = copy.deepcopy(frozen[2])
    manifest["content_hash"] = "0" * 64
    result = checks_for(frozen, manifest=manifest)["P8"]
    assert result["detail"] == "the content hash does not match the assignments"


def test_p9_settings_no_longer_produce_the_manifest(frozen):
    cfg, data_cfg, _, examples = frozen
    rebuilt = build_split(examples, dataclasses.replace(cfg, seed=cfg.seed + 1), data_cfg)
    result = checks_for(frozen, rebuilt=rebuilt)["P9"]
    assert not result["passed"]
    assert rebuilt["content_hash"][:12] in result["detail"]


def test_p10_reports_a_valid_unlock_and_fails_a_stray_file(frozen):
    cfg, _, manifest, _ = frozen
    cfg.unlock_file.write_text("{broken", encoding="utf-8")
    stray = checks_for(frozen)["P10"]
    assert not stray["passed"] and "is not valid JSON" in stray["detail"]
    cfg.unlock_file.write_text(
        json.dumps({"manifest_hash": manifest["content_hash"]}), encoding="utf-8"
    )
    valid = checks_for(frozen)["P10"]
    assert valid["passed"] and valid["detail"].startswith("unlocked: ")


# --------------------------------------------------------------------------- output


def test_output_is_deterministic_and_holds_no_example_text(frozen, tmp_path):
    cfg, data_cfg, manifest, examples = frozen
    first = build_report(manifest, manifest, examples, cfg, data_cfg)
    second = build_report(manifest, manifest, examples, cfg, data_cfg)
    assert render_json(first) == render_json(second)
    assert render_markdown(first) == render_markdown(second)
    for text in (render_json(first), render_markdown(first)):
        for word in SECRET_WORDS:
            assert word not in text.lower(), word
        assert str(tmp_path) not in text and tmp_path.name not in text
        assert text.isascii()
        assert "AAA" not in text  # counts only: not even the company symbols


def test_markdown_has_every_section_and_formats_shares(frozen):
    cfg, data_cfg, manifest, examples = frozen
    markdown = render_markdown(build_report(manifest, manifest, examples, cfg, data_cfg))
    assert markdown.startswith("# FinQA split report\n")
    assert f"- Content hash: `{manifest['content_hash']}`" in markdown
    assert "- Grouped by: company" in markdown
    assert "10 of 10 passed." in markdown
    assert "| Measure | fit | dev | quick-dev | train | test |" in markdown
    assert "| Share of all examples percent | 50.00 | 13.33 | 5.00 | 63.33 | 36.67 |" in markdown
    assert "| `subtract` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |" in markdown
    assert "| Measure | test against train | dev against fit |" in markdown
    assert "| Company also on the seen side | 0 | 0 |" in markdown


def test_write_report_and_stale_files(frozen):
    cfg, data_cfg, manifest, examples = frozen
    report = build_report(manifest, manifest, examples, cfg, data_cfg)
    markdown_path, json_path = report_paths(cfg.report_dir)
    assert (markdown_path.name, json_path.name) == (
        "finqa_split_report.md",
        "finqa_split_report.json",
    )
    assert stale_files(report, cfg.report_dir) == [markdown_path.name, json_path.name]
    assert write_report(report, cfg.report_dir) == (markdown_path, json_path)
    assert stale_files(report, cfg.report_dir) == []
    markdown_path.write_bytes(markdown_path.read_bytes().replace(b"\n", b"\r\n"))
    assert stale_files(report, cfg.report_dir) == []
    json_path.write_text("{}\n", encoding="utf-8")
    assert stale_files(report, cfg.report_dir) == [json_path.name]
