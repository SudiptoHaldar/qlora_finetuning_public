"""Tests for the schema report: counting, flags, checks, and what the output may contain.

All rows are invented. The one test that reads the real snapshot skips when it is absent.
"""

from __future__ import annotations

import json
import re

import pytest

from qlora_finetuning.data import load_data_config, load_raw_examples, read_columns, verify_snapshot
from qlora_finetuning.data.schema_report import (
    FLAGS,
    argument_kind,
    build_report,
    render_json,
    render_markdown,
    report_paths,
    run_checks,
    split_program,
    stale_files,
    write_report,
)
from tests.data.synthetic import CLEAN_EXPECTED, SECRET_WORDS, build_snapshot, clean_rows, row


def report_for(tmp_path, rows=None, expected=None):
    cfg = build_snapshot(tmp_path, rows, expected=expected)
    return build_report(load_raw_examples(cfg), cfg, read_columns(cfg)), cfg


def check(report, check_id):
    return next(item for item in report["checks"] if item["id"] == check_id)


def odd_rows():
    """One example of every oddity the report counts, spread over two official splits."""
    return {
        "train": [
            row("ACME/2015/page_10.pdf-1", question="same zebrafish question"),
            row("ACME/2015/page_10.pdf-2", question="same zebrafish question"),
            row("ACME/2015/page_11.pdf-1", answer="  "),
            row("ACME/2015/page_12.pdf-1", gold_inds={"text_-1": "marmalade", "table_1": "x"}),
            row("ACME/2015/page_13.pdf-1", gold_inds={"table_9": "x", "text_7": "y"}),
            row("ACME/2015/page_14.pdf-1", gold_inds="not json"),
            row("ACME/2015/page_15.pdf-1", program="divide(6.9%, const_100)"),
            row("ACME/2015/page_16.pdf-1", program="table_max(parsnip yield, 5)"),
            row(
                "ACME/2015/page_17.pdf-1",
                program="add(1, 2), add(#0, 3), add(#1, 4), add(#2, 5), add(#3, 6), add(#4, 7)",
            ),
        ],
        "test": [
            row("BOLT/2016/page_1.pdf-1", program="modulo(7, const_77)"),
            row("BOLT/2016/page_2.pdf-1", program="subtract(920; 95) then stop"),
            row("BOLT/2016/page_3.pdf-1", answer="yes", exe_ans="4.39e+07"),
            row("BOLT/2016/page_4.pdf-1", program="greater(5, 3)", answer="", exe_ans="yes"),
            row("BOLT/2016/page_5.pdf-1", program="greater(5, 3), add(#0, 1)", answer="2.43:1"),
            row("BOLT/2016/page_6.pdf-1", program="add(1, 2, 3)", exe_ans="n/a"),
            row("BOLT/2016/page_7.pdf-1", program="add(1, banana)", answer="$ 5 million"),
            row("oops", filename="BOLT/2016/page_8.pdf"),
            row("ACME/2015/page_10.pdf-1", filename="ACME/2015/page_99.pdf", question="other"),
        ],
    }


# --------------------------------------------------------------------------- programs


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("subtract(920, 95)", [("subtract", ["920", "95"])]),
        (
            "subtract(920, 95), divide(#0, const_100)",
            [("subtract", ["920", "95"]), ("divide", ["#0", "const_100"])],
        ),
        ("table_max(expected volatility, none)", [("table_max", ["expected volatility", "none"])]),
        ("  add( 1 ,2 )  ", [("add", ["1", "2"])]),
        ("add(1, 2, 3)", [("add", ["1", "2", "3"])]),
    ],
)
def test_split_program(text, expected):
    assert split_program(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "add(1, 2) add(3, 4)",
        "add(1, 2),",
        "add(subtract(1, 2), 3)",
        "add 1 2",
        "add(1, 2",
        None,
    ],
)
def test_split_program_rejects_other_forms(text):
    assert split_program(text) is None


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("920", "integer"),
        ("-121", "integer"),
        ("4474.2", "decimal"),
        (".5", "decimal"),
        ("#0", "reference"),
        ("const_100", "constant"),
        ("const_m1", "constant"),
        ("6.9%", "percent"),
        ("12%", "percent"),
        ("none", "none"),
        ("4.39e+07", "other"),
        ("net sales", "other"),
        ("", "other"),
    ],
)
def test_argument_kind(text, kind):
    assert argument_kind(text) == kind


# --------------------------------------------------------------------------- clean data


def test_clean_rows_pass_every_check(tmp_path):
    report, _ = report_for(tmp_path)
    assert [item["id"] for item in report["checks"]] == [f"S{n}" for n in range(1, 11)]
    assert all(item["passed"] for item in report["checks"]), report["checks"]
    assert all(ids == [] or name == "longest_programs" for name, ids in report["flagged"].items())
    assert list(report["flagged"]) == list(FLAGS)


def test_counts_on_clean_rows(tmp_path):
    report, cfg = report_for(tmp_path)
    total = report["total"]
    assert total["examples"] == 5
    assert report["source"]["revision"] == cfg.revision
    assert total["identity"]["companies"] == 3
    assert total["identity"]["reports"] == 4
    assert total["identity"]["pages"] == 4
    assert total["identity"]["examples_per_page"] == {"min": 1, "median": 1, "max": 2}
    assert (total["identity"]["first_year"], total["identity"]["last_year"]) == (2015, 2017)
    assert total["columns"]["explanation"] == {"null": 0, "empty": 5}
    assert total["columns"]["ann_text_rows"]["empty"] == 4
    assert total["tables"]["tables_with_an_empty_cell"] == 5
    assert total["tables"]["rows"] == {"min": 3, "median": 3, "max": 3}

    programs = total["programs"]
    assert programs["steps_in_total"] == 7
    assert programs["operations"] == {
        "subtract": 2, "divide": 2, "add": 1, "greater": 1, "table_sum": 1,
    }  # fmt: skip
    assert list(programs["operations"])[:2] == ["divide", "subtract"]  # count, then name
    assert programs["steps_per_program"] == {"1": 3, "2": 2}
    assert programs["arguments_per_step"] == {"2": 7}
    assert programs["argument_kinds"] == {
        "integer": 8, "reference": 2, "constant": 1, "decimal": 1, "none": 1, "row_name": 1,
    }  # fmt: skip
    assert programs["constants"] == {"const_100": 1}
    assert programs["table_row_names_found_in_the_table"] == 1
    assert programs["table_second_argument"] == {"none": 1}
    assert programs["examples_with_greater"] == 1

    evidence = total["evidence"]
    assert evidence["entries_per_example"] == {"1": 4, "2": 1}
    assert (evidence["table_entries"], evidence["text_entries"]) == (5, 1)
    assert evidence["examples_with_table_evidence_only"] == 4
    assert evidence["examples_with_both"] == 1
    assert evidence["examples_where_ann_rows_disagree_with_gold_inds"] == 0

    answers = total["answers"]
    assert (answers["exe_ans_numeric"], answers["exe_ans_yes"]) == (4, 1)
    assert answers["answer_contains_percent_sign"] == 1
    assert answers["answer_is_a_plain_number"] == 3
    assert answers["answer_is_yes_or_no"] == 1


def test_per_split_counts_add_up_to_the_total(tmp_path):
    report, _ = report_for(tmp_path, odd_rows(), expected={})
    total, splits = report["total"], report["by_split"].values()
    assert total["examples"] == sum(split["examples"] for split in splits)
    for section, key in (
        ("programs", "steps_in_total"),
        ("answers", "answer_empty"),
        ("evidence", "malformed_keys"),
        ("tables", "tables_with_an_empty_cell"),
    ):
        assert total[section][key] == sum(split[section][key] for split in splits)
    for name, count in total["programs"]["operations"].items():
        assert count == sum(s["programs"]["operations"].get(name, 0) for s in splits)


def test_groups_spanning_official_splits(tmp_path):
    report, _ = report_for(tmp_path)
    spanning = report["spanning_official_splits"]
    assert spanning["company"] == {"groups": 3, "in_more_than_one_official_split": 1}
    assert spanning["report"] == {"groups": 4, "in_more_than_one_official_split": 0}
    assert spanning["page"] == {"groups": 4, "in_more_than_one_official_split": 0}


# --------------------------------------------------------------------------- oddities


def test_every_oddity_is_counted_and_flagged(tmp_path):
    report, _ = report_for(tmp_path, odd_rows(), expected={})
    total, flagged = report["total"], report["flagged"]

    assert total["duplicates"]["groups_with_the_same_page_and_question"] == 1
    assert total["duplicates"]["examples_in_those_groups"] == 2
    assert total["duplicates"]["groups_also_sharing_program_and_result"] == 1
    assert flagged["duplicate_page_and_question"] == [
        "ACME/2015/page_10.pdf-1",
        "ACME/2015/page_10.pdf-2",
    ]
    assert flagged["id_repeated"] == ["ACME/2015/page_10.pdf-1"] * 2
    # "oops" matches no pattern; the last row's id does not extend its filename.
    assert flagged["id_pattern_mismatch"] == ["oops", "ACME/2015/page_10.pdf-1"]
    assert flagged["answer_empty"] == ["ACME/2015/page_11.pdf-1", "BOLT/2016/page_4.pdf-1"]
    assert flagged["answer_other_form"] == ["BOLT/2016/page_5.pdf-1"]

    assert flagged["evidence_malformed_key"] == ["ACME/2015/page_12.pdf-1"]
    assert flagged["evidence_key_out_of_range"] == ["ACME/2015/page_13.pdf-1"]
    assert flagged["evidence_unparsed"] == ["ACME/2015/page_14.pdf-1"]
    assert total["evidence"]["keys_out_of_range"] == 2
    assert total["evidence"]["gold_inds_not_a_json_object"] == 1
    # The malformed text key does not count as text evidence.
    assert total["evidence"]["malformed_keys"] == 1

    assert flagged["percent_literal_argument"] == ["ACME/2015/page_15.pdf-1"]
    assert flagged["row_name_not_in_table"] == ["ACME/2015/page_16.pdf-1"]
    assert flagged["table_second_argument_not_none"] == ["ACME/2015/page_16.pdf-1"]
    assert flagged["longest_programs"] == ["ACME/2015/page_17.pdf-1"]
    assert flagged["operation_unknown"] == ["BOLT/2016/page_1.pdf-1"]
    assert flagged["constant_unknown"] == ["BOLT/2016/page_1.pdf-1"]
    assert flagged["program_unparsed"] == ["BOLT/2016/page_2.pdf-1"]
    assert flagged["argument_count_unexpected"] == ["BOLT/2016/page_6.pdf-1"]
    assert flagged["argument_other_kind"] == ["BOLT/2016/page_7.pdf-1"]

    assert flagged["result_scientific_notation"] == ["BOLT/2016/page_3.pdf-1"]
    assert flagged["yes_no_answer_without_greater"] == ["BOLT/2016/page_3.pdf-1"]
    assert flagged["greater_answer_disagrees"] == [
        "BOLT/2016/page_4.pdf-1",
        "BOLT/2016/page_5.pdf-1",
    ]
    assert flagged["greater_not_last_step"] == ["BOLT/2016/page_5.pdf-1"]
    assert flagged["result_not_numeric_or_yes_no"] == ["BOLT/2016/page_6.pdf-1"]
    assert total["answers"]["answer_contains_dollar_sign"] == 1
    assert total["programs"]["table_second_argument"] == {"integer": 1}


def test_oddities_fail_their_checks(tmp_path):
    report, _ = report_for(tmp_path, odd_rows(), expected={})
    failed = {item["id"]: item["detail"] for item in report["checks"] if not item["passed"]}
    assert "1 repeated ids" in failed["S4"]
    assert "2 ids do not match" in failed["S5"]
    assert "1 programs do not split" in failed["S6"]
    assert "1 steps do not have 2 arguments" in failed["S6"]
    assert "unexpected operations: modulo" in failed["S7"]
    assert "unexpected constants: const_77" in failed["S8"]
    assert "unexpected argument kinds: other" in failed["S9"]
    assert check(report, "S1")["passed"] and check(report, "S2")["passed"]
    assert check(report, "S3")["passed"]


def test_an_expected_item_that_never_occurs_fails_the_check(tmp_path):
    expected = {**CLEAN_EXPECTED, "operations": [*CLEAN_EXPECTED["operations"], "exp"]}
    report, _ = report_for(tmp_path, expected=expected)
    assert check(report, "S7")["detail"] == "expected operations not seen: exp"


def test_program_longer_than_expected_fails(tmp_path):
    report, _ = report_for(tmp_path, expected={**CLEAN_EXPECTED, "max_program_steps": 1})
    assert check(report, "S10")["detail"] == "longest program has 2 steps, expected at most 1"


def test_row_count_check_uses_the_configured_rows(tmp_path):
    report, _ = report_for(tmp_path)
    report["source"]["files"][0]["rows"] = 99
    result = run_checks(report, CLEAN_EXPECTED | {"columns": {}, "arguments_per_step": 2})
    assert result[0]["detail"] == "train has 3 rows, expected 99"


def test_column_check_names_what_changed(tmp_path):
    cfg = build_snapshot(tmp_path)
    columns = read_columns(cfg)
    columns["test"] = {k: v for k, v in columns["test"].items() if k != "explanation"}
    columns["test"]["surprise"] = "string"
    columns["validation"] = {**columns["validation"], "answer": "int64"}
    report = build_report(load_raw_examples(cfg), cfg, columns)
    detail = check(report, "S2")["detail"]
    assert "validation: retyped answer" in detail
    assert "test: missing explanation; extra surprise" in detail
    assert "train" not in detail


def test_null_values_fail_the_check(tmp_path):
    rows = clean_rows()
    rows["train"][0]["explanation"] = None
    rows["train"][1]["table"] = None
    report, _ = report_for(tmp_path, rows)
    assert check(report, "S3")["detail"] == "nulls in table: 1, explanation: 1"


# --------------------------------------------------------------------------- output


def test_same_input_gives_the_same_text(tmp_path):
    first, _ = report_for(tmp_path / "a", odd_rows(), expected={})
    second, _ = report_for(tmp_path / "b", odd_rows(), expected={})
    assert render_json(first) == render_json(second)
    assert render_markdown(first) == render_markdown(second)
    assert render_json(first).endswith("}\n")
    assert json.loads(render_json(first)) == json.loads(json.dumps(first))


def test_output_holds_no_example_text_and_no_local_path(tmp_path):
    report, _ = report_for(tmp_path, odd_rows(), expected={})
    for text in (render_json(report), render_markdown(report)):
        lowered = text.lower()
        for word in (*SECRET_WORDS, "banana", "million"):
            assert word not in lowered, word
        assert str(tmp_path) not in text
        assert tmp_path.name not in text
        assert text.isascii()


def test_markdown_lists_ids_only_for_small_classes(tmp_path):
    rows = {"train": [row(f"ACME/2015/page_{n}.pdf-1", answer="") for n in range(1, 26)]}
    report, cfg = report_for(tmp_path, rows, expected={})
    assert cfg.report_max_ids_listed == 20
    markdown = render_markdown(report)
    line = next(text for text in markdown.splitlines() if text.startswith("| `answer_empty`"))
    assert "| 25 | see the JSON file |" in line
    assert len(report["flagged"]["answer_empty"]) == 25
    assert "| `evidence_unparsed` | gold_inds is not a JSON object | 0 | - |" in markdown


def test_markdown_shows_sections_checks_and_plain_years(tmp_path):
    report, _ = report_for(tmp_path)
    markdown = render_markdown(report)
    assert markdown.startswith("# FinQA schema report\n")
    assert "10 of 10 passed." in markdown
    assert "| Measure | Total | train | validation | test |" in markdown
    assert "| First year | 2015 | 2015 | 2016 | 2017 |" in markdown
    assert "| Operations: `table_sum` | 1 | 1 | 0 | 0 |" in markdown
    assert "| `exe_ans` numeric | 4 | 3 | 1 | 0 |" in markdown
    assert "| `table` | `list[list[string]]` | 0 | 0 |" in markdown
    assert "| Company | 3 | 1 |" in markdown


def test_write_report_and_stale_files(tmp_path):
    report, cfg = report_for(tmp_path)
    markdown_path, json_path = report_paths(cfg.report_dir)
    assert stale_files(report, cfg.report_dir) == [markdown_path.name, json_path.name]

    assert write_report(report, cfg.report_dir) == (markdown_path, json_path)
    assert b"\r\n" not in markdown_path.read_bytes()
    assert stale_files(report, cfg.report_dir) == []

    # A checkout that converts line endings still counts as current.
    markdown_path.write_bytes(markdown_path.read_bytes().replace(b"\n", b"\r\n"))
    assert stale_files(report, cfg.report_dir) == []

    json_path.write_text("{}\n", encoding="utf-8")
    assert stale_files(report, cfg.report_dir) == [json_path.name]


# --------------------------------------------------------------------------- real snapshot


def test_committed_report_matches_the_real_snapshot():
    """Skips where the snapshot has not been fetched (CI); the files are git-ignored."""
    cfg = load_data_config()
    if not all(item.ok for item in verify_snapshot(cfg)):
        pytest.skip("no FinQA snapshot here; fetch it with: python -m scripts.finqa_fetch")
    report = build_report(load_raw_examples(cfg), cfg, read_columns(cfg))
    assert all(item["passed"] for item in report["checks"])
    assert report["total"]["examples"] == 8281
    assert stale_files(report, cfg.report_dir) == []


def test_committed_report_exists_and_holds_only_counts_and_ids():
    cfg = load_data_config()
    markdown_path, json_path = report_paths(cfg.report_dir)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["source"]["revision"] == cfg.revision
    assert report["total"]["examples"] == sum(item.rows for item in cfg.files)
    assert all(item["passed"] for item in report["checks"])
    assert set(report) == {
        "schema", "generated_by", "source", "column_types", "total", "by_split",
        "spanning_official_splits", "flagged", "max_ids_listed", "checks",
    }  # fmt: skip
    # Flagged entries are ids and nothing else; no question (they end in "?") is in the text.
    id_pattern = re.compile(cfg.expected["id_pattern"])
    assert all(id_pattern.fullmatch(item) for ids in report["flagged"].values() for item in ids)
    assert "?" not in markdown_path.read_text(encoding="utf-8")
    assert "?" not in json_path.read_text(encoding="utf-8")
