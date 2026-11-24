"""Tests for the gold self-check: scoring gold replies, flags, the report, checks K1 to K7.

Rows are invented. A hand-made manifest stands in for the split, so no file is needed except
where a test writes the report or the flags file under a temporary folder.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from qlora_finetuning.data import (
    gold_check,
    load_data_config,
    load_prompt,
    load_raw_examples,
    load_split_config,
    verify_snapshot,
)
from qlora_finetuning.data.gold_check import (
    REPORT_COMMAND,
    build_flags,
    build_report,
    check_example,
    flags_content_hash,
    load_gold_flags,
    read_flags,
    render_json,
    render_markdown,
    report_paths,
    stale_files,
    write_flags,
    write_report,
)
from qlora_finetuning.data.split import read_manifest
from qlora_finetuning.evaluation.metrics import (
    Allowance,
    EvaluationError,
    load_evaluation_config,
    summarize,
)
from tests.data.synthetic import SECRET_WORDS, row, to_examples

FIT_1 = "AAA/2015/page_1.pdf-1"
FIT_2 = "AAA/2015/page_1.pdf-2"
FIT_3 = "BBB/2016/page_2.pdf-1"
DEV_1 = "CCC/2015/page_3.pdf-1"
DEV_2 = "CCC/2015/page_3.pdf-2"
TEST_1 = "DDD/2017/page_4.pdf-1"
TEST_2 = "DDD/2017/page_4.pdf-2"
ASSIGNMENTS = {
    FIT_1: "fit",
    FIT_2: "fit",
    FIT_3: "fit",
    DEV_1: "dev",
    DEV_2: "dev",
    TEST_1: "test",
    TEST_2: "test",
}
MANIFEST_HASH = "c0ffee" * 10 + "beef"

# One broken gold of each reason a flag can have.
BROKEN = {
    "value_mismatch": {"exe_ans": "900"},
    "division_by_zero": {"program": "divide(920, 0)", "exe_ans": "5"},
    "row_not_found": {"program": "table_sum(quokka row, none)", "exe_ans": "5"},
    "wrong_arity": {"program": "add(1)", "exe_ans": "3"},
    "unserializable": {"exe_ans": "n/a"},
    "answer_unreadable": {"exe_ans": "1_000"},
}


def clean_rows() -> dict[str, dict]:
    return {
        FIT_1: row(FIT_1),
        FIT_2: row(FIT_2, program="subtract(920, 95), divide(#0, 95)", exe_ans="8.68421"),
        FIT_3: row(FIT_3, program="table_sum(xylophone sales, none)", exe_ans="1015"),
        DEV_1: row(DEV_1, program="greater(920, 95)", exe_ans="yes"),
        DEV_2: row(
            DEV_2, program="table_max(total, none), multiply(#0, const_1000000)", exe_ans="1e+09"
        ),
        TEST_1: row(TEST_1, program="divide(920, const_100)", exe_ans="9.2"),
        TEST_2: row(TEST_2),
    }


def examples_with(**changes_by_id):
    """The clean examples, with the rows named by id rebuilt from keyword changes."""
    rows = clean_rows()
    for example_id, changes in changes_by_id.items():
        rows[example_id] = row(example_id, **changes)
    return to_examples(list(rows.values()))


def manifest(assignments=None, quick_dev=(DEV_1,)):
    return {
        "assignments": dict(ASSIGNMENTS if assignments is None else assignments),
        "quick_dev": list(quick_dev),
        "content_hash": MANIFEST_HASH,
    }


@pytest.fixture(scope="module")
def prompt():
    return load_prompt()


@pytest.fixture
def config(tmp_path):
    return load_evaluation_config(root=tmp_path)


def report_for(examples, prompt, config, **more):
    return build_report(examples, more.pop("manifest", manifest()), prompt, config, **more)


def by_id(report):
    return {check["id"]: check for check in report["checks"]}


def one(example_id, prompt, config, **changes):
    return check_example(to_examples([row(example_id, **changes)])[0], prompt, config)


# --------------------------------------------------------------------------- one example


def test_a_sound_gold_reply_scores_true_on_everything(prompt, config):
    changes = {"program": "subtract(920, 95), divide(#0, 95)", "exe_ans": "8.68421"}
    check = one(FIT_2, prompt, config, **changes)
    assert not check.flagged and check.reason is None
    assert check.strict and check.canonical
    assert check.score.all_correct and check.score.example_id == FIT_2
    assert 0 < check.allowance_used < 0.01
    assert not check.yes_no and not check.duplicate_row


def test_a_yes_or_no_gold_reply(prompt, config):
    check = one(DEV_1, prompt, config, program="greater(920, 95)", exe_ans="yes")
    assert check.score.all_correct and check.yes_no
    assert check.allowance_used is None


def test_a_result_in_scientific_notation_is_written_out_and_read_back(prompt, config):
    check = one(DEV_2, prompt, config, program="multiply(920, const_1000000)", exe_ans="9.2e+08")
    assert check.score.all_correct and check.allowance_used == 0.0


@pytest.mark.parametrize("reason", sorted(BROKEN))
def test_a_broken_gold_is_flagged_with_its_reason(prompt, config, reason):
    check = one(TEST_1, prompt, config, **BROKEN[reason])
    assert check.flagged and check.reason == reason
    assert not check.score.execution_correct and not check.score.all_correct
    assert check.score.example_id == TEST_1


def test_a_value_mismatch_still_reports_what_was_right(prompt, config):
    check = one(TEST_1, prompt, config, **BROKEN["value_mismatch"])
    assert check.score.executed and check.score.answer_correct and check.score.program_match
    assert check.strict and check.canonical
    assert check.allowance_used > 1.0


def test_an_example_with_no_program_is_unserializable(prompt, config):
    check = one(TEST_1, prompt, config, program="")
    assert check.reason == "unserializable"
    assert check.score.error_code == "unserializable" and not check.score.executed


def test_the_execution_allowance_decides_a_flag(prompt, config):
    # 825.5 against a gold of 825 is within 0.5% and outside 0.001%.
    changes = {"program": "add(825, 0.5)", "exe_ans": "825"}
    assert not one(TEST_1, prompt, config, **changes).flagged
    tight = dataclasses.replace(config, execution=Allowance(absolute=0.00001, relative=0.00001))
    assert one(TEST_1, prompt, tight, **changes).reason == "value_mismatch"


def test_a_program_naming_a_repeated_row_is_counted(prompt, config):
    table = [["", "2015", "2014"], ["total", "1", "2"], ["total", "10", "20"]]
    check = one(FIT_1, prompt, config, table=table, program="table_sum(total, none)", exe_ans="30")
    assert check.duplicate_row and check.score.all_correct


# --------------------------------------------------------------------------- the report


def test_clean_rows_pass_every_check(prompt, config):
    report = report_for(examples_with(), prompt, config)
    assert [check["id"] for check in report["checks"]] == [f"K{n}" for n in range(1, 8)]
    assert all(check["passed"] for check in report["checks"]), report["checks"]
    assert report["flagged"] == []
    assert {name: block["examples"] for name, block in report["splits"].items()} == {
        "fit": 3,
        "dev": 2,
        "quick_dev": 1,
        "test": 2,
    }
    total = report["total"]
    assert total["examples"] == 7
    assert total["gold_programs_that_reproduce_the_gold_answer"] == 7
    assert total["flagged_examples"] == 0 and total["flagged_by_reason"] == {}
    for metric in ("format_validity", "execution_accuracy", "answer_accuracy", "program_match"):
        assert total[metric] == 7
    assert total["yes_or_no_examples"] == 1 and report["splits"]["dev"]["yes_or_no_examples"] == 1
    assert total["programs_naming_a_repeated_row"] == 0
    assert 0 < total["largest_share_of_the_allowance_used"] < 0.01
    assert report["settings"] == {
        "manifest_hash": MANIFEST_HASH,
        "prompt_id": prompt.prompt_id,
        "execution": {"absolute": 0.00001, "relative": 0.005},
        "answer": {"absolute": 0.00001, "relative": 0.005},
    }
    assert by_id(report)["K1"]["detail"] == "7 examples, each checked once"


@pytest.mark.parametrize("reason", sorted(BROKEN))
def test_a_flagged_test_example_is_reported_and_never_dropped(prompt, config, reason):
    report = report_for(examples_with(**{TEST_1: BROKEN[reason]}), prompt, config)
    assert report["flagged"] == [{"example_id": TEST_1, "split": "test", "reason": reason}]
    test = report["splits"]["test"]
    assert test["examples"] == 2  # still counted
    assert test["gold_programs_that_reproduce_the_gold_answer"] == 1
    assert test["flagged_examples"] == 1 and test["flagged_by_reason"] == {reason: 1}
    assert test["execution_accuracy"] == 1
    assert report["total"]["examples"] == 7 and report["total"]["execution_accuracy"] == 6
    checks = by_id(report)
    # A flag in the test split is reported, not required to be absent (D23).
    assert all(checks[name]["passed"] for name in ("K1", "K3", "K4", "K5", "K6", "K7"))
    assert checks["K4"]["detail"].startswith("6 unflagged examples")


def test_a_gap_beyond_the_allowance_is_left_out_of_the_largest_share(prompt, config):
    report = report_for(examples_with(**{TEST_1: BROKEN["value_mismatch"]}), prompt, config)
    assert report["splits"]["test"]["largest_share_of_the_allowance_used"] == 0.0


def test_k1_fails_on_an_example_without_a_split_a_missing_one_and_a_repeat(prompt, config):
    examples = examples_with()
    without = {key: value for key, value in ASSIGNMENTS.items() if key != TEST_2}
    check = by_id(report_for(examples, prompt, config, manifest=manifest(without)))["K1"]
    assert not check["passed"]
    assert f"1 examples have no split in the manifest: {TEST_2}" in check["detail"]

    check = by_id(report_for(examples[:-1], prompt, config))["K1"]
    assert not check["passed"]
    assert f"1 examples of the manifest were not checked: {TEST_2}" in check["detail"]

    check = by_id(report_for(examples + examples[:1], prompt, config))["K1"]
    assert not check["passed"]
    assert f"1 examples were checked more than once: {FIT_1}" in check["detail"]


def test_k2_fails_on_a_program_that_is_not_in_canonical_form(prompt, config):
    examples = examples_with(**{FIT_1: {"program": "subtract( 920,95 )"}})
    report = report_for(examples, prompt, config)
    checks = by_id(report)
    assert not checks["K2"]["passed"] and FIT_1 in checks["K2"]["detail"]
    assert report["flagged"] == []  # it executes and matches, so it is not a flag (D19)
    assert checks["K3"]["passed"] and checks["K4"]["passed"]


def test_k3_fails_when_a_fit_or_dev_example_is_flagged(prompt, config):
    for example_id in (FIT_2, DEV_2):
        report = report_for(
            examples_with(**{example_id: BROKEN["value_mismatch"]}), prompt, config
        )
        check = by_id(report)["K3"]
        assert not check["passed"]
        assert check["detail"] == f"1 examples of fit or dev are flagged: {example_id}"
        assert report["total"]["examples"] == 7


def test_k4_fails_when_an_unflagged_example_is_not_right_on_everything(prompt, config):
    # A second line after the program breaks the two-line format; the program still runs.
    examples = examples_with(**{FIT_1: {"program": "subtract(920, 95)\nplus a remark"}})
    report = report_for(examples, prompt, config)
    checks = by_id(report)
    assert report["flagged"] == []
    assert not checks["K4"]["passed"] and FIT_1 in checks["K4"]["detail"]
    assert not checks["K2"]["passed"]
    assert report["splits"]["fit"]["format_validity"] == 2


def test_k5_compares_with_the_flags_file_as_it_was(prompt, config):
    examples = examples_with(**{TEST_1: BROKEN["value_mismatch"]})
    report = report_for(examples, prompt, config)
    assert by_id(report)["K5"]["passed"]  # no flags file before
    flags = build_flags(report)

    same = report_for(examples, prompt, config, previous_flags=flags)
    assert by_id(same)["K5"]["passed"]
    assert same == report  # so a second run rewrites nothing

    clean = report_for(examples_with(), prompt, config, previous_flags=flags)
    check = by_id(clean)["K5"]
    assert not check["passed"]
    assert "the flagged examples changed (1 before, 0 now)" in check["detail"]
    assert "review and commit the rewritten files" in check["detail"]

    tight = dataclasses.replace(config, execution=Allowance(absolute=0.00001, relative=0.00001))
    check = by_id(report_for(examples, prompt, tight, previous_flags=flags))["K5"]
    assert not check["passed"]
    assert "the settings changed since the flags file was written" in check["detail"]


def test_k6_reports_what_the_caller_found_about_the_data(prompt, config):
    problem = "the split manifest was built from another dataset revision"
    check = by_id(report_for(examples_with(), prompt, config, data_problem=problem))["K6"]
    assert not check["passed"] and check["detail"] == problem
    passing = by_id(report_for(examples_with(), prompt, config))["K6"]
    assert passing["passed"] and MANIFEST_HASH[:12] in passing["detail"]


def test_k7_fails_when_a_second_pass_differs(prompt, config, monkeypatch):
    real = gold_check.check_example
    calls = []

    def unsteady(example, *args, **kwargs):
        calls.append(example.id)
        check = real(example, *args, **kwargs)
        return dataclasses.replace(check, strict=len(calls) <= 7)

    monkeypatch.setattr(gold_check, "check_example", unsteady)
    check = by_id(report_for(examples_with(), prompt, config))["K7"]
    assert not check["passed"] and len(calls) == 14


def test_output_is_deterministic_and_holds_no_example_text(prompt, config):
    examples = examples_with(**{TEST_1: BROKEN["row_not_found"], FIT_3: BROKEN["value_mismatch"]})
    report = report_for(examples, prompt, config)
    assert report_for(examples, prompt, config) == report
    texts = [render_markdown(report), render_json(report), render_json(build_flags(report))]
    for text in texts:
        assert text.isascii() and text.endswith("\n")
        for word in SECRET_WORDS + ("subtract", "table_sum", "8.68421", "1015", "what was"):
            assert word not in text
    assert json.loads(texts[1]) == report


def test_markdown_sections(prompt, config):
    clean = render_markdown(report_for(examples_with(), prompt, config))
    assert clean.startswith("# FinQA gold self-check report\n")
    assert f"Generated by `{REPORT_COMMAND}`" in clean
    assert "7 of 7 passed." in clean
    assert "- Execution accuracy: absolute 1e-05, relative 0.005" in clean
    assert "| Measure | fit | dev | quick_dev | test | all |" in clean
    assert "| Gold programs that reproduce the gold answer | 3 | 2 | 1 | 2 | 7 |" in clean
    assert "| Flagged examples | 0 | 0 | 0 | 0 | 0 |" in clean
    assert clean.rstrip().endswith("## Flagged examples\n\nNone.")

    report = report_for(examples_with(**{TEST_1: BROKEN["division_by_zero"]}), prompt, config)
    flagged = render_markdown(report)
    assert f"| test | `{TEST_1}` | `division_by_zero` |" in flagged
    assert "| Flagged: `division_by_zero` | 0 | 0 | 0 | 1 | 1 |" in flagged
    assert "| K3 | Fit and dev reproduce | PASS |" in flagged

    report["flagged"] = report["flagged"] * 60
    assert "| ... | 10 more in the JSON file | |" in render_markdown(report)


# --------------------------------------------------------------------------- flags file


def test_flags_hold_ids_reasons_and_the_settings(prompt, config):
    report = report_for(examples_with(**{TEST_1: BROKEN["wrong_arity"]}), prompt, config)
    flags = build_flags(report)
    assert flags["flagged"] == [{"example_id": TEST_1, "split": "test", "reason": "wrong_arity"}]
    assert flags["counts"] == {"fit": 0, "dev": 0, "test": 1, "total": 1}
    assert flags["settings"] == report["settings"]
    assert flags["content_hash"] == flags_content_hash(flags["flagged"])
    assert flags["generated_by"] == REPORT_COMMAND

    empty = build_flags(report_for(examples_with(), prompt, config))
    assert empty["flagged"] == [] and empty["counts"]["total"] == 0
    assert empty["content_hash"] != flags["content_hash"]


def test_flags_round_trip_and_feed_the_totals(prompt, config):
    examples = examples_with(**{TEST_1: BROKEN["value_mismatch"]})
    flags = build_flags(report_for(examples, prompt, config))
    assert read_flags(config) is None
    path = write_flags(flags, config)
    assert path == config.flags_file and b"\r" not in path.read_bytes()
    assert read_flags(config) == flags
    assert load_gold_flags(config) == frozenset({TEST_1})

    scores = [check_example(example, prompt, config).score for example in examples]
    totals = summarize(scores, flagged=load_gold_flags(config))
    assert totals["all"]["examples"] == 7  # the headline keeps the flagged example
    assert totals["all"]["execution_accuracy"]["count"] == 6
    assert totals["unflagged"]["examples"] == 6
    assert totals["unflagged"]["execution_accuracy"] == {"count": 6, "rate": 1.0}
    assert totals["flagged_examples"] == 1


def test_no_flags_file_is_an_error_that_names_the_command(config):
    with pytest.raises(EvaluationError, match="python -m scripts.finqa_gold_check"):
        load_gold_flags(config)


def test_an_empty_flags_file_gives_no_ids(prompt, config):
    write_flags(build_flags(report_for(examples_with(), prompt, config)), config)
    assert load_gold_flags(config) == frozenset()


@pytest.mark.parametrize("damage", ["edited", "not json", "no hash", "not a list"])
def test_a_damaged_or_edited_flags_file_is_refused(prompt, config, damage):
    examples = examples_with(**{TEST_1: BROKEN["value_mismatch"]})
    flags = build_flags(report_for(examples, prompt, config))
    if damage == "edited":
        flags["flagged"] = []  # someone removed a flag by hand
    elif damage == "no hash":
        del flags["content_hash"]
    elif damage == "not a list":
        flags["flagged"] = "none"
    write_flags(flags, config)
    if damage == "not json":
        config.flags_file.write_text("{", encoding="utf-8")
    with pytest.raises(EvaluationError, match="damaged or was edited by hand"):
        load_gold_flags(config)


def test_write_and_stale(prompt, config):
    report = report_for(examples_with(**{TEST_1: BROKEN["value_mismatch"]}), prompt, config)
    markdown_path, json_path = report_paths(config.report_dir)
    names = [markdown_path.name, json_path.name, config.flags_file.name]
    assert names == [
        "finqa_gold_check_report.md",
        "finqa_gold_check_report.json",
        "finqa_gold_flags.json",
    ]
    assert stale_files(report, config) == names

    assert write_report(report, config) == (markdown_path, json_path)
    assert stale_files(report, config) == names[2:]
    write_flags(build_flags(report), config)
    assert stale_files(report, config) == []
    assert b"\r" not in markdown_path.read_bytes()
    assert json.loads(json_path.read_text(encoding="utf-8")) == report

    markdown_path.write_text("edited\n", encoding="utf-8")
    assert stale_files(report, config) == names[:1]
    assert stale_files(report, config, names=names[1:]) == []


# --------------------------------------------------------------------------- the real data


def test_committed_report_and_flags_match_a_fresh_run():
    """Counts and ids only: nothing from an example is written or shown here."""
    data_cfg = load_data_config()
    if not all(item.ok for item in verify_snapshot(data_cfg)):
        pytest.skip("no FinQA snapshot here; fetch it with: python -m scripts.finqa_fetch")
    config = load_evaluation_config()
    report = build_report(
        load_raw_examples(data_cfg),
        read_manifest(load_split_config()),
        load_prompt(),
        config,
        previous_flags=read_flags(config),
    )
    failed = [check["id"] for check in report["checks"] if not check["passed"]]
    assert failed == []
    assert stale_files(report, config) == []
    assert report["total"]["examples"] == 8281
    for name, size in (("fit", 4884), ("dev", 499), ("quick_dev", 200), ("test", 2898)):
        assert report["splits"][name]["examples"] == size
    for name in ("fit", "dev"):
        assert report["splits"][name]["flagged_examples"] == 0
        assert report["splits"][name]["gold_programs_that_reproduce_the_gold_answer"] == (
            report["splits"][name]["examples"]
        )
    assert load_gold_flags(config) == frozenset(item["example_id"] for item in report["flagged"])
