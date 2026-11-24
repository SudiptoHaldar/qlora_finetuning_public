"""Tests for the serialization report: lengths, counts, digests, and checks C1 to C10."""

from __future__ import annotations

import dataclasses
import json

import pytest

from qlora_finetuning.data import (
    load_data_config,
    load_prompt,
    load_split,
    load_split_config,
    split_ids,
    verify_snapshot,
)
from qlora_finetuning.data.serialization import build_record
from qlora_finetuning.data.serialization_report import (
    build_report,
    length_stats,
    read_committed_digests,
    render_json,
    render_markdown,
    report_paths,
    run_checks,
    split_stats,
    stale_files,
    write_report,
)
from tests.data.synthetic import GOLDEN_ROWS, SECRET_WORDS, row, to_examples


@pytest.fixture(scope="module")
def prompt():
    return load_prompt()


@pytest.fixture
def examples():
    return to_examples(list(GOLDEN_ROWS.values()))


def records_for(examples, prompt):
    return [build_record(item, prompt) for item in examples]


def checks(examples, records, prompt, previous=None, allowed=None, test_ids=()):
    allowed = {item.id for item in examples} if allowed is None else allowed
    results = run_checks({"fit": (examples, records)}, prompt, previous, allowed, test_ids)
    return {item["id"]: item for item in results}


def replaced(records, position, **changes):
    changed = list(records)
    changed[position] = dataclasses.replace(records[position], **changes)
    return changed


# --------------------------------------------------------------------------- statistics


def test_length_stats_use_the_nearest_rank():
    assert length_stats(list(range(1, 101))) == {
        "min": 1, "median": 50.5, "p90": 90, "p99": 99, "max": 100,
    }  # fmt: skip
    assert length_stats([7]) == {"min": 7, "median": 7, "p90": 7, "p99": 7, "max": 7}
    assert length_stats([3, 1, 2])["median"] == 2
    assert length_stats([]) == {"min": None, "median": None, "p90": None, "p99": None, "max": None}


def test_split_statistics(examples, prompt):
    examples = [
        *examples,
        to_examples([row("DOVE/2015/page_1.pdf-1", gold_inds={"text_-1": "x"})])[0],
    ]
    records = records_for(examples, prompt)
    stats = split_stats(examples, records)
    assert stats["records"] == 5
    assert stats["system_characters"] == len(prompt.system)
    assert stats["user_characters"]["max"] == max(len(record.user) for record in records)
    assert stats["target_characters"]["min"] == min(len(record.target) for record in records)
    assert stats["records_without_text_evidence"] == 3
    assert stats["records_by_number_of_passages"] == {"0": 3, "1": 1, "2": 1, "3_or_more": 0}
    assert (stats["most_table_rows"], stats["most_table_columns"]) == (3, 3)
    assert stats["answers_written_out_from_scientific_notation"] == 1
    assert stats["yes_or_no_answers"] == 1
    assert stats["skipped_evidence_keys"] == [
        {"example_id": "DOVE/2015/page_1.pdf-1", "key": "text_-1"}
    ]
    assert len(stats["digest"]) == 64


# --------------------------------------------------------------------------- checks


def test_sound_records_pass_every_check(examples, prompt):
    results = checks(examples, records_for(examples, prompt), prompt)
    assert list(results) == [f"C{n}" for n in range(1, 11)]
    assert all(item["passed"] for item in results.values()), results
    assert results["C1"]["detail"] == "4 targets"
    assert results["C9"]["detail"].startswith("fit ")


def failing(examples, prompt, position=0, **changes):
    records = replaced(records_for(examples, prompt), position, **changes)
    results = checks(examples, records, prompt)
    return {key for key, item in results.items() if not item["passed"]}, results


def test_c1_target_that_is_not_two_lines(examples, prompt):
    failed, results = failing(examples, prompt, target="PROGRAM: add(1, 2)")
    assert {"C1", "C8"} == failed
    assert results["C1"]["detail"] == "1 records are not two lines: ACME/2015/page_10.pdf-1"
    failed, _ = failing(examples, prompt, target="PROGRAM: add(1, 2)\nANSWER: 3\nextra")
    assert "C1" in failed


def test_c2_program_that_differs_from_the_dataset(examples, prompt):
    failed, _ = failing(examples, prompt, target="PROGRAM: subtract(920,95)\nANSWER: 8.68421")
    assert {"C2", "C8"} == failed


@pytest.mark.parametrize("answer", ["4.39e+07", "8.4%", "1,000", "Yes", ".5"])
def test_c3_answer_in_another_form(examples, prompt, answer):
    target = f"PROGRAM: subtract(920, 95), divide(#0, 95)\nANSWER: {answer}"
    failed, _ = failing(examples, prompt, target=target)
    assert {"C3", "C8"} == failed


def test_c4_table_with_a_missing_row_or_cell(examples, prompt):
    good = records_for(examples, prompt)[0].user
    failed, _ = failing(examples, prompt, user=good.replace("| total | 1000 | 900 |\n", ""))
    assert "C4" in failed
    failed, _ = failing(
        examples, prompt, user=good.replace("| total | 1000 | 900 |", "| total | 1000 |")
    )
    assert {"C4", "C8"} == failed
    failed, _ = failing(examples, prompt, user=good.replace("Table:", "Tabelle:"))
    assert "C4" in failed


def test_c5_evidence_missing_or_added(examples, prompt):
    good = records_for(examples, prompt)[0].user
    failed, _ = failing(
        examples, prompt, user=good.replace("- see the quokka note for details .\n", "")
    )
    assert {"C5", "C8"} == failed
    table_only = records_for(examples, prompt)[1].user
    failed, _ = failing(
        examples,
        prompt,
        position=1,
        user=table_only.replace("(none)", "- a passage that is not evidence ."),
    )
    assert {"C5", "C8"} == failed


def test_c6_question_missing(examples, prompt):
    good = records_for(examples, prompt)[0].user
    failed, _ = failing(examples, prompt, user=good + " please")
    assert {"C6", "C8"} == failed


def test_c7_another_system_text_or_prompt_id(examples, prompt):
    failed, _ = failing(examples, prompt, system="You are a pirate.")
    assert {"C7", "C8"} == failed
    failed, results = failing(examples, prompt, prompt_id="0" * 64)
    assert {"C7", "C8"} == failed
    assert "carry another system text or prompt id" in results["C7"]["detail"]


def test_c9_compares_with_the_committed_digests(examples, prompt):
    records = records_for(examples, prompt)
    report = build_report({"fit": (examples, records)}, prompt, None, {e.id for e in examples}, ())
    same = {"prompt_id": prompt.prompt_id, "fit": report["splits"]["fit"]["digest"]}
    assert checks(examples, records, prompt, previous=same)["C9"]["passed"]

    moved = checks(examples, records, prompt, previous={**same, "fit": "0" * 64})["C9"]
    assert not moved["passed"]
    assert "the fit records changed (digest 000000000000... is now " in moved["detail"]
    assert "review and commit the rewritten report" in moved["detail"]

    reworded = checks(examples, records, prompt, previous={**same, "prompt_id": "f" * 64})["C9"]
    assert "the prompt id changed from ffffffffffff... to " in reworded["detail"]


def test_c10_record_outside_fit_and_dev(examples, prompt):
    records = records_for(examples, prompt)
    inside = {item.id for item in examples}
    results = checks(examples, records, prompt, test_ids={examples[3].id})
    assert results["C10"]["detail"] == "1 records are not in fit or dev: CRUX/2017/page_8.pdf-1"
    results = checks(examples, records, prompt, allowed=inside - {examples[0].id})
    assert not results["C10"]["passed"]


def test_many_failures_are_summarized(prompt):
    examples = to_examples([row(f"ACME/2015/page_{n}.pdf-1") for n in range(1, 7)])
    records = [dataclasses.replace(r, target="broken") for r in records_for(examples, prompt)]
    detail = checks(examples, records, prompt)["C1"]["detail"]
    assert detail.startswith("6 records are not two lines: ACME/2015/page_1.pdf-1, ")
    assert detail.endswith(", ...")


# --------------------------------------------------------------------------- output


def report_for(examples, prompt, previous=None):
    records = records_for(examples, prompt)
    quick = (examples[:1], records[:1])
    by_split = {
        "fit": (examples, records),
        "dev": (examples[1:2], records[1:2]),
        "quick_dev": quick,
    }
    return build_report(by_split, prompt, previous, {e.id for e in examples}, ())


def test_quick_dev_is_reported_but_not_checked_twice(examples, prompt):
    report = report_for(examples, prompt)
    assert list(report["splits"]) == ["fit", "dev", "quick_dev"]
    assert report["checks"][0]["detail"] == "5 targets"  # fit 4 + dev 1, not quick-dev
    assert report["prompt"] == {
        "prompt_id": prompt.prompt_id,
        "version": 1,
        "system_characters": len(prompt.system),
    }


def test_output_is_deterministic_and_holds_no_example_text(examples, prompt, tmp_path):
    first, second = report_for(examples, prompt), report_for(examples, prompt)
    assert render_json(first) == render_json(second)
    assert render_markdown(first) == render_markdown(second)
    for text in (render_json(first), render_markdown(first)):
        for word in SECRET_WORDS:
            assert word not in text.lower(), word
        assert text.isascii()
        assert "?" not in text
        assert str(tmp_path) not in text


def test_markdown_sections(examples, prompt):
    examples = [
        *examples,
        to_examples([row("DOVE/2015/page_1.pdf-1", gold_inds={"text_-1": "x"})])[0],
    ]
    markdown = render_markdown(report_for(examples, prompt))
    assert markdown.startswith("# FinQA serialization report\n")
    assert f"- Prompt id: `{prompt.prompt_id}`" in markdown
    assert "10 of 10 passed." in markdown
    assert "| Measure | fit | dev | quick_dev |" in markdown
    assert "| Records | 5 | 1 | 1 |" in markdown
    assert "| Records by number of passages: `3_or_more` | 0 | 0 | 0 |" in markdown
    assert "| fit | `DOVE/2015/page_1.pdf-1` | `text_-1` |" in markdown
    assert markdown.count("| `") >= 3  # the digest rows
    assert "None." not in markdown
    assert "None." in render_markdown(report_for(examples[:4], prompt))


def test_write_read_back_and_stale(examples, prompt, tmp_path):
    report = report_for(examples, prompt)
    markdown_path, json_path = report_paths(tmp_path)
    assert (markdown_path.name, json_path.name) == (
        "finqa_serialization_report.md",
        "finqa_serialization_report.json",
    )
    assert read_committed_digests(tmp_path) is None
    assert stale_files(report, tmp_path) == [markdown_path.name, json_path.name]
    write_report(report, tmp_path)
    assert stale_files(report, tmp_path) == []
    digests = read_committed_digests(tmp_path)
    assert digests["prompt_id"] == prompt.prompt_id
    assert digests["fit"] == report["splits"]["fit"]["digest"]
    assert set(digests) == {"prompt_id", "fit", "dev", "quick_dev"}
    markdown_path.write_bytes(markdown_path.read_bytes().replace(b"\n", b"\r\n"))
    assert stale_files(report, tmp_path) == []
    json_path.write_text("{}", encoding="utf-8")
    assert read_committed_digests(tmp_path) is None
    assert stale_files(report, tmp_path) == [json_path.name]


# --------------------------------------------------------------------------- repository


def test_committed_report_names_the_current_prompt_and_passes(prompt):
    path = report_paths(prompt.report_dir)[1]
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["prompt"]["prompt_id"] == prompt.prompt_id
    assert all(item["passed"] for item in report["checks"])
    assert report["splits"]["fit"]["records"] == 4884
    assert report["splits"]["dev"]["records"] == 499
    assert report["splits"]["quick_dev"]["records"] == 200
    text = path.read_text(encoding="utf-8") + report_paths(prompt.report_dir)[0].read_text(
        encoding="utf-8"
    )
    assert "?" not in text  # no question found its way in


def test_committed_digests_match_the_real_records(prompt):
    """Skips where the snapshot has not been fetched (CI)."""
    if not all(item.ok for item in verify_snapshot(load_data_config())):
        pytest.skip("no FinQA snapshot here; fetch it with: python -m scripts.finqa_fetch")
    by_split = {}
    for name in ("fit", "dev", "quick_dev"):
        examples = load_split(name)
        by_split[name] = (examples, [build_record(item, prompt) for item in examples])
    cfg = load_split_config()
    allowed = set(split_ids("fit", cfg)) | set(split_ids("dev", cfg))
    report = build_report(by_split, prompt, None, allowed, set(split_ids("test", cfg)))
    assert all(item["passed"] for item in report["checks"])
    assert stale_files(report, prompt.report_dir) == []
