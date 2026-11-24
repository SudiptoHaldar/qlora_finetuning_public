"""Tests for canonical serialization. Golden records and every other row are invented."""

from __future__ import annotations

import dataclasses

import pytest
import yaml

from qlora_finetuning.data import (
    TestLockedError,
    load_data_config,
    load_prompt,
    serialize_split,
    split_ids,
    verify_snapshot,
)
from qlora_finetuning.data.serialization import (
    SerializationError,
    build_record,
    format_answer,
    format_table,
    format_target,
    format_user,
    records_digest,
    text_evidence,
)
from qlora_finetuning.settings import CONFIG_DIR, REPO_ROOT, ConfigError
from tests.data.synthetic import (
    GOLDEN_DIR,
    GOLDEN_ROWS,
    build_frozen_split,
    golden_text,
    row,
    to_examples,
)

# The id of the wording in config/prompt.yaml. Changing the wording changes this value: update
# it here on purpose, record the new id in the req_001 roadmap, and rerun every arm that was
# already run on the old wording.
PROMPT_ID = "d63742a62b76d526ecf3d92df4cbad832846d4da7aa1ef13d15401ac28391641"


@pytest.fixture(scope="module")
def prompt():
    return load_prompt()


def example(**overrides):
    return to_examples([row(**overrides)])[0]


def write_prompt(tmp_path, change=lambda raw: None, comment="", sort_keys=False):
    raw = yaml.safe_load((CONFIG_DIR / "prompt.yaml").read_text(encoding="utf-8"))
    change(raw)
    tmp_path.mkdir(parents=True, exist_ok=True)
    text = comment + yaml.safe_dump(raw, sort_keys=sort_keys)
    (tmp_path / "prompt.yaml").write_text(text, encoding="utf-8")
    return load_prompt(root=tmp_path, config_dir=tmp_path)


# --------------------------------------------------------------------------- prompt definition


def test_real_prompt_loads_and_its_id_is_pinned(prompt):
    assert prompt.prompt_id == PROMPT_ID
    assert prompt.version == 1
    assert prompt.system.startswith("You are a financial calculation assistant.")
    assert prompt.system.endswith(
        "ANSWER: <the result of the last step: a plain number, or yes or no>"
    )
    assert (prompt.context_label, prompt.table_label, prompt.question_label) == (
        "Context:",
        "Table:",
        "Question:",
    )
    assert (prompt.list_marker, prompt.no_context) == ("- ", "(none)")
    assert prompt.report_dir == REPO_ROOT / "reports" / "data"
    assert prompt.dump_dir == REPO_ROOT / "artifacts" / "serialized"


def test_system_text_names_the_whole_output_language(prompt):
    expected = yaml.safe_load((CONFIG_DIR / "data.yaml").read_text(encoding="utf-8"))
    for name in expected["finqa"]["expected"]["operations"]:
        assert name in prompt.system, name
    for name in expected["finqa"]["expected"]["constants"]:
        assert name in prompt.system, name
    for phrase in ("#N", "none", "12.5%", "PROGRAM:", "ANSWER:", "yes", "exactly two lines"):
        assert phrase in prompt.system, phrase
    assert prompt.system.isascii()


def test_prompt_id_ignores_comments_key_order_and_folders(tmp_path):
    base = write_prompt(tmp_path / "a")
    assert base.prompt_id == PROMPT_ID
    commented = write_prompt(tmp_path / "b", comment="# a note\n", sort_keys=True)
    assert commented.prompt_id == PROMPT_ID
    moved = write_prompt(tmp_path / "c", lambda raw: raw.update(report_dir="elsewhere"))
    assert moved.prompt_id == PROMPT_ID
    assert moved.report_dir == tmp_path / "c" / "elsewhere"


@pytest.mark.parametrize(
    "change",
    [
        lambda raw: raw.update(system=raw["system"].replace("financial", "careful")),
        lambda raw: raw.update(system=raw["system"] + "\nThink first."),
        lambda raw: raw["user"].update(question_label="Q:"),
        lambda raw: raw["user"].update(no_context="(no text)"),
        lambda raw: raw["target"].update(answer_label="RESULT:"),
        lambda raw: raw.update(version=2),
    ],
)
def test_prompt_id_changes_with_the_wording_the_layout_or_the_version(tmp_path, change):
    assert write_prompt(tmp_path, change).prompt_id != PROMPT_ID


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: raw.update(system="   \n"), "system must not be empty"),
        (lambda raw: raw.pop("system"), "is missing: system"),
        (lambda raw: raw["user"].pop("table_label"), "is missing: user.table_label"),
        (lambda raw: raw["target"].update(program_label="PROGRAM\nNOW:"), "must be one line"),
        (lambda raw: raw.update(version=0), "version must be 1 or more"),
        (lambda raw: raw.update(version="one"), "integer 'version'"),
    ],
)
def test_malformed_prompt_file_is_rejected(tmp_path, change, message):
    with pytest.raises(ConfigError, match=message):
        write_prompt(tmp_path, change)


# --------------------------------------------------------------------------- evidence


def test_text_evidence_is_in_document_order_whatever_the_key_order():
    found, skipped = text_evidence(
        example(gold_inds={"text_2": "x", "table_1": "x", "text_0": "x", "table_0": "x"})
    )
    assert found == ["zebrafish revenue rose in the year .", "see the quokka note for details ."]
    assert skipped == []


def test_passages_come_from_the_page_not_from_the_stored_value():
    found, _ = text_evidence(example(gold_inds={"text_1": "something else entirely"}))
    assert found == ["the marmalade segment grew ."]


def test_the_same_passage_text_is_shown_once():
    item = example(pre_text=["same words .", "other words ."], post_text=["same words ."],
                   gold_inds={"text_0": "x", "text_2": "x", "text_1": "x"})  # fmt: skip
    assert text_evidence(item)[0] == ["same words .", "other words ."]


def test_malformed_and_out_of_range_keys_are_skipped_and_reported():
    item = example(gold_inds={"text_-1": "x", "text_0": "x", "text_9": "x", "note": "x"})
    found, skipped = text_evidence(item)
    assert found == ["zebrafish revenue rose in the year ."]
    assert skipped == ["text_-1", "text_9", "note"]


def test_table_only_and_unparsed_evidence_give_no_passages():
    assert text_evidence(example()) == ([], [])
    assert text_evidence(example(gold_inds="not json")) == ([], ["gold_inds"])


# --------------------------------------------------------------------------- table


def test_table_has_a_header_a_separator_and_the_rows():
    table = [["", "2015", "2014"], ["net sales", "$ 920", "95"], ["total", "", "900"]]
    assert format_table(table).splitlines() == [
        "|  | 2015 | 2014 |",
        "|---|---|---|",
        "| net sales | $ 920 | 95 |",
        "| total |  | 900 |",
    ]


def test_one_row_table_is_a_header_and_a_separator():
    assert format_table([["metric", "2015"]]) == "| metric | 2015 |\n|---|---|"


def test_cells_are_copied_word_for_word():
    cell = "$ -23158 ( 23158 ) in millions 10% ( 10 % )"
    assert format_table([["a", "b"], ["name", cell]]).splitlines()[2] == f"| name | {cell} |"


def test_a_bar_is_escaped_and_a_line_break_becomes_a_space():
    lines = format_table([["a", "b"], ["x | y", "first\n second\r\nthird"]]).splitlines()
    assert lines[2] == "| x \\| y | first second third |"
    assert len(lines) == 3


def test_wide_table_keeps_every_column():
    header = [f"c{n}" for n in range(14)]
    lines = format_table([header, [str(n) for n in range(14)]]).splitlines()
    assert lines[1] == "|" + "---|" * 14
    assert lines[2].count("|") == 15


@pytest.mark.parametrize("table", [[], None, [[]]])
def test_empty_table_is_rejected(table):
    with pytest.raises(SerializationError, match="example X has an empty table"):
        format_table(table, "example X")


# --------------------------------------------------------------------------- answer


@pytest.mark.parametrize(
    ("stored", "shown"),
    [
        ("856", "856"),
        ("0.44659", "0.44659"),
        ("-0.05", "-0.05"),
        ("0", "0"),
        ("yes", "yes"),
        ("no", "no"),
        ("4.39e+07", "43900000"),
        ("1.23456e+06", "1234560"),
        ("3.60038e+13", "36003800000000"),
        ("-9.999e+08", "-999900000"),
        ("5e-05", "0.00005"),
        ("9E+06", "9000000"),
    ],
)
def test_answer_forms(stored, shown):
    assert format_answer(stored) == shown


@pytest.mark.parametrize("stored", ["", "n/a", "1,000", "12%", "Yes", "nan", "inf", None])
def test_result_that_is_not_a_number_or_yes_no_is_rejected(stored):
    with pytest.raises(SerializationError, match="neither a number nor yes/no"):
        format_answer(stored, "example X")


# --------------------------------------------------------------------------- records


@pytest.mark.parametrize("name", sorted(GOLDEN_ROWS))
def test_golden_records(prompt, name):
    record = build_record(to_examples([GOLDEN_ROWS[name]])[0], prompt)
    expected = (GOLDEN_DIR / f"{name}.txt").read_text(encoding="utf-8")
    assert golden_text(record) == expected
    assert record.system == prompt.system
    assert record.prompt_id == PROMPT_ID
    assert record.example_id == GOLDEN_ROWS[name]["id"]


def test_golden_folder_holds_exactly_the_golden_rows():
    assert sorted(path.stem for path in GOLDEN_DIR.glob("*.txt")) == sorted(GOLDEN_ROWS)


def test_program_is_copied_unchanged(prompt):
    program = "table_max(expected volatility, none), divide(6.9%, const_m1), exp(#1, #0)"
    record = build_record(example(program=program, exe_ans="1.5"), prompt)
    assert record.target == f"PROGRAM: {program}\nANSWER: 1.5"


def test_user_text_layout(prompt):
    user = format_user(
        ["first .", "second ."], [["", "2024"], ["Revenue", "10"]], "How much?", prompt
    )
    assert user == (
        "Context:\n- first .\n- second .\n\n"
        "Table:\n|  | 2024 |\n|---|---|\n| Revenue | 10 |\n\n"
        "Question: How much?"
    )
    assert format_user([], [["a"]], "q", prompt).startswith("Context:\n(none)\n\nTable:\n")
    assert format_target("add(1, 2)", "3", prompt) == "PROGRAM: add(1, 2)\nANSWER: 3"


def test_record_without_a_program_or_with_a_bad_result_names_the_example(prompt):
    with pytest.raises(SerializationError, match="example ACME/2015/page_10.pdf-1 has no program"):
        build_record(example(program=" "), prompt)
    with pytest.raises(SerializationError, match="example ACME/2015/page_10.pdf-1 has a result"):
        build_record(example(exe_ans="n/a"), prompt)


def test_serialization_is_deterministic_and_the_digest_sees_every_change(prompt):
    examples = to_examples(list(GOLDEN_ROWS.values()))
    first = [build_record(item, prompt) for item in examples]
    assert [build_record(item, prompt) for item in examples] == first
    digest = records_digest(first)
    assert len(digest) == 64 and records_digest(first) == digest
    assert records_digest(list(reversed(first))) != digest
    edited = [dataclasses.replace(first[0], target=first[0].target + " "), *first[1:]]
    assert records_digest(edited) != digest
    assert first[0].to_dict() == {
        "example_id": first[0].example_id,
        "prompt_id": PROMPT_ID,
        "system": prompt.system,
        "user": first[0].user,
        "target": first[0].target,
    }


# --------------------------------------------------------------------------- by split


def test_splits_are_serialized_in_dataset_order(tmp_path, prompt):
    cfg, data_cfg, _ = build_frozen_split(tmp_path)
    for name in ("fit", "dev", "quick_dev"):
        records = serialize_split(name, prompt, cfg, data_cfg)
        assert [record.example_id for record in records] == split_ids(name, cfg)
        assert all(record.prompt_id == PROMPT_ID for record in records)


def test_the_test_split_cannot_be_serialized(tmp_path, prompt):
    cfg, data_cfg, _ = build_frozen_split(tmp_path)
    with pytest.raises(TestLockedError):
        serialize_split("test", prompt, cfg, data_cfg)


def test_real_fit_and_dev_serialize_to_the_expected_counts():
    """Skips where the snapshot has not been fetched (CI)."""
    if not all(item.ok for item in verify_snapshot(load_data_config())):
        pytest.skip("no FinQA snapshot here; fetch it with: python -m scripts.finqa_fetch")
    assert len(serialize_split("fit")) == 4884
    assert len(serialize_split("dev")) == 499
    with pytest.raises(TestLockedError):
        serialize_split("test")
