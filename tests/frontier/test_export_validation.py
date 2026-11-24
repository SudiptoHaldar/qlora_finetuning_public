"""Tests for qlora_finetuning.frontier.export_validation: each check passing and failing.

The files are tampered with on disk after the export, because the validator reads the disk.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from decimal import Decimal

import pytest

from qlora_finetuning.data import load_data_config, verify_snapshot
from qlora_finetuning.evaluation.runner import RunnerContext
from qlora_finetuning.frontier import export_validation
from qlora_finetuning.frontier.config import load_frontier_config
from qlora_finetuning.frontier.export import RESERVED_WORDS, export_files
from qlora_finetuning.frontier.export_validation import load_token_volumes, validate_export
from qlora_finetuning.settings import REPO_ROOT, ConfigError
from tests.evaluation.conftest import ids_of, records_for
from tests.frontier.export_helpers import RATIO, detail, failed, volumes_for

TOKEN_REPORT = REPO_ROOT / "reports" / "data" / "finqa_token_budget_report.json"


@pytest.fixture
def export(context, cfg):
    return export_files(cfg, context)[0]


def validate(export, cfg, context, ratio=RATIO, **volume_changes):
    return validate_export(export, cfg, context, volumes_for(context, **volume_changes), ratio)


def lines_of(file) -> list[str]:
    return file.path.read_text(encoding="utf-8").splitlines()


def write_lines(file, lines: list[str]) -> None:
    file.path.write_bytes("".join(line + "\n" for line in lines).encode("utf-8"))


def change_line(file, index: int, change) -> None:
    """Apply ``change`` to the parsed line ``index`` and write it back in the same form."""
    lines = lines_of(file)
    data = json.loads(lines[index])
    change(data)
    lines[index] = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    write_lines(file, lines)


def set_text(part: str, text: str):
    def change(data):
        if part == "system":
            data["system"][0]["text"] = text
        else:
            data["messages"][0 if part == "user" else 1]["content"][0]["text"] = text

    return change


# --------------------------------------------------------------------------- the good case


def test_a_fresh_export_passes_every_check(export, cfg, context):
    checks = validate(export, cfg, context)
    assert [item["id"] for item in checks] == ["X1", "X2", "X3", "X4", "X5", "X6", "X7"]
    assert failed(checks) == []
    assert detail(checks, "X1") == "one line per record, in order: fit 30, dev 8"
    assert detail(checks, "X2") == "none of 38 exported ids is among the 22 test ids"
    assert "30 training and 8 validation records" in detail(checks, "X5")
    assert "about 1,980 of 32,000 tokens" in detail(checks, "X5")  # 1,200 * 1.5 * 1.1


def test_no_detail_ever_shows_a_text(export, cfg, context):
    file = export.files["training"]
    change_line(file, 2, set_text("user", "zebrafish quokka User: marmalade"))
    text = json.dumps(validate(export, cfg, context))
    for word in ("zebrafish", "quokka", "marmalade", "subtract", "PROGRAM"):
        assert word not in text, word
    assert ids_of("fit", context)[2] in text  # the example is named by its id


def test_only_the_ids_of_the_test_split_are_read(export, cfg, context, monkeypatch):
    asked = []
    real = export_validation.split_ids
    monkeypatch.setattr(
        export_validation, "split_ids", lambda name, *a: asked.append(name) or real(name, *a)
    )
    assert failed(validate(export, cfg, context)) == []
    assert asked == ["test"]  # ids only; the loader refuses the rows while the split is locked


# --------------------------------------------------------------------------- X1


def test_x1_fails_for_one_changed_character(export, cfg, context):
    record = records_for("dev", context)[3]
    change_line(export.files["validation"], 3, set_text("user", record.user + " "))
    checks = validate(export, cfg, context)
    assert "X1" in failed(checks) and "X7" in failed(checks)
    assert detail(checks, "X1") == (
        f"1 validation lines differ from their canonical record: {record.example_id}"
    )


@pytest.mark.parametrize("part", ["system", "target"])
def test_x1_fails_when_the_system_or_the_target_is_changed(export, cfg, context, part):
    change_line(export.files["training"], 0, set_text(part, "another text"))
    assert "X1" in failed(validate(export, cfg, context))


def test_x1_fails_for_a_missing_an_extra_and_two_swapped_lines(export, cfg, context):
    file = export.files["training"]
    original = lines_of(file)

    write_lines(file, original[:-1])
    assert "has 29 lines for 30 records" in detail(validate(export, cfg, context), "X1")

    write_lines(file, [*original, original[0]])
    assert "has 31 lines for 30 records" in detail(validate(export, cfg, context), "X1")

    swapped = [original[1], original[0], *original[2:]]
    write_lines(file, swapped)
    checks = validate(export, cfg, context)
    # The invented rows differ in their question, so a swap is two differing lines.
    assert "2 training lines differ from their canonical record" in detail(checks, "X1")


def test_x1_fails_when_the_id_list_is_not_the_split(export, cfg, context):
    file = export.files["validation"]
    ids = file.ids_path.read_text(encoding="utf-8").splitlines()
    file.ids_path.write_text("\n".join(reversed(ids)) + "\n", encoding="utf-8")
    assert "the validation id list is not the dev split in order" in detail(
        validate(export, cfg, context), "X1"
    )


def test_a_missing_file_fails_without_a_traceback(export, cfg, context):
    export.files["training"].path.unlink()
    checks = validate(export, cfg, context)
    assert {"X1", "X3", "X5", "X7"} <= set(failed(checks))
    assert "the training file or its id list is missing" in detail(checks, "X3")


# --------------------------------------------------------------------------- X2


def test_x2_fails_when_an_exported_id_is_a_test_id(export, cfg, context, monkeypatch):
    leaked = ids_of("fit", context)[4]
    monkeypatch.setattr(export_validation, "split_ids", lambda name, *a: [leaked, "other-id"])
    checks = validate(export, cfg, context)
    assert failed(checks) == ["X2"]
    assert detail(checks, "X2") == f"1 training ids are test ids: {leaked}"


def test_x2_fails_when_an_id_is_in_both_files(export, cfg, context):
    shared = ids_of("fit", context)[0]
    file = export.files["validation"]
    ids = file.ids_path.read_text(encoding="utf-8").splitlines()
    file.ids_path.write_text("\n".join([shared, *ids[1:]]) + "\n", encoding="utf-8")
    checks = validate(export, cfg, context)
    assert "X2" in failed(checks)
    assert f"1 ids are in both files: {shared}" in detail(checks, "X2")


# --------------------------------------------------------------------------- X3


def swap_roles(data):
    data["messages"][0]["role"], data["messages"][1]["role"] = "assistant", "user"


def add_key(data):
    data["example_id"] = "ACME/2015/page_10.pdf-1"


def drop_system(data):
    del data["system"]


def third_turn(data):
    data["messages"].append({"role": "user", "content": [{"text": "more"}]})


def other_schema(data):
    data["schemaVersion"] = "bedrock-conversation-2023"


def extra_content_key(data):
    data["messages"][0]["content"][0]["image"] = {}


@pytest.mark.parametrize(
    "change",
    [
        swap_roles,
        add_key,
        drop_system,
        third_turn,
        other_schema,
        extra_content_key,
        set_text("user", ""),
        set_text("target", "   "),
        set_text("system", 7),
    ],
    ids=lambda change: getattr(change, "__name__", "text"),
)
def test_x3_fails_for_a_line_of_another_shape(export, cfg, context, change):
    change_line(export.files["training"], 5, change)
    checks = validate(export, cfg, context)
    assert "X3" in failed(checks)
    assert "1 training lines are not a conversation of the expected shape: line 6" in (
        detail(checks, "X3")
    )


def test_x3_fails_for_a_line_that_is_not_json(export, cfg, context):
    file = export.files["validation"]
    lines = lines_of(file)
    write_lines(file, [lines[0], "{not json", *lines[2:]])
    assert "validation lines are not a conversation of the expected shape: line 2" in detail(
        validate(export, cfg, context), "X3"
    )


def test_x3_fails_for_a_carriage_return_a_missing_final_break_and_bad_bytes(export, cfg, context):
    file = export.files["training"]
    original = file.path.read_bytes()

    file.path.write_bytes(original.replace(b"\n", b"\r\n"))
    assert "the training file holds a carriage return" in detail(
        validate(export, cfg, context), "X3"
    )

    file.path.write_bytes(original[:-1])
    assert "the training file does not end with a line break" in detail(
        validate(export, cfg, context), "X3"
    )

    file.path.write_bytes(original + b"\xff\xfe\n")
    assert "the training file is not UTF-8" in detail(validate(export, cfg, context), "X3")


def test_x3_fails_for_a_file_name_aws_does_not_allow(export, cfg, context):
    file = export.files["training"]
    renamed = file.path.with_name("finqa fit(1).jsonl")
    shutil.copy(file.path, renamed)
    other = dataclasses.replace(
        export, files={**export.files, "training": dataclasses.replace(file, path=renamed)}
    )
    checks = validate(other, cfg, context)
    assert failed(checks) == ["X3"]
    assert "file name 'finqa fit(1).jsonl' has a character AWS does not allow" in (
        detail(checks, "X3")
    )
    assert "the S3 key of the training file has a character AWS does not allow" in (
        detail(checks, "X3")
    )


# --------------------------------------------------------------------------- X4


@pytest.mark.parametrize("word", RESERVED_WORDS)
@pytest.mark.parametrize("part", ["system", "user", "target"])
def test_x4_fails_for_each_reserved_word_in_each_text_in_any_case(
    export, cfg, context, word, part
):
    file = export.files["training"]
    change_line(file, 7, set_text(part, f"a line\nthen {word.upper()} in the middle"))
    checks = validate(export, cfg, context)
    assert "X4" in failed(checks)
    example_id = ids_of("fit", context)[7]
    assert detail(checks, "X4") == f"1 records hold a reserved word: {example_id} ({word})"


def test_x4_passes_for_words_that_only_look_alike(export, cfg, context):
    file = export.files["training"]
    record = records_for("fit", context)[1]
    harmless = record.user + "\nusers : 4 ; systems ; bots, [eo s] <images> assistant 's"
    change_line(file, 1, set_text("user", harmless))
    assert "X4" not in failed(validate(export, cfg, context))


def test_x4_reads_a_malformed_line_as_raw_text(export, cfg, context):
    file = export.files["validation"]
    lines = lines_of(file)
    write_lines(file, ['{"oops": "Bot: hello"}', *lines[1:]])
    checks = validate(export, cfg, context)
    assert "X4" in failed(checks) and "(bot:)" in detail(checks, "X4")


# --------------------------------------------------------------------------- X5


def test_x5_fails_below_the_minimum_and_above_the_maximum(export, cfg, context):
    few = validate(export, dataclasses.replace(cfg, min_records=31), context)
    assert failed(few) == ["X5"]
    assert detail(few, "X5").startswith("30 training records; AWS takes 31 to 20,000")

    many = validate(export, dataclasses.replace(cfg, max_records=29), context)
    assert "30 training records; AWS takes 8 to 29" in detail(many, "X5")

    both = validate(export, dataclasses.replace(cfg, max_records=37), context)
    assert failed(both) == ["X5"]
    assert detail(both, "X5") == "38 training and validation records; AWS takes at most 37"

    edge = validate(export, dataclasses.replace(cfg, min_records=30, max_records=38), context)
    assert failed(edge) == []


def test_x5_fails_when_the_ratio_pushes_the_longest_record_past_the_limit(export, cfg, context):
    # 1,200 Qwen tokens * ratio * 1.1 must stay within 32,000.
    assert failed(validate(export, cfg, context, ratio=Decimal("24.2"))) == []  # 31,944
    checks = validate(export, cfg, context, ratio=Decimal("24.3"))  # 32,076
    assert failed(checks) == ["X5"]
    assert "the longest fit record is about 32,076 tokens; AWS takes at most 32,000" in (
        detail(checks, "X5")
    )


def test_x5_fails_when_the_token_report_lacks_a_split(export, cfg, context):
    volumes = volumes_for(context)
    volumes = dataclasses.replace(volumes, splits={"fit": volumes.splits["fit"]})
    checks = validate_export(export, cfg, context, volumes, RATIO)
    assert failed(checks) == ["X5"] and "has no dev split" in detail(checks, "X5")


# --------------------------------------------------------------------------- X6 and X7


def test_x6_fails_when_the_token_report_is_for_another_prompt_or_split(export, cfg, context):
    checks = validate(export, cfg, context, prompt_id="f" * 64)
    assert failed(checks) == ["X6"]
    assert "the token budget report was made from another prompt wording" in detail(checks, "X6")
    checks = validate(export, cfg, context, split_hash="0" * 64)
    assert failed(checks) == ["X6"]
    assert "the token budget report was made from another split manifest" in detail(checks, "X6")


def test_x6_fails_when_the_files_were_made_from_another_prompt_or_split(export, cfg, context):
    checks = validate(dataclasses.replace(export, prompt_id="e" * 64), cfg, context)
    assert failed(checks) == ["X6"]
    assert "the files were made from another prompt wording" in detail(checks, "X6")
    checks = validate(dataclasses.replace(export, split_hash="1" * 64), cfg, context)
    assert "the files were made from another split manifest" in detail(checks, "X6")


def test_x7_fails_when_the_recorded_hash_is_not_the_files(export, cfg, context):
    file = export.files["training"]
    other = dataclasses.replace(
        export, files={**export.files, "training": dataclasses.replace(file, sha256="a" * 64)}
    )
    checks = validate(other, cfg, context)
    assert failed(checks) == ["X7"]
    assert "the training file on disk is not what rendering gives now" in detail(checks, "X7")


def test_x7_fails_for_any_changed_byte(export, cfg, context):
    file = export.files["validation"]
    lines = lines_of(file)
    # The same JSON with a space added: the texts are equal, the bytes are not.
    write_lines(file, [lines[0].replace('"system":', '"system": '), *lines[1:]])
    assert failed(validate(export, cfg, context)) == ["X7"]


# --------------------------------------------------------------------------- token volumes


def test_the_committed_token_budget_report_gives_the_volumes():
    volumes = load_token_volumes(TOKEN_REPORT)
    fit, dev = volumes.splits["fit"], volumes.splits["dev"]
    assert (fit.records, fit.tokens_in_one_pass, fit.longest) == (4884, 3_254_181, 1862)
    assert (dev.records, dev.tokens_in_one_pass, dev.longest) == (499, 333_677, 916)
    assert fit.prompt_tokens_in_one_pass == 3_103_361
    assert len(volumes.prompt_id) == 64 and len(volumes.split_hash) == 64


def test_a_missing_or_damaged_token_budget_report_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="token budget report not found"):
        load_token_volumes(tmp_path / "absent.json")
    damaged = tmp_path / "report.json"
    for content in ("{not json", "{}", '{"inputs": {}, "splits": {}}', "[]"):
        damaged.write_text(content, encoding="utf-8")
        with pytest.raises(ConfigError, match="token budget report cannot be read"):
            load_token_volumes(damaged)


# --------------------------------------------------------------------------- the real data


def test_the_real_export_passes_every_check(tmp_path):
    if not all(item.ok for item in verify_snapshot(load_data_config())):
        pytest.skip("no FinQA snapshot here; fetch it with: python -m scripts.finqa_fetch")
    cfg = load_frontier_config(root=tmp_path)  # the files go under tmp_path
    context = RunnerContext().resolved()
    export, written = export_files(cfg, context)
    assert written and str(export.folder).startswith(str(tmp_path))
    assert (export.files["training"].records, export.files["validation"].records) == (4884, 499)
    checks = validate_export(
        export, cfg, context, load_token_volumes(TOKEN_REPORT), cfg.assumed_token_ratio
    )
    assert failed(checks) == []
    assert detail(checks, "X2") == "none of 5,383 exported ids is among the 2,898 test ids"
