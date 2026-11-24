"""Tests for qlora_finetuning.frontier.export, on the invented split."""

from __future__ import annotations

import dataclasses
import hashlib
import json

from qlora_finetuning.data import CanonicalRecord
from qlora_finetuning.frontier import export as export_module
from qlora_finetuning.frontier.export import (
    RESERVED_WORDS,
    conversation_line,
    export_files,
    render_file,
)
from tests.evaluation.conftest import ids_of, records_for

SCHEMA = "bedrock-conversation-2024"
# Invented texts with everything a JSON writer could change.
AWKWARD = CanonicalRecord(
    example_id="ACME/2031/page_7.pdf-1",
    prompt_id="p" * 64,
    system='Reply with "two" lines.\n\tBackslash \\ and a unicode pound £. ',
    user="Context:\n- (none)\n\nTable:\n| kelp | 920 |\n\nQuestion: why € / {x} <b>?\n",
    target="PROGRAM: subtract(920, 95)\nANSWER: 825",
)


def test_a_line_holds_the_three_texts_character_for_character():
    line = conversation_line(AWKWARD, SCHEMA)
    assert "\n" not in line and "\r" not in line  # one record, one line
    data = json.loads(line)
    assert data["system"] == [{"text": AWKWARD.system}]
    assert data["messages"] == [
        {"role": "user", "content": [{"text": AWKWARD.user}]},
        {"role": "assistant", "content": [{"text": AWKWARD.target}]},
    ]
    assert data["schemaVersion"] == SCHEMA
    assert "ACME/2031" not in line and "p" * 64 not in line  # no id, no prompt id


def test_a_line_has_fixed_key_order_compact_form_and_unescaped_characters():
    line = conversation_line(AWKWARD, SCHEMA)
    assert list(json.loads(line)) == ["schemaVersion", "system", "messages"]
    assert line.startswith('{"schemaVersion":"bedrock-conversation-2024","system":[{"text":"')
    assert '"messages":[{"role":"user","content":[{"text":"' in line
    assert "£" in line and "€" in line and "\\u00a3" not in line
    assert ", " not in line.split('"text"')[0] and '": ' not in line.split("Reply")[0]
    assert conversation_line(AWKWARD, SCHEMA) == line  # the same record, the same bytes


def test_a_golden_line():
    record = CanonicalRecord("id-1", "pid", "S", 'U "q"\n', "PROGRAM: add(1, 2)\nANSWER: 3")
    assert conversation_line(record, SCHEMA) == (
        '{"schemaVersion":"bedrock-conversation-2024","system":[{"text":"S"}],'
        '"messages":[{"role":"user","content":[{"text":"U \\"q\\"\\n"}]},'
        '{"role":"assistant","content":[{"text":"PROGRAM: add(1, 2)\\nANSWER: 3"}]}]}'
    )


def test_a_file_is_one_line_per_record_each_ended_by_a_line_break():
    content = render_file([AWKWARD, AWKWARD], SCHEMA)
    assert content.count(b"\n") == 2 and content.endswith(b"\n") and b"\r" not in content
    assert content.decode("utf-8").splitlines()[0] == conversation_line(AWKWARD, SCHEMA)
    assert render_file([], SCHEMA) == b""


def test_export_writes_fit_and_dev_in_dataset_order(context, cfg):
    export, written = export_files(cfg, context)
    assert written is True
    training, validation = export.files["training"], export.files["validation"]
    assert (training.split, training.records) == ("fit", 30)
    assert (validation.split, validation.records) == ("dev", 8)
    assert training.path.name == "finqa_fit.jsonl" and validation.path.name == "finqa_dev.jsonl"
    for file, split in ((training, "fit"), (validation, "dev")):
        lines = file.path.read_text(encoding="utf-8").splitlines()
        records = records_for(split, context)
        assert lines == [conversation_line(record, SCHEMA) for record in records]
        assert file.ids_path.read_text(encoding="utf-8").splitlines() == ids_of(split, context)
        raw = file.path.read_bytes()
        assert (file.size, file.sha256) == (len(raw), hashlib.sha256(raw).hexdigest())
    assert export.prompt_id == context.prompt.prompt_id and export.schema_version == SCHEMA


def test_the_folder_is_named_after_the_content_and_is_under_the_export_dir(context, cfg, tmp_path):
    export, _ = export_files(cfg, context)
    digests = "\n".join(export.files[role].sha256 for role in ("training", "validation"))
    assert export.export_id == hashlib.sha256(digests.encode()).hexdigest()[:12]
    assert export.folder == tmp_path / "artifacts" / "bedrock" / export.export_id
    assert sorted(path.name for path in export.folder.iterdir()) == [
        "finqa_dev.jsonl",
        "finqa_dev.jsonl.ids.txt",
        "finqa_fit.jsonl",
        "finqa_fit.jsonl.ids.txt",
    ]


def test_a_second_export_writes_nothing_and_gives_the_same_export(context, cfg):
    first, _ = export_files(cfg, context)
    stamp = first.files["training"].path.stat().st_mtime_ns
    second, written = export_files(cfg, context)
    assert written is False and second == first
    assert second.files["training"].path.stat().st_mtime_ns == stamp


def test_a_file_changed_on_disk_is_written_again(context, cfg):
    export, _ = export_files(cfg, context)
    path = export.files["validation"].path
    path.write_bytes(path.read_bytes()[:-20])
    again, written = export_files(cfg, context)
    assert written is True
    assert hashlib.sha256(path.read_bytes()).hexdigest() == again.files["validation"].sha256


def test_only_fit_and_dev_are_ever_loaded(context, cfg, monkeypatch):
    asked = []
    real = export_module.serialize_split

    def watched(name, *arguments):
        asked.append(name)
        return real(name, *arguments)

    monkeypatch.setattr(export_module, "serialize_split", watched)
    export_files(cfg, context)
    assert asked == ["fit", "dev"]
    assert set(export_module.SPLITS.values()) == {"fit", "dev"}


def test_another_schema_version_gives_other_bytes_and_another_folder(context, cfg):
    first, _ = export_files(cfg, context)
    other, _ = export_files(dataclasses.replace(cfg, schema_version="v-next"), context)
    assert other.export_id != first.export_id and other.folder != first.folder


def test_repeated_lines_are_kept_and_counted(context, cfg):
    export, _ = export_files(cfg, context)
    for file in export.files.values():
        lines = file.path.read_bytes().splitlines()
        assert file.repeated_lines == len(lines) - len(set(lines))
        assert len(lines) == file.records  # nothing was dropped


def test_the_reserved_words_are_the_nine_of_awss_script_in_lower_case():
    assert RESERVED_WORDS == (
        "system:", "user:", "bot:", "assistant:", "thought:", "[eos]", "<image>", "<video>",
        "<unk>",
    )  # fmt: skip
    assert all(word == word.lower() for word in RESERVED_WORDS)
