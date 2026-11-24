"""The parser and the executor on the real fit and dev targets (req_002 v2.01).

Skipped where the snapshot has not been fetched (CI). Nothing from an example is written
here: the test compares counts with the committed split report and holds no example text.
The test split is not read.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from qlora_finetuning.data import (
    load_data_config,
    load_prompt,
    load_split,
    serialize_split,
    verify_snapshot,
)
from qlora_finetuning.dsl import (
    ReplyFormat,
    execute,
    format_program,
    load_dsl_config,
    parse_program,
    parse_reply,
)
from qlora_finetuning.settings import REPO_ROOT

EXPECTED_EXAMPLES = {"fit": 4884, "dev": 499}


@pytest.fixture(scope="module")
def split_report() -> dict:
    if not all(item.ok for item in verify_snapshot(load_data_config())):
        pytest.skip("no FinQA snapshot here; fetch it with: python -m scripts.finqa_fetch")
    path = REPO_ROOT / "reports" / "data" / "finqa_split_report.json"
    return json.loads(path.read_text(encoding="utf-8"))["columns"]


@pytest.mark.parametrize("name", ["fit", "dev"])
def test_every_gold_target_parses_executes_and_matches_the_inventory(name, split_report):
    config = load_dsl_config()
    reply_format = ReplyFormat.from_prompt(load_prompt())
    examples = load_split(name)
    records = serialize_split(name)
    assert len(examples) == len(records) == EXPECTED_EXAMPLES[name]

    operations: Counter = Counter()
    lengths: Counter = Counter()
    kinds: Counter = Counter()
    not_strict, not_canonical, failed = [], [], []
    for example, record in zip(examples, records, strict=True):
        assert example.id == record.example_id
        parsed = parse_reply(record.target, reply_format, config)
        if not parsed.strict:
            not_strict.append(example.id)
        program = parse_program(parsed.program_text, config)
        if format_program(program) != example.program:
            not_canonical.append(example.id)
        lengths[str(len(program.steps))] += 1
        for step in program.steps:
            operations[step.operation] += 1
            kinds.update(argument.kind for argument in step.arguments)
        try:
            execute(program, example.table)
        except Exception as error:  # noqa: BLE001 - report the id, never the example's text
            failed.append((example.id, type(error).__name__))

    assert not_strict == []
    assert not_canonical == []
    assert failed == []
    # v1.01 D14: the parser agrees with the counting pattern of the data reports.
    assert dict(operations) == split_report[name]["operations"]
    assert dict(lengths) == split_report[name]["program_length"]
    assert dict(kinds) == split_report[name]["argument_kinds"]
