"""The FinQA schema report: what is actually in the pinned snapshot (req_001 v1.01).

Pure functions over the raw records. The report holds **counts and example ids only**: no
question, passage, table cell, or answer text, because any example may be assigned to the test
split later. Of the program text it shows operation names, constant names, and argument kinds.
It carries no timestamp, so the same snapshot always gives the same bytes.

Usage::

    from qlora_finetuning.data import load_data_config, load_raw_examples, read_columns
    from qlora_finetuning.data.schema_report import build_report, write_report

    cfg = load_data_config()
    report = build_report(load_raw_examples(cfg), cfg, read_columns(cfg))
    write_report(report, cfg.report_dir)
    report["total"]["programs"]["operations"]      # {"divide": 5901, ...}

Programs are split here with one small pattern, only to count operations and arguments. It is
not the DSL parser (req_002 v2.01), which must agree with this inventory.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from qlora_finetuning.data.snapshot import DataConfig, RawExample

REPORT_STEM = "finqa_schema_report"
REPORT_COMMAND = "python -m scripts.finqa_report"

_STEP = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\(([^()]*)\)\s*")
_INTEGER = re.compile(r"-?\d+")
_DECIMAL = re.compile(r"-?\d*\.\d+")
_PERCENT = re.compile(r"-?\d*\.?\d+%")
_REFERENCE = re.compile(r"#\d+")
_CONSTANT = re.compile(r"const_\w+")
_EVIDENCE_KEY = re.compile(r"(table|text)_(\d+)")
_PLAIN_NUMBER = re.compile(r"-?[\d,]*\.?\d+")
_YES_NO = ("yes", "no")

# Oddity classes listed by id in the report, in this order, with their plain description.
FLAGS = {
    "id_repeated": "The id occurs more than once",
    "id_pattern_mismatch": "The id does not match the pattern or does not extend the filename",
    "duplicate_page_and_question": "Another example has the same page and the same question",
    "answer_empty": "The answer field is empty",
    "answer_other_form": "The answer is not a number, a percentage, a dollar amount, or yes/no",
    "evidence_unparsed": "gold_inds is not a JSON object",
    "evidence_empty": "gold_inds has no entries",
    "evidence_malformed_key": "A gold_inds key is not table_N or text_N",
    "evidence_key_out_of_range": "A gold_inds key points past the end of the table or the passages",
    "program_unparsed": "The program does not have the form op(a, b), op(a, b), ...",
    "operation_unknown": "The program uses an operation outside the expected set",
    "constant_unknown": "The program uses a constant outside the expected set",
    "argument_other_kind": "An argument is of none of the known kinds",
    "argument_count_unexpected": "A step does not have the expected number of arguments",
    "percent_literal_argument": "An argument is a percent literal such as 6.9%",
    "row_name_not_in_table": "A table operation names a row that is not in the example's table",
    "table_second_argument_not_none": "A table operation's second argument is not none",
    "longest_programs": "The program has the largest number of steps seen",
    "result_scientific_notation": "exe_ans is written in scientific notation",
    "result_not_numeric_or_yes_no": "exe_ans is neither a number nor yes/no",
    "yes_no_answer_without_greater": "The answer is yes or no but the program has no greater step",
    "greater_answer_disagrees": "The program has a greater step and the answer is not exe_ans",
    "greater_not_last_step": "A greater step is not the last step of the program",
}


# --------------------------------------------------------------------------- programs


def split_program(text: Any) -> list[tuple[str, list[str]]] | None:
    """Split ``op(a, b), op(c, d)`` into ``[(op, [a, b]), ...]``; ``None`` for any other form."""
    if not isinstance(text, str):
        return None
    steps: list[tuple[str, list[str]]] = []
    position = 0
    while True:
        match = _STEP.match(text, position)
        if match is None:
            return None
        steps.append((match.group(1), [part.strip() for part in match.group(2).split(",")]))
        position = match.end()
        if position == len(text):
            return steps
        if text[position] != ",":
            return None
        position += 1


def argument_kind(text: str) -> str:
    """Classify one argument of an arithmetic step."""
    if _REFERENCE.fullmatch(text):
        return "reference"
    if _CONSTANT.fullmatch(text):
        return "constant"
    if _INTEGER.fullmatch(text):
        return "integer"
    if _DECIMAL.fullmatch(text):
        return "decimal"
    if _PERCENT.fullmatch(text):
        return "percent"
    if text == "none":
        return "none"
    return "other"


def is_table_operation(name: str) -> bool:
    return name.startswith("table_")


# --------------------------------------------------------------------------- per example


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _items(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _is_number(text: str) -> bool:
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


def _facts(example: RawExample) -> dict[str, Any]:
    """Everything the report counts about one example. Holds no text from the example."""
    table = _items(example.table)
    row_names = {str(row[0]).strip() for row in table if _items(row)}
    passages = len(_items(example.pre_text)) + len(_items(example.post_text))

    steps = split_program(example.program)
    operations: list[str] = []
    constants: list[str] = []
    kinds: list[str] = []
    arities: list[int] = []
    second_arguments: list[str] = []
    rows_found = rows_missing = 0
    for name, arguments in steps or []:
        operations.append(name)
        arities.append(len(arguments))
        if is_table_operation(name):
            kinds.append("row_name")
            if arguments[0] in row_names:
                rows_found += 1
            else:
                rows_missing += 1
            rest = arguments[1:]
            second_arguments.extend(argument_kind(argument) for argument in rest[:1])
        else:
            rest = arguments
        for argument in rest:
            kind = argument_kind(argument)
            kinds.append(kind)
            if kind == "constant":
                constants.append(argument)
    greater_at = [i for i, name in enumerate(operations) if name == "greater"]

    gold = example.gold_inds
    table_keys: set[int] = set()
    text_keys: set[int] = set()
    malformed = out_of_range = 0
    for key in gold or {}:
        match = _EVIDENCE_KEY.fullmatch(str(key))
        if match is None:
            malformed += 1
            continue
        index = int(match.group(2))
        if match.group(1) == "table":
            table_keys.add(index)
            out_of_range += index >= len(table)
        else:
            text_keys.add(index)
            out_of_range += index >= passages

    answer = _text(example.answer)
    answer_forms = set()
    if answer.strip() == "":
        answer_forms.add("empty")
    if "%" in answer:
        answer_forms.add("percent")
    if "$" in answer:
        answer_forms.add("dollar")
    if answer.strip().lower() in _YES_NO:
        answer_forms.add("yes_no")
    if _PLAIN_NUMBER.fullmatch(answer):
        answer_forms.add("plain_number")
    if not answer_forms:
        answer_forms.add("other")

    result = _text(example.exe_ans)
    if result in _YES_NO:
        result_form = result
    elif _is_number(result):
        result_form = "numeric"
    else:
        result_form = "other"

    return {
        "steps": steps,
        "operations": operations,
        "constants": constants,
        "argument_kinds": kinds,
        "arities": arities,
        "second_arguments": second_arguments,
        "rows_found": rows_found,
        "rows_missing": rows_missing,
        "has_greater": bool(greater_at),
        "greater_not_last": any(i != len(operations) - 1 for i in greater_at),
        "evidence_entries": None if gold is None else len(gold),
        "table_keys": table_keys,
        "text_keys": text_keys,
        "malformed_keys": malformed,
        "out_of_range_keys": out_of_range,
        "answer_forms": answer_forms,
        "result_form": result_form,
        "result_scientific": result_form == "numeric" and "e" in result.lower(),
        "id_ok": example.ticker is not None
        and _text(example.id).startswith(_text(example.filename) + "-"),
    }


def example_facts(example: RawExample) -> dict[str, Any]:
    """What the reports count about one example; pair it with the example for ``profile``."""
    return _facts(example)


# --------------------------------------------------------------------------- aggregation


def _by_count(counter: Counter) -> dict[str, int]:
    """Largest count first, then by key, so the order never depends on hashing."""
    return {str(k): v for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))}


def _by_number(counter: Counter) -> dict[str, int]:
    return {str(k): counter[k] for k in sorted(counter)}


def _spread(values: Iterable[int]) -> dict[str, Any]:
    values = list(values)
    if not values:
        return {"min": None, "median": None, "max": None}
    median = statistics.median(values)
    return {
        "min": min(values),
        "median": int(median) if float(median).is_integer() else float(median),
        "max": max(values),
    }


def _is_empty(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, dict) and "op" in value:
        return len(_items(value["op"])) == 0
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


COLUMNS = (
    "id", "filename", "pre_text", "post_text", "table", "table_ori", "question", "answer",
    "exe_ans", "explanation", "ann_table_rows", "ann_text_rows", "steps", "program",
    "program_re", "gold_inds",
)  # fmt: skip


def profile(pairs: Sequence[tuple[RawExample, dict[str, Any]]]) -> dict[str, Any]:
    """Counts over ``(example, facts)`` pairs: the whole dataset or one official split."""
    examples = [example for example, _ in pairs]
    facts = [fact for _, fact in pairs]

    columns = {}
    for name in COLUMNS:
        values = [getattr(example, name) for example in examples]
        if name == "gold_inds":  # None here means "not a JSON object", counted under evidence
            columns[name] = {"null": 0, "empty": sum(value == {} for value in values)}
        else:
            columns[name] = {
                "null": sum(value is None for value in values),
                "empty": sum(_is_empty(value) for value in values),
            }

    ids = Counter(example.id for example in examples)
    years = [example.year for example in examples if example.year is not None]
    identity = {
        "unique_ids": len(ids),
        "repeated_ids": len(examples) - len(ids),
        "ids_not_matching_the_pattern": sum(not fact["id_ok"] for fact in facts),
        "companies": len({e.ticker for e in examples if e.ticker is not None}),
        "reports": len({e.report for e in examples if e.report is not None}),
        "pages": len({e.filename for e in examples}),
        "first_year": min(years) if years else None,
        "last_year": max(years) if years else None,
        "examples_per_company": _spread(
            Counter(e.ticker for e in examples if e.ticker is not None).values()
        ),
        "examples_per_report": _spread(
            Counter(e.report for e in examples if e.report is not None).values()
        ),
        "examples_per_page": _spread(Counter(e.filename for e in examples).values()),
    }

    groups: dict[tuple[Any, Any], list[RawExample]] = {}
    for example in examples:
        groups.setdefault((example.filename, example.question), []).append(example)
    repeated = [group for group in groups.values() if len(group) > 1]
    questions = Counter(example.question for example in examples)
    duplicates = {
        "groups_with_the_same_page_and_question": len(repeated),
        "examples_in_those_groups": sum(len(group) for group in repeated),
        "groups_also_sharing_program_and_result": sum(
            len({(e.program, e.exe_ans) for e in group}) == 1 for group in repeated
        ),
        "examples_sharing_question_wording_with_another": sum(
            questions[example.question] > 1 for example in examples
        ),
    }

    def not_ascii(texts: Iterable[Any]) -> bool:
        return any(not _text(text).isascii() for text in texts)

    text = {
        "passages_before_the_table": _spread(len(_items(e.pre_text)) for e in examples),
        "passages_after_the_table": _spread(len(_items(e.post_text)) for e in examples),
        "examples_with_a_non_ascii_question": sum(not_ascii([e.question]) for e in examples),
        "examples_with_a_non_ascii_passage": sum(
            not_ascii(_items(e.pre_text) + _items(e.post_text)) for e in examples
        ),
        "examples_with_a_non_ascii_table_cell": sum(
            not_ascii(cell for row in _items(e.table) for cell in _items(row)) for e in examples
        ),
    }

    def cells(example: RawExample) -> list[str]:
        return [_text(cell) for row in _items(example.table) for cell in _items(row)]

    tables = {
        "rows": _spread(len(_items(e.table)) for e in examples),
        "columns": _spread(len(_items(e.table)[0]) for e in examples if _items(e.table)),
        "ragged_tables": sum(
            len({len(_items(row)) for row in _items(e.table)}) > 1 for e in examples
        ),
        "tables_with_an_empty_cell": sum(
            any(cell.strip() == "" for cell in cells(e)) for e in examples
        ),
        "tables_with_a_pipe_character": sum(
            any("|" in cell for cell in cells(e)) for e in examples
        ),
        "tables_equal_to_table_ori": sum(e.table == e.table_ori for e in examples),
    }

    def ann_disagrees(example: RawExample, fact: dict[str, Any]) -> bool:
        return fact["table_keys"] != set(_items(example.ann_table_rows)) or fact[
            "text_keys"
        ] != set(_items(example.ann_text_rows))

    def ann_empty(example: RawExample) -> bool:
        return not _items(example.ann_table_rows) and not _items(example.ann_text_rows)

    evidence = {
        "gold_inds_not_a_json_object": sum(fact["evidence_entries"] is None for fact in facts),
        "entries_per_example": _by_number(
            Counter(f["evidence_entries"] for f in facts if f["evidence_entries"] is not None)
        ),
        "table_entries": sum(len(fact["table_keys"]) for fact in facts),
        "text_entries": sum(len(fact["text_keys"]) for fact in facts),
        "malformed_keys": sum(fact["malformed_keys"] for fact in facts),
        "keys_out_of_range": sum(fact["out_of_range_keys"] for fact in facts),
        "examples_with_table_evidence_only": sum(
            bool(f["table_keys"]) and not f["text_keys"] for f in facts
        ),
        "examples_with_text_evidence_only": sum(
            bool(f["text_keys"]) and not f["table_keys"] for f in facts
        ),
        "examples_with_both": sum(bool(f["table_keys"]) and bool(f["text_keys"]) for f in facts),
        "examples_where_ann_rows_disagree_with_gold_inds": sum(
            ann_disagrees(example, fact) for example, fact in pairs
        ),
        "examples_with_empty_ann_rows_but_gold_inds_entries": sum(
            ann_empty(example) and bool(fact["evidence_entries"]) for example, fact in pairs
        ),
    }

    table_facts = [f for f in facts if any(is_table_operation(n) for n in f["operations"])]
    programs = {
        "steps_in_total": sum(len(fact["operations"]) for fact in facts),
        "programs_that_do_not_split": sum(fact["steps"] is None for fact in facts),
        "operations": _by_count(Counter(n for fact in facts for n in fact["operations"])),
        "steps_per_program": _by_number(
            Counter(len(fact["steps"]) for fact in facts if fact["steps"] is not None)
        ),
        "arguments_per_step": _by_number(Counter(n for fact in facts for n in fact["arities"])),
        "argument_kinds": _by_count(Counter(k for fact in facts for k in fact["argument_kinds"])),
        "constants": _by_count(Counter(c for fact in facts for c in fact["constants"])),
        "table_operation_steps": sum(f["rows_found"] + f["rows_missing"] for f in facts),
        "examples_with_a_table_operation": len(table_facts),
        "table_row_names_found_in_the_table": sum(fact["rows_found"] for fact in facts),
        "table_row_names_not_in_the_table": sum(fact["rows_missing"] for fact in facts),
        "table_second_argument": _by_count(
            Counter(k for fact in facts for k in fact["second_arguments"])
        ),
        "examples_with_greater": sum(fact["has_greater"] for fact in facts),
        "examples_where_greater_is_not_the_last_step": sum(f["greater_not_last"] for f in facts),
    }

    forms = Counter(form for fact in facts for form in fact["answer_forms"])
    results = Counter(fact["result_form"] for fact in facts)
    answers = {
        "exe_ans_numeric": results["numeric"],
        "exe_ans_yes": results["yes"],
        "exe_ans_no": results["no"],
        "exe_ans_neither": results["other"],
        "exe_ans_in_scientific_notation": sum(fact["result_scientific"] for fact in facts),
        "answer_empty": forms["empty"],
        "answer_contains_percent_sign": forms["percent"],
        "answer_contains_dollar_sign": forms["dollar"],
        "answer_is_a_plain_number": forms["plain_number"],
        "answer_is_yes_or_no": forms["yes_no"],
        "answer_in_another_form": forms["other"],
        "yes_or_no_answer_without_greater": sum(
            "yes_no" in f["answer_forms"] and not f["has_greater"] for f in facts
        ),
        "greater_examples_whose_answer_differs_from_exe_ans": sum(
            fact["has_greater"] and _text(example.answer).strip().lower() != example.exe_ans
            for example, fact in pairs
        ),
    }

    return {
        "examples": len(examples),
        "columns": columns,
        "identity": identity,
        "duplicates": duplicates,
        "text": text,
        "tables": tables,
        "evidence": evidence,
        "programs": programs,
        "answers": answers,
    }


def spanning_official_splits(examples: Sequence[RawExample]) -> dict[str, dict[str, int]]:
    """How many companies, reports, and pages have examples in more than one official split."""
    result = {}
    for level, key in (
        ("company", lambda e: e.ticker),
        ("report", lambda e: e.report),
        ("page", lambda e: e.filename),
    ):
        splits: dict[Any, set[str]] = {}
        for example in examples:
            if key(example) is not None:
                splits.setdefault(key(example), set()).add(example.official_split)
        result[level] = {
            "groups": len(splits),
            "in_more_than_one_official_split": sum(len(found) > 1 for found in splits.values()),
        }
    return result


def flagged_ids(
    pairs: Sequence[tuple[RawExample, dict[str, Any]]], expected: dict[str, Any]
) -> dict[str, list[str]]:
    """The ids of the examples in each oddity class, in dataset order."""
    operations = set(expected["operations"])
    constants = set(expected["constants"])
    arity = int(expected["arguments_per_step"])
    id_counts = Counter(example.id for example, _ in pairs)
    page_question = Counter((example.filename, example.question) for example, _ in pairs)
    longest = max((len(fact["operations"]) for _, fact in pairs), default=0)

    tests = {
        "id_repeated": lambda e, f: id_counts[e.id] > 1,
        "id_pattern_mismatch": lambda e, f: not f["id_ok"],
        "duplicate_page_and_question": lambda e, f: page_question[(e.filename, e.question)] > 1,
        "answer_empty": lambda e, f: "empty" in f["answer_forms"],
        "answer_other_form": lambda e, f: "other" in f["answer_forms"],
        "evidence_unparsed": lambda e, f: f["evidence_entries"] is None,
        "evidence_empty": lambda e, f: f["evidence_entries"] == 0,
        "evidence_malformed_key": lambda e, f: f["malformed_keys"] > 0,
        "evidence_key_out_of_range": lambda e, f: f["out_of_range_keys"] > 0,
        "program_unparsed": lambda e, f: f["steps"] is None,
        "operation_unknown": lambda e, f: any(n not in operations for n in f["operations"]),
        "constant_unknown": lambda e, f: any(c not in constants for c in f["constants"]),
        "argument_other_kind": lambda e, f: "other" in f["argument_kinds"],
        "argument_count_unexpected": lambda e, f: any(n != arity for n in f["arities"]),
        "percent_literal_argument": lambda e, f: "percent" in f["argument_kinds"],
        "row_name_not_in_table": lambda e, f: f["rows_missing"] > 0,
        "table_second_argument_not_none": lambda e, f: any(
            kind != "none" for kind in f["second_arguments"]
        ),
        "longest_programs": lambda e, f: longest > 0 and len(f["operations"]) == longest,
        "result_scientific_notation": lambda e, f: f["result_scientific"],
        "result_not_numeric_or_yes_no": lambda e, f: f["result_form"] == "other",
        "yes_no_answer_without_greater": lambda e, f: (
            "yes_no" in f["answer_forms"] and not f["has_greater"]
        ),
        "greater_answer_disagrees": lambda e, f: (
            f["has_greater"] and _text(e.answer).strip().lower() != e.exe_ans
        ),
        "greater_not_last_step": lambda e, f: f["greater_not_last"],
    }
    return {
        name: [str(example.id) for example, fact in pairs if tests[name](example, fact)]
        for name in FLAGS
    }


# --------------------------------------------------------------------------- checks


def _check(check_id: str, name: str, problems: list[str], ok_detail: str) -> dict[str, Any]:
    return {
        "id": check_id,
        "name": name,
        "passed": not problems,
        "detail": "; ".join(problems) if problems else ok_detail,
    }


def _set_difference(name: str, seen: Iterable[str], expected: Iterable[str]) -> list[str]:
    seen, expected = set(seen), {str(item) for item in expected}
    problems = []
    if seen - expected:
        problems.append(f"unexpected {name}: {', '.join(sorted(seen - expected))}")
    if expected - seen:
        problems.append(f"expected {name} not seen: {', '.join(sorted(expected - seen))}")
    return problems


def run_checks(report: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare the report with the expected facts in ``config/data.yaml`` (S1 to S10)."""
    total = report["total"]
    programs = total["programs"]
    identity = total["identity"]

    rows = [
        f"{item['official_split']} has {report['by_split'][item['official_split']]['examples']:,} "
        f"rows, expected {item['rows']:,}"
        for item in report["source"]["files"]
        if report["by_split"][item["official_split"]]["examples"] != item["rows"]
    ]

    want = {str(name): str(kind) for name, kind in expected["columns"].items()}
    columns = []
    for split, have in report["column_types"].items():
        if list(have) != list(want) or have != want:
            missing = sorted(set(want) - set(have))
            extra = sorted(set(have) - set(want))
            changed = sorted(n for n in set(have) & set(want) if have[n] != want[n])
            parts = [
                f"{label} {', '.join(names)}"
                for label, names in (("missing", missing), ("extra", extra), ("retyped", changed))
                if names
            ]
            columns.append(f"{split}: {'; '.join(parts) or 'columns are in a different order'}")

    nulls = [
        f"{name}: {counts['null']:,}" for name, counts in total["columns"].items() if counts["null"]
    ]

    arity = str(int(expected["arguments_per_step"]))
    form = []
    if programs["programs_that_do_not_split"]:
        form.append(f"{programs['programs_that_do_not_split']:,} programs do not split")
    other_arity = sum(n for key, n in programs["arguments_per_step"].items() if key != arity)
    if other_arity:
        form.append(f"{other_arity:,} steps do not have {arity} arguments")

    longest = max((int(key) for key in programs["steps_per_program"]), default=0)
    limit = int(expected["max_program_steps"])

    return [
        _check("S1", "Rows per file", rows, f"{total['examples']:,} examples in total"),
        _check("S2", "Columns", columns, f"{len(want)} columns with the expected types"),
        _check("S3", "No null values", [f"nulls in {', '.join(nulls)}"] if nulls else [], "none"),
        _check(
            "S4",
            "Unique ids",
            [f"{identity['repeated_ids']:,} repeated ids"] if identity["repeated_ids"] else [],
            f"{identity['unique_ids']:,} unique ids",
        ),
        _check(
            "S5",
            "Id format",
            [f"{identity['ids_not_matching_the_pattern']:,} ids do not match"]
            if identity["ids_not_matching_the_pattern"]
            else [],
            "every id matches the pattern and extends its filename",
        ),
        _check(
            "S6",
            "Program form",
            form,
            f"{programs['steps_in_total']:,} steps, {arity} arguments each",
        ),
        _check(
            "S7",
            "Operations",
            _set_difference("operations", programs["operations"], expected["operations"]),
            f"{len(programs['operations'])} operations, as expected",
        ),
        _check(
            "S8",
            "Constants",
            _set_difference("constants", programs["constants"], expected["constants"]),
            f"{len(programs['constants'])} constants, as expected",
        ),
        _check(
            "S9",
            "Argument kinds",
            _set_difference(
                "argument kinds", programs["argument_kinds"], expected["argument_kinds"]
            ),
            f"{len(programs['argument_kinds'])} kinds, as expected",
        ),
        _check(
            "S10",
            "Program length",
            [f"longest program has {longest} steps, expected at most {limit}"]
            if longest > limit
            else [],
            f"longest program has {longest} steps",
        ),
    ]


# --------------------------------------------------------------------------- report


def build_report(
    examples: Sequence[RawExample], cfg: DataConfig, columns: dict[str, dict[str, str]]
) -> dict[str, Any]:
    """The whole report as a dict. ``columns`` is ``read_columns(cfg)``: types per file."""
    pairs = [(example, _facts(example)) for example in examples]
    report: dict[str, Any] = {
        "schema": 1,
        "generated_by": REPORT_COMMAND,
        "source": {
            "repo_id": cfg.repo_id,
            "former_repo_id": cfg.former_repo_id,
            "revision": cfg.revision,
            "files": [
                {
                    "official_split": item.official_split,
                    "path": item.path,
                    "sha256": item.sha256,
                    "bytes": item.size,
                    "rows": item.rows,
                }
                for item in cfg.files
            ],
        },
        "column_types": columns,
        "total": profile(pairs),
        "by_split": {
            split: profile([pair for pair in pairs if pair[0].official_split == split])
            for split in cfg.splits
        },
        "spanning_official_splits": spanning_official_splits(examples),
        "flagged": flagged_ids(pairs, cfg.expected),
        "max_ids_listed": cfg.report_max_ids_listed,
    }
    report["checks"] = run_checks(report, cfg.expected)
    return report


def render_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=True) + "\n"


def _number(value: Any, label: str = "") -> str:
    if value is None:
        return "-"
    if label.endswith("year"):  # a year, not a count: no thousands separator
        return str(value)
    if isinstance(value, float):
        return f"{value:,.1f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


# Field names and abbreviations that a plain "underscores to spaces" would mangle.
_LABEL_FIXES = (
    ("gold inds", "`gold_inds`"),
    ("exe ans", "`exe_ans`"),
    ("table ori", "`table_ori`"),
    ("ann rows", "`ann_*` rows"),
    ("non ascii", "non-ASCII"),
    ("json", "JSON"),
)


def _label(key: str) -> str:
    text = key.replace("_", " ")
    for old, new in _LABEL_FIXES:
        text = text.replace(old, new)
    return text[:1].upper() + text[1:]


def _rows(section: dict[str, Any]) -> dict[str, Any]:
    """Flatten one level: ``{"operations": {"add": 3}}`` becomes ``{"Operations: `add`": 3}``."""
    flat: dict[str, Any] = {}
    for key, value in section.items():
        if isinstance(value, dict):
            for inner, number in value.items():
                flat[f"{_label(key)}: `{inner}`"] = number
        else:
            flat[_label(key)] = value
    return flat


def _section_table(report: dict[str, Any], section: str) -> list[str]:
    splits = list(report["by_split"])
    total = _rows(report["total"][section])
    per_split = {split: _rows(report["by_split"][split][section]) for split in splits}
    lines = [
        "| Measure | Total | " + " | ".join(splits) + " |",
        "|---|---:|" + "---:|" * len(splits),
    ]
    for label, value in total.items():
        cells = [_number(per_split[split].get(label, 0), label) for split in splits]
        lines.append(f"| {label} | {_number(value, label)} | " + " | ".join(cells) + " |")
    return lines


SECTIONS = (
    ("identity", "Ids, companies, reports, and pages"),
    ("duplicates", "Duplicates"),
    ("text", "Text"),
    ("tables", "Tables"),
    ("evidence", "Evidence (`gold_inds`)"),
    ("programs", "Programs"),
    ("answers", "Answers"),
)


def render_markdown(report: dict[str, Any]) -> str:
    source = report["source"]
    total = report["total"]
    types = next(iter(report["column_types"].values()), {})
    limit = int(report["max_ids_listed"])
    failed = [check for check in report["checks"] if not check["passed"]]

    lines = [
        "# FinQA schema report",
        "",
        f"Generated by `{report['generated_by']}` from the pinned snapshot. Do not edit by hand.",
        "",
        "This file holds counts and example ids only. It contains no question, passage, table,",
        "or answer text, because any example may be assigned to the test split. The same",
        f"numbers, with every flagged id, are in `{REPORT_STEM}.json`.",
        "",
        "## Source",
        "",
        f"- Repository: `{source['repo_id']}`"
        + (f" (formerly `{source['former_repo_id']}`)" if source["former_repo_id"] else ""),
        f"- Commit: `{source['revision']}`",
        f"- Examples: {total['examples']:,}",
        "",
        "| Official split | File | Rows | Bytes | SHA-256 |",
        "|---|---|---:|---:|---|",
    ]
    lines += [
        f"| {item['official_split']} | `{item['path']}` | {item['rows']:,} | {item['bytes']:,} "
        f"| `{item['sha256']}` |"
        for item in source["files"]
    ]

    lines += [
        "",
        "## Checks",
        "",
        f"{len(report['checks']) - len(failed)} of {len(report['checks'])} passed.",
        "",
        "| Id | Check | Result | Detail |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {check['id']} | {check['name']} | {'PASS' if check['passed'] else 'FAIL'} "
        f"| {check['detail']} |"
        for check in report["checks"]
    ]

    lines += ["", "## Columns", "", "| Column | Type | Null | Empty |", "|---|---|---:|---:|"]
    lines += [
        f"| `{name}` | `{types.get(name, '?')}` | {counts['null']:,} | {counts['empty']:,} |"
        for name, counts in total["columns"].items()
    ]
    lines += [
        "",
        "Empty means a blank string, a list with no items, or `steps` with no step. For",
        "`gold_inds` it means a JSON object with no entries.",
    ]

    for section, title in SECTIONS:
        lines += ["", f"## {title}", ""]
        lines += _section_table(report, section)
        if section == "identity":
            lines += [
                "",
                "Groups with examples in more than one official split:",
                "",
                "| Level | Groups | In more than one official split |",
                "|---|---:|---:|",
            ]
            lines += [
                f"| {_label(level)} | {counts['groups']:,} "
                f"| {counts['in_more_than_one_official_split']:,} |"
                for level, counts in report["spanning_official_splits"].items()
            ]

    lines += [
        "",
        "## Flagged examples",
        "",
        f"Ids are listed here for a class of up to {limit} examples; the JSON file lists all.",
        "",
        "| Class | Meaning | Examples | Ids |",
        "|---|---|---:|---|",
    ]
    for name, ids in report["flagged"].items():
        if not ids:
            shown = "-"
        elif len(ids) <= limit:
            shown = ", ".join(f"`{item}`" for item in ids)
        else:
            shown = "see the JSON file"
        lines.append(f"| `{name}` | {FLAGS[name]} | {len(ids):,} | {shown} |")
    return "\n".join(lines) + "\n"


def report_paths(report_dir: Path | str) -> tuple[Path, Path]:
    report_dir = Path(report_dir)
    return report_dir / f"{REPORT_STEM}.md", report_dir / f"{REPORT_STEM}.json"


def write_report(report: dict[str, Any], report_dir: Path | str) -> tuple[Path, Path]:
    """Write the Markdown and JSON files and return their paths."""
    markdown_path, json_path = report_paths(report_dir)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    json_path.write_text(render_json(report), encoding="utf-8", newline="\n")
    return markdown_path, json_path


def stale_files(report: dict[str, Any], report_dir: Path | str) -> list[str]:
    """Names of the report files that are missing or differ from a fresh rendering.

    Compared as text, so a checkout that converts line endings still matches.
    """
    stale = []
    markdown_path, json_path = report_paths(report_dir)
    for path, fresh in ((markdown_path, render_markdown(report)), (json_path, render_json(report))):
        if not path.is_file() or path.read_text(encoding="utf-8") != fresh:
            stale.append(path.name)
    return stale
