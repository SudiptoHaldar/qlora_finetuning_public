"""Tests for the report's tables, sentences, template, check R8, and files (req_005 v5.03).

The inputs are the invented ones of ``tests/comparison/report_helpers.py``.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import re
from pathlib import Path

import pytest

from qlora_finetuning.comparison import (
    chart,
    outcomes,
    report_build,
    report_config,
    report_examples,
    report_record,
    report_tables,
    statistics,
)
from qlora_finetuning.comparison.report_build import (
    END,
    START,
    TEMPLATE,
    build_outputs,
    fill,
    merge_readme,
    readme_block,
    write_outputs,
    write_readme,
)
from qlora_finetuning.comparison.report_record import build_statistics
from qlora_finetuning.comparison.report_tables import (
    findings,
    pair_sentence,
    placeholders,
    render_statistics,
    shown_p,
    signed,
    table,
)
from qlora_finetuning.settings import ConfigError
from tests.comparison import report_helpers as r

NEW_MODULES = (
    chart,
    outcomes,
    report_build,
    report_config,
    report_examples,
    report_record,
    report_tables,
    statistics,
)


@pytest.fixture
def cfg(tmp_path):
    return r.report_cfg(tmp_path)


@pytest.fixture
def record(cfg):
    return build_statistics(r.inputs(), cfg)


def built(record, cfg, tmp_path, **options):
    settings = {"test_ids": r.IDS}
    return build_outputs(
        record,
        options.pop("cost", r.cost_record()),
        r.DEVIATIONS,
        cfg,
        tmp_path / "reports" / "comparison",
        **(settings | options),
    )


# --------------------------------------------------------------------------- the build


def test_a_build_fills_the_report_and_passes_r8(record, cfg, tmp_path):
    final, outputs = built(record, cfg, tmp_path)
    assert final["passed"] and r.check(final, "R8")["passed"]
    assert "placeholders filled" in r.check(final, "R8")["detail"]
    report = outputs.report[1]
    assert report.startswith("# The comparison (provisional)\n")
    assert "{{" not in report and "}}" not in report
    assert "| A1 | Qwen3-4B with QLoRA | 70.00% |" in report
    assert "An invented deviation." in report
    assert "per 1,000 questions](comparison/accuracy_cost.svg)" in report
    assert outputs.report[0] == cfg.report_file
    assert outputs.chart[0] == tmp_path / "reports" / "comparison" / "accuracy_cost.svg"
    assert '"id": "R8"' in outputs.statistics_json[1]


def test_a_reconciled_cost_record_gives_a_report_that_is_not_provisional(cfg, tmp_path):
    cost = r.cost_record(final=True)
    final, outputs = built(build_statistics(r.inputs(cost=cost), cfg), cfg, tmp_path, cost=cost)
    assert not final["provisional"]
    assert outputs.report[1].startswith("# The comparison\n")
    assert "reconciled with the AWS bill" in outputs.report[1]
    assert outputs.readme_block[1].splitlines()[1] == "## Results"
    assert outputs.statistics_markdown[1].startswith("# The comparison's statistics\n")


def test_no_output_holds_a_reply_a_resource_name_or_a_stray_placeholder(record, cfg, tmp_path):
    _, outputs = built(record, cfg, tmp_path)
    for _, text in dataclasses.astuple(outputs):
        assert "PROGRAM" not in text and "ANSWER" not in text
        assert "arn:aws" not in text and "{{" not in text


def test_the_same_record_gives_the_same_bytes(record, cfg, tmp_path):
    assert built(record, cfg, tmp_path)[1] == built(record, cfg, tmp_path)[1]


def test_a_failed_record_is_not_built_and_cannot_be_written(cfg, tmp_path):
    failed = build_statistics(r.inputs(cost=None), cfg)
    final, outputs = built(failed, cfg, tmp_path)
    assert outputs is None and not final["passed"]
    assert r.check(final, "R8")["detail"] == "not built: an earlier check failed"
    with pytest.raises(ConfigError, match="only when R1 to R8 pass"):
        write_outputs(final, outputs)
    assert not (tmp_path / "reports").exists()
    assert "Checks" in render_statistics(final, cfg) and "## The tested pairs" not in (
        render_statistics(final, cfg)
    )


# --------------------------------------------------------------------------- R8


def test_r8_fails_for_a_placeholder_without_a_value_and_a_value_without_a_place(
    record, cfg, tmp_path
):
    final, _ = built(record, cfg, tmp_path, template="# {{title}}\n\n{{nothing_like_this}}\n")
    detail = r.check(final, "R8")["detail"]
    assert "the template names {{nothing_like_this}}, which has no value" in detail
    assert "the value summary_table is used nowhere in the template" in detail
    assert not final["passed"]


def test_r8_fails_for_a_resource_name_or_an_account_number(record, cfg, tmp_path):
    named = copy.deepcopy(record)
    named["arms"]["B1-Pro"]["label"] = "arn:aws:bedrock:us-east-1:custom-model-deployment/x"
    detail = r.check(built(named, cfg, tmp_path)[0], "R8")["detail"]
    assert "the report holds an AWS resource name" in detail
    assert "the statistics record holds an AWS resource name" in detail

    numbered = copy.deepcopy(record)
    numbered["arms"]["B1-Pro"]["label"] = "the model of 123456789012"
    detail = r.check(built(numbered, cfg, tmp_path)[0], "R8")["detail"]
    assert "holds what looks like an account number" in detail


def test_r8_leaves_run_ids_and_hashes_alone(record, cfg, tmp_path):
    final, _ = built(record, cfg, tmp_path)
    assert "20261101T000000Z" in final["arms"]["A1"]["run_id"] and final["passed"]


def test_r8_fails_for_an_id_that_is_not_a_test_example(record, cfg, tmp_path):
    final, _ = built(record, cfg, tmp_path, test_ids=r.IDS[:900])
    assert "example ids are not test examples" in r.check(final, "R8")["detail"]


def test_r8_fails_for_an_address_in_the_record(record, cfg, tmp_path):
    with_address = copy.deepcopy(record)
    with_address["arms"]["A1"]["label"] = "ask someone@example.com"
    assert "email address" in r.check(built(with_address, cfg, tmp_path)[0], "R8")["detail"]


# --------------------------------------------------------------------------- the template


def test_the_template_holds_no_percentage_and_no_dollar_amount():
    text = TEMPLATE.read_text(encoding="utf-8")
    assert not re.findall(r"\d\s*%", text)
    assert not re.findall(r"\$\s*\d", text)
    counts = re.findall(r"\d{1,3},\d{3}(?! questions)", text)  # "per 1,000 questions" is a unit
    assert not counts


def test_every_placeholder_of_the_template_has_a_value_and_every_value_a_place(record, cfg):
    values = placeholders(record, r.cost_record(), r.DEVIATIONS, cfg, "chart.svg")
    _, problems = fill(TEMPLATE.read_text(encoding="utf-8"), values)
    assert problems == []


def test_fill_replaces_names_and_keeps_what_it_cannot_fill():
    text, problems = fill("{{a}} and {{b}}", {"a": "one", "c": "three"})
    assert text == "one and {{b}}"
    assert problems == [
        "the template names {{b}}, which has no value",
        "the value c is used nowhere in the template",
    ]


# --------------------------------------------------------------------------- sentences


def test_a_settled_pair_is_said_to_hold_under_both_intervals(record, cfg):
    sentence = pair_sentence(r.pair(record, "A1", "A0"), record, cfg)
    assert sentence.startswith("A1 is ahead of A0 by 69.00 points (interval by example ")
    assert sentence.endswith("The difference is settled: it holds under both intervals.")


def test_the_figures_point_the_way_the_sentence_does(record, cfg):
    pair = r.pair(record, "A1", "B1-Pro")  # A1 minus B1-Pro is below zero
    sentence = pair_sentence(pair, record, cfg)
    assert sentence.startswith("B1-Pro is ahead of A1 by 5.00 points")
    low = pair["interval"]["high"].lstrip("-")
    high = pair["interval"]["low"].lstrip("-")
    assert f"interval by example {low} to {high}" in sentence and "-" not in sentence.split("(")[1]


def test_a_pair_that_is_not_settled_says_why(cfg):
    lite = r.outcomes("B1-Lite", right=range(0, 660))
    arms = r.all_outcomes(**{"B1-Lite": lite})
    record = build_statistics(r.inputs(arms=arms, six_arm=r.six_arm(arms)), cfg)
    sentence = pair_sentence(r.pair(record, "A1", "B1-Lite"), record, cfg)
    assert "The difference is **not settled**: the interval by company includes zero." in sentence

    same = r.outcomes("B1-Lite", right=range(0, 700))
    arms = r.all_outcomes(**{"B1-Lite": same})
    record = build_statistics(r.inputs(arms=arms, six_arm=r.six_arm(arms)), cfg)
    sentence = pair_sentence(r.pair(record, "A1", "B1-Lite"), record, cfg)
    assert sentence.startswith("A1 and B1-Lite have the same execution accuracy")


def test_the_findings_follow_the_figures(record, cfg):
    lines = findings(record, r.cost_record(), cfg)
    assert lines[0].startswith("**Tuning is the large effect.** It raised execution accuracy")
    assert "in all 3 models, by 35.00 to 69.00 points" in lines[0]
    assert lines[1] == (
        "**The tuned arms, by execution accuracy:** B1-Pro 75.00%, A1 70.00% and B1-Lite 65.00%."
    )
    assert any(line.startswith("**Cost per 1,000 questions, tuned arms:**") for line in lines)
    assert any("batch job, a month" in line for line in lines)


def test_tuning_is_not_called_the_large_effect_when_a_gain_is_not_settled(cfg):
    weak = r.outcomes("B1-Lite", right=range(0, 203))  # three more than untuned Nova Lite
    arms = r.all_outcomes(**{"B1-Lite": weak})
    record = build_statistics(r.inputs(arms=arms, six_arm=r.six_arm(arms)), cfg)
    lines = findings(record, r.cost_record(), cfg)
    assert not any("large effect" in line for line in lines)
    assert sum(line.startswith("**Tuning.**") for line in lines) == 3


def test_the_always_on_finding_depends_on_what_can_be_reached(record, cfg):
    cost = r.cost_record()
    assert all(not e["reachable"] for e in cost["break_evens"] if e["scenario"] == "always_on")
    lines = findings(record, cost, cfg)
    assert any("does not pay off in this account" in line for line in lines)

    reachable = copy.deepcopy(cost)
    for entry in reachable["break_evens"]:
        entry["reachable"] = True
    lines = findings(record, reachable, cfg)
    assert not any("does not pay off" in line for line in lines)
    assert any("always-on endpoint:**" in line for line in lines)
    assert not any("not reachable in this account" in line for line in lines)


def test_small_formatters():
    assert signed("1.79") == "+1.79" and signed("-2.00") == "-2.00" and signed("0.00") == "0.00"
    smallest = r.report_cfg(Path(".")).smallest_p
    assert shown_p("3.57e-3", smallest) == "0.0036"
    assert shown_p("1.21e-625", smallest) == "below 0.0001"
    assert table(["a", "b"], [["x|y", 1]], right=(1,)) == "| a | b |\n|---|---:|\n| x\\|y | 1 |"


# --------------------------------------------------------------------------- the README


BLOCK = f"{START}\n## Results\n\nnew\n{END}\n"


def test_the_block_replaces_only_what_is_between_the_markers():
    existing = f"# Title\n\nbefore\n\n{START}\nold\nolder\n{END}\n\nafter\n"
    merged = merge_readme(existing, BLOCK)
    assert merged == f"# Title\n\nbefore\n\n{START}\n## Results\n\nnew\n{END}\n\nafter\n"
    assert merge_readme(merged, BLOCK) == merged


def test_a_readme_without_markers_gets_the_block_once():
    merged = merge_readme("# Title\n\ntext", BLOCK)
    assert merged == f"# Title\n\ntext\n\n{BLOCK}"
    assert merge_readme(merged, BLOCK) == merged
    assert merge_readme("", BLOCK) == BLOCK


def test_one_marker_without_the_other_is_refused():
    with pytest.raises(ConfigError, match="one comparison marker without the other"):
        merge_readme(f"# Title\n{START}\n", BLOCK)


def test_the_readme_keeps_its_byte_order_mark_and_its_line_ends(tmp_path):
    path = tmp_path / "README.md"
    path.write_bytes("﻿# Title\r\n\r\ntext\r\n".encode())
    write_readme(path, BLOCK)
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf# Title\r\n")
    assert b"\r\n## Results\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")
    write_readme(path, BLOCK)
    assert path.read_bytes() == raw


def test_the_block_links_to_the_report_and_names_its_command(record, cfg):
    block = readme_block(record, r.cost_record(), cfg)
    assert block.startswith(f"{START}\n## Results (provisional)\n") and block.endswith(f"{END}\n")
    assert "[reports/comparison_report.md](reports/comparison_report.md)" in block
    assert "python -m scripts.comparison_report build" in block
    assert block.count("\n- **") == 3


# --------------------------------------------------------------------------- writing


def test_every_file_is_written_and_a_second_build_changes_no_byte(record, cfg, tmp_path):
    final, outputs = built(record, cfg, tmp_path)
    written = write_outputs(final, outputs)
    folder = tmp_path / "reports" / "comparison"
    assert written == [
        folder / "comparison_statistics.json",
        folder / "comparison_statistics.md",
        folder / "accuracy_cost.svg",
        tmp_path / "reports" / "comparison_report.md",
        tmp_path / "README.md",
    ]
    before = {path: path.read_bytes() for path in written}
    assert all(b"\r\n" not in content for content in before.values())
    again, outputs = built(build_statistics(r.inputs(), cfg), cfg, tmp_path)
    write_outputs(again, outputs)
    assert {path: path.read_bytes() for path in written} == before


# --------------------------------------------------------------------------- guards


def _imports(module) -> set[str]:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_new_module_imports_aws_a_plotting_library_or_the_training_stack():
    banned = {"boto3", "botocore", "matplotlib", "scipy", "statsmodels", "torch", "transformers"}
    for module in NEW_MODULES:
        assert not _imports(module) & banned, module.__name__


def test_only_the_examples_module_touches_text():
    """What feeds a committed file never reads a reply, a question, or a split."""
    for module in (chart, report_build, report_config, report_record, report_tables, statistics):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "load_split" not in source and ".reply" not in source, module.__name__
        assert ".question" not in source and "run_predictions" not in source, module.__name__
    source = Path(outcomes.__file__).read_text(encoding="utf-8")
    assert ".reply" not in source and ".question" not in source and "load_split" not in source
