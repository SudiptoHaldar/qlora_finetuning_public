"""Tests for qlora_finetuning.frontier.baseline_report, on runs made by the real runner."""

from __future__ import annotations

import dataclasses
import json
from decimal import Decimal

import pytest

from qlora_finetuning.data import manifest_hash
from qlora_finetuning.evaluation.fake import gold_predictor
from qlora_finetuning.frontier.baseline_report import (
    build_report,
    load_qwen_prompt_tokens,
    render_markdown,
    report_paths,
    write_report,
)
from qlora_finetuning.frontier.pricing import load_pricing
from qlora_finetuning.settings import REPO_ROOT, ConfigError
from tests.evaluation.conftest import evaluate, ledger_records, records_for
from tests.frontier.conftest import (
    INPUT_TOKENS,
    OUTPUT_TOKENS,
    QWEN_PROMPT_TOKENS,
    baseline_run,
    make_predictor,
)


def report_for(summaries, context, cfg, **changes):
    arguments = {
        "prompt_id": context.prompt.prompt_id,
        "manifest_hash": manifest_hash(context.split_cfg),
    }
    arguments.update(changes)
    return build_report(
        summaries, ledger_records(context), cfg, load_pricing(), QWEN_PROMPT_TOKENS, **arguments
    )


def check(report, check_id):
    return next(c for c in report["checks"] if c["id"] == check_id)


def failed(report):
    return [c["id"] for c in report["checks"] if not c["passed"]]


# --------------------------------------------------------------------------- the good case


def test_four_baseline_runs_pass_every_check(four_runs, context, cfg):
    report = report_for(four_runs, context, cfg)
    assert report["passed"] and failed(report) == []
    assert [c["id"] for c in report["checks"]] == ["B1", "B2", "B3", "B4"]
    assert check(report, "B1")["name"] == "4 results"
    assert [(r["arm"], r["split"]) for r in report["runs"]] == [
        ("B0-Pro", "quick_dev"),
        ("B0-Lite", "quick_dev"),
        ("B0-Pro", "dev"),
        ("B0-Lite", "dev"),
    ]
    assert report["inputs"]["price_list"]["published"] == "2026-11-11"
    assert report["inputs"]["settings"] == cfg.settings


def test_runs_are_ordered_whatever_order_they_are_given_in(four_runs, context, cfg):
    report = report_for(list(reversed(four_runs)), context, cfg)
    assert report["passed"]
    assert [r["run_id"] for r in report["runs"]] == [s["run_id"] for s in four_runs]


def test_each_run_reports_scores_tokens_ratio_and_exact_cost(four_runs, context, cfg):
    report = report_for(four_runs, context, cfg)
    pro, lite = report["runs"][2], report["runs"][3]  # dev: 8 examples
    assert (pro["model"], pro["model_id"]) == ("nova-pro", "amazon.nova-pro-v1:0")
    assert pro["examples"] == 8 and pro["result"] is True
    for name in ("format_validity", "execution_accuracy", "answer_accuracy", "program_match"):
        assert pro[name] == {"count": 8, "rate": 1.0}, name
    assert pro["execution_accuracy_tight"] == {"count": 8, "rate": 1.0}
    assert pro["stop_reasons"] == {"end": 8} and pro["error_codes"] == {}
    assert (pro["input_tokens"], pro["output_tokens"]) == (8 * INPUT_TOKENS, 8 * OUTPUT_TOKENS)
    assert pro["qwen_prompt_tokens"] == 80
    assert pro["nova_to_qwen_input_ratio"] == "4.0000"  # 320 / 80

    # 320 input and 96 output tokens: Pro (320 * 0.80 + 96 * 3.20) / 1M, Lite at 0.06 / 0.24.
    assert Decimal(pro["cost_usd"]) == Decimal("0.000563")  # 0.0005632, shown to six places
    assert Decimal(pro["cost_per_1000_questions_usd"]) == Decimal("0.070400")
    assert Decimal(lite["cost_usd"]) == Decimal("0.000042")  # 0.00004224
    assert Decimal(lite["cost_per_1000_questions_usd"]) == Decimal("0.005280")
    assert pro["latency_seconds"]["mean"] == 0.25

    quick = report["runs"][0]
    assert quick["examples"] == 3 and quick["nova_to_qwen_input_ratio"] == "4.0000"  # 120 / 30


def test_wrong_replies_show_up_in_the_counts_not_in_the_checks(context, cfg):
    from tests.frontier.conftest import StubProvider

    wrong = StubProvider(text="The answer is probably 825.")
    summaries = [
        baseline_run(
            context, cfg, name, split, predictor=make_predictor(cfg, name, provider=wrong)
        )
        for split in ("quick_dev", "dev")
        for name in cfg.models
    ]
    report = report_for(summaries, context, cfg)
    assert report["passed"]  # a bad score is a result, not a failed check
    run = report["runs"][2]
    assert run["execution_accuracy"] == {"count": 0, "rate": 0.0}
    assert run["format_validity"]["count"] == 0 and sum(run["error_codes"].values()) == 8


# --------------------------------------------------------------------------- B1


def test_b1_fails_when_a_run_is_missing(four_runs, context, cfg):
    report = report_for(four_runs[:3], context, cfg)
    assert failed(report) == ["B1"] and not report["passed"]
    assert "nova-lite on dev: 0 runs, expected 1" in check(report, "B1")["detail"]


def test_b1_fails_when_a_model_and_split_has_two_runs(four_runs, context, cfg):
    again = baseline_run(context, cfg, "nova-pro", "dev")
    report = report_for([*four_runs, again], context, cfg)
    assert failed(report) == ["B1"]
    assert "nova-pro on dev: 2 runs, expected 1" in check(report, "B1")["detail"]


def test_b1_fails_for_a_partial_run(four_runs, context, cfg):
    partial = baseline_run(context, cfg, "nova-lite", "dev", limit=4)
    report = report_for([*four_runs[:3], partial], context, cfg)
    assert failed(report) == ["B1"]
    assert f"run {partial['run_id']} is not a result" in check(report, "B1")["detail"]


def test_b1_fails_for_a_run_recorded_under_another_arm(four_runs, context, cfg):
    mislabelled = baseline_run(context, cfg, "nova-lite", "dev", arm="B1-Lite")
    report = report_for([*four_runs[:3], mislabelled], context, cfg)
    assert failed(report) == ["B1"]
    assert "is arm B1-Lite, expected B0-Lite" in check(report, "B1")["detail"]


def test_b1_and_b3_fail_for_a_run_that_is_not_a_nova_baseline(four_runs, context, cfg):
    fake = evaluate(gold_predictor(records_for("dev", context)), "dev", context)
    report = report_for([*four_runs, fake], context, cfg)
    assert failed(report) == ["B1", "B2", "B3"]  # another predictor, other settings, no tokens
    assert "(gold on dev) is not a baseline of a configured model" in check(report, "B1")["detail"]
    fit = baseline_run(context, cfg, "nova-pro", "fit")
    assert "(nova-pro on fit) is not a baseline" in check(
        report_for([*four_runs, fit], context, cfg), "B1"
    )["detail"]


# --------------------------------------------------------------------------- B2


def test_b2_fails_for_another_prompt_or_split(four_runs, context, cfg):
    report = report_for(four_runs, context, cfg, prompt_id="f" * 64)
    assert failed(report) == ["B2"]
    assert "made with another prompt wording" in check(report, "B2")["detail"]
    report = report_for(four_runs, context, cfg, manifest_hash="0" * 64)
    assert failed(report) == ["B2"]
    assert "made with another split manifest" in check(report, "B2")["detail"]


def test_b2_and_b4_fail_when_one_run_used_other_settings(four_runs, context, cfg):
    warmer = make_predictor(cfg, "nova-lite", settings=dict(cfg.settings, temperature=0.7))
    other = baseline_run(context, cfg, "nova-lite", "dev", predictor=warmer)
    report = report_for([*four_runs[:3], other], context, cfg)
    assert failed(report) == ["B2", "B4"]
    assert "different settings" in check(report, "B2")["detail"]
    assert f"run {other['run_id']} used settings other than the config's" in (
        check(report, "B4")["detail"]
    )


def test_b2_fails_when_reply_limits_differ(four_runs, context, cfg):
    shorter = dataclasses.replace(context, max_new_tokens=64)
    other = baseline_run(shorter, cfg, "nova-lite", "dev")
    report = report_for([*four_runs[:3], other], context, cfg)
    assert failed(report) == ["B2"]
    assert "reply limits differ: 192, 64" in check(report, "B2")["detail"]


# --------------------------------------------------------------------------- B3 and B4


def test_b3_fails_when_the_token_counts_are_not_from_aws(four_runs, context, cfg):
    measured = make_predictor(cfg, "nova-lite")
    measured.usage_source = "measured"
    other = baseline_run(context, cfg, "nova-lite", "dev", predictor=measured)
    report = report_for([*four_runs[:3], other], context, cfg)
    assert failed(report) == ["B3"]
    assert f"run {other['run_id']} has a ledger record that is not aws_api tokens" in (
        check(report, "B3")["detail"]
    )


def test_b3_fails_when_a_run_has_no_ledger_record(four_runs, context, cfg):
    kept = [r for r in ledger_records(context) if r.run_id != four_runs[0]["run_id"]]
    report = build_report(
        four_runs,
        kept,
        cfg,
        load_pricing(),
        QWEN_PROMPT_TOKENS,
        prompt_id=context.prompt.prompt_id,
        manifest_hash=manifest_hash(context.split_cfg),
    )
    assert failed(report) == ["B3"]
    assert f"run {four_runs[0]['run_id']} has no ledger record" in check(report, "B3")["detail"]


def test_b4_fails_when_a_run_used_another_model_id(four_runs, context, cfg):
    other_id = make_predictor(cfg, "nova-pro", model_id="us.amazon.nova-pro-v1:0")
    other = baseline_run(context, cfg, "nova-pro", "dev", predictor=other_id)
    report = report_for([*four_runs[:2], other, four_runs[3]], context, cfg)
    assert failed(report) == ["B4"]
    assert "used us.amazon.nova-pro-v1:0, config says amazon.nova-pro-v1:0" in (
        check(report, "B4")["detail"]
    )
    # No price for that id: the cost is left empty, never guessed from the base model.
    run = next(r for r in report["runs"] if r["run_id"] == other["run_id"])
    assert run["cost_usd"] is None and run["cost_per_1000_questions_usd"] is None


# --------------------------------------------------------------------------- files


def test_the_files_hold_counts_and_no_reply_or_example_text(four_runs, context, cfg):
    report = report_for(four_runs, context, cfg)
    markdown_path, json_path = write_report(report, cfg.report_dir)
    assert (markdown_path, json_path) == report_paths(cfg.report_dir)
    assert markdown_path.parent == context.tracking_cfg.root / "reports" / "frontier"
    markdown, stored = markdown_path.read_text(encoding="utf-8"), json_path.read_text("utf-8")
    assert json.loads(stored) == report and markdown == render_markdown(report)
    for text in (markdown, stored):
        assert text.isascii() and "\r" not in text
        assert "subtract" not in text and "PROGRAM" not in text and "zebrafish" not in text
        for record in records_for("dev", context)[:2]:
            assert record.user not in text
    assert "4 of 4 passed." in markdown
    assert "| B0-Pro | dev | 8 | 8 (100.0%) | 8 (100.0%) |" in markdown
    assert "| B0-Lite | dev | 320 | 96 | 80 | 4.0000 | $0.0000 | $0.0053 |" in markdown
    assert "| B0-Pro | dev | 320 | 96 | 80 | 4.0000 | $0.0006 | $0.0704 |" in markdown
    assert "published 2026-11-11, read 2026-11-16" in markdown
    for run in four_runs:
        assert f"`{run['run_id']}`" in markdown


def test_a_failed_report_renders_its_fail_line(four_runs, context, cfg):
    markdown = render_markdown(report_for(four_runs[:3], context, cfg))
    assert "3 of 4 passed." in markdown
    assert "| B1 | 4 results | FAIL | nova-lite on dev: 0 runs, expected 1 |" in markdown


# --------------------------------------------------------------------------- the Qwen counts


def test_the_committed_token_budget_report_gives_the_qwen_prompt_tokens():
    tokens = load_qwen_prompt_tokens(REPO_ROOT / "reports/data/finqa_token_budget_report.json")
    assert tokens["quick_dev"] == 128_495 and tokens["dev"] == 317_882


def test_a_missing_or_damaged_token_budget_report_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="token budget report not found"):
        load_qwen_prompt_tokens(tmp_path / "absent.json")
    damaged = tmp_path / "report.json"
    for content in ("{not json", '{"splits": {"dev": {}}}', '{"splits": []}', "[]"):
        damaged.write_text(content, encoding="utf-8")
        with pytest.raises(ConfigError, match="token budget report cannot be read"):
            load_qwen_prompt_tokens(damaged)


def test_without_a_qwen_count_the_ratio_is_left_empty(four_runs, context, cfg):
    report = build_report(
        four_runs,
        ledger_records(context),
        cfg,
        load_pricing(),
        {},
        prompt_id=context.prompt.prompt_id,
        manifest_hash=manifest_hash(context.split_cfg),
    )
    assert report["passed"]
    assert {r["nova_to_qwen_input_ratio"] for r in report["runs"]} == {None}
    assert "| B0-Pro | dev | 320 | 96 | - | - | $0.0006 |" in render_markdown(report)
