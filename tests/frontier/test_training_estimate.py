"""Tests for qlora_finetuning.frontier.training_estimate: the ratio, the estimate, the gate."""

from __future__ import annotations

import dataclasses
import json
from decimal import Decimal

import pytest

from qlora_finetuning.evaluation.fake import gold_predictor
from qlora_finetuning.frontier.baseline_report import report_paths as baseline_paths
from qlora_finetuning.frontier.export import export_files
from qlora_finetuning.frontier.export_validation import validate_export
from qlora_finetuning.frontier.pricing import load_pricing
from qlora_finetuning.frontier.training_estimate import (
    TokenRatio,
    build_gate_record,
    estimate_path_b,
    gate_checks,
    load_token_ratio,
    read_gate_record,
    render_markdown,
    report_paths,
    staged_before,
    stale_files,
    write_gate_record,
)
from qlora_finetuning.settings import ConfigError
from tests.evaluation.conftest import evaluate, ledger_records, records_for
from tests.frontier.conftest import baseline_run, make_predictor
from tests.frontier.export_helpers import (
    detail,
    failed,
    volumes_for,
    write_baseline_report,
)

PRO, LITE = "amazon.nova-pro-v1:0", "amazon.nova-lite-v1:0"
MEASURED = TokenRatio(Decimal("1.25"), True, "measured in a test")


def ids(context):
    from qlora_finetuning.data import manifest_hash

    return context.prompt.prompt_id, manifest_hash(context.split_cfg)


def estimate(cfg, context, ratio=MEASURED, records=(), **options):
    arguments = {"test_examples": 22, "max_new_tokens": 192}
    arguments.update(options)
    return estimate_path_b(
        cfg, load_pricing(), volumes_for(context), ratio, list(records), **arguments
    )


# --------------------------------------------------------------------------- the ratio


def test_the_largest_measured_ratio_is_taken(cfg, context):
    write_baseline_report(cfg, context)
    ratio = load_token_ratio(cfg, *ids(context))
    assert ratio.measured and ratio.value == Decimal("1.2500")
    assert ratio.source == "the largest of the 4 ratios measured in nova_baseline_report.json"


def test_without_a_report_the_ratio_is_the_assumed_one_and_says_so(cfg, context):
    ratio = load_token_ratio(cfg, *ids(context))
    assert (ratio.measured, ratio.value) == (False, Decimal("1.5"))
    assert ratio.source.startswith("not measured: no nova_baseline_report.json; run the v4.01")


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"passed": False}, "did not pass its checks"),
        ({"passed": "yes"}, "did not pass its checks"),
        ({"inputs": {"prompt_id": "f" * 64, "split_content_hash": "x"}}, "another prompt wording"),
        ({"runs": []}, "has no usable token ratio"),
        ({"runs": [{"nova_to_qwen_input_ratio": None}]}, "cannot be read or has a run without"),
        ({"runs": [{"nova_to_qwen_input_ratio": "0"}]}, "has no usable token ratio"),
        ({"runs": [{"arm": "B0-Pro"}]}, "cannot be read or has a run without"),
        ({"runs": "none"}, "cannot be read or has a run without"),
    ],
)
def test_a_report_that_cannot_be_trusted_gives_the_assumed_ratio(cfg, context, changes, reason):
    write_baseline_report(cfg, context, **changes)
    ratio = load_token_ratio(cfg, *ids(context))
    assert (ratio.measured, ratio.value) == (False, cfg.assumed_token_ratio)
    assert reason in ratio.source


def test_a_report_for_another_split_or_a_damaged_file_is_not_measured(cfg, context):
    prompt_id, _ = ids(context)
    write_baseline_report(cfg, context)
    assert "another split manifest" in load_token_ratio(cfg, prompt_id, "0" * 64).source
    baseline_paths(cfg.report_dir)[1].write_text("{not json", encoding="utf-8")
    assert "cannot be read" in load_token_ratio(cfg, *ids(context)).source


# --------------------------------------------------------------------------- the estimate


def test_the_estimate_is_exact_decimal_arithmetic(cfg, context):
    result = estimate(cfg, context)
    # 30,000 Qwen tokens * 1.25 = 37,500 Nova tokens an epoch; 2 epochs.
    assert result["training_tokens_one_epoch"] == 37_500
    assert result["training_tokens"] == {PRO: 75_000, LITE: 75_000}
    assert result["training_usd"] == {PRO: "0.6", LITE: "0.15"}  # $8 and $2 per 1M
    assert result["training_total_usd"] == "0.75"
    assert result["training_with_margin_usd"] == "0.825"  # 10% more
    assert result["training_total_by_epochs_usd"] == {
        "1": "0.375", "2": "0.75", "3": "1.125", "4": "1.5", "5": "1.875",
    }  # fmt: skip
    # Evaluation to come, per model: dev once and test twice. Input (7,200 + 2 * 22 * 900)
    # * 1.25 = 58,500 tokens; output (8 + 44) replies of 192 tokens = 9,984.
    assert result["evaluation_to_come"]["input_tokens_per_model"] == 58_500
    assert result["evaluation_to_come"]["output_tokens_per_model"] == 9_984
    assert result["evaluation_to_come_usd"] == {PRO: "0.0787488", LITE: "0.00590616"}
    assert result["evaluation_to_come_total_usd"] == "0.08465496"
    assert result["storage_usd"] == {PRO: "1.95", LITE: "1.95"}
    assert result["storage_total_usd"] == "3.9"
    assert (result["spent_usd"], result["spent_ledger_records"]) == ("0", 0)
    assert result["total_usd"] == "4.80965496"  # 0.825 + 0.08465496 + 3.9
    assert result["cap_usd"] == "200"
    assert (result["token_ratio"], result["ratio_measured"]) == ("1.25", True)
    assert json.loads(json.dumps(result)) == result  # plain values only


def test_epochs_margin_and_storage_months_come_from_the_config(cfg, context):
    other = dataclasses.replace(
        cfg, epoch_count=3, margin_percent=Decimal("0"), storage_months=0
    )
    result = estimate(other, context)
    assert result["training_total_usd"] == result["training_with_margin_usd"] == "1.125"
    assert result["storage_total_usd"] == "0" and result["epoch_count"] == 3


def test_a_ratio_that_gives_a_fraction_of_a_token_is_rounded_up(cfg, context):
    result = estimate(cfg, context, ratio=TokenRatio(Decimal("1.00001"), True, "x"))
    assert result["training_tokens_one_epoch"] == 30_001  # 30,000.3 tokens


def test_the_ledgers_path_b_tokens_are_counted_once_at_list_price(four_runs, cfg, context):
    # A run of another path, and one whose model has no price, are not in the amount.
    evaluate(gold_predictor(records_for("dev", context)), "dev", context)
    foreign = make_predictor(cfg, "nova-pro", model_id="arn:aws:bedrock:us-east-1:1:deployment/x")
    baseline_run(context, cfg, "nova-pro", "fit", predictor=foreign)
    result = estimate(cfg, context, records=ledger_records(context))
    # Per model 11 examples (3 + 8) of 40 input and 12 output tokens.
    assert result["spent_usd"] == "0.00083248"  # 0.0007744 + 0.00005808
    assert result["spent_ledger_records"] == 4 and result["unpriced_ledger_records"] == 1
    assert Decimal(result["total_usd"]) == Decimal("4.80965496") + Decimal("0.00083248")


def test_a_token_report_without_fit_or_dev_is_a_config_error(cfg, context):
    volumes = dataclasses.replace(volumes_for(context), splits={})
    with pytest.raises(ConfigError, match="the token budget report has no fit split"):
        estimate_path_b(
            cfg, load_pricing(), volumes, MEASURED, [], test_examples=22, max_new_tokens=192
        )


# --------------------------------------------------------------------------- the gate


def test_g1_passes_only_for_a_measured_ratio(cfg, context):
    checks = gate_checks(estimate(cfg, context), cfg)
    assert [item["id"] for item in checks] == ["G1", "G2"] and failed(checks) == []
    assert detail(checks, "G1") == "Nova counts 1.25 tokens per Qwen token: measured in a test"

    assumed = TokenRatio(Decimal("1.5"), False, "not measured: no report")
    checks = gate_checks(estimate(cfg, context, ratio=assumed), cfg)
    assert failed(checks) == ["G1"] and detail(checks, "G1") == "not measured: no report"


def test_g2_passes_at_the_cap_and_fails_one_cent_above_it(cfg, context):
    result = estimate(cfg, context)
    total = Decimal(result["total_usd"])
    at_cap = dataclasses.replace(cfg, path_b_cap_usd=total)
    assert failed(gate_checks(result, at_cap)) == []
    below = dataclasses.replace(cfg, path_b_cap_usd=total - Decimal("0.01"))
    checks = gate_checks(result, below)
    assert failed(checks) == ["G2"]
    assert detail(checks, "G2") == "the Path B estimate is $4.81, above the cap of $4.80"


# --------------------------------------------------------------------------- the record


def record_for(cfg, context, ratio=MEASURED, staged=None):
    export, _ = export_files(cfg, context)
    checks = validate_export(export, cfg, context, volumes_for(context), ratio.value)
    result = estimate(cfg, context, ratio=ratio)
    checks += gate_checks(result, cfg)
    return export, build_gate_record(export, checks, result, cfg, load_pricing(), staged)


def test_the_record_passes_when_every_check_does(cfg, context):
    export, record = record_for(cfg, context)
    assert record["passed"] is True and len(record["checks"]) == 9
    assert record["export_id"] == export.export_id
    training = record["files"]["training"]
    assert training == {
        "name": "finqa_fit.jsonl",
        "split": "fit",
        "records": 30,
        "bytes": export.files["training"].size,
        "sha256": export.files["training"].sha256,
        "repeated_lines": export.files["training"].repeated_lines,
    }
    assert record["inputs"]["epoch_count"] == 2 and record["inputs"]["cap_usd"] == "200"
    assert record["inputs"]["price_list"]["published"] == "2026-11-11"
    assert record["estimate"]["training_usd"] == {PRO: "0.6", LITE: "0.15"}
    assert record["staged"] is None


def test_the_record_does_not_pass_when_any_check_fails(cfg, context):
    assumed = TokenRatio(Decimal("1.5"), False, "not measured: no report")
    _, record = record_for(cfg, context, ratio=assumed)
    assert record["passed"] is False
    assert [c["id"] for c in record["checks"] if not c["passed"]] == ["G1"]
    assert "NOT PASSED: no fine-tuning job may be started" in render_markdown(record)
    assert "NOT MEASURED, assumed" in render_markdown(record)


def test_the_record_is_reproducible_and_holds_no_text_path_or_time(cfg, context, tmp_path):
    _, first = record_for(cfg, context)
    _, second = record_for(cfg, context)
    assert first == second
    markdown_path, json_path = write_gate_record(first, cfg.report_dir)
    assert (markdown_path, json_path) == report_paths(cfg.report_dir)
    markdown, stored = markdown_path.read_text(encoding="utf-8"), json_path.read_text("utf-8")
    assert json.loads(stored) == first and stale_files(first, cfg.report_dir) == []
    for text in (markdown, stored):
        assert text.isascii() and "\r" not in text and str(tmp_path) not in text
        assert "artifacts" not in text and "2026-11-16T" not in text
        for word in ("zebrafish", "xylophone", "subtract", "PROGRAM"):
            assert word not in text, word
    assert "**PASSED: a fine-tuning job may be started on these files.**" in markdown
    assert "| `finqa_fit.jsonl` (training) | fit | 30 |" in markdown
    assert "| `amazon.nova-pro-v1:0` | $0.60 | $0.08 | $1.95 |" in markdown
    assert "| Training with the 10% margin | $0.83 |" in markdown
    assert "| **Total** (ledger, training with margin, evaluation, storage) | **$4.81** |" in (
        markdown
    )
    assert "| Cost | $0.38 | $0.75 | $1.13 | $1.50 | $1.88 |" in markdown
    assert "Not staged yet" in markdown


def test_read_gate_record_returns_the_written_record_or_a_config_error(cfg, context):
    with pytest.raises(ConfigError, match="no gate record"):
        read_gate_record(cfg)
    _, record = record_for(cfg, context)
    write_gate_record(record, cfg.report_dir)
    assert read_gate_record(cfg) == record
    report_paths(cfg.report_dir)[1].write_text("[]", encoding="utf-8")
    with pytest.raises(ConfigError, match="is damaged"):
        read_gate_record(cfg)
    report_paths(cfg.report_dir)[1].write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="is damaged"):
        read_gate_record(cfg)


def test_a_stale_record_is_detected(cfg, context):
    _, record = record_for(cfg, context)
    assert stale_files(record, cfg.report_dir) == [
        "nova_training_gate.md",
        "nova_training_gate.json",
    ]
    write_gate_record(record, cfg.report_dir)
    changed = dict(record, passed=False)
    assert stale_files(changed, cfg.report_dir) == [
        "nova_training_gate.md",
        "nova_training_gate.json",
    ]


def test_staged_locations_are_shown_and_carried_over_only_for_the_same_files(cfg, context):
    staged = {
        "training": {"s3_uri": "s3://bucket/data/bedrock/x/finqa_fit.jsonl", "bytes": 10,
                     "sha256": "a" * 64},
        "validation": {"s3_uri": "s3://bucket/data/bedrock/x/finqa_dev.jsonl", "bytes": 5,
                       "sha256": "b" * 64},
    }  # fmt: skip
    export, record = record_for(cfg, context, staged=staged)
    assert staged_before(cfg, export) is None  # nothing is written yet
    write_gate_record(record, cfg.report_dir)
    assert staged_before(cfg, export) == staged
    assert "| training | `s3://bucket/data/bedrock/x/finqa_fit.jsonl` | 10 |" in (
        render_markdown(record)
    )
    # Other files: what was staged before is not these files.
    changed = dataclasses.replace(
        export,
        files={
            **export.files,
            "training": dataclasses.replace(export.files["training"], sha256="c" * 64),
        },
    )
    assert staged_before(cfg, changed) is None
