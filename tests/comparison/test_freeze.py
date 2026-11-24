"""Tests for the configuration freeze and the unlock (req_005 v5.01), on an invented split."""

from __future__ import annotations

import dataclasses
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from qlora_finetuning.comparison.config import load_comparison_config
from qlora_finetuning.comparison.freeze import (
    REPORT_NAME,
    Sources,
    build_freeze,
    current_arms,
    drift,
    estimate_test_passes,
    freeze_digest,
    freeze_status,
    render_markdown,
    unlock,
    write_freeze,
)
from qlora_finetuning.comparison.gate import file_sha256, read_freeze, read_unlock
from qlora_finetuning.data import load_split
from qlora_finetuning.data.bundle import BUNDLE_ENV
from qlora_finetuning.evaluation.fake import gold_predictor
from qlora_finetuning.evaluation.runner import RunnerContext
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking import GitInfo
from tests.comparison.helpers import (
    PREDICTORS,
    arms_now,
    comparison_cfg,
    dev_run,
    six_dev_runs,
    write_unlock,
)
from tests.data.synthetic import SECRET_WORDS
from tests.evaluation.conftest import GIT, evaluate, make_context, records_for

DIRTY = GitInfo(commit=GIT.commit, branch="main", dirty=True, changed=("config/prompt.yaml",))
TODAY = date(2026, 11, 22)


def estimator(spent_a="8.24", cap_a="100", spent_b="66.84", cap_b="200", measured=True):
    def estimate(dev):
        return estimate_test_passes(
            load_comparison_config(),
            dev,
            test_examples=22,
            path_a={
                "estimate": lambda tuned: (
                    (2681 if tuned else 3949, Decimal("1.05"), Decimal("2.82"))
                    if measured
                    else None
                ),
                "spent": Decimal(spent_a),
                "cap": Decimal(cap_a),
            },
            path_b={
                "cost": lambda model, tokens_in, tokens_out: Decimal(tokens_in) / 1000,
                "spent": Decimal(spent_b),
                "cap": Decimal(cap_b),
            },
        )

    return estimate


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.delenv(BUNDLE_ENV, raising=False)
    return make_context(tmp_path)


@pytest.fixture
def sources(context):
    return Sources(arms=arms_now(), dev_runs=six_dev_runs(context), estimator=estimator(), git=GIT)


def check(record, check_id):
    return next(item for item in record["checks"] if item["id"] == check_id)


def failed(record):
    return [item["id"] for item in record["checks"] if not item["passed"]]


# --------------------------------------------------------------------------- the record


def test_a_consistent_state_passes_every_check(context, sources):
    record = build_freeze(comparison_cfg(context), context, sources)
    assert record["passed"] and failed(record) == []
    assert [item["id"] for item in record["checks"]] == [f"F{n}" for n in range(1, 9)]
    assert list(record["arms"]) == list(PREDICTORS)
    a1 = record["arms"]["A1"]
    assert a1["predictor"] == PREDICTORS["A1"] and a1["where"] == "sagemaker"
    assert a1["dev_run"] == sources.dev_runs["A1"]
    assert a1["dev"] == {
        "examples": 8, "format_validity": 8, "execution_accuracy": 8,
        "answer_accuracy": 8, "program_match": 8,
    }  # fmt: skip
    assert record["contract"]["max_new_tokens"] == 192 and record["contract"]["dev_examples"] == 8
    assert set(record["shared"]["configs"]) == set(comparison_cfg(context).shared_configs)
    assert record["estimate"]["paths"]["A"] == {
        "estimate_usd": "$2.10", "at_most_usd": "$5.64", "spent_usd": "$8.24",
        "cap_usd": "$100.00",
    }  # fmt: skip


def test_the_same_state_gives_the_same_bytes_and_no_example_text(context, sources):
    cfg = comparison_cfg(context)
    first, second = build_freeze(cfg, context, sources), build_freeze(cfg, context, sources)
    assert first == second
    digest = write_freeze(first, cfg, context.split_cfg)
    assert digest == freeze_digest(first) == file_sha256(cfg.freeze_file)
    assert read_freeze(cfg)["arms"] == first["arms"]
    markdown = (cfg.report_dir / REPORT_NAME).read_text(encoding="utf-8")
    stored = cfg.freeze_file.read_text(encoding="utf-8")
    assert digest in markdown and render_markdown(first) == markdown
    for arm, run_id in sources.dev_runs.items():
        assert f"| {arm} |" in markdown and run_id in markdown and run_id in stored
    for text in (markdown, stored):
        assert not any(word in text for word in SECRET_WORDS)
        assert "@" not in text and "started_at" not in text


@pytest.mark.parametrize("check_id", ["F1", "F2"])
def test_a_path_that_cannot_be_worked_out_fails_its_check(context, sources, check_id):
    sources.problems[check_id] = ["no checkpoint has been selected yet"]
    record = build_freeze(comparison_cfg(context), context, sources)
    assert failed(record) == [check_id]
    assert "no checkpoint has been selected" in check(record, check_id)["detail"]


def test_each_arm_needs_one_dev_result(context, sources):
    cfg = comparison_cfg(context)
    del sources.dev_runs["B0-Pro"]
    record = build_freeze(cfg, context, sources)
    assert (
        "F3" in failed(record)
        and "arm B0-Pro has no dev result run" in check(record, "F3")["detail"]
    )

    sources.dev_runs["B0-Pro"] = dev_run("B0-Pro", context, limit=3)["run_id"]
    assert "it is partial" in check(build_freeze(cfg, context, sources), "F3")["detail"]

    sources.dev_runs["B0-Pro"] = sources.dev_runs["B0-Lite"]
    detail = check(build_freeze(cfg, context, sources), "F3")["detail"]
    assert "was recorded as arm B0-Lite" in detail

    sources.dev_runs["B0-Pro"] = "20260101T000000Z-eval-absent-0000"
    assert "cannot be read here" in check(build_freeze(cfg, context, sources), "F3")["detail"]


def test_the_dev_runs_must_share_one_contract(context, sources):
    other = dataclasses.replace(context, max_new_tokens=64)
    sources.dev_runs["A0"] = dev_run("A0", other)["run_id"]
    record = build_freeze(comparison_cfg(context), context, sources)
    assert failed(record) == ["F4"]
    assert (
        "arm A0: its dev run was made with another max new tokens" in check(record, "F4")["detail"]
    )


def test_todays_configuration_must_be_the_dev_runs(context, sources):
    sources.arms["B1-Lite"]["predictor"]["model_id"] = "another-deployment"
    sources.arms["A0"]["predictor"]["settings"] = {"adapter": None, "batch_size": 1}
    record = build_freeze(comparison_cfg(context), context, sources)
    detail = check(record, "F5")["detail"]
    assert failed(record) == ["F5"]
    assert "arm B1-Lite: today's configuration differs from its dev run in model id" in detail
    assert "arm A0" in detail and "settings (batch_size, revision)" in detail


def test_the_freeze_comes_before_the_unlock_and_before_any_test_run(context, sources):
    cfg = comparison_cfg(context)
    good = build_freeze(cfg, context, sources)
    write_unlock(context, None)
    record = build_freeze(cfg, context, sources)
    assert failed(record) == ["F6"] and "an unlock record exists" in check(record, "F6")["detail"]
    with pytest.raises(ConfigError, match="never rewritten after it"):
        write_freeze(good, cfg, context.split_cfg)
    assert not cfg.freeze_file.exists()


def test_uncommitted_changes_fail_f7(context, sources):
    sources.git = DIRTY
    record = build_freeze(comparison_cfg(context), context, sources)
    assert failed(record) == ["F7"] and "config/prompt.yaml" in check(record, "F7")["detail"]
    with pytest.raises(ConfigError, match="a freeze check failed"):
        write_freeze(record, comparison_cfg(context), context.split_cfg)


def test_the_estimate_must_fit_in_the_caps(context, sources):
    cfg = comparison_cfg(context)
    sources.estimator = estimator(spent_a="96")
    record = build_freeze(cfg, context, sources)
    assert failed(record) == ["F8"]
    assert "Path A: $96.00 spent plus at most $5.64" in check(record, "F8")["detail"]

    sources.estimator = estimator(measured=False)
    detail = check(build_freeze(cfg, context, sources), "F8")["detail"]
    assert "arm A0: no pulled job has measured this kind of work" in detail


def test_nova_tokens_are_the_dev_runs_scaled_to_the_test_split(context, sources):
    record = build_freeze(comparison_cfg(context), context, sources)
    summary_tokens = json.loads(
        (context.tracking_cfg.runs_dir / sources.dev_runs["B0-Lite"] / "summary.json").read_text()
    )["usage"]["input_tokens"]
    entry = record["estimate"]["arms"]["B0-Lite"]
    assert entry["input_tokens"] == round(Decimal(summary_tokens) * 22 / 8)
    assert entry["basis"].startswith("the dev run's tokens")


# --------------------------------------------------------------------------- the unlock


@pytest.fixture
def frozen(context, sources):
    cfg = comparison_cfg(context)
    digest = write_freeze(build_freeze(cfg, context, sources), cfg, context.split_cfg)
    return cfg, digest


def open_it(context, sources, cfg, **options):
    settings = {"yes": True, "tracked": lambda path: True, "today": TODAY}
    settings.update(options)
    return unlock(cfg, context, sources, **settings)


def test_the_owner_unlocks_by_naming_the_freeze_record(context, sources, frozen):
    cfg, digest = frozen
    record = open_it(context, sources, cfg, freeze=digest[:12])
    assert record == {
        "schema": 1,
        "manifest_hash": context.manifest["content_hash"],
        "freeze_sha256": digest,
        "freeze_commit": GIT.commit,
        "unlocked_on": "2026-11-22",
        "unlocked_by": "the owner, req_005 v5.01",
    }
    assert read_unlock(context.split_cfg) == record
    assert len(load_split("test", context.split_cfg, context.data_cfg)) == 22
    assert evaluate(gold_predictor(records_for("test", context)), "test", context)["result"]
    with pytest.raises(ConfigError, match="the test split is unlocked once"):
        open_it(context, sources, cfg, freeze=digest[:12])


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"yes": False, "freeze": "x" * 12}, "is run by the owner"),
        ({"freeze": None}, "unlock --yes --freeze"),
        ({"freeze": "0" * 12}, "is not the freeze record on disk"),
        ({"freeze": "PREFIX"}, "give at least 12 characters"),
        ({"tracked": lambda path: False, "freeze": "FULL"}, "is not committed"),
    ],
)
def test_an_unlock_that_is_not_signed_off_changes_nothing(
    context, sources, frozen, options, message
):
    cfg, digest = frozen
    options = dict(options)
    if options.get("freeze") == "PREFIX":
        options["freeze"] = digest[:8]
    elif options.get("freeze") == "FULL":
        options["freeze"] = digest
    with pytest.raises(ConfigError, match=message):
        open_it(context, sources, cfg, **options)
    assert not Path(context.split_cfg.unlock_file).exists()


def test_the_unlock_needs_a_clean_tree_and_an_unchanged_configuration(context, sources, frozen):
    cfg, digest = frozen
    sources.git = DIRTY
    with pytest.raises(ConfigError, match="commit the freeze record first"):
        open_it(context, sources, cfg, freeze=digest[:12])
    sources.git = GIT
    sources.arms["B1-Pro"]["predictor"]["model_id"] = "a-new-deployment"
    with pytest.raises(ConfigError, match="arm B1-Pro: model id changed"):
        open_it(context, sources, cfg, freeze=digest[:12])
    sources.arms = arms_now()
    sources.problems["F2"] = ["the deployment of nova-pro (attempt 1) is Failed"]
    with pytest.raises(ConfigError, match="the deployment of nova-pro"):
        open_it(context, sources, cfg, freeze=digest[:12])
    assert not Path(context.split_cfg.unlock_file).exists()


def test_there_is_nothing_to_unlock_without_a_freeze_record(context, sources):
    with pytest.raises(ConfigError, match="there is no freeze record yet"):
        open_it(context, sources, comparison_cfg(context), freeze="0" * 12)


# --------------------------------------------------------------------------- does it still hold


def test_status_follows_the_freeze_the_unlock_and_any_change(context, sources, frozen):
    cfg, digest = frozen
    status = freeze_status(cfg, context, sources)
    assert status["frozen"] and status["freeze_sha256"] == digest
    assert not status["unlocked"] and status["differences"] == []

    open_it(context, sources, cfg, freeze=digest)
    status = freeze_status(cfg, context, sources)
    assert status["unlocked"] and status["unlock_names_freeze"]
    assert status["unlocked_on"] == "2026-11-22" and status["freeze_commit"] == GIT.commit

    flags = Path(context.config.flags_file)
    flags.write_text(flags.read_text(encoding="utf-8") + " ", encoding="utf-8")
    sources.arms["A1"]["predictor"]["settings"] = {"adapter": "0" * 64}
    differences = freeze_status(cfg, context, sources)["differences"]
    assert "the gold flags changed" in differences
    assert any(text.startswith("arm A1: settings (adapter") for text in differences)
    del sources.arms["A0"]
    assert "arm A0 can no longer be worked out from the configuration" in drift(
        read_freeze(cfg), cfg, context, sources.arms
    )


def test_nothing_is_frozen_at_first(context, sources):
    status = freeze_status(comparison_cfg(context), context, sources)
    assert not status["frozen"] and status["freeze_sha256"] is None and not status["unlocked"]


# --------------------------------------------------------------------------- the real repository


def test_the_six_arms_can_be_worked_out_from_the_committed_records():
    """Config and committed records only: no run folder, no AWS, no model.

    Once a freeze record is committed this is also the guard on it: a frozen value that was
    changed afterwards makes this test fail.
    """
    cfg = load_comparison_config()
    context = RunnerContext().resolved()
    arms, problems = current_arms(cfg, context)
    assert problems == {"F1": [], "F2": []} and list(arms) == list(cfg.arms)
    assert arms["A0"]["predictor"]["settings"]["adapter"] is None
    assert (
        arms["A1"]["predictor"]["settings"]["adapter"] == arms["A1"]["identity"]["adapter_sha256"]
    )
    for arm in ("B1-Lite", "B1-Pro"):
        assert arms[arm]["predictor"]["model_id"].startswith("arn:aws:bedrock:")
        assert arms[arm]["predictor"]["settings"] == arms["B0-Lite"]["predictor"]["settings"]
    stored = read_freeze(cfg)
    if stored is not None:
        assert drift(stored, cfg, context, arms) == []
