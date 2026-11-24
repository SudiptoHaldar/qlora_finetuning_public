"""Checks P1 to P10, the two cards, and the package record, on a stand-in project."""

from __future__ import annotations

import json
from datetime import date

import pytest

from qlora_finetuning.packaging import workflow
from qlora_finetuning.packaging.cards import ADAPTER, MERGED, header, usage_code
from qlora_finetuning.packaging.merged_package import merged_weights
from qlora_finetuning.packaging.record import (
    NOT_BUILT,
    NOT_READY,
    READY,
    WITHHELD,
    gather,
    render_markdown,
    write_record,
)
from qlora_finetuning.packaging.scan import scan_text

from .helpers import (
    GOLD_REPLY,
    TODAY,
    confirmed,
    make_run,
    make_world,
    write_card_runs,
)
from .test_packages import build_adapter, build_merged

DAY = date.fromisoformat(TODAY)


def state_of(world, cfg=None):
    return gather(
        cfg or world.cfg,
        world.inputs,
        ctx=world.runner,
        today=DAY,
        base=world.base,
        fingerprint=lambda folder: world.inputs.budget.fingerprint,
    )


def codes(world):
    budget, local_cfg = world.inputs.budget, world.inputs.local_cfg
    adapter = usage_code(ADAPTER, world.cfg, budget, local_cfg)[0]
    merged = usage_code(MERGED, world.cfg, budget, local_cfg)[0]
    return {ADAPTER: adapter, "merged_4bit": merged, "merged_full": merged}


def full_world(tmp_path, *, r2_wrong=0, r3_wrong=0, r4_wrong=0, merged=True):
    """Both packages built, four runs made, the cards' code run."""
    world = make_world(tmp_path)
    build_adapter(world)
    make_run(world, "R1")
    make_run(world, "R2", wrong=r2_wrong)
    if merged:
        folder = build_merged(world)
        weights = merged_weights(folder)["sha256"]
        make_run(world, "R3", wrong=r3_wrong, weights_sha256=weights)
        make_run(world, "R4", wrong=r4_wrong, weights_sha256=weights)
    write_card_runs(world, codes(world))
    return world


def by_id(state):
    return {check["id"]: check for check in state.record["checks"]}


def write_cards(world, monkeypatch):
    """``workflow.write_cards`` over the stand-in world."""
    monkeypatch.setattr(workflow, "gather", lambda cfg, inputs=None: state_of(world, cfg))
    return workflow.write_cards(world.cfg, world.inputs)


# --------------------------------------------------------------------------- the checks


def test_before_anything_is_run_the_record_says_what_is_missing(tmp_path):
    world = make_world(tmp_path)
    state = state_of(world)
    checks = by_id(state)
    assert [check["id"] for check in state.record["checks"]] == [f"P{n}" for n in range(1, 11)]
    assert checks["P1"]["passed"] and checks["P2"]["passed"] and checks["P7"]["passed"]
    assert "build the package first" in checks["P4"]["detail"]
    assert "(R1)" in checks["P8"]["detail"] and "(R2)" in checks["P8"]["detail"]
    assert "hf_package merge" in checks["P9"]["detail"]
    assert state.record["adapter"]["status"] == NOT_READY
    assert state.record["merged"]["status"] == NOT_BUILT
    assert state.record["passed"] is False and state.cards == {ADAPTER: None, MERGED: None}


def test_everything_in_place_makes_both_packages_ready(tmp_path, monkeypatch):
    world = full_world(tmp_path)
    before = state_of(world)
    assert by_id(before)["P8"]["passed"] and by_id(before)["P9"]["passed"]
    assert not by_id(before)["P10"]["passed"]  # the cards are built but not written yet
    assert before.cards[ADAPTER] and before.cards[MERGED]

    state, written = write_cards(world, monkeypatch)
    assert [check["passed"] for check in state.record["checks"]] == [True] * 10
    assert state.record["passed"] is True
    assert state.record["adapter"]["status"] == READY
    assert state.record["merged"]["status"] == READY
    names = sorted(path.relative_to(tmp_path).as_posix() for path in written)
    assert names == [
        "artifacts/release/adapter/README.md",
        "artifacts/release/merged/README.md",
        "reports/packaging/adapter_card.md",
        "reports/packaging/merged_card.md",
        "reports/packaging/package_record.json",
        "reports/packaging/package_record.md",
    ]
    assert world.cfg.adapter_card.read_bytes() == (world.cfg.adapter_dir / "README.md").read_bytes()
    assert b"\r\n" not in world.cfg.adapter_card.read_bytes()
    assert "README.md" in [entry["name"] for entry in state.record["adapter"]["files"]]


def test_one_reply_of_the_package_that_differs_fails_p8_and_is_named(tmp_path):
    world = full_world(tmp_path, r2_wrong=1)
    state = state_of(world)
    check = by_id(state)["P8"]
    difference = state.record["comparisons"]["R2"]
    assert not check["passed"] and "R2 differs from R1 on 1 replies" in check["detail"]
    assert difference["differing_ids"][0] in check["detail"]
    assert state.record["adapter"]["status"] == NOT_READY
    assert "another reply" in state.cards[ADAPTER]  # the card would say so


def test_r1_far_from_the_recorded_score_fails_p8(tmp_path):
    world = full_world(tmp_path)
    recorded = world.inputs.records.statistics["arms"]["A1"]["dev"]
    recorded["execution_accuracy"] = 8 + 10  # ten apart from R1's eight: not under the noise
    check = by_id(state_of(world))["P8"]
    assert not check["passed"] and "10 apart, not under 10" in check["detail"]
    recorded["execution_accuracy"] = 8 + 9
    assert by_id(state_of(world))["P8"]["passed"]


@pytest.mark.parametrize("label", ["R3", "R4"])
def test_a_merged_copy_outside_the_allowance_is_withheld_and_the_adapter_stays_ready(
    tmp_path, monkeypatch, label
):
    wrong = {"r3_wrong": 2} if label == "R3" else {"r4_wrong": 2}
    world = full_world(tmp_path, **wrong)
    import dataclasses

    world.cfg = dataclasses.replace(world.cfg, allowance_examples=2)
    state, written = write_cards(world, monkeypatch)
    checks = by_id(state)
    assert not checks["P9"]["passed"] and f"{label} has 2 fewer right" in checks["P9"]["detail"]
    assert state.record["merged"]["status"] == WITHHELD
    assert state.record["merged"]["withheld_because"]
    assert state.record["adapter"]["status"] == READY and checks["P10"]["passed"]
    assert not world.cfg.merged_card.exists() and not (world.cfg.merged_dir / "README.md").exists()
    assert "built and not published" in world.cfg.adapter_card.read_text(encoding="utf-8")
    assert world.cfg.merged_repo not in world.cfg.adapter_card.read_text(encoding="utf-8")

    world.cfg = dataclasses.replace(world.cfg, allowance_examples=3)  # two fewer is now inside
    assert state_of(world).record["merged"]["status"] in (READY, NOT_READY)
    assert by_id(state_of(world))["P9"]["passed"]


def test_a_merged_copy_that_scores_higher_is_not_withheld(tmp_path):
    world = make_world(tmp_path)
    build_adapter(world)
    make_run(world, "R1", wrong=3)
    make_run(world, "R2", wrong=3)
    weights = merged_weights(build_merged(world))["sha256"]
    make_run(world, "R3", weights_sha256=weights)
    make_run(world, "R4", wrong=1, weights_sha256=weights)
    write_card_runs(world, codes(world))
    state = state_of(world)
    assert by_id(state)["P9"]["passed"] and state.record["merged"]["withheld_because"] == []
    assert state.record["comparisons"]["R3"]["wrong_to_right"] == 3
    assert "3 more right than the adapter" in state.cards[MERGED]
    assert "2 more right than the adapter" in state.cards[MERGED]


def test_card_code_that_was_not_run_changed_or_failed_fails_its_check(tmp_path):
    world = full_world(tmp_path)
    world.cfg.card_runs_file.unlink()
    state = state_of(world)
    assert "was not run yet (adapter)" in by_id(state)["P8"]["detail"]
    assert "was not run yet (merged_full)" in by_id(state)["P9"]["detail"]
    assert state.cards[ADAPTER] is None  # no reply of the sample to print

    stale = codes(world)
    stale[ADAPTER] += "# another line\n"
    write_card_runs(world, stale)
    assert "changed since it was run (adapter)" in by_id(state_of(world))["P8"]["detail"]

    write_card_runs(world, codes(world), merged_full=["the reply's program did not run (x)"])
    state = state_of(world)
    assert by_id(state)["P8"]["passed"] and "did not run" in by_id(state)["P9"]["detail"]
    assert state.record["merged"]["status"] == NOT_READY


def test_private_strings_extra_files_and_a_wrong_tokenizer_fail(tmp_path):
    world = full_world(tmp_path)
    (world.cfg.adapter_dir / "notes.txt").write_text("see s3://bucket/key", encoding="utf-8")
    (world.cfg.merged_dir / "vocab.json").write_bytes(b"{}")
    checks = by_id(state_of(world))
    assert "adapter: notes.txt does not belong in the package" in checks["P4"]["detail"]
    assert "adapter: notes.txt holds an S3 path" in checks["P5"]["detail"]
    assert "merged: vocab.json is not the base model's file" in checks["P6"]["detail"]

    world = full_world(tmp_path / "second")
    state = gather(
        world.cfg, world.inputs, ctx=world.runner, today=DAY, base=world.base,
        fingerprint=lambda folder: "another tokenizer",
    )  # fmt: skip
    assert "not the one pinned" in by_id(state)["P6"]["detail"]


# --------------------------------------------------------------------------- the cards


def test_unconfirmed_licences_give_drafts_only(tmp_path, monkeypatch):
    world = full_world(tmp_path)
    world.cfg = confirmed(world.cfg, None)
    monkeypatch.setattr(workflow, "date", type("D", (), {"today": staticmethod(lambda: DAY)}))
    state, written = write_cards(world, monkeypatch)
    assert not by_id(state)["P7"]["passed"] and not by_id(state)["P10"]["passed"]
    assert state.record["adapter"]["status"] == NOT_READY
    names = sorted(path.relative_to(tmp_path).as_posix() for path in written)
    assert names == [
        "artifacts/release/drafts/adapter_card.md",
        "artifacts/release/drafts/merged_card.md",
        "reports/packaging/package_record.json",
        "reports/packaging/package_record.md",
    ]
    assert not world.cfg.adapter_card.exists()
    assert not (world.cfg.adapter_dir / "README.md").exists()
    assert "**not yet**" in world.cfg.record_markdown.read_text(encoding="utf-8")


def test_the_adapter_card_says_what_it_must(tmp_path):
    world = full_world(tmp_path)
    card = state_of(world).cards[ADAPTER]
    front = header(card)
    assert front["base_model"] == "Qwen/Qwen3-4B" and front["base_model_relation"] == "adapter"
    assert front["library_name"] == "peft" and front["license"] == "apache-2.0"
    result = front["model-index"][0]["results"][0]
    assert "not the official test split" in result["dataset"]["name"]
    assert [metric["type"] for metric in result["metrics"]][0] == "execution_accuracy"
    arm = world.inputs.records.statistics["arms"]["A1"]["metrics"]
    assert result["metrics"][0]["value"] == float(arm["execution_accuracy"]["percent"])
    assert "{{" not in card and scan_text(card, world.cfg.forbidden, "card") == []
    warning = card.index("Do not report this model's score on the official FinQA test split")
    assert warning < card.index("## What it does")  # near the top
    for words in (
        "not comparable with published FinQA results",
        "not financial advice",
        "Trust the program, not the answer line",
        "cannot be\ndownloaded",
        "enable_thinking=False",
        "gave the same reply as the\noriginal checkpoint on every one of the 8 development",
        "@article{chen2021finqa",
        world.cfg.merged_repo,
        world.cfg.code_repository,
        world.source.sha256,
    ):
        assert words.replace("\n", " ") in card.replace("\n", " "), words
    assert usage_code(ADAPTER, world.cfg, world.inputs.budget, world.inputs.local_cfg)[0] in card
    assert "Northfield Tools is an invented company" in card  # the sample, never a dataset row
    assert GOLD_REPLY.splitlines()[0] not in card  # no reply of a run


def test_the_merged_card_gives_its_own_dev_figures_and_labels_the_test_ones(tmp_path):
    world = full_world(tmp_path, r3_wrong=1)
    card = state_of(world).cards[MERGED]
    front = header(card)
    assert front["library_name"] == "transformers" and front["base_model_relation"] == "finetune"
    metrics = front["model-index"][0]["results"][0]
    assert metrics["dataset"]["split"].startswith("dev")
    assert [metric["value"] for metric in metrics["metrics"]] == [87.5, 100.0]
    for words in (
        "This copy is not the model that was measured on the test split",
        "These figures are **the adapter's**",
        "Do not report this model's score on the official FinQA test split",
        "1 fewer right than the adapter",
        "as many right as the adapter",
        "generation_config.json",
        "QUANTIZED",
    ):
        assert words in card.replace("\n", " "), words
    assert "{{" not in card and scan_text(card, world.cfg.forbidden, "card") == []


def test_a_card_on_disk_that_is_not_the_card_as_built_fails_p10(tmp_path, monkeypatch):
    world = full_world(tmp_path)
    write_cards(world, monkeypatch)
    world.cfg.adapter_card.write_text("# edited by hand\n", encoding="utf-8")
    detail = by_id(state_of(world))["P10"]["detail"]
    assert "the committed card (adapter) is out of date" in detail
    (world.cfg.merged_dir / "README.md").unlink()
    checks = by_id(state_of(world))
    assert "the card in the package (merged) is not written" in checks["P10"]["detail"]
    assert "merged: README.md is missing" in checks["P4"]["detail"]


# --------------------------------------------------------------------------- the record


def test_the_record_is_the_same_bytes_each_time_and_holds_no_reply(tmp_path, monkeypatch):
    world = full_world(tmp_path, r3_wrong=1)
    state, _ = write_cards(world, monkeypatch)
    first = (world.cfg.record_json.read_bytes(), world.cfg.record_markdown.read_bytes())
    write_record(world.cfg, state_of(world).record)
    assert (world.cfg.record_json.read_bytes(), world.cfg.record_markdown.read_bytes()) == first
    text = first[0].decode("utf-8") + first[1].decode("utf-8")
    assert "PROGRAM:" not in text and "subtract(" not in text and str(tmp_path) not in text
    assert scan_text(text, world.cfg.forbidden, "the record") == []
    record = json.loads(first[0])
    assert set(record["runs"]) == {"R1", "R2", "R3", "R4"}
    assert record["runs"]["R3"]["counts"]["execution_accuracy"] == 7
    assert record["runs"]["R3"]["percent"]["execution_accuracy"] == "87.50"
    assert record["comparisons"]["R3"]["right_to_wrong"] == 1
    assert record["card_code"]["adapter"]["value"] == 0.1256140350877193
    assert "reply" not in record["card_code"]["adapter"]
    assert record["merged"]["weights_sha256"] == merged_weights(world.cfg.merged_dir)["sha256"]
    markdown = render_markdown(record)
    assert "| R3 |" in markdown and "**ready**" in markdown and "Do not edit by hand" in markdown
