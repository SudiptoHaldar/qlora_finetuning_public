"""Tests for reading an arm's outcomes and for the local examples file (req_005 v5.03).

Runs are made on the invented, frozen, unlocked split of ``tests/comparison/helpers.py``
(22 test examples from invented companies) with stand-in predictors.
"""

from __future__ import annotations

import dataclasses

import pytest

from qlora_finetuning.comparison.outcomes import METRICS, companies, read_arm, read_arms
from qlora_finetuning.comparison.report_config import load_report_config
from qlora_finetuning.comparison.report_examples import NOTICE, render_examples, write_examples
from qlora_finetuning.data.bundle import BUNDLE_ENV, export_bundle
from qlora_finetuning.evaluation.fake import FakePredictor
from qlora_finetuning.evaluation.runner import summarize_run
from qlora_finetuning.settings import ConfigError
from tests.comparison.helpers import PREDICTORS, USAGE, comparison_cfg, freeze_and_unlock, run_arm
from tests.evaluation.conftest import GOLD_REPLY, ids_of, make_context

REVERSED = "PROGRAM: subtract(95, 920)\nANSWER: -825"
CHATTY = "Sure!\n" + GOLD_REPLY
NOTHING = "I cannot tell."


@pytest.fixture
def context(tmp_path, monkeypatch):
    """A frozen, unlocked invented split, in a process that looks like a test job."""
    monkeypatch.delenv(BUNDLE_ENV, raising=False)
    context = make_context(tmp_path)
    freeze_and_unlock(context)
    folder = tmp_path / "bundle_test"
    export_bundle(folder, ("test",), cfg=context.split_cfg, data_cfg=context.data_cfg)
    monkeypatch.setenv(BUNDLE_ENV, str(folder))
    return context


def scripted(arm: str, context, replies: dict[str, str]) -> str:
    """One test run of ``arm`` that answers with gold except where ``replies`` says."""
    ids = ids_of("test", context)
    frozen = PREDICTORS[arm]
    predictor = FakePredictor(
        {example_id: replies.get(example_id, GOLD_REPLY) for example_id in ids},
        name=frozen["name"],
        model_id=frozen["model_id"],
        settings=frozen["settings"],
        usage_source=USAGE[comparison_cfg(context).arms[arm].path],
    )
    return run_arm(arm, context, predictor=predictor)["run_id"]


@pytest.fixture
def run(context):
    ids = ids_of("test", context)
    return scripted("B0-Lite", context, {ids[0]: REVERSED, ids[1]: CHATTY, ids[2]: NOTHING})


def test_an_arm_is_read_example_by_example(context, run):
    ids = ids_of("test", context)
    arm = read_arm("B0-Lite", run, context)
    assert arm.arm == "B0-Lite" and arm.run_id == run
    assert list(arm.example_ids) == ids and len(ids) == 22
    assert set(arm.right) == set(METRICS)
    assert arm.right["execution_accuracy"][:4] == (False, True, False, True)
    assert arm.right["format_validity"][:4] == (True, False, False, True)
    assert arm.categories[:4] == ("order", "right", "no_program", "right")
    assert set(arm.gold_steps) == {1} and set(arm.stop_reasons) == {"end"}


def test_the_totals_are_the_runners_own(context, run):
    arm = read_arm("B0-Lite", run, context)
    block = summarize_run(run, context)["all"]
    assert arm.counts == {"examples": 22} | {
        metric: block[metric]["count"] for metric in METRICS
    }
    assert arm.counts["execution_accuracy"] == 20 and arm.counts["format_validity"] == 20


def test_the_outcomes_hold_no_text(context, run):
    shown = repr(read_arm("B0-Lite", run, context))
    assert "subtract" not in shown and "Sure" not in shown and "825" not in shown
    names = {field.name for field in dataclasses.fields(read_arm("B0-Lite", run, context))}
    assert not names & {"reply", "replies", "question", "gold", "text"}


def test_several_arms_are_read_and_a_missing_run_is_a_problem(context, run):
    other = scripted("B0-Pro", context, {})
    arms, problems = read_arms(
        {"B0-Lite": run, "A1": "20261101T000000Z-eval-nothing-test-0000", "B0-Pro": other},
        context,
    )
    assert list(arms) == ["B0-Lite", "B0-Pro"]
    assert arms["B0-Pro"].counts["execution_accuracy"] == 22
    assert arms["B0-Pro"].example_ids == arms["B0-Lite"].example_ids
    assert len(problems) == 1 and problems[0].startswith(
        "arm A1: run 20261101T000000Z-eval-nothing-test-0000 cannot be read here"
    )


def test_the_company_of_an_example_comes_from_its_id(context):
    ids = ids_of("test", context)
    found = companies(ids)
    assert len(found) == 22 and all(name == example_id.split("/")[0] for name, example_id in
                                    zip(found, ids, strict=True))
    assert companies(["no-slash"]) == ["(unknown)"]


# --------------------------------------------------------------------------- the examples file


def stored_record(context, run) -> dict:
    ids = ids_of("test", context)
    return {
        "arms": {
            "B0-Lite": {
                "label": "Nova Lite, untuned",
                "run_id": run,
                "categories": {"order": 1, "no_program": 1},
                "example_ids": {"order": [ids[0]], "no_program": [ids[2]]},
            }
        }
    }


def test_the_examples_file_holds_the_text_and_says_it_is_not_for_the_repository(context, run):
    text = render_examples(stored_record(context, run), context)
    ids = ids_of("test", context)
    assert NOTICE in text and f"**{ids[0]}**" in text and f"**{ids[2]}**" in text
    assert "subtract(95, 920)" in text and NOTHING in text  # the arm's replies
    assert "subtract(920, 95)" in text  # the gold reply
    assert "what was the change in xylophone sales" in text  # the invented question
    assert "### B0-Lite, `order` (1 replies)" in text


def test_the_examples_file_is_written_only_under_artifacts(context, run, tmp_path):
    cfg = load_report_config(root=tmp_path)
    path = write_examples(stored_record(context, run), cfg, context)
    assert path == tmp_path / "artifacts" / "comparison" / "error_examples.md"
    assert "subtract(95, 920)" in path.read_text(encoding="utf-8")

    outside = dataclasses.replace(cfg, examples_file=tmp_path / "reports" / "examples.md")
    with pytest.raises(ConfigError, match="only under artifacts/"):
        write_examples(stored_record(context, run), outside, context)
    assert not (tmp_path / "reports" / "examples.md").exists()

    with pytest.raises(ConfigError, match="names no examples"):
        write_examples({"arms": {}}, cfg, context)
