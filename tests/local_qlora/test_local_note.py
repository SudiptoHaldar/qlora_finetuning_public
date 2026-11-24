"""Tests for qlora_finetuning.local_qlora.local_note: the committed record of the optional
full run on this machine (req_003 v3.06). A side note, never arm A1.

The stand-in split's quick-dev is the first 3 of dev's 8 examples.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qlora_finetuning.local_qlora import local_note as local_note_module
from qlora_finetuning.local_qlora.adapters import ADAPTER_WEIGHTS
from qlora_finetuning.local_qlora.local_note import local_note, render_note
from qlora_finetuning.local_qlora.local_run import LABEL
from qlora_finetuning.settings import ConfigError

from .selection_helpers import REVERSED, World, scored_world
from .training_helpers import holds_example_text

RIGHT = {2: 1, 4: 3, 6: 2}  # on quick-dev's 3: checkpoint 4 is best


def check(outcome, check_id):
    return next(item for item in outcome["checks"] if item["id"] == check_id)


def local_world(root, right=None, dev=True, **options):
    """A full run made here, its checkpoints scored on quick-dev, the best one on dev."""
    world = World(root, right=right or RIGHT, where="local", scope="dev", **options)
    world.wrong = {4: REVERSED}
    world.score(split="quick_dev")
    if dev:
        world.evaluate_checkpoint(4)
    return world


def note(world, **options):
    return local_note(world.run, **world.stand_ins(), **options)


# --------------------------------------------------------------------------- the note


def test_a_scored_local_run_gets_its_note(tmp_path):
    world = local_world(tmp_path)
    outcome = note(world)
    assert outcome["passed"] and [item["id"] for item in outcome["checks"]] == [
        f"N{n}" for n in range(1, 6)
    ]
    assert outcome["label"] == LABEL == "local run: not an arm of the comparison"
    path = world.training.training_cfg.local_note
    assert outcome["written"] == [str(path), str(path.with_suffix(".md"))]
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record == json.loads(json.dumps(outcome["note"]))

    assert record["label"] == LABEL and record["training_run"] == world.run
    assert (record["chain"], record["interruptions"]) == ([world.run], 0)
    assert (record["records"], record["epochs"], record["steps"]) == (30, 3, 6)
    assert record["settings_sha256"] == world.training.training_cfg.settings_sha256(False)
    assert record["selection_split"] == "quick_dev"
    assert [row["execution_accuracy"] for row in record["checkpoints"]] == [1, 3, 2]
    assert record["untuned"]["execution_accuracy"] == 0
    selected = record["selected"]
    assert (selected["step"], selected["examples"]) == (4, 3)
    assert selected["adapter"] == f"{world.run}/checkpoint-4"
    dev = record["selected_on_dev"]
    assert (dev["examples"], dev["execution_accuracy"], dev["format_validity"]) == (8, 3, 8)
    assert dev["categories"]["right"] == 3 and dev["categories"]["order"] == 5
    assert record["selected_categories"]["right"] == 3
    assert record["seconds_per_sequence"] == 0.03 and record["peak_vram_allocated_gib"] == 1.5
    assert record["against_cloud"] is None


def test_the_page_says_what_it_is_and_what_it_is_not(tmp_path):
    world = local_world(tmp_path)
    outcome = note(world)
    page = Path(outcome["written"][1]).read_text(encoding="utf-8")
    assert page == render_note(outcome["note"])
    assert f"**A {LABEL}.**" in page and "It is never arm A1" in page
    assert "## The checkpoints on `quick_dev` (3 examples)" in page
    assert "| 4 **(selected)** | 3 | 3 |" in page and "| untuned | 0 |" in page
    assert "3 of 8 right on execution (37.50%); 8 in the format (100.00%)" in page
    assert "| `order` | 5 |" in page
    assert "Not yet: no cloud checkpoint has been selected." in page
    assert "not comparable row by row" in page and "No test example was used." in page


def test_the_files_hold_no_reply_and_are_the_same_twice(tmp_path):
    world = local_world(tmp_path)
    files = [Path(name) for name in note(world)["written"]]
    before = [path.read_bytes() for path in files]
    note(world)
    assert [path.read_bytes() for path in files] == before
    for content in before:
        text = content.decode("utf-8")
        assert "PROGRAM" not in text and "ANSWER" not in text and "subtract(" not in text
        assert "\r\n" not in text
    assert holds_example_text(files[0].parent) == []


def test_check_writes_nothing(tmp_path):
    world = local_world(tmp_path)
    outcome = note(world, check=True)
    assert outcome["passed"] and outcome["note"]["selected"]["step"] == 4
    assert outcome["written"] == [] and not world.training.training_cfg.local_note.exists()


def test_a_local_note_is_never_a_selection(tmp_path):
    world = local_world(tmp_path)
    note(world)
    assert not world.training.training_cfg.selection_record.exists()
    source = Path(local_note_module.__file__).read_text(encoding="utf-8")
    assert "selection_record.write" not in source and "boto" not in source
    # The selection itself still refuses the run: it is local.
    selection = world.select(check=True, split="quick_dev")
    assert not selection["passed"] and selection["written"] == []


# --------------------------------------------------------------------------- N1 to N5


def test_n1_nothing_scored(tmp_path):
    world = World(tmp_path, where="local", scope="dev")
    outcome = note(world)
    assert [item["id"] for item in outcome["checks"]] == ["N1", "N2", "N3"]
    assert "L1: checkpoint 2 has 0 complete results on quick_dev" in check(outcome, "N1")["detail"]
    assert check(outcome, "N2")["passed"] and check(outcome, "N3")["passed"]
    assert outcome["note"] is None and outcome["written"] == []


def test_n1_weights_that_changed_after_the_scoring(tmp_path):
    world = local_world(tmp_path)
    (world.checkpoint(2) / ADAPTER_WEIGHTS).write_bytes(b"other weights")
    outcome = note(world)
    assert "L3:" in check(outcome, "N1")["detail"] and outcome["written"] == []


def test_n2_a_cloud_run_is_refused(tmp_path):
    world = World(tmp_path, right=RIGHT)  # made in SageMaker, as the stand-in records it
    world.score(split="quick_dev")
    world.evaluate_checkpoint(4)
    outcome = note(world)
    detail = check(outcome, "N2")["detail"]
    assert not check(outcome, "N2")["passed"] and f"run {world.run} is sagemaker" in detail
    assert outcome["written"] == [] and not outcome["passed"]


def test_n3_a_smoke_run_is_refused(tmp_path):
    world = World(tmp_path, right=RIGHT, where="local", scope="dev", smoke=True)
    world.score(split="quick_dev")
    outcome = note(world)
    assert check(outcome, "N3")["detail"] == "a smoke run is a rehearsal, not the full run"
    assert outcome["written"] == []


def test_n4_the_selected_checkpoint_must_be_scored_on_dev_once(tmp_path):
    world = local_world(tmp_path, dev=False)
    outcome = note(world)
    detail = check(outcome, "N4")["detail"]
    assert "checkpoint 4 has no complete result on dev" in detail
    assert f"--adapter {world.run}/checkpoint-4 --split dev" in detail
    assert outcome["note"] is None and outcome["written"] == []

    world.evaluate_checkpoint(2)  # another checkpoint on dev does not count
    assert not check(note(world), "N4")["passed"]
    world.evaluate_checkpoint(4)
    assert note(world, check=True)["passed"]
    world.evaluate_checkpoint(4)
    assert "checkpoint 4 has 2 results on dev" in check(note(world), "N4")["detail"]


def test_checkpoints_scored_on_dev_need_no_second_result(tmp_path):
    world = World(tmp_path, right={2: 3, 4: 5, 6: 4}, where="local", scope="dev")
    world.score(split="dev")
    outcome = note(world, split="dev")
    assert outcome["passed"] and "scored on dev" in check(outcome, "N4")["detail"]
    record = outcome["note"]
    assert record["selection_split"] == "dev" and record["selected"]["step"] == 4
    assert record["selected_on_dev"]["execution_accuracy"] == 5
    assert record["selected_on_dev"]["eval_run_id"] == record["selected"]["eval_run_id"]
    with pytest.raises(ConfigError, match="the training run's record"):
        local_note("20261120T000000Z-train-x-0000", **world.stand_ins())


# --------------------------------------------------------------------------- local against cloud


def test_with_a_cloud_selection_the_two_are_put_side_by_side(tmp_path):
    cloud = scored_world(tmp_path, right={2: 3, 4: 5, 6: 4})
    cloud.select()
    selection = cloud.first_cfg.selection_record.read_bytes()

    world = local_world(tmp_path)  # the same root: the same split, registry, and reports
    outcome = note(world)
    assert outcome["passed"], outcome["checks"]
    against = outcome["note"]["against_cloud"]
    assert against["cloud_training_run"] == cloud.run != world.run
    assert (against["cloud_selected_step"], against["local_selected_step"]) == (4, 4)
    assert (against["cloud_execution_accuracy"], against["cloud_examples"]) == (5, 8)
    assert against["execution_gap_examples"] == 3 - 5
    assert against["noise_examples"] == 1 and against["within_noise"] is False
    assert against["cloud_training_billable_seconds"] == 9000
    assert against["cloud_seconds_per_sequence"] == 0.03

    page = Path(outcome["written"][1]).read_text(encoding="utf-8")
    assert "| Right on execution, dev | 3 of 8 | 5 of 8 |" in page
    assert "The local checkpoint has -2 examples right against the cloud's: more than noise" in page
    assert "no price is put on it" in page and "$" not in page
    assert cloud.first_cfg.selection_record.read_bytes() == selection  # untouched


def test_a_gap_under_the_noise_is_said_to_be_one(tmp_path):
    cloud = scored_world(tmp_path, right={2: 3, 4: 3, 6: 3})
    cloud.select()
    world = local_world(tmp_path)
    world_cfg = world.training.training_cfg
    import dataclasses

    stand_ins = world.stand_ins() | {
        "training_cfg": dataclasses.replace(world_cfg, noise_examples=2)
    }
    outcome = local_note(world.run, check=True, **stand_ins)
    against = outcome["note"]["against_cloud"]
    assert (against["execution_gap_examples"], against["within_noise"]) == (0, True)
    assert "the two runs are not told apart" in render_note(outcome["note"])
