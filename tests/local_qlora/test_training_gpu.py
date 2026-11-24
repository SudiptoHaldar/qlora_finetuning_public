"""The real training engine on a GPU (req_003 v3.02). Deselected by default: ``pytest -m gpu``.

It trains the small smoke model of ``config/local_env.yaml`` on the invented split for a few
steps: the same code path as the 4B model, in under a minute. Everything is written under
the temporary folder; nothing is downloaded.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from qlora_finetuning.data.token_budget import load_token_budget_config, load_tokenizer
from qlora_finetuning.local_qlora.adapters import (
    ADAPTER_WEIGHTS,
    CHECKPOINTS,
    METADATA_NAME,
    adapter_sha256,
)
from qlora_finetuning.local_qlora.backend import load_backend
from qlora_finetuning.local_qlora.render import IGNORE
from qlora_finetuning.local_qlora.training import (
    LOG_NAME,
    TRAINER_STATE,
    TrainingStopped,
    prepare,
    run_training,
)
from qlora_finetuning.settings import load_config
from tests.evaluation.conftest import ledger_records

from .training_helpers import holds_example_text, make_training_context

pytestmark = pytest.mark.gpu


@pytest.fixture
def context(tmp_path):
    """The invented split, the real tokenizer, and the small model. 8 optimizer steps."""
    smoke = load_config("local_env", required=["smoke.model_id", "smoke.revision"])["smoke"]
    budget = load_token_budget_config()
    base = make_training_context(tmp_path)
    cfg = dataclasses.replace(
        base.training_cfg,
        accumulation=2,  # 16 records in batches of 2: 4 steps an epoch
        smoke=dataclasses.replace(
            base.training_cfg.smoke,
            examples=16,
            epochs=2,
            save_steps=3,
            dev_examples=4,
            generate_examples=3,
            generate_batch_size=2,
        ),
    )
    return dataclasses.replace(
        base,
        training_cfg=cfg,
        budget=budget,
        tokenizer=load_tokenizer(budget),
        model_id=smoke["model_id"],
        revision=smoke["revision"],
    )


def test_the_collator_leaves_labels_on_the_reply_only(context):
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    rows = prepare(context.resolved(), smoke=True).rows[:2]
    collate = DataCollatorForLanguageModeling(pad_token_id=context.tokenizer.pad_token_id)
    batch = collate([{"input_ids": row.input_ids, "labels": row.labels} for row in rows])
    for position, row in enumerate(rows):
        labels = batch["labels"][position].tolist()
        assert labels[: row.prompt_tokens] == [IGNORE] * row.prompt_tokens
        assert labels[row.prompt_tokens : row.length] == row.input_ids[row.prompt_tokens :]
        assert all(label == IGNORE for label in labels[row.length :])  # padding
        assert int(batch["attention_mask"][position].sum()) == row.length


def test_a_run_is_stopped_resumed_and_its_adapter_reloaded(context):
    with pytest.raises(TrainingStopped) as caught:
        run_training(smoke=True, context=context, stop_after_steps=4)
    first = caught.value.run_id
    checkpoints = context.tracking_cfg.runs_dir / first / CHECKPOINTS
    assert (checkpoints / "checkpoint-3" / TRAINER_STATE).is_file()
    assert (checkpoints / "checkpoint-3" / ADAPTER_WEIGHTS).is_file()
    assert not (checkpoints / "checkpoint-6").exists()

    result = run_training(smoke=True, context=context, resume=first)
    assert (result["start_step"], result["steps"], result["continues"]) == (3, 8, first)
    checks = {check["id"]: check for check in result["checks"]}
    for check_id in ("K1", "K2", "K3", "K4", "K5", "K7"):
        assert checks[check_id]["passed"], checks[check_id]
    assert result["plan"]["checkpoint_steps"] == [3, 6, 8]

    run_dir = context.tracking_cfg.runs_dir / result["run_id"]
    assert sorted(path.name for path in (run_dir / CHECKPOINTS).glob("checkpoint-*")) == [
        "checkpoint-6",
        "checkpoint-8",
    ]
    log = [json.loads(line) for line in (run_dir / LOG_NAME).read_text("utf-8").splitlines()]
    assert [entry["step"] for entry in log if "loss" in entry] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert {entry["step"] for entry in log if "eval_loss" in entry} == {3, 6, 8}

    measurements = result["measurements"]
    assert measurements["trainer"]["max_steps"] == 8  # the trainer's arithmetic is the plan's
    assert measurements["trainable_parameters"] > 0
    assert measurements["probe"]["peak_vram_allocated_gib"] > 0
    assert measurements["reload"]["replies"] == 3 and measurements["reload"]["think_markers"] == 0

    # The saved adapter is one the inference backend loads, under the hash in its metadata.
    adapter_dir = context.training_cfg.adapters_dir / result["run_id"]
    metadata = json.loads((adapter_dir / METADATA_NAME).read_text(encoding="utf-8"))
    assert metadata["adapter_sha256"] == adapter_sha256(adapter_dir) == result["adapter_sha256"]
    backend = load_backend(
        context.local_cfg, context.budget, context.model_id, context.revision, adapter_dir
    )
    assert backend.settings["adapter"] == metadata["adapter_sha256"]
    backend.model = None

    assert [record.examples for record in ledger_records(context)] == [16, 16]
    assert holds_example_text(run_dir) == [] and holds_example_text(adapter_dir) == []
