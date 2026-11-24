"""Tests for qlora_finetuning.local_qlora.training_config and plan."""

from __future__ import annotations

import dataclasses

import pytest
import yaml

from qlora_finetuning.local_qlora.plan import longest, select_smoke, training_plan
from qlora_finetuning.local_qlora.training_config import (
    REQUIRED_KEYS,
    load_training_config,
    sha256_of,
)
from qlora_finetuning.settings import CONFIG_DIR, REPO_ROOT, ConfigError


def write_config(tmp_path, change=None):
    raw = yaml.safe_load((CONFIG_DIR / "training.yaml").read_text(encoding="utf-8"))
    if change:
        change(raw)
    (tmp_path / "training.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------- the config


def test_the_committed_file_holds_the_plans_defaults():
    cfg = load_training_config()
    assert (cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout, cfg.target_modules) == (
        16, 32, 0.05, "all-linear",
    )  # fmt: skip
    assert (cfg.optimizer, cfg.learning_rate, cfg.weight_decay, cfg.max_grad_norm) == (
        "paged_adamw_8bit", 2e-4, 0.0, 1.0,
    )  # fmt: skip
    assert (cfg.schedule, cfg.warmup_fraction) == ("cosine", 0.05)
    assert (cfg.batch_size, cfg.accumulation, cfg.epochs, cfg.seed) == (2, 8, 3, 42)
    assert (cfg.precision, cfg.gradient_checkpointing) == ("bf16", True)
    assert (cfg.save_steps, cfg.dev_loss, cfg.logging_steps) == (100, True, 10)
    assert cfg.adapters_dir == REPO_ROOT / "artifacts" / "adapters"
    assert (cfg.smoke.examples, cfg.smoke.epochs, cfg.smoke.save_steps) == (64, 3, 4)
    assert (cfg.smoke.generate_examples, cfg.smoke.generate_batch_size) == (24, 8)


def test_a_smoke_run_has_its_own_sizes_and_the_same_hyperparameters():
    cfg = load_training_config()
    assert (cfg.epochs_for(True), cfg.save_steps_for(True), cfg.logging_steps_for(True)) == (
        3,
        4,
        1,
    )
    assert (cfg.save_steps_for(False), cfg.logging_steps_for(False)) == (100, 10)
    full, smoke = cfg.settings(False), cfg.settings(True)
    for name in ("lora", "optimizer", "schedule", "batch", "seed", "precision"):
        assert full[name] == smoke[name]
    assert (full["smoke_examples"], smoke["smoke_examples"]) == (None, 64)


def test_the_settings_hash_is_stable_and_follows_every_value(tmp_path):
    cfg = load_training_config()
    assert cfg.settings_sha256(False) == sha256_of(cfg.settings(False))
    assert cfg.settings_sha256(False) == load_training_config().settings_sha256(False)
    assert cfg.settings_sha256(False) != cfg.settings_sha256(True)
    changed = dataclasses.replace(cfg, learning_rate=1e-4)
    assert changed.settings_sha256(False) != cfg.settings_sha256(False)
    # Where adapters are kept does not decide the weights.
    moved = load_training_config(root=tmp_path)
    assert moved.adapters_dir == tmp_path / "artifacts" / "adapters"
    assert moved.settings_sha256(False) == cfg.settings_sha256(False)


def test_target_modules_may_be_a_list(tmp_path):
    def change(raw):
        raw["lora"]["target_modules"] = ["q_proj", "v_proj"]

    cfg = load_training_config(config_dir=write_config(tmp_path, change))
    assert cfg.target_modules == ("q_proj", "v_proj")
    assert cfg.settings(False)["lora"]["target_modules"] == ["q_proj", "v_proj"]


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_a_missing_key_is_named(tmp_path, key):
    def drop(raw):
        *parents, last = key.split(".")
        for parent in parents:
            raw = raw[parent]
        del raw[last]

    with pytest.raises(ConfigError, match=key.replace(".", r"\.")):
        load_training_config(config_dir=write_config(tmp_path, drop))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("lora", "r"), 0),
        (("lora", "r"), 2.5),
        (("lora", "dropout"), 0.9),
        (("lora", "target_modules"), []),
        (("optimizer", "name"), "sgd"),
        (("optimizer", "learning_rate"), 0),
        (("optimizer", "learning_rate"), "fast"),
        (("optimizer", "max_grad_norm"), 0),
        (("schedule", "type"), "step"),
        (("schedule", "warmup_fraction"), 0.9),
        (("batch", "size"), 0),
        (("batch", "accumulation"), True),
        (("epochs",), 0),
        (("seed",), -1),
        (("precision",), "fp32"),
        (("gradient_checkpointing",), "yes"),
        (("checkpoint", "save_steps"), 0),
        (("checkpoint", "dev_loss"), 1),
        (("logging_steps",), 0),
        (("smoke", "examples"), 1),  # cannot fill a batch of 2
        (("smoke", "generate_batch_size"), 1),
        (("smoke", "max_final_loss_ratio"), 0),
        (("smoke", "max_final_loss_ratio"), 1.5),
    ],
)
def test_a_value_out_of_range_is_refused(tmp_path, path, value):
    def change(raw):
        *parents, last = path
        for parent in parents:
            raw = raw[parent]
        raw[last] = value

    with pytest.raises(ConfigError, match=path[-2] if path[-1] == "examples" else path[-1]):
        load_training_config(config_dir=write_config(tmp_path, change))


# --------------------------------------------------------------------------- the plan


def test_the_plan_of_the_full_run():
    """4,884 records in batches of 2, 8 batches a step: the trainer's own arithmetic."""
    cfg = load_training_config()
    lengths = [650] * 4882 + [1862, 1851]
    plan = training_plan(lengths, cfg, smoke=False)
    assert (plan.records, plan.steps_per_epoch, plan.total_steps) == (4884, 306, 918)
    assert plan.warmup_steps == 46  # 5% of 918, rounded up
    assert plan.checkpoint_steps == (100, 200, 300, 400, 500, 600, 700, 800, 900, 918)
    assert plan.worst_batch == (4882, 4883) and plan.worst_batch_width == 1862
    assert plan.tokens_per_pass == sum(lengths)
    assert plan.to_dict()["checkpoint_steps"][-1] == 918


def test_the_plan_of_the_smoke_run():
    cfg = load_training_config()
    plan = training_plan([600] * 64, cfg, smoke=True)
    assert (plan.steps_per_epoch, plan.total_steps, plan.warmup_steps) == (4, 12, 1)
    assert plan.checkpoint_steps == (4, 8, 12) and plan.save_steps == 4


def test_a_last_smaller_group_of_batches_is_a_step_and_the_last_step_is_a_checkpoint():
    cfg = dataclasses.replace(load_training_config(), epochs=2, save_steps=3)
    plan = training_plan([10] * 35, cfg, smoke=False)  # 18 batches: 2 full steps and one of 2
    assert (plan.steps_per_epoch, plan.total_steps) == (3, 6)
    assert plan.checkpoint_steps == (3, 6)
    odd = training_plan([10] * 35, dataclasses.replace(cfg, save_steps=4), smoke=False)
    assert odd.checkpoint_steps == (4, 6)
    none_due = training_plan([10] * 35, dataclasses.replace(cfg, save_steps=50), smoke=False)
    assert none_due.checkpoint_steps == (6,)


def test_tokens_for_a_share_of_the_steps():
    cfg = load_training_config()
    plan = training_plan([100] * 64, cfg, smoke=True)  # 6,400 tokens a pass, 3 epochs, 12 steps
    assert plan.tokens_for_steps(12) == 19_200
    assert plan.tokens_for_steps(6) == 9_600 and plan.tokens_for_steps(0) == 0
    assert plan.tokens_for_steps(99) == 19_200  # never more than the plan


def test_the_worst_batch_and_the_smoke_selection():
    lengths = [5, 9, 3, 9, 7, 1]
    assert longest(lengths, 2) == (1, 3)  # a tie goes to the earlier record
    assert longest(lengths, 3) == (1, 3, 4)
    assert select_smoke(lengths, examples=4, batch_size=2) == [1, 3, 0, 2]
    assert select_smoke(lengths, examples=99, batch_size=2) == [1, 3, 0, 2, 4, 5]
    assert select_smoke(lengths, examples=2, batch_size=2) == [1, 3]


def test_fewer_records_than_a_batch_cannot_be_planned():
    with pytest.raises(ValueError, match="cannot fill a batch"):
        training_plan([10], load_training_config(), smoke=False)
