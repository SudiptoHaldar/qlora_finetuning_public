"""Tests for qlora_finetuning.local_qlora.training, with a stand-in engine and no model."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from qlora_finetuning.local_qlora import training
from qlora_finetuning.local_qlora.adapters import (
    ADAPTER_WEIGHTS,
    CHECKPOINTS,
    METADATA_NAME,
    TRAINING_NAME,
    adapter_sha256,
)
from qlora_finetuning.local_qlora.render import LengthError
from qlora_finetuning.local_qlora.training import (
    LOG_NAME,
    MEASUREMENTS_NAME,
    ResumeError,
    TrainingError,
    TrainingStopped,
    prepare,
    run_training,
    trainer_arguments,
)
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking import load_manifest
from tests.evaluation.conftest import ledger_records, run_folders

from .conftest import OutOfMemoryError
from .training_helpers import (
    StubEngine,
    StubReloader,
    holds_example_text,
    make_training_context,
    with_training,
)

MODEL = "Qwen/Qwen3-4B"


@pytest.fixture
def context(tmp_path):
    return make_training_context(tmp_path)


def smoke(context, **options):
    options.setdefault("engine", StubEngine())
    options.setdefault("reloader", StubReloader())
    return run_training(smoke=True, context=context, **options)


def log_of(run_id, context):
    path = context.tracking_cfg.runs_dir / run_id / LOG_NAME
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read(run_id, name, context):
    return json.loads((context.tracking_cfg.runs_dir / run_id / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- preparing


def test_prepare_encodes_fit_and_plans_without_a_model(context):
    prepared = prepare(context.resolved(), smoke=False)
    assert len(prepared.rows) == 30 and len(prepared.dev_rows) == 8
    plan = prepared.plan
    assert (plan.records, plan.steps_per_epoch, plan.total_steps) == (30, 2, 6)
    assert plan.tokens_per_pass == sum(row.length for row in prepared.rows)
    assert plan.checkpoint_steps == (6,)  # save_steps is 100
    assert prepared.key["model_id"] == MODEL and prepared.key["smoke"] is False
    assert prepared.key["prompt_id"] == context.prompt.prompt_id
    assert re.fullmatch(r"[0-9a-f]{64}", prepared.key["selection_sha256"])


def test_a_smoke_run_takes_the_longest_records_first_and_a_few_dev_records(context):
    context = with_training(
        context, smoke=dataclasses.replace(context.training_cfg.smoke, examples=10, dev_examples=3)
    )
    full = prepare(context.resolved(), smoke=False)
    prepared = prepare(context.resolved(), smoke=True)
    assert len(prepared.rows) == 10 and len(prepared.dev_rows) == 3
    longest = sorted((row.length for row in full.rows), reverse=True)[:2]
    assert [row.length for row in prepared.rows[:2]] == longest
    assert prepared.key["selection_sha256"] != full.key["selection_sha256"]
    assert prepared.key["settings_sha256"] != full.key["settings_sha256"]


def test_without_dev_loss_no_dev_record_is_encoded(context):
    prepared = prepare(with_training(context, dev_loss=False).resolved(), smoke=False)
    assert prepared.dev_rows == []


def test_a_record_over_the_training_length_stops_everything_before_a_run(context):
    short = dataclasses.replace(context, budget=dataclasses.replace(context.budget, max_length=40))
    engine = StubEngine()
    with pytest.raises(LengthError, match="Nothing is truncated") as caught:
        run_training(context=short, engine=engine)
    assert "30 training sequences are longer than max_length 40" in str(caught.value)
    assert engine.jobs == [] and run_folders(context) == [] and ledger_records(context) == []


# --------------------------------------------------------------------------- a run


def test_a_smoke_run_leaves_a_run_an_adapter_and_passing_checks(context):
    engine, reloader = StubEngine(), StubReloader()
    result = smoke(context, engine=engine, reloader=reloader)
    run_id = result["run_id"]
    assert result["passed"] and [check["id"] for check in result["checks"]] == [
        "K1", "K2", "K3", "K4", "K5", "K6", "K7",
    ]  # fmt: skip
    assert all(check["passed"] for check in result["checks"])
    assert (result["kind"], result["steps"], result["start_step"]) == ("smoke", 6, 0)
    assert result["plan"]["checkpoint_steps"] == [4, 6]

    manifest = load_manifest(run_id, context.tracking_cfg)
    assert (manifest["kind"], manifest["path"], manifest["arm"]) == ("smoke", "A", "none")
    assert (manifest["status"], manifest["where"], manifest["scope"]) == (
        "completed",
        "local",
        "dev",
    )
    assert set(manifest["configs"]) == {"training", "local_qlora", "token_budget", "prompt"}
    assert manifest["outputs"]["adapter"] == run_id
    assert manifest["split_manifest_hash"] == result_key(run_id, context)["manifest_hash"]

    folder = context.tracking_cfg.runs_dir / run_id
    assert sorted(path.name for path in (folder / CHECKPOINTS).iterdir()) == [
        "checkpoint-4",
        "checkpoint-6",
    ]
    assert [entry["step"] for entry in log_of(run_id, context) if "loss" in entry] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert [entry["step"] for entry in log_of(run_id, context) if "eval_loss" in entry] == [4, 6]
    measurements = read(run_id, MEASUREMENTS_NAME, context)
    assert measurements["steps_done"] == 6 and measurements["probe"]["width"] > 0
    assert measurements["reload"]["programs_that_run"] == 3

    adapter_dir = Path(result["adapter_dir"])
    assert adapter_dir == context.training_cfg.adapters_dir / run_id
    assert (adapter_dir / ADAPTER_WEIGHTS).read_bytes() == b"weights after step 6"
    assert reloader.calls == [adapter_dir]
    metadata = json.loads((adapter_dir / METADATA_NAME).read_text(encoding="utf-8"))
    assert metadata["adapter_sha256"] == adapter_sha256(adapter_dir) == result["adapter_sha256"]
    assert (metadata["run_id"], metadata["steps"], metadata["smoke"]) == (run_id, 6, True)
    assert (metadata["model_id"], metadata["continues"]) == (MODEL, None)
    assert metadata["prompt_id"] == context.prompt.prompt_id
    assert set(metadata["libraries"]) >= {"torch", "transformers", "peft", "trl"}

    # Ids, counts, and losses only: nothing of the examples themselves.
    assert holds_example_text(folder) == [] and holds_example_text(adapter_dir) == []


def result_key(run_id, context):
    return read(run_id, TRAINING_NAME, context)["key"]


def test_the_ledger_gets_the_time_the_records_and_the_training_tokens(context):
    result = smoke(context)
    (record,) = ledger_records(context)
    plan = result["plan"]
    assert (record.phase, record.path, record.arm, record.where) == ("train", "A", "none", "local")
    assert (record.model_id, record.source, record.examples) == (MODEL, "computed", 30)
    assert record.training_tokens == plan["tokens_per_pass"] * 3  # every planned step was made
    assert record.wall_clock_seconds >= 0 and record.input_tokens is None
    assert "6 of 6" in record.source_detail


def test_a_full_run_is_arm_a1_and_is_not_reloaded(context):
    reloader = StubReloader()
    result = run_training(context=context, engine=StubEngine(), reloader=reloader)
    manifest = load_manifest(result["run_id"], context.tracking_cfg)
    assert (manifest["kind"], manifest["arm"]) == ("train", "A1")
    assert [check["id"] for check in result["checks"]] == ["K1", "K2", "K3", "K4", "K5"]
    assert result["passed"] and reloader.calls == []
    metadata = json.loads((Path(result["adapter_dir"]) / METADATA_NAME).read_text(encoding="utf-8"))
    assert metadata["smoke"] is False and metadata["epochs"] == 3


def test_the_engine_is_given_the_rows_the_plan_and_the_settings(context):
    engine = StubEngine()
    result = smoke(context, engine=engine)
    (job,) = engine.jobs
    assert len(job.rows) == 30 and len(job.dev_rows) == 8
    assert (job.model_id, job.smoke, job.start_step, job.resume_checkpoint) == (
        MODEL,
        True,
        0,
        None,
    )
    assert job.checkpoints_dir == context.tracking_cfg.runs_dir / result["run_id"] / CHECKPOINTS
    assert job.plan.total_steps == 6


# --------------------------------------------------------------------------- failures


def test_a_failure_keeps_checkpoints_and_records_what_was_consumed(context):
    engine = StubEngine(fail_at=6)
    with pytest.raises(TrainingError) as caught:
        smoke(context, engine=engine)
    run_id = caught.value.run_id
    assert str(caught.value) == "RuntimeError after 5 of 6 steps"  # none of the library's text
    manifest = load_manifest(run_id, context.tracking_cfg)
    assert manifest["status"] == "failed" and "adapter" not in manifest["outputs"]
    folder = context.tracking_cfg.runs_dir / run_id
    assert (folder / CHECKPOINTS / "checkpoint-4" / ADAPTER_WEIGHTS).is_file()
    assert read(run_id, MEASUREMENTS_NAME, context)["steps_done"] == 5
    (record,) = ledger_records(context)
    plan_tokens = read(run_id, TRAINING_NAME, context)["plan"]["tokens_per_pass"] * 3
    assert record.training_tokens == round(plan_tokens * 5 / 6)
    assert not (context.training_cfg.adapters_dir / run_id).exists()
    assert holds_example_text(folder) == []


def test_out_of_memory_says_what_to_do(context):
    engine = StubEngine(fail_at=1, error=OutOfMemoryError("CUDA out of memory"))
    with pytest.raises(TrainingError, match="ran out of memory.*Nothing is truncated"):
        smoke(context, engine=engine)
    (record,) = ledger_records(context)
    assert record.training_tokens == 0 and record.examples == 30


def test_a_trainer_that_ends_before_the_planned_step_fails_the_run(context):
    with pytest.raises(TrainingError, match="ended at step 5, not at the planned 6") as caught:
        smoke(context, engine=StubEngine(end_early=True))
    assert load_manifest(caught.value.run_id, context.tracking_cfg)["status"] == "failed"


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_stop_after_steps_must_be_a_positive_whole_number(context, steps):
    with pytest.raises(ConfigError, match="stop_after_steps"):
        smoke(context, stop_after_steps=steps)
    assert run_folders(context) == []


# --------------------------------------------------------------------------- the smoke checks


def check(result, check_id):
    return next(item for item in result["checks"] if item["id"] == check_id)


def test_k6_fails_when_the_loss_does_not_fall(context):
    result = smoke(context, engine=StubEngine(losses=lambda step: 2.0))
    assert not result["passed"] and not check(result, "K6")["passed"]
    assert "must be under 0.5 times the first" in check(result, "K6")["detail"]
    # The run itself completed: the adapter is there.
    assert (Path(result["adapter_dir"]) / ADAPTER_WEIGHTS).is_file()


@pytest.mark.parametrize(
    ("reloader", "message"),
    [
        (StubReloader(error=RuntimeError("zebrafish")), "RuntimeError while reloading"),
        (StubReloader(adapter_sha256="0" * 64), "hash is not the metadata's"),
        (StubReloader(replies=2), "2 replies for 3 prompts"),
        (StubReloader(think_markers=1), "1 replies hold a thinking marker"),
    ],
)
def test_k7_fails_when_the_adapter_does_not_reload_cleanly(context, reloader, message):
    result = smoke(context, reloader=reloader)
    assert not result["passed"] and message in check(result, "K7")["detail"]
    assert "zebrafish" not in json.dumps(result)


def test_k3_and_k5_report_what_the_engine_found(context):
    result = smoke(context, engine=StubEngine(unchanged=False))
    assert not check(result, "K3")["passed"]
    assert "peak GPU memory 1.5 GiB allocated, 1.75 reserved" in check(result, "K5")["detail"]
    assert "1,234 trainable parameters" in check(result, "K4")["detail"]


# --------------------------------------------------------------------------- resuming


def stopped_run(context, after=5):
    with pytest.raises(TrainingStopped) as caught:
        smoke(context, stop_after_steps=after)
    return caught.value.run_id


def test_a_stopped_run_is_continued_in_a_new_run_from_its_last_checkpoint(context):
    first = stopped_run(context)
    assert load_manifest(first, context.tracking_cfg)["status"] == "failed"
    engine = StubEngine()
    result = smoke(context, engine=engine, resume=first)
    second = result["run_id"]
    assert second != first and result["continues"] == first
    assert (result["start_step"], result["steps"], result["passed"]) == (4, 6, True)

    (job,) = engine.jobs
    assert job.start_step == 4
    assert job.resume_checkpoint == (
        context.tracking_cfg.runs_dir / first / CHECKPOINTS / "checkpoint-4"
    )
    # One log for the whole training: the earlier lines up to the checkpoint, then the new.
    steps = [entry["step"] for entry in log_of(second, context) if "loss" in entry]
    assert steps == [1, 2, 3, 4, 5, 6]
    assert read(second, TRAINING_NAME, context)["continues"] == first
    assert load_manifest(second, context.tracking_cfg)["description"] == f"continues {first}"
    metadata = json.loads((Path(result["adapter_dir"]) / METADATA_NAME).read_text(encoding="utf-8"))
    assert metadata["continues"] == first and metadata["steps"] == 6

    # Step 5 was made twice (once before the stop, once after), and both are recorded.
    first_record, second_record = ledger_records(context)
    one_step = result["plan"]["tokens_per_pass"] * 3 / 6
    assert first_record.training_tokens == round(5 * one_step)
    assert second_record.training_tokens == round(2 * one_step)


def test_resume_is_refused_under_other_conditions_and_starts_no_run(context, tmp_path):
    first = stopped_run(context)
    before = run_folders(context)
    cases = {
        "training settings": with_training(context, learning_rate=1e-4),
        "prompt wording": dataclasses.replace(
            context, prompt=dataclasses.replace(context.prompt, prompt_id="f" * 64)
        ),
        "commit of the base model": dataclasses.replace(context, revision="a" * 40),
    }
    for words, other in cases.items():
        engine = StubEngine()
        with pytest.raises(ResumeError, match=f"another {words}"):
            run_training(smoke=True, resume=first, context=other, engine=engine)
        assert engine.jobs == []
    with pytest.raises(ResumeError, match="kind of run"):
        run_training(resume=first, context=context, engine=StubEngine())
    assert run_folders(context) == before and len(ledger_records(context)) == 1


def test_resume_needs_a_training_run_with_a_checkpoint(context):
    early = stopped_run(context, after=2)  # stopped before the first checkpoint
    with pytest.raises(ResumeError, match="has no checkpoint"):
        smoke(context, resume=early)
    with pytest.raises(ResumeError, match="not a training run on this machine"):
        smoke(context, resume="20261119T000000Z-smoke-none-0000")
    for bad in ("", "../elsewhere", "a/b"):
        with pytest.raises(ResumeError, match="is not a run id"):
            smoke(context, resume=bad)
    done = smoke(context)["run_id"]
    with pytest.raises(ResumeError, match="already reached its last step"):
        smoke(context, resume=done)


# --------------------------------------------------------------------------- trainer arguments


def job_for(context, smoke_run=False):
    engine = StubEngine()
    run_training(smoke=smoke_run, context=context, engine=engine, reloader=StubReloader())
    return engine.jobs[0]


def test_every_trainer_argument_comes_from_the_config_and_nothing_is_cut(context):
    arguments = trainer_arguments(job_for(context))
    assert arguments["max_length"] is None  # the trainer's own default would cut rows
    assert arguments["packing"] is False and arguments["padding_free"] is False
    expected = {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 3,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit",
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "seed": 42,
        "data_seed": 42,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": None,
        "eval_strategy": "steps",
        "eval_steps": 100,
        "report_to": "none",
        "warmup_steps": 1,  # 5% of 6 steps, rounded up
    }
    assert {name: arguments[name] for name in expected} == expected
    assert "warmup_ratio" not in arguments and "group_by_length" not in arguments


def test_a_smoke_run_and_a_run_without_dev_loss_change_the_arguments(context):
    smoke_arguments = trainer_arguments(job_for(context, smoke_run=True))
    assert (smoke_arguments["save_steps"], smoke_arguments["logging_steps"]) == (4, 1)
    plain = trainer_arguments(job_for(with_training(context, dev_loss=False, precision="fp16")))
    assert plain["eval_strategy"] == "no" and "eval_steps" not in plain
    assert (plain["bf16"], plain["fp16"]) == (False, True)


# --------------------------------------------------------------------------- guards


def test_training_never_asks_for_the_test_split():
    source = Path(training.__file__).read_text(encoding="utf-8")
    assert '"test"' not in source and "'test'" not in source
    assert set(re.findall(r'_records\("(\w+)"', source)) == {"fit", "dev"}
