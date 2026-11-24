"""QLoRA training of the pinned model on the ``fit`` records (req_003 v3.02).

One call is one run in the registry. It encodes every record as the sequence the token
budget counted (the prompt, the reply, the end token; the loss on the reply only), probes
the worst batch, trains a LoRA adapter on the 4-bit model with the settings of
``config/training.yaml``, saves a checkpoint every ``save_steps`` optimizer steps, and
leaves the final adapter in ``artifacts/adapters/<run_id>/`` with its metadata.

Usage::

    from qlora_finetuning.local_qlora.training import run_training

    result = run_training(smoke=True)             # 64 records, a few minutes on a GPU
    result["run_id"], result["steps"], result["adapter"], result["passed"]
    result["checks"]                              # K1 to K7 for a smoke run
    result["measurements"]["probe"]["peak_vram_allocated_gib"]

    run_training()                                # the full run
    run_training(resume="<run_id>")               # a new run from that run's last checkpoint

A run cannot be reopened, so resuming is a new run that names the one it continues; it is
refused unless the model, the prompt, the split, the settings, and the records are the same.
Nothing is ever truncated: a record over the training length is a ``LengthError`` before the
model is loaded. Checkpoints are scored afterwards, through the evaluation runner
(``scripts.evaluate --predictor qwen3-4b-tuned --adapter <run_id>[/checkpoint-N]``).

This module and ``backend.py`` are the only ones of Path A that import the training stack,
and only inside functions. A test passes a stand-in ``engine`` and needs no model. No file
written here holds example text: ids, counts, and losses only.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import math
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from qlora_finetuning.data import build_record, load_prompt, load_split, manifest_hash
from qlora_finetuning.data.token_budget import (
    count_split,
    load_token_budget_config,
    load_tokenizer,
)
from qlora_finetuning.environment import installed_versions
from qlora_finetuning.local_qlora.adapters import (
    ADAPTER_FILES,
    ADAPTER_WEIGHTS,
    CHECKPOINTS,
    METADATA_NAME,
    TRAINING_NAME,
    adapter_sha256,
    read_json,
    write_json,
)
from qlora_finetuning.local_qlora.config import load_local_qlora_config
from qlora_finetuning.local_qlora.plan import TrainingPlan, select_smoke, training_plan
from qlora_finetuning.local_qlora.render import IGNORE, EncodedExample, encode_records
from qlora_finetuning.local_qlora.training_config import (
    TrainingConfig,
    load_training_config,
    sha256_of,
)
from qlora_finetuning.logging_setup import get_logger
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking import load_tracking_config, start_run
from qlora_finetuning.tracking.config import timestamp

SCHEMA = 1
LOG_NAME = "training_log.jsonl"
MEASUREMENTS_NAME = "measurements.json"
TRAINER_STATE = "trainer_state.json"  # what a checkpoint needs to be resumed from
PHASE = "train"
GIB = 2**30
# Every config file a run reads; each is copied into the run folder and hashed.
RUN_CONFIGS = ("training", "local_qlora", "token_budget", "prompt", "split", "data")
LIBRARIES = ("torch", "transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes")
LOG_FIELDS = ("epoch", "loss", "learning_rate", "grad_norm", "eval_loss")
PASS, FAIL = "PASS", "FAIL"
KEY_WORDS = {
    "model_id": "base model",
    "revision": "commit of the base model",
    "prompt_id": "prompt wording",
    "manifest_hash": "split",
    "settings_sha256": "training settings",
    "selection_sha256": "selection of records",
    "smoke": "kind of run (smoke or full)",
}

logger = get_logger(__name__)


class TrainingError(Exception):
    """A training run failed. ``run_id`` names the run, so it can be resumed."""

    run_id: str | None = None


class TrainingStopped(TrainingError):
    """The run was stopped on request (``stop_after_steps``): a rehearsed interruption."""


class ResumeError(ConfigError):
    """An earlier run cannot be continued: other settings, or no checkpoint to start from."""


@dataclass
class TrainingContext:
    """Everything a run reads from the project. ``None`` means "load the real one".

    A test fills it with a temporary split, a temporary tracking root, a stand-in tokenizer,
    a fixed git state, and an empty environment, so nothing is written to the repository.
    ``model_id`` and ``revision`` default to the pin of ``config/token_budget.yaml``.
    """

    training_cfg: TrainingConfig | None = None
    local_cfg: Any = None
    budget: Any = None
    prompt: Any = None
    split_cfg: Any = None
    data_cfg: Any = None
    tracking_cfg: Any = None
    tokenizer: Any = None
    model_id: str | None = None
    revision: str | None = None
    git: Any = None
    environment: dict[str, Any] | None = None
    config_dir: Path | str | None = None
    configs: tuple[str, ...] = RUN_CONFIGS

    def resolved(self) -> TrainingContext:
        budget = self.budget or load_token_budget_config()
        return dataclasses.replace(
            self,
            training_cfg=self.training_cfg or load_training_config(),
            local_cfg=self.local_cfg or load_local_qlora_config(),
            budget=budget,
            prompt=self.prompt or load_prompt(),
            tracking_cfg=self.tracking_cfg or load_tracking_config(),
            model_id=self.model_id or budget.model_id,
            revision=self.revision or budget.revision,
        )


@dataclass
class TrainingJob:
    """What an engine is asked to do. ``progress`` is filled as it goes, also on a failure."""

    rows: list[EncodedExample]
    dev_rows: list[EncodedExample]
    plan: TrainingPlan
    cfg: TrainingConfig
    local_cfg: Any
    budget: Any
    model_id: str
    revision: str
    smoke: bool
    checkpoints_dir: Path
    resume_checkpoint: Path | None
    start_step: int
    stop_after_steps: int | None
    log: Callable[[dict[str, Any]], None]
    progress: dict[str, Any] = field(default_factory=dict)


class Engine(Protocol):
    """The part that needs the model. ``TrlEngine`` is the real one."""

    def fit(self, job: TrainingJob) -> None:
        """Train. Fill ``job.progress``; raise on any failure, leaving checkpoints behind."""
        ...


# --------------------------------------------------------------------------- preparing


def _records(split: str, ctx: TrainingContext) -> list[Any]:
    return [
        build_record(example, ctx.prompt)
        for example in load_split(split, ctx.split_cfg, ctx.data_cfg)
    ]


def _labels_ok(example: EncodedExample) -> bool:
    """Exactly the reply (the target and the end token) carries a label."""
    cut = example.prompt_tokens
    return (
        0 < cut < example.length
        and len(example.labels) == example.length
        and all(label == IGNORE for label in example.labels[:cut])
        and example.labels[cut:] == example.input_ids[cut:]
    )


@dataclass
class Prepared:
    rows: list[EncodedExample]
    dev_rows: list[EncodedExample]
    plan: TrainingPlan
    key: dict[str, Any]
    tokens_checked: int


def prepare(ctx: TrainingContext, smoke: bool) -> Prepared:
    """Encode the records and plan the run. Needs the tokenizer; no model and no GPU.

    Raises ``LengthError`` for a record over the training length, and ``ConfigError`` if the
    sequences are not the ones the token budget counted.
    """
    cfg, budget = ctx.training_cfg, ctx.budget
    tokenizer = ctx.tokenizer or load_tokenizer(budget)
    suffix = budget.thinking_off_suffix

    records = _records("fit", ctx)
    rows = encode_records(records, tokenizer, suffix, budget.max_length)
    if smoke:
        positions = select_smoke([row.length for row in rows], cfg.smoke.examples, cfg.batch_size)
        records, rows = [records[p] for p in positions], [rows[p] for p in positions]

    counts = count_split(records, tokenizer, suffix)
    differ = [
        row.example_id
        for row, count in zip(rows, counts, strict=True)
        if row.length != count.total or row.target_tokens != count.target
    ]
    if differ:
        raise ConfigError(
            f"{len(differ):,} training sequences do not have the token budget's length "
            f"(first: {differ[0]}): the encoder and the token budget disagree"
        )
    unlabelled = [row.example_id for row in rows if not _labels_ok(row)]
    if unlabelled:
        raise ConfigError(
            f"{len(unlabelled):,} training sequences do not have their labels on the reply "
            f"alone (first: {unlabelled[0]})"
        )

    dev_rows: list[EncodedExample] = []
    if cfg.dev_loss:
        dev_records = _records("dev", ctx)
        if smoke:
            dev_records = dev_records[: cfg.smoke.dev_examples]
        dev_rows = encode_records(dev_records, tokenizer, suffix, budget.max_length)

    plan = training_plan([row.length for row in rows], cfg, smoke)
    key = {
        "model_id": ctx.model_id,
        "revision": ctx.revision,
        "prompt_id": ctx.prompt.prompt_id,
        "manifest_hash": manifest_hash(ctx.split_cfg),
        "settings_sha256": cfg.settings_sha256(smoke),
        "selection_sha256": sha256_of([row.example_id for row in rows]),
        "smoke": smoke,
    }
    return Prepared(rows, dev_rows, plan, key, sum(row.length for row in rows))


# --------------------------------------------------------------------------- resuming


def resumable_steps(checkpoints_dir: Path) -> list[int]:
    """The steps whose checkpoint holds the trainer's state as well as the adapter."""
    steps = []
    if checkpoints_dir.is_dir():
        for folder in checkpoints_dir.iterdir():
            name, _, number = folder.name.partition("-")
            if (
                name == "checkpoint"
                and number.isdigit()
                and (folder / TRAINER_STATE).is_file()
                and (folder / ADAPTER_WEIGHTS).is_file()
            ):
                steps.append(int(number))
    return sorted(steps)


def _check_resume(resume: str, prepared: Prepared, ctx: TrainingContext) -> tuple[Path, int]:
    """The checkpoint an earlier run is continued from, and its step. Raises ``ResumeError``."""
    if not isinstance(resume, str) or not resume or any(c in resume for c in ("..", "/", "\\")):
        raise ResumeError(f"{resume!r} is not a run id")
    folder = ctx.tracking_cfg.runs_dir / resume
    try:
        earlier = read_json(folder / TRAINING_NAME, "the training run's record")
    except ConfigError as error:
        raise ResumeError(
            f"run {resume} cannot be continued: it is not a training run on this machine ({error})"
        ) from error
    differs = [
        words
        for name, words in KEY_WORDS.items()
        if (earlier.get("key") or {}).get(name) != prepared.key[name]
    ]
    if differs:
        raise ResumeError(
            f"run {resume} cannot be continued: it was made with another "
            f"{', another '.join(differs)}. Start a new run instead"
        )
    steps = resumable_steps(folder / CHECKPOINTS)
    if not steps:
        raise ResumeError(
            f"run {resume} cannot be continued: it has no checkpoint. Start a new run instead"
        )
    step = steps[-1]
    if step >= prepared.plan.total_steps:
        raise ResumeError(f"run {resume} already reached its last step ({step:,})")
    return folder / CHECKPOINTS / f"checkpoint-{step}", step


def _read_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue  # a line cut off by a killed process
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _append_log(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=True, allow_nan=False) + "\n")
        handle.flush()


# --------------------------------------------------------------------------- checks


def _check(check_id: str, name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"id": check_id, "name": name, "passed": bool(passed), "detail": detail}


def _loss_check(losses: Sequence[float], ratio: float) -> dict[str, Any]:
    if len(losses) < 4:
        return _check("K6", "Loss falls", False, f"only {len(losses)} losses were logged")
    last = losses[-max(len(losses) // 4, 1) :]
    mean_last = math.fsum(last) / len(last)
    passed = all(math.isfinite(x) for x in losses) and mean_last < ratio * losses[0]
    return _check(
        "K6",
        "Loss falls",
        passed,
        f"first {losses[0]:.3f}, mean of the last {len(last)} {mean_last:.3f} "
        f"(must be under {ratio:g} times the first)",
    )


def _reload_check(reload: dict[str, Any] | None, error: str | None, sha256: str) -> dict[str, Any]:
    name = "Adapter reloads"
    if reload is None:
        return _check("K7", name, False, error or "the adapter was not reloaded")
    problems = []
    if reload.get("adapter_sha256") != sha256:
        problems.append("the loaded adapter's hash is not the metadata's")
    if reload.get("replies") != reload.get("asked"):
        problems.append(f"{reload.get('replies')} replies for {reload.get('asked')} prompts")
    if reload.get("think_markers"):
        problems.append(f"{reload['think_markers']} replies hold a thinking marker")
    detail = "; ".join(problems) or (
        f"{reload['asked']} prompts answered; {reload['programs_that_run']} programs run "
        f"without an error; {reload['identical_replies']} replies identical one at a time "
        f"and in batches of {reload['batch_size']}"
    )
    return _check("K7", name, not problems, detail)


# --------------------------------------------------------------------------- the run


def run_training(
    *,
    smoke: bool = False,
    resume: str | None = None,
    stop_after_steps: int | None = None,
    where: str = "local",
    scope: str = "dev",
    label: str | None = None,
    context: TrainingContext | None = None,
    engine: Engine | None = None,
    reloader: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Train an adapter; return what the run did.

    ``smoke`` trains on a few records and then reloads the adapter and asks it (checks K6
    and K7). ``resume=<run_id>`` continues an earlier run from its last checkpoint, in a new
    run. ``stop_after_steps=N`` fails the run after N optimizer steps, to rehearse an
    interruption.

    Raises ``ConfigError`` (a ``LengthError`` or ``ResumeError`` among them) before any run
    exists, and ``TrainingError`` with ``run_id`` set when a run fails: its checkpoints are
    kept and its usage is recorded.
    """
    ctx = (context or TrainingContext()).resolved()
    cfg = ctx.training_cfg
    if stop_after_steps is not None and (
        isinstance(stop_after_steps, bool)
        or not isinstance(stop_after_steps, int)
        or stop_after_steps < 1
    ):
        raise ConfigError(f"stop_after_steps must be a positive whole number: {stop_after_steps!r}")

    prepared = prepare(ctx, smoke)
    plan = prepared.plan
    resume_checkpoint, start_step = (None, 0)
    if resume is not None:
        resume_checkpoint, start_step = _check_resume(resume, prepared, ctx)
    engine = engine or TrlEngine()

    kind = "smoke" if smoke else "train"
    with start_run(
        kind,
        label or f"qwen3-4b {'smoke' if smoke else 'fit'}",
        path="A",
        arm="none" if smoke else "A1",
        where=where,
        scope=scope,
        configs=ctx.configs,
        split_manifest_hash=prepared.key["manifest_hash"],
        description=f"continues {resume}" if resume else "",
        cfg=ctx.tracking_cfg,
        config_dir=ctx.config_dir,
        git=ctx.git,
        environment=ctx.environment,
    ) as run:
        log_path = run.run_dir / LOG_NAME
        write_json(
            run.run_dir / TRAINING_NAME,
            {
                "schema": SCHEMA,
                "run_id": run.run_id,
                "smoke": smoke,
                "continues": resume,
                "start_step": start_step,
                "key": prepared.key,
                "settings": cfg.settings(smoke),
                "plan": plan.to_dict(),
                "dev_records": len(prepared.dev_rows),
            },
        )
        run.outputs.update(training=TRAINING_NAME, log=LOG_NAME, checkpoints=CHECKPOINTS)
        if resume is not None:  # the earlier lines, so one log covers the whole training
            earlier_log = ctx.tracking_cfg.runs_dir / resume / LOG_NAME
            for entry in _read_log(earlier_log):
                if entry.get("step", 0) <= start_step:
                    _append_log(log_path, entry)

        job = TrainingJob(
            rows=prepared.rows,
            dev_rows=prepared.dev_rows,
            plan=plan,
            cfg=cfg,
            local_cfg=ctx.local_cfg,
            budget=ctx.budget,
            model_id=ctx.model_id,
            revision=ctx.revision,
            smoke=smoke,
            checkpoints_dir=run.run_dir / CHECKPOINTS,
            resume_checkpoint=resume_checkpoint,
            start_step=start_step,
            stop_after_steps=stop_after_steps,
            log=lambda entry: _append_log(log_path, entry),
            progress={"steps_done": start_step},
        )
        job.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        try:
            try:
                engine.fit(job)
            except TrainingError as error:
                error.run_id = run.run_id
                raise
            except Exception as error:
                # The library's own text goes to the log only.
                logger.error("training failed: %s: %s", type(error).__name__, error)
                kind_name = type(error).__name__
                advice = (
                    ": the GPU ran out of memory; close other programs that use it and "
                    "start again. Nothing is truncated to make a batch fit"
                    if "OutOfMemory" in kind_name
                    else ""
                )
                failure = TrainingError(
                    f"{kind_name} after {job.progress.get('steps_done', start_step):,} of "
                    f"{plan.total_steps:,} steps{advice}"
                )
                failure.run_id = run.run_id
                raise failure from error
            steps_done = int(job.progress.get("steps_done", start_step))
            if steps_done != plan.total_steps:
                failure = TrainingError(
                    f"training ended at step {steps_done:,}, "
                    f"not at the planned {plan.total_steps:,}"
                )
                failure.run_id = run.run_id
                raise failure
        finally:
            seconds = time.perf_counter() - started
            steps_done = int(job.progress.get("steps_done", start_step))
            steps_here = max(steps_done - start_step, 0)
            run.record(
                phase=PHASE,
                wall_clock_seconds=round(seconds, 3),
                examples=plan.records,
                training_tokens=plan.tokens_for_steps(steps_here),
                model_id=ctx.model_id,
                source="computed",
                source_detail=(
                    f"wall clock by time.perf_counter around the training; training tokens "
                    f"are the tokens of one pass over the records times the epochs, times "
                    f"the share of the planned optimizer steps this run made "
                    f"({steps_here} of {plan.total_steps})"
                ),
            )
            measurements = _plain(
                {
                    "steps_done": steps_done,
                    "steps_in_this_run": steps_here,
                    "wall_clock_seconds": round(seconds, 3),
                    **{
                        name: job.progress[name]
                        for name in ("trainable_parameters", "probe", "trainer")
                        if name in job.progress
                    },
                }
            )
            write_json(run.run_dir / MEASUREMENTS_NAME, measurements)
            run.outputs["measurements"] = MEASUREMENTS_NAME

        # Completed: the final adapter, its metadata, and the checks.
        final_checkpoint = Path(job.progress["final_checkpoint"])
        adapter_dir = cfg.adapters_dir / run.run_id
        adapter_dir.mkdir(parents=True, exist_ok=True)
        for name in ADAPTER_FILES:
            shutil.copyfile(final_checkpoint / name, adapter_dir / name)
        sha256 = adapter_sha256(adapter_dir)
        metadata = {
            "schema": SCHEMA,
            **prepared.key,
            "run_id": run.run_id,
            "continues": resume,
            "steps": steps_done,
            "epochs": plan.epochs,
            "records": plan.records,
            "tokens_per_pass": plan.tokens_per_pass,
            "trainable_parameters": job.progress.get("trainable_parameters"),
            "adapter_sha256": sha256,
            "created_at": timestamp(),
            "libraries": installed_versions(LIBRARIES),
        }
        write_json(adapter_dir / METADATA_NAME, metadata)
        run.outputs.update(adapter=run.run_id, adapter_sha256=sha256)

        probe = job.progress.get("probe") or {}
        checks = [
            _check(
                "K1",
                "Sequences",
                True,
                f"{plan.records:,} sequences of {plan.tokens_per_pass:,} tokens, each of the "
                f"token budget's length; the longest is {plan.worst_batch_width:,} of "
                f"{ctx.budget.max_length:,}",
            ),
            _check("K2", "Labels", True, "only the reply and its end token carry a label"),
            _check(
                "K3",
                "Trainer left the data alone",
                job.progress.get("dataset_unchanged") is True,
                f"the trainer's dataset has the pipeline's {plan.records:,} rows and lengths",
            ),
            _check(
                "K4",
                "Only the adapter trains",
                job.progress.get("only_adapter_trains") is True,
                f"{job.progress.get('trainable_parameters', 0):,} trainable parameters, all LoRA",
            ),
            _check(
                "K5",
                "Worst batch fits",
                bool(probe) and math.isfinite(probe.get("loss", math.nan)),
                (
                    f"{probe.get('sequences')} sequences, {probe.get('width'):,} tokens wide: "
                    f"peak GPU memory {probe.get('peak_vram_allocated_gib')} GiB allocated, "
                    f"{probe.get('peak_vram_reserved_gib')} reserved"
                    if probe
                    else "the probe did not run"
                ),
            ),
        ]
        reload_result = None
        if smoke:
            losses = [entry["loss"] for entry in _read_log(log_path) if "loss" in entry]
            checks.append(_loss_check(losses, cfg.smoke.max_final_loss_ratio))
            reload_error = None
            try:
                reload_result = _plain((reloader or reload_adapter)(ctx, adapter_dir))
            except Exception as error:  # the check fails; the trained adapter is kept
                logger.error(
                    "the adapter could not be reloaded: %s: %s", type(error).__name__, error
                )
                reload_error = f"{type(error).__name__} while reloading the adapter"
            checks.append(_reload_check(reload_result, reload_error, sha256))
            measurements["reload"] = reload_result
            write_json(run.run_dir / MEASUREMENTS_NAME, measurements)

    return {
        "run_id": run.run_id,
        "kind": kind,
        "smoke": smoke,
        "continues": resume,
        "start_step": start_step,
        "steps": steps_done,
        "plan": plan.to_dict(),
        "adapter": run.run_id,
        "adapter_dir": str(adapter_dir),
        "adapter_sha256": sha256,
        "measurements": measurements,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }


def _plain(value: Any) -> Any:
    """Plain values only, with no NaN: what may be written to a run's files."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


# --------------------------------------------------------------------------- the real engine


def trainer_arguments(job: TrainingJob) -> dict[str, Any]:
    """Every argument the trainer is given. Nothing is left to a library default.

    ``max_length`` is ``None`` on purpose: the trainer's own default (1,024) cuts rows of a
    prepared dataset without an error, and nothing in this project is truncated.
    """
    cfg, plan = job.cfg, job.plan
    arguments: dict[str, Any] = {
        "output_dir": str(job.checkpoints_dir),
        "max_length": None,
        "packing": False,
        "padding_free": False,
        "per_device_train_batch_size": cfg.batch_size,
        "per_device_eval_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.accumulation,
        "num_train_epochs": plan.epochs,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler_type": cfg.schedule,
        "warmup_steps": plan.warmup_steps,
        "optim": cfg.optimizer,
        "weight_decay": cfg.weight_decay,
        "max_grad_norm": cfg.max_grad_norm,
        "bf16": cfg.precision == "bf16",
        "fp16": cfg.precision == "fp16",
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "seed": cfg.seed,
        "data_seed": cfg.seed,
        "logging_steps": cfg.logging_steps_for(job.smoke),
        "save_strategy": "steps",
        "save_steps": plan.save_steps,
        "save_total_limit": None,
        "save_only_model": False,
        "load_best_model_at_end": False,
        "eval_strategy": "steps" if job.dev_rows else "no",
        "report_to": "none",
        "dataloader_num_workers": 0,
        "disable_tqdm": True,
    }
    if job.dev_rows:
        arguments["eval_steps"] = plan.save_steps
    return arguments


def _batch(rows: Sequence[EncodedExample], pad_id: int, torch: Any) -> dict[str, Any]:
    width = max(row.length for row in rows)
    return {
        "input_ids": torch.tensor(
            [row.input_ids + [pad_id] * (width - row.length) for row in rows]
        ),
        "labels": torch.tensor([row.labels + [IGNORE] * (width - row.length) for row in rows]),
        "attention_mask": torch.tensor(
            [[1] * row.length + [0] * (width - row.length) for row in rows]
        ),
    }


class TrlEngine:
    """Trains with TRL's ``SFTTrainer`` on a dataset this pipeline prepared itself."""

    def fit(self, job: TrainingJob) -> None:
        from qlora_finetuning.local_qlora.backend import check_ready, load_quantized_model

        stack, tokenizer = check_ready(job.budget, job.model_id, job.revision)
        import datasets
        import peft
        import transformers
        import trl

        torch, cfg, progress = stack.torch, job.cfg, job.progress
        model = trainer = None
        # A run never touches the network. Saving an adapter makes PEFT ask the Hub for the
        # base model's config; offline, it reads the local cache and carries on.
        import huggingface_hub.constants as hub_constants

        was_offline = hub_constants.HF_HUB_OFFLINE
        hub_constants.HF_HUB_OFFLINE = True
        try:
            torch.cuda.reset_peak_memory_stats(0)
            started = time.perf_counter()
            logger.info("loading %s at %s in 4-bit for training", job.model_id, job.revision[:12])
            model = load_quantized_model(stack, job.local_cfg, job.model_id, job.revision)
            model = peft.prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=cfg.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            modules = cfg.target_modules
            model = peft.get_peft_model(
                model,
                peft.LoraConfig(
                    r=cfg.lora_r,
                    lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    target_modules=modules if isinstance(modules, str) else list(modules),
                    task_type="CAUSAL_LM",
                ),
            )
            load_seconds = time.perf_counter() - started
            trainable = [
                (name, p.numel()) for name, p in model.named_parameters() if p.requires_grad
            ]
            progress["trainable_parameters"] = sum(count for _, count in trainable)
            progress["only_adapter_trains"] = bool(trainable) and all(
                "lora_" in name for name, _ in trainable
            )
            if not progress["only_adapter_trains"]:
                raise TrainingError("a weight outside the LoRA adapter is set to train")

            progress["probe"] = self._probe(job, model, tokenizer, torch)

            arguments = trainer_arguments(job)
            dataset = datasets.Dataset.from_list(
                [{"input_ids": row.input_ids, "labels": row.labels} for row in job.rows]
            )
            dev = (
                datasets.Dataset.from_list(
                    [{"input_ids": row.input_ids, "labels": row.labels} for row in job.dev_rows]
                )
                if job.dev_rows
                else None
            )
            trainer = trl.SFTTrainer(
                model=model,
                args=trl.SFTConfig(**arguments),
                train_dataset=dataset,
                eval_dataset=dev,
                processing_class=tokenizer,
                callbacks=[_recorder(job, transformers)],
            )
            # The trainer prints every log entry to standard output by default.
            for noisy in (transformers.PrinterCallback, transformers.ProgressCallback):
                trainer.remove_callback(noisy)
            progress["dataset_unchanged"] = _same(trainer.train_dataset, job.rows) and (
                dev is None or _same(trainer.eval_dataset, job.dev_rows)
            )
            if not progress["dataset_unchanged"]:
                raise TrainingError(
                    "the trainer changed the prepared dataset (rows or lengths differ): "
                    "refusing to train on it"
                )

            torch.cuda.reset_peak_memory_stats(0)
            started = time.perf_counter()
            resume_from = str(job.resume_checkpoint) if job.resume_checkpoint else None
            trainer.train(resume_from_checkpoint=resume_from)
            train_seconds = time.perf_counter() - started
            steps_done = int(trainer.state.global_step)
            progress["steps_done"] = steps_done
            steps_here = max(steps_done - job.start_step, 1)
            progress["trainer"] = {
                "gpu_name": torch.cuda.get_device_name(0),
                "load_seconds": round(load_seconds, 3),
                "train_seconds": round(train_seconds, 3),
                "seconds_per_step": round(train_seconds / steps_here, 3),
                "seconds_per_sequence": round(
                    train_seconds / (steps_here * cfg.batch_size * cfg.accumulation), 4
                ),
                "peak_vram_allocated_gib": round(torch.cuda.max_memory_allocated(0) / GIB, 3),
                "peak_vram_reserved_gib": round(torch.cuda.max_memory_reserved(0) / GIB, 3),
                "max_steps": int(trainer.state.max_steps),
            }
            final = job.checkpoints_dir / f"checkpoint-{steps_done}"
            if not (final / ADAPTER_WEIGHTS).is_file():  # the last step is not a save step
                model.save_pretrained(str(final))
            progress["final_checkpoint"] = str(final)
        finally:
            hub_constants.HF_HUB_OFFLINE = was_offline
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()

    @staticmethod
    def _probe(job: TrainingJob, model: Any, tokenizer: Any, torch: Any) -> dict[str, Any]:
        """One forward and backward pass over the longest sequences; no optimizer step."""
        rows = [job.rows[position] for position in job.plan.worst_batch]
        batch = {
            name: tensor.to(model.device)
            for name, tensor in _batch(rows, tokenizer.pad_token_id, torch).items()
        }
        torch.cuda.reset_peak_memory_stats(0)
        model.train()
        started = time.perf_counter()
        loss = model(**batch).loss
        loss.backward()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        value = float(loss.detach())
        model.zero_grad(set_to_none=True)
        del batch, loss
        result = {
            "sequences": len(rows),
            "width": max(row.length for row in rows),
            "loss": round(value, 4) if math.isfinite(value) else None,
            "seconds": round(seconds, 3),
            "peak_vram_allocated_gib": round(torch.cuda.max_memory_allocated(0) / GIB, 3),
            "peak_vram_reserved_gib": round(torch.cuda.max_memory_reserved(0) / GIB, 3),
        }
        torch.cuda.empty_cache()
        logger.info(
            "worst batch: %d sequences, %d tokens wide, peak %.2f GiB",
            result["sequences"],
            result["width"],
            result["peak_vram_allocated_gib"],
        )
        if result["loss"] is None:
            raise TrainingError("the worst batch gave a loss that is not a number")
        return result


def _same(dataset: Any, rows: Sequence[EncodedExample]) -> bool:
    """The trainer's dataset is the one it was given: same rows, same ids, same labels."""
    if dataset is None or len(dataset) != len(rows):
        return False
    return list(dataset["input_ids"]) == [row.input_ids for row in rows] and list(
        dataset["labels"]
    ) == [row.labels for row in rows]


def _recorder(job: TrainingJob, transformers: Any) -> Any:
    """A trainer callback that writes the training log and follows the step count."""

    class Recorder(transformers.TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if int(state.max_steps) != job.plan.total_steps:
                raise TrainingError(
                    f"the trainer plans {state.max_steps:,} steps and the pipeline "
                    f"{job.plan.total_steps:,}: the step arithmetic no longer agrees"
                )

        def on_step_begin(self, args, state, control, **kwargs):
            # Stopping here, the step before is complete: logged, and saved if it was due.
            stop = job.stop_after_steps
            if stop is not None and state.global_step >= stop:
                raise TrainingStopped(
                    f"stopped on request after {state.global_step:,} of "
                    f"{job.plan.total_steps:,} steps"
                )

        def on_step_end(self, args, state, control, **kwargs):
            job.progress["steps_done"] = int(state.global_step)

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" not in logs and "eval_loss" not in logs:
                return
            entry: dict[str, Any] = {"step": int(state.global_step)}
            for name in LOG_FIELDS:
                value = logs.get(name)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    entry[name] = round(float(value), 6)
            job.log(entry)
            logger.info("step %s", " ".join(f"{k}={v}" for k, v in entry.items()))

    return Recorder()


def reload_adapter(ctx: TrainingContext, adapter_dir: Path) -> dict[str, Any]:
    """Load a saved adapter through the inference backend and ask it a few quick-dev prompts.

    Each prompt is asked alone and again in a batch. Nothing is scored and no predictions
    file is written: this shows that the adapter reloads and generates, and how much its
    replies depend on the batch. Counts only.
    """
    from qlora_finetuning.dsl import run_reply
    from qlora_finetuning.local_qlora.backend import load_backend

    smoke = ctx.training_cfg.smoke
    examples = load_split("quick_dev", ctx.split_cfg, ctx.data_cfg)[: smoke.generate_examples]
    prompts = [
        (record.system, record.user) for record in (build_record(e, ctx.prompt) for e in examples)
    ]
    limit, size = ctx.budget.max_new_tokens, smoke.generate_batch_size
    backend = load_backend(
        ctx.local_cfg, ctx.budget, ctx.model_id, ctx.revision, adapter_dir=adapter_dir
    )
    try:
        started = time.perf_counter()
        single = [backend.generate([prompt], limit)[0] for prompt in prompts]
        single_seconds = time.perf_counter() - started
        started = time.perf_counter()
        batched = [
            reply
            for first in range(0, len(prompts), size)
            for reply in backend.generate(prompts[first : first + size], limit)
        ]
        batched_seconds = time.perf_counter() - started
        notes = backend.measurements()
        return {
            "adapter_sha256": backend.settings["adapter"],
            "asked": len(prompts),
            "replies": len(single),
            "think_markers": sum("<think>" in reply.text for reply in single + batched),
            "stopped_at_the_limit": sum(reply.stop_reason == "length" for reply in single),
            "programs_that_run": sum(
                run_reply(reply.text, example.table).error_code is None
                for reply, example in zip(single, examples, strict=True)
            ),
            "identical_replies": sum(
                one.text == other.text for one, other in zip(single, batched, strict=True)
            ),
            "batch_size": size,
            "seconds_per_example_single": round(single_seconds / max(len(prompts), 1), 3),
            "seconds_per_example_batched": round(batched_seconds / max(len(prompts), 1), 3),
            "peak_vram_allocated_gib": notes["peak_vram_allocated_gib"],
        }
    finally:
        backend.model = None
        gc.collect()
        if backend._torch is not None:
            backend._torch.cuda.empty_cache()
