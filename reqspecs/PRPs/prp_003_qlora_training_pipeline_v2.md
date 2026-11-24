# PRP: QLoRA training pipeline (req_003 v3.02)

**Feature Name**: qlora_training_pipeline (qlora_local v3.02)  
**Version**: 2  
**Created**: 2026-11-19  
**Status**: ✅ IMPLEMENTED  
**Implemented**: 2026-11-19  
**Implementation Summary**: [impl_003_20261119_041502.md](../impl_summary/impl_003_20261119_041502.md)  
**Confidence Score**: 7/10  
**Estimated Time**: 6–8 hours of code and documents, plus about 15 minutes of local GPU time  
**Origin**: `reqspecs/reqs/req_003_qlora_local/req_003_02_spec.md`  
**Template**: `prp_base.md` (shortened)  

---

## Goal

Build the pipeline that fits a LoRA adapter to 4-bit `Qwen/Qwen3-4B` on the `fit` records:
`venv\Scripts\python -m scripts.qlora_train` with a dry run, a smoke run, checkpoints, and
resuming; a saved adapter with metadata; and `scripts.evaluate --predictor qwen3-4b-tuned
--adapter <run>` to score one. Prove it with a smoke run of 64 examples on this machine.

## Why

- v3.01 measured the untuned model at 3 of 499 on dev: almost no valid program. Training is
  the whole of Path A's case.
- Every later version of req_003 runs this code: the optional local full run (v3.06), the
  SageMaker job (v3.03, v3.04). A fault found here costs nothing; there it costs GPU hours.

## What

The owner started this PRP and its execution without changing the two questions of spec
section 8, so they are **taken as recommended**: the config says **3 epochs**, and
**checkpoints are scored after training** as ordinary evaluation runs, not inside the loop.
By spec D29 the agent runs the smoke run, an interrupted run with its resume, and a partial
evaluation of the smoke adapter. Nothing is billed; no AWS service is used. **No full
training run is started.**

### Success Criteria

- [x] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
      and without the training stack blocked; `pytest -m gpu` passes.
- [x] A test shows only the reply carries a label and a sequence has the token budget's length.
- [x] `qlora_train --dry-run` prints the full plan (4,884 records, 3,254,181 tokens) and
      starts no run.
- [x] `qlora_train --smoke` passes K1 to K7: 33,030,144 trainable parameters, worst batch
      1,862 tokens wide.
- [x] A smoke run stopped after 6 steps exits 1 with a checkpoint at step 4 and a ledger
      record; `--resume` completes it at step 12.
- [x] The smoke adapter is scored on 24 quick-dev examples as `qwen3-4b-tuned-smoke`, a
      partial run with no results line; an adapter for another prompt is refused, exit 2.
- [x] `--predictor gold` and `--predictor qwen3-4b` behave as before.
- [x] Spec section 11, roadmap, execution plan, runbook, references, implementation summary.

## All Needed Context

### Documentation & References

```yaml
- file: reqspecs/reqs/req_003_qlora_local/req_003_02_spec.md
  why: decisions D1 to D29, the config file, the smoke checks K1 to K7, the trial's figures
- file: reqspecs/references/tech/hf_training_stack.md
  why: the 4-bit load, LoRA on a 4-bit model, generation; warnings that can be ignored
- file: reqspecs/references/project_patterns/run_tracking.md
  why: start_run / run.record; tests use a temporary root, a fixed GitInfo, environment={}
- file: reqspecs/references/project_patterns/canonical_record.md
  why: texts unchanged; reuse checklist
- file: reqspecs/references/project_patterns/check_script.md
  why: PASS / FAIL lines and exit codes 0, 1, 2
- file: reqspecs/references/common/gotchas.md
  why: sections 1 and 4 (heredocs mangle backslashes: write files with the file tool)

- file: qlora_finetuning/local_qlora/render.py, backend.py, predictor.py, config.py
  why: render_chat, load_backend (take load_quantized_model out of it), QwenPredictor
- file: qlora_finetuning/evaluation/runner.py
  why: RunnerContext with .resolved(): the shape TrainingContext copies
- file: qlora_finetuning/frontier/predictor.py (build_tuned_predictor), frontier/tuned.py
  why: a tuned predictor that refuses a model tuned for another experiment
- file: scripts/evaluate.py
  why: PredictorEntry.build(split, context); add --adapter
- file: scripts/finqa_tokens.py
  why: a script that needs the tokenizer and exits 2 with the fix when it is missing
- file: tests/evaluation/conftest.py, tests/local_qlora/conftest.py, tests/data/synthetic.py
  why: make_context, StubBackend, FakeTokenizer, fake_budget, build_frozen_split

- url: https://huggingface.co/docs/trl/sft_trainer
  why: SFTTrainer and SFTConfig; a dataset with input_ids is taken as prepared
- url: https://huggingface.co/docs/transformers/main_classes/callback
  why: TrainerCallback.on_log, on_step_end
- url: https://huggingface.co/docs/peft/package_reference/lora
  why: LoraConfig
```

### Verified in the installed libraries (TRL 1.18.0, transformers 5.23.0, peft 0.21.3)

```python
# A dataset with "input_ids" is not tokenized; "labels" is left as is. BUT max_length
# (default 1024) still cuts every row and drops rows left without a label. Pass None.
SFTConfig(output_dir=..., max_length=None, per_device_train_batch_size=2,
          gradient_accumulation_steps=8, learning_rate=2e-4, lr_scheduler_type="cosine",
          warmup_steps=1, optim="paged_adamw_8bit", bf16=True, logging_steps=1,
          num_train_epochs=3, save_strategy="no", report_to="none", seed=42)   # ran in the trial
SFTTrainer(model=peft_model, args=args, train_dataset=Dataset.from_list(rows),
           eval_dataset=..., processing_class=tokenizer, callbacks=[...])
trainer.train(resume_from_checkpoint=None | "<folder>")
trainer.state.global_step, trainer.state.max_steps, trainer.state.log_history   # floats
# warmup_ratio and group_by_length do not exist. SFTConfig defaults: gradient_checkpointing
# True, optim adamw_torch_fused, learning_rate 2e-5, linear, save_steps 500, eval_strategy no.

# Step arithmetic of transformers.Trainer.set_initial_training_values:
batches = ceil(records / batch_size)
steps_per_epoch = max(batches // accumulation + (batches % accumulation > 0), 1)
max_steps = ceil(epochs * steps_per_epoch)          # 64 records -> 12; 4,884 records -> 918
```

Trial on this machine (spec 2.3): 33,030,144 trainable parameters; a 66 MB adapter
(`adapter_model.safetensors`, `adapter_config.json`); worst batch 11.00 GiB allocated; 12.6 s
per step; loss 2.27 falling to about 0.3 in 12 steps.

### Known Gotchas

```python
# CRITICAL: only backend.py and training.py may import the training stack (torch,
#   transformers, peft, trl, datasets, bitsandbytes), and only inside functions.
# CRITICAL: nothing is truncated. Over max_length is a LengthError naming the records,
#   raised before the model is loaded.
# CRITICAL: after building the trainer, compare trainer.train_dataset (and eval_dataset)
#   with our rows: count and every length. Refuse to train on a difference.
# CRITICAL: no file this version writes may hold example text: ids, counts, losses only.
# CRITICAL: a run in the registry finishes once. Resuming is a new run that names the old one.
# GOTCHA: whether the trainer saves a checkpoint at the last step when it is not a multiple
#   of save_steps is not relied on: after training, save the final checkpoint if absent.
# GOTCHA: the training model must be freed (del, gc, empty_cache) before the adapter is
#   reloaded through load_backend in the same process.
# GOTCHA: a ledger token count needs model_id; a record needs at least one quantity.
# GOTCHA: tests that start a run pass a tracking root under tmp_path, a fixed GitInfo, and
#   environment={}. load_training_config(root=tmp_path) puts adapters under tmp_path.
# GOTCHA: scripts print ASCII only.
```

## Implementation Blueprint

### Files

```yaml
CREATE:
  - config/training.yaml
  - qlora_finetuning/local_qlora/training_config.py   # load_training_config, TrainingConfig
  - qlora_finetuning/local_qlora/plan.py              # training_plan, select_smoke, TrainingPlan
  - qlora_finetuning/local_qlora/adapters.py          # metadata, resolve_adapter, check_adapter
  - qlora_finetuning/local_qlora/training.py          # run_training, TrainingContext, the TRL engine
  - scripts/qlora_train.py
  - tests/local_qlora/test_training_config.py, test_plan.py, test_encode.py, test_adapters.py,
    test_training.py, test_training_gpu.py; tests/scripts/test_qlora_train.py

MODIFY:
  - qlora_finetuning/local_qlora/render.py      # encode_example, encode_records, LengthError
  - qlora_finetuning/local_qlora/backend.py     # load_quantized_model taken out of load_backend
  - qlora_finetuning/local_qlora/predictor.py   # details; build_tuned_predictor; qwen3_4b_tuned
  - scripts/evaluate.py                         # --adapter; PredictorEntry.takes_adapter; the entry
  - tests/local_qlora/test_predictor.py (guard: two stack modules), tests/scripts/test_evaluate.py
  - runbook section 5, hf_training_stack.md, gotchas.md, canonical_record.md, coding.md,
    architecture overview, spec section 11, roadmap, execution plan
```

### Data models

```python
@dataclass(frozen=True)
class EncodedExample:            # render.py
    example_id: str
    input_ids: list[int]         # prompt ids, then target ids, then the end token
    labels: list[int]            # -100 over the prompt, the ids over the reply
    prompt_tokens: int

@dataclass(frozen=True)
class TrainingPlan:              # plan.py
    records: int; tokens_per_pass: int; batch_size: int; accumulation: int; epochs: float
    steps_per_epoch: int; total_steps: int; warmup_steps: int
    checkpoint_steps: tuple[int, ...]      # multiples of save_steps, and the last step
    worst_batch: tuple[int, ...]           # positions of the batch_size longest sequences
    worst_batch_width: int

# The engine does everything that needs the model; a test passes a stand-in.
class TrainingJob:   rows, dev_rows, plan, cfg, local_cfg, budget, checkpoints_dir,
                     resume_checkpoint, stop_after_steps, log(entry), progress (steps_done)
class EngineResult:  steps_done, dataset_unchanged, trainable_parameters, only_adapter_trains,
                     probe {width, loss, seconds, peak_vram_*}, measurements, final_checkpoint

# training.json in the run folder: what was asked, and what a resume compares.
{"schema": 1, "run_id": ..., "smoke": bool, "continues": run_id | None, "start_step": int,
 "key": {"model_id", "revision", "prompt_id", "manifest_hash", "settings_sha256",
         "selection_sha256", "smoke"}, "plan": {...}}

# adapter_metadata.json beside the final adapter: the key, plus run_id, steps, epochs,
# records, adapter_sha256, continues, library versions.
```

### Tasks, in order

```yaml
Task 1 - config: config/training.yaml (spec 4.1, plus smoke.dev_examples) and its loader;
  settings(smoke) and its SHA-256.
Task 2 - encoder in render.py; plan.py. Pure, tested with the stand-in tokenizer.
Task 3 - adapters.py: metadata, resolve_adapter("RUN" | "RUN/checkpoint-N"), check_adapter.
Task 4 - backend.py: load_quantized_model; load_backend calls it. v3.01 tests still pass.
Task 5 - training.py: TrainingContext, run_training (run, key, resume, log, ledger record
  in a finally, final adapter and metadata, the checks), then the TRL engine (load, LoRA,
  probe, trainer, dataset comparison, callbacks, final checkpoint, free the model) and the
  reload check for K7.
Task 6 - predictor.py and scripts/evaluate.py: the tuned predictor and --adapter.
Task 7 - scripts/qlora_train.py.
Task 8 - tests without a GPU, lint, the suite with and without the stack.
Task 9 - gpu tests on the 0.6B model, then the local runs of spec section 9.
Task 10 - documents, spec section 11, roadmap, execution plan, implementation summary.
```

### Pseudocode

```python
# render.py
def encode_example(record, tokenizer, suffix):
    prompt = tokenizer(render_chat(record.system, record.user, tokenizer, suffix),
                       add_special_tokens=False)["input_ids"]
    reply = tokenizer(record.target + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    return EncodedExample(record.example_id, prompt + reply, [-100] * len(prompt) + reply, len(prompt))

# training.py
def run_training(*, smoke=False, resume=None, stop_after_steps=None, where="local",
                 scope="dev", context=None, engine=None, reloader=None):
    ctx = (context or TrainingContext()).resolved()
    rows, dev_rows, plan, key = prepare(ctx, smoke)          # LengthError before any run
    earlier = check_resume(resume, key, ctx) if resume else None     # ResumeError before any run
    with start_run("smoke" if smoke else "train", ...) as run:
        write training.json; copy the earlier log lines
        try:
            result = engine.fit(job)                          # raises on a failure or a stop
        finally:
            run.record(phase="train", wall_clock_seconds=..., examples=plan.records,
                       training_tokens=share_of_planned_steps, model_id=..., source="computed")
            write measurements.json
        copy the final adapter to adapters_dir/<run_id>/ and write adapter_metadata.json
        checks = K1..K5 (+ K6, K7 for a smoke run)
    return {...}

# the TRL engine
model = prepare_model_for_kbit_training(load_quantized_model(...), ...)
model = get_peft_model(model, LoraConfig(...))
probe(worst batch): forward, backward, no optimizer step; zero_grad; record peak memory
trainer = SFTTrainer(model=model, args=SFTConfig(max_length=None, ...), train_dataset=...,
                     eval_dataset=..., processing_class=tokenizer, callbacks=[Recorder(job)])
assert lengths(trainer.train_dataset) == lengths(rows)        # else refuse
trainer.train(resume_from_checkpoint=job.resume_checkpoint)
```

### Error handling

| Where | What | Becomes |
|---|---|---|
| Before a run | A record over `max_length`; no stack, weights, or GPU; a bad config | `ConfigError`: exit 2, no run |
| Before a run | A resume under other settings, or from a run without a checkpoint | `ResumeError`: exit 2, no run |
| In a run | The probe or a step fails (out of memory), or `--stop-after-steps` | The run fails; checkpoints kept; a ledger record for the steps done; exit 1 |
| In a run | The trainer's dataset differs from ours | `TrainingError` before the first step; exit 1 |
| Evaluate | No `--adapter`, an unknown one, or one for another prompt, split, or model | `ConfigError`: exit 2, no run |

## Validation Loop

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest tests/local_qlora tests/scripts -q
venv\Scripts\python -m pytest -q
venv\Scripts\python -m pytest -m gpu -q
venv\Scripts\python -m scripts.qlora_train --dry-run
venv\Scripts\python -m scripts.qlora_train --smoke
venv\Scripts\python -m scripts.qlora_train --smoke --stop-after-steps 6
venv\Scripts\python -m scripts.qlora_train --smoke --resume <the stopped run>
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b-tuned --adapter <the smoke run> --split quick_dev --limit 24
venv\Scripts\python -m scripts.evaluate --predictor gold --split quick_dev --limit 3
```

The suite is also run once with the training stack blocked (a throwaway pytest plugin).

## Final validation Checklist

- [x] Tests and lint green, with and without the stack; `gpu` tests green
- [x] Guards: only `backend.py` and `training.py` name the stack; no file written holds example text
- [x] The smoke run passes K1 to K7; the stopped run resumes and completes
- [x] The tuned predictor refuses an adapter made for another experiment
- [x] Documents, spec section 11, roadmap, execution plan, implementation summary

---

## Time Estimate

**Total**: 6–8 hours of work, plus about 15 minutes of GPU time.

| Task | Time |
|---|---|
| Config and loader | 30 min |
| Encoder and plan | 40 min |
| Adapters and metadata | 35 min |
| Backend refactor | 10 min |
| Training: the run, resume, ledger, checks | 80 min |
| Training: the TRL engine | 60 min |
| Tuned predictor and `--adapter` | 35 min |
| Script | 30 min |
| Tests | 100 min |
| Local runs | 20 min |
| Documents and status files | 45 min |

**Risks**: resuming from a checkpoint may fail on Windows for a reason the trial did not
exercise (optimizer or random state files) (+1 hour); the trainer may or may not save at
the last step (handled); evaluation loss on a prepared dataset may behave differently from
training (+30 min).

## Anti-Patterns to Avoid

- ❌ No truncation, and no `max_length` other than `None` given to the trainer
- ❌ No hyperparameter in code, and none left to a library default
- ❌ No scoring inside the training loop; no second evaluation loop
- ❌ No full training run in this version
- ❌ No training-stack import outside `backend.py` and `training.py`
- ❌ No example text in a log, a measurement, metadata, or a committed file

## Time Taken for PRP Generation

**~12 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 3 min |
| Research existing PRPs and patterns, installed library source | 3 min |
| Writing PRP document | 4 min |
