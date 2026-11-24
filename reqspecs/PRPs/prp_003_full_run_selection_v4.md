# PRP: Full training run and checkpoint selection (req_003 v3.04)

**Feature Name**: full_run_selection (qlora_local v3.04)  
**Version**: 4  
**Created**: 2026-11-20  
**Status**: ✅ IMPLEMENTED (code; the owner's jobs remain)  
**Implemented**: 2026-11-20  
**Implementation Summary**: [impl_003_20261120_015558.md](../impl_summary/impl_003_20261120_015558.md)  
**Confidence Score**: 7/10  
**Estimated Time**: 6–8 hours of code and documents, about 10 minutes of local GPU time, then the owner's live steps  
**Origin**: `reqspecs/reqs/req_003_qlora_local/req_003_04_spec.md`  
**Template**: `prp_base.md` (shortened)  

---

## Goal

Build what the full Path A run needs beyond v3.03: a `score` task that scores the untuned
model and every checkpoint of a training run in one job; an estimate in the launcher's plan
with a ninth check; `scripts/qlora_select.py`, which picks the checkpoint with the best dev
execution accuracy and writes a committed selection record; `--adapter selected`;
`pull --checkpoint`; and `keep`. Rehearse scoring and selection on a local smoke run.
**Start no job**: the two paid jobs are the owner's, after the v3.03 smoke job.

## Why

- The full run is the point of Path A, and its two jobs are the path's largest spend. The
  owner must see hours and dollars first, and a stopped job must cost nothing twice.
- Ten checkpoints scored on dev are ten results with arm A1. One record has to say which
  of them is arm A1, or every later version will guess.

## What

The owner started this PRP and its execution with the v3.03 smoke job still not run.
Questions 1 to 3 of spec section 8 are **taken as recommended**: every checkpoint on all of
dev; "missed" means execution accuracy under 40% or format validity under 98%; two jobs.
On question 4 the owner chose to build now. So this PRP builds, tests, and rehearses the
code locally, and its paid steps wait: `plan train` cannot give an estimate until a smoke
job has been pulled, and check J9 says so.

The AWS sign-in is expired. Nothing is created in AWS and no job is launched.

### Success Criteria

Code (the agent):

- [x] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
      and without the training stack blocked.
- [x] `job_run score --run <a local smoke run> --split quick_dev --limit 24` scores the
      untuned model and that run's three checkpoints; run again, it skips all four.
- [x] `qlora_select --run <that run> --split quick_dev --limit 24 --check` reports L1 to L6
      on those runs (L5 fails, as it must for local runs) and writes nothing.
- [x] `sagemaker_job plan train`, with no pulled job, fails J9 and says to run the smoke job.
- [x] The existing scripts behave as before.

Live (the owner, spec section 9): the smoke job first; then `plan train`, `launch train`,
`pull`, `plan score`, `launch score`, `pull`, `qlora_select`, `pull --checkpoint`, `keep`.

## All Needed Context

### Documentation & References

```yaml
- file: reqspecs/reqs/req_003_qlora_local/req_003_04_spec.md
  why: decisions D1 to D21, the config additions, checks L1 to L6 and J9, acceptance
- file: reqspecs/reqs/req_003_qlora_local/req_003_03_spec.md (section 11)
  why: what the launcher is, and that nothing has run in AWS yet
- file: reqspecs/references/tech/sagemaker.md, reqspecs/references/common/gotchas.md
- file: reqspecs/references/project_patterns/evaluation_run.md, run_tracking.md, check_script.md

- file: qlora_finetuning/local_qlora/job.py
  why: run_task and _evaluate; add the score task beside smoke, train, evaluate
- file: qlora_finetuning/local_qlora/sagemaker.py
  why: JobOptions, task_arguments, job_definition, plan (checks J1 to J8), pull; add the
       estimate, J9, --max-runtime, pull_checkpoint, keep
- file: qlora_finetuning/local_qlora/adapters.py
  why: resolve_adapter, checkpoint_steps, read_json; add the chain of a resumed run and
       reading the selection record
- file: qlora_finetuning/local_qlora/predictor.py
  why: build_tuned_predictor: "selected" resolves through the selection record
- file: qlora_finetuning/evaluation/runner.py
  why: summarize_run(run_id, context) recomputes every count; evaluation.json holds the
       key (split, prompt, manifest, reply limit, predictor name, model, settings) and the
       predictor's details (training_run, checkpoint, steps)
- file: qlora_finetuning/frontier/tuned_report.py, scripts/nova_tuned.py
  why: a committed report with checks and a verdict that starts nothing: the shape to mirror
- file: tests/local_qlora/training_helpers.py, sagemaker_helpers.py, tests/evaluation/conftest.py
  why: StubEngine (training runs with checkpoint folders), stand-in AWS clients, make_context
```

### Facts the code rests on

```python
# A training run folder: artifacts/runs/<run_id>/{training.json, training_log.jsonl,
#   measurements.json, checkpoints/checkpoint-N/{adapter_model.safetensors, ...}}.
#   training.json: {"key": {...}, "plan": {"total_steps", "checkpoint_steps", ...},
#   "continues": <earlier run or None>, "start_step", "smoke"}.
# A resumed run's folder holds only the checkpoints made after the resume: the earlier ones
#   are in the run it continues. The checkpoints of a training are the union along the chain.
# An evaluation run folder: evaluation.json {"key": {"split", "prompt_id", "manifest_hash",
#   "max_new_tokens", "predictor": {"name", "model_id", "settings"}}, "limit",
#   "predictor": {..., "details": {"training_run", "checkpoint", "steps", "smoke"}}}.
# summarize_run -> "complete" (R1 to R5 pass), "partial", "result", "all" counts,
#   "diagnostic", "backend" (the predictor's measurements: generate_seconds, prompts, ...).
# A predictor's expected settings are known without loading a model:
#   load_local_qlora_config().settings(budget.revision, adapter_sha256_or_None).
# job_result.json of a pulled job is at artifacts/jobs/<job name>/job_result.json; a train
#   entry has "trainer" {"seconds_per_sequence", "train_seconds"}; an eval entry has "backend".
# The job record's entries have "pulled", "instance_type", "billable_seconds", "runs".
```

### Known Gotchas

```python
# CRITICAL: nothing in this PRP launches a job or needs a sign-in. launch, stop, and keep
#   are the owner's; tests use stand-in clients.
# CRITICAL: the selection record is the only thing that says which run is arm A1. It is
#   written only when L1 to L6 pass, and never from partial runs (a rehearsal uses --check).
# CRITICAL: the selection record and report hold counts, ids, hashes, and amounts. No reply,
#   no example text, no time stamp (the same input must give the same files).
# CRITICAL: choose by dev execution accuracy. Dev loss is reported, never used.
# GOTCHA: a scan of evaluation runs must match the whole of the expected settings (the
#   batch size too), the split, the prompt, the manifest, the reply limit, and the limit.
# GOTCHA: a resumed evaluation is a new run; the run it continues stays incomplete. Prefer
#   a complete run; otherwise resume the newest incomplete one.
# GOTCHA: a model must be released before the next is loaded (_release in job.py).
# GOTCHA: prices and amounts are Decimal; the ledger never gets dollars.
# GOTCHA: selection keys added to config/training.yaml must stay out of TrainingConfig.settings,
#   or every existing adapter's settings hash would change.
```

## Implementation Blueprint

### Files

```yaml
CREATE:
  - qlora_finetuning/local_qlora/selection.py      # gather, checks L1-L6, choose, verdict, report
  - qlora_finetuning/local_qlora/estimate.py       # measured rates from pulled jobs; the estimate
  - scripts/qlora_select.py
  - tests/local_qlora/selection_helpers.py, test_selection.py, test_score.py, test_estimate.py,
    tests/scripts/test_qlora_select.py

MODIFY:
  - config/sagemaker.yaml (tasks.score, estimate), config/training.yaml (selection)
  - qlora_finetuning/local_qlora/sagemaker_config.py, training_config.py
  - qlora_finetuning/local_qlora/adapters.py       # training_chain, chain_checkpoints, read_selection, selected_adapter
  - qlora_finetuning/local_qlora/job.py            # the score task
  - qlora_finetuning/local_qlora/predictor.py      # --adapter selected
  - qlora_finetuning/local_qlora/sagemaker.py      # run, max_runtime, estimate, J9, pull_checkpoint, keep
  - scripts/job_run.py, scripts/sagemaker_job.py, scripts/evaluate.py (help text)
  - the tests of the modules above; documents; spec, roadmap, execution plan
```

### Tasks, in order

```yaml
Task 1 - config: the score task and the estimate block; the selection block and its loader.
Task 2 - adapters.py: the chain of a resumed training; the selection record's readers.
Task 3 - job.py and job_run.py: the score task (find an existing run, skip or resume).
Task 4 - estimate.py; sagemaker.py: JobOptions.run and .max_runtime, the estimate in the
         plan, J9, a workspace required where a job continues another's work.
Task 5 - selection.py and scripts/qlora_select.py.
Task 6 - --adapter selected (predictor and launcher); pull_checkpoint; keep.
Task 7 - tests, lint, the suite with and without the training stack.
Task 8 - on this machine: score a local smoke run twice; qlora_select --check; plan train.
Task 9 - documents, spec section 11 (the code part), roadmap, execution plan, summary.
```

### Pseudocode

```python
# job.py, task "score"
steps = chain_checkpoints(run, runs_dir)            # {step: "RUN/checkpoint-N"}, along a resumed chain
for name, adapter in [(UNTUNED, None)] + [(TUNED, steps[s]) for s in sorted(steps)]:
    sha = None if adapter is None else adapter_sha256(folder_of(adapter))
    found = find_evaluation(ctx, split, limit, expected_settings(sha))   # complete | incomplete | None
    if found complete: entry from its summary, "reused": True
    else: _evaluate(name, split, adapter=adapter, limit=limit, resume=found incomplete or None)

# estimate.py
rates = measured_rates(job record, artifacts/jobs, instance type)   # newest pulled job first
train:  records(fit) * epochs * rates.train_seconds_per_sequence + rates.overhead_seconds
score:  examples * (rates.untuned + checkpoints * rates.tuned) + rates.overhead_seconds
proposed limit = max(minimum, ceil(estimate * margin))

# selection.py
rows = one complete result per checkpoint of the chain, and one for the untuned model
checks L1..L6; stop if any fails
selected = max(rows, key=(execution count, format-valid count, -step))
verdict = "met" if execution% >= min and format% >= min else "missed"
write adapter_selection.json and .md (only without --check, only for the config's split, no limit)
```

### Error handling

| Where | What | Becomes |
|---|---|---|
| `score` | An unknown run, or one without checkpoints | `config_error`, nothing started |
| `score` | A predictor fails mid-way | `failed`; the runs made are listed; the job is launched again and continues |
| `plan` | No pulled job to measure from; an estimate over the limit | J9 FAIL with what to do |
| `plan` | `score`, `--resume`, or a tuned `evaluate` without `--workspace` | `ConfigError`: name the training job's workspace |
| `qlora_select` | A check fails | The FAIL lines; nothing written; exit 1 |
| `qlora_select` | Partial runs or another split without `--check` | `ConfigError`: a rehearsal writes nothing |
| `--adapter selected` | No record, or a record for another prompt, split, or commit | `ConfigError` before any model is loaded |
| `keep` | No `--yes`; no selection; the adapter not pulled | `ConfigError`, nothing uploaded |

## Validation Loop

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest tests/local_qlora tests/scripts -q
venv\Scripts\python -m pytest -q
venv\Scripts\python -m scripts.job_run score --run <local smoke run> --split quick_dev --limit 24
venv\Scripts\python -m scripts.job_run score --run <local smoke run> --split quick_dev --limit 24   # skips all
venv\Scripts\python -m scripts.qlora_select --run <local smoke run> --split quick_dev --limit 24 --check
venv\Scripts\python -m scripts.sagemaker_job plan train --scope measured                            # J9 fails
```

**Not run by the agent**: `launch`, `stop`, `keep`, and anything that needs a sign-in.

## Final validation Checklist

- [x] Tests and lint green, with and without the stack
- [x] Scoring a local run twice makes four runs, then none
- [x] A selection is written only from complete, checked runs; the rehearsal writes nothing
- [x] Nothing was created in AWS
- [x] Documents, spec, roadmap, execution plan, implementation summary

---

## Time Estimate

**Total**: 6–8 hours of work, about 10 minutes of GPU, then the owner's steps.

| Task | Time |
|---|---|
| Config and loaders | 30 min |
| Chain and selection-record readers | 30 min |
| The score task | 60 min |
| Estimate, J9, launcher options | 70 min |
| Selection and its script | 90 min |
| `selected`, `pull --checkpoint`, `keep` | 45 min |
| Tests | 110 min |
| Local rehearsal | 15 min |
| Documents and status files | 45 min |

**Risks**: the smoke job, not yet run, may show a fault in the launcher that this version
builds on (+1 to 2 hours, shared with v3.03); the estimate's formulas cannot be compared
with a real job until one is pulled.

## Anti-Patterns to Avoid

- ❌ No selection by loss, by hand, or from partial runs
- ❌ No reading of an A1 result by its arm alone
- ❌ No job launched, and no S3 object written, by the agent
- ❌ No hyperparameter changed: this is the first, default run
- ❌ No second evaluation loop: every score is `run_evaluation`

## Time Taken for PRP Generation

**~10 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 1 min |
| Codebase exploration | 3 min |
| Research existing PRPs and patterns | 1 min |
| Writing PRP document | 5 min |
