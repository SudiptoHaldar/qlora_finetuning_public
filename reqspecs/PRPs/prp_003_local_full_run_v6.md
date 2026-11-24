# PRP: Full local training run (req_003 v3.06, optional)

**Feature Name**: local_full_run (qlora_local v3.06)  
**Version**: 6  
**Created**: 2026-11-20  
**Status**: ✅ IMPLEMENTED (the owner's run was made 2026-11-20; spec section 12)  
**Implemented**: 2026-11-20  
**Implementation Summary**: [impl_003_20261120_122043.md](../impl_summary/impl_003_20261120_122043.md)  
**Confidence Score**: 8/10  
**Estimated Time**: 3–4 hours of code and documents, no GPU time; then the owner's run, about 5 hours of the laptop  
**Origin**: `reqspecs/reqs/req_003_qlora_local/req_003_06_spec.md`  

Template: `prp_lean.md` (medium: two small modules and one script over existing pipelines).

## Goal

Make the optional local full run safe to start and easy to follow, and give it a committed
record of its own that can never be taken for arm A1: a **preflight** (P1 to P6), a
**progress line**, and a **local note** (N1 to N5) with a local-against-cloud section once
v3.04's selection exists. No change to training, scoring, selection, or settings.

**The agent does not start the five-hour run.** The owner does, in their own terminal
(spec section 8, question 2, taken as recommended). Nothing is billed and no AWS call is made.

## Why

- A laptop run fails for reasons a cloud job does not have: GPU memory held by other
  programs (2.4 GiB of 16 today, against 13.6 GiB reserved in a resumed run), battery,
  sleep, a run that was interrupted and should be resumed, not restarted.
- `qlora_select` and `qlora_analyze` refuse to write for local runs, by design. The local
  run needs its own record, labelled as a side note.
- It would be the project's first fully trained adapter, before any dollar is spent.

## What

The five decisions of spec section 8 are **taken as recommended**: do it, now; the owner
starts the run; the selected checkpoint is scored once on dev; quick-dev for the ten
checkpoints; 14.5 GiB of GPU memory must be free.

Success criteria

Code (the agent):

- [x] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
      and without the training stack blocked.
- [x] `qlora_local preflight` on this machine prints P1 to P6 with real values and starts
      nothing; as the machine stands it fails P3 and says what holds the memory.
- [x] `qlora_local status` with no full local run says so; with `--run <the local smoke run>`
      it prints 12 of 12 steps.
- [x] `qlora_local note --run <the local smoke run> --check` fails N3 and writes nothing.
- [x] The selection record, `--adapter selected`, and `keep` are untouched by a local note.

Live (the owner): the run sheet of spec section 4.2.

## All Needed Context

| File | Why |
|---|---|
| `reqspecs/reqs/req_003_qlora_local/req_003_06_spec.md` | D1 to D17, P1 to P6, N1 to N5, the run sheet |
| `qlora_finetuning/local_qlora/training.py` | `TrainingContext`, `prepare(ctx, smoke)` (the dry run: needs the tokenizer, no GPU), `LOG_NAME`, `MEASUREMENTS_NAME` |
| `qlora_finetuning/local_qlora/selection.py` | `gather_scored(run, split=..., check=True)` and `best_row`: the note reuses them; L5 fails for local runs and is not asked |
| `qlora_finetuning/local_qlora/scoring.py` | `evaluation_runs(runs_dir)`: finding the selected checkpoint's run on dev by its settings |
| `qlora_finetuning/evaluation/failures.py` | `run_failures`, `count_failures`: the categories in the note |
| `qlora_finetuning/local_qlora/adapters.py` | `training_chain`, `read_json`, `write_json`, `read_selection`, `checkpoint_steps` |
| `qlora_finetuning/tracking/registry.py` | `list_runs` (a run with no `finished` line is `started`), `load_manifest`, `read_git_info(root, env_var, (ledger_prefix,))` |
| `scripts/qlora_select.py` | The shape of a script: module run, `**stand_ins`, exit codes 0 / 1 / 2 |
| `tests/local_qlora/selection_helpers.py`, `training_helpers.py` | `World(where="local", scope="dev")`; `make_training_context`, `StubEngine(fail_at=)` |

Facts established for this PRP:

- **A training log line has no time**: `{"step", "epoch", "loss", ...}` and
  `{"step", "eval_loss"}`. Seconds per step of a run in progress are the log file's age
  against the manifest's `started_at`, over the steps logged; a finished run has
  `trainer.seconds_per_step` in `measurements.json`.
- **`nvidia-smi` answers without the training stack**:
  `--query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits`
  gives `NVIDIA GeForce RTX 4090 Laptop GPU, 16376, 2480, 13569` today, and
  `--query-compute-apps=process_name --format=csv,noheader` lists the programs (a browser,
  chat and office programs), with per-program memory `[N/A]` on Windows. So the check
  names programs, not amounts.
- **Power on Windows**: `GetSystemPowerStatus` (ctypes) gives the AC line status;
  `powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE` prints
  `Current AC Power Setting Index: 0x00000000` for "never". Both are read-only. Elsewhere
  both are unknown and the check is skipped.
- The stand-in split has `quick_dev` (3 examples, the first of dev's 8), so the note's two
  splits can be tested.
- A killed run stays `started` in the run index for ever. "Running" therefore means
  started, unfinished, and its log written in the last 15 minutes.

Gotchas:

- The preflight and the status line must not import the training stack at module level;
  `prepare` loads the tokenizer when called.
- Probes for the GPU, the power state, and the disk are parameters, so tests need no
  hardware. A probe that cannot answer returns "unknown", never raises.
- The note is written with no time stamp; the same runs give the same bytes. It holds no
  reply and no example text.
- Never write, read for a decision, or rename the selection record here.
- The owner's smoke job is in progress in AWS while this is built: make no AWS call.

## Implementation Blueprint

```
config/training.yaml                         + local_run: min_free_gpu_gib, min_free_disk_gb, note_file
qlora_finetuning/local_qlora/training_config.py   + the three fields (not in settings())
qlora_finetuning/local_qlora/local_run.py    probe_gpu, probe_power, preflight (P1-P6), run_status
qlora_finetuning/local_qlora/local_note.py   local_note (N1-N5), render_note, the cloud section
scripts/qlora_local.py                       preflight | status | note
```

```python
# local_run.py
def preflight(*, context=None, gpu=probe_gpu, power=probe_power, disk=shutil.disk_usage, now=None):
    P1  prepare(ctx, smoke=False) -> plan; estimate from the newest local run's seconds per sequence
    P2  ctx.git or read_git_info(root, env, (ledger_prefix,)): not dirty
    P3  gpu() free >= cfg.min_free_gpu_gib; name the programs
    P4  disk(root).free >= cfg.min_free_disk_gb
    P5  on AC; sleep on AC "never"; SKIP what cannot be read
    P6  no run that is running; no interrupted full local run that should be resumed instead
def run_status(run=None, *, context=None, now=None):   # newest full local run when none is named
    step, total, seconds per step, hours left, last loss, last dev loss, state, resume command

# local_note.py
def local_note(run, *, split="quick_dev", check=False, ...):
    scored = gather_scored(run, split=split, check=True)       # L1-L4, L6; L5 is not asked
    N1 scored; N2 every run is local; N3 a full run over all of fit
    N4 the selected checkpoint has one complete result on dev, same settings; N5 scores current
    write reports/local_qlora/local_run.json and .md only if all pass and not check
```

Errors: an unknown run, or no local run at all, is a `ConfigError` (exit code 2). A failed
check is reported (exit code 1). Nothing is raised for hardware that cannot be read.

Tasks, in order:

1. Config and loader.
2. `local_run.py`: probes, preflight, status.
3. `local_note.py`: the note, its page, local against cloud.
4. `scripts/qlora_local.py`.
5. Tests: `tests/local_qlora/test_local_run.py`, `test_local_note.py`,
   `tests/scripts/test_qlora_local.py`.
6. On this machine: `preflight`, `status`, `status --run <smoke run>`,
   `note --run <smoke run> --check`. **Do not start `qlora_train`.**
7. Documents (runbook, gotchas, `coding.md`, architecture), spec section 11 (the code),
   roadmap, execution plan, implementation summary.

## Validation Loop

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest tests/local_qlora tests/scripts -q
venv\Scripts\python -m pytest -q
venv\Scripts\python -m pytest -q -p block_stack          # the session's plugin on PYTHONPATH

venv\Scripts\python -m scripts.qlora_local preflight
venv\Scripts\python -m scripts.qlora_local status
venv\Scripts\python -m scripts.qlora_local status --run 20261119T083236Z-smoke-qwen3-4b-smoke-5477
venv\Scripts\python -m scripts.qlora_local note --run 20261119T083236Z-smoke-qwen3-4b-smoke-5477 --check
```

## Final validation Checklist

- [x] Tests and lint green, with and without the stack
- [x] The preflight reads this machine and starts nothing
- [x] The note cannot be written for a smoke run or a cloud run, and holds no reply
- [x] Nothing was started on the GPU beyond the tests, and no AWS call was made
- [x] Documents, spec, roadmap, execution plan, implementation summary

## Time Estimate

| Task | Time |
|---|---|
| Config and loader | 10 min |
| Preflight and status | 50 min |
| The local note | 50 min |
| Script | 20 min |
| Tests | 60 min |
| Rehearsals | 10 min |
| Documents | 30 min |
| **Total** | **3–4 hours** |

Confidence 8/10: no model, no AWS, and every piece it reads exists. What can go wrong is
the machine-reading (`nvidia-smi`, `powercfg`), which is why each probe may answer "unknown".

## Anti-Patterns to Avoid

- Do not start the training run, and do not change `qlora_train`.
- Do not let anything read the local note as a selection, and do not set `--scope measured`.
- Do not put a price on the local run.
- Do not make a preflight check fail the script with a traceback because a tool is missing.

## Time Taken for PRP Generation

**~10 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 1 min |
| Codebase exploration | 5 min |
| Research existing PRPs and patterns | 1 min |
| Writing PRP document | 3 min |
