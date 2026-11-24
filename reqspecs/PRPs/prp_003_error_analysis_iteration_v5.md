# PRP: Error analysis and one corrective iteration (req_003 v3.05)

**Feature Name**: error_analysis_iteration (qlora_local v3.05)  
**Version**: 5  
**Created**: 2026-11-20  
**Status**: ✅ IMPLEMENTED (code; conditional, and not yet triggered)  
**Implemented**: 2026-11-20  
**Implementation Summary**: [impl_003_20261120_034140.md](../impl_summary/impl_003_20261120_034140.md)  
**Confidence Score**: 7/10  
**Estimated Time**: 6–8 hours of code and documents, no GPU time; then the owner's steps, which wait for v3.04's verdict  
**Origin**: `reqspecs/reqs/req_003_qlora_local/req_003_05_spec.md`  

Template: `prp_base.md` (complex: three new modules, two scripts, a launch check, a change to
the selection).

## Goal

Build everything v3.05 needs so that, the day v3.04's selection exists, the owner can (a)
get a committed error analysis of arm A1 on dev with one free command, and (b) if the
verdict is `missed`, register one change, run the second pair of jobs, and select across
both runs, with the tooling refusing anything that is not exactly one registered change.

**Nothing is launched and nothing is created in AWS by the agent.** The trigger does not
exist yet: no cloud job has run. The agent's proof is tests and a rehearsal on stored runs.

## Why

- v5.03 reports error categories for six arms; the classifier is needed whatever the verdict.
- "Change one variable" and "one iteration only" are promises that only code can keep once
  money is involved: a launch check (J10) and a registered record do that.
- The selection record must go on saying, alone, which run is arm A1 after a second run.

## What

The decisions of spec section 8 are **taken as recommended** (the owner started this PRP
without changing them): build now; no iteration after `met`; the five variables and the
table as written; no margin, a tie keeps the first run; the spend accepted as a ceiling.

Success criteria

Code (the agent):

- [x] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
      and without the training stack blocked.
- [x] `qlora_analyze --run 20261119T083236Z-smoke-qwen3-4b-smoke-5477 --split quick_dev
      --limit 24 --check` prints, for checkpoint 8: 10 right, 5 `steps_more`, 4 `order`,
      3 `evidence`, 1 `steps_fewer`, 1 `operation`; writes nothing.
- [x] `failure_counts` on the stored untuned dev runs gives the figures of spec section 2.2
      (Qwen 3 right / 45 cut off / 23 more steps; Nova Pro 190 / 3 / 72; Nova Lite 123 / 2 / 100).
- [x] `qlora_iterate propose` with no selection record fails I1 and writes nothing.
- [x] `sagemaker_job plan train` shows J10 passing with the note that no run is selected yet.
- [x] The existing scripts behave as before; v3.04's tests pass unchanged except where a
      count of checks goes from nine to ten.

Live (the owner, spec section 9): `qlora_analyze --run <first run>` always; the iteration's
commands only on a `missed` verdict.

## All Needed Context

### Read first

| File | Why |
|---|---|
| `reqspecs/reqs/req_003_qlora_local/req_003_05_spec.md` | D1 to D20, the category table (4.1), the finding table (4.3), I1 to I5, J10 |
| `qlora_finetuning/local_qlora/selection.py` | `select_checkpoint`: gathers the scored runs, L1 to L6, the choice, the record. To be split into gathering and choosing |
| `qlora_finetuning/local_qlora/adapters.py` | `read_selection`, `selected_adapter`, `training_chain`, `read_json`, `write_json`, `SELECTION_SCHEMA` |
| `qlora_finetuning/evaluation/metrics.py` | `ExampleScore`, `score_reply`, `Gold`, `_same_argument`, `mismatch_label` |
| `qlora_finetuning/evaluation/runner.py` | `_load_gold`, `run_predictions`, `summarize_run`: how a run's lines meet their gold replies |
| `qlora_finetuning/evaluation/predictions.py` | `PredictionRecord`, `outcome_from_dict` |
| `qlora_finetuning/local_qlora/sagemaker.py` | `plan`, `_check_estimate` (the pattern for J10), `launch` (the job entry) |
| `qlora_finetuning/local_qlora/training_config.py` | `settings(smoke)`, `settings_sha256`, the loader and its `REQUIRED_KEYS` |
| `tests/local_qlora/selection_helpers.py` | `World`: a stand-in training run, scored through the real task and runner |
| `reqspecs/references/project_patterns/evaluation_run.md`, `run_tracking.md` | What a run leaves; what may be committed |
| `reqspecs/references/common/gotchas.md` | Ten runs carry arm A1; the loss is not the selector |

No web research is needed: no new library, no new AWS call.

### Facts established while writing the spec and this PRP

- A predictions line holds `outcome` (`program`, `value`, `error_stage` in `reply` /
  `program` / `execution`, `error_code`, `deviations`, `answer_text`) and `stop_reason`
  (`end`, `length`, `other`). A score is recomputed with `score_reply`; never trust the
  stored one for a report.
- `training.json` holds the run's `settings` dict in full and `key.settings_sha256`.
  `TrainingConfig.settings(False)` gives the current one: a flat diff of dotted keys says
  what changed.
- `check_adapter` and `selected_adapter` compare model, revision, prompt, and split, not
  the training settings: `--adapter selected` keeps working across an iteration.
- L4 compares a run's settings hash with `config/training.yaml` now. After an iteration
  the file must be the selected run's (D19).
- The stand-in split's dev examples all have the gold program `subtract(920, 95)`
  (answer 825), so a test can produce any category with a crafted reply:
  `subtract(95, 920)` is `order`, `subtract(920, 96)` is `evidence`, `add(920, 95)` is
  `operation`, `subtract(920, 95), divide(#0, 95)` is `steps_more`.
- The throwaway probe (`probe_errors.py` in the session scratchpad) is the reference for
  the category rules; its counts are in spec section 2.2.

### Gotchas for this version

- **A committed file never holds a reply.** `--show` prints and writes nothing. A test
  scans `reports/` under the temporary root for the stand-in replies and secret words.
- **Category precedence is the spec's order**: cut off, no program, bad program, not
  executable, kind, steps, operation, evidence, order, then scale and sign for a program
  equal to gold. `evidence` beats `order` when both occur.
- **A reversed division is `order`, not `sign`**: compare programs, not values.
- **`local_qlora/` must not import `frontier/`**, and `evaluation/failures.py` imports
  neither path.
- **No import cycle**: `analysis` and `iteration` may import `selection`'s gathering and
  `adapters`; `selection` imports `iteration` only for the record's read and write.
  Keep those two functions free of any import of `analysis`.
- **J10 changes every plan's check count** from nine to ten. Update the tests that count.
- **A selection over another run than the recorded one is refused without an iteration
  record**, except with `--check` (the rehearsal on a local smoke run must keep working
  when a real record exists).
- Shell heredocs mangle quotes on this machine: edit with scripts written by the Write tool.

## Implementation Blueprint

### Modules

```
qlora_finetuning/evaluation/failures.py      classify, run_failures, failure_counts      (no model, any arm)
qlora_finetuning/evaluation/runner.py        + run_gold(run_id, context): the gold replies of a run's examples
qlora_finetuning/local_qlora/selection.py    gather_scored() split out; choose across two runs; schema 2
qlora_finetuning/local_qlora/analysis.py     analyze_run, render_analysis                 (report: counts and ids)
qlora_finetuning/local_qlora/iteration.py    changed_variables, propose, iteration_status, read/write record
qlora_finetuning/local_qlora/sagemaker.py    + _check_registered (J10); the job entry says "iteration"
qlora_finetuning/local_qlora/training_config.py   + noise_examples, iteration record and analysis paths, the table
scripts/qlora_analyze.py, scripts/qlora_iterate.py
```

### Pseudocode

```python
# failures.py
def classify(outcome, stop_reason, gold, config, dsl_config) -> str:
    score = score_reply(outcome, gold, config, dsl_config=dsl_config)
    if score.execution_correct: return RIGHT
    if stop_reason != END: return CUT_OFF
    if not score.executed:
        return {REPLY: NO_PROGRAM, PROGRAM: BAD_PROGRAM, EXECUTION: NOT_EXECUTABLE}.get(stage, OTHER)
    if score.mismatch == KIND: return KIND
    predicted = parse_program(outcome.program); g = gold.program        # None -> OTHER
    if len differ: return STEPS_MORE / STEPS_FEWER
    if operations differ: return OPERATION
    per step: same | reversed | differs (a table step compares its first argument only)
    if any differs: return EVIDENCE
    if any reversed: return ORDER
    return SCALE / SIGN by score.mismatch, else OTHER

# selection.py
scored = gather_scored(run, ...)           # everything up to and including L6, as a dataclass
best = choose(scored.rows)                 # v3.04's rule, unchanged
existing = the record on disk, if any
if existing names another run:             # not this run, not in its chain
    iteration = read_iteration(...)        # must name existing's first run and this run's settings
    winner = second if (exec, format) of best > incumbent's else first      # a tie keeps the first
    write schema 2; keep the first record as adapter_selection_first.*; note the outcome in the iteration record

# iteration.py
def propose(finding, reason, ...):
    checks = [I1 triggered, I2 analysed, I3 one variable, I4 the first time, I5 the same experiment]
    if all pass: write reports/local_qlora/iteration.json

# sagemaker.py
def _check_registered(options):            # J10
    not a full train from the start        -> PASS
    no selection record                    -> PASS "no run is selected yet: this is the first full run"
    no iteration record for these settings -> FAIL "register the change first"
    a second run already made              -> FAIL "the one corrective iteration has been made"
    not on the first job's workspace       -> FAIL
```

### Error handling

- Everything a user can cause is a `ConfigError` (exit code 2 in the scripts): an unknown
  run, no selection record where one is required, a rehearsal that would write, a damaged
  record, a second run without a registered change.
- A failed check is reported, not raised (exit code 1), and nothing is written.
- A run whose predictions cannot be read is an `EvaluationError` (a `ConfigError`).

### Tasks, in order

```yaml
Task 1 - config and loader:
  config/training.yaml: selection.noise_examples: 10; an "iteration" block (record_file, analysis_file,
    the table T1 to T3 as variable and value)
  training_config.py: noise_examples, iteration_record, analysis_record, iteration_changes; none of it in settings()

Task 2 - evaluation/failures.py and runner.run_gold:
  the categories of spec 4.1; run_failures (one entry per example, with what the analysis groups by);
  failure_counts; export from qlora_finetuning.evaluation

Task 3 - selection.py:
  gather_scored() returning a dataclass; select_checkpoint keeps its outcome and its record for one run;
  the comparison with a recorded first run (D16 to D18); schema 2; read_selection reads 1 and 2

Task 4 - analysis.py and scripts/qlora_analyze.py:
  the report of spec 4.2; the finding; --check, --show, --split/--limit as rehearsal

Task 5 - iteration.py and scripts/qlora_iterate.py:
  changed_variables; I1 to I5; propose; status with D19

Task 6 - sagemaker.py:
  J10; "iteration": true in the job entry of a second run; the script needs no new flag

Task 7 - tests:
  tests/evaluation/test_failures.py; tests/local_qlora/test_analysis.py, test_iteration.py,
  test_selection_iteration.py, J10 in test_sagemaker_selection.py; tests/scripts for both scripts;
  update the counts of checks

Task 8 - on this machine (free): the two rehearsals of the success criteria; plan train; iterate propose

Task 9 - documents: runbook, AWS run sheet, evaluation_run.md, coding.md, gotchas.md, architecture,
  req_004 roadmap note; spec section 11 (the code), roadmap, execution plan, implementation summary
```

## Validation Loop

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest tests/evaluation tests/local_qlora tests/scripts -q
venv\Scripts\python -m pytest -q
# with the training stack blocked (the session's block_stack plugin on PYTHONPATH):
venv\Scripts\python -m pytest -q -p block_stack

venv\Scripts\python -m scripts.qlora_analyze --run 20261119T083236Z-smoke-qwen3-4b-smoke-5477 --split quick_dev --limit 24 --check
venv\Scripts\python -m scripts.qlora_iterate propose --finding T1 --reason "rehearsal"
venv\Scripts\python -m scripts.qlora_iterate status
venv\Scripts\python -m scripts.sagemaker_job plan train --scope measured        # read-only; signed out it fails J1
```

## Final validation Checklist

- [x] Tests and lint green, with and without the stack
- [x] The categories reproduce spec section 2.2 on the stored runs
- [x] No file under `reports/` holds a reply; `--show` writes nothing
- [x] A second run cannot be selected or launched without a registered single change
- [x] Nothing was created in AWS
- [x] Documents, spec, roadmap, execution plan, implementation summary

## Time Estimate

| Task | Time |
|---|---|
| 1 Config and loader | 20 min |
| 2 Categories | 45 min |
| 3 Selection across runs | 75 min |
| 4 Analysis and its script | 75 min |
| 5 Iteration and its script | 60 min |
| 6 J10 | 30 min |
| 7 Tests | 120 min |
| 8 Rehearsals | 15 min |
| 9 Documents | 45 min |
| **Total** | **6–8 hours** |

The risk is in task 3: `select_checkpoint` is split while 34 tests watch it, and the
two-run comparison has several ways to be re-run (the first run again, the second again,
after a revert). Confidence 7/10: nothing here meets AWS or a model, and the stand-in world
of v3.04's tests already exists; the open part is the number of states the selection record
can be in.

## Anti-Patterns to Avoid

- Do not launch, stop, or keep anything; do not sign in.
- Do not write a reply, a gold program, or any example text to a committed file.
- Do not choose the change in code: the analysis reports a finding, the owner decides.
- Do not let a category depend on reading: every rule is computed.
- Do not add a setting of spec D9 to the allowed variables, and do not compare settings by
  hash where the variable must be named.
- Do not change what `select_checkpoint` returns or writes for a single run.

## Time Taken for PRP Generation

**~15 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 6 min |
| Research existing PRPs and patterns | 2 min |
| Writing PRP document | 5 min |
