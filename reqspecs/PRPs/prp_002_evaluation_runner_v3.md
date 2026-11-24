# PRP: Evaluation runner (req_002 v2.03)

**Feature Name**: evaluation_runner (dsl_evaluation v2.03)  
**Version**: 3  
**Created**: 2026-11-16  
**Status**: ✅ IMPLEMENTED  
**Implemented**: 2026-11-16  
**Implementation Summary**: [impl_002_20261116_011831.md](../impl_summary/impl_002_20261116_011831.md)  
**Confidence Score**: 8/10  
**Estimated Time**: 4–5 hours  
**Origin**: `reqspecs/reqs/req_002_dsl_evaluation/req_002_03_spec.md`  
**Template**: `prp_lean.md` (expanded, medium complexity)  

---

## Goal

One piece of code that evaluates any model: it takes a split and a predictor, writes a
predictions file as it goes, records the run and its usage, can be resumed after an
interruption, and gives totals recomputed from the file. A fake predictor proves the whole
path without a model, which is milestone M3.

## Why & What

- **User value**: all six arms are scored by the same loop, an interrupted paid run loses at
  most one batch, and a result is only a result when every example has a prediction.
- **Technical**: a protocol, a fake, a predictions-file module, a runner, a script, three new
  config keys, tests on invented rows.

The owner invoked this PRP and its execution without changing the three items of spec
section 8, so they are taken **as recommended**: a predictor failure stops the run and it is
resumed (never scored as wrong); every summary shows execution accuracy under the tight rule
beside the 0.5% headline; each complete run adds one line of counts to the committed
`ledger/evaluations.jsonl`.

## Success Criteria

- [x] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, also
      with the training libraries blocked.
- [x] `venv\Scripts\python -m scripts.evaluate --predictor gold --split dev`: 499 of 499 on
      all four metrics, R1 to R5 pass, 499 lines, a run in the registry, one ledger record.
- [x] The same on `quick_dev`: 200 of 200.
- [x] A test shows an interrupted run, resumed, equal to an uninterrupted one except for run
      ids, with ledger records that add up and no example asked twice.
- [x] `--summary <run_id>` reproduces a run's totals from its file.
- [x] Asking for `test` fails with the lock's message and predicts nothing.
- [x] M3 recorded in the execution plan.

---

## Context

### References

```yaml
refs:
  - reqspecs/references/project_patterns/run_tracking.md     # start_run, run.record, tests
  - reqspecs/references/project_patterns/reply_scoring.md    # run_reply, score_reply, summarize
  - reqspecs/references/project_patterns/canonical_record.md # texts unchanged; prompt id
  - reqspecs/references/project_patterns/split_loading.md    # load_split; the test lock
  - reqspecs/references/project_patterns/check_script.md     # PASS/FAIL lines, exit codes
  - reqspecs/references/common/gotchas.md

code:
  - file: qlora_finetuning/tracking/registry.py
    why: start_run arguments; a Run finishes once; leaving the block marks completed or failed
  - file: qlora_finetuning/tracking/ledger.py
    why: validate() rules; Ledger(cfg).effective(); LedgerRecord for a recovery record
  - file: qlora_finetuning/tracking/jsonl.py
    why: append_line writes one line and forces it to disk
  - file: qlora_finetuning/evaluation/metrics.py
    why: EvaluationConfig, Gold, score_reply, summarize; gains three config fields
  - file: qlora_finetuning/data/gold_check.py
    why: the report, check, and file conventions; load_gold_flags, build_flags, write_flags
  - file: scripts/finqa_gold_check.py
    why: main(argv, ...) shape, by-path hint, exit codes
  - file: tests/tracking/conftest.py
    why: a fixed GitInfo and environment={} so tests write under tmp_path only
  - file: tests/data/synthetic.py
    why: build_frozen_split(tmp_path, rows=...): 60 invented examples, fit 30 / dev 8 /
         quick_dev 3 / test 22
  - file: tests/data/test_split_loader.py
    why: how a test writes an unlock record under the temporary root
```

No web research was needed: there is no new library.

### Facts verified 2026-11-16

| Fact | Value |
|---|---|
| `start_run(kind, label, *, path, arm, where, scope, configs, split_manifest_hash, cfg, config_dir, git, environment, ...)` | Kinds include `eval` and `smoke`; paths and arms include `none` |
| A ledger record | Needs one quantity; a token count needs `model_id`; `run.record` refuses another run's id, but `Ledger(cfg).append(LedgerRecord(...))` accepts any |
| `TestLockedError` | A `SplitError`, which is a `ConfigError`. Raised by `load_split("test")` before data is read |
| `split_ids(name, cfg)` | Works for `test` while locked: ids only |
| `ReplyOutcome.to_dict()` | Tuples become lists; the value is a float or `yes` / `no` |
| `load_token_budget_config()` | Plain YAML; imports nothing from the training stack |
| Trial loop on dev | The gold reply scores 499 of 499; 0.03 seconds; about 825 bytes per line |
| `ledger/` | README only: this version's fake run writes the first lines |

### Critical Gotchas

- **Do not import the runner from `qlora_finetuning/evaluation/__init__.py`.** The runner
  imports the data package, and `qlora_finetuning.data.gold_check` imports
  `qlora_finetuning.evaluation.metrics`, which loads the package's `__init__` first: a cycle.
  `predictor`, `fake`, and `predictions` do not import the data package and may be exported.
- **A run cannot be reopened.** Resuming is a new run that copies the valid lines of the
  earlier file into its own folder and continues.
- **Refuse a resume before `start_run`**, so a refusal leaves no run behind.
- **Count an example as predicted only after its line is on disk.** The ledger record is
  written in a `finally`, so a failed run still records what it consumed.
- **A killed process writes no ledger record.** On resume, if the earlier run has lines in
  the file and no eval record, append a `computed` record for that run from its lines, or
  check R4 can never pass for that evaluation.
- **A line counts only if it ends in a newline.** A last piece without one is a cut-off line
  and is dropped; any other damaged line is an error naming its number.
- **`EvaluationConfig` gains fields**: `tests/evaluation/test_metrics.py` builds one by hand
  and must be updated. The allowances do not change, so the gold flags stay valid.
- **`summarize_run` needs the gold flags file**: tests write an empty one under the
  temporary root (`write_flags(build_flags({"flagged": [], "settings": {}}), config)`).
- **No raw reader** in `evaluation/`, `scripts/evaluate.py`, or their tests.
- **JSON**: `allow_nan=False`, `ensure_ascii=True`, `newline="\n"`.

---

## Implementation

### Files

```yaml
CREATE:
  - qlora_finetuning/evaluation/predictor.py: PredictionRequest, Prediction, Predictor,
      PredictorError, check_predictions, describe, STOP_REASONS, USAGE_SOURCES
  - qlora_finetuning/evaluation/fake.py: FakePredictor, gold_predictor
  - qlora_finetuning/evaluation/predictions.py: PredictionRecord, PredictionsError,
      append_records, read_predictions, outcome_from_dict
  - qlora_finetuning/evaluation/runner.py: RunnerContext, ResumeError, run_evaluation,
      summarize_run, latency_stats
  - scripts/evaluate.py
  - tests/evaluation/conftest.py, test_predictor.py, test_predictions.py, test_runner.py,
      test_runner_resume.py; tests/scripts/test_evaluate.py
  - reqspecs/references/project_patterns/evaluation_run.md
  - reqspecs/impl_summary/impl_002_<TIMESTAMP>.md

MODIFY:
  - config/evaluation.yaml: version 2; diagnostic.execution_relative, runner.progress_every,
      results_file
  - qlora_finetuning/evaluation/metrics.py: the three fields and their validation
  - qlora_finetuning/evaluation/__init__.py: docstring; export predictor, fake, predictions
  - tests/evaluation/test_metrics.py: the hand-built config; tests of the new keys
  - docs/runbook.md (section 9), coding.md, gotchas.md, reply_scoring.md,
    run_tracking.md, project_patterns/README.md, ledger/README.md,
    architecture/architecture_overview.md
  - the spec, req_002_roadmap.md, execution_plan.md (M3)
```

### Design

```python
# predictor.py
@dataclass(frozen=True)
class PredictionRequest: example_id: str; system: str; user: str; max_new_tokens: int
@dataclass(frozen=True)
class Prediction:
    text: str; input_tokens: int | None = None; output_tokens: int | None = None
    latency_seconds: float = 0.0; stop_reason: str = "end"      # end | length | other
class Predictor(Protocol):
    name: str; model_id: str | None; settings: Mapping[str, Any]
    batch_size: int; usage_source: str                           # measured | aws_api | computed
    def predict(self, requests: Sequence[PredictionRequest]) -> Sequence[Prediction]: ...
class PredictorError(Exception): run_id: str | None = None
def check_predictions(requests, predictions) -> list[Prediction]   # raises PredictorError
def describe(predictor) -> dict     # name, model_id, settings (JSON-plain), batch_size, usage_source

# runner.py
@dataclass
class RunnerContext:      # everything a test replaces; None means "load the real one"
    config; split_cfg; data_cfg; prompt; tracking_cfg; dsl_config
    max_new_tokens; git; environment; config_dir
    configs = ("evaluation", "prompt", "dsl", "token_budget", "split", "data")

def run_evaluation(predictor, split, *, label, path, arm, where, scope, kind="eval",
                   limit=None, resume=None, description="", context=None) -> dict:
    examples = load_split(split, ...)            # TestLockedError is not caught
    records  = [build_record(e, prompt) for e in examples]
    golds    = [Gold.from_target(r.target, ...) for r in records]     # fail before any run
    key      = {split, manifest_hash, prompt_id, max_new_tokens, predictor name/model/settings}
    done     = [] if resume is None else resumable_lines(resume, key, ids)   # ResumeError
    with start_run(kind, label, ...) as run:
        write evaluation.json; write the copied lines; recover the earlier run's ledger record
        try:
            for batch in batches(remaining, predictor.batch_size):
                predictions = check_predictions(batch, predictor.predict(batch))
                lines = [record(example, prediction, run_reply(...), score_reply(...))]
                append_records(path, lines); predicted += len(lines)
        except PredictorError as error:
            error.run_id = run.run_id; raise
        finally:
            if predicted: run.record(phase="eval", examples=predicted, wall_clock_seconds=...,
                                     tokens when model_id is set and every count is known)
            summary = summarize_run(run.run_id, context); write summary.json
        if summary["result"]: append the results line
    return summary

def summarize_run(run_id, context=None) -> dict:
    # read evaluation.json and the file; score every stored outcome again; totals by
    # summarize(scores, flagged=load_gold_flags(config)); the tight diagnostic; usage;
    # latency; stop reasons; checks R1 to R5; complete, partial, result
```

Checks, with one adjustment to be written back into the spec:

- **R1 is "every example asked for"**: with `limit`, the first N. A limited run can pass its
  checks; it is `partial`, and `result` (what D23 calls a result, and what gets a results
  line) is "checks pass and not partial". Otherwise an asked-for partial run would print FAIL.
- R4 compares, for each run id found in the file's lines, the examples of that run's ledger
  records with its lines.

### Script

`main(argv=None, context=None, predictors=None)`. `--predictor NAME --split NAME [--limit N]
[--resume RUN_ID] [--label TEXT] [--scope dev|measured] [--where local]`, or
`--summary RUN_ID` alone. `PREDICTORS` maps a name to a factory path (`module:function`,
imported when named), a kind, a path, and an arm; v2.03 registers `gold`. Exit 0, 1, 2 as in
spec 4.5. On a predictor failure it prints the command that resumes the run.

### Tests

| File | Cases |
|---|---|
| `test_predictor.py` | Requests; each malformed prediction; `describe`; the fake by table and by function, batches, fail after N, token counts |
| `test_predictions.py` | Round trip; the valid beginning; cut-off last line; damaged middle line; repeated, out-of-order, and unknown ids; too many lines |
| `test_runner.py` | Gold predictor; each wrong kind of reply; limit; summary equals `summarize`; changed allowance; tight diagnostic; latency; unknown tokens; ledger record; results line; R1 to R5 passing and failing; the test lock; guards; real data |
| `test_runner_resume.py` | Fail after N for several N and batch sizes; resumed file equals the uninterrupted one; never asked twice; ledger adds up; recovery record after a kill; each refusal |
| `test_evaluate.py` | Exit codes; `--summary`; `--resume`; unknown predictor; module form; by-path hint |

### Tasks (~4.5 hours total)

1. **Config keys and `metrics.py` fields; update the metrics tests** (~20 min).
2. **`predictor.py`, `fake.py`, tests** (~35 min).
3. **`predictions.py`, tests** (~35 min).
4. **`runner.py`: run, summary, checks** (~60 min).
5. **Runner and resume tests** (~55 min).
6. **Script and tests** (~30 min).
7. **Real runs on quick-dev and dev with the gold predictor** (~5 min).
8. **Documents** (~30 min).
9. **Validation with the stack blocked, status documents, summaries** (~15 min).

### Error handling

| Situation | Behavior |
|---|---|
| Predictor raises `PredictorError`, or returns something malformed | Run failed; lines kept; usage recorded; error carries the run id; script exits 1 with the resume command |
| Any other exception in a predictor | Propagates; the run is failed and the usage recorded the same way |
| Locked test split | `TestLockedError` from the loader, before any run is started; script exits 2 |
| Resume refused | `ResumeError` naming what differs or which line is damaged; no run is created; exit 2 |
| Resume of a run that already has every example | `ResumeError` pointing to `--summary` |
| An example whose record or gold reply cannot be built | An error naming the example before any run starts. None exists today (gold self-check) |
| Config value malformed | `ConfigError` naming the key |

---

## Validation

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q
venv\Scripts\python -m scripts.evaluate --predictor gold --split quick_dev
venv\Scripts\python -m scripts.evaluate --predictor gold --split dev
venv\Scripts\python -m scripts.finqa_gold_check --check
```

**Expected**: ruff clean; all tests pass; five PASS lines per run; 200 of 200 and 499 of 499;
the gold check still current. Also the unit tests with the training libraries blocked.

---

## Checklist

- [x] Files created and modified as listed
- [x] Every function and every check has passing and failing tests
- [x] Lint clean; tests green with and without the training stack
- [x] Real runs shown; M3 recorded
- [x] Runbook, references, `coding.md`, architecture updated
- [x] Roadmap, execution plan, spec status updated; implementation summary created

---

## Implementation Time Estimate

**Total: 4–5 hours**

| Task | Time |
|---|---|
| Config and metrics fields | 20 min |
| Predictor and fake | 35 min |
| Predictions file | 35 min |
| Runner | 60 min |
| Runner and resume tests | 55 min |
| Script and tests | 30 min |
| Real runs | 5 min |
| Documents | 30 min |
| Validation and status documents | 15–30 min |

## Confidence Score: 8/10

Everything the runner calls exists and was read; the loop itself was tried on dev. Two points
are held back: resuming has many edge cases (a cut-off line, a killed process with no ledger
record, a batch that fails half-way), and the import cycle between the evaluation package
and the gold self-check must be avoided on purpose.

## Time Taken for PRP Generation

**~12 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 5 min |
| Research existing PRPs and patterns | 1 min |
| Writing PRP document | 4 min |
