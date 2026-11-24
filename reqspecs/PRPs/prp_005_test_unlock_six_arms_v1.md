# PRP: Test unlock and six-arm evaluation (req_005 v5.01)

**Feature Name**: test_unlock_six_arms (comparison v5.01)  
**Version**: 1  
**Created**: 2026-11-22  
**Status**: ✅ IMPLEMENTED (code; the test split is still locked: the freeze, the unlock, and the runs are the owner's)  
**Implemented**: 2026-11-22  
**Implementation Summary**: [impl_005_20261122_122611.md](../impl_summary/impl_005_20261122_122611.md)  
**Confidence Score**: 7/10  
**Estimated Time**: 6–7.5 hours of code and documents, plus the owner's runs (about 7 hours of waiting, most of it Nova Pro at 18 requests a minute; about $6.50)  
**Origin**: `reqspecs/reqs/req_005_comparison/req_005_01_spec.md`  
**Template**: `prp_base.md` (complex: three new modules, two scripts, changes to the lock, the bundle, the runner, and the launcher)  

---

## Goal

Everything needed to read the test split once and safely:

- `python -m scripts.comparison_freeze` (`check`, `freeze`, `status`, `unlock`): a committed
  freeze record of the six arms' configurations, and the owner's unlock that names it.
- A gate, `check_test_run`, inside `run_evaluation`: after the unlock, a test run is possible
  only at the frozen configuration, once per arm.
- A test-only data bundle and a launch check J11, so A0 and A1 can answer the test
  questions inside SageMaker jobs.
- `python -m scripts.comparison_runs` (`preflight`, `status`, `record`, `void`): the
  committed six-arm record that later versions take their run ids from.

## Why & What

- **User value**: the comparison rests on one measurement. This version makes it hard to
  take that measurement wrongly: at a changed configuration, twice, partially, or on the
  wrong machine.
- **Technical**: the lock is the only guard today. Once `splits/test_unlock.json` exists,
  today's `scripts.evaluate --split test` accepts any predictor, any `--limit`, any number
  of times. The gate replaces the lock as the guard.

**The owner started this PRP without changing spec section 8, so its six decisions are
taken as recommended**: one complete result per arm, resume allowed, rerun only after a
void with a reason; sign-off by typing the freeze record's hash; two SageMaker jobs; a
five-question trial on quick-dev, never on test; the estimate of about $6.50 accepted; no
waiting for Nova Pro's training charge.

**The agent does not write the unlock record and reads no test row.** On the day this is
built the test split stays locked. Every refusal below is expected to be seen as a refusal.

## Success Criteria

- [x] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, also
      with the training libraries blocked.
- [x] `comparison_freeze check` runs on the real repository, prints F1 to F8 and the
      estimate, and writes nothing. (F7 fails until the build is committed: expected.)
- [x] `scripts.evaluate --predictor gold --split test` still answers `LOCKED`, exit 2.
- [x] `sagemaker_job bundle --split test` is refused: the split is locked.
- [x] In tests, on invented rows: each gate condition refuses with its own message; a
      second run is refused and is allowed after a void; a `fit` and `dev` bundle never
      serves `test`; five arms write no record.
- [x] No committed file holds example text, a name, or an address.

Live (the owner, spec section 9): freeze, commit, unlock, commit, preflight, four Nova
runs, two SageMaker jobs, record.

---

## Context

### References

```yaml
- file: reqspecs/reqs/req_005_comparison/req_005_01_spec.md   # D1 to D27, F1 to F8, C1 to C8
- file: qlora_finetuning/data/split_loader.py     # unlock_state, load_split: the lock
- file: qlora_finetuning/data/bundle.py           # export_bundle, read_bundle, bundle_split
- file: qlora_finetuning/evaluation/runner.py     # run_evaluation: where the gate goes; summarize_run
- file: qlora_finetuning/local_qlora/scoring.py   # expected_settings: a Qwen predictor's settings without the model
- file: qlora_finetuning/local_qlora/sagemaker.py # plan, launch, checks J1 to J10, SOURCE_PATHS
- file: qlora_finetuning/local_qlora/estimate.py  # measured_rates, estimate_job
- file: qlora_finetuning/local_qlora/adapters.py  # read_selection, selected_adapter
- file: qlora_finetuning/frontier/tuned.py        # tuned_attempt: the B1 of this experiment
- file: qlora_finetuning/frontier/tuned_report.py # the report pattern: checks, Markdown and JSON twins
- file: scripts/nova_tuned.py, scripts/qlora_select.py   # the script pattern
- file: tests/data/synthetic.py, tests/evaluation/conftest.py, tests/local_qlora/sagemaker_helpers.py
- file: reqspecs/references/project_patterns/split_loading.md, evaluation_run.md, check_script.md
- file: reqspecs/references/tech/sagemaker.md, reqspecs/references/common/gotchas.md
```

No web research was needed: this version adds no AWS call. The launcher's upload and job
creation are the ones v3.03 built and tested.

### Facts verified 2026-11-22

- **The six frozen dev runs** (ids in spec section 2.2) are on this machine with their
  `evaluation.json`. Their predictor blocks:
  - A0: `qwen3-4b`, model `Qwen/Qwen3-4B`, settings with `adapter: null`, `batch_size: 8`,
    the model revision. A1: `qwen3-4b-tuned`, the same settings with `adapter` = the
    weights' SHA-256 `a8c9561b...`. Both equal `expected_settings(local_cfg, revision, sha)`.
  - B0: `nova-lite` / `nova-pro`, model id the plain id, settings `{api: converse,
    temperature: 0, top_k: 1, ...}`. B1: name `<model>-tuned`, `model_id` the deployment
    ARN, the same settings, `usage_model_id` the base id.
- **The dev runs do not share config bytes.** The two B0 runs (2026-11-16) snapshotted
  `evaluation.yaml` `683372b7...` and `dsl.yaml` `7d2495e2...`; the four later runs
  `e5574d33...` and `a311fe59...`. Scores are equal under today's files: every run still
  passes R3 ("scores current"). So the freeze ties the dev runs to today's configuration by
  rescoring (R1 to R5 through `summarize_run`), and freezes the hashes of **today's** files.
  This replaces "the six runs agree on the hashes" in spec F4.
- **A job's code is `git archive` of `qlora_finetuning`, `scripts`, `config`, `splits`**, and
  two requirements files. `reports/` is not in a job. The gate runs inside the job, so the
  two records it reads live under `splits/`: `splits/config_freeze.json` and
  `splits/test_run_voids.json`. The Markdown twin of the freeze record goes to
  `reports/comparison/`. This is a change from the spec's file location, for that reason.
- ~~Config files have the same bytes here and in a job.~~ **Wrong, found while building**: git
  checks them out with CRLF here and a job has LF. Hashes are taken with Unix line ends.
- In a job the ledger and the run registry are the workspace's, not the repository's. The
  in-job gate sees only its workspace; J11 at plan time sees the local ledger and the job
  record.
- A job's name is `<prefix>-<task>-<commit 8>-a<attempt>`. Two `evaluate` jobs from one
  commit would have one name and the second would be adopted as the first. A launch writes
  the job record, which must be committed before the next launch (J2), so the commits
  differ in practice; otherwise `--attempt 2`.
- `estimate_job("evaluate", ...)` already works for 2,898 examples: A0 3,949 s ($1.54), A1
  2,681 s ($1.05), from job `qlora-score-63b89247-a1`.
- `split_ids("test")` works while locked (ids only). `load_gold_flags` already covers test
  examples.
- `scripts/` is not collected by pytest, but a script must still not be named `test_*`.

### Critical Gotchas

- **Never create `splits/test_unlock.json` in the repository during the build**, and never
  call `load_split("test")` on the real data. Tests use `tests/data/synthetic.py` rows and a
  temporary unlock record.
- **The gate must not import the runner** (the runner imports the gate, lazily, when the
  split is `test`). It imports `data`, `tracking`, `settings`, and reads JSON.
- **A refusal must come before a model is loaded and before a run exists.** `scripts.evaluate`
  and the `evaluate` task call the cheap part of the gate first; `run_evaluation` calls the
  whole gate with the predictor's description.
- **Only `sagemaker.py` and `staging.py` use botocore under `local_qlora/`; only `bedrock.py`,
  `staging.py`, `tuning.py` under `frontier/`.** `comparison/` imports none of them at module
  level. What F8 needs from the launcher (what Path A has cost) moves to `estimate.py`.
- **Canonical JSON for the freeze record**: `json.dumps(..., indent=2, sort_keys=True,
  ensure_ascii=True) + "\n"`, written with `newline="\n"`; its SHA-256 is over those bytes.
  No time stamp inside, so the same state gives the same hash.
- **Existing tests** that unlock a temporary split and then load `test` through a bundle
  (`tests/data/test_bundle.py`) change meaning: a `fit` and `dev` bundle still refuses.
- `config/sagemaker.yaml` gains a key: update the loader's required keys and its tests.
- Windows: `Path.write_text(..., newline="\n")` everywhere a hash is taken.

---

## Implementation

### Files

```
config/comparison.yaml                         CREATE  where the records live; the six arms
config/sagemaker.yaml                          MODIFY  test_bundle_dir
qlora_finetuning/comparison/config.py          CREATE  ComparisonConfig, load_comparison_config
qlora_finetuning/comparison/freeze.py          CREATE  freeze record, F1 to F8, estimate, unlock
qlora_finetuning/comparison/gate.py            CREATE  check_test_run, voids, prior test runs
qlora_finetuning/comparison/runs.py            CREATE  six-arm record, C1 to C8, void, preflight
qlora_finetuning/data/bundle.py                MODIFY  a test-only bundle
qlora_finetuning/data/split_loader.py          MODIFY  test from a test bundle, when unlocked
qlora_finetuning/evaluation/runner.py          MODIFY  the gate, when split is test
qlora_finetuning/local_qlora/estimate.py       MODIFY  committed_path_a (moved, no botocore)
qlora_finetuning/local_qlora/sagemaker.py      MODIFY  test bundle, J11, rehearse refuses test
qlora_finetuning/local_qlora/sagemaker_config.py  MODIFY  test_bundle_dir
qlora_finetuning/local_qlora/job.py            MODIFY  the cheap gate before the model loads
scripts/comparison_freeze.py, scripts/comparison_runs.py   CREATE
scripts/evaluate.py, scripts/sagemaker_job.py  MODIFY  REFUSED; bundle --split test
tests/comparison/ (helpers, test_config, test_freeze, test_gate, test_runs)   CREATE
tests/scripts/test_comparison_freeze.py, test_comparison_runs.py              CREATE
tests/data/test_bundle.py, test_split_loader.py, tests/evaluation/test_runner.py,
tests/local_qlora/test_sagemaker*.py, tests/scripts/test_evaluate.py, test_sagemaker_job.py  MODIFY
docs/runbook.md, aws_commands_run.md                MODIFY
reqspecs/1. PROMPTS/coding.md, splits/README.md,
reqspecs/references/project_patterns/split_loading.md, tech/sagemaker.md, common/gotchas.md  MODIFY
```

### Design

```python
# config/comparison.yaml
#   freeze_file: splits/config_freeze.json        voids_file: splits/test_run_voids.json
#   report_dir: reports/comparison                 arms: the six, each with path, predictor,
#   where (sagemaker | bedrock), and for Path B its model name in frontier.yaml
#   shared_configs: [prompt, split, token_budget, dsl, evaluation, data]

# comparison/gate.py   (no runner import, no AWS)
class TestRunRefused(ConfigError): __test__ = False
def read_freeze(cfg) -> dict | None                  # the record, shape checked
def freeze_sha256(cfg) -> str | None                 # over the file's bytes
def unlock_problems(cfg, split_cfg) -> list[str]     # unlocked, and by a record naming this freeze
def shared_state(config_dir, flags_file, names) -> dict   # sha256 per shared config, gold flags
def read_voids(cfg) -> list[dict]; def voided_ids(cfg) -> set[str]
def prior_test_runs(arm, tracking_cfg, voided) -> list[dict]   # registry runs of the arm on test
def complete_test_results(arm, results_file, voided) -> list[str]
def check_test_request(*, arm, name, where, scope, limit, resume, ctx) -> None
    # cheap: unlocked + freeze named; arm is one of the six (or gold, arm "none");
    # scope measured; no limit; A arms: where sagemaker and QLORA_DATA_BUNDLE set;
    # B arms: where bedrock; shared_state equals the freeze's; prompt id, manifest hash,
    # reply limit equal; no complete unvoided result; an unfinished earlier run -> resume it
def check_test_run(described, *, arm, where, scope, limit, resume, ctx) -> None
    # check_test_request, then name, model_id, settings == freeze["arms"][arm]["predictor"]

# evaluation/runner.py, in run_evaluation, before _load_gold:
#   if split == "test": from ...comparison.gate import check_test_run; check_test_run(...)
# RunnerContext gains comparison_cfg: Any = None

# comparison/freeze.py
def current_arms(cfg, ...) -> dict[arm, {"predictor": {name, model_id, settings}, "identity": {...}}]
    # A0/A1: local_qlora config + expected_settings (+ the selection record's hash);
    # B0: frontier config; B1: tuned_attempt() -> deployment ARN, job ARN, export id, attempt
def dev_runs(cfg) -> dict[arm, run_id]     # selection record (untuned, selected);
                                           # tuned report (the dev pair of each model)
def build_freeze(...) -> dict              # arms, shared, contract, dev results (counts),
                                           # estimate, checks F1 to F8, passed
def estimate_test_passes(...) -> dict      # Path A: estimate_job; Path B: dev tokens x
                                           # test ids / dev examples x pricing
def write_freeze(record, cfg) -> str       # JSON under splits/, Markdown under reports/; the hash
def unlock(cfg, split_cfg, *, yes, freeze_prefix, git) -> dict   # D7, D8; writes the unlock record
def freeze_status(...) -> dict             # frozen? unlocked? does it still hold?

# comparison/runs.py
def arm_status(...) -> list[dict]          # per arm: done | unfinished | missing, run ids
def build_record(...) -> dict              # C1 to C8; per arm counts, tokens, latency, job
def write_record / render_markdown / render_json
def void_run(run_id, reason, *, yes, ...)  # the run and the runs it continues; committed file
def preflight(...)                         # gold on test through the runner; prompt lengths
```

Freeze checks, as built (F4 and F5 reworded for the fact above):

| Id | Passes when |
|---|---|
| F1 | Selection record readable, verdict present, `selected_adapter(...)` resolves for today's model, prompt, and manifest; no iteration registered and unfinished |
| F2 | `tuned_attempt()` succeeds for both models |
| F3 | Each arm has one dev run; `summarize_run` says `result: true` for each (R1 to R5 under today's configuration) |
| F4 | The six runs agree on prompt id, manifest hash, reply limit, example count, and example order |
| F5 | Each run's recorded predictor (name, model id, settings) equals `current_arms()` |
| F6 | No unlock record; no run in the registry or the results file with split `test` |
| F7 | Working tree clean (the ledger prefix ignored, as the registry does) |
| F8 | Estimate fits in what is left of each path's cap |

Unlock record (`splits/test_unlock.json`): `{"schema": 1, "manifest_hash", "freeze_sha256",
"freeze_commit", "unlocked_on", "unlocked_by": "the owner, req_005 v5.01"}`.

Launcher:

```python
# sagemaker.py
def _bundle_dir(self, options): test_bundle_dir if task == "evaluate" and split == "test" else bundle_dir
# plan(): bundle prefix and J3 from that folder; launch(): uploads that folder
def _check_test_job(self, options) -> J11:
    # not a test job -> PASS. score/train/smoke with split test -> FAIL.
    # evaluate on test: check_test_request(where="sagemaker", in_job=True-equivalent) with the
    # local ledger; expected predictor (expected_settings) == freeze; for A1 the adapter is
    # the selected one; no test job of this arm in the job record that is unpulled
# rehearse(): refuses split test (D17)
```

### Tasks (~7 hours)

1. `config/comparison.yaml`, `comparison/config.py`, tests. (25 min)
2. `gate.py` with voids, prior runs, both checks; `RunnerContext.comparison_cfg`; the call in
   `run_evaluation`; tests for every refusal, gold, resume, void. (80 min)
3. Bundle and loader: test-only bundle, `load_split("test")` in a job; tests. (40 min)
4. `freeze.py`: current arms, dev runs, F1 to F8, estimate, write, unlock, status; tests on a
   constructed repository state. (100 min)
5. `scripts/comparison_freeze.py`; tests; run `check` on the real repository. (30 min)
6. Launcher: `test_bundle_dir`, bundle choice, J11, `rehearse`, `bundle --split test`,
   `committed_path_a` in `estimate.py`; `job.py` cheap gate; `scripts.evaluate` REFUSED;
   tests. (70 min)
7. `runs.py`: status, C1 to C8, record, void, preflight; `scripts/comparison_runs.py`;
   tests. (80 min)
8. Guards: who may ask for `test`; frozen values against a committed freeze record; no
   boto3 in `comparison/`. (20 min)
9. Documents: runbook, run sheet stage, `coding.md`, `splits/README.md`, pattern and tech
   references, gotchas. (35 min)
10. Full tests with and without the training stack, lint, the manual gates, spec section 11,
    roadmap, execution plan, implementation summary. (30 min)

### Error handling

| Situation | Behaviour |
|---|---|
| Test split locked | Unchanged: `LOCKED`, exit 2 |
| Unlocked by a record that names no freeze, or another freeze | `REFUSED`: "the unlock record does not name the freeze record ..."; exit 2 |
| A frozen value changed (a shared config, a setting, an adapter, a deployment) | `REFUSED`, naming what differs and that nothing frozen may change; exit 2 |
| `--limit`, scope `dev`, a Qwen arm outside a job, a Nova arm not on Bedrock | `REFUSED`, each with its own sentence; exit 2 |
| The arm has a complete result | `REFUSED`: names the run and `comparison_runs void` |
| The arm has an unfinished run and no `--resume` | `REFUSED`: prints the `--resume` command |
| Freeze: a check fails | Checks printed, nothing written, exit 1 |
| Freeze after the unlock | Refused: the record is never rewritten; exit 2 |
| Unlock without `--yes` or `--freeze`, wrong hash, dirty tree, freeze not committed | What it would do, or the reason; exit 2; nothing written |
| `bundle --split test` while locked | `LOCKED`, exit 2 |
| Record with fewer than six arms, or a failed C check | Status printed, nothing written, exit 1 |
| Void without `--yes` or without a reason | Refused, exit 2 |

---

## Validation

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q
venv\Scripts\python -m scripts.comparison_freeze check                 # F1 to F8 and the estimate; writes nothing
venv\Scripts\python -m scripts.comparison_freeze status                # not frozen, locked
venv\Scripts\python -m scripts.evaluate --predictor gold --split test   # LOCKED, exit 2
venv\Scripts\python -m scripts.sagemaker_job bundle --split test        # LOCKED, exit 2
venv\Scripts\python -m scripts.comparison_runs status                   # six arms missing
```

The second pytest run blocks the training stack as the earlier versions did.

## Checklist

- [x] The repository is still locked after the build; no test row was read
- [x] Every gate condition has a test and its own message
- [x] Gold passes the gate after an unlock; dev and quick-dev never reach it
- [x] A void lets exactly one more run through, and shows as a deviation
- [x] A `fit` and `dev` bundle never serves `test`; a test bundle holds nothing else
- [x] J11 passes and fails; `rehearse` refuses `test`
- [x] F1 to F8 and C1 to C8 each pass and fail in a test
- [x] The freeze record is byte-stable; Markdown and JSON agree; no example text
- [x] Import guards hold; lint clean; both pytest runs green
- [x] Runbook, run sheet, `coding.md`, references, spec section 11, roadmap, execution plan

## Implementation Time Estimate

**6–7.5 hours** of code and documents.

| Task | Time |
|---|---|
| Config | 25 min |
| Gate and runner | 80 min |
| Bundle and loader | 40 min |
| Freeze and unlock | 100 min |
| Freeze script | 30 min |
| Launcher, job, evaluate | 70 min |
| Six-arm record and its script | 80 min |
| Guards | 20 min |
| Documents | 35 min |
| Validation and status documents | 30 min |

The owner's part: a few minutes for the freeze, the unlock, and the preflight; about 30
minutes for each Nova Lite run; 2 hours 41 minutes for each Nova Pro run; about 1.1 and
0.75 hours for the two SageMaker jobs, which can run while the Nova runs go. About $6.50.

## Confidence Score: 7/10

Every building block exists and is tested: the lock, the bundle, the runner, the launcher,
the selection and tuning records. Three points off. The gate sits in the shared runner,
where a mistake would stop every arm. The freeze reads six real runs and three real records
whose shapes were made by four earlier versions. And the path that matters most, a real
test run after a real unlock, cannot be exercised during the build: it is covered on
invented rows only.

## Time Taken for PRP Generation

**~35 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 5 min |
| Codebase exploration | 18 min |
| Research existing PRPs and patterns | 4 min |
| Writing PRP document | 8 min |
