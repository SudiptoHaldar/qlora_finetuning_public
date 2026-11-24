# PRP: Tuned-model dev evaluation (req_004 v4.04)

**Feature Name**: tuned_dev_evaluation (frontier_tuning v4.04)  
**Version**: 4  
**Created**: 2026-11-18  
**Status**: ✅ IMPLEMENTED (code; no tuned model asked: the runs are the owner's)  
**Implemented**: 2026-11-18  
**Implementation Summary**: [impl_004_20261118_100412.md](../impl_summary/impl_004_20261118_100412.md)  
**Confidence Score**: 8/10  
**Estimated Time**: 3.5–4.5 hours of code, plus the owner's runs (about an hour, once v4.03 has two Active deployments)  
**Origin**: `reqspecs/reqs/req_004_frontier_tuning/req_004_04_spec.md`  
**Template**: `prp_lean.md` (expanded, medium complexity)  

---

## Goal

Two predictors, `nova-lite-tuned` and `nova-pro-tuned`, that ask the deployment named in the
tuning record, and `python -m scripts.nova_tuned`, which pairs each tuned run with its
baseline run and writes `reports/frontier/nova_tuned_report.md` and `.json` with a verdict
per model.

## Why & What

- **User value**: one command per arm scores a tuned model exactly as its baseline was
  scored, refuses to run against the wrong or a missing deployment, and one report says
  whether tuning helped by enough.
- **Technical**: an identity module (which attempt is this experiment's B1), a tuned
  factory over the existing `FrontierPredictor`, a small extension of the predictor
  description so usage is recorded under the base model, a report module, a script.

The owner decided spec section 8 on 2026-11-18: a tuned model has **missed** when it is less
than **10 points** above its own B0 on dev execution accuracy or its format validity is
under **98%**; **quick-dev and dev** are both scored; the **deployments are kept**.

**The agent asks no tuned model anything.** No deployment exists, so the tuned predictors
are expected to refuse on the day this is built.

## Success Criteria

- [x] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, also
      with the training libraries blocked.
- [x] `venv\Scripts\python -m scripts.evaluate --predictor nova-lite-tuned --split quick_dev`
      exits 2 with a message naming the missing step, and leaves no run.
- [x] A test shows the request for a tuned model equals the untuned one except for `modelId`.
- [x] A test shows usage of a tuned run is recorded under the base model id, so
      `committed_path_b` prices it.
- [x] The verdict flips exactly at 50 more correct of 499 and at 490 well-formed of 499.
- [x] `scripts.nova_tuned` makes no AWS call; the report holds no example text.

Live (the owner, spec section 9): trial, quick-dev, dev per model; then the report.

---

## Context

### References

```yaml
- file: reqspecs/reqs/req_004_frontier_tuning/req_004_04_spec.md      # D1 to D17, V1 to V6
- file: qlora_finetuning/frontier/predictor.py        # FrontierPredictor, build_predictor: extend
- file: qlora_finetuning/evaluation/predictor.py      # describe(): two optional fields
- file: qlora_finetuning/evaluation/runner.py         # key, ledger model_id, summarize_run
- file: qlora_finetuning/frontier/baseline_report.py  # the report pattern to mirror
- file: scripts/nova_baseline.py                      # the script pattern to mirror
- file: qlora_finetuning/frontier/tuning_record.py    # Attempt, read_tuning_record, COMPLETED, ACTIVE
- file: qlora_finetuning/frontier/training_estimate.py  # read_gate_record: export_id, inputs
- file: qlora_finetuning/frontier/tuning.py           # committed_path_b: unknown model_id counts as 0
- file: tests/frontier/conftest.py, tests/frontier/tuning_helpers.py  # StubProvider, FakeBedrock
- file: reqspecs/references/tech/bedrock_nova.md, reqspecs/references/common/gotchas.md
```

No web research was needed: this version adds no AWS call beyond one
`get_custom_model_deployment` read that v4.03 already uses and tests.

### Facts verified 2026-11-18

- The runner's resume key is `{name, model_id, settings}` of the predictor. With the
  deployment ARN as `model_id`, a resume against another deployment is refused with no new
  code (spec D5).
- The runner writes the ledger's `model_id` from `described["model_id"]`.
  `committed_path_b` counts a record whose model has no price as nothing, so an ARN there
  would hide tuned inference from the cap check. Usage must be recorded under the base id.
- `tests/frontier/test_predictor.py` asserts `describe(predictor)` of an untuned predictor
  equals exactly five keys. New description fields must be absent when not set.
- `Attempt.base_model_id` is the inference id that prices are keyed by.
- The gate record carries `export_id` and `inputs.prompt_id`, `inputs.split_content_hash`.
- `ExampleScore` fields in a prediction line's `score`: `format_valid`,
  `execution_correct`. A run that is a result passed R3, so stored scores are current.
- Account limits (Service Quotas, 2026-11-16): tuned Nova Pro deployment 20 requests a
  minute, Nova Lite 200.
- Baselines: dev execution accuracy B0-Lite 123 of 499, B0-Pro 190 of 499.

### Critical Gotchas

- **Only `bedrock.py`, `staging.py`, `tuning.py` may import botocore**, and no other module
  may import those three at module level. The live deployment read goes into `bedrock.py`
  and is imported inside the factory.
- **`settings` must stay equal to the baseline's**: the report compares them. The tuned
  model's identity goes into a separate `details` field, not into `settings`.
- **Counts decide, not rounded percentages**: compare `(b1 - b0) * 100 >= points * n`.
- **No example text in the report**: example ids, counts, hashes only.
- **`frontier.yaml` gains keys**: update `REQUIRED_KEYS`, `GOOD` in `test_config.py`, and
  check `nova_export --check` still says the gate record is current.
- The tuning record's ARNs contain the account id. That is already committed; it is not a
  credential.

---

## Implementation

### Files

```
config/frontier.yaml                          MODIFY  version 4; tuned_evaluation block
qlora_finetuning/frontier/config.py           MODIFY  four fields
qlora_finetuning/evaluation/predictor.py      MODIFY  describe(): optional usage_model_id, details
qlora_finetuning/evaluation/runner.py         MODIFY  ledger model id; run_predictions()
qlora_finetuning/frontier/tuned.py            CREATE  which attempt is the B1 of this experiment
qlora_finetuning/frontier/bedrock.py          MODIFY  deployment_status(): one read
qlora_finetuning/frontier/predictor.py        MODIFY  build_tuned_predictor, two factories
qlora_finetuning/frontier/tuned_report.py     CREATE  pairs, sign test, cost, verdict, V1 to V6
scripts/evaluate.py                           MODIFY  two entries
scripts/nova_tuned.py                         CREATE
tests/frontier/test_tuned.py, test_tuned_predictor.py, test_tuned_report.py   CREATE
tests/scripts/test_nova_tuned.py              CREATE
tests/evaluation/test_predictor.py, test_runner.py, tests/frontier/test_config.py,
tests/frontier/test_predictor.py, tests/scripts/test_evaluate.py   MODIFY
docs/runbook.md, aws_commands_run.md, aws_console_primer.md   MODIFY
reqspecs/references/tech/bedrock_nova.md, common/gotchas.md, 1. PROMPTS/coding.md,
architecture/architecture_overview.md         MODIFY
```

### Design

```python
# evaluation/predictor.py: describe() adds, only when the predictor has them:
#   "usage_model_id": text   the id the ledger records usage under (default: model_id)
#   "details": plain mapping what else identifies the model (never part of the resume key)
# runner: ledger model id = described.get("usage_model_id") or described["model_id"]
# runner: run_predictions(run_id, context) -> list[PredictionRecord]   (public, for reports)

# frontier/tuned.py  (no AWS)
def tuned_attempt(model_name, cfg, prompt_id, split_hash, record=None, gate=None) -> Attempt:
    # ConfigError, naming the command to run, unless: an attempt exists; job Completed; a
    # deployment ARN, recorded Active, not deleted; attempt.export_id == gate["export_id"];
    # gate inputs' prompt id and split hash are today's.
def tuned_details(attempt) -> dict   # model, attempt, base_model_id, ARNs, export id, hyperparameters

# frontier/bedrock.py
def deployment_status(control_client, deployment_arn, profile) -> str | None  # None: not found

# frontier/predictor.py
def build_tuned_predictor(model_name, *, context, cfg=None, aws=None, session=None):
    attempt = tuned_attempt(...)                      # local refusals first: no sign-in needed
    check_identity(sts); status = deployment_status(bedrock control, attempt.deployment_arn)
    if status != ACTIVE: raise ConfigError(... "run python -m scripts.nova_tune status")
    return FrontierPredictor(provider, attempt.deployment_arn, f"{model_name}-tuned",
                             cfg.settings, usage_model_id=attempt.base_model_id,
                             details=tuned_details(attempt))

# frontier/tuned_report.py  (no AWS, no file reads except write_report)
def paired(b0_lines, b1_lines) -> dict      # both_right, both_wrong, fixed, broken, ids, p_value
def sign_test(fixed, broken) -> float | None   # exact, two-sided; None when nothing changed
def verdict(b0, b1, cfg) -> dict            # met | missed, with the two conditions and reasons
def build_report(tuned, baselines, lines, record, gate, ledger, cfg, pricing, *, prompt_id,
                 manifest_hash) -> dict     # pairs, verdicts, checks V1 to V6, passed, partial
def render_markdown / render_json / report_paths / write_report
```

Verdict (spec D14, D15), on the dev pair only, from counts with `Decimal`:
`gain_met = (b1_exec - b0_exec) * 100 >= min_points * examples`;
`format_met = b1_format * 100 >= min_percent * examples`.

`scripts/nova_tuned.py --runs <B1 ids> --baselines <B0 ids>`: exit 0 the report was
written and every scored model met; 1 a check failed (nothing written) or a model missed
(report written); 2 a config problem or an unreadable run.

### Tasks (~4 hours)

1. Config: `tuned_evaluation` block, four fields, tests. (20 min)
2. `describe()` optional fields; runner ledger id and `run_predictions`; tests. (25 min)
3. `tuned.py` with every refusal; `deployment_status`; tests. (35 min)
4. `build_tuned_predictor`, factories, `evaluate.py` entries; tests through the runner. (40 min)
5. `tuned_report.py`: pairs, sign test, verdict, cost, checks, rendering; tests. (70 min)
6. `scripts/nova_tuned.py`; tests; run the two commands by hand. (30 min)
7. Documents: runbook, run sheet stage G, primer, references, gotchas, coding.md,
   architecture. (25 min)
8. Full tests with and without the training stack, lint, spec section 11, roadmap,
   execution plan, implementation summary. (25 min)

### Error handling

| Situation | Behaviour |
|---|---|
| No tuning record, or no attempt for the model | `ConfigError`: "run python -m scripts.nova_tune launch --model M --yes"; exit 2, no run |
| Job not Completed; no deployment; deployment not Active or deleted | `ConfigError` naming `nova_tune status` or `deploy`; exit 2 |
| Recorded Active, Bedrock says otherwise or not found | `ConfigError` with Bedrock's status; exit 2 |
| Tuned on another export, or the gate's prompt or split is not today's | `ConfigError`; exit 2 |
| No sign-in | The existing sign-in message, after the local refusals; exit 2 |
| Throttled out during a run | Unchanged: `PREDICTOR FAILED`, the `--resume` command |
| Report: a check fails | Checks printed, nothing written, exit 1 |
| Report: a model missed | Report written, `MISSED` printed, exit 1 |
| Report: one model only | Report written with `partial: true`, `PARTIAL` printed |

---

## Validation

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q
venv\Scripts\python -m scripts.evaluate --predictor nova-lite-tuned --split quick_dev   # exit 2: nothing launched
venv\Scripts\python -m scripts.nova_tuned --runs x --baselines y                          # exit 2: unknown run
venv\Scripts\python -m scripts.nova_export --check                                        # the gate record is still current
```

The second pytest run blocks the training stack as the earlier versions did.

## Checklist

- [x] Untuned predictors describe themselves exactly as before
- [x] Tuned request differs from untuned only in `modelId`
- [x] Ledger records of tuned runs carry the base model id and the tuned arm
- [x] Every refusal of D2 and D3 has a test and names a command
- [x] V1 to V6 each pass and fail in a test
- [x] Threshold boundaries; sign test against hand-computed values
- [x] No example text in the report; no AWS call in `nova_tuned`
- [x] Import guard still holds; `test` split still refused
- [x] Runbook, run sheet, primer, references, spec section 11, roadmap, execution plan

## Implementation Time Estimate

**3.5–4.5 hours** of code and documents.

| Task | Time |
|---|---|
| Config | 20 min |
| Predictor description and runner | 25 min |
| Tuned identity and the deployment read | 35 min |
| Tuned predictors and `evaluate` entries | 40 min |
| Report module | 70 min |
| Script | 30 min |
| Documents | 25 min |
| Validation and status documents | 25 min |

The owner's part, once v4.03 is done: about 10 minutes for Nova Lite's three runs, about 40
for Nova Pro's at 20 requests a minute, a minute for the report. About $0.50.

## Confidence Score: 8/10

The predictor, runner, report pattern, and both test stand-ins exist, and the only AWS call
added is a read v4.03 already makes. The two points off: nothing can be tried against a
real deployment (a cold start, the metric dimension, and the billing rate are unverified),
and the description change touches the shared runner, where an oversight would show up in
every arm.

## Time Taken for PRP Generation

**~25 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 4 min |
| Codebase exploration | 12 min |
| Research existing PRPs and patterns | 3 min |
| Writing PRP document | 6 min |
