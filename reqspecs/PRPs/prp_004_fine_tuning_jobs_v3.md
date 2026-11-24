# PRP: Fine-tuning jobs (req_004 v4.03)

**Feature Name**: fine_tuning_jobs (frontier_tuning v4.03)  
**Version**: 3  
**Created**: 2026-11-16  
**Status**: ✅ IMPLEMENTED (code; nothing launched: the jobs are the owner's)  
**Implemented**: 2026-11-16  
**Implementation Summary**: [impl_004_20261116_045338.md](../impl_summary/impl_004_20261116_045338.md)  
**Confidence Score**: 7/10  
**Estimated Time**: 5–6 hours of code, plus the owner's launches (hours of waiting, a day for the bill)  
**Origin**: `reqspecs/reqs/req_004_frontier_tuning/req_004_03_spec.md`  
**Template**: `prp_lean.md` (expanded, medium to high complexity)  

---

## Goal

`python -m scripts.nova_tune` with seven commands: `plan` (what a launch would do, with
checks T1 to T9, starting nothing), `launch` (one Bedrock customization job, created once
however often it is asked for), `status` (read the jobs and deployments, fetch the loss
curves), `deploy`, `stop`, `bill`, and `teardown`. A committed tuning record holds the job,
model, and deployment ARNs.

## Why & What

- **User value**: the paid step of Path B can be seen before it happens, cannot happen
  twice by accident, survives a closed terminal, and leaves the ARNs and the cost where
  v4.04, v5.01, and v5.02 will look for them.
- **Technical**: config keys, a tuning-record module, a job-control module over stub-able
  Bedrock and S3 clients, a script, in-memory fakes of both services for the tests.

The owner decided spec section 8 on 2026-11-16: **Bedrock's default learning rate and
warmup steps** (1.00E-5, 10) for both models; **Nova Lite first**; **Nova Pro only after
Nova Lite's charge has been checked on the bill**, enforced as check T9.

**The agent launches nothing.** It runs no command that creates, stops, or deletes anything
in AWS, and with the sign-in expired it cannot even read. The gate of v4.02 is closed, so
`plan` is expected to fail on T1 on the day this is built.

## Success Criteria

- [x] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, also
      with the training libraries blocked.
- [x] `venv\Scripts\python -m scripts.nova_tune plan` prints the plan for both models with
      T1 failing (gate closed) and the AWS checks skipped (no sign-in); exit 1.
- [x] A test shows a repeated launch ends with one job, and a launch with a failing check
      creates nothing and writes nothing.
- [x] `create_provisioned_model_throughput` appears nowhere in the package or scripts.
- [x] No command changes AWS without `--yes`.

Live (the owner, spec section 9): the whole chain from the baselines to two Active
deployments.

---

## Context

### References

```yaml
refs:
  - reqspecs/references/tech/bedrock_nova.md, boto3.md
  - reqspecs/references/project_patterns/run_tracking.md   # a train run; computed, then bill
  - reqspecs/references/project_patterns/check_script.md   # PASS/FAIL lines, exit codes
  - reqspecs/references/common/gotchas.md

code:
  - file: qlora_finetuning/frontier/training_estimate.py
    why: read_gate_record(); gate["estimate"]["training_tokens_one_epoch"], gate["inputs"],
         gate["files"][role]{bytes, sha256}, gate["staged"][role]["s3_uri"]; usd(); check()
  - file: qlora_finetuning/frontier/staging.py
    why: the checksum form S3 returns (base64 of the digest); the _call error mapping
  - file: qlora_finetuning/frontier/bedrock.py
    why: aws_session, client_config, check_identity, AUTH_EXCEPTIONS, AUTH_ERROR_CODES
  - file: qlora_finetuning/frontier/config.py, pricing.py
    why: FrontierConfig and FrontierModel to extend; ModelPrice.training_per_million
  - file: qlora_finetuning/tracking/registry.py, ledger.py
    why: start_run(scope="measured") refuses a dirty tree; read_git_info; LedgerRecord with
         training_tokens, external_id, billed_amount_usd (source bill only), replaces
  - file: scripts/nova_export.py
    why: the shape of a script with injected context, config, and client factory
  - file: tests/frontier/export_helpers.py, tests/frontier/conftest.py
    why: FakeS3, volumes_for, write_baseline_report, baseline_run; the invented split

docs (read 2026-11-16; summarized in spec sections 2.2 and 2.3):
  - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_CreateModelCustomizationJob.html
  - https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models-hp.html
  - https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-submit.html
  - https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-monitor.html
  - https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-stop.html
  - https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-analyze.html
  - https://docs.aws.amazon.com/bedrock/latest/userguide/deploy-custom-model-on-demand.html
```

### Facts verified 2026-11-16 (installed boto3 1.44.3; no network)

| Fact | Value |
|---|---|
| `create_model_customization_job` | `jobName`, `customModelName` (1 to 63 characters; letters, digits, hyphens), `roleArn`, `baseModelIdentifier`, `customizationType`, `trainingDataConfig={"s3Uri"}`, `validationDataConfig={"validators": [{"s3Uri"}]}`, `outputDataConfig={"s3Uri"}`, `hyperParameters` (text to text), `clientRequestToken` (letters, digits, inner hyphens, at most 256), `jobTags`, `customModelTags` as `[{"key", "value"}]`. Returns `{"jobArn"}` |
| `get_model_customization_job(jobIdentifier=)` | ARN or name. `status` in `InProgress`, `Completed`, `Failed`, `Stopping`, `Stopped`; `statusDetails.{validationDetails, dataProcessingDetails, trainingDetails}.status`; `failureMessage`; `outputModelArn`; `clientRequestToken`; `creationTime`, `endTime` (datetimes); `trainingMetrics.trainingLoss`; `validationMetrics[].validationLoss` |
| Deployment | `create_custom_model_deployment(modelDeploymentName, modelArn, clientRequestToken, tags)` returns `{"customModelDeploymentArn"}`; `get_custom_model_deployment(customModelDeploymentIdentifier=)` gives `status` in `Creating`, `Active`, `Failed`, and `failureMessage` |
| Deleting | `delete_custom_model_deployment(customModelDeploymentIdentifier=)`, `delete_custom_model(modelIdentifier=)` |
| Provisioned Throughput | `list_provisioned_model_throughputs()` returns `provisionedModelSummaries` |
| Metrics files | `step_wise_training_metrics.csv` and `validation_metrics.csv` under the output location, in a folder named after the job; columns `step_number`, `epoch_number`, `training_loss` or `validation_loss`, `perplexity` |
| Unknown | What `get_model_customization_job` raises for a name that does not exist (`ResourceNotFoundException` is declared; treat `ValidationException` the same and say so) |
| Today | Gate record: `passed: false`, `staged: null`. No tuning record. Sign-in expired |

### Critical Gotchas

- **A lost reply must not become a second job.** If Bedrock already has a job of the
  planned name with our request token, a launch adopts it and creates nothing.
- **Write the tuning record the moment the ARN is known**, inside the run, before anything
  else can fail.
- **`scope="measured"` refuses a dirty tree**, and the tuning record is a committed file:
  after one launch the owner commits before the next. T8 says so before the run is tried.
- **`bill` uses `replaces`**: the estimate's tokens then leave the effective ledger, so the
  cap check must price a `bill` record by its amount and a `computed` one by its tokens.
- **`import botocore` is allowed in `bedrock.py`, `staging.py`, and now `tuning.py` only.**
  The guard in `tests/frontier/test_predictor.py` names them.
- **Datetimes from boto3 are not JSON.** Store them as UTC text.
- **`tests/frontier/test_config.py` pins the committed `frontier.yaml`**: its `GOOD` dict
  changes with the file. `settings` must stay as it is.
- **Never print or store an example's text.** Loss values, ARNs, counts, and dollars only.
- **The agent runs `plan` and `status` only**, and only because they cannot change AWS.

---

## Implementation

### Files

```yaml
CREATE:
  - qlora_finetuning/frontier/tuning_record.py: Attempt, TuningRecord, read_tuning_record,
      write_tuning_record, render_markdown, report_paths
  - qlora_finetuning/frontier/tuning.py: JobPlan, build_plan, request_token,
      committed_path_b, BedrockTuning (preflight, launch, refresh, wait, deploy, stop,
      bill, teardown), summarize_metrics
  - scripts/nova_tune.py
  - tests/frontier/tuning_helpers.py: FakeBedrock, a staged gate record on the invented split
  - tests/frontier/test_tuning_record.py, test_tuning_plan.py, test_tuning_launch.py,
      test_tuning_status.py; tests/scripts/test_nova_tune.py
  - reqspecs/impl_summary/impl_004_<TIMESTAMP>.md

MODIFY:
  - config/frontier.yaml (version 3), qlora_finetuning/frontier/config.py,
    tests/frontier/test_config.py
  - qlora_finetuning/frontier/bedrock.py: bedrock_control_client
  - tests/frontier/export_helpers.py: FakeS3 gains list_objects_v2 and get_object
  - tests/frontier/test_predictor.py: the import guard
  - docs/runbook.md (section 6), aws_commands_run.md, coding.md, gotchas.md,
    bedrock_nova.md, architecture overview
  - the spec, req_004_roadmap.md, execution_plan.md
```

### Design

```python
# tuning.py
def build_plan(model_name, attempt, cfg, gate, pricing, bucket, role_arn) -> JobPlan:
    # names  f"{prefix}-{model}-{export_id}-a{attempt}"
    # token  sha256 of [model, export id, hyperparameters, attempt]
    # hyperParameters {"epochCount": "2", "learningRate": "0.00001",
    #                  "learningRateWarmupSteps": "10"}
    # output s3://<bucket>/<output_prefix><export_id>/<model>-a<attempt>/
    # estimated tokens = gate training_tokens_one_epoch * epochs; dollars from pricing

class BedrockTuning:
    def __init__(self, cfg, pricing, context, bucket, role_arn, bedrock=None, s3=None,
                 profile="qlora", sleep=time.sleep)
    def preflight(self, plan) -> list[check]      # T1..T9; AWS checks "skipped" without clients
    def launch(self, plan) -> Attempt             # adopt or create; record; ledger; run
    def refresh(self) -> TuningRecord             # jobs, deployments, metrics; reads only
    def wait(self) -> TuningRecord                # refresh until nothing is in progress
    def deploy(self, model_name) -> Attempt
    def stop(self, model_name) -> Attempt
    def bill(self, model_name, usd) -> Attempt    # ledger record, source bill, replaces
    def teardown(self, model_name, everything) -> Attempt
```

A check that needs AWS and has no client is `{"passed": None, "detail": "skipped: ..."}`.
`plan` prints `SKIP` for it and exits 1; `launch` requires clients, so nothing is skipped.

### Tasks (~5.5 hours)

1. **Config v3 and its tests** (~25 min).
2. **`tuning_record.py` and tests** (~35 min).
3. **`FakeBedrock`, the staged-gate fixture, `FakeS3` additions** (~35 min).
4. **`build_plan`, the token, T1 to T9, and tests** (~60 min).
5. **`launch` with adoption, the ledger record, the run; tests** (~50 min).
6. **`refresh`, `wait`, metrics; `deploy`, `stop`, `bill`, `teardown`; tests** (~70 min).
7. **Script and tests; the guards** (~40 min).
8. **Documents** (~30 min).
9. **Validation, status documents, summaries** (~20 min).

### Error handling

| Situation | Behavior |
|---|---|
| A check fails | `plan`: FAIL line, exit 1. `launch`: `REFUSED`, nothing created, nothing written, exit 1 |
| No sign-in | `plan`: AWS checks skipped, exit 1. Every other command that needs AWS: `CONFIG ERROR` with the `aws sso login` command, exit 2 |
| Missing `--yes` on a command that changes AWS | Prints what it would do and the cost; exit 1 |
| `create` raises (conflict, quota, validation, throttling) | Reported with the code; the run is failed; no tuning record entry; a repeat adopts the job if it was in fact created |
| A job of the planned name exists with another token | T6 fails |
| Job `Failed` or `Stopped` | Recorded with the phase and message; `status` exits 1; a new attempt needs `--attempt` |
| Metrics files missing or damaged | A note in the record; the job's status is kept |
| `deploy` for a model that has not completed; `stop` for a job that is not running; `teardown` for something not in the record | `REFUSED`, exit 1 |
| `delete` of something already gone | Treated as deleted |

---

## Validation

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q
venv\Scripts\python -m scripts.nova_tune plan
venv\Scripts\python -m scripts.nova_tune status
```

**Expected**: ruff clean; all tests pass; `plan` shows both models, `FAIL  T1` (the gate is
closed), `SKIP` for T3 to T6, and exits 1; `status` says no job has been launched.

---

## Checklist

- [x] Files created and modified as listed
- [x] Every command and every check has passing and failing tests
- [x] Lint clean; tests green with and without the training stack
- [x] `plan` and `status` shown
- [x] Runbook (with teardown), references, `coding.md`, architecture updated
- [x] Roadmap, execution plan, spec status updated; implementation summary created
- [ ] Owner: the whole chain of spec section 9, from the baselines to two Active deployments

---

## Implementation Time Estimate

**Total: 5–6 hours**

| Task | Time |
|---|---|
| Config | 25 min |
| Tuning record | 35 min |
| Fakes and fixtures | 35 min |
| Plan and checks | 60 min |
| Launch | 50 min |
| Status, deploy, stop, bill, teardown | 70 min |
| Script and guards | 40 min |
| Documents | 30 min |
| Validation and status documents | 20–40 min |

## Confidence Score: 7/10

The request and response shapes are read from the installed boto3 and every precondition
has an existing source of truth. Three points are held back, more than usual, because
**none of this can be tried against Bedrock before the owner's first launch**: what Bedrock
answers for a job name that does not exist; where exactly the metrics files land; and
whether a real job's `statusDetails` and times look as the definitions say. The first
launch is Nova Lite's on purpose.

## Time Taken for PRP Generation

**~10 minutes** (the research was done while writing the spec the same day)

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 3 min |
| Research existing PRPs and patterns | 1 min |
| Writing PRP document | 4 min |
