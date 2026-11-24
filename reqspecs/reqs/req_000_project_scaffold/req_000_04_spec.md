# Project Scaffold

Spec Group      :   req_000_project_scaffold  
Feature Name    :   project_scaffold  
Version         :   4  
Status          :   ✅ Complete 2026-11-15 (see `reqspecs/PRPs/prp_000_run_registry_ledger_v4.md`); CI green on pull request #2 and on `main` (2026-11-15)  
Last updated    :   2026-11-15  

## 1. v0.04 — Run registry and cost ledger

The project's deliverable is a cost and accuracy comparison. The cost half is only as good as
the record of what each run used: how long it ran, on what, how many tokens it consumed, and
which code and configuration produced it. Both training paths, the evaluation runner, and the
cost model all need to write to or read from that record, so it is built once, here, before
any of them.

This version adds two things to `qlora_finetuning/tracking/`:

- a **run registry**: every training or evaluation run gets an id, a folder, and a manifest
  saying exactly what was run;
- a **cost ledger**: append-only records of what each run consumed, in the units AWS bills.

It records quantities. It does not turn them into dollars; that is the cost model (v5.02).

**Depends on**: v0.02 (done). It uses `environment_report()` from v0.03, which is built.

## 2. What exists today (verified 2026-11-15)

| Item | State |
|---|---|
| `qlora_finetuning/tracking/` | An empty sub-package |
| `artifacts/` | Git-ignored. Holds `env_check/report.json` only |
| `qlora_finetuning/environment.py` | `environment_report()` returns versions and GPU facts as a plain dict |
| `settings.load_config()` | Reads `config/<name>.yaml`; each file has an integer `version` |
| Split manifest | Does not exist yet (req_001 v1.02) |

How AWS reports billed usage, read from the installed boto3 service definitions:

| Usage | Where AWS reports it | Field |
|---|---|---|
| SageMaker training or evaluation job time | `sagemaker:DescribeTrainingJob` | `BillableTimeInSeconds` (per instance; multiply by `ResourceConfig.InstanceCount`), plus `TrainingTimeInSeconds` |
| Bedrock inference tokens, base or custom model | The response of each `bedrock-runtime:Converse` call | `usage.inputTokens`, `usage.outputTokens` |
| Bedrock fine-tuning tokens | **Not reported by any API.** `bedrock:GetModelCustomizationJob` returns status, hyperparameters, and loss metrics, but no token count | — |

So billed training tokens for Nova cannot be read back. They can only be computed (tokens in
the training file times the number of epochs) or taken from the AWS bill afterwards.

## 3. Locked decisions

| # | Decision | Reason |
|---|---|---|
| D1 | The ledger stores quantities, not derived costs: seconds, instance type and count, token counts, number of questions. Dollars appear only when an amount was read from the AWS bill | Rates change and live in `config/`; the cost model multiplies. Keeping them apart means a rate correction never rewrites the ledger |
| D2 | Every ledger record says where its numbers came from: `measured` (our own clock or counter), `aws_api` (a field AWS returned), `computed` (derived, for example tokens times epochs), or `bill` (read from the AWS bill by the owner) | Section 2: one kind of usage cannot be read back, so the report must be able to tell reported numbers from estimates |
| D3 | The ledger is append-only. A wrong record is never edited; a new record names the one it replaces, and readers skip replaced records | An audit trail for money figures |
| D4 | The ledger and the index of runs are **committed** to git, in a top-level `ledger/` folder. Everything bulky about a run (config snapshot, full manifest, predictions, checkpoints) stays under git-ignored `artifacts/runs/<run_id>/` | The two files are small text and are the evidence behind every cost figure in the report. Committed, the figures reproduce from a clean checkout and cannot be lost with a local folder. **This departs from the plan**, which put all run data under `artifacts/` |
| D5 | A run id is `<UTC time>-<kind>-<label>-<4 hex>`, for example `20261115T061500Z-train-a1-3f9a` | Sorts by time, readable, and unique without a counter |
| D6 | A run's manifest records: what it was (kind, path, arm, where it ran, free-text label), the git commit and whether the working tree had uncommitted changes, the package version, `environment_report()`, a hash and a copy of every config file it used, and the split-manifest hash | The plan's list, plus the environment report, so a cloud run and a local run can be compared (v3.03) |
| D7 | A run marked `measured` (one whose numbers go into the report) refuses to start from a working tree with uncommitted changes, unless the caller passes an explicit override, which is recorded. Runs marked `dev` only record the fact | A reported number must trace to a commit |
| D8 | The split-manifest hash is passed in by the caller and may be absent until req_001 v1.02 exists | This version must not depend on req_001 |
| D9 | Where git is not available (inside a SageMaker job), the commit is read from the `QLORA_GIT_COMMIT` environment variable, which the launcher sets; otherwise it is recorded as `unknown` | v3.03 runs the same entry point in the cloud |
| D10 | This version makes no AWS call. It provides two pure functions that turn an AWS response (passed in as a dict) into a ledger record: one for `DescribeTrainingJob`, one for a `Converse` usage block. The calls themselves belong to v3.03 and v4.01 | Testable without AWS; one place defines the mapping |
| D11 | Token usage is recorded once per run as totals, not once per question | Per-question detail already lives in the predictions file (v2.03); the ledger stays small |
| D12 | Bedrock fine-tuning tokens are recorded as `computed` when the job is launched, and a second record with source `bill` replaces it when the owner reads the billed figure | Section 2 |
| D13 | Paths and the allowed values for kind, arm, and phase live in `config/tracking.yaml` | Working agreement 13 |
| D14 | Records never contain a credential, token, or email address. Account ids and resource names are allowed; they are already in `config/aws.yaml` | Working agreement 15 |
| D15 | Timestamps are UTC, ISO 8601, ending in `Z`. Dollar amounts are stored as text to avoid rounding | Unambiguous across the laptop and the cloud |

## 4. Scope to build

### 4.1 `config/tracking.yaml`

`version: 1`; `runs_dir: artifacts/runs`; `ledger_file: ledger/ledger.jsonl`;
`runs_index_file: ledger/runs.jsonl`; the allowed kinds (`train`, `eval`, `export`, `smoke`,
`other`), arms (`A0`, `A1`, `B0-Pro`, `B1-Pro`, `B0-Lite`, `B1-Lite`, `none`), places
(`local`, `sagemaker`, `bedrock`), scopes (`dev`, `measured`), and ledger phases (`train`,
`eval`, `deploy`, `storage`, `other`).

### 4.2 Run registry: `qlora_finetuning/tracking/registry.py`

| Function | Behavior |
|---|---|
| `new_run_id(kind, label)` | Per D5. The label is reduced to lower-case letters, digits, and hyphens |
| `start_run(kind, label, path, arm, where, scope, configs, split_manifest_hash=None, allow_dirty=False)` | Creates `artifacts/runs/<run_id>/`, copies each named config file into its `config/` folder, writes `manifest.json` with status `started`, appends one line to the runs index, and returns a `Run`. Enforces D7. An unknown kind, arm, place, or scope raises `ConfigError` |
| `Run.finish(status, outputs=None)` | Sets status `completed` or `failed`, the end time, and named outputs (for example an adapter path or a custom model ARN); rewrites the manifest and appends a closing line to the runs index |
| `Run` as a context manager | An exception inside the block finishes the run as `failed` with the error type and message, then re-raises |
| `Run.record(...)` | Appends a ledger record for this run (4.3) |
| `load_manifest(run_id)`, `list_runs()` | Read back a manifest; list runs from the index, newest first, with their final status |

The manifest is one JSON file. The runs index is JSON Lines: a `started` line and a `finished`
line per run.

### 4.3 Cost ledger: `qlora_finetuning/tracking/ledger.py`

One record, one line of JSON:

| Field | Meaning |
|---|---|
| `schema` | 1 |
| `record_id` | Unique id of this record |
| `run_id` | The run it belongs to |
| `recorded_at` | UTC time |
| `phase` | `train`, `eval`, `deploy`, `storage`, or `other` |
| `path`, `arm`, `where` | Copied from the run |
| `instance_type`, `instance_count` | For compute billed by time |
| `model_id` | For usage billed by token |
| `wall_clock_seconds` | Elapsed time by our own clock |
| `billable_seconds` | Per instance, as AWS reports it |
| `input_tokens`, `output_tokens`, `training_tokens` | Token totals |
| `examples` | Number of questions answered, for cost per 1,000 questions |
| `billed_amount_usd` | Only with source `bill` |
| `source` | Per D2 |
| `source_detail` | Which API call, formula, or bill line the numbers came from |
| `external_id` | Job name or ARN |
| `replaces` | The `record_id` this record corrects, if any |
| `note` | Free text |

Rules: at least one quantity must be present; no quantity may be negative; `billable_seconds`
needs an instance type and count; any token count needs a model id; `billed_amount_usd` needs
source `bill`; `replaces` must name an existing record. A record that breaks a rule is rejected
before anything is written.

| Function | Behavior |
|---|---|
| `append(record)` | Validates, then appends one line and flushes it to disk |
| `read()` | Returns all records in order; a line that is not valid JSON or has an unknown schema raises `ConfigError` naming the line number |
| `effective()` | The records left after replaced ones are removed |
| `summarize(by=("arm", "phase"))` | Totals of every quantity per group over the effective records, with billable instance-seconds computed as `billable_seconds` times `instance_count`, and a count of records by source |
| `record_from_sagemaker_job(description, run, phase)` | Per D10: reads `BillableTimeInSeconds`, `TrainingTimeInSeconds`, the instance type and count, and the job name; source `aws_api`. A job that has not finished raises `ConfigError` |
| `record_from_converse_usage(totals, run, model_id, examples, wall_clock_seconds)` | Per D10 and D11: `totals` is the sum of the `usage` blocks of a run; source `aws_api` |

### 4.4 `scripts/ledger_summary.py`

`venv\Scripts\python -m scripts.ledger_summary` prints the runs (id, kind, arm, scope, status)
and the ledger totals per arm and phase, and says how many records are estimates
(`computed`) against reported (`aws_api`, `bill`, `measured`). Options: `--by` to change the
grouping, `--json` for machine-readable output. An empty or missing ledger prints "no records"
and exits 0. It follows the operator-script conventions: module form, by-path hint, exit code
2 for a config problem.

### 4.5 Committed `ledger/` folder

`ledger/README.md` explains the two files, the append-only rule, how to correct a record, and
that the folder is committed while `artifacts/` is not. The two `.jsonl` files are created by
the first real run, not by this version.

### 4.6 Tests

All tests write to temporary folders, need no network, and do not depend on the repository's
git state: git information is supplied by a function the tests replace.

- Registry: id format and label cleaning; the folder, config copies, and manifest contents;
  index lines; `finish`; the context manager on success and on an exception; unknown values;
  the uncommitted-changes rule for `measured` and `dev`, and the override; the
  `QLORA_GIT_COMMIT` fallback; `list_runs` order and status.
- Ledger: a valid record round-trips; each rule in 4.3 rejects its violation and writes
  nothing; `replaces` and `effective`; `summarize` totals including instance-seconds; a corrupt
  line; both mapping functions against response dicts shaped like section 2, including an
  unfinished job.
- One end-to-end test: a dummy run with two ledger records, then `summarize`, then the script.
- A guard: no record or manifest written by the tests contains a key or value that looks like
  a credential.

### 4.7 Documentation

- Runbook: a "Runs and the cost ledger" section: where things are, how to read the summary,
  how to correct a record, how to add a billed amount from the AWS bill.
- `high_level_plan.md` section 5: add `ledger/` to the repository layout.
- `coding.md`: every training and evaluation entry point starts a run and records what it
  used.
- `reqspecs/references/`: a short pattern note on how an entry point uses the registry and
  ledger, for v2.03, v3.02, v3.03, and v4.01 to follow.

## 5. Out of scope

- Any AWS call (v3.03, v4.01, v4.03).
- Prices, rates, and any dollar figure derived from them (v4.02 for the estimate, v5.02 for
  the cost model).
- Reconciling the ledger with the AWS bill (v5.02).
- Predictions, scores, and per-question records (v2.03).
- Uploading run folders to S3 (v3.03).

## 6. Known limits and follow-ups

- **No file locking.** Two processes appending at the same moment could interleave. The
  project has one operator and runs one job at a time; a cloud job writes its own files and
  the launcher appends locally afterwards.
- **Committing the ledger means a commit after each measured run.** Development runs add lines
  too. If that proves noisy, `dev` records can be moved to a git-ignored file later without
  changing the record format.
- **`computed` training-token records are estimates** until replaced from the bill. The summary
  shows how many remain.
- **The manifest trusts the caller** for the split-manifest hash and for which config files a
  run used.

## 7. Acceptance

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with and
  without the training stack installed.
- A dummy run in a temporary folder produces a run folder with a manifest and config copies, two
  index lines, and ledger records; `summarize` returns the totals written.
- A `measured` run refuses to start from a working tree with uncommitted changes.
- The two mapping functions produce valid records from response dicts shaped as in section 2.
- `venv\Scripts\python -m scripts.ledger_summary` runs against the real (empty) ledger and
  prints "no records".
- No AWS call is made anywhere in this version.

## 8. Next step

```
/generate-prp reqspecs/reqs/req_000_project_scaffold --version 4
```
