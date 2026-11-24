# PRP: SageMaker job packaging (req_003 v3.03)

**Feature Name**: sagemaker_job_packaging (qlora_local v3.03)  
**Version**: 3  
**Created**: 2026-11-19  
**Status**: ✅ IMPLEMENTED (code; the owner's smoke job remains)  
**Implemented**: 2026-11-19  
**Implementation Summary**: [impl_003_20261119_235853.md](../impl_summary/impl_003_20261119_235853.md)  
**Confidence Score**: 6/10  
**Estimated Time**: 7–9 hours of code and documents, about 10 minutes of local GPU time, then the owner's live steps  
**Origin**: `reqspecs/reqs/req_003_qlora_local/req_003_03_spec.md`  
**Template**: `prp_base.md` (shortened)  

---

## Goal

Make the training pipeline and the evaluation runner run as SageMaker training jobs: a data
bundle of `fit` and `dev`, a job runner (`scripts/job_run.py`), a launcher
(`scripts/sagemaker_job.py`: `bundle`, `rehearse`, `plan`, `launch`, `status`, `logs`,
`pull`, `stop`), an S3 workspace shared by a chain of jobs, a Path A budget check, a
committed job record, and billable seconds in the ledger. Rehearse the smoke task on this
machine. **Stop before anything is created in AWS**: the paid smoke job is the owner's.

## Why

- Every number of the comparison must come from AWS, so that Path A has a bill.
- The snapshot holds the test rows and must never be uploaded; a job needs its own, smaller
  input.
- A fault found in a free local rehearsal costs nothing; in a job it costs GPU time.

## What

The owner started this PRP and its execution without changing the five questions of spec
section 8, so they are **taken as recommended**: SageMaker training jobs created with plain
boto3 (Q8); `ml.g5.xlarge` (Q9); inference batch size 8 in `config/local_qlora.yaml`; AWS's
PyTorch 2.10 container with the pins installed at job start; a $100 cap for Path A and a
one-hour maximum for the smoke job.

The AWS sign-in is expired. The agent builds, tests, exports the bundle, rehearses the smoke
task locally, and runs `plan` signed out. It runs no command that needs a sign-in and
creates nothing in AWS. The version ends 🟡 operator-gated.

### Success Criteria

Code (the agent):

- [x] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
      and without the training stack blocked.
- [x] `load_split` gives the same examples from a bundle as from the snapshot; a bundle
      cannot hold a test row; without `QLORA_DATA_BUNDLE` nothing changes.
- [x] `sagemaker_job bundle`: 4,884 and 499 examples, no test id, the same SHA-256 twice.
- [x] `sagemaker_job rehearse smoke`: K1 to K7 pass and 48 examples are scored, reading the
      bundle in a folder that has no snapshot.
- [x] `sagemaker_job plan smoke`, signed out: J1 fails with the sign-in message, nothing created.
- [x] `scripts.evaluate`, `scripts.qlora_train`, and the Path B scripts behave as before.

Live (the owner, spec section 9): sign in, `aws_check`, `plan smoke`, `launch smoke --yes`,
`status`, `pull`.

## All Needed Context

### Documentation & References

```yaml
- file: reqspecs/reqs/req_003_qlora_local/req_003_03_spec.md
  why: decisions D1 to D28, the config, checks J1 to J8, sections 2.2 to 2.5 (API shapes,
       what AWS documents, prices, what could not be checked)
- file: reqspecs/references/tech/boto3.md
  why: sessions, error shapes, stub clients in tests
- file: reqspecs/references/project_patterns/run_tracking.md, split_loading.md, check_script.md
- file: reqspecs/references/common/gotchas.md
  why: section 2 (AWS), section 4 (heredocs mangle backslashes: write files with the file tool)

- file: qlora_finetuning/frontier/bedrock.py
  why: AUTH_EXCEPTIONS, AUTH_ERROR_CODES, sign_in_message, aws_session, check_identity:
       move to a shared module that bedrock.py re-exports
- file: qlora_finetuning/frontier/staging.py
  why: put_object with ChecksumSHA256, head_object with ChecksumMode, then verify
- file: qlora_finetuning/frontier/tuning.py, tuning_record.py, scripts/nova_tune.py
  why: plan with checks, launch with --yes, adopt an existing job, a committed record,
       committed_path_b (the cap arithmetic), the CLI's exit codes
- file: qlora_finetuning/tracking/ledger.py
  why: record_from_sagemaker_job(description, run, phase); Ledger.append validates
- file: qlora_finetuning/data/split_loader.py, snapshot.py
  why: load_split, _ids, RawExample (a frozen dataclass of plain values)
- file: qlora_finetuning/local_qlora/training.py, predictor.py; evaluation/runner.py
  why: run_training, build_predictor, build_tuned_predictor, run_evaluation, RunnerContext
- file: scripts/aws_check.py
  why: the role-trust and quota checks to mirror (the package may not import a script)

- url: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html
- url: https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html
- url: https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html
- url: https://github.com/aws/deep-learning-containers/tree/main/docs/src/data/pytorch-training
```

### Verified (2026-11-19)

```python
# Installed boto3 1.44.3, service model "sagemaker":
#   create_training_job(TrainingJobName*, RoleArn*, OutputDataConfig{S3OutputPath*},
#     AlgorithmSpecification{TrainingImage, TrainingInputMode*, ContainerEntrypoint,
#     ContainerArguments}, InputDataConfig[{ChannelName*, DataSource*{S3DataSource{
#     S3DataType*, S3Uri*}}}], ResourceConfig{InstanceType, InstanceCount, VolumeSizeInGB},
#     StoppingCondition{MaxRuntimeInSeconds}, CheckpointConfig{S3Uri*, LocalPath},
#     Environment, Tags)
#   describe_training_job -> TrainingJobArn, TrainingJobStatus (InProgress | Completed |
#     Failed | Stopping | Stopped), SecondaryStatus, FailureReason, TrainingStartTime,
#     TrainingEndTime, TrainingTimeInSeconds, BillableTimeInSeconds, ResourceConfig
#   stop_training_job, list_training_jobs(NameContains, StatusEquals)
#   Job name: 1 to 63 of [a-zA-Z0-9-]. An existing name raises ClientError "ResourceInUse".
# AWS documentation: ContainerEntrypoint and ContainerArguments are 1 to 100 strings of at
#   most 256 characters each; channels land in /opt/ml/input/data/<channel>; the checkpoint
#   folder (default /opt/ml/checkpoints) is kept in step with CheckpointConfig.S3Uri and
#   existing objects are written to it at the start; billing is TrainingStartTime to
#   TrainingEndTime, per instance; at the maximum runtime: SIGTERM, 120 s, status Stopped.
# Newest AWS PyTorch training container: 2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker.
# The committed source (package, scripts, config, splits, requirements): 253 KB compressed.
```

### Known Gotchas

```python
# CRITICAL: git archive converts line endings on Windows when core.autocrlf is on. A shell
#   script with CRLF does not run in the container. Archive with
#   `git -c core.autocrlf=false archive ...`, and test that the .sh in it has no "\r".
# CRITICAL: an entry-point string may be at most 256 characters: keep the bootstrap in a
#   committed script inside the archive, and the entry point a short `bash -c` line.
# CRITICAL: a bundle holds fit and dev only. Check every id against the manifest's test ids
#   on writing AND on reading. In bundle mode "test" is refused even with an unlock record.
# CRITICAL: nothing in this PRP may be run that needs a sign-in. launch and stop are the
#   owner's; tests use stand-in clients.
# CRITICAL: nothing under local_qlora/ imports frontier/, and the package never imports a
#   script. The sign-in helper goes to a shared module.
# GOTCHA: the model cache and pip's files must stay outside /opt/ml/checkpoints, or 14 GB
#   is copied to S3. A file deleted under the checkpoint folder is deleted in S3.
# GOTCHA: a job has no git: the commit comes from QLORA_GIT_COMMIT (tracking already reads it).
# GOTCHA: `settings.get_str(name, "")` reads the environment, then .env.
# GOTCHA: a ledger record needs a quantity; billable_seconds needs instance_type and count.
# GOTCHA: the code is uncommitted while it is being built, so `rehearse` must be able to
#   copy the working tree's files (and say so) when HEAD does not hold them yet.
# GOTCHA: scripts print ASCII only; a job's log lines may hold anything, so encode them.
```

## Implementation Blueprint

### Files

```yaml
CREATE:
  - config/sagemaker.yaml
  - qlora_finetuning/aws_common.py                 # sign-in helpers, shared by both paths
  - qlora_finetuning/data/bundle.py                # export_bundle, read_bundle, bundle_split
  - qlora_finetuning/local_qlora/job.py            # run_task: smoke | train | evaluate
  - qlora_finetuning/local_qlora/sagemaker_config.py
  - qlora_finetuning/local_qlora/sagemaker.py      # source, job definition, SageMakerJobs
  - qlora_finetuning/local_qlora/staging.py        # S3 upload with checksums; workspace download
  - qlora_finetuning/local_qlora/job_record.py     # the committed job record
  - scripts/job_bootstrap.sh, scripts/job_run.py, scripts/sagemaker_job.py
  - tests/data/test_bundle.py, tests/local_qlora/test_job.py, test_sagemaker_config.py,
    test_sagemaker.py, test_sagemaker_pull.py, sagemaker_helpers.py;
    tests/scripts/test_job_run.py, test_sagemaker_job.py
  - reqspecs/references/tech/sagemaker.md

MODIFY:
  - qlora_finetuning/data/split_loader.py   # QLORA_DATA_BUNDLE
  - qlora_finetuning/frontier/bedrock.py    # import the sign-in helpers from aws_common
  - qlora_finetuning/local_qlora/backend.py # fetch_weights (the job downloads the model)
  - config/pricing.yaml (sagemaker.training), config/local_qlora.yaml (batch_size 8)
  - runbook, gotchas, run_tracking.md, split_loading.md, coding.md, architecture, spec,
    roadmap, execution plan
```

### Tasks, in order

```yaml
Task 1 - aws_common.py; bedrock.py re-exports. The Path B tests must pass untouched.
Task 2 - the bundle and load_split's bundle mode, with tests.
Task 3 - config/sagemaker.yaml, prices, the loader; batch size 8.
Task 4 - job.py and scripts/job_run.py (pins check, the three tasks, job_result.json);
         backend.fetch_weights; scripts/job_bootstrap.sh.
Task 5 - sagemaker.py: build_source, job_name, job_definition, the checks, SageMakerJobs
         (plan, launch, status, logs, pull, stop); staging.py; job_record.py.
Task 6 - scripts/sagemaker_job.py with bundle and rehearse.
Task 7 - tests, lint, the suite with and without the training stack.
Task 8 - on this machine: bundle (twice), rehearse smoke, plan smoke signed out.
Task 9 - documents, spec section 11 (the code part), roadmap, execution plan, summary.
```

### Pseudocode

```python
# data/bundle.py
def export_bundle(folder, splits=("fit", "dev"), cfg=None, data_cfg=None):
    refuse any split other than fit and dev
    rows = [asdict(example) for name in splits for example in load_split(name, ...)]
    refuse if any row's id is a test id of the manifest
    write examples.jsonl (one JSON object per line, key order kept), then bundle.json:
      dataset repo and commit, manifest content hash, counts per split, sha256, bytes

# split_loader.load_split
folder = get_str("QLORA_DATA_BUNDLE", "")
if folder: return bundle_split(name, Path(folder), cfg, manifest, data_cfg)   # "test" refused

# scripts/job_bootstrap.sh (runs in the container, from the unpacked archive)
link artifacts/ and ledger/ into /opt/ml/checkpoints; pip install the two requirements
files; python -m scripts.job_run fetch-model; python -m scripts.job_run "$@";
copy job_result.json to /opt/ml/output/data; exit with the task's status

# local_qlora/sagemaker.py
def job_definition(cfg, aws, task, arguments, name, workspace, commit, code_uri, bundle_uri):
    return {"TrainingJobName": name, "RoleArn": ...,
            "AlgorithmSpecification": {"TrainingImage": cfg.image, "TrainingInputMode": "File",
                "ContainerEntrypoint": ["bash", "-c"],
                "ContainerArguments": [ENTRY_LINE, "job", task, "--where", "sagemaker", *arguments]},
            "InputDataConfig": [channel("code", code_uri), channel("bundle", bundle_uri)],
            "OutputDataConfig": {"S3OutputPath": ...},
            "ResourceConfig": {...}, "StoppingCondition": {"MaxRuntimeInSeconds": ...},
            "CheckpointConfig": {"S3Uri": workspace_uri, "LocalPath": "/opt/ml/checkpoints"},
            "Environment": {"QLORA_GIT_COMMIT": commit, "QLORA_JOB_NAME": name,
                            "QLORA_DATA_BUNDLE": "/opt/ml/input/data/bundle", ...},
            "Tags": [...]}

class SageMakerJobs:      # every client is passed in
    plan(task) -> the definition, checks J1..J8, the most the job can cost
    launch(task, yes) -> stage code and bundle; start_run("other", ...); create the job
                         (ResourceInUse: adopt); write the job record
    status(), logs(tail), stop(yes)
    pull(checkpoints=False) -> download the job's run folders and adapters; append its
                         ledger and results lines once; add the billable-seconds record
```

### Error handling

| Where | What | Becomes |
|---|---|---|
| Any AWS call | No sign-in, or an expired one | `ConfigError` with the `aws sso login` line; exit 2 |
| `plan`, `launch` | A check fails | The check's FAIL line; nothing created; exit 1 |
| `launch` | No `--yes` | Refused before any call; exit 2 |
| `launch` | The job name exists | Adopted from `DescribeTrainingJob`; no second job |
| `pull` | The job has not ended | A message; no billable record; exit 0 |
| `job_run` | A pin differs, a run fails | `job_result.json` still written; exit 1 or 2 |
| Bundle | A test id, another manifest, a changed file | `SplitError` naming the fault |

## Validation Loop

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest tests/data/test_bundle.py tests/local_qlora tests/scripts -q
venv\Scripts\python -m pytest -q
venv\Scripts\python -m scripts.sagemaker_job bundle
venv\Scripts\python -m scripts.sagemaker_job rehearse smoke
venv\Scripts\python -m scripts.sagemaker_job plan smoke          # signed out: J1 fails
venv\Scripts\python -m scripts.evaluate --predictor gold --split quick_dev --limit 3
```

The suite is also run once with the training stack blocked. **Not run by the agent**:
`launch`, `stop`, and anything that needs a sign-in.

## Final validation Checklist

- [x] Tests and lint green, with and without the stack
- [x] The bundle has no test id; the rehearsal reads the bundle and no snapshot
- [x] Guards: no `frontier` import under `local_qlora/`; boto3 only where allowed
- [x] Nothing was created in AWS; the owner's commands are written down with their cost
- [x] Documents, spec, roadmap, execution plan, implementation summary

---

## Time Estimate

**Total**: 7–9 hours of work, about 10 minutes of GPU, then the owner's six commands.

| Task | Time |
|---|---|
| Shared sign-in module | 20 min |
| Bundle and loader | 50 min |
| Config and prices | 30 min |
| Job runner and bootstrap | 60 min |
| Launcher: source, definition, checks, launch, status, logs, stop | 110 min |
| Pull, staging, job record | 70 min |
| Script | 40 min |
| Tests | 110 min |
| Bundle, rehearsal, plan | 20 min |
| Documents and status files | 50 min |

**Risks**: none of the AWS behavior can be tried before the owner's job (the entry point,
the checkpoint sync, pip in the container, the driver against torch 2.14.1): the smoke job
may need one or two corrected attempts at about $0.20 each. The rehearsal on Windows cannot
exercise the bootstrap script or the links.

## Anti-Patterns to Avoid

- ❌ No upload of the snapshot, and no test row in a bundle
- ❌ No call that needs a sign-in, and no AWS resource created, by the agent
- ❌ No training or evaluation logic written for the cloud: the job calls the same functions
- ❌ No dollar amount in the ledger; no price in code
- ❌ No second job when the first reply was lost: adopt by name
- ❌ No `sagemaker` SDK, no Docker

## Time Taken for PRP Generation

**~12 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 5 min |
| Research existing PRPs and patterns | 1 min |
| Writing PRP document | 4 min |
