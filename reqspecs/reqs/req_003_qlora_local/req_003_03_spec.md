# QLoRA Local

Spec Group      :   req_003_qlora_local  
Feature Name    :   qlora_local  
Version         :   3  
Status          :   ✅ Complete 2026-11-20: code built and rehearsed 2026-11-19 (`prp_003_sagemaker_job_packaging_v3.md`); the second smoke job, `qlora-smoke-6f2dd19d-a1`, ended `Completed` and was pulled (792 billable seconds, $0.31). The first failed before its first step ($0.11). Section 11 has the figures  
Last updated    :   2026-11-19  

## 1. v3.03 — SageMaker job packaging

The training pipeline and the evaluation runner work on this machine. Every number of the
comparison must come from AWS, so that Path A has a bill to set beside Path B's. This
version makes the same code run as a SageMaker training job, and proves the round trip with
one short paid job.

It adds:

- a **data bundle**: the `fit` and `dev` examples in one file, which is all of the dataset a
  job is ever given. No test row leaves this machine;
- a **job runner**, `scripts/job_run.py`: the thing a job executes. It runs a named task
  (`smoke`, `train`, or `evaluate`) by calling `run_training` and `run_evaluation` unchanged;
- a **launcher**, `scripts/sagemaker_job.py`: `plan`, `launch`, `status`, `logs`, `pull`,
  and `stop`;
- a **job workspace** in S3 that a chain of jobs shares, so a later job can resume a
  training run or score a checkpoint that an earlier job made;
- **`config/sagemaker.yaml`**, SageMaker prices in `config/pricing.yaml`, and a **Path A
  budget check** before every launch;
- a committed **job record**, and billable seconds in the ledger;
- the **cloud smoke job**: the 64-example smoke training, then a few evaluated examples
  with and without the adapter, on one GPU instance for about 25 minutes and about $0.60.

It starts no full training run (v3.04) and reads no test row (v5.01). **The smoke job is
the first billed GPU use of the project and is started by the owner.**

**Depends on**: v3.02, v0.05 (both built).

## 2. What exists today (verified 2026-11-19)

### 2.1 What earlier versions provide

| Item | State |
|---|---|
| `run_training(...)`, `scripts.qlora_train` | Dry run, smoke run with checks K1 to K7, full run, resume. Takes `where` and `scope`. Reads the examples through `load_split` |
| `run_evaluation(...)`, `scripts.evaluate` | `qwen3-4b` (arm A0) and `qwen3-4b-tuned --adapter RUN_ID[/checkpoint-N]` (arm A1). `--where sagemaker` is an allowed place. Resumable |
| Run registry and ledger | Everything a run writes is under two folders of the repository root: `artifacts/` (git-ignored) and `ledger/` (committed). `LedgerRecord` has `instance_type`, `instance_count`, `billable_seconds`, `external_id` |
| `record_from_sagemaker_job(description, run, phase)` | Builds a ledger record from `DescribeTrainingJob`: billable seconds per instance, source `aws_api`. A stopped or failed job is recorded too |
| `QLORA_GIT_COMMIT` | `config/tracking.yaml`: "set by the SageMaker launcher so a cloud job, which has no git, can record its commit" |
| `load_split(name)` | Reads the snapshot of all 8,281 rows and checks its three files against pinned checksums. **The snapshot holds the test rows**, so it cannot be given to a job |
| `config/aws.yaml` | Profile `qlora`, us-east-1, bucket `qlora-sagemaker-123456789012` with prefixes `data/`, `sagemaker/`, `adapters/`, role `qlora-sagemaker-execution` (it has `AmazonSageMakerFullAccess`), quota of 1 for `ml.g5.xlarge` and `ml.g6.xlarge` |
| `scripts/aws_check.py` | C1 to C8: identity, bucket, both roles, the GPU quota, the Nova models, the budget alarm |
| Path B's AWS code | `frontier/bedrock.py` has `aws_session` and the sign-in message; `frontier/staging.py` uploads with SHA-256 checksums and reads back. Nothing under `local_qlora/` may import `frontier/` |
| `config/frontier.yaml` | `budget.path_b_cap_usd: 200`, of the $300 overall cap |
| `requirements-train.txt` | Ten exact pins, torch 2.14.1 first. Python 3.13.7 here |
| Installed locally | boto3 1.44.3. **No `sagemaker` SDK and no Docker client in the venv** |
| The committed source a job needs | `qlora_finetuning/`, `scripts/`, `config/`, `splits/`, and the two requirements files: 253 KB as a compressed archive of `HEAD` |
| Measured on this machine (v3.01, v3.02) | Training 0.88 to 0.96 s per sequence; the worst batch 12.35 GiB reserved; a tuned reply about 5 s at batch size 1; batch size 8 is 3.2 times faster untuned, and 21 or 22 of 24 tuned replies are the same as at batch size 1 |

### 2.2 The training-job API, from the installed boto3 (no network)

| Part | Shape |
|---|---|
| `CreateTrainingJob` | Required: `TrainingJobName`, `RoleArn`, `OutputDataConfig.S3OutputPath`. Also `AlgorithmSpecification` (`TrainingImage`, `TrainingInputMode`, `ContainerEntrypoint`, `ContainerArguments`), `InputDataConfig` (channels), `ResourceConfig` (`InstanceType`, `InstanceCount`, `VolumeSizeInGB`), `StoppingCondition.MaxRuntimeInSeconds`, `CheckpointConfig` (`S3Uri`, `LocalPath`), `Environment`, `Tags` |
| Job name | 1 to 63 characters, letters, digits, and hyphens. Unique in the account and region |
| `DescribeTrainingJob` | `TrainingJobStatus` (`InProgress`, `Completed`, `Failed`, `Stopping`, `Stopped`), `SecondaryStatus` (among them `Downloading`, `Training`, `Uploading`, `MaxRuntimeExceeded`), `FailureReason`, `TrainingStartTime`, `TrainingEndTime`, `TrainingTimeInSeconds`, `BillableTimeInSeconds`, `ModelArtifacts` |
| Other operations | `StopTrainingJob`, `ListTrainingJobs`, `DeleteTrainingJob` |

### 2.3 What AWS documents (read 2026-11-19)

| Topic | Finding | Source |
|---|---|---|
| Checkpoints | With a `CheckpointConfig`, SageMaker "automatically syncs" the local folder (default `/opt/ml/checkpoints`) with the S3 location during training. "Existing checkpoints in S3 are written to the container at the start of the job." A file deleted in the container is deleted in S3. Objects added to S3 after the job started are not copied in | SageMaker guide: checkpoints |
| Entry point | `ContainerEntrypoint` and `ContainerArguments` "override the corresponding `ENTRYPOINT` or `CMD`" of the image. Each is 1 to 100 strings of at most 256 characters | API reference: `AlgorithmSpecification` |
| Folders in the container | Input channels at `/opt/ml/input/data/<channel>`; final model files in `/opt/ml/model`; other output in `/opt/ml/output/data`. Each is uploaded "as a single object in compressed tar format" | SageMaker guide: training containers |
| Billing | "You are billed for the time interval between" `TrainingStartTime` and `TrainingEndTime`. `BillableTimeInSeconds` is per instance. Whether pulling the image is inside that interval is **not stated** | API reference: `DescribeTrainingJob` |
| Maximum runtime | Default 1 day, maximum 28 days. When reached, the job gets `SIGTERM`, 120 seconds, then is ended: status `Stopped`, secondary status `MaxRuntimeExceeded` | API reference: `StoppingCondition` |
| Environment | Up to 100 entries, values up to 512 characters. "Do not include any security-sensitive information" | API reference |
| Internet access from a job | **Not stated** on the pages read. The job needs PyPI and the Hugging Face Hub | to be settled by the smoke job |
| Prebuilt containers | The newest PyTorch training container for SageMaker is **2.10.0, Python 3.13, CUDA 13.0, Ubuntu 22.04**, available since 2026-01-21 and patched until 2027-01-21: tag `2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker`. None carries torch 2.14.1 | `aws/deep-learning-containers`, the container definitions |

### 2.4 Prices and the cost of this version

From the AWS price list for SageMaker in us-east-1 published 2026-11-14, as recorded in the
execution plan. Not read again today (section 2.5).

| Instance | GPU | Per hour | Per second |
|---|---|---:|---:|
| `ml.g5.xlarge` | One A10G, 24 GB; 4 vCPU, 16 GB of memory | $1.408 | $0.000391 |
| `ml.g6.xlarge` | One L4, 24 GB; 4 vCPU, 16 GB of memory | $1.127 | $0.000313 |

The smoke job, on `ml.g5.xlarge`:

| Step | Estimate |
|---|---|
| Install the pinned libraries; download the model (8.04 GB) | 8 minutes |
| Smoke training, 12 steps, with the reload of check K7 | 9 minutes |
| 24 quick-dev examples with the smoke adapter, and 24 without | 6 minutes |
| **Total** | **about 25 minutes, about $0.60** |
| Maximum runtime 1 hour: the most the job can cost | $1.41 |

S3 storage for the job's files is under 2 GB: about $0.05 a month until deleted.

### 2.5 What could not be checked

The AWS sign-in had expired, so nothing was asked of the live account. These are settled by
the build or by the smoke job, not by this spec:

1. that the container tag exists in us-east-1 for this account, and its digest;
2. today's price of the two instance types;
3. that a job can reach PyPI and the Hugging Face Hub;
4. that torch 2.14.1 from PyPI runs on the container's GPU driver;
5. how fast an A10G trains and generates, compared with this laptop;
6. whether 16 GB of machine memory is enough to load the model for 4-bit quantization.

Points 3 to 6 are what the smoke job is for.

## 3. Locked decisions

### What a job is given

| # | Decision | Reason |
|---|---|---|
| D1 | Path A runs as **SageMaker training jobs, created with plain boto3**. The `sagemaker` Python SDK is not used. **Decision for the owner in section 8** (Q8) | A job bills by the second only while it runs, stops by itself, and reports billable seconds through the API, which the ledger needs. boto3 is already pinned and tested; the SDK would be a large second client for one call |
| D2 | The image is AWS's own PyTorch training container, by tag in `config/sagemaker.yaml`. The job installs `requirements-train.txt` over it before anything else and then checks every installed version against the pins. **Decision for the owner in section 8** | No AWS image has torch 2.14.1 (v0.03, section 6). Installing the pins in the job needs no Docker, no ECR repository, and no image to maintain; it costs a few minutes per job. The pins stay in one file |
| D3 | The code a job runs is `git archive` of a commit: the package, the scripts, `config/`, `splits/`, and the requirements files. The launcher refuses to launch from uncommitted changes. The commit goes to the job as `QLORA_GIT_COMMIT` | A job's result must trace to a commit, and the archive is that commit and nothing else |
| D4 | **The data a job gets is a bundle of `fit` and `dev` only**: one JSON-lines file of the raw examples and a small record of what it holds (dataset commit, split manifest hash, counts, SHA-256). `export_bundle` writes it and refuses to write a test row | Working agreement 6 and the roadmap's guard: nothing uploaded to S3 contains test rows before v5.01 |
| D5 | In a job, `load_split` reads the bundle, named by the environment variable `QLORA_DATA_BUNDLE`. It checks the bundle's hash and that its ids are exactly the manifest's for the split, in order. Asking for a split the bundle does not hold is an error. Callers do not change | The loader stays the only way to get examples. In a job the test split is not merely locked: it is not there |
| D6 | The model weights are downloaded from the Hugging Face Hub inside the job, at the pinned commit, before the task starts. The task itself loads them with `local_files_only`, as on this machine | Free and a few minutes. Keeping 8 GB in S3 would mean uploading it from the laptop for no gain; if the Hub proves unreliable, an S3 channel is a small change |
| D7 | The job's entry point overrides the container's: unpack the code, install the pins, download the model, then `python -m scripts.job_run <task> ...`. The code arrives as an input channel | `ContainerEntrypoint` works for any image and does not depend on the container's own toolkit |

### The workspace

| # | Decision | Reason |
|---|---|---|
| D8 | Each chain of jobs has a **workspace**, an S3 prefix `sagemaker/workspaces/<name>/`, given to the job as its `CheckpointConfig`. In the container, the repository's `artifacts/` and `ledger/` folders are links into the checkpoint folder | SageMaker then copies everything a run writes (run folders, checkpoints, adapters, ledger lines) to S3 as it is written, with no upload code of ours, and a job that dies loses nothing |
| D9 | A later job launched on the same workspace finds the earlier files in place. So `--resume <run_id>` and `--adapter <run_id>/checkpoint-N` work in a job exactly as they do here | Section 2.3: existing objects are written to the container at the start. This is how v3.04 scores checkpoints and how a stopped training job is continued |
| D10 | The model cache and the installed libraries are outside the workspace | They are 14 GB that must not be copied to S3 |
| D11 | A job starts with an empty `ledger/`. The lines it writes are the new lines, and `pull` appends them to the committed ledger here, skipping any already present | The ledger stays the one committed record, and pulling twice changes nothing |

### Tasks

| # | Decision | Reason |
|---|---|---|
| D12 | `scripts/job_run.py` runs one task and is the same program here and in a job. `smoke`: `run_training(smoke=True)`, then 24 quick-dev examples with that adapter and 24 with none. `train`: `run_training()`, or with `--resume`. `evaluate`: `run_evaluation` for `qwen3-4b` or `qwen3-4b-tuned --adapter`, with `--split`, `--limit`, `--resume` | "Entry points that run unchanged locally and in a job" (roadmap). No training or evaluation logic is written for the cloud |
| D13 | A task first compares the installed libraries with the pins and stops on a difference. It ends by writing `job_result.json`: the task, the commit, the library versions, each run it made with its status, and the GPU's name | The execution plan's check: "library versions in the job log equal the local pins" |
| D14 | Runs made in a job record `where: sagemaker`. `scope` is `dev` for the smoke job | The numbers of the comparison come from `measured` runs, in v3.04 and v5.01 |
| D15 | The smoke task is rehearsed on this machine before any launch: the same archive, unpacked into a temporary folder, with the same bundle, run by `job_run smoke` | Everything except AWS itself is then proven for free |

### Launching and paying

| # | Decision | Reason |
|---|---|---|
| D16 | `sagemaker_job plan <task>` prints the job definition and runs checks J1 to J8 (section 4.5). It creates nothing and may be run by anyone | The dry run the roadmap asks for |
| D17 | `sagemaker_job launch <task> --yes` uploads the code and the bundle, creates one job, and writes the job record. **It is run by the owner.** It refuses if a check fails | Working agreement 10: anything billed is started by the owner |
| D18 | Every job has a maximum runtime from `config/sagemaker.yaml`, per task. The plan prints the most the job can cost: that runtime times the price | A GPU bills whether busy or idle. The limit is what caps a job, not the budget alarm |
| D19 | **A Path A cap**, `budget.path_a_cap_usd`, is checked before every launch: what Path A has cost so far (billable seconds in the ledger times the price) plus the most this job can cost must be within it. **Decision for the owner in section 8** | Path B has a cap of $200 of the $300. Path A has had none |
| D20 | One job at a time: a launch is refused while another job of this project is in progress | The quota is one instance, and two jobs on one workspace would write over each other |
| D21 | A job's name is `qlora-<task>-<commit, 8 characters>-<attempt>`. If a job of that name exists, `launch` adopts it and creates nothing | A lost reply to "create the job" must not start a second one (v4.03 D5) |
| D22 | The job record `reports/local_qlora/sagemaker_jobs.json` is committed. It is written the moment a job's ARN is known and updated by `status` and `pull`: name, ARN, task, commit, image, instance type, workspace, maximum runtime, status, billable seconds, the runs it made | Where later code and the owner find a job. No example text |
| D23 | A launch is itself a run in the registry (kind `other`, place `sagemaker`). When the job has ended, `pull` adds that run's ledger record from `DescribeTrainingJob`: instance type, count, billable seconds, source `aws_api`. No dollar amount | v0.04: the ledger holds quantities. The job's billable time is one figure, whatever runs it held |
| D24 | `pull` downloads the run folders and adapters of the job's runs (checkpoints only on request), appends the job's ledger and results lines, and records the billable seconds. It changes nothing in AWS | The roadmap: "pulls results and billable seconds into the run registry and ledger" |
| D25 | `stop --yes` stops a job and is the owner's. `status` and `logs` read only | A stopped job is billed for the time it ran; its workspace keeps what it wrote |
| D26 | Prices go into `config/pricing.yaml` under `sagemaker`, per instance type and hour, as text, with the price list's date | v4.01 D22 |
| D27 | Only `local_qlora/sagemaker.py` and `local_qlora/staging.py` import boto3 under `local_qlora/`. The sign-in helper moves from `frontier/bedrock.py` to a shared module that both paths import | One place per kind of AWS call; the paths still do not import each other |

### The inference batch size

| # | Decision | Reason |
|---|---|---|
| D28 | **The cloud runs generate in batches of 8**: `inference.batch_size` in `config/local_qlora.yaml` becomes 8, for A0 and A1 alike. **Decision for the owner in section 8** | Open since v3.01. Path A is billed by the second, so the batch size sets its cost per question; the two Qwen arms must use the same value |

## 4. Scope to build

### 4.1 Config

`config/sagemaker.yaml`:

```yaml
version: 1
verified_on: "2026-11-19"
image: "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker"
instance_type: ml.g5.xlarge        # section 8, question 2
instance_count: 1
volume_gb: 60
workspace_prefix: sagemaker/workspaces
code_prefix: sagemaker/code
bundle_prefix: data/bundles
output_prefix: sagemaker/output
name_prefix: qlora
poll_seconds: 30
tasks:
  smoke:    {max_runtime_seconds: 3600,  phase: other, arm: none}
  train:    {max_runtime_seconds: 43200, phase: train, arm: A1}
  evaluate: {max_runtime_seconds: 14400, phase: eval}
smoke:
  evaluate_examples: 24
budget:
  path_a_cap_usd: 100              # section 8, question 5
record_file: reports/local_qlora/sagemaker_jobs.json
tags: {project: qlora-finetuning}
```

`config/pricing.yaml` gains `sagemaker.training`: `ml.g5.xlarge: "1.408"`,
`ml.g6.xlarge: "1.127"` per hour, with `published: "2026-11-14"`.
`config/local_qlora.yaml`: `inference.batch_size: 8` (D28).

### 4.2 The bundle: `qlora_finetuning/data/bundle.py`

| Item | Behavior |
|---|---|
| `export_bundle(folder, splits=("fit", "dev"))` | Writes `examples.jsonl` and `bundle.json`. `test` is refused, and every id written is checked against the manifest's test ids. Deterministic: the same inputs give the same bytes |
| `read_bundle(folder)` | Checks the record, the file's SHA-256, the dataset commit, and the split manifest hash against the current ones |
| `load_split` | With `QLORA_DATA_BUNDLE` set, serves `fit`, `dev`, and `quick_dev` from the bundle (D5). Without it, exactly as today |

### 4.3 The job runner: `qlora_finetuning/local_qlora/job.py`, `scripts/job_run.py`

```powershell
venv\Scripts\python -m scripts.job_run smoke
venv\Scripts\python -m scripts.job_run train [--resume RUN_ID]
venv\Scripts\python -m scripts.job_run evaluate --predictor qwen3-4b-tuned --adapter RUN_ID/checkpoint-300 --split dev
```

`run_task(task, options)` implements D12 to D14 and returns the content of
`job_result.json`. Exit codes as the scripts it wraps: 0, 1, 2. `--where` defaults to
`local`; the job's entry point passes `sagemaker`.

### 4.4 The launcher: `qlora_finetuning/local_qlora/sagemaker.py`, `staging.py`, `scripts/sagemaker_job.py`

| Item | Behavior |
|---|---|
| `load_sagemaker_config()` | Typed values; a missing key is named |
| `build_source(commit)` | The `git archive` of D3, as bytes, with its SHA-256 |
| `job_definition(task, ...)` | The `CreateTrainingJob` request, as a plain mapping. Pure: no client, no clock |
| `entrypoint(task, arguments)` | The container entry point of D7, within the API's limits |
| `SageMakerJobs(sagemaker, s3, logs, ...)` | `plan`, `launch`, `status`, `logs`, `pull`, `stop`. Every client is passed in |
| `staging.py` | Uploads the archive and the bundle with SHA-256 checksums and reads them back; downloads a workspace's files for `pull` |
| Job record | `read_job_record()`, and the writes of D22 |

```powershell
venv\Scripts\python -m scripts.sagemaker_job bundle                    # write the bundle under artifacts/; no AWS
venv\Scripts\python -m scripts.sagemaker_job rehearse smoke            # D15: run the task here from the archive and the bundle; no AWS
venv\Scripts\python -m scripts.sagemaker_job plan smoke
venv\Scripts\python -m scripts.sagemaker_job launch smoke --yes        # the owner
venv\Scripts\python -m scripts.sagemaker_job status
venv\Scripts\python -m scripts.sagemaker_job logs --tail 50
venv\Scripts\python -m scripts.sagemaker_job pull
venv\Scripts\python -m scripts.sagemaker_job stop --yes                # the owner
```

### 4.5 The checks before a launch

| Id | Check | Passes when |
|---|---|---|
| J1 | Signed in | The caller's account is the one in `config/aws.yaml` |
| J2 | A commit | The working tree is clean (changes under `ledger/` aside) and the commit is known |
| J3 | Bundle | It holds `fit` and `dev` only, no test id, and was built from the current split manifest and dataset commit |
| J4 | Role | The execution role exists and trusts SageMaker |
| J5 | Quota | The account's quota for the instance type and training jobs is at least the instance count |
| J6 | Nothing running | No job with this project's name prefix is in progress |
| J7 | Within the cap | Path A's cost so far plus the most this job can cost is within `path_a_cap_usd`. The line prints all three amounts |
| J8 | Priced and pinned | The instance type has a price in `config/pricing.yaml`, and the image is a full URI with a tag |

Signed out, `plan` fails J1 and reports J4 to J6 as skipped. Exit codes of
`sagemaker_job`: 0 done, or a plan whose checks pass; 1 a check failed, or the job failed;
2 a config problem or no sign-in.

### 4.6 Tests

No test needs the network, credentials, a GPU, or a paid call. Clients are stand-ins.

- Bundle: round trip on the invented split; the same bytes twice; `test` refused; a test id
  in the rows refused; a changed file, another manifest, or another dataset commit refused
  on reading; `load_split` from a bundle returns the same examples in the same order as
  from the snapshot, serves `quick_dev`, and refuses `test` even with an unlock record.
- Job runner, with stand-in training and evaluation: each task makes the calls of D12 with
  `where` passed through; a pin that differs stops the task before any run; the result
  file lists the runs; a failed run gives exit 1 and still writes the file.
- Config: each missing key; an unknown task; a maximum runtime of zero; an image without a
  tag; an instance type without a price.
- Job definition: the role, image, instance, volume, maximum runtime, channels, workspace,
  environment (the commit, the bundle path, no credential), and tags; the entry point's
  strings within 256 characters; the job name within 63 characters and the pattern.
- Source: the archive holds the package, scripts, config, and splits, and holds no
  `artifacts/`, `ledger/`, `reqspecs/`, or `.env`.
- Checks J1 to J8: each passes and fails; signed out; the cap arithmetic with `Decimal`.
- Launch: refused without `--yes`, and on each failed check; uploads then creates; writes
  the job record before returning; a job that already exists is adopted, not created
  twice; an expired sign-in gives the sign-in message.
- Status, logs, stop: read from stand-in responses; `stop` needs `--yes`.
- Pull: run folders and adapters land under `artifacts/`; ledger and results lines are
  appended once however often it is run; the billable-seconds record has the instance
  type and source `aws_api`; a job still running adds no such record; nothing with example
  text is written to a committed file.
- Guards: only `sagemaker.py` and `staging.py` import boto3 under `local_qlora/`; nothing
  there imports `frontier`; no code names the test split for upload.

### 4.7 Documentation

- Runbook section 5: export the bundle, rehearse the task, `plan`, `launch`, watch, `pull`,
  `stop`, what each costs, how to delete a workspace, troubleshooting.
- A new tech note, `reqspecs/references/tech/sagemaker.md`: sections 2.2 and 2.3 with their
  links, and what the smoke job measured.
- `gotchas.md`: the container has no torch 2.14.1; a deleted checkpoint file is deleted in
  S3; billing runs from `TrainingStartTime`; entry-point strings are at most 256 characters.
- `run_tracking.md`: a job's runs, the launch run, and the billable-seconds record.
  `split_loading.md`: the bundle.
- `coding.md`: launch only through `SageMakerJobs`; the owner runs `launch` and `stop`; a
  job is given the bundle, never the snapshot; no test row in S3 before v5.01.
- `administration/aws_commands_run.md`: the live commands of section 9.
- Architecture overview; `aws_check` is unchanged.

## 5. Out of scope

- The full training run, scoring its checkpoints, and choosing one (v3.04).
- The test split in a bundle, and any `measured` run (v5.01).
- A custom image, Docker, or an ECR repository (section 8, question 4).
- Managed spot training. It would cut the price by more than half and the resume code
  could carry it, but it needs its own quota and a job that tolerates interruption.
- More than one instance, distributed training, a warm pool, a VPC.
- A SageMaker endpoint for the adapter (req_007).
- The cost model (v5.02). This version multiplies seconds by a price for one check.
- Copying the selected adapter to `adapters/` in the bucket (v3.04).

## 6. What this hands to later versions

| For | What |
|---|---|
| v3.04 | `launch train` for the full run; `launch evaluate` on the same workspace to score each checkpoint; measured seconds per training sequence and per reply on an A10G, for the time and cost estimate the owner sees first; the cap check |
| v3.05 | `launch train` again with one setting changed, on a new workspace |
| v5.01 | A bundle that includes `test` at the unlock; `launch evaluate` with scope `measured` for A0 and A1 |
| v5.02 | Billable seconds per job and instance type in the ledger; prices with their date in `config/pricing.yaml` |
| v6.01 | `pull` for the selected adapter |

## 7. Known limits and follow-ups

- **The image tag can be repointed by AWS.** The job record keeps the image a job ran
  with; the digest is recorded when the account can be asked.
- **Each job spends about 8 minutes on setup**, about $0.19, before it does any work. A
  custom image would save most of it (section 8, question 4).
- **A job needs the internet**, for PyPI and the Hub. If either is down, the job fails
  early and cheaply.
- **Everything in the workspace is copied to S3, and a file deleted in the job is deleted
  there.** Nothing in the pipeline deletes a checkpoint.
- **A job that reaches its maximum runtime is stopped mid-task.** What it wrote is in the
  workspace, and a later job resumes it.
- **Billable time may include pulling the image.** The documentation does not say. The
  smoke job's own figures settle it.
- **The price is from the list of 2026-11-14.** The build reads it again once signed in.
- **Local and cloud runs differ.** Another GPU can turn a near tie the other way; the
  smoke job's losses will be close to this machine's, not equal.
- **The budget alarm does not stop a job**, and the cap check counts only what the ledger
  holds. A job that was launched and never pulled is counted at its maximum.
- **Looking ahead to v3.04**, at this machine's speeds: training about 4 hours; scoring
  ten checkpoints on dev about 7 hours at batch size 1 or about 2 at batch size 8. On
  `ml.g5.xlarge` that is roughly $15 against $9. The smoke job replaces these guesses.

## 8. Decisions for the owner

**Taken as recommended on 2026-11-19**: the owner started the PRP and its execution
without changing any of the five. Q8 and Q9 are closed: SageMaker training jobs created
with boto3, on `ml.g5.xlarge`. The table is kept as the record of what was asked.

| # | Question | Recommendation |
|---|---|---|
| 1 | **How does Path A run in AWS (Q8)?** SageMaker training jobs, a notebook instance, or an EC2 instance | Training jobs, created with boto3 (D1). They stop by themselves and report billable seconds; an instance you start by hand bills until you remember to stop it |
| 2 | **Which GPU instance (Q9)?** `ml.g5.xlarge` (A10G, $1.408 an hour) or `ml.g6.xlarge` (L4, $1.127) | `ml.g5.xlarge` for the smoke job and the full run. The L4 is 20% cheaper per hour and slower by more than that on this kind of work, so it should cost more per run; the quota allows either. If you want it measured, a second smoke job on `ml.g6.xlarge` is another $0.50 |
| 3 | **Inference batch size in the cloud: 8 or 1 (D28)?** Open since v3.01 | 8, for both Qwen arms. Tuned replies agreed with batch size 1 in 21 or 22 of 24, the differences are not a bias toward either answer, and Path A's cost per 1,000 questions is about a third at 8. That is also how such a model would be served. The report (v5.03) should say that replies depend on the batch. The alternative keeps each reply independent and makes Path A look about three times dearer than it need be |
| 4 | **AWS's container with the pins installed at job start (D2), or our own image?** | AWS's container. It needs nothing new on this machine or in the account, and the pins stay in one file. Our own image saves about 8 minutes a job, worth it only if there turn out to be many jobs; it can be added later without changing the launcher |
| 5 | **A cap of $100 for Path A (D19)**, the rest of the $300 after Path B's $200? And a maximum runtime of 1 hour for the smoke job ($1.41 at most)? | Yes to both. The whole of Path A is estimated at $15 to $30, so $100 stops a runaway without getting in the way |

**Who runs what**: the owner signs in and runs `launch` (and `stop` if needed). Claude
builds, tests, rehearses the task on this machine (free, about ten minutes of GPU), and may
run `plan`, `status`, `logs`, and `pull`, which read from AWS and bill nothing.

## 9. Acceptance

Code, by Claude:

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
  and without the training stack blocked, with no network and no credentials.
- A test shows `load_split` gives the same examples from a bundle as from the snapshot,
  and that a bundle cannot hold a test row.
- `scripts.evaluate` and `scripts.qlora_train` behave as before when no bundle is named.

On this machine, by Claude (free):

```powershell
venv\Scripts\python -m scripts.sagemaker_job bundle                    # fit and dev, no test id
venv\Scripts\python -m scripts.sagemaker_job rehearse smoke            # the archive and the bundle, in a temporary folder
venv\Scripts\python -m scripts.sagemaker_job plan smoke                # signed out: J1 fails, nothing created
```

- The bundle holds 4,884 and 499 examples and no test id; exporting it twice gives the
  same SHA-256.
- The rehearsal passes K1 to K7 and scores its 48 examples, reading the bundle and never
  the snapshot.

Live, by the owner (🟡), each command logged in `administration/aws_commands_run.md`:

```powershell
aws sso login --profile qlora
venv\Scripts\python -m scripts.aws_check                               # 8 of 8, free
venv\Scripts\python -m scripts.sagemaker_job plan smoke                # J1 to J8 pass; prints "at most $1.41"
venv\Scripts\python -m scripts.sagemaker_job launch smoke --yes        # about 25 minutes, about $0.60
venv\Scripts\python -m scripts.sagemaker_job status
venv\Scripts\python -m scripts.sagemaker_job pull
```

- The job ends `Completed`. Its log shows the installed versions equal to the pins.
- In the job, the smoke run passes K1 to K7, with 33,030,144 trainable parameters and a
  worst batch 1,862 tokens wide, as here.
- `pull` brings the three runs into the registry with `where: sagemaker`, and the ledger
  has one record with the instance type and the billable seconds, source `aws_api`.
- The S3 objects under the bundle prefix hold `fit` and `dev` ids only.
- The job record is committed. The budget alarm's page shows non-zero SageMaker spend
  within a day (the open row of `gotchas.md`).
- Section 11, "As built", is added: what the job cost and how long each step took,
  seconds per training sequence and per reply on the A10G, peak GPU memory, and which of
  section 2.5 turned out how. The roadmap and the execution plan are updated, and Q8 and
  Q9 are closed.

## 10. Next step

The version is complete (2026-11-20). Next is v3.04, the owner's: `sagemaker_job plan
train --scope measured` now has the smoke job's measured times for its estimate (check J9).

## 11. As built (2026-11-19)

Built by `prp_003_sagemaker_job_packaging_v3.md`. **Nothing was created in AWS and no call
reached it**: the sign-in was expired throughout. The smoke job is the owner's, and this
section is completed with its figures after it has run.

What was run:

- `venv\Scripts\python -m pytest -q`: 2,266 passed (136 new). With the training stack
  blocked: 2,260 passed, 6 skipped. `pytest -m gpu -q`: 8 passed.
  `venv\Scripts\python -m ruff check .`: clean. The Path B tests pass unchanged after the
  sign-in helpers moved to `qlora_finetuning/aws_common.py`.
- `scripts.sagemaker_job bundle`, twice: `fit` 4,884 and `dev` 499, 5,383 examples,
  29,621,381 bytes, no test id, SHA-256 `d1ad8eed5445...` both times.
- `scripts.sagemaker_job rehearse smoke`: the smoke task from the code and the bundle, in a
  temporary folder with no snapshot. About 6 minutes of GPU.
- `scripts.sagemaker_job plan smoke`, signed out: J1 failed with the sign-in message, J4 to
  J6 were skipped, nothing was created, exit 1.
- `scripts.evaluate --predictor gold` and `scripts.qlora_train --dry-run`: as before.

### The rehearsal

| Run | Result |
|---|---|
| Smoke training (`20261120T043401Z-smoke-qwen3-4b-smoke-6c98`) | 12 steps; K1 to K7 pass |
| 24 quick-dev examples with the smoke adapter, batch size 8 | 10 right on execution; 24 in the format |
| The same 24 without an adapter, batch size 8 | 0 right on execution; 22 in the format |
| Snapshot in the folder | None: every example came from the bundle |
| Ledger and run registry here | Untouched: a rehearsal's runs stay in its temporary folder |

The rehearsal used the working tree's files, because the code was not committed yet. From
a clean tree it uses `git archive` of the commit, as a job does.

What the rehearsal cannot show, because it runs on Windows without the container: the
bootstrap script, the links into the checkpoint folder, pip over the container's
libraries, the model download in the job, and everything of section 2.5.

### The plan a launch will make

`qlora-smoke-<commit>-a1` on one `ml.g5.xlarge` with a 60 GB volume, the PyTorch 2.10
container, at most 60 minutes: **at most $1.41**. Path A has cost $0.00 so far, of the $100
cap. The job's name follows the commit, so it is fixed once this version is committed.

### Differences from sections 3 and 4

- **The entry point is a short `bash -c` line and a committed bootstrap script**
  (`scripts/job_bootstrap.sh`, inside the archive), because an entry-point string may hold
  at most 256 characters. D7's steps are in the script.
- **The archive is made with `core.autocrlf` switched off.** On this machine git would
  otherwise write Windows line endings into it, and the shell script would not run.
- **`rehearse` copies the working tree when there are uncommitted changes**, and says so.
- **Writing a bundle always reads the snapshot** (`load_split_from_snapshot`), also when
  `QLORA_DATA_BUNDLE` is set, so a bundle is never made from another bundle.
- **J7 counts a job that was launched and not yet pulled at its maximum cost.**
- **`pull` without a result file** (a job that died before its task ended) takes every run
  folder the workspace holds, and still records the billable seconds.
- The job runner is `qlora_finetuning/local_qlora/job.py` with `scripts/job_run.py` over
  it, which also has `fetch-model` (the model download of D6). The loader and the job
  record have modules of their own (`sagemaker_config.py`, `job_record.py`).
- `pin_differences` accepts a local build suffix (`2.14.1+cu126`), as `env_check` does.
- The sign-in helpers are in `qlora_finetuning/aws_common.py`; `frontier/bedrock.py`
  imports them from there.
- `inference.batch_size` is 8 (D28). The local A0 baseline of v3.01 was run at 1.

### The first smoke job failed (2026-11-20)

`qlora-smoke-8d397d1b-a1`, launched by the owner from commit `8d397d1`, ended `Failed`
after 281 billable seconds (**$0.11**). It was pulled; Path A has cost $0.11 of its $100.

| Step in the job | Result |
|---|---|
| Waiting for the instance | About 8 minutes, not billed |
| The bootstrap script, the links, pip over the container's libraries | Worked. Every pinned library at its pinned version |
| Internet access (PyPI, the Hugging Face Hub) | Worked, with no token |
| GPU | NVIDIA A10G, 23,028 MiB, driver 595.112.07 |
| Model download | About 40 seconds |
| The smoke task | **Failed before step 1**: `the training stack is not installed (None is missing)` |
| Workspace sync and `pull` | Worked: the failed run, the result file, and the billable seconds came home |

**The cause, as far as the log shows.** A library was installed and failed while loading;
the message hid the reason, and no traceback was logged. The likely cause is the
container's own `torchvision 0.25.0+cu130` and `torchaudio 2.10.0+cu130`, built for torch
2.10.0 and left in place when pip installed the pinned torch 2.14.1 (pip warned of both).
transformers loads torchvision when it finds it, and reports a failure there as a missing
module without a name. This is a reading of the log, **not confirmed**: the second job
confirms or refutes it.

**What was changed.**

- `scripts/job_bootstrap.sh` removes `torchvision` and `torchaudio` after the install
  (nothing in the project uses them), then loads the training stack before the model is
  downloaded. A library that cannot be loaded now stops the job there, with the whole
  error in the log.
- `local_qlora/backend.py`: when a library is installed and cannot be loaded, the message
  says so and gives the error and its cause, not "None is missing". One new test.
- 2,441 tests pass. Lint is clean for the changed files.

### The second smoke job completed (2026-11-20)

`qlora-smoke-6f2dd19d-a1`, launched by the owner from commit `6f2dd19`, ended `Completed`
and was pulled. **792 billable seconds, $0.31** at the list price, against an estimate of
about 25 minutes and $0.60. Path A has cost $0.42 of its $100 (the two smoke jobs).

Section 9's points:

| Point | Result |
|---|---|
| The job ends `Completed`; installed versions equal the pins | Yes. The ten pinned libraries are at their pins in the log and in `job_result.json`; torch is the build `2.14.1+cu130` |
| The smoke run passes K1 to K7, with 33,030,144 trainable parameters and a worst batch 1,862 tokens wide | Yes: `checks failed: none`, 12 steps, both figures as on this machine |
| `pull` brings the three runs home with `where: sagemaker`; one ledger record with the instance type and billable seconds, source `aws_api` | Yes: 39 files, 6 run lines, 3 ledger lines, and the record for `ml.g5.xlarge`, 792 seconds |
| The S3 objects under the bundle prefix hold `fit` and `dev` ids only | Yes. `data/bundles/d34fb22e37b7-d1ad8eed5445/examples.jsonl` has the size and the SHA-256 of the local bundle, which check J3 found free of test ids |
| The job record is committed | With this version |
| The budget alarm's page shows SageMaker spend within a day | **Not checked yet**: the spend is an hour old. It stays the open row of `gotchas.md` |

What the job did, and how long each step took:

| Step | Time |
|---|---|
| Waiting for the instance (not billed) | About 6 minutes |
| From the start of billing to the libraries installed and loaded | About 4 minutes |
| Model download (8.04 GB) | About 30 seconds |
| Smoke training: load, worst-batch probe, 12 steps with 3 dev-loss passes | 4 minutes 20 seconds (260 s; the 12 steps alone 214 s) |
| The reload of check K7: 24 replies at batch size 8 and again singly | About 2 minutes 40 seconds |
| 24 quick-dev examples with the smoke adapter | 46 seconds |
| The same 24 without an adapter | 23 seconds |
| **Billed in all** | **13 minutes 12 seconds** |

On the A10G:

| Measure | A10G | This machine |
|---|---|---|
| Seconds per training sequence | 1.115 | 0.79 in the trial; 0.88 to 0.96 in later runs |
| Seconds per reply, tuned, batch size 8 | 1.93 | |
| Seconds per reply, tuned, batch size 1 | 4.19 | about 5 |
| Seconds per reply, untuned, batch size 8 | 0.97 | |
| Peak GPU memory, worst training batch | 11.00 GiB allocated, 12.35 reserved | the same |
| Peak GPU memory, training steps | 5.99 GiB allocated | |
| Peak GPU memory, generating | 5.78 GiB allocated | |
| Loss, step 1 to step 12 | 2.27 to 0.29; dev loss 0.33, 0.24, 0.24 | 2.27 to 0.28 |
| Smoke adapter on 24 quick-dev examples | 10 right on execution, 24 in the format | the same |
| Untuned on the same 24 | 0 right, 22 in the format | the same |
| Tuned replies identical at batch size 8 and 1 | 17 of 24 | 22 of 24 in the trial |

The A10G trained more slowly per sequence than this machine's GPU. v3.04's estimate reads
these figures.

Section 2.5, point by point:

1. **The container tag** exists for the account: both jobs ran on it. Its digest was not read.
2. **The prices** were not read again; the costs here are billable seconds at the list of
   2026-11-14.
3. **A job reaches PyPI and the Hugging Face Hub**, with no token.
4. **torch 2.14.1 from PyPI runs on the container's GPU driver** (595.112.07), once the
   container's `torchvision` and `torchaudio` are removed. That removal was the one change
   between the job that failed and the job that ran, which confirms the reading above.
5. **The A10G's speeds** are in the table.
6. **16 GB of machine memory is enough** to load the model in 4-bit.

Not settled: whether the image pull is inside the billed interval. Billing began about four
minutes before the model download, and the log does not say how those minutes divide.
