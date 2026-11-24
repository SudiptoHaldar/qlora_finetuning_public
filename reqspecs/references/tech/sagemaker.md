# SageMaker training jobs

**Last verified**: 2026-11-19 against boto3 1.44.3 (the service definition it ships) and the
AWS documentation, and on 2026-11-20 against the live account by two smoke jobs: the first
failed, the second (`qlora-smoke-6f2dd19d-a1`) completed. `create_training_job`,
`describe_training_job`, the log calls, the checkpoint sync, and `pull` worked as written
here. `stop_training_job` and the adoption of an existing job name have not been used live.
**Used by**: `qlora_finetuning/local_qlora/sagemaker.py`, `staging.py`, `scripts/sagemaker_job.py`.

The project creates training jobs with plain boto3. The `sagemaker` Python SDK and Docker
are not used.

## The API

```python
sagemaker.create_training_job(
    TrainingJobName=name,                      # 1 to 63 of [a-zA-Z0-9-]; unique in account and region
    RoleArn=role,
    AlgorithmSpecification={"TrainingImage": image, "TrainingInputMode": "File",
                            "ContainerEntrypoint": ["bash", "-c"],
                            "ContainerArguments": [line, "job", task, ...]},
    InputDataConfig=[{"ChannelName": "code", "InputMode": "File",
                      "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": uri,
                                                      "S3DataDistributionType": "FullyReplicated"}}}],
    OutputDataConfig={"S3OutputPath": uri},
    ResourceConfig={"InstanceType": "ml.g5.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 60},
    StoppingCondition={"MaxRuntimeInSeconds": 3600},
    CheckpointConfig={"S3Uri": workspace_uri, "LocalPath": "/opt/ml/checkpoints"},
    Environment={...}, Tags=[...],
)
sagemaker.describe_training_job(TrainingJobName=name)
    # TrainingJobArn, TrainingJobStatus (InProgress | Completed | Failed | Stopping | Stopped),
    # SecondaryStatus (Starting, Downloading, Training, Uploading, MaxRuntimeExceeded, ...),
    # FailureReason, TrainingStartTime, TrainingEndTime, TrainingTimeInSeconds,
    # BillableTimeInSeconds, ResourceConfig
sagemaker.list_training_jobs(NameContains=prefix, StatusEquals="InProgress")
sagemaker.stop_training_job(TrainingJobName=name)
logs.describe_log_streams(logGroupName="/aws/sagemaker/TrainingJobs", logStreamNamePrefix=name + "/")
logs.get_log_events(logGroupName=..., logStreamName=..., limit=50, startFromHead=False)
```

- A job name that exists raises `ClientError` `ResourceInUse`: describe it and adopt it.
- Describing a job that does not exist raises `ValidationException`, not a not-found code.

## What AWS documents

| Topic | Finding |
|---|---|
| Entry point | `ContainerEntrypoint` and `ContainerArguments` override the image's `ENTRYPOINT` and `CMD`. Each is 1 to 100 strings of **at most 256 characters** |
| Folders | Input channels at `/opt/ml/input/data/<channel>`; model files in `/opt/ml/model`; other output in `/opt/ml/output/data`; each uploaded as one compressed tar object when the job ends |
| Checkpoints | The local folder (default `/opt/ml/checkpoints`) is kept in step with `CheckpointConfig.S3Uri` during the job. **Objects already there are written to the container at the start.** A file deleted in the container is deleted in S3. Objects added to S3 after the start are not copied in |
| Billing | From `TrainingStartTime` to `TrainingEndTime`. `BillableTimeInSeconds` is per instance: multiply by the instance count |
| Maximum runtime | Default 1 day, at most 28. When reached: `SIGTERM`, 120 seconds, then status `Stopped` with secondary status `MaxRuntimeExceeded` |
| Environment | Up to 100 entries, values up to 512 characters. No credentials in it |
| Containers | AWS's newest PyTorch training container for SageMaker (2026-11-19): `2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker` in `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training`. None has torch 2.14.1 |

Sources: the SageMaker developer guide (checkpoints; how SageMaker runs a training image;
training output), the API reference (`CreateTrainingJob`, `DescribeTrainingJob`,
`AlgorithmSpecification`, `StoppingCondition`), and the container definitions under
`docs/src/data/pytorch-training/` in `github.com/aws/deep-learning-containers`.

## How the project uses it

- **The code** is `git -c core.autocrlf=false archive --format=tar.gz <commit> -- <paths>`,
  uploaded to `sagemaker/code/<commit>/source.tar.gz` and given to the job as the channel
  `code`. With `core.autocrlf` on, git would write Windows line endings into the archive and
  the shell script in it would not run.
- **The entry point** is one short `bash -c` line that unpacks the archive and starts
  `scripts/job_bootstrap.sh`, because of the 256-character limit.
- **The data** is the bundle (`fit` and `dev`), the channel `bundle`; `QLORA_DATA_BUNDLE`
  points `load_split` at it.
- **The workspace**: the bootstrap links the repository's `artifacts/` and `ledger/` into
  `/opt/ml/checkpoints`, so everything a run writes is copied to
  `sagemaker/workspaces/<name>/` as it is written, and a later job on the same workspace
  starts with it in place. The model cache and the installed libraries stay outside.
- **The libraries**: `pip install -r requirements.txt -r requirements-train.txt` over the
  container's; then `torchvision` and `torchaudio` are removed, because the container's are
  built for its own torch and break the loading of transformers beside the pinned one; then
  the training stack is loaded once, so that a broken library stops the job before the
  model download; then the task compares every installed version with the pins.
- **The model**: `python -m scripts.job_run fetch-model` downloads it from the Hub in the job.

## Settled by the smoke job (2026-11-20)

| Question | Answer |
|---|---|
| Does the image tag exist for the account, and what is its digest? | It exists: both jobs ran on it. The digest was not read |
| Can a job reach PyPI and the Hugging Face Hub? | Yes, with no token |
| Does torch 2.14.1 from PyPI run on the container's GPU driver? | Yes: `2.14.1+cu130` on an A10G with driver 595.112.07, once `torchvision` and `torchaudio` are removed |
| Is image pull time inside the billed interval? | Not settled. Billing began about 4 minutes before the model download; the log does not divide them |
| Is 16 GB of machine memory enough to load the model for 4-bit quantization? | Yes |
| Seconds per training sequence and per reply on an A10G | 1.115 per training sequence. Per reply at batch size 8: 1.93 tuned, 0.97 untuned; 4.19 tuned at batch size 1 |
| How long the setup takes (pip, model download) | About 4 minutes to the libraries loaded, then about 30 seconds for the model |
| How long a job waits for its instance | 6 and 8 minutes in the two jobs, not billed |
| Peak GPU memory | 11.00 GiB allocated (12.35 reserved) for the worst training batch; 5.78 GiB generating |

The smoke job was billed 792 seconds ($0.31) against an estimate of 25 minutes.

## From req_005 v5.01: the two test jobs

- **The test bundle.** `sagemaker_job bundle --split test` writes the test examples alone to
  `artifacts/bundle_test/` (`test_bundle_dir` in `config/sagemaker.yaml`), only after the
  unlock. An `evaluate` job with `--split test` is given that bundle in the `bundle`
  channel, under its own key prefix in `data/bundles/`; every other job is given `fit` and
  `dev`. The launch uploads it: the one upload of test rows to S3.
- **Check J11** asks the gate of `comparison/gate.py` at plan time, with what the job cannot
  see: the local results file and the job record. It fails while the split is locked, for a
  task other than `evaluate`, without `--scope measured`, with `--limit`, when the arm's
  expected settings (`expected_settings`) differ from the freeze record, when the arm has
  its test result, and when an earlier test job of the arm has not been pulled.
- **Inside the job** the gate runs again with the real predictor. It reads
  `splits/config_freeze.json` and `splits/test_unlock.json` from the job's code, which is
  why those two files are under `splits/` and must be committed before the launch. In a job
  the ledger and the run registry are the workspace's, so "this arm already has a result"
  is only fully known at plan time.
- **Two jobs, two commits.** A job's name holds the commit, so two `evaluate` jobs from one
  commit would be one job. The job record written by the first launch is committed before
  the second, which gives a new commit; otherwise `--attempt 2`.
- **`rehearse` refuses `--split test`**: a Qwen model answers test questions inside a job
  only, once.
- **`committed_path_a`** (what Path A has cost and may yet cost) now lives in `estimate.py`,
  without botocore, so the freeze can check the cap; `SageMakerJobs.committed` calls it.

## From req_003 v3.04: estimates, scoring, and keeping an adapter

- **The estimate** (`local_qlora/estimate.py`). A pulled job's `job_result.json` holds the
  seconds per training sequence and the seconds and prompts of every evaluation run. With
  the job's billable seconds, the rest is overhead (pip, the model download, model loads).
  `measured_rates` takes the newest figure of each kind from pulled jobs on the configured
  instance type; `estimate_job` multiplies. No figure, no estimate, and check J9 fails.
  A scoring job loads a model eleven times and the smoke job twice, so the overhead of the
  first scoring estimate is on the low side by a few minutes; the margin covers it.
- **`--max-runtime`** sets `StoppingCondition.MaxRuntimeInSeconds` for one job. The cap
  check (J7) and the job record use it. The task's value in `config/sagemaker.yaml` is the
  default.
- **A job on another job's workspace** starts with that job's `artifacts/` and `ledger/`
  in place (the checkpoint folder is written back at the start). That is how `score` finds
  the checkpoints, and why `score`, a resumed `train`, and a tuned `evaluate` are refused
  without `--workspace`.
- **`keep`** writes three objects under `adapters/<run>/checkpoint-N/`: the adapter's two
  files and the selection record, each with a checksum. It is the only launcher command
  besides `launch` that writes to S3, and it is the owner's.
- **`pull --checkpoint N`** lists a workspace and downloads one checkpoint's adapter files
  and the run's `training.json`. Read-only.

Settled by the first training and scoring jobs (2026-11-20 and 2026-11-21):

| Job | Estimate | Billed |
|---|---|---|
| Training, 918 steps | 4.7 hours, $6.59 | 15,037 seconds (4.2 hours), $5.88 |
| Scoring, eleven models on 499 examples | 2.9 hours, $4.09 | 4,965 seconds (1.4 hours), $1.94 |

The training estimate was close. The scoring estimate was more than twice too high: the
smoke job's 1.91 s per tuned reply comes from 24 examples, and over 499 a tuned reply took
about 0.8 s. A later scoring plan reads the scoring job's own times. A job on another job's
workspace found its checkpoints in place, as described above. The peak GPU memory of each
run is in its `measurements.json`.
