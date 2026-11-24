# Project Runbook

> **Written for the project's owner.** This is the project's own runbook, published as it is. The steps that are billed or that change AWS need an AWS account of your own; the account number, the bucket, and the roles shown here are placeholders. See [PUBLIC_SNAPSHOT.md](../PUBLIC_SNAPSHOT.md).

Operator steps for `qlora_finetuning`. Commands are PowerShell, run from the repo root.

## 1. AWS (req_000 v0.05)

Both training paths run in AWS account 123456789012 (shared with velaHita staging), region
us-east-1. The resource names live in `config/aws.yaml`. Every AWS command run for the project
is logged in `administration/aws_commands_run.md`.

### 1.1 One-time setup

```powershell
py -3.13 -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
```

Every Python command in this runbook is written as `venv\Scripts\python -m <tool>`, so the venv
never needs activating.

The first `venv/` on this machine was copied from another project, and its `Scripts\` folder
still holds that project's launchers (for example `alembic.exe`, `uvicorn.exe`). They do
nothing. To clear them, delete `venv/` and run the two commands above again.

The `qlora` AWS CLI profile must exist in `~/.aws/config` (see section 1 of
`administration/aws_commands_run.md` for its contents).

### 1.2 Sign in

```powershell
aws sso login --profile qlora
```

Opens the browser. Needed again whenever the session expires; `aws_check` tells you when.

### 1.3 Verify the AWS foundation

```powershell
venv\Scripts\python -m scripts.aws_check
```

Run it from the repository root, as a module (`-m`). The script imports the
`qlora_finetuning` package, which a run by file path cannot find.

The script only reads. It creates, changes, and deletes nothing, and no call it makes is billed.
It prints one line per check:

| Check | Passes when |
|---|---|
| C1 Caller identity | The profile signs in to the account in `config/aws.yaml` |
| C2 Bucket | The bucket exists in us-east-1, all four public-access blocks are on, default encryption is set |
| C3 SageMaker role | The role exists and trusts `sagemaker.amazonaws.com` |
| C4 Bedrock role | The role exists, trusts `bedrock.amazonaws.com`, and its inline policy grants read on `data/*` and read and write on `bedrock/*` |
| C5 GPU quota | The training-job quota is at least 1 for `ml.g5.xlarge` and `ml.g6.xlarge` |
| C6 Nova models | Nova Pro and Nova Lite are listed as fine-tunable |
| C7 Budget | The budget exists with the $300 cap and at least one alert has a subscriber |
| C8 Nova inference | Each model of `config/frontier.yaml` (`amazon.nova-pro-v1:0`, `amazon.nova-lite-v1:0`) is Active and can be called on demand. The line also prints an end-of-life date if AWS has set one (req_004 v4.01) |

| Exit code | Meaning |
|---|---|
| 0 | All eight checks passed (C1 to C7 are milestone M1b) |
| 1 | At least one check failed; the FAIL line gives the reason |
| 2 | Sign-in or config problem; nothing was checked |

Options: `--config <path>` to use another config file, `--profile <name>` to override the profile.

The same checks as a test, for a valid sign-in: `venv\Scripts\python -m pytest -m network -q`.
Unit tests and lint are in section 2.

### 1.4 What each resource is for

| Resource | Name | Purpose |
|---|---|---|
| CLI profile | `qlora` | Signs in through the `velahita` SSO session with the `AdministratorAccess` role |
| S3 bucket | `qlora-sagemaker-123456789012` | The project's only bucket. Prefixes: `data/` training and validation files, `sagemaker/` SageMaker job outputs, `bedrock/` Bedrock customization outputs, `adapters/` selected LoRA adapters |
| SageMaker execution role | `qlora-sagemaker-execution` | The identity SageMaker training jobs run as. Carries `AmazonSageMakerFullAccess`, which reaches S3 only in buckets with "sagemaker" in the name |
| Bedrock customization role | `qlora-bedrock-customization` | The identity Bedrock fine-tuning jobs run as. Inline policy `qlora-bedrock-s3`: list the bucket, read `data/*`, read and write `bedrock/*` |
| GPU training quota | `ml.g5` and `ml.g6`, 1 instance each | Lets one training job run at a time. An account property; nothing was requested |
| Nova fine-tuning access | Nova Pro, Nova Lite | An account property; nothing was requested |
| Budget alarm | `qlora-finetuning` | Emails at 80% and 100% of $300 actual SageMaker plus Bedrock spend. It does not stop anything. Yearly period: the counter resets on 2027-01-01 |

Policy documents are kept for reference in `administration/`: `trust.json` (SageMaker trust),
`bedrock-trust.json`, `bedrock-s3-policy.json`.

### 1.5 Teardown

Run by the owner, in this order. Each command deletes something; log each in
`administration/aws_commands_run.md`. Check first that no SageMaker or Bedrock job is running
and that any custom Nova model has been deleted (those belong to later versions).

```powershell
# 1. Bedrock customization role: remove its inline policy, then the role
aws iam delete-role-policy --role-name qlora-bedrock-customization --policy-name qlora-bedrock-s3 --profile qlora
aws iam delete-role --role-name qlora-bedrock-customization --profile qlora

# 2. SageMaker execution role: detach the managed policy, then the role
aws iam detach-role-policy --role-name qlora-sagemaker-execution `
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess --profile qlora
aws iam delete-role --role-name qlora-sagemaker-execution --profile qlora

# 3. S3 bucket: delete every object, then the bucket (copy out anything worth keeping first)
aws s3 rm s3://qlora-sagemaker-123456789012 --recursive --profile qlora
aws s3 rb s3://qlora-sagemaker-123456789012 --profile qlora

# 4. Budget alarm (its alerts and subscribers go with it)
aws budgets delete-budget --account-id 123456789012 --budget-name qlora-finetuning --profile qlora
```

Not removed by these steps:

- The `qlora` profile: delete its `[profile qlora]` block from `~/.aws/config` by hand.
- GPU quota and Nova fine-tuning access: account properties, nothing to delete.
- The account and the `velahita` SSO session: shared with velaHita staging, left as they are.

After teardown, `aws_check` should report FAIL for C2, C3, C4, and C7.

### 1.6 Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `SIGN-IN REQUIRED` | The SSO session expired or the profile is missing. Run `aws sso login --profile qlora` |
| `CONFIG ERROR` | `config/aws.yaml` is missing or lacks a key; the message names it |
| C1 fails with a different account | The profile points at another account. Check `sso_account_id` in `~/.aws/config` |
| C5 says a quota was not found | AWS renamed the quota. Compare `quota.usage` in the config with `aws service-quotas list-service-quotas --service-code sagemaker` |
| C7 fails after the cap is changed | Update `budget.cap_usd` in `config/aws.yaml` to match the new cap |
| `No module named pip` or `boto3` | The venv is incomplete; repeat 1.1 |
| "Run this script as a module from the repository root" | The script was started by file path. Use `venv\Scripts\python -m scripts.aws_check` |

## 2. Tests, lint, and CI (req_000 v0.02)

### 2.1 Run locally

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q
```

These are the same two commands CI runs. The lint rules are in `ruff.toml` (line length 100;
rule sets E, W, F, I, UP, B). `venv\Scripts\python -m ruff format <file>` wraps over-long lines.

Three kinds of test are skipped unless asked for by name:

| Marker | Needs | Run with |
|---|---|---|
| `network` | Network access and a valid AWS sign-in | `venv\Scripts\python -m pytest -m network -q` |
| `gpu` | A CUDA GPU | `venv\Scripts\python -m pytest -m gpu -q` |
| `paid` | A billed AWS call. Run only by the owner | `venv\Scripts\python -m pytest -m paid -q` |

### 2.2 Package, configuration, and logging

- **Version**: `venv\Scripts\python -c "import qlora_finetuning; print(qlora_finetuning.__version__)"`
  prints the content of `release_number.md`. `/release-tag` is the only thing that changes it.
- **Config files**: `config/<name>.yaml`, each with an integer `version:` key. Code reads them
  with `load_config("<name>")` from `qlora_finetuning/settings.py`.
- **Environment variables**: optional. Copy `.env.example` to `.env` to set `LOG_LEVEL`
  (DEBUG, INFO, WARNING, ERROR). A variable set in the shell wins over `.env`.
- **Log lines** go to standard error; a script's results go to standard output.
- **Scripts** are run as modules from the repository root: `venv\Scripts\python -m scripts.<name>`.

### 2.3 CI

`.github/workflows/ci.yml` runs on every pull request to `main` and on every push to `main`:
install `requirements.txt`, `ruff check .`, `pytest -q`, on Ubuntu with Python 3.13. It uses no
secret and no AWS credential. A push to a branch with no open pull request does not start it.

CI does not decide when a version is complete (since 2026-11-15). A version is complete when
the tests and the lint pass locally and the work is committed; pull requests are opened when
convenient, usually for several versions at once, and CI checks them then.

To read a failed run:

```powershell
gh run list --workflow ci --limit 5
gh run view <run-id> --log-failed
```

The failed step is Install, Lint, or Test. Run the same command locally to reproduce it. If it
passes locally and fails in CI, the usual causes are a test that depends on something only
this machine has (a `.env` file, the `qlora` AWS profile, a Windows path) or a file name whose
upper and lower case differ from how the code spells it.

Owner steps on GitHub:

- **First push of the workflow file** needs a token with the `workflow` scope:
  `gh auth refresh -h github.com -s workflow`.
- **CI is a required check** since 2026-11-15: the `protect-main` ruleset requires
  `lint and test` to pass before a pull request can merge. If the job is ever renamed in
  `ci.yml`, update the check name in the ruleset in the same pull request window (Settings,
  Rules, Rulesets, `protect-main`), or no pull request will be able to merge.

## 3. Local GPU environment (req_000 v0.03)

Path A is developed and smoke-tested on this machine (native Windows; RTX 4090 Laptop GPU,
16 GB). Measured runs happen on SageMaker.

### 3.1 Install the training stack

```powershell
venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
venv\Scripts\python -m pip install -r requirements-train.txt
```

The first command installs the CUDA build of torch (several gigabytes). The second installs the
other pinned libraries on top of it without replacing it. `requirements-train.txt` holds exact
versions; the same versions go into the cloud image. CI does not install this file.

### 3.2 Check the machine

```powershell
venv\Scripts\python -m scripts.env_check
```

Fast and offline. It prints one line per check:

| Check | Passes when |
|---|---|
| E1 Python | The interpreter is 3.11 or later |
| E2 Pinned packages | Every package in `requirements-train.txt` is installed at its pinned version |
| E3 CUDA build of torch | torch is the `cu126` build and can reach the GPU |
| E4 GPU | A GPU with at least 15 GiB is present; the line shows its name, memory, and driver |
| E5 BF16 | Compute capability is 8.0 or higher and torch reports BF16 as supported |
| E6 bitsandbytes | A small layer is quantized to 4-bit on the GPU and run |
| E7 Smoke | Skipped unless `--smoke` is given |

A check whose prerequisite failed is shown as SKIP. Exit code 0 means nothing failed, 1 means a
check failed, 2 means `config/local_env.yaml` is missing or incomplete.

### 3.3 Prove the stack end to end

```powershell
venv\Scripts\python -m scripts.env_check --smoke
venv\Scripts\python -m scripts.env_check --smoke --report artifacts/env_check/report.json
```

The smoke loads `Qwen/Qwen3-0.6B` in 4-bit, generates a few tokens with thinking disabled,
attaches LoRA, and runs one training step. The first run downloads about 1.5 GB from Hugging
Face (no account needed) and takes about half a minute; later runs take a few seconds. It uses
about 1.2 GiB of GPU memory. `--report` saves the versions, GPU facts, and results as JSON
under the git-ignored `artifacts/` folder.

Passing this on 2026-11-15 was milestone M1 and settled that local development stays on native
Windows.

The same checks as tests: `venv\Scripts\python -m pytest -m gpu -q`.

### 3.4 Housekeeping

- **Model cache**: `%USERPROFILE%\.cache\huggingface\hub`. Delete the
  `models--Qwen--Qwen3-0.6B` folder there to free the space; the next smoke downloads it again.
- **Moving the cache**: set `HF_HOME` in the shell before running. Hugging Face libraries do
  not read `.env`.
- **Changing a pin**: edit `requirements-train.txt`, reinstall, run `env_check --smoke`, and
  record the change in the roadmap. The cloud image must be changed to match.

### 3.5 Troubleshooting

| Symptom | Cause and fix |
|---|---|
| E2 lists packages as not installed | The stack is not installed, or `venv/` was recreated. Repeat 3.1 |
| E3 says torch "is not the cu126 build" | A CPU build of torch was installed, usually by running the second command of 3.1 alone. Run the first command, then the second |
| E3 says torch "cannot reach a CUDA device" | Driver problem or the GPU is disabled. Check `nvidia-smi` |
| E6 fails with a CUDA setup error | bitsandbytes cannot load its GPU library. Reinstall it at the pinned version; if it persists, record it in the roadmap, since the fallback is a decision for the owner |
| E7 fails with a connection error | No route to huggingface.co. The model is only downloaded once; retry when online |
| E7 fails with out-of-memory | Another program is using the GPU. Close it and check `nvidia-smi` |
| "triton not found" at start | Printed by torch on Windows. Harmless |
| `FutureWarning` about `gradient_checkpointing_enable` during the smoke | A library calls a name that transformers has deprecated. Harmless |

## 4. Data pipeline (req_001)

All four parts exist: the dataset snapshot and its schema report (v1.01), the frozen split
(v1.02), canonical serialization (v1.03), and the token budget (v1.04).

### 4.1 What is pinned

The project's FinQA is the Hugging Face dataset `bevaya/FinQA` (formerly
`rootsautomation/FinQA`, which now redirects) at one commit. `config/data.yaml` holds the
commit and, for each of the three Parquet files, its SHA-256, size, and row count. The dataset
is public: no account or token is needed.

| Thing | Where | In git |
|---|---|---|
| Pin and expected facts | `config/data.yaml` | Yes |
| The three data files and `snapshot.json` | `artifacts/data/finqa/<commit>/` | No |
| Schema report | `reports/data/finqa_schema_report.md` and `.json` | Yes |
| Split settings | `config/split.yaml` | Yes |
| Split manifest (frozen) | `splits/finqa_split_manifest.json` | Yes |
| Split report | `reports/data/finqa_split_report.md` and `.json` | Yes |
| Prompt wording and layout | `config/prompt.yaml` | Yes |
| Serialization report (lengths, digests) | `reports/data/finqa_serialization_report.md` and `.json` | Yes |
| Full records, for reading | `artifacts/serialized/` | No, and never to be committed |
| Tokenizer pin, training length, reply limit | `config/token_budget.yaml` | Yes |
| Token budget report | `reports/data/finqa_token_budget_report.md` and `.json` | Yes |

### 4.2 Fetch the snapshot

Needs the training stack (section 3.1) for the download library, and network access. About
40 MB, once.

```powershell
venv\Scripts\python -m scripts.finqa_fetch
```

One line per file, for example
`PASS  train  data/train-00000-of-00001.parquet: 30,089,143 bytes, 6,251 rows, sha256 55dc8ef7405e...`.
Files that are present and correct are not downloaded again.

| Option | Effect |
|---|---|
| `--verify-only` | Checks the files on disk. No network, no download library |
| `--force` | Downloads every file again, replacing what is there |

Exit codes: 0 every file verified, 1 a file is missing or wrong or the download failed, 2 a
config problem.

Every load of the data repeats the check, so code can never read a changed file.

### 4.3 Build or check the schema report

Needs the snapshot; no network.

```powershell
venv\Scripts\python -m scripts.finqa_report
venv\Scripts\python -m scripts.finqa_report --check
```

The first command runs checks S1 to S10 against the expected facts in `config/data.yaml` and
rewrites the two report files. The second writes nothing and fails if the committed files
differ from a fresh report. The report holds counts and example ids only, and no date, so on
the same snapshot it never changes.

| Check | Passes when |
|---|---|
| S1 Rows per file | Each official split has its configured row count |
| S2 Columns | Every file has exactly the expected column names and types |
| S3 No null values | No column holds a null |
| S4 Unique ids | No id repeats |
| S5 Id format | Every id matches the pattern and extends its `filename` |
| S6 Program form | Every program splits into steps of two arguments |
| S7 Operations | The operations seen equal the expected set |
| S8 Constants | The constants seen equal the expected set |
| S9 Argument kinds | The argument kinds seen equal the expected set |
| S10 Program length | No program is longer than the expected maximum |

Exit codes: 0 every check passed (and, with `--check`, the files are current), 1 a check
failed or a file is out of date, 2 a config problem or no verified snapshot.

### 4.4 The frozen split

Every example is in exactly one of `fit`, `dev`, or `test`. All examples of one company are on
the same side. The assignments are in `splits/finqa_split_manifest.json`, which was written
once and is never replaced by a command.

| Split | Examples | Companies | Used for |
|---|---|---|---|
| fit | 4,884 | 79 | Training |
| dev | 499 | 15 | Checkpoint selection, early stopping, validation files |
| quick-dev | 200 (inside dev) | 12 | Fast baselines |
| test | 2,898 | 43 | The final comparison, once (req_005 v5.01) |

Check that the settings, the data, and the committed files still agree. Needs the snapshot; no
network:

```powershell
venv\Scripts\python -m scripts.finqa_split --check
```

It runs checks P1 to P10, prints the counts and the content hash, and writes nothing. Exit
codes: 0 all checks pass and the files are current, 1 a check fails or a committed file
differs, 2 a config problem or no verified snapshot.

Without `--check` the same command rebuilds the split report and confirms the manifest. It
writes the manifest only if none exists.

| Check | Passes when |
|---|---|
| P1 Complete | Every example in the snapshot has exactly one assignment |
| P2 Same data | The manifest names the dataset commit and checksums in `config/data.yaml` |
| P3 Ratio | The training side is within 0.5 points of 65% |
| P4 Groups whole | No company has examples in two splits |
| P5 Dev size | Dev is at most 500 examples and at least 475 |
| P6 Quick-dev | Exactly 200 ids, all in dev |
| P7 Pages and duplicates whole | No page and no duplicate question pair is divided |
| P8 Hash | The content hash matches the assignments |
| P9 Reproducible | Building from the settings and the snapshot gives the committed manifest |
| P10 Test locked | No unlock record exists (after v5.01: the record names this manifest) |

**The test split is locked.** `load_split("test")` raises until req_005 v5.01 creates
`splits/test_unlock.json`. Do not create that file before then; a test fails if it appears.

**Changing the split** invalidates every result. It is done only by a decision recorded in the
req_001 roadmap: then delete the manifest by hand and run the command without `--check`. There
is no force option.

Content hash of the split (milestone M2):
`d34fb22e37b78a31ac22287cc8e199ab942d93125f9a80cb0b323862e599a07a`

### 4.5 Canonical records

Every arm reads the same three texts per example: a system text, a user text (text evidence,
the table, the question), and a target (`PROGRAM:` and `ANSWER:` lines). The wording and the
layout are in `config/prompt.yaml`. Their hash is the **prompt id**:

`d63742a62b76d526ecf3d92df4cbad832846d4da7aa1ef13d15401ac28391641`

Build the records for fit and dev, check them, and write the report. Needs the snapshot and
the split manifest; no network. The test split is never read:

```powershell
venv\Scripts\python -m scripts.finqa_serialize
venv\Scripts\python -m scripts.finqa_serialize --check
```

| Check | Passes when |
|---|---|
| C1 Target shape | Every target is exactly two lines, `PROGRAM:` then `ANSWER:` |
| C2 Program unchanged | Every program is the dataset's, character for character |
| C3 Answer form | Every answer is a plain number or yes/no, with no exponent |
| C4 Table shape | Every table has its header, separator, and rows, each with the same number of cells |
| C5 Evidence complete | Every record shows exactly its text evidence, or `(none)` |
| C6 Question present | Every user text ends with its question |
| C7 One system text | Every record carries the configured system text and prompt id |
| C8 Deterministic | Serializing again gives identical records |
| C9 Digests | The fit and dev records are unchanged since the committed report |
| C10 No test record | Every record is a fit or dev example |

Exit codes: 0 every check passed and the files are current, 1 a check failed, the report is
out of date, or an id cannot be shown, 2 a config problem, no snapshot, or no split manifest.

**Read real prompts** before approving or changing the wording. Both stay on this machine:

```powershell
venv\Scripts\python -m scripts.finqa_serialize --show "<example id>"
venv\Scripts\python -m scripts.finqa_serialize --dump
```

`--show` prints one fit or dev record to the terminal; ids are listed in
`splits/finqa_split_manifest.json`, and a test id is refused. `--dump` writes every fit and
dev record to `artifacts/serialized/fit.jsonl` and `dev.jsonl`, which are git-ignored.

**Changing the wording.** Until the first model has been run on it, edit `config/prompt.yaml`
and run the build: check C9 fails once to announce that every record changed, and the report
is rewritten; run it again to confirm, update the pinned id in
`tests/data/test_serialization.py`, and record the new id in the req_001 roadmap. Once any arm
has been run on a wording, a change also needs a new `version` in the file and a rerun of
every arm.

### 4.6 Token budget

How long every fit and dev record is in tokens, counted with the tokenizer of `Qwen/Qwen3-4B`
at a pinned commit. Only the tokenizer is loaded: no model weights, no GPU.

| Limit | Value | Where |
|---|---|---|
| Longest training sequence (`max_length`) | 2,048 tokens | `config/token_budget.yaml` |
| Longest reply (`max_new_tokens`) | 192 tokens | `config/token_budget.yaml` |

Nothing is ever truncated to fit. A record longer than `max_length` is an error.

Needs the training stack (section 3.1) for the tokenizer library, the snapshot, the split
manifest, and the serialization report. The first time, fetch the tokenizer files (a few
megabytes); after that no network is used:

```powershell
venv\Scripts\python -m scripts.finqa_tokens --download
venv\Scripts\python -m scripts.finqa_tokens
venv\Scripts\python -m scripts.finqa_tokens --check
```

| Check | Passes when |
|---|---|
| T1 Fits | No fit or dev full sequence is longer than `max_length` |
| T2 Reply limit | No target with its end token is longer than `max_new_tokens` |
| T3 Tokenizer | The tokenizer in use has the pinned fingerprint |
| T4 Thinking off | Every prompt ends with the empty think block |
| T5 Template constant | The chat template adds the same number of tokens to every record |
| T6 Same records | The records counted are the ones the serialization report pins |

Exit codes: 0 every check passed and the files are current, 1 a check failed or the report is
out of date, 2 a config problem, no training stack, no tokenizer, no snapshot, or no manifest.

Measured on 2026-11-15: a full sequence in fit is 643 tokens at the 50th percentile, 1,060 at
the 99th, and 1,862 at most. One pass over fit is 3,254,181 tokens, 56% of it the system text.

**If T1 fails** after a change of wording or data: do not raise `max_length` by reflex and
never truncate. Look at the named records, then decide with the owner between a longer limit
(memory on the GPU must be re-proven) and leaving those records out of training on both paths.

### 4.7 Housekeeping

- **The snapshot is the project's only pinned copy.** The repository belongs to a third party
  and has been renamed once. Keep `artifacts/data/finqa/`; if it is lost and the repository is
  gone, the pinned bytes cannot be recovered.
- **Changing the pin** is a decision, not a routine step: every result depends on it. Change
  `revision` and the three file entries in `config/data.yaml`, fetch, rebuild the report, read
  the difference in the report, and record the change in the req_001 roadmap.
- **The download folder also holds a `.cache` folder** written by the download library. It can
  be deleted.

### 4.8 Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `huggingface_hub is not installed` | The training stack is missing from this venv. Section 3.1 |
| `SNAPSHOT ERROR: download of ... failed` | No route to huggingface.co, or the repository moved. Retry when online; if it has moved, find the new name and update `repo_id` |
| A FAIL line with `sha256 is ..., expected ...` after a download | The file on the Hub is not the one the config describes. Nothing was deleted. Do not edit the checksum to make it pass; compare the commit on the Hub with `revision` first |
| "a file ... does not match config/data.yaml ... run the fetch again with --force" | A file on disk was changed or the download was cut short. `--force` replaces it |
| `finqa_report` prints `CONFIG ERROR: the FinQA snapshot ... is not valid` | There is no snapshot here, or a file changed. Run section 4.2 |
| `finqa_report --check` prints `OUT OF DATE` | The committed report was edited by hand, or the report code changed. Rebuild it and review the difference before committing |
| `SPLIT ERROR: finqa_split_manifest.json exists and differs` | `config/split.yaml` or the data changed since the manifest was written. Nothing was overwritten. Restore the settings (`git diff config/split.yaml`); the split is frozen |
| `finqa_split --check` prints `OUT OF DATE` | Same cause, or a report file was edited by hand. If only a report file is named, rebuild it without `--check` and review the difference |
| `the split manifest does not match its content hash` | The manifest was edited. Restore it: `git checkout -- splits/finqa_split_manifest.json` |
| `TestLockedError: the test split is locked` | Working as intended. Use `fit`, `dev`, or `quick_dev`. The test split is read once, in req_005 v5.01 |
| A test says a file "reads FinQA rows directly" | Code outside `qlora_finetuning/data/` used the raw reader. Use `load_split` |
| `FAIL  C9  Digests` from `finqa_serialize` | The records differ from the committed report: the wording, the split, or the data changed. If the wording was changed on purpose, the run has rewritten the report; run again and review the difference. Otherwise find what changed with `git diff config/` |
| A test fails comparing the prompt id | `config/prompt.yaml` changed. If intended, follow "Changing the wording" in section 4.5 |
| `SERIALIZATION ERROR: example ... has a result that is neither a number nor yes/no` | The data holds a value the serializer does not know. Expected only after a change of the dataset pin |
| `REFUSED: ... is in the test split` | `--show` was given a test id. Only fit and dev records can be read |
| `finqa_tokens` prints `CONFIG ERROR: transformers is not installed` | The training stack is missing from this venv. Section 3.1 |
| `finqa_tokens` prints `CONFIG ERROR: the tokenizer ... could not be loaded` | The tokenizer files are not in the local cache. Run it once with `--download` |
| `FAIL  T3  Tokenizer` | A different tokenizer was loaded than the one pinned. Check `tokenizer.revision` in `config/token_budget.yaml`; do not edit the fingerprint to make it pass |
| `FAIL  T6  Same records` | The serialization report is missing or older than the wording. Run `finqa_serialize` first |
| A check S1 to S10 fails | The data differs from `config/data.yaml`. The detail names what. Expected only after a deliberate change of the pin |

## 5. Path A: Qwen3-4B with QLoRA (req_003)

Path A asks `Qwen/Qwen3-4B`, loaded in 4-bit on a GPU, for replies. On this machine it is
free and uses no network once the weights are downloaded. Training, SageMaker jobs, and
checkpoint selection are added by later versions of req_003; paid jobs are started by the
owner.

### 5.1 What is fixed

| Item | Where |
|---|---|
| The model and its commit (one pin for the tokenizer that counts and the model that answers) | `config/token_budget.yaml` (`tokenizer.model_id`, `tokenizer.revision`) |
| 4-bit NF4, double quantization, BF16 compute | `config/local_qlora.yaml` |
| Greedy decoding, batch size, and the two end tokens. Generation stops at an end token or the reply limit and nowhere else | `config/local_qlora.yaml` |
| The reply limit, 192 tokens, the same as every other arm | `config/token_budget.yaml` |

The prompt is the canonical record's system and user texts in the model's chat template,
with thinking switched off. Every value above is part of a run's settings: a run made under
other values cannot be resumed.

### 5.2 Download the weights (once, 8.04 GB, free)

```powershell
venv\Scripts\hf download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c
```

The files go to the Hugging Face cache (`%USERPROFILE%\.cache\huggingface\hub`), outside the
repository. The model is public; no account is needed. A run never downloads anything: without
the weights it stops with this command and exit code 2, before any run exists.

### 5.3 A trial of five examples

```powershell
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b --split quick_dev --limit 5
```

It loads the model (about 7 seconds), asks five questions, and ends with `PARTIAL`, which is
right for a limited run. The `measurements:` line gives peak GPU memory, load time, and tokens
per second. Look at the replies in the run's `predictions.jsonl` (never commit that file).

### 5.4 The untuned baseline, arm A0

```powershell
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b --split quick_dev
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b --split dev
```

Measured on this machine on 2026-11-19 at batch size 1: about 4.1 seconds per example, so 14
minutes for quick-dev and 33 minutes for dev; peak GPU memory 2.7 GiB. A complete run ends
with `RESULT: complete` and adds a line to `ledger/evaluations.jsonl`. Its input tokens must
equal the token budget report's prompt tokens for the split (128,495 for quick-dev, 317,882
for dev): a difference means the model was not asked what the budget counted.

These are development figures. The A0 number of the comparison comes from the cloud run.

A run that stops is continued with the command it prints (`--resume <run_id>`).

### 5.5 The batch size

`inference.batch_size` in `config/local_qlora.yaml` is how many prompts are generated
together. It is 1. At 8 the same 40 examples took 1.3 seconds each instead of 4.1, but 16 of
the 40 replies differed from the ones generated one at a time. Changing it is a change of
settings: earlier runs cannot be resumed under the new value, and runs at different values
should not be compared reply by reply.

### 5.6 The GPU tests

```powershell
venv\Scripts\python -m pytest -m gpu -q
```

They load the small smoke model, and the 4B model when its weights are in the cache.

### 5.7 Housekeeping

- **Free the disk space**: delete `models--Qwen--Qwen3-4B` in the cache folder. The next run
  then asks for the download again.
- **Moving the cache**: set `HF_HOME` in the shell; Hugging Face libraries do not read `.env`.
- **Predictions and measurements** stay under git-ignored `artifacts/runs/<run_id>/`. A local
  run adds lines to the committed `ledger/` files; commit them with the work.

### 5.8 Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `CONFIG ERROR: the weights of Qwen/Qwen3-4B ... are not in the local Hugging Face cache` | Not downloaded, or only partly. Run the command of 5.2; it continues a partial download |
| `CONFIG ERROR: the training stack is not installed` | Section 3.1 |
| `CONFIG ERROR: the tokenizer ... is not the one pinned` | The tokenizer files differ from the fingerprint in `config/token_budget.yaml`. Check the commit; do not edit the fingerprint to make it pass |
| `CONFIG ERROR: no CUDA GPU is available` | `venv\Scripts\python -m scripts.env_check` |
| `CONFIG ERROR: ... could not be loaded in 4-bit (OutOfMemoryError ...)` | Another program holds the GPU's memory. Check `nvidia-smi` and close it |
| `PREDICTOR FAILED: OutOfMemoryError while generating a batch of N` | Lower `inference.batch_size` and start a new run; the stopped run cannot be resumed under another batch size |
| Many replies stop with `length` | The untuned model sometimes repeats itself until the reply limit (45 of 499 on dev). It is scored as it is; do not add a stop string for Qwen alone |

### 5.9 Training: what is fixed (req_003 v3.02)

| Item | Where |
|---|---|
| Every hyperparameter: LoRA r 16, alpha 32, dropout 0.05 on all linear layers; batch 2 with accumulation 8; learning rate 2e-4, cosine, 5% warmup; 8-bit paged AdamW; 3 epochs; seed 42 | `config/training.yaml` |
| The training length, 2,048 tokens. **Nothing is truncated**: a longer record is an error that names it | `config/token_budget.yaml` (`max_length`) |
| A checkpoint every 100 optimizer steps, all kept, and the loss on dev at each | `config/training.yaml` (`checkpoint`) |
| Where finished adapters go | `artifacts/adapters/<run_id>/` (git-ignored) |

A training sequence is the prompt of section 5.1, then the reply, then the end token; only the
reply carries the loss. Its length is the one the token budget counted.

### 5.10 The plan (no GPU)

```powershell
venv\Scripts\python -m scripts.qlora_train --dry-run
venv\Scripts\python -m scripts.qlora_train --dry-run --smoke
```

It encodes the records with the tokenizer, prints the plan, and starts no run. For the full
run: 4,884 records, 3,254,181 tokens in one pass, 918 optimizer steps (306 an epoch), 46
warmup steps, 10 checkpoints (every 100 steps and step 918), worst batch 1,862 tokens wide.

### 5.11 The smoke run (about 5 minutes on this machine, free)

```powershell
venv\Scripts\python -m scripts.qlora_train --smoke
```

It trains on 64 fit records (the two longest first) for 12 steps, then loads the adapter
through the inference backend and asks it 24 quick-dev prompts. It prints seven checks:

| Check | Means |
|---|---|
| K1 Sequences | Every sequence has the token budget's length; none is over the training length |
| K2 Labels | Only the reply and its end token carry a label |
| K3 Trainer left the data alone | The trainer's dataset is the pipeline's, row for row. The trainer's own default would cut rows to 1,024 tokens without saying so |
| K4 Only the adapter trains | Every trainable parameter is a LoRA weight (33,030,144 of them) |
| K5 Worst batch fits | The longest records went through one training step; the line gives the peak GPU memory |
| K6 Loss falls | The last losses are under half the first |
| K7 Adapter reloads | The saved adapter loads, answers, and writes no thinking block; the line says how many programs run and how many replies are the same alone and in a batch |

Run it after any change to the training stack, the config, or the pipeline, and before any
paid job. A smoke adapter is not a result and is never arm A1.

### 5.12 The full run on this machine (hours of GPU time; optional, v3.06)

```powershell
venv\Scripts\python -m scripts.qlora_train
```

Measured in the smoke runs: 0.88 s per sequence, so about 71 minutes an epoch and 3.6 hours
for the 3 epochs. The worst batch needs 11.0 GiB allocated (12.4 reserved, 13.6 in one
resumed run) of the card's 16: close other programs that use the GPU first. Every run starts
by probing that batch, so one that cannot fit fails in its first seconds.

### 5.13 Resuming an interrupted run

A failed or stopped run prints the command:

```powershell
venv\Scripts\python -m scripts.qlora_train --resume <run_id>            # add --smoke for a smoke run
```

It starts a **new run** from the last checkpoint of the old one and carries the log on. It
is refused (exit 2) unless the model commit, the prompt, the split, the training settings,
and the records are the same. To rehearse an interruption: `--stop-after-steps N`.

### 5.14 Scoring an adapter

Checkpoints are scored after training, through the same runner as every other arm:

```powershell
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b-tuned --adapter <run_id> --split quick_dev
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b-tuned --adapter <run_id>/checkpoint-300 --split dev
```

`--adapter` names a training run (its final adapter) or one of its checkpoints, never a
path. An adapter trained for another prompt wording, split, or model commit is refused. A
smoke adapter runs under the name `qwen3-4b-tuned-smoke`. A tuned model answers more slowly
than the untuned one (8 to 14 tokens per second here, against 19.5), because the adapter is
not merged: allow about 5 seconds per example.

### 5.15 Where training files are

| What | Where | Committed |
|---|---|---|
| What a run was asked, its plan | `artifacts/runs/<run_id>/training.json` | No |
| Loss, learning rate, dev loss per logged step | `artifacts/runs/<run_id>/training_log.jsonl` | No |
| The probe, peak memory, seconds per step | `artifacts/runs/<run_id>/measurements.json` | No |
| Checkpoints (adapter, optimizer, trainer state): about 140 MB each | `artifacts/runs/<run_id>/checkpoints/checkpoint-N/` | No |
| The final adapter and `adapter_metadata.json` | `artifacts/adapters/<run_id>/` | No |
| The run, and its time, records, and training tokens | `ledger/runs.jsonl`, `ledger/ledger.jsonl` | Yes |

To free space, delete a run's `checkpoints/` folder once its adapter has been chosen; the
final adapter under `artifacts/adapters/` is separate. Smoke runs can be deleted whole.

### 5.16 Training troubleshooting

| Symptom | Cause and fix |
|---|---|
| `CONFIG ERROR: N training sequences are longer than max_length` | A record does not fit the training length. Nothing is truncated. Raising the limit, or leaving records out on both paths, is a decision for the owner |
| `TRAINING FAILED: OutOfMemoryError after 0 of N steps` | The worst batch does not fit beside what else is on the GPU. Check `nvidia-smi`, close other programs, start again |
| `TRAINING FAILED: ... the trainer changed the prepared dataset` | A library update changed how the trainer prepares data. Do not train; check `trainer_arguments` in `local_qlora/training.py` against the installed TRL |
| `TRAINING FAILED: ... the step arithmetic no longer agrees` | The trainer counts steps differently from `local_qlora/plan.py`. Fix the plan before training |
| `CONFIG ERROR: run ... cannot be continued: it was made with another ...` | The settings changed since the run. Restore them, or start a new run |
| `CONFIG ERROR: adapter ... was trained for another ...` | The adapter belongs to another experiment. Train again under the current settings |
| `UserWarning: Unable to fetch remote file ... offline mode is enabled` while saving | Expected. Training switches the Hugging Face Hub to offline so that a run never uses the network; the adapter is saved correctly |

### 5.17 Path A in AWS: what is fixed (req_003 v3.03)

| Item | Where |
|---|---|
| The image (AWS's PyTorch training container), the instance type, the volume, the S3 prefixes, each task's maximum runtime, the Path A cap | `config/sagemaker.yaml` |
| The price of an instance hour, with the date of the price list | `config/pricing.yaml` (`sagemaker.training`) |
| The profile, bucket, execution role, and quota names | `config/aws.yaml` |
| The inference batch size, 8 | `config/local_qlora.yaml` |

A job is given three things: the code of a commit, a **data bundle** of `fit` and `dev`, and
an S3 **workspace**. It installs the pinned libraries over the container's, downloads the
model, and runs `scripts.job_run` with a task. **No test row is ever uploaded.** A job bills
by the second while it runs, up to its maximum runtime.

**Who runs what.** `launch` and `stop` change AWS and are run by the owner, with `--yes`.
`bundle`, `rehearse`, `plan`, `status`, `logs`, and `pull` create nothing in AWS.

### 5.18 The data bundle and the rehearsal (no AWS, free)

```powershell
venv\Scripts\python -m scripts.sagemaker_job bundle
venv\Scripts\python -m scripts.sagemaker_job rehearse smoke
```

`bundle` writes `artifacts/bundle/` (git-ignored): the 4,884 `fit` and 499 `dev` examples in
one file of about 30 MB, and a record of what it holds. Write it again after any change of
the split manifest or the dataset pin.

`rehearse` runs a task on this machine the way a job would: the code is unpacked into a
temporary folder that has no snapshot, the examples come from the bundle, and
`scripts.job_run` runs there. The smoke task takes about 6 minutes of GPU. It leaves
nothing in the repository or the ledger. From a clean tree it uses the commit's code; with
uncommitted changes it uses the working tree's and says so. Rehearse before every launch
of a changed pipeline.

### 5.19 Plan a job (needs a sign-in; creates nothing)

```powershell
aws sso login --profile qlora
venv\Scripts\python -m scripts.sagemaker_job plan smoke
```

It prints the job, the most it can cost, and ten checks (from v3.04 also an estimate; see 5.24):

| Check | Means |
|---|---|
| J1 Signed in | The sign-in is valid and is the configured account |
| J2 A commit | The working tree is clean: a job runs a commit. Commit first (changes under `ledger/` do not count) |
| J3 Bundle | The bundle holds `fit` and `dev` only, no test id, and was built from the current split and dataset |
| J4 Role | The execution role exists and trusts SageMaker |
| J5 Quota | The account may run the instance type as a training job |
| J6 Nothing running | No other job of this project is in progress |
| J7 Within the cap | What Path A has cost, plus the most this job can cost, is within `budget.path_a_cap_usd` ($100) |
| J8 Priced and pinned | The instance type has a price, and the image has a tag |
| J9 Estimated | For `train`, `score`, and `evaluate`: a pulled job has measured this kind of work, and the estimate fits in the job's limit. The smoke job has no estimate and passes |
| J10 Registered | Once a checkpoint is selected, a full training run from its start is the one corrective iteration: registered with `qlora_iterate propose`, on the first job's workspace, and made once (5.30). A first run, a resume, and the other tasks pass |

### 5.20 Launch, follow, and pull a job (the owner launches; billed)

```powershell
venv\Scripts\python -m scripts.sagemaker_job launch smoke --yes     # THE OWNER: about 25 minutes, about $0.60, at most $1.41
venv\Scripts\python -m scripts.sagemaker_job status
venv\Scripts\python -m scripts.sagemaker_job logs --tail 50
venv\Scripts\python -m scripts.sagemaker_job pull
git add reports/local_qlora ledger ; git commit -m "SageMaker smoke job"
```

`launch` uploads the code archive and the bundle, creates one training job, and writes the
job record `reports/local_qlora/sagemaker_jobs.json`. It refuses unless every check passes.
Running it again for the same commit adopts the job that exists; `--attempt 2` makes a new
one.

`pull`, once the job has ended, downloads its run folders and adapters into `artifacts/`,
appends its lines to the ledger, and records the job's billable seconds. Pulling twice adds
nothing. Checkpoints stay in S3 unless `--checkpoints` is given.

The other tasks, for later versions:

```powershell
venv\Scripts\python -m scripts.sagemaker_job launch train --yes                                   # the full run: hours
venv\Scripts\python -m scripts.sagemaker_job launch train --resume <run_id> --workspace <job name> --attempt 2 --yes
venv\Scripts\python -m scripts.sagemaker_job launch evaluate --predictor qwen3-4b-tuned --adapter <run_id>/checkpoint-300 --split dev --workspace <job name> --yes
```

A job on an earlier job's `--workspace` finds that job's runs and checkpoints in place.

### 5.21 Stop a job, and clean up

```powershell
venv\Scripts\python -m scripts.sagemaker_job stop --yes            # THE OWNER
aws s3 rm s3://qlora-sagemaker-123456789012/sagemaker/workspaces/<job name>/ --recursive --profile qlora   # THE OWNER
```

A stopped job is billed for the time it ran, and its workspace keeps what it wrote. A
training job leaves nothing running when it ends: there is no endpoint and no instance to
delete. What remains is S3 storage, a few cents a month per workspace; delete a workspace
once its runs are pulled and its adapter is no longer needed there.

### 5.22 Troubleshooting a job

| Symptom | Cause and fix |
|---|---|
| `CONFIG ERROR: the AWS sign-in is missing or has expired` | `aws sso login --profile qlora` |
| `FAIL  J2  A commit: uncommitted changes` | Commit, then plan again |
| `FAIL  J3  Bundle: ... built from another split manifest` | `sagemaker_job bundle` again |
| `FAIL  J6  Nothing running` | Wait for the other job, or stop it |
| `AWS ERROR: AWS answered ResourceLimitExceeded when asked to create the job` | The quota for the instance type is in use or zero. `aws_check`, check C5 |
| The job fails in its first minutes | `sagemaker_job logs`. Look for the pip step, the model download, and the line that lists installed versions. A failed job is billed only for the minutes it ran |
| `STATUS: config_error ... not the pinned one` in the log | The container installed another version than `requirements-train.txt` pins. Do not loosen the pin; find out why pip chose otherwise |
| `OutOfMemoryError after 0 of N steps` in the log | The worst batch did not fit. It fits in 24 GB by a wide margin, so check the instance type |
| `pull` says `no result file` | The job ended before its task did (it was stopped, or hit its maximum runtime). Its runs and checkpoints are in the workspace; launch again on the same `--workspace` with `--resume` |
| The job is `Stopped` with `MaxRuntimeExceeded` | Raise the task's `max_runtime_seconds` only after reading why it ran long; resume on the same workspace |

### 5.23 The full run: what it takes and who runs it (req_003 v3.04)

Arm A1 is made by two billed jobs and one free command. **Both jobs are the owner's.** The
v3.03 smoke job must have been run and pulled first: the plan computes its estimate from
the seconds that job measured, and check J9 fails without them.

| Step | Command | Billed |
|---|---|---|
| Train on `fit`, 3 epochs, ten checkpoints | `sagemaker_job launch train` | Yes: hours of one `ml.g5.xlarge` |
| Score the untuned model and all ten checkpoints on `dev` | `sagemaker_job launch score` | Yes |
| Choose the checkpoint | `qlora_select` | No: it reads the pulled runs |
| Fetch the chosen adapter, and store it for good | `sagemaker_job pull --checkpoint N`, `keep` | Pennies of S3 |

The thresholds a selection is judged by, and where its record goes, are the `selection`
block of `config/training.yaml`.

### 5.24 The training job (the owner; billed)

```powershell
aws sso login --profile qlora
venv\Scripts\python -m scripts.sagemaker_job plan train --scope measured
venv\Scripts\python -m scripts.sagemaker_job launch train --scope measured --max-runtime <proposed> --yes     # THE OWNER
venv\Scripts\python -m scripts.sagemaker_job status
venv\Scripts\python -m scripts.sagemaker_job logs --tail 20
venv\Scripts\python -m scripts.sagemaker_job pull
git add reports/local_qlora ledger ; git commit -m "Path A: the full training run"
```

The plan prints two lines the smoke plan did not have:

```
estimate    14,652 training sequences at 0.90 s, plus 0.4 hours of overhead: about 4.0 hours: about $5.65
proposed    --max-runtime 21720
```

The estimate is the work times the seconds the last pulled job measured on this instance
type, plus that job's overhead. `proposed` is half as much again, and at least two hours.
Give it to `launch` as `--max-runtime`: a job is billed for what it runs, never for its
limit, and the limit is what stops a job that hangs. `--scope measured` marks the run as
one whose figures the comparison may use.

If the job stops before step 918 (a limit, an AWS fault), nothing is lost: its checkpoints
are in its workspace. Continue it there, as a new attempt:

```powershell
venv\Scripts\python -m scripts.sagemaker_job launch train --scope measured --resume <run_id> --workspace <training job> --attempt 2 --yes   # THE OWNER
```

`pull` brings the training run's folder (its log with the loss on dev at every checkpoint,
its measurements) and the final adapter. The checkpoints stay in S3.

### 5.25 The scoring job (the owner; billed)

```powershell
venv\Scripts\python -m scripts.sagemaker_job plan score --run <run_id> --workspace <training job> --scope measured
venv\Scripts\python -m scripts.sagemaker_job launch score --run <run_id> --workspace <training job> --scope measured --yes     # THE OWNER
venv\Scripts\python -m scripts.sagemaker_job pull
git add reports/local_qlora ledger ; git commit -m "Path A: the checkpoints scored on dev"
```

`--workspace` is the training job's name: the checkpoints are there. `--run` is the
training run's id (`sagemaker_job status --job <training job>` lists it; for a resumed
training, give the last run, and the earlier checkpoints are found through it). The job
asks the untuned model and then each checkpoint all 499 `dev` examples; each is an
ordinary evaluation run. A scoring job that stops is launched again with `--attempt 2` on
the same workspace: what is already scored is skipped, and a run that was cut short is
resumed.

To try the task here, free, on a local smoke run:

```powershell
venv\Scripts\python -m scripts.job_run score --run <a local smoke run> --split quick_dev --limit 24
```

### 5.26 Choose the checkpoint (free; no AWS, no GPU)

```powershell
venv\Scripts\python -m scripts.qlora_select --run <run_id> --check     # report only
venv\Scripts\python -m scripts.qlora_select --run <run_id>
git add reports/local_qlora ; git commit -m "Path A: arm A1 selected"
```

It takes the checkpoint with the most `dev` examples right by **execution accuracy**; a tie
goes to the higher format validity, then to the earlier step. The loss on dev is printed
beside it and never chooses. Six checks must pass before anything is written:

| Check | Means |
|---|---|
| L1 Every checkpoint scored | Each checkpoint, and the untuned model, has exactly one complete result on `dev` |
| L2 Same question | All of them were asked with one prompt, one split, one reply limit, and the same settings |
| L3 The right weights | Each run used the weights of its own checkpoint, by hash |
| L4 The run it claims | The training run completed at its last step and was made for the current model, prompt, split, and settings |
| L5 Measured in the cloud | Every run was made in SageMaker with `--scope measured`, in a job that is pulled and whose billable seconds are in the ledger |
| L6 Scores current | Every run passes the runner's checks R1 to R5 today |

It writes `reports/local_qlora/adapter_selection.json` and `.md`. **That record is what
says which run is arm A1**: ten runs carry the arm, and later code asks for
`--adapter selected`. The verdict is `met` or `missed` against the two thresholds
(execution accuracy at least 40%, format validity at least 98%). Exit code 0 is selected
and met; 1 is a failed check or `missed`; 2 is a config problem. `missed` starts nothing:
it offers the one corrective iteration of v3.05, which is the owner's to decide.

`--split` and `--limit` rehearse the selection on partial runs and need `--check`. A smoke
run can only be rehearsed.

### 5.27 Fetch the selected adapter, use it, and keep it

```powershell
venv\Scripts\python -m scripts.sagemaker_job pull --job <training job> --checkpoint <N>
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b-tuned --adapter selected --split quick_dev --limit 24
venv\Scripts\python -m scripts.sagemaker_job keep --yes          # THE OWNER: writes to S3
git add reports/local_qlora ; git commit -m "Path A: the selected adapter kept"
```

`pull --checkpoint` downloads one checkpoint's adapter (about 130 MB) and reads only.
`--adapter selected` works wherever an adapter is named, in `evaluate` and in
`sagemaker_job launch evaluate`; it is refused if the weights here are not the ones that
were scored, or if the record was made for another prompt, split, or model commit.
`keep` copies the adapter and the selection record to `adapters/<run>/checkpoint-N/` in the
bucket, where it stays when workspaces are deleted, and notes it in the job record.

### 5.28 Troubleshooting the full run

| Symptom | Cause and fix |
|---|---|
| `FAIL  J9  Estimated: no pulled job ... has measured this kind of work` | The smoke job has not been run and pulled. Do that first (5.20) |
| `FAIL  J9 ... over this job's limit ... launch with --max-runtime N` | The estimate is longer than the limit. Add the `--max-runtime` it names, and read J7 again: the most the job can cost has grown |
| `CONFIG ERROR: this job continues an earlier job's work ... add --workspace` | A `score` job, a resumed training, and a tuned evaluation need the workspace of the job that made the checkpoints |
| `FAIL  L1 ... checkpoint N has 0 complete results` | The scoring job did not finish, or is not pulled. `pull`; if it stopped, launch `score` again with `--attempt 2` |
| `FAIL  L1 ... has 2 complete results` | A checkpoint was scored twice under the same settings. Move the unwanted run's folder out of `artifacts/runs/` |
| `FAIL  L5 ... is local and dev` | The runs were made here, or without `--scope measured`. Only cloud runs with that scope can be arm A1 |
| `FAIL  L5 ... is not pulled, or its billable seconds are not recorded` | `sagemaker_job pull --job <name>` |
| `CONFIG ERROR: no checkpoint has been selected yet` | `--adapter selected` or `keep` before `qlora_select` has written its record |
| `the weights of ... here are not the ones the selection was made on` | `pull --checkpoint N` again |
| `VERDICT: missed` | Not a fault of the tooling. Read the report; v3.05 is offered |

### 5.29 Where arm A1 goes wrong: the error analysis (req_003 v3.05; free)

```powershell
venv\Scripts\python -m scripts.qlora_analyze --run <run_id> --check       # a look; writes nothing
venv\Scripts\python -m scripts.qlora_analyze --run <run_id>
venv\Scripts\python -m scripts.qlora_analyze --run <run_id> --check --show order --n 5
git add reports/local_qlora ; git commit -m "Path A: error analysis on dev"
```

Run it after `qlora_select`, whatever the verdict: the comparison report needs it. It reads
the pulled runs, asks no model, and uses no AWS. Every wrong reply gets one category, by the
first rule that fits:

| Category | Means |
|---|---|
| `cut_off` | The reply hit the token limit |
| `no_program`, `bad_program` | No program line, or one that does not parse (the error code says how) |
| `not_executable` | It parses and cannot be run: a row that is not there, a division by zero |
| `kind` | A number where yes or no was due, or the reverse |
| `steps_more`, `steps_fewer` | More, or fewer, steps than the gold program |
| `operation` | The same number of steps, another operation |
| `evidence` | The same operations, other numbers |
| `order` | The same operations and numbers, two arguments reversed |
| `scale`, `sign` | The gold program itself, and a result off by a factor, or negated |

It writes `reports/local_qlora/error_analysis.json` and `.md`: the categories of the
selected checkpoint beside the untuned model, the categories at every checkpoint, the
failures by length of the gold program, by operation, by company, and by prompt length, and
the examples no checkpoint got right. **The files hold counts and example ids, never a
reply.** `--show CATEGORY` prints replies beside their gold programs to the terminal only;
do not paste them into a committed file.

The last line is the **finding**: which row of the table below the figures fit. It decides
nothing.

### 5.30 One corrective iteration (only after `VERDICT: missed`; the owner's; billed)

Path A gets one more training run, with **one** setting changed, and only when the
selection missed a threshold. A fault in the pipeline is not an iteration: fix it, and make
the first run again.

| Finding | What the analysis shows | The one change |
|---|---|---|
| T1 | The best checkpoint is at the end, and the last three checkpoints still rise | `epochs` 3 to 5 |
| T2 | The best checkpoint is in the first half, later ones are lower, and the dev loss rises | `optimizer.learning_rate` 2e-4 to 1e-4 |
| T3 | Flat from the middle on, and most failures are steps, operation, or evidence | `lora.r` 16 to 32, with `lora.alpha` 64 |
| T4 | Format validity missed and accuracy did not: read the curve as T1 or T2 | as T1 or T2 |
| T5 | Most failures are cut off, or not programs | None: look for a fault first |

The values are in `config/training.yaml` (`iteration.changes`). You may choose otherwise
(`--finding other`) among `epochs`, `optimizer.learning_rate`, `lora.r` (alpha stays at
twice it), `lora.dropout`, and `schedule.warmup_fraction`. Nothing else may change: the
prompt, the split, the reply limit, and the inference settings are shared with the other
arms.

```powershell
# 1. edit ONE value in config/training.yaml, then register it:
venv\Scripts\python -m scripts.qlora_iterate propose --finding T2 --reason "peaks at step 300 and falls"
git add -A ; git commit -m "Path A: one corrective change registered"

# 2. the second pair of jobs, on the FIRST training job's workspace (THE OWNER; billed):
venv\Scripts\python -m scripts.sagemaker_job plan train --scope measured --workspace <first training job>
venv\Scripts\python -m scripts.sagemaker_job launch train --scope measured --workspace <first training job> --max-runtime <proposed> --yes
venv\Scripts\python -m scripts.sagemaker_job pull
venv\Scripts\python -m scripts.sagemaker_job launch score --run <second run> --workspace <first training job> --scope measured --yes
venv\Scripts\python -m scripts.sagemaker_job pull

# 3. select across both runs, analyse the winner, and check the config:
venv\Scripts\python -m scripts.qlora_select --run <second run>
venv\Scripts\python -m scripts.qlora_analyze --run <selected run>
venv\Scripts\python -m scripts.qlora_iterate status
git add -A ; git commit -m "Path A: the corrective iteration"
```

`propose` runs five checks and writes `reports/local_qlora/iteration.json` only if all pass:

| Check | Means |
|---|---|
| I1 Triggered | A checkpoint is selected for the current experiment, and its verdict is `missed` |
| I2 Analysed | The error analysis of that checkpoint exists |
| I3 One variable | `config/training.yaml` differs from the first run in exactly one allowed variable |
| I4 The first time | No second run has been made |
| I5 The same experiment | Prompt, split, model, reply limit, and inference settings are the first run's |

The plan of a training job has a tenth check, **J10 Registered**: once a checkpoint is
selected, a full training run needs that record, the first job's workspace, and that no
second run has completed. On that workspace the scoring job finds the untuned model's
result and does not ask it again.

`qlora_select --run <second run>` compares the second run's best checkpoint with the one
already selected, by the same rule. **A tie keeps the first run's.** The record then lists
both runs, says whether the gap is within noise (`selection.noise_examples`, 10 of 499),
and the first record is kept as `adapter_selection_first.json`. If the first run's
checkpoint stays, revert the change in `config/training.yaml` and commit: `qlora_iterate
status` fails until the file holds the selected run's settings. Whatever the second
verdict, Path A goes on to the test evaluation: there is no second iteration.

| Symptom | Cause and fix |
|---|---|
| `FAIL  I1 ... the verdict is met` | Nothing was missed, so there is nothing to correct |
| `FAIL  I3 ... is the first run's` | The YAML is unchanged. Edit one value, then propose |
| `FAIL  I3 ... N variables differ`, or `... may not change` | Revert all but one allowed change |
| `FAIL  I3 ... lora.alpha must stay at 2 times lora.r` | Change both, or neither |
| `FAIL  J10 ... is registered first` | `qlora_iterate propose`, or the YAML is not the registered change |
| `FAIL  J10 ... add --workspace <name>` | The second run goes on the first training job's workspace |
| `FAIL  J10 ... has been made` | The one iteration is used. A stopped second run is continued with `--resume` |
| `CONFIG ERROR: ... only as the registered corrective iteration` | `qlora_select` was given another run than the selected one, and no change is registered |
| `CONFIG ERROR: ... select with the second run's id` | After an iteration, select with the second run: it compares both |
| `FAIL  S1` in `status` | The YAML is not the selected run's settings: revert the change, or register it |

### 5.31 The full run on this machine (req_003 v3.06; optional, free, about 5 hours)

A learning exercise: the full fine-tune on the laptop's GPU, with the cloud run's config.
**It is a side note: not an arm of the comparison, never arm A1, and without a price.**
Nothing is billed and nothing reaches AWS. The machine is not free for other GPU work
while it runs.

**Before**: plug in, and close what holds GPU memory (a browser with hardware
acceleration, chat and office programs). Training reserved up to 13.6 GiB of the card's
16; with less free it spills into shared memory, which does not fail and is many times
slower.

```powershell
venv\Scripts\python -m scripts.qlora_local preflight
```

It starts nothing and prints six checks:

| Check | Means |
|---|---|
| P1 The plan | The dry run succeeds: 4,884 records, 918 steps, ten checkpoints. Hours are estimated from the last local run's measured speed |
| P2 A clean tree | No uncommitted change outside `ledger/`: the run is a commit's |
| P3 GPU memory | At least `local_run.min_free_gpu_gib` (14.0) is free. It names the programs that hold memory. With every program closed this laptop had 14.3 to 14.5 GiB free: the Windows desktop (`explorer.exe`) holds 1.1 to 1.4, and that cannot be closed |
| P4 Disk | At least `local_run.min_free_disk_gb` (10) free. Ten checkpoints are 1.4 GB |
| P5 Power | On AC power and not set to sleep there. Skipped where the system does not say |
| P6 Nothing running | No training run is in progress, and no stopped full run is waiting to be resumed |

**The run** (in a terminal of your own; about 4 hours):

```powershell
venv\Scripts\python -m scripts.qlora_train
venv\Scripts\python -m scripts.qlora_local status               # from a second terminal, any time
venv\Scripts\python -m scripts.qlora_train --resume <run_id>    # only if it stopped; never start again
```

`status` prints the step, the hours left at the speed so far, the last loss, the loss on
dev at the last checkpoint, and, for a run that stopped, the command that continues it. A
run can be resumed as often as needed; the checkpoints of the whole chain are found later.

**Scoring and the note** (about an hour):

```powershell
venv\Scripts\python -m scripts.job_run score --run <run_id> --split quick_dev             # about 50 minutes
venv\Scripts\python -m scripts.qlora_select --run <run_id> --split quick_dev --check      # the table; L5 fails by design
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b-tuned --adapter <run_id>/checkpoint-<N> --split dev
venv\Scripts\python -m scripts.qlora_analyze --run <run_id> --split quick_dev --check     # why replies were wrong
venv\Scripts\python -m scripts.qlora_local note --run <run_id>
git add reports/local_qlora ledger ; git commit -m "Path A: the local full run (a side note)"
```

`qlora_select --check` shows which checkpoint is best on quick-dev (`<N>`); it never
writes, because a local run cannot be arm A1. That checkpoint is then scored once on all
of dev, so that it can be set beside the cloud's. `note` writes
`reports/local_qlora/local_run.json` and `.md` when five checks pass:

| Check | Means |
|---|---|
| N1 Scored | Every checkpoint and the untuned model have one complete result on the split, with the right weights, for a run that completed under the current settings |
| N2 Local | Every run was made on this machine |
| N3 The full run | Not a smoke run, and trained on all of `fit` |
| N4 On dev | The selected checkpoint has exactly one complete result on all of dev |
| N5 Scores current | Every run passes the runner's checks today |

The note holds hours, speed, peak GPU memory, the per-checkpoint table, the selected
checkpoint, its result on dev, and the categories of its failures: no reply, no price.
Once `qlora_select` has selected a cloud checkpoint, run `note` again: it adds **local
against cloud** (hours, seconds per training sequence, the step each selected, the two
results on dev, and whether they are within noise).

The adapter stays on this machine under `artifacts/`; use it by name
(`--adapter <run_id>/checkpoint-<N>`). `--adapter selected` is the cloud's and is never
this one.

| Symptom | Cause and fix |
|---|---|
| `FAIL  P3  GPU memory` | Close the programs it names, and run the preflight again |
| `FAIL  P6 ... continue it, do not start again` | A full run stopped with checkpoints: `qlora_train --resume <run_id>` |
| `FAIL  P6 ... is in progress` | A training run wrote its files in the last 15 minutes. One at a time |
| Steps take several times longer than the preflight estimated | Something took GPU memory after the start and training spilled into shared memory. Close it; the speed recovers |
| `OutOfMemoryError` at the first step | The worst batch did not fit beside what else is on the card. Close programs; `--resume` is not needed, nothing was trained |
| The machine restarted or slept | `qlora_local status` shows the last checkpoint and the resume command |
| `FAIL  N4 ... no complete result on dev` | Run the `evaluate` command it prints |
| `FAIL  N4 ... has 2 results on dev` | The checkpoint was scored twice: move one run's folder out of `artifacts/runs/` |
| `FAIL  N2 ... is sagemaker` | The run was made in the cloud. Its record is the selection, not this note |

## 6. Path B: Amazon Nova on Bedrock (req_004)

Path B asks Amazon Nova Pro and Amazon Nova Lite for replies through the Bedrock Converse
API in us-east-1. **Every call to Bedrock is billed, so every command in this section that
reaches Bedrock is run by the owner.** Building a predictor, `aws_check`, and the report
script cost nothing.

### 6.1 What is fixed

| Item | Where |
|---|---|
| The model ids (plain in-region ids) and each model's untuned arm | `config/frontier.yaml` |
| Decoding: temperature 0 and `top_k` 1 (greedy). Nothing else is sent: no `top_p`, stop sequence, cache point, reasoning setting, or service tier | `config/frontier.yaml`, `qlora_finetuning/frontier/bedrock.py` |
| Retrying: botocore's `adaptive` mode (it paces calls after a throttle), 12 attempts, 120-second read timeout | `config/frontier.yaml` |
| The reply limit, 192 tokens, the same as every other arm | `config/token_budget.yaml` |
| Prices per 1M tokens, with the date of the AWS price list | `config/pricing.yaml` |

The prompt is the canonical record's system and user texts, unchanged. Changing a decoding
value makes earlier runs another experiment: a run made under other values cannot be resumed.

### 6.2 Before the first call

```powershell
aws sso login --profile qlora
venv\Scripts\python -m scripts.aws_check
```

Check C8 must pass: both models Active and callable on demand. It is read-only and free.

### 6.3 A trial of five examples (billed, under $0.01)

```powershell
venv\Scripts\python -m scripts.evaluate --predictor nova-lite --split quick_dev --limit 5
```

It proves the sign-in, the model access, and the request. It ends with `PARTIAL`, which is
right for a limited run. Look at the replies in the run's `predictions.jsonl`.

### 6.4 The untuned baselines (billed)

```powershell
venv\Scripts\python -m scripts.evaluate --predictor nova-lite --split quick_dev
venv\Scripts\python -m scripts.evaluate --predictor nova-pro --split quick_dev
venv\Scripts\python -m scripts.evaluate --predictor nova-lite --split dev
venv\Scripts\python -m scripts.evaluate --predictor nova-pro --split dev
```

| Run | Examples | Worst-case cost | Time |
|---|---:|---:|---|
| Nova Lite, quick-dev | 200 | $0.02 | a few minutes |
| Nova Pro, quick-dev | 200 | $0.28 | a few minutes |
| Nova Lite, dev | 499 | $0.05 | about ten minutes |
| Nova Pro, dev | 499 | $0.69 | about ten minutes |

The worst case assumes Nova counts 1.5 times as many tokens as the Qwen tokenizer and every
reply runs to the 192-token limit. A run cannot cost more than its split allows: one call per
example. Each run must end with `RESULT: complete; this run counts as a result`. The place
recorded is `bedrock` and the arm is `B0-Lite` or `B0-Pro`, without any extra option. Calls
are made one at a time. Log each command in `administration/aws_commands_run.md`.

### 6.5 The baseline report (free)

```powershell
venv\Scripts\python -m scripts.nova_baseline --runs <run_id> <run_id> <run_id> <run_id>
```

Give it the four run ids of 6.4 in any order. It calls no AWS service. When checks B1 to B4
pass it writes `reports/frontier/nova_baseline_report.md` and `.json`, which are committed:
scores, Bedrock's token counts, their ratio to the Qwen counts, cost at list price, latency.

| Check | Passes when |
|---|---|
| B1 Four results | Exactly one complete run per model and per split (`quick_dev`, `dev`), on the model's untuned arm |
| B2 Same question | All runs have the current prompt id and split manifest, one reply limit, and one set of settings |
| B3 Usage from AWS | Every prediction has both token counts and each run's ledger record carries Bedrock's counts |
| B4 As configured | Each run used the model id and decoding values of `config/frontier.yaml` |

| Exit code | Meaning |
|---|---|
| 0 | Every check passed and the report was written |
| 1 | A check failed. Nothing is written |
| 2 | A config problem, or a run id that is not on this machine |

### 6.6 Troubleshooting

| Symptom | What to do |
|---|---|
| `CONFIG ERROR: the AWS sign-in is missing or has expired` | `aws sso login --profile qlora`. No run was started and nothing was billed |
| `PREDICTOR FAILED: the AWS sign-in is missing or has expired` | The sign-in ran out during a run. Sign in, then run the printed `--resume` command. Nothing is lost and no example is asked twice |
| `PREDICTOR FAILED: Bedrock answered ThrottlingException ...` | Bedrock was still refusing after twelve attempts. Nova Pro allows 25 requests a minute on this account, its tuned deployment 20. Since 2026-11-22 the predictor paces Nova Pro at 18 a minute (`models.nova-pro.requests_per_minute` in `config/frontier.yaml`), so this should be rare. Wait a minute and run the printed `--resume` command |
| `PREDICTOR FAILED: Bedrock answered ValidationException ...` | The log line above it has AWS's own message. If it names the temperature, set `inference.temperature` to `0.00001` in `config/frontier.yaml` (the documented minimum; with `top_k` 1 the reply is the same) and start the run again |
| `PREDICTOR FAILED: Bedrock answered AccessDeniedException ...` | The account may not call that model. Run `aws_check` and read C8 |
| `PREDICTOR FAILED: ... reports cached tokens` | Bedrock reused part of the prompt and counted it apart, so the input count is not the whole prompt. Stop and report it: the cost figures would be wrong |
| `REFUSED: run ... cannot be continued: it was made with another predictor` | A decoding value or the model id changed since the run began. Start a new run |
| Many replies end with stop reason `length` | The 192-token limit is in Nova's tokens. Look at the replies before changing anything: a reply cut at 192 was almost certainly wrong already |

### 6.7 The training files and the budget gate (req_004 v4.02)

```powershell
venv\Scripts\python -m scripts.nova_export
```

Free; calls no AWS service. It writes the two files Bedrock will train on under the
git-ignored `artifacts/bedrock/<export id>/` (`finqa_fit.jsonl`, 4,884 lines, the training
file; `finqa_dev.jsonl`, 499 lines, the validation file), reads them back, prices Path B,
and writes the committed gate record `reports/frontier/nova_training_gate.md` and `.json`.
**No fine-tuning job may be started unless that record says passed.**

| Setting | Where |
|---|---|
| Epochs the gate prices, and v4.03 launches with: 2 | `config/frontier.yaml`, `customization.epoch_count` |
| The cap for Path B: $200 of the $300 for both paths | `config/frontier.yaml`, `budget.path_b_cap_usd` |
| The margin added to the training estimate: 10% | `config/frontier.yaml`, `estimate.margin_percent` |
| The ratio shown when none was measured: 1.5 (never enough to open the gate) | `config/frontier.yaml`, `estimate.assumed_token_ratio` |
| AWS's limits: 8 to 20,000 records, 32,000 tokens a record | `config/frontier.yaml`, `export` |

| Check | Passes when |
|---|---|
| X1 Same records | Each file has one line per record of its split, in order, with the canonical record's three texts |
| X2 No test rows | No exported id is a test id, and the two files share no id |
| X3 Format | Every line is a Bedrock conversation (system, one user turn, one assistant turn); UTF-8; no carriage return; allowed file names |
| X4 Reserved words | No text holds `system:`, `user:`, `bot:`, `assistant:`, `thought:`, `[eos]`, `<image>`, `<video>`, or `<unk>`, in any case |
| X5 Limits | The record counts and the longest record are inside AWS's limits |
| X6 Same question | The prompt and the split are the current ones, and those the token budget report was made from |
| X7 Reproducible | Rendering again gives the bytes on disk |
| G1 Ratio measured | The baseline report of 6.5 exists, passed, and is for the current prompt and split |
| G2 Within the cap | Path B in total (ledger, training with margin, evaluation to come, storage) is at most the cap |

| Exit code | Meaning |
|---|---|
| 0 | Every check passed: the gate is open |
| 1 | A check failed. The record is still written and says not passed. With `--check`: also when the record is out of date |
| 2 | A config problem, no snapshot, or with `--upload` no sign-in |

**Until the baselines of 6.4 are run and the report of 6.5 is written, G1 fails and the gate
is closed. That is intended.** The estimate is still printed, at the assumed ratio.

```powershell
venv\Scripts\python -m scripts.nova_export --check
```

Writes no committed file; fails if the gate record differs from a fresh one. Run it after
changing the config, the prices, or the prompt.

### 6.8 Stage the training files in S3 (the owner)

```powershell
aws sso login --profile qlora
venv\Scripts\python -m scripts.nova_export --upload
```

Creates two objects, `s3://qlora-sagemaker-123456789012/data/bedrock/<export id>/finqa_fit.jsonl`
and `.../finqa_dev.jsonl` (about 12 MB together; storing them is well under a cent a month).
Each is sent with its SHA-256, read back, and compared with the local file; the locations
are written into the gate record, which is then committed. Uploading is refused if a check
X1 to X7 failed. It does not need the gate to be open, so the files can be staged before the
baselines are run. Files already there and equal are not sent again. Log the command in
`administration/aws_commands_run.md`.

To delete the staged files (they hold dataset text from fit and dev, never from test):

```powershell
aws s3 rm s3://qlora-sagemaker-123456789012/data/bedrock/ --recursive --profile qlora
```

| Symptom | What to do |
|---|---|
| `FAIL  G1  Ratio measured: not measured: no nova_baseline_report.json` | Run the baselines (6.4) and the report (6.5), commit the report, run the export again |
| `FAIL  G1 ... was made from another prompt wording` | The prompt changed after the baselines. They must be run again |
| `FAIL  X6 ... the token budget report was made from another prompt wording` | `venv\Scripts\python -m scripts.finqa_tokens`, then the export again |
| `FAIL  X4  Reserved words` naming an example | A text holds a word that makes a Nova job fail. Do not edit the file: the wording or the data has to be dealt with for every arm. Report it |
| `FAIL  G2  Within the cap` | The estimate is above `budget.path_b_cap_usd`. Fewer epochs, or a higher cap, is the owner's decision |
| `CONFIG ERROR: ... already exists with other content` | An object under that key is not this file. Delete it (command above) and upload again |
| `STALE: nova_training_gate...` from `--check` | Something the record depends on changed. Run the export without `--check` and commit the record |

### 6.9 The fine-tuning jobs (req_004 v4.03)

This is where Path B's money is spent. **`launch`, `deploy`, `stop`, and `teardown` change
AWS and are run by the owner; each needs `--yes`, and without it only prints what it would
do.** `plan` and `status` only read. Nothing here creates Provisioned Throughput, and no
command launches more than the one model it is told to.

| Setting | Where |
|---|---|
| Epochs 2, learning rate 0.00001, warmup steps 10: Bedrock's defaults, for both models | `config/frontier.yaml`, `customization` |
| The order: Nova Lite, then Nova Pro | `config/frontier.yaml`, `customization.launch_order` |
| The customization model ids (they end in `:300k`) and the tuned arms `B1-Lite`, `B1-Pro` | `config/frontier.yaml`, `models` |
| The role the jobs run as, and the bucket | `config/aws.yaml` |

**Before anything can be launched**: the baselines and their report (6.4, 6.5), the gate open
(6.7), the files staged (6.8), and everything committed.

```powershell
aws sso login --profile qlora
venv\Scripts\python -m scripts.nova_tune plan
```

Free. Per model it prints the job name, the files, the hyperparameters, the estimated
tokens and cost, and nine checks. Without a sign-in the checks that need AWS show `SKIP`.

| Check | Passes when |
|---|---|
| T1 Gate open | The gate record says passed and names both S3 objects |
| T2 Gate current | The prompt, split, epoch count, cap, and price-list date are the ones the gate priced |
| T3 Files in place | Both S3 objects have the recorded size and SHA-256 |
| T4 Model ready | The model is listed as fine-tunable and is Active |
| T5 No Provisioned Throughput | The account has none ($55 to $60.50 an hour if it had) |
| T6 Not launched yet | This attempt is not in the tuning record, and Bedrock has no other job of that name |
| T7 Within the cap | What Path B has cost so far, plus this job with the margin, is at most the cap |
| T8 Committed | No uncommitted change (the `ledger/` folder aside) |
| T9 Earlier bill seen | For Nova Pro: Nova Lite's job has completed and its billed amount has been entered |

**Nova Lite first** ($13 to $20):

```powershell
venv\Scripts\python -m scripts.nova_tune launch --model nova-lite --yes
git add reports/frontier ledger ; git commit -m "Nova Lite fine-tuning job launched"
venv\Scripts\python -m scripts.nova_tune status --wait
```

`launch` runs T1 to T9 again, creates the job, and writes the tuning record
`reports/frontier/nova_tuning.md` and `.json` and one ledger record with the estimated
training tokens. Commit the record: it is the only memory of the job. `status --wait` looks
every 60 seconds until the job has ended; it can be stopped with Ctrl+C and started again at
any time, in any session. A job takes hours. When it has completed, `status` also downloads
the loss curves to `artifacts/bedrock/<export id>/nova-lite-a1/`.

| Job status | Meaning |
|---|---|
| `InProgress` | Running: validation, then data processing, then training |
| `Completed` | The tuned model exists. Storage ($1.95 a month) has started |
| `Failed` | No model. `status` says in which phase: validation or data processing points at the files, training at the service. Read the message in the tuning record |
| `Stopping`, `Stopped` | Stopped by `stop`. No model; the tokens used are billed |

**Then wait for Nova Lite's charge** (the owner's decision: Nova Pro is not launched before
it has been checked). It appears on the AWS bill within about a day (Billing, Bills, Amazon
Bedrock, the model customization line). Enter it:

```powershell
venv\Scripts\python -m scripts.nova_tune bill --model nova-lite --usd <amount>
venv\Scripts\python -m scripts.nova_tune plan --model nova-pro
```

`bill` needs no sign-in. It replaces the estimate in the ledger with the amount. `plan` then
shows the amount beside the estimate and what Nova Pro would cost at the same proportion.
**If the two are far apart, stop and revisit the estimate before launching Nova Pro.**

**Nova Pro** ($52 to $78), after committing:

```powershell
venv\Scripts\python -m scripts.nova_tune launch --model nova-pro --yes
venv\Scripts\python -m scripts.nova_tune status --wait
```

**Deploy** each finished model so that it can be asked questions (v4.04):

```powershell
venv\Scripts\python -m scripts.nova_tune deploy --model nova-lite --yes
venv\Scripts\python -m scripts.nova_tune deploy --model nova-pro --yes
venv\Scripts\python -m scripts.nova_tune status --wait
```

A deployment is on demand: paid per token when used. When `status` shows it `Active`, its ARN
in the tuning record is the model id to ask. Commit the tuning record.

| Exit code | Meaning |
|---|---|
| 0 | Done; for `plan` every check passed; for `status` every job has completed and every deployment is Active |
| 1 | A check failed, a command was refused or lacked `--yes`, a job failed or is still running |
| 2 | A config problem or no sign-in |

| Symptom | What to do |
|---|---|
| `REFUSED: a check failed, so nothing was launched` | Read the FAIL lines below it. Nothing was created |
| `FAIL  T6 ... already launched ...; another launch needs --attempt 2` | That attempt exists. A second attempt of a model is a decision (req_004 v4.05), not a retry |
| `FAIL  T8  Committed` | Commit first. The tuning record of the previous launch is usually what is uncommitted |
| `CONFIG ERROR: AWS answered ServiceQuotaExceededException when asked to create the job` | Another job is running, or a quota is reached. Nothing was created; try again later |
| A launch fails with another `AWS answered ...` | Run the same command again. If Bedrock did create the job, it is adopted, not created twice |
| `status` shows `Failed` in validation or data processing | The files were refused. Read the message in `reports/frontier/nova_tuning.md` and report it; do not edit the files |
| A job must be stopped | `venv\Scripts\python -m scripts.nova_tune stop --model <model> --yes`. It cannot be resumed, and the tokens used so far are billed |
| `note: the loss curves could not be read` | The job is fine; the metrics files were not where expected. Look under the job's output location in S3 |

### 6.10 Teardown: leave nothing of Path B that bills

Do this when the comparison is finished (after req_005), or earlier for a model that is no
longer needed. Deleting a tuned model cannot be undone.

```powershell
venv\Scripts\python -m scripts.nova_tune teardown --model nova-lite --deployment --yes
venv\Scripts\python -m scripts.nova_tune teardown --model nova-lite --all --yes
venv\Scripts\python -m scripts.nova_tune teardown --model nova-pro --all --yes
aws s3 rm s3://qlora-sagemaker-123456789012/data/bedrock/ --recursive --profile qlora
aws s3 rm s3://qlora-sagemaker-123456789012/bedrock/ --recursive --profile qlora
```

`--deployment` deletes the deployment and keeps the tuned model (it can be deployed again).
`--all` deletes the deployment and then the tuned model, which ends the $1.95 a month. Only
what the tuning record names is touched. The two `aws s3 rm` commands delete the staged
training files and the jobs' outputs; the tool never deletes from S3.

To confirm nothing is left:

```powershell
aws bedrock list-custom-model-deployments --region us-east-1 --profile qlora --query "modelDeploymentSummaries[].customModelDeploymentName"
aws bedrock list-custom-models --region us-east-1 --profile qlora --query "modelSummaries[].modelName"
aws bedrock list-provisioned-model-throughputs --region us-east-1 --profile qlora --query "provisionedModelSummaries[].provisionedModelName"
aws bedrock list-model-customization-jobs --region us-east-1 --profile qlora --status-equals InProgress --query "modelCustomizationJobSummaries[].jobName"
```

All four should print an empty list for this project's names (`qlora-finqa-...`). Finished
jobs stay listed in Bedrock; they cost nothing.

### 6.11 The tuned models on dev (req_004 v4.04)

Once a model's job has completed and its deployment is Active (6.9), the tuned model is
scored exactly as its baseline was. **These runs call Bedrock and are billed, so the owner
runs them.** The report at the end is free.

| Predictor | Arm | Asks |
|---|---|---|
| `nova-lite-tuned` | B1-Lite | The deployment of the latest `nova-lite` attempt in the tuning record |
| `nova-pro-tuned` | B1-Pro | The deployment of the latest `nova-pro` attempt |

The request is the untuned one with another model id: the deployment's ARN. Prompt,
decoding values, reply limit, and scoring are the same. Tokens are recorded under the base
model's id, which is what Bedrock bills a tuned model as.

**Before a run starts**, with nothing billed, the predictor refuses unless:

- the tuning record has a job for the model, and it completed;
- the tuned model has a deployment that is recorded as Active and was not deleted;
- the model was tuned on the export the gate record names, made for today's prompt and split;
- Bedrock, asked once, says the deployment is Active.

Each refusal names the command to run (`nova_tune status`, `nova_tune deploy`, or the
launch). Exit code 2, and no run is left behind.

```powershell
aws sso login --profile qlora
venv\Scripts\python -m scripts.evaluate --predictor nova-lite-tuned --split quick_dev --limit 5
venv\Scripts\python -m scripts.evaluate --predictor nova-lite-tuned --split quick_dev
venv\Scripts\python -m scripts.evaluate --predictor nova-lite-tuned --split dev
venv\Scripts\python -m scripts.evaluate --predictor nova-pro-tuned --split quick_dev --limit 5
venv\Scripts\python -m scripts.evaluate --predictor nova-pro-tuned --split quick_dev
venv\Scripts\python -m scripts.evaluate --predictor nova-pro-tuned --split dev
```

| Run | Questions | Cost at the baselines' token counts | Time |
|---|---:|---:|---|
| Nova Lite, quick-dev | 200 | $0.01 | about 2 minutes |
| Nova Lite, dev | 499 | $0.02 | about 5 minutes |
| Nova Pro, quick-dev | 200 | $0.13 | about 10 minutes |
| Nova Pro, dev | 499 | $0.32 | about 25 minutes |

Nova Pro is slow because this account allows a tuned Nova Pro deployment 20 requests a
minute (Nova Lite 200). A run that is throttled out prints its `--resume` command; the
resume is refused against any other deployment. Note the `run ...` id of each complete run.

**The report** pairs each tuned run with the baseline run of the same model and split:

```powershell
venv\Scripts\python -m scripts.nova_tuned --runs <four B1 run ids> --baselines <four B0 run ids>
```

It calls no AWS service. It prints checks V1 to V6, then per model the dev scores and the
verdict, and writes `reports/frontier/nova_tuned_report.md` and `.json` (commit both).

| Check | Passes when |
|---|---|
| V1 Complete | Every run named is a complete result |
| V2 Paired | Each tuned run has one baseline run of its model and split, with the same prompt, split, settings, reply limit, and examples in the same order |
| V3 The right model | Each tuned run asked the deployment of its model's latest, completed attempt |
| V4 The right files | That attempt was tuned on the export the gate record names, for today's prompt and split |
| V5 Usage adds up | Every run has Bedrock's token counts in the ledger, under the base model |
| V6 dev present | Every model scored has a dev pair |

**The verdict** comes from dev only (`config/frontier.yaml`, `tuned_evaluation`; the
owner's decision, 2026-11-18). A tuned model has **met** its threshold when its execution
accuracy is at least 10 points above its own baseline's **and** its format validity is at
least 98%. With 499 dev questions: Nova Lite needs 173 correct (its baseline had 123), Nova
Pro 240 (190), and each 490 well-formed replies. Otherwise it has **missed**, and v4.05
(one corrective iteration) is eligible for that model. Nothing starts by itself.

| Exit code | Meaning |
|---|---|
| 0 | The report was written and every model in it met its threshold |
| 1 | A check failed and nothing was written, or the report was written and a model missed |
| 2 | A config problem, or a run that cannot be read |

With runs for one model only the report is written and marked partial; milestone M6 needs
both. The report also gives, per pair, how many questions tuning fixed and how many it
broke with an exact sign test, the cost per example and per correct answer, and the cost
of the tuning job beside them (the billed amount once `nova_tune bill` has entered it,
otherwise the estimate, labelled).

| Message | What to do |
|---|---|
| `CONFIG ERROR: nova-lite has not been fine-tuned` | Nothing was launched for it: section 6.9 |
| `... is InProgress, not Completed; run python -m scripts.nova_tune status` | Wait for the job, then `status` |
| `... has no deployment` | `nova_tune deploy --model M --yes` (the owner) |
| `... is Creating according to Bedrock, not Active` | Wait, then `nova_tune status` |
| `... is not there according to Bedrock` | The deployment was deleted outside `nova_tune`: `teardown --deployment`, then `deploy` again |
| `... was tuned on export X, and the gate record names export Y` | The training files changed after the job. The tuned model belongs to the earlier export; restore the files or tune again |
| `FAIL  V3 ... did not ask the deployment of ...'s latest attempt` | The runs are of an earlier attempt. Run the tuned evaluations again |

The deployments are kept after this version (the owner's decision, 2026-11-18): the test
runs of req_005 use them.

### 6.12 Still to come

From later versions of req_004: at most one corrective job per model (v4.05), only for a
model whose verdict is "missed", and only if the owner starts it.

## 7. Comparison and report (req_005)

The cost model (v5.02) and the report (v5.03) are not written yet. This section covers
v5.01: freezing the configuration, unlocking the test split, and the six test runs.

### 7.1 What is fixed (req_005 v5.01)

- The test split (2,898 examples) is read **once per arm**, by six arms: A0, A1, B0-Lite,
  B1-Lite, B0-Pro, B1-Pro. They are listed in `config/comparison.yaml`.
- The order is: **freeze, commit, unlock, commit, check, run, record**. Each step refuses
  until the one before it is done.
- **The owner unlocks, and the owner starts every test run.** Claude runs `check`,
  `status`, and `plan`, and nothing else on the test split.
- After the unlock nothing frozen may change: the prompt, the split, the token budget, the
  DSL limits, the evaluation allowances, the gold flags, each predictor's settings, the
  selected adapter, the two deployments. A changed value makes every test run refuse.
- There is no command to lock again.

Files:

| File | What | Committed |
|---|---|---|
| `splits/config_freeze.json` | The freeze record the code reads | yes |
| `reports/comparison/config_freeze.md` | The same, for reading | yes |
| `splits/test_unlock.json` | The unlock record: names the freeze record and its commit | yes |
| `splits/test_run_voids.json` | Voided test runs with the owner's reasons (only if any) | yes |
| `reports/comparison/six_arm_test_runs.json`, `.md` | The six runs: ids, counts, tokens | yes |
| `artifacts/bundle_test/` | The test examples, for the two SageMaker jobs | no (dataset text) |

### 7.2 Freeze (free; no AWS, no sign-in)

```powershell
venv\Scripts\python -m scripts.comparison_freeze check
venv\Scripts\python -m scripts.comparison_freeze freeze
```

`check` prints checks F1 to F8 and the estimate of the six test passes, and writes nothing.
`freeze` writes the two freeze files and prints the record's SHA-256. The working tree must
be clean (F7), and the six dev result runs must be on this machine under `artifacts/runs/`.

| Check | Fails when | Then |
|---|---|---|
| F1 | No selection record, or it was made for another prompt, split, or model commit | `qlora_select`, section 5.26 |
| F2 | A tuned Nova model has no completed job or no Active deployment | `nova_tune status`, section 6.9 |
| F3 | A dev run's folder is not here, or the run is not a result any more | `sagemaker_job pull` for Path A; for Path B the run was made on this machine |
| F4 | The dev runs were made with different prompts, splits, or reply limits | Something changed between runs: the arm is run again on dev first |
| F5 | A setting, an adapter, or a deployment differs from the arm's dev run | Undo the change, or run the arm again on dev |
| F6 | The split is already unlocked, or a test run exists | Too late to freeze: see 7.8 |
| F7 | Uncommitted changes | Commit, then freeze |
| F8 | The estimate does not fit in a path's cap | Raise the cap in `config/` (the owner's decision), or do not run |

Read `reports/comparison/config_freeze.md`, then commit both files:

```powershell
git add splits/config_freeze.json reports/comparison/config_freeze.md
git commit -m "Configuration frozen for the test evaluation (req_005 v5.01)"
```

The freeze can be repeated as long as the split is locked. Each time the SHA-256 may change
(the estimate holds what was spent so far), and the newest committed record is the one to
unlock with.

### 7.3 Unlock (the owner, once)

```powershell
venv\Scripts\python -m scripts.comparison_freeze unlock --yes --freeze <first 12 characters of the SHA-256>
git add splits/test_unlock.json
git commit -m "Test split unlocked (req_005 v5.01)"
```

Typing the hash is the sign-off that the freeze record was read and accepted. The command
refuses unless the freeze record is committed, the tree is clean, and nothing frozen has
changed since. **Commit the unlock record before anything else**: a SageMaker job runs a
commit, and a measured run refuses uncommitted changes.

```powershell
venv\Scripts\python -m scripts.comparison_freeze status
```

Says whether the configuration is frozen, whether the split is unlocked, and lists every
frozen value that has changed (`CHANGED ...`, exit code 1).

### 7.4 The check after the unlock (free)

```powershell
venv\Scripts\python -m scripts.comparison_runs preflight
venv\Scripts\python -m scripts.finqa_split
```

`preflight` gives each test example its own gold reply through the runner: it must print
`gold on test: 2,898 of 2,898 right on execution` and `PREFLIGHT PASSED`. It also counts the
test sequences in Qwen tokens, which could not be done while the split was locked. A
sequence over the training length of 2,048 is reported, not refused; note the count for the
report. `finqa_split` writes the split report again, because its check P10 now reads
"unlocked"; commit it.

### 7.5 The four Nova runs (the owner; billed, about $3.90)

Sign in first (`aws sso login --profile qlora`). Before each run, a trial of five quick-dev
questions shows that the model answers today. **Never a trial on test**: `--limit` is
refused there.

```powershell
venv\Scripts\python -m scripts.evaluate --predictor nova-lite --split quick_dev --limit 5
venv\Scripts\python -m scripts.evaluate --predictor nova-lite --split test --scope measured
venv\Scripts\python -m scripts.evaluate --predictor nova-lite-tuned --split quick_dev --limit 5
venv\Scripts\python -m scripts.evaluate --predictor nova-lite-tuned --split test --scope measured
venv\Scripts\python -m scripts.evaluate --predictor nova-pro --split quick_dev --limit 5
venv\Scripts\python -m scripts.evaluate --predictor nova-pro --split test --scope measured
venv\Scripts\python -m scripts.evaluate --predictor nova-pro-tuned --split quick_dev --limit 5
venv\Scripts\python -m scripts.evaluate --predictor nova-pro-tuned --split test --scope measured
```

| Run | Time | Estimate |
|---|---|---:|
| `nova-lite`, `nova-lite-tuned` on test | about 30 minutes each | $0.14, $0.13 |
| `nova-pro`, `nova-pro-tuned` on test | about 2 hours 41 minutes each (18 requests a minute) | $1.87, $1.75 |

A trial on quick-dev changes the ledger, and a measured run refuses a dirty tree only
outside `ledger/`, so the trials need no commit. A run that stops (a throttle, a lost
connection, an expired sign-in) prints its `--resume` command: **always resume, never start
again**. A second start is refused and names the run to resume.

### 7.6 The two SageMaker jobs (the owner; billed, about $2.60)

```powershell
venv\Scripts\python -m scripts.sagemaker_job bundle --split test
venv\Scripts\python -m scripts.sagemaker_job plan evaluate --predictor qwen3-4b --split test --scope measured
venv\Scripts\python -m scripts.sagemaker_job launch evaluate --predictor qwen3-4b --split test --scope measured --yes
venv\Scripts\python -m scripts.sagemaker_job status
venv\Scripts\python -m scripts.sagemaker_job pull
git add reports/local_qlora ledger
git commit -m "Arm A0 scored on test"
venv\Scripts\python -m scripts.sagemaker_job plan evaluate --predictor qwen3-4b-tuned --adapter selected --workspace qlora-train-958bed10-a1 --split test --scope measured
venv\Scripts\python -m scripts.sagemaker_job launch evaluate --predictor qwen3-4b-tuned --adapter selected --workspace qlora-train-958bed10-a1 --split test --scope measured --yes
venv\Scripts\python -m scripts.sagemaker_job pull
```

- `bundle --split test` writes the test examples, and nothing else, to
  `artifacts/bundle_test/`. The first launch uploads it: the only upload of test rows to S3.
- The plan must pass J1 to J11. **J11** is the gate, asked from this machine: the arm is at
  its frozen configuration, has no test result, and no earlier test job of the arm is
  waiting to be pulled. **J3** names the test bundle: `test 2,898; ... test rows only`.
- Estimates: A0 about 1.1 hours and $1.54, A1 about 0.75 hours and $1.05; each at most
  $2.82 at the proposed limit of 7,200 seconds.
- **Commit the job record between the two launches.** A launch writes it, and the next
  launch refuses a dirty tree. The new commit also gives the second job its own name.
- The jobs can run while the Nova runs go: Bedrock and SageMaker share no limit. Only one
  SageMaker job runs at a time (J6).
- A job that stops at its limit: `pull`, then launch again with `--resume <run id>
  --workspace <that job's name> --max-runtime <more>`.

### 7.7 The six-arm record (free)

```powershell
venv\Scripts\python -m scripts.comparison_runs status
venv\Scripts\python -m scripts.comparison_runs record
git add reports/comparison ledger reports/local_qlora
git commit -m "Six arms scored on the test split (req_005 v5.01, M7)"
```

`status` lists each arm as `done`, `unfinished` (with the run to resume), or `missing`.
`record` runs checks C1 to C8 and writes `reports/comparison/six_arm_test_runs.json` and
`.md` only when all six arms pass. It holds counts, tokens, and billable seconds: no
statistics and no dollars. **Do not compare arms or read replies before it is written.**

| Check | Fails when |
|---|---|
| C1 | An arm has no complete result, or more than one |
| C2 | A run lacks a prediction for a test example |
| C3 | Runs were made with different prompts or splits, or two arms of one model were sent different numbers of input tokens |
| C4 | A run was not made at the frozen configuration, or a shared file changed afterwards |
| C5 | The unlock record is not committed, or a run's commit does not contain it |
| C6 | A1 did not ask the selected adapter, or a B1 did not ask the tuning record's deployment |
| C7 | A Nova prediction lacks AWS's token counts, or a Qwen job is not pulled |
| C8 | Another test run exists that is neither one of the six nor voided |

### 7.8 When a run is worthless: void it (the owner)

Only for a fault in the pipeline or the service: a deployment deleted mid-run, corrupted
files, a bug that made every reply unreadable. **Never because the answers disappoint.**

```powershell
venv\Scripts\python -m scripts.comparison_runs void --run <run id> --reason "<what went wrong>" --yes
git add splits/test_run_voids.json
git commit -m "Test run voided: <arm>"
```

The run and every run it continues are voided together; name the newest run of the chain.
The arm may then answer the test split once more. The reason is committed and appears in the
six-arm record under "Deviations", and the report will carry it.

### 7.9 Troubleshooting

| Message | Meaning |
|---|---|
| `LOCKED: the test split is locked` | Not unlocked yet: 7.2 and 7.3 |
| `REFUSED: ... there is no freeze record` | An unlock record exists that `comparison_freeze unlock` did not write. Remove it and start at 7.2 |
| `REFUSED: changed since the freeze: config/...` | A shared file changed. `git diff <freeze commit> -- config splits`, and undo it |
| `REFUSED: the predictor of arm ... is not the frozen one` | A setting, the adapter, or a deployment changed. `comparison_freeze status` names it |
| `REFUSED: add --scope measured` | A test run is a measurement |
| `REFUSED: ... inside a SageMaker job only` | A Qwen arm was started here: use `sagemaker_job launch evaluate` |
| `REFUSED: ... already has its test result` | The arm is done. A rerun needs a void (7.8) |
| `REFUSED: ... has an unfinished test run` | Add the `--resume` the message names |
| `a measured run cannot start from uncommitted changes` | Commit (the unlock record, the job record), then run |
| A deployment is not Active any more | `nova_tune status`; if it was deleted, `nova_tune deploy --model <name> --yes` creates another one, which is **not** the frozen deployment: the arm cannot be run without a spec change |

### 7.10 The cost model (req_005 v5.02; free, no AWS call, no sign-in)

```powershell
venv\Scripts\python -m scripts.comparison_cost check     # K1 to K8 and the figures; writes nothing
venv\Scripts\python -m scripts.comparison_cost report    # writes the record
```

`report` writes `reports/comparison/cost_model.json` and `.md`: what each tuned model cost
to make, what a question costs for each arm, what serving Qwen costs as a batch job and as
an always-on endpoint, the break-even volumes against each Nova model, the project's spend
per path and category, and the comparison with the AWS bill. Every figure is a quantity
from the ledger times a price from `config/pricing.yaml`. Settings are in
`config/cost_model.yaml`.

Exit codes: 0 done; 1 a check failed and nothing was written; 2 a config problem, a missing
record, or a refused bill entry.

**A record whose title says "provisional"** passed with a reservation: a figure is still an
estimate, or the bill has not been entered for the days of the last runs. It is complete
and usable; it is final once the reservations are gone.

**Entering the bill (the owner).** AWS bills by service and day, not by run, so only the
totals can be compared. Runs reach the bill about a day after they are made.

1. In the console: Billing and Cost Management, Bills, the month, "Charges by service".
2. Read the amount on the **Bedrock** row and on the **SageMaker** row.
3. Check that the last runs are on it. `comparison_cost check` prints what the model
   expects under K8 ("the model says $..."); a total below that means the bill is still
   catching up.
4. Enter each, with the last day (UTC) whose usage the total covers:

```powershell
venv\Scripts\python -m scripts.comparison_cost bill --service bedrock --usd <amount> --through <date>
venv\Scripts\python -m scripts.comparison_cost bill --service sagemaker --usd <amount> --through <date>
venv\Scripts\python -m scripts.comparison_cost report
git add reports/comparison ; git commit -m "Cost model reconciled with the bill"
```

`bill` appends to `reports/comparison/aws_bill.json` and never changes an earlier entry. An
entry agrees when it is within the larger of $0.50 and 2% of the billed amount. For Bedrock
the model's total includes storage: one model-month ($1.95) for each tuned model in each
calendar month it exists.

**A new price.** When AWS publishes a new price list, change the price and both dates in
`config/pricing.yaml`. The record shows the dates beside its figures. The hosting price of
`ml.g5.xlarge` (`sagemaker.hosting`) is read from the public price list, usage type
`USE1-Host:ml.g5.xlarge`; no sign-in is needed.

| Message | What to do |
|---|---|
| `FAIL  K1  ... there is no six-arm record` | Run `scripts.comparison_runs record` first (section 7.7) |
| `FAIL  K4  ... is not in the job record` | A SageMaker job's record is in the ledger but not in `reports/local_qlora/sagemaker_jobs.json`: restore that file from git |
| `FAIL  K4  ... nothing to price` or `unknown place` | A new kind of ledger record. It needs a branch in `cost.place` and a test; it is never skipped |
| `FAIL  K5  Totals agree` | A job was launched and not pulled: `sagemaker_job pull`. The cap check counts such a job at its maximum |
| `FAIL  K8  ... billed $X, the model $Y` | Read the bill's lines for that service and compare them with `ledger_summary --by arm,phase`. A charge the ledger does not know is the usual cause: something else in the account using the service, or a run that was not recorded |
| `CONFIG ERROR: --through ... is in the future` | Give the last day the total covers, today at the latest |

### 7.11 The comparison report (req_005 v5.03; free, no AWS call, no sign-in)

```powershell
venv\Scripts\python -m scripts.comparison_report check      # R1 to R8 and the main figures; writes nothing
venv\Scripts\python -m scripts.comparison_report build      # writes the record, the chart, the report, the README block
venv\Scripts\python -m scripts.comparison_report examples   # example text, to the git-ignored artifacts/ only
```

`build` reads the six test runs the six-arm record names, from their stored replies under
`artifacts/runs/`, and the cost record. It asks no model. It takes a few seconds and writes:

| File | What it is |
|---|---|
| `reports/comparison_report.md` | The report |
| `reports/comparison/comparison_statistics.json` and `.md` | Every interval, every pair's counts and p-values, the error categories, example ids |
| `reports/comparison/accuracy_cost.svg` | The chart |
| `README.md`, between `<!-- comparison:start -->` and `<!-- comparison:end -->` | The summary. Nothing outside the markers is touched |

The same replies and the same seed give the same bytes, so a second `build` changes nothing.
Commit all of them together.

Exit codes: 0 done; 1 a check failed and nothing was written; 2 a config problem or a
missing record.

**The report is generated: never edit it.** To change its wording, edit
`qlora_finetuning/comparison/report_template.md`; a figure in a sentence is a
`{{placeholder}}`, never a typed number. To add a deviation, add an entry to
`config/report_deviations.yaml`. Then build again. Settings (resamples, seed, the tested
pairs, colours) are in `config/report.yaml`.

**How to read the statistics.**

- An **interval** is a 95% bootstrap interval. *By example* resamples the questions; *by
  company* resamples whole companies and is wider, because questions about one company's
  report resemble each other.
- A **pair** is tested with McNemar's exact test on the questions that exactly one of the
  two arms got right. The p-values of the six tested pairs are adjusted together (Holm).
- A difference is **settled** when its adjusted p-value is below 0.05 and neither interval
  includes zero. Otherwise it is **not settled**, whatever its p-value.

**A report whose title says "provisional"** was built on a provisional cost record (the
bill is not entered for the days of the last runs). No accuracy figure depends on it. After
the bill is entered (section 7.10), run `comparison_cost report` and then
`comparison_report build` again; committing that report is milestone M8.

**`examples` writes dataset text.** `artifacts/comparison/error_examples.md` holds test
questions, gold programs, and replies, for the examples the statistics record names by id.
It is git-ignored. Do not copy it into a committed file.

| Message | What to do |
|---|---|
| `FAIL  R1  ... there is no cost record` | Run `scripts.comparison_cost report` first (section 7.10) |
| `FAIL  R1  ... built under another freeze record` | The cost record is older than the six-arm record: run `comparison_cost report` again |
| `FAIL  R2  ... cannot be read here` | A test run's folder is not under `artifacts/runs/` on this machine. For a Qwen arm, `sagemaker_job pull` its job again; a Nova run exists only on the machine that made it, so restore the folder from a backup |
| `FAIL  R3  Counts agree` | A reply scores differently today than when the six-arm record was written. A scoring file changed: see R4, and `comparison_freeze status` |
| `FAIL  R4  ... changed since the freeze` | Restore the named config file from git; nothing frozen may change |
| `FAIL  R7  ... priced another run` | The cost record and the six-arm record disagree: run `comparison_cost report` again |
| `FAIL  R8  ... the template names {{x}}, which has no value` | A placeholder in the template has no value in `report_tables.placeholders`: correct the name or add the value, with a test |
| `FAIL  R8  ... holds an AWS resource name` | A deployment name or an account number reached a written file. Name models by arm (`arm_labels` in `config/report.yaml`) |
| `CONFIG ERROR: the README has one comparison marker without the other` | Restore both marker comments in `README.md`, or remove the one that is left |

## 8. Runs and the cost ledger (req_000 v0.04)

Every training or evaluation run is registered and records what it consumed. The cost figures
in the final report are computed from these records.

### 8.1 Where things are

| What | Where | In git |
|---|---|---|
| Index of runs: one line at start, one at end | `ledger/runs.jsonl` | Yes |
| Cost ledger: one line per usage record | `ledger/ledger.jsonl` | Yes |
| A run's manifest, config copies, and outputs | `artifacts/runs/<run_id>/` | No |
| Paths and allowed values | `config/tracking.yaml` | Yes |

The two ledger files appear with the first real run.

### 8.2 Read the summary

```powershell
venv\Scripts\python -m scripts.ledger_summary
venv\Scripts\python -m scripts.ledger_summary --by arm,phase,where
venv\Scripts\python -m scripts.ledger_summary --json
```

It lists the runs, then the totals per group: wall-clock seconds, billable instance-seconds
(seconds times instances, which is what SageMaker charges for), tokens, and questions. The last
line says how many records are still estimates. It shows no dollars except amounts read from
the AWS bill; prices are applied by the cost model.

### 8.3 After a measured run

A run marked `measured` (its numbers go into the report) refuses to start while there are
uncommitted changes, so commit first. Afterwards commit `ledger/`, so the record of what it
used is in the history. Changes inside `ledger/` never block the next run.

### 8.4 Correct a record or add a billed amount

Lines are never edited. Append a record that names the one it replaces:

```powershell
venv\Scripts\python -c "from qlora_finetuning.tracking import Ledger, LedgerRecord; l = Ledger(); old = [r for r in l.read() if r.record_id.startswith('<first characters of the id>')][0]; l.append(LedgerRecord(run_id=old.run_id, phase=old.phase, path=old.path, arm=old.arm, where=old.where, model_id=old.model_id, training_tokens=<billed tokens>, billed_amount_usd='<amount, for example 72.96>', source='bill', source_detail='AWS bill <month>, Amazon Bedrock, model customization', replaces=old.record_id))"
```

This is how a fine-tuning estimate (`computed`) becomes a billed figure: AWS reports no token
count for a Bedrock fine-tuning job, so the first record is an estimate and the second, read
from the bill, replaces it. Find record ids with `--json`, or by opening `ledger/ledger.jsonl`.

### 8.5 Troubleshooting

| Symptom | Cause and fix |
|---|---|
| "a measured run cannot start from uncommitted changes" | Commit the listed files, or run with scope `dev` |
| "a measured run must trace to a commit" | The run is not in a git checkout. In a cloud job the launcher must set `QLORA_GIT_COMMIT` |
| `CONFIG ERROR: ledger.jsonl line N is not valid JSON` | The file was edited by hand or a write was cut short. Restore it from git; never repair a line in place |
| "is not an allowed value for arms" (or kinds, phases, places) | Use a value from `config/tracking.yaml`, or add the new value there |
| A run shows status `started` long after it should have ended | The process was killed before it could finish the run. Its manifest in `artifacts/runs/` shows how far it got |

## 9. Evaluation harness (req_002)

The code that reads a model's reply, runs its program, and decides whether the result is
right. It is the scorer for all six arms.
Nothing here needs the network, a GPU, AWS, or the training stack.

### 9.1 What is fixed

| Item | Where |
|---|---|
| The reply labels (`PROGRAM:`, `ANSWER:`) | `config/prompt.yaml` |
| The safety limits: reply length 4,000 characters, 32 steps | `config/dsl.yaml` |
| The ten operations and seventeen constants | `qlora_finetuning/dsl/language.py`; a test ties them to `config/data.yaml` and to the system text |
| When a number counts as right: absolute 0.00001 or relative 0.5%, for the executed value and for the answer line | `config/evaluation.yaml` |
| Which examples have a gold program that does not reproduce its gold answer (none today) | `splits/finqa_gold_flags.json` |
| The reply limit, 192 new tokens, passed to every predictor | `config/token_budget.yaml` |
| The tight rule shown beside the headline (relative 0.001%), and how often the runner logs progress | `config/evaluation.yaml` |

The limits are far above what a correct reply needs (145 characters, 6 steps). Raising them is
safe; they are not tuning knobs and do not change any score of a well-formed reply.

### 9.2 Try a reply by hand

```powershell
venv\Scripts\python -c "from qlora_finetuning.dsl import run_reply; print(run_reply('PROGRAM: subtract(920, 95), divide(#0, 95)\nANSWER: 8.68421'))"
```

The outcome shows the executed value, whether the reply kept the format exactly (`strict`),
and, when something failed, an `error_code` with its stage. For a table operation pass the
table as a list of rows as the second argument.

### 9.3 Check the harness against the gold programs

```powershell
venv\Scripts\python -m pytest tests\dsl -q -rs
```

With the snapshot present, `tests\dsl\test_real_data.py` parses and executes all 5,383 fit and
dev targets and compares the operation counts with the committed split report. It must run,
not skip, before any model is scored on this machine.

### 9.4 The gold self-check

Before any model is scored, the scorer is tried on the replies whose score is known: the
dataset's own. Needs the snapshot (section 4.2) and the split manifest (section 4.4); no
network, no model. It takes a few seconds.

```powershell
venv\Scripts\python -m scripts.finqa_gold_check           # check, then write the report and the flags
venv\Scripts\python -m scripts.finqa_gold_check --check   # write nothing; fail if a file is out of date
```

It builds each example's gold reply, runs it, and scores it against itself, for all 8,281
examples. Test rows are included: the script reads their gold programs and gold answers and
prints counts and example ids only. It writes three committed files:

| File | Content |
|---|---|
| `reports/data/finqa_gold_check_report.md` and `.json` | Counts per split, the checks, the settings, the flagged ids |
| `splits/finqa_gold_flags.json` | The flagged examples with their reasons; what scoring reads through `load_gold_flags()` |

| Check | Passes when |
|---|---|
| K1 Complete | Every example of the manifest was checked, once |
| K2 Gold replies are well formed | Every gold reply keeps the two-line format and its program is in canonical form |
| K3 Fit and dev reproduce | No fit or dev example is flagged |
| K4 Self-consistent | Every unflagged example is right on all four metrics |
| K5 Flags current | The flags file holds the same examples and settings as a fresh one |
| K6 Same data | The manifest describes the data on disk |
| K7 Deterministic | Checking twice gives the same result |

Exit codes: 0 every check passed (and, with `--check`, the files are current); 1 a check
failed or a file is out of date; 2 a config problem, no verified snapshot, no manifest, or
a damaged flags file.

Result on 2026-11-16: 7 of 7 checks pass; all 8,281 gold programs reproduce their gold
answer (fit 4,884, dev 499, test 2,898); no example is flagged.

**What a flagged example means.** Its gold program fails, or gives a value that does not
match its gold answer, so no reply can be scored right on execution for it. It is never
dropped: it stays in its split and in every total. Scoring reports the total over all
examples as the headline and the total over the unflagged ones beside it. A flag in the
test split is reported and leaves the exit code at 0; a flag in fit or dev fails K3,
because it would mean training on a target whose program does not give its answer.

**When to run it again.** After changing `config/evaluation.yaml`, the executor, the
prompt wording, or the data pin. If the flags or the settings changed, K5 fails once and
the files are rewritten: review the diff, commit it, and run again. Changing an allowance
after a model has been scored also means rescoring every arm.

### 9.5 Score a reply by hand

```powershell
venv\Scripts\python -c "from qlora_finetuning.dsl import ReplyFormat, run_reply; from qlora_finetuning.evaluation import Gold, load_evaluation_config, score_reply; f = ReplyFormat('PROGRAM:', 'ANSWER:'); g = Gold.from_target('PROGRAM: subtract(920, 95), divide(#0, 95)\nANSWER: 8.68421', f); print(score_reply(run_reply('PROGRAM: divide(825, 95)\nANSWER: 8.68'), g, load_evaluation_config()))"
```

The score shows the four results (`format_valid`, `execution_correct`, `answer_correct`,
`program_match`) and, for a wrong numeric result, a label: `scale` (off by 100, 1,000, or
1,000,000), `sign`, `kind` (a number where yes or no was due, or the reverse), or `other`.
Here the program is right but is not the gold program, so `program_match` is false.

### 9.6 Run an evaluation

One command evaluates any model on a split. Needs the snapshot and the manifest. With the
`gold` predictor it needs nothing else and takes about a second.

```powershell
venv\Scripts\python -m scripts.evaluate --predictor gold --split quick_dev
venv\Scripts\python -m scripts.evaluate --predictor gold --split dev
venv\Scripts\python -m scripts.evaluate --predictor gold --split dev --limit 20
venv\Scripts\python -m scripts.evaluate --predictor gold --split dev --resume <run_id>
venv\Scripts\python -m scripts.evaluate --summary <run_id>
```

| Option | Meaning |
|---|---|
| `--predictor NAME` | An entry of `PREDICTORS` in `scripts/evaluate.py`. `gold` is the fake predictor that gives each example its own gold reply; a model backend adds its own name (v3.01, v4.01) |
| `--split NAME` | `quick_dev`, `dev`, `fit`, or `test`. `test` is refused until req_005 v5.01 unlocks it |
| `--limit N` | The first N examples only: a **partial** run, for a first look. Never a result |
| `--resume RUN_ID` | Continue an interrupted run. A new run copies the earlier run's lines and goes on after them |
| `--summary RUN_ID` | Predict nothing; compute a run's totals again from its file |
| `--scope measured` | For a run whose numbers go into the report. Refuses to start from uncommitted changes. Default `dev` |
| `--label TEXT`, `--where PLACE` | The run's label; `local`, `sagemaker`, or `bedrock` |

Progress goes to standard error; the result goes to standard output: five check lines, the four
rates, execution accuracy under the tight rule (a diagnostic beside the 0.5% headline), tokens,
latency, stop reasons, and where the predictions file is.

| Check | Passes when |
|---|---|
| R1 Complete | Every example asked for has a line, in dataset order |
| R2 Same question | The lines were predicted from the current prompt wording and split manifest |
| R3 Scores current | Every stored score equals the score computed now |
| R4 Usage adds up | The ledger records of the runs involved cover the lines |
| R5 Nothing unscored | Every line has an outcome and a score |

The last line says what the run is: `RESULT` (complete, and a number that may be quoted),
`PARTIAL` (a limited run), or `NOT A RESULT` (a check failed or the run is unfinished).

Exit codes: 0 the checks pass (a complete run, or the partial run that was asked for); 1 a check
failed or the predictor failed; 2 a config problem, a locked split, an unknown predictor or run
id, or a resume that is refused.

**What a run writes.**

| File | Content | Committed |
|---|---|---|
| `artifacts/runs/<run_id>/predictions.jsonl` | One line per example: the reply, the outcome with its unrounded value, the score, token counts, latency | No |
| `artifacts/runs/<run_id>/evaluation.json`, `summary.json`, `manifest.json` | What was asked, the totals, and the run's manifest | No |
| `ledger/runs.jsonl`, `ledger/ledger.jsonl` | The run and one record of what it consumed (section 8) | Yes |
| `ledger/evaluations.jsonl` | One line of counts, only for a run that is a result | Yes |

Commit the `ledger/` changes after a run whose numbers matter.

**When a run is interrupted.** If the predictor cannot answer (an endpoint that throttles, a GPU
error), the run stops, keeps every line written so far, records what it consumed, and prints the
command that resumes it. No example is scored as wrong because of the failure. At most one batch is
predicted again. A resume is refused if the split, the prompt, the reply limit, or the predictor's
name, model, or settings differ: start a new run instead.

**After changing an allowance.** Run `--summary <run_id>`: the totals are computed again from the
stored values, with no model. Check R3 then fails, because the scores stored in the lines are the
old ones; that is the reminder that the run was scored under another rule when it was made.

Result on 2026-11-16 with the `gold` predictor: quick-dev 200 of 200 and dev 499 of 499 on all four
metrics, R1 to R5 pass. Milestone M3.

### 9.7 Error codes

| Stage | Codes |
|---|---|
| Reply | `reply_too_long`, `reply_empty`, `no_program` |
| Program | `empty_program`, `too_many_steps`, `syntax`, `unknown_operation`, `nested_call`, `wrong_arity`, `bad_argument`, `unknown_constant`, `bad_reference`, `bad_row_name` |
| Execution | `division_by_zero`, `numeric_error`, `yes_no_operand`, `table_missing`, `row_not_found`, `row_not_numeric` |

Codes are stored with predictions and are never renamed once a model has been scored.

### 9.8 Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `test_real_data.py` is skipped | No snapshot on this machine. Section 4.2 |
| The real-data test fails on the inventory comparison | The split report or the parser changed. Run `finqa_split --check` (section 4.4) first; the parser must agree with the report, not the other way round |
| Many replies end in `no_program` | The model is not writing the `PROGRAM:` label at the start of a line. Read a few raw replies; a backend may need to remove model-specific text before the parser |
| Many replies are not `strict` but execute | Expected for untuned models. Format validity and execution accuracy are reported separately |
| `row_not_found` on a row that looks right | The name must equal the row's first cell exactly, in the same case and spacing |
| `row_not_numeric` | The named row has a text cell, an empty cell, or a bracketed negative. This is the rule, not a bug |
| `CONFIG ERROR` naming `dsl.yaml` | A limit is missing or is not a positive whole number |
| `CONFIG ERROR` naming `evaluation.yaml` | An allowance is missing, negative, or not a number; both parts of one allowance are zero; or a scale factor is not above 1 |
| `finqa_gold_check` prints `FAIL  K5  Flags current` | The flagged examples or the settings differ from the committed flags file. If you changed an allowance, the executor, or the wording on purpose, review the rewritten files, commit them, and run again; otherwise undo the change |
| `finqa_gold_check` prints `FAIL  K3` | A fit or dev gold program no longer reproduces its answer. Nothing in the data changes by itself, so look at the last change to the executor or to `config/evaluation.yaml` |
| `finqa_gold_check --check` prints `OUT OF DATE` | A committed file differs from a fresh one. Run without `--check`, review the diff, commit |
| "the gold flags file ... is damaged or was edited by hand" | Restore `splits/finqa_gold_flags.json` from git, or delete it and run `finqa_gold_check` |
| "no gold flags file" when scoring | Run `finqa_gold_check` once; the file is committed, so this means a fresh clone is missing it or it was deleted |
| `evaluate` prints `PREDICTOR FAILED` | The backend could not answer. Fix the cause (sign in again, free the GPU, wait out the throttling) and run the `resume with:` command it printed |
| `evaluate` prints `REFUSED: run ... cannot be continued` | The settings differ from the interrupted run's, or its predictions file is damaged in the middle. The message names which. Start a new run without `--resume` |
| `evaluate` prints `LOCKED` | The test split was asked for. It is read once, by req_005 v5.01 |
| `FAIL  R3  Scores current` after `--summary` | `config/evaluation.yaml` changed since the run. The printed totals are under the current rule |
| `FAIL  R4  Usage adds up` | A run's usage is missing from `ledger/ledger.jsonl`: the file was edited, or not committed and lost. Restore it from git |
| `CONFIG ERROR: run ... has no evaluation.json` | The run id is wrong, the run is not an evaluation, or its folder under `artifacts/runs/` is not on this machine |
| `UNKNOWN PREDICTOR` | The name is not in `PREDICTORS` in `scripts/evaluate.py` |
| A model's answer of `8.4` is marked wrong against a gold of `0.084` | Intended: without a `%` sign there is no scale leniency. The `scale` label counts these |

## 10. The demo (req_006 v6.01)

One question to the tuned Qwen model, on this machine's GPU: the program it proposes, the
result of running that program in the project's executor, and whether the model's own
answer agrees with it. **It is a demonstration, not a measurement**: it scores nothing,
starts no run, writes no file, and calls no AWS service. It is free and takes seconds
once the model is loaded (about 10 seconds to load, 2 to 5 for an answer).

### 10.1 What it needs

| Need | How to get it |
|---|---|
| A CUDA GPU with room for the 4-bit model (its weights take 2.4 GiB), and the training stack | Section 2 (`env_check`) |
| The base model in the local cache, at the pinned commit | Section 5.2 |
| The selected checkpoint under `artifacts/runs/` | Section 5.27: `sagemaker_job pull --checkpoint 500` (needs a sign-in; reads only). On the owner's machine it is there |

### 10.2 Check without loading a model

```powershell
venv\Scripts\python -m qlora_finetuning.demo --check
```

Prints one `PASS` line for each input file and one for the adapter (its name and the
start of its hash), then `CHECK ONLY: no model was loaded`. It needs no GPU.

### 10.3 Ask

```powershell
venv\Scripts\python -m qlora_finetuning.demo                                  # the bundled sample
venv\Scripts\python -m qlora_finetuning.demo --input my_question.json
venv\Scripts\python -m qlora_finetuning.demo --input a.json b.json --json     # one JSON object per input
venv\Scripts\python -m qlora_finetuning.demo --show-input                     # with the context and the table
venv\Scripts\python -m qlora_finetuning.demo --untuned                        # the base model, for contrast
venv\Scripts\python -m qlora_finetuning.demo --adapter RUN_ID/checkpoint-N    # another adapter, by name
```

| Option | Meaning |
|---|---|
| `--input FILE [FILE ...]` | JSON input files. Without it, the sample `qlora_finetuning/demo/sample_input.json` |
| `--adapter NAME` | `selected` (the default: arm A1's checkpoint), a training run's id, or `RUN_ID/checkpoint-N`. Never a path |
| `--untuned` | The base model without an adapter. Not together with `--adapter` |
| `--json` | One line of JSON per input, with the whole outcome |
| `--show-input` | Also print the context and the table the model was given: `Context` and `Table` lines before the question, or `context` and `table` in the JSON. Without it the output is unchanged. No effect with `--check` |
| `--check` | Check the inputs and the adapter; load no model |

Exit codes: 0 every input's program ran; 1 at least one did not; 2 the demo stopped
before a reply (a bad input file, no adapter, no GPU, no weights, a model failure).
Results go to standard output; log lines and the loading bar go to standard error, so
`2>$null` leaves the result alone.

### 10.4 An input file

```json
{
  "context": ["One passage of the report.", "Another passage."],
  "table": [
    ["", "2024", "2023"],
    ["revenue", "$ 1,150", "$ 920"]
  ],
  "question": "what was the percentage change in revenue from 2023 to 2024?"
}
```

`table` is required: a list of rows, the first one the header, each cell text or a
number. `context` may be empty or left out. `expected` is allowed and ignored. Any other
field is refused, as is a file without a question. A table operation names a row by the
exact text of its first cell.

**Do not put FinQA rows into an input file.** The demo is for text of your own; a test
example belongs to the six test runs and nowhere else.

### 10.5 Reading the output

| Line | Meaning |
|---|---|
| `Model` | The predictor and the adapter's name, or `(untuned)` |
| `Program` | The program the model proposed, in canonical form |
| `Steps`, `Executed` | Each step's value and the last one, exactly as computed: not rounded, and a percentage is a fraction (0.1256, not 12.56) |
| `Model said` | The model's own answer line, and whether it agrees with the executed result under the answer allowance of `config/evaluation.yaml`. It says nothing about whether the program is the right one |
| `Format` | `exact`, or which format rule the reply broke |
| `Reply` | The raw reply, shown when the program did not run or the format was not exact |
| `Warning` | The reply was cut off at the limit, or the prompt is longer than anything the adapter was trained on (it is answered all the same; nothing is ever truncated) |

Output is ASCII: any other character appears as a backslash escape.

### 10.6 Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `CONFIG ERROR: the selected checkpoint (...) is not on this machine` | Run the pull command the message names, after `aws sso login --profile qlora` |
| `CONFIG ERROR: the weights of ... are not the ones the selection was made on` | The checkpoint folder was changed. Pull it again |
| `CONFIG ERROR: input NAME: ...` | The file and the field are named. Fix the file |
| `CONFIG ERROR` naming the weights, the GPU, or the training stack | Sections 2 and 5.2 |
| `MODEL ERROR: OutOfMemoryError ...` | Something else holds the GPU's memory, or the input is very long. Close what holds it (`nvidia-smi`) |
| `Executed    nothing: row_not_found` | The program names a row that is not in the table, letter for letter. Section 9.7 lists the codes |
| The same question gives another reply than in a test run | The demo asks one question at a time; the measured runs used batches of 8, and a reply can depend on its batch. Neither is wrong |
| The untuned model's program does not run | Expected: untuned, it got 27 of 2,898 test questions right |

## 11. The packages and their model cards (req_006 v6.02)

Two folders under git-ignored `artifacts/release/` that can be uploaded as they are:
`adapter/`, arm A1's adapter with its tokenizer, prompt, split manifest, training facts,
and model card; and `merged/`, the base model with that adapter merged into its weights,
with its own card. **Nothing here uploads anything, is billed, or calls AWS.** Publishing is
the owner's step (req_006 v6.03).

### 11.1 What it needs

| Need | How to get it |
|---|---|
| The selected checkpoint under `artifacts/runs/` | Section 5.27: `sagemaker_job pull --checkpoint 500` |
| The base model in the local cache, at the pinned commit | Section 5.2 |
| The training stack and a CUDA GPU | Section 3 (`env_check`). The merge itself runs on the CPU and needs about 16 GB of memory |
| The comparison's records, passed and not provisional | Section 7 |
| About 9 GB of disk for the merged copy | |

### 11.2 The steps, in order

```powershell
venv\Scripts\python -m scripts.hf_package check      # what a build needs; writes nothing
venv\Scripts\python -m scripts.hf_package adapter    # seconds
venv\Scripts\python -m scripts.hf_package merge      # about a minute, on the CPU
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b-tuned --adapter selected --split dev   # R1
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b-package --split dev                    # R2
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b-merged --split dev                     # R3
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b-merged-full --split dev                # R4
venv\Scripts\python -m scripts.hf_package verify     # runs each card's code on the GPU, about 3 minutes
venv\Scripts\python -m scripts.hf_package cards      # writes the cards and the record
venv\Scripts\python -m scripts.hf_package status     # checks P1 to P10; writes nothing
```

Each dev run takes 4 to 7 minutes on the laptop GPU and is free. Run them one at a time:
each loads a model. The full-precision run (R4) needs about 10 GB of GPU memory.

| Run | What it is | Must hold |
|---|---|---|
| R1 | The selected adapter, loaded by name | Within `noise_examples` (10) of the recorded 358 of 499 |
| R2 | The adapter package, loaded from its folder | **The same reply as R1 on every example** |
| R3 | The merged copy in 4-bit | Fewer than 10 examples below R1 |
| R4 | The merged copy at full precision | Fewer than 10 examples below R1 |

None of the four is an arm, none has a price, and none may be run on test: the three new
predictors refuse the test split, and so does the gate.

### 11.3 The licences, and why `cards` may write drafts

`config/packaging.yaml` lists three licences (`licences.entries`): what each source said and
on which day it was read. A card is written to its package and to `reports/packaging/` only
when the owner has confirmed every entry (`confirmed_on`, a day no more than 30 days back).
Until then `cards` writes the two cards under `artifacts/release/drafts/` for reading, and
check P7 fails. To confirm: read the three sources again, correct an entry if its source
now says something else, set `confirmed_on`, and run `cards` again.

### 11.4 The checks

| Check | Passes when |
|---|---|
| P1 | The six-arm record, the cost record, and the comparison's record passed; none is provisional |
| P2, P3 | The checkpoint on this machine is arm A1, for today's model commit, prompt, and split |
| P4 | Each folder holds its files and no other; the adapter files are the checkpoint's bytes |
| P5 | No text file holds an account number, ARN, S3 path, job name, key, token, email address, or machine path |
| P6 | The tokenizer files are the base model's and give the pinned fingerprint |
| P7 | Every licence entry is confirmed by the owner |
| P8 | R1 and R2 exist; R2 equals R1; the adapter card's code ran to the sample's expected result |
| P9 | R3 and R4 exist and are inside the allowance; the merged card's code ran both ways |
| P10 | Each card on disk is its template filled from today's records |

`status` ends with two lines: `adapter package: ready` or `not ready`, and
`merged copy: ready`, `withheld`, `not ready`, or `not built`. A withheld merged copy gets
no card and does not hold the adapter back.

### 11.5 What is written

| File | In git | Holds |
|---|---|---|
| `artifacts/release/adapter/`, `artifacts/release/merged/` | no | The two packages |
| `artifacts/release/card_runs.json` | no | What running the cards' code gave |
| `artifacts/release/drafts/` | no | Cards built before the licences are confirmed |
| `reports/packaging/package_record.json`, `.md` | yes | Files with sizes and hashes, the runs, the comparisons, the checks, the licences |
| `reports/packaging/adapter_card.md`, `merged_card.md` | yes | The cards, as built |

**Never edit a card or the record.** Wording is in `qlora_finetuning/packaging/adapter_card.md`
and `merged_card.md`; the code a card prints is in `adapter_usage.py.txt` and
`merged_usage.py.txt`. After changing either, run `verify` and `cards` again.

### 11.6 After something changed

| What changed | Run again |
|---|---|
| A card template's wording | `cards` |
| A card's usage code | `verify`, then `cards` |
| The comparison's records (a rebuild of the report or the cost record) | `cards` |
| The repository names or the author in `config/packaging.yaml` | `adapter` is not needed; `merge` (its notice names the adapter), R3, R4, `verify`, `cards` |
| The merged copy was rebuilt | R3 and R4 (its weights are named in a run's settings), `verify`, `cards` |
| A licence entry | `cards` |

### 11.7 Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `CONFIG ERROR: ... not the ones the selection was made on` | The checkpoint folder was changed. Pull it again (section 5.27) |
| `FAIL  P8 ... R2 differs from R1 on N replies` | The package does not hold the adapter that R1 loaded, or one of the two runs was made under other settings. Run `adapter` again, then R2 |
| `FAIL  P8 ... no complete local run` | The run was not made here, was cut short, or was made before the package was rebuilt. Make it again |
| `FAIL  P9 ... R3 has N fewer right than R1` | The merged copy is withheld. This is a result, not a fault; the adapter is published without it |
| `the card's code changed since it was run` | `verify` |
| `the card's code failed (exit code 1): ...` | The last lines of its output follow. Usually the GPU's memory is held by another process (`nvidia-smi`) |
| `FAIL  P5 ... holds an S3 path` | A file of the package names something private. Find where the text came from; never remove it from the package by hand |
| The merge runs out of memory | It needs about 16 GB of free memory. Close what holds it |
| `the template names {{...}}, which has no value` | A record or a run the card quotes is missing; the earlier checks say which |

## 12. The public copy of the repository (req_006 v6.04)

The repository on GitHub is private. A second, public one,
`SudiptoHaldar/qlora_finetuning_public`, is made from it by export: a commit's files without
history, without the folders that are left out, with the AWS account number replaced by a
placeholder. **This repository is the source of truth. The public one is never edited by
hand.** The tool makes no commit, no push, and no network call.

### 12.1 When to export

After a commit here that is worth publishing: a release, a corrected report, a fixed README.
Each export becomes one commit in the public repository.

### 12.2 The steps

```powershell
venv\Scripts\python -m scripts.public_export plan                    # what would be copied; writes nothing
venv\Scripts\python -m scripts.public_export build --to <folder>     # seconds
venv\Scripts\python -m scripts.public_export check --to <folder>     # about 7 minutes: it runs the suite
```

`<folder>` is the working folder of the public repository, outside this one. It can be
named once in `.env` as `QLORA_PUBLIC_EXPORT_DIR`, and `--to` left out.

- `build` refuses while tracked files here have uncommitted changes: an export is of a
  commit. It writes into an empty folder (a `.git` may be there) or over a previous export.
  It removes only files that the previous export wrote and this commit no longer has.
- `check` runs checks E1 to E10 and writes `reports/public_export/export_record.json` and
  `.md` here. Commit the record afterwards. It is not part of a copy.

### 12.3 The owner's steps, after `check` passes

The first time, in the public folder:

```powershell
git init -b main
git config user.name "SudiptoHaldar"
git config user.email "<your GitHub noreply address>"
git add -A
git commit -m "Snapshot of the private repository at <short hash>"
git remote add origin https://github.com/SudiptoHaldar/qlora_finetuning_public.git
git push -u origin main
```

**Set the noreply address before the first commit.** Git on this machine is set to a
personal address, and a commit in a public repository shows its address to everyone. The
address is under GitHub, Settings, Emails. The setting is for that folder only.

Later exports, in the public folder: `git add -A`, `git commit -m "Snapshot of the private
repository at <short hash>"`, `git push`. `check` prints the short hash as part of E1.

Then, on GitHub while the repository is still private: see that the README renders, that
its two diagrams draw, and that the CI run passes. Then switch it to public.

### 12.4 What is left out and what is replaced

All of it is in `config/public_export.yaml`:

| | |
|---|---|
| Left out | `.claude/`, `track_progress/`, `docs/old_references/`, `administration/` but this runbook, the planning material and plan references under `reqspecs/0. PLANNING/`, `tests/test_process_scaffold.py`, the assistant instructions under `.github/`, `reports/public_export/` |
| Moved | This runbook, to `docs/runbook.md`; every mention of its old path is rewritten |
| Replaced | The AWS account number (read from `config/aws.yaml`) by `123456789012`; the private repository's name by the public one; a folder of the owner's machine by a neutral one |
| Generated | `PUBLIC_SNAPSHOT.md`, and one line under the title of the README and of the runbook |

**A new file that holds something private needs a rule here before it is committed**: a
folder to leave out, or a literal to replace. `check` fails on a string it knows to look
for (E3); it cannot know a secret nobody told it about.

### 12.5 The checks

| Check | Passes when |
|---|---|
| E1 | The copy names a commit of this repository and was built from a clean tree |
| E2 | The folder is, byte for byte, what that commit exports: nothing missing, extra, or edited |
| E3 | No file holds a replaced value, an access key, a token, an email address outside the allowed domains, or a path on someone's machine |
| E4 | Every replacement was made at least once |
| E5 | The moved files are in place; every link of the README leads to a file of the copy |
| E6 | `PUBLIC_SNAPSHOT.md` names the commit and the README points to it |
| E7 | Lint and the test suite pass in a temporary copy of the folder, with the training stack blocked as in CI |
| E8 | The tests skipped there are exactly the ones marked `private_records` |
| E9 | No tracked file here has an uncommitted change, the export record aside |
| E10 | No file over 5 MB; no archive, PDF, or weight file |

### 12.6 The freeze in the public copy

The freeze record names AWS resources, so its content changes with the placeholder and its
hash is no longer the one the unlock record names. In the copy, the tests that verify those
records (`@pytest.mark.private_records`) are skipped, and the gate refuses any test run.
The freeze can be verified here only. `PUBLIC_SNAPSHOT.md` says so.

### 12.7 Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `CONFIG ERROR: tracked files have uncommitted changes` | Commit them, or put them aside (`git stash`), and build again |
| `... is not empty and is not a previous export` | The folder holds something an export did not write. Name another folder, or empty this one yourself. The tool will not |
| `FAIL  E2 ... not part of the export` | A file was added to the public folder by hand. Remove it, or add it here and export again |
| `FAIL  E2 ... changed since the export` | A file of the copy was edited. Build again; make the change here |
| `FAIL  E3 ... holds ...` | Something private got through. Add a rule to `config/public_export.yaml` (leave the file out, or replace the literal), commit, build, check. **Do not push the folder** |
| `FAIL  E7 ... the suite did not pass in the copy` | A test depends on something the copy does not have. Fix the test here. Only a test of the committed freeze records gets the marker |
| `FAIL  E8` | A test was marked that does not need it, or a marked one was not skipped. See section 12.6 |
| `FAIL  E9` | Work is in progress here. Commit it and export again, or check with a clean tree |
| CI fails in the public repository and passed here | Usually the ten-minute limit of the workflow, or Linux. Change it here and export again |
