# Project Scaffold

Spec Group      :   req_000_project_scaffold  
Feature Name    :   project_scaffold  
Version         :   5  
Status          :   ✅ Complete 2026-11-14 (built ahead of v0.02, self-contained; see `reqspecs/PRPs/prp_000_aws_foundation_v5.md`)  
Last updated    :   2026-11-14  

## 1. v0.05 — AWS foundation

Both training paths run in one AWS account and one region, so they need a shared foundation: a
CLI profile, a bucket, two execution roles, GPU quota, and a budget alarm. This version has two
parts:

- **Operator part (done 2026-11-14).** The user created the resources; Claude prepared the
  commands and verified each one read-only. Every command is logged with a plain-English
  description in `administration/aws_commands_run.md`.
- **Code and documentation part (to build).** A config file that names the resources, a
  read-only `scripts/aws_check.py` that re-verifies them on demand, and a runbook section.

**Depends on**: v0.02 (package skeleton, `settings.py`, `load_config()`, pytest markers).

## 2. What exists in AWS (verified 2026-11-14)

| Item | Value |
|---|---|
| Account | 123456789012. This is the velaHita staging account, shared with that project |
| Region | us-east-1 |
| CLI profile | `qlora`: IAM Identity Center sign-in through the existing `velahita` SSO session, role `AdministratorAccess`. Refresh with `aws sso login --profile qlora` |
| S3 bucket | `qlora-sagemaker-123456789012`: us-east-1, all four public-access blocks on, AES256 default encryption |
| SageMaker execution role | `arn:aws:iam::123456789012:role/qlora-sagemaker-execution`: trusted by `sagemaker.amazonaws.com`; managed policy `AmazonSageMakerFullAccess` |
| Bedrock customization role | `arn:aws:iam::123456789012:role/qlora-bedrock-customization`: trusted by `bedrock.amazonaws.com` for model-customization jobs in this account and region only; inline policy `qlora-bedrock-s3` |
| GPU training quota | 1 on-demand training instance each for `ml.g5` and `ml.g6` at `xlarge`, `2xlarge`, `4xlarge`, `8xlarge`, `12xlarge`, `16xlarge`. Already in place; no request was filed |
| Nova fine-tuning | `amazon.nova-pro-v1:0:300k` and `amazon.nova-lite-v1:0:300k` are listed as fine-tunable for the account |
| Budget alarm | `qlora-finetuning`: $300, cost of Amazon SageMaker plus Amazon Bedrock only, email alerts at 80% and 100% of actual spend |

## 3. Locked decisions

| # | Decision | Reason |
|---|---|---|
| D1 | Use the existing velaHita staging account, not a new dedicated one | Owner's choice. GPU quota and Bedrock access were already in place, so nothing had to be requested |
| D2 | One bucket with a fixed prefix layout: `data/` (training and validation files), `sagemaker/` (job outputs), `bedrock/` (customization outputs), `adapters/` (selected LoRA adapters) | The Bedrock role's permissions are written against these prefixes |
| D3 | The bucket name contains "sagemaker" | `AmazonSageMakerFullAccess` only grants S3 access to buckets with that word in the name |
| D4 | The SageMaker role keeps `AmazonSageMakerFullAccess` | **Deviation** from the plan's "least privilege". Accepted for a single-operator project; a scoped policy can replace it later without changing any code |
| D5 | The Bedrock role is least privilege: list the bucket, read `data/*`, read and write `bedrock/*` | Matches the plan |
| D6 | Budget cap $300 (resolves Q2), alerts at 80% and 100% of actual spend, scoped to SageMaker and Bedrock by service filter | velaHita staging uses neither service, so the filter isolates this project. S3 is left out as negligible |
| D7 | The budget period is yearly | AWS Budgets has no whole-project period; monthly would reset the cap each month |
| D8 | No quota increase is requested | 1 instance of `ml.g5.xlarge` and of `ml.g6.xlarge` covers the plan. The choice between them stays with Q9 in v3.03 |
| D9 | The alert email address is not written to any committed file | Working agreement 15 (secrets and personal details stay out of the repo) |

## 4. Scope to build

### 4.1 `config/aws.yaml`

Holds the non-secret names the code needs: profile name, account id, region, bucket, the four
prefixes, both role ARNs, the instance types whose quota is checked (`ml.g5.xlarge`,
`ml.g6.xlarge`), the two Nova model ids, and the budget name and cap. The file name and
versioning follow whatever scheme v0.02 locks. No credentials and no email address.

### 4.2 `scripts/aws_check.py`

A read-only script, run by the operator, that prints one PASS or FAIL line per check and exits
non-zero if any check fails. It creates, changes, and deletes nothing, and makes no call that
is billed (so no Cost Explorer calls). It reads everything it compares against from
`config/aws.yaml`.

| # | Check | Passes when |
|---|---|---|
| C1 | Caller identity | The profile signs in and the account id matches config |
| C2 | Bucket | It exists, is in the configured region, has all four public-access blocks on, and has default encryption |
| C3 | SageMaker role | It exists and its trust policy allows `sagemaker.amazonaws.com` |
| C4 | Bedrock role | It exists, its trust policy allows `bedrock.amazonaws.com`, and its inline policy grants read on `data/*` and read and write on `bedrock/*` |
| C5 | GPU quota | The training-job quota is at least 1 for every configured instance type |
| C6 | Nova models | Both configured model ids are listed as fine-tunable |
| C7 | Budget | The named budget exists with the configured cap and at least one alert that has a subscriber |

An expired sign-in is reported as a clear message telling the operator to run
`aws sso login --profile qlora`, not as a stack trace. "Assumable by its service" is checked by
reading the trust policy; the script does not try to assume either role.

`boto3` is added to `requirements.txt` with a pinned version.

### 4.3 Tests

Unit tests run without network or credentials, per working agreement 9: each check is a
function that takes a client, and tests pass it a stubbed client covering one passing and at
least one failing case per check. A single live end-to-end run sits behind
`@pytest.mark.network` and is skipped by default.

### 4.4 Documentation

- `docs/runbook.md` gains an "AWS" section: how to sign in, how to run
  `aws_check`, what each resource is for, and teardown steps (delete the two roles, empty and
  delete the bucket, delete the budget).
- `administration/aws_commands_run.md` stays the command log and is updated whenever a new AWS
  command is run.
- The policy documents are kept under `administration/`: `trust.json` (SageMaker trust, already
  there) plus the Bedrock trust and S3 policy documents, saved from the live role.

## 5. Out of scope

- Starting any SageMaker or Bedrock job (v3.03, v4.01 onward).
- Reading billed seconds or tokens back from AWS (v0.04).
- Uploading any data. No test row is uploaded before the v5.01 unlock.
- Replacing `AmazonSageMakerFullAccess` with a scoped policy (see D4).

## 6. Known limits and follow-ups

- **The alarm only emails.** It does not stop jobs. Job runtime limits (v3.03) and the budget
  gate (v4.02) are what cap spend.
- **The budget follows the calendar year.** Its counter resets on 2027-01-01. If paid runs
  fall on both sides of that date the alarm sees each side separately; the ledger does not.
- **The service filter is unconfirmed.** It assumes Nova charges are billed under "Amazon
  Bedrock". After the first billed job, confirm the budget shows non-zero spend and correct the
  filter if it does not.
- **Shared account.** If velaHita staging starts using SageMaker or Bedrock, the budget counts
  that spend too and should be re-scoped by tag. The v5.02 reconciliation with the AWS bill
  must filter to this project's resources.

## 7. Acceptance

- `python scripts/aws_check.py` prints PASS for C1–C7 against the live account (milestone M1b).
- `pytest -q` and `ruff check .` pass with no network and no credentials.
- The runbook AWS section exists and its teardown steps name every resource in §2.
- No credential, access key, or email address appears in any committed file.

## 8. Next step

```
/generate-prp reqspecs/reqs/req_000_project_scaffold --version 5
```
