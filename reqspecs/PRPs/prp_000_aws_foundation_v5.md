# PRP: AWS foundation (req_000 v0.05)

**Feature Name**: aws_foundation (project_scaffold v0.05)  
**Version**: 5  
**Created**: 2026-11-14  
**Status**: ✅ IMPLEMENTED  
**Implemented**: 2026-11-14  
**Implementation Summary**: [impl_000_20261114_165800.md](../impl_summary/impl_000_20261114_165800.md)  
**Confidence Score**: 9/10  
**Estimated Time**: 1.5–2 hours  
**Origin**: `reqspecs/reqs/req_000_project_scaffold/req_000_05_spec.md`  
**Template**: `prp_lean.md` (expanded, medium complexity)  

---

## Goal

Add the code and documentation half of the AWS foundation: `config/aws.yaml` naming the
already-created resources, a read-only `scripts/aws_check.py` that prints PASS or FAIL for
checks C1–C7 and exits non-zero on any failure, network-free unit tests, and a runbook "AWS"
section with teardown. The AWS resources themselves already exist and are not touched.

## Why & What

- **User value**: one command re-verifies the whole AWS setup before any billed job
  (milestone M1b), instead of ten hand-typed CLI checks.
- **Technical**: a YAML config, seven check functions that each take a boto3 client, a CLI
  wrapper, stub-client tests, two saved policy documents, runbook and roadmap updates.

## Success Criteria

- [ ] `venv\Scripts\python scripts\aws_check.py` prints PASS for C1–C7 against the live
      account and exits 0.
- [ ] Any failing check prints a FAIL line with the reason and the exit code is 1.
- [ ] An expired or missing sign-in prints one line telling the operator to run
      `aws sso login --profile qlora`, exits 2, and shows no stack trace.
- [ ] `venv\Scripts\python -m pytest -q` passes with no network and no credentials; the one
      live test is deselected by default and runs with `-m network`.
- [ ] `venv\Scripts\python -m ruff check .` reports no errors.
- [ ] The script makes only read calls (`Get*`, `List*`, `Describe*`) and no Cost Explorer call.
- [ ] `docs/runbook.md` has an AWS section whose teardown names both roles, the
      bucket, and the budget.
- [ ] No credential, access key, or email address is in any committed file; the script never
      prints a budget subscriber's address.

---

## Context

### The v0.02 dependency (read this first)

The spec depends on v0.02 (`qlora_finetuning` package, `settings.py`, `load_config()`, pytest
markers, ruff). **None of it exists yet**: there is no `qlora_finetuning/` package, `config/`
and `tests/` are empty apart from `tests/__init__.py`, and there is no `scripts/` folder. The
owner asked for this PRP ahead of v0.02, so v0.05 is built **self-contained**, with one seam:

| v0.02 item | What v0.05 does instead | What v0.02 changes later |
|---|---|---|
| `load_config()` | `load_aws_config(path)` inside `scripts/aws_check.py`, plain `yaml.safe_load` plus required-key validation | Replace the body with `load_config("aws")`; rename the file if a versioned naming scheme is locked |
| Config naming scheme | `config/aws.yaml` with a top-level `version: 1` | May rename |
| pytest markers | Register only `network` in `pytest.ini` and deselect it by default | Adds `gpu`, `paid`, removes inherited `db` |
| ruff | Default rules, no config file | Adds its configuration |
| Package skeleton | Not created. All logic lives in `scripts/aws_check.py` | May move the checks into the package |

Do **not** create `qlora_finetuning/` in this version.

### Environment facts (verified 2026-11-14)

- `venv/` is a broken shell: it was copied from another project, has no `Lib/site-packages`
  and no `pip`. Rebuild it in place with `py -3.13 -m venv venv` (no `--clear`), then install
  only what this version needs. `uv` exists on the machine but the project decision (Q6) is
  `venv` + `requirements.txt`; do not use `uv run`.
- Do not `pip install -r requirements.txt`: it still lists inherited PostgreSQL and FastAPI
  packages that v0.01 removes. Install `boto3==1.44.3 PyYAML pytest ruff` directly.
- Latest versions on PyPI today: boto3 1.44.3, ruff 0.17.5, pytest 9.2.0, PyYAML 6.0.3.
- AWS CLI 2.35.7 is installed and the `qlora` profile sign-in is currently valid.
- Python 3.13.7, Windows 11, PowerShell.

### References

```yaml
refs:   # reqspecs/references/ does not exist yet (v0.01 creates it); nothing local to read
  - url: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/budgets.html
    why: describe_budget, describe_notifications_for_budget, describe_subscribers_for_notification
  - url: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/service-quotas/paginator/ListServiceQuotas.html
    why: paginator shape for C5
  - url: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/list_foundation_models.html
    why: byCustomizationType filter for C6
  - url: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/error-handling.html
    why: ClientError codes and botocore exception classes

code:
  - file: administration/aws_commands_run.md
    why: the CLI equivalent of every check, with the observed results; also the log to update
  - file: reqspecs/reqs/req_000_project_scaffold/req_000_05_spec.md
    why: resource values (section 2), check table (4.2), acceptance (7)
```

### Live response shapes (captured read-only 2026-11-14)

```text
sts.get_caller_identity()                  -> {"Account": "123456789012", ...}
s3.get_bucket_location(Bucket)             -> {"LocationConstraint": None}      # None means us-east-1
s3.get_public_access_block(Bucket)         -> {"PublicAccessBlockConfiguration": {BlockPublicAcls, IgnorePublicAcls, BlockPublicPolicy, RestrictPublicBuckets: True}}
s3.get_bucket_encryption(Bucket)           -> {"ServerSideEncryptionConfiguration": {"Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]}}
iam.get_role(RoleName)                     -> {"Role": {"AssumeRolePolicyDocument": {...dict, already decoded...}}}
iam.get_role_policy(RoleName, PolicyName)  -> {"PolicyDocument": {...dict...}}
service-quotas list_service_quotas pages   -> {"Quotas": [{"QuotaName": "ml.g5.xlarge for training job usage", "Value": 1.0}, ...]}
bedrock.list_foundation_models(byCustomizationType="FINE_TUNING") -> {"modelSummaries": [{"modelId": "amazon.nova-pro-v1:0:300k"}, ...]}
budgets.describe_budget(AccountId, BudgetName) -> {"Budget": {"BudgetLimit": {"Amount": "300.0", "Unit": "USD"}, "TimeUnit": "ANNUALLY"}}
budgets.describe_notifications_for_budget  -> {"Notifications": [{"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN", "Threshold": 100.0}, {... 80.0}]}
budgets.describe_subscribers_for_notification(AccountId, BudgetName, Notification) -> {"Subscribers": [{"SubscriptionType": "EMAIL", "Address": "<never print>"}]}
```

Bedrock role, live trust policy (save as `administration/bedrock-trust.json`):

```json
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"bedrock.amazonaws.com"},"Action":"sts:AssumeRole","Condition":{"StringEquals":{"aws:SourceAccount":"123456789012"},"ArnLike":{"aws:SourceArn":"arn:aws:bedrock:us-east-1:123456789012:model-customization-job/*"}}}]}
```

Bedrock role, live inline policy `qlora-bedrock-s3` (save as `administration/bedrock-s3-policy.json`):

```json
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"s3:ListBucket","Resource":"arn:aws:s3:::qlora-sagemaker-123456789012"},{"Effect":"Allow","Action":"s3:GetObject","Resource":"arn:aws:s3:::qlora-sagemaker-123456789012/data/*"},{"Effect":"Allow","Action":["s3:GetObject","s3:PutObject"],"Resource":"arn:aws:s3:::qlora-sagemaker-123456789012/bedrock/*"}]}
```

### Critical Gotchas

- **Account id must be a quoted string in YAML.** Unquoted, `123456789012` loads as an int and
  never equals the string STS returns. The loader also coerces with `str()`.
- **us-east-1 reports its region as `None`** in `get_bucket_location`. Map `None` to `us-east-1`.
- **Quota names collide by substring.** `ml.g5.xlarge for spot training job usage` also contains
  "training job usage". Match the full name exactly: `f"{instance_type} for training job usage"`.
- **SageMaker has thousands of quotas.** Use the `list_service_quotas` paginator; a single call
  returns only the first page.
- **Policy fields are string or list.** `Action`, `Resource`, and `Principal.Service` each come
  back as a string when there is one value and a list when there are several. Normalize first.
- **Budget amount is the string `"300.0"`.** Compare as `Decimal`, not as text.
- **Notifications carry no `ThresholdType`** in the response. Pass each notification dict back
  to `describe_subscribers_for_notification` unchanged; do not add keys with `None` values.
- **Never print a subscriber address.** Report only the count (decision D9).
- **Expired sign-in surfaces in two ways**: botocore exceptions raised before any request
  (`UnauthorizedSSOTokenError`, `SSOTokenLoadError`, `TokenRetrievalError`,
  `NoCredentialsError`, `CredentialRetrievalError`; `ProfileNotFound` at session creation) and
  `ClientError` with code `ExpiredToken`, `ExpiredTokenException`, `RequestExpired`,
  `InvalidClientTokenId`, or `UnrecognizedClientException`.
- **Windows console is cp1252.** Print ASCII only (no arrows, check marks, or em dashes).
- **Import path for tests.** `tests/__init__.py` exists, so pytest puts the repo root on
  `sys.path`; with `scripts/__init__.py` present, `from scripts import aws_check` works.
- **`-m` in `addopts` is overridden** by a later `-m` on the command line, so
  `pytest -m network` runs the live test.

---

## Implementation

### Files

```yaml
CREATE:
  - config/aws.yaml: non-secret AWS names the checks compare against
  - scripts/__init__.py: empty; makes scripts importable from tests
  - scripts/aws_check.py: config loader, seven checks, runner, CLI (target under 350 lines)
  - tests/scripts/__init__.py: empty
  - tests/scripts/test_aws_check.py: stub-client unit tests plus one live test
  - administration/bedrock-trust.json: Bedrock role trust policy, as saved from the live role
  - administration/bedrock-s3-policy.json: Bedrock role inline policy, as saved from the live role
  - reqspecs/impl_summary/impl_000_<TIMESTAMP>.md: implementation summary (execute-prp step 7.1)

MODIFY:
  - requirements.txt: add "boto3==1.44.3" under a new "# AWS" heading; add "ruff>=0.17" under Testing
  - pytest.ini: register marker "network"; add addopts = -m "not network"
  - docs/runbook.md: add the AWS section (sign-in, aws_check, resources, teardown)
  - administration/aws_commands_run.md: log the read-only commands run for this PRP and the aws_check run
  - reqspecs/reqs/req_000_project_scaffold/req_000_roadmap.md: v0.05 status, detail, version log, header
  - reqspecs/0. PLANNING/execution_plan.md: v0.05 row in the status board and the M1b line
```

### `config/aws.yaml`

```yaml
version: 1
verified_on: "2026-11-14"
profile: qlora
account_id: "123456789012"          # keep quoted
region: us-east-1
bucket:
  name: qlora-sagemaker-123456789012
  prefixes: {data: data/, sagemaker: sagemaker/, bedrock: bedrock/, adapters: adapters/}
roles:
  sagemaker_execution:
    arn: arn:aws:iam::123456789012:role/qlora-sagemaker-execution
    trusted_service: sagemaker.amazonaws.com
  bedrock_customization:
    arn: arn:aws:iam::123456789012:role/qlora-bedrock-customization
    trusted_service: bedrock.amazonaws.com
    inline_policy: qlora-bedrock-s3
quota:
  service_code: sagemaker
  usage: training job usage
  minimum: 1
  instance_types: [ml.g5.xlarge, ml.g6.xlarge]
bedrock:
  fine_tunable_models: ["amazon.nova-pro-v1:0:300k", "amazon.nova-lite-v1:0:300k"]
budget:
  name: qlora-finetuning
  cap_usd: 300
```

### Tasks (~105 min total)

1. **Rebuild the venv and install tools** (~10 min): `py -3.13 -m venv venv`, then
   `venv\Scripts\python -m pip install boto3==1.44.3 PyYAML pytest ruff`.
2. **Config and requirements** (~10 min): create `config/aws.yaml`; edit `requirements.txt`
   and `pytest.ini`.
3. **`scripts/aws_check.py`** (~35 min): per the pseudocode below.
4. **Tests** (~25 min): per the test list below; run until green.
5. **Policy documents** (~3 min): write the two JSON files from the captured content above.
6. **Lint and unit tests** (~5 min): validation commands, fix findings.
7. **Live run** (~5 min): run the script against the account; run `pytest -m network`.
8. **Documentation** (~12 min): runbook AWS section, command log, roadmap, execution plan,
   implementation summary, PRP status.

### Pseudocode (`scripts/aws_check.py`)

```python
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "aws.yaml"
AUTH_ERROR_CODES = {"ExpiredToken", "ExpiredTokenException", "RequestExpired",
                    "InvalidClientTokenId", "UnrecognizedClientException"}
AUTH_EXCEPTIONS = (UnauthorizedSSOTokenError, SSOTokenLoadError, TokenRetrievalError,
                   NoCredentialsError, CredentialRetrievalError, ProfileNotFound)

class ConfigError(Exception): ...
class SignInError(Exception): ...          # carries the profile name

@dataclass(frozen=True)
class CheckResult: check_id: str; name: str; passed: bool; detail: str

def load_aws_config(path=DEFAULT_CONFIG) -> dict:
    # yaml.safe_load; raise ConfigError if the file is missing, is not a mapping, or lacks any
    # required dotted key (profile, account_id, region, bucket.name, bucket.prefixes.data,
    # bucket.prefixes.bedrock, both role arns and trusted_service, inline_policy,
    # quota.instance_types, quota.minimum, bedrock.fine_tunable_models, budget.name,
    # budget.cap_usd). Coerce account_id to str.
    # SEAM: v0.02 replaces this with settings.load_config("aws").

def _as_list(v): return [] if v is None else v if isinstance(v, list) else [v]
def _role_name(arn): return arn.rsplit("/", 1)[-1]
def _trusts(doc, service) -> bool:
    # any statement: Effect == "Allow", "sts:AssumeRole" in actions, service in Principal.Service
def _allows(doc, action, resource_arn) -> bool:
    # any Allow statement where some action pattern fnmatches `action` (case-insensitive)
    # and some resource pattern fnmatchcase-matches resource_arn

# Each check takes the one client it needs plus cfg, returns CheckResult, never raises for an
# expected AWS "not found" answer; those become FAIL with the error code in the detail.
def check_identity(sts, cfg)        # C1: Account == cfg["account_id"]
def check_bucket(s3, cfg)           # C2: location (None -> us-east-1) == region; all four
                                    #     blocks True; at least one rule with an SSEAlgorithm
def check_sagemaker_role(iam, cfg)  # C3: get_role; _trusts(doc, trusted_service)
def check_bedrock_role(iam, cfg)    # C4: get_role + _trusts; get_role_policy; _allows for
                                    #     s3:GetObject on <bucket>/<data>probe,
                                    #     s3:GetObject and s3:PutObject on <bucket>/<bedrock>probe
def check_gpu_quota(quotas, cfg)    # C5: paginate ListServiceQuotas into {name: value};
                                    #     every f"{t} for {usage}" present and >= minimum
def check_nova_models(bedrock, cfg) # C6: configured ids are a subset of FINE_TUNING modelIds
def check_budget(budgets, cfg)      # C7: Decimal(Amount) == Decimal(str(cap_usd)), Unit USD;
                                    #     at least one notification with >= 1 subscriber;
                                    #     detail reports counts only

def _guard(check_id, name, fn, client, cfg) -> CheckResult:
    try: return fn(client, cfg)
    except AUTH_EXCEPTIONS as e: raise SignInError(cfg["profile"]) from e
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in AUTH_ERROR_CODES: raise SignInError(cfg["profile"]) from e
        return CheckResult(check_id, name, False, f"{code}: {message}")
    except BotoCoreError as e: return CheckResult(check_id, name, False, str(e))

def run_checks(session, cfg) -> list[CheckResult]:
    # one client per service from session.client(name, config=Config(retries, timeouts));
    # budgets and iam are global services, the session region (us-east-1) is fine for both.
    # Runs C1..C7 in order through _guard. C1 failing does not stop the others.

def main(argv=None) -> int:
    # argparse: --config PATH, --profile NAME (overrides the config's profile)
    # ConfigError -> print "CONFIG ERROR: ...", return 2
    # boto3.Session(profile_name=..., region_name=...) inside the same try as run_checks
    # SignInError / AUTH_EXCEPTIONS -> print
    #   "SIGN-IN REQUIRED: run 'aws sso login --profile <profile>' and try again", return 2
    # print f"{'PASS' if r.passed else 'FAIL'}  {r.check_id}  {r.name}: {r.detail}" per result
    # print "N/7 checks passed"; return 0 if all passed else 1

if __name__ == "__main__": sys.exit(main())
```

### Tests (`tests/scripts/test_aws_check.py`)

Stub clients are `unittest.mock.MagicMock` objects with return values or
`side_effect=ClientError({"Error": {"Code": ..., "Message": ...}}, "Op")`. No test creates a
real boto3 client. A `cfg` fixture returns a plain dict equal in shape to `config/aws.yaml`.

| Area | Passing case | Failing cases |
|---|---|---|
| Config | The committed `config/aws.yaml` loads; `account_id` is a `str` | Missing file; missing key; non-mapping YAML. Guard: the committed file contains no `@` and no `AKIA` |
| C1 | Account matches | Account differs |
| C2 | `None` location, four blocks on, AES256 | Wrong region; one block off; `NoSuchPublicAccessBlockConfiguration`; `ServerSideEncryptionConfigurationNotFoundError`; `NoSuchBucket` |
| C3 | Trust allows the service (string and list principal) | Other service; `NoSuchEntity` |
| C4 | Live policy shape | Trust wrong; no write on `bedrock/*`; no read on `data/*`; `NoSuchEntity` on the policy |
| C5 | Both at 1.0 across two pages | One at 0; one missing; only the spot quota present |
| C6 | Both listed | One missing |
| C7 | Cap 300.0, one alert with a subscriber | Cap differs; no notifications; notification with no subscribers; `NotFoundException`. Passing detail must not contain the address |
| Guard and main | `main` returns 0 with a fake session, prints seven PASS lines | One FAIL gives exit 1; `UnauthorizedSSOTokenError` and `ClientError(ExpiredToken)` give exit 2 with the `aws sso login --profile qlora` message; bad config path gives exit 2 |
| Live | `@pytest.mark.network`: `main([])` returns 0 | — |

`main` tests replace `aws_check.boto3.Session` with a fake via `monkeypatch` and read output
with `capsys`.

### Error handling

| Situation | Behavior | Exit |
|---|---|---|
| All checks pass | Seven PASS lines, summary | 0 |
| A resource is missing or misconfigured | FAIL line with reason; other checks still run | 1 |
| A non-auth AWS or network error in one check | FAIL line with the error code or message | 1 |
| Sign-in expired, missing, or profile not found | One `SIGN-IN REQUIRED` line, no traceback | 2 |
| Config missing or invalid | One `CONFIG ERROR` line | 2 |

### Runbook AWS section (content to write)

1. Sign in: `aws sso login --profile qlora`; when it is needed.
2. One-time environment setup: rebuild the venv, install the four packages.
3. Run the check: `venv\Scripts\python scripts\aws_check.py`; what C1–C7 mean; exit codes.
4. What each resource is for (profile, bucket and four prefixes, two roles, quota, budget).
5. Teardown, in order, each a command run by the owner: delete the inline policy and the
   Bedrock role; detach `AmazonSageMakerFullAccess` and delete the SageMaker role; empty and
   delete the bucket; delete the budget. Note: the `qlora` profile entry in `~/.aws/config` is
   removed by hand; quota and Nova access are account properties and need no teardown; the
   account itself is shared with velaHita staging and is not closed.
6. Troubleshooting: SIGN-IN REQUIRED, CONFIG ERROR, a C7 failure after a cap change.

---

## Validation

```powershell
# Lint
venv\Scripts\python -m ruff check .

# Unit tests (no network, no credentials; the live test is deselected)
venv\Scripts\python -m pytest -q

# Live gate (needs a valid qlora sign-in; read-only, not billed)
venv\Scripts\python scripts\aws_check.py
venv\Scripts\python -m pytest -m network -q

# Secret scan of committed text
git grep -n -I -E "AKIA[0-9A-Z]{16}|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}" -- . ":!LICENSE" ":!reqspecs/PRPs/templates"
```

**Expected**: ruff clean; all unit tests pass; the script prints seven PASS lines and exits 0;
the secret scan returns nothing new (the `noreply`/`example.com` placeholders in inherited
command files are not secrets).

---

## Checklist

- [ ] Files created and modified as listed
- [ ] Every check has one passing and at least one failing test
- [ ] Lint clean, unit tests green, live run green
- [ ] The AWS command log lists every AWS command run during this work
- [ ] Roadmap and execution plan updated in the same turn
- [ ] Implementation summary created; PRP status set to implemented

---

## Implementation Time Estimate

**Total: 1.5–2 hours**

| Task | Time |
|---|---|
| Rebuild venv, install tools | 10 min |
| `config/aws.yaml`, `requirements.txt`, `pytest.ini` | 10 min |
| `scripts/aws_check.py` | 35 min |
| Unit tests | 25 min |
| Policy documents | 3 min |
| Lint, test, live run, fixes | 10–25 min |
| Runbook, command log, roadmap, execution plan, summaries | 12–15 min |

Risk: +15 min if a live response differs from the captured shapes (most likely the paginated
quota listing or the budget notification round trip).

## Confidence Score: 9/10

Every resource value and response shape was read from the live account today, the boto3 calls
are stable, and the checks are small pure functions. The one point held back is for the missing
v0.02 base: the venv must be rebuilt and the config loader is a stand-in.

## Time Taken for PRP Generation

**~14 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 3 min |
| Codebase exploration | 3 min |
| Research existing PRPs and patterns (none exist; live AWS shapes and package versions instead) | 3 min |
| Writing PRP document | 5 min |
