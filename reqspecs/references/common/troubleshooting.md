# Troubleshooting

**Last verified**: 2026-11-14  
**Sources**: problems met while building req_000 v0.05 and v0.01, plus the AWS CLI's standard
sign-in messages.

## Local environment

| Symptom | Cause | Fix |
|---|---|---|
| `No module named pip`, or every import fails | `venv/` is incomplete. It was once copied from another project without its packages | `py -3.13 -m venv venv`, then `venv\Scripts\python -m pip install -r requirements.txt` |
| `venv\Scripts\` holds launchers such as `alembic.exe` or `uvicorn.exe` | Left over from the copied venv. They do nothing | Optional: delete `venv/` and recreate it as above |
| pytest reports tests as "deselected" | Tests marked `network` are skipped by default | Run them on purpose: `venv\Scripts\python -m pytest -m network -q` |
| "Run this script as a module from the repository root" | A script was started by file path | `venv\Scripts\python -m scripts.<name>` from the repository root |
| `ConfigError: config file needs an integer 'version' key` | A file under `config/` has no `version:` line, or its value is quoted | Add `version: 1` |
| ruff `E501` line too long | The line is over 100 columns | `venv\Scripts\python -m ruff format <file>`, or wrap it by hand |
| CI fails but the same command passes locally | The test depends on something only this machine has (`.env`, the `qlora` profile, a Windows path) or on file-name case | `gh run view <run-id> --log-failed`, then remove the dependency |
| ruff `UP035` on `from typing import Callable` | ruff's default rules prefer `collections.abc` | `from collections.abc import Callable` |
| `env_check` E2 lists packages as not installed | The training stack is missing from this venv | Runbook section 3.1: torch from the `cu126` index first, then `requirements-train.txt` |
| `env_check` E3: torch "is not the cu126 build" | A CPU build was installed because the two install commands were run out of order | Run the torch command, then the requirements command |
| A test fails in CI with `No module named 'torch'` | A module or test imports the training stack at module level | Move the import inside the function; mark a test that needs the real library `gpu` |
| A logging test finds a handler it did not add | An earlier test ran a script's `main()`, which calls `setup_logging()` | The logging fixture removes the handler before and after each test |
| `finqa_report` or a data loader says the FinQA snapshot "is not valid" | The dataset files are not on this machine, or one changed | `venv\Scripts\python -m scripts.finqa_fetch` (runbook section 4.2) |
| `finqa_fetch` says `huggingface_hub is not installed` | The download uses a library from the training stack | Runbook section 3.1 |
| `finqa_report --check` prints `OUT OF DATE` | The committed schema report differs from a fresh one | Rebuild with `venv\Scripts\python -m scripts.finqa_report` and review the difference |
| A test fails in CI with `No module named 'pyarrow'` | `requirements.txt` lost its `pyarrow` line | Restore it; the data reader and its tests need it |
| `SplitError: ... exists and differs from the split these settings produce` | `config/split.yaml` or the data changed after the manifest was written | Restore the settings; the split is frozen (runbook section 4.4) |
| `TestLockedError` | Code asked for the test split | Use `fit`, `dev`, or `quick_dev`; test is read once, in req_005 v5.01 |
| `SplitError: the split manifest does not match the snapshot` | The data on disk is not the data the split was built from | `venv\Scripts\python -m scripts.finqa_fetch --verify-only`, then `venv\Scripts\python -m scripts.finqa_split --check` |
| A test fails comparing the prompt id, or `finqa_serialize` prints `FAIL  C9  Digests` | `config/prompt.yaml` changed | If intended, follow "Changing the wording" in runbook section 4.5; otherwise restore the file |
| `env_check --smoke` reports a loss of `nan` after the prompt grew | The smoke example no longer fits `smoke.max_seq_length`, so the target was cut off | Raise it in `config/local_env.yaml` |
| `PermissionError: [WinError 5] Access is denied` while a run manifest is written | Another program held the file for a moment | The registry now retries by itself. If it still fails after eight attempts, something is holding the file open: close editors and viewers on `artifacts/runs/` and run again |
| `finqa_tokens` cannot load the tokenizer | The files are not in the local Hugging Face cache, or the training stack is missing | `venv\Scripts\python -m scripts.finqa_tokens --download`; for the stack, runbook section 3.1 |

## AWS sign-in

| Symptom | Cause | Fix |
|---|---|---|
| `aws_check` prints `SIGN-IN REQUIRED` | The SSO session expired or was never started | `aws sso login --profile qlora` |
| `Error loading SSO Token` or `Token has expired and refresh failed` from the AWS CLI | Same cause | Same fix |
| `The config profile (qlora) could not be found` | The profile block is missing from `~/.aws/config` | Add it; its contents are in section 1 of `administration/aws_commands_run.md` |

## AWS checks and resources

| Symptom | Cause | Fix |
|---|---|---|
| `get-bucket-location` returns `null` | Normal for us-east-1 | Nothing to fix |
| `aws_check` C5 says a quota was not found | AWS changed the quota's name | Compare `quota.usage` in `config/aws.yaml` with `aws service-quotas list-service-quotas --service-code sagemaker` |
| `aws_check` C7 fails after the budget cap is changed | The config still holds the old cap | Update `budget.cap_usd` in `config/aws.yaml` |
| `create-budget` is rejected with "EMAIL address is invalid" | The alert file still holds the placeholder address | Put the real address in the temporary file (never in a committed file) and run again |
| `aws_check` C1 reports a different account | The profile points at another account | Check `sso_account_id` in `~/.aws/config` |
