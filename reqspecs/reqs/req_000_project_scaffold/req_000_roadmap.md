# req_000 Project Scaffold — Roadmap

**Spec group**: req_000_project_scaffold  
**Last updated**: 2026-11-15 — every version in this group is ✅ Complete. Later the same day a defect in v0.04 was fixed: the manifest write now retries when Windows briefly refuses to replace the file. Milestones M0, M1, and M1b reached; phase P0 done.  

Version log:

- **2026-11-13** — group created from `reqspecs/0. PLANNING/execution_plan.md` §2; `req_000_01_spec.md` is a placeholder.
- **2026-11-14** — plan revised before any version was specced: measured Qwen runs move to AWS SageMaker; the hosted path is Amazon Nova Pro and Nova Lite on Bedrock (OpenAI dropped); Q5 and Q6 resolved. Version scopes updated in place.
- **2026-11-14** — v0.05 operator part done ahead of v0.02: profile `qlora`, bucket, SageMaker and Bedrock roles, and the $300 budget alarm created by the user and verified read-only; GPU quota and Nova fine-tuning access were already in place, so no quota request was filed. Q2 resolved ($300, alerts at 80% and 100%). `req_000_05_spec.md` written; commands logged in `administration/aws_commands_run.md`.
- **2026-11-14** — v0.05 PRP written (`prp_000_aws_foundation_v5.md`) and executed the same day, ahead of v0.02 at the owner's request. Built self-contained: `config/aws.yaml`, `scripts/aws_check.py` with its own YAML loader, 44 unit tests plus one live test behind `@pytest.mark.network`, runbook AWS section with teardown, Bedrock policy documents saved. `aws_check` passes C1–C7 live: milestone M1b reached. Found on the way: `venv/` was an empty copy from another project and was rebuilt in place. Carried to v0.02: swap the stand-in loader for `load_config()`, settle the config naming scheme, add the `gpu` and `paid` markers and the ruff configuration.
- **2026-11-14** — v0.05 "Depends on" set to none: as built it depends on nothing. The v0.02 row notes what v0.05 already put in place.
- **2026-11-14** — `req_000_01_spec.md` written for v0.01 (📝 Spec ready): eleven decisions locked in draft, pending the owner's review. Scope adjusted for what v0.05 already put in place (`boto3`, `ruff`, the `network` marker, the rebuilt venv, runbook section 1).
- **2026-11-14** — v0.01 PRP written (`prp_000_process_bootstrap_v1.md`) and executed. Five command files, three PRP templates, and `coding.md` retargeted; `alembic.ini`, `alembic/`, six pins, `DATABASE_URL`, and the `db` marker removed; `reqspecs/references/` (gotchas, troubleshooting, boto3) and `track_progress/` created; runbook stub sections added; `tests/test_process_scaffold.py` guards against inherited content returning. 65 tests pass. Milestone M0 reached. v0.02 is now unblocked.
- **2026-11-14** — `req_000_02_spec.md` written for v0.02 (📝 Spec ready). Agreed with the owner: plain config names with a `version:` key; `__version__` read from `release_number.md`; scripts run as `python -m scripts.<name>`; the AWS checks stay in `scripts/`; `ruff.toml` at 100 columns; one GitHub Actions workflow. The `protect-main` ruleset was verified as already active (blocks deletion and force push, requires a pull request, no required status check).
- **2026-11-14** — v0.02 PRP written (`prp_000_package_skeleton_v2.md`) and executed. Built: the `qlora_finetuning` package with `__version__`, `settings.py`, `logging_setup.py`, and eight sub-package stubs; `aws_check` moved onto `load_config()` and run as a module; `gpu` and `paid` markers; `ruff.toml`; `.github/workflows/ci.yml`. 120 tests pass and lint is clean. Status 🟡: one owner step remains: open the pull request so CI runs (the live `aws_check` run passed 7/7 after the owner signed in again).
- **2026-11-15** — v0.02 ✅ Complete. The owner signed in again and the live `aws_check` passed 7/7 in module form (2026-11-14); pull request #1 was merged to `main`, and the `ci` workflow passed on the pull request and on `main`. v0.03, v0.04, and the first versions of req_001 and req_002 are unblocked.
- **2026-11-15** — `lint and test` added to the `protect-main` ruleset as a required status check, at the owner's request. A pull request to `main` can no longer merge with failing lint or tests. This closes the follow-up recorded in the v0.02 spec.
- **2026-11-15** — `req_000_03_spec.md` written for v0.03 (📝 Spec ready). Checked on the day: RTX 4090 Laptop GPU, 16 GB, compute capability 8.9; no Linux distribution under WSL2; the whole training stack (torch 2.14.1 `cu126`, transformers 5.23.0, datasets 5.2.0, peft 0.21.3, trl 1.18.0, accelerate 1.15.0, bitsandbytes 0.50.2) resolves with Windows wheels for Python 3.13. Fifteen decisions drafted, pending the owner's review.
- **2026-11-15** — v0.03 PRP written (`prp_000_local_environment_v3.md`) and executed. The training stack is pinned in `requirements-train.txt` and installed; `scripts/env_check.py` passes E1–E6, and `--smoke` passes E7 on native Windows: `Qwen/Qwen3-0.6B` loads in 4-bit, answers without thinking, and takes one LoRA step with the 8-bit optimizer (loss 2.071, peak VRAM 1.2 GiB). Milestone M1 reached; Q3 resolved as native Windows. 189 unit tests pass, also with the training libraries blocked as in CI. ruff target moved to `py311`. Status 🟡 until CI passes on the pull request.
- **2026-11-15** — `req_000_04_spec.md` written for v0.04 (📝 Spec ready). Checked in the installed boto3 definitions: SageMaker reports `BillableTimeInSeconds` and Bedrock `Converse` reports token usage, but no API reports the tokens billed for a Bedrock fine-tuning job, so those are recorded as computed and later replaced from the bill. Proposed departure from the plan: the ledger and the runs index are committed in a top-level `ledger/` folder instead of living under git-ignored `artifacts/`. Fifteen decisions drafted, pending the owner's review.
- **2026-11-15** — v0.04 PRP written (`prp_000_run_registry_ledger_v4.md`) and executed. `qlora_finetuning/tracking/` now holds the run registry (`start_run`, `Run`, manifests, the runs index) and the cost ledger (`Ledger`, `LedgerRecord`, validation, replacing records, `summarize`, and two functions that turn a SageMaker job description or a Bedrock usage block into a record). `scripts/ledger_summary.py` prints runs and totals. The committed `ledger/` folder holds a README; its two files appear with the first real run. One refinement of the spec: the clean-tree check for measured runs ignores changes inside `ledger/`. 286 tests pass, also with the training libraries blocked. Status 🟡 until CI passes on the pull request.
- **2026-11-15** — v0.03 and v0.04 ✅ Complete. Pull request #2 (branch `setup/aws`, release v1.1) was merged to `main`; the `ci` workflow passed on the pull request and on `main` (385 passed, 1 skipped, lint clean). Every version of req_000 is complete, so phase P0 is done. One note carried forward: a tracking test failed once on Windows with `PermissionError` in `os.replace` and has not recurred (`gotchas.md` section 5).
- **2026-11-15** — Fix to v0.04, made while building req_001 v1.03. A tracking test had been failing now and then on this machine; it was traced to `os.replace` in `tracking/registry.py`, which Windows refuses with "Access is denied" when another program briefly holds a manifest that was just written (3 of 25 runs of the tracking tests). The same failure could have stopped a real run from recording its result. `_write_json` now tries the replace up to eight times with a short, growing wait; other errors are not retried. Three tests added; 0 failures in 40 runs afterwards. Also in v0.03's area: the `env_check` smoke example is now built by the req_001 serializer, and `smoke.max_seq_length` went from 256 to 1024 to fit the full system text; `env_check --smoke` passes 7/7.

This file is the **single source of truth** for the req_000 version sequence. The group owns
**the repository skeleton, the development-process scaffolding, the verified training environment, and the cross-cutting run registry and cost ledger. It extends nothing; every other group depends on it**.

## Why this group exists

The repository was cloned from an earlier project's scaffold. Its command files, PRP templates,
`requirements.txt`, `alembic.ini`, and `coding.md` still describe a different system (SEC EDGAR MCP
server, PostgreSQL, FastAPI). Nothing can be specced cleanly until that is retargeted, and no
training work is safe to plan until the stack is shown to run. Since 2026-11-14 the measured
Path A runs happen on AWS (SageMaker training jobs), in the same account and region as Path B
(Amazon Nova on Bedrock, us-east-1), so both paths are costed in billed dollars. This machine
(Windows 11, RTX 4090 Laptop GPU with 16 GB, Python 3.13) is the development and smoke-test
environment only.

Standing posture: file-based storage only (Parquet, JSONL, YAML); tunable numbers in versioned
YAML under `config/`; tests run without GPU, network, or API keys. The run registry and cost
ledger live here because both training paths and the comparison depend on them.

**Status legend** (in lifecycle order):

- ⏸ **Conditional** — only starts when a specific trigger fires
- 📋 **Planned** — next-up; depends on a prior version landing
- 📝 **Spec ready** — req_000_NN_spec.md written + decisions locked; PRP pending
- 🛠 **PRP ready** — PRP authored + validated; execution pending
- 🚧 **In flight** — PRP being executed or impl in progress
- 🟡 **Operator-gated** — prepared; waiting on a step only the user performs
- ✅ **Complete** — spec + impl shipped
- 🗄 **Shelved** — parked indefinitely; revival trigger noted

---

## Quick-reference table (version sequence)

| # | Title | Status | Depends on | Trigger to start |
|---|---|---|---|---|
| **v0.01** | **Process bootstrap** — Retarget inherited command files and PRP templates (leftovers listed in `docs/old_references/stale_project_name_references.md` §3); remove the inherited database stack; references and tracking folders | ✅ Complete | — | — |
| **v0.02** | **Package skeleton, settings, CI** — `qlora_finetuning/__init__.py` with `__version__`; `settings.py` (`get_str`/`get_int`, `load_config()` over versioned `config/*.yaml`); logging; sub-package stubs (`tracking`, `data`, `dsl`, `evaluation`, `local_qlora`, `frontier`, `comparison`, `demo`); `tests/` mirror; pytest markers `gpu`, `network`, `paid`; `ruff`; minimal CPU-only GitHub Actions workflow | ✅ Complete | v0.01 | — |
| **v0.03** | **Local development environment and training stack** — `scripts/env_check.py` (GPU, VRAM, driver, compute capability, BF16, torch CUDA build); pinned `requirements-train.txt`; opt-in 4-bit LoRA smoke step | ✅ Complete | v0.02 | — |
| **v0.04** | **Run registry and cost ledger** — `tracking/`: run ids, `artifacts/runs/<run_id>/` layout, run manifest (config snapshot, git commit, split-manifest hash); append-only ledger of billed quantities; summary script | ✅ Complete | v0.02 | — |
| **v0.05** | **AWS foundation** — One region (us-east-1) for both paths. AWS resources created and verified 2026-11-14; `config/aws.yaml`, `scripts/aws_check.py` (7/7 live), and the runbook AWS section built the same day, ahead of v0.02 | ✅ Complete | — | — |

---

## Detail per version

### v0.01 — Process bootstrap  ✅

**Status**: ✅ Complete 2026-11-14 (spec `req_000_01_spec.md`, PRP `reqspecs/PRPs/prp_000_process_bootstrap_v1.md`, summary `reqspecs/impl_summary/impl_000_20261114_203322.md`)  
**Depends on**: —  

**Scope**: Retarget inherited command files and PRP templates (leftovers listed in `docs/old_references/stale_project_name_references.md` §3). Correct `coding.md` (plan path `reqspecs/0. PLANNING/`, module list, GPU/network/paid test rules). Create `reqspecs/references/` skeleton with `common/gotchas.md` seeded from both guides, `track_progress/wip/`, `track_progress/roadmap_views/`, runbook stub sections. Add `artifacts/` to `.gitignore`.

**Decisions and open questions**: Decided 2026-11-13: Q5, remove the inherited PostgreSQL/FastAPI/alembic files; Q6, keep `venv` + `requirements.txt` and rewrite the `uv run` commands to match. Locked in the spec draft (2026-11-14): commands written as `venv\Scripts\python -m <tool>`; 1000-line file guideline everywhere; `release_number.md` stays the only release number (whether `__version__` mirrors it is a v0.02 decision); the roll commands report milestones, AWS spend against the cap, and operator-gated steps; `pytest.ini` keeps only `network` until v0.02 adds `gpu` and `paid`.

**Built 2026-11-14**: everything in the scope above. Three things differ from the spec text: the acceptance scan matches `registry` and `pypi` only in their publication senses (the spec allowed this); the inherited example names in `generate-prp.md` were replaced with names from this repository; and the scan was added as a test (`tests/test_process_scaffold.py`). Left for the owner: stale launchers in `venv/Scripts/` (delete and recreate `venv/` to clear them).

### v0.02 — Package skeleton, settings, CI  ✅

**Status**: ✅ Complete 2026-11-15 (spec `req_000_02_spec.md`, PRP `reqspecs/PRPs/prp_000_package_skeleton_v2.md`, summary `reqspecs/impl_summary/impl_000_20261114_233357.md`). Built 2026-11-14; live `aws_check` 7/7; CI green on pull request #1 and on `main` (2026-11-15)  
**Depends on**: v0.01  

**Scope**: `qlora_finetuning/__init__.py` with `__version__`; `settings.py` (`get_str`/`get_int`, `load_config()` over versioned `config/*.yaml`); logging; sub-package stubs (`tracking`, `data`, `dsl`, `evaluation`, `local_qlora`, `frontier`, `comparison`, `demo`); `tests/` mirror; pytest markers `gpu`, `network`, `paid`; `ruff`; minimal CPU-only GitHub Actions workflow.

**Carried in from v0.05 (2026-11-14)**: `scripts/aws_check.py` has a stand-in `load_aws_config()`; replace it with `load_config()` and rename `config/aws.yaml` if the naming scheme requires. `pytest.ini` already registers `network` and deselects it by default; add `gpu` and `paid` and remove the inherited `db` marker. `requirements.txt` already lists `boto3==1.44.3` and `ruff>=0.17`; ruff still runs with default rules. `venv/` was rebuilt with only `boto3`, `PyYAML`, `pytest`, and `ruff` installed.

**Decisions and open questions**: Locked in the spec (2026-11-14): config files keep plain names with an integer `version:` key; `__version__` is read from `release_number.md`; scripts run as `venv\Scripts\python -m scripts.<name>` and no `pyproject.toml` is added; the AWS checks stay in `scripts/aws_check.py` with only the loader swapped; `ruff.toml` with line length 100 and rule sets E, W, F, I, UP, B; CI on `ubuntu-latest`, Python 3.13, no secrets. The CI job was made a required check in `protect-main` on 2026-11-15. No open question.

**Built 2026-11-14**: everything in the spec's scope. Two additions: a script started by file path prints a one-line hint instead of a traceback, and `pytest.ini` sets `--strict-markers`. Existing Python files were passed through `ruff format` to meet the 100-column limit.

### v0.03 — Local development environment and training stack  ✅

**Status**: ✅ Complete 2026-11-15 (spec `req_000_03_spec.md`, PRP `reqspecs/PRPs/prp_000_local_environment_v3.md`, summary `reqspecs/impl_summary/impl_000_20261115_005208.md`). Built and passing on this machine, including the smoke; CI green on pull request #2 and on `main` (2026-11-15)  
**Depends on**: v0.02  

**Scope**: `scripts/env_check.py` (GPU, VRAM, driver, compute capability, BF16, torch CUDA build). CUDA torch install kept out of `requirements.txt`: `requirements-train.txt` plus a runbook step. Pin `torch`, `transformers`, `datasets`, `peft`, `trl`, `accelerate`, `bitsandbytes`. Smoke gate: load a small Qwen3 model in 4-bit NF4 and run one LoRA training step.

**Decisions and open questions**: Locked in the spec draft (2026-11-15): native Windows first, and if the smoke fails the work stops for the owner to choose a fallback; exact pins in `requirements-train.txt` with torch from the `cu126` index installed by a runbook step; CI never installs the training stack; `env_check` follows the `aws_check` shape with an opt-in `--smoke` on `Qwen/Qwen3-0.6B`; ruff target moves to `py311`. **Q3 resolved 2026-11-15: native Windows** (the smoke passes). Earlier note: Q3 (native Windows vs WSL2), now lower stakes: this machine only runs development and smoke tests; measured runs use the Linux training image on SageMaker (v3.03). The same pinned versions must be used locally and in the cloud image. Reaching it is milestone M1.

### v0.04 — Run registry and cost ledger  ✅

**Status**: ✅ Complete 2026-11-15 (spec `req_000_04_spec.md`, PRP `reqspecs/PRPs/prp_000_run_registry_ledger_v4.md`, summary `reqspecs/impl_summary/impl_000_20261115_011500.md`). CI green on pull request #2 and on `main` (2026-11-15)  
**Depends on**: v0.02  

**Scope**: `tracking/`: run ids, `artifacts/runs/<run_id>/` layout, run manifest (config snapshot, git commit, split-manifest hash). Append-only JSONL ledger records for wall-clock, billed instance-seconds with instance type, token counts, and provider-billed amounts. Summary reader.

**Decisions and open questions**: Locked in the spec draft (2026-11-15): the ledger stores quantities, not dollars, and every record names its source (`measured`, `aws_api`, `computed`, `bill`); append-only with replacing records; SageMaker time comes from `DescribeTrainingJob.BillableTimeInSeconds` and Bedrock inference tokens from the `Converse` usage block; Bedrock fine-tuning tokens are not reported by any API, so they are computed and later replaced from the bill; `measured` runs refuse a working tree with uncommitted changes; no AWS call in this version. Committing `ledger/` (a departure from the plan's git-ignored `artifacts/`) was accepted by the owner on 2026-11-15 by proceeding to the PRP; `high_level_plan.md` section 5 now lists the folder.

### v0.05 — AWS foundation  ✅

**Status**: ✅ Complete 2026-11-14 (spec `req_000_05_spec.md`, PRP `reqspecs/PRPs/prp_000_aws_foundation_v5.md`, summary `reqspecs/impl_summary/impl_000_20261114_165800.md`)  
**Depends on**: — (the spec named v0.02; the version was built self-contained, so the dependency no longer applies)  

**Scope**: One region (us-east-1) for both paths. Named AWS CLI profile; S3 bucket with a fixed prefix layout for data, job outputs, and adapters; IAM execution roles for SageMaker and for Bedrock model customization; Service Quotas check for the SageMaker training instance types; AWS Budgets alarm at the agreed cap; `config/aws.yaml`; `scripts/aws_check.py` that verifies all of it read-only; runbook section.

**Done 2026-11-14 (operator part)**: profile `qlora` on account 123456789012 (velaHita staging, shared); bucket `qlora-sagemaker-123456789012`; roles `qlora-sagemaker-execution` and `qlora-bedrock-customization`; budget `qlora-finetuning` ($300, alerts at 80% and 100%, SageMaker plus Bedrock only). GPU training quota was already 1 for `ml.g5` and `ml.g6` (`xlarge` to `16xlarge`), and Nova Pro and Nova Lite were already fine-tunable, so no request was needed. Command log: `administration/aws_commands_run.md`.

**Built 2026-11-14 (code and documentation part)**: `config/aws.yaml`; `scripts/aws_check.py` (checks C1–C7, read-only, exit 0/1/2); `tests/scripts/test_aws_check.py` (44 unit tests with stubbed clients, one live test behind `@pytest.mark.network`); runbook AWS section with teardown; `administration/bedrock-trust.json` and `bedrock-s3-policy.json`. `aws_check` passes 7/7 against the live account, so milestone M1b is reached.

**Built ahead of v0.02**: the owner asked for the PRP before v0.02 existed, so the version is self-contained and no `qlora_finetuning/` package was created. The follow-up items are listed under v0.02.

**Decisions and open questions**: Q2 resolved ($300). Deviation accepted: the SageMaker role uses `AmazonSageMakerFullAccess`, not a least-privilege policy. Follow-ups recorded in the spec: the budget counter resets on 2027-01-01; the budget's service filter is to be confirmed after the first billed job. Q8 (SageMaker training jobs vs EC2) and Q9 (instance type) stay open for v3.03; neither blocks this version.
