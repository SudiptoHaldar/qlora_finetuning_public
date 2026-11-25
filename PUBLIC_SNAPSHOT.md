# Public snapshot

This repository is a copy of a private one, written by `python -m scripts.public_export`. It is not
edited by hand: a change is made in the private repository and exported again.

| | |
|---|---|
| Exported commit | `45b2006886aee16458c99668110ef39a9ece16c9` of the private repository, committed 2026-11-25 |
| Files | 552 copied, 30 left out, 1 generated (this one) |

## How it differs from the private repository

- **No history.** Each commit here is one snapshot.
- **The AWS account number is a placeholder**, `123456789012`, in every file: configs, reports, records, the ledger, tests, and documents. It is the account number AWS uses in its own documentation. The bucket and the roles whose names are built on it are placeholders too.
- **The private repository's name** was replaced by `SudiptoHaldar/qlora_finetuning_public`.
- **A folder on the owner's machine** was replaced by `C:\projects`.
- **Reports and records were redacted, not rebuilt.** A file that says "do not edit by hand" was changed by the replacements above and by nothing else.
- **The checks of the configuration freeze are skipped here.** The freeze record names AWS resources, so its content changed with the placeholder and its hash is no longer the one the unlock record names. Tests marked `private_records` are skipped in this copy, and say so in the test output. The gate on the test split refuses a test run here. The freeze can be verified in the private repository only.
- **Some paths changed.**
  - `docs/runbook.md` (in the private repository it is `runbook.md` under `administration/`)

## Left out

A document here may name a file under one of these. It is not in this copy.

- `.claude/`
- `track_progress/`
- `docs/old_references/`
- `administration/`
- `reqspecs/0. PLANNING/planning_material/`
- `reqspecs/0. PLANNING/execution_plan_reference/`
- `tests/test_process_scaffold.py`
- `.github/copilot-instructions.md`
- `.github/instructions/`
- `reports/public_export/`

## Using it

The tests need no GPU, no network, and no credentials. The steps that are billed or that change AWS need an AWS account of your own, with its own bucket, roles, and budget in `config/aws.yaml`. The dataset, the runs, the replies, and the model weights are not in the repository.
