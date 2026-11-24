# PRP: Public repository export (req_006 v6.04)

**Feature Name**: public_export (packaging_demo v6.04)  
**Version**: 4  
**Created**: 2026-11-24  
**Status**: ✅ IMPLEMENTED  
**Implemented**: 2026-11-24  
**Implementation Summary**: [impl_006_20261124_083000.md](../impl_summary/impl_006_20261124_083000.md)  
**Confidence Score**: 7/10  
**Estimated Time**: 3–4 hours of code, tests, and documents; about 20 minutes of this machine (the suite inside the copy, twice); nothing billed  
**Origin**: `reqspecs/reqs/req_006_packaging_demo/req_006_04_spec.md`  

Template: `prp_lean.md`, extended (medium: one small package and a script; the risk is in
what the redaction breaks, which is found by running the suite in the copy).

## Goal

`venv\Scripts\python -m scripts.public_export build --to <folder>` writes a copy of a
commit that can be published: without history, without the folders the owner left out,
with the AWS account number and two other private strings replaced, the runbook at
`docs/runbook.md`, and a notice that it is a redacted snapshot. `check` proves that
nothing private remains and that lint and the suite pass inside the copy.

## Why

- The owner wants the code public and the private repository unchanged.
- The two model cards link to the public repository.

## What

All six decisions of spec section 8 are taken: the placeholder is `123456789012`; the
private repository's name becomes `SudiptoHaldar/qlora_finetuning_public`; the owner
commits and pushes, with the GitHub noreply address; a pytest marker skips the freeze
tests in the copy; one new commit per export.

Success criteria

- [x] `pytest -q` and `ruff check .` pass here, with and without the training stack. Tests
      marked `private_records` still run here.
- [x] `public_export plan` lists what is kept, left out, and generated, from the commit.
- [x] `public_export build --to` the owner's folder, from a clean commit.
- [x] `public_export check --to` that folder: E1 to E10 pass; the record is written under
      `reports/public_export/`.
- [x] Nothing frozen changed. No commit, push, or network call is made by the tool.

## All Needed Context

| File | Why |
|---|---|
| `reqspecs/reqs/req_006_packaging_demo/req_006_04_spec.md` | D1 to D22, checks E1 to E10, the six decisions, the owner's steps (section 9) |
| `qlora_finetuning/packaging/scan.py`, `files.py` | `scan_text`, `sha256_file`, `write_text_lf`, `read_json`: reused |
| `qlora_finetuning/packaging/record.py` | `check_result`, the shape of a record with numbered checks and a Markdown rendering |
| `scripts/hf_package.py` | The script conventions: `main(argv, cfg, steps)`, ASCII output, exit codes 0, 1, 2 |
| `qlora_finetuning/settings.py` | `load_config`, `get_str`, `REPO_ROOT`, `ConfigError` |
| `config/aws.yaml` | `account_id`: the value to replace, read at export time |
| `tests/test_readme.py` | Runs in the copy too: the notice in the README may hold no twelve-digit number |
| `reqspecs/references/project_patterns/release_package.md`, `check_script.md` | The conventions for something that leaves the project, and for a check script |

### Facts found (2026-11-24)

- 552 tracked files; 526 kept and the runbook; 5.5 MB. The account number is in 37 kept
  files (164 occurrences); `docs/runbook.md` is named in 59.
- The kept files hold a path on the owner's machine in three documents
  (`C:\projects\...`, the export's own destination). It is replaced by a
  neutral prefix: one more literal replacement, reported to the owner.
- Tests plant private-looking strings on purpose (an access key, a token, a machine path)
  in `tests/packaging/test_config_and_scan.py`. They are written in pieces so the scan of
  the copy does not find them; new tests do the same.
- Tests already use `123456789012` and `111122223333` as invented accounts, never as "the
  other account" against the real one, so the placeholder creates no clash.
- There is no `tests/conftest.py`. The package is not installed in the environment, so
  `python -m pytest` in another folder imports that folder's code.
- `reports/public_export/` is left out of the copy: the record describes a publication and
  always names an earlier commit than the one that holds it.

### Gotchas

- **Export a tree, not the folder**: `git archive <commit>` piped into `tarfile`. Git stores
  LF; bytes are written as they are, in binary mode.
- **The clean-tree rule looks at tracked files only.** An untracked file is not in the
  archive; it is counted and named, not refused.
- **A build needs a commit of this very code**, and `check` writes the record into the
  tracked tree. The owner is asked once for leave to commit before the real build.
- **The suite is run in a temporary copy of the destination**, so no cache or bytecode is
  written into the folder that will be pushed.
- **Block the training stack with a folder of stub packages on `PYTHONPATH`**, each raising
  `ModuleNotFoundError` with its name: no pytest plugin of the project is needed.
- **The marker is given only to tests that fail in the copy because the freeze records are
  redacted.** Find them by exporting the working tree through a temporary index and
  running the suite there. Any other failure is a fault to fix, not to mark.
- **Never delete what the tool did not write.** The destination carries a git-ignored
  manifest of the last export; only paths in it are removed.
- **Windows console is cp1252**: ASCII output. No dataset text exists in the commit.

## Implementation Blueprint

```
config/public_export.yaml          left_out, moved, replacements, path_rewrites, may_not_remain, notice, limits
qlora_finetuning/public_export/
  __init__.py     config.py        the typed config
  source.py       the files of a commit (git archive); the commit's hash and date; dirty paths
  plan.py         kept / left out / moved / generated
  transform.py    replacements and rewrites on text; the notice, the README section, the runbook note
  build.py        expected_files(...) -> {path: bytes}; write the copy; the manifest; safe removal
  checks.py       E1 to E10; the suite in a temporary copy; the record and its Markdown
  snapshot.py     NOTICE_NAME; skip_private_records(items, root) for tests/conftest.py
scripts/public_export.py           plan | build --to | check --to
tests/conftest.py, pytest.ini      the marker and its skip
tests/public_export/               a stand-in git repository per test
```

```python
files = commit_files(root, commit)                       # {posix path: bytes}
plan = make_plan(files, cfg)                             # kept, left_out, moved
expected = expected_files(files, plan, cfg, commit_info, account)   # transformed + generated
write_copy(dest, expected)                               # refuses an unknown folder; removes only manifest paths
problems = compare(dest, expected)                       # E2 and D22 in one: the copy is what this commit exports
```

Tasks, in order

1. Config and its loader; `source.py` over `git`; tests with a stand-in repository.
2. `plan.py`, `transform.py`, `build.py`; tests.
3. `snapshot.py`, `tests/conftest.py`, the marker in `pytest.ini`; tests.
4. `checks.py` and the script; tests.
5. Export the working tree through a temporary index to a scratch folder; run the suite
   there; mark what fails for the freeze; fix anything else.
6. Break up planted strings in existing tests.
7. With the owner's leave: commit, `build`, `check` on the real folder; record.
8. Runbook, `coding.md`, architecture, README sentence, `.env.example`; spec section 11,
   roadmap, execution plan, implementation summary.

Error handling: a refusal is a `ConfigError` on one line (exit code 2); a failed check is
exit code 1 and never stops the record being written; `build` writes nothing if the
destination is refused.

## Validation Loop

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q
venv\Scripts\python -m scripts.public_export plan
venv\Scripts\python -m scripts.public_export build --to <folder>
venv\Scripts\python -m scripts.public_export check --to <folder>
venv\Scripts\python -m scripts.comparison_freeze status
```

Nothing is billed. The tool makes no commit, no push, and no network call; pushing and
making the repository public are the owner's.

## Anti-Patterns to Avoid

- Copying the working folder, or anything under `artifacts/`.
- A replacement by pattern. Replacements are literal and counted.
- Marking a test `private_records` to make a failure go away.
- Deleting in the destination by walking it.
- Editing a file of the copy by hand.

## Implementation Time Estimate

**3–4 hours**, and about 20 minutes of machine time.

| Task | Time |
|---|---|
| Config, source, plan, transform, build | 70 min |
| Checks, the suite in a copy, the record, the script | 50 min |
| The marker; finding and marking the freeze tests | 30 min |
| Tests | 50 min |
| Documents | 30 min |
| Machine: the suite in the copy, twice | 20 min |

## Confidence

**7/10.** The copying is simple. The risk is the number of tests that fail in a redacted
copy for reasons other than the freeze, and CI on Linux, which cannot be tried before the
owner pushes.

## Time Taken for PRP Generation

**~12 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 5 min |
| Research existing PRPs and patterns | 2 min |
| Writing PRP document | 3 min |
