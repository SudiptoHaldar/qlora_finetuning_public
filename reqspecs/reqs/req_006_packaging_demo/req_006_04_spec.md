# Packaging Demo

Spec Group      :   req_006_packaging_demo  
Feature Name    :   packaging_demo  
Version         :   4  
Status          :   🟡 Built and verified 2026-11-24 (section 11): the public folder is written and checked; the owner commits there, pushes, and makes the repository public  
Last updated    :   2026-11-24  

## 1. v6.04 — Public repository export

The project's repository is private. This version makes a second, public one from it:
a copy of the files a reader needs, without the git history, with the AWS account number
replaced by a placeholder.

```powershell
venv\Scripts\python -m scripts.public_export plan                 # what would be copied; writes nothing
venv\Scripts\python -m scripts.public_export build --to <folder>  # write the copy
venv\Scripts\python -m scripts.public_export check --to <folder>  # prove the copy is clean and its tests pass
```

The copy is written to a folder outside this repository
(`C:\projects\AIML_projects\qlora_finetuning_public`), which becomes the
working tree of `SudiptoHaldar/qlora_finetuning_public` on GitHub. That repository exists
already, empty and private. Pushing to it and making it public are the owner's steps.

The private repository stays the source of truth. The public one is never edited by hand:
a change is made here, committed, and exported again.

**Depends on**: v6.02 (complete 2026-11-24, M9): the README that the public copy shows, and
the cards that link to it. **Recommended before v6.03**, so that the link on the two model
cards leads somewhere when the models are uploaded.

## 2. What exists today (verified 2026-11-24)

### 2.1 What the owner decided on 2026-11-24

Recorded in `req_006_02_spec.md` section 8.1. They are locked here (section 3) and not
asked again.

| Topic | Decision |
|---|---|
| Destination | `C:\projects\AIML_projects\qlora_finetuning_public`; it exists and is empty |
| GitHub repository | `SudiptoHaldar/qlora_finetuning_public`, created empty and private; made public after the copy is pushed and reviewed |
| History | None is exported |
| Left out | `.claude/`, `track_progress/`, `docs/old_references/`, everything in `administration/` but the runbook, `reqspecs/0. PLANNING/planning_material/`, `reqspecs/0. PLANNING/execution_plan_reference/`, `tests/test_process_scaffold.py`, the two instruction files under `.github/` |
| Kept | The rest of `reqspecs/`, the CI workflow, `ledger/`, every other tracked file |
| Runbook | Published as `docs/runbook.md`, a reader's version; pointers to `docs/runbook.md` rewritten by rule |
| Account number | Replaced by a placeholder everywhere, the bucket name included |
| Freeze checks | Skipped in the public copy, which is a redacted snapshot and says so in its README. The freeze stays verifiable in the private repository |

### 2.2 The numbers

| | |
|---|---|
| Tracked files | 552 |
| Kept by the rules above | 526, and the runbook: 5.5 MB together |
| Files that hold the account number | 37 of the kept ones, 164 occurrences |
| Files that hold the bucket's name | 23 (the name ends with the account number) |
| Files that hold an AWS resource name (`arn:aws`) | 37 |
| Files that point to `docs/runbook.md` | 59, 65 occurrences: documents, and messages in code and tests |
| Files that name another file of `administration/` | 36 |
| Files that name `.claude/` or `track_progress/` | 13 and 12 |
| Email addresses in the kept files | Three, all invented for tests (`example.com`, `example.org`) |
| Paths on the owner's machine in the kept files | One, an invented one in a test |
| The private repository's name (`SudiptoHaldar/qlora_finetuning_public`) | 3 kept files |

### 2.3 What makes this more than a copy

- **The account number is inside other values**: every ARN, the bucket's name, the two IAM
  roles in `config/aws.yaml`. A placeholder that is not twelve digits breaks code that
  checks an ARN's shape, and tests that compare a config with a fixture.
- **The freeze record holds deployment ARNs**, and the unlock record names the freeze
  record's hash. Replacing the account number changes that hash, so the checks of the
  freeze fail in the copy. The owner's decision is to skip them there.
- **Messages in code name the runbook's path** (for example the hint that says where the
  training stack's installation is described), and tests assert those messages. The
  rewrite must change both together.
- **Reports say "do not edit by hand"** and the copy does edit them. The copy is not the
  place they are built; its README says so.
- **The git identity on this machine uses a personal email address.** A commit made in the
  public folder with it would publish that address.
- **CI runs the whole suite on Linux with a ten-minute limit.** On this machine the suite
  takes about seven minutes.
- The package is not installed in the virtual environment, so `python -m pytest` run
  inside the copy imports the copy's own code. The copy can be tested with this
  environment.

## 3. Locked decisions

### What the export is

| # | Decision | Reason |
|---|---|---|
| D1 | The export is of a **commit**, taken with `git archive`, never of the working tree. A tree with uncommitted changes is refused | What is published can be named. The SageMaker launcher packs code the same way |
| D2 | One command builds the copy from a config file, `config/public_export.yaml`: what is left out, what is replaced, which paths are rewritten, which strings may not remain. Nothing is decided in code | The export will be repeated after every change worth publishing |
| D3 | The public copy is never the source of anything. No code reads it, and nothing is copied back | One source of truth |
| D4 | The destination is given on the command line (`--to`) or by `QLORA_PUBLIC_EXPORT_DIR` in `.env`. It is not written in a committed file | A path on the owner's machine is not project configuration |
| D5 | `build` writes into a folder that is empty, or that holds a previous export (it carries the snapshot notice of D12). Any other folder is refused. The folder's `.git` is never read, written, or removed | The tool deletes files that a new export no longer has. It must not be able to do that anywhere else |
| D6 | The tool makes no git commit, no push, and no call to GitHub | Publishing is outward-facing and is the owner's (decision 4) |

### What is copied and what is changed

| # | Decision | Reason |
|---|---|---|
| D7 | The file set is: every file of the commit, minus the paths of section 2.1, plus the files the export generates (D11, D12). `plan` prints the three lists with counts | The owner's decisions, made checkable |
| D8 | Replacements are literal and listed in the config: the account number, and whatever decision 2 adds. The account number to replace is read from `config/aws.yaml` at export time, so it is written in one place | A second copy of the number in the export's own config would be one more file to redact |
| D9 | A replacement is made in every text file of the copy: code, tests, configs, reports, records, the ledger, documents. A file that is not UTF-8 text is copied as it is and listed | The number is in all of those kinds. One rule, no exceptions to forget |
| D10 | Path rewrites are literal too: `docs/runbook.md` becomes `docs/runbook.md`, in documents, in code, and in tests alike | A message and the test that asserts it change together |
| D11 | `docs/runbook.md` is the private runbook with the same replacements, and a short generated note at its top: it was written for the project's owner, the billed steps need the reader's own AWS account, and the placeholder is not a real account | The owner's decision. A mechanical transform cannot drift from the original |
| D12 | The copy gets a generated notice: a file `PUBLIC_SNAPSHOT.md` at its root and a short section at the top of its README. It says: this is a snapshot of a private repository at a stated commit and date; the account number is a placeholder; committed reports and records were redacted, not rebuilt; the checks of the configuration freeze are skipped here; which folders were left out | The owner's decision on the freeze. A reader must be able to tell a redacted record from an original |
| D13 | A mention of a file that was left out stays as it is written. The notice lists the folders left out, so a reader who follows one knows why it leads nowhere | Rewriting sixty documents' cross-references by guesswork would damage them. The working notes are history |
| D14 | Nothing under `artifacts/` is ever copied, and no dataset text exists in the commit to copy | Git ignores the folder; the rule on dataset text already holds for every tracked file |

### The freeze checks in the copy

| # | Decision | Reason |
|---|---|---|
| D15 | Tests that verify the committed freeze, unlock, and frozen-value records against the real repository get a pytest marker, `private_records`. In the private repository they run as today | The set that cannot hold in a redacted copy is named once, in the code |
| D16 | In a folder that carries `PUBLIC_SNAPSHOT.md`, tests with that marker are skipped, with the notice as the reason. Nothing else is skipped, and no test file is removed for this | A skip shows up in the test output with its reason. A missing file shows nothing |
| D17 | The marker is given only to a test that fails in the copy for this reason. The export's own check runs the suite in the copy and reports any other failure as a failure | "Skipped in public" must not become a way to hide a broken test |
| D18 | The gate on the test split is not weakened. In the copy it refuses a test run, because the freeze record no longer matches. That is the right behaviour for a redacted copy | Nobody should run the comparison's test pass from a redacted record |

### Proving the copy

| # | Decision | Reason |
|---|---|---|
| D19 | `check` runs after `build` and reads the destination only: checks E1 to E10 (section 4.3). It also runs `ruff` and the test suite **inside the copy**, with the training stack blocked, as CI has it | The copy is a different repository. It must pass on its own |
| D20 | The strings that may not remain are in the config: the real account number, an AWS access key, a Hugging Face token, an email address outside `example.com`, `example.org`, and GitHub's `users.noreply.github.com`, a path on someone's machine, and the private repository's name (decision 2) | An ARN, a bucket name, and a job name are allowed in the copy; they carry the placeholder |
| D21 | The export writes a record in the **private** repository, `reports/public_export/export_record.json` and `.md`: the commit that was exported, the file counts, how many replacements each rule made in how many files, the skipped tests, the checks. It holds no private value | What was published, and from what |
| D22 | The same commit exported twice gives the same bytes | A later export shows only what changed |

## 4. Scope to build

### 4.1 New code

| File | Holds |
|---|---|
| `config/public_export.yaml` | Paths left out; literal replacements; path rewrites; strings that may not remain; the texts of the notice |
| `qlora_finetuning/public_export/config.py` | The typed config |
| `qlora_finetuning/public_export/plan.py` | The file set of a commit: kept, left out, generated |
| `qlora_finetuning/public_export/transform.py` | Replacements and rewrites on one file's text; the runbook's note; the notice |
| `qlora_finetuning/public_export/build.py` | Write the copy; remove what a previous export had and this one has not |
| `qlora_finetuning/public_export/checks.py` | E1 to E10, and the record |
| `scripts/public_export.py` | `plan`, `build`, `check` |
| `tests/conftest.py`, `pytest.ini` | The marker `private_records` and its skip in a snapshot (D15, D16) |

The scan for strings reuses `packaging/scan.py`. Existing tests gain the marker where D17
says so; nothing else in the project changes.

### 4.2 The commands

```powershell
venv\Scripts\python -m scripts.public_export plan
venv\Scripts\python -m scripts.public_export build --to C:\projects\AIML_projects\qlora_finetuning_public
venv\Scripts\python -m scripts.public_export check --to C:\projects\AIML_projects\qlora_finetuning_public
```

`plan` and `check` write nothing in the destination. `build` writes only there. `check`
writes the export record in this repository. No command calls AWS, GitHub, or any network.

### 4.3 Checks

| # | Check | Passes when |
|---|---|---|
| E1 | Source | The export was made from a commit, and the working tree had no uncommitted change |
| E2 | File set | The copy holds exactly the kept files and the generated ones; nothing of a left-out path; nothing from `artifacts/` |
| E3 | Nothing private | No file of the copy holds a string of D20 |
| E4 | Replacements | Each replacement was made at least once and none of its originals remains; the counts are in the record |
| E5 | Pointers | No `docs/runbook.md` remains; `docs/runbook.md` exists; every link of the copy's README leads to a file of the copy |
| E6 | The notice | `PUBLIC_SNAPSHOT.md` and the README section are present and name the exported commit |
| E7 | Lint and tests | `ruff check .` and `pytest -q` pass inside the copy with the training stack blocked |
| E8 | Skipped tests | The tests skipped by the notice are exactly those marked `private_records`; their count is in the record |
| E9 | The private repository is untouched | `comparison_freeze status` here reports no change; no tracked file here changed but the export record |
| E10 | Size and kinds | No file over 5 MB; no PDF, archive, or weight file |

### 4.4 Tests

No GPU, no network, no credentials. A stand-in repository is made in a temporary folder
with `git init`, a few files, and one commit.

- Plan: left-out paths are left out; a kept file is kept; a generated file is listed; a
  path that would escape the destination is refused.
- Transform: the account number is replaced inside an ARN, a bucket name, JSON, YAML, and
  Markdown; a number that only resembles it is not; the runbook pointer is rewritten in a
  document, a code string, and a test; a file that is not text is copied unchanged.
- Build: an empty destination is written; a second build removes a file the commit no
  longer has; a destination with unknown files is refused and left as it was; `.git` in
  the destination is not touched; a dirty source tree is refused.
- Checks: each of E1 to E6, E8, and E10 passes and fails; a planted account number, token,
  email address, and machine path are found.
- The marker: with the notice file present, a marked test is skipped with its reason and
  an unmarked one runs; without it both run.
- Reproducible: two builds of one commit are the same bytes.
- Guards: the package imports no boto3 module, no network library, and none of the training
  stack; it never runs `git commit`, `git push`, or `gh`.
- Script: exit codes 0, 1, and 2; ASCII output.

### 4.5 Documentation

- Runbook: a section on the export. When to export, the three commands, the owner's steps
  after `check` (section 9), what to do when a check fails.
- `reqspecs/1. PROMPTS/coding.md`: one rule (D3, D5, D6, D15 to D17).
- `architecture/architecture_overview.md`: the new package; that the public repository is
  a product of this one.
- `README.md`: one sentence that the public repository is exported from a private one. The
  notice itself is generated into the copy only.
- `.env.example`: `QLORA_PUBLIC_EXPORT_DIR`.
- `administration/aws_commands_run.md`: nothing. No AWS command is run.

## 5. Out of scope

- Pushing to GitHub, making the repository public, settings of the GitHub repository
  (description, topics, branch rules). The owner's.
- Uploading the models to Hugging Face. That is v6.03.
- Rewriting the history of the private repository, or removing the account number from it.
- Rebuilding reports in the public copy. They are redacted copies of the private ones.
- Making the freeze verifiable in the public copy.
- Rewriting the working notes under `reqspecs/` for a public reader, beyond the
  replacements.
- A copy of the dataset, of runs, of replies, or of weights.
- Keeping the two repositories in step automatically (a hook, a scheduled job).
- Accepting changes made in the public repository. Pull requests there are answered by a
  change here and a new export.

## 6. What this hands to later versions

| For | What |
|---|---|
| v6.03 | A public address for the cards' "Code" link; the README a visitor of the model page lands on |
| Later exports | One command to repeat, and a record of what the last one published |
| req_007 and later | A rule for new files: what must be left out or replaced is declared in `config/public_export.yaml` |

## 7. Known limits and follow-ups

- **The public copy cannot prove the freeze.** Its freeze record is redacted. The claim
  that every arm's configuration was fixed before the test split was read rests on the
  private repository and on the report's own account of it.
- **Identifiers other than the account number stay** unless decision 2 removes them: the
  ids of the two Bedrock deployments and custom models, SageMaker job names, run ids.
  None of them can be used without credentials for the account.
- **The working notes were written for the owner.** They name files that are not in the
  copy, and they describe steps only the owner can take.
- **A reader cannot rerun the cloud steps as they are.** They need an AWS account of their
  own, with its own bucket, roles, quotas, and budget.
- **The exported commit's hash is published** in the notice. It identifies a commit of a
  private repository and reveals nothing else.
- **CI in the public repository is new.** It has the private workflow's ten-minute limit,
  and the suite takes about seven minutes on this machine. If it times out there, the
  limit is raised here and exported again.
- **An export is a snapshot.** Work done here after it is not public until the next one.

## 8. Decisions for the owner

**All six are taken (2026-11-24).** The table below them is kept as the record of what was
asked.

| # | Taken as |
|---|---|
| 1 | The placeholder is `123456789012` |
| 2 | The private repository's name, `SudiptoHaldar/qlora_finetuning_public`, is replaced by the public one, `SudiptoHaldar/qlora_finetuning_public`. Nothing else is replaced: the Bedrock deployment ids and the SageMaker job names stay. Where a working note calls the repository private, the copy then says that of the public name; the snapshot notice explains the replacement |
| 3 | The public commit carries the name `SudiptoHaldar` and the GitHub noreply address `16464190+SudiptoHaldar@users.noreply.github.com`, set in the public folder only. The private repository's git setting is not touched |
| 4 | As recommended: Claude builds and checks the folder; the owner commits and pushes |
| 5 | As recommended: the marker `private_records` |
| 6 | As recommended: one new commit on `main` for each export |

| # | Question | Recommendation |
|---|---|---|
| 1 | **What is the placeholder for the account number?** | `123456789012`, the account number AWS uses in its own documentation. Every ARN and the bucket name stay well formed, so code that checks their shape and tests that compare a config with a fixture keep passing. The alternative, a word such as `ACCOUNT_ID`, is plainer to read and breaks those checks, which would then need the skip marker too |
| 2 | **Is anything besides the account number replaced?** The kept files also hold the private repository's name (`SudiptoHaldar/qlora_finetuning_public`, 3 files), the ids of the two Bedrock deployments and custom models, and SageMaker job names | Replace the private repository's name with `a private repository`. Keep the deployment ids and job names: they are in the frozen records and the reports, nobody can use them without your credentials, and removing them means redacting many more records |
| 3 | **Which name and email address does the commit in the public repository carry?** Git on this machine is set to a personal address | Your GitHub "noreply" address, set for that folder only (`git config user.email` inside it). GitHub shows it under Settings, Emails. The private repository's setting is not touched |
| 4 | **Who commits and pushes?** | I build and check the folder. You run `git init`, set the identity, commit, add the remote, and push: five commands, listed in section 9. Pushing publishes to an outside service, and switching the repository to public is yours in any case |
| 5 | **How are the freeze checks skipped: a marker on the tests, or leaving those test files out?** | The marker (D15 to D17). The tests stay in the copy, and its test output says "skipped: redacted snapshot" for each. Leaving files out hides that anything was skipped |
| 6 | **What does a later export look like in the public repository?** | One new commit on `main` for each export, with the message "Snapshot of the private repository at `<short hash>`". The public history is then a list of snapshots. The alternative replaces the single commit each time, which loses that list and needs a forced push |

## 9. Acceptance

Code, by Claude:

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass here,
  with and without the training stack blocked. The marked tests still run here.
- `scripts.public_export plan` on the real repository lists the file set of section 2.2.
- `comparison_freeze status`: nothing frozen has changed.

On the owner's machine, by Claude:

1. `public_export build --to <the folder>` from a clean commit.
2. `public_export check --to <the folder>`: E1 to E10 pass. The record is written and
   committed here.

By the owner:

3. Look through the folder: the README with its notice, `docs/runbook.md`, one report, one
   config file.
4. In the folder, once:

   ```powershell
   git init -b main
   git config user.name "SudiptoHaldar"
   git config user.email "16464190+SudiptoHaldar@users.noreply.github.com"
   git add -A
   git commit -m "Snapshot of the private repository at <short hash>"
   git remote add origin https://github.com/SudiptoHaldar/qlora_finetuning_public.git
   git push -u origin main
   ```

5. Check the repository on GitHub while it is private: the README renders, the diagrams
   draw, CI passes.
6. Switch it to public.

A check that fails stops at step 2: the folder is not pushed.

## 10. Next step

The six decisions are taken (section 8) and the PRP was generated and executed on
2026-11-24. What is left is the owner's (section 11.4).

```
/generate-prp reqspecs/reqs/req_006_packaging_demo --version 4
```

## 11. As built (2026-11-24)

Built by `prp_006_public_export_v4.md`. The tool makes no commit in the public folder, no
push, and no network call; nothing is billed and no frozen value changed
(`comparison_freeze status`: nothing frozen has changed). The public folder is written and
checked from the commit that holds this text; what that check found is in
`reports/public_export/export_record.md` in the private repository. **Pushing the folder and
making the repository public are the owner's steps** (section 9, runbook section 12.3).

### 11.1 What was built

| File | Holds |
|---|---|
| `config/public_export.yaml` | What is left out, moved, replaced, and rewritten; what may not remain; the invented values tests plant |
| `qlora_finetuning/public_export/source.py` | A commit's files through `git archive`; the commit's hash and day; tracked changes |
| `transform.py` | The plan, the literal rules, the snapshot notice, `expected_files` |
| `build.py` | `write_copy`, the manifest, `compare` |
| `checks.py` | E1 to E10, the scan, lint and the suite in a scratch copy, the record |
| `snapshot.py`, `tests/conftest.py`, `pytest.ini` | The marker `private_records` and its skip where the notice is |
| `scripts/public_export.py` | `plan`, `build --to`, `check --to` |

Where the build differs from sections 3 and 4:

- **A third replacement.** Three documents name a folder of the owner's machine (the
  export's own destination). It is replaced by a neutral folder. It was not among the
  owner's decisions; it is the same kind of thing as decision 1 and is reported here.
- **The account number is replaced only as a whole number.** A longer number that merely
  contains its digits is left alone, and the scan reads the same way.
- **`plan` and `transform` are one module** (`transform.py`): the plan is three lines.
- **The suite is run in a scratch copy that is made a git repository**, with one throwaway
  commit in a temporary folder. Two tests read "this repository's commit" and fail in a
  folder that is not a repository; a checkout in CI is one. D6 stands as meant: no commit
  in the public folder or in this repository.
- **`reports/public_export/` is left out of a copy.** The record always names an earlier
  commit than the one that holds it.
- **Invented values are listed in the config.** Four tests plant an access key, a token, or
  a machine path to prove a scan finds it. Those values are set aside before a file is
  scanned; one is the access key of AWS's own documentation.
- **The email scan looks only around each `@`.** A pattern over the whole text took
  minutes on a file of a million letters.
- **The destination keeps a git-ignored manifest** of what the last export wrote, so a
  later export removes those files and no other.

### 11.2 Which tests carry the marker

Five tests fail in a redacted copy because the freeze record's hash no longer matches:

- `tests/data/test_split_loader.py::test_the_repository_is_locked_or_unlocked_against_its_freeze_record`
- `tests/scripts/test_comparison_scripts.py::test_the_repository_is_locked_or_unlocked_by_a_record_that_names_the_freeze`
- `tests/packaging/test_guards_and_script.py::test_the_gate_refuses_a_package_on_the_test_split` (three cases)

Nothing else failed in the copy. Every other test passes there unchanged, the ones that
compare a config with a fixture included, because the placeholder is twelve digits.

### 11.3 Tests

| Run | Result |
|---|---|
| `venv\Scripts\python -m pytest -q` here | 3,099 passed, 1 skipped, 12 deselected |
| The same with the training stack blocked | 3,091 passed, 9 skipped, 12 deselected |
| Lint and the suite in a trial copy of the working tree, stack blocked | lint clean; 3,047 passed, 24 skipped, of which 5 by the snapshot notice |
| `venv\Scripts\python -m ruff check .` | clean |

New: `tests/public_export/` (37 tests in three files).

### 11.4 For the owner

1. Look through the public folder: `README.md` with its snapshot line, `PUBLIC_SNAPSHOT.md`,
   `docs/runbook.md`, one report, `config/aws.yaml`.
2. Run the seven git commands of section 9 there. **Set the noreply address before the
   first commit.**
3. On GitHub, while the repository is private: the README renders, its two diagrams draw,
   CI passes. The workflow has a ten-minute limit and the suite takes about six minutes
   here.
4. Switch the repository to public.
