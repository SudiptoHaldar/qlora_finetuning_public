# Packaging Demo

Spec Group      :   req_006_packaging_demo  
Feature Name    :   packaging_demo  
Version         :   3  
Status          :   ✅ Complete 2026-11-25 (section 11): both models are public and the final check passed  
Last updated    :   2026-11-24  

## 1. v6.03 — Publish the adapter and the merged copy

The two packages of v6.02 are ready on the owner's machine and checked. This version puts
them on the Hugging Face Hub, where anyone can load them:

| Repository | Folder | Size |
|---|---|---|
| `sudiptoHaldar/qwen3-4b-finqa-program-qlora` | `artifacts/release/adapter/` | 82 MB |
| `sudiptoHaldar/qwen3-4b-finqa-program-merged` | `artifacts/release/merged/` | 8.06 GB |

Each is uploaded as a private repository, checked against what was built, read by the owner
on the Hub, and only then made public. Afterwards the code printed on each card is run
once more, this time loading from the Hub by name.

The trigger for this version was the owner's decision to publish, taken on 2026-11-24.

**Depends on**: v6.02 (both packages ready, checks P1 to P10 pass; complete 2026-11-24).
v6.04 is complete too, so the "Code" link on both cards leads to a public repository.

## 2. What exists today (verified 2026-11-24)

| Fact | |
|---|---|
| The packages | Both `ready` in `reports/packaging/package_record.json`, which holds every file's size and SHA-256 |
| The Hub account | `sudiptoHaldar`. Neither repository exists yet |
| The token | Created by the owner, with write access. It goes in `.env` as `HF_TOKEN`; `.env.example` has the empty placeholder. `.env` is git-ignored |
| The upload tool | The `hf` command of `huggingface_hub` 2.4.0, already in the environment: `hf upload REPO_ID LOCAL_PATH --private` creates the repository as private if it does not exist, and resumes an interrupted upload when run again |
| How the token is read | `hf` and the library read `HF_TOKEN` from the process environment. They do not read `.env`. `settings.get_str` does |
| Storage | A free account has 100 GB of private storage and no stated limit on public models. Both packages fit either way |
| The card's code | Already loads by name: it calls `snapshot_download` when the name is not a local folder. `verify.run_card_code` runs it with the folder substituted and the Hub switched off |
| The base model | In the local cache. A run of the adapter card's code from the Hub downloads 82 MB; the merged card's downloads 8 GB into the cache, beside the copy under `artifacts/` |

What is missing:

- No code uploads anything, by design: a guard test forbids upload calls in
  `qlora_finetuning/packaging/`.
- Nothing compares what is on the Hub with what was built.
- The README says the packages are "packaged for the Hugging Face Hub" and to build them
  locally "until the two repositories exist".

**The tuned Nova models are kept** (owner's decision, 2026-11-24): keeping them costs $1.95
a month each, and getting one back means paying for its training again. Nothing in this
version depends on them, and no teardown is part of it.

## 3. Locked decisions

### What is uploaded

| # | Decision | Reason |
|---|---|---|
| D1 | Exactly the two folders of v6.02 are uploaded, and only after `hf_package status` passes P1 to P10 and every file on disk has the SHA-256 the package record holds | What is published is what was scored and reviewed. A folder touched since then is not |
| D2 | A repository that already exists is uploaded to only if every file it holds is a file of the package with the same hash. Anything else is refused and left alone | An upload adds a commit to whatever is there. A repository with someone's other files is not ours to overwrite |
| D3 | Both repositories are created **private**. No command of this version makes one public | The owner's review comes first (decision 2), and switching is the owner's (decision 3) |
| D4 | The adapter is uploaded and checked first, then the merged copy | The merged card links to the adapter. The small one shows any problem with the token or the account before 8 GB are sent |
| D5 | Nothing but the package's files goes up: no code of the project, no record, no run | The cards link to the code repository |

### The token

| # | Decision | Reason |
|---|---|---|
| D6 | The token is read from `HF_TOKEN` in `.env` or the environment, and handed to the upload through the environment of the process that needs it. It is never a command-line argument, never printed, never logged, and never written to a file | A command line is visible to other programs and stays in shell history |
| D7 | A command that needs the token and does not find it stops with one line that says where to put it | As every other missing setting |
| D8 | Every line the tool prints or writes is scanned for the token's value before it leaves the process, and the record is scanned before it is written | A library's error message can echo a header |
| D9 | The token is the owner's. A command that uses it is run by the owner, with `--yes` (decision 1) | As with AWS: what changes something outside the machine is the owner's |

### After the upload

| # | Decision | Reason |
|---|---|---|
| D10 | `check` lists the repository's files on the Hub and compares names, sizes, and SHA-256 hashes with the package record. A large file's hash is the one the Hub stores for it; a small file is downloaded and hashed | The upload is proved by content, not by "it finished" |
| D11 | `check` also runs the card's code **by repository name**, in an empty directory, with nothing substituted: the adapter card's once, the merged card's in 4-bit and at full precision. Each reply must run to the sample's expected result | It is the first time the code runs as a visitor will run it. v6.02 ran it against the folder |
| D12 | `check` records the Hub's commit hash of each repository and whether it was private or public at that moment | The record says what was checked. A card edited on the website later has another commit |
| D13 | The owner reads each rendered card on the Hub while it is private: the header's badges, the results widget, the code block, the tables, the licence | The Hub renders the header into things the Markdown preview does not show |
| D14 | After both are public, `check` is run once more without a token, and that run is the one the record keeps | Without a token it sees what the world sees |
| D15 | The record is `reports/packaging/publish_record.json` and `.md`: repository names, Hub commits, the file comparison, the card-code runs, visibility, dates. Committed. It holds no token | As the package record |

### Afterwards

| # | Decision | Reason |
|---|---|---|
| D16 | The README's section on the tuned model is changed to say the models are published, with links, and the public repository is exported again | The README is the first thing a visitor of the code reads |
| D17 | A card is never edited on the Hub. A change is made in the template, `hf_package cards` builds it, and the folder is uploaded again | One source for the card. D10 would otherwise report the edited card as a difference |
| D18 | A later upload of a rebuilt package is a new commit in the same repository, after the same checks | The Hub keeps the history; a link to the repository keeps working |

## 4. Scope to build

### 4.1 New code

| File | Holds |
|---|---|
| `qlora_finetuning/packaging/publish.py` | The pre-upload checks (D1, D2), the file comparison (D10), the record. It takes a Hub client as an argument and imports none |
| `qlora_finetuning/packaging/hub.py` | The one module that talks to the Hub: list a repository's files, upload a folder, download a small file. The library is imported inside functions |
| `scripts/hf_publish.py` | `plan`, `upload`, `check` |

The guard test of `packaging/` is changed to allow upload calls in `hub.py` only.
`verify.run_card_code` gains a way to run with the Hub switched on.

### 4.2 The commands

```powershell
venv\Scripts\python -m scripts.hf_publish plan                    # what would be uploaded; no network, no token
venv\Scripts\python -m scripts.hf_publish upload adapter --yes    # THE OWNER: creates the private repository, uploads
venv\Scripts\python -m scripts.hf_publish check adapter           # compares the Hub with the record; runs the card's code
venv\Scripts\python -m scripts.hf_publish upload merged --yes     # THE OWNER: 8 GB
venv\Scripts\python -m scripts.hf_publish check merged
venv\Scripts\python -m scripts.hf_publish check                   # both, after they are public: no token needed
```

`plan` touches nothing outside the machine. `upload` and a `check` of a private repository
need the token. A `check` of a public repository needs none.

### 4.3 Checks

| # | Check | Passes when |
|---|---|---|
| U1 | The package is the one that was checked | `hf_package status` passes P1 to P10; every file on disk has the size and hash of the package record |
| U2 | The repository is ours to upload to | It does not exist, or it holds only files of the package with the same hashes |
| U3 | The token is there | `HF_TOKEN` is set. The check prints only that it is |
| U4 | The Hub holds the package | Same file names, sizes, and hashes as the package record; nothing extra |
| U5 | The card's code runs from the Hub | By repository name, in an empty directory, to the sample's expected result; both ways for the merged copy |
| U6 | Visibility | Recorded. For the final run: public |
| U7 | Nothing leaked | The token's value is in no line printed and in no file written |

### 4.4 Tests

No network, no token, no GPU: a stand-in Hub client that keeps files in memory.

- Plan: lists both packages with file counts and sizes; a package that is not `ready`, or a
  file changed since the record, fails U1 and nothing is uploaded.
- Upload: refused without `--yes`; refused without a token, with a line that names `.env`;
  a repository that holds a foreign file is refused (U2); one that holds part of the
  package is resumed; the repository is created private.
- The token: never in an argument list, never in output, never in the record; an error
  message that contains it is printed with the value removed.
- Check: equal files pass; a missing, extra, or changed file is named; the Hub's commit
  and the visibility are recorded.
- The card's code: run with the Hub switched on when asked, off otherwise (v6.02's check
  is unchanged).
- Guards: only `hub.py` names an upload call; no module deletes a repository or changes
  its visibility.
- Script: exit codes 0, 1, and 2; ASCII output.

### 4.5 Documentation

- Runbook: a section on publishing. The token, the order of the commands, what to look at
  on the Hub, how to make a repository public, how to publish a rebuilt package.
- `reqspecs/1. PROMPTS/coding.md`: one rule (D1, D6, D9, D17).
- `README.md`: D16. Then one more public export.
- `architecture/architecture_overview.md`: the two Hub repositories as products.

## 5. Out of scope

- Making a repository public, deleting one, renaming one, gating access to one.
- Editing a card on the Hub.
- A GGUF copy, a quantized copy, an inference endpoint, a demo Space.
- Publishing the Nova models. They cannot leave Bedrock.
- Tearing down anything in AWS. The tuned Nova models are kept (owner, 2026-11-24).
- A collection, a paper page, an announcement.
- Changing a package or a card. That is v6.02's `hf_package`.

## 6. What this hands on

| For | What |
|---|---|
| A reader | Two public model repositories with cards whose code has been run from the Hub |
| A later release | `hf_publish upload` and `check` to repeat after `hf_package` rebuilds a package |
| The public code repository | A README that links to the published models, after one more export |

## 7. Known limits and follow-ups

- **A name is hard to change once public.** The Hub redirects after a rename, but cards and
  copied links keep the old name.
- **The Hub checks a card's header when it is uploaded** and may warn about a field. The
  metric types in `model-index` are this project's own names, so the results widget may
  show them plainly or not at all. D13 is where that is seen.
- **The merged card's code downloads 8 GB** the first time it runs from the Hub, into the
  Hugging Face cache, beside the copy under `artifacts/`.
- **A visitor needs a GPU and the libraries** the card names. Nothing here serves the
  model.
- **Download counts and discussions are the Hub's.** The project does not read them.
- **The Hub can change how it stores large files.** D10 compares the hash the Hub reports
  for a large file; if it stops reporting one, the file is downloaded and hashed instead.

## 8. Decisions for the owner

**All six are taken as recommended (2026-11-24).** The owner runs the uploads, with
`--yes`; each repository is private first; the owner makes it public on the Hub; the upload
is checked while private, by the owner, and once more after publishing without a token;
both packages in one sitting, the adapter first; the default repository settings. The
table is kept as the record of what was asked.

| # | Question | Recommendation |
|---|---|---|
| 1 | **Who runs the uploads?** They use your token and create repositories under your name | You, with `--yes`. I build and test the tool, run `plan`, and run the token-free `check` once the repositories are public. The alternative is that I run `upload` with the token from `.env` each time you say so; the token then passes through a process I start |
| 2 | **Private first, or straight to public?** | Private first. The rendered card, the badges, and the results widget can only be seen on the Hub, and a mistake is easier to correct before anyone has downloaded the model |
| 3 | **Who makes a repository public, and how?** | You, on the repository's Settings page on the Hub. No command of the tool changes visibility, so nothing can become public by a slip |
| 4 | **Is the upload checked while still private?** That check needs your token | Yes, and you run it: a broken upload is then never public. The final check after publishing needs no token and I run it |
| 5 | **Both packages in one sitting, or the adapter now and the merged copy later?** | Both, the adapter first. The adapter's card links to the merged copy, so the link is dead until the second one is public |
| 6 | **Any setting other than the defaults?** Gated access (visitors must ask), discussions, a licence other than the card's | Defaults: not gated, discussions on, the licence the card states (`apache-2.0`) |

## 9. Acceptance

Code, by Claude:

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
  and without the training stack blocked, with no network and no token.
- `scripts.hf_publish plan` on the real repository: both packages listed, U1 passes.
- `comparison_freeze status`: nothing frozen has changed.

By the owner, in order:

1. Put the token in `.env` as `HF_TOKEN`.
2. `hf_publish upload adapter --yes`, then `hf_publish check adapter`: U1 to U5 and U7 pass.
3. `hf_publish upload merged --yes`, then `hf_publish check merged`.
4. Read both cards on the Hub (D13).
5. Make both repositories public, the merged copy first, so the adapter's link never
   leads nowhere.

By Claude, after that:

6. `hf_publish check` without a token: U4 to U7 pass for both, visibility public. The
   record is committed.
7. The README (D16), and one more export of the public repository; the owner commits and
   pushes that folder.

## 10. Next step

The six decisions are taken (section 8) and the PRP was generated and executed on
2026-11-24. What is left is the owner's (section 11.3).

```
/generate-prp reqspecs/reqs/req_006_packaging_demo --version 3
```

## 11. As built (2026-11-24)

Built by `prp_006_publish_models_v3.md`. Nothing was uploaded by Claude, no token was read,
and nothing is billed. **The uploads are the owner's** (section 9, runbook section 13).

### 11.1 What was built

| File | Holds |
|---|---|
| `qlora_finetuning/packaging/hub.py` | `HubClient`: `state`, `create_private`, `upload`, `fetch`. The one module that imports the Hub's library |
| `qlora_finetuning/packaging/publish.py` | `read_packages`, `local_problems` (U1), `remote_problems` (U2, U4), `upload_package`, `check_package`, `card_runs` (U5), the record, `redact` |
| `scripts/hf_publish.py` | `plan`, `upload {adapter,merged} --yes`, `check [adapter|merged]` |
| `packaging/verify.py` | `run_card_code(..., online=, token=)` |
| `config/packaging.yaml` | `publish`: the token's variable name, the Hub's own files, the timeout |

Where the build differs from sections 3 and 4:

- **The upload uses the library, not the `hf` command.** The token is then an argument of a
  function in the same process, never of a command line.
- **`.gitattributes` is allowed on the Hub.** The Hub adds it to every repository.
- **A file's hash on the Hub**: for a large file the Hub's own SHA-256 is compared; a small
  file is fetched and hashed. Sizes are compared for both.
- **"Without a token" means none at all.** The check of a public repository switches off a
  saved login as well, in the client and in the process that runs the card's code.
- **The token is read only when a repository cannot be seen without it**, so the final
  check never touches it.
- **U2 and U3 are conditions of `upload`**; `check` reports U1 and U4 to U7.
- **A `check` of one package keeps the record's entry for the other.**

### 11.2 On this machine

`hf_publish plan`: U1 passes for both. The adapter package is 12 files, 82.3 MB; the merged
copy is 16 files, 8,060.9 MB; every file has the size and hash of the package record.
`hf_publish upload adapter` without `--yes` is refused with exit code 2.

### 11.3 For the owner

1. Put the token in `.env` as `HF_TOKEN`.
2. `hf_publish upload adapter --yes`, then `hf_publish check adapter`.
3. `hf_publish upload merged --yes`, then `hf_publish check merged`.
4. Read both cards on the Hub. Make both repositories public, the merged copy first.

Then Claude: `hf_publish check` without a token, the README, and one more export of the
public repository.

### 11.4 Published (2026-11-25)

The owner uploaded both packages on 2026-11-24, checked each while private, read the cards on the Hub, and made both repositories public. The final check, run on 2026-11-25 without a token, passed U1 and U4 to U7 for both: the Hub holds the same files by name, size, and hash, and each card's code ran by repository name to the sample's expected result (three runs).

| Model | Address | Commit on the Hub |
|---|---|---|
| The adapter | https://huggingface.co/sudiptoHaldar/qwen3-4b-finqa-program-qlora | `789ee8536389` |
| The merged copy | https://huggingface.co/sudiptoHaldar/qwen3-4b-finqa-program-merged | `16a585311263` |

The Hub read the card's header as intended: the task, the library, the dataset, the
tags, and the licence appear as badges, and it added three of its own (conversational,
the Qwen3 paper's arXiv number, and "Eval Results (legacy)" for the results in the
header). The Hub reported a hash for every large file, so no file was compared by size
alone. The record is `reports/packaging/publish_record.md`. The README now links to
both models; the public code repository gets that change with its next export.
