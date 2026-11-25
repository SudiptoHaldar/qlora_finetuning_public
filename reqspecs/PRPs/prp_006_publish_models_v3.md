# PRP: Publish the adapter and the merged copy (req_006 v6.03)

**Feature Name**: publish_models (packaging_demo v6.03)  
**Version**: 3  
**Created**: 2026-11-24  
**Status**: ✅ IMPLEMENTED (the uploads are the owner's)  
**Implemented**: 2026-11-24  
**Implementation Summary**: [impl_006_20261124_093000.md](../impl_summary/impl_006_20261124_093000.md)  
**Confidence Score**: 8/10  
**Estimated Time**: 2–3 hours of code, tests, and documents; the uploads and the checks on the Hub are the owner's and take about 30 minutes plus the time to send 8 GB  
**Origin**: `reqspecs/reqs/req_006_packaging_demo/req_006_03_spec.md`  

Template: `prp_lean.md` (medium: two modules and a script over what v6.02 built).

## Goal

`venv\Scripts\python -m scripts.hf_publish` uploads the two package folders of v6.02 to
private repositories on the Hugging Face Hub, and proves that what is there is what was
built: the same files by name, size, and hash, and the card's code running by repository
name. No command makes a repository public.

## Why

- The owner decided to publish, so that others can use the tuned model.
- Both packages are ready and the code repository is public: the cards' links resolve.

## What

All six decisions of spec section 8 are taken as recommended (2026-11-24): the owner runs
the uploads with `--yes`; private first; the owner makes each repository public on the Hub;
the upload is checked while private, by the owner, and once more after publishing without
a token; both packages in one sitting, the adapter first; default repository settings.

Success criteria

- [x] `pytest -q` and `ruff check .` pass, with and without the training stack, with no
      network and no token.
- [x] `hf_publish plan` on the real repository lists both packages and passes U1.
- [x] `upload` is refused without `--yes` and without a token; the token is never an
      argument, never printed, never in a record.
- [x] `check` compares the Hub with the package record and runs the card's code by name.
- [x] Nothing frozen changed. Claude starts no upload.

## All Needed Context

| File | Why |
|---|---|
| `reqspecs/reqs/req_006_packaging_demo/req_006_03_spec.md` | D1 to D18, checks U1 to U7, the owner's steps (section 9) |
| `reports/packaging/package_record.json` | `adapter.files` and `merged.files` (`name`, `bytes`, `sha256`), `adapter.status`, `merged.status`, `passed` |
| `qlora_finetuning/packaging/config.py` | `PackagingConfig`: `adapter_dir`, `merged_dir`, `adapter_repo`, `merged_repo`, `report_dir` |
| `qlora_finetuning/packaging/verify.py` | `run_card_code(code, values, timeout=...)` sets `HF_HUB_OFFLINE`; it needs a way to run with the Hub on |
| `qlora_finetuning/packaging/workflow.py`, `cards.py` | `usage_code(kind, cfg, budget, local_cfg)`, `check_card_run`, `load_inputs` |
| `qlora_finetuning/packaging/files.py`, `record.py` | `sha256_file`, `write_json_lf`; `check_result`; the record's shape |
| `tests/packaging/test_guards_and_script.py` | The guard that forbids upload calls in `packaging/`: it must allow them in `hub.py` only |
| `qlora_finetuning/settings.py` | `get_str` reads the environment, then `.env` |
| `reqspecs/references/project_patterns/release_package.md` | The conventions for something that leaves the project |
| https://huggingface.co/docs/huggingface_hub/guides/upload | `upload_folder`; resuming |
| https://huggingface.co/docs/huggingface_hub/package_reference/hf_api | `model_info(files_metadata=True)`, `create_repo(private=True)` |

### Facts found (2026-11-24, `huggingface_hub` 2.4.0, read from the installed library)

- `HfApi(token=...)`: `create_repo(repo_id, private=True, exist_ok=True)`;
  `upload_folder(repo_id=, folder_path=, commit_message=)` returns the commit;
  `model_info(repo_id, files_metadata=True)` gives `.sha`, `.private`, and `.siblings`, each
  with `rfilename`, `size`, `blob_id`, and `lfs` (`sha256`, `size`) for a large file;
  `hf_hub_download(repo_id, filename, revision=, local_dir=)`.
- `HfApi(token=False)` and `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` switch off a token saved by
  `hf auth login`, so a check "without a token" really has none.
- A missing or private repository raises `RepositoryNotFoundError` for a caller without
  access: the two cases look the same from outside.
- **The Hub adds `.gitattributes` to every repository.** It is not a file of the package and
  is allowed.
- `settings._env` reads `.env` whole on the first lookup. The tool asks for `HF_TOKEN` only
  when a repository cannot be read without it.

### Gotchas

- **The token**: in memory and in a child process's environment only. Never in an argument
  list, a log line, an exception message that is printed, or a file. Every printed line and
  the record pass through `redact`.
- **An agent never runs `upload`**, and never a `check` that needs the token.
- **Imports**: `huggingface_hub` is imported inside functions of `hub.py` only. The guard
  test of `packaging/` is widened for that one module and for upload names in it; deleting
  a repository or changing its visibility is forbidden everywhere.
- **`run_card_code` with the Hub on** must not inherit a saved login when no token is meant.
- **The merged card's code downloads 8 GB** on its first run from the Hub: a long timeout.
- **Windows console is cp1252**: ASCII output. Records are UTF-8 with LF.

## Implementation Blueprint

```
config/packaging.yaml              publish: token_env, hub_files, card_code_timeout_seconds
qlora_finetuning/packaging/hub.py      HubClient over HfApi: state, create_private, upload, fetch
qlora_finetuning/packaging/publish.py  U1 to U7, the comparison, the record, redact
scripts/hf_publish.py                  plan | upload {adapter,merged} --yes | check [adapter|merged]
tests/packaging/test_publish.py        a stand-in Hub that keeps files in memory
```

```python
state = client.state(repo_id)            # None: not there, or not visible to this caller
local_problems(package)                  # U1: the record says ready; every file has its recorded size and hash
foreign_files(state, package)            # U2: a file on the Hub that is not the package's
client.create_private(repo_id); client.upload(repo_id, folder, message)
remote_problems(state, package, fetch)   # U4: names, sizes, hashes; .gitattributes allowed
card_runs(package, online=True, token)   # U5: the card's code by repository name
```

Tasks, in order

1. Config keys and loader; `hub.py`; `publish.py`; `run_card_code(online=, token=)`.
2. `scripts/hf_publish.py`.
3. Tests with the stand-in Hub; the guard test widened.
4. `hf_publish plan` on the real repository.
5. Runbook, `coding.md`, architecture; spec section 11, roadmap, execution plan,
   implementation summary. The version ends 🟡: the uploads are the owner's.
6. After the owner publishes: the token-free `check`, the README (D16), one more export.

Error handling: a refusal is a `ConfigError` on one line (exit code 2); a failed check is
exit code 1; a library error is caught, redacted, and reported as a failed check.

## Validation Loop

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q
venv\Scripts\python -m scripts.hf_publish plan
venv\Scripts\python -m scripts.comparison_freeze status
```

`upload` and a `check` of a private repository are the owner's. Nothing is billed.

## Anti-Patterns to Avoid

- `--token` on a command line, or the token in an f-string that is printed.
- Uploading a folder whose files were not compared with the package record.
- A command that makes a repository public, deletes one, or edits a card on the Hub.
- Calling an upload from anywhere but `hub.py`.

## Implementation Time Estimate

**2–3 hours.** The owner's part: about 30 minutes, plus the upload of 8 GB.

| Task | Time |
|---|---|
| `hub.py`, `publish.py`, config, `run_card_code` | 60 min |
| The script | 20 min |
| Tests | 50 min |
| Documents | 30 min |

## Confidence

**8/10.** The library's calls were read from the installed version. What cannot be tried
here is the Hub itself: how it reports a large file's hash, and what its card checker says
about the header.

## Time Taken for PRP Generation

**~8 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 1 min |
| Codebase exploration | 3 min |
| Research existing PRPs and patterns | 1 min |
| Writing PRP document | 3 min |
