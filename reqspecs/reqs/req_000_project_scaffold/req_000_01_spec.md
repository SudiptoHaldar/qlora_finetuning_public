# Project Scaffold

Spec Group      :   req_000_project_scaffold  
Feature Name    :   project_scaffold  
Version         :   1  
Status          :   ✅ Complete 2026-11-14 (see `reqspecs/PRPs/prp_000_process_bootstrap_v1.md`)  
Last updated    :   2026-11-14  

## 1. v0.01 — Process bootstrap

The repository was cloned from an earlier project (an SEC EDGAR MCP server with a PostgreSQL
and FastAPI backend). The project name was replaced on 2026-11-13, but the command files, PRP
templates, `coding.md`, and several root files still describe that system. Every PRP written
or executed today has to work around instructions that do not apply: run commands through
`uv`, read an EDGAR guide that is not in the repo, validate an MCP server.

This version makes the process files describe this project and removes the inherited database
stack. It writes no application code.

**Depends on**: nothing.

## 2. What is wrong today (verified 2026-11-14)

| Area | Problem |
|---|---|
| `.claude/commands/generate-prp.md`, `execute-prp.md` | Tell the agent to read `planning_material/edgar_filings_mcp_registry_guide.pdf` (absent), run everything through `uv run`, validate with `mcp dev qlora_finetuning/server.py` (absent), and write runbook entries about SEC access, PyPI, `server.json`, and `mcp-publisher`. State a 500-line file limit; `coding.md` says 1000 |
| `.claude/commands/release-tag.md` | Describes a published package version in `pyproject.toml` and `server.json`, and a PR test plan with `uv run` and MCP Inspector |
| `.claude/commands/wip-roll.md`, `roadmap_view-roll.md` | Release-status sections and examples built around PyPI and MCP Registry publication and SEC data quirks |
| `reqspecs/PRPs/templates/prp_base.md`, `prp_lean.md` | `uv run` commands, `src/` paths, FastAPI and `curl` examples, a `mypy` gate the project does not use |
| `reqspecs/PRPs/templates/GUIDELINES.md` | Examples about charts, date pickers, and UI components |
| `reqspecs/1. PROMPTS/coding.md` | Points at `reqspecs/planning/high_level_plan.md` (the folder is `reqspecs/0. PLANNING/`); lists one module, described as a database and charting backend; test rule is about a live database and `@pytest.mark.db` |
| Root files | `alembic.ini` and an empty `alembic/` folder; `requirements.txt` pins SQLAlchemy, alembic, psycopg, FastAPI, uvicorn, httpx; `.env.example` holds `DATABASE_URL`; `pytest.ini` registers a `db` marker |
| Missing folders | `reqspecs/references/` (both PRP commands read from it), `track_progress/wip/` and `track_progress/roadmap_views/` (the two roll commands write to them) |
| `.gitignore` | Does not cover `artifacts/`, where dataset snapshots, runs, and adapters will go |

The full line-by-line list is in `docs/old_references/stale_project_name_references.md` §3.

Already in place from v0.05, and to be kept: `boto3==1.44.3` and `ruff>=0.17` in
`requirements.txt`; the `network` marker and `addopts = -m "not network"` in `pytest.ini`; a
rebuilt `venv/`; `docs/runbook.md` section 1 (AWS).

## 3. Locked decisions

| # | Decision | Reason |
|---|---|---|
| D1 | Every Python command in commands and templates is written as `venv\Scripts\python -m <tool>` (for example `venv\Scripts\python -m pytest -q`), with the POSIX form `venv/bin/python -m <tool>` given once in `coding.md` | Q6, resolved 2026-11-13: `venv` plus `requirements.txt`, not `uv`. The `python -m` form works without activating the venv |
| D2 | The inherited database stack is deleted, not archived: `alembic.ini`, `alembic/`, the six pins, `DATABASE_URL`, the `db` marker | Q5, resolved 2026-11-13. Storage is file-based only (Parquet, JSONL, YAML) |
| D3 | The file-size guideline is 1000 lines everywhere | `coding.md` is the source of truth for coding rules; the 500 in the two commands is a leftover |
| D4 | The planning material the commands point to is the two guides in `reqspecs/0. PLANNING/planning_material/`, read when a PRP touches training, data preparation, or evaluation | They are the project's actual playbooks |
| D5 | The commands' manual gate becomes: when a script's command line or output changed, run that script once and show its output. Anything that needs network or credentials stays behind `-m network`; anything billed is run by the owner | Replaces the MCP Inspector gate; matches working agreements 9 and 10 |
| D6 | `release_number.md` stays the only release number. All package-version, PyPI, and Registry text is removed from `release-tag.md`. Whether `qlora_finetuning.__version__` mirrors it is decided in the v0.02 spec | There is no published package |
| D7 | In `wip-roll.md` the "Release status" section becomes "Project status": milestones reached (M0 to M7), AWS spend so far against the $300 cap, and operator-gated steps waiting on the owner. In `roadmap_view-roll.md` the operator-gated examples become paid SageMaker and Bedrock launches | These are the things this project waits on |
| D8 | The `pytest.ini` marker set after this version is `network` only. `gpu` and `paid` are added in v0.02 | Keeps v0.02's scope intact; `db` goes with the database stack |
| D9 | `numpy`, `pandas`, `python-dotenv`, `PyYAML`, `pytest`, `ruff`, and `boto3` stay in `requirements.txt`. No ML library is added here | The training stack is pinned in v0.03 (`requirements-train.txt`) |
| D10 | Empty folders are kept in git with a `.gitkeep` file | Git does not track empty folders |
| D11 | Historical documents are not rewritten: `docs/old_references/`, existing PRPs, spec summaries, and implementation summaries keep whatever they say | They record what was true when written |

## 4. Scope to build

### 4.1 Command files

Rewrite the stale parts of the five files below. Their structure, step numbering, output
formats, and file-naming rules do not change.

| File | Change |
|---|---|
| `generate-prp.md` | Step 0 reads the plan, `coding.md`, and (per D4) the two guides. Example technologies and patterns become this project's (boto3, Hugging Face `datasets`, `transformers`, `peft`, `trl`, `bitsandbytes`, Bedrock; split manifest, DSL executor, cost ledger). Validation gates per D1 and D5. Limit per D3 |
| `execute-prp.md` | Same reading list and gates. The runbook step lists what a runbook entry needs here: setup steps, new dependencies (`pip install` plus the `requirements.txt` line), config files touched, how to run the new script, operator-gated or billed steps, troubleshooting. The example runbook entry is replaced with one modelled on the AWS section. "Next steps" examples use D1 commands. The rule that every AWS command is logged in `administration/aws_commands_run.md` is added |
| `release-tag.md` | Per D6. The PR test-plan examples use D1 commands and name no MCP or publish step |
| `wip-roll.md` | Per D7. The "load-bearing knowledge" example becomes a project one (for example a library version quirk or an AWS limit) |
| `roadmap_view-roll.md` | Per D7 |

### 4.2 PRP templates

- `prp_base.md` and `prp_lean.md`: commands per D1; paths use `qlora_finetuning/`, `scripts/`,
  and `tests/`; the FastAPI, `curl`, and `mypy` lines are removed; the integration-points
  example lists `config/*.yaml`, `requirements.txt`, `pytest.ini` markers, and the runbook.
- `GUIDELINES.md`: the chart and UI examples become project examples; nothing else changes.

### 4.3 `coding.md`

- Plan path corrected to `reqspecs/0. PLANNING/high_level_plan.md` (two places).
- Module list replaced with the eight sub-packages from plan §5, plus `scripts/` and `config/`.
- Venv rule states D1.
- Testing rule replaced: tests need no GPU, network, credentials, or paid call; such checks sit
  behind `@pytest.mark.gpu`, `@pytest.mark.network`, or `@pytest.mark.paid` and are deselected
  by default. A note says `gpu` and `paid` are registered in v0.02.
- New short rules taken from the working agreements: no tunable numbers in code; secrets and
  personal details stay out of the repo; every AWS command is logged.

### 4.4 Inherited database stack (D2)

- Delete `alembic.ini` and the empty `alembic/` folder.
- `requirements.txt`: remove `SQLAlchemy`, `alembic`, `psycopg[binary]`, `fastapi`,
  `uvicorn[standard]`, `httpx` and their headings.
- `.env.example`: remove `DATABASE_URL`; keep `LOG_LEVEL`.
- `pytest.ini`: remove the `db` marker (D8).

### 4.5 New folders and files

```
reqspecs/references/
  README.md                  what goes in each folder; the reference-first rule
  common/gotchas.md          seeded, see below
  common/troubleshooting.md  seeded with the venv and sign-in issues met in v0.05
  tech/boto3.md              the response shapes and pitfalls recorded in the v0.05 PRP
  patterns/.gitkeep
  project_patterns/README.md states that no project pattern exists yet
track_progress/wip/.gitkeep
track_progress/roadmap_views/.gitkeep
```

`common/gotchas.md` is seeded from three sources, each entry one or two lines with its source:

- **The two guides**: Qwen3 thinking mode is on by default; silent truncation at the maximum
  sequence length; the completion-only loss API has changed across TRL releases; BF16 versus
  FP16; bitsandbytes on Windows; `max_new_tokens` cutting programs short; never tune against
  the test split.
- **The plan**: Provisioned Throughput cost trap; GPU quota lead time; a custom Nova model
  costs $1.95 per month while kept.
- **v0.05**: us-east-1 is reported as a null region; quota names collide by substring; a YAML
  account id must be quoted; budget amounts come back as text; the Windows console is cp1252.

### 4.6 Other files

- `.gitignore`: add `artifacts/`.
- `docs/runbook.md`: section 1.1 now installs with
  `venv\Scripts\python -m pip install -r requirements.txt`. Stub sections are appended, each a
  heading and one line naming the version that fills it: local GPU environment (v0.03), data
  pipeline (req_001), Path A (req_003), Path B (req_004), comparison and report (req_005).
- `docs/old_references/stale_project_name_references.md`: a closing section records that §3 was
  resolved by this version, with the date. Sections 1 to 5 are left as they are (D11).

## 5. Out of scope

- The `qlora_finetuning/` package, `settings.py`, the `gpu` and `paid` markers, ruff
  configuration, and CI (v0.02).
- ML library pins and `requirements-train.txt` (v0.03).
- `README.md` content (req_006).
- Changes to `.claude/CLAUDE.md`, `.claude/settings.json`, or `.claude/key_value.md`.
- Any change to what the commands produce: file names, headers, and summary formats stay.

## 6. Known limits and follow-ups

- **`venv/Scripts/` still holds launchers from another project** (`alembic.exe`,
  `uvicorn.exe`, and others). They do nothing. Deleting `venv/` and recreating it clears them;
  that is an owner step, noted in the runbook, not done by this version.
- **`gotchas.md` entries from the guides are unverified against installed libraries.** v0.03
  confirms or corrects them when the training stack is pinned.
- **`coding.md` will name markers that do not exist until v0.02.** The note in 4.3 covers the gap.

## 7. Acceptance

- The stale-reference scan returns nothing:

  ```bash
  grep -rniE -- 'uv run|uv sync|uv add|uvx|edgar|mcp|pypi|server\.json|registry|postgres|psycopg|sqlalchemy|fastapi|uvicorn|alembic|DATABASE_URL|mark\.db|500-line|reqspecs/planning/|date picker|chart' \
    .claude/commands reqspecs/PRPs/templates "reqspecs/1. PROMPTS" \
    requirements.txt pytest.ini .env.example
  ```

  "Run registry" is this project's term and would match `registry`; the command files must
  word around it or the PRP narrows the pattern to `MCP Registry`.
- `alembic.ini` and `alembic/` no longer exist.
- `venv\Scripts\python -m pytest -q` still passes (44 tests) and
  `venv\Scripts\python -m ruff check .` is clean.
- `venv\Scripts\python -m pip install -r requirements.txt` succeeds in the existing venv.
- Every path in section 4.5 exists; `git status` shows the new folders as trackable.
- `/generate-prp` and `/execute-prp` contain no instruction that names a file, tool, or command
  absent from this repository.
- Milestone M0 (process ready) is reached.

## 8. Next step

```
/generate-prp reqspecs/reqs/req_000_project_scaffold --version 1
```
