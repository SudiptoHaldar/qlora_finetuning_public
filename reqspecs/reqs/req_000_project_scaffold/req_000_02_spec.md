# Project Scaffold

Spec Group      :   req_000_project_scaffold  
Feature Name    :   project_scaffold  
Version         :   2  
Status          :   ✅ Complete 2026-11-15 (see `reqspecs/PRPs/prp_000_package_skeleton_v2.md`); live `aws_check` 7/7; CI green on pull request #1 and on `main`  
Last updated    :   2026-11-14  

## 1. v0.02 — Package skeleton, settings, CI

Every later version imports from a `qlora_finetuning` package, reads tunable values through
one config loader, and relies on tests and lint running the same way locally and on GitHub.
None of that exists yet: there is no package, `scripts/aws_check.py` carries its own YAML
loader, ruff runs on default rules, and nothing runs on a pull request.

This version adds the package with its eight empty sub-packages, `settings.py`, logging, the
`gpu` and `paid` test markers, a ruff configuration, and a GitHub Actions workflow. It moves
`aws_check` onto the shared loader.

**Depends on**: v0.01 (done).

## 2. What exists today (verified 2026-11-14)

| Item | State |
|---|---|
| `qlora_finetuning/` | Does not exist |
| `config/aws.yaml` | Exists, with a top-level `version: 1` |
| `scripts/aws_check.py` | Works (7/7). Has a stand-in `load_aws_config()` and its own `ConfigError` |
| `tests/` | 65 tests in `tests/scripts/test_aws_check.py` and `tests/test_process_scaffold.py` |
| `pytest.ini` | Registers `network`; `addopts = -m "not network"` |
| ruff | Installed (`ruff>=0.17`), no configuration file |
| `requirements.txt` | numpy, pandas, python-dotenv, PyYAML, boto3, pytest, ruff |
| `release_number.md` | `0.2` |
| `.github/` | Does not exist. GitHub Actions is enabled for the repository |
| Branch ruleset `protect-main` | Exists and is active: blocks deletion and force push, requires a pull request with zero approvals. It requires no status check |
| Repository | `SudiptoHaldar/qlora_finetuning_public`, private |

## 3. Locked decisions

| # | Decision | Reason |
|---|---|---|
| D1 | Config files keep plain names (`config/aws.yaml`) and carry an integer `version:` key inside. `load_config("aws")` reads `config/aws.yaml` | Agreed 2026-11-14. Matches the file that exists; git history holds older versions |
| D2 | `qlora_finetuning.__version__` is read from `release_number.md`. If that file is absent the value is `"0+unknown"` | Agreed 2026-11-14. One release number; `/release-tag` is unchanged. The fallback covers a copy of the package shipped without the repository root (the SageMaker job, v3.03) |
| D3 | Scripts are run as modules: `venv\Scripts\python -m scripts.aws_check`. Running a script by file path is no longer supported | Agreed 2026-11-14. Module form puts the repository root on the import path, so a script can import `qlora_finetuning` without packaging or an install step |
| D4 | The AWS check code stays in `scripts/aws_check.py`. Only its config loading changes | Agreed 2026-11-14. Nothing else needs those checks |
| D5 | ruff is configured in `ruff.toml`: line length 100, rule sets `E`, `W`, `F`, `I`, `UP`, `B` | Agreed 2026-11-14. Existing code is written to about 100 columns |
| D6 | CI is one GitHub Actions workflow on pull requests to `main` and pushes to `main`: install `requirements.txt`, run ruff, run pytest | Agreed 2026-11-14 |
| D7 | No `pyproject.toml`, no editable install, no build step | Follows from D3. The project publishes no package |
| D8 | CI runs on `ubuntu-latest` with Python 3.13 only | Linux minutes cost half of Windows minutes on a private repository, and the measured runs are on Linux. Windows is covered by local runs |
| D9 | CI uses no secret and no AWS credential, and declares `permissions: contents: read` | Working agreement 9. The deselected markers keep network, GPU, and paid tests out |
| D10 | `load_config()` returns a new plain `dict` on every call and does no caching. It fails with `ConfigError` on a missing file, invalid YAML, a non-mapping file, a missing or non-integer `version`, or any missing key the caller listed as required | "Missing key fails loudly" is the plan's check for this version. No cache keeps tests independent |
| D11 | ruff `target-version` is `py310` | The Python version of the SageMaker training image is not known until v3.03; 3.10 is a safe floor. Revisit there |
| D12 | Logging goes to standard error | Script output on standard output (for example the PASS and FAIL lines) stays clean |

## 4. Scope to build

### 4.1 Package

```
qlora_finetuning/
  __init__.py        __version__ (D2)
  settings.py        4.2
  logging_setup.py   4.3
  tracking/__init__.py     run registry, cost ledger (v0.04)
  data/__init__.py         FinQA snapshot, split manifest, serialization, token budget (req_001)
  dsl/__init__.py          program parser and safe executor (req_002)
  evaluation/__init__.py   metrics, Predictor protocol, evaluation runner (req_002)
  local_qlora/__init__.py  Path A (req_003)
  frontier/__init__.py     Path B (req_004)
  comparison/__init__.py   cost model, statistics, report builder (req_005)
  demo/__init__.py         demo CLI (req_006)
```

Each sub-package `__init__.py` holds a one-line docstring naming what it will contain and the
spec group that fills it. No other code.

### 4.2 `settings.py`

| Name | Behavior |
|---|---|
| `REPO_ROOT`, `CONFIG_DIR` | Paths to the repository root and `config/` |
| `ConfigError` | Raised for every configuration problem below |
| `get_str(name, default=None)` | Returns the environment variable. `.env` at the repository root is loaded once, without overriding variables already set. With no value and no default, raises `ConfigError` naming the variable |
| `get_int(name, default=None)` | As `get_str`, converted to `int`. A non-integer value raises `ConfigError` naming the variable and the value |
| `load_config(name, required=(), config_dir=None)` | Per D1 and D10. `required` is a list of dotted keys such as `"bucket.name"`; an empty string, empty list, or null counts as missing. The error message lists every missing key |

### 4.3 `logging_setup.py`

- `setup_logging(level=None)`: configures the root logger once (calling it again changes only
  the level). The level comes from the argument, else `LOG_LEVEL`, else `INFO`. An unknown
  level name raises `ConfigError`. Format: time, level, logger name, message. Output per D12.
- `get_logger(name)`: returns `logging.getLogger(name)`.

### 4.4 `scripts/aws_check.py`

- `load_aws_config()` calls `load_config()` with the existing required-key list and keeps the
  account-id coercion to `str`. The stand-in YAML code is removed.
- `aws_check.ConfigError` becomes the `settings.ConfigError` class, under the same name.
- `--config <path>` keeps working: the file's folder and stem are passed to `load_config()`.
- Output, checks, and exit codes do not change.

### 4.5 Tests and markers

- `pytest.ini` registers `gpu` and `paid` beside `network`, and deselects all three by default.
- `tests/` gains an empty package folder for each sub-package, plus:

| File | Covers |
|---|---|
| `tests/test_package.py` | `__version__` equals the content of `release_number.md`; the fallback when the file is absent; every sub-package imports |
| `tests/test_settings.py` | `get_str` and `get_int`: value, default, missing, non-integer; `.env` loading without override. `load_config`: the committed `aws.yaml`, each failure in D10, the full list of missing keys, a fresh dict per call |
| `tests/test_logging_setup.py` | Level from argument, from `LOG_LEVEL`, default; unknown level; no duplicate handlers on a second call; output on standard error |
| `tests/scripts/test_aws_check.py` | Existing tests keep passing. One new test runs `python -m scripts.aws_check --config <missing file>` as a subprocess and expects exit code 2 and `CONFIG ERROR` (no network involved) |
| `tests/test_process_scaffold.py` | The marker test expects `network`, `gpu`, and `paid` |

### 4.6 ruff

`ruff.toml` per D5 and D11, with `venv`, `artifacts`, and `track_progress` excluded. Existing
files are brought into line (import order and any line over 100 columns). No rule is switched
off for a whole file.

### 4.7 CI

`.github/workflows/ci.yml` per D6, D8, D9:

- Triggers: `pull_request` targeting `main`; `push` to `main`.
- One job: check out, set up Python 3.13 with pip caching, `pip install -r requirements.txt`,
  `python -m ruff check .`, `python -m pytest -q`.
- A concurrency group cancels a superseded run on the same branch.
- Action versions are checked on the day the PRP is written and pinned to a major version.

### 4.8 Documentation

Live documents switch to the module form of D3 and drop their "arrives in v0.02" notes:

- `docs/runbook.md`: commands; a short "Tests, lint, CI" part; how to read a failed
  CI run.
- `reqspecs/1. PROMPTS/coding.md`: `settings.py`, markers, and script command form are stated
  as current.
- `.claude/commands/generate-prp.md`, `execute-prp.md`, the two PRP templates: script examples
  use `-m scripts.<name>`.
- `.claude/commands/release-tag.md`: the "if no ruleset exists yet" paragraph is replaced with
  the ruleset's actual state (section 2) and the note that `__version__` follows
  `release_number.md` by itself.
- `reqspecs/references/`: the script command in `troubleshooting.md` and `tech/boto3.md`; a
  gotcha that scripts must be run with `-m`.

Historical documents (earlier PRPs, spec summaries, implementation summaries, the AWS command
log's past entries) are not rewritten.

## 5. Out of scope

- Any code inside the eight sub-packages.
- `pyproject.toml`, packaging, type checking, pre-commit hooks.
- The training stack and `requirements-train.txt` (v0.03).
- Windows or GPU runners in CI.
- Changing the `protect-main` ruleset (see section 6).

## 6. Known limits and follow-ups

- **CI was not a required check when this version shipped.** Closed 2026-11-15: the
  `protect-main` ruleset now requires the `lint and test` check.
- **CI does not run on feature-branch pushes**, only on pull requests and on `main`. A branch
  without an open pull request is checked locally.
- **Actions minutes.** The repository is private, so each run uses the organisation's included
  minutes. A run is expected to take about one minute.
- **`python scripts\aws_check.py` stops working.** It fails with `ModuleNotFoundError` for
  `qlora_finetuning`. The runbook and the troubleshooting reference say so.
- **`__version__` in the cloud.** A training job that receives only the package folder reports
  `0+unknown`. v3.03 decides whether to ship `release_number.md` with the job.

## 7. Acceptance

- `venv\Scripts\python -m pytest -q` passes with no GPU, network, or credentials.
- `venv\Scripts\python -m ruff check .` is clean under the new `ruff.toml`.
- `venv\Scripts\python -m scripts.aws_check` prints PASS for C1–C7 against the live account.
- `venv\Scripts\python -c "import qlora_finetuning; print(qlora_finetuning.__version__)"`
  prints the content of `release_number.md`.
- `load_config("aws")` returns the committed values; a missing required key raises
  `ConfigError` naming it.
- The CI workflow runs on the pull request that carries this version and finishes green.
- No live document tells the reader to run a script by file path.

## 8. Next step

```
/generate-prp reqspecs/reqs/req_000_project_scaffold --version 2
```
