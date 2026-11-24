# PRP: Package skeleton, settings, CI (req_000 v0.02)

**Feature Name**: package_skeleton (project_scaffold v0.02)  
**Version**: 2  
**Created**: 2026-11-14  
**Status**: ✅ IMPLEMENTED  
**Implemented**: 2026-11-14  
**Implementation Summary**: [impl_000_20261114_233357.md](../impl_summary/impl_000_20261114_233357.md)  
**Confidence Score**: 8/10  
**Estimated Time**: 2–3 hours  
**Origin**: `reqspecs/reqs/req_000_project_scaffold/req_000_02_spec.md`  
**Template**: `prp_lean.md` (expanded, medium complexity)  

---

## Goal

Create the `qlora_finetuning` package with eight empty sub-packages, `settings.py`
(environment accessors and `load_config()`), `logging_setup.py`, and `__version__`. Move
`scripts/aws_check.py` onto the shared loader and run scripts as modules. Add the `gpu` and
`paid` markers, a ruff configuration, and a GitHub Actions workflow.

## Why & What

- **User value**: every later version has one package to import from, one way to read config,
  and the same lint and test gate locally and on each pull request.
- **Technical**: 3 package modules, 8 sub-package stubs, 3 new test files, `ruff.toml`,
  `.github/workflows/ci.yml`, a loader swap in `aws_check`, and command-form updates in live
  documents.

## Success Criteria

- [ ] `venv\Scripts\python -m pytest -q` passes with no GPU, network, or credentials.
- [ ] `venv\Scripts\python -m ruff check .` is clean under the new `ruff.toml`.
- [ ] `venv\Scripts\python -m scripts.aws_check` prints PASS for C1–C7 against the live account.
- [ ] `venv\Scripts\python -c "import qlora_finetuning; print(qlora_finetuning.__version__)"`
      prints the content of `release_number.md`.
- [ ] `load_config("aws")` returns the committed values; a missing required key raises
      `ConfigError` naming every missing key.
- [ ] No live document tells the reader to run a script by file path (enforced by a test).
- [ ] `.github/workflows/ci.yml` parses and its three commands pass when run locally.
- [ ] **Owner step, cannot be done by the agent**: the workflow finishes green on the pull
      request that carries this version.

---

## Context

### References

```yaml
refs:
  - reqspecs/references/common/gotchas.md        # section 4, local environment
  - reqspecs/references/common/troubleshooting.md
  - reqspecs/references/project_patterns/README.md  # read-only check script candidate; unchanged here
  - url: https://docs.astral.sh/ruff/configuration/
    why: ruff.toml layout (top-level settings, [lint], [lint.isort])
  - url: https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions
    why: on, permissions, concurrency keys
  - url: https://github.com/actions/setup-python#caching-packages-dependencies
    why: `cache: pip` keys on requirements.txt

code:
  - file: scripts/aws_check.py
    why: load_aws_config(), _lookup(), ConfigError move into settings.py almost unchanged
  - file: tests/scripts/test_aws_check.py
    why: test style (MagicMock stubs, capsys, monkeypatch); config tests need a `version` key
  - file: tests/test_process_scaffold.py
    why: guard-test style; marker test and a new by-path scan go here
```

### Facts verified 2026-11-14

- Latest action releases: `actions/checkout` v7.0.1, `actions/setup-python` v7.1.0. Pin `@v7`.
- GitHub Actions is enabled for the private repository `SudiptoHaldar/qlora_finetuning_public`.
- The `protect-main` ruleset is active: deletion, non_fast_forward, pull_request (zero
  approvals). It requires no status check.
- Under the planned rules (`E,W,F,I,UP,B`, line length 100, target `py310`) ruff reports
  12 findings in existing code, all `E501`.
- `release_number.md` holds `0.2`.
- `python-dotenv` and `PyYAML` are installed and already in `requirements.txt`. No new
  dependency is needed.

### Critical Gotchas

- **Do not write `.env` values into `os.environ`.** Use `dotenv_values()` into a module-level
  dict and look up `os.environ` first. This gives "does not override variables already set"
  for free and keeps tests from leaking environment changes.
- **`bool` is an `int` in Python.** `version: true` must be rejected: check
  `isinstance(v, int) and not isinstance(v, bool)`.
- **Existing config tests write YAML without a `version` key.** Add `"version": 1` to the
  `cfg` fixture in `tests/scripts/test_aws_check.py`, or three tests fail for the wrong reason.
- **`load_config()` builds the file name as `<name>.yaml`.** For `--config <path>`,
  `aws_check` passes the path's stem and parent, and rejects a path that does not end in
  `.yaml` with `ConfigError`.
- **A script run by file path cannot import the package.** Catch that one
  `ModuleNotFoundError` at the top of `aws_check.py` and exit with a one-line hint. This is
  friendlier than the bare traceback the spec describes.
- **Log level lookup.** `logging.getLevelNamesMapping()` needs Python 3.11; the ruff target is
  3.10. Use an explicit dict of the five standard levels.
- **pytest captures `sys.stderr` per test.** Create the `StreamHandler(sys.stderr)` inside
  `setup_logging()`, and remove it in a test fixture, so `capsys` sees the output and no
  handler outlives its test.
- **Keep `qlora_finetuning/__init__.py` light.** It reads one file and imports nothing from
  the package, so `import qlora_finetuning` never pulls in YAML or boto3.
- **ruff excludes `venv` by default.** Only `artifacts` and `track_progress` need adding.
- **The guard test scans command files for `pyproject`.** Do not mention that word in
  `coding.md`, the commands, or the templates.
- **CI on Linux is case-sensitive** and has no `.env`, no AWS profile, and no GPU. Every test
  must pass in that state; the subprocess test must use `sys.executable`, not a venv path.

---

## Implementation

### Files

```yaml
CREATE:
  - qlora_finetuning/__init__.py: __version__ from release_number.md, fallback "0+unknown"
  - qlora_finetuning/settings.py: REPO_ROOT, CONFIG_DIR, ConfigError, get_str, get_int, load_config
  - qlora_finetuning/logging_setup.py: setup_logging, get_logger
  - qlora_finetuning/{tracking,data,dsl,evaluation,local_qlora,frontier,comparison,demo}/__init__.py: docstring only
  - tests/{tracking,data,dsl,evaluation,local_qlora,frontier,comparison,demo}/__init__.py: empty
  - tests/test_package.py
  - tests/test_settings.py
  - tests/test_logging_setup.py
  - ruff.toml
  - .github/workflows/ci.yml
  - reqspecs/impl_summary/impl_000_<TIMESTAMP>.md

MODIFY:
  - scripts/aws_check.py: use settings.load_config and settings.ConfigError; by-path hint; docstring
  - tests/scripts/test_aws_check.py: version in fixture; two subprocess tests; long lines
  - tests/test_process_scaffold.py: three markers; scan for by-path script commands
  - pytest.ini: gpu and paid markers; all three deselected; --strict-markers
  - docs/runbook.md: module-form commands; new "Tests, lint, and CI" section; stubs renumbered
  - reqspecs/1. PROMPTS/coding.md: settings, markers, and script form stated as current; logging rule
  - .claude/commands/generate-prp.md, execute-prp.md: script examples use -m scripts.<name>
  - .claude/commands/release-tag.md: ruleset paragraph states the verified state; __version__ note
  - reqspecs/PRPs/templates/prp_base.md, prp_lean.md: script examples use -m scripts.<name>
  - reqspecs/references/common/gotchas.md, troubleshooting.md: module-form entries
  - roadmap, execution plan, spec status
```

### Pseudocode

```python
# qlora_finetuning/__init__.py
RELEASE_FILE = Path(__file__).resolve().parents[1] / "release_number.md"
UNKNOWN_VERSION = "0+unknown"
def _read_version(path=RELEASE_FILE) -> str:
    try: text = path.read_text(encoding="utf-8").strip()
    except OSError: return UNKNOWN_VERSION
    return text or UNKNOWN_VERSION
__version__ = _read_version()

# qlora_finetuning/settings.py
REPO_ROOT = Path(__file__).resolve().parents[1]; CONFIG_DIR = REPO_ROOT / "config"
ENV_FILE = REPO_ROOT / ".env"; _dotenv: dict | None = None
class ConfigError(Exception): ...
def _env(name):                       # os.environ wins; then .env (read once); "" counts as unset
    global _dotenv
    value = os.environ.get(name)
    if value: return value
    if _dotenv is None: _dotenv = dict(dotenv_values(ENV_FILE)) if ENV_FILE.is_file() else {}
    return _dotenv.get(name) or None
def get_str(name, default=None): value or default, else ConfigError(f"environment variable {name} is not set")
def get_int(name, default=None): int(value) -> ConfigError naming variable and value on ValueError
def load_config(name, required=(), config_dir=None) -> dict:
    path = Path(config_dir or CONFIG_DIR) / f"{name}.yaml"
    # ConfigError for: file not found; invalid YAML; not a mapping;
    # version missing or not a non-bool int; any required dotted key that is None, "" or []
    # (message lists every missing key). Returns the freshly parsed dict.

# qlora_finetuning/logging_setup.py
LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
HANDLER_NAME = "qlora_finetuning"
def setup_logging(level=None) -> int:
    name = (level or get_str("LOG_LEVEL", "INFO")).upper()
    if name not in LEVELS: raise ConfigError(...)
    root = logging.getLogger()
    if no handler on root named HANDLER_NAME: add StreamHandler(sys.stderr) with FORMAT, that name
    root.setLevel(LEVELS[name]); return LEVELS[name]
def get_logger(name): return logging.getLogger(name)

# scripts/aws_check.py (top of file)
try:
    from qlora_finetuning.settings import CONFIG_DIR, ConfigError, load_config
except ModuleNotFoundError as exc:
    if exc.name != "qlora_finetuning": raise
    raise SystemExit("Run this script as a module from the repository root: "
                     "python -m scripts.aws_check") from None
DEFAULT_CONFIG = CONFIG_DIR / "aws.yaml"
def load_aws_config(path=DEFAULT_CONFIG):
    path = Path(path)
    if path.suffix != ".yaml": raise ConfigError(f"config file must end in .yaml: {path}")
    cfg = load_config(path.stem, required=REQUIRED_KEYS, config_dir=path.parent)
    cfg["account_id"] = str(cfg["account_id"]); return cfg
```

### `ruff.toml`

```toml
line-length = 100
target-version = "py310"
extend-exclude = ["artifacts", "track_progress"]

[lint]
select = ["E", "W", "F", "I", "UP", "B"]

[lint.isort]
known-first-party = ["qlora_finetuning", "scripts"]
```

### `.github/workflows/ci.yml`

```yaml
name: ci
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]
permissions:
  contents: read
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true
jobs:
  ci:
    name: lint and test
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: "3.13"
          cache: pip
      - name: Install
        run: python -m pip install -r requirements.txt
      - name: Lint
        run: python -m ruff check .
      - name: Test
        run: python -m pytest -q
```

### `pytest.ini`

```ini
[pytest]
addopts = --strict-markers -m "not network and not gpu and not paid"
markers =
    network: needs network access and a valid AWS sign-in; deselected by default, run with -m network
    gpu: needs a CUDA GPU; deselected by default, run with -m gpu
    paid: makes a billed AWS call; deselected by default, run only by the owner with -m paid
```

### Tests

| File | Cases |
|---|---|
| `tests/test_package.py` | `__version__` equals stripped `release_number.md`; `_read_version` on a missing file and on an empty file returns `0+unknown`; each of the eight sub-packages imports and has a docstring; importing the package does not import `yaml` or `boto3` (checked in a subprocess) |
| `tests/test_settings.py` | `get_str`: from environment, default, missing raises, empty string counts as unset. `get_int`: value, default, non-integer raises naming the variable. `.env`: value read from a temp `ENV_FILE`; environment wins over `.env`; `os.environ` is not modified. `load_config`: committed `aws.yaml`; missing file; invalid YAML; non-mapping; missing `version`; `version: true`; `version: "1"`; all missing required keys listed; empty string and empty list count as missing; `0` and `false` do not; two calls return different objects |
| `tests/test_logging_setup.py` | Level from argument, from `LOG_LEVEL`, default INFO; lower-case accepted; unknown level raises; second call adds no handler and changes the level; a logged line appears on stderr and not on stdout; `get_logger` returns the named logger |
| `tests/scripts/test_aws_check.py` | Existing 44 pass. New: `python -m scripts.aws_check --config <missing>.yaml` exits 2 with `CONFIG ERROR`; running the file by path exits non-zero with the "as a module" hint; `--config x.yml` gives `CONFIG ERROR` |
| `tests/test_process_scaffold.py` | Markers: all three registered and deselected. New scan: no `python scripts\` or `python scripts/` in the commands, templates, `coding.md`, runbook, references, or `scripts/*.py` |

Fixtures: `monkeypatch.setattr(settings, "ENV_FILE", tmp)` and
`monkeypatch.setattr(settings, "_dotenv", None)` reset the `.env` cache; a logging fixture
removes the named handler and restores the root level after each test.

### Tasks (~2.5 hours total)

1. **Package** (~30 min): `__init__.py`, `settings.py`, `logging_setup.py`, eight stubs.
2. **Package tests** (~35 min): three test files, eight empty test packages.
3. **`aws_check` swap** (~20 min): loader, error class, hint, docstring; update its tests.
4. **Markers and ruff** (~20 min): `pytest.ini`, `ruff.toml`, fix the 12 long lines and any
   import-order findings.
5. **CI workflow** (~10 min): write the file; parse it with PyYAML; run its three commands.
6. **Documents** (~25 min): runbook, `coding.md`, commands, templates, references.
7. **Guard tests** (~10 min).
8. **Validation, live run, status documents** (~20 min).

### Error handling

| Situation | Behavior |
|---|---|
| Environment variable unset and no default | `ConfigError` naming the variable |
| Non-integer value for `get_int` | `ConfigError` naming the variable and the value |
| Config file missing, invalid, not a mapping, bad `version`, or missing keys | `ConfigError` with the path and, for keys, the full list |
| Unknown log level | `ConfigError` listing the allowed names |
| Script run by file path | One-line hint, non-zero exit, no traceback |
| `aws_check` behavior | Unchanged: exit 0, 1, or 2 as before |

---

## Validation

```powershell
# Lint and tests (exactly what CI runs)
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q

# Package and loader
venv\Scripts\python -c "import qlora_finetuning; print(qlora_finetuning.__version__)"
venv\Scripts\python -c "from qlora_finetuning.settings import load_config; print(load_config('aws')['bucket']['name'])"

# Workflow file parses
venv\Scripts\python -c "import yaml; print(list(yaml.safe_load(open('.github/workflows/ci.yml'))['jobs']))"

# Script, module form (read-only; needs a valid sign-in) and the live test
venv\Scripts\python -m scripts.aws_check
venv\Scripts\python -m pytest -m network -q
```

**Expected**: ruff clean; all tests pass; `0.2`; the bucket name; `['ci']`; seven PASS lines.

The CI run itself needs a push and a pull request, which the owner makes. Pushing a workflow
file needs a GitHub token with the `workflow` scope.

---

## Checklist

- [ ] Files created and modified as listed
- [ ] Each new function has expected, edge, and failure tests
- [ ] Lint clean, tests green, `aws_check` 7/7 in module form
- [ ] Live documents use the module form; guard test added
- [ ] Roadmap, execution plan, spec status updated in the same turn
- [ ] Implementation summary created; PRP status set

---

## Implementation Time Estimate

**Total: 2–3 hours**

| Task | Time |
|---|---|
| Package modules and stubs | 30 min |
| Package tests | 35 min |
| `aws_check` loader swap and tests | 20 min |
| Markers, `ruff.toml`, lint fixes | 20 min |
| CI workflow | 10 min |
| Documents | 25 min |
| Guard tests | 10 min |
| Validation, live run, status documents | 20–40 min |

Risk: +20 min if the new rule sets flag more than the 12 known lines once the new files exist.

## Confidence Score: 8/10

The code is small and the patterns already exist in `aws_check`. Two points are held back: the
CI workflow cannot be run from here, so its first real run may need a fix, and the change
touches many documents, where a missed by-path command is likely on the first pass (the new
guard test catches it).

## Time Taken for PRP Generation

**~9 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 1 min |
| Codebase exploration (script-path mentions, ruff preview) | 2 min |
| Research existing PRPs and patterns (action versions, ruleset state) | 1 min |
| Writing PRP document | 5 min |
