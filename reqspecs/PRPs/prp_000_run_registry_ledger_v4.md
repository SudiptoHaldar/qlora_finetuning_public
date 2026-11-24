# PRP: Run registry and cost ledger (req_000 v0.04)

**Feature Name**: run_registry_ledger (project_scaffold v0.04)  
**Version**: 4  
**Created**: 2026-11-15  
**Status**: ✅ IMPLEMENTED  
**Implemented**: 2026-11-15  
**Implementation Summary**: [impl_000_20261115_011500.md](../impl_summary/impl_000_20261115_011500.md)  
**Confidence Score**: 9/10  
**Estimated Time**: 3–4 hours  
**Origin**: `reqspecs/reqs/req_000_project_scaffold/req_000_04_spec.md`  
**Template**: `prp_lean.md` (expanded, medium complexity)  

---

## Goal

Fill `qlora_finetuning/tracking/` with a run registry (id, folder, manifest per run) and an
append-only cost ledger of billed quantities, plus `scripts/ledger_summary.py`. No AWS call,
no prices, no dollars derived from rates.

## Why & What

- **User value**: every cost figure in the final report traces to a ledger line, and every
  ledger line to a run, a commit, and a configuration.
- **Technical**: four small modules, one config file, one script, a committed `ledger/`
  folder, tests in temporary folders.

## Success Criteria

- [ ] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, also
      with the training libraries blocked (what CI sees).
- [ ] A dummy run in a temporary folder leaves a run folder with `manifest.json` and config
      copies, two index lines, and ledger records; `summarize` returns the totals written.
- [ ] A `measured` run refuses a working tree with uncommitted changes; the override is
      recorded in the manifest.
- [ ] Each validation rule in spec 4.3 rejects its violation and writes nothing.
- [ ] Both mapping functions produce valid records from AWS-shaped dicts.
- [ ] `venv\Scripts\python -m scripts.ledger_summary` prints "no records" on the empty ledger.
- [ ] No module in `qlora_finetuning/tracking/` imports boto3.

---

## Context

### References

```yaml
refs:
  - reqspecs/references/project_patterns/check_script.md   # script conventions: -m form, by-path hint, exit code 2
  - reqspecs/references/tech/boto3.md                       # response shapes; no call is made here
  - reqspecs/references/common/gotchas.md                   # section 4: heredoc backslashes, logger names

code:
  - file: qlora_finetuning/settings.py
    why: load_config, ConfigError, REPO_ROOT, CONFIG_DIR, get_str
  - file: qlora_finetuning/environment.py
    why: environment_report() goes into the manifest
  - file: scripts/env_check.py
    why: main(argv) shape, by-path hint, ConfigError -> exit 2
  - file: tests/scripts/test_env_check.py
    why: main() tests with capsys; subprocess tests for module and by-path runs
```

### Facts verified 2026-11-15 (installed botocore definitions, read offline)

| Call | Fields used |
|---|---|
| `sagemaker:DescribeTrainingJob` | `TrainingJobName`, `TrainingJobStatus` (`InProgress`, `Completed`, `Failed`, `Stopping`, `Stopped`, `Deleting`), `ResourceConfig.InstanceType`, `ResourceConfig.InstanceCount`, `BillableTimeInSeconds` (integer, per instance), `TrainingTimeInSeconds` (integer). The last two are not required members, so they can be absent |
| `bedrock-runtime:Converse` | `usage.inputTokens`, `usage.outputTokens`, `usage.totalTokens` (all integers, all required) |
| `bedrock:GetModelCustomizationJob` | No token count. Fine-tuning tokens are `computed`, later replaced by a `bill` record |

### Critical Gotchas

- **Appending to the committed ledger makes the working tree dirty.** If the dirty check
  counted that, a second `measured` run could never start without a commit in between. The
  check therefore ignores changes under the ledger folder. This refines spec D7.
- **`environment_report()` imports torch when it is installed** (a few seconds, and it touches
  the GPU). `start_run` takes an optional `environment` argument; tests pass `{}`.
- **`input_tokens` contains the word "token".** The secret guard must look at values (access
  key ids, email addresses), not at key names.
- **Dollar amounts as floats lose cents.** `billed_amount_usd` accepts text, `Decimal`, or
  `int`; a float is rejected. Sums use `Decimal`.
- **`bool` is an `int`.** Reject booleans where a count is expected.
- **Validate before writing.** `start_run` checks every value and every config file before it
  creates the folder; `append` checks the record before it opens the file.
- **JSON Lines on Windows.** Open with `newline="\n"` and `encoding="utf-8"`, one
  `json.dumps` per line, then flush and `os.fsync`.
- **The manifest is rewritten on finish.** Write to a temporary file in the same folder and
  `os.replace` it, so a crash never leaves half a manifest.
- **`test_subpackage_imports_and_is_described`** expects every sub-package docstring to say
  "Filled by req_". `tracking` is now built; loosen the test to "names its spec group".
- **The stale-content guard matches "registry entry".** Do not use that phrase in `coding.md`,
  the commands, or the templates; say "starts a run".
- **No backslashes through a shell heredoc.** Write Python files with the file tool.
- **Git may be absent or the folder may not be a repository** (a cloud job, a temporary test
  folder). Then the commit comes from `QLORA_GIT_COMMIT`, else `unknown`, and dirtiness is
  unknown (`None`).

---

## Implementation

### Files

```yaml
CREATE:
  - config/tracking.yaml: paths and allowed values
  - qlora_finetuning/tracking/config.py: TrackingConfig, load_tracking_config, TrackingError, utc helpers, check_no_secrets
  - qlora_finetuning/tracking/jsonl.py: append_line, read_lines
  - qlora_finetuning/tracking/ledger.py: LedgerRecord, Ledger, sum_converse_usage, the two mappers
  - qlora_finetuning/tracking/registry.py: new_run_id, GitInfo, read_git_info, start_run, Run, load_manifest, list_runs
  - scripts/ledger_summary.py
  - ledger/README.md
  - tests/tracking/test_ledger.py, tests/tracking/test_registry.py, tests/scripts/test_ledger_summary.py
  - reqspecs/references/project_patterns/run_tracking.md
  - reqspecs/impl_summary/impl_000_<TIMESTAMP>.md

MODIFY:
  - qlora_finetuning/tracking/__init__.py: docstring and public names
  - tests/test_package.py: sub-package docstring expectation
  - docs/runbook.md: new section "Runs and the cost ledger"
  - reqspecs/0. PLANNING/high_level_plan.md: section 5 layout gains ledger/
  - reqspecs/1. PROMPTS/coding.md: entry points start a run and record usage
  - reqspecs/references/project_patterns/README.md
  - roadmap, execution plan, spec status
```

### Design

```python
# tracking/config.py
class TrackingError(ConfigError): ...
@dataclass(frozen=True)
class TrackingConfig:
    root: Path; runs_dir: Path; ledger_file: Path; runs_index_file: Path
    git_commit_env: str
    kinds, paths, arms, places, scopes, phases: frozenset[str]
def load_tracking_config(root=REPO_ROOT, config_dir=None) -> TrackingConfig
    # reads config/tracking.yaml; relative paths resolve under `root` (tests pass tmp_path)
def utc_now() -> datetime;  def timestamp(dt=None) -> "2026-11-15T06:15:00Z"
def check_no_secrets(value, where)   # recursive; AKIA/ASIA key ids and email addresses -> TrackingError

# tracking/jsonl.py
def append_line(path, obj)           # mkdir parents, append one JSON line, flush, fsync
def read_lines(path) -> list[dict]   # missing file -> []; bad line -> TrackingError with its number

# tracking/ledger.py
SOURCES = ("measured", "aws_api", "computed", "bill")
@dataclass(frozen=True)
class LedgerRecord:                  # fields exactly as spec 4.3; record_id and recorded_at default
    def to_dict(self) -> dict        # omits None and "" so lines stay short
    @classmethod
    def from_dict(cls, data) -> LedgerRecord   # unknown schema or field -> TrackingError
def validate(record, cfg, known_ids) # the rules of spec 4.3, in one place
class Ledger:
    def __init__(self, cfg=None)
    def append(self, record) -> LedgerRecord
    def read(self) -> list[LedgerRecord]
    def effective(self) -> list[LedgerRecord]          # drops records named by a `replaces`
    def summarize(self, by=("arm", "phase")) -> list[dict]
        # per group: records, wall_clock_seconds, billable_instance_seconds
        # (= billable_seconds * instance_count), input/output/training tokens, examples,
        # billed_amount_usd (Decimal sum as text), sources {name: count}
def sum_converse_usage(usages) -> {"inputTokens": n, "outputTokens": m}
def record_from_sagemaker_job(description, run, phase) -> LedgerRecord   # not appended
def record_from_converse_usage(totals, run, model_id, examples, wall_clock_seconds=None, phase="eval")

# tracking/registry.py
@dataclass(frozen=True)
class GitInfo: commit: str; branch: str | None; dirty: bool | None
def read_git_info(root, env_var, ignore_prefixes=()) -> GitInfo
def dirty_paths(porcelain_text, ignore_prefixes) -> list[str]     # pure; unit-tested
def new_run_id(kind, label, now=None, suffix=None) -> "20261115T061500Z-train-a1-3f9a"
def start_run(kind, label, *, path, arm, where, scope, configs=(), split_manifest_hash=None,
              allow_dirty=False, description="", cfg=None, config_dir=None, git=None,
              environment=None, now=None) -> Run
class Run:                            # run_id, run_dir, manifest (dict), cfg
    def new_record(**fields) -> LedgerRecord      # fills run_id, path, arm, where
    def record(**fields) -> LedgerRecord          # new_record + append
    def append(record) -> LedgerRecord
    def finish(status, outputs=None, error=None)  # once; rewrites manifest, appends "finished" line
    __enter__ / __exit__                          # exception -> failed and re-raise; else completed
def load_manifest(run_id, cfg=None) -> dict
def list_runs(cfg=None) -> list[dict]             # newest first, with final status
```

Rule for `measured` runs: refuse when `git.dirty` is true, or when the commit is `unknown`,
unless `allow_dirty=True`; the manifest then records `"dirty_override": true`. `dev` runs
never refuse.

### `scripts/ledger_summary.py`

`main(argv=None, cfg=None) -> int`. Options `--by arm,phase` and `--json`. Prints the runs
table, then the totals table, then "N of M records are estimates (computed)". With no runs and
no records prints "no records". `ConfigError` (which `TrackingError` extends) gives a
`CONFIG ERROR` line and exit 2. Same by-path hint as the other scripts.

### Tests

| File | Cases |
|---|---|
| `tests/tracking/test_ledger.py` | Round trip; `to_dict` omits empty fields; each rule (no quantity, negative, billable seconds without instance, tokens without model, amount without `bill`, float amount, unknown phase/arm/source, boolean count, duplicate id, `replaces` unknown) rejects and leaves the file untouched; `effective` after a replacement; `summarize` totals, instance-seconds, Decimal sums, sources, custom `by`, bad `by`; corrupt line and unknown schema with line numbers; `sum_converse_usage`; both mappers, an unfinished job, a job without billable time; secret guard |
| `tests/tracking/test_registry.py` | Id format and label cleaning; folder, config copies with hashes and versions, manifest fields; index lines; `finish` once only; context manager on success and on exception; unknown kind/arm/place/scope; missing config file leaves nothing behind; the dirty rule for `measured`, `dev`, and the override; unknown commit; `dirty_paths` with ignored ledger changes, renames, and quoted paths; `read_git_info` in a folder without git falls back to the environment variable; `read_git_info` on this repository returns a 40-character commit; `list_runs` order and status; `load_manifest` of an unknown run |
| `tests/scripts/test_ledger_summary.py` | Empty ledger; a dummy run end to end with two records; `--by`; `--json` parses; estimate count; corrupt ledger gives exit 2; module run; by-path hint |

All use `load_tracking_config(root=tmp_path)`, a fixed `GitInfo`, and `environment={}`.

### Tasks (~3.5 hours total)

1. **Config and `tracking/config.py`, `jsonl.py`** (~25 min).
2. **`ledger.py`** (~45 min).
3. **`registry.py`** (~50 min).
4. **Tests for ledger and registry** (~50 min).
5. **`ledger_summary.py` and its tests** (~25 min).
6. **`ledger/README.md`, runbook, plan layout, `coding.md`, pattern note** (~25 min).
7. **Validation, status documents, summaries** (~15 min).

### Error handling

| Situation | Behavior |
|---|---|
| Invalid record | `TrackingError` naming the rule; nothing written |
| Unknown kind, arm, place, scope, phase | `TrackingError` listing the allowed values |
| Config file named for the snapshot is missing | `ConfigError`; no run folder is created |
| `measured` run on uncommitted changes or unknown commit | `TrackingError` naming up to five changed paths and the override |
| Corrupt ledger or index line | `TrackingError` with file name and line number |
| Finishing a run twice | `TrackingError` |
| Exception inside `with start_run(...)` | Run finished as `failed` with the error text; exception re-raised |

---

## Validation

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q
venv\Scripts\python -m scripts.ledger_summary
```

**Expected**: ruff clean; all tests pass; "no records".

Also run the unit tests with the training libraries blocked, as for v0.03.

---

## Checklist

- [ ] Files created and modified as listed
- [ ] Every rule and every function has passing and failing tests
- [ ] Lint clean; tests green with and without the training stack
- [ ] Script run shown
- [ ] Runbook, plan layout, `coding.md`, pattern note updated
- [ ] Roadmap, execution plan, spec status updated; implementation summary created

---

## Implementation Time Estimate

**Total: 3–4 hours**

| Task | Time |
|---|---|
| Config, `config.py`, `jsonl.py` | 25 min |
| `ledger.py` | 45 min |
| `registry.py` | 50 min |
| Ledger and registry tests | 50 min |
| `ledger_summary.py` and tests | 25 min |
| Documents | 25 min |
| Validation and status documents | 15–30 min |

## Confidence Score: 9/10

Pure Python over files, no external service, with the AWS field names confirmed from the
installed definitions. The point held back is for the amount of validation logic, where an
edge case is likely to need a second pass.

## Time Taken for PRP Generation

**~8 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 1 min |
| Codebase exploration | 1 min |
| Research existing PRPs and patterns (AWS field names) | 1 min |
| Writing PRP document | 5 min |
