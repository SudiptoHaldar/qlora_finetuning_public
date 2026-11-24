# PRP: Dataset snapshot and schema report (req_001 v1.01)

**Feature Name**: dataset_snapshot (finqa_data v1.01)  
**Version**: 1  
**Created**: 2026-11-15  
**Status**: ✅ IMPLEMENTED  
**Implemented**: 2026-11-15  
**Implementation Summary**: [impl_001_20261115_063228.md](../impl_summary/impl_001_20261115_063228.md)  
**Confidence Score**: 8/10  
**Estimated Time**: 4–5 hours  
**Origin**: `reqspecs/reqs/req_001_finqa_data/req_001_01_spec.md`  
**Template**: `prp_lean.md` (expanded, medium complexity)  

---

## Goal

Pin FinQA to one commit and three checksums, read it as typed records, and commit a
reproducible schema report holding counts and example ids only. Nothing about the split, the
prompt, or scoring.

## Why & What

- **User value**: every later version (split, prompts, executor, scorer) works from the same
  verified bytes and from measured facts about them.
- **Technical**: one config file, two modules in `qlora_finetuning/data/`, two scripts, a
  committed report under `reports/data/`, tests on synthetic Parquet files.

## Success Criteria

- [ ] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, also
      with the training libraries blocked (what CI sees).
- [ ] `venv\Scripts\python -m scripts.finqa_fetch` leaves three verified files and a
      `snapshot.json` under `artifacts/data/finqa/<commit>/`; a second run downloads nothing.
- [ ] `venv\Scripts\python -m scripts.finqa_report` passes every check and writes the two
      report files; `--check` straight afterwards exits 0.
- [ ] The report shows 8,281 examples (6,251 / 883 / 1,147), sixteen columns, ten operations,
      seventeen constants, and agrees with spec section 2. Any difference is investigated and
      the spec corrected.
- [ ] The committed report contains no question, passage, table, or answer text.
- [ ] `load_raw_examples()` returns 8,281 records and refuses a snapshot with an altered file.

---

## Context

### References

```yaml
refs:
  - reqspecs/references/project_patterns/check_script.md   # PASS/FAIL lines, exit codes 0/1/2, -m form
  - reqspecs/references/common/gotchas.md                   # section 4: ASCII output, heredoc backslashes, logger names
  - reqspecs/references/tech/hf_training_stack.md           # huggingface_hub 2.4.0 is training stack; noisy loggers

code:
  - file: qlora_finetuning/tracking/config.py
    why: a frozen config dataclass loaded with root= and config_dir= so tests use tmp_path
  - file: scripts/env_check.py
    why: CheckResult, main(argv) shape, by-path hint, ConfigError -> exit 2, NOISY_LOGGERS
  - file: tests/scripts/test_ledger_summary.py
    why: main() tests with capsys; subprocess tests for module and by-path runs
  - file: tests/test_process_scaffold.py
    why: guards this work must keep green (no module-level training-stack import; module-form commands)
```

### Facts verified 2026-11-15

Spec section 2 is the reference for every count. Added here, read from the real files:

| Fact | Value |
|---|---|
| Parquet column types, as `str(field.type)` | `string`; `list<element: string>`; `list<element: list<element: string>>`; `list<element: int32>`; `steps` is `struct<op: list<element: string>, arg1: ..., arg2: ..., res: ...>` |
| `pyarrow.parquet.read_table(path).to_pylist()` | Returns plain `str`, `list`, and `dict` values. No NumPy arrays, so no conversion step is needed |
| Row count without reading | `pyarrow.parquet.ParquetFile(path).metadata.num_rows` |
| Hub file listing, no library needed | `https://huggingface.co/api/datasets/bevaya/FinQA/tree/<commit>/data` returns a JSON list; each item has `path`, `size`, and `lfs.oid` (the SHA-256) |
| `hf_hub_download(repo_id=, filename=, repo_type="dataset", revision=, local_dir=)` | Writes `local_dir/<filename>` keeping the `data/` folder, and a `.cache/` folder beside it |
| Program form | `", ".join(re.findall(r"[a-z_]+\([^()]*\)", p)) == p.strip()` holds for all 8,281 |
| git on this machine | `core.autocrlf=true`: a committed text file is checked out with CRLF |

### Critical Gotchas

- **Read with pyarrow alone.** Spec D6 names pandas and pyarrow; `to_pylist()` already gives
  the plain values D8 asks for, so pandas is not used. Import pyarrow inside functions.
- **`huggingface_hub` is training stack.** Import it inside the default downloader only. When
  it is missing, raise `SnapshotError` pointing at runbook section 3.1.
- **Compare the report as text, not bytes.** `--check` and the freshness test read the
  committed file with `read_text`, which folds CRLF to LF; files are written with
  `newline="\n"`.
- **Determinism.** No timestamp, no absolute path, no set or dict ordering that depends on
  hashing. Counted maps are sorted by count descending, then key. JSON is written with
  `indent=2`, `ensure_ascii=True`, and a trailing newline.
- **No example text in the report** (D11). Only counts, ids, operation names, constant names,
  and argument kinds. Row names, answers, and questions never appear. A test checks this.
- **Tests never use real rows.** Synthetic rows are written to Parquet with pyarrow in a
  temporary folder, with a config whose checksums are computed from those files.
- **`SnapshotError` extends `ConfigError`.** Scripts catch it first.
- **The stale-content guard matches `\bticker` and `\bchart`** in the command files, the PRP
  templates, `coding.md`, `requirements.txt`, and `pytest.ini`. Code may use `ticker`; those
  files may not.
- **ASCII output only** from scripts (Windows console).
- **A YAML id pattern** goes in single quotes so backslashes survive.
- **`gold_inds` that is not a JSON object** becomes `None` on the record, never `{}`: the
  reader does not repair, and the report counts it.

---

## Implementation

### Files

```yaml
CREATE:
  - config/data.yaml
  - qlora_finetuning/data/snapshot.py: DataConfig, SnapshotError, RawExample, fetch, verify, load
  - qlora_finetuning/data/schema_report.py: split_program, profile, build_report, render_markdown
  - scripts/finqa_fetch.py, scripts/finqa_report.py
  - tests/data/synthetic.py: synthetic rows, Parquet writer, config builder
  - tests/data/test_snapshot.py, tests/data/test_schema_report.py
  - tests/scripts/test_finqa_fetch.py, tests/scripts/test_finqa_report.py
  - reports/data/finqa_schema_report.md, reports/data/finqa_schema_report.json (generated)
  - reqspecs/references/tech/finqa_dataset.md
  - reqspecs/impl_summary/impl_001_<TIMESTAMP>.md

MODIFY:
  - requirements.txt: pyarrow==26.0.1
  - qlora_finetuning/data/__init__.py: docstring, public names
  - docs/runbook.md: section 4 "Data pipeline"
  - reqspecs/references/common/gotchas.md, troubleshooting.md, references/README.md
  - reqspecs/0. PLANNING/high_level_plan.md, execution_plan.md: dataset name; percent literals in 4.2
  - roadmap, execution plan status, spec status
```

### Design

```python
# data/snapshot.py
class SnapshotError(ConfigError): ...
@dataclass(frozen=True)
class SnapshotFile: official_split: str; path: str; sha256: str; size: int; rows: int
@dataclass(frozen=True)
class DataConfig:
    root: Path; repo_id: str; former_repo_id: str; revision: str
    snapshot_dir: Path; report_dir: Path; files: tuple[SnapshotFile, ...]; expected: dict
def load_data_config(root=REPO_ROOT, config_dir=None) -> DataConfig
    # config/data.yaml; relative paths resolve under root. Rejects: a revision that is not 40
    # lower-case hex characters, a sha256 that is not 64, non-positive bytes or rows, a
    # repeated split, an absolute path or one containing "..".
def snapshot_path(cfg) -> Path                      # snapshot_dir / revision
def sha256_of(path) -> str                          # 1 MiB chunks
@dataclass(frozen=True)
class FileCheck: official_split; path; ok: bool; detail: str
def verify_snapshot(cfg) -> list[FileCheck]         # exists, size, sha256, row count
def fetch_snapshot(cfg, download=None, force=False, now=None) -> list[str]   # paths downloaded
    # download(repo_id, filename, revision, local_dir, force). A present, correct file is kept.
    # A present, wrong file raises unless force. After downloading, verify; any failure raises
    # SnapshotError with expected and actual values. Nothing is deleted. Writes snapshot.json
    # when something was downloaded or the file is absent.
def read_columns(cfg) -> dict[split, dict[name, type]]   # type_name(): "list[string]", "struct{op: list[string], ...}"
@dataclass(frozen=True)
class RawExample:        # spec 4.2 fields; ticker/year/page/index are None when the id does not match
    @property
    def report(self) -> str | None                  # "<ticker>/<year>"
def load_raw_examples(cfg=None) -> list[RawExample] # verifies first; train, validation, test; file order

# data/schema_report.py   (pure; no pyarrow, no file access except write_report)
def split_program(text) -> list[tuple[str, list[str]]] | None    # None = cannot be split
def argument_kind(text) -> "integer"|"decimal"|"reference"|"constant"|"percent"|"none"|"other"
def profile(records, expected) -> dict     # sections: examples, columns, identity, duplicates,
                                           # text, tables, evidence, programs, answers
def flagged_ids(records, expected) -> dict[str, list[str]]
def run_checks(report, expected) -> list[dict]      # S1..S10: id, name, passed, detail
def build_report(records, cfg, columns) -> dict
    # {"schema": 1, "source": {...}, "column_types": {...}, "total": profile(all),
    #  "by_split": {split: profile(subset)}, "spanning_official_splits": {...},
    #  "flagged": {...}, "checks": [...]}
def render_json(report) -> str;  def render_markdown(report, max_ids=20) -> str
def write_report(report, report_dir) -> tuple[Path, Path]
```

Checks (all must pass on the real snapshot):

| Id | Check | Passes when |
|---|---|---|
| S1 | Rows per file | Each official split has its configured row count |
| S2 | Columns | Every file has exactly the expected names and types |
| S3 | No null values | No column holds a null |
| S4 | Unique ids | No id repeats |
| S5 | Id format | Every id matches the pattern and extends its `filename` |
| S6 | Program form | Every program splits, and every step has the expected number of arguments |
| S7 | Operations | The set seen equals the expected set |
| S8 | Constants | The set seen equals the expected set |
| S9 | Argument kinds | The set seen equals the expected set |
| S10 | Program length | No program is longer than the expected maximum |

Counting rules, chosen to reproduce spec section 2: a `gold_inds` key is well-formed when it
matches `^(table|text)_(\d+)$`; anything else is malformed (so `text_-1` is malformed, not out
of range). Disagreement with the `ann_*` lists compares the sets of well-formed indices. A
table operation's first argument is kind `row_name`; it is "found" when it equals the first
cell of a row of the example's table. Answer categories: empty (blank after stripping),
contains `%`, contains `$`, yes/no (lower-case equals), plain number
(`-?[\d,]*\.?\d+`), other (none of these); `%` and `$` can overlap.

Flagged id lists: duplicate page-and-question groups, empty answer, other-form answer,
malformed evidence key, unparsed `gold_inds`, out-of-range evidence key, unparsed program,
unknown operation, unknown constant, other-kind argument, percent-literal argument,
row name not in table, table second argument not `none`, longest programs, scientific-notation
result, yes/no answer without `greater`, `greater` whose answer disagrees. The Markdown lists
ids for a class of up to 20 and gives the count otherwise; the JSON lists all.

### Scripts

`finqa_fetch`: `main(argv=None, cfg=None, download=None)`. Lines
`PASS  train  data/...parquet: 30,089,143 bytes, 6,251 rows, sha256 55dc8ef7...`. Options
`--verify-only`, `--force`. Exit 0 all verified; 1 a file is missing or wrong, or the download
failed; 2 config problem.

`finqa_report`: `main(argv=None, cfg=None)`. Prints the check lines and a summary, writes both
files. `--check` writes nothing and exits 1 when a committed file is missing or differs. Exit
1 when a check fails; 2 when the config is broken or there is no verified snapshot (the message
names the fetch command).

### Tests

| File | Cases |
|---|---|
| `tests/data/test_snapshot.py` | Config: real file loads, paths under `root`, bad revision, bad checksum, repeated split, unsafe path. Fetch: stand-in downloads missing files only; nothing downloaded on a second call; `snapshot.json` contents and that it is not rewritten; wrong checksum after download raises with both values and deletes nothing; present wrong file without `force`; `force`; downloader exception; missing library message. Verify: missing, wrong size, wrong checksum, wrong row count. Load: order, plain lists, parsed `gold_inds`, non-JSON `gold_inds` is `None`, id parts, unmatched id, missing snapshot message, altered file refused. `type_name`. One `network` test against the Hub listing |
| `tests/data/test_schema_report.py` | `split_program` and `argument_kind` tables; a synthetic set with one of each oddity gives the expected counts and flags; each check passes on clean rows and fails on its violation; per-split counts add up to the total; same input gives the same text; no synthetic question, passage, cell, or answer text in either output; Markdown id-list cut-off; freshness of the committed report (skips without a snapshot) |
| `tests/scripts/test_finqa_fetch.py`, `test_finqa_report.py` | Each exit code with `capsys`; `--verify-only` makes no download call; `--check` on equal, different, and missing files; ASCII output; module form; by-path hint |

### Tasks (~4.5 hours total)

1. **`config/data.yaml`, `requirements.txt`, `snapshot.py`** (~50 min).
2. **`tests/data/synthetic.py`, `test_snapshot.py`** (~40 min).
3. **`schema_report.py`** (~60 min).
4. **`test_schema_report.py`** (~40 min).
5. **Both scripts and their tests** (~35 min).
6. **Real run**: fetch, report, compare with spec section 2, reconcile, `-m network` (~20 min).
7. **Documents**: runbook, reference note, gotchas, plan names (~25 min).
8. **Validation with the stack blocked, status documents, summaries** (~15 min).

### Error handling

| Situation | Behavior |
|---|---|
| Config value malformed (revision, checksum, path) | `ConfigError` naming the key |
| `huggingface_hub` not installed | `SnapshotError`: install the training stack, runbook 3.1 |
| Download raises | `SnapshotError` with the error type and first line; exit 1 |
| File missing, wrong size, checksum, or row count | `SnapshotError` naming the file, expected and actual; nothing deleted |
| No snapshot when loading | `SnapshotError` naming `python -m scripts.finqa_fetch` |
| A report check fails | FAIL line with the detail; files still written; exit 1 |

---

## Validation

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q
venv\Scripts\python -m scripts.finqa_fetch
venv\Scripts\python -m scripts.finqa_report
venv\Scripts\python -m scripts.finqa_report --check
venv\Scripts\python -m pytest -m network -q tests/data
```

**Expected**: ruff clean; all tests pass; three PASS lines; ten PASS checks; `--check` exits 0.

Also run the unit tests with the training libraries blocked, as for v0.03 and v0.04.

---

## Checklist

- [ ] Files created and modified as listed
- [ ] Every function and every check has passing and failing tests
- [ ] Lint clean; tests green with and without the training stack
- [ ] Real fetch and report run shown; numbers reconciled with the spec
- [ ] Runbook, reference note, gotchas, plan documents updated
- [ ] Roadmap, execution plan, spec status updated; implementation summary created

---

## Implementation Time Estimate

**Total: 4–5 hours**

| Task | Time |
|---|---|
| Config, requirement, `snapshot.py` | 50 min |
| Synthetic data helper and snapshot tests | 40 min |
| `schema_report.py` | 60 min |
| Report tests | 40 min |
| Scripts and their tests | 35 min |
| Real run and reconciliation | 20 min |
| Documents | 25 min |
| Validation and status documents | 15–30 min |

## Confidence Score: 8/10

The data was profiled before the spec was written, so the expected numbers and the file
schema are known. The points held back: the report has many counting rules, and a few of
them may not reproduce the spec's figures on the first pass; and the download goes through a
library call that has not yet been run from this repository.

## Time Taken for PRP Generation

**~10 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 1 min |
| Codebase exploration | 2 min |
| Research existing PRPs and patterns (file schema, script and test conventions) | 2 min |
| Writing PRP document | 5 min |
