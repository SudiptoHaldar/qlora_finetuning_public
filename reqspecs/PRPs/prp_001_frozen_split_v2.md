# PRP: Frozen split manifest (req_001 v1.02)

**Feature Name**: frozen_split (finqa_data v1.02)  
**Version**: 2  
**Created**: 2026-11-15  
**Status**: ✅ IMPLEMENTED  
**Implemented**: 2026-11-15  
**Implementation Summary**: [impl_001_20261115_125357.md](../impl_summary/impl_001_20261115_125357.md)  
**Confidence Score**: 8/10  
**Estimated Time**: 4–5 hours  
**Origin**: `reqspecs/reqs/req_001_finqa_data/req_001_02_spec.md`  
**Template**: `prp_lean.md` (expanded, medium complexity)  

---

## Goal

Assign every FinQA example to `fit`, `dev`, or `test` by a deterministic rule grouped by
company, commit the result as a manifest that is written once, and make every later stage ask
for examples by split name, with `test` locked until req_005 v5.01.

## Why & What

- **User value**: both training paths and the final comparison use exactly the same examples,
  and the test set cannot be read or tuned against by accident.
- **Technical**: one config file, three modules in `qlora_finetuning/data/`, one script, a
  committed `splits/` folder and split report, tests on invented rows.

## Success Criteria

- [ ] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, also
      with the training libraries blocked.
- [ ] `venv\Scripts\python -m scripts.finqa_split` writes the manifest and the report; P1 to
      P10 pass; a second run changes nothing; `--check` exits 0.
- [ ] The manifest reproduces spec section 2.4: fit 4,884 / dev 499 / test 2,898, quick-dev
      200, 79 / 15 / 43 companies. Any difference is investigated before anything is kept.
- [ ] `load_split("test")` raises `TestLockedError`; `fit`, `dev`, `quick_dev` return 4,884,
      499, and 200 records.
- [ ] No file outside the allowed list uses the raw reader.
- [ ] The content hash is recorded in the req_001 roadmap and in the execution plan (M2).

---

## Context

### References

```yaml
refs:
  - reqspecs/references/tech/finqa_dataset.md            # columns, grouping levels, the rule on committed text
  - reqspecs/references/project_patterns/check_script.md # PASS/FAIL lines, exit codes, -m form
  - reqspecs/references/project_patterns/run_tracking.md # start_run(split_manifest_hash=...)
  - reqspecs/references/common/gotchas.md                # sections 4 and 5

code:
  - file: qlora_finetuning/data/snapshot.py
    why: DataConfig, RawExample (ticker, report, filename), load_raw_examples, config loading with root= and config_dir=
  - file: qlora_finetuning/data/schema_report.py
    why: profile() and the per-example facts are reused for the split report; write/stale helpers show the text comparison
  - file: scripts/finqa_report.py
    why: main(argv, cfg) shape, --check, exit codes
  - file: tests/data/synthetic.py
    why: invented rows and build_snapshot; extended here with many companies and a split config
```

### Facts verified 2026-11-15

| Fact | Value |
|---|---|
| Rule on the real snapshot, key = company, seed 20261115 | train 5,383 in 94 companies; test 2,898 in 43; fit 4,884 in 79; dev 499 in 15; quick-dev 200 from 12 |
| Train target | `round(0.65 * 8281)` = 5,383 |
| Leakage, test against train, key = company | company in train 0; shared (row name, 4+ digit number) 0 of 1,903 with the same company and 65 with any company; same question wording 1 |
| Digest | `hashlib.sha256(f"{seed}:{salt}:{key}".encode()).hexdigest()`; salts `train`, `dev`, `quick_dev` |
| Fill | Sort group keys by digest; take a group when `total + size <= target` |
| Dev | Same rule over the train groups' sizes, target 500 |
| Quick-dev | Dev examples sorted by `digest("quick_dev", example.id)`, first 200 |
| Shared-number pairs | For each `table_N` key of `gold_inds` with `N < len(table)`: for each cell after the first, keep only its digits; if 4 or more, the pair is `(first cell stripped, digits)`. Training side: the same pairs over every row of every table |
| `.gitattributes` | `* text=auto`: committed text is stored with LF and may be checked out with CRLF |

### Critical Gotchas

- **pytest collects any class named `Test*` that a test module imports.** `TestLockedError`
  needs `__test__ = False`, or every test file importing it warns.
- **Fail the lock before reading data.** `load_split("test")` checks the unlock record first,
  so a locked call never touches a test row.
- **Never write the manifest from a test into the repository.** Tests pass a config rooted in
  `tmp_path`. The only writer of `splits/` is the script, run once by hand in this PRP.
- **The manifest is written once.** `write_manifest` writes when absent, does nothing when
  identical, raises when different. There is no force flag anywhere.
- **Compare committed text as text** (`read_text`), never bytes; write with `newline="\n"`.
- **Determinism.** No clock, no absolute path, no random module. `json.dumps(indent=1)` puts
  one assignment per line.
- **Float targets.** `round(n * fraction)`; test the real 8,281 case gives 5,383.
- **No example text in committed files**: the manifest holds ids and company symbols; the
  report holds counts.
- **The stale-content guard matches `\bticker` and `\bchart`** in `coding.md`, the command
  files, and the PRP templates. Say "company" there.
- **A stray unlock file must not unlock.** It counts only when it is a JSON object whose
  `manifest_hash` equals the current content hash.
- **`build_split` rejects** an example with no group key, a repeated id, a dev size that does
  not fit in train, and a quick-dev size larger than dev.

---

## Implementation

### Files

```yaml
CREATE:
  - config/split.yaml
  - qlora_finetuning/data/split.py: SplitConfig, SplitError, order_key, fill, build_split, content_hash, manifest read/write
  - qlora_finetuning/data/split_loader.py: load_split, split_ids, manifest_hash, is_test_unlocked, TestLockedError
  - qlora_finetuning/data/split_report.py: split statistics, leakage, checks P1-P10, JSON and Markdown
  - scripts/finqa_split.py
  - splits/README.md, splits/finqa_split_manifest.json (generated once)
  - reports/data/finqa_split_report.md, .json (generated)
  - tests/data/test_split.py, test_split_loader.py, test_split_report.py, tests/scripts/test_finqa_split.py
  - reqspecs/references/project_patterns/split_loading.md
  - reqspecs/impl_summary/impl_001_<TIMESTAMP>.md

MODIFY:
  - qlora_finetuning/data/__init__.py: public names
  - qlora_finetuning/data/schema_report.py: a public name for the per-example facts
  - tests/data/synthetic.py: many-company rows, split config builder
  - docs/runbook.md (section 4), reqspecs/1. PROMPTS/coding.md
  - reqspecs/references: gotchas, troubleshooting, finqa_dataset.md, project_patterns/README.md
  - high_level_plan.md (4.1, 5), execution_plan.md (Q4, M2, status), roadmap, spec status
```

### Design

```python
# data/split.py
class SplitError(ConfigError): ...
SPLITS = ("fit", "dev", "test");  RULE = "sha256-order-fill-v1"
GROUP_KEYS = {"company": e.ticker, "report": e.report, "page": e.filename}
@dataclass(frozen=True)
class SplitConfig: root, group_by, seed, train_fraction, train_fraction_tolerance, dev_size,
                   dev_size_tolerance, quick_dev_size, leakage_min_digits,
                   manifest_file, unlock_file, report_dir
def load_split_config(root=REPO_ROOT, config_dir=None) -> SplitConfig
def group_key(example, group_by) -> str            # SplitError when the example has none
def order_key(seed, salt, key) -> str
def fill(sizes, seed, salt, target) -> list[str]   # chosen keys, in digest order
def content_hash(assignments, quick_dev) -> str    # "<id>\t<split>\t<1|0>\n" in dataset order
def build_split(examples, cfg, data_cfg) -> dict   # pure
def render_manifest(manifest) -> str               # json, indent=1, trailing newline
def validate_manifest(manifest, where) -> None     # shape, values, quick-dev inside dev, hash
def read_manifest(cfg) -> dict                     # missing file names the build command
def write_manifest(manifest, cfg) -> bool          # True when written; SplitError when it differs

# data/split_loader.py
class TestLockedError(SplitError): __test__ = False
LOADABLE = ("fit", "dev", "quick_dev", "test")
def check_manifest_against_data(manifest, examples, data_cfg)   # D15
def unlock_state(cfg, manifest) -> tuple[bool, str]             # (unlocked, why)
def is_test_unlocked(cfg=None) -> bool
def split_ids(name, cfg=None) -> list[str]
def manifest_hash(cfg=None) -> str
def load_split(name, cfg=None, data_cfg=None) -> list[RawExample]

# data/split_report.py
def gold_number_pairs(example, min_digits), table_number_pairs(example, min_digits)
def leakage(held_out, seen, min_digits) -> dict
def split_stats(pairs, group_by, total) -> dict
def run_checks(manifest, rebuilt, examples, cfg, data_cfg) -> list[dict]   # P1..P10
def build_report(manifest, rebuilt, examples, cfg, data_cfg) -> dict
def render_json, render_markdown, write_report, stale_files
```

Manifest keys, in this order: `schema`, `dataset`, `settings`, `counts`, `content_hash`,
`groups` (sorted by key), `quick_dev`, `assignments` (dataset order).

Unlock record (created only by v5.01): a JSON object with at least
`{"manifest_hash": "<content hash>"}`.

### Script

`main(argv=None, cfg=None, data_cfg=None)`. Load the examples, build the split, then:
without `--check`, `write_manifest` (a different existing manifest prints `SPLIT ERROR` and
exits 1 without touching anything), read the manifest back, run the checks, write the report.
With `--check`, write nothing and report stale files. Prints the P lines, the counts per
split, and the content hash. Exit 0, 1, 2 as in the spec.

### Tests

| File | Cases |
|---|---|
| `tests/data/test_split.py` | Config loads and validates (bad key, fraction, sizes); `order_key` against fixed values; `fill` never exceeds, takes later groups that fit, empty input; each grouping key; groups whole; dev inside train; quick-dev size and membership; dev or quick-dev too large raises; no group key raises; repeated id raises; same input same manifest; another seed differs; `round(0.65 * 8281)`; content hash changes with one assignment and with one quick-dev flag; manifest round trip; write once, identical accepted, different refused; hand-edited file fails; missing file names the command |
| `tests/data/test_split_loader.py` | Each split's ids in dataset order; quick-dev; unknown name; manifest for another commit; missing id; extra id; locked without a record, with another hash, with a malformed record, with a list; unlocked with a valid record; a locked call reads no data; `split_ids("test")` works while locked; `manifest_hash`; repository guards: no unlock file, raw reader used only by allowed files, committed manifest valid, committed manifest matches the snapshot when present |
| `tests/data/test_split_report.py` | Number pairs; leakage on a constructed case; stats per split; every check passes on a clean build and fails on its violation; deterministic text; no invented words in the output; write and stale |
| `tests/scripts/test_finqa_split.py` | First run writes; second run unchanged; `--check` current, stale, and missing; a different existing manifest exits 1 and leaves it untouched; no snapshot exits 2; module form; by-path hint |

### Tasks (~4.5 hours total)

1. **Config and `split.py`** (~50 min).
2. **`split_loader.py`** (~30 min).
3. **Synthetic helpers, builder and loader tests** (~50 min).
4. **`split_report.py` and its tests** (~55 min).
5. **Script and its tests** (~30 min).
6. **Real run**: build, compare with spec 2.4, `--check`, keep the manifest (~15 min).
7. **Documents** (~30 min).
8. **Validation with the stack blocked, status documents, summaries** (~15 min).

### Error handling

| Situation | Behavior |
|---|---|
| Config value out of range or unknown key | `ConfigError` naming the key |
| Example without a group key, repeated id | `SplitError` naming the id |
| Dev or quick-dev size cannot be met | `SplitError` with the numbers |
| Manifest missing | `SplitError` naming `python -m scripts.finqa_split` |
| Manifest edited, wrong hash, wrong shape | `SplitError` saying what failed |
| Manifest exists and differs from the build | `SplitError`: the split is frozen; nothing written |
| Manifest does not match the data | `SplitError`; nothing loaded |
| `load_split("test")` while locked | `TestLockedError`; no data read |

---

## Validation

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q
venv\Scripts\python -m scripts.finqa_split
venv\Scripts\python -m scripts.finqa_split --check
```

**Expected**: ruff clean; all tests pass; ten PASS lines; fit 4,884, dev 499, test 2,898;
`--check` exits 0. Also the unit tests with the training libraries blocked.

---

## Checklist

- [ ] Files created and modified as listed
- [ ] Every function and every check has passing and failing tests
- [ ] Lint clean; tests green with and without the training stack
- [ ] Real build shown and reconciled with the spec; content hash recorded
- [ ] Runbook, `coding.md`, references, plan documents updated
- [ ] Roadmap, execution plan, spec status updated; implementation summary created

---

## Implementation Time Estimate

**Total: 4–5 hours**

| Task | Time |
|---|---|
| Config and `split.py` | 50 min |
| `split_loader.py` | 30 min |
| Synthetic helpers, builder and loader tests | 50 min |
| `split_report.py` and tests | 55 min |
| Script and tests | 30 min |
| Real run | 15 min |
| Documents | 30 min |
| Validation and status documents | 15–30 min |

## Confidence Score: 8/10

The rule was run on the real data before the spec was written, so the expected counts are
known and the algorithm is a few lines. The points held back: the manifest is written once,
so the first real build has to be right before it is kept; and the lock and its guard tests
have several edge cases.

## Time Taken for PRP Generation

**~8 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 1 min |
| Codebase exploration | 1 min |
| Research existing PRPs and patterns | 1 min |
| Writing PRP document | 5 min |
