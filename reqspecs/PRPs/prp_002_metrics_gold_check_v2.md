# PRP: Metrics and gold self-check (req_002 v2.02)

**Feature Name**: metrics_gold_check (dsl_evaluation v2.02)  
**Version**: 2  
**Created**: 2026-11-15  
**Status**: ✅ IMPLEMENTED  
**Implemented**: 2026-11-16  
**Implementation Summary**: [impl_002_20261116_004319.md](../impl_summary/impl_002_20261116_004319.md)  
**Confidence Score**: 8/10  
**Estimated Time**: 3.5–4.5 hours  
**Origin**: `reqspecs/reqs/req_002_dsl_evaluation/req_002_02_spec.md`  
**Template**: `prp_lean.md` (expanded, medium complexity)  

---

## Goal

Say when an executed value, an answer text, and a program are right, in one module with its
allowances in config; then score every gold reply of the dataset with it, commit the result
as a report, and list the examples whose gold program does not reproduce its gold answer in
a flags file.

## Why & What

- **User value**: the rules that decide every number in the comparison are fixed and proven
  before any model is run, and no example is ever silently left out.
- **Technical**: one config file, one module in `evaluation/`, one module in `data/`, one
  script, a committed report and flags file, tests on invented data.

This PRP was written assuming the three items of spec section 8 as recommended. **The owner
decided them on 2026-11-16, before the build was finished**: the Gemini guide's 0.5% for
execution accuracy (not the tight allowance); flagged examples stay in every total with the
unflagged total beside it; the self-check reads test rows now, for counts and ids only. The
build follows the owner's decisions: `execution.relative` is 0.005.

## Success Criteria

- [x] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, also
      with the training libraries blocked.
- [x] `venv\Scripts\python -m scripts.finqa_gold_check` checks 8,281 examples; the checks
      pass; a second run changes nothing; `--check` exits 0.
- [x] Fit and dev: 5,383 of 5,383 reproduce, none flagged, largest gap at most half the
      allowance.
- [x] The test split's rate and flagged ids are in the report and in the roadmap.
- [x] The report and the flags file hold no example text.
- [x] The raw-reader guard lists exactly one more file than before.

---

## Context

### References

```yaml
refs:
  - reqspecs/references/project_patterns/reply_scoring.md   # run_reply and its outcome
  - reqspecs/references/project_patterns/split_loading.md   # flags go in their own file
  - reqspecs/references/project_patterns/canonical_record.md
  - reqspecs/references/project_patterns/check_script.md    # PASS/FAIL lines, exit codes
  - reqspecs/references/common/gotchas.md                   # section 5

code:
  - file: qlora_finetuning/data/serialization_report.py
    why: report shape, _check, write_report, stale_files, a check that fails once when a
         committed value changes (C9), rendering with schema_report._label and _number
  - file: scripts/finqa_serialize.py
    why: main(argv, cfg, data_cfg, prompt) shape, the by-path hint, exit codes
  - file: scripts/finqa_split.py
    why: a script that reads every row with the raw reader and checks the manifest
  - file: qlora_finetuning/data/split_loader.py
    why: check_manifest_against_data; reading the manifest's assignments and quick_dev
  - file: tests/data/test_split_loader.py
    why: RAW_READER_ALLOWED, the list that gains one entry
  - file: tests/data/synthetic.py
    why: build_frozen_split, to_examples, row(); gains a rows argument
  - file: tests/scripts/test_finqa_serialize.py
    why: script tests on a synthetic snapshot
```

No web research was needed.

### Facts verified 2026-11-15

| Fact | Value |
|---|---|
| Spec section 2 | Measured with the built v2.01 executor on fit and dev |
| Rule that reproduces all 5,270 numeric gold answers | `abs(v - g) <= max(0.00001, 0.00001 * abs(g))`; largest gap is 0.5 of the allowance |
| `run_reply` over 5,383 replies | 0.14 seconds, so checking 8,281 twice is cheap |
| Manifest | `manifest["assignments"]` maps id to `fit` / `dev` / `test` in dataset order; `manifest["quick_dev"]` lists ids; `manifest["content_hash"]` |
| `build_record(example, prompt)` | Works on any `RawExample`, test rows included; raises `SerializationError` for an unusable row |
| `load_prompt(root=tmp_path)` | Real wording, report paths under a temporary root; the tests' pattern |

### Critical Gotchas

- **`evaluation/metrics.py` must not import `qlora_finetuning.data` at module level**, or the
  data package's self-check, which imports the metrics, becomes a cycle. It needs only
  `qlora_finetuning.dsl` and `settings`.
- **Do not export the self-check from `qlora_finetuning/data/__init__.py`**: import it as
  `qlora_finetuning.data.gold_check`.
- **`6.9 / 100` and `0.069` may differ in the last bit.** Program match compares numbers with
  `math.isclose(rel_tol=1e-12, abs_tol=0)`, not `==`.
- **The answer reader meets model text.** Check characters by hand before `float`; cap the
  exponent's length; reject a result that is not finite.
- **A file that mentions the raw reader's name must be on the guard's list.** Keep the name
  out of `tests/scripts/test_finqa_gold_check.py`; build its data through
  `tests/data/synthetic.py`.
- **A check that compares with a committed file fails once and rewrites** (as C9 does): the
  operator reviews the diff. `--check` never writes.
- **No example text in the report or the flags file**: ids, splits, reasons, counts.
- **Write files with `newline="\n"`; compare as text.**
- **The real run reads test rows.** Print counts and ids only; never a program or an answer.

---

## Implementation

### Files

```yaml
CREATE:
  - config/evaluation.yaml
  - qlora_finetuning/evaluation/metrics.py: Allowance, EvaluationConfig, EvaluationError,
      load_evaluation_config, numbers_match, values_match, read_answer, Gold, programs_match,
      mismatch_label, ExampleScore, score_reply, summarize
  - qlora_finetuning/data/gold_check.py: check_example, build_report, build_flags,
      render_markdown, write_report, write_flags, stale_files, load_gold_flags
  - scripts/finqa_gold_check.py
  - tests/evaluation/test_metrics.py, tests/data/test_gold_check.py,
      tests/scripts/test_finqa_gold_check.py
  - reports/data/finqa_gold_check_report.md, .json; splits/finqa_gold_flags.json (generated)
  - reqspecs/impl_summary/impl_002_<TIMESTAMP>.md

MODIFY:
  - qlora_finetuning/evaluation/__init__.py: docstring, public names
  - tests/data/synthetic.py: build_frozen_split(rows=...)
  - tests/data/test_split_loader.py: one new allowed file, and a test of the list's size
  - docs/runbook.md (section 9), coding.md, gotchas.md, reply_scoring.md,
    split_loading.md, splits/README.md, high_level_plan.md (4.4)
  - the spec, req_002_roadmap.md, execution_plan.md
```

### Design

```python
# evaluation/metrics.py
@dataclass(frozen=True)
class Allowance: absolute: float; relative: float
def numbers_match(value, gold, allowance):      # D1
    return abs(value - gold) <= max(allowance.absolute, allowance.relative * abs(gold))
def values_match(value, gold, allowance)        # yes/no exactly; a number never matches yes/no
def read_answer(text) -> tuple[Value, ...]      # (), ("yes",), (n,), or (n, n / 100) for "n%"
class Gold: program: Program | None; value: Value; program_text; answer_text
    from_target(target, reply_format, dsl_config=None)
def programs_match(first, second)               # D14, D15; numbers by value
def mismatch_label(value, gold, config)         # kind, sign, scale, other; None when right
def score_reply(outcome, gold, config, example_id=None) -> ExampleScore
def summarize(scores, flagged=()) -> {"all": block, "unflagged": block, "flagged_examples": n}

# data/gold_check.py
def check_example(example, prompt, config, reply_format, dsl_config) -> GoldCheck
#   record = build_record(example, prompt)          (SerializationError -> "unserializable")
#   outcome = run_reply(record.target, example.table, reply_format, dsl_config)
#   gold = Gold.from_target(record.target, ...)     (EvaluationError -> "answer_unreadable")
#   score = score_reply(outcome, gold, config, example.id)
#   reason = outcome.error_code, or "value_mismatch", or None
def build_report(examples, manifest, prompt, config, previous_flags=None, data_problem=None)
def build_flags(report) -> dict                 # ids, splits, reasons, settings, a hash
def load_gold_flags(config=None) -> frozenset[str]
```

Checks K1 to K7 as in spec 4.3, with two adjustments to be written back into it:

- **K6 is "the manifest describes this data"**, computed by the script with
  `check_manifest_against_data` and passed in. Comparing with two other committed reports
  would tie the pure report function to files it does not own.
- **Reasons for a flag** are an error code, `value_mismatch`, `answer_unreadable`, or
  `unserializable`. A gold reply that is not strict or not canonical fails K2; it is not a
  flag, because D19 defines a flag by execution only.

### Script

`main(argv=None, cfg=None, data_cfg=None, prompt=None, config=None)`. Reads every row with
the raw reader and the manifest, builds the report twice for K7, prints the check lines and
one line per split, writes the report and the flags file unless `--check`. Exit 0, 1, 2.

### Tests

| File | Cases |
|---|---|
| `test_metrics.py` | Spec 4.5: comparison, answer reader, program match, mismatch labels, `score_reply`, `summarize`, config; the spec's usage example; no module-level import of the data package |
| `test_gold_check.py` | Clean rows; one broken gold of each reason; nothing dropped; each check passing and failing; no invented text in the output; flags round trip and hash; deterministic; `load_gold_flags`; committed files match a fresh run (skips without the snapshot) |
| `test_finqa_gold_check.py` | Exit codes; second run unchanged; `--check` before a build and after an edit; a flagged test example leaves exit 0; a flagged fit example exits 1; module form; by-path hint |

### Tasks (~4 hours total)

1. **Config and `metrics.py`** (~50 min).
2. **Metrics tests** (~45 min).
3. **`gold_check.py`** (~45 min).
4. **Script, synthetic helper, tests** (~50 min).
5. **Real run over 8,281 examples; reconcile fit and dev with spec section 2** (~10 min).
6. **Documents** (~30 min).
7. **Validation with the stack blocked, status documents, summaries** (~15 min).

### Error handling

| Situation | Behavior |
|---|---|
| Config value malformed | `ConfigError` naming the key |
| A gold target that cannot be read at all | `EvaluationError` from `Gold.from_target`; the self-check turns it into a flag |
| A gold program that fails or does not match | The example is flagged; the script still exits 0 unless it is in fit or dev (K3) |
| Manifest missing or for other data | Exit 2 naming the command to run, or K6 fails |
| Flags file missing when `load_gold_flags` is called | `EvaluationError` naming `python -m scripts.finqa_gold_check` |
| Committed report or flags differ under `--check` | "OUT OF DATE", exit 1 |

---

## Validation

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q
venv\Scripts\python -m scripts.finqa_gold_check
venv\Scripts\python -m scripts.finqa_gold_check --check
```

**Expected**: ruff clean; all tests pass; seven PASS lines; fit 4,884 and dev 499 reproduce
in full; `--check` exits 0. Also the unit tests with the training libraries blocked.

---

## Checklist

- [x] Files created and modified as listed
- [x] Every function and every check has passing and failing tests
- [x] Lint clean; tests green with and without the training stack
- [x] Real run shown and reconciled with the spec; the test split's result recorded
- [x] Runbook, references, plan, `coding.md` updated
- [x] Roadmap, execution plan, spec status updated; implementation summary created

---

## Implementation Time Estimate

**Total: 3.5–4.5 hours**

| Task | Time |
|---|---|
| Config and `metrics.py` | 50 min |
| Metrics tests | 45 min |
| `gold_check.py` | 45 min |
| Script, synthetic helper, tests | 50 min |
| Real run | 10 min |
| Documents | 30 min |
| Validation and status documents | 15–30 min |

## Confidence Score: 8/10

The comparison rule was measured on fit and dev with the built executor, and the report and
script follow three existing ones. Two points are held back: the test split's gold programs
have never been executed, so the first real run may show forms that fit and dev do not have;
and the answer reader is new code that meets model text.

## Time Taken for PRP Generation

**~9 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 3 min |
| Research existing PRPs and patterns | 1 min |
| Writing PRP document | 3 min |
