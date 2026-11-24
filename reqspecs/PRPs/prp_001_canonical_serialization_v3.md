# PRP: Canonical serialization (req_001 v1.03)

**Feature Name**: canonical_serialization (finqa_data v1.03)  
**Version**: 3  
**Created**: 2026-11-15  
**Status**: ✅ IMPLEMENTED  
**Implemented**: 2026-11-15  
**Implementation Summary**: [impl_001_20261115_133935.md](../impl_summary/impl_001_20261115_133935.md)  
**Confidence Score**: 8/10  
**Estimated Time**: 4–5 hours  
**Origin**: `reqspecs/reqs/req_001_finqa_data/req_001_03_spec.md`  
**Template**: `prp_lean.md` (expanded, medium complexity)  

---

## Goal

Turn each FinQA example into one canonical record (system, user, and target text) from a
prompt definition kept in `config/prompt.yaml`, the same bytes every time, so that all six
arms read identical words and are asked for identical output.

## Why & What

- **User value**: the comparison between the paths cannot be skewed by a difference in
  wording, and any change of wording is visible as a new prompt id.
- **Technical**: one config file, two modules in `qlora_finetuning/data/`, one script, a
  committed report with digests, golden tests on invented rows.

## Success Criteria

- [ ] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, also
      with the training libraries blocked.
- [ ] `venv\Scripts\python -m scripts.finqa_serialize` produces 4,884 fit and 499 dev records;
      C1 to C10 pass; a second run changes nothing; `--check` exits 0.
- [ ] The report agrees with spec section 2: 3,387 records without text evidence, one skipped
      key, 138 answers rewritten from scientific notation.
- [ ] `serialize_split("test")` raises `TestLockedError`.
- [ ] No committed file contains real example text; the golden files are invented.
- [ ] The prompt id is recorded in the req_001 roadmap.
- [ ] `env_check --smoke` still passes with its example built by the serializer.

---

## Context

### References

```yaml
refs:
  - reqspecs/references/project_patterns/split_loading.md  # load_split, the lock, tests on build_frozen_split
  - reqspecs/references/tech/finqa_dataset.md              # columns; the rule on committed text
  - reqspecs/references/project_patterns/check_script.md   # PASS/FAIL lines, exit codes
  - reqspecs/references/tech/hf_training_stack.md          # what the smoke step does with a prompt
  - reqspecs/references/common/gotchas.md                  # sections 4 and 5

code:
  - file: qlora_finetuning/data/split_loader.py
    why: load_split and split_ids; serialize_split goes through them
  - file: qlora_finetuning/data/split_report.py
    why: report shape, checks list, write_report and stale_files compared as text
  - file: scripts/finqa_split.py
    why: main(argv, cfg, data_cfg) shape and --check
  - file: scripts/env_check.py
    why: holds a hand-written example (SMOKE_SYSTEM/USER/TARGET) that this version replaces
  - file: tests/data/synthetic.py
    why: invented rows, build_frozen_split
```

### Facts verified 2026-11-15 (fit and dev, through `load_split`)

| Fact | Value |
|---|---|
| `text_N` evidence equals passage `N` of `pre_text + post_text` | All 2,694 entries |
| Examples without text evidence | 3,387 of 5,383 |
| Keys out of document order; an example with the same passage text twice | 59 examples; 1 example |
| Malformed key | `text_-1` in `RE/2010/page_120.pdf-2`, which is in fit |
| Cells with a line break, `\|`, backslash, or outer spaces | None |
| `exe_ans` forms | `-?\d+` 1,074; `-?\d+\.\d+` 4,058; scientific 138; `yes`/`no` 113; nothing else |
| `Decimal("4.39e+07")` formatted with `"f"` | `43900000`; `Decimal("5e-05")` gives `0.00005`. No digit changes |
| Programs | Already uniform; longest 122 characters |
| `config/local_env.yaml` | `smoke.max_seq_length: 256`, which the longer system text will exceed |

### Critical Gotchas

- **A YAML block scalar ends with a line break.** Strip the system text's trailing white space
  when loading; reject an empty one.
- **The prompt id must ignore comments and key order** but change with any character of the
  system text, a label, or the version: hash a canonical JSON of the parsed values, not the
  file bytes. Paths in the file are not part of it.
- **Use `Decimal`, never `float`,** to write out scientific notation.
- **Text evidence comes from the passage list by index**, sorted, each text once. A key that
  is not `table_N` or `text_N`, or a text index outside the passages, is skipped and counted.
- **Cells are copied word for word.** Only a vertical bar is escaped and a line break turned
  into a space; neither occurs in the data.
- **Golden files hold invented rows only.** Generate them once, read them, then freeze them.
- **Real prompts may go to the terminal and to `artifacts/`, never to a committed file.**
- **`--show` refuses a test id** by looking at `split_ids("test")`, before loading anything.
- **The smoke step truncates at `smoke.max_seq_length`.** With a longer prompt the target
  would be cut off and the loss would be undefined: raise the value and run
  `env_check --smoke` to prove it.
- **The stale-content guard matches `\bticker` and `\bchart`** in `coding.md`.
- **Digest check C9** compares with the digests in the committed report. A deliberate change
  of wording fails it once, loudly, and rewrites the report; the next run passes.

---

## Implementation

### Files

```yaml
CREATE:
  - config/prompt.yaml
  - qlora_finetuning/data/serialization.py: PromptDefinition, CanonicalRecord, load_prompt,
      text_evidence, format_table, format_answer, format_user, format_target, build_record,
      serialize_split, records_digest
  - qlora_finetuning/data/serialization_report.py: statistics, checks C1-C10, JSON and Markdown
  - scripts/finqa_serialize.py
  - tests/data/test_serialization.py, test_serialization_report.py, tests/scripts/test_finqa_serialize.py
  - tests/data/golden/*.txt: four invented records
  - reports/data/finqa_serialization_report.md, .json (generated)
  - reqspecs/references/project_patterns/canonical_record.md
  - reqspecs/impl_summary/impl_001_<TIMESTAMP>.md

MODIFY:
  - qlora_finetuning/data/__init__.py: public names
  - scripts/env_check.py, config/local_env.yaml: smoke example built by the serializer
  - tests/data/synthetic.py: rows for the golden records
  - docs/runbook.md, reqspecs/1. PROMPTS/coding.md, references, high_level_plan.md 4.2
  - roadmap, execution plan, spec status
```

### Design

```python
# data/serialization.py
class SerializationError(ConfigError): ...
@dataclass(frozen=True)
class PromptDefinition: root, version, system, context_label, list_marker, no_context,
                        table_label, question_label, program_label, answer_label,
                        report_dir, dump_dir, prompt_id
def load_prompt(root=REPO_ROOT, config_dir=None) -> PromptDefinition
@dataclass(frozen=True)
class CanonicalRecord: example_id, prompt_id, system, user, target;  to_dict()
def text_evidence(example) -> tuple[list[str], list[str]]        # passages, skipped keys
def format_table(table) -> str
def format_answer(exe_ans, where="") -> str
def format_user(passages, table, question, prompt) -> str        # also serves free input later
def format_target(program, answer, prompt) -> str
def build_record(example, prompt) -> CanonicalRecord
def serialize_split(name, prompt=None, cfg=None, data_cfg=None) -> list[CanonicalRecord]
def records_digest(records) -> str

# data/serialization_report.py
def length_stats(values) -> {min, median, p90, p99, max}         # nearest rank, whole numbers
def split_stats(examples, records, prompt) -> dict
def run_checks(...) -> list[dict]                                # C1..C10
def build_report(by_split, prompt, previous_digests, test_ids) -> dict
def render_markdown, write_report, stale_files, read_committed_digests
```

Layout (exact):

```
Context:\n- <p1>\n- <p2>\n\nTable:\n| h1 | h2 |\n|---|---|\n| a | b |\n\nQuestion: <q>
Context:\n(none)\n\nTable:\n...
PROGRAM: <program>\nANSWER: <answer>
```

A table line is `"| " + " | ".join(cells) + " |"`; the separator is `"|" + "---|" * columns`.

### Script

`main(argv=None, cfg=None, data_cfg=None, prompt=None)`: serialize fit, dev, and quick-dev;
run the checks; write the report unless `--check`; `--dump` writes `fit.jsonl` and `dev.jsonl`
under the dump folder; `--show <id>` prints one record and does nothing else. Exit 0, 1, 2.

### Tests

| File | Cases |
|---|---|
| `test_serialization.py` | Prompt: real file loads; id stable against a comment and key order, changes with text, label, version; missing key, empty system, bad version. Evidence: order, repeats, `text_-1`, out of range, table keys ignored, unparsed `gold_inds`. Table: header, separator, empty cells, one row, bar and line break, wide, empty raises. Answer: each form, scientific large, small, negative, digits kept, bad value raises. Record: four golden files; program unchanged; `(none)`; determinism; digest order-sensitive. Split: fit and dev order; `test` locked. Pin of the real prompt id |
| `test_serialization_report.py` | Length statistics; statistics on invented rows; every check passes and fails; no invented words; deterministic; write and stale; real digests when the snapshot is present |
| `test_finqa_serialize.py` | Exit codes; second run unchanged; `--check`; changed wording fails C9 once; `--dump` location; `--show` for a fit id, a test id, an unknown id; module form; by-path hint |

### Tasks (~4.5 hours total)

1. **`config/prompt.yaml` and `serialization.py`** (~50 min).
2. **Golden records and serializer tests** (~50 min).
3. **`serialization_report.py` and tests** (~50 min).
4. **Script and tests** (~30 min).
5. **Real run**: build, compare with spec section 2, read a few prompts with `--show` (~15 min).
6. **Smoke example through the serializer; `env_check --smoke`** (~20 min).
7. **Documents** (~30 min).
8. **Validation with the stack blocked, status documents, summaries** (~15 min).

### Error handling

| Situation | Behavior |
|---|---|
| Prompt file missing a key, empty system text, bad version | `ConfigError` naming the key |
| Table with no rows; result that is neither a number nor yes/no | `SerializationError` naming the example |
| Malformed or out-of-range evidence key | Skipped, counted, listed in the report |
| `serialize_split("test")`, `--show` with a test id | `TestLockedError`; refusal with exit 1 |
| A check fails, or the digests changed | FAIL line; report still written; exit 1 |
| No snapshot or no manifest | `CONFIG ERROR` or `SPLIT ERROR` naming the command to run |

---

## Validation

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q
venv\Scripts\python -m scripts.finqa_serialize
venv\Scripts\python -m scripts.finqa_serialize --check
venv\Scripts\python -m scripts.env_check --smoke
```

**Expected**: ruff clean; all tests pass; ten PASS lines; 4,884 and 499 records; `--check`
exits 0; smoke 7/7. Also the unit tests with the training libraries blocked.

---

## Checklist

- [ ] Files created and modified as listed
- [ ] Every function and every check has passing and failing tests
- [ ] Lint clean; tests green with and without the training stack
- [ ] Real build shown and reconciled with the spec; prompt id recorded
- [ ] Smoke step passes on the serializer's record
- [ ] Runbook, `coding.md`, references, plan documents updated
- [ ] Roadmap, execution plan, spec status updated; implementation summary created

---

## Implementation Time Estimate

**Total: 4–5 hours**

| Task | Time |
|---|---|
| Prompt definition and `serialization.py` | 50 min |
| Golden records and serializer tests | 50 min |
| `serialization_report.py` and tests | 50 min |
| Script and tests | 30 min |
| Real run | 15 min |
| Smoke example | 20 min |
| Documents | 30 min |
| Validation and status documents | 15–30 min |

## Confidence Score: 8/10

The data was measured for the spec and needs no escaping, and the functions are small and
pure. The points held back: the smoke step in `env_check` changes and has to be proven on the
GPU, and the report's checks parse the rendered text, where an off-by-one is easy.

## Time Taken for PRP Generation

**~8 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 1 min |
| Codebase exploration | 1 min |
| Research existing PRPs and patterns | 1 min |
| Writing PRP document | 5 min |
