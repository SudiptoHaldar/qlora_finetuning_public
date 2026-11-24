# PRP: Token budget report (req_001 v1.04)

**Feature Name**: token_budget (finqa_data v1.04)  
**Version**: 4  
**Created**: 2026-11-15  
**Status**: ✅ IMPLEMENTED  
**Implemented**: 2026-11-15  
**Implementation Summary**: [impl_001_20261115_144212.md](../impl_summary/impl_001_20261115_144212.md)  
**Confidence Score**: 9/10  
**Estimated Time**: 3–4 hours  
**Origin**: `reqspecs/reqs/req_001_finqa_data/req_001_04_spec.md`  
**Template**: `prp_lean.md` (expanded, medium complexity)  

---

## Goal

Count, with the pinned Qwen3-4B tokenizer, how long every fit and dev record is as a full
training sequence; commit the result as a report with checks; and put the training length
(2,048) and the reply limit (192) in config for the two paths to read.

## Why & What

- **User value**: no example is silently cut, the reply limit is known to be long enough, and
  the hosted path's training volume is a measured number before any money is spent.
- **Technical**: one config file, two modules in `qlora_finetuning/data/`, one script, a
  committed report, tests with a stand-in tokenizer.

## Success Criteria

- [ ] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, also
      with the training libraries blocked.
- [ ] `venv\Scripts\python -m scripts.finqa_tokens` passes T1 to T6 and writes the report; a
      second run changes nothing; `--check` exits 0.
- [ ] The report agrees with spec section 2: system text 372 tokens; fit full sequences 643 at
      the 50th percentile and 1,862 at most; 2 fit records over 1,536, none over 2,048;
      3,254,181 tokens in one pass over fit.
- [ ] The report holds no example text and no dollar figure.
- [ ] `high_level_plan.md` and the execution plan's cost table carry the measured values.

---

## Context

### References

```yaml
refs:
  - reqspecs/references/project_patterns/canonical_record.md  # the three texts; never edited
  - reqspecs/references/project_patterns/split_loading.md     # load_split; tests on build_frozen_split
  - reqspecs/references/tech/hf_training_stack.md             # chat template, enable_thinking=False, eos
  - reqspecs/references/project_patterns/check_script.md      # PASS/FAIL lines, exit codes
  - reqspecs/references/common/gotchas.md                     # sections 4 and 5

code:
  - file: qlora_finetuning/data/serialization_report.py
    why: length_stats (nearest rank), report shape, checks, write_report, stale_files, read_committed_digests
  - file: scripts/finqa_serialize.py
    why: main(argv, cfg, data_cfg, prompt) shape; how records are built per split
  - file: scripts/env_check.py
    why: importing the training stack inside a function; apply_chat_template call
```

### Facts verified 2026-11-15

| Fact | Value |
|---|---|
| Tokenizer | `Qwen/Qwen3-4B` at `1cfa9a7208912126459214e8b04321603b3df60c`; files are in the local Hugging Face cache; `local_files_only=True` loads them |
| Content fingerprint | SHA-256 of `json.dumps([sorted(vocab.items()), chat_template, eos_token], ensure_ascii=True)` is `e2d3cbac3923643bb0a5d1f0045aad8728663a6ff2f6df7a4872ee6a4a218fae` |
| Batch call | `tokenizer(list_of_texts, add_special_tokens=False)["input_ids"]` is a list of id lists |
| Prompt | `tokenizer.apply_chat_template([{"role": "system", ...}, {"role": "user", ...}], tokenize=False, add_generation_prompt=True, enable_thinking=False)` ends with `<think>\n\n</think>\n\n` |
| Template overhead | `prompt - system - user` is 17 tokens for all 5,383 records |
| Target | Counted as `target + tokenizer.eos_token` |
| Spec section 2 | The expected numbers; percentiles are nearest rank |

### Critical Gotchas

- **`transformers` is training stack**: import it inside `load_tokenizer`. A missing library
  or missing cached files raises `TokenBudgetError` naming the fix; the script exits 2.
- **A loaded tokenizer does not reliably expose the commit it came from.** Check T3 uses the
  content fingerprint above, pinned in config, which proves the tokenizer is the right one
  however it was loaded.
- **The stand-in tokenizer in tests** splits on spaces and has its own tiny chat template; the
  thinking-off suffix and the fingerprint are config values, so tests replace them.
- **Percentiles are nearest rank** (`length_stats` in `serialization_report.py`), so the
  report's 50th percentile is a whole number.
- **No dollars and no example text in the report**; ids and counts only.
- **T6 reads the committed serialization report.** If it is missing, the check fails with the
  command to run first.
- **The test with the real tokenizer skips** when `transformers`, the cached tokenizer, or
  the snapshot is missing, so CI and the blocked run stay green.
- **Write files with `newline="\n"`; compare as text.**

---

## Implementation

### Files

```yaml
CREATE:
  - config/token_budget.yaml
  - qlora_finetuning/data/token_budget.py: TokenBudgetConfig, TokenBudgetError, TokenCounts,
      load_token_budget_config, load_tokenizer, tokenizer_fingerprint, render_prompt,
      count_record, count_split
  - qlora_finetuning/data/token_budget_report.py: statistics, checks T1-T6, JSON and Markdown
  - scripts/finqa_tokens.py
  - tests/data/test_token_budget.py, test_token_budget_report.py, tests/scripts/test_finqa_tokens.py
  - reports/data/finqa_token_budget_report.md, .json (generated)
  - reqspecs/impl_summary/impl_001_<TIMESTAMP>.md

MODIFY:
  - qlora_finetuning/data/__init__.py: public names
  - tests/data/synthetic.py: stand-in tokenizer
  - docs/runbook.md, reqspecs/1. PROMPTS/coding.md, gotchas.md, troubleshooting.md,
    hf_training_stack.md
  - high_level_plan.md (3, 4.6), execution_plan.md (cost table, status, P1), roadmap, spec
```

### Design

```python
# data/token_budget.py
class TokenBudgetError(ConfigError): ...
@dataclass(frozen=True)
class TokenBudgetConfig: root, model_id, revision, fingerprint, thinking_off_suffix,
                         max_length, max_new_tokens, thresholds, longest_listed, report_dir
def load_token_budget_config(root=REPO_ROOT, config_dir=None) -> TokenBudgetConfig
def load_tokenizer(cfg, download=False) -> Any
def tokenizer_fingerprint(tokenizer) -> str
def render_prompt(record, tokenizer) -> str
@dataclass(frozen=True)
class TokenCounts: example_id, system, user, target, prompt, thinking_off
    total = prompt + target;  template = prompt - system - user
def count_split(records, tokenizer, thinking_off_suffix) -> list[TokenCounts]   # batched
def count_record(record, tokenizer, thinking_off_suffix) -> TokenCounts

# data/token_budget_report.py
def split_stats(records, counts, cfg) -> dict
def run_checks(by_split, cfg, fingerprint, prompt_id, committed_digests) -> list[dict]  # T1..T6
def build_report(by_split, cfg, inputs, committed_digests) -> dict
def render_markdown, write_report, stale_files
```

### Script

`main(argv=None, cfg=None, data_cfg=None, prompt=None, budget=None, tokenizer=None)`. Loads
the tokenizer unless one is passed, builds the fit, dev, and quick-dev records, counts, runs
the checks over fit and dev, writes the report unless `--check`. `--download` lets the
tokenizer be fetched. Exit 0, 1, 2.

### Tests

| File | Cases |
|---|---|
| `test_token_budget.py` | Real config loads; bad revision, fingerprint, limits, thresholds; counts for a known record with the stand-in; batch equals single; target includes the end token; thinking flag; fingerprint stable and sensitive; `load_tokenizer` without the library and without cached files |
| `test_token_budget_report.py` | Statistics, thresholds, longest list, totals and system share; each of T1 to T6 passing and failing; deterministic; no example text, no dollar sign; write and stale; committed report facts; real tokenizer comparison (skips) |
| `test_finqa_tokens.py` | Exit codes; second run unchanged; `--check`; missing tokenizer exits 2; module form; by-path hint |

### Tasks (~3.5 hours total)

1. **Config and `token_budget.py`** (~40 min).
2. **`token_budget_report.py`** (~45 min).
3. **Stand-in tokenizer and tests** (~55 min).
4. **Script and tests** (~25 min).
5. **Real run and comparison with the spec** (~10 min).
6. **Documents, cost table** (~30 min).
7. **Validation with the stack blocked, status documents, summaries** (~15 min).

### Error handling

| Situation | Behavior |
|---|---|
| Config value malformed | `ConfigError` naming the key |
| `transformers` missing | `TokenBudgetError`: install the training stack (runbook 3.1); exit 2 |
| Tokenizer not in the local cache | `TokenBudgetError`: run with `--download`; exit 2 |
| A record longer than `max_length`, or a reply longer than `max_new_tokens` | T1 or T2 fails naming the ids; report written; exit 1 |
| Wrong tokenizer | T3 fails |
| Serialization report missing or for other records | T6 fails naming the command to run |

---

## Validation

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q
venv\Scripts\python -m scripts.finqa_tokens
venv\Scripts\python -m scripts.finqa_tokens --check
```

**Expected**: ruff clean; all tests pass; six PASS lines; the numbers of spec section 2;
`--check` exits 0. Also the unit tests with the training libraries blocked.

---

## Checklist

- [ ] Files created and modified as listed
- [ ] Every function and every check has passing and failing tests
- [ ] Lint clean; tests green with and without the training stack
- [ ] Real run shown and reconciled with the spec
- [ ] Runbook, plan, cost table, references updated
- [ ] Roadmap, execution plan, spec status updated; implementation summary created

---

## Implementation Time Estimate

**Total: 3–4 hours**

| Task | Time |
|---|---|
| Config and `token_budget.py` | 40 min |
| `token_budget_report.py` | 45 min |
| Stand-in tokenizer and tests | 55 min |
| Script and tests | 25 min |
| Real run | 10 min |
| Documents and cost table | 30 min |
| Validation and status documents | 15–30 min |

## Confidence Score: 9/10

Every number was measured for the spec with the same calls the code will make, and the report
follows two existing report modules. The point held back is for the stand-in tokenizer, which
has to behave enough like the real one for the checks to mean something.

## Time Taken for PRP Generation

**~6 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 1 min |
| Codebase exploration | 1 min |
| Research existing PRPs and patterns | 1 min |
| Writing PRP document | 3 min |
