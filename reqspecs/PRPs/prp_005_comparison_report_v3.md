# PRP: Comparison report (req_005 v5.03)

**Feature Name**: comparison_report (comparison v5.03)  
**Version**: 3  
**Created**: 2026-11-23  
**Status**: ✅ IMPLEMENTED (rebuilt on the reconciled cost record on 2026-11-24; the report is final, M8)  
**Implemented**: 2026-11-23  
**Implementation Summary**: [impl_005_20261124_015500.md](../impl_summary/impl_005_20261124_015500.md)  
**Confidence Score**: 8/10  
**Estimated Time**: 5–6 hours of code and documents; then the owner's reading of the report and a rebuild after the bill is entered (minutes, no charge)  
**Origin**: `reqspecs/reqs/req_005_comparison/req_005_03_spec.md`  
**Template**: `prp_base.md` (seven new modules, one script, two config files, one template)  

---

## Goal

`python -m scripts.comparison_report` (`check`, `build`, `examples`) writes the answer to
the project's question from what is already stored:

1. `reports/comparison/comparison_statistics.json` and `.md`: a confidence interval for
   every accuracy figure, a paired test for six named pairs of arms, the thirteen error
   categories for all six arms, accuracy by gold steps.
2. `reports/comparison/accuracy_cost.svg`: execution accuracy against cost per 1,000
   questions.
3. `reports/comparison_report.md`: the report, built from a template that holds no number.
4. A generated block in `README.md`, between two marker comments.
5. `artifacts/comparison/error_examples.md` (git-ignored): example text for the owner.

**No AWS call, no sign-in, nothing billed, no model run, no frozen file changed.**

## Why

The six arms have test results (v5.01, M7) and prices (v5.02). Nothing yet says how sure a
difference between two arms is, or where each arm goes wrong on test, and there is no
report. Committing the report is milestone M8.

## What is decided

The spec's D1 to D20 and the owner's nine decisions (section 8), all taken as recommended on
2026-11-23:

- Six tested pairs: each tuned arm against its untuned model, and the three tuned arms
  against each other. Holm's method over the six.
- Two intervals: by example (main) and by company. A difference whose company interval
  includes zero is called **not settled**, whatever its p-value.
- Committed files hold counts and example ids only. Example text goes to a git-ignored file.
- The chart is an SVG from the project's own code. No new dependency.
- The report is `reports/comparison_report.md`, with a generated README block.
- It is built now on the provisional cost record and is itself marked provisional.
- It names no single winner.
- The PRP drafts the deviations list; the owner reads it.
- This version reads test rows on this machine (principle 6 of the execution plan was
  amended the same day).

## All needed context

### Read first

| File | Why |
|---|---|
| `reqspecs/reqs/req_005_comparison/req_005_03_spec.md` | The spec: D1 to D20, R1 to R8, the report's twelve sections |
| `reqspecs/1. PROMPTS/coding.md` | Conventions: scoring only through `evaluation`, no dataset text committed, scripts as modules |
| `qlora_finetuning/comparison/cost_config.py`, `cost_report.py`, `scripts/comparison_cost.py` | The pattern to mirror: typed config, `render_markdown`, `write_record`, `stand_ins` in `main`, exit codes 0/1/2 |
| `qlora_finetuning/comparison/gate.py` | `canonical_json`, `file_sha256`, `shared_state`, `shared_differences`, `read_freeze` |
| `qlora_finetuning/evaluation/failures.py` | `classify`, `CATEGORIES`, `company_of`; how `run_failures` rescored a stored run |
| `qlora_finetuning/evaluation/runner.py` | `RunnerContext`, `run_predictions`, `run_gold` |
| `qlora_finetuning/evaluation/metrics.py` | `score_reply`, `summarize`, `ExampleScore` |
| `tests/comparison/helpers.py`, `tests/comparison/test_runs.py` | A frozen, unlocked invented split with 22 test examples; `run_arm` with a stand-in predictor |
| `reqspecs/references/project_patterns/canonical_record.md`, `check_script.md`, `pricing_a_ledger_record.md` | How a committed record and a check script are built here |

### What the real data looks like (verified 2026-11-23)

- **Six prediction files** are on this machine, `artifacts/runs/<run id>/predictions.jsonl`,
  2,898 lines each, the same example ids in the same order. A line has `example_id`,
  `outcome`, `score`, `stop_reason`, `reply`. `run_predictions(run_id, ctx)` checks a file
  against the split's ids; `run_gold(run_id, ctx)` gives the gold replies in the same order.
  Both load the test split, which is unlocked.
- **Six-arm record**: `arms[arm]` has `run_id`, `all.<metric>.count` for
  `execution_accuracy`, `answer_accuracy`, `program_match`, `format_validity`,
  `all.examples`, `latency_seconds` (mean, median, p95), `stop_reasons`, `model_id`
  (a deployment ARN for a tuned Nova arm: never printed), `path`; top level `passed`,
  `freeze_sha256`, `test_examples`, `deviations`.
- **Cost record**: `passed`, `provisional`, `six_arm_freeze_sha256`, `test_pass[arm]`
  (`run_id`, `correct`, `usd` or `billable_seconds`, `per_1000_questions`,
  `per_1000_questions_steady`, `per_1000_correct`), `making[arm]` (`train`, `select`,
  `making_usd`, `storage`), `serving[arm]`, `break_evens[]` (`pair`, `scenario`, `kind`,
  `questions`, `cheaper_below`, `cheaper_above`, `cheaper_always`, `reachable`,
  `limit_note`), `project.paths`, `prices`. Amounts are text with six decimals.
- **Freeze record** (`splits/config_freeze.json`): `arms[arm].dev` holds the dev counts
  (`examples`, `execution_accuracy`, `answer_accuracy`, `program_match`,
  `format_validity`); `shared` holds the hashes that `shared_differences` compares.
- **The split is grouped by company** (`config/split.yaml`: `group_by: company`). The test
  split has 43 companies; `company_of(example_id)` gives the company.
- **Libraries**: `numpy` and `pandas` are in `requirements.txt`. `matplotlib`, `scipy`, and
  `statsmodels` are not installed and must not be added.

### Hand figures the record must reproduce

Right on execution, of 2,898: A0 27, A1 2,121, B0-Lite 484, B1-Lite 2,069, B0-Pro 1,025,
B1-Pro 2,179.

| First, second | Both right | Only first | Only second | Both wrong | Exact p |
|---|---:|---:|---:|---:|---:|
| A1, A0 | 25 | 2,096 | 2 | 775 | below 0.0001 |
| B1-Lite, B0-Lite | 452 | 1,617 | 32 | 797 | below 0.0001 |
| B1-Pro, B0-Pro | 974 | 1,205 | 51 | 668 | below 0.0001 |
| A1, B1-Pro | 1,958 | 163 | 221 | 556 | 0.00357 |
| A1, B1-Lite | 1,883 | 238 | 186 | 591 | 0.0132 |
| B1-Pro, B1-Lite | 1,934 | 245 | 135 | 584 | 1.8e-08 |

All three tuned arms right: 1,807. At least one: 2,401. None of the six: 467.

### Gotchas

- **The bootstrap must give the same bytes everywhere.** Use `numpy.random.RandomState(seed)`
  and its `randint`: the legacy generator's stream is frozen (NEP 19). `default_rng` may
  change between numpy versions. Resample in chunks: 10,000 by 2,898 indices at once is
  230 MB.
- **A company resample changes the denominator.** The rate of a resample is right answers
  over examples in the resampled companies, not over 2,898.
- **The exact test uses whole numbers** (`math.comb`, `fractions.Fraction`). A float
  underflows: one p-value here is near 1e-287.
- **No rate by hand for an arm's totals**: the counts come from `summarize(scores)` and must
  equal the six-arm record (R3). The bootstrap resamples the per-example results.
- **A tuned Nova arm's `model_id` is a deployment ARN with the account number.** Name
  models by arm and base model; R8 fails on `arn:aws` in any written file.
- **`README.md` starts with a byte order mark and may have CRLF line ends.** Keep both.
- **Git checks files out with CRLF here**: write records with `newline="\n"`.
- **A script or helper name must not start with `test_`.** The module is `statistics.py`
  inside the package; import it absolutely (`qlora_finetuning.comparison.statistics`) so the
  standard library's `statistics` is never shadowed.
- **Another session may edit the same tree** (req_006). Touch only the README block.

## Implementation blueprint

### Files

| File | Holds |
|---|---|
| `config/report.yaml` | Record paths, output paths, bootstrap (resamples 10,000, seed 20261115, confidence 95), the six pairs, the step groups, examples per category, the smallest p shown, chart size |
| `config/report_deviations.yaml` | The deviations list: date, what, where it is recorded |
| `qlora_finetuning/comparison/report_config.py` | `ReportConfig`, `load_report_config`, `load_deviations` |
| `qlora_finetuning/comparison/statistics.py` | `bootstrap_rates`, `interval`, `mcnemar_exact`, `holm`. Pure: lists of true and false in, numbers out |
| `qlora_finetuning/comparison/outcomes.py` | `ArmOutcomes`, `read_arm`, `read_arms`: per example the four scores, the category, the gold steps, the company. No text |
| `qlora_finetuning/comparison/report_record.py` | `ReportInputs`, checks R1 to R7, `build_statistics` |
| `qlora_finetuning/comparison/report_tables.py` | Markdown tables and generated sentences from the record and the cost record; `placeholders` |
| `qlora_finetuning/comparison/chart.py` | `render_chart`: the SVG |
| `qlora_finetuning/comparison/report_build.py` | The template, R8, the README block, the examples file, `write_outputs` |
| `qlora_finetuning/comparison/report_template.md` | The report's prose with `{{placeholders}}` |
| `scripts/comparison_report.py` | `check`, `build`, `examples` |
| `tests/comparison/test_statistics.py`, `test_outcomes.py`, `test_report_record.py`, `test_report_build.py`, `test_chart.py`, `test_report_config.py`, `report_helpers.py`; `tests/scripts/test_comparison_report.py` | Tests, invented data only |

### Approach

```python
# statistics.py
def bootstrap_rates(columns, groups, resamples, seed, chunk=500):
    # columns: {name: [bool]} of equal length; groups: None, or one group label per example
    # returns {name: ndarray of `resamples` rates}; one set of resampled indices for all columns
    state = numpy.random.RandomState(seed)
    # by example: idx = state.randint(0, n, size=(chunk, n)); counts = matrix[idx].sum(axis=1)
    # by group:   per-group sums and sizes; idx over groups; rate = sums / sizes
def interval(values, confidence) -> (low, high)        # percentiles of the resampled values
def mcnemar_exact(only_first, only_second) -> Fraction # two-sided, min(1, 2 * tail)
def holm(p_values) -> list[Fraction]                   # step-down, monotone, capped at 1

# outcomes.py
def read_arm(arm, run_id, ctx, golds=None) -> ArmOutcomes
    # lines = run_predictions(run_id, ctx); golds = run_gold(run_id, ctx) once
    # score_reply + classify per line, as failures.run_failures does; summarize() for totals

# report_record.py
def build_statistics(inputs, cfg) -> dict
    # R1 records; R2 replies; R3 counts; R4 frozen files; then the figures; R5; R6; R7
    # record: arms (counts, rates, both intervals, categories, by steps, latency, dev),
    #         pairs (four counts, difference, both intervals, p, adjusted p, settled),
    #         overlap, costs copied as text, checks, passed, provisional
```

- **Rates in the record** are text percentages with two decimals; p-values are text in
  scientific form with three digits, and shown as "below 0.0001" under the smallest p.
- **Settled** (decision 2): the adjusted p is below 1 minus the confidence, and neither
  interval of the difference includes zero. Otherwise "not settled".
- **Seeds**: the example bootstrap uses `seed`, the company bootstrap `seed + 1`.
- **The template** has placeholders `{{name}}`. `placeholders(record, cost, deviations)`
  returns every value. R8: no placeholder unfilled, no value unused, no `arn:aws`, no
  account-number-like run of 12 digits, every example id is a test id, `check_no_secrets`.
- **Findings** are generated sentences with branches (tuning gains, each tuned pair with
  its verdict, the cost per 1,000 questions, each break-even with its reachability), so the
  template prose claims nothing a changed record could make false.
- **README block**: between `<!-- comparison:start -->` and `<!-- comparison:end -->`;
  appended once when the markers are missing.
- **`examples`** writes only under `artifacts/`; it refuses any other folder.

### Error handling

- `ConfigError` for a bad config, a missing record, an unreadable run: `CONFIG ERROR: ...`,
  exit 2.
- A failed check: every check is printed, nothing is written, exit 1.
- A provisional cost record: R1 passes with a reservation; outputs carry **provisional**.

### Tasks, in order

1. `config/report.yaml`, `report_config.py`, tests.
2. `statistics.py`, tests (bootstrap, exact test, Holm).
3. `outcomes.py`, tests on the invented split with stand-in predictors.
4. `report_record.py`: R1 to R7 and the record; tests.
5. `chart.py`, tests.
6. `report_tables.py`, the template, `report_build.py` (R8, README block, examples); tests.
7. `scripts/comparison_report.py`, script tests, guards.
8. `config/report_deviations.yaml`, drafted from the roadmaps' version logs.
9. Run `check` and `build` on the real repository; compare with the hand figures; a second
   build changes no byte; read the report.
10. Documents: runbook 7.11, run sheet, a project pattern file, coding notes.
11. Lint, full tests, with and without the training stack; spec section 11, roadmap,
    execution plan, WIP, implementation summary.

## Validation gates

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q
venv\Scripts\python -m pytest -q tests/comparison tests/scripts/test_comparison_report.py
venv\Scripts\python -m scripts.comparison_report check
venv\Scripts\python -m scripts.comparison_report build
venv\Scripts\python -m scripts.comparison_report build      # no byte changes
```

## Success criteria

- [x] `pytest -q` and `ruff check .` pass, with and without the training stack blocked.
- [x] `comparison_report check` passes R1 to R8 on the real repository, with a reservation
      only for the provisional cost record.
- [x] The pairs' counts and p-values equal the hand figures above.
- [x] A second `build` changes no byte.
- [x] No committed output holds a reply, a question, a gold program, or an ARN.
- [x] No AWS call, no model run, no frozen file changed; no new dependency.
- [x] Runbook, run sheet, pattern file, coding notes, spec section 11, roadmap, execution
      plan updated.
- [x] For the owner: read the report and the deviations list; after the bill is entered,
      build again and commit (M8).

## Implementation Time Estimate

**5–6 hours**

| Task | Time |
|---|---|
| Config and loaders, tests | 25 min |
| Statistics, tests | 45 min |
| Outcomes, tests | 40 min |
| The record and checks R1 to R7, tests | 50 min |
| Chart, tests | 35 min |
| Tables, template, build, README block, examples, tests | 75 min |
| Script, script tests, guards | 25 min |
| Deviations list | 15 min |
| Real-repository run and reading the report | 20 min |
| Documents | 25 min |
| Validation and status documents | 25 min |

## Confidence: 8/10

The inputs are verified and there are hand figures for the pairs. The statistics are small
and exact. The risks: the generated findings have branches that the real data exercises
only one way; the rescored counts must equal the six-arm record on the first run (R3); and
the report's prose needs a careful reading that tests cannot give.

## Time Taken for PRP Generation

**~4 minutes** by the clock (23:29 to 23:33)

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 1 min |
| Codebase exploration | 1.5 min |
| Research existing PRPs and patterns | 0.5 min |
| Writing PRP document | 1 min |

The spec was written in the same session, so the stored runs' shapes and the paired counts
were already known.
