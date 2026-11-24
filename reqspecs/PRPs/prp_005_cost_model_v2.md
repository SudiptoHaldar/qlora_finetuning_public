# PRP: Cost model (req_005 v5.02)

**Feature Name**: cost_model (comparison v5.02)  
**Version**: 2  
**Created**: 2026-11-23  
**Status**: ✅ IMPLEMENTED (the owner's two bill entries were made on 2026-11-24; the record is final)  
**Implemented**: 2026-11-23  
**Implementation Summary**: [impl_005_20261123_173726.md](../impl_summary/impl_005_20261123_173726.md)  
**Confidence Score**: 8/10  
**Estimated Time**: 4.5–5.5 hours of code and documents; then the owner's two bill entries (minutes, no charge)  
**Origin**: `reqspecs/reqs/req_005_comparison/req_005_02_spec.md`  
**Template**: `prp_base.md` (two new modules, one script, one config file, one price added)  

---

## Goal

`python -m scripts.comparison_cost` (`check`, `report`, `bill`): one committed record,
`reports/comparison/cost_model.json` and `.md`, that states in dollars

1. what each tuned model cost to make,
2. what a question costs for each of the six arms (per 1,000 questions and per 1,000 correct
   answers),
3. when Path A or a Nova model is cheaper (break-even volumes, each marked reachable or
   not), and
4. whether the model's totals agree with the AWS bill.

Every figure is a ledger quantity times a price from `config/pricing.yaml`, as `Decimal`.
**No AWS call, no sign-in, nothing billed, no new run.**

## Why

The six arms have their test results (v5.01, M7). v5.03's report needs one place to take
every dollar figure from, as it takes every run id from the six-arm record.

## What is decided

The spec's D1 to D18, and the owner's seven decisions of its section 8, all taken as
recommended on 2026-11-23. In short:

- A tuned model's cost is its one training run, with A1's checkpoint selection beside it.
  Everything else is **development**, per path.
- Path A's serving cost: a **batch job**, and an **always-on endpoint** at 100%, 50%, 10%,
  and 1% utilization over 730 hours a month.
- The endpoint's hourly price is read from the public SageMaker price list and added to
  `config/pricing.yaml`. **Read on 2026-11-23 while writing this PRP**: `USE1-Host:ml.g5.xlarge`
  is **$1.408** an hour (price list version `20261123184802`, published 2026-11-23), the
  same amount as the training rate.
- S3, CloudWatch, and data transfer are left out and named as left out.
- Nova Pro's training charge is entered: $58.75 (`source: bill`).
- The model is reconciled with the bill by service. Storage shown per month, at 1 and 12.

## All needed context

### Read first

| File | Why |
|---|---|
| `reqspecs/reqs/req_005_comparison/req_005_02_spec.md` | The spec: D1 to D18, K1 to K8, the record's seven tables |
| `reqspecs/1. PROMPTS/coding.md` | Conventions: `Decimal` money, no boto3 in pure modules, scripts as modules |
| `qlora_finetuning/comparison/runs.py`, `scripts/comparison_runs.py` | The pattern to mirror: `_check`, `build_record`, `render_markdown`, `write_record`, exit codes 0/1/2, `stand_ins` in `main` |
| `qlora_finetuning/comparison/gate.py` | `canonical_json`, `write_canonical` (it runs `check_no_secrets`) |
| `qlora_finetuning/frontier/pricing.py` | `load_pricing`, `inference_cost`, `MILLION`: use, do not copy |
| `qlora_finetuning/frontier/tuning.py` | `committed_path_b(ledger_records, pricing)` |
| `qlora_finetuning/local_qlora/estimate.py` | `committed_path_a(cfg, tracking_cfg, record)`: adds the maximum of any unpulled job |
| `qlora_finetuning/tracking/ledger.py` | `LedgerRecord` fields; `Ledger(cfg).effective()` |
| `reqspecs/references/project_patterns/canonical_record.md`, `check_script.md` | How a committed record and a check script are built here |
| `tests/comparison/test_runs.py`, `tests/scripts/test_comparison_scripts.py` | Test style |

### What the real data looks like (verified 2026-11-23)

- **Ledger**: 87 lines, 2 replaced. A SageMaker job's record has `billable_seconds`,
  `instance_type`, `instance_count`, and `external_id` = the job's name; its `run_id` is the
  job's `launch_run_id`. A run made inside a job has its own record with `where: sagemaker`
  and **no** billable seconds: it is priced at zero, because its job is priced. A Bedrock
  evaluation record has `input_tokens`, `output_tokens`, and the **base** `model_id` even
  for a tuned model. A customization job's record has `phase: train`, `external_id` = the
  job's ARN, and either `billed_amount_usd` (`source: bill`) or `training_tokens`
  (`source: computed`). Both are billed now: $14.60 and $58.75.
- **Run date**: every `run_id` begins `YYYYMMDDThhmmssZ`. Use that date, not `recorded_at`:
  a bill record is written days after its job.
- **Six-arm record** (`reports/comparison/six_arm_test_runs.json`): `passed`, `arms[arm]`
  with `run_id`, `continues`, `path`, `model_id`, `input_tokens`, `output_tokens`,
  `all.examples`, `all.execution_accuracy.count`, and for a Qwen arm
  `jobs[{name, instance_type, billable_seconds}]`.
- **Job record** (`reports/local_qlora/sagemaker_jobs.json`): `jobs[]` with `name`, `task`
  (`smoke`, `train`, `score`, `evaluate`), `arm`, `launch_run_id`, `runs[].run_id`,
  `billable_seconds`, `pulled`.
- **Selection record** (`reports/local_qlora/adapter_selection.json`): `training_run`,
  `selected.eval_run_id`.
- **Tuning record** (`reports/frontier/nova_tuning.json`): `attempts[]` with `model`, `arm`,
  `attempt`, `job_arn`, `base_model_id`, `ended_at`, `billed_usd`, `model_deleted`.
- **Prices**: `pricing.yaml` has `bedrock.<model id>` and `sagemaker.training.<instance>`.
  `SageMakerConfig.cost_of` is seconds times the hourly price over 3,600.
- **The bill, read by the owner on 2026-11-23**: Bedrock $78.19 and SageMaker $8.24. Both
  equal the ledger **through 2026-11-22** to the cent, with storage counted as one
  model-month per tuned model ($1.95 each; both models were made in November, on the 19th
  and the 22nd, and each line reads "1 Model/month"). The test runs of 2026-11-23 were not
  on the bill yet. Expected once they are: Bedrock $81.98, SageMaker $10.87.
- **Account limits** (Service Quotas, read 2026-11-16, not adjustable): Nova Pro 25
  requests a minute, tuned Nova Pro 20, Nova Lite and tuned Nova Lite 200.

### Hand figures the record must reproduce (to the cent)

| Arm | Test pass | Per 1,000 questions |
|---|---:|---:|
| A0 | $1.63 (4,164 s) | $0.56 as run; $0.52 steady (3,833 s) |
| A1 | $1.00 (2,564 s) | $0.35 as run; $0.30 steady (2,240 s) |
| B0-Lite | $0.14 | $0.047 |
| B1-Lite | $0.13 | $0.044 |
| B0-Pro | $1.81 | $0.62 |
| B1-Pro | $1.71 | $0.59 |

Making: A1 train $5.88 (15,037 s), select $1.94 (4,965 s); B1-Lite $14.60; B1-Pro $58.75.
Totals: `committed_path_a` $10.87; Bedrock records $78.08; with storage $81.98.

### Gotchas

- **Never a float for money.** `wall_clock_seconds` is a float in the ledger: convert with
  `Decimal(str(value))`.
- **`committed_path_a` counts an unpulled job's maximum.** If K5 differs, say so; do not
  "fix" the difference.
- **An arm name does not identify a run** (dozens carry A1). Take the test runs from the
  six-arm record, the training job from the selection record, the tuning jobs from the
  tuning record.
- **Division by zero**: A0 has 27 correct answers, an invented arm may have none; a Nova
  price may be zero in a test.
- **A script name must not start with `test_`.** Helper names in modules must not either.
- Git checks files out with CRLF here; write records with `newline="\n"` as `runs.py` does.
- `cost.py` must not import boto3, the training stack, or `load_split`.

## Implementation blueprint

### Files

| File | Holds |
|---|---|
| `config/cost_model.yaml` | Record file names, the committed records' paths, scenarios (hours a month, utilizations, batch sizes, jobs a month), account limits with their source and date, the services of the reconciliation, the tolerance |
| `config/pricing.yaml` | **Add** `sagemaker.hosting` (its own `price_list` dates, `instances.ml.g5.xlarge: "1.408"`). Change nothing else |
| `qlora_finetuning/comparison/cost_config.py` | `CostConfig`, `load_cost_config`; `InstancePrices`, `load_instance_prices` |
| `qlora_finetuning/comparison/cost.py` | Categories, pricing a record, per-arm figures, scenarios, break-evens, storage, reconciliation, K1 to K8, `build_record`, `render_markdown`, `write_record`, the bill file |
| `scripts/comparison_cost.py` | `check`, `report`, `bill` |
| `tests/comparison/test_cost_config.py`, `test_cost.py`, `tests/scripts/test_comparison_cost.py` | Tests, invented data only |

### Approach

```python
# cost.py (pure; inputs are passed in, so tests need no files)
@dataclass
class CostInputs:
    ledger: list[LedgerRecord]          # effective records
    six_arm: dict | None                # the six-arm record
    jobs: dict                          # the job record
    selection: dict | None              # the selection record
    tuning: list[dict]                  # the tuning record's attempts
    pricing: Pricing                    # Bedrock prices
    instances: InstancePrices           # SageMaker training and hosting prices
    bills: list[dict]                   # the owner's entries
    committed: dict[str, Decimal]       # {"A": committed_path_a(...), "B": committed_path_b(...)}
    caps: dict[str, Decimal]

def place(record, index) -> Placed      # category, amount, basis, estimate?, problem?
#   where == local                      -> no_charge, 0
#   sagemaker with billable seconds     -> the job by external_id: test job (six-arm record),
#                                          the training job (holds selection.training_run),
#                                          the scoring job (holds selection.selected.eval_run_id),
#                                          any other -> development; unknown job -> problem
#   sagemaker without billable seconds  -> its job's category, amount 0 ("priced through its job")
#   bedrock, phase train                -> current attempt's job ARN -> train, else development;
#                                          billed amount, else tokens x training price (estimate)
#   bedrock with tokens                 -> test if the run is in the six-arm record, else development
#   anything else                       -> problem (K4 fails)

def arm_figures(...)      # test pass, per 1,000 questions (as run / steady), per 1,000 correct
def serving(...)          # batch at each size; always-on at each utilization
def crossing(fixed_a, slope_a, fixed_b, slope_b)   # two straight lines: volume, or who is always cheaper
def break_evens(...)      # D13 always-on and batch (one job a month), D15 with the making cost; D14 marks
def storage_months(attempts, through)              # one model-month per calendar month a model exists
def reconcile(placed, bills, ...)                  # per entry: model total through its date, difference
def build_record(inputs, cfg) -> dict              # K1..K8, tables, `passed`, `provisional`
```

- **Money in the record** is text: six decimal places in JSON, shown in Markdown with two
  decimals at $1 or more and three below.
- **Break-even arithmetic**: a month's cost is `fixed + volume x slope`. Always-on Qwen:
  fixed = hosting price x hours, slope 0. Batch Qwen: fixed = one job's start-up, slope =
  steady cost per question. Nova: fixed = storage (tuned) or 0, slope = test cost per
  question. D15: fixed = making cost (train plus select for A1), slopes as steady; no
  monthly amounts. Each result says who is cheaper below and above, or who is cheaper at
  every volume.
- **Reachable** (D14): the volume is at most the instance's measured capacity a month and
  at most the Nova arm's requests a minute x 60 x hours a month. D15 is a lifetime volume:
  it is marked reachable, with the months it takes at the smaller limit.
- **K8**: an entry passes within `max($0.50, 2%)`. A service with no entry that covers its
  last priced run date passes **with a reservation**; the record is then provisional.
- **`bill`** appends one entry to `reports/comparison/aws_bill.json` (canonical JSON,
  earlier entries untouched). It refuses an unknown service, a negative or malformed
  amount, and a date in the future.

### Error handling

- `ConfigError` for a bad config, a missing record file, a refused `bill`: the script
  prints `CONFIG ERROR: ...` and exits 2.
- A failed check: every check is printed, nothing is written, exit 1.
- A reservation: the record is written with **provisional** in its title; exit 0, with a
  line that says what is still missing.

### Tasks, in order

1. `config/cost_model.yaml`, the hosting price in `pricing.yaml`, `cost_config.py`, tests.
2. `cost.py`: `place`, the index of committed records, pricing; tests for each category.
3. Per-arm figures, storage, the project totals; tests.
4. Serving scenarios and break-evens with reachability; tests.
5. Reconciliation and the bill file; tests.
6. K1 to K8, `build_record`, `render_markdown`, `write_record`; tests.
7. `scripts/comparison_cost.py`; script tests; guard tests (no boto3, no `load_split`).
8. Run `check` and `report` on the real repository; compare with the hand figures.
9. Documents: runbook 7.10, run sheet, `ledger/README.md`, a project pattern file, coding
   notes where the rules changed.
10. Lint, full tests, with and without the training stack; spec section 11, roadmap,
    execution plan, implementation summary.

## Validation gates

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q
venv\Scripts\python -m pytest -q tests/comparison tests/scripts/test_comparison_cost.py
venv\Scripts\python -m scripts.comparison_cost check
venv\Scripts\python -m scripts.comparison_cost report
```

## Success criteria

- [x] `pytest -q` and `ruff check .` pass, with and without the training stack blocked.
- [x] `comparison_cost check` passes K1 to K8 on the real repository, with a reservation
      only for the bill.
- [x] Per-path totals equal `committed_path_a` and `committed_path_b`.
- [x] The test-pass figures equal the hand figures above to the cent.
- [x] Every break-even carries a reachable mark.
- [x] No AWS call; no new module imports boto3; `cost.py` does not ask for a split.
- [x] Runbook, run sheet, ledger readme, pattern file, spec section 11, roadmap, execution
      plan updated.
- [x] For the owner: the two `comparison_cost bill` entries, then a record that is not
      provisional.

## Implementation Time Estimate

**4.5–5.5 hours**

| Task | Time |
|---|---|
| Config, hosting price, loaders, tests | 30 min |
| Categories and pricing a record, tests | 50 min |
| Per-arm figures, storage, totals | 35 min |
| Scenarios and break-evens | 50 min |
| Reconciliation and the bill file | 30 min |
| Checks, the record, Markdown | 45 min |
| Script and its tests, guards | 30 min |
| Real-repository run and comparison | 15 min |
| Documents | 30 min |
| Validation and status documents | 25 min |

## Confidence: 8/10

The work is arithmetic on records whose shapes are verified, with hand figures to check
against, and it touches no existing behaviour. The risks: a ledger record that fits no
category on the real repository (K4 will show it), and the break-even cases, where the
"cheaper at every volume" branches are easy to get backwards.

## Time Taken for PRP Generation

**~15 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 6 min |
| Research existing PRPs and patterns, the public price list | 3 min |
| Writing PRP document | 4 min |

The spec was written in the same session an hour earlier, so the records' shapes and the
ledger's content were already known.
