# Pricing a ledger record

**Defined by**: req_005 v5.02 (the cost model)  
**Code**: `qlora_finetuning/comparison/cost.py` (`place`, `place_all`, `path_totals`)  

The ledger holds quantities, never prices. Any code that needs a dollar figure asks the cost
model for it, or takes it from `reports/comparison/cost_model.json`. It does not multiply
tokens or seconds by a price of its own.

## The rule

A dollar figure is one of three things, and says which:

| Basis | What it is |
|---|---|
| `billed` | The amount of a ledger record with `source: bill`, as the owner read it from the AWS bill. Used as it stands |
| `list price` | A quantity AWS reported (`aws_api`) times a price from `config/pricing.yaml` |
| `estimate` | A quantity we computed ourselves (`computed`) times a price. Marked as an estimate wherever it is shown |

## How a record is priced

| The record | Its price |
|---|---|
| `where: local` | Zero. A run on the owner's machine has no AWS charge |
| `where: sagemaker`, with `billable_seconds` | Seconds times instance count times the hourly training price, over 3,600 |
| `where: sagemaker`, without `billable_seconds` | Zero. The run was made inside a job, and the job's record is priced |
| `where: bedrock`, `phase: train` | The billed amount; without one, training tokens times the training price |
| `where: bedrock`, with tokens | `frontier.pricing.inference_cost`: input and output tokens times their prices |

## Which run is which

An arm name does not identify a run: dozens of runs carry arm A1. Take the identity from the
committed records:

| Question | Record |
|---|---|
| Which six runs are the test results? | `reports/comparison/six_arm_test_runs.json` |
| Which job trained A1, and which chose its checkpoint? | `reports/local_qlora/adapter_selection.json` (`training_run`, `selected.eval_run_id`), looked up in `reports/local_qlora/sagemaker_jobs.json` |
| Which customization job made a tuned Nova model? | The latest attempt per model in `reports/frontier/nova_tuning.json` |

## Reuse checklist

- [ ] Read `Ledger(cfg).effective()`, never `read()`: a replaced record must not be counted.
- [ ] Money is `Decimal`. `wall_clock_seconds` is a float in the ledger: `Decimal(str(value))`.
- [ ] A run's day is the date in its `run_id`, not `recorded_at`: a bill record is written
      days after its job.
- [ ] A new kind of ledger record needs a branch in `place`, and a test. A record that fits
      no category fails check K4; it is never skipped.
- [ ] A new price goes into `config/pricing.yaml` as quoted text with its price list's date.
- [ ] Guard against a division by zero: an arm may have no correct answer.
