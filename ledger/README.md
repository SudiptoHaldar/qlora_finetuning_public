# Ledger

The project's record of what every run consumed. This folder is **committed**, unlike
`artifacts/`, because the cost figures in the final report are computed from it.

| File | Holds | Written by |
|---|---|---|
| `runs.jsonl` | One line when a run starts and one when it ends: id, kind, arm, where it ran, scope, git commit, final status | `qlora_finetuning.tracking.start_run` and `Run.finish` |
| `ledger.jsonl` | One line per usage record: seconds, instance type and count, token counts, number of questions, and where the numbers came from | `Run.record` or `Ledger.append` |
| `evaluations.jsonl` | One line per evaluation run that is a result: run id, arm, split, number of examples, the count right on each of the four metrics, the count under the tight rule, tokens, the split manifest hash, and the prompt id. Counts only: no question, reply, or answer text | `run_evaluation` (req_002 v2.03) |

Each line is one JSON object. The first lines, of 2026-11-16, are two smoke runs of the evaluation
runner with the fake `gold` predictor (kind `smoke`, arm `none`): they carry no tokens and no cost.

The full detail of a run (manifest, a copy of each config file it used, predictions,
checkpoints) is in `artifacts/runs/<run_id>/`, which is not committed. A results line is the only
copy of a result that survives the loss of that folder.

## Rules

- **Append only.** Never edit or delete a line. To correct a record, append a new one whose
  `replaces` field holds the old record's `record_id`; readers then skip the old one.
- **Quantities, not prices.** The ledger records seconds and tokens. Dollars appear only in
  `billed_amount_usd`, and only for a figure read from the AWS bill (`source: bill`). Rates
  live in `config/` and are applied by the cost model.
- **Every record names its source**: `measured` (our own clock or counter), `aws_api` (a field
  AWS returned), `computed` (derived, for example training tokens from the file size and
  epochs), or `bill` (read from the AWS bill).
- **No secrets.** No credential, token, or email address. Job names, ARNs, and the account id
  are fine.
- **Commit after a measured run**, so the figures it produced are in the history.

## Reading it

```powershell
venv\Scripts\python -m scripts.ledger_summary
venv\Scripts\python -m scripts.ledger_summary --by arm,phase,where
venv\Scripts\python -m scripts.ledger_summary --json
```

The summary shows the runs, the totals per group, and how many records are still estimates.
It shows quantities only.

## Dollars

```powershell
venv\Scripts\python -m scripts.comparison_cost check
venv\Scripts\python -m scripts.comparison_cost report
```

The cost model (req_005 v5.02) prices every effective record with `config/pricing.yaml` and
puts it in exactly one category:

| Category | Records |
|---|---|
| `train` | The training job of A1; the customization job of each tuned Nova model |
| `select` | The scoring job that chose A1's checkpoint |
| `test` | The six runs of the six-arm record; for a Qwen arm, its test job |
| `development` | Every other billed record: smoke jobs, runs on dev and quick-dev, trials, failed runs |
| `no charge` | Runs on the owner's machine |

A run made inside a SageMaker job has its own record without billable seconds. It is priced
at zero, because its job's record is priced. A record that fits no category fails the cost
model's check K4. The pattern is in
`reqspecs/references/project_patterns/pricing_a_ledger_record.md`.

Operator steps, including how to add a billed amount, are in `docs/runbook.md`.
