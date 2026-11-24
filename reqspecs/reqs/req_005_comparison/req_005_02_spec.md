# Comparison

Spec Group      :   req_005_comparison  
Feature Name    :   comparison  
Version         :   2  
Status          :   ✅ Complete 2026-11-24: code built 2026-11-23 (`prp_005_cost_model_v2.md`, section 11); the owner entered the Bedrock and SageMaker totals from the bill on 2026-11-24; K1 to K8 pass with no reservation (commit `3dbf2b2`)  
Last updated    :   2026-11-24  

## 1. v5.02 — Cost model

The six arms have answered the test split (v5.01, milestone M7). This version says what each
of them cost, in dollars, and what each would cost to use.

Both paths are billed by AWS, but in different units. Qwen on SageMaker is billed by the
second an instance runs, whether it is answering or idle. Nova on Bedrock is billed by the
token, and a tuned Nova model also costs a fixed amount a month for as long as it exists.
The cost model puts both on one scale.

It answers four questions:

1. **What did it cost to make each tuned model?** One training run per tuned arm.
2. **What does a question cost?** Cost per 1,000 questions for each of the six arms, from
   the six test runs.
3. **When is which path cheaper?** Break-even volumes between Path A and each Nova model,
   under stated ways of serving Qwen.
4. **Does it agree with the bill?** The total the model computes for the project, beside
   the total AWS charged.

It writes one committed record, `reports/comparison/cost_model.json` and `.md`. It calls no
AWS service and starts nothing billed. Every figure is a quantity from the ledger times a
price from `config/pricing.yaml`.

**Depends on**: v5.01 (complete 2026-11-23) and v0.04 (the ledger).

This version computes no statistics and draws no chart. The confidence intervals, the tests
between arms, and the accuracy-versus-cost chart are v5.03.

## 2. What exists today (verified 2026-11-23)

### 2.1 The quantities

The ledger holds quantities, never prices. The effective records (a replaced record is not
counted) that this version prices:

| What | Arm | Quantity | Source |
|---|---|---|---|
| Training job `qlora-train-958bed10-a1` | A1 | 15,037 billable seconds, `ml.g5.xlarge` | `aws_api` |
| Scoring job `qlora-score-63b89247-a1` (ten checkpoints and A0 on dev) | A1 | 4,965 billable seconds | `aws_api` |
| Test job `qlora-evaluate-1a60194b-a1` | A0 | 4,164 billable seconds; the run inside it took 3,833 | `aws_api`, `measured` |
| Test job `qlora-evaluate-d604229c-a1` | A1 | 2,564 billable seconds; the run inside it took 2,240 | `aws_api`, `measured` |
| Two smoke jobs | none | 281 and 792 billable seconds | `aws_api` |
| Nova Lite customization job | B1-Lite | **$14.60, read from the bill** (the estimate was 6,411,388 tokens, $12.82) | `bill` |
| Nova Pro customization job | B1-Pro | **$58.75, read from the bill** and entered 2026-11-23, for 7,343,826 tokens (the estimate was 6,411,388 tokens, $51.29) | `bill` |
| Four Nova test runs | B0-Lite, B1-Lite, B0-Pro, B1-Pro | 1,773,547 input tokens each; 126,528, 87,972, 121,823, and 90,516 output tokens | `aws_api` |
| Every other Bedrock run (dev, quick-dev, trials, two failed runs) | the four B arms | input and output tokens per run | `aws_api` |
| Runs on the owner's machine | A0, A1 | wall-clock seconds; no AWS charge | `measured` |

The six test runs are the ones the six-arm record names. Dozens of other runs carry the
same arm names.

### 2.2 The prices

`config/pricing.yaml`, version 1:

| Item | Price | Price list |
|---|---|---|
| Nova Pro | $0.80 input, $3.20 output, $8.00 training, per million tokens; $1.95 a month per custom model | `AmazonBedrock`, published 2026-11-11 |
| Nova Lite | $0.06 input, $0.24 output, $2.00 training, per million tokens; $1.95 a month per custom model | the same |
| `ml.g5.xlarge`, SageMaker training | $1.408 an hour | `AmazonSageMaker`, published 2026-11-14 |

A tuned Nova model is billed at its base model's token prices. **There is no price for a
SageMaker real-time endpoint in the file.** An always-on scenario needs one (decision 3).

### 2.3 What already computes dollars

| Code | What it does | Used by |
|---|---|---|
| `frontier/pricing.py`: `load_pricing`, `inference_cost` | Prices as `Decimal`; tokens times price | Nova reports |
| `frontier/tuning.py`: `committed_path_b` | What Path B has spent, for the cap | The freeze, the tuning checks |
| `local_qlora/estimate.py`: `committed_path_a` | What Path A has spent, for the cap | The freeze, launch check J7 |
| `frontier/tuned_report.py`: `training_cost` | A customization job's cost: the billed amount, or the estimate | The tuned report |
| `scripts/ledger_summary.py` | Totals of quantities; no dollars | The owner |

No code yet states a cost per arm, a cost per 1,000 questions, or a break-even.

### 2.4 What the figures look like, by hand

Worked from the quantities and prices above, to show the size of things. The model's
output replaces these; none is a result.

| Arm | Test pass, 2,898 questions | Per 1,000 questions |
|---|---:|---:|
| A0 | $1.63 | $0.56 |
| A1 | $1.00 | $0.35 |
| B0-Lite | $0.14 | $0.047 |
| B1-Lite | $0.13 | $0.044 |
| B0-Pro | $1.81 | $0.62 |
| B1-Pro | $1.71 | $0.59 |

| Tuned arm | Training run |
|---|---:|
| A1 | $5.88 |
| B1-Lite | $14.60 (billed) |
| B1-Pro | $58.75 (billed) |

What this already shows:

- **A Qwen job pays for time that answers nothing.** Each test job had about 325 seconds of
  start-up and loading, $0.13. Without it A1 is about $0.30 per 1,000 questions.
- **Against Nova Pro, A1 is cheaper to train and cheaper per question** when the instance
  is kept busy. An idle instance reverses that: one always-on instance costs about $1,030 a
  month at the training rate, which Nova Pro's token price only reaches at about 1.7 million
  questions a month.
- **Against Nova Lite, A1 is cheaper to train and about seven times dearer per question.**
  Nova Lite overtakes it after a few tens of thousands of questions.
- **The account's limits bound the volumes.** Tuned Nova Pro is allowed 20 requests a
  minute here, under 900,000 questions a month. One `ml.g5.xlarge` answered about 4,650
  questions an hour, about 3.4 million a month.

## 3. Locked decisions

### What is priced

| # | Decision | Reason |
|---|---|---|
| D1 | Every dollar figure is a ledger quantity times a price from `config/pricing.yaml`, computed as `Decimal`, rounded only when shown. The one exception is a record with `source: bill`, whose amount is used as it stands | The ledger's rule. A billed amount is better than a product of an estimate |
| D2 | The six test runs are taken from `reports/comparison/six_arm_test_runs.json` and nowhere else. The model refuses to run if that record is missing or did not pass | v5.01 D27. An arm name does not identify a run |
| D3 | Every effective ledger record is put in exactly one category: **train**, **select**, **test**, **development**, or **no charge**. The categories are listed in section 4.2. A record that fits none stops the model | Nothing unpriced, nothing counted twice. The total can then be compared with the bill |
| D4 | A run on the owner's machine has no AWS charge. It is listed with its seconds and priced at zero. No electricity or hardware cost is invented for it | The project compares two AWS bills |
| D5 | A figure built on a `computed` quantity is marked **estimate** wherever it is shown, and the record says how many figures are estimates | When this spec was written that was Nova Pro's training cost; with its bill entered, no priced figure is an estimate today. The rule stays for any later record |
| D6 | Prices are those of the file's price lists, with their dates shown beside every table. The model does not fetch prices | A price change after the runs must not silently change a result |

### What a question costs

| # | Decision | Reason |
|---|---|---|
| D7 | For a Nova arm, cost per 1,000 questions is the test run's input and output tokens times their prices, divided by 2,898, times 1,000 | Billed per token; no fixed part per question |
| D8 | For a Qwen arm, two figures are given: **as run** (the job's billable seconds, start-up included) and **steady** (the run's own wall-clock seconds inside the job). Both use the test job, on `ml.g5.xlarge`, at batch size 8 | The first is what was paid. The second is what a longer job approaches |
| D9 | Cost per 1,000 **correct** answers is also given, per arm: cost per 1,000 questions divided by the arm's execution accuracy in the six-arm record | It is the figure v5.03's accuracy-versus-cost chart needs |
| D10 | A tuned Nova model's storage is a cost per month, shown on its own line and added to break-even figures as a fixed monthly amount. It is never spread over the test questions | It is paid whether or not a question is asked |

### Serving Qwen, and the break-even

| # | Decision | Reason |
|---|---|---|
| D11 | Path A's serving cost is stated for two scenarios (question Q10). **Batch job**: a job is started for a batch of questions and stops when done; its cost is start-up plus the steady time for the batch. **Always-on endpoint**: one instance runs every hour of the month; its cost does not depend on how many questions arrive | These are the two ways the adapter would be used. The batch job is the measured case |
| D12 | The always-on scenario is given at the utilizations in `config/cost_model.yaml`: 100%, 50%, 10%, and 1% of the instance's measured capacity, over 730 hours a month | One figure would hide that the answer depends on load |
| D13 | A break-even is a number of questions a month at which two arms cost the same. It is given for A1 against B1-Lite and against B1-Pro, in each scenario, and for the untuned arms in the same way. When one arm is cheaper at every volume the model says so and gives no number | Some pairs never cross |
| D14 | Each break-even is checked against what either side can serve: the instance's measured capacity, and the account's requests a minute for the Nova model (`config/frontier.yaml` and the quota table in `reqspecs/references/tech/bedrock_nova.md`). A break-even above either limit is shown as **not reachable here** | A volume the account cannot send is not a choice |
| D15 | A second break-even counts the training run too: the number of questions after which the dearer-to-train arm has made up its training cost. It is given for A1 against each tuned Nova model in the batch scenario | "Cheaper to train, dearer to run" needs a volume to mean anything |

### The record

| # | Decision | Reason |
|---|---|---|
| D16 | The record is written by `comparison_cost report`, only when checks K1 to K8 pass. `K2` and `K8` may pass with a stated reservation (an estimate, a bill not entered); the record then carries the word **provisional** in its title | The model is useful before the last charge shows on the bill, and must not look final |
| D17 | The record holds, per arm: the training run, the selection, the test pass, the cost per 1,000 questions and per 1,000 correct answers; per path: the development total and the project total; the scenarios and break-evens; the reconciliation; the price list dates; the list of estimates. No example text, no name, no address | It is committed |
| D18 | v5.03 takes every dollar figure from this record's JSON file and computes none itself | One place for costs, as the six-arm record is for runs |

## 4. Scope to build

New code: `qlora_finetuning/comparison/cost.py` (categories, per-arm costs, scenarios,
break-evens, checks K1 to K8, the record) and `qlora_finetuning/comparison/cost_config.py`
(the loader of `config/cost_model.yaml`). One script, `scripts/comparison_cost.py`. The
existing `load_pricing`, `inference_cost`, `committed_path_a`, and `committed_path_b` are
used, not copied.

### 4.1 The commands

```powershell
venv\Scripts\python -m scripts.comparison_cost check     # K1 to K8 and the figures; writes nothing
venv\Scripts\python -m scripts.comparison_cost report    # writes the record
venv\Scripts\python -m scripts.comparison_cost bill --service bedrock --usd 70.61 --through 2026-11-23   # THE OWNER
```

`report` writes `reports/comparison/cost_model.json` and `.md`. `bill` appends one line to
`reports/comparison/aws_bill.json`: a service, an amount, and the last day it covers, as
the owner read them from the AWS bill. None of the three calls AWS or needs a sign-in.

### 4.2 The categories

| Category | Records in it | Priced as |
|---|---|---|
| **train** | The training job of A1; the customization job of each tuned Nova model | Billable seconds times the instance rate; the billed amount, or training tokens times the training price |
| **select** | The scoring job that chose A1's checkpoint | Billable seconds times the instance rate |
| **test** | The six runs of the six-arm record; for a Qwen arm, its test job | D7 and D8 |
| **development** | Every other billed record: smoke jobs, baselines on dev and quick-dev, tuned runs on dev, five-question trials, failed runs | The same prices |
| **no charge** | Runs on the owner's machine; the `gold` predictor's runs | Zero |

Which run is which is decided from the committed records, never from an arm name alone:
the job record (`reports/local_qlora/sagemaker_jobs.json`), the selection record, the
tuning record, and the six-arm record. `config/cost_model.yaml` names those files and the
scenarios; it holds no run id.

A run's own usage record and its job's record describe the same seconds. For a run made
inside a SageMaker job, the job's billable seconds are what is priced, and the run's
wall-clock seconds are used only for the steady figure of D8.

### 4.3 The checks

| Id | Check | Passes when |
|---|---|---|
| K1 | Six test runs | The six-arm record exists, passed C1 to C8, and names one run per arm |
| K2 | Quantities found | Every run and job the model prices has its effective ledger record, and the tokens and billable seconds there equal those in the six-arm record and the job record. A `computed` quantity passes with a reservation and is listed |
| K3 | Prices found | Every model id and instance type priced has a price in `config/pricing.yaml`, with a price list and its date |
| K4 | Every record placed | Each effective ledger record is in exactly one category |
| K5 | Totals agree | The sum over categories per path equals what `committed_path_a` and `committed_path_b` return for the same ledger |
| K6 | Same questions | The six test runs have 2,898 examples each; the four Nova runs have equal input tokens, and so have the two Qwen runs |
| K7 | Reachable | Each break-even is marked reachable or not, from the measured capacity and the account's limits; none is shown without the mark |
| K8 | Agrees with the bill | For each service with an entered bill amount, the model's total for the same days is within the tolerance in `config/cost_model.yaml` (proposed: the larger of $0.50 and 2%). A service with no entered amount passes with a reservation |

### 4.4 The record's tables

1. **Making each tuned model**: training run, selection, and for a Nova model the storage
   per month.
2. **The test pass**: per arm, the quantity, the cost, the cost per 1,000 questions (as run
   and steady for Qwen), and per 1,000 correct answers.
3. **Serving Qwen**: the batch job at several batch sizes (100, 1,000, 2,898, 10,000, and
   100,000 questions), and the always-on endpoint at the four utilizations.
4. **Break-evens**: D13, D14, and D15.
5. **The project so far**: per path and category, with the caps ($100 and $200).
6. **Against the bill**: per service, the model's total, the billed amount, the difference.
7. **What is an estimate**, and the price lists with their dates.

### 4.5 Tests

No network, no credentials, no GPU, invented ledger lines and prices only.

- Categories: each kind of record lands in its category; an unknown record stops the model;
  a replaced record is not counted; a local run is priced at zero.
- Costs: a Nova run's cost equals `inference_cost`; a job's cost is seconds times the hourly
  rate over 3,600; a billed amount wins over an estimate; `Decimal` throughout, no float.
- Per 1,000: as run and steady differ by the start-up; per 1,000 correct answers uses the
  six-arm record's count; an arm with no correct answer gives no figure, not a division by
  zero.
- Scenarios: the always-on cost per 1,000 doubles when utilization halves; the batch cost
  per 1,000 falls toward the steady figure as the batch grows.
- Break-evens: a pair that crosses gives the volume at which both costs are equal, checked
  by pricing both sides at that volume; a pair that never crosses gives none; a volume above
  a limit is marked not reachable.
- Checks: each of K1 to K8 passes and fails; a reservation gives a provisional record.
- Record: the same state gives the same bytes; Markdown and JSON agree; no example text,
  name, or address in either; `bill` appends and never rewrites.
- Guards: no new module imports boto3; `cost.py` does not ask for the test split.

### 4.6 Documentation

- Runbook: a section 7.10, with how to read the bill in the console and enter it.
- `administration/aws_commands_run.md`: no AWS command is run by this version; the owner's
  reading of the bill in the console is noted as a step.
- `ledger/README.md`: the five categories, and that dollars are now computed by
  `comparison_cost`.
- `reqspecs/references/project_patterns/`: a short pattern file for pricing a ledger record.

## 5. Out of scope

- Confidence intervals on any figure, tests between arms, the accuracy-versus-cost chart,
  and the report's prose (v5.03).
- Any new run, on AWS or here. No endpoint is created to measure one: the always-on
  scenario is computed from the test job's throughput.
- Other instance types, quantization settings, or batch sizes for Qwen. One measured
  configuration is priced.
- Provisioned Throughput, batch inference on Bedrock, and prompt caching: none was used.
- The cost of the owner's time, of the owner's machine, and of electricity.
- Deleting the tuned models or the deployments (the runbook's teardown). The record shows
  what keeping them costs.
- Changing `config/pricing.yaml`'s existing prices. One price is added (decision 3).

## 6. What this hands to later versions

| For | What |
|---|---|
| v5.03 | `reports/comparison/cost_model.json`: per arm, the cost per 1,000 questions and per 1,000 correct answers for the chart; the cost table; the break-evens and their reachability; the list of estimates and the price list dates for the method section |
| v6.02 | The training cost and the cost per 1,000 questions of A1, for the model card |
| The teardown | What each kept resource costs a month |

## 7. Known limits and follow-ups

- **List prices, not the bill, per run.** AWS bills by service and day, not by run, so a
  single run's cost is always quantity times list price. Only the totals can be compared
  with the bill (K8).
- **The always-on figure is computed, not measured.** It assumes an endpoint answers as
  fast as the test job did at batch size 8. An endpoint that answers one request at a time
  is slower, so the real cost per question is higher. The record says so beside the table.
- **One job's start-up is a sample of one** per arm (331 and 324 seconds). It varies with
  the time to get an instance and to download the image.
- **Both training charges are billed amounts.** Each was above its estimate: Nova Lite by
  14% and Nova Pro by 15%, because Bedrock counted more training tokens than the Qwen
  tokenizer did.
- **Token prices and limits are this account's and this month's.** The limits are a tenth
  of the documented defaults and are not adjustable here.
- **Storage in S3, logs in CloudWatch, and data transfer are not in the model** (decision
  4). They appear in the bill, and so in the reconciliation's difference.
- **The questions are FinQA test questions.** Cost per question scales with prompt length:
  the test prompts average about 612 Nova input tokens.

## 8. Decisions for the owner

**All seven taken as recommended by the owner on 2026-11-23.** In the owner's words: 1, agree on the split between development and training; 2, ok; 3, go ahead with reading the public price list; 4, no; 5, done (the bill is entered); 6, yes; 7, ok. The table is kept as the record of what was asked.

| # | Question | Recommendation |
|---|---|---|
| 1 | **What counts as the cost of making a tuned model?** | The one run that made it (train), with the selection of A1's checkpoint shown on its own line beside it. Everything else spent on the way (smoke jobs, dev runs, trials, failed runs) is shown per path as development, not charged to a model. A reader who wants to repeat the work pays the first two; the third is what finding the way cost |
| 2 | **How is Path A's serving cost stated (Q10)?** | D11 and D12: a batch job, and an always-on endpoint at 100%, 50%, 10%, and 1% utilization, over 730 hours a month, with the break-even against each Nova model |
| 3 | **Which hourly price for the always-on endpoint?** | The SageMaker real-time inference price of `ml.g5.xlarge` in us-east-1, read from the public price list and added to `config/pricing.yaml` as `sagemaker.hosting`, with the list's date. Reading the public list needs no sign-in and costs nothing. The alternative, reusing the training rate of $1.408, is simpler but prices a different product |
| 4 | **Are S3, CloudWatch, and data transfer included?** | No. They are left out of the model and named as left out. They are small beside the other figures, the ledger does not record them, and they show up in the reconciliation's difference |
| 5 | **Nova Pro's training charge is not in the ledger.** | **Done 2026-11-23**: the owner read $58.75 from the bill (7,343,826 tokens at $8.00 per million) and entered it with `nova_tune bill --model nova-pro --usd 58.75`; commit `a0a855a`. What was recommended: enter it before the record is final |
| 6 | **Reconcile with the bill?** | Yes, by service: the owner reads the Bedrock and SageMaker totals from the bill for the project's days and enters them with `comparison_cost bill`. Tolerance: the larger of $0.50 and 2%. Without the entries the record is provisional |
| 7 | **How long are the tuned Nova models kept, for the storage line?** | Show the cost per month, and the total at 1 and 12 months. No end date is assumed in a break-even: storage enters as a fixed amount a month |

## 9. Acceptance

Code, by Claude:

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
  and without the training stack blocked, with no network and no credentials.
- `comparison_cost check` passes K1 to K8 on the real repository, with reservations only
  for what the owner has not yet entered, and its totals per path equal those of
  `committed_path_a` and `committed_path_b`.
- `comparison_cost report` writes a record whose test-pass figures equal the hand figures
  of section 2.4 to the cent.
- No AWS service is called and nothing billed is started.

By the owner (🟡):

1. Read Nova Pro's training charge from the bill and enter it with `nova_tune bill`.
   **Done 2026-11-23: $58.75.**
2. Read the Bedrock and SageMaker totals from the bill and enter them with
   `comparison_cost bill`.
3. `comparison_cost report`: K1 to K8 pass with no reservation; the record is not
   provisional. Commit the record, the bill file, and the ledger.

- The record's figures are written into this spec as a section 11.

## 10. Next step

The code is built and the record is written, provisional (section 11). The owner, once the test runs
of 2026-11-23 show on the bill (runbook section 7.10): read the Bedrock and SageMaker totals, enter
them with `comparison_cost bill`, run `comparison_cost report`, and commit. That completes this
version.

After this version comes v5.03, the comparison report, which reaches milestone M8. It needs its spec
and does not have to wait for the bill: it reads `cost_model.json`, whose figures the bill entries
do not change.

## 11. As built (2026-11-23)

The code is built (`prp_005_cost_model_v2.md`) and the record is written:
`reports/comparison/cost_model.json` and `.md`. It is **provisional**: check K8 passes with
a reservation, because no bill amount is entered yet for the day of the test runs. Nothing
was sent to AWS. Tests: 2,763 passed in the full suite, 79 of them new (one guard test failed once, on files of the req_006 demo work then in progress in the same tree; it passes on a rerun); with the training stack blocked, 2,757 passed and 7 skipped, with the same one failure; lint clean.

### The figures

| Arm | Test pass | Per 1,000 questions | Steady | Per 1,000 correct answers |
|---|---:|---:|---:|---:|
| A0 | $1.63 | $0.562 | $0.517 | $60.32 |
| A1 | $1.00 | $0.346 | $0.302 | $0.473 |
| B0-Lite | $0.137 | $0.047 | | $0.283 |
| B1-Lite | $0.128 | $0.044 | | $0.062 |
| B0-Pro | $1.81 | $0.624 | | $1.76 |
| B1-Pro | $1.71 | $0.590 | | $0.784 |

They equal the hand figures of section 2.4 to the cent.

| Tuned arm | Training run | Selection | Storage a month |
|---|---:|---:|---:|
| A1 | $5.88 | $1.94 | |
| B1-Lite | $14.60 (billed) | | $1.95 |
| B1-Pro | $58.75 (billed) | | $1.95 |

The project so far: Path A $10.87 of $100, Path B $78.08 of $200 (with storage, $81.98).
Both equal what the cap checks count (K5). No priced figure is an estimate.

Serving A1: 0.773 seconds a question, about 4,658 an hour. An always-on `ml.g5.xlarge`
costs $1,027.84 a month, so $0.302 per 1,000 questions at full use and $30.23 at 1%.

### The break-evens

| Pair | Always-on endpoint | Batch job, a month | With the cost of making |
|---|---|---|---|
| A1 and B1-Pro | 1,740,152 questions a month, **not reachable**: tuned Nova Pro is allowed 876,000 | A1 is cheaper at every volume | A1 is cheaper at every volume |
| A1 and B1-Lite | 23,313,104, **not reachable**: one instance serves 3,400,174 | Below 7,059 questions A1 is cheaper, above it B1-Lite | Below 26,239 questions A1 is cheaper, above it B1-Lite |
| A0 and B0-Pro | 1,646,889, **not reachable**: Nova Pro is allowed 1,095,000 | Above 1,212 questions A0 is cheaper | |
| A0 and B0-Lite | 21,777,236, **not reachable** | B0-Lite is cheaper at every volume | |

One result the hand figures did not show: against tuned Nova Lite, a batch job of A1 is the
cheaper below about 7,000 questions a month, because Nova Lite's model costs $1.95 a month
to keep whether or not it is asked anything.

No always-on break-even can be reached in this account: each needs more questions a month
than the account may send to the Nova model, or than one instance can answer.

### Against the bill

The owner read the bill on 2026-11-23: Bedrock $78.19 and SageMaker $8.24. Both equal the
ledger through 2026-11-22 to the cent, storage included. The test runs of 2026-11-23 were
not on the bill yet; once they are, the model expects Bedrock $81.98 and SageMaker $10.87.
These two amounts have not been entered with `comparison_cost bill`: that is the owner's
step, and the record stays provisional until an entry covers 2026-11-23.

### Differences from this spec

- **Five modules, not two.** `cost.py` (categories, per-arm figures, the checks),
  `cost_scenarios.py` (serving and break-evens), `cost_bill.py` (the bill file),
  `cost_report.py` (the Markdown and the files), `cost_config.py`.
- **The account's limits are in `config/cost_model.yaml`**, with the day they were read.
  D14 named `config/frontier.yaml` and a reference file; the first holds only the pacing of
  Nova Pro and the second is prose.
- **Storage in the reconciliation** is one model-month for each calendar month a tuned
  model exists, which is how the November bill charges it ("1 Model/month" for each model,
  made on the 19th and the 22nd).
- **A run's day is the date in its run id**, not the day its record was written: the two
  bill records were written days after their jobs.
- **K8 knows whether an entry covers the last run.** An amount entered through an earlier
  day is compared with the model for those days, and the record stays provisional.
- **The tolerance is 2% of the billed amount** (or $0.50, the larger).
- **The batch break-even assumes one job a month** (`batch_jobs_per_month`).
- **D15 counts A1's selection with its training run**: $7.82 to make A1.
- **The hosting price equals the training price**: $1.408 an hour for `ml.g5.xlarge`, from
  the public price list of 2026-11-23. The always-on figures are the same as they would
  have been at the training rate.

### Not verified

- The reconciliation for 2026-11-23 itself: it needs the bill to show the test runs.
- The always-on figures are computed from the test job's throughput; no endpoint was made.

### Closed (2026-11-24)

The bill caught up a day after the test runs, as expected. The owner read it and entered
both totals with `comparison_cost bill`, through 2026-11-23:

| Service | Billed | The model | Difference |
|---|---:|---:|---:|
| Bedrock | $81.98 | $81.98 | under half a cent |
| SageMaker | $10.87 | $10.87 | under half a cent |

`comparison_cost report` then passed K1 to K8 with no reservation, and the record is no
longer provisional. No figure outside the reconciliation table changed. The reconciliation
for 2026-11-23, listed above as not verified, is now verified. Committed as `3dbf2b2`.
This version is complete.
