# Frontier Tuning

Spec Group      :   req_004_frontier_tuning  
Feature Name    :   frontier_tuning  
Version         :   4  
Status          :   ✅ Complete: code built 2026-11-18 (`prp_004_tuned_dev_evaluation_v4.md`); on 2026-11-22 the owner ran the four tuned runs and the report was written: both models met their threshold (section 11)  
Last updated    :   2026-11-22  

## 1. v4.04 — Tuned-model dev evaluation

This version asks the two fine-tuned Nova models the dev questions and says whether tuning
helped. It scores arms **B1-Lite** and **B1-Pro** with the runner that scored their untuned
baselines (B0), on the same questions, with the same prompt and settings, and compares each
tuned model with its own baseline.

It adds:

- two **predictors**, `nova-lite-tuned` and `nova-pro-tuned`, that take the deployment to
  call from the tuning record of v4.03;
- a **tuned-model report**: per model and split, B0 beside B1 on every metric, the paired
  comparison question by question, the cost per example, and a **verdict** per model
  against the thresholds of section 3;
- that verdict is what decides whether v4.05 (one corrective iteration) fires.

The evaluation runs call Bedrock and are billed. **The owner runs them.** The report calls
no AWS service.

**Depends on**: v4.03, with a completed job and an Active deployment for the model being
scored. Neither exists today: no job has been launched. The code of this version can be
built and tested before that.

Reaching the end of this version, with both models scored, is milestone **M6**.

## 2. What exists today (verified 2026-11-18)

### 2.1 What earlier versions provide

| Item | State |
|---|---|
| Evaluation runner (v2.03) | `run_evaluation`, resumable, checks R1 to R5, one summary per run in `ledger/evaluations.jsonl` |
| `FrontierPredictor` (v4.01) | One Converse call per question; usage from AWS into the ledger. Takes the `modelId` to call as an argument |
| `scripts/evaluate.py` | Predictors by name: `gold`, `nova-pro`, `nova-lite`. A resume is refused when the predictor's name, model id, or settings differ |
| Tuning record (v4.03) | `read_tuning_record().latest(model)`: job status, model ARN, deployment ARN and status, hyperparameters, export id, attempt. **Today: no record; nothing launched** |
| `config/frontier.yaml` | Per model `tuned_arm`: `B1-Pro`, `B1-Lite`. Retry mode `adaptive`, 12 attempts (changed 2026-11-16 after a throttled run) |
| `config/pricing.yaml` | "A tuned model is billed at the same token prices as its base model" (the price list carries custom-model rows with the same amounts) |
| Baseline report code (v4.01) | `frontier/baseline_report.py`: checks, token and cost figures, Markdown and JSON twins. **The report itself has not been written yet** (`nova_baseline` not run) |

### 2.2 The baselines the tuned models are compared with

The four B0 runs were made by the owner on 2026-11-16. From `ledger/evaluations.jsonl`:

| Arm | Split | Run | Format validity | Execution accuracy | Answer accuracy |
|---|---|---|---:|---:|---:|
| B0-Lite | quick-dev | `20261116T201734Z-eval-nova-lite-quick-dev-700e` | 192 of 200, 96.00% | 47 of 200, 23.50% | 75 of 200, 37.50% |
| B0-Lite | dev | `20261116T202551Z-eval-nova-lite-dev-d37e` | 466 of 499, 93.39% | 123 of 499, 24.65% | 189 of 499, 37.88% |
| B0-Pro | quick-dev | `20261116T214132Z-eval-nova-pro-quick-dev-156c` | 199 of 200, 99.50% | 75 of 200, 37.50% | 103 of 200, 51.50% |
| B0-Pro | dev | `20261116T203032Z-eval-nova-pro-dev-d704` | 497 of 499, 99.60% | 190 of 499, 38.08% | 259 of 499, 51.90% |

All four used prompt `d63742a62b76...` and split manifest `d34fb22e37b7...`. A dev run sent
313,156 input tokens and received about 22,000 to 23,000 output tokens.

### 2.3 The account's limits (read from Service Quotas, 2026-11-16)

| Deployment of a tuned | Requests a minute | Tokens a minute |
|---|---:|---:|
| Nova Lite | 200 | 4,000,000 |
| Nova Pro | 20 | 800,000 |

Neither is adjustable. These are a tenth of the documented defaults in `bedrock_nova.md`.
At 20 a minute, Nova Pro's dev run takes about 25 minutes and its quick-dev run about 10.

### 2.4 What the runs will cost

At the baselines' token counts and the base models' prices:

| Arm | quick-dev | dev |
|---|---:|---:|
| B1-Lite | $0.01 | $0.02 |
| B1-Pro | $0.13 | $0.32 |

About $0.50 for all four. A tuned model should answer with a program and little else, so
the output tokens, and the cost, are more likely lower than higher. Storage of the tuned
models ($1.95 each a month) runs regardless and belongs to v4.03.

Not confirmed until the first bill: that tuned-model inference is charged at the base
model's prices on this account; whether the first call to an idle deployment is slow or
refused while it warms up.

## 3. Locked decisions

### What is run

| # | Decision | Reason |
|---|---|---|
| D1 | The tuned predictors are `FrontierPredictor` with a different `modelId`: the deployment ARN from the tuning record. Prompt, settings, parser, executor, and scoring are untouched | The comparison is only fair if nothing but the model changes |
| D2 | The model scored is the **latest attempt** of that model in the tuning record. It must show job `Completed` and deployment `Active`, and the deployment is asked for its status once (a read) before the first question. Anything else is a `ConfigError` that names the `nova_tune` command to run | A run must not start against a deployment that is gone or still being created |
| D3 | The tuned model's export id must be the export id of the current gate record, and the prompt id and split hash recorded at export must be today's | A model tuned on other files, or for another prompt, is not this experiment's B1 |
| D4 | A run's evaluation summary names the deployment ARN, the base model id, the model name, the attempt, and the job ARN. Usage is priced at the **base model's** prices | The bill is by base model; the report must say which tuned model answered |
| D5 | A run cannot be resumed against another deployment: the deployment ARN is part of what a resume compares | A second attempt is another model |
| D6 | Splits: **dev** is the result. quick-dev is run first as a check that costs cents (the owner's decision, 2026-11-18) | 200 questions show a broken deployment or a format collapse before the 25-minute run |
| D7 | The runs are the owner's. Each begins with a five-question trial (`--limit 5`), as v4.01 did | Billed calls (working agreement); the first call to a new deployment is the unverified one |
| D8 | The test split is not touched. `evaluate` refuses it as before; this version adds no way round | Locked until v5.01 |

### What is compared

| # | Decision | Reason |
|---|---|---|
| D9 | Each B1 run is compared with the B0 run of the same model and split, named on the command line. The report refuses the pair unless prompt id, split manifest hash, settings, example count, and example order are equal | "Same questions" is checked, not assumed |
| D10 | The **paired comparison** is per question: right in both, wrong in both, fixed by tuning (wrong in B0, right in B1), broken by tuning (right in B0, wrong in B1), on execution accuracy. The report gives the four counts and the exact two-sided sign test on fixed against broken | 499 questions answered by both models say more as pairs than as two percentages. The test needs only the standard library |
| D11 | Differences are in percentage points, from counts, with `Decimal` | As every figure in this project |
| D12 | Cost per example is input and output tokens at list price over the number of examples, for B0 and for B1, and also per correct answer. The training cost of the model is shown beside it, from the ledger (`bill` if entered, else the estimate, and labelled which), and is **not** spread over examples | Spreading a one-off cost over 499 questions gives a number that means nothing; the final report (v5.02) owns the cost comparison |
| D13 | The report holds counts, rates, ids, hashes, and example ids. No question, context, program, or reply text | Dataset text is never committed |

### The verdict

| # | Decision | Reason |
|---|---|---|
| D14 | A model **meets** its threshold on dev when both hold: its execution accuracy is at least **10.00 points** above its B0's, and its format validity is at least **98.00%**. Otherwise it has **missed** (the owner's decision, 2026-11-18). Both figures are in `config/frontier.yaml` | See section 8. On 499 questions that is at least 50 more correct than B0 (Lite 173, Pro 240) and at least 490 well-formed |
| D15 | The verdict is computed from dev only. quick-dev figures are shown and never decide | One split decides; the other is a preview |
| D16 | A missed threshold makes v4.05 **eligible** for that model. Nothing launches: v4.05 is a spec, a decision of the owner, and a paid job | Working agreement: nothing billed starts by itself |
| D17 | The report also states, without judging, whether tuning broke more than a tenth as many questions as it fixed, and the stop reasons and format failures of B1 | These are what v4.05 would read to choose its one change |

## 4. Scope to build

### 4.1 Predictors

`scripts/evaluate.py` gains two entries; `frontier/predictor.py` gains their factories.

| Predictor | Arm | Calls |
|---|---|---|
| `nova-lite-tuned` | B1-Lite | The deployment of the latest `nova-lite` attempt |
| `nova-pro-tuned` | B1-Pro | The deployment of the latest `nova-pro` attempt |

```powershell
venv\Scripts\python -m scripts.evaluate --predictor nova-lite-tuned --split quick_dev --limit 5
venv\Scripts\python -m scripts.evaluate --predictor nova-lite-tuned --split dev
```

Reading the deployment's status uses the Bedrock control client, through `BedrockTuning`
or a read beside it; `predictor.py` still imports no botocore.

### 4.2 The report

A new module `frontier/tuned_report.py` and a script `scripts/nova_tuned.py`.

```powershell
venv\Scripts\python -m scripts.nova_tuned --runs <B1 run ids> --baselines <B0 run ids>
```

It reads the runs' summaries and prediction files, the tuning record, the gate record, the
ledger, and the price file. It calls no AWS service and needs no sign-in. It writes
`reports/frontier/nova_tuned_report.md` and `.json` (committed) and prints the checks and
the verdicts.

Checks, all of which must pass for the report to be written as a result:

| Id | Check | Passes when |
|---|---|---|
| V1 | Complete | Every run named is complete and counts as a result (R1 to R5) |
| V2 | Paired | Each B1 run has exactly one B0 run of the same model and split, and D9 holds |
| V3 | The right model | Each B1 run's deployment ARN is the latest attempt's in the tuning record, with job `Completed` |
| V4 | The right files | D3 |
| V5 | Usage adds up | The ledger's records for each run are from AWS and cover every prediction |
| V6 | Dev present | Both models have a dev pair. With one model only, the report is written as partial and says so |

Exit codes: 0 the report was written and every model met its threshold; 1 a check failed,
or a model missed (the report is still written when the checks pass); 2 a config problem or
a run that cannot be read.

The report's sections: the verdict per model; B0 beside B1 per split and metric with the
difference; the paired table; cost per example and per correct answer; training cost;
latency and stop reasons; the tuned model's identity (names, ARNs, hyperparameters, final
losses, attempt); the checks.

### 4.3 Tests

No network, no credentials: the stub provider of v4.01 and the stub Bedrock of v4.03.

- Predictors: the deployment ARN is the `modelId` sent; the request is otherwise byte for
  byte the untuned one; each refusal of D2 and D3 (no record, job not completed, deployment
  not Active, recorded Active but Bedrock says otherwise, another export id); the arm is
  the tuned arm; usage is priced at the base model's prices; a resume against another
  deployment is refused.
- Report: each of V1 to V6 passes and fails; the paired counts on a constructed pair; the
  sign test against hand-computed values, including no discordant pairs; the threshold at
  its boundary (49 and 50 more correct; 489 and 490 well-formed); a missed verdict exits 1
  with the report written; one model only gives a partial report; Markdown and JSON agree;
  no example text in either.
- Guards: `test` is refused; `nova_tuned` makes no AWS call; only `bedrock.py`,
  `staging.py`, and `tuning.py` import botocore.

### 4.4 Documentation

- Runbook section 6: the tuned evaluations, the report, how to read the verdict, what the
  rate limit means for the Nova Pro runs.
- `administration/aws_commands_run.md`: the owner's commands, added to the run sheet as
  stage G, with where to watch (CloudWatch shows a deployment's calls under its own ARN).
- `bedrock_nova.md`: the account's limits for deployments; what the first tuned call showed.
- `gotchas.md`: the documented limits are ten times this account's.
- `aws_console_primer.md`: where a deployment's metrics appear.

## 5. Out of scope

- Any change to a job, a hyperparameter, or the training files (v4.05).
- Error categories for the failures (format, sign, scale, evidence). v4.05 does that if it
  fires; this version gives it the counts and the example ids.
- The test split and the six-arm comparison (v5.01); the cost comparison between paths
  and the reconciliation with the bill (v5.02).
- Comparing Nova with Qwen. Path A has no dev result yet.
- Deleting deployments or models (v4.03's `teardown`).

## 6. What this hands to later versions

| For | What |
|---|---|
| v4.05 | The verdict per model; fixed and broken example ids; format failures and stop reasons; the loss curves stay in v4.03's record |
| v5.01 | The two tuned predictors, unchanged, for the test runs; the deployments, still Active (the owner's decision, 2026-11-18) |
| v5.02 | Tokens and cost per example for B0 and B1 on dev; the training cost beside them |
| v6.02 | The dev figures for the model card |

## 7. Known limits and follow-ups

- **A dev result is not the project's answer.** Dev was the validation file of the jobs
  (v4.02), so Bedrock saw its loss during training. It selected nothing here, since there
  is one job per model, but the test split in v5.01 is the only clean measure.
- **The baselines were run under retry mode `standard`, the tuned runs under `adaptive`.**
  Retries do not change a reply, and the settings a resume compares are equal.
- **Greedy decoding is not a promise of identical replies.** Bedrock does not guarantee
  determinism; a rerun can differ in a few questions. One run per arm and split is the
  result.
- **Nova Pro's tuned runs are slow** (20 requests a minute), and a run that is throttled
  out prints its `--resume` command.
- **The threshold is a judgment.** Ten points is well outside what two runs of 499
  questions differ by chance, but it is not derived from anything.
- **If only one model has a deployment**, the report is partial and M6 is not reached.

## 8. Decisions for the owner

**Decided by the owner on 2026-11-18**, each as recommended: (1) a tuned model has missed when it is less than 10 points above its own baseline on dev execution accuracy, or its format validity is under 98% ("10 points is a good improvement benchmark"); (2) both quick-dev and dev are scored; (3) the deployments are kept. The table is kept as the record of what was asked.

| # | Question | Recommendation |
|---|---|---|
| 1 | **What dev result means a tuned model has missed, so that v4.05 may fire?** | Missed when the tuned model is less than **10 points** above its own baseline on dev execution accuracy, or its format validity is under **98%**. In counts: Nova Lite needs 173 of 499 (from 123), Nova Pro 240 of 499 (from 190), and each 490 well-formed replies. Relative to its own baseline, because the two models start 13 points apart and one absolute bar would be easy for one and hard for the other. 98% format validity, because a tuned model that cannot produce the format has not learned the task; untuned Nova Lite is at 93.39%. The alternative is an absolute bar per model, which needs a target the plan does not give |
| 2 | **Score quick-dev as well as dev?** The roadmap says dev | Yes, quick-dev first for each model. It costs about $0.14 in all and ten minutes for Nova Pro, and it shows a broken deployment or a collapsed format before the long run. It also gives v5.02 a second pair of figures |
| 3 | **Keep the deployments after this version?** | Keep them until the test runs of v5.01. The price list has no hourly charge for an on-demand deployment, and the tuned models cost $1.95 a month each whether deployed or not. If the first bill shows an idle deployment does cost something, delete them with `nova_tune teardown --deployment` and create them again before v5.01 |

## 9. Acceptance

Code, by Claude:

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
  and without the training stack blocked, with no network and no credentials.
- `venv\Scripts\python -m scripts.evaluate --predictor nova-lite-tuned --split quick_dev`
  refuses with a message that names the missing step, because nothing has been launched.
  **That is the expected state on the day the code is built.**
- No test and no command run by Claude asks a tuned model anything.

Live, by the owner (🟡), each command logged in `administration/aws_commands_run.md`:

1. v4.03 complete for the model: job `Completed`, deployment `Active`, record committed.
2. Trial: `--predictor nova-lite-tuned --split quick_dev --limit 5`.
3. `nova-lite-tuned` on quick-dev, then on dev. The same three for `nova-pro-tuned`.
4. `nova_tuned --runs ... --baselines ...`: V1 to V6 pass. Commit the report and ledger.

- The report shows B0 beside B1 for both models on dev, the paired counts, the cost per
  example, and a verdict per model.
- The billed charge for the tuned runs, when it appears, is compared with the report's
  cost; a difference is written into section 11.
- The roadmap records, per model, met or missed, and with it whether v4.05 is eligible.
- The figures are written into this spec as section 11. With both models scored, M6 is
  reached.

## 10. Next step

This version is complete: both tuned models are scored on dev and both met their threshold,
so v4.05 is not needed and milestone M6 is reached. Path B is frozen at attempt 1 of each
model. Next is req_005 v5.01, the configuration freeze and the single test-set evaluation of
all six arms, which needs its spec. Before it: Nova Pro's training charge entered with
`nova_tune bill`.

## 11. As built (2026-11-18)

No tuned model was asked anything during the build: none existed. The scores, the verdicts,
and the cost of the owner's tuned runs are at the end of this section.

What was run:

- `venv\Scripts\python -m pytest -q`: 1,925 passed (103 new). With the training stack
  blocked: 1,924 passed, 1 skipped. `venv\Scripts\python -m ruff check .`: clean.
- `venv\Scripts\python -m scripts.evaluate --predictor nova-lite-tuned --split quick_dev`
  (and `nova-pro-tuned` on dev): "nova-lite has not been fine-tuned: the tuning record has
  no job for it. The owner runs python -m scripts.nova_tune launch --model nova-lite --yes
  (billed)", exit 2, no run left behind. As expected.
- `venv\Scripts\python -m scripts.nova_tuned --runs x --baselines y`: exit 2, "run x has no
  evaluation.json".
- `venv\Scripts\python -m scripts.nova_export --check`: the gate record is reported stale.
  That is from the baseline runs, not from this version: the committed record says $0 spent
  and the ledger now holds $0.49. The owner's next `nova_export` (run sheet, stage C)
  rewrites it.

Differences from sections 3 and 4:

- **The thresholds are config**, not constants: `config/frontier.yaml` version 4 gains
  `tuned_evaluation` (the splits, the verdict split, 10 points, 98%).
- **A tuned run's `model_id` is the deployment's ARN**, because that is what is called and
  what a resume must compare (D5 then needs no new code). The base model id travels beside
  it as `usage_model_id`, which the runner writes into the ledger. An ARN in the ledger
  would have no price, and the v4.03 cap check counts an unpriced record as nothing.
- **The predictor description gained two optional fields**, `usage_model_id` and `details`
  (which tuned model: attempt, ARNs, export id, hyperparameters). They appear only when a
  predictor has them, so every earlier run and predictor describes itself as before.
  `details` is not part of what a resume compares. This is a small change to the shared
  runner of v2.03.
- **The live read of the deployment is a function in `bedrock.py`** (`deployment_status`),
  not a call through `BedrockTuning`: the factory needs one read and none of job control.
- **The record and gate checks run before the sign-in**, so "nothing was launched" needs no
  `aws sso login` to be told.
- **`run_predictions(run_id)` was added to the runner**: the report needs the per-question
  scores of two runs, and reading a predictions file is the runner's business.
- **V2 checks more than section 4.2 lists**: also the two arms, the reply limit, that the
  baseline asked the configured base model, that no tuned run appears twice, and that no
  baseline is left unpaired.
- **V3 does not require the deployment to be Active today**, only that the runs asked the
  latest, completed attempt's deployment. The report can be written again after a teardown.
- **V6 passes when every model that was scored has a dev pair.** A report for one model is
  written, marked `partial`, and exits 0 if that model met its threshold; the output and
  the report say that M6 needs both.
- **The sign test's value is stored as a number with six significant digits**; with no
  changed question it is absent and the report says so. The cost per correct answer is
  absent when a run got nothing right.
- **Example ids are in the JSON file only.** The Markdown file gives the counts.
- Not built: nothing of section 4 was left out.

Still unverified, because only a real deployment can show it: whether the first call to an
idle deployment is slow or refused; under which `ModelId` CloudWatch lists a deployment's
calls; that the bill charges tuned inference at the base model's prices on this account.

### Live results (the owner's runs, 2026-11-22)

Both tuned models were asked every question of quick-dev and dev, and `python -m
scripts.nova_tuned` wrote the report: V1 to V6 passed, and **both models met their
threshold**. v4.05's trigger did not fire.

| Arm | Split | Run | Well-formed | Execution accuracy | Untuned | Gain in points |
|---|---|---|---:|---:|---:|---:|
| B1-Pro | dev | `20261122T114511Z-eval-nova-pro-tuned-dev-276f` | 499 of 499 | 359 of 499, 71.94% | 190, 38.08% | +33.87 |
| B1-Lite | dev | `20261122T103321Z-eval-nova-lite-tuned-dev-f97f` | 499 of 499 | 329 of 499, 65.93% | 123, 24.65% | +41.28 |
| B1-Pro | quick-dev | `20261122T103709Z-eval-nova-pro-tuned-quick-dev-5d0c` | 200 of 200 | 141 of 200, 70.50% | 75, 37.50% | +33.00 |
| B1-Lite | quick-dev | `20261122T103142Z-eval-nova-lite-tuned-quick-dev-7daa` | 200 of 200 | 140 of 200, 70.00% | 47, 23.50% | +46.50 |

Needed on dev: 240 correct for Nova Pro, 173 for Nova Lite, 490 well-formed replies each.
Question by question on dev, tuning fixed 190 and broke 21 for Nova Pro, and fixed 228 and
broke 22 for Nova Lite (sign test p under 0.000001 for both). The four tuned runs cost
$0.45 at list price ($0.3005, $0.1222, $0.0225, $0.0091). The full figures are in
`reports/frontier/nova_tuned_report.md`.

What the runs settled, of the things section 7 left open:

- **A deployment has almost no cold start.** The first call of each five-question trial
  took 3.2 and 3.7 seconds; later calls 0.5 and 0.6. No call failed while warming up.
- **Bedrock reports token counts for a deployment** as for a base model, and the input
  counts equal the baselines' exactly (313,156 on dev).
- **The tuned replies are shorter**: about 31 output tokens per question against 44 to 46
  untuned, so asking a tuned model costs slightly less than asking its base model.
- **Tuned Nova Pro stops on throttling at its 20 requests a minute.** Its dev run stopped
  at 205 and again at 440 of 499 after botocore's 12 adaptive attempts, on different
  questions, and was resumed twice; the result is one run of 499. Tuned Nova Lite (200 a
  minute) never stopped. The follow-up was built the same day: Nova Pro, tuned and untuned,
  is paced at 18 requests a minute (`models.nova-pro.requests_per_minute` in
  `config/frontier.yaml`, the owner's decision). Pacing is not part of a run's settings.
- Tuned Nova Lite is level with tuned Nova Pro on quick-dev and 6 points behind on dev,
  for about a quarter of the training cost.

Still open: whether the bill charges the tuned calls at the base model's prices, and Nova
Pro's training charge (estimate $51.29); both are read from the bill when they show.
