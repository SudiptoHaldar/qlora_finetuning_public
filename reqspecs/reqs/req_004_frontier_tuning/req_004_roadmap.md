# req_004 Frontier Tuning — Roadmap

**Spec group**: req_004_frontier_tuning  
**Last updated**: 2026-11-22 — v4.01 to v4.04 complete (✅): both Nova models tuned, deployed, and scored on dev; both met their threshold, so v4.05 (conditional) is not needed; 5 envisioned versions.  

Version log:

- **2026-11-13** — group created from `reqspecs/0. PLANNING/execution_plan.md` §2; `req_004_01_spec.md` is a placeholder.
- **2026-11-14** — plan revised before any version was specced: measured Qwen runs move to AWS SageMaker; the hosted path is Amazon Nova Pro and Nova Lite on Bedrock (OpenAI dropped); Q5 and Q6 resolved. Version scopes updated in place.
- **2026-11-16** — `req_004_01_spec.md` written for v4.01 (📝) from the Converse definition in the installed boto3, the AWS documentation, and the public price list (published 2026-11-11, prices unchanged). The plain model ids `amazon.nova-pro-v1:0` and `amazon.nova-lite-v1:0` can be called on demand in us-east-1; decoding is greedy (`topK: 1`, temperature 0); retrying is botocore's; the runner already records Bedrock's token counts, so the adapter writes nothing to the ledger. The version also adds `config/pricing.yaml`, an `aws_check` check C8, and a committed baseline report with the Nova to Qwen token ratio. Found on the way: the documentation gives 0.00001 as the lowest temperature and recommends 0; both models are Active but past their guaranteed date, and a Legacy model cannot be fine-tuned; the fine-tuning limits (8 to 20,000 samples, 32k context) hold our 5,383 records, which settles the open part of Q1. The AWS sign-in had expired, so nothing was checked against the live account: that is an owner step in the spec. Three decisions wait for the owner (who runs the billed calls; B0 on dev now or in v4.04; in-region id or cross-region profile). The two quick-dev baselines are estimated at under $0.35.
- **2026-11-16** — the owner decided the three questions, each as recommended: the owner runs the billed calls; B0 is scored on dev in v4.01 as well as on quick-dev (at most $0.75 more, so under $1.10 for the version); the plain in-region model ids are used. The spec's baseline report now takes four runs.
- **2026-11-16** — `prp_004_bedrock_nova_adapter_v1.md` written and executed the same day (🟡): `config/frontier.yaml`, `config/pricing.yaml`, `qlora_finetuning/frontier/` (`config`, `provider`, `bedrock`, `predictor`, `pricing`, `baseline_report`), `scripts/nova_baseline.py`, the `nova-pro` and `nova-lite` entries of `scripts/evaluate.py`, and `aws_check` check C8; 168 new tests, 1,560 in total, lint clean, also with the training stack blocked. **No call has been made to Bedrock**: the AWS sign-in was expired throughout, and the owner runs every billed call. Signed out, the Nova command stops with the sign-in message before any run exists. Still open until the owner's five-example trial: whether Bedrock accepts temperature 0, and whether the account may call both plain model ids. Remaining owner steps: sign in, `aws_check` (8 of 8), the trial, four baseline runs (under $1.10 in total), `scripts.nova_baseline`, commit the report. Found on the way: botocore's `max_attempts` counts retries only, so the client uses `total_max_attempts`.
- **2026-11-16** — `req_004_02_spec.md` written for v4.02 (📝) from the AWS pages on preparing Nova fine-tuning data, AWS's own dataset validation script (read, not run), and a trial export. The files are 10.7 MB (fit, 4,884 lines) and 1.1 MB (dev, 499); no id is a test id; no text holds a reserved word (nine words, any case, which AWS's script checks more strictly than its guide says); every line is ASCII. Found on the way: 52 fit records are exact copies of another and 7 more share a prompt with a different gold reply (all are kept, so both paths train on the same examples); Bedrock may bill validation tokens, which is not documented; a free `CountTokens` call exists but its support for Nova is not confirmed. The version adds a validator of the files on disk (X1 to X7), a Path B cost estimate, and a committed gate record that v4.03 must find passed. Training both models costs $65 to $98 for 2 epochs and $98 to $146 for 3, depending on the token ratio, which is not known until the v4.01 baselines are run; without it the gate fails on purpose. Three decisions wait for the owner (epochs; the Path B cap; dev as the validation file).
- **2026-11-16** — the owner started the v4.02 PRP and its execution without changing the three decisions, so they are taken as recommended (2 epochs, a $200 cap for Path B, dev as the validation file). `prp_004_training_export_gate_v2.md` written and executed the same day (🟡): `frontier/export.py`, `export_validation.py`, `training_estimate.py`, `staging.py`, `scripts/nova_export.py`, and new keys in `config/frontier.yaml` (version 2); 144 new tests, 1,704 in total, lint clean, also with the training stack blocked. **The real export passes X1 to X7**: 4,884 and 499 lines (10.7 MB and 1.1 MB), no test id, no reserved word; export id `c2d67fe65c7f`. **The gate is closed, as intended**: G1 fails because the v4.01 baselines have not been run, so no token ratio is measured. At the assumed ratio of 1.5 the estimate is $120.61 against the $200 cap (training $97.63, $107.39 with the margin; evaluation to come $9.32; storage $3.90). Nothing was uploaded and nothing was asked of AWS. Remaining owner steps: the v4.01 baselines and report, then `scripts.nova_export` (gate open) and `scripts.nova_export --upload`. Found on the way: `Decimal` rounds half a cent to the even cent, so amounts are shown rounded half up; a rerun must not add a run to the committed ledger each time, so an export is recorded once per gate record and once per staging.
- **2026-11-16** — v4.02 committed as release v1.6. `req_004_03_spec.md` written for v4.03 (📝) from the job, deployment, and deletion calls in the installed boto3 and the AWS pages on hyperparameters, stopping, monitoring, and on-demand deployment. A `plan` command checks eight preconditions (T1 to T8: gate open and current, files in S3 with the recorded checksums, model fine-tunable and Active, no Provisioned Throughput, not launched yet, within the cap, committed tree) and launches nothing; `launch` creates one job per command with a request token, so a repeat cannot create a second job; `status` only reads and is how an interruption is survived; a committed tuning record holds the job, model, and deployment ARNs. Found on the way: a job reports no token count and no amount, so the ledger holds the estimate until the owner enters the bill; a stopped job is billed and makes no model; AWS recommends both "start with the defaults" (warmup 10) and warmup of dataset size over 160 or 320 (31 for Lite, 15 for Pro). Job control is a class of its own, not part of `FrontierProvider`: a departure from plan §4.7, stated in the spec. Three decisions wait for the owner (learning rate and warmup; which model first; what must be true before the second launch). Nothing can be launched until the v4.01 baselines are run, the gate is open, and the files are staged.
- **2026-11-16** — the owner decided the three v4.03 questions, each as recommended: Bedrock's default learning rate and warmup steps for both models; Nova Lite first; Nova Pro only after Nova Lite's charge has been checked on the AWS bill. The spec gains check T9: a `nova-pro` launch is refused until the tuning record shows a completed `nova-lite` job and the ledger holds its `bill` record; `plan` then shows the billed amount beside the estimate.
- **2026-11-16** — `prp_004_fine_tuning_jobs_v3.md` written and executed the same day (🟡): `frontier/tuning.py` (`BedrockTuning`: plan, checks T1 to T9, launch, refresh, deploy, stop, bill, teardown), `frontier/tuning_record.py`, `scripts/nova_tune.py`, `config/frontier.yaml` version 3 (customization model ids, tuned arms, hyperparameters, launch order), and in-memory stand-ins for Bedrock and S3 in the tests; 118 new tests, 1,822 in total, lint clean, also with the training stack blocked. **Nothing was launched and nothing was asked of Bedrock**: the sign-in was expired, the gate is closed, and the files are not staged. `plan`, signed out, fails T1 and skips the checks that need AWS, as intended. Found on the way: a reply to "create the job" can be lost after Bedrock has created it, so a launch first looks for a job of the planned name with its own request token and adopts it; botocore prints a traceback for an expired sign-in, now silenced. Unverified until a real job: what Bedrock answers for an unknown job name, and where the metrics files land. Remaining owner steps, in order: the v4.01 baselines and report; the v4.02 gate and upload; `nova_tune plan`; launch Nova Lite; status; read its charge and enter it with `bill`; launch Nova Pro; deploy both.
- **2026-11-18** — the owner ran the four v4.01 baseline runs on 2026-11-16 (dev execution accuracy: B0-Lite 123 of 499, 24.65%; B0-Pro 190 of 499, 38.08%). Nova Pro's quick-dev run was throttled at 182 of 200 and resumed: the account allows Nova Pro 25 requests a minute, so `config/frontier.yaml` now retries in `adaptive` mode with 12 attempts. The baseline report (`nova_baseline`) is not written yet, so v4.01 stays 🟡. `req_004_04_spec.md` written for v4.04 (📝): two tuned predictors that call the deployments named in the tuning record, a tuned-model report with a paired comparison against each B0, and a verdict per model. Three decisions are open for the owner: the threshold that makes v4.05 eligible (recommended: less than 10 points above its own B0 on dev execution accuracy, or format validity under 98%), whether quick-dev is scored too, and whether the deployments are kept until v5.01.
- **2026-11-18** — the owner decided the three v4.04 questions, each as recommended: a tuned model has missed when it is less than 10 points above its own B0 on dev execution accuracy or its format validity is under 98%; quick-dev and dev are both scored; the deployments are kept. `prp_004_tuned_dev_evaluation_v4.md` written and executed the same day (🟡): `frontier/tuned.py` (which attempt is this experiment's tuned model), `build_tuned_predictor` and the predictors `nova-lite-tuned` and `nova-pro-tuned`, `frontier/tuned_report.py` (pairs, sign test, cost, verdict, checks V1 to V6), `scripts/nova_tuned.py`, and `config/frontier.yaml` version 4 with the thresholds. A predictor may now name a `usage_model_id`: a tuned model is called by its deployment's ARN and recorded in the ledger under its base model, which is what it is billed as. 1,925 tests, lint clean. No tuned model was asked anything: none exists until v4.03's jobs are launched.
- **2026-11-18** — **v4.01 complete (✅)**. The owner ran `nova_baseline` on the four baseline runs: B1 to B4 passed and `reports/frontier/nova_baseline_report.md` is written. The four runs cost $0.49 at list price. Measured Nova to Qwen token ratio: 0.985 (0.9848 on quick-dev, 0.9851 on dev), against the 1.5 every estimate had assumed, so the Path B training estimate falls by about a third when v4.02's gate reprices. The B0 part of milestone M4 is reached.
- **2026-11-18** — **v4.02 complete (✅)**. The owner ran `nova_export` (gate open: 9 of 9 checks, measured ratio 0.9851, training for both models $64.11 or $70.53 with the margin, Path B total $82.46 of the $200 cap) and `nova_export --upload` (both files staged under `data/bedrock/c2d67fe65c7f/`, run `20261119T015413Z-export-nova-export-c2d67fe65c7f-95c9`). v4.03's first launch, Nova Lite, is now possible once the gate record is committed.
- **2026-11-22** — **v4.03 complete (✅)**. The owner ran both fine-tuning jobs (Nova Lite 2026-11-19, 2 h 19 min; Nova Pro 2026-11-21 to 2026-11-22, 18 h 40 min, most of it waiting to start) and deployed both tuned models on-demand; both deployments are Active. Nova Lite was billed $14.60 against an estimate of $12.82 (1.138 times); Nova Pro's charge is not on the bill yet (estimate $51.29, about $58 at that proportion). Bedrock reports -1 for a Nova job's losses, so they are read from the loss curves (fixed in `frontier/tuning.py`). v4.04's tuned runs are now possible.
- **2026-11-22** — **v4.04 complete (✅)**; milestone M6 reached. The owner ran the four tuned runs ($0.45) and `nova_tuned` wrote the report: V1 to V6 passed and both models met their threshold. Dev execution accuracy: B1-Pro 359 of 499 (71.94%, +33.87 points over B0-Pro), B1-Lite 329 of 499 (65.93%, +41.28 over B0-Lite); every reply well-formed. v4.05's trigger did not fire, so it stays conditional and unneeded. Tuned Nova Pro's dev run stopped twice on its 20 requests a minute and was resumed. Follow-up built the same day (the owner's decision): Nova Pro, tuned and untuned, is paced at 18 requests a minute by `models.nova-pro.requests_per_minute` in `config/frontier.yaml`; pacing is not part of a run's settings.

This file is the **single source of truth** for the req_004 version sequence. The group owns
**Path B, the hosted path (Amazon Nova Pro and Amazon Nova Lite, each fine-tuned through Amazon Bedrock), up to two tuned models evaluated on dev. It extends `req_001_finqa_data` and `req_002_dsl_evaluation`; it does NOT own the test evaluation or the report (`req_005_comparison`)**.

## Why this group exists

The brief asks for a second path: train a frontier model on the same FinQA data and compare
cost and accuracy with the QLoRA path. Neither planning guide covers this, so the specs here are
written from current AWS documentation.

Decided 2026-11-13/14 (Q1): OpenAI is out (its self-serve fine-tuning is closed to new users
and ends 2027-01-06). The provider is Amazon Bedrock in us-east-1, with **two hosted models,
Nova Pro and Nova Lite**, each with an untuned and a tuned arm, to show how accuracy scales with
price. Prices from the AWS price list published 2026-11-11, per 1M tokens (training / tuned
input / tuned output): Pro $8.00 / $0.80 / $3.20; Lite $2.00 / $0.06 / $0.24; storage $1.95 per
custom model per month. Estimated total for both models at 3 epochs: about $155 (token volumes
are estimates until v1.04). Provisioned Throughput ($55.00–$60.50 per hour) must not be used;
tuned models are deployed on-demand.

Standing decisions: "training" means supervised fine-tuning through Bedrock model
customization; prompts are byte-identical to Path A's canonical records; the provider sits
behind a small `FrontierProvider` interface; the untuned model is measured first; nothing is
uploaded before the training file is validated and its cost estimated; launching a paid job is
operator-gated and bounded by a budget cap; the API key stays in `.env`.

**Status legend** (in lifecycle order):

- ⏸ **Conditional** — only starts when a specific trigger fires
- 📋 **Planned** — next-up; depends on a prior version landing
- 📝 **Spec ready** — req_004_NN_spec.md written + decisions locked; PRP pending
- 🛠 **PRP ready** — PRP authored + validated; execution pending
- 🚧 **In flight** — PRP being executed or impl in progress
- 🟡 **Operator-gated** — prepared; waiting on a step only the user performs
- ✅ **Complete** — spec + impl shipped
- 🗄 **Shelved** — parked indefinitely; revival trigger noted

---

## Quick-reference table (version sequence)

| # | Title | Status | Depends on | Trigger to start |
|---|---|---|---|---|
| **v4.01** | **Bedrock Nova adapter and zero-shot baselines** — Record model availability, limits, and prices for Nova Pro and Nova Lite checked on the day | ✅ Complete (2026-11-18) | v2.03, v1.03, v0.05 | — |
| **v4.02** | **Training file export and budget gate** — Render fit and dev once to Bedrock conversation JSONL (`bedrock-conversation-2024` schema), shared by both models, and upload to S3; validate against the format rules; count tokens; estimate training and evaluation cost from `config/` pricing with a retrieval date; refuse above the budget cap; dry-run report | ✅ Complete (2026-11-18) | v4.01 | — |
| **v4.03** | **Fine-tuning job** — Create one Bedrock model-customization job per model (Pro, Lite) with recorded hyperparameters (`epochCount`, `learningRate`, `learningRateWarmupSteps`); poll; create the on-demand deployment for each custom model; capture job ARN, custom model ARN, billed tokens and amount into the ledger; resumable after interruption; runbook entry including how to delete the deployments and the stored models | ✅ Complete (2026-11-22) | v4.02 | — |
| **v4.04** | **Tuned-model dev evaluation** — Arms B1-Pro and B1-Lite scored on dev with the shared runner; each compared with its B0; cost per example recorded | ✅ Complete (2026-11-22) | v4.03 | — |
| **v4.05** | **One corrective iteration** — For the model that missed, change one job setting (epoch count or learning rate) and rerun | ⏸ Conditional | v4.04 | A B1 dev result below its v4.04 threshold, and budget remaining |

---

## Detail per version

### v4.01 — Bedrock Nova adapter and zero-shot baselines  ✅

**Status**: ✅ Complete: code built 2026-11-16 (`prp_004_bedrock_nova_adapter_v1.md`); the owner's four baseline runs 2026-11-16 and the baseline report 2026-11-18 (spec section 11)  
**Depends on**: v2.03, v1.03, v0.05  

**Scope**: Record model availability, limits, and prices for Nova Pro and Nova Lite checked on the day. `FrontierProvider` interface (one method, `converse`) and a Bedrock implementation (Converse API), model id from `config/frontier.yaml`; inference `Predictor` with greedy decoding and nothing else sent; botocore retries; stub-client tests. `config/pricing.yaml`; `aws_check` C8 (both models active and callable on demand); `scripts/nova_baseline.py` and a committed baseline report. Arms B0-Pro and B0-Lite scored on quick-dev and on dev.

**Decisions and open questions**: Confirmed from AWS docs in the spec: fine-tuning limits for Pro and Lite (8 to 20,000 samples, 32k context) and the on-demand deployment steps for custom models (`CreateCustomModelDeployment`, the deployment ARN as model id, us-east-1 only). Decided by the owner 2026-11-16 (spec section 8): the owner runs the billed calls; B0 on dev now as well; in-region model ids. Settled by the build, because the sign-in had expired: that the account may call both ids, and that temperature 0 is accepted. 🟡 The baseline runs are billed (under $1.10). Contributes to milestone M4.

### v4.02 — Training file export and budget gate  ✅

**Status**: ✅ Complete: code built 2026-11-16 (`prp_004_training_export_gate_v2.md`); the owner opened the gate and staged the files on 2026-11-18 (spec section 11)  
**Depends on**: v4.01  

**Scope**: Render fit and dev once to Bedrock conversation JSONL (`bedrock-conversation-2024` schema), shared by both models, and upload to S3; validate against the format rules; count tokens; estimate training and evaluation cost from `config/` pricing with a retrieval date; refuse above the budget cap; dry-run report. Guard: zero test ids in any exported file.

**Decisions and open questions**: Built 2026-11-16 (`prp_004_training_export_gate_v2.md`). Taken as recommended, because the owner started the PRP without changing them: the gate prices 2 epochs, the cap for Path B is $200 of Q2's $300, and dev is the validation file. Open: whether Bedrock bills validation tokens (covered by the 10% margin), and whether `CountTokens` supports Nova. The gate cannot pass until the v4.01 baselines have been run, because the Nova to Qwen token ratio comes from them. 🟡 The upload to S3 is the owner's.

### v4.03 — Fine-tuning job  ✅

**Status**: ✅ Complete: code built 2026-11-16 (`prp_004_fine_tuning_jobs_v3.md`); the owner ran both jobs and deployed both tuned models; both deployments Active 2026-11-22 (spec section 11)  
**Depends on**: v4.02  

**Scope**: Create one Bedrock model-customization job per model (Pro, Lite) with recorded hyperparameters (`epochCount`, `learningRate`, `learningRateWarmupSteps`); poll; create the on-demand deployment for each custom model; capture job ARN, custom model ARN, billed tokens and amount into the ledger; resumable after interruption; runbook entry including how to delete the deployments and the stored models.

**Decisions and open questions**: 🟡 Operator-gated: the user launches the paid jobs. Never create Provisioned Throughput (checked before every launch and deploy; a guard test forbids the create call). Locked in the spec: both models use the same hyperparameters, with 2 epochs, which is each model's documented default. Decided by the owner 2026-11-16 (spec section 8): Bedrock's defaults for the learning rate and warmup steps (1.00E-5 and 10); Nova Lite first; Nova Pro only after Nova Lite's charge has been checked on the bill, enforced as check T9. Not confirmed in the AWS documentation: whether a failed job is billed, how many jobs may run at once, and whether an idle deployment costs anything.

### v4.04 — Tuned-model dev evaluation  ✅

**Status**: ✅ Complete: code built 2026-11-18 (`prp_004_tuned_dev_evaluation_v4.md`); the owner's four tuned runs and the report 2026-11-22; both models met their threshold (spec section 11)  
**Depends on**: v4.03  

**Scope**: Arms B1-Pro and B1-Lite scored on dev with the shared runner; each compared with its B0; cost per example recorded.

**Decisions and open questions**: Decided by the owner 2026-11-18 (spec section 8), each as recommended: (1) the dev threshold that makes v4.05 eligible: a tuned model has missed when it is less than 10 points above its own B0 on dev execution accuracy, or its format validity is under 98%; (2) quick-dev is scored before dev; (3) the deployments are kept until v5.01. Locked in the spec: the latest attempt in the tuning record is the model scored; each B1 run is paired with its B0 run and refused unless prompt, split, settings, and order are equal; the verdict comes from dev only and launches nothing. The owner runs the billed evaluations (about $0.50 in all; Nova Pro's deployment allows 20 requests a minute). Reaching it is milestone M6.

### v4.05 — One corrective iteration  ⏸

**Status**: ⏸ Conditional; the trigger did not fire: both models met their v4.04 threshold on 2026-11-22  
**Depends on**: v4.04  
**Trigger**: A B1 dev result below its v4.04 threshold, and budget remaining  

**Scope**: For the model that missed, change one job setting (epoch count or learning rate) and rerun.

**Decisions and open questions**: One iteration per model at most. Available from req_003 v3.05 (2026-11-20): `qlora_finetuning/evaluation/failures.py` sorts the wrong replies of any stored run, Nova's included, into computed categories (`run_failures`, `count_failures`), and Path A's pattern of registering one change before it is paid for. On the stored untuned dev runs: Nova Pro 190 right, 168 programs that do not parse (132 of them nested calls), 72 with more steps than gold; Nova Lite 123, 212, and 100. Among the wrong results, 56 of Nova Pro's and 52 of Nova Lite's are the gold value off by a factor.
