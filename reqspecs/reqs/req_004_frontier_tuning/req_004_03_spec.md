# Frontier Tuning

Spec Group      :   req_004_frontier_tuning  
Feature Name    :   frontier_tuning  
Version         :   3  
Status          :   ✅ Complete: code built 2026-11-16 (`prp_004_fine_tuning_jobs_v3.md`); the owner ran both jobs (Nova Lite 2026-11-19, Nova Pro 2026-11-21 to 2026-11-22) and deployed both tuned models on-demand; both deployments Active 2026-11-22 (section 11)  
Last updated    :   2026-11-22  

## 1. v4.03 — Fine-tuning jobs

This version spends the money. It starts one Bedrock model-customization job for Nova Lite
and one for Nova Pro on the files v4.02 staged, follows each to its end, and puts each tuned
model behind an on-demand deployment that v4.04 can ask questions of.

It adds:

- a **plan** command: what would be launched and what it would cost, with every
  precondition checked, and nothing started;
- **launch**, one model at a time: the job is created once, however often the command is
  repeated, and its estimated tokens go into the ledger;
- **status**: reads the jobs from Bedrock, records how they ended, and fetches the loss
  curves. It can be run any number of times and from a new session, which is how an
  interruption is survived;
- **deploy**: an on-demand deployment for each finished model;
- **stop**, **bill**, and **teardown**: stop a running job, enter the amount AWS billed,
  and delete deployments and tuned models when they are no longer needed;
- a committed **tuning record** that holds the job, model, and deployment ARNs.

Launch, deploy, stop, and teardown create, change, or delete AWS resources, and launch is
billed. **The owner runs them.** Plan and status only read.

**Depends on**: v4.02, whose gate must be open and whose files must be staged. Neither is
true today: the v4.01 baselines have not been run. The code of this version can be built
and tested before that; nothing can be launched before it.

## 2. What exists today (verified 2026-11-16)

### 2.1 What earlier versions provide

| Item | State |
|---|---|
| Gate record (v4.02) | `reports/frontier/nova_training_gate.json`: file hashes, the estimate, the cap, the verdict, and after an upload the S3 locations. **Today: `passed: false`, not staged** |
| `config/frontier.yaml` | `customization.epoch_count: 2`; `budget.path_b_cap_usd: 200`; the inference model ids |
| `config/aws.yaml` | The customization model ids `amazon.nova-pro-v1:0:300k` and `amazon.nova-lite-v1:0:300k`; the role `qlora-bedrock-customization`; the bucket and its `bedrock/` prefix |
| Bedrock role (v0.05) | Trusted by Bedrock for customization jobs of this account in us-east-1 only; may read `data/*` and read and write `bedrock/*` |
| `stage_export`, `FakeS3` | Checksum comparison against S3, and its test stand-in |
| `aws_check` | C4 the role, C6 both models fine-tunable, C8 both models Active |
| Ledger (v0.04) | Fields `training_tokens`, `external_id`, `billed_amount_usd` (only with source `bill`), `replaces`; phases `train`, `deploy`, `storage`; arms `B1-Pro`, `B1-Lite` |
| `scope="measured"` | A run refuses to start from uncommitted changes |
| `config/pricing.yaml` | Training $8.00 and $2.00 per 1M tokens; storage $1.95 per model per month |

### 2.2 The API, from the installed boto3 (no network)

| Call | What matters |
|---|---|
| `create_model_customization_job` | Needs `jobName` and `customModelName` (63 characters, letters, digits, hyphens), `roleArn`, `baseModelIdentifier`, `trainingDataConfig.s3Uri`, `outputDataConfig.s3Uri`. Takes `validationDataConfig.validators[].s3Uri`, `hyperParameters` (text to text), `customizationType`, `jobTags`, `customModelTags`, and **`clientRequestToken`**. Returns `jobArn`. Errors include `ConflictException`, `ServiceQuotaExceededException`, `ValidationException` |
| `get_model_customization_job` | `status`: `InProgress`, `Completed`, `Failed`, `Stopping`, `Stopped`. Also `statusDetails` (validation, data processing, training), `failureMessage`, `outputModelArn`, `creationTime`, `endTime`, `trainingMetrics.trainingLoss`, `validationMetrics[].validationLoss`. **No token count and no amount** |
| `stop_model_customization_job` | By job ARN |
| `create_custom_model_deployment` | `modelDeploymentName`, `modelArn`, `clientRequestToken`, `tags`. Returns `customModelDeploymentArn` |
| `get_custom_model_deployment` | `status`: `Creating`, `Active`, `Failed` |
| `delete_custom_model_deployment`, `delete_custom_model` | By ARN |
| `list_provisioned_model_throughputs` | Lists Provisioned Throughput in the account: a read |

### 2.3 What AWS documents (read 2026-11-16)

| Topic | Finding |
|---|---|
| Hyperparameters, Nova Pro and Lite | `epochCount` default 2; `learningRate` default 1.00E-5, at most 1.00E-4; `learningRateWarmupSteps` default 10, at most 100. "We recommend starting with the default hyperparameters". For warmup: "We recommend setting the warmup steps by dividing the dataset size by ... 160 for Amazon Nova Lite, and 320 for Amazon Nova Pro". For 4,884 records that is 31 and 15, so **the two recommendations disagree** |
| Cost of epochs | "Each epoch processes the entire training dataset once" |
| Duration | "The customization job can take several hours" |
| Phases | Validation, data processing, training; a failed job says where it failed |
| Stopping | "You can't resume a stopped job. Amazon Bedrock charges for the tokens that it used to train the model before you stopped the job", and creates no model |
| Repeating a request | "To prevent the request from completing more than once, include a `clientRequestToken`" |
| On-demand deployment | Nova Pro and Nova Lite: us-east-1 only. The model must have been customized on or after 7/16/2025. When the deployment is `Active`, its ARN is the `modelId` for inference. "You only pay for what you use" |
| Deleting | A deployment is deleted by its ARN. A tuned model can be deleted; for some model kinds its deployments must go first |

Not confirmed: whether a job that fails in validation or data processing is billed; how
many customization jobs may run at once; whether a deployment that receives no request
costs anything (the price list has no hourly item for it).

### 2.4 What the two jobs will cost

Training, 2 epochs, from the v4.02 estimate. The token ratio is still unmeasured.

| | If Nova counts as Qwen does | If Nova counts 1.5 times as many |
|---|---:|---:|
| Nova Lite | $13.02 | $19.53 |
| Nova Pro | $52.07 | $78.10 |
| Both | $65.08 | $97.63 |
| Storage of the two tuned models, per month | $3.90 | $3.90 |

Nova Lite's job is a quarter of Nova Pro's and exercises exactly the same files and code.

## 3. Locked decisions

### What is launched

| # | Decision | Reason |
|---|---|---|
| D1 | One job per model, each on the training and validation objects named in the gate record, with `customizationType: FINE_TUNING` | Plan §4.7; v4.02 section 8 (dev is the validation file) |
| D2 | `epochCount` is the count the gate priced. All three hyperparameters are sent explicitly, as text, from `config/frontier.yaml`, and recorded | A default that AWS changes later must not change what was run. The other two are Bedrock's defaults, learning rate 1.00E-5 and 10 warmup steps (the owner's decision, 2026-11-16) |
| D3 | Both models get the same hyperparameters | One recipe, two sizes: a difference in results is then the model's. v4.05 may change one setting for one model |
| D4 | Names: job and model `qlora-finqa-<model>-<export id>-a<attempt>`, deployment the same; output under `s3://<bucket>/bedrock/<export id>/<model>-a<attempt>/` | A name says which files and which attempt; the role may write under `bedrock/` |
| D5 | The `clientRequestToken` is derived from the model, the export id, the hyperparameters, and the attempt number | Repeating a launch after a crash or a lost reply asks for the same job, not a second one |
| D6 | Jobs and tuned models are tagged `project: qlora-finetuning` | The budget of v0.05 may need re-scoping by tag |
| D7 | The base model for a job is the customization id (`...:300k`), from `config/frontier.yaml` beside the inference id | Two ids per model, one place (v4.01 gotcha) |

### Before a launch

| # | Decision | Reason |
|---|---|---|
| D8 | A launch runs every check of section 4.3 first and refuses if one fails. `plan` runs the same checks and launches nothing | The owner sees exactly what a launch would do |
| D9 | The cap is checked again per launch: what Path B has already committed (the ledger's inference tokens and training tokens at list price) plus this job with the margin must be at most the cap | The gate priced each model once. A second attempt, or a changed config, must not pass on the first attempt's verdict |
| D10 | A launch is a run of kind `train` with scope `measured` | The tuned model's numbers go into the report, so the job must trace to a commit. It refuses from uncommitted changes |
| D11 | **Nothing is launched automatically.** No command launches more than the model it was told to, and no command launches because another finished | Working agreement 10 |
| D12 | One model per launch command. **Nova Lite is launched first, and Nova Pro only after Nova Lite's job has completed and the amount AWS billed for it has been entered with `bill`** (check T9). `plan` then shows that amount beside the estimate and what the same proportion would make Nova Pro cost; whether it is close enough is the owner's judgment, made by launching or not | The owner's decision, 2026-11-16. Lite's job costs a quarter of Pro's, and its bill is the only check of the token estimate before the larger job |

### After a launch

| # | Decision | Reason |
|---|---|---|
| D13 | The **tuning record**, `reports/frontier/nova_tuning.json` with a Markdown twin, is written the moment Bedrock returns a job ARN, and is committed. Per model and attempt: names, token, hyperparameters, S3 locations, the gate record's export id, job ARN and status, where it failed if it did, model ARN, final losses, deployment ARN and status, and the times AWS reports | It is the only memory of a job across sessions. v4.04 reads the deployment ARN from it |
| D14 | At launch one ledger record: phase `train`, the estimated training tokens (tokens of one epoch from the gate record times the epochs), source `computed`, the job ARN as `external_id`, the base inference model id as `model_id`, arm `B1-Pro` or `B1-Lite` | No API reports the tokens billed (v0.04 gotcha) |
| D15 | `bill` appends a record with the amount the owner read from the AWS bill, source `bill`, replacing the estimate | v0.04: dollars enter the ledger only from the bill |
| D16 | `status` only reads. It updates the tuning record from `get_model_customization_job` and `get_custom_model_deployment`. `--wait` repeats it at the interval in config until every job has ended | Resumable by construction: there is nothing to resume but looking again |
| D17 | When a job has completed, `status` downloads its `step_wise_training_metrics.csv` and `validation_metrics.csv` to git-ignored `artifacts/bedrock/<export id>/<model>-a<attempt>/` and writes the first, last, and lowest loss into the tuning record | v4.05 chooses between "more epochs" and "another learning rate" from the curve |
| D18 | A failed or stopped job stays in the record with its message. A new launch for that model is the next attempt number and needs the owner to ask for it (`--attempt`) | A failure is information; a retry is a decision |
| D19 | `deploy` creates one on-demand deployment for a completed model, with its own request token. The model is usable when `status` shows the deployment `Active` | AWS documentation |
| D20 | **No code path creates Provisioned Throughput.** Before a launch and before a deploy, `list_provisioned_model_throughputs` must be empty | $55 to $60.50 an hour (working agreement 12). A guard test fails on the create call anywhere in the package |
| D21 | `teardown` deletes deployments, then tuned models, only those in the tuning record, and only when told which. It never deletes S3 objects | Storage is $1.95 per model per month; deleting a model is not reversible |
| D22 | Job control is a class over the Bedrock control client, `BedrockTuning`, with the client passed in. `FrontierProvider` stays the inference interface | **A departure from plan §4.7**, which put staging, jobs, and deployment in the provider. There is one provider; an interface for five calls that only it makes would be a guess. Tests stub the client |

## 4. Scope to build

### 4.1 Config

`config/frontier.yaml`, `version: 3`. `settings` does not change.

```yaml
models:
  nova-pro:
    customization_model_id: "amazon.nova-pro-v1:0:300k"
    tuned_arm: B1-Pro
  nova-lite:
    customization_model_id: "amazon.nova-lite-v1:0:300k"
    tuned_arm: B1-Lite
customization:
  epoch_count: 2
  learning_rate: "0.00001"          # Bedrock's default (the owner, 2026-11-16)
  learning_rate_warmup_steps: 10    # Bedrock's default (the owner, 2026-11-16)
  launch_order: [nova-lite, nova-pro]   # a later model waits for the earlier one's bill
  name_prefix: qlora-finqa
  output_prefix: bedrock/
  poll_seconds: 60
  tags: {project: qlora-finetuning}
```

`config/aws.yaml` keeps its list of fine-tunable ids for check C6; a test ties the two files.

### 4.2 Modules

| Module | Holds |
|---|---|
| `frontier/tuning_record.py` | The tuning record: read, update one model's attempt, write, render. No AWS |
| `frontier/tuning.py` | `BedrockTuning` (D22): `plan`, `preflight`, `launch`, `refresh`, `deploy`, `stop`, `teardown`. Imports botocore for error codes; the third module allowed to |
| `frontier/bedrock.py` | Gains `bedrock_control_client(aws, cfg)` |

```python
from qlora_finetuning.frontier.tuning_record import read_tuning_record

attempt = read_tuning_record().latest("nova-lite")
attempt.job_status            # "Completed"
attempt.deployment_arn        # the modelId v4.04 passes to the predictor
attempt.deployment_status     # "Active"
```

### 4.3 The checks before a launch

| Id | Check | Passes when | Needs AWS |
|---|---|---|---|
| T1 | Gate open | The gate record says passed and names both S3 objects | No |
| T2 | Gate current | Its prompt id, split hash, epoch count, cap, and price-list date are today's | No |
| T3 | Files in place | `head_object` gives the recorded size and SHA-256 for both objects | Read |
| T4 | Model ready | The customization id is listed as fine-tunable and the model is Active | Read |
| T5 | No Provisioned Throughput | The account has none | Read |
| T6 | Not launched yet | The tuning record has no attempt with this number for this model, and Bedrock has no job of that name | Read |
| T7 | Within the cap | D9 | No |
| T8 | Committed | The working tree is clean (changes under `ledger/` aside) | No |
| T9 | Lite's bill seen | For `nova-pro` only: the tuning record shows a completed `nova-lite` job, and the ledger holds a `bill` record for it (D12). Always passes for `nova-lite` | No |

### 4.4 `scripts/nova_tune.py`

| Command | Does | Who |
|---|---|---|
| `plan [--model M]` | Prints, per model: names, base model, the three S3 locations, hyperparameters, estimated tokens and cost, and T1 to T9; for Nova Pro, once Lite's bill is entered, Lite's billed amount over its estimate and Pro's cost at that proportion. Without a sign-in it runs the local checks and says which were skipped | Anyone |
| `launch --model M [--attempt N]` | T1 to T9, then the job. **Billed** | Owner |
| `status [--wait]` | D16, D17. Prints one line per model and attempt | Anyone |
| `deploy --model M` | D19 | Owner |
| `stop --model M` | Stops a running job. The tokens used so far are billed | Owner |
| `bill --model M --usd AMOUNT` | D15 | Owner |
| `teardown --model M (--deployment \| --all)` | D21 | Owner |

Exit codes: 0 done, or for `plan` every check passed; 1 a check failed, a job failed, or
with `status` a job is still running; 2 a config problem or no sign-in. Every command that
changes AWS prints what it is about to do and the cost first, and needs `--yes`.

### 4.5 Tests

Stub clients only: an in-memory Bedrock that walks a job through its states, and `FakeS3`.

- Names, token, and request: the exact `create_model_customization_job` arguments for
  each model; hyperparameters as text; a second launch sends the same token; another
  attempt, export, or hyperparameter gives another token; names fit AWS's patterns.
- T1 to T9: each passes and fails; Nova Pro is refused while Nova Lite is unfinished, failed, or unbilled, and Nova Lite never waits on Nova Pro; `plan` creates nothing; a launch with a failing check
  creates nothing and writes nothing.
- Launch: the tuning record and the ledger record after a success; a lost reply followed
  by a repeat ends with one job; `ConflictException` and `ServiceQuotaExceededException`
  are reported, not retried into a second job; a dirty tree is refused.
- Status: each job state; where a failed job failed; `--wait` stops when all have ended
  (the clock is passed in); the metrics files are fetched once and summarized; a damaged
  metrics file is reported and does not lose the job's status.
- Deploy: only for a completed model; repeat is the same deployment; `Failed` is shown.
- Stop, bill, teardown: only what the record names; a bill replaces the estimate and can
  itself be replaced; teardown order; a model with a live deployment is not deleted
  without `--all`.
- Cap (D9): a second attempt that would pass the cap is refused; the boundary.
- Guards: no `create_provisioned_model_throughput` anywhere in the package or scripts; no
  command changes AWS without `--yes`; the record holds no example text; only
  `bedrock.py`, `staging.py`, and `tuning.py` import botocore.

### 4.6 Documentation

- Runbook section 6: the sequence from `plan` to `deploy`, what each state means, how long
  to expect, what a failure in each phase suggests, stopping, and **teardown**: delete the
  deployments, delete the tuned models, delete the staged files, and how to confirm that
  nothing of Path B is left that bills.
- `bedrock_nova.md`: sections 2.2 and 2.3. `gotchas.md`: a stopped job is billed and makes
  no model; the two warmup recommendations; storage starts when the model exists.
- `coding.md`: jobs only through `BedrockTuning`; never Provisioned Throughput.
- `administration/aws_commands_run.md`: every live command, as the owner runs it.

## 5. Out of scope

- Asking a tuned model anything, and its dev score (v4.04).
- Choosing a second attempt's setting (v4.05). This version only makes a second attempt
  possible and priced.
- Reconciling the ledger with the bill beyond the one `bill` record per job (v5.02).
- Provisioned Throughput, batch inference, cross-region anything, encryption with a
  customer key, a VPC.
- Notifications. `status --wait` in a terminal is the monitor.

## 6. What this hands to later versions

| For | What |
|---|---|
| v4.04 | The deployment ARN of each model, from the tuning record, as the `modelId` for the predictor of v4.01; the mapping from a deployment to its base model's prices; the tuned arm names |
| v4.05 | The loss curves; attempt numbers; the cost of another attempt checked against the cap |
| v5.01 | The same deployments for the test runs |
| v5.02 | `computed` and `bill` records per job; the job ARNs to find them on the bill |
| v6.02 | Hyperparameters, losses, and dates for the model card; the teardown steps |

## 7. Known limits and follow-ups

- **A failed job may still cost something.** AWS says a stopped job is billed for the
  tokens used; it says nothing about a job that fails by itself. The ledger keeps the
  estimate until the bill replaces it.
- **The estimate is not the bill.** The ledger's `computed` tokens rest on a measured
  ratio and an unpublished tokenizer. Only the `bill` record is what was paid.
- **A job takes hours and cannot be paused.** Stopping it loses it.
- **How many jobs may run at once is not known.** A second launch may be refused with
  `ServiceQuotaExceededException`; nothing is lost by waiting.
- **Storage starts when a model exists**, deployed or not: $1.95 a month each until
  `teardown --all`.
- **A deployment may be slow to become Active**, and whether an idle one costs anything is
  not confirmed. The budget alarm of v0.05 is the backstop.
- **The models can be retired.** A Legacy model cannot be fine-tuned; T4 checks on the day.
- **The tuning record is the only link to a job.** If it is lost, the jobs are still in
  Bedrock under their names and can be read back by hand.

## 8. Decisions for the owner

**Decided by the owner on 2026-11-16**, each as recommended: (1) Bedrock's defaults, learning
rate 1.00E-5 and 10 warmup steps, for both models; (2) Nova Lite first; (3) Nova Pro is
launched only after Nova Lite's charge has been checked on the AWS bill, which the tool
enforces as check T9. The table is kept as the record of what was asked.

| # | Question | Recommendation |
|---|---|---|
| 1 | **Learning rate and warmup steps.** Bedrock's defaults are 1.00E-5 and 10. AWS also recommends warmup of dataset size over 160 for Lite (31) and over 320 for Pro (15), which contradicts "start with the defaults" | The defaults for both models: 1.00E-5 and 10. It is the service as offered, it keeps one recipe for two models (D3), and it leaves warmup as a clean single change for v4.05. The cost is that warmup may be shorter than AWS's formula suggests |
| 2 | **Which model first?** | Nova Lite. Its job costs $13 to $20 against $52 to $78 for Pro and runs through the same files, role, and code. Anything wrong with the data or the setup shows up for a quarter of the price |
| 3 | **What must be true before Nova Pro is launched?** (a) Lite's job has passed validation and data processing; (b) Lite's job has completed; (c) Lite's charge is on the AWS bill and is close to the estimate | (c). It is the only check of the token estimate before the larger job, and the estimate rests on a tokenizer Amazon does not publish and a billing rule it does not fully state. It costs about a day of waiting for the charge to appear. If you would rather not wait, (b) still catches every data and setup problem |

## 9. Acceptance

Code, by Claude:

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
  and without the training stack blocked, with no network and no credentials.
- `venv\Scripts\python -m scripts.nova_tune plan` prints the plan for both models and
  T1 failing, because the gate is closed. **That is the expected state on the day the code
  is built.**
- No test and no command run by Claude creates, stops, or deletes anything in AWS.

Live, by the owner (🟡), each command logged in `administration/aws_commands_run.md`:

1. The v4.01 baselines and report; `nova_export` with the gate open; `nova_export --upload`.
   Commit.
2. `nova_tune plan`: T1 to T9 pass for Nova Lite. For Nova Pro, T9 fails until step 5.
3. `nova_tune launch --model nova-lite --yes`. Commit the tuning record.
4. `nova_tune status --wait` until the job has completed; the loss curve is fetched.
5. Wait for Nova Lite's charge to appear on the AWS bill (about a day), read it, and enter it:
   `nova_tune bill --model nova-lite --usd <amount>`. `nova_tune plan` then shows it beside the
   estimate. If it is far off, stop and revisit the estimate before going on.
6. `nova_tune launch --model nova-pro --yes`, then status as before.
7. `nova_tune deploy --model ... --yes` for each; `status` until both are `Active`.

- The tuning record names a completed job, a model ARN, and an Active deployment for both
  models, and is committed.
- The ledger holds a `train` record per job with its job ARN.
- `aws_check` still passes, and T5 confirms there is no Provisioned Throughput.
- The results (durations, losses, billed amounts) are written into this spec as section 11.
  Reaching it, with v4.04, is milestone M6.

## 10. Next step

This version is complete: both jobs ran and both tuned models are deployed and Active.
Next is v4.04's live part (run sheet, stage G): the five-question trial of each tuned
model, the four runs on quick-dev and dev, then `python -m scripts.nova_tuned`. Nova Pro's
charge is entered with `nova_tune bill --model nova-pro` when it shows on the bill.

## 11. As built (2026-11-16)

No job was launched and nothing was asked of Bedrock during the build: the sign-in was
expired throughout, the gate was closed, and the files were not staged. The durations,
losses, and billed amounts of the owner's launches are at the end of this section.

What was run:

- `venv\Scripts\python -m pytest -q`: 1,822 passed (118 new). With the training stack
  blocked: 1,821 passed, 1 skipped. `venv\Scripts\python -m ruff check .`: clean.
- `venv\Scripts\python -m scripts.nova_tune plan`, signed out: both plans printed; T1
  fails (the gate is closed, the files are not staged), T3 fails with it, T4 to T6 show
  `SKIP`, T8 fails while this version is uncommitted, and for Nova Pro T9 fails (Nova Lite
  has not been launched). Exit 1; "nothing was started". As expected.
- `venv\Scripts\python -m scripts.nova_tune status`: "no job has been launched", exit 0.

The estimates `plan` shows today are at the assumed ratio of 1.5, because the gate has no
measured one: 9,762,544 training tokens per model, Nova Lite $19.53, Nova Pro $78.10.

Differences from sections 3 and 4:

- **A job that already exists with our request token is adopted** (D5 said a repeat "asks
  for the same job"). Before creating, a launch looks the planned name up in Bedrock; if a
  job is there with this launch's token, the tuning record and the ledger record are written
  for it and nothing is created. T6 passes in that case and says so. A job of that name with
  another token fails T6.
- **A check that needs AWS and has no client is "skipped"**, a third outcome beside pass
  and fail. It never counts as a pass. `launch` requires both clients, so it skips nothing.
- **`status` with nothing launched needs no sign-in** and exits 0.
- **`bill` needs no sign-in**: it changes only the ledger and the tuning record, so it has
  no `--yes`. The `bill` record carries the amount and no token count.
- **The cap check prices a `bill` record by its amount** and every other record by its
  tokens at list price.
- **`teardown` writes the record after each deletion**, so a failure between the two
  leaves the truth: the deployment gone, the model still there.
- **`status` does not ask again about a job that has ended**, nor about a deleted
  deployment. The loss curves are fetched once.
- **The deployment's request token** is the job's token with `-deploy` appended.
- **botocore's own traceback for an expired sign-in is silenced** (`botocore.tokens` and
  `botocore.credentials` at ERROR, in `aws_session`); the operator sees one line with the
  `aws sso login` command. This also quiets `scripts.evaluate` and `scripts.nova_export`.
- Not built: nothing of section 4 was left out.

Still unverified, because only a real job can show it (section 7 has the rest): what
Bedrock answers for a job name that does not exist; where the two metrics files land under
the output location (the code looks for them by file name anywhere under it); whether a
real job's phases and times look as the definitions say.

### Live results (the owner's runs, 2026-11-19 to 2026-11-22)

Both jobs completed and both tuned models are deployed on-demand and Active. No Provisioned
Throughput was created. All times are UTC.

| | Nova Lite (B1-Lite) | Nova Pro (B1-Pro) |
|---|---|---|
| Job | `qlora-finqa-nova-lite-c2d67fe65c7f-a1` | `qlora-finqa-nova-pro-c2d67fe65c7f-a1` |
| Launch run | `20261119T021002Z-train-qlora-finqa-nova-lite-c2d67fe65c7f-a1-ca6d` | `20261121T095540Z-train-qlora-finqa-nova-pro-c2d67fe65c7f-a1-752d` |
| Launched | 2026-11-19 02:10 | 2026-11-21 09:55 |
| Completed | 2026-11-19 04:29 (2 h 19 min) | 2026-11-22 04:35 (18 h 40 min) |
| Tuned model | `arn:aws:bedrock:us-east-1:123456789012:custom-model/amazon.nova-lite-v1:0:300k/gmwz1fhs0vbt` | `arn:aws:bedrock:us-east-1:123456789012:custom-model/amazon.nova-pro-v1:0:300k/z86zmzl5wkz8` |
| Deployment | `arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/308jz60uh4a5`, Active | `arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/62v0vii1ge23`, Active |
| Training loss, first to last | 0.4394 to 0.0589 over 152 steps | 0.3580 to 0.0411 over 304 steps |
| Validation loss, epoch 1 to epoch 2 | 0.1083 to 0.1035 | 0.0794 to 0.0821 |
| Estimate | $12.82 (6,411,388 tokens) | $51.29 (6,411,388 tokens) |
| Billed | $14.60 (7,298.1 thousand tokens) | Not on the bill yet; about $58.38 in Nova Lite's proportion |

What the runs settled, of the things section 7 left open:

- **The bill is 1.138 times the estimate** for Nova Lite: Bedrock billed about 887,000 more
  training tokens than Qwen tokens times the measured ratio times the epochs. One job does
  not say whether that is the validation file, per-record overhead, or both. It is above
  the 10% margin of the gate; the Path B total moves from $82.46 to about $91 of the $200
  cap.
- **Bedrock answers -1 for both losses of a Nova job** in `get_model_customization_job`.
  The losses come from the two metrics files in the job's output, at
  `<output>/model-customization-job-<id>/training_artifacts/step_wise_training_metrics.csv`
  and `.../validation_artifacts/post_fine_tuning_validation/validation/validation_metrics.csv`.
  `BedrockTuning` now takes a loss Bedrock does not report from the last value of its curve.
- **Validation loss is reported once per epoch**, so each job has two values.
- **A job can wait many hours before training starts.** Nova Pro's validation phase ended
  ten minutes after the launch, and its training had not begun eleven hours later. The
  wait is not billed.
- **Storage is billed from the day the model exists**: $1.95 for Nova Lite's model-month
  was on the bill three days after its job.
- Nova Pro's validation loss rose slightly in its second epoch, where Nova Lite's fell.
  v4.04's dev runs say whether that matters.

Still open: Nova Pro's charge (entered with `nova_tune bill --model nova-pro` when it shows),
and everything about calling a tuned model, which is v4.04.
