# QLoRA Local

Spec Group      :   req_003_qlora_local  
Feature Name    :   qlora_local  
Version         :   4  
Status          :   ✅ Complete 2026-11-21: the training job and the scoring job completed and were pulled ($5.88 and $1.94); arm A1 is checkpoint 500, 358 of 499 right on dev (71.74%), verdict `met`; the adapter is kept in the bucket. M5 reached (section 12)  
Last updated    :   2026-11-21  

## 1. v3.04 — Full training run and checkpoint selection

Everything Path A needs exists: the pipeline trains (v3.02) and a job can run it (v3.03).
This version does the run that matters: the full fit on SageMaker, every checkpoint scored
on dev, one checkpoint chosen, and that choice written down as arm **A1**.

It adds:

- a **`score` task**: one job that scores the untuned model and every checkpoint of a
  training run on dev, through the ordinary evaluation runner, and can be resumed;
- an **estimate in the plan**, from the times the smoke job measured, so the owner sees
  hours and dollars before launching;
- a **selection**: `scripts/qlora_select.py` reads the scored runs, picks the checkpoint
  with the best dev execution accuracy, checks the result against thresholds, and writes a
  committed **selection record and report**;
- `--adapter selected`, so that later versions ask for arm A1 by name and never by a
  checkpoint typed by hand;
- keeping the selected adapter: pulled here, and copied to the bucket's `adapters/` folder.

Its two paid jobs are the largest spend of Path A: roughly $9 to $14 together at this
machine's speeds (section 2.3). **Both are started by the owner.** Reaching the selection
is milestone M5.

**Depends on**: v3.03. Its code is built; **its smoke job has not been run**. This version's
code can be built and rehearsed without it. Its jobs cannot be planned without it: the
estimate of D9 reads the smoke job's measurements.

## 2. What exists today (verified 2026-11-20)

### 2.1 What earlier versions provide

| Item | State |
|---|---|
| `scripts.sagemaker_job launch train` | Runs `run_training()` in a job: 4,884 records, 3 epochs, 918 optimizer steps, a checkpoint every 100 steps and at 918 (ten in all), the loss on dev at each. `--resume RUN_ID --workspace NAME --attempt 2` continues a run in a new job on the same workspace |
| `scripts.sagemaker_job launch evaluate` | One evaluation run per job: `--predictor qwen3-4b-tuned --adapter RUN_ID/checkpoint-N --split dev --workspace NAME`. Each launch pays about 8 minutes of setup |
| The workspace | Everything a job writes under `artifacts/` and `ledger/` is in S3, and a later job on the same workspace starts with it in place |
| `pull` | Brings run folders and adapters here, checkpoints only on request and then all of them; appends ledger lines once; records the job's billable seconds |
| `build_tuned_predictor(adapter, context)` | Names an adapter as `RUN_ID` or `RUN_ID/checkpoint-N`; refuses one trained for another model commit, prompt, or split. `details` name the training run, checkpoint, and step; `settings["adapter"]` is the hash of the weights |
| `run_evaluation` | Each complete run of a split is a result and adds a line to `ledger/evaluations.jsonl` with its arm. **Ten checkpoints scored on dev are ten lines with arm A1** |
| `config/sagemaker.yaml` | Tasks `smoke`, `train` (at most 12 hours), `evaluate` (at most 4); `ml.g5.xlarge` at $1.408 an hour; Path A cap $100; nothing spent yet |
| `config/training.yaml` | 3 epochs; no selection rule and no threshold |
| `config/aws.yaml` | The bucket has an `adapters/` prefix "for selected LoRA adapters". Nothing is in it |
| Plan and roadmap | Select "the checkpoint with the best dev execution accuracy, not the lowest loss". "Decision to lock: acceptance thresholds on dev that decide whether v3.05 fires" |
| Path B's thresholds (v4.04) | A tuned Nova model has "missed" under 10 points above its own untuned score on dev, or under 98% format validity |
| Dev baselines | Untuned Qwen here, batch size 1: 3 of 499 (0.60%). Untuned Nova Pro 190 of 499 (38.08%); Nova Lite 123 (24.65%) |

### 2.2 Measured on this machine (v3.01 to v3.03)

| Quantity | Value |
|---|---|
| Training | 0.88 to 0.96 s per sequence: 14.0 to 15.3 s per optimizer step |
| Dev loss at a checkpoint (16 records) | 3.5 s: about 0.2 s per record |
| A tuned reply, batch size 1 | 5.0 s (8 to 14 generated tokens per second) |
| An untuned reply, batch size 8 against 1 | 1.28 s against 4.14 s: 3.2 times faster |
| The smoke adapter (12 steps on 64 records), first 24 of quick-dev | 10 right on execution, 24 in the format; untuned 0 and 22 |
| Loading the model for a predictor | 6 to 8 s |

A tuned reply at batch size 8 has not been timed on its own. Taking the untuned ratio, it
is about 1.6 s.

### 2.3 What the two jobs would cost, at this machine's speeds

The cloud GPU is an A10G, a different card. **These are placeholders**: the smoke job
measures seconds per training sequence and per reply on it, and the plan of D9 computes
from those.

| Job | Work | Hours here | If the A10G is 1.5 times slower | Cost at $1.408 |
|---|---|---:|---:|---:|
| Training | 918 steps, ten dev-loss passes over 499 records, setup | 4.0 | 5.9 | $5.60 to $8.30 |
| Scoring | The untuned model and ten checkpoints on dev: 11 runs of 499 replies, setup | 2.4 | 3.5 | $3.40 to $4.90 |
| **Both** | | **6.4** | **9.4** | **$9 to $14** |

At batch size 1 the scoring job alone would be about 7.5 hours and $10.50.

### 2.4 What is not known

1. Everything the smoke job settles (v3.03, section 2.5): whether a job runs at all, and
   how fast an A10G trains and generates.
2. How good the tuned model will be. The only evidence is the smoke adapter's 10 of 24.
3. Whether accuracy peaks before the last checkpoint.

## 3. Locked decisions

### The training run

| # | Decision | Reason |
|---|---|---|
| D1 | The full run is `launch train --scope measured`, with the settings of `config/training.yaml` as they are: nothing is tuned before a first result | Plan §4.6: defaults first. Its billable time is Path A's training cost in the comparison, so it is a measured run, from a clean commit |
| D2 | The run is made once. A job that stops (maximum runtime, a failure) is continued with `--resume` on the same workspace, not started again | The checkpoints are in the workspace, and every billed second is recorded either way |
| D3 | The training job's maximum runtime is set from the estimate of D9 before launch: the estimate plus half again, and never under 2 hours | 12 hours is a guess that allows $17. The limit should be near what the run needs |

### Scoring

| # | Decision | Reason |
|---|---|---|
| D4 | A new task, **`score`**, runs in one job: the untuned model on the split, then every checkpoint of a training run, each as its own evaluation run through `run_evaluation`. Default split `dev`, scope `measured` | Eleven separate jobs would each pay 8 minutes of setup and need a command from the owner. One job, one estimate, one launch |
| D5 | **Every checkpoint is scored on all of dev.** No shortlist from quick-dev. **Decision for the owner in section 8** | The plan says the best dev execution accuracy. A shortlist would save about $1 and choose on 200 examples from 15 companies |
| D6 | `score` skips a checkpoint that already has a complete result for the split in the workspace, and resumes one whose run was cut short | A scoring job that stops is launched again on the same workspace and loses at most one batch |
| D7 | Scoring uses the batch size of `config/local_qlora.yaml` (8) for the untuned model and every checkpoint alike | One setting for both Qwen arms (v3.03 D28). The selection compares like with like |
| D8 | The ten checkpoint runs are recorded with arm A1 and are told apart by their predictor's `details` (training run, checkpoint, step). **Which of them is arm A1 is said by the selection record alone** | The registry has no arm for a candidate. Nothing downstream may read an A1 result by its arm; it reads the record (D13) |

### The estimate

| # | Decision | Reason |
|---|---|---|
| D9 | `plan train` and `plan score` print an estimate: the work in sequences and replies times the seconds measured by the last pulled job of this project on the same instance type, plus the setup time that job took; then the cost at the price. The maximum cost is printed beside it, as now | "The user starts the run after seeing the time and cost estimate from the smoke job" (roadmap) |
| D10 | A new check, **J9, Estimated**: a measurement exists, and the estimate is within the task's maximum runtime. Without a pulled job to measure from, J9 fails | A run whose estimate exceeds its limit would be cut and resumed for nothing. No measurement means the smoke job has not been run |

### The selection

| # | Decision | Reason |
|---|---|---|
| D11 | The selected checkpoint is the one with the most examples right on dev by **execution accuracy**, over all 499. Ties go to the higher format validity, then to the earlier step | Plan §4.6. An earlier step has seen the data fewer times |
| D12 | Selection runs here, not in a job: `scripts/qlora_select.py` reads the pulled runs and recomputes every count from the predictions files with `summarize_run` | It costs nothing, can be run again, and a changed allowance needs no model |
| D13 | It writes `reports/local_qlora/adapter_selection.json` and `.md`, committed: the training run, one row per checkpoint (step, the four counts, the tight count, dev loss, the evaluation run), the selected checkpoint and the hash of its weights, the untuned model's row, the verdict, and what the two jobs were billed. Counts, ids, and amounts only | The one place that says what arm A1 is. v5.01 freezes it |
| D14 | Checks L1 to L6 (section 4.4) must pass before anything is written | A selection over nine of ten checkpoints, or over runs made with different settings, is not a selection |
| D15 | **The verdict.** A1 has "missed" if the selected checkpoint's execution accuracy on dev is under `selection.min_execution_accuracy_percent`, or its format validity is under `selection.min_format_validity_percent`. Otherwise it is "met". **Decision for the owner in section 8** | The roadmap's decision to lock. The second threshold is Path B's |
| D16 | A missed verdict starts nothing. It is the trigger of v3.05, which the owner decides on | As v4.04 |
| D17 | The report also says, without it changing the verdict: whether the selected checkpoint is the last one (the model may still have been improving), where the dev loss was lowest, and how far A1 is above the untuned model | What v3.05 or a longer run would be decided from |

### After the selection

| # | Decision | Reason |
|---|---|---|
| D18 | `--adapter selected` means the checkpoint of the selection record, in `scripts.evaluate` and in `sagemaker_job launch evaluate` (which sends the job the resolved name). It is refused when no record exists, or when the record's training run was made for another prompt, split, or model commit | As Path B takes a deployment from the tuning record and never from a pasted ARN |
| D19 | `pull --checkpoint N` downloads one checkpoint's adapter. `qlora_select` says which to pull | Pulling all ten is 1.4 GB for no use |
| D20 | `sagemaker_job keep --yes` copies the selected adapter and its metadata to `s3://<bucket>/adapters/<training run>/checkpoint-N/`, with checksums. **It is run by the owner** | A workspace is working storage and can be deleted; `adapters/` is where a chosen adapter stays (v0.05 D2) |
| D21 | Peak GPU memory is recorded in each run's measurements and in the selection report, not in the ledger | The ledger has no such field, and the roadmap's wording does not justify a schema change |

## 4. Scope to build

### 4.1 Config

`config/sagemaker.yaml`: a fourth task, and what the estimate needs.

```yaml
tasks:
  score:
    max_runtime_seconds: 21600
    phase: eval
    arm: A1
estimate:
  runtime_margin: 1.5          # D3: the limit proposed for a job is its estimate times this
  minimum_runtime_seconds: 7200
```

`config/training.yaml`: the selection rule.

```yaml
selection:
  split: dev
  min_execution_accuracy_percent: 40    # section 8, question 2
  min_format_validity_percent: 98
```

### 4.2 The `score` task: `qlora_finetuning/local_qlora/job.py`, `scripts/job_run.py`

```powershell
venv\Scripts\python -m scripts.job_run score --run <training run id> [--split dev] [--limit N]
```

It lists the run's checkpoints, scores the untuned model, then each checkpoint in step
order (D4, D6), and writes into `job_result.json` one entry per run with the step, the
hash of the checkpoint's weights, and the counts. `--limit` makes partial runs, for a
rehearsal. It runs here as it does in a job.

### 4.3 The launcher: `qlora_finetuning/local_qlora/sagemaker.py`, `scripts/sagemaker_job.py`

| Item | Behavior |
|---|---|
| `estimate(task, options)` | D9: the work, the measured seconds it used, the hours, the cost. Reads `job_result.json` of the last pulled job on the same instance type |
| Check J9 | D10 |
| `launch score --run RUN_ID --workspace NAME` | The scoring job, on the training job's workspace |
| `--adapter selected` | D18 |
| `pull --checkpoint N` | D19 |
| `keep --yes` | D20. The owner's |
| `--max-runtime SECONDS` on `launch` | D3: the owner may set a job's limit from the plan's proposal; the config value is the default |

### 4.4 Selection: `qlora_finetuning/local_qlora/selection.py`, `scripts/qlora_select.py`

```powershell
venv\Scripts\python -m scripts.qlora_select --run <training run id>
venv\Scripts\python -m scripts.qlora_select --run <training run id> --check     # writes nothing
```

| Id | Check | Passes when |
|---|---|---|
| L1 | Every checkpoint scored | Each checkpoint of the training run's plan has exactly one complete result on the split; the untuned model has one too |
| L2 | Same question | All those runs have the current prompt id and split manifest, the same reply limit, and the same settings except the adapter |
| L3 | The right weights | Each run's adapter hash is the hash the scoring job recorded for that checkpoint, and no two checkpoints share one |
| L4 | The run it claims | The training run completed all its planned steps, and its key (model commit, prompt, split, settings) is the current one |
| L5 | Measured in the cloud | Every run is `where: sagemaker` and `scope: measured`, and the jobs that made them are pulled, with billable seconds in the ledger |
| L6 | Scores current | Every run passes R1 to R5 of the evaluation runner now |

Exit codes: 0 selected, and the verdict is "met"; 1 a check failed, or the verdict is
"missed"; 2 a config problem or an unknown run. `read_selection()` is how other code reads
the record.

### 4.5 Tests

No test needs the network, credentials, a GPU, or a paid call.

- `score`, with stand-in training folders and evaluation: checkpoints in step order after
  the untuned model; a checkpoint with a complete result is skipped and one with a partial
  run is resumed; an unknown run, or one without checkpoints, is a config error; the
  result lists every run with its step and hash; `--limit` passes through.
- Estimate: from a stand-in job result, the hours and the cost for `train` and `score`, in
  `Decimal`; no measurement, or one from another instance type, fails J9; an estimate over
  the limit fails J9; the proposed limit is the estimate with its margin.
- Launcher: the `score` job's definition and name; `--adapter selected` resolved before
  the job is defined, and refused without a record; `pull --checkpoint` downloads one
  checkpoint's adapter files and nothing else; `keep` needs `--yes`, uploads with
  checksums, and refuses a selection that is not there.
- Selection, on invented runs: the best execution accuracy wins; each tie-break; each of
  L1 to L6 passing and failing; a verdict at, above, and below each threshold; the notes
  of D17; nothing written when a check fails or with `--check`; the same input gives the
  same files; the files hold no reply and no example text.
- `read_selection` and the selected predictor: the record's checkpoint is what is loaded;
  a record for another prompt is refused.
- Guards as before: no boto3 outside the launcher and staging; nothing uploads a test row.

### 4.6 Documentation

- Runbook section 5: the full run step by step with what each step costs, resuming a
  stopped job, scoring, selecting, pulling and keeping the adapter, what to delete after.
- `administration/aws_commands_run.md`: a run sheet for the two jobs, as for Path B.
- `sagemaker.md`, `gotchas.md`: what the two jobs measured.
- `evaluation_run.md`: several results can carry arm A1; the selection record says which.
- `coding.md`: ask for arm A1 as `--adapter selected`; choose a checkpoint only through
  `qlora_select`, by dev execution accuracy, never by loss and never by hand.
- Architecture overview; the execution plan's cost table with the billed figures.

## 5. Out of scope

- Any change to a hyperparameter, the prompt, or the data (v3.05, if triggered).
- A second training run, a longer one, or a sweep (v3.05; req_007).
- The test split (v5.01), confidence intervals and paired tests (v5.02), error categories
  (v3.05, v5.03).
- Scoring on `fit`, or any use of the dev loss to choose.
- Merging the adapter, or serving it from an endpoint (req_007).
- The optional local full run (v3.06). The `score` task and `qlora_select` work on a local
  run too, which is all v3.06 needs from here.

## 6. What this hands to later versions

| For | What |
|---|---|
| v3.05 | The verdict and its trigger; ten checkpoints' predictions on dev to categorize failures from; the notes of D17 |
| v3.06 | `job_run score` and `qlora_select` on a local training run |
| v5.01 | The selection record to freeze; `--adapter selected` for arm A1 on test; the workspace and the kept adapter in `adapters/` |
| v5.02 | Billable seconds of the training job and the scoring job; measured seconds per training sequence and per reply on the instance |
| v5.03 | The per-checkpoint table and the dev loss curve |
| v6.01, v6.02 | The selected adapter here, with its metadata |

## 7. Known limits and follow-ups

- **The A1 figure on dev is the best of ten on dev.** Choosing and scoring on the same
  examples flatters it a little. The test split, which no checkpoint has seen, is what the
  comparison reports.
- **Dev is 499 examples from 15 companies.** Two checkpoints a few examples apart are not
  really different, and the rule still has to pick one.
- **Replies depend on the batch** (v3.01). Every checkpoint is scored the same way, so the
  ranking is fair, but another batch size could order close checkpoints differently.
- **Two training runs with the same seed differ** (v3.02). The selected adapter is
  identified by the hash of its weights; a rerun would not reproduce it.
- **Section 2.3 is not an estimate of the cloud.** If the smoke job shows the A10G to be
  much slower, the costs rise in proportion; the cap check and J9 stand in the way of a
  surprise.
- **Ten results with arm A1 are in the results file** (D8). Anything that reads it by arm
  alone will be wrong.
- **A scoring job that is stopped repeats at most one batch**, and its billable time is
  recorded each time.
- **The thresholds are judgment.** No published figure applies to this split and prompt.

## 8. Decisions for the owner

**Taken as recommended on 2026-11-20**: the owner started the PRP and its execution
without changing questions 1 to 3. On question 4 the build went ahead before the smoke
job; section 11 says what that leaves open. The table is kept as the record of what was
asked.

| # | Question | Recommendation |
|---|---|---|
| 1 | **Score all ten checkpoints on all of dev (D5), or shortlist on quick-dev first?** | All of dev. The shortlist saves about $1 of a $9 to $14 total and chooses the finalists on 200 examples. The plan's rule is dev execution accuracy; this applies it as written |
| 2 | **The thresholds of D15**: what dev result makes A1 "missed" and offers v3.05? | Execution accuracy under 40%, or format validity under 98%. 40% is about what untuned Nova Pro scores (38.08%): a tuned 4B model that cannot pass an untuned large one has something wrong worth one corrective look. The smoke adapter reached 10 of 24 after 12 steps, so a sound run should clear this easily. 98% is Path B's figure. A stricter bar, say 60%, would be a hope about the outcome, not a test for a fault |
| 3 | **One job for training and a second for scoring (D4), or one job that does both?** | Two. You see the loss curve and the training bill before paying for scoring, a failure in one does not cost the other, and `score` can be run again alone after a change of the evaluation allowances. The price is a second 8-minute setup, about $0.19 |
| 4 | **Run the v3.03 smoke job before this version is built, or build now?** | Run the smoke job first if you can: it costs $0.60, it may show something the launcher has to change, and this version's plan cannot produce an estimate without it. Building first is possible, since everything here except the two jobs can be tested and rehearsed locally, but a fault found by the smoke job would then be fixed under two versions at once |

## 9. Acceptance

Code, by Claude:

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
  and without the training stack blocked.
- On this machine, free: `job_run score --run <a local smoke run> --split quick_dev
  --limit 24` scores the untuned model and that run's three checkpoints; run again, it
  skips them all; `qlora_select --check` on those runs reports L1 to L6 (L5 fails, as it
  must for local runs) and writes nothing.
- `sagemaker_job plan train`, with no pulled job, fails J9 and says to run the smoke job.

Live, by the owner (🟡), each command logged in `administration/aws_commands_run.md`. The
v3.03 smoke job is pulled first.

```powershell
aws sso login --profile qlora
venv\Scripts\python -m scripts.sagemaker_job plan train --scope measured      # the estimate; J1 to J9
venv\Scripts\python -m scripts.sagemaker_job launch train --scope measured --max-runtime <proposed> --yes
venv\Scripts\python -m scripts.sagemaker_job status                            # hours later
venv\Scripts\python -m scripts.sagemaker_job pull
venv\Scripts\python -m scripts.sagemaker_job plan score --run <run> --workspace <job> --scope measured
venv\Scripts\python -m scripts.sagemaker_job launch score --run <run> --workspace <job> --scope measured --yes
venv\Scripts\python -m scripts.sagemaker_job pull
venv\Scripts\python -m scripts.qlora_select --run <run>
venv\Scripts\python -m scripts.sagemaker_job pull --job <training job> --checkpoint <N>
venv\Scripts\python -m scripts.sagemaker_job keep --yes
```

- The training job ends `Completed` at step 918 with ten checkpoints; checks K1 to K5 pass
  in the job; the ledger has its billable seconds.
- The scoring job ends `Completed` with eleven results on dev.
- `qlora_select` passes L1 to L6 and writes the selection record and report; the record
  is committed.
- `scripts.evaluate --predictor qwen3-4b-tuned --adapter selected --split quick_dev
  --limit 24` runs here with the pulled adapter.
- The adapter is in `adapters/` in the bucket with the recorded hash.
- Section 11, "As built", is added: the per-checkpoint table, the selected checkpoint,
  A1 beside A0 and the two untuned Nova models on dev, the verdict, what each job took
  and was billed. Milestone M5 is recorded, and the roadmap says whether v3.05 is offered.

## 10. Next step

The version is complete (2026-11-21): section 12 has the figures, and M5 is reached. The
verdict is `met`, so v3.05 is its error analysis only. Path A then waits for v5.01.

## 11. As built (2026-11-20): the code

Built by `prp_003_full_run_selection_v4.md`. **Nothing was created in AWS and no job was
launched**: the sign-in was expired throughout, and the v3.03 smoke job has still not been
run. This section covers the code. The per-checkpoint table, the selected checkpoint, A1
beside A0 and the Nova models, the verdict, and the bills are added after the owner's jobs.

What was run:

- `venv\Scripts\python -m pytest -q`: 2,343 passed (77 new). With the training stack
  blocked: 2,337 passed, 6 skipped. `pytest -m gpu -q`: 8 passed.
  `venv\Scripts\python -m ruff check .`: clean.
- `scripts.job_run score --run 20261119T083236Z-smoke-qwen3-4b-smoke-5477 --split quick_dev
  --limit 24`, twice, on this machine's GPU.
- `scripts.qlora_select` on that run, with `--split quick_dev --limit 24 --check`, and
  once without `--check`.
- `scripts.sagemaker_job plan train --scope measured`, signed out.

### The rehearsal on this machine

| Model | Right on execution, of 24 | In the format | Loss on dev | Run |
|---|---:|---:|---:|---|
| Untuned | 0 | 22 | | `20261120T062707Z-eval-qwen3-4b-quick-dev-3c9c` |
| Checkpoint 4 | 9 | 24 | 0.3295 | `20261120T062742Z-eval-qwen3-4b-tuned-quick-dev-c5b0` |
| Checkpoint 8 | 10 | 24 | 0.2414 | `20261120T062824Z-eval-qwen3-4b-tuned-quick-dev-56a3` |
| Checkpoint 12 | 10 | 24 | 0.2357 | `20261120T062853Z-eval-qwen3-4b-tuned-quick-dev-a9ab` |

- The second `score` made no run and loaded no model: all four were found and reused.
- `qlora_select --check` passed L1 to L4 and L6, failed L5 (the runs are local and `dev`
  scope, as they must be), showed checkpoint 8 as best, and wrote nothing. Checkpoints 8
  and 12 tie on accuracy and the earlier step wins; the loss was lowest at 12, so on this
  small sample the loss and the accuracy already disagree about which to take.
- Without `--check` it refused: a smoke run is never arm A1.
- `plan train` failed J1 (signed out), J2 (uncommitted changes), and J9 (no pulled job has
  measured anything), skipped J4 to J6, and created nothing. The job as it stands: at most
  720 minutes and $16.90; Path A at $0.00 of its $100 cap.

These figures are 24 examples of a 12-step smoke adapter. They show that the tooling
works. They say nothing about arm A1.

### The decisions of section 8

Taken as recommended, with one exception that the order of events decided: question 4
recommended running the smoke job before building, and the build went first. The cost of
that is stated in "Still to come".

### Differences from sections 3 and 4

- **A smoke run can only be rehearsed.** `qlora_select` on a smoke run without `--check`
  is a config error, whatever its scores.
- **`--adapter selected` also checks the weights.** It is refused when the adapter on this
  machine does not have the hash the selection record names, not only when the record was
  made for another prompt, split, or model commit.
- **L3 falls back on the job's own record.** When a checkpoint's weights are not on this
  machine (the usual case: checkpoints stay in S3), the hash is the one the scoring job
  wrote in its result file.
- **Only the untuned run made the way the checkpoints were is counted.** An earlier
  baseline at another batch size is a different question and is ignored by L1.
- **A job that continues another's work must name its workspace.** `score`, a resumed
  `train`, and a tuned `evaluate` are refused without `--workspace`, at `plan` already.
- **The smoke job passes J9 without an estimate**: it is the job that measures.
- The look-up of existing runs and the estimate have modules of their own
  (`local_qlora/scoring.py`, `estimate.py`). `keep` notes what it stored under `kept` in
  the job record.

## 12. As run (2026-11-20 and 2026-11-21): the owner's jobs

The version is complete. **Arm A1 is checkpoint 500 of training run
`20261120T182710Z-train-qwen3-4b-fit-4557`: 358 of 499 right on dev by execution accuracy,
71.74%. The verdict is `met`. Milestone M5 is reached.** The committed record is
`reports/local_qlora/adapter_selection.json`, with a readable page beside it.

### The two jobs

| Job | What it did | Estimate | Billed |
|---|---|---|---|
| `qlora-train-958bed10-a1` | 918 steps, three epochs over the 4,884 `fit` examples, ten checkpoints; checks failed: none | 4.7 hours, $6.59 | 15,037 seconds (4 hours 11 minutes), **$5.88** |
| `qlora-score-63b89247-a1` | The untuned model and the ten checkpoints on the 499 `dev` examples: eleven results | 2.9 hours, $4.09 | 4,965 seconds (1 hour 23 minutes), **$1.94** |

Both ended `Completed` on their first attempt and were pulled. The two cost $7.82; Path A
has cost $8.24 of its $100 cap, the two smoke jobs included.

The scoring estimate was more than twice too high. The smoke job measured 1.91 s per tuned
reply on 24 examples; over 499 examples a checkpoint took about 6 minutes 45 seconds, 0.8 s
per reply. The training estimate was close: about 15 s per optimizer step, as the smoke
job's 1.115 s per sequence predicts.

### The checkpoints on dev

| Checkpoint | Right on execution | In the format | Answer line right | Program match | Dev loss |
|---:|---:|---:|---:|---:|---:|
| untuned | 6 (1.20%) | 457 | 60 | 2 | |
| 100 | 321 (64.33%) | 499 | 287 | 291 | 0.1643 |
| 200 | 322 (64.53%) | 498 | 290 | 299 | 0.1531 |
| 300 | 338 (67.74%) | 499 | 317 | 308 | 0.1414 |
| 400 | 350 (70.14%) | 499 | 321 | 320 | 0.1397 |
| **500 (selected)** | **358 (71.74%)** | 499 | 338 | 333 | 0.1321 |
| 600 | 356 (71.34%) | 499 | 341 | 328 | 0.1286 |
| 700 | 356 (71.34%) | 499 | 340 | 326 | 0.1330 |
| 800 | 358 (71.74%) | 499 | 343 | 329 | 0.1352 |
| 900 | 356 (71.34%) | 499 | 342 | 327 | 0.1351 |
| 918 | 356 (71.34%) | 499 | 342 | 329 | 0.1348 |

- Checkpoints 500 and 800 tie at 358 with every reply in the format; the rule (D11) takes
  the earlier step.
- From step 500 on, every checkpoint is within two examples of the best. The third epoch
  changed nothing that dev can measure.
- The dev loss was lowest at step 600, not at the selected step. It is shown and never used.
- `qlora_select` passed L1 to L6: 12 runs in 2 pulled jobs.

### A1 beside the untuned models, on dev (499 examples)

| Arm | Right on execution | In the format |
|---|---:|---:|
| A0, Qwen3-4B untuned (this scoring job, batch size 8) | 6 (1.20%) | 457 (91.58%) |
| B0-Lite, Nova Lite untuned (v4.01) | 123 (24.6%) | 466 (93.4%) |
| B0-Pro, Nova Pro untuned (v4.01) | 190 (38.1%) | 497 (99.6%) |
| **A1, Qwen3-4B tuned, checkpoint 500** | **358 (71.74%)** | **499 (100.00%)** |

A1 is 70.54 points above A0. These are development figures: A1's is the best of ten
checkpoints on the examples it was chosen on, so it flatters a little. The comparison
reports the test split (v5.01). A0 was 3 of 499 on this machine at batch size 1 (v3.01).

### The verdict

`met`: execution accuracy 71.74% against the 40% asked, format validity 100.00% against
98%. v3.05's corrective iteration is therefore not offered; its error analysis is run
all the same.

### Section 9's points

| Point | Result |
|---|---|
| The training job ends `Completed` at step 918 with ten checkpoints; billable seconds in the ledger | Yes |
| The scoring job ends `Completed` with eleven results on dev | Yes |
| `qlora_select` passes L1 to L6; the record is committed | Yes (commit `bf5d11e`) |
| `scripts.evaluate --predictor qwen3-4b-tuned --adapter selected --split quick_dev --limit 24` runs here with the pulled adapter | Yes: run `20261121T083447Z-eval-qwen3-4b-tuned-quick-dev-6117`, completed, with the adapter of hash `a8c9561b4fc6...` |
| The adapter is in `adapters/` in the bucket with the recorded hash | Yes: `adapters/20261120T182710Z-train-qwen3-4b-fit-4557/checkpoint-500/`, three objects; `adapter_model.safetensors` is 66,127,776 bytes and its SHA-256 in S3 is the record's `a8c9561b4fc6...902b` |

### What the jobs settled

- **The `score` task and a job on another job's workspace work in a container**: the
  scoring job found the training job's checkpoints in place and scored all ten.
- **The adapter is 66 MB**, not the 130 MB the documents guessed.
- Nothing needed correcting after the smoke job's fix of v3.03.
