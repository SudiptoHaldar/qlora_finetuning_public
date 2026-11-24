# Comparison

Spec Group      :   req_005_comparison  
Feature Name    :   comparison  
Version         :   1  
Status          :   ✅ Complete 2026-11-23: code built 2026-11-22 (`prp_005_test_unlock_six_arms_v1.md`); frozen, unlocked, and all six arms run on test by the owner 2026-11-23; six-arm record written, C1 to C8 pass (section 11); M7 reached  
Last updated    :   2026-11-22  

## 1. v5.01 — Test unlock and six-arm evaluation

This version reads the test split, once. Until now every number in the project came from
dev: prompts, checkpoints, and thresholds were chosen there. The 2,898 test examples have
never been shown to a model. This version asks all six arms those questions and stores the
answers. It is the measurement the project's comparison rests on.

It does four things, in this order:

1. **Freeze.** It writes a committed record of the configuration of both paths: which model
   each arm is, and every value that decides a reply or a score.
2. **Unlock.** The owner creates the unlock record, which names the freeze record. From then
   on a test run is possible, and only at the frozen configuration.
3. **Run.** Each arm answers all 2,898 test questions through the shared runner: A0 and A1
   as SageMaker evaluation jobs, the four Nova arms through Bedrock.
4. **Record.** A committed record names the six result runs and shows that they are
   complete, comparable, and made at the frozen configuration.

The six arms:

| Arm | Model | Predictor | Runs in |
|---|---|---|---|
| A0 | Qwen3-4B, untuned | `qwen3-4b` | SageMaker job |
| A1 | Qwen3-4B with the selected adapter (checkpoint 500) | `qwen3-4b-tuned --adapter selected` | SageMaker job |
| B0-Lite | Nova Lite, untuned | `nova-lite` | Bedrock |
| B1-Lite | Nova Lite, tuned (attempt 1) | `nova-lite-tuned` | Bedrock |
| B0-Pro | Nova Pro, untuned | `nova-pro` | Bedrock |
| B1-Pro | Nova Pro, tuned (attempt 1) | `nova-pro-tuned` | Bedrock |

All six passes are billed. **The owner unlocks and the owner starts every run.** The freeze,
the checks, and the final record call no AWS service.

**Depends on**: v3.04 and v4.04, both complete. Path A is frozen at checkpoint 500 (v3.05
ended as analysis only). Path B is frozen at attempt 1 of each model (v4.05 did not fire).

Reaching the end of this version, with six complete result runs, is milestone **M7**.

This version computes no statistics and no costs. It shows each arm's counts and rates and
stops there: the cost model is v5.02 and the report is v5.03.

## 2. What exists today (verified 2026-11-22)

### 2.1 The lock

| Item | State |
|---|---|
| `load_split("test")` | Raises `TestLockedError` before any data is read, until `splits/test_unlock.json` exists and its `manifest_hash` equals the split manifest's content hash. The file does not exist |
| The split | 8,281 examples: fit 4,884, dev 499, **test 2,898** (35.00%), 43 companies, none shared with training. Manifest hash `d34fb22e37b7...` |
| `scripts.evaluate --split test` | Reports `LOCKED` and exits 2. Once unlocked it would accept any predictor, any `--limit`, any number of times: the lock is the only guard today |
| A job's data | The bundle of `fit` and `dev`. `export_bundle` refuses `test`; in a job `load_split("test")` is refused whatever the unlock record says |
| S3 | No test row has been uploaded |
| Gold check (v2.02) | Already covered the test examples by count: flagged ids are in `splits/finqa_gold_flags.json` |
| Token budget (v1.04) | Counted fit and dev only. **The length of the test prompts is not known** |

### 2.2 What each arm is, and its dev result

Every run below is complete, counts as a result, and used prompt `d63742a62b76...` and
manifest `d34fb22e37b7...` on the same 499 dev examples.

| Arm | What decides it | Dev run | Execution accuracy on dev |
|---|---|---|---:|
| A0 | Model commit `1cfa9a720891...`, 4-bit, batch size 8 | `20261121T070030Z-eval-qwen3-4b-dev-2416` | 6 of 499, 1.20% |
| A1 | The selection record: `20261120T182710Z-train-qwen3-4b-fit-4557/checkpoint-500`, weights `a8c9561b4fc6...` | `20261121T073743Z-eval-qwen3-4b-tuned-dev-fd5b` | 358 of 499, 71.74% |
| B0-Lite | `amazon.nova-lite-v1:0`, temperature 0, top_k 1 | `20261116T202551Z-eval-nova-lite-dev-d37e` | 123 of 499, 24.65% |
| B1-Lite | The tuning record: latest attempt of `nova-lite`, its deployment | `20261122T103321Z-eval-nova-lite-tuned-dev-f97f` | 329 of 499, 65.93% |
| B0-Pro | `amazon.nova-pro-v1:0`, temperature 0, top_k 1 | `20261116T203032Z-eval-nova-pro-dev-d704` | 190 of 499, 38.08% |
| B1-Pro | The tuning record: latest attempt of `nova-pro`, its deployment | `20261122T114511Z-eval-nova-pro-tuned-dev-276f` | 359 of 499, 71.94% |

Both Qwen dev runs were made in SageMaker with scope `measured`. The four Nova dev runs
have scope `dev`. Both deployments are Active and were kept for this version (the owner's
decision, 2026-11-18).

### 2.3 Speeds and limits

| Item | Measured |
|---|---|
| Qwen on `ml.g5.xlarge`, batch size 8 | 1.24 s per untuned reply, 0.80 s per tuned reply, 369 s of job overhead (job `qlora-score-63b89247-a1`) |
| Nova prompts | 313,156 input tokens for the 499 dev questions: 627.6 a question |
| Nova replies | About 45 output tokens a question untuned, 31 tuned |
| Nova Pro, tuned and untuned | Paced at 18 requests a minute (`config/frontier.yaml`). Nova Lite is not paced |

### 2.4 What the six passes will cost

Computed from section 2.3 and `config/pricing.yaml`. The test prompts are assumed as long
as dev's, on average: that cannot be checked while the split is locked.

| Arm | Work | Time | Estimate | At most |
|---|---|---:|---:|---:|
| A0 | 2,898 replies at 1.24 s, plus overhead | 1.1 hours | $1.54 | $2.82 (a 7,200 s limit) |
| A1 | 2,898 replies at 0.80 s, plus overhead | 0.75 hours | $1.05 | $2.82 |
| B0-Lite | 1.82M input tokens, 0.13M output | about 30 minutes | $0.14 | |
| B1-Lite | 1.82M input, 0.09M output | about 30 minutes | $0.13 | |
| B0-Pro | 1.82M input, 0.13M output | 2 hours 41 minutes at 18 a minute | $1.87 | |
| B1-Pro | 1.82M input, 0.09M output | 2 hours 41 minutes | $1.75 | |
| **All six** | | | **about $6.50** | about $9.50 |

Spent so far: Path A $8.24 of its $100 cap. Path B about $67 of its $200 cap, counting Nova
Pro's training at its estimate ($51.29; its charge is not on the bill yet).

## 3. Locked decisions

### The freeze

| # | Decision | Reason |
|---|---|---|
| D1 | The frozen configuration of an arm is **its dev result run of section 2.2**. The freeze record copies from that run what the runner records and what a resume compares: predictor name, model id, and settings; and beside them the prompt id, the split manifest hash, and the hashes of the config files the run snapshotted that decide a reply or a score (`prompt`, `split`, `token_budget`, `dsl`, `evaluation`) and of the gold flags | A test run is then "the dev run, on other questions", and that can be checked field by field. Nothing new has to be described by hand |
| D2 | A1 is the checkpoint the selection record names, with its weights hash. B1 of each model is the latest attempt in the tuning record, with its job ARN, deployment ARN, and export id. The freeze record stores these beside the dev run | These are the two records the project already treats as the only source for a tuned model |
| D3 | The freeze is refused unless checks F1 to F8 pass (section 4.1) | A freeze of an inconsistent state would be signed and then unusable |
| D4 | The freeze record holds ids, hashes, counts, settings, and ARNs. No example text, no email address, no person's name | As every committed file |
| D5 | Frozen means values, not bytes. A config file may still gain keys that decide no reply (a price, a pacing rate, a report folder). What may not change is listed in the freeze record and checked by the gate (D15) and by a test that runs with the suite | `config/pricing.yaml` and `config/frontier.yaml` will change in v5.02 and v5.03 without touching a reply |
| D6 | Writing the freeze record changes nothing in AWS and costs nothing. It can be written again as long as the test split is locked. Once the unlock record exists, the freeze record is never rewritten | The plan asks that the freeze predates the unlock |

### The unlock

| # | Decision | Reason |
|---|---|---|
| D7 | **The owner unlocks**, with `comparison_freeze unlock --yes --freeze <first 12 characters of the freeze record's SHA-256>`. Typing the hash is the sign-off: it says which record was read and accepted | The plan asks for a signed-off record. A hash in a command is a signature that needs no name in a committed file |
| D8 | The unlock record `splits/test_unlock.json` holds the manifest hash (which is what `load_split` checks), the freeze record's SHA-256, the commit that contains the freeze record, and the date. The unlock is refused unless the freeze record is committed, the tree is clean, F1 to F8 still pass, and the hash given is the record's | "Predates" is then a fact of the git history, not a time stamp |
| D9 | There is no command to lock again. Unlocking is done once | A split that can be locked and unlocked is not locked |

### What "once" means

| # | Decision | Reason |
|---|---|---|
| D10 | Each arm gets **one complete result run** on test. A run that was interrupted (a throttle, a lost connection, a job at its time limit) is resumed, as on dev: the result is still one run over 2,898 questions | Nova Pro's dev run stopped twice at its rate limit. Forbidding a resume would make the measurement a matter of luck |
| D11 | `--limit` is refused on test. A trial before a billed run uses five questions of quick-dev | A partial test run is a look at the test split that is not the measurement |
| D12 | A second complete run of an arm is refused. The one exception: a run that is void because of a fault in the pipeline or the service, never because of what the model answered. The owner voids it with a written reason (`comparison_runs void`), the reason is committed in the six-arm record as a deviation, and then one more run is allowed | Things can break. A rerun must be visible in the report, not silent |
| D13 | No reply is looked at, and no run is compared with another, before all six are complete and the six-arm record is written. Failure categories and examples are v5.03's | The order of the runs must not be able to influence anything |
| D14 | One run per arm is the result. Bedrock does not promise identical replies for identical requests, and a Qwen reply can depend on its batch. Neither is averaged away | As on dev (v4.04 section 7, v3.01) |

### The gate on a test run

| # | Decision | Reason |
|---|---|---|
| D15 | Every way of starting a test run passes one gate, `check_test_run`, before a run exists. It refuses unless all of these hold: the split is unlocked by a record that names a committed freeze record with that hash; the arm is one of the six and its predictor name, model id, and settings equal the frozen ones; the shared values of D1 equal the frozen ones; the scope is `measured` (so the tree is clean); no `--limit`; A0 and A1 run inside a SageMaker job, the Nova arms against Bedrock; no complete, unvoided test result exists for the arm | After the unlock the lock no longer protects anything. The gate is what keeps "no configuration changes afterwards" true |
| D16 | The `gold` predictor may run on test, locally, after the unlock. It is not an arm and has no settings to freeze | It reads gold replies only, says nothing about any model, and proves the runner end to end on the 2,898 questions before a billed call |
| D17 | A Qwen model is never asked a test question on this machine. `rehearse` refuses `--split test` | Working agreement 11: reported numbers come from the cloud. A local test run would also be a second run |
| D18 | Code that trains, exports, selects, or counts keeps refusing test rows after the unlock: `run_training`, `export_files`, the `score` task, `qlora_select`, `finqa_tokens`, `finqa_serialize`. A guard test lists the modules that may ask for `test` | The unlock opens the test split to evaluation, not to everything |

### The runs

| # | Decision | Reason |
|---|---|---|
| D19 | A0 and A1 are **two `evaluate` jobs**, one per arm | Each job's billable seconds are then recorded under its own arm, which is what v5.02 prices. One job for both would save one overhead, about $0.14 |
| D20 | A test job is given a **test bundle**: the 2,898 test examples and nothing else, written by `sagemaker_job bundle --split test` after the unlock and uploaded by the owner's launch. This is the only upload of test rows to S3. In a job, `load_split("test")` is served only from a bundle that names `test`, and only when the unlock record inside the job's code names the manifest | The job needs the rows. Keeping them apart from `fit` and `dev` means no training job can ever be handed them |
| D21 | A1's job is `--adapter selected` on the workspace of the training job, `qlora-train-958bed10-a1` | The checkpoint is there, with the hash the selection record names |
| D22 | A new launch check, J11, applies the gate of D15 to a test job at plan time, and also refuses when an earlier test job for the arm is unpulled or holds a complete result | A job's run is not in the local ledger until it is pulled, so the local gate alone cannot see it |
| D23 | The Nova runs are `scripts.evaluate --split test --scope measured`, unchanged otherwise. Nova Pro stays paced at 18 requests a minute | Nothing but the questions differs from the dev runs |
| D24 | The order is free. Recommended: the gold check, then the two Nova Lite runs, then the two Nova Pro runs, with the two SageMaker jobs launched one after the other while the Nova runs go | The cheap and fast runs show a fault first. Bedrock and SageMaker do not share a limit |

### The record

| # | Decision | Reason |
|---|---|---|
| D25 | The six-arm record is written by `comparison_runs record`, from the ledger, the run folders, the job record, the freeze record, and the unlock record. It calls no AWS service. It is written only when checks C1 to C8 pass (section 4.5); with fewer than six arms it prints what is missing and writes nothing | A partial record would invite reading partial results |
| D26 | The record holds, per arm: the run id, the counts and rates of every metric over all examples and over the unflagged ones, tokens, latency, stop reasons, and for a Qwen arm the job name and billable seconds. No confidence interval, no test between arms, no dollar amount, no example text | Statistics are v5.03's and costs are v5.02's. Counts are what both are computed from |
| D27 | The six runs it names are the only test runs later versions read. v5.02 and v5.03 take run ids from this record and nowhere else | Dozens of runs carry arm A1 today. An arm name does not identify a run |

## 4. Scope to build

New code lives in `qlora_finetuning/comparison/`: `freeze.py` (the freeze record, checks F1
to F8, the unlock), `gate.py` (`check_test_run`), and `runs.py` (the six-arm record, checks
C1 to C8, voiding). Two scripts: `scripts/comparison_freeze.py` and
`scripts/comparison_runs.py`. A script name does not start with `test_`, so that pytest
never collects one.

### 4.1 The freeze

```powershell
venv\Scripts\python -m scripts.comparison_freeze check     # F1 to F8 and the estimate; writes nothing
venv\Scripts\python -m scripts.comparison_freeze freeze    # writes the record
venv\Scripts\python -m scripts.comparison_freeze status    # frozen? unlocked? does it still hold?
```

`freeze` writes `reports/comparison/config_freeze.json` and `.md` (committed) and prints
the record's SHA-256. It needs no sign-in.

| Id | Check | Passes when |
|---|---|---|
| F1 | Path A selected | The selection record exists, its checks passed, and its adapter is the one `--adapter selected` resolves to for the current model commit, prompt, and manifest. No corrective iteration is registered and unfinished |
| F2 | Path B tuned | For both models the tuning record's latest attempt shows job `Completed` and deployment `Active`, tuned on the export of the current gate record |
| F3 | Six dev results | Each arm has exactly one dev run to freeze (A0 and A1 from the selection record's table, B1 from the tuned report, B0 from the baseline report). Each is complete and counts as a result |
| F4 | One contract | The six runs agree on prompt id, manifest hash, reply limit, example count and order, and on the hashes of the snapshotted `dsl` and `evaluation` config |
| F5 | Nothing changed since | The current config values, prompt id, manifest hash, and gold flags equal what those runs recorded |
| F6 | Still locked | No unlock record exists and no run in the registry has split `test` |
| F7 | Committed | The working tree is clean |
| F8 | Within the caps | The estimate of section 2.4, computed again from the job record and the ledger, fits in what is left of each path's cap |

### 4.2 The unlock

```powershell
venv\Scripts\python -m scripts.comparison_freeze unlock --yes --freeze <12 characters>
```

The owner's. It runs the checks of D8, prints the estimate once more, writes
`splits/test_unlock.json`, and reminds the owner to commit it: a SageMaker job runs a
commit, so the unlock record must be in it. Without `--yes` or without `--freeze` it prints
what it would do and exits 2.

After the unlock, free and local:

```powershell
venv\Scripts\python -m scripts.evaluate --predictor gold --split test
venv\Scripts\python -m scripts.finqa_split
```

The gold run must score 2,898 of 2,898 and pass R1 to R5. The split report is written again,
because its check P10 now reads "unlocked". `comparison_runs preflight` wraps the gold run
and also reports, as counts only, the number of test prompts and the length of the longest
one in Qwen tokens. A prompt over the training length of 2,048 is reported and not refused:
inference has no such limit, but the adapter never saw one that long. That would be a
caveat for the report.

### 4.3 The gate

`check_test_run` (D15) is called by `scripts.evaluate`, by the `evaluate` task of
`scripts.job_run`, and by the launcher's check J11. It returns the reason for a refusal as
a sentence that names the next step. `scripts.evaluate` prints it as `REFUSED` and exits 2.

On dev, quick-dev, and fit the gate is not called and nothing changes.

### 4.4 Path A's test jobs

- `export_bundle` accepts `("test",)` alone, only when unlocked, into `artifacts/bundle_test/`
  (a new key in `config/sagemaker.yaml`). A bundle is still either `fit` and `dev` or `test`,
  never a mix. `read_bundle` and `bundle_split` follow.
- `load_split("test")` in a job: D20.
- An `evaluate` job with `--split test` takes the test bundle (check J3 reads that one), is
  estimated on 2,898 examples, needs `--scope measured`, and passes J11.

```powershell
venv\Scripts\python -m scripts.sagemaker_job bundle --split test
venv\Scripts\python -m scripts.sagemaker_job plan evaluate --predictor qwen3-4b --split test --scope measured
venv\Scripts\python -m scripts.sagemaker_job launch evaluate --predictor qwen3-4b --split test --scope measured --yes
venv\Scripts\python -m scripts.sagemaker_job launch evaluate --predictor qwen3-4b-tuned --adapter selected --workspace qlora-train-958bed10-a1 --split test --scope measured --yes
venv\Scripts\python -m scripts.sagemaker_job pull
```

The changed `evaluate` task is rehearsed on dev with `--limit`, as before. Its test path is
covered by unit tests on invented rows.

### 4.5 The Nova runs and the six-arm record

```powershell
venv\Scripts\python -m scripts.evaluate --predictor nova-lite --split quick_dev --limit 5
venv\Scripts\python -m scripts.evaluate --predictor nova-lite --split test --scope measured
```

and the same for `nova-lite-tuned`, `nova-pro`, and `nova-pro-tuned`. A stopped run prints
its `--resume` command.

```powershell
venv\Scripts\python -m scripts.comparison_runs status      # which arms are done, running, missing
venv\Scripts\python -m scripts.comparison_runs record      # C1 to C8; writes the record
venv\Scripts\python -m scripts.comparison_runs void --run <run id> --reason "<why>" --yes
```

`record` writes `reports/comparison/six_arm_test_runs.json` and `.md` (committed).

| Id | Check | Passes when |
|---|---|---|
| C1 | Six arms | Each arm has exactly one complete, unvoided result run on test (R1 to R5) |
| C2 | Every question | Each run has one prediction for each of the 2,898 test ids, in the manifest's order |
| C3 | Same questions | The six runs agree on prompt id and manifest hash. A0 and A1 report the same input tokens; so do B0 and B1 of each model |
| C4 | Frozen | Each run's predictor name, model id, settings, and shared values equal the freeze record's |
| C5 | After the unlock | Each run's commit contains the unlock record, and the unlock record names the freeze record |
| C6 | The right models | A1's adapter is the selection record's; each B1's deployment is the tuning record's latest attempt |
| C7 | Usage recorded | Each Nova run has AWS's token counts for every prediction. Each Qwen run belongs to a pulled job with billable seconds |
| C8 | No stray run | No other complete test run exists that is not voided with a reason |

Exit codes for both scripts: 0 done; 1 a check failed; 2 a config problem, no `--yes`, or a
refusal.

### 4.6 Tests

No network, no credentials, no GPU, invented rows only.

- Freeze: each of F1 to F8 passes and fails; the record is the same bytes for the same
  state; it is refused once an unlock record exists; Markdown and JSON agree; no example
  text, name, or address in either.
- Unlock: refused without `--yes`, without the hash, with another hash, with a dirty tree,
  with an uncommitted freeze record, and when a frozen value changed after the freeze; the
  record written unlocks `load_split("test")`.
- Gate: each condition of D15 refuses with its own message; `gold` passes; a resume of an
  unfinished run passes; a second run is refused and is allowed after a void; dev and
  quick-dev never reach the gate.
- Bundle: `test` is refused while locked and with any other split beside it; a test bundle
  serves `test` in a job only with an unlock record that names the manifest; a `fit` and
  `dev` bundle still refuses `test`.
- Launcher: J11 passes and fails; a test job's definition names the test bundle; `rehearse`
  refuses `test`.
- Record: each of C1 to C8 passes and fails; five arms write nothing; a voided run appears
  as a deviation.
- Guards: only the listed modules ask for `test` (D18); a frozen value that differs from a
  committed freeze record fails a test in the suite; no new module imports boto3.
- Existing tests that assert the repository is locked are changed to accept "locked, or
  unlocked by a record that names a committed freeze record".

### 4.7 Documentation

- Runbook: a section for this version, in the order of section 9, with what to do when a
  run stops and when a run must be voided.
- `administration/aws_commands_run.md`: the owner's commands as a new stage of the run
  sheet.
- `reqspecs/1. PROMPTS/coding.md`: the rules on the test lock, the bundle, and "no test row
  in S3" are rewritten for the state after the unlock: who may read test, through which
  gate, and that nothing frozen may change.
- `splits/README.md` and `reqspecs/references/project_patterns/split_loading.md`: the
  unlock record and what it names.
- `reqspecs/references/tech/sagemaker.md`: the test bundle and J11.

## 5. Out of scope

- Costs: per arm, per 1,000 questions, break-even, and the reconciliation with the bill
  (v5.02).
- Confidence intervals, McNemar tests, comparisons between arms, error categories, and
  examples (v5.03).
- Any change to a prompt, a setting, an adapter, or a tuned model. If the test results
  disappoint, they are the results.
- A second look at dev, or a new dev run.
- Deleting the deployments, the tuned models, or the bundles (the runbook's teardown,
  after v5.03 and v6.01 have what they need).
- The local full run of v3.06: it is not an arm and gets no test run.

## 6. What this hands to later versions

| For | What |
|---|---|
| v5.02 | Per Nova arm, AWS's input and output tokens for 2,898 questions. Per Qwen arm, the billable seconds of its own job. The run ids to price |
| v5.03 | Six prediction files over the same 2,898 ids in the same order, ready for paired statistics and for `classify`. The freeze record, the unlock record, and any deviation, for the report's method section |
| v6.02 | The test figures for the model card |

## 7. Known limits and follow-ups

- **The gate is a convention enforced by code and tests, not access control.** The snapshot
  on this machine holds the test rows, as it always has.
- **The cost estimate assumes test prompts as long as dev's.** They have never been
  counted. The first Nova Lite run shows the real figure for cents.
- **Nova Pro's two runs take about 2 hours 41 minutes each.** The AWS sign-in must outlast
  a run, or the run stops and is resumed.
- **Dev flatters A1 a little and test should not.** Checkpoint 500 is the best of ten on
  dev. A test figure below 71.74% is expected and is not a fault.
- **The Nova dev figures were also seen by Bedrock** as validation loss during tuning. Test
  was not.
- **The oracle-evidence setting still holds on test**: every arm is given the gold
  supporting facts. The report says so (v5.03).
- **A deployment could be deleted or a model retired between the freeze and a run.** The
  tuned predictors already refuse a deployment that is not Active. If one cannot be
  restored, that arm has no test result, and the record cannot be written until the owner
  decides what to do: a deviation, in a spec change.

## 8. Decisions for the owner

**Taken as recommended on 2026-11-22**: the owner started the PRP and its execution without changing any of the six. The table is kept as the record of what was asked.

| # | Question | Recommendation |
|---|---|---|
| 1 | **What does "once" allow?** | D10 to D12: one complete result per arm; an interrupted run is resumed; no `--limit` on test; a rerun only after the owner voids a run for a pipeline or service fault, with the reason committed as a deviation |
| 2 | **How is the freeze signed off?** | D7: the owner's `unlock --yes --freeze <hash>`. No name is written into a committed file |
| 3 | **One SageMaker job for both Qwen arms, or two?** | Two (D19). About $0.14 more, and each arm's cost is its own job's |
| 4 | **A trial before each billed run?** | Yes, five questions of quick-dev per Nova predictor on the day, a few cents in all. Never a trial on test (D11) |
| 5 | **The estimate: about $6.50, at most about $9.50, for all six passes.** Confirm before the unlock | Go ahead. Both caps have ample room |
| 6 | **Wait for Nova Pro's training charge to show on the bill before unlocking?** | No. v5.02 needs that amount; this version does not. Enter it with `nova_tune bill` when it shows |

## 9. Acceptance

Code, by Claude:

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
  and without the training stack blocked, with no network and no credentials.
- `comparison_freeze check` passes F1 to F8 on the real repository and prints the estimate.
- `scripts.evaluate --predictor gold --split test` still answers `LOCKED`. **Claude does not
  write the unlock record, and no test row is read during the build.**
- `sagemaker_job bundle --split test` is refused, because the split is locked.

Live, by the owner (🟡), each AWS command logged in `administration/aws_commands_run.md`:

1. `comparison_freeze freeze`; read the record; commit it.
2. `comparison_freeze unlock --yes --freeze <hash>`; commit the unlock record.
3. `comparison_runs preflight`: gold 2,898 of 2,898. `finqa_split`; commit the report.
4. Sign in. The four Nova runs, each after its five-question trial on quick-dev.
5. `sagemaker_job bundle --split test`; plan, launch, and pull the A0 job, then the A1 job.
6. `comparison_runs record`: C1 to C8 pass. Commit the record and the ledger.

- Six complete result runs exist on test, one per arm, each over 2,898 questions.
- The freeze record's commit is an ancestor of the unlock record's, which is an ancestor of
  every test run's.
- No frozen value differs between the freeze record and any run.
- The figures are written into this spec as section 11, with any deviation. M7 is reached.

## 10. Next step

**Done on 2026-11-23 (section 11).** Next is req_005 v5.02, the cost model, which needs its spec. What this section said before the runs: the code is built. The owner's steps, in order (runbook section 7, run sheet section 19): commit the build; `comparison_freeze freeze` and commit; `comparison_freeze unlock --yes --freeze <hash>` and commit; `comparison_runs preflight`; the four Nova runs; the two SageMaker jobs; `comparison_runs record` and commit. That reaches M7. Then req_005 v5.02, the cost model, which needs its spec.

## 11. As built (2026-11-22)

The code is built (`prp_005_test_unlock_six_arms_v1.md`). **The test split is still locked:
no freeze record and no unlock record was written, no test row was read, and nothing was
sent to AWS.** The owner's steps of section 9 remain; the results of the six runs will be
added to this section when they are made.

**2026-11-23, the owner's freeze and unlock.** The sentence above describes the build as it
was on 2026-11-22. Since then: the build was committed (`51ddcea`); `comparison_freeze
freeze` passed F1 to F8 and wrote the freeze record, SHA-256
`1785a84a3817ca63878462a91178d28e6d35762a3896e2afb5abd62658f64ea4` (commit `8cc1c18`);
`comparison_freeze unlock --yes --freeze 1785a84a3817` wrote `splits/test_unlock.json`
(commit `e041952`). The test split is unlocked and no test run exists yet.

**2026-11-23, the six runs and the record: v5.01 complete.** All six arms answered the 2,898
test examples once, each at the frozen configuration; no run stopped and none was voided.
`comparison_runs record` passed C1 to C8 and wrote
`reports/comparison/six_arm_test_runs.json` and `.md`. Milestone M7 is reached.

| Arm | Run | Execution accuracy | In the format |
|---|---|---:|---:|
| A0 | `20261123T183543Z-eval-qwen3-4b-test-ea44` | 27 (0.93%) | 2,630 |
| A1 | `20261123T201754Z-eval-qwen3-4b-tuned-test-074f` | 2,121 (73.19%) | 2,898 |
| B0-Lite | `20261123T072634Z-eval-nova-lite-test-e25a` | 484 (16.70%) | 2,764 |
| B1-Lite | `20261123T165542Z-eval-nova-lite-tuned-test-b4bf` | 2,069 (71.39%) | 2,898 |
| B0-Pro | `20261123T172210Z-eval-nova-pro-test-6176` | 1,025 (35.37%) | 2,890 |
| B1-Pro | `20261123T181752Z-eval-nova-pro-tuned-test-79c0` | 2,179 (75.19%) | 2,898 |

These are counts. Whether one arm is better than another is for v5.03, which adds the
confidence intervals and the paired tests.

What the real runs showed that the build could not:

- **The gate works inside a real SageMaker job.** Both jobs found the unlock record in their
  commit, read the test bundle, and passed `check_test_run`.
- **A1 passes J11.** Its plan passed all eleven checks at commit `d604229`.
- **The test prompts fit.** The longest sequence is 1,606 Qwen tokens; none is over 2,048.
  The estimate's assumption (test prompts as long as dev's) held: the two jobs cost $2.63 at
  the list price against an estimate of $2.59 (4,164 and 2,564 billable seconds).
- **Both deployments were Active** and answered every question.
- **The two Nova Pro runs can go side by side** at 18 requests a minute each: untuned Nova
  Pro and the tuned deployment have separate quotas.

One thing to know when running the like again: a permission rule saved by the coding
assistant into the tracked `.claude/settings.json` made one measured run refuse to start
("uncommitted changes"). `git checkout -- .claude/settings.json` cleared it.

After the six runs the full suite was run again with the test split unlocked: 2,585 passed, lint clean.

The owner started the PRP without changing section 8, so its six decisions were taken as
recommended.

What was run:

- `venv\Scripts\python -m pytest -q`: 2,585 passed (95 new). With the training stack blocked:
  2,579 passed, 6 skipped. `venv\Scripts\python -m ruff check .`: clean.
- `venv\Scripts\python -m scripts.comparison_freeze check`: F1 to F6 and F8 pass on the real
  repository; F7 fails on the uncommitted build, as it must. The six dev runs are the ones
  of section 2.2. Estimate: **about $6.48, at most $9.52** (A0 $1.54, A1 $1.05, B0-Lite
  $0.14, B1-Lite $0.13, B0-Pro $1.87, B1-Pro $1.75). Path A has $8.24 of $100 spent, Path B
  $66.84 of $200. Nothing was written.
- `comparison_freeze status`: not frozen, locked. `comparison_runs status`: six arms missing.
- `scripts.evaluate --predictor gold --split test`, `scripts.evaluate --predictor nova-lite
  --split test --scope measured`, `comparison_runs preflight`, and `sagemaker_job bundle
  --split test`: each answers `LOCKED` and exits 2, before any sign-in, call, or file.
- `sagemaker_job rehearse evaluate --predictor qwen3-4b --split test`: refused, exit 2.

Differences from sections 3 and 4:

- **The freeze record the code reads is `splits/config_freeze.json`**, not
  `reports/comparison/config_freeze.json`, and voided runs are in
  `splits/test_run_voids.json`. The gate also runs inside a SageMaker job, and a job's code
  holds `config/` and `splits/` but not `reports/`. The Markdown copy is under
  `reports/comparison/` as specified.
- **F4 does not compare config file hashes between the dev runs.** The two Nova baselines of
  2026-11-16 snapshotted other bytes of `evaluation.yaml` and `dsl.yaml` than the four later
  runs. Each dev run is instead summarized again under today's configuration (F3: checks R1
  to R5, which rescore every stored reply), and the freeze holds the hashes of today's
  files.
- **Hashes are taken with Unix line ends.** Git checks the config files out with CRLF on
  this machine, and a job has them with LF. The same file must give the same hash in both
  places, or the gate would refuse every test job.
- **The gate is inside `run_evaluation`**, not only in the three callers of section 4.3:
  any code that evaluates on `test` passes it. `scripts.evaluate` and the `evaluate` task
  also ask its cheap part first, so a refusal comes before a model is loaded.
- **A run that was started and did not finish must be resumed.** A new start is refused and
  names the run (D10 says a resume is allowed; this makes it the only way on).
- **`serialize_split("test")` now raises whatever the lock says** (D18): it is how training
  files and token counts are made. A real-data test relied on the lock for this.
- **C4 does not compare a run's snapshotted config hashes** with the freeze (they are over
  raw bytes, see above). It compares the run's predictor, reply limit, scope, and place
  with the freeze, and today's shared files with the frozen ones. The gate checked the
  files when the run started.
- **J11 also checks, for A1, that `--adapter` is the selected checkpoint**, and that no
  earlier test job of the arm is waiting to be pulled.
- **`comparison_runs preflight` exists as a command** (section 4.2 names it): the gold run
  and the token count in one step. The count is skipped, and says so, when the tokenizer is
  not on the machine.
- **`committed_path_a` moved to `local_qlora/estimate.py`** so that the freeze can check
  Path A's cap without importing botocore.
- Not built: nothing of section 4 was left out. Arm A1's path through J11 (the selected
  adapter) is covered by a refusal test only; its passing path needs a selection record and
  is first exercised by the owner's `plan`.

Still unverified, because only a real unlock can show it: the gate inside a real SageMaker
job (the freeze and unlock records arriving through `git archive`, and the hashes agreeing
there); the real length of the test prompts; that a tuned deployment is still Active on
the day.
