# QLoRA Local

Spec Group      :   req_003_qlora_local  
Feature Name    :   qlora_local  
Version         :   6  
Status          :   ✅ Complete: code built 2026-11-20 (`prp_003_local_full_run_v6.md`); the owner's run was made the same day and the local note is written (section 12). Checkpoint 900: 360 of 499 right on dev (72.14%). Not an arm. Optional: no other version depends on it  
Last updated    :   2026-11-21  

## 1. v3.06 — Full local training run (optional)

The owner's learning exercise: the full fine-tune on the laptop's own GPU, with the config
the cloud run will use, scored and selected the way the cloud run will be. It costs
electricity and about five hours of the machine. **Its result is a side note. It is not an
arm, it is never arm A1, and it enters no cost comparison**: a local run has no AWS bill
(working agreement 11).

Almost everything it needs exists. `scripts.qlora_train` is the full run on this machine,
`--resume` continues one, `job_run score` scores a run's checkpoints, and `qlora_select
--check` and `qlora_analyze --check` look at the result. What is missing is small:

- a **preflight**, because a five-hour run on a laptop fails for reasons a cloud job does
  not have: another program holding GPU memory, the lid, the battery, sleep;
- a **progress line** for a run that is under way or was interrupted;
- a **local note**: a committed record and page of the local run, which the existing
  selection can never write, because it writes arm A1 and refuses local runs by design;
- later, a **local-against-cloud section** in that note, once v3.04's selection exists.

It is worth more today than when it was planned. It was to run "after v3.02 and before
v3.03"; instead three versions of cloud tooling have been built with no real trained
adapter anywhere. A local full run gives one: ten real checkpoints, a real accuracy curve,
and real failures for the error analysis, before any dollar is spent.

**Depends on**: v3.02. It uses v3.04's `score` task and v3.05's analysis, which are built.
It needs nothing from AWS.

## 2. What exists today (verified 2026-11-20)

### 2.1 What earlier versions provide

| Item | State |
|---|---|
| `scripts.qlora_train` | Without a flag: the full run here. 4,884 `fit` records, 3 epochs, 918 optimizer steps, a checkpoint every 100 steps and at 918, the loss on dev at each. Checks K1 to K5. `--dry-run` prints the plan with no GPU |
| `scripts.qlora_train --resume RUN_ID` | A new run from the last checkpoint. Refused under changed settings |
| `scripts.job_run score --run RUN_ID --split quick_dev` | The untuned model and every checkpoint on a split, each an ordinary evaluation run, `where: local`. What is already scored is skipped; a run cut short is resumed. Checkpoints are found along the chain of a resumed training |
| `scripts.evaluate --predictor qwen3-4b-tuned --adapter RUN_ID/checkpoint-N --split dev` | One checkpoint on all of dev |
| `scripts.qlora_select --run RUN_ID --split quick_dev --check` | The per-checkpoint table and the best checkpoint. Check L5 fails for local runs, and nothing is ever written: a local run cannot become arm A1 |
| `scripts.qlora_analyze --run RUN_ID --split quick_dev --check` | The failure categories per checkpoint, and the finding. Writes nothing |
| The run registry and the ledger | A local run records wall-clock seconds, tokens, the GPU, and peak memory; no billable seconds |
| `config/training.yaml` | The settings both runs share. A local full run has the same settings hash as the cloud run; the two adapters differ by the hash of their weights |
| Roadmap | "Execution accuracy per checkpoint on the 200-example quick-dev subset." "To confirm in the spec (recommended): score the selected local checkpoint once on the full dev slice" |

### 2.2 Measured on this machine

| Quantity | Value |
|---|---|
| Training, per sequence | 0.88 to 0.96 s (the smoke run: 0.8755) |
| Loss on dev at a checkpoint | About 0.2 s per record |
| A tuned reply at batch size 8 | 1.05 s (the smoke run's reload); 1.4 s with model loading, over 24 examples |
| An untuned reply at batch size 8 | 1.28 s |
| GPU memory in training | Worst batch 11.00 GiB allocated, 12.35 reserved; **13.6 GiB reserved in a resumed run**. The card has 16 GiB |
| GPU memory held by other programs right now | **2.4 GiB**, with no training running |
| A checkpoint on disk | 139 MB (adapter 63, optimizer state 65, tokenizer 11): ten are 1.4 GB |
| Disk free | 1,097 GB |
| Sleep on AC power | Never (the machine's current setting) |

### 2.3 What the exercise would take

| Step | Work | Time |
|---|---|---:|
| Training | 14,652 sequences; ten passes of the loss over dev | 3.9 to 4.2 hours |
| Scoring on quick-dev | The untuned model and ten checkpoints, 200 examples each | 45 to 55 minutes |
| The selected checkpoint on dev | 499 examples | 10 to 12 minutes |
| **All** | | **about 5 hours** |

Scoring all ten checkpoints on all of dev instead would be 2.4 hours of scoring, 6.5 hours
in all.

### 2.4 What is not known

1. **Whether the run fits beside what else is on the GPU.** 13.6 GiB reserved plus the
   2.4 GiB other programs hold today is the whole card. Windows lets a GPU spill into
   shared memory, which does not fail and is many times slower.
2. Whether the laptop holds 0.9 s per sequence for four hours, or slows as it heats.
3. How good a fully trained adapter is. Still only the smoke adapter's 10 of 24.

## 3. Locked decisions

### The run

| # | Decision | Reason |
|---|---|---|
| D1 | The run is `scripts.qlora_train`, unchanged, with `config/training.yaml` as committed, from a clean working tree | "The same config and pinned versions the cloud run will use." A run from a dirty tree cannot be tied to a commit |
| D2 | It is recorded as `where: local`, `scope: dev`. Nothing here can set `measured` | Only cloud runs with that scope enter the comparison (v3.04 L5). The label is what keeps the side note out |
| D3 | An interrupted run is continued with `--resume`, as many times as needed. It is never started again while a checkpoint exists | Every earlier checkpoint stays valid, and the chain is what the scoring and the note follow |
| D4 | **A preflight runs first** (section 4.1) and the run is not started while a check fails. It is a separate command; `qlora_train` itself does not change | The failures it catches cost hours when they are found at step 400 |
| D5 | **The owner starts the run**, in a terminal of their own. **Decision for the owner in section 8** | It is the owner's exercise, it takes five hours, and it should not depend on an assistant's session staying open. Nothing is billed, so this is a recommendation, not a rule |

### Scoring and selecting

| # | Decision | Reason |
|---|---|---|
| D6 | Every checkpoint and the untuned model are scored on **quick-dev** (200 examples) with `job_run score`, at the batch size of `config/local_qlora.yaml` (8) | The owner's decision of 2026-11-14, to shorten the laptop run |
| D7 | The local selection is the rule of v3.04 applied to quick-dev: most right on execution, then format validity, then the earlier step | One rule. The note says which split chose |
| D8 | **The selected local checkpoint is scored once on all of dev.** **Decision for the owner in section 8** | The roadmap's recommendation: the cloud run selects on dev, so only a dev figure compares like with like. Ten minutes |
| D9 | The untuned model is not scored on dev locally at batch size 8 | The note compares two tuned adapters. The untuned baseline on dev exists at batch size 1 (v3.01) and will exist from the cloud at 8 |
| D10 | The error analysis is looked at with `qlora_analyze --check` and is not written | The committed analysis is arm A1's. The local one is printed, and its category counts go into the local note |

### The local note

| # | Decision | Reason |
|---|---|---|
| D11 | A new command writes `reports/local_qlora/local_run.json` and `.md`, committed: the run and its chain, the commit, the settings hash, hours, seconds per sequence, peak GPU memory, the GPU's name, interruptions, the per-checkpoint table on quick-dev with the loss on dev, the selected checkpoint with the hash of its weights, its result on dev, and the category counts of the selected checkpoint. Counts, ids, hashes, and times only | The selection record cannot hold it: that record is arm A1. A separate file cannot be mistaken for it |
| D12 | The note is written only when checks N1 to N5 pass (section 4.3). It is **never** read by `--adapter selected`, by `keep`, or by anything of req_005 | A side note. `--adapter RUN_ID/checkpoint-N` loads the local adapter for anyone who wants it, by name |
| D13 | Every page and every printed summary of the local run carries the words "local run: not an arm of the comparison" | The figures will look like arm A1's. They are not |
| D14 | When a selection record exists (v3.04), the same command adds **local against cloud**: hours, seconds per training sequence, the step each selected, execution accuracy and format validity on dev, and whether the two are within `selection.noise_examples`. No dollars | The roadmap's closing note. Electricity is not measured, and a made-up figure would be worse than none |
| D15 | The cloud side of that section is read from the selection record and the pulled training run. If they are not there, the section says "not yet" and the note is still written | The local run should not wait for the cloud run, nor the reverse |

### Keeping

| # | Decision | Reason |
|---|---|---|
| D16 | The final adapter and all checkpoints stay under git-ignored `artifacts/`. Nothing is uploaded. Deleting the checkpoints afterwards is the owner's choice; the note keeps the hash of the selected one | 1.4 GB of 1,097 GB free. S3 holds cloud adapters only |
| D17 | The local run's ledger lines are committed as every local run's are | The ledger is the record of what was run, billed or not |

## 4. Scope to build

### 4.1 The preflight and the progress line: `qlora_finetuning/local_qlora/local_run.py`, `scripts/qlora_local.py`

```powershell
venv\Scripts\python -m scripts.qlora_local preflight
venv\Scripts\python -m scripts.qlora_local status [--run RUN_ID]
venv\Scripts\python -m scripts.qlora_local note --run RUN_ID [--check]
```

`preflight` starts nothing and needs the GPU only to ask it for its memory:

| Id | Check | Passes when |
|---|---|---|
| P1 | The plan | `qlora_train --dry-run` succeeds: every record fits, 918 steps, ten checkpoints. The estimate of section 2.3 is printed from the last local run's measured seconds per sequence |
| P2 | A clean tree | No uncommitted change outside `ledger/` |
| P3 | GPU memory | The free memory of the GPU is at least `local_run.min_free_gpu_gib` (14.0; first 14.5, see section 11). The check names what is in use, so the owner can close it |
| P4 | Disk | At least `local_run.min_free_disk_gb` (10) free where `artifacts/` is |
| P5 | Power | On AC power. If the sleep setting on AC can be read, it is "never"; if it cannot, the check is skipped and says so |
| P6 | Nothing running | No training run of this project is recorded as started and unfinished in the last 24 hours |

`status` reads a run's log and prints the step, the share done, the mean seconds per step
so far, the hours left at that rate, the last loss, the loss on dev at the last checkpoint,
and whether the run ended, and how. For a run that was interrupted it prints the `--resume`
command. Without `--run` it takes the newest full local run.

`config/training.yaml` gets a block `local_run` with the two thresholds and the note's
path. None of it is part of a run's settings.

### 4.2 The run sheet (no new code)

```powershell
venv\Scripts\python -m scripts.qlora_local preflight
venv\Scripts\python -m scripts.qlora_train                                   # about 4 hours
venv\Scripts\python -m scripts.qlora_local status                            # from another terminal
venv\Scripts\python -m scripts.qlora_train --resume <run_id>                 # only if it stopped
venv\Scripts\python -m scripts.job_run score --run <run_id> --split quick_dev          # about 50 minutes
venv\Scripts\python -m scripts.qlora_select --run <run_id> --split quick_dev --check   # the table; L5 fails by design
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b-tuned --adapter <run_id>/checkpoint-<N> --split dev
venv\Scripts\python -m scripts.qlora_analyze --run <run_id> --split quick_dev --check
venv\Scripts\python -m scripts.qlora_local note --run <run_id>
git add reports/local_qlora ledger ; git commit -m "Path A: the local full run (a side note)"
```

### 4.3 The local note

`note` gathers the run's scored checkpoints with v3.04's `gather_scored` on quick-dev, so
checks L1 to L4 and L6 apply as they are. In place of L5 and for the rest:

| Id | Check | Passes when |
|---|---|---|
| N1 | Scored | L1 to L4 and L6 pass on quick-dev for the run: every checkpoint and the untuned model, one complete result each, the right weights, the run completed under the current settings |
| N2 | Local | Every run is `where: local`. A run made in the cloud is refused: it has a selection record of its own |
| N3 | The full run | The training run is not a smoke run, and trained all of `fit` |
| N4 | On dev | The selected checkpoint has exactly one complete result on all of dev, made with the same settings as its quick-dev result (skipped, and said so, if the owner decides against D8) |
| N5 | Scores current | Those runs pass R1 to R5 now |

`--check` prints and writes nothing. The same runs give the same files: the note carries no
time stamp of its writing. With a selection record present it adds the section of D14.

### 4.4 Tests

No test needs the network, credentials, a GPU, or a paid call.

- Preflight: each of P1 to P6 passing and failing, with stand-ins for the GPU's memory, the
  disk, the power status, and the registry; P5 skipped when the setting cannot be read;
  the estimate from a stand-in measurement; nothing is started and nothing written.
- Status: a run in progress, completed, failed, and resumed along a chain; hours left from
  the log; a run with no log yet; no local run at all.
- The note, on the stand-in world of v3.04's tests made local: each of N1 to N5; the
  selected checkpoint by the rule on quick-dev; the dev result beside it; the same files
  twice; **no reply and no example text in them**; the words of D13 on the page; `--check`
  writes nothing; a cloud run is refused; a smoke run is refused.
- Local against cloud: without a selection record the section says "not yet"; with one,
  the figures of both sides and the noise note at the limit.
- Guards: a local note never changes the selection record; `--adapter selected` is
  unaffected by it; nothing here imports boto3 or uploads anything.

### 4.5 Documentation

- Runbook section 5: the local full run step by step, what to close first, how to follow
  it, what to do when it stops, how to read the note.
- `gotchas.md`: GPU memory held by other programs; shared GPU memory on Windows; a local
  run has the cloud run's settings hash and another adapter hash.
- `coding.md`: a local run is never an arm; its record is the local note and nothing else.
- Architecture overview. `administration/aws_commands_run.md` is not touched: no AWS
  command is involved.

## 5. Out of scope

- Any change to training, scoring, selection, or the settings.
- Counting the local run in the comparison, its cost tables, or its statistics (req_005).
- A dollar figure for electricity.
- Using the local adapter as arm A1 if the cloud run is never made. That would be a change
  of the plan (working agreement 11), and the owner's to raise.
- Uploading anything, and any use of `--scope measured` on this machine.
- Tuning: a second local run with other settings. One corrective iteration belongs to the
  cloud path (v3.05).
- Running under WSL or Linux, or any driver or power-plan change made by code.

## 6. What this hands to later versions

| For | What |
|---|---|
| v3.04 | Nothing it depends on. If the local run is made first: a real ten-checkpoint run to rehearse `score`, `qlora_select --check`, and the curve's shape before the billed jobs |
| v3.05 | Real failures of a trained adapter, through `qlora_analyze --check`, before the table of findings is ever applied to arm A1 |
| v5.03 | The local-against-cloud section, as a side note of the report, labelled as such |
| v6.01 | A trained adapter on this machine that the demo can be developed against, by name. The demo's shipped adapter is the selected cloud one |

## 7. Known limits and follow-ups

- **The local selection is on quick-dev and the cloud's is on dev.** Their per-checkpoint
  tables are not comparable row by row; only the two selected checkpoints on dev are (D8).
  Quick-dev is 200 examples: two checkpoints five examples apart are not told apart.
- **Quick-dev is part of dev.** The local checkpoint is chosen on 200 of the 499 examples
  it is then scored on, so its dev figure flatters it a little, as arm A1's does.
- **Two runs with one config and one seed are two adapters** (v3.02). A difference between
  the local and the cloud result within noise is the expected outcome, and says the
  pipeline is the same in both places, not that one GPU trains better.
- **Speeds compare hardware, not cost.** The laptop is not billed; the A10G is. The note
  gives no price for the local run.
- **A laptop run can stop for reasons the preflight cannot see**: a driver update, a
  restart forced by the system, heat. `--resume` is the answer to all of them.
- **P5 reads the power state at the start only.** Unplugging later is not noticed.
- **The machine is not free for other GPU work for five hours**, and anything else that
  takes GPU memory during the run may push it into shared memory and slow it severalfold
  without failing.

## 8. Decisions for the owner

**Taken as recommended on 2026-11-20**: the owner started the PRP and its execution
without changing any of the five. The table is kept as the record of what was asked.

| # | Question | Recommendation |
|---|---|---|
| 1 | **Do this optional version at all, and when?** | Yes, and now, before the smoke job. It is free, it needs no sign-in, and it is the first time the whole pipeline produces a real adapter: ten checkpoints, a real curve, real failures for the analysis. Its original purpose, "the run also exercises the whole pipeline before any billed job", still holds |
| 2 | **Who starts the five-hour run (D5)?** | You, in your own terminal, after `preflight`. I can start it if you prefer; it would then run as a background task of a session, and if the session closes the run is orphaned, though `--resume` recovers it |
| 3 | **Score the selected local checkpoint on all of dev (D8)?** | Yes: ten minutes, and without it the local and cloud figures cannot be put side by side |
| 4 | **Quick-dev for the ten checkpoints (D6), as decided on 2026-11-14, or all of dev?** | Keep quick-dev: 50 minutes against 2.4 hours. If you would rather have the local table comparable row by row with the cloud's, all of dev is one flag (`--split dev`) and the note records which was used |
| 5 | **How much GPU memory must be free (P3)?** | 14.5 GiB of the 16. Today 2.4 GiB is held by other programs, so the preflight would fail as the machine stands: close them, or lower the threshold knowing that the run may spill into shared memory and crawl |

## 9. Acceptance

Code, by Claude:

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
  and without the training stack blocked.
- `qlora_local preflight` on this machine prints P1 to P6 with real values and starts
  nothing. As the machine stands it is expected to fail P3, and to say what holds the memory.
- `qlora_local status` with no full local run says so; on the local smoke run, with
  `--run`, it prints 12 of 12 steps.
- `qlora_local note --run <the local smoke run> --check` fails N3 (a smoke run) and writes
  nothing.

Live, by the owner: the run sheet of section 4.2.

- Training ends at step 918 with ten checkpoints; K1 to K5 pass; an interruption, if any,
  was resumed.
- Eleven results on quick-dev and one on dev.
- `qlora_local note` passes N1 to N5 and writes the note; it is committed with the ledger.
- Section 11, "As built", is added: hours, seconds per sequence, peak memory, the
  per-checkpoint table, the selected checkpoint, its dev result, and the categories of its
  failures. After v3.04's selection, `note` is run again and the local-against-cloud
  section is added there and in the roadmap.

## 10. Next step

The run is made and the note is written (section 12). What is left is small: after
v3.04's selection, `note` is run once more for the local-against-cloud section. The
group's next work is v3.04's training and scoring jobs in the cloud, which are the owner's.

## 11. As built (2026-11-20): the code

Built by `prp_003_local_full_run_v6.md`. **The run itself has not been made**: it is the
owner's to start. Nothing was trained, nothing is billed, and no AWS call was made. This
section covers the code; the run's figures are added after it.

What was run:

- `venv\Scripts\python -m pytest -q`: 2,485 passed (44 new for this version). With the
  training stack blocked: 2,479 passed, 6 skipped.
  `venv\Scripts\python -m ruff check .`: clean.
- `scripts.qlora_local preflight`, `status`, `status --run <the local smoke run>`, and
  `note --run <the local smoke run> --check`, on this machine.

### The rehearsals on this machine

| Command | Result |
|---|---|
| `preflight` | P1 passes: 4,884 records, 918 steps, ten checkpoints, about 3.9 hours of training at 0.96 s per sequence. **P3 fails**: 13.5 GiB free of 16.0, 14.5 asked; it names Acrobat, the browser, Explorer, the security app, the phone link, and the chat app. P2 fails on the uncommitted files of this version. P4 (1,178 GB free), P5 (on AC, sleep never), and P6 pass. Nothing started |
| `status` | `CONFIG ERROR: no full training run has been made on this machine yet` |
| `status --run <smoke run>` | Step 12 of 12, completed, 14.0 s per step, loss on dev 0.2357, checkpoints at 4, 8, 12 |
| `note --run <smoke run> --check` | N3 fails (a smoke run is a rehearsal), N1 fails (its checkpoints were scored on 24 examples, not on quick-dev); nothing written |

### The decisions of section 8

Taken as recommended: the version is done now; the owner starts the run; the selected
checkpoint is scored once on dev; the ten checkpoints on quick-dev; 14.5 GiB must be free.

**Changed the same day, by the owner: 14.0 GiB, not 14.5.** With every program closed, the
preflight read 14.3 GiB free of 16.0 twice, and 14.5 a little later: the Windows desktop
(`explorer.exe`) holds 1.1 to 1.4 GiB of the card and cannot be closed. So 14.5 sat on the
edge of what this machine can free; it had been set before that was measured. 14.0 is
under what the machine can free and above the
13.6 GiB that a resumed run reserved, the one measurement there is. The value is not one
of the training settings, so the run is the same run. Whether 13.6 holds for four hours
is for the run to show: `qlora_local status` after the first minutes gives the seconds
per step, and a figure several times the expected one means memory has spilled over.

### Differences from sections 3 and 4

- **P6 also fails for a stopped full run that should be resumed** (D3), and prints the
  `--resume` command. A run counts as "in progress" while its files were written in the
  last 15 minutes: a killed run stays `started` in the run index for ever, so the index
  alone cannot say.
- **The newest run is the one no other run continues**, not the last id in order: two runs
  can start in the same second. A test found it.
- **P1's estimate is of training only.** The loss on dev at each checkpoint adds a few
  minutes that no run has measured on its own.
- **N1 is the selection's checks L1 to L4; N5 is L6 and the dev result.** L5 is not asked.
- **N4 cannot be skipped.** The owner decided for D8, so the option to skip it was not built.
  `note --split dev` covers a run whose checkpoints were all scored on dev.
- **The note also says whether the run was made from a clean tree**, and, in the JSON, the
  cloud training job's billable seconds. It holds no dollars.
- The preflight and status are `local_qlora/local_run.py`; the note is
  `local_qlora/local_note.py`.
- Speed for a run in progress is the age of its log over the steps logged: a training log
  line carries no time.

### While this was built

The owner's first SageMaker smoke job had been launched, and changes for it were in the
working tree (`local_qlora/backend.py`, `scripts/job_bootstrap.sh`, a test, the AWS
commands log). They are not part of this version and were not touched.

### The owner's run (made: see section 12)

```powershell
# commit, and close the programs that hold GPU memory
venv\Scripts\python -m scripts.qlora_local preflight
venv\Scripts\python -m scripts.qlora_train
```

then the scoring and the note of section 4.2 (the runbook, 5.31). **Do not start it while
anything else needs the GPU**, and not on battery.

## 12. As built (2026-11-20): the run

The owner started the run at 17:30 local time on 2026-11-20, from commit `56dd25c89511`
with a clean tree and all six preflight checks passing. It completed at 20:22 without a
stop. The owner then scored the checkpoints and the selected checkpoint on dev; the note
was written from those runs. Nothing was billed and no AWS call was made.

| | Estimated | Measured |
|---|---|---|
| Training | about 3.9 hours | 2.87 hours: 918 steps at 11.2 s per step, 0.70 s per sequence |
| Scoring ten checkpoints and the untuned model on quick-dev | about 50 minutes | about 27 minutes |
| The selected checkpoint on dev | about 10 minutes | about 6 minutes |
| Peak GPU memory | 13.6 GiB reserved (a 12-step run) | 11.00 GiB allocated, 15.41 GiB reserved |

Training run `20261120T223038Z-train-qwen3-4b-fit-db4b`. The record is `reports/local_qlora/local_run.json` and `.md`.

### The checkpoints on quick-dev (200 examples)

| Checkpoint | Right on execution | In the format | Loss on dev |
|---:|---:|---:|---:|
| untuned | 2 (1.0%) | 186 | |
| 100 | 127 (63.5%) | 200 | 0.1655 |
| 200 | 133 (66.5%) | 200 | 0.1506 |
| 300 | 136 (68.0%) | 200 | 0.1385 |
| 400 | 139 (69.5%) | 200 | 0.1383 |
| 500 | 142 (71.0%) | 200 | 0.1331 |
| 600 | 140 (70.0%) | 200 | 0.1279 |
| 700 | 142 (71.0%) | 200 | 0.1310 |
| 800 | 145 (72.5%) | 200 | 0.1303 |
| 900 (selected here) | 146 (73.0%) | 200 | 0.1300 |
| 918 | 146 (73.0%) | 200 | 0.1299 |

The selection rule picks checkpoint 900: it ties with 918 and the earlier step wins. The
lowest loss on dev is at step 600, which is not the most accurate checkpoint.
`qlora_select --check` passes L1 to L4 and L6 and fails L5, as designed for a local run;
it wrote nothing.

### Checkpoint 900 on all of dev

360 of 499 right on execution (72.14%); 499 in the format. Run `20261121T050005Z-eval-qwen3-4b-tuned-dev-abf8`.
Both thresholds of v3.04 (40% execution, 98% format) would be met, though this run is not
the one they are asked of.

The 139 wrong replies, by computed category: `steps_fewer` 54, `evidence` 37,
`steps_more` 20, `operation` 16, `order` 8, `bad_program` 3, `not_executable` 1.

### What the run settled

- **The laptop holds its speed.** 11.2 s per step over 2.87 hours, faster than the 12-step
  smoke runs suggested (0.70 s per sequence against 0.96). The GPU stayed at 55 to 65 °C.
- **Peak memory is higher than the smoke runs showed**: 15.41 GiB reserved of 16. It rose
  from 8.6 to 15.6 GiB in use over the first ten minutes, as longer sequences arrived, and
  stayed there. The speed did not drop, so nothing spilled into shared memory, but the
  margin is thin: with the 2.4 GiB that other programs held before, the run would not
  have fitted.
- **The first accuracy curve of a fully trained adapter**: 63.5% after 100 steps, rising
  slowly to 73% at the end.

### Still open

After v3.04's selection, run `venv\Scripts\python -m scripts.qlora_local note --run 20261120T223038Z-train-qwen3-4b-fit-db4b`
once more: it adds the local-against-cloud section.
