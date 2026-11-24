# QLoRA Local

Spec Group      :   req_003_qlora_local  
Feature Name    :   qlora_local  
Version         :   5  
Status          :   ✅ Complete 2026-11-21, as analysis only: v3.04's verdict is `met`, so the corrective iteration was not triggered and nothing was billed. The error analysis of arm A1 on dev is written (`reports/local_qlora/error_analysis.md`; commit pending). Finding T3 (section 12)  
Last updated    :   2026-11-21  

## 1. v3.05 — Error analysis and one corrective iteration

v3.04 ends with a verdict on arm A1: `met` or `missed`. This version is what happens after
`missed`. It has two parts, and they are not conditional in the same way.

- **The error analysis** sorts every wrong reply of the scored checkpoints into a fixed set
  of categories, from the stored predictions, with no model and no AWS. It is free, it can
  be built and rehearsed today, and the comparison report (v5.03) needs "error categories
  with examples" whatever the verdict.
- **The corrective iteration** changes **one** training setting, trains once more, scores
  the new checkpoints, and selects again across both runs. It is two more billed jobs, both
  the owner's, and it runs **only if the verdict is `missed`**. It runs once: further
  tuning belongs to `req_007_future_extensions`.

It adds:

- `qlora_finetuning/evaluation/failures.py`: one category per wrong reply, for any run of
  any arm;
- `scripts/qlora_analyze.py`: the analysis of a scored training run, with a committed
  report of counts and example ids;
- `scripts/qlora_iterate.py`: the change is written down, with its reason, **before** the
  second run is paid for, and is checked to be one setting;
- a tenth launch check, J10, so that a second full training run cannot be launched
  without that record, nor a third at all;
- a selection across two runs: the second run replaces the first only if it is better by
  the rule that chose the first.

**Depends on**: v3.04. Its code is built; **its jobs, and the v3.03 smoke job before them,
have not been run**. There is no selection record, so nothing says whether this version is
triggered.

**Trigger** (v3.04 D15): the selected checkpoint's execution accuracy on dev is under 40%,
or its format validity is under 98%.

## 2. What exists today (verified 2026-11-20)

### 2.1 What earlier versions provide

| Item | State |
|---|---|
| A predictions line (`artifacts/runs/<run>/predictions.jsonl`) | The reply, the outcome of running it (the program as read, the value, the error stage and code), the score, the stop reason, tokens, latency. Git-ignored: a reply can repeat dataset text |
| `ExampleScore` | `format_valid`, `executed`, `execution_correct`, `program_match`, `error_code`, `deviations`, `mismatch` (`kind`, `sign`, `scale`, `other`), `relative_gap` |
| `run_predictions(run_id)` | A run's lines in dataset order, checked as `summarize_run` checks them |
| `programs_match`, `parse_program` | Programs compared step by step, numbers by value |
| The selection record (`read_selection`) | Which checkpoint is arm A1, the per-checkpoint table, the verdict, three notes: whether the selected checkpoint is the last, where the dev loss was lowest, the gain over the untuned model. **It does not exist yet** |
| `training.json` in a training run's folder | The run's settings in full (`lora`, `optimizer`, `schedule`, `batch`, `epochs`, `seed`, ...) and their hash |
| `config/training.yaml` | r 16, alpha 32, dropout 0.05, learning rate 2e-4, cosine with 5% warmup, 3 epochs, batch 2 x 8. A resume under changed settings is refused; a changed file means a new run |
| `score` | Reuses a complete result found in the workspace: on the first training job's workspace the untuned model is not asked again |
| `qlora_select`, check L4 | The training run's settings must be the ones in `config/training.yaml` now |
| Checks J1 to J9; the Path A cap | $100; nothing spent |
| Plan and roadmap | "Categorize dev failures (format, sign/order reversal, scale/unit mismatch, wrong evidence); change one variable; rerun and re-select." "One iteration only" |
| Path B's v4.05 | Planned, not specced: "change one job setting (epoch count or learning rate) and rerun" |

### 2.2 Measured for this spec: the categories work on stored runs

A throwaway script sorted the wrong replies of seven stored runs by the rules of section
4.1. No model was asked anything. Counts of replies:

| Run | Right | Not a program that runs | Cut off | More steps than gold | Fewer steps | Other operation | Arguments reversed | Other numbers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Untuned Qwen, dev (499), batch size 1 | 3 | 426 | 45 | 23 | 2 | 0 | 0 | 0 |
| Untuned Nova Pro, dev (499) | 190 | 189 | 3 | 72 | 21 | 7 | 6 | 11 |
| Untuned Nova Lite, dev (499) | 123 | 236 | 2 | 100 | 15 | 14 | 3 | 6 |
| Untuned Qwen, 24 of quick-dev, batch size 8 | 0 | 20 | 2 | 1 | 1 | 0 | 0 | 0 |
| Smoke adapter, checkpoint 4, the same 24 | 9 | 2 | 0 | 2 | 5 | 0 | 4 | 2 |
| Smoke adapter, checkpoint 8, the same 24 | 10 | 0 | 0 | 5 | 1 | 1 | 4 | 3 |
| Smoke adapter, checkpoint 12, the same 24 | 10 | 0 | 0 | 6 | 1 | 1 | 4 | 2 |

- Every wrong reply fell into a category; none needed reading.
- The untuned models fail on the language: nested calls (132 of Nova Pro's failures), bad
  references (201 of Qwen's), unknown constants. Twelve training steps remove that
  entirely, and what is left is the calculation: too many steps, and the two arguments of
  a subtraction or division the wrong way round.
- The label `sign` alone would have missed the reversed arguments: a reversed division
  gives the reciprocal, not the negative. The categories compare programs, not only values.
- Failures by the length of the gold program are just as cheap: at checkpoint 12 the smoke
  adapter is wrong on 8 of 15 one-step examples and on all of the two examples with four
  or five steps.

These are a smoke adapter's figures on 24 examples. They show the method. They say nothing
about arm A1.

### 2.3 What an iteration would cost, at this machine's speeds

Placeholders, as in v3.04 section 2.3; the plan's estimate (J9) replaces them.

| Change | Training job | Scoring job | Both |
|---|---:|---:|---:|
| Learning rate, or LoRA rank: 918 steps, ten checkpoints | $5.60 to $8.30 | $3.10 to $4.50 (the untuned result is reused) | **$9 to $13** |
| Epochs 3 to 5: 1,530 steps, sixteen checkpoints | $9.30 to $13.80 | $4.90 to $7.30 | **$14 to $21** |

With the first pair of jobs ($9 to $14), Path A would stand at $18 to $35 of its $100 cap.

### 2.4 What is not known

1. **Whether this version is triggered.** Nothing has been trained in the cloud.
2. What the failures of a fully trained adapter look like. Section 2.2's smoke adapter has
   seen 64 records.
3. Which setting would help. That is what the analysis is for, and why the change is not
   chosen in this spec.

## 3. Locked decisions

### The analysis

| # | Decision | Reason |
|---|---|---|
| D1 | A wrong reply gets exactly one category, by the first rule of section 4.1 that fits, computed from the stored predictions line and the gold program. No model is asked and nothing is judged by reading | Repeatable, free, and the same for every arm. Section 2.2 shows it covers every case |
| D2 | The classifier is in `qlora_finetuning/evaluation/`, not in `local_qlora/`: it takes a run id and knows nothing of Qwen | v4.05 and v5.03 need the same categories for the Nova arms. `local_qlora/` and `frontier/` do not import each other |
| D3 | The analysis of a training run covers: the selected checkpoint in full; the untuned model beside it; the category counts of every checkpoint; failures by gold program length, by operation, by company, and by prompt length; examples no checkpoint got right | The corrective change is chosen from the curve over checkpoints and from where the failures sit, not from one table |
| D4 | The committed report (`reports/local_qlora/error_analysis.json` and `.md`) holds counts, rates, run ids, and example ids. **No reply and no example text.** `--show CATEGORY` prints replies beside their gold programs to the terminal only | A reply can repeat dataset text, which is never committed. Example ids are in the committed split manifest already |
| D5 | The analysis is made from the runs the selection was made from: it requires the selection record, and that checks L1 to L6 still pass. `--check` on other runs (a local smoke run, a partial split) prints and writes nothing | One set of runs is arm A1's evidence. A rehearsal must not look like it |
| D6 | The analysis is written whatever the verdict | v5.03 reports error categories for every arm; `met` does not remove the need |

### The trigger and the change

| # | Decision | Reason |
|---|---|---|
| D7 | The iteration is allowed only when the selection record's verdict is `missed`. With `met`, `qlora_iterate` refuses. **Decision for the owner in section 8** | The plan makes the iteration a response to a fault, not a second try at a better number. A second run chosen on dev after a good first one would be tuning on dev |
| D8 | **One variable** is one of: `epochs`; `optimizer.learning_rate`; `lora.r` (with `lora.alpha` kept at twice `r`); `lora.dropout`; `schedule.warmup_fraction`. Nothing else may differ between the two runs | These are the training settings of plan §4.6 that the guides name as the first to revisit. Alpha follows r because the guides fix their ratio; changing r alone would change the effective scale as well |
| D9 | **Out of bounds**: the prompt, the split, the serialization, the reply limit, the inference batch size, decoding, the evaluation allowances, the seed, the batch size, the model | The first seven are shared with arm A0 and the Nova arms: changing one would end the comparison. A new seed is a second draw, not a correction. The batch settings are sized for memory |
| D10 | The change is chosen by the owner with the table of section 4.3, which maps what the analysis finds to one variable and one new value. A choice outside the table is allowed and is recorded as such, with the reason in the owner's words | The table is written now, before any result exists, so that the change is not fitted to the outcome. It cannot foresee everything |
| D11 | **A fault in the pipeline is not an iteration.** If the analysis shows a defect (training sequences cut, a template that differs between training and inference, a check K1 to K7 that should have failed), it is fixed under the version that owns it, and the first run is repeated. The iteration is still available afterwards | The one iteration is for the recipe. Spending it on a bug would leave the recipe untried |
| D12 | The change is registered before it is paid for: `qlora_iterate propose` checks I1 to I5 (section 4.3) and writes the committed `reports/local_qlora/iteration.json`: the first run, the finding, the variable, the old and new value, both settings hashes, the reason | "Change one variable" is only checkable if the variable is named first |
| D13 | A new launch check, **J10, Registered**: a full `train` job that is not a resume, while a selection record exists, needs an iteration record whose new settings are the ones in `config/training.yaml`, and no second run that has completed | Nothing else stops a second, third, or unregistered run. A resume of the second run passes |

### The second run

| # | Decision | Reason |
|---|---|---|
| D14 | The second training job and its scoring job run **on the first training job's workspace** | `score` then finds the untuned model's result and does not ask it again. A second untuned result with the same settings would also fail L1 |
| D15 | Both jobs are `--scope measured`, planned with J1 to J10, and started by the owner. Their billable seconds count towards Path A's training cost in the comparison, beside the first run's | The iteration is part of what arm A1 cost |
| D16 | **Selecting again.** `qlora_select --run <second run>` checks L1 to L6 for the second run, picks its best checkpoint by v3.04's rule, and compares it with the checkpoint already selected, by the same rule: more examples right on execution, then format validity. **A tie keeps the first run's checkpoint.** **Decision for the owner in section 8** | One rule for every choice. The first run's checkpoint is the one that came from the recipe as documented |
| D17 | The selection record becomes schema 2: `selected` is the winner, and `iterations` lists both runs with their settings, their best checkpoint, and their counts. The first record is kept beside it as `adapter_selection_first.json` and `.md`. `read_selection` reads both schemas | The record still says, alone, which run is arm A1; the history of how is not lost |
| D18 | The record says whether the two best checkpoints differ by fewer than `selection.noise_examples` (10 of 499, two points). It changes neither the choice nor the verdict | At 40% the standard error on 499 examples is 2.2 points. A change that moved 6 examples has not been shown to work, and the report must say so |
| D19 | `config/training.yaml` ends as the selected run's settings. If the first run's checkpoint stays, the change is reverted in a commit; `qlora_iterate status` fails until the file and the record agree | v5.01 freezes the config. It must describe the adapter that is arm A1 |
| D20 | The verdict is computed again for the winner, and **the path goes on to v5.01 whatever it is**. A second `missed` is reported as it stands | One iteration only. The comparison is of a documented recipe with one correction, not of the best Qwen adapter that can be made |

## 4. Scope to build

### 4.1 Categories: `qlora_finetuning/evaluation/failures.py`

`classify(line, gold, dsl_config) -> str`, and `failure_counts(run_id, context)`. A reply
that is right on execution is `right`. For the others, the first rule that fits:

| Category | Rule | The roadmap's word |
|---|---|---|
| `cut_off` | The stop reason is not `end`: the reply hit the token limit | format |
| `no_program` | No program line was read (error stage `reply`) | format |
| `bad_program` | The program does not parse (error stage `program`); the error code is kept as a sub-count | format |
| `not_executable` | It parses and cannot be run (error stage `execution`: a row that is not there, a division by zero) | wrong evidence |
| `kind` | A number where yes or no was due, or the reverse | |
| `steps_more`, `steps_fewer` | The program has more, or fewer, steps than the gold program | |
| `operation` | The same number of steps, another operation in at least one | |
| `evidence` | The same operations, and an argument that is neither the gold one nor the gold pair reversed | wrong evidence |
| `order` | The same operations and numbers, with the two arguments of at least one step reversed | sign/order reversal |
| `scale`, `sign` | The same program as gold and a result off by a configured factor, or negated (possible only where the gold answer is not what the gold program gives) | scale/unit mismatch |
| `other` | Anything left | |

Beside the category, three things are counted for every wrong reply without changing it:
the mismatch label of the value (`sign`, `scale`), whether the reply was in the format, and
whether the answer line was right although the program was not.

Format validity is the second threshold and gets its own table: replies not in the strict
format, by deviation (`text_after`, `answer_missing`, ...), whether or not they were right.

### 4.2 The analysis: `qlora_finetuning/local_qlora/analysis.py`, `scripts/qlora_analyze.py`

```powershell
venv\Scripts\python -m scripts.qlora_analyze --run <training run id>
venv\Scripts\python -m scripts.qlora_analyze --run <training run id> --check                 # writes nothing
venv\Scripts\python -m scripts.qlora_analyze --run <training run id> --show order --n 5       # replies, to the terminal only
venv\Scripts\python -m scripts.qlora_analyze --run <run> --split quick_dev --limit 24 --check # a rehearsal
```

The report, per D3:

| Part | Content |
|---|---|
| The selected checkpoint | Category counts and rates over all of dev; the format table; stop reasons |
| Beside the untuned model | The same counts for the untuned run; how many examples were fixed, broken, right in both, wrong in both |
| Over the checkpoints | One row per checkpoint: right, and each category. The step where execution accuracy was highest, where it was within `noise_examples` of the highest for the first time, and whether the last three checkpoints rise, fall, or stay within noise |
| Where the failures sit | Right and wrong by gold program length (1, 2, 3, 4 or more steps), by gold operation, by company (from the example id), and by quarter of prompt length in tokens |
| Never right | How many examples no checkpoint got right, and their ids |
| Example ids | The wrong examples of the selected checkpoint, by category |
| The finding | Which row of section 4.3's table the figures fit, or "none" |

The gathering of a training run's scored runs moves out of `select_checkpoint` into a
function both use; the selection's behavior does not change.

### 4.3 The iteration: `qlora_finetuning/local_qlora/iteration.py`, `scripts/qlora_iterate.py`

The table of D10. "Noise" is `selection.noise_examples`.

| # | What the analysis shows | The one change |
|---|---|---|
| T1 | The best checkpoint is the last or the one before it, and the last three checkpoints rise by more than noise | `epochs` 3 to 5 |
| T2 | The best checkpoint is in the first half, later ones are lower by more than noise, and the dev loss rises after it | `optimizer.learning_rate` 2e-4 to 1e-4 |
| T3 | The checkpoints from the middle on are within noise of each other, and most failures are `steps_more`, `steps_fewer`, `operation`, or `evidence` | `lora.r` 16 to 32 (alpha 64) |
| T4 | Format validity is under its threshold and execution accuracy is not: read the curve as in T1 and T2 and take that row | as T1 or T2 |
| T5 | Most failures are `cut_off`, or `bad_program` that training on valid programs should have removed | Not a change of recipe: look for a fault first (D11) |
| | None of the above | The owner's choice within D8, recorded as outside the table |

```powershell
# the owner edits config/training.yaml (one value), then:
venv\Scripts\python -m scripts.qlora_iterate propose --finding T2 --reason "accuracy peaks at step 300 and falls"
venv\Scripts\python -m scripts.qlora_iterate status
```

| Id | Check | Passes when |
|---|---|---|
| I1 | Triggered | A selection record exists, made for the current model, prompt, and split, and its verdict is `missed` |
| I2 | Analysed | `error_analysis.json` exists for that record's training run and selected checkpoint |
| I3 | One variable | The settings in `config/training.yaml` differ from the first run's recorded settings in exactly one variable of D8 |
| I4 | The first time | No iteration record names a second run that has completed |
| I5 | The same experiment | Nothing of D9 has changed since the first run: prompt id, split manifest, model commit, reply limit, inference settings |

`propose` writes the iteration record only if all five pass. `status` prints the record,
the second run and its jobs once they exist, and after the second selection checks D19.
Exit codes as elsewhere: 0, 1 a check failed, 2 a config problem.

### 4.4 The launcher and the selection

| Item | Behavior |
|---|---|
| Check J10 | D13. Passes with a note when no selection record exists (a first run), for `smoke`, `score`, `evaluate`, and a resume |
| The second jobs | No new command: `launch train --scope measured --workspace <first training job>`, then `launch score --run <second run> --workspace <first training job>` |
| `select_checkpoint` | D16 to D18: with a selection record for another run of the same experiment and an iteration record that names it, the second run's best is compared with the selected checkpoint. Without the iteration record, a selection over a second run is refused |
| `config/training.yaml` | `selection.noise_examples: 10` |

### 4.5 Tests

No test needs the network, credentials, a GPU, or a paid call.

- Categories: one invented reply per category and per precedence between two that both
  fit; reversed arguments in one step of several; a table operation; a gold program that
  does not parse; a right reply is `right` however it looks; the counts of a run add up to
  its examples.
- Analysis, on the stand-in scored run of v3.04's tests: each part of the report; the
  trend over checkpoints rising, falling, and flat at the noise limit; fixed and broken
  against the untuned run; the finding for T1 to T5 and for none; refused without a
  selection record unless `--check`; the same runs give the same files; **the files hold
  no reply and no example text**, and `--show` writes no file.
- Iteration: each of I1 to I5 passing and failing; alpha following r is one variable and
  alpha changed alone is none; two variables are refused; a variable outside D8 is
  refused; `met` is refused; the record is written only when all pass.
- J10: a first run passes; a second without a record, with a record for other settings,
  and after a completed second run all fail; a resume passes.
- Selection across runs: the second wins, loses, and ties (the first stays); schema 2 and
  the kept first record; `read_selection` on both schemas; `--adapter selected` follows the
  winner; the noise note at 9, 10, and 11 examples; D19 detected both ways.
- Guards as before, and one more: nothing under `reports/` holds a reply.

### 4.6 Documentation

- Runbook section 5: reading the analysis; deciding on an iteration; the run sheet of the
  second pair of jobs; reverting the config when the first run stays.
- `administration/aws_commands_run.md`: the second jobs, as the owner's steps.
- `evaluation_run.md`: the categories, and that they are computed, not judged.
- `coding.md`: categorize failures only through `classify`; a second full run only through
  an iteration record; never change a setting of D9 inside Path A.
- `gotchas.md`: a reversed division is not a sign error; ten examples of 499 are noise.
- Architecture overview; the roadmap of req_004, which can take the classifier for v4.05.

## 5. Out of scope

- A second iteration, a sweep, or a search over settings (req_007).
- Any change to what the arms share (D9). If the analysis says the prompt is at fault,
  that is a finding for the report and a question for the owner about all six arms.
- Categories for the Nova arms' failures as a report (v4.05, v5.03). The classifier works
  on their runs; this version writes no Path B file.
- The test split, confidence intervals, and paired tests (v5.01, v5.02). The noise figure
  of D18 is a yardstick, not a test.
- Reading replies to judge them, or any category that needs a model to assign.
- The optional local full run (v3.06).

## 6. What this hands to later versions

| For | What |
|---|---|
| v4.05 | `classify` and `failure_counts` for the Nova runs; the pattern of registering one change before it is paid for |
| v5.01 | The selection record, schema 2 or 1, and `qlora_iterate status` as part of the freeze: the config is the selected run's |
| v5.02 | Billable seconds of the second pair of jobs, as part of Path A's training cost |
| v5.03 | The categories for all six arms on test; the dev analysis; the iteration record, as a stated deviation from the first recipe |
| req_007 | What one change did and did not move |

## 7. Known limits and follow-ups

- **Written before its trigger can be read.** If the verdict is `met`, only the analysis
  is used, and sections 4.3 and 4.4 are code that never runs for a result. That is the
  price of having it ready.
- **Four versions now rest on a job that has never run.** v3.03's smoke job is the first
  time any of this meets a container. A fault there may reach this version's assumptions
  about workspaces and reuse (D14).
- **One change is one data point.** If the second run is better, the change may have
  helped, or the run may have drawn better: two runs with one seed already differ (v3.02).
  D18 says when the gap is within noise; it cannot say why.
- **Choosing between two runs on dev flatters the winner again**, as choosing among ten
  checkpoints did. Test is the measure.
- **The table of section 4.3 is judgment.** It encodes common practice, not a result on
  this data.
- **A category is the first rule that fits.** A reply with too many steps and reversed
  arguments counts once, as `steps_more`. The counts describe where replies first go
  wrong, not everything wrong with them.
- **`evidence` cannot tell a wrong number from a right number written another way** that
  the program comparison does not treat as equal. Such a reply would usually be right on
  execution and never be classified; the remainder is small in section 2.2.
- **A second run on the first workspace downloads the first run's checkpoints at its
  start**: some minutes of billed time, in exchange for not scoring the untuned model again.

## 8. Decisions for the owner

**Taken as recommended on 2026-11-20**: the owner started the PRP and its execution
without changing any of the five. The table is kept as the record of what was asked.

| # | Question | Recommendation |
|---|---|---|
| 1 | **Build this version now, before the verdict exists, or wait?** | Build the analysis now: it is free, it runs on stored runs today, and v5.03 needs it either way. The iteration tooling comes with it in one PRP, because the selection and the launch check must be tested together. But **run the v3.03 smoke job first if you can**: this would be the third version built on a launcher that has not met a container |
| 2 | **With a `met` verdict, is the iteration forbidden (D7)?** | Yes. If A1 clears 40% and 98%, a second run picked on dev is tuning for the number. If you want it open, the alternative is a flag `--although-met` that the record carries into the report as a deviation |
| 3 | **The variables of D8 and the table of section 4.3** | As written. The one I am least sure of is T3 (rank 16 to 32): it doubles the trainable parameters and may do nothing. Say so if you would rather it read "the owner's choice" |
| 4 | **Does the second run need a margin to replace the first (D16)?** | No margin: more examples right wins, a tie keeps the first, and the record says when the gap is within noise. A required margin of 10 examples is the stricter alternative; it would keep a slightly worse adapter as arm A1 in order to stay with the documented recipe |
| 5 | **The spend**: an iteration is $9 to $21 at laptop speeds, on top of $9 to $14 | Accept it as the ceiling of Path A's tuning. With both, Path A stays under $35 of its $100 cap; J7 and J9 check each job |

## 9. Acceptance

Code, by Claude:

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
  and without the training stack blocked.
- On this machine, free: `qlora_analyze --run 20261119T083236Z-smoke-qwen3-4b-smoke-5477
  --split quick_dev --limit 24 --check` prints, for checkpoint 8: 10 right, 5 `steps_more`,
  4 `order`, 3 `evidence`, 1 `steps_fewer`, 1 `operation`; and writes nothing.
- `failure_counts` on the stored untuned dev runs of Qwen, Nova Pro, and Nova Lite gives
  the figures of section 2.2.
- `qlora_iterate propose`, with no selection record, fails I1 and writes nothing.
- `sagemaker_job plan train` shows J10 passing with the note that no run is selected yet.

Live, by the owner, after v3.04's selection:

```powershell
venv\Scripts\python -m scripts.qlora_analyze --run <first run>          # always: free
git add reports/local_qlora ; git commit -m "Path A: error analysis on dev"
```

Only if the verdict is `missed` (🟡 billed), each command logged in
`administration/aws_commands_run.md`:

```powershell
# edit one value in config/training.yaml
venv\Scripts\python -m scripts.qlora_iterate propose --finding <T1..T4> --reason "<why>"
git add -A ; git commit -m "Path A: one corrective change registered"
venv\Scripts\python -m scripts.sagemaker_job plan train --scope measured --workspace <first training job>
venv\Scripts\python -m scripts.sagemaker_job launch train --scope measured --workspace <first training job> --max-runtime <proposed> --yes
venv\Scripts\python -m scripts.sagemaker_job pull
venv\Scripts\python -m scripts.sagemaker_job launch score --run <second run> --workspace <first training job> --scope measured --yes
venv\Scripts\python -m scripts.sagemaker_job pull
venv\Scripts\python -m scripts.qlora_select --run <second run>
venv\Scripts\python -m scripts.qlora_analyze --run <selected run>
venv\Scripts\python -m scripts.qlora_iterate status
venv\Scripts\python -m scripts.sagemaker_job pull --job <first training job> --checkpoint <N>
venv\Scripts\python -m scripts.sagemaker_job keep --yes
```

- The analysis report is committed and holds no reply.
- If triggered: the iteration record names one variable; the second run completed; the
  selection record is schema 2 and names the winner; `config/training.yaml` is the
  winner's; `qlora_iterate status` passes.
- Section 11, "As built", is added: the category table for A1 on dev; and, if triggered,
  the finding, the change, both runs side by side, the verdict, and the bills.
- The roadmap records how the version ended: analysis only, or iterated; and that Path A
  goes to v5.01.

## 10. Next step

The version is complete (2026-11-21), as analysis only: section 12 has the figures. Path A
goes to v5.01 with checkpoint 500 as arm A1.

## 11. As built (2026-11-20): the code

Built by `prp_003_error_analysis_iteration_v5.md`. **Nothing was created in AWS, no job was
launched, and nothing has triggered this version**: there is still no selection, because no
cloud job has run. This section covers the code. The category table of arm A1 on dev, and
the iteration if one is made, are added after v3.04's jobs.

What was run:

- `venv\Scripts\python -m pytest -q`: 2,437 passed (94 new). With the training stack
  blocked: 2,431 passed, 6 skipped. `pytest -m gpu -q`: 8 passed.
  `venv\Scripts\python -m ruff check .`: clean.
- `scripts.qlora_analyze --run 20261119T083236Z-smoke-qwen3-4b-smoke-5477 --split quick_dev
  --limit 24 --check`, on this machine's stored runs.
- `failure_counts` on the stored untuned dev runs of Qwen, Nova Pro, and Nova Lite.
- `scripts.qlora_iterate propose --finding T1 --reason "rehearsal"` and `status`.
- `scripts.sagemaker_job plan train --scope measured`, signed out.

### The rehearsals

| Command | Result |
|---|---|
| `qlora_analyze --check` on the local smoke run, checkpoint 8 | 10 right, 5 `steps_more`, 4 `order`, 3 `evidence`, 1 `steps_fewer`, 1 `operation`: the figures of section 2.2. Untuned beside it: 19 `bad_program`, 2 `cut_off`. 13 of the 24 examples were right at no checkpoint. Finding T3. L5 fails, as it must for local runs; nothing written |
| `failure_counts` on the stored dev runs | Qwen 3 right, 382 `bad_program`, 44 `not_executable`, 45 `cut_off`, 23 `steps_more`; Nova Pro 190, 168, 21, 3, 72; Nova Lite 123, 212, 24, 2, 100: section 2.2 |
| `qlora_iterate propose` | `FAIL  I1`: no checkpoint has been selected yet; nothing written |
| `qlora_iterate status` | `state: nothing selected` |
| `plan train` | `PASS  J10  Registered: no run is selected yet: this is the first full run`; J1, J2, and J9 fail as before; nothing created |

The finding T3 on the smoke run is the table applied to three checkpoints of a 12-step
adapter on 24 examples. It shows that the reading works, and means nothing.

### The decisions of section 8

Taken as recommended: built now; no iteration after `met`; the five variables and the
table as written; no margin, a tie keeps the first run; the spend accepted as a ceiling.
The smoke job was again not run first.

### Differences from sections 3 and 4

- **The table's values are in `config/training.yaml`** (`iteration.changes`), with the two
  record paths. The spec named only `selection.noise_examples`.
- **`propose` takes T1 to T4 or `other`.** T5 is not a change and is refused as a finding.
  The record also keeps what the analysis itself read, and whether the change is the
  table's for the finding given.
- **"A second run has been made" is read from two places**: the iteration record once the
  selection has compared a second run, and the job record, where a second training job is
  marked `iteration` when it is launched. A second job that failed or was stopped does not
  count: it is resumed, or launched again.
- **J10 also checks the workspace** (D14), and names the first training job's.
- **The selection refuses the first run's id after an iteration**, unless `--check`: the
  second run's id compares both.
- **A second run made with other settings than the registered ones is refused** by the
  selection, not only at launch.
- **`qlora_iterate status` checks the config whenever no second run is pending**, also
  before a change is registered: an edited file that is neither registered nor reverted
  fails it.
- **The analysis that is written is of the selected run.** Another run can be looked at
  with `--check`. When a second run becomes the selected one, the first run's analysis is
  kept as `error_analysis_first.json`, as the first selection record is.
- `run_gold(run_id)` was added to the runner, so that a report can meet a run's lines with
  their gold replies without reaching into the runner's internals. The iteration record's
  reading and writing is `local_qlora/iteration_record.py`, which the selection, the
  launcher, and the checks share.
- The finding's "most failures" is more than half of the selected checkpoint's wrong
  replies; "the first half" is a step at or before half the last step.

## 12. As run (2026-11-21): analysis only

**How the version ended: analysis only.** v3.04's selection gave the verdict `met` (71.74%
on dev against the 40% asked, format validity 100.00% against 98%), so the corrective
iteration was not triggered: no value was changed, no second run was made, and nothing was
billed. Path A goes to v5.01 with checkpoint 500 as arm A1.

The owner ran

```powershell
venv\Scripts\python -m scripts.qlora_analyze --run 20261120T182710Z-train-qwen3-4b-fit-4557
```

which passed L1 to L6 and wrote `reports/local_qlora/error_analysis.md` and `.json`. It
makes no AWS call. The report holds counts and ids, and no reply.

### Arm A1 on dev: why replies are wrong

Checkpoint 500: 358 of 499 right; all 499 in the format; every reply ended at the end token.

| Category | Checkpoint 500 | Share of all 499 | Untuned model |
|---|---:|---:|---:|
| Cut off at the token limit | 0 | 0.0% | 45 |
| Program does not parse | 0 | 0.0% | 383 |
| Program cannot be run | 2 | 0.4% | 39 |
| More steps than the gold program | 21 | 4.2% | 24 |
| Fewer steps than the gold program | 65 | 13.0% | 2 |
| Another operation | 12 | 2.4% | 0 |
| Other numbers, same operations | 34 | 6.8% | 0 |
| Arguments reversed | 7 | 1.4% | 0 |
| **Right** | **358** | 71.7% | 6 |

- Against the untuned model: 354 examples fixed, 2 broken.
- The 141 wrong replies are all in the format. What is left is reasoning, not form: 139 of
  them are a program that runs and gives another result.
- **By the length of the gold program**: one step 232 of 286 (81.1%); two steps 112 of 160
  (70.0%); three steps 7 of 35 (20.0%); four or more 7 of 18 (38.9%).
- **By operation in the gold program**: `multiply` is the weakest of the common ones, 18
  of 59 (30.5%); `add`, `subtract`, and `divide` are between 63% and 71%.
- **By prompt length**: 81.5% right in the shortest quarter, 63.2% in the longest.
- Of the wrong results, 31 are the gold value off by a factor and 10 are it negated.
- 100 examples were wrong at every checkpoint.
- The last three checkpoints are flat, and accuracy was first within noise of its best at
  step 400.

### The finding

**T3**: accuracy is flat from the middle on and most failures are calculation. The
table's change would be `lora.r` to 32. It was not made: a corrective iteration is allowed
only on a `missed` verdict. The finding is kept for v5.03 and for any later work on Path A.

### Section 9's points

| Point | Result |
|---|---|
| The analysis report is committed and holds no reply | Written; the commit is the owner's and is pending |
| If triggered: the iteration record, the second run, the selection of schema 2 | Not triggered |
| The roadmap records how the version ended | Analysis only; Path A goes to v5.01 |

**Left unproven**, because the iteration never ran: `qlora_iterate propose`, launch check
J10 against a registered change, a second training run on the first job's workspace, and
the selection across two runs. They are tested on stand-ins only.
