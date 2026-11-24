# DSL Evaluation

Spec Group      :   req_002_dsl_evaluation  
Feature Name    :   dsl_evaluation  
Version         :   3  
Status          :   ✅ Complete 2026-11-16; tests and lint green locally; marked complete by the owner (section 8 taken as recommended; built by `prp_002_evaluation_runner_v3.md`; results in section 11)  
Last updated    :   2026-11-16  

## 1. v2.03 — Evaluation runner

v2.01 runs a reply and v2.02 says whether it is right. This version asks a model for the
replies: it takes the examples of a split, hands each prompt to a **predictor**, stores what
came back, scores it, and adds the totals up. All six arms go through this one piece of code.

It adds:

- the **`Predictor` protocol**: prompt in; reply text, token counts, and latency out;
- the **runner**: one split, one predictor, a predictions file written as it goes, and a
  run in the registry with its usage in the ledger;
- **resuming** after an interruption without repeating or skipping an example;
- a **summary** of a finished or unfinished run, recomputed from the stored predictions;
- a **fake predictor**, which makes the whole path testable without a model.

It contains no model. The Qwen backend is v3.01 and the Bedrock adapter is v4.01; each
supplies a predictor and nothing else. Reaching this version's acceptance is milestone M3.

**Depends on**: v2.02, v1.03, v0.04 (all built). It also reads the reply limit of v1.04.

## 2. What exists today (verified 2026-11-16)

### 2.1 What earlier versions provide

| Item | State |
|---|---|
| `load_split(name)`, `serialize_split(name)` | Examples and canonical records of `fit`, `dev`, `quick_dev`, in dataset order. `test` raises `TestLockedError` until req_005 v5.01 |
| `CanonicalRecord` | `example_id`, `prompt_id`, `system`, `user`, `target` |
| `run_reply(text, table)` | A `ReplyOutcome` for any string; `to_dict()` gives plain values |
| `Gold.from_target`, `score_reply`, `summarize`, `load_gold_flags` | One `ExampleScore` per reply and totals over all and over unflagged examples (v2.02) |
| `config/evaluation.yaml` | Execution and answer accuracy: absolute 0.00001 or relative 0.5% (the owner's decision, 2026-11-16) |
| `config/token_budget.yaml` | `max_new_tokens: 192`; the longest correct reply is 80 tokens |
| `start_run(...)` | A run folder under `artifacts/runs/<run_id>/`, a manifest, config copies, a line in the committed runs index. **A run finishes once**: it cannot be reopened |
| `run.record(...)` | One ledger record per run and phase. A token count needs `model_id`; a record needs at least one quantity; dollars only with source `bill` |
| `scope="measured"` | Refuses to start from uncommitted changes |
| `ledger/` | Holds a README only: **no run has been recorded yet**. `artifacts/runs/` does not exist yet |
| `qlora_finetuning/evaluation/` | `metrics.py` only. `qlora_finetuning.data.gold_check` imports it, and it must not import the data package |
| Plan | "`Predictor` protocol (prompt in; text, token counts, latency out); runner over a named split; resumable predictions JSONL with raw output, parsed program, executed value, per-example scores; writes to the run registry and cost ledger; a fake predictor for tests" |

### 2.2 A trial of the loop

A throwaway loop over quick-dev (200) and dev (499), with made-up predictors that derive a
reply from the gold one. Counts only; the test split was not read.

| Reply given | Format valid | Execution | Answer | Program match | Note |
|---|---|---|---|---|---|
| The gold reply | 499 | 499 | 499 | 499 | |
| The gold reply with `program:` in lower case | 0 | 499 | 499 | 499 | deviation `label_case` 499 times |
| A sentence, then the gold reply | 0 | 499 | 499 | 499 | deviation `text_before` 499 times |
| Nothing | 0 | 0 | 0 | 0 | error `reply_empty` 499 times |

Quick-dev gives the same pattern with 200.

| Fact | Value |
|---|---|
| Running and scoring 499 replies | 0.03 seconds. The model's time is all that matters |
| One prediction line (reply, outcome, score, usage) | about 825 bytes |
| So a predictions file | 0.2 MB for quick-dev, 0.4 MB for dev, 2.4 MB for test |
| Longest user text in dev | 1,763 characters; longest target 145 |

So the predictions file can hold everything about each reply, a run can be summarized again
from its file in well under a second, and resuming is worth having only for the model's
time and money, which is where it must be exact.

## 3. Locked decisions

### The predictor

| # | Decision | Reason |
|---|---|---|
| D1 | A predictor has one method: `predict(requests) -> predictions`, **a list in, a list of the same length out, in the same order** | A local GPU and a hosted endpoint both gain from batches; the runner must not care how |
| D2 | A request holds `example_id`, `system`, `user`, and `max_new_tokens`. The two texts are the canonical record's, unchanged. The predictor wraps them in its model's format and adds no word | v1.03: every arm reads the same words |
| D3 | A prediction holds `text` (the reply, with whatever is specific to the model already removed), `input_tokens`, `output_tokens` (each a whole number or unknown), `latency_seconds`, and `stop_reason` (`end`, `length`, or `other`) | Plan; v2.01: a backend removes its model's own markers before the parser sees the text |
| D4 | A predictor describes itself: `name`, `model_id`, `settings` (decoding and anything else that changes a reply), `batch_size`, and where its token counts come from (`measured`, `aws_api`, or `computed`) | The run manifest and the resume check need it; the ledger needs a truthful source |
| D5 | The request's `example_id` is for logs and for the fake predictor. A real predictor must not let it change the reply | An arm must answer from the prompt alone |
| D6 | The runner checks what comes back: the right number of predictions, `text` a string, counts whole and not negative. Anything else is a predictor failure (D9) | A broken backend must not produce scores |

### The run

| # | Decision | Reason |
|---|---|---|
| D7 | One invocation is one run in the registry (kind `eval`; `smoke` for a fake predictor). Its folder holds `evaluation.json` (what was asked), `predictions.jsonl`, and `summary.json` | v0.04 convention |
| D8 | Examples are predicted in dataset order, in batches of the predictor's size. After each batch the lines are written and flushed before the next batch starts | An interruption loses at most one batch |
| D9 | **A predictor failure stops the run.** The run is marked failed, everything predicted so far is kept, the usage so far is recorded, and the run can be resumed. An example is never scored as wrong because the infrastructure failed | A throttled endpoint or a full GPU says nothing about the model. **Decision for the owner in section 8** |
| D10 | A reply the model did give is always scored, however bad: empty text, a reply cut off at the limit (`stop_reason: length`), text in the wrong format | v2.02 D16: nothing leaves a denominator |
| D11 | The split is named: `quick_dev`, `dev`, `fit`, or `test`. The runner calls `load_split` and never catches `TestLockedError` | The lock is the loader's; the runner adds no second way in |
| D12 | `limit=N` predicts the first N examples only. Such a run is **partial**: its summary says so and it is never listed as a result (D21) | For a first look at a new backend |
| D13 | The reply limit is `max_new_tokens` of `config/token_budget.yaml`, passed in every request | One limit for every arm |
| D14 | The usage of a run is one ledger record: examples predicted **in this run**, wall-clock seconds, and the token totals when the predictor reports them. It is written whether the run completes or fails | A resumed evaluation is several runs; their records must add up to what was consumed, once |

### The predictions file

| # | Decision | Reason |
|---|---|---|
| D15 | One JSON line per example: `example_id`, its position, `prompt_id`, the run that predicted it, the reply text as returned, the outcome of `run_reply` (program text, canonical program, **unrounded value**, answer text, strict flag, deviations, notes, error code), the `ExampleScore`, token counts, latency, stop reason | Plan. v2.02 D6: with the unrounded value stored, a changed allowance needs no model run |
| D16 | The file holds **no prompt text and no gold text**. The prompt is the prompt id plus the example id; the gold reply comes from the serializer again | The file stays small, and a test-split file holds nothing of the test set but what the model wrote |
| D17 | Predictions live under git-ignored `artifacts/` and are never committed | A reply can repeat dataset text |
| D18 | `evaluation.json` records what decides a reply: split, manifest hash, prompt id, `max_new_tokens`, and the predictor's name, model id, and settings. It also records the limit, if any | The resume check (D19) |

### Resuming

| # | Decision | Reason |
|---|---|---|
| D19 | `resume=<run_id>` starts a **new run** that continues an earlier one. It is refused unless everything in D18 is equal | A run cannot be reopened (section 2.1), and predictions made under other settings are another experiment |
| D20 | The earlier file must be a clean beginning of the split: its ids are the split's first ids, in order, each once. A last line that is cut off is dropped; any other damage is an error. The valid lines are copied into the new run's folder, which names the run it continues | Never repeat or skip an example; never guess at a damaged file |

### The summary

| # | Decision | Reason |
|---|---|---|
| D21 | The summary is computed **from the predictions file**, not from memory: stored outcomes are scored again under the current `config/evaluation.yaml` and totalled with `summarize(scores, flagged=load_gold_flags())` | One code path for a fresh run, a resumed run, and a run looked at again later |
| D22 | It holds the totals of v2.02 for all and for unflagged examples, plus: examples predicted and expected, whether the run is complete or partial, token totals, latency (mean, median, 95th percentile), and counts by stop reason | What v3.01 and v4.01 report for a baseline |
| D23 | A run is **complete** when checks R1 to R5 (section 4.4) pass. Only a complete run is a result | A number over 480 of 499 examples must never be read as the dev score |
| D24 | The summary also gives execution accuracy under the **tight rule** (absolute 0.00001 or relative 0.001%) as a diagnostic beside the headline | The owner chose 0.5%; v2.02 measured that it lets ten times as many wrong-operand programs through. **Decision for the owner in section 8** |
| D25 | Each complete run adds one line to the committed `ledger/evaluations.jsonl`: run id, arm, split, examples, the four counts, the tight count, tokens, the manifest hash, and the prompt id. Counts only | Results must survive the loss of a git-ignored folder. **Decision for the owner in section 8** |

### The fake predictor

| # | Decision | Reason |
|---|---|---|
| D26 | `FakePredictor` answers from a table of replies by example id, or from a function of the request. It can be told to fail after N requests, and it counts a token per word | Every path of the runner, including the interruption, without a model |
| D27 | The script offers one fake predictor by name, `gold`: each example's own gold reply. On any split it must score every example right | Plan, milestone M3 |
| D28 | A fake run is recorded like any other, as kind `smoke` with path and arm `none` and no token counts | The plan asks that the runner be shown writing to the registry and the ledger |

## 4. Scope to build

### 4.1 `config/evaluation.yaml`

`version: 2`. New keys: `diagnostic.execution_relative: 0.00001` (D24),
`runner.progress_every: 50` (a log line every so many examples),
`results_file: ledger/evaluations.jsonl` (D25). The allowances do not change, so the gold
flags file stays valid; `load_evaluation_config` reads the new keys.

### 4.2 Predictor: `qlora_finetuning/evaluation/predictor.py`

| Item | Behavior |
|---|---|
| `PredictionRequest` | D2 |
| `Prediction` | D3 |
| `Predictor` | A protocol: `name`, `model_id`, `settings`, `batch_size`, `usage_source`, `predict(requests)` |
| `PredictorError` | A predictor could not answer. Carries a short reason; a backend raises it after its own retries |
| `check_predictions(requests, predictions)` | D6 |
| `describe(predictor)` | The plain values of D4, for the manifest and for D18 |

### 4.3 Fake: `qlora_finetuning/evaluation/fake.py`

`FakePredictor(replies, name="fake", batch_size=1, fail_after=None, latency_seconds=0.0)`
(D26) and `gold_predictor(records)`, which builds one from canonical records (D27).

### 4.4 Predictions and runner

`qlora_finetuning/evaluation/predictions.py`:

| Item | Behavior |
|---|---|
| `PredictionRecord` | D15; `to_dict()` and `from_dict()` |
| `append_records(path, records)` | Writes lines and flushes |
| `read_predictions(path, expected_ids)` | D20: the valid beginning, or an error naming the line |

`qlora_finetuning/evaluation/runner.py`:

| Item | Behavior |
|---|---|
| `run_evaluation(predictor, split, *, label, path, arm, where, scope, limit=None, resume=None, ...)` | D7 to D14, D19, D20. Returns the summary |
| `summarize_run(run_id)` | D21 to D24. Works on any run folder, finished or not |

Usage, with the fake predictor:

```python
from qlora_finetuning.data import serialize_split
from qlora_finetuning.evaluation.fake import gold_predictor
from qlora_finetuning.evaluation.runner import run_evaluation, summarize_run

predictor = gold_predictor(serialize_split("quick_dev"))
summary = run_evaluation(predictor, "quick_dev", label="gold quick-dev",
                         path="none", arm="none", where="local", scope="dev")
summary["complete"]                                   # True
summary["all"]["execution_accuracy"]                  # {"count": 200, "rate": 1.0}
summary["diagnostic"]["execution_accuracy_tight"]     # {"count": 200, "rate": 1.0}
summary["run_id"]                                     # where the predictions file is

summarize_run(summary["run_id"])                      # the same totals, from the file
```

| Id | Check | Passes when |
|---|---|---|
| R1 | Complete | The file has one line for every example asked for, in dataset order: the whole split, or with `limit` its first N. As built: a limited run can pass its checks; it is marked partial, and only a run that passes R1 to R5 **and is not partial** is a result (`result` in the summary) |
| R2 | Same question | Every line's prompt id is the current one, and the manifest hash is the run's |
| R3 | Scores current | Every stored score equals the score computed again from the stored outcome |
| R4 | Usage adds up | For every run that predicted a line of the file, the examples in its ledger records are at least its lines. As built: fewer means usage went unrecorded and fails; more is possible and truthful (a prediction paid for and then lost with a cut-off line, so paid for again) |
| R5 | Nothing unscored | Every line has an outcome and a score; no line records a predictor failure |

### 4.5 `scripts/evaluate.py`

| Command | Behavior |
|---|---|
| `venv\Scripts\python -m scripts.evaluate --predictor gold --split quick_dev` | Runs, prints the check lines and the four rates, and where the files are |
| `... --limit 20` | A partial run (D12) |
| `... --resume <run_id>` | Continues an interrupted run (D19) |
| `venv\Scripts\python -m scripts.evaluate --summary <run_id>` | Prints a run's summary again from its file; predicts nothing |

`--predictor` takes a name from a table in the script. v2.03 registers `gold`; v3.01 and
v4.01 add theirs, each imported only when named. `--scope` is `dev` unless `measured` is
given. Exit codes: 0 a complete run whose checks pass (or a partial run that was asked for);
1 a check failed or the predictor failed; 2 a config problem, a locked split, or a resume
that is refused.

### 4.6 Tests

On invented rows and the fake predictor. No test needs the network, a GPU, or the training
stack, and every test writes under a temporary root.

- Protocol: a request carries the record's texts character for character; each malformed
  prediction (wrong count, text not a string, negative or fractional tokens) is a predictor
  failure.
- Fake predictor: by table and by function; batches; fails after N; token counts.
- Runner with the gold predictor: every example right; one line per example in order; the
  manifest names the split hash, the prompt id, and the configs; one ledger record.
- Runner with replies that are wrong in each way of section 2.2: the totals are as expected,
  and nothing is left out of a denominator.
- **Interruption**: fail after N at several N, including 0, mid-batch, and the last example;
  resume; the final file equals the file of an uninterrupted run except for run ids; the
  ledger records add up; the predictor is never asked twice for an example.
- Resume refused: another split, prompt, predictor setting, or reply limit; a file with a
  repeated id, an id out of order, or a damaged line in the middle. A cut-off last line is
  dropped and the example predicted again.
- Limit: a partial run is marked partial, exits 0, and adds no line to the results file.
- Summary: equal to `summarize` over the same scores; recomputed under a changed allowance
  without a predictor; the tight diagnostic differs from the headline on a reply built to
  sit between the two rules; latency statistics on known values; unknown token counts.
- Checks R1 to R5: each passes and fails.
- Test lock: asking for `test` raises `TestLockedError` before the predictor is called; with
  an unlock record in the temporary root it runs.
- Script: exit codes; `--summary`; `--resume`; an unknown predictor name; module form and
  by-path hint.
- Guards: `qlora_finetuning.evaluation` still loads without the data package (the runner is
  imported by its own module name, never from the package's `__init__`); no file under
  `evaluation/` imports `local_qlora` or `frontier`; no predictions file is written outside
  the run folder.
- On the real data when the snapshot is present: the gold predictor on quick-dev and dev
  scores 200 of 200 and 499 of 499, under a temporary tracking root.

### 4.7 Documentation

- Runbook section 9: run an evaluation, resume one, read a summary, what "partial" means.
- A new project-pattern note, `evaluation_run.md`: how a backend supplies a predictor and how
  an entry point calls the runner. `reply_scoring.md` and `run_tracking.md` point to it.
- `coding.md`: a model is evaluated only through `run_evaluation`; a backend implements
  `Predictor` and nothing of the loop.
- `gotchas.md`: a run cannot be reopened; the runner is not exported from the package.
- Architecture overview: the predictor interface as built.
- `ledger/README.md`: the results file.

## 5. Out of scope

- Any model, tokenizer, or AWS call (v3.01, v4.01). Retries and throttling belong to the
  backend, inside `predict`.
- Running as a SageMaker job, and the instance cost of one (v3.03).
- Unlocking the test split (v5.01).
- Confidence intervals and paired tests (v5.02); breakdowns by operation or length (v5.03).
- Running examples in parallel from the runner. A backend may do so inside one batch.
- Storing prompts in the predictions file, or committing predictions.

## 6. What this hands to later versions

| For | What |
|---|---|
| v3.01 | Implement `Predictor` for Qwen3-4B and register it; arm A0 on quick-dev is one command. Peak memory and seconds per example come from the summary's latency and the backend's own notes |
| v3.03, v3.04 | The same `run_evaluation` inside a SageMaker job; `--resume` after a lost job |
| v4.01 | Implement `Predictor` over Bedrock; token counts with source `aws_api`; throttling raises `PredictorError` only after the backend's retries |
| v4.04 | Arms B1 on dev with the same command |
| v5.01 | Six runs on `test`, each complete by R1 to R5, each a line in the results file |
| v5.02 | Per-example true or false for every metric and arm, from the predictions files |
| v5.03 | Error codes, deviations, mismatch labels, stop reasons, and latency per example |
| v6.01 | The demo uses a predictor directly, with `run_reply`; it does not need the runner |

## 7. Known limits and follow-ups

- **Predictions are not backed up.** They are in a git-ignored folder on one machine. The
  results file (D25) keeps the counts; the replies themselves are lost with the folder.
  Copying run folders to S3 is v3.03's.
- **A batch is all or nothing.** If a backend fails on the fifth request of eight, the first
  four are predicted again on resume, and their tokens are paid twice. Small next to a lost
  run; a backend that wants better can use a batch size of 1.
- **Latency is what the predictor reports.** For a batch, it is the backend's own division of
  the batch's time. Local and hosted latency are not comparable without that in mind.
- **The resume check trusts the predictor's description.** A backend that leaves a setting
  out of `settings` can be resumed under a different value without notice.
- **A fake run leaves lines in the committed ledger** (D28): the first lines it will ever
  hold. They carry no tokens and no cost.
- **The tight diagnostic is not a second headline** (D24). It is there to show how much of an
  arm's score rests on the wider allowance.
- **The figures in section 2.2 come from a throwaway script.** The built runner must
  reproduce the first row; any difference is investigated first.

## 8. Decisions for the owner

**Taken as recommended on 2026-11-16**: the owner started the PRP and its execution
without changing any of the three. (1) A predictor failure stops the run and it is
resumed. (2) Every summary shows execution accuracy under the tight rule beside the
headline. (3) Each run that is a result adds one line to `ledger/evaluations.jsonl`.
The table is kept as the record of what was asked.

| # | Question | Recommendation |
|---|---|---|
| 1 | **When a predictor cannot answer** (throttling after its retries, a GPU error): stop the run and resume later (D9), or record the example as wrong and go on? | Stop and resume. Scoring a failed call as a wrong answer would make an arm's accuracy depend on how busy the endpoint was that day. The cost is that a run can need more than one attempt |
| 2 | **Show execution accuracy under the tight rule beside the headline** in every summary (D24)? | Yes. It costs nothing, since every prediction keeps its unrounded value, and it shows at a glance whether the 0.5% allowance is doing any work for an arm. The headline stays the 0.5% figure you chose |
| 3 | **Commit one line of counts per complete run** in `ledger/evaluations.jsonl` (D25), or keep results only in the git-ignored run folders? | Commit the line. It holds counts and ids of runs, no example text, and it is the only copy of a result that survives a lost folder. The alternative keeps the repository quieter but leaves baseline numbers in a place git does not track |

## 9. Acceptance

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with and
  without the training stack blocked.
- `venv\Scripts\python -m scripts.evaluate --predictor gold --split dev` completes: 499 of
  499 on all four metrics, checks R1 to R5 pass, a predictions file of 499 lines, a run in
  the registry, and one ledger record.
- The same on `quick_dev`: 200 of 200.
- An interrupted run, resumed, gives the same predictions as an uninterrupted one, and its
  ledger records add up to the number of examples.
- `--summary <run_id>` reproduces a run's totals from its file.
- Asking for `test` fails with the lock's message and predicts nothing.
- Milestone M3 (harness trusted) is recorded in the execution plan.

## 10. Next step

Built. req_002 is complete with this version; milestone M3 (harness trusted) is reached. The
next versions are v3.01 (Qwen backend, arm A0) and v4.01 (Bedrock adapter, arms B0).

## 11. As built (2026-11-16)

`venv\Scripts\python -m scripts.evaluate --predictor gold --split <name>`:

| | quick-dev | dev |
|---|---:|---:|
| Examples | 200 | 499 |
| Format validity | 200 | 499 |
| Execution accuracy | 200 | 499 |
| Answer accuracy | 200 | 499 |
| Program match | 200 | 499 |
| Execution accuracy, tight rule (diagnostic) | 200 | 499 |
| Checks R1 to R5 | pass | pass |
| Wall clock | 0.18 s | 0.49 s |

- These two runs wrote the first lines of the committed ledger: two runs, two usage records
  (no tokens, no cost), and two lines in `ledger/evaluations.jsonl`. They are kind `smoke`.
- `--summary <run_id>` reproduces a run's output from its file, and asking for `test`
  prints the lock's message and exits 2 with nothing predicted.
- Section 2.2's first row is reproduced by the built runner.

Differences from sections 3 and 4:

- **R1 and R4** as written into section 4.4 above.
- **The usage of a killed run is rebuilt when it is resumed.** A process that is killed
  writes no ledger record, so check R4 could never pass for that evaluation. The resuming
  run appends a record for the earlier run from its lines, marked `computed`, with no wall
  clock (it is not known).
- **A line that `run_reply` could not have written** (neither a value nor an error) counts as
  wrong on everything and fails R5; it is never scored as it stands.
- **Token counts reach the ledger only when the predictor names a model** and every count of
  the run is known. That is the ledger's own rule, and it is how a fake run records none
  (D28).
- **The resume check covers the predictor's name, model id, and settings, not its batch
  size.** A backend whose replies depend on the batch size puts it in `settings`.
- **`PredictorEntry`** in the script carries the kind, path, and arm with the factory, so
  `--predictor gold` is recorded as a smoke run without more flags.
- `run_evaluation` takes one `context` argument (`RunnerContext`) for everything a test
  replaces, in place of the `...` of section 4.4.
