# Comparison

Spec Group      :   req_005_comparison  
Feature Name    :   comparison  
Version         :   3  
Status          :   ✅ Complete 2026-11-24: code built 2026-11-23 (`prp_005_comparison_report_v3.md`, section 11); rebuilt by the owner on the reconciled cost record on 2026-11-24; R1 to R8 pass with no reservation; report committed (`3dbf2b2`): milestone M8  
Last updated    :   2026-11-24  

## 1. v5.03 — Comparison report

The six arms have answered the test split (v5.01) and each has a price (v5.02). This
version writes the answer to the project's question: how the two paths compare on accuracy
and on cost, how sure the comparison is, and where each model goes wrong.

It adds three things that do not exist yet:

1. **How sure each figure is.** A confidence interval for every accuracy figure, and a
   paired test for each comparison between two arms.
2. **Where each arm goes wrong.** The thirteen error categories of v3.05, counted for all
   six arms on the test split.
3. **The report itself.** One Markdown file with the accuracy table, the cost table, an
   accuracy-versus-cost chart, the error categories, who holds the tuned model, the
   caveats, and the deviations from the plan. A short summary goes into the README.

It calls no AWS service and starts nothing billed. It runs no model: every figure comes from
the six stored test runs, the six-arm record, and the cost record.

**Depends on**: v5.02 (🟡: built, its record provisional until the bill is entered). The
bill entries change only the reconciliation table of the cost record, so this version can be
built now. A report built on a provisional cost record is itself marked provisional
(decision 6).

Committing the report is milestone **M8**.

## 2. What exists today (verified 2026-11-23)

### 2.1 The inputs

| What | Where | Committed |
|---|---|---|
| The six test runs: per arm, the run id and the counts of the four metrics | `reports/comparison/six_arm_test_runs.json` | Yes |
| Every reply of the six runs, 2,898 lines each, with its stored score | `artifacts/runs/<run id>/predictions.jsonl` | No: replies are kept on this machine and in S3 |
| Every dollar figure | `reports/comparison/cost_model.json` | Yes |
| Dev results of the arms | `reports/local_qlora/adapter_selection.json`, `reports/frontier/nova_baseline_report.json`, `reports/frontier/nova_tuned_report.json` | Yes |
| The frozen configuration | `splits/config_freeze.json` | Yes |

All six prediction files are on this machine today, and hold the same 2,898 example ids in
the same order. The test split has examples from **43 companies**.

### 2.2 Code this version uses

| Code | What it gives |
|---|---|
| `evaluation/failures.py`: `run_failures`, `count_failures`, `CATEGORIES` | One computed category per reply, from the stored reply and the gold program. Written for A1 on dev (v3.05); it works on any run |
| `evaluation/failures.py`: `company_of` | The company of an example, from its id |
| `comparison/runs.py`, `comparison/cost_report.py` | The pattern for a checked record written as JSON and Markdown |
| `scripts/qlora_analyze.py` | The pattern for an error table per run |

### 2.3 What is missing

- No code computes a confidence interval or a test between two arms.
- No code draws a chart. `matplotlib` and `scipy` are not installed and are not project
  dependencies; `numpy` and `pandas` are.
- The README has one section, on the demo. It says nothing about results.
- There is no list of the project's deviations from the plan in one place. They are spread
  over the roadmaps' version logs.

### 2.4 What the comparison looks like, by hand

Counted from the stored scores on 2026-11-23, to show the size of things. The built report
replaces these; none is a result yet.

Right on execution, of 2,898: A0 27, A1 2,121, B0-Lite 484, B1-Lite 2,069, B0-Pro 1,025,
B1-Pro 2,179.

| First arm, second arm | Both right | Only the first | Only the second | Both wrong | Difference | Exact test, p |
|---|---:|---:|---:|---:|---:|---:|
| A1, A0 | 25 | 2,096 | 2 | 775 | +72.26 points | below 0.0001 |
| B1-Lite, B0-Lite | 452 | 1,617 | 32 | 797 | +54.69 | below 0.0001 |
| B1-Pro, B0-Pro | 974 | 1,205 | 51 | 668 | +39.82 | below 0.0001 |
| A1, B1-Pro | 1,958 | 163 | 221 | 556 | −2.00 | 0.0036 |
| A1, B1-Lite | 1,883 | 238 | 186 | 591 | +1.79 | 0.013 |
| B1-Pro, B1-Lite | 1,934 | 245 | 135 | 584 | +3.80 | below 0.0001 |

What this already shows:

- **Tuning is the large effect** in all three models: 40 to 72 points.
- **The three tuned arms are within four points of each other**, and each difference
  passes the paired test when every question is treated as independent. A rough interval
  for A1 minus tuned Nova Pro is −3.3 to −0.7 points.
- **The questions are not independent.** They come from 43 companies, and questions about
  one company's report resemble each other. An interval that resamples companies will be
  wider, and the two-point differences may not survive it (decision 2).
- **The tuned arms fail on different questions.** 1,807 questions are right in all three
  tuned arms, 2,401 in at least one. 467 questions are right in none of the six arms.

## 3. Locked decisions

### The statistics

| # | Decision | Reason |
|---|---|---|
| D1 | Every figure is computed from the six runs the six-arm record names, by rescoring the stored replies. The counts must equal the six-arm record's, or nothing is written | One source for the runs (v5.01 D27). A drifted score would be caught |
| D2 | A confidence interval is a **paired percentile bootstrap**: the examples are resampled with replacement, the same resample is used for all six arms, and the interval is the 2.5th to 97.5th percentile. Resamples, seed, and level are in `config/report.yaml` (proposed: 10,000, the project's seed 20261115, 95%) | The plan names the bootstrap. One resample for all arms keeps a difference between two arms paired |
| D3 | Intervals are given for the four metrics of each arm, and for the difference in execution accuracy of each tested pair | The primary metric gets the comparisons; the others are shown with their intervals only |
| D4 | A second interval resamples **companies** instead of examples, and is shown beside the first for execution accuracy and its differences (decision 2) | The split was made by company, so the examples of one company are not independent |
| D5 | A test between two arms is **McNemar's exact test**: the two-sided binomial test on the questions that exactly one of the two arms got right. It is computed with whole numbers, with no approximation and no new dependency | Both arms answered the same questions. The exact form needs no assumption about the count |
| D6 | The tested pairs are named in `config/report.yaml` before any test is computed (proposed: the six of section 2.4). The p-values of that family are adjusted with **Holm's method**, and both the raw and the adjusted value are shown | Six tests on one data set. Holm controls the chance of any false finding and assumes nothing about how the tests depend on each other |
| D7 | The report says that the pairs were named **after** the counts of the six-arm record were known | That is what happened. A reader should know the tests were not planned blind |
| D8 | A p-value below 0.0001 is shown as "below 0.0001". No figure is called "significant" without its interval and its p-value beside it | A bare verdict hides the size of the effect |
| D9 | The same seed and the same replies give the same record, byte for byte, on any machine | The record is committed and must be reproducible |

### The error analysis

| # | Decision | Reason |
|---|---|---|
| D10 | Each wrong reply of each arm gets one category from `run_failures`, the code of v3.05, unchanged. No new category is added and none is assigned by reading | The categories are the same for every arm only if one code assigns them |
| D11 | The report also gives execution accuracy by the number of steps in the gold program (1, 2, 3 or more), per arm | It shows whether an arm fails on long programs or on all of them |
| D12 | A committed file holds counts and example ids, never a question, a passage, a table, a gold program, or a reply (decision 3) | The project's rule. Every example here is a test example |
| D12a | The categories need each test example's gold program, so this version **reads test rows on this machine**. It is not a test run: no model answers, nothing is uploaded, and no frozen file changes. The execution plan's principle 6 ("only v5.01 loads test rows") is amended to name this reading (decision 9) | A wrong program can only be compared with the gold one. The principle protects against tuning on test, which cannot happen after the record is written |

### The costs and the chart

| # | Decision | Reason |
|---|---|---|
| D13 | Every dollar figure is read from `cost_model.json`. This version multiplies no quantity by a price | v5.02 D18 |
| D14 | The chart plots the six arms: cost per 1,000 questions across (logarithmic, because the figures span a factor of 14), execution accuracy up, each point with its interval. It is an SVG file written by the project's own code (decision 4) | Deterministic bytes, readable on GitHub, no new dependency |
| D15 | A Qwen arm is plotted at its cost as run, the batch job that was paid for. The report says beside the chart that an always-on endpoint costs more per question at low use, and points to the serving table | One point per arm; the caveat is the cost record's |

### The report

| # | Decision | Reason |
|---|---|---|
| D16 | The report is built from a committed template with named placeholders. Every table, every figure in a sentence, and the chart are filled in by the build. The template itself holds no percentage and no dollar amount | A number typed by hand goes stale when a record changes |
| D17 | Models are named by arm and base model (`A1`, Qwen3-4B with QLoRA). No deployment ARN, account number, name, or address appears | The report is public material |
| D18 | The report states its findings and the conditions under which each path is the better choice. It names no single winner (decision 7) | The answer depends on volume, on the accuracy needed, and on who must hold the model |
| D19 | Nothing is written unless checks R1 to R8 pass. R1 may pass with a reservation (a provisional cost record); the report then carries **provisional** in its title | The same rule as the cost record |
| D20 | The README gets one generated block between two marker comments: the six-arm table, three sentences, and a link to the report. Nothing outside the markers is touched | The README is also edited by req_006 |

## 4. Scope to build

New code under `qlora_finetuning/comparison/`, split by concern:

| Module | Holds |
|---|---|
| `report_config.py` | The loader of `config/report.yaml` |
| `statistics.py` | The bootstrap, by example and by company; the exact test; Holm's method. Pure functions on lists of true and false; they know nothing of runs |
| `outcomes.py` | Reads the six runs: per example and arm, the four scores, the category, the gold step count, the company. Counts only; no text leaves it |
| `chart.py` | The SVG chart |
| `report_record.py` | Checks R1 to R8 and the statistics record |
| `report_build.py` | The template, the report, the README block |

One script, `scripts/comparison_report.py`.

### 4.1 The commands

```powershell
venv\Scripts\python -m scripts.comparison_report check      # R1 to R8 and the main figures; writes nothing
venv\Scripts\python -m scripts.comparison_report build      # writes the record, the chart, the report, the README block
venv\Scripts\python -m scripts.comparison_report examples   # writes the local file of example text; never committed
```

None calls AWS or needs a sign-in. `build` takes under a minute.

### 4.2 What is written

| File | Content | Committed |
|---|---|---|
| `reports/comparison/comparison_statistics.json` and `.md` | Every interval, every pair's four counts and p-values, the category counts, accuracy by steps | Yes |
| `reports/comparison/accuracy_cost.svg` | The chart | Yes |
| `reports/comparison_report.md` | The report | Yes |
| `README.md`, between the markers | The summary | Yes |
| `artifacts/comparison/error_examples.md` | For each arm and category, a few example ids with the question, the gold program, and the reply | **No**: git-ignored, for the owner to read |

### 4.3 The report's sections

1. **Summary**: the six-arm table and the findings in a few sentences.
2. **What was compared**: the task, the six arms, the split (4,884 to fit, 499 dev, 2,898
   test, by company), the single test pass, the frozen configuration and its hash.
3. **Accuracy**: the four metrics per arm with intervals; dev beside test for each arm.
4. **Between arms**: for each tested pair, the four counts, the difference with both
   intervals, the raw and the adjusted p-value. Then how the tuned arms overlap.
5. **Cost**: making each tuned model, the test pass, cost per 1,000 questions and per 1,000
   correct answers, the serving scenarios and break-evens with their reachability, storage.
6. **Accuracy against cost**: the chart, and how to read it.
7. **Where each arm goes wrong**: the thirteen categories per arm; accuracy by gold steps;
   replies cut off at the token limit; the format.
8. **Speed**: latency per arm from the six-arm record, with the caveat that the three
   settings are not comparable (a batch of 8 on one GPU; on-demand calls; Nova Pro paced at
   18 requests a minute).
9. **Who holds the tuned model**: the table of the roadmap.
10. **Caveats**: oracle evidence, so not comparable to published FinQA results; one
    dataset; one training run per arm and no repeated seeds; prices and limits of one
    account in one month; the always-on figures are computed, not measured; the tests were
    named after the counts were known.
11. **Deviations from the plan**: the list of section 4.5.
12. **How to reproduce**: the commands, the records, the commit.

### 4.4 The checks

| Id | Check | Passes when |
|---|---|---|
| R1 | Records in place | The six-arm record passed C1 to C8. The cost record passed K1 to K8; a provisional cost record passes with a reservation |
| R2 | Replies found | Each of the six runs has its predictions file here, with 2,898 lines, and all six hold the same example ids in the same order |
| R3 | Counts agree | The rescored counts of the four metrics, per arm, equal the six-arm record |
| R4 | Nothing frozen has changed | The shared config files still have the hashes of the freeze record |
| R5 | Intervals hold their figures | Every interval contains its point figure, and its lower end is not above its upper end |
| R6 | Pairs add up | For every pair the four counts sum to 2,898, and their margins equal the two arms' counts |
| R7 | Costs taken, not computed | Every dollar figure in the record and the report equals the figure in `cost_model.json` |
| R8 | No text, nothing by hand | Every placeholder of the template is filled and none is left over; the committed outputs hold example ids of the test split and no reply, question, or gold text |

### 4.5 The deviations list

A committed file, `config/report_deviations.yaml`: one entry per deviation, with a date, a
sentence, and the document that records it. The build prints them as a table. The PRP fills
it from the roadmaps' version logs; the owner reviews it (decision 8). Known entries:

- The hosted path is Nova Pro and Nova Lite on Bedrock, not OpenAI; six arms, not four.
- Measured Qwen runs are SageMaker jobs, not runs on the owner's machine.
- Path A's second training iteration (v3.05) ended as analysis only; Path B's (v4.05) did
  not fire. Each tuned arm is one training run.
- Nova Pro was paced at 18 requests a minute, because the account's limits are a tenth of
  the documented defaults.
- Both Bedrock training charges were above their estimates (14% and 15%).
- The always-on scenario is computed from the test job; no endpoint was made.
- No test run was voided.

### 4.6 Tests

No network, no credentials, no GPU. Invented replies and scores only.

- Bootstrap: the same seed gives the same interval; an arm right on every example gives an
  interval of one point; the paired difference of an arm with itself is zero; the company
  interval equals the example interval when every company has one example, and is wider
  when companies are all right or all wrong.
- Exact test: known small cases by hand (no discordant pair gives 1; 0 against 10 gives
  2/1,024); symmetric in the two arms; whole-number arithmetic for counts in the thousands.
- Holm: a worked case; adjusted values never fall below the raw ones and never exceed 1;
  the order of the pairs does not change the result.
- Outcomes: counts equal a hand-made run's; a missing or short predictions file fails R2;
  ids in another order fail R2.
- Checks: each of R1 to R8 passes and fails; a provisional cost record gives a provisional
  report.
- Chart: valid XML; six points; the same input gives the same bytes; no text from a reply.
- Report: every placeholder filled; the template holds no percentage and no dollar amount;
  the README block replaces only what is between the markers, and a README without markers
  gets the block appended once.
- Guards: no new module imports boto3, matplotlib, or scipy; no committed output holds
  reply text; the `examples` file is written only under `artifacts/`.

### 4.7 Documentation

- Runbook: a section 7.11, with the three commands and what each check's failure means.
- `administration/aws_commands_run.md`: a note that this version runs no AWS command.
- `reqspecs/references/project_patterns/`: a short pattern file for a paired comparison of
  two runs.
- `coding.md`: the report is built, never edited; statistics come from `statistics.py`.

## 5. Out of scope

- Any new run, on AWS or here, and any change to a frozen file.
- A new error category, or a category assigned by reading.
- Intervals on dollar figures. The costs are counted, not sampled; their uncertainty is in
  the cost record's limits.
- Repeated training runs, other seeds, other datasets, a full-context arm (req_007).
- The model card, the adapter package, and the rest of the README (v6.02).
- Deleting the tuned Nova models (the runbook's teardown, the owner's).
- A PDF or HTML version of the report.

## 6. What this hands to later versions

| For | What |
|---|---|
| v6.02 | The test results of A1 with their intervals, the error categories, and the caveats, for the model card; the README block |
| req_007 | The 467 questions no arm got right, as ids, and the category counts: where a further arm could gain |
| The teardown | The report does not need the tuned Nova models again; after M8 nothing in the plan asks them a question |

## 7. Known limits and follow-ups

- **The interval by example is too narrow** if questions about one company succeed or fail
  together. The interval by company corrects for that, but 43 companies are few, so it is
  itself rough. Both are shown.
- **One training run per tuned arm.** The intervals describe the test questions, not what
  another training run with another seed would give. A two-point difference between tuned
  arms could be within that run-to-run variation, which this project did not measure.
- **The tests were named after the counts were known.** Holm's method corrects for six
  tests, not for the choice of which six.
- **The categories are computed, not read.** They say how a program differs from the gold
  program, not why the model wrote it.
- **The six-arm record holds the two deployment ARNs**, with the account number. It was
  committed in v5.01. The report does not repeat them; whether to remove them from that
  record before the repository is published is a question for v6.02.
- **The replies are not in the repository.** Rebuilding the statistics needs the six
  prediction files, which are on this machine and in the private bucket.

## 8. Decisions for the owner

**Taken by the owner on 2026-11-23, as recommended: 2, 3, and 9.** So: both intervals are
shown, and a difference whose company interval includes zero is called "not settled";
committed examples are ids only, with the text in a git-ignored file; this version reads
test rows on this machine, and principle 6 of the execution plan is amended (done the same
day). **The other six (1, 4, 5, 6, 7, and 8) were taken as recommended later the same
day**, so all nine are locked: six tested pairs with Holm's method; the chart as an SVG
from the project's own code; the report at `reports/comparison_report.md` with a block in
the README; a build now on the provisional cost record, with M8 at the commit of the final
report; no single winner named; the deviations list drafted by the PRP and read by the
owner. The table is kept as the record of what was asked.

| # | Question | Recommendation |
|---|---|---|
| 1 | **Which pairs are tested?** | The six of section 2.4: each tuned arm against its untuned model, and the three tuned arms against each other. Holm's method over the six. The other nine pairs are shown as counts only |
| 2 | **Are questions resampled, or companies?** | Both are shown. The interval by example is the main one, as the plan says and as the exact test assumes. The interval by company stands beside it, and a difference whose company interval includes zero is described as "not settled", whatever its p-value. This matters for A1 against tuned Nova Pro and against tuned Nova Lite |
| 3 | **"Error categories with examples": what is an example?** | In the committed report, example ids only. The text goes to a git-ignored file on this machine, written by `comparison_report examples`. The alternative is invented examples that illustrate each category, which would be committed but are not real replies |
| 4 | **How is the chart made?** | An SVG written by the project's own code: no new dependency, the same bytes on every build. The alternative, adding `matplotlib`, gives a PNG that differs between versions |
| 5 | **Where does the report live?** | `reports/comparison_report.md`, with its record and chart in `reports/comparison/`, and a generated block in the README |
| 6 | **Build now, on the provisional cost record?** | Yes. The report is marked provisional and rebuilt after the bill is entered; M8 is the commit of the report that is not provisional |
| 7 | **Does the report name a winner?** | No. It states the findings and when each path fits: by volume, by the accuracy needed, and by who must hold the model |
| 8 | **Who writes the deviations list?** | The PRP drafts it from the version logs; the owner reads it before the report is built for good |
| 9 | **May this version read test rows on this machine?** The error categories compare each wrong program with its gold program. The execution plan's principle 6 says only v5.01 loads test rows | Yes, and amend the principle: v5.03 reads them locally to score and categorize stored replies; no model answers, nothing is uploaded. Without it the report has the accuracy tables and tests but no error categories on test |

## 9. Acceptance

Code, by Claude:

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
  and without the training stack blocked, with no network and no credentials.
- `comparison_report check` passes R1 to R8 on the real repository, with a reservation only
  for a provisional cost record.
- `comparison_report build` writes the record, the chart, the report, and the README block;
  a second build changes no byte.
- The pairs' counts in the record equal the hand figures of section 2.4.
- No AWS service is called, no model is run, and no frozen file changes.

By the owner (🟡):

1. Read the report and the deviations list.
2. Enter the bill (v5.02), then `comparison_report build` again: the report is no longer
   provisional. Commit. That commit is **M8**.

- The report's main figures are written into this spec as a section 11.

## 10. Next step

The code is built and the report is written, provisional (section 11). The owner: read
`reports/comparison_report.md` and `config/report_deviations.yaml`; after the bill is entered
(runbook section 7.10), run `comparison_cost report` and `comparison_report build` again, and
commit. That commit is milestone M8 and completes this version.

After this version come v6.02 (model card, adapter package, README; M9) and, for the owner,
the Path B teardown.

## 11. As built (2026-11-23)

The code is built (`prp_005_comparison_report_v3.md`) and the report is written:
`reports/comparison_report.md`, with `reports/comparison/comparison_statistics.json` and
`.md`, `reports/comparison/accuracy_cost.svg`, and the block in `README.md`. It is
**provisional**: check R1 passes with a reservation, because the cost record it reads is
provisional. Nothing was sent to AWS, no model was asked, and no frozen file changed. Tests:
2,896 passed and 1 skipped in the full suite, 171 of them new; with the training stack blocked, 2,890 passed and 7 skipped; lint clean.

### The figures

Checks R1 to R8 pass on the real repository. The rescored counts equal the six-arm record,
and the pairs' counts equal the hand figures of section 2.4. A second build changes no byte.

| Arm | Execution accuracy | Interval by example | Interval by company |
|---|---:|---:|---:|
| A0 | 0.93% | 0.59 to 1.31 | 0.40 to 1.56 |
| A1 | 73.19% | 71.60 to 74.78 | 69.63 to 76.47 |
| B0-Lite | 16.70% | 15.36 to 18.08 | 14.66 to 18.61 |
| B0-Pro | 35.37% | 33.64 to 37.09 | 32.23 to 37.87 |
| B1-Lite | 71.39% | 69.74 to 73.02 | 67.78 to 74.91 |
| B1-Pro | 75.19% | 73.60 to 76.74 | 72.11 to 77.69 |

| Pair | Difference, points | By example | By company | p, adjusted | Verdict |
|---|---:|---:|---:|---:|---|
| A1 minus A0 | +72.26 | 70.63 to 73.84 | 68.76 to 75.40 | below 0.0001 | settled |
| B1-Lite minus B0-Lite | +54.69 | 52.76 to 56.56 | 51.14 to 58.54 | below 0.0001 | settled |
| B1-Pro minus B0-Pro | +39.82 | 37.89 to 41.75 | 37.00 to 43.04 | below 0.0001 | settled |
| A1 minus B1-Pro | -2.00 | -3.31 to -0.66 | -3.55 to -0.17 | 0.0072 | settled |
| A1 minus B1-Lite | +1.79 | 0.38 to 3.21 | -0.03 to 3.52 | 0.0132 | **not settled** |
| B1-Pro minus B1-Lite | +3.80 | 2.48 to 5.11 | 2.24 to 5.17 | below 0.0001 | settled |

What the report finds:

- **Tuning is the large effect**: 39.82 to 72.26 points in the three models, all settled.
- **Tuned Nova Pro is ahead of A1 by 2.00 points, and that is settled**: both intervals
  exclude zero, the one by company only just (its end nearest zero is 0.17 points).
- **A1 is ahead of tuned Nova Lite by 1.79 points, and that is not settled**: the interval
  by company includes zero, as decision 2 foresaw.
- **Tuned Nova Pro is ahead of tuned Nova Lite by 3.80 points**, settled.
- **Long programs are where the tuned arms differ most.** On questions whose gold program
  has three or more steps (223 of them): A1 43.05%, tuned Nova Pro 34.08%, tuned Nova Lite
  26.01%. On one-step questions the three are within 2.4 points of each other.
- **The tuned arms fail in the same ways**: too few steps first (215 to 288 replies), then
  other numbers with the right operations (199 to 247). Almost none of their replies fails
  to parse or to run. The untuned arms mostly write programs that do not parse.
- **Test is not below dev for any tuned arm**: A1 +1.45 points, tuned Nova Lite +5.46,
  tuned Nova Pro +3.25. Choosing on dev did not flatter them.
- 1,807 questions are right in all three tuned arms, 2,401 in at least one, 467 in none of
  the six arms.

### Differences from this spec

- **Nine modules, not six.** `report_tables.py` (every table and generated sentence) and
  `report_examples.py` (the only code that handles example text) are separate from
  `report_build.py`; the template is `report_template.md` beside them.
- **R7 is checked as "the same six runs".** The cost record's run ids, counts of right
  answers, and freeze hash must equal the six-arm record's; its figures are then copied as
  text. This version multiplies nothing by a price.
- **The findings are generated sentences**, with branches for a gain that is not settled,
  equal arms, and a reachable always-on break-even. The template's own prose claims nothing
  that a changed record could make false.
- **Example ids are in the statistics file**, not in the report, which points to them.
  The ids of the 467 questions no arm got right are in the JSON file (`unanswered_ids`).
- **The README block has three of the findings** (tuning, the tuned arms in order, cost per
  1,000 questions) and the summary table.
- **The bootstrap uses numpy's legacy `RandomState`** with a fixed integer type, because
  its stream is frozen; the company bootstrap uses the seed plus one.
- **A sentence about a pair turns its intervals** to point the way the sentence does
  ("B1-Pro is ahead of A1 by 2.00 points, interval 0.66 to 3.31"); the table keeps first
  minus second.
- **The chart says when the whiskers are hidden**: here every interval is narrower than its
  marker.
- **The arms are listed in the six-arm record's order** (A0, A1, B0-Lite, B0-Pro, B1-Lite,
  B1-Pro).
- **`cost_report._break_even_text` was renamed `break_even_text`** so the report reuses the
  cost record's wording.
- **The deviations list has twelve entries**, drafted from the version logs. The owner has
  not read it yet.

### Not verified

- The report's prose has been read once by Claude, not by the owner.
- The chart was rendered once in a headless browser and looked at; how GitHub shows it in
  a dark theme was not checked (its surface is fixed and light).
- The run-to-run variation of training was not measured: the intervals describe the test
  questions only.

### Closed (2026-11-24): milestone M8

The owner entered the bill (v5.02), ran `comparison_cost report` and
`comparison_report build`, and committed the result as `3dbf2b2`. R1 to R8 pass with no
reservation; the report, the statistics record, and the README block no longer say
"provisional". No accuracy figure and no interval changed; a further build changes no byte.
The report is at `reports/comparison_report.md`. This version is complete, and with it
phase P5.

Still as listed under "Not verified": the owner's reading of the prose and of the
deviations list is the owner's to confirm, and the figure was not checked in GitHub's dark
theme.
