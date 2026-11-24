# DSL Evaluation

Spec Group      :   req_002_dsl_evaluation  
Feature Name    :   dsl_evaluation  
Version         :   2  
Status          :   ✅ Complete 2026-11-16; tests and lint green locally; committed (the three decisions of section 8 were made by the owner; built by `prp_002_metrics_gold_check_v2.md`; results in section 11)  
Last updated    :   2026-11-16  

## 1. v2.02 — Metrics and gold self-check

v2.01 turns a reply into a value or a named error. This version says when that value is
**right**, and proves the whole harness on the one set of replies whose score is known in
advance: the gold ones.

It adds:

- a **comparison rule**: when two numbers count as the same, fixed in config before any model
  is scored;
- an **answer reader**: the text after `ANSWER:` as a number or `yes` / `no`;
- the **four metrics** of plan §4.4 for one reply, and their totals over many;
- a **gold self-check**: every gold reply is scored by the harness and must come out right.
  Examples whose gold program does not reproduce the gold answer are listed in a flags file,
  never dropped.

It runs no model and scores no prediction. The runner is v2.03.

**Depends on**: v2.01 and v1.02 (both built). It also uses the canonical record of v1.03.

## 2. What exists today (verified 2026-11-15)

Measured on fit and dev together (5,383 examples) with the built v2.01 executor, read through
`load_split`. The test split was not read: every rule below was fixed without it, and its
part of the self-check is first seen when this version is built (D21).

### 2.1 What earlier versions provide

| Item | State |
|---|---|
| `run_reply(text, table)` | A `ReplyOutcome`: unrounded value, canonical program, answer text, strict flag, error code |
| `parse_program`, `Program` | Each argument with its kind and its value |
| `CanonicalRecord.target` | The gold reply: the dataset's program and `exe_ans`, scientific notation written out |
| `config/split.yaml`, the manifest | Frozen. v1.02 D18: per-example flags go in their own file, never in the manifest |
| The raw reader | Allowed in `qlora_finetuning/data/`, the two data scripts, and their tests only (v1.02 D13). The split report is "the one sanctioned use" of test rows before the unlock (v1.02 D14) |
| `qlora_finetuning/evaluation/` | An empty package |
| Plan §4.4 | Four metrics; tolerance and normalization "fixed in req_002 before any model is scored" |
| The guides | Gemini: numeric answer "within ±0.5% relative tolerance to handle rounding". OpenAI: "normalize commas, percent signs, and rounding; exact match for yes/no and a defined numeric tolerance"; for programs, "normalize whitespace and numeric formatting, then compare" |

### 2.2 How exact the gold answers are

5,270 gold answers are numbers and 113 are `yes` / `no`. 19 are exactly zero, and the
executed value is exactly zero in all 19.

`exe_ans` is the true result **rounded to five decimal places, and to about six significant
digits when it is large**. So a small result has few digits: `0.00002` stands for anything
from 0.000015 to 0.000025. 92 gold answers are below 0.01 in size, 13 below 0.001.

How many of the 5,270 executed gold programs match their own `exe_ans`, by rule:

| Allowed difference | Matches |
|---|---|
| Relative 0.001% only | 4,009 |
| Relative 0.5% only | 5,259 |
| Absolute 0.00001 only | 4,843 |
| Absolute 0.000005 or relative 0.001% | 5,269 |
| **Absolute 0.00001 or relative 0.001%** | **5,270** |

A relative rule alone fails on small results whatever its size; an absolute rule alone fails
on large ones. With both, every gold program matches, and the largest gap uses half of what
is allowed.

**This corrects v2.01**, which said every result was "within a relative 0.0005%" of the gold
one. That holds for 438 results; the other 4,945 equal the gold one only after rounding to
five decimal places, and some of those differ by far more in relative terms.

### 2.3 What a looser rule lets through

For each gold program, one number in it was replaced by another number from the same table:
a wrong operand, the commonest kind of model error. 164,877 such wrong programs over 5,074
examples.

| Relative part of the rule (absolute 0.00001 throughout) | Wrong programs accepted | Examples with at least one |
|---|---|---|
| 0.001% | 512 (0.31%) | 44 (0.87%) |
| 0.01% | 541 (0.33%) | 56 (1.10%) |
| 0.1% | 728 (0.44%) | 138 (2.72%) |
| 0.5% (the Gemini guide) | 1,319 (0.80%) | 439 (8.65%) |

At 0.5%, one example in twelve has a wrong operand that scores as right; at 0.001%, one in
115. Of the 44 examples that remain at 0.001%: in 19 the result is so small that the absolute
allowance decides, in 12 the changed step is one the gold program computes and never uses,
and 16 have another cause, such as an operand too small to move a large result. (An example
can be in more than one group.)

### 2.4 The answer line

Significant digits on the gold answer line:

| Digits | 1 | 2 | 3 | 4 | 5 | 6 | 7 or more |
|---|---|---|---|---|---|---|---|
| Examples | 179 | 480 | 665 | 1,150 | 1,886 | 757 | 134 |

How often an answer written with fewer digits than the gold one would still be accepted:

| The answer is the gold value rounded to | Relative 0.001% | Relative 0.5% |
|---|---|---|
| 2 decimal places | 38.6% | 64.4% |
| 4 decimal places | 58.1% | 99.2% |
| 3 significant digits | 29.0% | 100.0% |
| 4 significant digits | 52.2% | 100.0% |

2,834 numeric results are below 1 in size (ratios and rates). 96 gold programs end by
multiplying by `const_100` and 94 by dividing by it, so the dataset itself is not consistent
about writing a rate as 0.084 or as 8.4.

### 2.5 Programs

| Fact | Count |
|---|---|
| Decimal literals with a whole value, such as `4850.0` | 273 |
| Decimal literals with a trailing zero, such as `3.50` | 126 |
| Decimal literals starting with the point, such as `.35` | 19 |
| Constants | 1,117 |
| Percent literals | 235 |
| `add` and `multiply` steps, whose arguments could be swapped | 1,893 of 8,283 |
| Pairs of examples with the same page and question | 61; 54 have the same result, and all 54 have the same program text |

A model that writes `4850` for `4850.0` has written the same program. The duplicate pairs
give no evidence that annotators wrote equivalent programs in different ways.

## 3. Locked decisions

### Comparing numbers

| # | Decision | Reason |
|---|---|---|
| D1 | Two numbers match when their difference is at most **the larger of an absolute allowance and a relative allowance times the gold value**. Both come from `config/evaluation.yaml` | Section 2.2: neither part works alone |
| D2 | The **gold value is `exe_ans`**, read from the canonical record's answer line. It is not the gold program executed again | One gold value per example, the one on the answer line the models are trained on, and independent of the executor being tested |
| D3 | **Execution accuracy: absolute 0.00001, relative 0.5%.** | **The owner's decision (section 8, 2026-11-16)**: the Gemini guide's ±0.5%, so a program that works from a number the text gives rounded is not marked wrong for it. The absolute part stays at the dataset's own precision, because a relative rule alone fails on 11 small gold results (section 2.2). The cost is stated in section 2.3: 8.65% of examples have a wrong operand that scores as right, against 0.87% under the tight rule this spec first proposed (absolute 0.00001, relative 0.001%) |
| D4 | **Answer accuracy: absolute 0.00001, relative 0.5%.** | The answer line is the model's own arithmetic and rounding. The Gemini guide sets ±0.5% for it "to handle rounding", and section 2.4 shows a tight rule would mostly measure how many digits were printed. With D3 as decided, the two allowances are equal; they stay two settings in the config, because they answer different questions |
| D5 | `yes` and `no` match exactly. A number never matches `yes` / `no` | Both guides |
| D6 | The allowances are fixed before any model is scored. Changing one afterwards means rescoring every arm, which is possible because every prediction keeps its unrounded value (v2.03) | Plan §4.4 |

### Reading the answer line

| # | Decision | Reason |
|---|---|---|
| D7 | The answer text is trimmed, a final full stop is dropped, and case is ignored. `yes` and `no` are read as such | The prompt asks for "a plain number, or yes or no" |
| D8 | A number may carry a `$`, thousands separators, a leading plus or minus sign, and an exponent. Anything else, such as `3.8 million` or `( 5 )`, is not an answer and scores as wrong | OpenAI guide: normalize commas and percent signs. Words are not guessed at |
| D9 | **An answer ending in `%` matches when either the number as written or the number divided by 100 matches the gold value** | Section 2.4: gold results are written both ways. `8.4%` is a fair answer to a gold of 8.4 and to a gold of 0.084 |
| D10 | Without a `%` sign there is no scale leniency: 8.4 does not match 0.084 | The answer is "the result of the last step". Off-by-100 is the error the guides warn about, and it must stay visible |

### The four metrics

| # | Decision | Reason |
|---|---|---|
| D11 | **Format validity**: the reply is strict (v2.01 D7) | Plan §4.4 |
| D12 | **Execution accuracy** (primary): the program was found, parsed, and executed, and its value matches the gold value under D3. A reply that broke the format but had a readable program counts (v2.01 D8) | Plan §4.4; accepted with v2.01 |
| D13 | **Answer accuracy**: the answer text, read by D7 to D9, matches the gold value under D4. It is judged whether or not the program ran | It measures the model's stated result, independently |
| D14 | **Program match**: same number of steps, same operations in the same order, and equal arguments. Numbers are compared **by value**, however written: `4850.0` and `4850`, `.35` and `0.35`, `const_100` and `100`, `6.9%` and `0.069`. References by step number, row names by text. The second argument of a table step is ignored | OpenAI guide: "normalize whitespace and numeric formatting, then compare". Section 2.5 |
| D15 | Program match does **not** treat `add(a, b)` and `add(b, a)` as equal, or reorder steps | The Gemini guide keeps "exact match" for the canonical form and credits equivalent programs under execution accuracy. Two metrics that mean different things are more use than two that nearly agree |
| D16 | Every rate is a count over **all** examples given. A reply with an error is wrong on what could not be judged; nothing is left out of a denominator | An arm must not gain by failing to answer |
| D17 | A wrong numeric result gets one **mismatch label** for error analysis: `scale` (the value is the gold one times or divided by 100, 1,000, or 1,000,000), `sign` (the negative of it), `kind` (a number where `yes` / `no` was due, or the reverse), or `other`. A label never makes a result right | The guides name scale and operand errors as the first things to look for |

### The gold self-check

| # | Decision | Reason |
|---|---|---|
| D18 | The self-check builds each example's gold reply with the serializer, runs it through `run_reply`, and scores it against itself. Format validity, execution accuracy, answer accuracy, and program match must all be true | The harness is trusted only after it scores known-right replies as right. It also exercises the answer reader on every gold answer form |
| D19 | An example fails the self-check when its gold program does not execute or its value does not match `exe_ans` under D3. Such an example is **flagged**: its id and a reason go into `splits/finqa_gold_flags.json` | v1.02 D18. The manifest and its hash do not change |
| D20 | **A flagged example is never dropped.** It stays in its split and in every denominator. Every total is reported twice: over all examples, and over the unflagged ones. The first is the headline | Plan §7 and v1.02 D2: nothing is silently removed. A flagged example cannot be scored right on execution, so the first number has a ceiling below 100%, and the report states it. **Decision for the owner in section 8** |
| D21 | **The self-check covers all 8,281 examples, test rows included**, and reports counts and example ids only. It reads a test row's gold program and gold answer; no model is involved. This is a second sanctioned use of test rows before the unlock, beside the split report | The plan asks for the rate "over all 8,281". The rules were fixed on fit and dev (section 2) before any test row was checked. **Decision for the owner in section 8** |
| D22 | The self-check lives in `qlora_finetuning/data/` and its script joins the two already allowed to use the raw reader. The guard test's list grows by that one script | v1.02 D13. No other code gains access to test rows |
| D23 | Fit and dev must show no flagged example; that is a check, and it holds today (section 2.2). The test split's count is reported, not required to be zero | A flagged fit example would mean training on a target whose program does not give its answer. What to do then is decided if it ever happens; today it does not |
| D24 | The report is committed as `reports/data/finqa_gold_check_report.md` and `.json`: counts, ids, the settings, no example text, no timestamp. The flags file is committed too, with the manifest hash, the prompt id, and the allowances it was made with | Same rules as the other reports. A flag is only meaningful for the rule that produced it |
| D25 | The statements in the system text about what a percent literal and each operation mean count as confirmed when fit and dev pass the self-check (v1.03 D16) | No model has seen the prompt yet |

## 4. Scope to build

### 4.1 `config/evaluation.yaml`

`version: 1`; `execution.absolute: 0.00001`, `execution.relative: 0.005`;
`answer.absolute: 0.00001`, `answer.relative: 0.005`; `scale_factors: [100, 1000, 1000000]`;
`flags_file: splits/finqa_gold_flags.json`; `report_dir: reports/data`.

### 4.2 Metrics: `qlora_finetuning/evaluation/metrics.py`

Pure functions. No import of the data package at module level, no file access except the
config loader.

| Item | Behavior |
|---|---|
| `load_evaluation_config()` | Reads and validates the file: allowances are numbers of zero or more, at least one of each pair above zero |
| `numbers_match(value, gold, allowance)` | D1 |
| `read_answer(text)` | D7 to D9. Returns `yes` / `no`, or the one or two numbers the text can mean, or nothing |
| `Gold.from_target(target, reply_format)` | The gold program (parsed) and the gold value, from a canonical record's target text |
| `programs_match(first, second)` | D14, D15 |
| `mismatch_label(value, gold, config)` | D17 |
| `score_reply(outcome, gold, config)` | An `ExampleScore`: `format_valid`, `executed`, `execution_correct`, `answer_correct`, `program_match`, `error_code`, `mismatch`, `relative_gap` |
| `summarize(scores, flagged=())` | D16, D20. For all examples and for the unflagged ones: the number of examples, the count and rate of each metric, how many executed, and counts by error code, by deviation, and by mismatch label |

Usage, on an invented example:

```python
from qlora_finetuning.dsl import ReplyFormat, run_reply
from qlora_finetuning.evaluation.metrics import (
    Gold, load_evaluation_config, score_reply, summarize,
)

config = load_evaluation_config()
labels = ReplyFormat(program_label="PROGRAM:", answer_label="ANSWER:")
gold = Gold.from_target("PROGRAM: subtract(920, 95), divide(#0, 95)\nANSWER: 8.68421", labels)

reply = "PROGRAM: subtract(920.0, 95), divide(#0, 95)\nANSWER: 8.68"
score = score_reply(run_reply(reply, table=None), gold, config)
score.execution_correct   # True: 8.684210... matches 8.68421
score.answer_correct      # True: 8.68 is within 0.5%
score.program_match       # True: 920.0 and 920 are the same number
score.format_valid        # True

summarize([score])["all"]["execution_accuracy"]   # {"count": 1, "rate": 1.0}
```

### 4.3 Self-check: `qlora_finetuning/data/gold_check.py`

| Function | Behavior |
|---|---|
| `check_example(example, prompt, config)` | D18, D19: the example's score and, if it fails, a reason: an error code of v2.01, `value_mismatch`, `answer_unreadable` (the gold answer is neither a plain number nor `yes` / `no`), or `unserializable` (no gold reply can be built). As built: a gold reply that is not strict or not in canonical form fails K2 and is **not** a flag, because D19 defines a flag by execution only |
| `build_report(examples, manifest, prompt, config)` | Per split (fit, dev, quick-dev, test) and in total: examples checked; each metric's count; flagged examples by reason; the largest gap as a share of the allowance; `yes` / `no` examples; programs with the `duplicate_row` note. Then the settings and the checks |
| `build_flags(report)` | The flags file: one entry per flagged id with its split and reason, the settings, and a hash |
| `load_gold_flags()` | The flagged ids, for `summarize`. Usable everywhere: it reads the committed file, not the dataset |

| Id | Check | Passes when |
|---|---|---|
| K1 | Complete | Every example of the manifest was checked, once |
| K2 | Gold replies are well formed | Every gold reply is strict and its program equals its canonical form |
| K3 | Fit and dev reproduce | No fit or dev example is flagged (D23) |
| K4 | Self-consistent | Every unflagged example scores true on all four metrics |
| K5 | Flags current | The flags file as it was before this run holds the same flagged examples and settings as a fresh one, or there was none. A difference fails once and the file is rewritten for review, as the serialization report's digest check does; `--check` writes nothing |
| K6 | Same data | The manifest describes exactly the data on disk (`check_manifest_against_data`, run by the script and passed in). As built: comparing with two other committed reports would have tied the report function to files it does not own; the manifest hash and the prompt id are recorded in the report and in the flags file instead |
| K7 | Deterministic | Checking twice gives identical output |

### 4.4 `scripts/finqa_gold_check.py`

| Command | Behavior |
|---|---|
| `venv\Scripts\python -m scripts.finqa_gold_check` | Checks all 8,281 examples, runs K1 to K7, writes the report and the flags file, prints the check lines and the rate per split |
| `... --check` | Writes nothing; fails if the report or the flags file differs from a fresh one |

Exit codes 0, 1, 2 as in the other data scripts.

### 4.5 Tests

On invented programs, tables, and rows. No test needs the network, a GPU, or the training
stack.

- Comparison: each side of the absolute and the relative allowance, at the boundary and just
  past it; a gold of zero; negative values; a very large value; `yes` / `no` against each
  other and against a number.
- Answer reader: plain, negative, decimal, `$`, thousands separators, exponent, a final full
  stop, upper case `YES`; a `%` answer against a gold written each way; words, brackets, an
  empty answer, two numbers.
- Program match: each equivalence of D14; each thing D15 keeps apart; a different operation,
  step count, reference, and row name; a table step's second argument ignored.
- Mismatch labels: each scale factor both ways, a sign flip, each kind mix-up, `other`; a
  right result gets none.
- `score_reply`: a perfect reply; right program and wrong answer; wrong program and right
  answer; a recovered reply (execution right, format wrong); an error at each stage; an
  equivalent program that is not a program match.
- `summarize`: counts and rates on a small set; all-wrong and empty input; flagged ids change
  only the unflagged block; rates are counts over the number of examples.
- Config: a negative allowance, both parts zero, a missing key.
- Self-check on invented rows with one broken gold program of each reason: flagged, reported,
  never dropped; each check passes and fails; the report holds no invented text; the flags
  file round-trips; deterministic.
- Script: exit codes; `--check`; module form and by-path hint.
- Guards: the raw-reader list has exactly one new entry; `evaluation/metrics.py` does not
  import the data package at module level.
- On the real data when the snapshot is present: the committed report and flags file match a
  fresh run.

### 4.6 Documentation

- Runbook section 9: run and check the self-check; what a flagged example means.
- `reply_scoring.md`: scoring goes through `score_reply` and `summarize`; never compare
  values or round in a caller.
- `gotchas.md`: `exe_ans` is rounded to five decimal places **and** to about six significant
  digits; a relative tolerance alone fails on small results; the answer line may end in `%`.
- `high_level_plan.md` §4.4: the two allowances, program match by value, the mismatch labels.
- `coding.md`: allowances come from `config/evaluation.yaml`; totals come from `summarize`.
- `split_loading.md`: the gold flags file and `load_gold_flags()`.

## 5. Out of scope

- Running a model, the `Predictor` protocol, the predictions file, run tracking (v2.03).
- Confidence intervals and paired tests between arms (v5.02).
- Breaking results down by operation, program length, or evidence kind (v5.03).
- Unlocking the test split or scoring any prediction on it (v5.01).
- Crediting an answer for being off by a scale factor, or a program for being equivalent in
  some way D14 does not name.

## 6. What this hands to later versions

| For | What |
|---|---|
| v2.03 | `score_reply`, `summarize`, `Gold.from_target`, `load_gold_flags`; the runner stores each `ExampleScore` and the outcome's unrounded value |
| v3.01, v4.01 | The baselines' first numbers come from `summarize`, on quick-dev |
| v3.04 | Checkpoint selection on dev execution accuracy, as D12 defines it |
| v3.05, v4.05 | Mismatch labels, error codes, and deviations as the first cut of an error analysis |
| v5.01 | The flags for the test split already exist; the headline is over all test examples, with the unflagged figure beside it |
| v5.02 | Per-example true / false for each metric and arm, which is what paired statistics need |
| v1.03 D16 | The system text's statements are confirmed (D25) |

## 7. Known limits and follow-ups

- **Small results are judged loosely.** With an absolute allowance of 0.00001, a gold of
  0.00002 accepts anything from 0.00001 to 0.00003. That is the gold value's own precision;
  13 fit examples are below 0.001 in size.
- **Some wrong programs still pass.** Section 2.3: at the chosen rule (relative 0.5%) 8.65%
  of examples have at least one wrong operand that gives a matching value, and 0.80% of all
  single-operand swaps do. The tight rule would leave 0.87% of examples: small results,
  steps the gold program never uses, and operands too small to matter. Every prediction
  keeps its unrounded value (D6), so the tight figure can be computed beside the headline
  at any time without running a model again.
- **A right calculation on another scale is wrong** unless the answer carries a `%` sign
  (D9, D10). The `scale` label counts these so they can be reported.
- **Program match is strict about order** (D15). It will read lower than execution accuracy
  for every arm; the gap is the share of right answers reached another way.
- **The test split's self-check result is now known** (D21, section 11): no test example is
  flagged, so the headline has no ceiling below 100% and the two totals are equal today.
- **Answer accuracy is lenient by design** (D4). It says the model's stated number is about
  right, not that its arithmetic is exact.
- **The figures in section 2 come from throwaway scripts.** The built report must reproduce
  those it covers; any difference is investigated first.

## 8. Decisions for the owner

**Decided by the owner on 2026-11-16**: (1) the Gemini guide's **0.5%** for execution
accuracy, not the recommended tight rule; (2) flagged examples are **kept** in every total;
(3) **yes**, the self-check reads test rows now. D3, section 4.1, and section 7 were
rewritten for the first; the table below is kept as the record of what was asked.

| # | Question | Recommendation |
|---|---|---|
| 1 | **Tolerance for execution accuracy**: the dataset's own precision (absolute 0.00001 or relative 0.001%, D3), or the Gemini guide's 0.5%? | The tight one. At 0.5%, ten times as many examples have a wrong operand that scores as right (8.65% against 0.87%), and the guide proposed 0.5% for the answer line, where it is kept (D4). The loose one is kinder to a model that works from a rounded number in the text |
| 2 | **Flagged examples**: keep them in every total and show the unflagged total beside it (D20), or leave them out of scoring? | Keep them. The test set stays the brief's 35% with nothing removed, and the two numbers show the size of the effect. Leaving them out gives a cleaner number but makes "the test set" depend on the executor |
| 3 | **May the self-check read test rows now** (D21), for counts and ids only? | Yes. No model is involved, the rules were fixed on fit and dev first, and the plan asks for all 8,281. The alternative is to check fit and dev now and the test rows at the unlock (v5.01), which keeps the lock absolute but lets flagged test examples appear on the day the final numbers are made |

## 9. Acceptance

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with and
  without the training stack blocked.
- `venv\Scripts\python -m scripts.finqa_gold_check` checks 8,281 examples; K1 to K7 pass; a
  second run changes nothing; `--check` exits 0.
- Fit and dev: 5,383 of 5,383 reproduce, with no flagged example, and the largest gap is at
  most half the allowance (section 2.2: it is exactly half under the tight rule, so it
  cannot be more under the decided one).
- The test split's rate and its flagged ids, if any, are in the report and recorded in the
  roadmap.
- The report and the flags file hold no example text.
- The raw-reader guard lists exactly one more file than before.

## 10. Next step

Built. The next version is v2.03, the evaluation runner (`req_002_roadmap.md`).

## 11. As built (2026-11-16)

`venv\Scripts\python -m scripts.finqa_gold_check`, all 8,281 examples, checks K1 to K7 pass:

| | fit | dev | quick-dev | test | all |
|---|---:|---:|---:|---:|---:|
| Examples | 4,884 | 499 | 200 | 2,898 | 8,281 |
| Gold programs that reproduce the gold answer | 4,884 | 499 | 200 | 2,898 | 8,281 |
| Flagged | 0 | 0 | 0 | 0 | 0 |
| Largest share of the allowance used | 0.490 | 0.494 | 0.494 | 0.454 | 0.494 |
| `yes` / `no` examples | 100 | 13 | 5 | 41 | 154 |
| Programs naming a repeated row | 1 | 0 | 0 | 2 | 3 |

- **No example is flagged in any split**, so `splits/finqa_gold_flags.json` is empty and the
  totals over all examples and over the unflagged ones are the same.
- The figures of section 2 that the report covers are reproduced: 5,383 fit and dev examples,
  113 of them `yes` / `no`, every gold program matching.
- The same check run once under the tight rule (absolute 0.00001, relative 0.001%; counts
  only, nothing written) also flags nothing, in the test split as well, with a largest share
  of exactly 0.5. So the choice between the two rules changes no flag; it changes only how a
  model's near-miss is scored.
- Differences from sections 4.2 to 4.4 are written into those sections (the reasons for a
  flag, K5, K6). Two additions: `ExampleScore` also carries the example id and the reply's
  `deviations`, and `summarize` reports `executed` beside the four metrics.
