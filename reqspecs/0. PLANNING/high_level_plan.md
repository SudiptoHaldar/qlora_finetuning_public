# High-level architecture and plan

**Project**: `qlora_finetuning` (QLoRA Finetuning)  
**Last updated**: 2026-11-14 — measured Qwen runs moved to AWS SageMaker; hosted path set to Amazon Nova Pro and Nova Lite on Bedrock; six arms. (2026-11-13: plan expanded from the original brief; spec groups req_000–req_006 created.)  
**Companion documents**: `execution_plan.md` (ordered versions, phases, status board, open questions), `../../architecture/architecture_overview.md` (diagrams), `../1. PROMPTS/coding.md` (coding rules).

---

## 0. Original brief (owner's words, kept verbatim)

Read the documents in folder `/../reqspecs/0. PLANNING/planning_material`
  - Gemini FinQA QLoRA Fine-Tuning Execution Guide.pdf
  - openai_model_fine_tuning_qlora_project_guide.pdf

I want to use the FinQA dataset for
 - finetuning Qwen/Qwen3-4B
 - training a frontier AI model

I want to compare the cost and accuracy of the two paths.

Use 65% of the data for training purpose, and the rest for testing predictive performance. Use the same split for both paths, i.e. define the split between training and validation data beforehand and use the same split for both methods.

---

## 1. Goal

Build a **financial report calculation assistant** two ways on the same data, then compare them:

| Path | What is trained | Where it runs |
|---|---|---|
| **A. Open-weight QLoRA** | A LoRA adapter on 4-bit-quantized `Qwen/Qwen3-4B` | AWS SageMaker GPU jobs (developed and smoke-tested on the local RTX 4090 Laptop GPU) |
| **B. Hosted fine-tuning** | Amazon Nova Pro and Amazon Nova Lite, each fine-tuned through Amazon Bedrock | AWS Bedrock, us-east-1 |

The brief's "frontier AI model" is realized as Amazon Nova. OpenAI was the first choice but its
self-serve fine-tuning is closed to new users and ends in January 2027. Nova Pro is the
strongest Nova model Bedrock currently lets you fine-tune; Nova Premier is not offered for it.

Given a report excerpt, a financial table, and a question, each model returns a short arithmetic
program plus an answer. A deterministic Python executor runs the program, so the model proposes
the calculation and never acts as the calculator.

The deliverable is a **cost and accuracy comparison** of the two paths on one frozen test set,
with the untuned version of every model as a reference point (six arms in total):

| Arm | Model | Tuned |
|---|---|---|
| A0 | Qwen3-4B, 4-bit | No (zero-shot) |
| A1 | Qwen3-4B, 4-bit + QLoRA adapter | Yes |
| B0-Pro | Amazon Nova Pro | No (zero-shot) |
| B1-Pro | Amazon Nova Pro, fine-tuned | Yes |
| B0-Lite | Amazon Nova Lite | No (zero-shot) |
| B1-Lite | Amazon Nova Lite, fine-tuned | Yes |

Two hosted models are included to show how accuracy scales with price: Nova Lite costs about a
quarter of Nova Pro to train and about a fourteenth to serve.

## 2. Scope

**In scope**

- One frozen, shared data split and one shared prompt/target contract for both paths.
- A safe arithmetic DSL executor and a model-agnostic evaluation harness.
- QLoRA training of Qwen3-4B on a SageMaker GPU instance, including zero-shot baseline and checkpoint selection.
- Fine-tuning Amazon Nova Pro and Nova Lite through Bedrock, including zero-shot baselines.
- A shared AWS foundation: one region, one S3 bucket, least-privilege roles, a budget alarm.
- A run registry and cost ledger that both paths write to.
- A comparison report: accuracy, cost to train, cost to serve, error analysis.
- A small demo CLI, a model card, and a README (deliverables listed in both guides).

**Out of scope (candidates for `req_007_future_extensions`)**

- Retrieval or full-filing context (the primary comparison uses gold evidence; see §4.3).
- Hyperparameter sweeps beyond one corrective iteration per path.
- Other hosted models or providers, other open-weight base models, serving infrastructure, a web UI.
- Any claim that the result is a production financial adviser. FinQA is a research benchmark.

## 3. The two planning guides and where this plan departs from them

Both guides describe Path A only. This plan adopts their engineering advice and overrides them
where the brief says otherwise.

| Topic | Guides say | This plan | Why |
|---|---|---|---|
| Data split | Official 6,251 / 883 / 1,147 | Pool all 8,281, re-split **65% train / 35% test** | The brief sets the ratio |
| Validation data | Official validation split | A **dev slice carved out of the 65%** | Keeps the 35% test untouched until the end |
| Paths | QLoRA only | QLoRA **and** hosted fine-tuning of Nova Pro and Nova Lite | The brief |
| Hardware | Local 16 GB GPU | Measured runs on a 24 GB SageMaker GPU; local GPU for development only | Owner's decision 2026-11-14: with both paths billed in AWS dollars the cost comparison is like-for-like |
| Thinking mode | Gemini guide: disable | `enable_thinking=False` for Qwen3; inference settings fixed and recorded for Nova | Stable, parseable output |
| Context | Gemini guide: gold evidence + Markdown table, 1,536 tokens | Adopted for all models, with a training length of 2,048 tokens | Identical prompts make the comparison fair. Measured in v1.04: two training examples exceed 1,536 and none exceeds 2,048, and nothing is ever truncated |
| Loss | Completion-only | Adopted | Verify the API against the installed TRL; `DataCollatorForCompletionOnlyLM` has been removed from recent TRL releases |
| DSL | FinQA native syntax with `#k` references | Adopted | Unambiguous multi-step programs |
| OS | Linux or WSL2 | Linux on SageMaker for measured runs; native Windows first for local development, WSL2 as fallback (Q3) | The v0.03 environment check decides the local question; it no longer affects reported numbers |
| Benchmark expectations | Gemini guide's accuracy ranges | Treated as rough expectations only | They assume the official split; ours differs |

## 4. Design decisions

### 4.1 Frozen split (binding for both paths)

- Pool the three official splits of `bevaya/FinQA` (formerly `rootsautomation/FinQA`, which now
  redirects) at the commit and checksums pinned in `config/data.yaml` (8,281 examples).
- Assign **65% to train and 35% to test** (5,383 / 2,898), deterministic from a fixed seed
  (20261115). Groups are ordered by SHA-256 of seed and group key, not by a random generator,
  and taken while they fit under the target.
- **Group by company** so no company is on both sides (Q4, resolved 2026-11-15). Grouping by
  report was the first plan; it was dropped when measurement showed annual reports repeat
  prior years' numbers, so 28.4% of the checkable test questions would have rested on a number
  seen in training.
- Carve a **dev slice** out of train (499 examples from 15 companies, also grouped) for checkpoint
  selection, early stopping, and the Bedrock validation file. A fixed 200-example
  **quick-dev** subset of dev is used for fast baselines.
- The result is a committed **split manifest** (`example_id → fit | dev | test`, plus seed,
  dataset revision, and a content hash). Every later stage reads the manifest; nothing re-splits.
- **Test lock**: training, selection, and prompt-iteration code cannot load test rows. The test
  set is unlocked once, in `req_005` v5.01, for all six arms together.

### 4.2 Shared input/output contract

- Input: gold-evidence text (63% of examples have none, and then the context reads `(none)`),
  the whole table as GitHub-Flavored Markdown with cells word for word, the question. A system
  text, the same for every example, describes the whole output language, so the untuned arms
  are told everything the tuned ones learn.
- Output, exactly two fields:

  ```
  PROGRAM: subtract(4850.0, 4474.2), divide(#0, 4474.2)
  ANSWER: 0.08402
  ```

- Allowed operations: `add`, `subtract`, `multiply`, `divide`, `exp`, `greater`, the FinQA
  table helpers (`table_sum`, `table_average`, `table_max`, `table_min`), `#k` step references,
  `const_*` literals, and percent literals such as `6.9%`. The inventory was confirmed against
  the dataset in v1.01 (`reports/data/finqa_schema_report.md`): ten operations, seventeen
  constants, always two arguments, up to six steps.
- The program is the dataset's own, unchanged. The answer is the executed result (the dataset's
  `exe_ans`), not the annotator's free-form answer, with scientific notation written out.
- The wording is in `config/prompt.yaml`; its hash, the prompt id, is recorded with every run.
- One **canonical example record** (system text, user text, target text) is produced once in
  `req_001`. Path A renders it through the Qwen chat template; Path B renders it into Bedrock
  conversation JSONL. Neither path edits the wording.

### 4.3 Evidence setting

Both paths receive the gold supporting facts (`gold_inds`) plus the table, at train and test
time. This is an **oracle-evidence** setting: it isolates program synthesis, which is what is
being compared, but the numbers are not comparable to published FinQA results that include
retrieval. The report states this. A full-context arm for the hosted models is a future
extension (Q7, resolved 2026-11-15: the owner accepted this setting for the primary comparison).

### 4.4 Metrics

| Metric | Definition | Role |
|---|---|---|
| Program execution accuracy | Predicted program parses, executes, and matches the gold answer within tolerance | **Primary** |
| Numeric answer accuracy | The `ANSWER` field matches gold after normalization | Secondary |
| Program exact match | Normalized predicted program equals the gold program | Secondary |
| Format validity | Output passes the strict two-field parser | Diagnostic |

Format and calculation are judged separately (req_002 v2.01): a reply that breaks the two-line
format is still read when it has a program line, so execution accuracy measures the
calculation and format validity measures obedience to the layout. The executor reproduces
every gold result of fit and dev: 4,945 are equal at five decimal places and the other 438 are
within a relative 0.0005%.

The rules were fixed in req_002 v2.02, before any model was scored, and live in
`config/evaluation.yaml`:

- **Two numbers match** when they differ by at most the larger of an absolute allowance and
  a relative allowance times the gold value. Execution accuracy and answer accuracy both
  allow absolute 0.00001 or relative 0.5% (the owner's decision, 2026-11-16: the Gemini
  guide's figure). The gold value is `exe_ans`.
- **`yes` and `no`** match exactly and never match a number.
- **The answer line** may carry a `$`, thousands separators, a sign, and an exponent. An
  answer ending in `%` matches on either scale; without the sign there is no scale
  leniency, so an answer off by 100 is wrong.
- **Program match** compares numbers by value (`4850.0` is `4850`, `const_100` is `100`)
  and keeps the order of arguments and steps; an equivalent program is credited by
  execution accuracy.
- **Every rate is a count over all examples.** A wrong numeric result gets a label for
  error analysis (`scale`, `sign`, `kind`, `other`), which never makes it right.

The harness first proved itself by scoring every **gold** reply: all 8,281 gold programs
reproduce their gold answer (`reports/data/finqa_gold_check_report.md`). An example that
did not would be flagged in `splits/finqa_gold_flags.json` and kept in every total, with
the total over the unflagged examples reported beside it; none is flagged today.

Because all arms answer the same test items, differences are reported with paired statistics
(bootstrap confidence intervals and McNemar's test).

### 4.5 Cost model

Both paths are billed by AWS, so every figure is a dollar amount from the same bill.

| Cost | Path A (Qwen on SageMaker) | Path B (Nova on Bedrock) |
|---|---|---|
| Training | Billable instance-seconds × instance hourly rate | Billed training tokens × per-token price |
| Inference | Billable instance-seconds of the evaluation job; per-question cost depends on utilization | Input and output tokens × per-token price (same as the base model) |
| Fixed / other | S3 storage (negligible) | $1.95 per custom model per month while kept |

Reference prices, us-east-1, AWS price list of 2026-11-11 and 2026-11-14: `ml.g5.xlarge` $1.408
per hour; Nova Pro $8.00 per 1M training tokens, $0.80 / $3.20 per 1M input / output tokens; Nova
Lite $2.00, $0.06 / $0.24. Estimated project spend: Path B about $102 (revised 2026-11-15 from the measured token volume), Path A roughly $7–$21
(unmeasured). Provisioned Throughput ($55–$60.50 per hour) is never used.

All rates live in versioned YAML under `config/` with the price-list date, never in code. Every
run writes time, instance type, and tokens to the **cost ledger**; the report derives cost per
training run, cost per 1,000 questions, and break-even volumes. Because a GPU instance bills per
hour whether busy or idle and Nova bills per token, Path A's serving cost is reported under
stated utilization scenarios (Q10).

### 4.6 Path A training defaults (from the guides; tuned only after a stable first run)

The guides sized these for a 16 GB GPU. They are kept unchanged for the first run on the 24 GB
cloud GPU so the result still reflects the documented recipe.

4-bit NF4 with double quantization, BF16 compute; LoRA `r=16`, `alpha=32`, `dropout=0.05` on
all linear layers; batch 2 × gradient accumulation 8; `paged_adamw_8bit`, LR `2e-4`, cosine
schedule with 5% warmup; max sequence length 2,048 (measured in v1.04; the guides' 1,536 would cut two examples); gradient checkpointing; 2–3 epochs; evaluate
every 100 steps and select the checkpoint with the best **dev execution accuracy**, not the
lowest loss. Greedy decoding, `max_new_tokens=192` (the longest correct reply is 80 tokens).
Both limits live in `config/token_budget.yaml`.

### 4.7 Path B defaults

Amazon Bedrock behind a small `FrontierProvider` interface (stage file, start job, poll,
deploy, infer), with the model id in config so Nova Pro and Nova Lite share all code. Zero-shot
baselines first. One training file in Bedrock conversation JSONL serves both models; it is
validated and its cost estimated before any job starts. Hyperparameters Bedrock exposes:
`epochCount` (1–5), `learningRate`, `learningRateWarmupSteps`. Tuned models are deployed
on-demand. Launching a paid job is an **operator-gated** step under a budget cap (Q2).

### 4.8 Where things run

| Activity | Where |
|---|---|
| Data pipeline, DSL executor, scoring, unit tests | Local machine, CPU |
| Qwen development, smoke runs, quick-dev baseline, demo | Local RTX 4090 Laptop GPU (16 GB) |
| Qwen full training and all reported Qwen evaluations | SageMaker training jobs, one 24 GB GPU instance, us-east-1 |
| Optional full Qwen training run as a learning exercise (v3.06) | Local RTX 4090 Laptop GPU; a side note, not one of the six arms |
| Nova fine-tuning and all Nova inference | Bedrock, us-east-1 |
| Training files, job outputs, adapters | One private S3 bucket, us-east-1; test rows are uploaded only at the v5.01 unlock |

Anything billed is started by the owner after seeing an estimate. SageMaker work runs as jobs
with a maximum runtime, never as an always-on instance.

## 5. Repository layout (target)

```
qlora_finetuning/            parent package (absolute imports: qlora_finetuning.*)
  settings.py                env accessors + load_config() over config/*.yaml
  tracking/                  run registry, cost ledger
  data/                      FinQA snapshot, split manifest, serialization, token budget
  dsl/                       program parser + safe executor
  evaluation/                metrics, Predictor protocol, evaluation runner
  local_qlora/               Path A: 4-bit inference backend, QLoRA training, SageMaker launcher
  frontier/                  Path B: Bedrock adapter, JSONL export, job control
  comparison/                cost model, statistics, report builder
  demo/                      CLI: context + question -> program -> executed answer
config/                      versioned YAML (split, prompt, training, AWS, pricing)
scripts/                     env_check, aws_check, finqa_fetch, finqa_report, finqa_split, finqa_serialize, and other entry points
tests/                       mirrors the package; no GPU, network, or API key required
architecture/                diagrams
artifacts/                   git-ignored: dataset snapshot, run folders, adapters, predictions
ledger/                      committed: index of runs and the cost ledger (JSON Lines)
splits/                      committed: the frozen split manifest; later the test unlock record
reports/                     committed: dataset schema report (reports/data/), comparison report and figures
reqspecs/                    planning, specs, PRPs, references
docs/runbook.md    operator steps
```

## 6. Spec groups

| Group | Owns |
|---|---|
| `req_000_project_scaffold` | Process bootstrap, package skeleton, local environment, run registry and cost ledger, AWS foundation |
| `req_001_finqa_data` | Dataset snapshot, frozen split manifest, canonical serialization, token budget |
| `req_002_dsl_evaluation` | DSL executor, metrics, model-agnostic evaluation runner |
| `req_003_qlora_local` | Path A: Qwen3-4B baseline, QLoRA pipeline, SageMaker packaging, full run and checkpoint selection |
| `req_004_frontier_tuning` | Path B: Bedrock adapter, baselines, training file, fine-tuning jobs for Nova Pro and Nova Lite |
| `req_005_comparison` | Test unlock for all six arms, cost model, comparison report |
| `req_006_packaging_demo` | Demo CLI, model card, adapter package, README |
| `req_007_future_extensions` | Not created; conditional follow-ups |

Version sequences, dependencies, phases, and milestones are in `execution_plan.md`; each group's
`req_NNN_roadmap.md` is the source of truth for its own sequence.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Training stack does not work on native Windows / Python 3.13 | v0.03 environment check; WSL2 fallback; reported runs use Linux on SageMaker regardless |
| SageMaker GPU quota is zero and approval is slow | Quota request is the first AWS step (v0.05), started before it is needed |
| Local and cloud environments drift apart | One set of pinned versions used in both; the cloud job logs its versions and v3.03 checks them |
| A forgotten AWS resource keeps billing | Jobs with a maximum runtime, no Provisioned Throughput, budget alarm, teardown steps in the runbook |
| Leakage between train and test | Grouped split, committed manifest, test-lock guard test |
| Test set tuned against by accident | Single unlock in v5.01; all iteration uses dev |
| Unfair comparison | Identical prompts, identical decoding contract, same scorer, paired statistics |
| Fine-tuning cost overrun | Dry-run cost estimate, budget cap, operator-gated launch |
| Bedrock limits or prices differ from what was read on 2026-11-13/14 | Availability, limits, and pricing re-verified when v4.01 is specced |
| Nova is not what a reader expects from "frontier" | The report names the models plainly and notes that Nova Premier is not fine-tunable |
| Gold programs that do not reproduce gold answers | Measured in v2.02; such examples are flagged in the manifest, not silently dropped |
| Inherited scaffolding from an earlier project (Postgres, FastAPI, EDGAR commands, `uv` vs `venv`) | Cleaned up in v0.01 (Q5, Q6) |

## 8. Licenses and positioning

Qwen3 is Apache 2.0. Amazon Nova is used under the AWS service terms. FinQA annotations are MIT; the underlying tables come from FinTabNet
(CDLA-Permissive-1.0). The model card cites the FinQA paper, states the oracle-evidence setting,
and says the adapter is a research artifact, not financial advice. Licenses are re-checked on the
model and dataset cards before anything is published.
