# FinQA Data

Spec Group      :   req_001_finqa_data  
Feature Name    :   finqa_data  
Version         :   4  
Status          :   ✅ Complete 2026-11-15 (see `reqspecs/PRPs/prp_001_token_budget_v4.md`); CI green on pull request #3 and on `main` (2026-11-15)  
Last updated    :   2026-11-15  

## 1. v1.04 — Token budget report

Before any GPU hour or hosted training token is paid for, this version answers three
questions with measurements instead of the guides' estimates:

- **Does every training example fit** in the sequence length the model is trained with, and
  if not, what happens to the ones that do not? (Never silent truncation.)
- **Is the reply limit long enough** for the longest correct reply?
- **How many tokens** does one pass over the training data hold? That number, times the
  price, is the cost of the hosted path.

It adds a token counter, a committed report, and the configuration values the two training
paths will read. It finishes phase P1.

**Depends on**: v1.03 (built).

## 2. What exists today (verified 2026-11-15)

Measured on the real fit and dev records with the Qwen3-4B tokenizer (tokenizer files only,
CPU). The test split was not read.

### 2.1 The tokenizer

| Item | State |
|---|---|
| Model | `Qwen/Qwen3-4B`, commit `1cfa9a7208912126459214e8b04321603b3df60c`. Public, not gated, Apache 2.0 |
| Tokenizer files | `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`; a few megabytes. No model weights are needed |
| Vocabulary | 151,669 entries. End of sequence is `<\|im_end\|>` |
| Compared with the smoke model's tokenizer (`Qwen/Qwen3-0.6B`) | Identical vocabulary and identical chat template |
| Chat template with `enable_thinking=False` | Adds 17 tokens around the system and user texts, the same for every record. The prompt ends with an empty think block, as `hf_training_stack.md` describes |

### 2.2 Lengths

The **full sequence** is what Path A trains on: the chat-templated system and user texts, the
target, and the end-of-sequence token.

| Tokens | fit (4,884) | dev (499) |
|---|---|---|
| System text | 372 | 372 |
| User text: median / 90th / 98th / 99th / longest | 223 / 383 / 539 / 630 / 1,428 | 247 / 382 / 449 / 476 / 497 |
| Target with end token: median / 99th / longest | 29 / 65 / 80 | 29 / 66 / 70 |
| **Full sequence: median / 90th / 98th / 99th / longest** | **643 / 804 / 962 / 1,060 / 1,862** | **667 / 803 / 866 / 891 / 916** |

| Full sequence longer than | fit | dev |
|---|---|---|
| 768 | 715 (14.6%) | 95 (19.0%) |
| 1,024 | 58 (1.2%) | 0 |
| 1,280 | 5 (0.1%) | 0 |
| **1,536** (the plan's limit) | **2 (0.04%)** | 0 |
| 2,048 | 0 | 0 |

The two fit examples over 1,536 are on one page (`SLG/2009/page_84.pdf-1` and `-2`), at 1,851
and 1,862 tokens. The next longest is 1,451.

One token is about 3.0 characters of these texts.

### 2.3 Volumes

| | Tokens |
|---|---|
| One pass over fit, full sequences | 3,254,181 |
| of which the system text (372 × 4,884) | 1,816,848 (56%) |
| Three passes over fit | 9,762,543 |
| All dev sequences once | 333,677 |
| All dev prompts once, without targets | 317,882 |

The system text is more than half of everything a model is trained on.

### 2.4 What the plan and the guides assumed

| Assumption | Where | Measured |
|---|---|---|
| Sequence length 1,536 | Plan §4.6, Gemini guide | 2 fit examples do not fit; 2,048 fits everything in fit and dev |
| Reply limit 192 tokens | Plan §4.6 | Longest correct reply is 80 tokens with its end token |
| About 1,000 tokens per example; 14.6M training tokens for three epochs | Execution plan cost table | 666 per example on average; 9.76M for three epochs |

## 3. Locked decisions

| # | Decision | Reason |
|---|---|---|
| D1 | Tokens are counted with the tokenizer of `Qwen/Qwen3-4B` at the commit in section 2.1, pinned in `config/token_budget.yaml`. v3.01 loads the model at the same commit | The budget must describe the model that is trained |
| D2 | The counted unit is the **full sequence**: the tokenizer's own chat template over the system and user texts, with the generation prompt and `enable_thinking=False`, then the target, then the end token. The parts are counted too | This is what Path A puts in memory. v3.01 owns the renderer used for models; its token counts must equal these for the same record, and a test there says so |
| D3 | The counter takes the tokenizer as an argument and needs no GPU and no model weights | Tests pass a stand-in; CI has no training stack |
| D4 | **Nothing is ever truncated.** A record longer than the training length is an error that names the example, in the report and later in the training code | Plan: never silently truncate the question or the table. A truncated prompt can lose its question, and a truncated target is a wrong program |
| D5 | **The training length becomes 2,048 tokens**, up from the plan's 1,536 | It fits every fit and dev record with 186 tokens to spare, so no example is dropped and both paths train on exactly the same examples. Only the two longest records are affected; every other batch is as long as before. **Alternative for the owner in section 8** |
| D6 | The peak memory of a batch holding the two longest records is measured in v3.02's smoke run, on the laptop and before any paid job. If it does not fit, the fallback is the alternative of section 8, by the owner's decision | 2,048 has not been run on a GPU. This version cannot test it |
| D7 | The reply limit stays at 192 new tokens | 2.4 times the longest correct reply. A wrong reply that runs on is cut there, which is the point of a limit |
| D8 | There is no length limit at inference. Dev and, later, test prompts are sent whole | The model's context is 131,072 tokens; the limit of D5 is about training memory only |
| D9 | The test split is not measured here. v5.01 measures it at the unlock, with this code, and reports it | The lock. A long test prompt is not a problem (D8) |
| D10 | Hosted-model tokens cannot be counted locally: Amazon publishes no Nova tokenizer. Qwen token counts stand in for them, and the real ratio is taken from the token usage that Bedrock returns for the quick-dev baselines (v4.01), before the paid training (v4.02) | Section 2.3 is a proxy for Path B until then |
| D11 | The report holds token counts and example ids. It holds no dollar figure; prices are applied by v4.02 and v5.02 | req_000 v0.04 D1: quantities and prices are kept apart |
| D12 | **The system text stays as it is.** Its cost is now known (section 2.3) and is inside the budget | Two of the six arms are untuned and need the full description. **Alternative for the owner in section 8** |
| D13 | The report is committed as `reports/data/finqa_token_budget_report.md` and `.json`, with no timestamp. It names the tokenizer commit, the prompt id, and the split's content hash | Same rules as the other reports; a change in any of the three inputs is visible |
| D14 | The limits live in `config/token_budget.yaml` (`max_length`, `max_new_tokens`) and are read from there by v3.01, v3.02, and v4.02 | Working agreement 13. `max_length` is the name TRL uses |
| D15 | The tokenizer is loaded from the local Hugging Face cache; downloading it is an explicit option of the script | The script otherwise runs offline, like the other data scripts |

## 4. Scope to build

### 4.1 `config/token_budget.yaml`

`version: 1`; `tokenizer.model_id`, `tokenizer.revision`; `max_length: 2048`;
`max_new_tokens: 192`; `thresholds: [768, 1024, 1280, 1536, 2048]`; `longest_listed: 10`;
`report_dir: reports/data`.

### 4.2 Counter: `qlora_finetuning/data/token_budget.py`

| Function | Behavior |
|---|---|
| `load_token_budget_config()` | Reads and validates the file; the revision must be a full commit hash |
| `load_tokenizer(cfg, download=False)` | Imports `transformers` inside the function and loads the pinned tokenizer from the local cache; with `download=True` it may fetch it. A missing library or missing files raises with the command to run |
| `render_prompt(record, tokenizer)` | D2: the chat-templated prompt text |
| `count_record(record, tokenizer)` | Returns the token counts of the system text, user text, target with end token, prompt, and full sequence |
| `count_split(records, tokenizer)` | The counts for every record, in order |

### 4.3 Report: `qlora_finetuning/data/token_budget_report.py`

For fit, dev, and quick-dev: records; system-text tokens; template tokens; minimum, median,
90th, 98th, 99th percentile, and maximum of user, target, prompt, and full-sequence tokens;
records over each threshold; the longest records by id; total tokens of full sequences and of
prompts alone; the share that is system text; characters per token. Then the inputs (tokenizer
commit, prompt id, split content hash) and the checks:

| Id | Check | Passes when |
|---|---|---|
| T1 | Fits | No fit or dev full sequence is longer than `max_length` |
| T2 | Reply limit | No target with its end token is longer than `max_new_tokens` |
| T3 | Tokenizer | The loaded tokenizer is the pinned model and commit |
| T4 | Thinking off | Every prompt ends with the empty think block, and no target contains one |
| T5 | Template constant | The template adds the same number of tokens to every record |
| T6 | Same records | The records counted are the ones the serialization report pins (same prompt id and digests) |

### 4.4 `scripts/finqa_tokens.py`

| Command | Behavior |
|---|---|
| `venv\Scripts\python -m scripts.finqa_tokens` | Counts fit and dev, runs T1 to T6, writes the report |
| `... --check` | Writes nothing; fails if the report differs |
| `... --download` | Fetches the tokenizer files first (a few megabytes, once) |

Exit codes 0, 1, 2. Needs the training stack for `transformers`; a missing stack or tokenizer
is exit 2 with the fix.

### 4.5 Tests

- Counter and report on invented records with a stand-in tokenizer (splits on spaces, has a
  small chat template): counts, percentiles, thresholds, longest list, totals, each check
  passing and failing, deterministic output, no example text in the output.
- Config: bad revision, limits that are not positive whole numbers.
- `load_tokenizer` without the library, and without cached files, raises with the fix.
- Script: exit codes; `--check`; module form; by-path hint.
- With the real tokenizer and the snapshot present: the committed report matches. Skips
  otherwise, so CI and a machine without the training stack stay green.

### 4.6 Documentation

- Runbook section 4: count tokens, check, what to do when a record does not fit.
- `high_level_plan.md` §3 and §4.6: sequence length 2,048; reply limit confirmed.
- `execution_plan.md` cost table: the measured volumes and the revised Path B estimate.
- `hf_training_stack.md`: the 4B tokenizer equals the 0.6B one; template adds 17 tokens.
- `gotchas.md`: the "long contexts are cut silently" row becomes measured.

## 5. Out of scope

- Loading the model, training, or measuring memory (v3.01, v3.02).
- The renderer used for models (v3.01) and the Bedrock training file (v4.02).
- Prices and dollar estimates in code or in the report (v4.02, v5.02).
- Changing the wording to save tokens: possible until the first baseline, but a decision of
  its own (section 8).

## 6. What this hands to later versions

| For | What |
|---|---|
| v3.01 | The tokenizer commit; `max_new_tokens`; the token counts its renderer must reproduce |
| v3.02 | `max_length`; "over the length is an error, never truncation"; the two longest fit records, to put in the smoke run's batch and measure memory |
| v4.01 | Prompt tokens of quick-dev (roughly 127,000 for the 200 prompts; this version measures it), to compare with what Bedrock reports and get the Nova-to-Qwen ratio |
| v4.02 | Fit and dev token volumes for the cost estimate and the budget gate |
| v5.01 | The same counter for the test split after the unlock |

## 7. What it means for cost

Not part of the build; planning information. With the reference prices in the execution plan
and Qwen tokens standing in for Nova tokens (D10):

| Item | Earlier estimate | From the measured volume |
|---|---|---|
| Nova Pro fine-tuning, three epochs | 14.6M tokens, about $117 | 9.76M tokens, about $78 |
| Nova Lite fine-tuning, three epochs | about $29 | about $20 |
| Share of that which is the system text | not known | 56%: about $44 for Pro and $11 for Lite |

Every 100 tokens of system text cost about $12 on Nova Pro and $3 on Nova Lite over three
epochs. The real figures depend on Nova's own tokenizer and move with the ratio of D10.

## 8. Decisions for the owner

All three were accepted on 2026-11-15, when the owner proceeded to the PRP.

| # | Question | Recommendation |
|---|---|---|
| 1 | **Training length**: raise it to 2,048 so everything fits (D5), or keep the plan's 1,536 and leave the two longest fit examples out of training on both paths? | Raise it. No example is dropped and no exclusion list is needed. Keeping 1,536 stays closer to the guides' recipe and avoids an untested memory peak, at the cost of 2 of 4,884 examples and a list both paths must honor. If the memory test of D6 fails, this is the fallback |
| 2 | **System text**: keep it at 372 tokens (D12), or shorten it before the first baseline? | Keep it. It is about $55 of an estimated $98 of hosted training, within the $300 cap, and the untuned arms depend on it. Shortening is still possible until the first baseline if the v4.02 estimate comes out higher |
| 3 | Reply limit stays 192 (D7)? | Yes |

## 9. Known limits and follow-ups

- **2,048 is untested on a GPU** (D6).
- **Nova token counts are a proxy** until v4.01 (D10). The cost figures of section 7 carry
  that uncertainty.
- **The test split's lengths are unknown** until the unlock. They do not affect training.
- **The figures in section 2 come from a throwaway script.** The built report must reproduce
  them; any difference is investigated first.
- **Batches are padded to their longest record.** The two long records sit on one page; if
  both land in one batch, that batch is the memory peak. v3.02 tests exactly that.

## 10. Acceptance

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with and
  without the training stack installed.
- `venv\Scripts\python -m scripts.finqa_tokens` passes T1 to T6 and writes the report; a
  second run changes nothing; `--check` exits 0.
- The report agrees with section 2: system text 372 tokens; fit full sequences 643 at the
  median and 1,862 at most; 2 fit records over 1,536 and none over 2,048; 3,254,181 tokens in
  one pass over fit.
- The report holds no example text and no dollar figure.
- `high_level_plan.md` and the execution plan's cost table carry the measured values.
- Phase P1 is complete.

## 11. Next step

Complete; CI green on pull request #3 and on `main` (2026-11-15). Phase P1 is complete; next is req_002 v2.01.
