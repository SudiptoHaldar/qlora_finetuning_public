# req_001 FinQA Data — Roadmap

**Spec group**: req_001_finqa_data  
**Last updated**: 2026-11-15 — every version of this group is ✅ Complete (v1.01 on pull request #2; v1.02, v1.03, and v1.04 on pull request #3). Phase P1 is complete. Training length 2,048 tokens, reply limit 192.  

Version log:

- **2026-11-13** — group created from `reqspecs/0. PLANNING/execution_plan.md` §2; `req_001_01_spec.md` is a placeholder.
- **2026-11-14** — plan revised before any version was specced: measured Qwen runs move to AWS SageMaker; the hosted path is Amazon Nova Pro and Nova Lite on Bedrock (OpenAI dropped); Q5 and Q6 resolved. Version scopes updated in place.
- **2026-11-15** — `req_001_01_spec.md` written for v1.01 (📝 Spec ready). The dataset was downloaded to a scratch folder and profiled on the day. Found: the repository is now `bevaya/FinQA` (`rootsautomation/FinQA` redirects to it), pinned at commit `3d6a736bc67e06bc15fbf3618d88204a57c5b25e`; 8,281 rows in sixteen columns; ten operations, matching the plan; percent literals such as `6.9%` occur as program arguments (not in the plan's list); `answer` is free-form and empty in 74 examples while `exe_ans` is always present; `gold_inds` is reliable and the `ann_*` row lists are not; the official splits group by page only, so 355 of 799 reports span two official splits. Sixteen decisions drafted, pending the owner's review. `requirements.txt` has no Parquet reader; the build adds `pyarrow`.
- **2026-11-15** — v1.01 PRP written (`prp_001_dataset_snapshot_v1.md`, 🛠) and executed the same day (🟡). Built: `config/data.yaml`, `qlora_finetuning/data/snapshot.py` (download, verification on every load, raw reader), `qlora_finetuning/data/schema_report.py`, `scripts/finqa_fetch.py`, `scripts/finqa_report.py`, and the committed report under `reports/data/`. The real snapshot was fetched (three files, checksums match) and the report passes checks S1–S10 on 8,281 examples. 386 tests pass, also with the training libraries blocked; the live Hub check passes. Differences from the spec: files are read with `pyarrow` alone (no `pandas`); one figure in spec section 2.4 was corrected (5,171 table-only and 1,171 both, not 5,170 and 1,172). One item remains for the owner: the CI run on the pull request, which also proves `pyarrow==26.0.1` installs there. New for later versions: `reqspecs/references/tech/finqa_dataset.md`.
- **2026-11-15** — v1.01 ✅ Complete. Pull request #2 was merged to `main`; CI passed on the pull request and on `main`, and `pyarrow==26.0.1` installed on the Linux runner. v1.02 is unblocked.
- **2026-11-15** — `req_001_02_spec.md` written for v1.02 (📝 Spec ready). Candidate splits were measured on the snapshot with the seed fixed beforehand (20261115). Main finding: grouping by report, the plan's default for Q4, still leaks. Annual reports print prior years, so 72% of adjacent-year reports of one company share a row name and number, and 28.4% of checkable test questions would rest on a number seen in training. Grouping by company removes that (0%), so the spec drafts **company** as the key, a change from the plan that needs the owner's confirmation. With a fill rule over SHA-256 order the split lands exactly on 5,383 / 2,898 (65.00% / 35.00%): fit 4,884 in 79 companies, dev 499 in 15, test 2,898 in 43, quick-dev 200. Nineteen decisions drafted: nothing excluded, no stratification, manifest written once, test lock through an unlock record that only v5.01 creates, raw reader restricted by a guard test.
- **2026-11-15** — v1.02 PRP written (`prp_001_frozen_split_v2.md`) and executed the same day (🟡). By proceeding to the PRP the owner accepted the spec's three decisions: **Q4 is resolved as company**, dev is 15 companies, and the seed is used once. Built: `config/split.yaml`; `qlora_finetuning/data/split.py` (SHA-256 ordering, fill rule, write-once manifest), `split_loader.py` (`load_split`, `split_ids`, `manifest_hash`, the test lock), `split_report.py` (statistics per split, leakage, checks P1–P10); `scripts/finqa_split.py`; `splits/` with the manifest and a README; the split report under `reports/data/`. The real build reproduced the spec: fit 4,884 in 79 companies, dev 499 in 15, test 2,898 in 43, quick-dev 200; train is 65.00%. **Content hash `d34fb22e37b78a31ac22287cc8e199ab942d93125f9a80cb0b323862e599a07a`**: milestone M2. `load_split("test")` raises until v5.01 writes the unlock record, and a guard test lists the only files allowed to use the raw reader. 495 tests pass, also with the training libraries blocked. Corrections to the spec made while building: the leakage figures were re-measured on the final rule with header rows excluded (report grouping 28.4%, not 28.6%; chance matches 58, not 65). Newly seen: `exp` has 1 step in fit and 8 in test. One item remains for the owner: the CI run on the pull request.
- **2026-11-15** — Q7 resolved by the owner: the oracle-evidence setting is accepted for the primary comparison. v1.03 has no open question from the plan; its remaining decisions (what goes on the `ANSWER:` line, yes/no answers, table edge cases, the malformed evidence key, programs kept verbatim, the system-prompt wording) are for its spec.
- **2026-11-15** — `req_001_03_spec.md` written for v1.03 (📝 Spec ready). Measured on fit and dev only, through `load_split`: every text evidence entry equals its indexed passage; 3,387 of 5,383 examples (63%) have no text evidence, so their prompt is table and question; no table cell needs escaping; `exe_ans` is in scientific notation in 138 examples; the scale cue ("in millions") is only in a non-evidence passage for 1,482 examples (28%), but that can change the right program in at most 31 (0.6%). Twenty-two decisions drafted. The main ones: context is text evidence only and no passage is added for scale cues; the whole table as Markdown, cells word for word; the program copied unchanged; the answer line is `exe_ans` with exponents written out; the system text describes the whole output language so the untuned arms are not handicapped; golden files use invented rows and the real records are pinned by a committed digest; the wording can change until the first baseline is run. Four items for the owner, in the spec's section 8.
- **2026-11-15** — v1.03 PRP written (`prp_001_canonical_serialization_v3.md`) and executed the same day (🟡). By proceeding to the PRP the owner accepted the spec's four items: no passage added for scale cues; the answer line is `exe_ans` with exponents written out; the draft system text; `(none)` for an empty context. Built: `config/prompt.yaml`; `qlora_finetuning/data/serialization.py` (`build_record`, `serialize_split`, `format_user` for input that is not a dataset example, the prompt id); `serialization_report.py` (lengths, counts, digests, checks C1–C10); `scripts/finqa_serialize.py` with `--check`, `--dump`, and `--show`; four golden records from invented rows; the committed report. The real build matched the spec: 4,884 fit and 499 dev records, 3,387 without text evidence, one skipped key (`text_-1`), 138 answers written out. **Prompt id `d63742a62b76d526ecf3d92df4cbad832846d4da7aa1ef13d15401ac28391641`**, pinned in a test. Fit digest `ce028ca2e912...`, dev digest `9036cbf62a14...`. The smoke example in `scripts/env_check.py` is now built by the serializer; `smoke.max_seq_length` went from 256 to 1024 to fit it, and `env_check --smoke` passes 7/7. 594 tests pass, also with the training libraries blocked (three of them cover a fix to the v0.04 run registry made on the way; see the req_000 roadmap). Measured on the way: the system text is 372 tokens with the Qwen3 tokenizer. One item remains for the owner: the CI run on the pull request.
- **2026-11-15** — `req_001_04_spec.md` written for v1.04 (📝 Spec ready). Measured on the real fit and dev records with the `Qwen/Qwen3-4B` tokenizer at commit `1cfa9a7208912126459214e8b04321603b3df60c` (identical to the smoke model's tokenizer). Full training sequences in fit: 643 tokens at the median, 1,060 at the 99th percentile, 1,862 at most; dev 916 at most. Two fit examples, on one page, exceed the plan's 1,536; none exceeds 2,048. The longest correct reply is 80 tokens, so the 192 limit holds. One pass over fit is 3,254,181 tokens, of which the 372-token system text is 56%; three epochs are 9.76M, against the 14.6M the cost table assumed. Fifteen decisions drafted. For the owner: raise the training length to 2,048 so nothing is dropped (recommended) or keep 1,536 and leave two examples out; keep the system text at its length (recommended) or shorten it before the first baseline; keep the reply limit.
- **2026-11-15** — v1.04 PRP written (`prp_001_token_budget_v4.md`) and executed the same day (🟡). By proceeding to the PRP the owner accepted the spec's three items: training length 2,048, the system text kept, reply limit 192. Built: `config/token_budget.yaml` (tokenizer pin with a content fingerprint, the two limits); `qlora_finetuning/data/token_budget.py` (loads the tokenizer only, renders the prompt with the chat template and thinking off, counts); `token_budget_report.py` (statistics, thresholds, longest records, volumes, checks T1–T6); `scripts/finqa_tokens.py` with `--check` and `--download`; the committed report. The real run reproduced the spec: full sequences in fit 643 at the 50th percentile, 1,060 at the 99th, 1,862 at most; 2 over 1,536, none over 2,048; longest reply 80 tokens; 3,254,181 tokens in one pass over fit, 55.8% of it the system text; quick-dev prompts total 128,495 tokens. 639 tests pass, also with the training libraries blocked. The execution plan's cost table now gives Path B about $102. Still open, for later versions: memory at 2,048 (v3.02 smoke) and the Nova-to-Qwen token ratio (v4.01). One item remains for the owner: the CI run on the pull request. With it, phase P1 is complete.
- **2026-11-15** — v1.02, v1.03, and v1.04 ✅ Complete. Pull request #3 ("Release v1.2") was merged to `main`; CI passed on the pull request and on `main`, on a runner with no snapshot, no tokenizer, and no training libraries. Every version of this group is complete, and so is phase P1. Work continues on branch `setup/training`; next is req_002 v2.01.

This file is the **single source of truth** for the req_001 version sequence. The group owns
**everything between the Hugging Face dataset and a model-ready example: snapshot, frozen split, canonical serialization, and token budget. It extends `req_000_project_scaffold`; it does NOT own scoring (`req_002_dsl_evaluation`) or path-specific rendering (chat template in `req_003_qlora_local`, provider JSONL in `req_004_frontier_tuning`)**.

## Why this group exists

The comparison is only meaningful if both paths train and test on exactly the same examples
with exactly the same wording. The brief sets the ratio: 65% of the data for training, the rest
for testing, fixed before either path runs. That overrides the official FinQA splits, so the
three official splits (6,251 / 883 / 1,147) are pooled and re-split.

Standing decisions: the split is grouped by source report to prevent leakage; a dev slice is
carved from the 65% so the 35% test stays untouched; the manifest is committed and nothing
re-splits; loaders refuse test rows until `req_005` v5.01 unlocks them; prompts use gold evidence
plus a Markdown table (oracle-evidence setting) and are never silently truncated.

**Status legend** (in lifecycle order):

- ⏸ **Conditional** — only starts when a specific trigger fires
- 📋 **Planned** — next-up; depends on a prior version landing
- 📝 **Spec ready** — req_001_NN_spec.md written + decisions locked; PRP pending
- 🛠 **PRP ready** — PRP authored + validated; execution pending
- 🚧 **In flight** — PRP being executed or impl in progress
- 🟡 **Operator-gated** — prepared; waiting on a step only the user performs
- ✅ **Complete** — spec + impl shipped
- 🗄 **Shelved** — parked indefinitely; revival trigger noted

---

## Quick-reference table (version sequence)

| # | Title | Status | Depends on | Trigger to start |
|---|---|---|---|---|
| **v1.01** | **Dataset snapshot and schema report** — Download `bevaya/FinQA` (formerly `rootsautomation/FinQA`) at a pinned commit into `artifacts/data/finqa/` with checksums; raw reader; committed schema report | ✅ Complete | v0.02 | — |
| **v1.02** | **Frozen split manifest** — Pool the official splits; 65/35 train/test grouped by report, seeded; carve dev (500) and quick-dev (200) from train; commit the manifest with seed, dataset revision, and content hash; split-aware loaders; test-lock guard and its test; split statistics | ✅ Complete | v1.01 | — |
| **v1.03** | **Canonical serialization** — Markdown table formatter; gold-evidence selection; system/user/target record (`PROGRAM:` / `ANSWER:`); unit cues preserved; prompt template in `config/`; deterministic output with golden-file tests | ✅ Complete | v1.02 | — |
| **v1.04** | **Token budget report** — Tokenize all fit and dev records with the pinned Qwen3-4B tokenizer (CPU only); committed report; training length and reply limit in config | ✅ Complete | v1.03 | — |

---

## Detail per version

### v1.01 — Dataset snapshot and schema report  ✅

**Status**: ✅ Complete 2026-11-15. CI green on pull request #2 and on `main` (2026-11-15)  
**PRP**: `reqspecs/PRPs/prp_001_dataset_snapshot_v1.md`  
**Implementation summary**: `reqspecs/impl_summary/impl_001_20261115_063228.md`  
**Depends on**: v0.02  
**Spec**: `req_001_01_spec.md`  

**Scope**: Download `bevaya/FinQA` at commit `3d6a736bc67e06bc15fbf3618d88204a57c5b25e` into `artifacts/data/finqa/<commit>/`, checked against recorded SHA-256 checksums on every load. A raw reader returns the 8,281 pooled examples as typed records, nothing dropped. A committed, reproducible schema report (`reports/data/`) gives column names, program syntax, operation and `const_*` inventory, boolean answers, missing or empty fields, duplicates, and examples per report, with counts and ids only. Expected facts live in `config/data.yaml` and are checked.

**Decisions and open questions**: Sixteen decisions in the spec, section 3. The main ones: the real repository name is pinned, not the redirecting one; files are read with `pandas` and `pyarrow` (added to `requirements.txt`), not `datasets`; the report is committed but carries no question, context, or answer text, because any example may become a test example in v1.02. Findings handed to v1.02, v1.03, v2.01, and v2.02 are in the spec, section 6. Open for the owner: whether to keep a second copy of the snapshot in the S3 bucket (spec section 7); until then `artifacts/data/finqa/` on this machine is the only pinned copy.

**How to run**: `venv\Scripts\python -m scripts.finqa_fetch`, then `venv\Scripts\python -m scripts.finqa_report` (runbook section 4).

### v1.02 — Frozen split manifest  ✅

**Status**: ✅ Complete 2026-11-15. CI green on pull request #3 and on `main` (2026-11-15)  
**PRP**: `reqspecs/PRPs/prp_001_frozen_split_v2.md`  
**Implementation summary**: `reqspecs/impl_summary/impl_001_20261115_125357.md`  
**Content hash (M2)**: `d34fb22e37b78a31ac22287cc8e199ab942d93125f9a80cb0b323862e599a07a`  
**Depends on**: v1.01  
**Spec**: `req_001_02_spec.md`  

**Scope**: Pool the official splits; 65/35 train/test grouped by report, seeded; carve dev (500) and quick-dev (200) from train; commit the manifest with seed, dataset revision, and content hash; split-aware loaders; test-lock guard and its test; split statistics.

**Decisions and open questions**: Nineteen decisions in the spec, section 3. **Q4, for the owner**: the spec drafts grouping by **company**, not by report as the plan recommended, because report grouping leaves 28.4% of checkable test questions resting on a number seen in training (spec section 2.3); switching back is one line of config before the manifest is first written. Also for the owner: a dev slice of 499 examples from 15 companies; seed 20261115 with no second try. Drafted: SHA-256 ordering instead of a random generator; a fill rule that lands on 65.00%; all 8,281 examples assigned; manifest in `splits/finqa_split_manifest.json`, written once; `load_split` for `fit`, `dev`, `quick_dev`; `test` locked until v5.01 writes `splits/test_unlock.json`. Reaching it is milestone M2.

### v1.03 — Canonical serialization  ✅

**Status**: ✅ Complete 2026-11-15. CI green on pull request #3 and on `main` (2026-11-15)  
**PRP**: `reqspecs/PRPs/prp_001_canonical_serialization_v3.md`  
**Implementation summary**: `reqspecs/impl_summary/impl_001_20261115_133935.md`  
**Prompt id**: `d63742a62b76d526ecf3d92df4cbad832846d4da7aa1ef13d15401ac28391641` (version 1; open to change until the first baseline is run)  
**Depends on**: v1.02  
**Spec**: `req_001_03_spec.md`  

**Scope**: Markdown table formatter; gold-evidence selection; system/user/target record (`PROGRAM:` / `ANSWER:`); unit cues preserved; prompt template in `config/`; deterministic output with golden-file tests.

**Decisions and open questions**: Q7 resolved 2026-11-15: the owner accepted the oracle-evidence setting for the primary comparison (gold supporting text plus the whole table for every arm), with the report to state that accuracy is higher than on full pages and that the setting may understate the hosted models' advantage on long documents. Drafted in the spec (section 3, twenty-two decisions) and **for the owner** (section 8): (1) scale cues: keep the prompt as evidence text plus table, or also show the table's caption passage when it holds a scale phrase (recommended: keep); (2) the answer line is `exe_ans` with scientific notation written out (recommended: yes); (3) the draft system text, to be read against real prompts after the build and adjusted before the first baseline; (4) `(none)` for an empty context (recommended). Statements in the system text about what operations and percent literals mean are confirmed by v2.02 before any model is run.

### v1.04 — Token budget report  ✅

**Status**: ✅ Complete 2026-11-15. CI green on pull request #3 and on `main` (2026-11-15)  
**PRP**: `reqspecs/PRPs/prp_001_token_budget_v4.md`  
**Implementation summary**: `reqspecs/impl_summary/impl_001_20261115_144212.md`  
**Tokenizer**: `Qwen/Qwen3-4B` at `1cfa9a7208912126459214e8b04321603b3df60c`  
**Depends on**: v1.03  
**Spec**: `req_001_04_spec.md`  

**Scope**: Tokenize all fit and dev prompts with the Qwen3-4B tokenizer (CPU only). Report P50/P90/P98/P99 and maximum, the share above 1,536 tokens, and the explicit policy for over-length examples.

**Decisions and open questions**: Fifteen decisions in the spec, section 3. **For the owner** (spec section 8): (1) training length 2,048 instead of the plan's 1,536, so the two over-length fit examples are kept (recommended), with memory to be proven in v3.02's smoke run; (2) keep the 372-token system text, which is 56% of the training tokens and about $55 of an estimated $98 of hosted training (recommended); (3) reply limit stays 192. Nova tokens are a proxy until v4.01 measures the ratio. Completing this version completes phase P1.
