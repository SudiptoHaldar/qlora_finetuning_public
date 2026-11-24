# Frontier Tuning

Spec Group      :   req_004_frontier_tuning  
Feature Name    :   frontier_tuning  
Version         :   2  
Status          :   ✅ Complete: code built 2026-11-16 (`prp_004_training_export_gate_v2.md`); on 2026-11-18 the owner opened the gate (Path B estimate $82.46 of the $200 cap) and staged both files in S3 (section 11)  
Last updated    :   2026-11-18  

## 1. v4.02 — Training file export and budget gate

Before any money is spent on fine-tuning, this version produces the two files Bedrock will
train on, proves they are what they should be, and says what the training will cost.

It adds:

- the **export**: the fit split as the training file and the dev split as the validation
  file, in Bedrock's conversation format, one file pair for both models;
- a **validator** that reads the files back from disk and checks them against the
  canonical records, the test lock, and AWS's format rules;
- a **cost estimate** for Path B from the token volume, the measured Nova to Qwen token
  ratio, and `config/pricing.yaml`;
- the **budget gate**: a committed record that says whether the estimate is inside the cap.
  v4.03 will refuse to start a job without a passing record for exactly these files;
- the **upload** of the two files to S3, with a check that what arrived is what was made.

It starts no job and calls no model. The upload is the only step that changes anything in
AWS, and the owner runs it. Storing 12 MB in S3 costs well under a cent a month.

**Depends on**: v4.01. Its code is built; **its baseline runs have not been made yet**, and
the gate cannot pass until they have, because the token ratio comes from them (D17).
Everything else in this version can be built and run before that.

## 2. What exists today (verified 2026-11-16)

### 2.1 What earlier versions provide

| Item | State |
|---|---|
| `serialize_split("fit" \| "dev")` | 4,884 and 499 canonical records (system, user, target), in dataset order |
| `split_ids("test")` | The 2,898 test ids, readable while the split is locked: ids only |
| `canonical_record.md` | Already says how Path B uses a record: the `system` field, one `user` turn, and the target as the `assistant` turn |
| Token budget report (v1.04) | Qwen tokens in one pass: fit 3,254,181, dev 333,677. Longest record: fit 1,862, dev 916. It names the prompt id and split hash it was made from |
| `config/pricing.yaml` (v4.01) | Training: Nova Pro $8.00, Nova Lite $2.00 per 1M tokens. Storage $1.95 per model per month. Price list published 2026-11-11 |
| `reports/frontier/nova_baseline_report.json` (v4.01) | **Does not exist yet.** It will hold `nova_to_qwen_input_ratio` for four runs |
| `config/aws.yaml` | Bucket `qlora-sagemaker-123456789012`, prefix `data/` for training and validation files; budget cap $300 for both paths together |
| Bedrock customization role | May read `data/*` and read and write `bedrock/*` in that bucket (v0.05) |
| `config/tracking.yaml` | Run kind `export` exists |
| `qlora_finetuning/frontier/` | `config`, `provider`, `bedrock`, `predictor`, `pricing`, `baseline_report` |

### 2.2 What AWS documents (read 2026-11-16)

From the Nova user guide page on preparing fine-tuning data, the Bedrock pages on preparing
data and on analyzing job results, and AWS's own dataset validation script
(`aws-samples/amazon-nova-samples`, `nova_ft_dataset_validator.py`), which was read, not run.

| Topic | Finding |
|---|---|
| Record | One JSON object per line: `schemaVersion` (`bedrock-conversation-2024` in every example; the guide says it "can be any string value"), an optional `system` list of `{"text"}`, and `messages`, each with a `role` and a `content` list of `{"text"}` |
| Turns | Roles are `user` and `assistant`, alternating, first `user`, last `assistant`. One user turn and one assistant turn is a valid record |
| Reserved words | The guide: `User:`, `Bot:`, `Assistant:`, `System:`, `<image>`, `<video>`, `[EOS]` anywhere in a user or system prompt make the job "fail due to data issues". The script also refuses `Thought:` and `<unk>`, compares **without regard to case**, and checks every text, the assistant's included |
| Text | A content item may not be empty |
| File names | "only alphanumeric characters, underscores, hyphens, slashes, and dots" |
| Size | 8 to 20,000 samples for Pro and for Lite; 32k tokens of context per sample; training plus validation records at most 20,000 |
| File size | Not stated for Nova. Other Bedrock models: 1 GB training, 100 MB validation |
| Advice | "Use the same system prompt during both training and inference"; "Deduplicate and review your data before training" |
| Job output | `step_wise_training_metrics.csv` and, for some jobs, `validation_metrics.csv`, written to the output location. Neither holds a billed token count |
| Billing | Training is priced per token (`config/pricing.yaml`). The plan's reading is tokens in the training file times epochs. Whether validation tokens are billed is **not confirmed** |
| Token counting | Bedrock has a free `CountTokens` call whose result "will match the token count that would be charged". Whether it supports Nova Pro and Lite is **not confirmed**: the page refers to each model's card |

### 2.3 A trial export

A throwaway script rendered both splits in the format of D2. Counts only.

| Fact | fit | dev |
|---|---:|---:|
| Records | 4,884 | 499 |
| File size | 10.7 MB | 1.1 MB |
| Longest line | 5,533 bytes | 3,330 bytes |
| Ids that are test ids | 0 | 0 |
| Records with a reserved word (nine words, any case, in system, user, or target) | 0 | 0 |
| Empty user or target texts | 0 | 0 |
| Records with a character outside ASCII | 0 | 0 |
| Carriage returns, or control characters other than line break and tab | 0 | 0 |
| Lines identical to an earlier line | 52 | 2 |
| Prompts identical to an earlier prompt | 59 | 2 |

So the files pass every documented rule as they are and are a hundredth of any stated size
limit. Two things stand out. FinQA repeats itself: 52 fit records are exact copies of
another, and **7 more have the same prompt as another record and a different gold reply**.
And the system text, 1,331 characters, is in every line: it is 56% of the training tokens.

### 2.4 What the training will cost

Qwen token counts times the price, for both models together. The Nova to Qwen ratio is not
known yet; 1.5 is the worst case v4.01 assumed.

| | If Nova counts as Qwen does | If Nova counts 1.5 times as many |
|---|---:|---:|
| One epoch, Nova Pro | $26.03 | $39.05 |
| One epoch, Nova Lite | $6.51 | $9.76 |
| **2 epochs, both models** | **$65.08** | **$97.63** |
| **3 epochs, both models** | **$97.63** | **$146.44** |
| Evaluation still to come (B1 on dev, four runs on test) | about $4 | about $6 |
| Storage of two tuned models, per month | $3.90 | $3.90 |

If validation tokens turn out to be billed like training tokens, training costs 10% more.
The evaluation figure estimates the locked test split from fit's average prompt.

## 3. Locked decisions

### The files

| # | Decision | Reason |
|---|---|---|
| D1 | Two files: `finqa_fit.jsonl` (training, 4,884 records) and `finqa_dev.jsonl` (validation, 499), made once and used by both models | Plan §4.7. **Whether dev is given to Bedrock at all is a decision for the owner (section 8)** |
| D2 | A line is `{"schemaVersion": "bedrock-conversation-2024", "system": [{"text": system}], "messages": [{"role": "user", "content": [{"text": user}]}, {"role": "assistant", "content": [{"text": target}]}]}` with the three texts of the canonical record, **character for character** | v1.03. It is also exactly what `BedrockProvider` sends at inference, so a tuned model is asked the way it was taught |
| D3 | Lines are in dataset order. Keys are in the order of D2, with compact separators, UTF-8, characters outside ASCII written as themselves, one line break (`\n`) after every line | The same input always gives the same bytes, so a file is known by its SHA-256 |
| D4 | **Every record of the split is exported, the repeated ones too** | Both paths must train on the same 4,884 examples. Dropping the 52 copies here and not on Path A would be a difference between the arms. The report counts them |
| D5 | The line holds no example id. A sidecar `finqa_fit.ids.txt` (and `finqa_dev.ids.txt`), one id per line in file order, stays local | Bedrock's format has no place for an id, and the validator needs to know which record a line is |
| D6 | Files are written under git-ignored `artifacts/bedrock/<export_id>/`, where the export id is the first 12 characters of the SHA-256 over both files' digests | Dataset text is never committed. The same content always lands in the same folder |
| D7 | Only `fit` and `dev` can be exported. The function takes no split name from the caller, and nothing in this version loads `test` | Working agreement 6: no test row leaves this machine before v5.01 |

### Validation

| # | Decision | Reason |
|---|---|---|
| D8 | The validator reads the **files on disk**, never the records in memory | What is uploaded is the file |
| D9 | It is our own code, with AWS's rules written out as checks (section 4.3). AWS's script is not vendored or run by the tools | It pulls in other libraries and changes without notice. Running it once by hand is an optional owner step (section 7) |
| D10 | Reserved words are checked the strict way: nine words (`system:`, `user:`, `bot:`, `assistant:`, `thought:`, `[eos]`, `<image>`, `<video>`, `<unk>`), any case, anywhere, in all three texts | The guide and the script disagree; a file that passes the stricter reading passes both |
| D11 | A reserved word found is an **error that names the example**. Nothing is rewritten | Changing a text for Nova alone would break "same words for every arm". None is found today; if the prompt wording ever changes, this check speaks first |
| D12 | The per-sample limit is checked from the committed token budget report: the longest record's Qwen tokens, times the token ratio, times the margin, must be at most 32,000 | No tokenizer needs loading. The longest record is 1,862 tokens, so the answer is not close |

### The estimate and the gate

| # | Decision | Reason |
|---|---|---|
| D13 | Training tokens of a model are estimated as: Qwen tokens in one pass over fit (from the token budget report) times the Nova to Qwen ratio times the epoch count | Amazon publishes no Nova tokenizer. v4.01 measures the ratio on real prompts |
| D14 | The ratio is the **largest** `nova_to_qwen_input_ratio` of the runs in the v4.01 baseline report | Four measurements, two models; the estimate should err high |
| D15 | A margin from config (10%) is added to the training estimate before it is compared with the cap | The ratio is measured on prompts, not on targets, and validation tokens may be billed (10% of fit) |
| D16 | The estimate covers all of Path B: what the ledger already holds for Path B at list price, training for both models, the evaluation runs still to come, and one month of storage for two models | The cap is for the path, not for one job |
| D17 | **Without a baseline report that passed its checks and matches the current prompt id and split hash, the gate fails.** The estimate is still printed, with the assumed ratio from config (1.5) and the words "not measured" | A gate that passes on a guess is not a gate |
| D18 | The epoch count comes from `config/frontier.yaml`. The report also shows the cost at 1 to 5 epochs | v4.03 launches with the count the gate saw. **Decision for the owner in section 8** |
| D19 | The cap is `budget.path_b_cap_usd` in `config/frontier.yaml`. **Decision for the owner in section 8** | Q2 set $300 for both paths and left the split to this version |
| D20 | The outcome is a committed **gate record**, `reports/frontier/nova_training_gate.json` with a Markdown twin: the two files' SHA-256, sizes, and record counts, the prompt id and split hash, every number of the estimate, the cap, the verdict, and after an upload the S3 locations. Counts, hashes, and dollars only | v4.03 starts a job only if this record says passed and names the very files in S3 |
| D21 | Dollars are computed with `Decimal` from `config/pricing.yaml` and never written to the ledger | v0.04 |

### The upload

| # | Decision | Reason |
|---|---|---|
| D22 | Upload is a separate, explicit step (`--upload`) and is refused unless validation passed. It does not need the gate to pass | The files can be staged while the baselines are still to run; nothing is billed by staging |
| D23 | Keys: `data/bedrock/<export_id>/finqa_fit.jsonl` and `.../finqa_dev.jsonl` | Under `data/`, which the Bedrock role may read; the name carries the content |
| D24 | Each object is sent with a SHA-256 checksum and read back with `head_object`; size and checksum must equal the local file's. An object already there with the same checksum is left alone | What was validated is what Bedrock will read |
| D25 | The export is one run in the registry (kind `export`, path B, arm `none`), with one ledger record: the records exported and the wall clock | v0.04: every entry point leaves a trace. No token count: training tokens are recorded by v4.03 when a job starts |
| D26 | The owner runs `--upload`. Everything else Claude may run | Coding rule: anything that creates or changes an AWS resource is the owner's |

## 4. Scope to build

### 4.1 Config

`config/frontier.yaml`, `version: 2`. New keys:

```yaml
customization:
  epoch_count: 2                 # section 8, decision 1
export:
  schema_version: bedrock-conversation-2024
  dir: artifacts/bedrock         # git-ignored
  s3_prefix: data/bedrock/
budget:
  path_b_cap_usd: 200            # section 8, decision 2
estimate:
  assumed_token_ratio: 1.5       # shown, never gated on, when no ratio has been measured
  margin_percent: 10
  storage_months: 1
```

Existing keys do not change, so runs made under version 1 can still be resumed: the
resume check compares settings, not the file's version.

### 4.2 Export: `qlora_finetuning/frontier/export.py`

| Item | Behavior |
|---|---|
| `conversation_line(record, schema_version)` | The line of D2 and D3, without the line break |
| `export_files(cfg, context)` | Renders fit and dev, writes the two files and the two id lists under the export folder, returns an `Export` (paths, SHA-256, bytes, record counts, export id, prompt id, split hash) |
| `RESERVED_WORDS` | The nine of D10 |

### 4.3 Validation: `qlora_finetuning/frontier/export_validation.py`

`validate_export(export, context) -> list of checks`. Each check reads the files again.

| Id | Check | Passes when |
|---|---|---|
| X1 | Same records | Each file has one line per record of its split, in order (fit 4,884; dev 499), and each line's three texts equal the canonical record's |
| X2 | No test rows | No id in either id list is a test id, and the two lists share no id |
| X3 | Format | Every line is one JSON object with exactly the keys of D2, roles `user` then `assistant`, one non-empty text each; the file is UTF-8, ends with a line break, and holds no carriage return; file names use only letters, digits, `_`, `-`, `.` |
| X4 | Reserved words | No text holds one (D10). A failure names the examples and the word |
| X5 | Limits | 8 to 20,000 training records; training plus validation at most 20,000; the longest record within 32,000 tokens (D12) |
| X6 | Same question | The prompt id and split hash are the current ones and those the token budget report was made from |
| X7 | Reproducible | Rendering again gives the same SHA-256 for both files |

### 4.4 Estimate and gate: `qlora_finetuning/frontier/training_estimate.py`

| Item | Behavior |
|---|---|
| `load_token_ratio(cfg)` | D14 and D17: the measured ratio and where it came from, or the assumed one marked as such |
| `estimate_path_b(...)` | D13 to D16, as plain values: per model and in total, at the configured epoch count and at 1 to 5 |
| `gate(estimate, checks, cfg)` | Two more checks: |

| Id | Check | Passes when |
|---|---|---|
| G1 | Ratio measured | D17 |
| G2 | Within the cap | The Path B total, training margin included, is at most the cap |

`write_gate_record(...)` writes D20. The record says `passed` only when X1 to X7, G1, and
G2 all pass.

```python
from qlora_finetuning.frontier.training_estimate import read_gate_record

record = read_gate_record()
record["passed"]                          # True only when every check passed
record["files"]["training"]["sha256"]     # what v4.03 compares with the object in S3
record["estimate"]["training_usd"]        # {"amazon.nova-pro-v1:0": "...", ...}
```

### 4.5 Upload: `qlora_finetuning/frontier/staging.py`

`stage_export(s3, export, aws, cfg)`: D22 to D24, with the S3 client passed in. It returns
the two S3 URIs and what `head_object` reported. `BedrockProvider` is not touched; staging
is the first step the `FrontierProvider` interface will gain in v4.03, where it is used.

### 4.6 `scripts/nova_export.py`

| Command | Behavior |
|---|---|
| `venv\Scripts\python -m scripts.nova_export` | Export, validate, estimate, write the gate record. Prints one line per check, the estimate, and the verdict. Calls no AWS service |
| `... --upload` | The same, then stages the files (the owner) and writes the S3 locations into the gate record |
| `... --check` | Writes nothing: fails if the committed gate record differs from a fresh one |

Exit codes: 0 every check passed, the gate included; 1 a check failed (the files and the
record are still written, the record says not passed); 2 a config problem, no snapshot, or,
with `--upload`, no sign-in.

### 4.7 Tests

On invented rows and stub clients. No test needs the network, credentials, or the snapshot.

- Line: the three texts round-trip character for character (line breaks, tabs, quotes,
  backslashes, characters outside ASCII); key order; the same record twice gives the same
  bytes; a golden line.
- Export: counts and order; the id lists; the folder name follows the content; a second
  export writes identical files; only fit and dev are ever loaded (the loader is watched).
- Each of X1 to X7 passes and fails: a changed character, a missing, extra, or swapped
  line, a test id, a wrong role, an extra key, an empty text, a carriage return, a missing
  final line break, each reserved word in each text and in upper case, 7 and 20,001
  records, a ratio that pushes the longest record past the limit, another prompt id.
- Estimate: exact `Decimal` arithmetic on known numbers; the largest ratio is taken; no
  report, a failed report, and a report for another prompt each fail G1 and print the
  assumed ratio; the cap boundary passes at equal and fails one cent above; the ledger's
  Path B tokens are counted once.
- Gate record: holds no text of any example; `passed` false when any check fails;
  `--check` detects a stale record.
- Staging: keys and checksums; an object that is already there and equal is not sent
  again; a size or checksum mismatch is an error; refused when validation failed.
- Script: each exit code; module form and by-path hint; without `--upload` no client is
  created.
- With the snapshot present: the real export gives 4,884 and 499 lines and passes X1 to X7.

### 4.8 Documentation

- Runbook section 6: make the files, read the estimate, upload, what each check means,
  how to delete the staged files.
- `bedrock_nova.md`: the format rules of section 2.2 with their sources.
- `canonical_record.md`: the line of D2 as built.
- `gotchas.md`: reserved words; repeated records; file names.
- `coding.md`: training files come only from `export_files`; no job starts without a
  passing gate record.
- `administration/aws_commands_run.md`: the upload, when the owner runs it.

## 5. Out of scope

- Starting, polling, or stopping a customization job; deployments; hyperparameters other
  than the epoch count (v4.03).
- Removing repeated records, or changing any text.
- Uploading test rows (v5.01), or anything for Path A (v3.03 uploads its own files).
- An exact token count from Bedrock's `CountTokens` (section 7).
- Reconciling the estimate with the AWS bill (v5.02).

## 6. What this hands to later versions

| For | What |
|---|---|
| v4.03 | The gate record: start a job only if it passed, the configured epoch count is the one it saw, and the objects in S3 have the recorded checksums. The S3 URIs for `trainingDataConfig` and `validationDataConfig`. The estimated training tokens, for the ledger record marked `computed` |
| v4.04 | The line format is the request format, so `nova-pro` and `nova-lite` with a deployment ARN ask a tuned model as it was taught |
| v4.05 | The cost of one more epoch, already in the report |
| v5.02 | The estimate beside the bill |

## 7. Known limits and follow-ups

- **The token count is an estimate.** It rests on a ratio measured on the 499 dev prompts and on
  the Qwen count of the targets. If a model's card says `CountTokens` supports it, an
  exact, free count is possible and would replace D13.
- **What Bedrock bills for is not fully documented.** Tokens times epochs is the plan's
  reading; validation tokens may be added. The margin covers that case, and v4.03 records
  what the bill says.
- **Our validator is not AWS's.** A rule AWS adds later is unknown to it. The first job is
  the real test; a job that fails on data issues is expected to cost nothing, which is also
  not confirmed. Running AWS's script by hand on the two files is cheap insurance.
- **Seven fit prompts have two different gold replies.** Both paths train on them. They
  are noise, not a fault of the export.
- **The system text is 56% of the training tokens.** It is the price of telling the
  untuned arms everything; training without it would break D2's "asked as taught".
- **The test estimate is a guess** from fit's average prompt until v5.01.
- **Staged files stay in S3** until deleted. They hold dataset text from fit and dev only.

## 8. Decisions for the owner

**Taken as recommended on 2026-11-16**: the owner started the PRP and its execution without
changing any of the three. (1) The gate prices 2 epochs. (2) The cap for Path B is $200.
(3) Dev is the validation file. The table is kept as the record of what was asked.

| # | Question | Recommendation |
|---|---|---|
| 1 | **How many epochs does the gate price?** 2 is Bedrock's default for both models; the plan's earlier estimate assumed 3. Both models: $65 to $98 for 2, $98 to $146 for 3, depending on the token ratio | 2. It is the documented default, so the first result reflects the service as offered, the way Path A keeps the guides' settings. It leaves room in the budget for the one corrective iteration of v4.05, which may add an epoch for a model that missed |
| 2 | **What is the cap for Path B?** $300 covers both paths; Path A is guessed at $7 to $21 | $200. It covers 2 epochs on both models at the worst-case ratio (about $119 with the margin, evaluation, and storage) and then a corrective rerun of Nova Lite. A corrective rerun of Nova Pro fits only if the measured ratio is nearer 1 than 1.5; otherwise the cap is revisited then. It leaves $100 for Path A, several times its estimate |
| 3 | **Is dev given to Bedrock as the validation file?** With it, the job reports a validation loss per step. It trains on fit only either way. The cost, if validation tokens are billed, is up to 10% of training | Yes. The loss curve is the only view into the job, and v4.05 needs it to choose between "more epochs" and "another learning rate". Path A also looks at dev while training, so the two paths stay alike |

## 9. Acceptance

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
  and without the training stack blocked, with no network and no credentials.
- `venv\Scripts\python -m scripts.nova_export` writes two files of 4,884 and 499 lines;
  X1 to X7 pass; the estimate is printed; the gate record is written.
- Before the v4.01 baselines exist, that run exits 1 with G1 failed and the estimate
  marked "not measured". **This is the expected state on the day the code is built.**
- No committed file holds a line of either training file.

Live, by the owner (🟡), after the v4.01 baseline report is committed:

```powershell
venv\Scripts\python -m scripts.nova_export            # G1 and G2 pass: the gate is open
aws sso login --profile qlora
venv\Scripts\python -m scripts.nova_export --upload   # stages two files; pennies a year
```

- The gate record says passed, names both S3 objects, and is committed.
- The estimate at the measured ratio is written into this spec as section 11.

## 10. Next step

This version is complete: the gate is open and the files are staged. Commit the gate record
and the ledger. Next is v4.03's live part, starting with Nova Lite: `python -m
scripts.nova_tune plan`, then `launch --model nova-lite --yes` (run sheet, stage D).

## 11. As built (2026-11-16)

`venv\Scripts\python -m scripts.nova_export`, on the real data, with no AWS call:

| | Training file | Validation file |
|---|---:|---:|
| Split | fit | dev |
| Records | 4,884 | 499 |
| Bytes | 10,715,093 | 1,106,662 |
| Repeated lines (kept) | 52 | 2 |
| Checks X1 to X7 | pass | pass |

Export id `c2d67fe65c7f`; the files are under git-ignored `artifacts/bedrock/c2d67fe65c7f/`.

**The gate is closed, as intended: G1 fails because no token ratio has been measured.** The
record `reports/frontier/nova_training_gate.json` says `passed: false`. At the assumed
ratio of 1.5:

| Part of Path B | Amount |
|---|---:|
| Training, 2 epochs, both models | $97.63 |
| The same with the 10% margin | $107.39 |
| Evaluation still to come | $9.32 |
| Storage, one month, two models | $3.90 |
| Already in the ledger | $0.00 |
| **Total** | **$120.61** of a $200.00 cap (G2 passes) |

Training by epoch count, both models: $48.81, $97.63, $146.44, $195.25, $244.06 for 1 to 5.
These are the worst-case figures; the measured ones replace them here after the baselines.

Tests: 1,704 passed (144 new); with the training stack blocked 1,703 passed, 1 skipped;
lint clean. Nothing was uploaded and nothing was asked of AWS.

Differences from sections 2 to 4:

- **Evaluation to come is priced with every reply at the 192-token limit**: $9.32, not the
  "about $6" of section 2.4, which assumed replies of the gold length. It errs high, like
  the rest of the estimate.
- **AWS's three limits are in `config/frontier.yaml`** (`export.min_records`,
  `max_records`, `max_sample_tokens`), not in code: they are service facts with a date.
- **`staging.py` imports botocore** (to tell "missing" from other S3 errors), so the rule
  is now "only `bedrock.py` and `staging.py` import boto3 or botocore"; the guard test
  names both.
- **An export is recorded as a run when its gate record is first written, and again when
  it is staged**, not on every repeat and never for `--check`. Otherwise each rerun would
  add lines to the committed ledger. The first real run is in the ledger.
- **`--check` writes no committed file** but does write the files under `artifacts/`,
  because the validator reads the disk.
- **The gate record carries its S3 locations over** from the committed record when the
  files' hashes are unchanged, so a later run without `--upload` does not lose them.
- **An object already in S3 under the key with other content is an error**, not replaced:
  the key carries the content's id, so a difference means something is wrong.
- **Dollar amounts are shown rounded half up**; `Decimal`'s default would show $0.825 as
  $0.82. Stored amounts are exact text.
- `export_files` returns the export and whether files were written. The cap, ratio, and
  margin are `Decimal` fields of `FrontierConfig`. `load_bucket()` reads the bucket name.

### Live results (the owner's runs, 2026-11-18)

With the v4.01 baseline report written, `python -m scripts.nova_export` opened the gate, and
`python -m scripts.nova_export --upload` staged the files (run
`20261119T015413Z-export-nova-export-c2d67fe65c7f-95c9`). X1 to X7, G1, and G2 passed both
times: 9 of 9.

| | Value |
|---|---|
| Export id | `c2d67fe65c7f` (unchanged since the build) |
| Training file | `s3://qlora-sagemaker-123456789012/data/bedrock/c2d67fe65c7f/finqa_fit.jsonl`, 4,884 records, 10,715,093 bytes |
| Validation file | `s3://qlora-sagemaker-123456789012/data/bedrock/c2d67fe65c7f/finqa_dev.jsonl`, 499 records, 1,106,662 bytes |
| Nova tokens per Qwen token | 0.9851, measured (the largest of the four baseline ratios); the build had assumed 1.5 |
| Training, both models, 2 epochs | $64.11; $70.53 with the 10% margin |
| By epoch count | 1: $32.06, 2: $64.11, 3: $96.17, 4: $128.23, 5: $160.28 |
| Already in the ledger | $0.49 (the baselines) |
| Evaluation to come | $7.55 |
| Storage, one month | $3.90 |
| Path B total | $82.46 of the $200.00 cap |

Each object was sent with its SHA-256 and read back by the script. The gate record under
`reports/frontier/` now says `passed: true` and names both S3 locations.

Still open, from section 7: whether Bedrock bills validation tokens, and whether the bill
matches tokens times epochs. Nova Lite's bill in v4.03 is the first evidence.
