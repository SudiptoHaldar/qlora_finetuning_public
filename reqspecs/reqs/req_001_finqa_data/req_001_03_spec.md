# FinQA Data

Spec Group      :   req_001_finqa_data  
Feature Name    :   finqa_data  
Version         :   3  
Status          :   ✅ Complete 2026-11-15 (see `reqspecs/PRPs/prp_001_canonical_serialization_v3.md`); CI green on pull request #3 and on `main` (2026-11-15). The wording stays open to change until the first baseline  
Last updated    :   2026-11-15  

## 1. v1.03 — Canonical serialization

Six arms are compared: two open-weight and four hosted. The comparison is fair only if every
one of them reads exactly the same words and is asked for exactly the same output. This
version defines those words once.

For each example it produces one **canonical record**: a system text, a user text, and a
target text. Path A later wraps the record in the Qwen chat template; Path B wraps it in
Bedrock's conversation format. Neither may change a character of it.

It adds:

- a **prompt definition** in `config/prompt.yaml`: the system text and the layout of the user
  and target texts;
- a **serializer**: example in, canonical record out, the same bytes every time;
- a **serialization report**: lengths in characters, what was left out and why, and a digest
  of all records, in counts only.

**Depends on**: v1.02 (built).

## 2. What exists today (verified 2026-11-15)

Measured on fit and dev together (5,383 examples), read through `load_split`. The test split
was not read. Facts about the whole dataset come from the committed schema report.

### 2.1 What earlier versions provide

| Item | State |
|---|---|
| `load_split("fit" \| "dev" \| "quick_dev")` | Returns `RawExample` records; `test` is locked |
| Q7 | Resolved 2026-11-15: every arm sees the annotated supporting text plus the whole table |
| Plan §4.2 | Output is two fields, `PROGRAM:` and `ANSWER:`; programs use FinQA's own syntax |
| `scripts/env_check.py` | Holds one hand-written example in roughly this layout for its smoke step |
| `config/prompt.yaml`, a serializer | Do not exist |

### 2.2 Text evidence

- Every `text_N` entry of `gold_inds` is identical to passage `N` of `pre_text` followed by
  `post_text`: 2,694 entries checked, none differs.
- **3,387 examples (63%) have no text evidence at all**; their evidence is in the table only.
  1,442 have one passage, 462 two, 92 three or more.
- In 59 examples the keys are not in document order. One example lists the same passage twice.
- The example with the malformed key `text_-1` (`RE/2010/page_120.pdf-2`) is in fit.
- Text evidence per example: up to 1,513 characters; 618 at the 99th percentile.
- No evidence passage contains a line break.

### 2.3 Tables

- No cell contains a line break, a `|`, or a backslash, and none has leading or trailing
  spaces. (The schema report shows the same for `|` over the whole dataset.)
- 2,185 of 132,424 cells are empty. In 1,794 tables the first header cell is empty. No header
  row is entirely empty.
- One table has a single row.
- 241 tables have two rows with the same name. This matters to the table operations, which
  name a row (v2.01).
- A table as Markdown is about 330 characters at the median, about 1,070 at the 99th
  percentile, and about 3,700 at most.

### 2.4 Scale cues

A scale cue is a phrase such as "in millions" or "( in thousands )".

| Where the cue is | Examples | Of the 190 whose program uses a scale constant |
|---|---|---|
| In the table | 1,831 | 83 |
| In the text evidence | 67 | 1 |
| **Only in a passage that is not evidence** | **1,482** | **31** |
| Nowhere on the page | 2,003 | 75 |

When the cue is in a passage before the table, it is the passage directly before the table in
1,976 of 2,391 cases: the table's caption.

So a prompt of evidence text plus table loses the scale cue in 1,482 examples (28%). It
changes the right program in at most 31 of them (0.6% of fit and dev), because a program
works on the table's numbers in the table's own units unless the question asks for a
conversion.

### 2.5 Programs, questions, and results

- Every program is already in one uniform form: steps joined by a comma and a space, no
  stray spaces. Up to 122 characters; 93 at the 99th percentile.
- Questions: up to 367 characters, no line breaks.
- `exe_ans`, the executed result:

| Form | Examples | Example of the form |
|---|---|---|
| Decimal | 4,058 | `0.44659` |
| Whole number | 1,074 | `856` |
| Scientific notation | 138 | `4.39e+07` |
| `yes` / `no` | 113 | `yes` |

  Of the decimals, 2,566 have exactly five decimal places, which is where the dataset rounded
  them. 134 of the scientific values are a million or more; 4 are tiny. No value has a
  trailing `.0` or stray spaces. 622 are negative.

## 3. Locked decisions

| # | Decision | Reason |
|---|---|---|
| D1 | A canonical record has three texts: **system**, **user**, **target**, plus the example id and the id of the prompt definition that produced it | Plan §4.2. Both paths have a system role and a user role |
| D2 | The system text is the same for every example and is a **complete description of the output language**: operations, argument kinds, constants, table operations, and the two-line reply | Two arms are untuned. They can only follow the format if the prompt states all of it; a prompt that relied on fine-tuning to teach the format would handicap them |
| D3 | The user text has three parts in a fixed order: `Context:`, `Table:`, `Question:` | The table and the question are closest to where the answer starts |
| D4 | **Context is the text evidence only**: the passages named by `text_N` keys of `gold_inds`, word for word, in document order, each once, as a list. Table evidence is not marked in any way | Q7. Showing which table rows matter would make the task easier than the plan intends |
| D5 | **No passage is added for scale cues.** "Unit cues preserved" means: nothing inside a table cell or an evidence passage is removed or rewritten (`$`, `%`, brackets, "in millions" in a header) | Section 2.4: the cue is lost in 28% of prompts but can change the program in at most 0.6%. Adding the caption passage would put non-evidence text into the prompt and blur the setting Q7 just fixed. **Alternative for the owner in section 8** |
| D6 | When an example has no text evidence, the context part reads `Context:` followed by `(none)` | 63% of examples. A fixed shape is easier to learn and to parse than a part that comes and goes |
| D7 | A `gold_inds` key that is not `table_N` or `text_N`, or that points outside the passages, is skipped and counted. The example is kept | One example in fit (`text_-1`). Nothing is dropped (v1.02 D2) |
| D8 | The table is the `table` column as GitHub-Flavored Markdown: the first row is the header, then a separator row, then the other rows. Cells are word for word; an empty cell stays empty | Plan §4.2. Row names must reach the model exactly, because table operations name them |
| D9 | Escaping exists even though the data needs none today: a vertical bar in a cell is written with a backslash before it, and a line break becomes a space | So a future change of the pin cannot silently break a table |
| D10 | The target is exactly two lines: `PROGRAM: <program>` and `ANSWER: <result>` | Plan §4.2 |
| D11 | The program is the dataset's `program` column, unchanged: percent literals, `none`, and row names included | Section 2.5: it is already uniform. This is what v2.01 must parse |
| D12 | **The answer is `exe_ans`**, the executed result, not the annotator's `answer` | `answer` is free-form, empty in 74 examples, and often on another scale. The answer line should be what the program evaluates to |
| D13 | `exe_ans` is copied as it is, except that scientific notation is rewritten as a plain number with the same digits (`4.39e+07` becomes `43900000`). `yes` and `no` are copied | 138 examples. A model should not have to learn to print exponents; converting through decimal arithmetic changes no digit |
| D14 | The prompt definition lives in `config/prompt.yaml` with a version. Its **prompt id** is a SHA-256 of the system text and the layout. Every record carries it, and every run records it | Working agreement 7: one contract for every arm, and a change is visible |
| D15 | The wording may still change until the first model is run on it (v3.01 or v4.01). From then on a change means a new version of the prompt definition and a rerun of every arm | Baselines must be comparable with tuned runs |
| D16 | The system text states what a percent literal and each operation mean. Those statements are confirmed by the gold self-check (v2.02) before any model is run; if the self-check contradicts one, the text is corrected then | v1.03 comes before the executor exists. No model sees the prompt before v2.02 is done |
| D17 | Serialization is a pure function of the example and the prompt definition: no clock, no randomness, no dependence on the split or on other examples | Deterministic output; the same function serves training, dev, and later test |
| D18 | Records are served by split through `serialize_split(name)`, which calls `load_split`. The test lock applies unchanged | v1.02 D11–D13 |
| D19 | Nothing is truncated here. An over-length example is v1.04's subject | Plan: never silently truncate the question or the table |
| D20 | Golden-file tests use invented rows only. For the real data, a **digest** of all fit records and of all dev records is committed; a test compares it when the snapshot is present | A committed golden file of real prompts would be dataset text (v1.01 D11). A digest proves nothing changed without containing any text |
| D21 | The report is committed as `reports/data/finqa_serialization_report.md` and `.json`: counts, character lengths, digests, the prompt id; no example text; no timestamp | Same rules as the other two reports |
| D22 | Full records can be written for inspection to `artifacts/serialized/`, which is git-ignored, for fit and dev only | The owner needs to read real prompts before approving the wording; they must not be committed |

## 4. Scope to build

### 4.1 `config/prompt.yaml`

`version: 1`; `system`: the text below; `user` layout: the three labels, the list marker, the
text for "no text evidence"; `target` layout: the two labels.

Draft system text (decision 3 in section 8):

```
You are a financial calculation assistant. You are given text from a company report, a table
from the same report, and a question. Write the calculation that answers the question as a
program, then give the result of the program.

A program is one or more steps separated by ", ". Each step is: operation(first, second)

Operations:
- add, subtract, multiply, divide
- exp: the first argument raised to the power of the second
- greater: yes if the first argument is larger than the second, otherwise no
- table_sum, table_average, table_max, table_min: the first argument is the exact name of a
  table row (the text of its first cell) and the second argument is none; the operation is
  applied to the numbers in that row

An argument is one of:
- a number from the text or the table, written without thousands separators; a percentage
  may be written as 12.5% and counts as 0.125
- #N, the result of step N, where the first step is #0
- a constant, for a number that is not in the text or the table: const_1, const_2, const_3,
  const_4, const_5, const_6, const_7, const_8, const_9, const_10, const_100, const_1000,
  const_10000, const_100000, const_1000000, const_1000000000, const_m1 (minus one)

Reply with exactly two lines and nothing else:
PROGRAM: <the program>
ANSWER: <the result of the last step: a plain number, or yes or no>
```

User text, with text evidence:

```
Context:
- <passage>
- <passage>

Table:
| <header cell> | <header cell> |
|---|---|
| <cell> | <cell> |

Question: <question>
```

Without text evidence the first part is `Context:` and `(none)` on the next line.

Target text:

```
PROGRAM: subtract(920, 95), divide(#0, 95)
ANSWER: 8.68421
```

### 4.2 Serializer: `qlora_finetuning/data/serialization.py`

| Function | Behavior |
|---|---|
| `load_prompt(config_dir=None)` | Reads and validates `config/prompt.yaml`; returns the definition with its prompt id |
| `text_evidence(example)` | D4 and D7: the passages, in document order, each once; also returns the skipped keys |
| `format_table(table)` | D8 and D9. A table with no row raises |
| `format_answer(exe_ans)` | D13. A value that is neither a number nor `yes` / `no` raises, naming the example |
| `build_record(example, prompt)` | Returns a `CanonicalRecord`: `example_id`, `prompt_id`, `system`, `user`, `target` |
| `serialize_split(name, prompt=None)` | `load_split(name)`, then `build_record` for each, in dataset order |
| `records_digest(records)` | SHA-256 over the records in order, for D20 |

### 4.3 Report: `qlora_finetuning/data/serialization_report.py`

For fit, dev, and quick-dev: the number of records; characters in the system, user, and target
texts (minimum, median, 90th, 99th percentile, maximum); examples without text evidence;
passages per example; skipped evidence keys with their example ids; answers rewritten from
scientific notation; yes/no targets; the largest table; the digest. Then the prompt id and the
checks:

| Id | Check | Passes when |
|---|---|---|
| C1 | Target shape | Every target is exactly two lines starting `PROGRAM: ` and `ANSWER: ` |
| C2 | Program unchanged | Every program in a target equals the example's `program` |
| C3 | Answer form | Every answer is a plain number or `yes` / `no`; none contains an exponent |
| C4 | Table shape | Every table has a header, a separator, and one line per remaining row, each with the same number of cells |
| C5 | Evidence complete | Every well-formed text evidence passage of an example is in its user text |
| C6 | Question present | Every user text ends with the example's question |
| C7 | One system text | Every record has the configured system text and prompt id |
| C8 | Deterministic | Serializing twice gives identical records |
| C9 | Digests | The fit and dev digests equal the committed ones |
| C10 | No test record | Every record's id is in fit or dev |

### 4.4 `scripts/finqa_serialize.py`

| Command | Behavior |
|---|---|
| `venv\Scripts\python -m scripts.finqa_serialize` | Serializes fit and dev, runs C1 to C10, writes the report (and, the first time, the digests) |
| `... --check` | Writes nothing; fails if the report or a digest differs |
| `... --dump` | Also writes `artifacts/serialized/fit.jsonl` and `dev.jsonl` for reading (D22) |
| `... --show <example id>` | Prints one fit or dev record to the terminal; a test id is refused |

Exit codes 0, 1, 2 as in the other data scripts.

### 4.5 Tests

On invented rows, in temporary folders.

- Evidence: document order from unordered keys; a repeated passage once; `text_-1` and an
  out-of-range key skipped and reported; table keys ignored; no text evidence gives `(none)`.
- Table: header and separator; empty cells; one-row table; a `|` and a line break in a cell;
  a wide table; an empty table raises.
- Answer: whole number, decimal, negative, zero, `yes`, `no`; scientific notation large and
  small, with every digit kept; a value that is none of these raises.
- Record: golden files for four invented examples (text and table; table only; a table
  operation with a row name; a yes/no result); the program is copied unchanged including a
  percent literal and `none`; the prompt id changes when the system text or a label changes
  and not when a comment in the file changes.
- Prompt definition: missing key, empty system text, wrong version.
- Split: `serialize_split("test")` raises `TestLockedError`; fit and dev order.
- Report: each check passes and fails; no invented words from the examples in the output;
  deterministic.
- Script: exit codes; `--check`; `--dump` writes under `artifacts/` only; `--show` refuses a
  test id.
- On the real data when the snapshot is present: the digests match; 5,383 records.
- The smoke example in `scripts/env_check.py` is replaced by a record built with this
  serializer from an invented example, so there is one layout in the repository.

### 4.6 Documentation

- Runbook section 4: serialize, check, read real prompts with `--dump` or `--show`.
- `reqspecs/references/project_patterns/`: a note on the canonical record for the two path
  renderers (v3.01, v4.01, v4.02): wrap it, never edit it; record the prompt id.
- `high_level_plan.md` §4.2: the answer is the executed result; context is text evidence only.
- `coding.md`: prompts come from `build_record`; no prompt text in code.

## 5. Out of scope

- Token counts and the over-length policy (v1.04).
- The Qwen chat template and thinking mode (v3.01); Bedrock conversation JSONL (v4.02).
- Parsing a model's reply, executing a program, scoring (v2.01, v2.02).
- Few-shot examples in the prompt (a req_007 candidate).
- Any change to which examples are in which split.

## 6. What this hands to later versions

| For | What |
|---|---|
| v1.04 | `serialize_split("fit")` and `("dev")` to tokenize; character lengths as a first estimate |
| v2.01 | The exact target grammar: two lines, the program form of D11, the answer forms of D13 |
| v2.02 | The statements about operations and percent literals in the system text, to confirm (D16); 241 tables with a repeated row name |
| v3.01, v4.01, v4.02 | `CanonicalRecord`, to wrap; the prompt id, to record |
| v5.01 | The same function for the test split after the unlock |

## 7. Known limits and follow-ups

- **The scale cue is missing from 28% of prompts** (D5). In at most 31 fit and dev examples
  the right program applies a scale conversion whose cue the prompt does not show; a further
  75 have no cue anywhere on the page. All arms are affected equally.
- **Table-only prompts are the majority.** 63% of examples give the model no text at all.
- **The answer line asks the model for arithmetic.** The program is executed by code; the
  answer is the model's own figure and is scored separately as a secondary metric (plan
  §4.4). Results rounded by the dataset to five decimal places are copied as they are.
- **The system text is long.** It is sent with every example, so it adds to training tokens
  on the hosted path. Measured when built: 1,331 characters, 372 tokens with the Qwen3
  tokenizer. v1.04 measures whole prompts.
- **The wording is a draft.** It has not been tried on any model. D15 allows changing it
  until the first baseline.
- **Statements about meaning are unconfirmed** until v2.02 (D16).

## 8. Decisions for the owner

All four were accepted on 2026-11-15, when the owner proceeded to the PRP. Item 3 stays open in the
sense the spec gives it: the wording can still be adjusted, after reading real prompts, until
the first baseline is run.

| # | Question | Recommendation |
|---|---|---|
| 1 | **Scale cues**: leave the prompt as evidence text plus table, or also show the passage directly before the table when it contains a scale phrase? | Leave it (D5). The alternative restores the cue for about 1,976 examples, at the cost of putting text that is not evidence into the prompt and of a rule that guesses what a caption is |
| 2 | **Answer line**: the executed result `exe_ans`, with scientific notation written out? | Yes (D12, D13) |
| 3 | **System text** of section 4.1 | Accept as a draft; read real prompts with `--show` after the build and adjust before the first baseline |
| 4 | `(none)` for an empty context, or leave the context part out? | `(none)` (D6) |

## 9. Acceptance

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with and
  without the training stack installed.
- `venv\Scripts\python -m scripts.finqa_serialize` produces 4,884 fit and 499 dev records;
  C1 to C10 pass; a second run changes nothing; `--check` exits 0.
- The report agrees with section 2: 3,387 records without text evidence, one skipped key, 138
  answers rewritten from scientific notation.
- `serialize_split("test")` raises `TestLockedError`.
- No committed file contains real example text; the golden files are invented.
- The prompt id is recorded in the req_001 roadmap.

## 10. Next step

Built. Prompt id: `d63742a62b76d526ecf3d92df4cbad832846d4da7aa1ef13d15401ac28391641`. CI green on pull request #3 and on `main` (2026-11-15). Then the v1.04 spec (token budget).
