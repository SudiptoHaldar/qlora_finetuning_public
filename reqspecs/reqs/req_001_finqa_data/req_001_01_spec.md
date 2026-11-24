# FinQA Data

Spec Group      :   req_001_finqa_data  
Feature Name    :   finqa_data  
Version         :   1  
Status          :   ✅ Complete 2026-11-15 (see `reqspecs/PRPs/prp_001_dataset_snapshot_v1.md`); CI green on pull request #2 and on `main` (2026-11-15)  
Last updated    :   2026-11-15  

## 1. v1.01 — Dataset snapshot and schema report

Every later stage reads FinQA: the split, the prompts, the executor's operation list, the
scorer. This version fixes **which bytes** the project means by "FinQA" and writes down **what
is actually in them**, so the later specs are written from facts and not from the two guides,
which describe the original GitHub release.

It adds three things:

- a **pinned snapshot**: the three dataset files, downloaded at one commit and checked against
  recorded checksums on every load;
- a **raw reader**: the 8,281 examples as typed records, with nothing dropped or repaired;
- a **schema report**: counts and edge cases, committed, and reproducible byte for byte.

It decides nothing about the split, the prompt, or what counts as a correct answer. It hands
those versions the facts they need (section 6).

**Depends on**: v0.02 (done).

## 2. What exists today (verified 2026-11-15)

The dataset was downloaded at the pinned commit into a scratch folder and profiled. Every number
below was measured on that copy, not taken from the dataset card.

### 2.1 The repository

| Item | State |
|---|---|
| Name | **`bevaya/FinQA`**. The name in the plan, `rootsautomation/FinQA`, now answers with a redirect to it. The dataset card's own usage example still shows the old name |
| Commit | `3d6a736bc67e06bc15fbf3618d88204a57c5b25e` (2026-05-14), the newest of three; the data files were added in the commit before it and have not changed |
| Access | Public, not gated, MIT. No Hugging Face token is needed |
| What it is | A third-party repackaging of the original release (`github.com/czyssrs/FinQA`) as Parquet. It was not compared with the original here; the split sizes match the published ones |

| File | Rows | Bytes | SHA-256 |
|---|---|---|---|
| `data/train-00000-of-00001.parquet` | 6,251 | 30,089,143 | `55dc8ef7405e8edb86743c14120f50280aed3fcb41505270160e0b77819ee339` |
| `data/validation-00000-of-00001.parquet` | 883 | 4,196,010 | `b21e85e33138739d6bd51c63144a5eaf30026d89e161092999f7c22daa76a1ea` |
| `data/test-00000-of-00001.parquet` | 1,147 | 5,415,145 | `4f5edc88f1e8bf34a005d2bd39bf1a0e7419eed72998ea2fdd31fa1a15f85bb0` |

The checksums of the downloaded files equal the ones the Hub lists for them. Total: 8,281 rows.

### 2.2 Columns

Sixteen columns, the same in all three files. None is ever null.

| Column | Type | Notes |
|---|---|---|
| `id` | string | `{TICKER}/{YEAR}/page_{N}.pdf-{idx}`. All 8,281 are unique and match the pattern |
| `filename` | string | The `id` without `-{idx}`: one page of one annual report |
| `pre_text`, `post_text` | list of string | Passages before and after the table. Never empty lists. Up to 450 and 489 passages |
| `table` | list of list of string | Cleaned table. 1–20 rows, 2–14 columns, never ragged |
| `table_ori` | list of list of string | Table with the original formatting. Differs from `table` in every example |
| `question` | string | Never empty |
| `answer` | string | Free-form, written by the annotator (2.5). Empty in 74 examples |
| `exe_ans` | string | The executed result: a number, or `yes` / `no`. Never empty |
| `explanation` | string | Empty in 6,988 examples (84%) |
| `ann_table_rows`, `ann_text_rows` | list of int | Annotated evidence rows. Unreliable (2.4) |
| `steps` | struct of four lists (`op`, `arg1`, `arg2`, `res`) | The program step by step, with different operation names (`minus2-1`, `compare_larger1-1`). Same number of steps as `program` in every example |
| `program` | string | Flat form: `subtract(920, 95), divide(#0, 5)` |
| `program_re` | string | Nested form: `divide(subtract(920, 95), 5)` |
| `gold_inds` | string | A JSON object: evidence key to evidence text |

The guides' field names (`pre_text`, `post_text`, `table`, `gold_inds`) are right, but the
original release nests the question fields under `qa`; here they are top-level columns.

All questions, table cells, and passages are plain ASCII. No table cell contains a `|`.

### 2.3 Reports, pages, and duplicates

| Level | Count | Examples per group |
|---|---|---|
| Company (ticker) | 137 | 2 to 472, median 36 |
| Report (ticker + year) | 799 | 1 to 95, median 7 |
| Page (`filename`) | 2,789 | 1 to 6 |

Years run from 1999 to 2019.

The **official splits are grouped by page only**: no page appears in two official splits, but
355 of the 799 reports and 117 of the 137 companies do. So the official split already lets
questions about the same annual report sit on both sides.

Duplicates: 73 pairs of examples (146 rows) share a page and a question word for word. In 64
pairs the program and the result are also identical; in 9 they differ. Each pair is on one
page, so any grouping key keeps a pair on one side of the split.

### 2.4 Evidence

- `gold_inds` parses as JSON in every example and is never empty. It holds 1 to 9 entries.
- Keys are `table_N` (9,901 in total) or `text_N` (4,206). A table entry's value is the row
  rewritten as a sentence, not the raw row.
- 5,171 examples have table evidence only, 1,939 text only, 1,171 both. (Corrected when the
  report was built: the first profile counted the malformed key below as text evidence.)
- Every index is inside its table or passage list, except one: `RE/2010/page_120.pdf-2` has
  the key `text_-1`.
- `ann_table_rows` and `ann_text_rows` disagree with the `gold_inds` keys in 1,302 examples. In
  631 of those both lists are empty while `gold_inds` has entries. **`gold_inds` is the
  evidence source to trust.**
- 3,289 tables have at least one empty cell (often the top-left header cell).

### 2.5 Programs and answers

Every one of the 8,281 programs has the form `op(a, b), op(a, b), ...`. Every step has exactly
two arguments. There are 12,732 steps in total.

| Operation | Steps | | Operation | Steps |
|---|---|---|---|---|
| `divide` | 5,901 | | `table_average` | 129 |
| `subtract` | 3,676 | | `table_max` | 66 |
| `add` | 1,952 | | `table_sum` | 50 |
| `multiply` | 759 | | `table_min` | 36 |
| `greater` | 154 | | `exp` | 9 |

These ten are the whole inventory, and they match the plan's list (§4.2).

| Steps in a program | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Examples | 4,894 | 2,709 | 429 | 118 | 125 | 6 |

The dataset card says programs have 1 to 5 steps; six have 6.

Arguments of the six arithmetic operations:

| Kind | Count | Example |
|---|---|---|
| Whole number | 12,115 | `920`, `-121` |
| Decimal | 6,242 | `4474.2` |
| Step reference | 4,465 | `#0` |
| Constant | 1,709 | `const_100` |
| **Percent literal** | 371 | `6.9%`, as in `divide(6.9%, const_100)` and `add(const_1, 2.0%)` |

Percent literals are not in the plan's list of argument kinds.

Seventeen constants occur: `const_1` to `const_10`, `const_100`, `const_1000`, `const_10000`,
`const_100000`, `const_1000000`, `const_1000000000`, and `const_m1` (minus one; 21 uses).

Table operations (281 steps in 272 examples): the first argument is a row name, and in all 281
it equals the first cell of a row of the example's table. Row names contain spaces
(`table_max(expected volatility, none)`) but never a comma or a bracket. The second argument is
`none` in 279 steps and a number in 2.

Answers:

- `exe_ans` is a number in 8,127 examples and `yes` or `no` in 154 (93 and 61). Those 154 are
  exactly the programs that contain `greater`, and `greater` is always their last step.
- 223 numeric `exe_ans` values are in scientific notation (`4.39e+07`). Values look rounded to
  about five significant digits (`0.44659` for `divide(18616, 41685)`).
- `answer` and `exe_ans` are often on different scales: `93.5%` against `0.935`; `380` against
  `3.8`.
- `answer` is free-form: 4,652 contain `%`, 3,262 are plain numbers, 98 contain `$`, 152 are
  yes or no, 74 are empty, and 44 are something else (ratios such as `2.43:1`, units such as
  `2.42 million`, whole sentences, a literal `\n` at the end).
- Two examples answer yes or no although their program has no `greater` and `exe_ans` is a
  number (`PPG/2018/page_85.pdf-1`, `ANET/2015/page_156.pdf-1`). Eight `greater` examples have
  an `answer` that does not match `exe_ans`, most of them empty.

### 2.6 The repository on this side

| Item | State |
|---|---|
| `qlora_finetuning/data/` | An empty sub-package |
| `artifacts/` | Git-ignored. No dataset in it |
| `config/` | No data config |
| `requirements.txt` | Has `pandas`; has **no Parquet reader**. `pyarrow` 26.0.1 is installed locally only because the training stack pulls it in, and CI does not install the training stack |
| `huggingface_hub` 2.4.0 | Pinned in `requirements-train.txt`. `hf_hub_download(repo_id, filename, repo_type=, revision=, local_dir=)` is available |
| pandas and nested columns | `pandas.read_parquet` returns list columns as NumPy arrays and the `steps` struct as a dict of arrays, not as Python lists |

## 3. Locked decisions

| # | Decision | Reason |
|---|---|---|
| D1 | The dataset is `bevaya/FinQA` at commit `3d6a736bc67e06bc15fbf3618d88204a57c5b25e`. The old name is recorded as a note only | Section 2.1. A redirect can be withdrawn; the real name cannot |
| D2 | The full commit hash is the pin. A branch name or a short hash is rejected by the config check | `main` can move |
| D3 | The three checksums of section 2.1, with file sizes and row counts, are written in `config/data.yaml`. The files are checked against them after download **and on every load** | Hashing 40 MB takes a fraction of a second; a silently changed file would invalidate every result |
| D4 | The snapshot lives at `artifacts/data/finqa/<commit>/`, keeping the Hub's file paths, with a `snapshot.json` beside the files. It is git-ignored | Plan §5. The commit in the path lets two revisions coexist if the pin ever changes |
| D5 | Files are downloaded with `huggingface_hub.hf_hub_download`, imported inside the function. The download function receives the downloader as an argument | Already pinned; handles redirects, retries, and resume. The argument lets tests pass a stand-in (coding rule) |
| D6 | Files are **read** with `pyarrow`, not with the `datasets` library. `pyarrow` is added to `requirements.txt`. (As built: `pyarrow` alone; its `to_pylist()` already returns plain lists, so `pandas` is not involved) | Reading must work in CI and in any later CPU-only step. `datasets` is training stack and stays out of module top levels |
| D7 | The raw reader returns the three files pooled, in the order train, validation, test and in file order, each record carrying its `official_split`. Nothing is dropped, deduplicated, or repaired | The plan pools all 8,281. Odd examples are counted and flagged by id, never removed (plan §7) |
| D8 | A record is a frozen dataclass with plain Python values: lists, not NumPy arrays; `gold_inds` parsed into a dict in its original key order; and `ticker`, `year`, `page`, `index` parsed from the `id` | Section 2.6. Later stages should not each repeat the conversion |
| D9 | The raw reader reads all rows, including rows that v1.02 will assign to test. It is named and documented as raw, and v1.02 puts the split-aware loaders and the test lock in front of it | There is no split yet, so there is nothing to lock. v1.02 owns the lock |
| D10 | The schema report is **committed**, as `reports/data/finqa_schema_report.md` and `.json` | v2.01 takes its operation list from it; v1.02 and v1.03 cite it. It is small and is evidence |
| D11 | The report contains counts and example **ids** only: no question, passage, table cell, or answer text. Program text appears only as operation names, constant names, and argument kinds | Any example may land in the test split in v1.02. A committed file must not carry test content |
| D12 | The report carries no timestamp, user name, or machine path. Running it twice gives identical bytes. The download date is in `snapshot.json` only | So "the report matches the snapshot" is a test, and a diff of the report means the data or the code changed |
| D13 | Expected facts live in `config/data.yaml` and are **checked**: row counts per file, column names and types, unique ids, the id pattern, and the sets of known operations, constants, and argument kinds. A mismatch is a failed check with exit code 1. Everything else in the report is a count, not a check | If the pin ever changes, the report says exactly what moved. Follows the check-script pattern |
| D14 | To count operations this version splits programs with one small pattern, private to the report module. A program the pattern cannot split is counted as unparsed (expected: zero). It is not the DSL parser | v2.01 owns the parser. v2.01 adds a test that its parser agrees with this inventory |
| D15 | Downloading and reporting are not tracked runs and write nothing to the cost ledger | They are free and are neither training nor evaluation. `snapshot.json` is their record |
| D16 | No Hugging Face token is read, asked for, or stored | The dataset is public |

## 4. Scope to build

### 4.1 `config/data.yaml`

`version: 1`; `verified_on`; under `finqa`: `repo_id`, `former_repo_id`, `revision`,
`snapshot_dir: artifacts/data/finqa`, `report_dir: reports/data`, and one entry per file with
`official_split`, `path`, `sha256`, `bytes`, `rows`. Under `finqa.expected`: the sixteen column
names with their types, the id pattern, the ten operations, the seventeen constants, the
argument kinds (whole number, decimal, step reference, constant, percent literal, row name,
`none`), and the largest program length seen (6).

### 4.2 Snapshot: `qlora_finetuning/data/snapshot.py`

| Function | Behavior |
|---|---|
| `snapshot_path(config)` | `artifacts/data/finqa/<commit>/` |
| `fetch_snapshot(config, download=None, force=False)` | Downloads each missing file at the pinned commit, checks size and SHA-256, and writes `snapshot.json` (repository, commit, files with checksums, download time in UTC, library versions). A file that is present and correct is not downloaded again. A wrong checksum deletes nothing, raises `SnapshotError` naming the file with the expected and actual values |
| `verify_snapshot(config)` | Checks that every file exists with the right size and checksum. Returns one result per file |
| `load_raw_examples(config)` | Verifies, then returns the 8,281 records per D7 and D8. A missing snapshot raises `SnapshotError` with the command that fetches it |

`RawExample` fields: `id`, `filename`, `official_split`, `ticker`, `year`, `page`, `index`,
`pre_text`, `post_text`, `table`, `table_ori`, `question`, `answer`, `exe_ans`, `explanation`,
`ann_table_rows`, `ann_text_rows`, `steps`, `program`, `program_re`, `gold_inds`.

### 4.3 Schema report: `qlora_finetuning/data/schema_report.py`

Pure functions over a list of records. `build_report(records, config)` returns a dict;
`render_markdown(report)` returns text. Sections:

| Section | Content |
|---|---|
| Source | Repository, commit, files, checksums, row counts |
| Columns | Name, type, null count, empty count |
| Identity | Unique ids, pattern mismatches; counts and examples-per-group (minimum, median, maximum) by company, report, and page; how many groups at each level span more than one official split |
| Duplicates | Same page and question; of those, same program and result |
| Tables | Row and column ranges, ragged tables, tables with empty cells, cells containing `\|` |
| Evidence | `gold_inds` parse failures, entries per example, key kinds, table / text / both, out-of-range and malformed keys, disagreement with the `ann_*` lists |
| Programs | Operation counts, steps per program, argument kinds, constants, table-operation arguments (row name found in the table or not, second argument), unparsed programs |
| Answers | `exe_ans`: numeric, `yes`, `no`, scientific notation. `answer`: empty, contains `%`, contains `$`, plain number, yes/no, other. Yes/no answers without `greater`; `greater` programs whose `answer` disagrees with `exe_ans` |
| Flagged examples | For each oddity class, the list of ids (D11) |
| Checks | The D13 checks, each pass or fail |

Every count is given in total and per official split.

### 4.4 Scripts

Both follow the operator-script conventions: module form, by-path hint, ASCII output, exit
codes 0, 1, 2.

| Command | Behavior |
|---|---|
| `venv\Scripts\python -m scripts.finqa_fetch` | Fetches and verifies. Prints one line per file. `--verify-only` makes no network call. `--force` downloads again |
| `venv\Scripts\python -m scripts.finqa_report` | Reads the snapshot, runs the checks, writes the two report files, prints the check lines and a summary. `--check` writes nothing and exits 1 if the committed report differs from a fresh one |

### 4.5 Tests

Tests build a small synthetic Parquet snapshot in a temporary folder: invented rows, never
real ones. No test needs the network or the training stack.

- Snapshot: paths; a stand-in downloader fetches missing files only; a wrong checksum, a wrong
  size, and a missing file each raise and name the file; `snapshot.json` contents; a branch
  name or short hash as the revision is rejected; records come back in order with Python lists,
  a parsed `gold_inds`, and parsed id parts.
- Report: synthetic rows covering each oddity (duplicate pair, empty answer, `text_-1`, percent
  literal, table operation with a missing row name, six steps, unknown operation, unparsable
  program, yes/no answer without `greater`); each check passes and fails as expected; the same
  input gives the same bytes; the report contains none of the question, passage, cell, or
  answer text of the synthetic rows.
- Scripts: each exit code; module form and by-path hint.
- One test marked `network`: the pinned commit exists on the Hub and lists the three files with
  the configured sizes and checksums.
- A test that the committed report, when present with a snapshot, matches a fresh one; it
  skips when there is no snapshot (CI).

### 4.6 Documentation

- Runbook: a "FinQA dataset snapshot" section: fetch, verify, rebuild the report, what to do if
  a checksum fails, and that the snapshot is the project's only copy of the data.
- `reqspecs/references/tech/finqa_dataset.md`: section 2 of this spec as a reference note.
- `common/gotchas.md`: the renamed repository; pandas returning arrays for list columns;
  `ann_*` lists unreliable; percent literals in programs; `answer` free-form and sometimes
  empty.
- `high_level_plan.md` and `execution_plan.md`: replace `rootsautomation/FinQA` with
  `bevaya/FinQA`; add percent literals to the argument kinds in plan §4.2.
- `requirements.txt`: `pyarrow`, at the version already installed.

## 5. Out of scope

- The split, the grouping key, the manifest, split-aware loaders, the test lock (v1.02).
- Table formatting, evidence selection, the prompt, the target text (v1.03).
- Tokenizing anything (v1.04).
- Parsing or executing programs, and checking that a gold program reproduces `exe_ans` (v2.01,
  v2.02).
- Uploading data to S3 (v3.03, v4.02).

## 6. What this hands to later versions

Findings that a later spec must decide on. This version only reports them.

| Finding | For |
|---|---|
| Group sizes: 799 reports, largest 95 examples (1.1% of the data); 137 companies, largest 472 (5.7%). The official split groups by page only | v1.02, Q4 |
| 73 duplicate pairs, each within one page | v1.02 |
| `gold_inds` is the evidence source; the `ann_*` lists are not. One key is `text_-1`. Table evidence is stored as sentences | v1.03 |
| `answer` cannot be the target: empty 74 times, free-form, on a different scale from `exe_ans`. `exe_ans` is always present | v1.03, v2.02 |
| 3,289 tables have empty cells; no cell contains a `\|` | v1.03 |
| Ten operations, always two arguments, up to six steps; percent literals as arguments; `none` as a second argument; row names with spaces; `const_m1` | v2.01 |
| `exe_ans` is rounded to about five significant digits, 223 times in scientific notation; yes/no only with `greater` as the last step; two yes/no answers without `greater` | v2.02 |

## 7. Known limits and follow-ups

- **One upstream copy.** The repository was renamed once already and is run by a third party.
  If it disappears, the local snapshot is the only pinned copy. A copy in the project's S3
  bucket would remove that risk; it is an AWS write, so it is the owner's call and is not part
  of this version.
- **Not compared with the original release.** The repackaging is trusted to match
  `czyssrs/FinQA`. Row counts match; contents were not diffed.
- **The profile in section 2 was made with a throwaway script.** The built report must
  reproduce those numbers; any difference is investigated before the spec's numbers or the
  report is accepted.
- **Flagged ids in a committed file.** The report lists ids of odd examples, some of which
  will be test examples. An id reveals the company, year, and page, not the question or the
  answer.

## 8. Acceptance

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with and
  without the training stack installed.
- `venv\Scripts\python -m scripts.finqa_fetch` leaves three verified files and a
  `snapshot.json` under `artifacts/data/finqa/<commit>/`; a second run downloads nothing.
- `venv\Scripts\python -m scripts.finqa_report` passes every check and writes the two report
  files; `--check` straight afterwards exits 0.
- The report shows 8,281 examples (6,251 / 883 / 1,147), sixteen columns, ten operations,
  seventeen constants, and agrees with section 2 of this spec.
- The committed report contains no question, passage, table, or answer text.
- `load_raw_examples()` returns 8,281 records and refuses a snapshot with an altered file.

## 9. Next step

Complete. Next: v1.02, the frozen split manifest (`req_001_02_spec.md`).
