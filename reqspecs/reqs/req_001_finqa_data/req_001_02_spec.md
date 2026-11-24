# FinQA Data

Spec Group      :   req_001_finqa_data  
Feature Name    :   finqa_data  
Version         :   2  
Status          :   ✅ Complete 2026-11-15 (see `reqspecs/PRPs/prp_001_frozen_split_v2.md`); CI green on pull request #3 and on `main` (2026-11-15). Q4 resolved as company  
Last updated    :   2026-11-15  

## 1. v1.02 — Frozen split manifest

The brief asks for 65% of the data for training and the rest for testing, decided once and
used by both paths. This version makes that decision and freezes it: every example gets one
label, the labels are committed, and from then on nothing re-splits.

It adds four things:

- a **split builder**: a deterministic rule that assigns every example to `fit`, `dev`, or
  `test`;
- a **committed manifest**: the assignments, the settings that produced them, and a hash;
- **split-aware loaders** with a **test lock**: code asks for a split by name, and asking for
  `test` fails until `req_005` v5.01 unlocks it;
- a **split report**: how the splits compare, in counts only.

Reaching it is milestone **M2**. After M2 the manifest changes only by a recorded decision,
and any change invalidates every earlier result (working agreement 5).

**Depends on**: v1.01 (done).

## 2. What exists today (verified 2026-11-15)

Measured on the pinned snapshot with a throwaway script. The seed, 20261115, was fixed before
any split was computed and was not changed afterwards.

### 2.1 What v1.01 provides

| Item | State |
|---|---|
| `load_raw_examples()` | Returns all 8,281 records, with `ticker`, `year`, `page`, `filename` on each. Raw: it reads every row, with no lock |
| `config/data.yaml` | Pins the dataset commit and file checksums |
| `reports/data/finqa_schema_report.md` | Group sizes, duplicates, operation counts |
| `start_run(..., split_manifest_hash=None)` | The run registry already accepts the manifest hash (req_000 v0.04, D8) |
| Split manifest, split config, loaders by split | Do not exist |

### 2.2 The three possible grouping keys

| Key | Groups | Largest group | Share of the data in the largest group |
|---|---|---|---|
| Page | 2,789 | 6 | 0.07% |
| Report (company + year) | 799 | 95 | 1.1% |
| Company | 137 | 472 | 5.7% |

A company has 1 to 16 reports in the data (median 5).

### 2.3 How much each key leaks

The same 65 / 35 split rule (section 4.2) was run with each key. "Shared number" means: a
test example's gold table row holds a number of four or more digits, and the same row name
with the same number also appears in a table of the training side, for the same company.
Rows without a name (header rows) are not counted. The figures below were re-measured when
the version was built; the first draft used a simpler cut rule and counted header rows, and
differed slightly (28.6% for report grouping).

| Measure | By page | By report | By company |
|---|---|---|---|
| Test examples whose company is also in train | 2,884 of 2,898 | 2,745 of 2,898 | 0 |
| Test examples whose annual report is also in train | 2,427 | 0 | 0 |
| Test examples with a shared number, of those that have one to share | 550 of 1,839 (29.9%) | 500 of 1,761 (28.4%) | 0 of 1,861 |
| Same, matching against any company's training tables (chance level) | 581 | 517 | 58 (3.1%) |
| Test examples whose exact question wording is in train | 42 | 30 | 1 |

Why grouping by report still leaks: an annual report prints the prior years beside the
current one. Of 457 pairs of reports from one company one year apart, 331 (72%) share at least
one row-name-and-number pair; two years apart, 228 of 378 (60%); three years apart, 106 of 332
(32%). Overall 571 of 792 reports share a number with another year of the same company.

So with report grouping, more than a quarter of test questions rest on a number the model saw
in training, under the same row name, for the same company. Grouping by company removes that.

### 2.4 What the proposed split looks like

Rule of section 4.2, key = company, seed 20261115:

| Split | Examples | Share | Companies |
|---|---|---|---|
| fit | 4,884 | 58.98% | 79 |
| dev | 499 | 6.03% | 15 |
| train (fit + dev) | 5,383 | 65.00% | 94 |
| test | 2,898 | 35.00% | 43 |
| quick-dev (inside dev) | 200 | | 12 |

The largest company on the train side has 389 examples, on the test side 472, in dev 149.
Test years run from 2001 to 2019. Of the test examples, 2,127 come from the official train
file, 311 from validation, and 460 from the official test file.

| | fit | dev | quick-dev | test |
|---|---|---|---|---|
| Programs of 1 / 2 / 3 / 4+ steps (%) | 59.4 / 32.4 / 5.4 / 2.8 | 57.3 / 32.1 / 7.0 / 3.6 | 58.0 / 31.0 / 8.5 / 2.5 | 59.0 / 33.3 / 4.5 / 3.2 |
| `divide` share of steps (%) | 45.4 | 43.0 | 45.9 | 48.6 |
| `subtract` (%) | 28.1 | 29.4 | 28.7 | 30.1 |
| `add` (%) | 16.6 | 16.8 | 15.0 | 12.9 |
| `multiply` (%) | 6.0 | 7.9 | 8.3 | 5.5 |
| Yes/no examples | 100 | 13 | 5 | 41 |
| Examples with a table operation | 181 | 10 | 2 | 81 |

The largest gap between fit and test is `add`, 3.7 points. `exp` (9 steps in the whole
dataset) has 1 step in fit, none in dev, and 8 in test.

For comparison, the same rule with key = report gives 4,883 / 500 / 2,898 with 456, 53, and
290 reports, and a largest fit-to-test gap of 1.1 points.

## 3. Locked decisions

| # | Decision | Reason |
|---|---|---|
| D1 | **The split is grouped by company** (the `ticker` part of the id). All examples of one company are on one side. This answers Q4 and **changes the plan's recommended default**, which was report (company + year) | Section 2.3: report grouping leaves 28.4% of checkable test questions resting on a number seen in training; company grouping leaves none. The price is coarser groups: the operation mix differs a little more between splits, and dev holds 15 companies |
| D2 | All 8,281 examples are assigned. Nothing is excluded: not the 74 empty answers, not the duplicate pairs, not the malformed-evidence example | Plan §4.1 and §7: odd examples are flagged, never dropped. Whether an example is scoreable is decided by the gold self-check (v2.02), not here |
| D3 | The seed is 20261115. It was fixed before any split was computed and is never changed to get a nicer split | Trying seeds until the statistics look good is tuning the test set |
| D4 | Order is defined by SHA-256, not by a random-number generator: a group's position is the hex digest of `"<seed>:<salt>:<group key>"`. Salts are `train`, `dev`, `quick_dev` | Identical on every Python version and operating system. `random.shuffle` is not guaranteed to be |
| D5 | Selection rule ("fill"): walk the groups in digest order and take a group whenever it still fits under the target; never exceed the target. Train target = 65% of the examples, rounded to a whole number (5,383). Test is everything else | With whole groups, a plain cut misses 65% by up to half a group. This rule lands on 5,383 exactly with either key (section 2.4) |
| D6 | Dev is carved from train with the same key and the same rule, target 500 examples (result: 499). Fit is train minus dev | Plan §4.1. Dev then measures what test measures: companies not seen in fitting |
| D7 | Quick-dev is 200 examples of dev: the first 200 in digest order of the example id. It is a label on dev examples, not a fourth split | Plan §4.1. It is never trained on, so it need not be grouped inside dev |
| D8 | The manifest is one committed JSON file, `splits/finqa_split_manifest.json`: the dataset pin, the settings, the company-to-split table, the id-to-split table, the quick-dev ids, the counts, and a content hash | Small (about 8,300 short lines), readable, and the evidence for every later number. Ids and company names are not example text (v1.01 D11) |
| D9 | The content hash is SHA-256 over one line per example, in dataset order: `<id>\t<split>\t<1 or 0 for quick-dev>\n`. It is what `start_run(split_manifest_hash=...)` receives and what the execution plan records at M2 | One value identifies the split everywhere |
| D10 | The manifest is written once. The build command refuses to overwrite an existing manifest that differs; there is no force option. Changing the split means deleting the file by hand and recording the decision in this group's roadmap | Working agreement 5 |
| D11 | Code gets examples through `load_split(name)` with name `fit`, `dev`, or `quick_dev`. There is no loader for "train": fit and dev are asked for separately | Keeps "what was trained on" unambiguous |
| D12 | **Test lock.** `load_split("test")` raises `TestLockedError` unless an unlock record exists at `splits/test_unlock.json` and names the current manifest hash. This version never creates that file; `req_005` v5.01 does, once. A test asserts the file is absent | Plan §4.1. The lock guards against accident. It cannot stop someone who deliberately bypasses it, and does not claim to |
| D13 | The raw reader stays, but a guard test lists the only files allowed to import it: the `qlora_finetuning/data/` package, the two data scripts, and their tests. Any other importer fails the test | Otherwise the lock has a side door. v1.01 D9 promised this |
| D14 | The builder and the split report read test rows, for counts only. This is the one sanctioned use before the unlock | The plan asks for split statistics per split. Counts of operations and program lengths reveal no question or answer, and the same counts per official split are already committed (v1.01) |
| D15 | Loading a split checks the manifest against the data first: same dataset commit and file checksums, every snapshot id assigned exactly once, no unknown id, content hash correct. A mismatch raises; nothing is loaded | A manifest for different bytes is worse than no manifest |
| D16 | Settings live in `config/split.yaml`: key, seed, fractions, sizes, tolerances, paths | Working agreement 13 |
| D17 | The split report is committed as `reports/data/finqa_split_report.md` and `.json`, counts and ids only, no timestamp | Same rules as the schema report (v1.01 D10–D12) |
| D18 | Per-example flags from later versions (for example "gold program does not reproduce the answer", v2.02) go in their own file, not in the manifest | So adding a flag never changes the manifest or its hash |
| D19 | The split is not stratified by operation or program length | Grouping by company and stratifying at once would need a search over assignments, which is a form of seed shopping. The report shows the mix per split instead |

## 4. Scope to build

### 4.1 `config/split.yaml`

`version: 1`; `group_by: company` (allowed: `company`, `report`, `page`); `seed: 20261115`;
`train_fraction: 0.65`; `train_fraction_tolerance: 0.005`; `dev_size: 500`;
`dev_size_tolerance: 25`; `quick_dev_size: 200`; `manifest_file:
splits/finqa_split_manifest.json`; `unlock_file: splits/test_unlock.json`; `report_dir:
reports/data`.

### 4.2 Builder: `qlora_finetuning/data/split.py`

| Function | Behavior |
|---|---|
| `group_key(example, group_by)` | `ticker`, `report`, or `filename`. An example without one (id that does not parse) raises |
| `order_key(seed, salt, key)` | The hex SHA-256 of `"<seed>:<salt>:<key>"` (D4) |
| `fill(sizes, seed, salt, target)` | D5. Returns the chosen groups |
| `build_split(examples, cfg)` | Applies D5 to D7 and returns the manifest as a dict. Pure: no file access, no clock |
| `content_hash(assignments)` | D9 |
| `write_manifest(manifest, cfg)` | D10: writes if absent; if present and identical, does nothing; if present and different, raises `SplitError` |
| `read_manifest(cfg)` | Reads and validates the file's shape and its hash |

The manifest:

| Key | Content |
|---|---|
| `schema` | 1 |
| `dataset` | `repo_id`, `revision`, and each file's `sha256`, copied from `config/data.yaml` |
| `settings` | `group_by`, `seed`, `train_fraction`, `dev_size`, `quick_dev_size`, and the rule's name |
| `counts` | Examples and groups per split, and the quick-dev count |
| `groups` | Group key to `fit`, `dev`, or `test`, sorted by key |
| `assignments` | Example id to `fit`, `dev`, or `test`, in dataset order, one per line |
| `quick_dev` | The quick-dev ids, in dataset order |
| `content_hash` | D9 |

### 4.3 Loaders and the lock: `qlora_finetuning/data/split_loader.py`

| Function | Behavior |
|---|---|
| `load_split(name, cfg=None)` | Checks the manifest against the data (D15), then returns that split's records in dataset order. `quick_dev` returns the 200. `test` is subject to D12. An unknown name raises |
| `split_ids(name)` | The ids of a split without loading the data. `test` ids are available: they are in the committed manifest |
| `manifest_hash()` | The content hash, for `start_run` |
| `is_test_unlocked()` | Whether a valid unlock record exists |
| `TestLockedError` | Raised by D12. Its message says the test split is read once, in req_005 v5.01 |

Both paths, the token budget (v1.04), and the Bedrock export (v4.02) use only these.

### 4.4 Split report: `qlora_finetuning/data/split_report.py`

Per split (fit, dev, quick-dev, test) and for train as a whole: examples, share, companies,
reports, pages; program length distribution; operation counts and shares; argument kinds;
yes/no and table-operation examples; evidence kind (table, text, both); empty answers; largest
group; official-file origin. Then the leakage table of section 2.3 for the chosen key, and
the checks:

| Id | Check | Passes when |
|---|---|---|
| P1 | Complete | Every snapshot id has exactly one assignment, and the manifest has no other id |
| P2 | Same data | The manifest's dataset pin equals `config/data.yaml` |
| P3 | Ratio | Train share is within `train_fraction_tolerance` of `train_fraction` |
| P4 | Groups whole | No group has examples in two of fit, dev, test |
| P5 | Dev size | Dev is at most `dev_size` and within `dev_size_tolerance` of it |
| P6 | Quick-dev | Exactly `quick_dev_size` ids, all in dev |
| P7 | Pages and duplicates whole | No page, and no duplicate page-and-question pair, is in two splits |
| P8 | Hash | The content hash matches the assignments |
| P9 | Reproducible | Building again from the config and the snapshot gives the committed manifest exactly |
| P10 | Test locked | No unlock record exists (reported as "unlocked" after v5.01, not as a failure) |

### 4.5 `scripts/finqa_split.py`

| Command | Behavior |
|---|---|
| `venv\Scripts\python -m scripts.finqa_split` | Builds the split. If no manifest exists, writes it. If one exists, it must equal the rebuilt one. Then runs P1 to P10, writes the split report, and prints the check lines, the counts, and the content hash |
| `venv\Scripts\python -m scripts.finqa_split --check` | Writes nothing; fails if the manifest or the report differs from a fresh build |

Exit codes: 0 all checks pass, 1 a check fails or a committed file differs, 2 config problem
or no verified snapshot.

### 4.6 `splits/README.md`

What the folder holds, that the manifest is frozen, how a change is made and recorded, what
the unlock record is and who creates it.

### 4.7 Tests

All on invented rows (`tests/data/synthetic.py`, extended to several companies), in temporary
folders.

- Builder: the digest order is stable (fixed expected values); `fill` never exceeds the
  target and takes later groups that fit; each key; groups whole; dev inside train; quick-dev
  inside dev and of the right size; too few examples for the dev or quick-dev size raises; an
  example with no group key raises; the same input gives the same manifest; a different seed
  gives a different one.
- Manifest: round trip; the hash changes when one assignment changes; refuses to overwrite a
  different manifest; accepts an identical one; a hand-edited file fails validation.
- Loaders: each split returns the right ids in dataset order; an unknown name; a manifest for
  another dataset commit; a snapshot id missing from the manifest; an extra id.
- Lock: `load_split("test")` raises without an unlock record, with a record naming another
  hash, and with a malformed record; it loads with a valid one (written by the test in a
  temporary folder).
- Guards on the repository itself: `splits/test_unlock.json` does not exist; only the allowed
  files import the raw reader (D13); the committed manifest, when a snapshot is present,
  passes P1 to P9 (skips in CI, which has no snapshot); the committed manifest's own shape and
  hash are valid (runs in CI).
- Report: counts on a small set; each check passes and fails; no example text in the output;
  deterministic.
- Script: each exit code; `--check`; module form and by-path hint.

### 4.8 Documentation

- Runbook section 4: build and check the split; what "frozen" means; what to do if `--check`
  fails.
- `coding.md`: get examples with `load_split`; never call the raw reader outside the data
  package; pass `manifest_hash()` to `start_run`.
- `reqspecs/references/project_patterns/`: a short note on the split loaders for v1.03, v1.04,
  v2.02, v3.x, and v4.x.
- `high_level_plan.md` §4.1: grouping by company, the fill rule, dev of 499; §5: `splits/`.
- `execution_plan.md`: Q4 resolved; M2 with the content hash.

## 5. Out of scope

- Prompts, targets, table formatting, evidence selection (v1.03).
- Token counts (v1.04).
- Whether a gold program reproduces its answer, and any per-example flag (v2.02).
- Uploading anything to S3 (v3.03, v4.02).
- Creating the unlock record (v5.01).

## 6. What this hands to later versions

| For | What |
|---|---|
| v1.03, v1.04 | `load_split("fit")`, `load_split("dev")`, `load_split("quick_dev")` |
| v2.02 | Split ids, so the gold self-check can report per split; flags go in a separate file (D18) |
| v2.03, v3.x, v4.x | `manifest_hash()` for every run's manifest |
| v3.03, v4.02 | Exports are built from `load_split`, so no test row can reach S3; each adds its own "zero test ids" check |
| v5.01 | Creates `splits/test_unlock.json` naming the manifest hash, in the same commit as the configuration freeze record |
| The report (v5.03) | The test set is 43 companies none of which was seen in training; results are not comparable with published FinQA numbers, which use the official split |

## 7. Known limits and follow-ups

- **Dev is small in companies.** 499 examples from 15 companies. Dev accuracy will be noisier
  than the same number of examples drawn across all companies, and dev has no `exp` example
  and only 10 table-operation examples. Checkpoint selection and the Bedrock validation file
  rest on it.
- **The operation mix differs between fit and test** by up to 3.7 points (`add`). This is what
  the companies happen to ask; it is reported, not corrected (D19).
- **A company is identified by its ticker.** One company under two tickers in different years
  would count as two. None was looked for.
- **Chance overlap remains.** 58 test examples (3.1% of those checked) share a row name and a
  number with some training table of another company.
- **`exp` is almost unseen in training.** The seed put 8 of the dataset's 9 `exp` steps in
  test and 1 in fit. The tuned models will have had one example of it. The seed is not
  re-rolled (D3); the report will show these examples separately if they matter.
- **The numbers in section 2 come from a throwaway script.** The built manifest must
  reproduce section 2.4; any difference is investigated before the manifest is committed.
- **The lock is a convention enforced by tests**, not access control.
- **The test set is harder than a page-grouped one.** Accuracy for every arm will likely be
  lower than the guides' ranges, which assume the official split.

## 8. Decisions for the owner

All three were accepted on 2026-11-15, when the owner proceeded to the PRP.

| # | Question | Recommendation |
|---|---|---|
| 1 | **Q4: group by company or by report?** | Company (D1). It departs from the plan's default but is the only key that keeps training numbers out of the test set. Switching to report is one line in `config/split.yaml` before the manifest is first written |
| 2 | Accept a dev slice of 15 companies? | Yes. The alternative, grouping dev by report inside the training companies, would make dev easier than test |
| 3 | Seed 20261115, with no second try? | Yes |

## 9. Acceptance

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with and
  without the training stack installed.
- `venv\Scripts\python -m scripts.finqa_split` writes the manifest and the report; P1 to P10
  pass; a second run changes nothing; `--check` exits 0.
- The manifest reproduces section 2.4: 4,884 / 499 / 2,898, quick-dev 200, 79 / 15 / 43
  companies.
- `load_split("test")` raises `TestLockedError`; `load_split("fit")`, `("dev")`, and
  `("quick_dev")` return 4,884, 499, and 200 records.
- No file outside the allowed list imports the raw reader.
- The content hash is recorded in the req_001 roadmap and in the execution plan as M2.

## 10. Next step

Complete; CI green on pull request #3 and on `main` (2026-11-15). Then the v1.03 spec (canonical serialization).
