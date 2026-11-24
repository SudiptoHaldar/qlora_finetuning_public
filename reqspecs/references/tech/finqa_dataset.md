# FinQA dataset

**Last verified**: 2026-11-15, on the pinned snapshot  
**Sources**: `reports/data/finqa_schema_report.md` (generated from the data; the authority for
every count), `config/data.yaml`, and the dataset card at
https://huggingface.co/datasets/bevaya/FinQA.

What the project's FinQA actually contains. The two planning guides describe the original
GitHub release; where they and this file disagree, this file describes the data in use.

## Source

- Repository `bevaya/FinQA`, commit `3d6a736bc67e06bc15fbf3618d88204a57c5b25e`. The old name,
  `rootsautomation/FinQA`, redirects to it.
- Public, not gated, MIT. A third-party repackaging of `github.com/czyssrs/FinQA` as Parquet.
- Three files: train 6,251 rows, validation 883, test 1,147. Total 8,281.

## Reading it

```python
from qlora_finetuning.data import load_split

fit = load_split("fit")   # verifies checksums and the split manifest, returns RawExample records
```

- Use `load_split` with `fit`, `dev`, or `quick_dev` (`../project_patterns/split_loading.md`).
  `load_raw_examples()` returns all 8,281 rows, test rows included, and is for the data
  package only.
- Files are read with `pyarrow.parquet.read_table(path).to_pylist()`, which gives plain lists
  and dicts. `pandas.read_parquet` would give NumPy arrays for the list columns.
- Download: `huggingface_hub.hf_hub_download(repo_id=, filename=, repo_type="dataset",
  revision=, local_dir=)`. It keeps the `data/` folder under `local_dir` and adds a `.cache`
  folder.
- The Hub lists a commit's files, with size and SHA-256 (`lfs.oid`), at
  `https://huggingface.co/api/datasets/<repo>/tree/<commit>/data`. No library is needed.

## Columns

Sixteen, the same in every file, never null.

| Column | Type | Notes |
|---|---|---|
| `id` | string | `<company>/<year>/page_<n>.pdf-<index>`; unique |
| `filename` | string | The id without `-<index>`: one page of one annual report |
| `pre_text`, `post_text` | list of string | Passages before and after the table; never empty |
| `table` | list of list of string | Cleaned table; 1–20 rows, 2–14 columns, never ragged |
| `table_ori` | list of list of string | Original formatting; always differs from `table` |
| `question` | string | Never empty |
| `answer` | string | Free-form; empty 74 times |
| `exe_ans` | string | A number, or `yes` / `no`; never empty |
| `explanation` | string | Empty in 84% |
| `ann_table_rows`, `ann_text_rows` | list of int | Unreliable; do not use |
| `steps` | struct of four lists | The program step by step, with other operation names |
| `program` | string | `subtract(920, 95), divide(#0, 5)` |
| `program_re` | string | Nested form of the same program |
| `gold_inds` | string | JSON object: evidence key to evidence text |

The original release nests the question fields under `qa`; here they are top-level.

## Programs

- Form: `op(a, b), op(a, b), ...`. Always two arguments. One to six steps.
- Ten operations: `add`, `subtract`, `multiply`, `divide`, `exp`, `greater`, `table_sum`,
  `table_average`, `table_max`, `table_min`.
- Argument kinds: whole number, decimal, step reference (`#0`), constant, **percent literal**
  (`6.9%`, 371 uses), and for table operations a row name and `none`.
- Seventeen constants: `const_1` to `const_10`, `const_100`, `const_1000`, `const_10000`,
  `const_100000`, `const_1000000`, `const_1000000000`, `const_m1` (minus one).
- Table operations take a row name that equals the first cell of a row of the example's table.
  Names contain spaces, never a comma or bracket. The second argument is `none`, except a
  number in two examples.
- `greater` is always the last step, and its result is `yes` or `no`.

## Answers

- `exe_ans` is the executed result: numeric in 8,127 examples, `yes` / `no` in 154. It looks
  rounded to about five significant digits and is in scientific notation 223 times.
- `answer` is what the annotator typed: percentages, plain numbers, dollar amounts, ratios,
  sentences, sometimes nothing. It is often on a different scale from `exe_ans` (`93.5%`
  against `0.935`).

## Evidence

- Use `gold_inds`. Keys are `table_N` (a table row) or `text_N` (a passage, counting through
  `pre_text` then `post_text`). A table entry's value is the row rewritten as a sentence.
- One example, `RE/2010/page_120.pdf-2`, has the malformed key `text_-1`.
- The `ann_*` lists disagree with `gold_inds` in 1,302 examples and are empty in 631 that have
  evidence.

## Grouping

| Level | Groups | Examples per group |
|---|---|---|
| Company | 137 | 2 to 472 |
| Report (company + year) | 799 | 1 to 95 |
| Page | 2,789 | 1 to 6 |

The official splits keep a page together but split reports (355 of 799) and companies (117 of
137) across splits. 73 pairs of examples share a page and a question word for word.

## The project's split

Frozen in `splits/finqa_split_manifest.json`; statistics in
`reports/data/finqa_split_report.md`. Grouped by company, seed 20261115.

| Split | Examples | Companies |
|---|---|---|
| fit | 4,884 | 79 |
| dev | 499 | 15 |
| test | 2,898 | 43 |
| quick-dev (inside dev) | 200 | 12 |

- Train (fit + dev) is 5,383, exactly 65.00%.
- No test company appears in training, so no test question rests on a number the model saw
  under the same row name for the same company. Grouping by report would have left 28.4% of
  the checkable test questions resting on such a number, because annual reports print prior
  years.
- The operation mix differs a little between splits: `add` is 16.6% of fit steps and 12.9% of
  test steps. `exp` has 1 step in fit, none in dev, and 8 in test.
- These results are not comparable with published FinQA numbers, which use the official
  split.

## How an example is shown to a model

`../project_patterns/canonical_record.md`. In short: text evidence only (63% of fit and dev
examples have none), the whole table as Markdown, the question; the reply is the dataset's
program unchanged and `exe_ans` as the answer. Lengths are in
`reports/data/finqa_serialization_report.md`.

## Rules for anything committed

Any example may be a test example. A committed file may hold counts and example ids, never
question, passage, table, or answer text. The schema report follows this and a test enforces
it. Tests use invented rows (`tests/data/synthetic.py`), never real ones.
