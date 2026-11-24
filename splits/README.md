# splits/

The frozen train/test split of FinQA (req_001 v1.02) and the gold flags (req_002 v2.02).
This folder is committed.

| File | What it is |
|---|---|
| `finqa_split_manifest.json` | Which example is in `fit`, `dev`, or `test`, which dev examples are quick-dev, the settings and dataset that produced it, and a content hash |
| `finqa_gold_flags.json` | The examples whose gold program does not reproduce its gold answer, with the reason and the settings that decided it. Empty today: all 8,281 reproduce |
| `test_unlock.json` | **Does not exist yet.** Created once, by req_005 v5.01, to unlock the test split |

## The manifest is frozen

Both training paths and the final comparison use exactly these assignments. The manifest was
written once, on 2026-11-15, by `python -m scripts.finqa_split`. That command never replaces
it: on later runs it only confirms that `config/split.yaml` and the pinned data still produce
the same file.

- Do not edit the file. It carries a hash of its own contents, and every loader refuses a
  manifest whose hash does not match.
- Changing the split invalidates every result obtained with it. If it must change, record the
  decision in `reqspecs/reqs/req_001_finqa_data/req_001_roadmap.md` first, then delete the
  file by hand and run the build command again. There is no force option, on purpose.

Content hash of the current manifest:
`d34fb22e37b78a31ac22287cc8e199ab942d93125f9a80cb0b323862e599a07a`

## What is in it

| Key | Content |
|---|---|
| `dataset` | The dataset repository, commit, and file checksums the split was built from |
| `settings` | Grouping key, seed, training fraction, dev and quick-dev sizes, rule name |
| `counts` | Examples and groups per split |
| `content_hash` | SHA-256 over one line per example: id, split, quick-dev flag |
| `groups` | Each company and the split all its examples are in |
| `quick_dev` | The ids of the 200 quick-dev examples |
| `assignments` | Each example id and its split, in dataset order |

It holds ids and company symbols only: no question, context, or answer text.

## The gold flags

`finqa_gold_flags.json` is written by `python -m scripts.finqa_gold_check`, which scores
every example's gold reply with the evaluation harness (req_002 v2.02). An example is
flagged when its gold program fails or its value does not match the gold answer.

- A flagged example is **never dropped**. It stays in its split and in every total; scoring
  reports the total over all examples, which is the headline, and the total over the
  unflagged ones beside it.
- The file is separate from the manifest so that the manifest and its hash never change.
- It holds ids, splits, reasons, counts, the manifest hash, the prompt id, and the
  allowances of `config/evaluation.yaml`. A flag is only meaningful for the rule that
  produced it, so changing an allowance means rebuilding the file.
- Do not edit it. It carries a hash of its entries, and `load_gold_flags()` refuses a file
  whose hash does not match.

```python
from qlora_finetuning.data.gold_check import load_gold_flags

flagged = load_gold_flags()    # a frozenset of example ids; empty today
```

The counts per split are in `reports/data/finqa_gold_check_report.md`. How to rebuild or
check is in `docs/runbook.md`, section 9.

## The test lock

`load_split("test")` raises `TestLockedError` until `test_unlock.json` exists and contains the
manifest's content hash:

```json
{"manifest_hash": "<content hash>", "unlocked_by": "req_005 v5.01"}
```

Creating that file is the single, deliberate act that opens the test set. It belongs to
req_005 v5.01 and to the owner. Any other file at that path, or one naming another hash,
does not unlock anything.

The lock protects against accidents. It is a convention enforced by tests, not access
control.

## The freeze and the unlock (req_005 v5.01)

Three more files appear here when the test split is opened. None is written by hand.

| File | Written by | What it holds |
|---|---|---|
| `config_freeze.json` | `python -m scripts.comparison_freeze freeze` | Each of the six arms' predictor (name, model id, settings), the prompt id, the manifest hash, the reply limit, and the SHA-256 of every shared config file and of the gold flags. Written before the unlock, never after it |
| `test_unlock.json` | `python -m scripts.comparison_freeze unlock --yes --freeze <hash>` (the owner, once) | `manifest_hash`, `freeze_sha256` (the freeze record it signs off), `freeze_commit`, `unlocked_on` |
| `test_run_voids.json` | `python -m scripts.comparison_runs void` (the owner) | Test runs voided for a pipeline or service fault, each with its reason. Absent unless a run was voided |

They live here, not under `reports/`, because the gate that reads them also runs inside a
SageMaker job, and a job's code holds `config/` and `splits/` only. A readable copy of the
freeze record is `reports/comparison/config_freeze.md`.

After the unlock, `load_split("test")` works, and every evaluation run on `test` must pass
the gate (`qlora_finetuning/comparison/gate.py`): the unlock record must name the freeze
record on disk, nothing frozen may have changed, and each arm answers once. An unlock record
that names no freeze record opens nothing. There is no way to lock again. The steps are in
`docs/runbook.md`, section 7.

## Reading the split

```python
from qlora_finetuning.data import load_split, manifest_hash

fit = load_split("fit")
dev = load_split("dev")
quick = load_split("quick_dev")
```

Statistics per split are in `reports/data/finqa_split_report.md`. How to rebuild or check is
in `docs/runbook.md`, section 4.
