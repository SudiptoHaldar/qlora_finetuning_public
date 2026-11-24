# Convention: getting FinQA examples by split

**Last verified**: 2026-11-16  
**Defined in**: `qlora_finetuning/data/split_loader.py` (req_001 v1.02). Recorded before its
second use so that v1.03, v1.04, v2.02, v2.03, v3.x, and v4.x all follow it.

Every stage outside the data package gets its examples by split name. Nothing else reads the
dataset.

## Use

```python
from qlora_finetuning.data import load_split, manifest_hash, split_ids
from qlora_finetuning.tracking import start_run

fit = load_split("fit")            # 4,884 records: the only examples a model is trained on
dev = load_split("dev")            # 499: checkpoint selection, early stopping, validation file
quick = load_split("quick_dev")    # 200 of the dev examples: fast baselines

with start_run("train", "a1", path="A", arm="A1", where="sagemaker", scope="measured",
               configs=["split", "data"], split_manifest_hash=manifest_hash()) as run:
    ...
```

Each record is a `RawExample` (`reqspecs/references/tech/finqa_dataset.md`). Records come back
in dataset order. A loader first checks that the committed manifest describes exactly the data
on disk and raises `SplitError` if not.

## In a cloud job

The snapshot holds every row, the test rows among them, so it is never uploaded. A SageMaker
job is given a **data bundle** instead (req_003 v3.03): the raw examples of `fit` and `dev`,
written by `python -m scripts.sagemaker_job bundle` (`export_bundle` in
`qlora_finetuning/data/bundle.py`). When the environment variable `QLORA_DATA_BUNDLE` names
its folder, `load_split` serves `fit`, `dev`, and `quick_dev` from it, after checking the
file's hash, the dataset commit, the split manifest hash, and that the ids are exactly the
manifest's, in order. Callers do not change. `test` is then refused whatever the unlock
record says: the rows are not there. Such a bundle cannot hold a test row: writing one is
refused, and one found on reading stops the load.

The two test jobs of req_005 v5.01 are given a **test bundle** instead: the test examples
and nothing else (`sagemaker_job bundle --split test`, possible only after the unlock).
From it `load_split("test")` is served when the unlock record in the job's code names the
manifest, and `fit` and `dev` are refused. A bundle is one kind or the other, never both.

## After the unlock (req_005 v5.01)

`load_split("test")` returning rows does not make a test run allowed. `run_evaluation` puts
every run on `test` through `check_test_run` (`qlora_finetuning/comparison/gate.py`): the
unlock record must name the freeze record, the predictor must be the frozen one, the run
must be measured, whole, in the cloud, and the arm's first. Code other than the runner has
no reason to load test rows: `serialize_split("test")` raises whatever the lock says, and a
guard test lists the two files that may name the test split in a load
(`comparison/runs.py` for the gold check, `scripts/sagemaker_job.py` for the test bundle).

## Rules

- **Three names only**: `fit`, `dev`, `quick_dev`. There is no "train" loader; when both are
  wanted, ask for `fit` and `dev` separately so it stays clear what was trained on.
- **`load_split("test")` raises `TestLockedError`** until the owner unlocks it (req_005
  v5.01). Do not catch the error, and do not work around it. After the unlock, do not load
  test rows either: evaluate through `run_evaluation`, which has the gate.
- **Never call `load_raw_examples`, read the Parquet files, or read the Hub** outside
  `qlora_finetuning/data/`. A guard test lists the files allowed to, and fails on any other.
- **Anything exported** (a training file, an S3 upload) is built from `load_split`, and the
  exporter adds its own check that the output holds no test id. `split_ids("test")` gives the
  ids to check against; it works while the split is locked.
- **Record the split** in every run: `split_manifest_hash=manifest_hash()` and `"split"` in
  `configs`.
- **Dev is the only place to look** when choosing a prompt, a checkpoint, or a setting.
- **Per-example flags** go in their own file keyed by example id, never in the manifest.
  The first is `splits/finqa_gold_flags.json` (req_002 v2.02): the examples whose gold
  program does not reproduce its gold answer, with the reason. Read it with
  `load_gold_flags()` from `qlora_finetuning.data.gold_check`; it reads the committed file,
  not the dataset, so any code may call it, and it covers test ids while the split is
  locked. A flagged example is never dropped: pass the ids to `summarize(scores,
  flagged=...)`, which reports the total over all examples and over the unflagged ones.
  The file is empty today.
- **Three scripts may read every row**, test rows included: `finqa_report`, `finqa_split`,
  and `finqa_gold_check`. Each reports counts and ids only. The guard test lists them, and
  a second test fails if the list grows.

## Tests

- Use invented rows: `tests/data/synthetic.py` has `build_frozen_split(tmp_path)`, which
  returns a split config, a data config, and a manifest, all under the temporary folder. Pass
  the two configs to `load_split(name, cfg, data_cfg)`. Its `rows` argument takes the rows
  of each official file, for a test that needs some of them changed.
- To exercise code that needs the test split, write an unlock record in the temporary folder
  (`{"manifest_hash": manifest["content_hash"]}`). Never in the repository. For a run on
  test, the gate also needs a freeze record: `tests/comparison/helpers.py` has
  `freeze_and_unlock(context)`, `arm_predictor(arm, context)`, and `run_arm(arm, context)`.

## Reuse checklist

- [ ] Examples come from `load_split`, with a name of `fit`, `dev`, or `quick_dev`
- [ ] The run records `manifest_hash()`
- [ ] No import of the raw reader; the guard test stays green
- [ ] An export checks itself against `split_ids("test")`
- [ ] Tests run on `build_frozen_split(tmp_path)`
