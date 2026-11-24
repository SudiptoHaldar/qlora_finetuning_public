# Spec summary: Dataset snapshot and schema report (req_001 v1.01)

**PRP**: [prp_001_dataset_snapshot_v1.md](../PRPs/prp_001_dataset_snapshot_v1.md)  
**Origin**: `reqspecs/reqs/req_001_finqa_data/req_001_01_spec.md`  
**Created**: 2026-11-15  

## Todo list

1. Write `config/data.yaml`, add `pyarrow` to `requirements.txt`, and write
   `data/snapshot.py`: config, download, verification, typed raw reader.
2. Write the synthetic-data helper and the snapshot tests.
3. Write `data/schema_report.py`: program splitting, counts, flagged ids, checks, JSON and
   Markdown output.
4. Write the report tests, including the "no example text" guard.
5. Write `scripts/finqa_fetch.py` and `scripts/finqa_report.py` with their tests.
6. Fetch the real snapshot, build the report, and reconcile it with spec section 2.
7. Write the runbook section, the dataset reference note, the gotchas, and correct the dataset
   name in the plan documents.
8. Validate with and without the training stack; update the roadmap, plan, and summaries.

## Summary

FinQA is pinned to `bevaya/FinQA` at one commit, with the SHA-256, size, and row count of each
of its three Parquet files in `config/data.yaml`. The files are downloaded once into
git-ignored `artifacts/data/finqa/<commit>/` and checked against those values on every load.
A raw reader returns the 8,281 examples as typed records with nothing dropped or repaired.

A schema report is generated from the snapshot and committed under `reports/data/`. It holds
counts and example ids only, never question, context, or answer text, because any example may
become a test example in v1.02. It carries no timestamp, so rebuilding it gives the same
bytes. Ten checks compare the data with the expected facts in the config and fail loudly if
the pinned data ever changes.

## Confidence score

8/10. The data was profiled before the spec was written. The risk is in reproducing every
count exactly and in the first real download through the library.

## Implementation Time Estimate

4–5 hours.

| Task | Time |
|---|---|
| Config, requirement, `snapshot.py` | 50 min |
| Synthetic data helper and snapshot tests | 40 min |
| `schema_report.py` | 60 min |
| Report tests | 40 min |
| Scripts and their tests | 35 min |
| Real run and reconciliation | 20 min |
| Documents | 25 min |
| Validation and status documents | 15–30 min |

## Time Taken for PRP Generation

~10 minutes.

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 1 min |
| Codebase exploration | 2 min |
| Research (file schema, script and test conventions) | 2 min |
| Writing PRP document | 5 min |
