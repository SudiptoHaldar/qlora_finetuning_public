# Spec summary: Run registry and cost ledger (req_000 v0.04)

**PRP**: [prp_000_run_registry_ledger_v4.md](../PRPs/prp_000_run_registry_ledger_v4.md)  
**Origin**: `reqspecs/reqs/req_000_project_scaffold/req_000_04_spec.md`  
**Created**: 2026-11-15  

## Todo list

1. Write `config/tracking.yaml` and the shared pieces: `tracking/config.py`, `tracking/jsonl.py`.
2. Write `tracking/ledger.py`: the record, its rules, append, read, effective records,
   summary, and the two functions that turn an AWS response into a record.
3. Write `tracking/registry.py`: run ids, git state, `start_run`, `Run`, manifests, the index.
4. Write the ledger and registry tests, all in temporary folders.
5. Write `scripts/ledger_summary.py` and its tests, including one dummy run end to end.
6. Write `ledger/README.md`, the runbook section, the plan layout line, the `coding.md` rule,
   and the run-tracking convention note.
7. Validate with and without the training stack; update the roadmap, plan, and summaries.

## Summary

The run registry gives every training or evaluation run an id, a folder under git-ignored
`artifacts/runs/`, and a manifest recording the git commit, package version, environment, a
copy and hash of each config file, and the split-manifest hash. The cost ledger is an
append-only JSON Lines file of what each run consumed, in the units AWS bills. It holds
quantities, never derived dollars, and every record names its source, because AWS reports no
token count for a Bedrock fine-tuning job. The ledger and the index of runs are committed in a
top-level `ledger/` folder.

No AWS call is made. Two pure functions turn a `DescribeTrainingJob` response or a `Converse`
usage block into a ledger record; the calls belong to later versions.

## Confidence score

9/10. Pure Python over files, with the AWS field names confirmed from the installed
definitions.

## Implementation Time Estimate

3–4 hours.

| Task | Time |
|---|---|
| Config, `config.py`, `jsonl.py` | 25 min |
| `ledger.py` | 45 min |
| `registry.py` | 50 min |
| Ledger and registry tests | 50 min |
| `ledger_summary.py` and tests | 25 min |
| Documents | 25 min |
| Validation and status documents | 15–30 min |

## Time Taken for PRP Generation

~8 minutes.

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 1 min |
| Codebase exploration | 1 min |
| Research (AWS field names) | 1 min |
| Writing PRP document | 5 min |
