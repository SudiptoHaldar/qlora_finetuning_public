# Spec summary: Process bootstrap (req_000 v0.01)

**PRP**: [prp_000_process_bootstrap_v1.md](../PRPs/prp_000_process_bootstrap_v1.md)  
**Origin**: `reqspecs/reqs/req_000_project_scaffold/req_000_01_spec.md`  
**Created**: 2026-11-14  

## Todo list

1. Rewrite the stale parts of the five command files (`generate-prp`, `execute-prp`,
   `release-tag`, `wip-roll`, `roadmap_view-roll`).
2. Rewrite the stale parts of the three PRP templates and rewrite `coding.md`.
3. Remove the inherited database stack: `alembic.ini`, `alembic/`, six pins in
   `requirements.txt`, `DATABASE_URL`, the `db` pytest marker.
4. Create `reqspecs/references/` (README, gotchas, troubleshooting, boto3, project patterns)
   and the two `track_progress/` folders.
5. Update the runbook (install from `requirements.txt`, stub sections), `.gitignore`
   (`artifacts/`), and the stale-reference report (closing section).
6. Add `tests/test_process_scaffold.py` as a guard.
7. Validate: stale-reference scan, `pip install -r requirements.txt`, ruff, pytest, `aws_check`.
8. Update the roadmap, execution plan, and spec status; write the implementation summary.

## Summary

The repository was cloned from an SEC EDGAR MCP-server project and its process files still
described that system. This PRP retargets them: commands run through the repo venv instead of
`uv`, the two planning guides replace an absent EDGAR guide, publication steps are removed, and
the examples are this project's. It deletes the unused database stack and creates the reference
and tracking folders the commands already expect. No application code is written.

Edits are applied by line range with a check that the old lines contain an expected phrase, so
a file that has changed stops the edit instead of being overwritten.

## Confidence score

9/10. All work is text editing against files read in full, with an executable scan as the gate.

## Implementation Time Estimate

1.5–2.5 hours.

| Task | Time |
|---|---|
| Command files | 35 min |
| Templates and `coding.md` | 25 min |
| Database stack removal | 5 min |
| References and tracking folders | 25 min |
| Runbook, `.gitignore`, stale-reference report | 10 min |
| Guard test | 10 min |
| Validation and fixes | 10–30 min |
| Roadmap, plan, summaries | 10 min |

## Time Taken for PRP Generation

~12 minutes.

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 3 min |
| Research (the two guides, v0.05 PRP) | 2 min |
| Writing PRP document | 5 min |
