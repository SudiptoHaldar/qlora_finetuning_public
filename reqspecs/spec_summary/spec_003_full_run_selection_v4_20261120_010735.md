# Spec summary: Full training run and checkpoint selection (req_003 v3.04)

**PRP**: [prp_003_full_run_selection_v4.md](../PRPs/prp_003_full_run_selection_v4.md)  
**Spec**: `reqspecs/reqs/req_003_qlora_local/req_003_04_spec.md`  
**Generated**: 2026-11-20  

## Todo list

- [ ] Config: the `score` task and the estimate block; the selection block and its loader
- [ ] The chain of a resumed training run; readers of the selection record
- [ ] The `score` task in `job.py` and `scripts/job_run.py`
- [ ] The estimate, check J9, `--run`, `--max-runtime` in the launcher
- [ ] `selection.py` and `scripts/qlora_select.py`
- [ ] `--adapter selected`; `pull --checkpoint`; `keep`
- [ ] Lint and tests, with and without the training stack
- [ ] On this machine: score a local smoke run twice; `qlora_select --check`; `plan train`
- [ ] Runbook, run sheet, tech note, gotchas, patterns, `coding.md`, architecture
- [ ] Spec, roadmap, execution plan (🟡), implementation summary
- [ ] Owner: the v3.03 smoke job; then the training job, the scoring job, the selection, `keep`

## Summary

A new `score` task scores the untuned model and every checkpoint of a training run on a
split, each as an ordinary evaluation run, in one job. It skips a checkpoint that already
has a complete result and resumes one that was cut short, so a stopped scoring job is
simply launched again. Checkpoints of a resumed training run are found along its chain.

The launcher's plan gains an estimate, computed from the seconds per training sequence and
per reply that the last pulled job measured on the same instance type, and a ninth check
that fails when there is nothing to measure from or the estimate exceeds the job's limit.

`scripts/qlora_select.py` reads the pulled runs, recomputes every count, runs six checks,
picks the checkpoint with the best dev execution accuracy, and judges it against two
thresholds (40% execution accuracy, 98% format validity). It writes a committed selection
record and report, which alone say which run is arm A1. `--adapter selected` reads it.

The owner started the PRP before the v3.03 smoke job was run. The code is built and
rehearsed locally; the two paid jobs wait for the smoke job. Nothing is created in AWS.

## Confidence score

7/10. The parts are extensions of code that is tested and, for scoring and selection,
rehearsable on local runs. The estimate cannot be compared with a real job yet, and the
launcher underneath has still never run in AWS.

## Implementation Time Estimate

6–8 hours of work, about 10 minutes of GPU, then the owner's steps.

| Task | Time |
|---|---|
| Config and loaders | 30 min |
| Chain and selection-record readers | 30 min |
| The score task | 60 min |
| Estimate, J9, launcher options | 70 min |
| Selection and its script | 90 min |
| `selected`, `pull --checkpoint`, `keep` | 45 min |
| Tests | 110 min |
| Local rehearsal | 15 min |
| Documents and status files | 45 min |

## Time Taken for PRP Generation

~10 minutes.

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 1 min |
| Codebase exploration | 3 min |
| Research existing PRPs and patterns | 1 min |
| Writing PRP document | 5 min |
