# Spec summary: SageMaker job packaging (req_003 v3.03)

**PRP**: [prp_003_sagemaker_job_packaging_v3.md](../PRPs/prp_003_sagemaker_job_packaging_v3.md)  
**Spec**: `reqspecs/reqs/req_003_qlora_local/req_003_03_spec.md`  
**Generated**: 2026-11-19  

## Todo list

- [ ] Shared sign-in module; `frontier/bedrock.py` re-exports it
- [ ] Data bundle (`fit` and `dev` only) and `load_split`'s bundle mode, with tests
- [ ] `config/sagemaker.yaml`, SageMaker prices, the loader; inference batch size 8
- [ ] Job runner (`job.py`, `scripts/job_run.py`) and `scripts/job_bootstrap.sh`
- [ ] Launcher: source archive, job definition, checks J1 to J8, `plan`, `launch`, `status`, `logs`, `stop`
- [ ] `pull`, staging, the committed job record, billable seconds in the ledger
- [ ] `scripts/sagemaker_job.py` with `bundle` and `rehearse`
- [ ] Lint and tests, with and without the training stack
- [ ] On this machine: the bundle, the rehearsal of the smoke task, `plan` signed out
- [ ] Runbook, tech note, gotchas, patterns, `coding.md`, architecture
- [ ] Spec, roadmap, execution plan (🟡), implementation summary
- [ ] Owner: sign in, `aws_check`, `plan smoke`, `launch smoke --yes`, `status`, `pull`

## Summary

A job is given three things: the code as `git archive` of a commit, a bundle of the `fit`
and `dev` examples (never the snapshot, which holds the test rows), and an S3 workspace.
In the job a short bootstrap links `artifacts/` and `ledger/` into the checkpoint folder,
which SageMaker keeps in step with the workspace, installs the pinned libraries over AWS's
PyTorch 2.10 container, downloads the model, and runs `scripts.job_run` with a task:
`smoke`, `train`, or `evaluate`. The task calls `run_training` and `run_evaluation`
unchanged.

The launcher plans a job (the definition, eight checks, the most it can cost), launches it
for the owner, reports status and logs, and pulls the runs, the ledger lines, and the
billable seconds home. A committed record keeps every job. A Path A cap of $100 is checked
before each launch.

The owner started the PRP without changing the spec's five questions, so they are taken as
recommended. The sign-in is expired: the agent builds, tests, and rehearses locally and
creates nothing in AWS. The smoke job, about $0.60, is the owner's.

## Confidence score

6/10. The code can be tested and rehearsed here, but nothing of AWS's side can be tried
before the owner's job: the entry point, the checkpoint sync, pip in the container, and
the driver against the pinned torch. One or two corrected attempts would not be surprising.

## Implementation Time Estimate

7–9 hours of work, about 10 minutes of GPU, then the owner's six commands.

| Task | Time |
|---|---|
| Shared sign-in module | 20 min |
| Bundle and loader | 50 min |
| Config and prices | 30 min |
| Job runner and bootstrap | 60 min |
| Launcher: source, definition, checks, launch, status, logs, stop | 110 min |
| Pull, staging, job record | 70 min |
| Script | 40 min |
| Tests | 110 min |
| Bundle, rehearsal, plan | 20 min |
| Documents and status files | 50 min |

## Time Taken for PRP Generation

~12 minutes.

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 5 min |
| Research existing PRPs and patterns | 1 min |
| Writing PRP document | 4 min |
