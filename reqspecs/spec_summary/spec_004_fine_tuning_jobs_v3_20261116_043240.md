# Spec summary: fine-tuning jobs (req_004 v4.03)

**PRP**: [prp_004_fine_tuning_jobs_v3.md](../PRPs/prp_004_fine_tuning_jobs_v3.md)  
**Spec**: `reqspecs/reqs/req_004_frontier_tuning/req_004_03_spec.md`  
**Generated**: 2026-11-16  

## Todo list

- [ ] Config v3: customization model ids, tuned arms, hyperparameters, names, launch order
- [ ] `tuning_record.py`: the committed record of jobs, models, and deployments
- [ ] In-memory Bedrock and S3 stand-ins for the tests
- [ ] The plan, the request token, checks T1 to T9
- [ ] `launch`: adopt or create, tuning record, ledger record, a `train` run
- [ ] `status` and `--wait`, the loss curves; `deploy`, `stop`, `bill`, `teardown`
- [ ] `scripts/nova_tune.py`; the guards (no Provisioned Throughput, `--yes`, imports)
- [ ] Runbook with teardown, references, `coding.md`, gotchas, architecture
- [ ] Lint and tests, with and without the training stack
- [ ] Spec, roadmap, execution plan, implementation summary
- [ ] Owner: baselines, gate, upload, then Lite: plan, launch, status, bill; then Pro; then deploy both

## Summary

`plan` shows what a launch would do and runs nine checks: the gate is open and current,
the files are in S3 with the recorded checksums, the model is fine-tunable and Active, the
account has no Provisioned Throughput, the job is not launched yet, the path stays within
its cap, the tree is committed, and, for Nova Pro, Nova Lite's job has completed and its
billed amount has been entered. It starts nothing.

`launch` creates one job per command with a request token derived from the model, the
files, the hyperparameters, and the attempt, so a repeat asks for the same job; if a job
with that token already exists it is adopted. The tuning record is written as soon as the
ARN is known, and the ledger gets a `train` record with the estimated tokens. `status` only
reads: it updates the record, and for a completed job it fetches the loss curves. `deploy`
puts a finished model behind an on-demand deployment. `stop`, `bill`, and `teardown` stop a
job, replace the estimate with the billed amount, and delete deployments and models.

The owner decided the spec's three questions on 2026-11-16: Bedrock's default learning
rate and warmup steps; Nova Lite first; Nova Pro only after Nova Lite's charge has been
checked on the bill. The agent launches nothing.

## Confidence score

7/10. The API shapes are read from the installed library and the preconditions all have a
source of truth. None of it can be tried against Bedrock before the owner's first launch,
which is Nova Lite's on purpose.

## Implementation Time Estimate

5–6 hours of code; then hours of waiting per job and about a day for Lite's bill.

| Task | Time |
|---|---|
| Config | 25 min |
| Tuning record | 35 min |
| Fakes and fixtures | 35 min |
| Plan and checks | 60 min |
| Launch | 50 min |
| Status, deploy, stop, bill, teardown | 70 min |
| Script and guards | 40 min |
| Documents | 30 min |
| Validation and status documents | 20–40 min |

## Time Taken for PRP Generation

~10 minutes.

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 3 min |
| Research (existing patterns) | 1 min |
| Writing PRP document | 4 min |
