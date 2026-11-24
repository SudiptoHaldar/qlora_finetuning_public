# Spec summary: test unlock and six-arm evaluation (req_005 v5.01)

**PRP**: [prp_005_test_unlock_six_arms_v1.md](../PRPs/prp_005_test_unlock_six_arms_v1.md)  
**Spec**: `reqspecs/reqs/req_005_comparison/req_005_01_spec.md`  
**Generated**: 2026-11-22  

## Todo list

- [ ] `config/comparison.yaml` and its loader: where the records live, the six arms
- [ ] The gate (`check_test_run`), called inside `run_evaluation` when the split is `test`
- [ ] A test-only data bundle; `load_split("test")` in a job, from that bundle, when unlocked
- [ ] The freeze record with checks F1 to F8 and the estimate; the owner's unlock
- [ ] `scripts/comparison_freeze.py` (`check`, `freeze`, `status`, `unlock`)
- [ ] Launcher: the test bundle for a test job, check J11, `rehearse` refuses `test`
- [ ] The six-arm record with checks C1 to C8, voiding a run, the preflight
- [ ] `scripts/comparison_runs.py` (`preflight`, `status`, `record`, `void`)
- [ ] Guard tests: who may ask for `test`; frozen values against a committed freeze record
- [ ] Runbook, run sheet, `coding.md`, `splits/README.md`, references, gotchas
- [ ] Lint and tests, with and without the training stack
- [ ] Spec section 11, roadmap, execution plan, implementation summary
- [ ] Owner: freeze, commit, unlock, commit, preflight, four Nova runs, two SageMaker jobs,
      record

## Summary

The test split has been locked since the split was frozen. This version unlocks it once and
asks all six arms the 2,898 test questions.

First a freeze record is written: per arm, the predictor name, model id, and settings of
its dev result run, with the hashes of the shared config files and the gold flags. The
owner then unlocks by typing the freeze record's hash, which is the sign-off. The unlock
record names the freeze record and the commit that holds it.

After the unlock the lock no longer guards anything, so a gate inside the runner takes
over. A test run needs the frozen configuration, scope `measured`, no `--limit`, the right
place (a SageMaker job for the Qwen arms, Bedrock for the Nova arms), and no earlier
complete run for its arm. An interrupted run is resumed. A rerun needs the owner to void
the earlier run with a written reason, which the final record shows as a deviation.

The two Qwen arms run as two SageMaker jobs on a bundle that holds test rows only: the
single upload of test rows to S3. A new launch check, J11, applies the gate at plan time.
The four Nova arms run through `scripts.evaluate`.

Last, a six-arm record is written when all six runs are complete and comparable. It holds
counts, tokens, and billable seconds; no statistics and no dollars. Later versions take
their run ids from it.

Two things differ from the spec. The six dev runs do not share config file bytes (two were
made before a comment changed), so the freeze ties them to today's files by rescoring. And
the two records the gate reads live under `splits/`, because a job's code does not include
`reports/`.

The owner started the PRP without changing the spec's six decisions, so they are taken as
recommended. The agent does not unlock and reads no test row.

## Confidence score

**7/10.** Every building block exists and is tested. The gate sits in the shared runner,
the freeze reads records made by four earlier versions, and a real test run cannot be
exercised during the build.

## Implementation Time Estimate

**6–7.5 hours** of code and documents.

| Task | Time |
|---|---|
| Config | 25 min |
| Gate and runner | 80 min |
| Bundle and loader | 40 min |
| Freeze and unlock | 100 min |
| Freeze script | 30 min |
| Launcher, job, evaluate | 70 min |
| Six-arm record and its script | 80 min |
| Guards | 20 min |
| Documents | 35 min |
| Validation and status documents | 30 min |

The owner's runs: about 7 hours of waiting, most of it Nova Pro at 18 requests a minute;
about $6.50.

## Time Taken for PRP Generation

**~35 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 5 min |
| Codebase exploration | 18 min |
| Research existing PRPs and patterns | 4 min |
| Writing PRP document | 8 min |
