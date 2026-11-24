# Spec summary: training file export and budget gate (req_004 v4.02)

**PRP**: [prp_004_training_export_gate_v2.md](../PRPs/prp_004_training_export_gate_v2.md)  
**Spec**: `reqspecs/reqs/req_004_frontier_tuning/req_004_02_spec.md`  
**Generated**: 2026-11-16  

## Todo list

- [ ] Config keys and `FrontierConfig` fields; update the config tests
- [ ] `export.py`: the line, the two files, the id lists, with tests
- [ ] `export_validation.py`: checks X1 to X7 from the files on disk, each passing and failing
- [ ] `training_estimate.py`: the token ratio, the Path B estimate, checks G1 and G2, the gate record
- [ ] `staging.py`: upload with checksums, verified; the import guard
- [ ] `scripts/nova_export.py` and its tests
- [ ] The real run (the gate is expected to fail on G1 until the baselines exist)
- [ ] Runbook, references, `coding.md`, gotchas, architecture
- [ ] Lint and tests, with and without the training stack
- [ ] Spec, roadmap, execution plan, implementation summary
- [ ] Owner: v4.01 baselines, then the plain command (gate open), then `--upload`

## Summary

The fit split becomes the training file and the dev split the validation file, in Bedrock's
conversation format: one line per record with the canonical system, user, and target texts
unchanged, which is also exactly what the predictor sends at inference. Every record is
exported, the repeated ones too, so both paths train on the same examples.

A validator reads the files back from disk: same records in order, no test id, the format,
no reserved word (AWS's nine, any case, every text), the size limits, the current prompt
and split, and that a second rendering gives the same bytes.

The estimate covers all of Path B: what the ledger already holds, training for both models
(Qwen tokens times the measured Nova to Qwen ratio times the epochs, plus a 10% margin),
the evaluation runs still to come, and a month of storage. A committed gate record holds
the files' hashes, the estimate, the cap, and the verdict; v4.03 will start a job only
with a passing record for those very files. Without a measured ratio the gate fails.

`--upload` stages the files under `data/bedrock/<export id>/` with a SHA-256 checksum and
reads them back. The owner runs it.

The owner started the PRP and its execution without changing the spec's three questions,
so they are taken as recommended: 2 epochs, a $200 cap for Path B, dev as the validation
file.

## Confidence score

8/10. A trial export ran clean and every input exists. The staging code cannot be tried
against S3 by the agent, and the gate's passing path waits for the baselines.

## Implementation Time Estimate

4–5 hours of code, then the owner's upload.

| Task | Time |
|---|---|
| Config | 25 min |
| Export | 40 min |
| Validation | 60 min |
| Estimate and gate | 60 min |
| Staging and the guard | 25 min |
| Script | 35 min |
| Real run | 5 min |
| Documents | 25 min |
| Validation and status documents | 20–40 min |

## Time Taken for PRP Generation

~10 minutes.

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 4 min |
| Research (existing patterns) | 1 min |
| Writing PRP document | 3 min |
