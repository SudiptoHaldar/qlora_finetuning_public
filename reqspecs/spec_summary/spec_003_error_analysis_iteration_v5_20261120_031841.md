# Spec summary: error analysis and one corrective iteration (req_003 v3.05)

**PRP**: [prp_003_error_analysis_iteration_v5.md](../PRPs/prp_003_error_analysis_iteration_v5.md)  
**Spec**: `reqspecs/reqs/req_003_qlora_local/req_003_05_spec.md`  
**Created**: 2026-11-20  

## Todo list

- [ ] Config: `selection.noise_examples`, the `iteration` block; the loader
- [ ] `evaluation/failures.py`: one category per wrong reply; `runner.run_gold`
- [ ] `selection.py`: gathering split out; a second run compared with the selected one; schema 2
- [ ] `analysis.py` and `scripts/qlora_analyze.py`: the report, the finding, `--check`, `--show`
- [ ] `iteration.py` and `scripts/qlora_iterate.py`: I1 to I5, `propose`, `status`
- [ ] Launch check J10 in `sagemaker.py`
- [ ] Tests; the counts of checks from nine to ten
- [ ] Rehearsals on this machine: the analysis on a local smoke run; `propose` refused; `plan train`
- [ ] Documents; spec section 11 (the code); roadmap; execution plan; implementation summary
- [ ] Owner, after v3.04's selection: `qlora_analyze --run <first run>`; and only on `missed`: one edit
      of `config/training.yaml`, `qlora_iterate propose`, the second training and scoring jobs,
      `qlora_select --run <second run>`, `qlora_iterate status`

## Summary

v3.05 is conditional, and its trigger cannot be read yet: no cloud job has run, so there is
no verdict. This PRP builds both of its parts all the same.

The **error analysis** is free and is needed by v5.03 whatever the verdict. A classifier
gives each wrong reply one category from the stored predictions, with no model: cut off, no
program, bad program, not executable, kind, more or fewer steps, another operation, other
numbers, arguments reversed, scale, sign. `qlora_analyze` writes a committed report of
counts and example ids for the selected checkpoint, the untuned model, every checkpoint,
and where the failures sit; it never writes a reply.

The **corrective iteration** runs only on a `missed` verdict and only once. The owner
changes one of five training settings, `qlora_iterate propose` checks that it is exactly
one and writes it down before anything is paid for, and a tenth launch check refuses a
second full run without that record and any third run. The second run is scored on the
first job's workspace and replaces the first only if it is better by the rule that chose
the first; a tie keeps the first.

The agent launches nothing. The decisions of spec section 8 are taken as recommended.

## Confidence score

**7/10.** No AWS, no model, and the stand-in world of v3.04's tests exists. The open part is
the selection record's states once two runs exist.

## Implementation Time Estimate

6–8 hours of code and documents, no GPU time; then the owner's steps, which wait for
v3.04's verdict.

| Task | Time |
|---|---|
| Config and loader | 20 min |
| Categories | 45 min |
| Selection across runs | 75 min |
| Analysis and its script | 75 min |
| Iteration and its script | 60 min |
| J10 | 30 min |
| Tests | 120 min |
| Rehearsals | 15 min |
| Documents | 45 min |

## Time Taken for PRP Generation

**~15 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 6 min |
| Research existing PRPs and patterns | 2 min |
| Writing PRP document | 5 min |
