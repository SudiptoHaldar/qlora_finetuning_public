# Spec summary: Metrics and gold self-check (req_002 v2.02)

**PRP**: [prp_002_metrics_gold_check_v2.md](../PRPs/prp_002_metrics_gold_check_v2.md)  
**Origin**: `reqspecs/reqs/req_002_dsl_evaluation/req_002_02_spec.md`  
**Created**: 2026-11-15  

## Todo list

1. Write `config/evaluation.yaml` and `evaluation/metrics.py`: the comparison rule, the
   answer reader, program match, mismatch labels, `score_reply`, `summarize`.
2. Write the metrics tests.
3. Write `data/gold_check.py`: check one example, build the report and the flags file.
4. Write `scripts/finqa_gold_check.py` (`--check`), extend the synthetic helper, write the
   tests; add the script to the raw-reader guard's list.
5. Run it over all 8,281 examples and reconcile fit and dev with spec section 2.
6. Update the runbook, references, `coding.md`, and plan §4.4.
7. Validate with and without the training stack; update the roadmap, plan, and summaries.

## Summary

Two numbers match when they differ by at most an absolute allowance or a relative one,
whichever is larger: gold answers are rounded to five decimal places and to about six
significant digits, so neither part works alone. Execution accuracy and answer accuracy both
use 0.00001 absolute and the guide's 0.5% relative (the owner's decision of 2026-11-16; the
spec first proposed the dataset's own precision, 0.001% relative, for execution accuracy). Program match compares numbers by value and
keeps the order of arguments and steps. Every rate is a count over all examples.

The gold self-check scores every gold reply with the same code. An example whose gold
program does not reproduce its gold answer is flagged in `splits/finqa_gold_flags.json`,
never dropped; totals are reported over all examples and over the unflagged ones.

The owner decided the spec's three questions on 2026-11-16: the guide's 0.5% for execution
accuracy (not the recommended tight allowance), flagged examples kept in every total, and
the self-check reading test rows now for counts and ids only.

## Confidence score

8/10. The rule was measured on fit and dev; the test split's gold programs have not been
executed before, and the answer reader is new.

## Implementation Time Estimate

3.5–4.5 hours.

| Task | Time |
|---|---|
| Config and `metrics.py` | 50 min |
| Metrics tests | 45 min |
| `gold_check.py` | 45 min |
| Script, synthetic helper, tests | 50 min |
| Real run | 10 min |
| Documents | 30 min |
| Validation and status documents | 15–30 min |

## Time Taken for PRP Generation

~9 minutes.

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 3 min |
| Research (existing patterns) | 1 min |
| Writing PRP document | 3 min |
