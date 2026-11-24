# Spec summary: comparison report (req_005 v5.03)

**PRP**: [prp_005_comparison_report_v3.md](../PRPs/prp_005_comparison_report_v3.md)  
**Spec**: `reqspecs/reqs/req_005_comparison/req_005_03_spec.md`  
**Created**: 2026-11-23  

## Todo list

- [ ] `config/report.yaml` and its loader
- [ ] Statistics: the paired bootstrap by example and by company, McNemar's exact test, Holm's method
- [ ] Outcomes: per example and arm, the four scores, the error category, the gold steps, the company
- [ ] Checks R1 to R7 and the statistics record, in JSON and Markdown
- [ ] The accuracy-versus-cost chart as SVG
- [ ] The report from a template that holds no number; the generated findings
- [ ] Check R8; the README block; the local file of example text
- [ ] `scripts/comparison_report.py` (`check`, `build`, `examples`)
- [ ] The deviations list, drafted from the version logs
- [ ] Tests and guards; runbook, run sheet, pattern file, coding notes
- [ ] The real build; spec section 11, roadmap, execution plan
- [ ] Owner: read the report and the deviations list; rebuild after the bill is entered; commit (M8)

## Summary

The report is built from the six stored test runs, the six-arm record, and the cost record.
It calls no AWS service and runs no model. It adds a confidence interval for every accuracy
figure (a paired bootstrap, by example and by company), a paired exact test for six named
pairs of arms with Holm's adjustment, and the thirteen error categories for all six arms.

The owner took all nine decisions of the spec as recommended. A difference whose company
interval includes zero is called "not settled". Committed files hold counts and example ids
only. The report names no single winner. It is built now on the provisional cost record and
rebuilt after the bill is entered; that commit is milestone M8.

Counted by hand before the build: A1 against tuned Nova Pro, 163 questions only A1 right
and 221 only Nova Pro (p 0.0036); A1 against tuned Nova Lite, 238 and 186 (p 0.013).

## Confidence score

8/10. The inputs are verified and the pairs have hand figures. The risks are the branches
of the generated findings, the rescored counts having to equal the six-arm record, and the
report's prose, which needs reading.

## Implementation Time Estimate

5–6 hours of code and documents, then the owner's reading and a rebuild after the bill.

## Time Taken for PRP Generation

About 4 minutes by the clock (23:29 to 23:33): 1 reading the spec and guidelines, 1.5
exploring the code and the records, 0.5 on earlier PRPs and patterns, 1 writing. The spec
had been written in the same session.
