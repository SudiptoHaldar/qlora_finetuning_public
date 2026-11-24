# Spec summary: DSL parser and safe executor (req_002 v2.01)

**PRP**: [prp_002_dsl_parser_executor_v1.md](../PRPs/prp_002_dsl_parser_executor_v1.md)  
**Origin**: `reqspecs/reqs/req_002_dsl_evaluation/req_002_01_spec.md`  
**Created**: 2026-11-15  

## Todo list

1. Write `config/dsl.yaml` and the `errors`, `config`, and `language` modules.
2. Write the reply parser (`reply.py`) and its tests.
3. Write the program parser (`program.py`) with the canonical form, and its tests.
4. Write the executor (`executor.py`) with the table operations, and its tests.
5. Write `run_reply` (`outcome.py`), the safety and fuzz tests, and the language-table test.
6. Write the real-data test; check the 4,945 / 438 figures and the cell rule by hand.
7. Update the runbook, references, `coding.md`, and plan §4.4.
8. Validate with and without the training stack; update the roadmap, plan, and summaries.

## Summary

A model's reply is read in three steps, each with named errors: the reply parser finds the
program and the answer and says whether the two-line format was kept exactly; the program
parser checks every step against ten allowed operations, seventeen constants, and the
argument forms the dataset uses; the executor runs the steps on floats, reading table rows by
name, and returns a number or `yes` / `no` without rounding. `run_reply` joins the three and
never raises on any string. No reply text is ever evaluated as Python.

The owner accepted the spec's three recommendations by proceeding: a reply that breaks the
format is still read, with a flag; a repeated row name means the last row; a constant outside
the list is an error.

## Confidence score

9/10. The rules reproduced every gold result of fit and dev in a trial run before the spec.

## Implementation Time Estimate

3–4 hours.

| Task | Time |
|---|---|
| Config, errors, language | 25 min |
| Reply parser and tests | 30 min |
| Program parser and tests | 50 min |
| Executor and tests | 40 min |
| `run_reply`, safety, fuzz, language tests | 30 min |
| Real-data test and hand check | 15 min |
| Documents | 25 min |
| Validation and status documents | 15–30 min |

## Time Taken for PRP Generation

~8 minutes.

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 2 min |
| Research (existing patterns) | 1 min |
| Writing PRP document | 3 min |
