# Spec summary: evaluation runner (req_002 v2.03)

**PRP**: [prp_002_evaluation_runner_v3.md](../PRPs/prp_002_evaluation_runner_v3.md)  
**Spec**: `reqspecs/reqs/req_002_dsl_evaluation/req_002_03_spec.md`  
**Generated**: 2026-11-16  

## Todo list

- [ ] Config keys and `metrics.py` fields; update the metrics tests
- [ ] `predictor.py` and `fake.py`, with tests
- [ ] `predictions.py`, with tests
- [ ] `runner.py`: the run, the summary, checks R1 to R5
- [ ] Runner and resume tests
- [ ] `scripts/evaluate.py` and its tests
- [ ] Real runs on quick-dev and dev with the gold predictor
- [ ] Runbook, project-pattern note, `coding.md`, gotchas, architecture, ledger README
- [ ] Lint and tests, with and without the training stack
- [ ] Spec, roadmap, execution plan (M3), implementation summary

## Summary

A predictor takes a list of requests (the canonical record's two texts and the reply limit)
and returns a list of predictions (reply text, token counts, latency, stop reason). The
runner predicts a split in dataset order, in batches, and writes each batch's lines to a
predictions file before the next batch starts. One invocation is one run in the registry
with one ledger record for what it consumed, written whether the run completes or fails.

A run cannot be reopened, so resuming is a new run that copies the valid lines of an earlier
file and continues. It is refused unless the split, the prompt, the reply limit, and the
predictor's settings are the same.

The summary is always computed from the file: stored outcomes are scored again under the
current config. It gives the totals of v2.02, execution accuracy under the tight rule as a
diagnostic, token totals, latency, and five checks. Only a run that passes them and is not
partial is a result, and it adds one line of counts to the committed
`ledger/evaluations.jsonl`.

The owner started the PRP and its execution without changing the spec's three questions, so
they are taken as recommended: a predictor failure stops the run and it is resumed; the
tight-rule figure is shown beside the 0.5% headline; one line of counts per complete run is
committed.

## Confidence score

8/10. The pieces the runner calls exist and the loop was tried on dev. Resuming has many
edge cases, and an import cycle between the evaluation package and the gold self-check must
be avoided on purpose.

## Implementation Time Estimate

4–5 hours.

| Task | Time |
|---|---|
| Config and metrics fields | 20 min |
| Predictor and fake | 35 min |
| Predictions file | 35 min |
| Runner | 60 min |
| Runner and resume tests | 55 min |
| Script and tests | 30 min |
| Real runs | 5 min |
| Documents | 30 min |
| Validation and status documents | 15–30 min |

## Time Taken for PRP Generation

~12 minutes.

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 5 min |
| Research (existing patterns) | 1 min |
| Writing PRP document | 4 min |
