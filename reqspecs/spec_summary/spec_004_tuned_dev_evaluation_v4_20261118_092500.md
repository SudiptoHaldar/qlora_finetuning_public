# Spec summary: tuned-model dev evaluation (req_004 v4.04)

**PRP**: [prp_004_tuned_dev_evaluation_v4.md](../PRPs/prp_004_tuned_dev_evaluation_v4.md)  
**Spec**: `reqspecs/reqs/req_004_frontier_tuning/req_004_04_spec.md`  
**Generated**: 2026-11-18  

## Todo list

- [ ] Config v4: the `tuned_evaluation` block (splits, verdict split, the two thresholds)
- [ ] Predictor description: optional `usage_model_id` and `details`; the runner records
      usage under the base model; `run_predictions`
- [ ] `tuned.py`: which attempt is this experiment's B1, with every refusal
- [ ] `deployment_status` in `bedrock.py`: one read before the first question
- [ ] `nova-lite-tuned` and `nova-pro-tuned` predictors and their `evaluate` entries
- [ ] `tuned_report.py`: pairs, sign test, cost, verdict, checks V1 to V6, Markdown and JSON
- [ ] `scripts/nova_tuned.py`
- [ ] Runbook, run sheet stage G, console primer, references, gotchas, `coding.md`
- [ ] Lint and tests, with and without the training stack
- [ ] Spec section 11, roadmap, execution plan, implementation summary
- [ ] Owner, after v4.03: trial, quick-dev, dev for each tuned model; then the report

## Summary

The tuned predictors are the existing Nova predictor with another `modelId`: the ARN of the
deployment that the tuning record names for the model's latest attempt. Before a run
starts, the attempt must show a completed job and an Active deployment, it must have been
tuned on the current export for today's prompt and split, and Bedrock is asked once whether
the deployment is really Active. Each refusal names the command to run. Usage is recorded
under the base model's id, because that is what the bill and the cap check price by, while
the run's description names the deployment.

`scripts.nova_tuned` takes the tuned runs and their baseline runs. It refuses a pair unless
prompt, split, settings, and example order are equal, then reports both scores, the
difference in points, how many questions tuning fixed and broke with an exact sign test,
the cost per example and per correct answer, the training cost beside them, and a verdict
per model from dev only. It calls no AWS service.

The owner decided the spec's three questions on 2026-11-18: missed means less than 10
points above the model's own baseline on dev execution accuracy, or format validity under
98%; quick-dev and dev are both scored; the deployments are kept until v5.01. The agent
asks no tuned model anything, and none exists yet.

## Confidence score

**8/10.** Everything it builds on exists and is tested. Nothing can be tried against a real
deployment, and one small change reaches the shared runner.

## Implementation Time Estimate

**3.5–4.5 hours** of code and documents.

| Task | Time |
|---|---|
| Config | 20 min |
| Predictor description and runner | 25 min |
| Tuned identity and the deployment read | 35 min |
| Tuned predictors and `evaluate` entries | 40 min |
| Report module | 70 min |
| Script | 30 min |
| Documents | 25 min |
| Validation and status documents | 25 min |

The owner's runs: about an hour once both deployments are Active, about $0.50.

## Time Taken for PRP Generation

**~25 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 4 min |
| Codebase exploration | 12 min |
| Research existing PRPs and patterns | 3 min |
| Writing PRP document | 6 min |
