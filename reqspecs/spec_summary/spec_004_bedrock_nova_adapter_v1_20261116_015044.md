# Spec summary: Bedrock Nova adapter and zero-shot baselines (req_004 v4.01)

**PRP**: [prp_004_bedrock_nova_adapter_v1.md](../PRPs/prp_004_bedrock_nova_adapter_v1.md)  
**Spec**: `reqspecs/reqs/req_004_frontier_tuning/req_004_01_spec.md`  
**Generated**: 2026-11-16  

## Todo list

- [ ] `config/frontier.yaml`, `config/pricing.yaml`, and their loaders, with tests
- [ ] `FrontierProvider` and `BedrockProvider`, with stub-client tests
- [ ] `FrontierPredictor` and the `nova-pro` / `nova-lite` factories, with tests through the runner
- [ ] `scripts/evaluate.py`: `PredictorEntry.where` and `.configs`; the two Nova entries
- [ ] `scripts/aws_check.py`: check C8
- [ ] Prices, the baseline report, `scripts/nova_baseline.py`, with tests
- [ ] Runbook section 6, tech note, gotchas, patterns, `coding.md`, architecture
- [ ] Lint and tests
- [ ] Spec, roadmap, execution plan (🟡), implementation summary
- [ ] Owner: sign in, `aws_check`, the trial, four baseline runs, the report

## Summary

A provider with one method, `converse`, wraps the Bedrock Converse API: the canonical
system and user texts unchanged, greedy decoding (`topK: 1`, temperature 0), nothing else
sent. A predictor over it gives the evaluation runner one reply per request with Bedrock's
own token counts, so the runner's ledger record carries them with source `aws_api`.
Retrying is botocore's; when it gives up the run stops and is resumed.

Before a run exists, the factory checks the sign-in and the account, so a missing sign-in
leaves no failed run behind. `aws_check` gains C8: both models active and callable on
demand. `config/pricing.yaml` records the prices of the list published 2026-11-11, and a
report script turns four baseline runs (two models, quick-dev and dev) into a committed
report with scores, tokens, cost at list price, and the Nova to Qwen token ratio.

The owner decided the spec's three questions on 2026-11-16: the owner runs the billed
calls, B0 is scored on dev as well as quick-dev, and the in-region model ids are used. The
agent therefore builds and tests the code and makes no call to Bedrock.

## Confidence score

8/10. Both interfaces were read (the runner's protocol, the Converse definition in the
installed boto3). Nothing could be tried against the live account: whether `temperature: 0`
is accepted and what a real response looks like are settled by the owner's trial run.

## Implementation Time Estimate

4–5 hours of code, then the owner's six commands.

| Task | Time |
|---|---|
| Config files and loaders | 30 min |
| Provider and Bedrock implementation | 60 min |
| Predictor | 35 min |
| Evaluate script | 20 min |
| `aws_check` C8 | 20 min |
| Prices, report, report script | 60 min |
| Documents | 30 min |
| Validation and status documents | 20–40 min |

## Time Taken for PRP Generation

~20 minutes.

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 3 min |
| Codebase exploration | 9 min |
| Research (existing patterns) | 3 min |
| Writing PRP document | 5 min |
