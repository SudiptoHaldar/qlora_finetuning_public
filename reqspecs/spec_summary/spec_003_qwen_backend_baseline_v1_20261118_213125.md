# Spec summary: Qwen backend and zero-shot baseline (req_003 v3.01)

**PRP**: [prp_003_qwen_backend_baseline_v1.md](../PRPs/prp_003_qwen_backend_baseline_v1.md)  
**Spec**: `reqspecs/reqs/req_003_qlora_local/req_003_01_spec.md`  
**Generated**: 2026-11-18  

## Todo list

- [ ] `config/local_qlora.yaml` and its loader, with tests
- [ ] Renderer and reply reader (`render.py`), with tests
- [ ] 4-bit backend with an optional adapter (`backend.py`), readiness checks, with tests
- [ ] `QwenPredictor` and the `qwen3-4b` factory, with tests through the runner
- [ ] Runner: optional `measurements()`; `scripts/evaluate.py`: the entry and one line
- [ ] Lint and tests, with and without the training stack; `gpu` tests
- [ ] Download the weights; the trial; the batch comparison; A0 on quick-dev and dev
- [ ] Runbook section 5, tech note, gotchas, patterns, `coding.md`, architecture
- [ ] Spec section 11, roadmap, execution plan (M4), implementation summary

## Summary

A renderer wraps the canonical system and user texts in Qwen's chat template with thinking
off. A backend loads `Qwen/Qwen3-4B` at the pinned commit in 4-bit NF4 from the local cache
and generates greedily in left-padded batches; it stops at the model's own end tokens or
the reply limit and nowhere else. A predictor over it gives the runner one reply per
request with measured token counts. Everything that can fail before a reply (no training
stack, no GPU, weights not downloaded, a changed tokenizer) fails in the factory, so no
failed run is left behind.

The runner gains one optional hook: a predictor's `measurements()` is stored with the run,
which is how peak GPU memory is recorded.

The owner started the PRP without changing the spec's three questions, so they are taken as
recommended: end token only, batch size 8 after measuring its agreement with 1, and A0 on
dev as well as quick-dev. The agent downloads the weights (8.04 GB) and runs the local GPU
runs. Nothing is billed.

## Confidence score

8/10. The installed library calls and the tokenizer were inspected. The 4B model has never
been loaded on this machine: memory, speed, and how the untuned model behaves are found by
the build.

## Implementation Time Estimate

4–5 hours of work, plus unattended GPU time.

| Task | Time |
|---|---|
| Config and loader | 20 min |
| Renderer and reply reader | 30 min |
| Backend | 50 min |
| Predictor and factory | 25 min |
| Runner hook and script | 25 min |
| Tests | 70 min |
| Local runs and the batch comparison | 30 min of attention |
| Documents and status files | 40 min |

## Time Taken for PRP Generation

~15 minutes.

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 6 min |
| Research existing PRPs and patterns, installed library APIs | 3 min |
| Writing PRP document | 4 min |
