# PRP: Qwen backend and zero-shot baseline (req_003 v3.01)

**Feature Name**: qwen_backend_baseline (qlora_local v3.01)  
**Version**: 1  
**Created**: 2026-11-18  
**Status**: ✅ IMPLEMENTED  
**Implemented**: 2026-11-19  
**Implementation Summary**: [impl_003_20261119_005334.md](../impl_summary/impl_003_20261119_005334.md)  
**Confidence Score**: 8/10  
**Estimated Time**: 4–5 hours of code and documents, plus unattended GPU time for the runs  
**Origin**: `reqspecs/reqs/req_003_qlora_local/req_003_01_spec.md`  
**Template**: `prp_lean.md` (expanded, medium complexity)  

---

## Goal

Let the evaluation runner ask `Qwen/Qwen3-4B`, loaded in 4-bit on the local GPU, for replies,
so that arm A0 can be scored with `python -m scripts.evaluate --predictor qwen3-4b`, and
record the A0 baseline on quick-dev and dev with peak GPU memory and seconds per example.

## Why & What

- **User value**: Path A gets its untuned reference point beside B0-Pro and B0-Lite
  (milestone M4), and the first measured memory and speed of the 4B model, which v3.02 and
  v3.03 need before any billed GPU job.
- **Technical**: one config file, a renderer, a 4-bit backend with an optional adapter, one
  predictor class, an optional `measurements()` hook in the runner, one entry in
  `scripts/evaluate.py`, tests with a stub backend, `gpu` tests on the 0.6B model.

The owner started this PRP and its execution without changing the three questions of spec
section 8, so they are **taken as recommended**: (1) a reply ends at the model's end token
or the reply limit only, with no cut after the `ANSWER` line; (2) batch size 8, after the
agreement between 8 and 1 is measured on 40 examples; (3) A0 is scored on dev as well as
quick-dev. By spec D30 the agent downloads the weights and runs the local GPU runs. Nothing
is billed and no AWS service is used.

## Success Criteria

Code:

- [x] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass,
      with and without the training stack blocked.
- [x] A test shows the rendered prompt holds the request's texts character for character
      and equals `token_budget.render_prompt` for the same record.
- [x] `--predictor gold` prints what it printed before, and its summary has no new key.
- [x] With the weights absent, `--predictor qwen3-4b` exits 2 with the download command and
      leaves no run.

On this machine (free):

- [x] `venv\Scripts\python -m pytest -m gpu -q` passes.
- [x] The five-example trial completes as a partial run and prints its measurements.
- [x] 40 examples at batch size 1 and at 8 are compared reply by reply: 16 differ, so the
      baseline ran at batch size 1 (spec section 11).
- [x] A0 on quick-dev and on dev: `RESULT: complete`, R1 to R5 pass, **input tokens 128,495
      and 317,882**, no thinking block in any reply, peak GPU memory under 16 GB.
- [x] Spec section 11 "As built", the roadmap, and milestone M4 are written.

---

## Context

### References

```yaml
refs:
  - reqspecs/references/project_patterns/evaluation_run.md   # what a backend implements
  - reqspecs/references/project_patterns/canonical_record.md # texts unchanged; reuse checklist
  - reqspecs/references/project_patterns/run_tracking.md     # the runner records the usage
  - reqspecs/references/tech/hf_training_stack.md            # the calls that work on this machine
  - reqspecs/references/common/gotchas.md                    # section 1 and section 4

code:
  - file: qlora_finetuning/evaluation/predictor.py
    why: Predictor, Prediction, PredictorError, describe(); optional attributes are read with getattr
  - file: qlora_finetuning/evaluation/runner.py
    why: the finally block that records usage is where measurements.json is written;
         summarize_run builds the summary; RUN_CONFIGS; RunnerContext
  - file: scripts/evaluate.py
    why: PREDICTORS, PredictorEntry(factory, kind, path, arm, where, configs); print_summary
  - file: qlora_finetuning/frontier/predictor.py and frontier/config.py
    why: the shape to mirror: a predictor over an injected backend, a factory that checks
         everything before a run exists, a typed config with a `settings` mapping
  - file: qlora_finetuning/data/token_budget.py
    why: load_token_budget_config (model id, commit, fingerprint, thinking_off_suffix,
         max_new_tokens), load_tokenizer, tokenizer_fingerprint, render_prompt, count_split
  - file: scripts/env_check.py (smoke_step)
    why: the 4-bit load and the generate call that already run on this machine
  - file: tests/evaluation/conftest.py, tests/data/synthetic.py
    why: make_context(tmp_path), evaluate(), FakeTokenizer, fake_budget
  - file: tests/frontier/test_predictor.py (guards), tests/scripts/test_evaluate.py
    why: the import-guard tests and the stub-factory tests to mirror

docs:
  - https://huggingface.co/docs/transformers/main_classes/text_generation   # GenerationConfig
  - https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
  - https://huggingface.co/docs/peft/package_reference/peft_model           # PeftModel.from_pretrained
  - https://huggingface.co/Qwen/Qwen3-4B                                    # model card
```

### Verified on this machine (2026-11-18)

Installed: transformers 5.23.0, peft 0.21.3, huggingface_hub 2.4.0, torch 2.14.1+cu126.

```python
# All four accept None; do_sample=False then gives plain greedy decoding with no warning.
GenerationConfig(do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None)
# generate(inputs, generation_config=..., **kwargs): max_new_tokens goes in kwargs.
# tokenizer.decode(token_ids, skip_special_tokens=..., **kwargs)
# PeftModel.from_pretrained(model, model_id, adapter_name=..., is_trainable=False, ...)
# huggingface_hub.try_to_load_from_cache(repo_id, filename, revision=...) -> path or None
#   Qwen3-4B before the download: config.json -> a path; model.safetensors.index.json -> None
# model.get_memory_footprint() exists.
```

Tokenizer of the pinned commit: `padding_side` is `right`; `eos` `<|im_end|>` 151645; `pad`
`<|endoftext|>` 151643; `<think>` 151667 and `</think>` 151668 are **not** special, so
`skip_special_tokens=True` leaves them; the thinking-off prompt ends
`<|im_start|>assistant\n<think>\n\n</think>\n\n`; tokenizing the rendered text with
`add_special_tokens=False` gives the ids of `apply_chat_template(tokenize=True)`.

Model: `generation_config.json` has `do_sample: true`, temperature 0.6, top-k 20, top-p 0.95,
`eos_token_id: [151645, 151643]`. Weights are three shards listed in
`model.safetensors.index.json` (8.04 GB). The 0.6B smoke model may ship one
`model.safetensors` and no index: the cache check must handle both.

### Gotchas

```python
# CRITICAL: only local_qlora/backend.py may import torch, transformers, peft, bitsandbytes,
#   huggingface_hub, and only inside functions. CI has none of them.
# CRITICAL: the factory loads the model BEFORE run_evaluation is called (entry.build is an
#   argument of the call), so a ConfigError there is exit 2 with no run and no ledger line.
# CRITICAL: set tokenizer.padding_side = "left" on the backend's own tokenizer object, and
#   pass attention_mask. Never truncate: no truncation argument, no max_length.
# CRITICAL: input_tokens is the row's own length (attention_mask.sum), not the padded width.
# GOTCHA: a row that ended early is padded with <|endoftext|>, which is itself an end token;
#   cut at the FIRST end token. A row with no end token ran to the limit.
# GOTCHA: decode the ids before the end token; then strip one complete leading
#   <think>...</think> block by hand, because decode leaves the markers in.
# GOTCHA: the runner's resume check ignores batch_size, so it is in settings.
# GOTCHA: environment_report() imports torch; tests that start a run pass environment={}
#   (make_context does).
# GOTCHA: scripts print ASCII only (cp1252 console).
# GOTCHA: never write prompt or reply text into a PredictorError, a log line above DEBUG,
#   a measurement, or any committed file.
# GOTCHA: a local run appends to the committed ledger/ files; that is expected.
```

---

## Implementation

### Files

```yaml
CREATE:
  - config/local_qlora.yaml
  - qlora_finetuning/local_qlora/config.py     # load_local_qlora_config, LocalQloraConfig
  - qlora_finetuning/local_qlora/render.py     # render_chat, read_generation, Generation
  - qlora_finetuning/local_qlora/backend.py    # check_ready, load_backend, QwenBackend
  - qlora_finetuning/local_qlora/predictor.py  # QwenPredictor, qwen3_4b
  - tests/local_qlora/conftest.py              # stand-in tokenizer with decode, StubBackend
  - tests/local_qlora/test_config.py, test_render.py, test_predictor.py, test_backend.py
  - tests/local_qlora/test_backend_gpu.py      # pytestmark = pytest.mark.gpu

MODIFY:
  - qlora_finetuning/evaluation/runner.py      # measurements.json; "backend" in the summary
  - scripts/evaluate.py                        # the qwen3-4b entry; one measurements line
  - tests/evaluation/test_runner.py, tests/scripts/test_evaluate.py
  - docs/runbook.md (section 5), reqspecs/references/tech/hf_training_stack.md,
    reqspecs/references/common/gotchas.md, reqspecs/references/project_patterns/evaluation_run.md,
    reqspecs/1. PROMPTS/coding.md, architecture/architecture_overview.md
  - req_003_01_spec.md (section 11), req_003_roadmap.md, execution_plan.md (status, M4, P3 check)
```

### Tasks, in order

```yaml
Task 1 - config:
  - config/local_qlora.yaml as in spec 4.1.
  - load_local_qlora_config(config_dir=None): required keys named in a ConfigError;
    refuse batch_size < 1 or not a whole number, do_sample not false, num_beams != 1,
    quantization.type not in (nf4, fp4), compute_dtype not in (bfloat16, float16),
    end_tokens not a non-empty list of text.
  - cfg.settings(revision, adapter=None): the mapping of spec D25.

Task 2 - render.py (no training-stack import; the tokenizer is an argument):
  - render_chat(system, user, tokenizer, thinking_off_suffix): apply_chat_template over a
    system and a user message, tokenize=False, add_generation_prompt=True,
    enable_thinking=False; raise RenderError if the text does not end with the suffix.
  - read_generation(token_ids, tokenizer, end_ids, limit) -> Generation(text,
    output_tokens, stop_reason, think_removed): spec D15 to D17.

Task 3 - backend.py:
  - check_ready(budget, model_id, revision): stack importable; weights in the cache (index
    plus every shard it lists, or a single model.safetensors); tokenizer loads from the
    cache and has the pinned fingerprint; CUDA available. Each failure is a ConfigError
    with the command to run. The download command is
    venv\Scripts\hf download <model> --revision <commit>.
  - load_backend(cfg, budget, model_id=None, revision=None, adapter_dir=None): check_ready,
    reset the peak-memory counter, load tokenizer and 4-bit model with
    local_files_only=True, attach the adapter unmerged, eval mode, resolve the end-token
    ids, render a probe prompt to prove the thinking-off suffix, time the load.
  - QwenBackend.generate(prompts, max_new_tokens) and .measurements() (spec D23).

Task 4 - predictor.py:
  - QwenPredictor(backend, name, settings): model_id from the backend, batch_size from
    settings, usage_source "measured"; predict() turns any backend error into a
    PredictorError naming the error type and the batch size; latency is the batch's
    seconds over its size; measurements() passes the backend's on.
  - qwen3_4b(split, context): the factory.

Task 5 - runner and script:
  - runner: in the finally block, after the usage record, write measurements.json when the
    predictor has a callable measurements(); failures are logged, never raised.
    summarize_run adds "backend" only when the file exists.
  - evaluate.py: the entry, a "measurements:" line when the summary has a backend block,
    the docstring.

Task 6 - tests (spec 4.4), then lint and the full suite, with and without the stack.

Task 7 - the local runs of spec section 9, in its order. The two 40-example runs differ
  only in inference.batch_size, edited for the first and restored for the second; compare
  the two predictions files reply by reply with a throwaway script.

Task 8 - documents (spec 4.5), spec section 11, roadmap, execution plan, runbook,
  implementation summary.
```

### Pseudocode

```python
# render.py
THINK = re.compile(r"\s*<think>.*?</think>\s*", re.DOTALL)

def read_generation(token_ids, tokenizer, end_ids, limit):
    ids = list(token_ids)[:limit]
    cut = next((i for i, token in enumerate(ids) if token in end_ids), None)
    body = ids if cut is None else ids[:cut]
    text = tokenizer.decode(body, skip_special_tokens=True)
    match = THINK.match(text)                 # only at the very start, only if closed
    if match: text = text[match.end():]
    return Generation(text, len(body) + (cut is not None),
                      END if cut is not None else LENGTH, bool(match))

# backend.py: every heavy import is inside the function that needs it
def generate(self, prompts, max_new_tokens):
    import torch
    texts = [render_chat(system, user, self.tokenizer, self.suffix) for system, user in prompts]
    batch = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    lengths = batch["attention_mask"].sum(dim=1).tolist()
    started = time.perf_counter()
    with torch.inference_mode():
        out = self.model.generate(**batch.to(self.model.device),
                                  generation_config=self.generation_config,
                                  max_new_tokens=max_new_tokens)
    new = out[:, batch["input_ids"].shape[1]:].tolist()      # tolist waits for the GPU
    seconds = time.perf_counter() - started
    return [Generated(read_generation(row, ...), input_tokens=n, seconds=seconds) ...]

# runner.py, inside finally, after run.record(**usage)
notes = _measurements(predictor)             # None if absent, not callable, raises, not plain
if notes is not None:
    _write_json(run.run_dir / MEASUREMENTS_NAME, notes)
    run.outputs["measurements"] = MEASUREMENTS_NAME
```

### Error handling

| Where | What | Becomes |
|---|---|---|
| Factory, before the run | No stack, no CUDA, weights or tokenizer not cached, wrong fingerprint, a template without the thinking-off suffix, an unknown end token | `ConfigError` with the command; exit 2, no run |
| `predict` | Any exception from the backend (out of memory, CUDA) | `PredictorError`: error type and batch size, no text; out of memory adds "lower inference.batch_size and start a new run" |
| Runner | `measurements()` raises or is not plain values | A warning in the log; the run is unaffected |

---

## Validation

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest tests/local_qlora tests/evaluation tests/scripts/test_evaluate.py -q
venv\Scripts\python -m pytest -q
venv\Scripts\python -m pytest -m gpu -q
venv\Scripts\python -m scripts.evaluate --predictor gold --split quick_dev --limit 3      # as before
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b --split quick_dev --limit 5
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b --split quick_dev
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b --split dev
```

The suite is also run once with the training stack blocked (a throwaway pytest plugin that
refuses to import the ten libraries), as earlier versions did.

## Final Checklist

- [x] Tests and lint green, with and without the stack; `gpu` tests green
- [x] Guards: only `backend.py` names the stack under `local_qlora/`; nothing there imports `frontier`
- [x] The gold predictor's output and summary are unchanged
- [x] A0 runs are results; input tokens equal the token budget's; figures written into the spec
- [x] Runbook section 5, tech note, gotchas, `evaluation_run.md`, `coding.md`, architecture
- [x] Roadmap, execution plan (status, M4, the reworded P3 check), implementation summary

---

## Time Estimate

**Total**: 4–5 hours of work, plus GPU time that runs unattended (not known until the trial).

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

**Risks**: the 4B model not fitting or loading on native Windows (+1 hour: lower the batch
size, then report); batched replies differing from single ones in many examples (the
baseline is then run at batch size 1 and the question goes back to the owner); a slow
download.

## Anti-Patterns to Avoid

- ❌ No stop string and no cut after the `ANSWER` line
- ❌ No truncation of a prompt, and no trimming of a reply beyond spec D17
- ❌ No loop over a split, predictions file, or accuracy outside the runner
- ❌ No training-stack import at module level, and none outside `backend.py`
- ❌ No download inside a run: `local_files_only=True` everywhere
- ❌ No test outside the `gpu` marker that loads a model

## Time Taken for PRP Generation

**~15 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 6 min |
| Research existing PRPs and patterns, installed library APIs | 3 min |
| Writing PRP document | 4 min |
