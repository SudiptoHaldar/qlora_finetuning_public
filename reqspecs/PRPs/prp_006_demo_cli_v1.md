# PRP: Demo CLI (req_006 v6.01)

**Feature Name**: demo_cli (packaging_demo v6.01)  
**Version**: 1  
**Created**: 2026-11-23  
**Status**: ✅ IMPLEMENTED  
**Implemented**: 2026-11-23  
**Implementation Summary**: [impl_006_20261123_174500.md](../impl_summary/impl_006_20261123_174500.md)  
**Confidence Score**: 9/10  
**Estimated Time**: 2.5–3.5 hours of code, tests, and documents; under 5 minutes of the local GPU; nothing billed  
**Origin**: `reqspecs/reqs/req_006_packaging_demo/req_006_01_spec.md`  

Template: `prp_lean.md` (medium: four small modules over existing pieces, one function
added to the metrics, one function split in the Path A predictor).

## Goal

`venv\Scripts\python -m qlora_finetuning.demo` takes a report excerpt, a table, and a
question from a JSON file, asks the Qwen model with the selected adapter on the local GPU,
and prints the proposed program, the executed result, and whether the model's own answer
agrees with it. With no argument it answers one invented sample.

## Why

- Both guides list a demo among the deliverables; phase P6 ends when the demo answers the
  bundled sample.
- v6.02's model card needs a sample that is not a FinQA row, and its clean-load check
  needs a function that turns a predictor and an input into a result.

## What

The five decisions of spec section 8 are **taken as recommended** (the owner started this
PRP): Qwen only; an over-length input is answered with a warning; a demo call is not
recorded; input is a JSON file; a checkpoint that is not on the machine stops the demo
with the pull command.

Success criteria

- [x] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass,
      with and without the training stack blocked.
- [x] `python -m qlora_finetuning.demo --check` passes on the real repository: the sample
      is valid and the selected checkpoint has the hash of the selection record.
- [x] On the GPU: the demo with no argument ends with exit code 0, the executed value
      matches the sample's `expected`, and the model's answer agrees.
- [x] On the GPU: `--untuned` ends without an exception.
- [x] `comparison_freeze status` reports no changed frozen value. No AWS call is made.
- [x] No run folder, registry line, or ledger line is written by a demo call.

## All Needed Context

| File | Why |
|---|---|
| `reqspecs/reqs/req_006_packaging_demo/req_006_01_spec.md` | D1 to D23, the output layout (4.3), the tests (4.4) |
| `reqspecs/references/project_patterns/canonical_record.md`, `reply_scoring.md` | The two conventions the demo follows; use their reuse checklists |
| `qlora_finetuning/data/serialization.py` | `load_prompt()`, `format_user(passages, table, question, prompt)`; `format_table` raises `SerializationError` on an empty table |
| `qlora_finetuning/local_qlora/predictor.py` | `build_tuned_predictor(adapter, context)`, `build_predictor()`, `QwenPredictor`; `context` needs `.prompt`, `.split_cfg` (None loads the real one), `.tracking_cfg` |
| `qlora_finetuning/local_qlora/adapters.py` | `resolve_adapter`, `check_adapter`, `selected_adapter`, `read_selection`, `adapter_sha256`, `SELECTED` |
| `qlora_finetuning/evaluation/predictor.py` | `PredictionRequest(example_id, system, user, max_new_tokens)`; `Prediction(text, input_tokens, output_tokens, latency_seconds, stop_reason)`; `PredictorError` |
| `qlora_finetuning/evaluation/metrics.py` | `read_answer(text)` gives `()`, one reading, or two for a `%` answer; `values_match(value, gold, allowance)`; `EvaluationConfig.answer` |
| `qlora_finetuning/dsl/outcome.py` | `run_reply(text, table)`, `ReplyOutcome` (`ok`, `value`, `step_values`, `program`, `answer_text`, `strict`, `deviations`, `notes`, `error_*`, `to_dict()`) |
| `scripts/qlora_select.py` | The shape of an entry point: `main(argv, **stand_ins)`, `CONFIG ERROR:` line, exit codes |
| `scripts/qlora_train.py` | `NOISY_LOGGERS`, set to WARNING after `setup_logging()` |
| `tests/local_qlora/conftest.py`, `test_adapters.py` | `StubBackend`, `MODEL_ID`, `REVISION`; `loaded` (replacing `load_backend`), `runner_context`, `trained` |
| `tests/local_qlora/test_predictor.py` (line 201 on) | The import guards to copy: an AST scan for the training stack, a subprocess that checks `sys.modules` |

Facts established for this PRP:

- The selected checkpoint, `20261120T182710Z-train-qwen3-4b-fit-4557/checkpoint-500`, is on
  this machine under `artifacts/runs/<run>/checkpoints/checkpoint-500/`.
- `build_tuned_predictor` does every check and then loads the model, in one function.
  `--check` (D23) needs the checks alone, so the function is split in two; nothing it
  refuses or returns changes.
- `Prediction.input_tokens` is the length of the chat-templated prompt. The length warning
  (D14) is `input_tokens + max_new_tokens > max_length`, known after the reply; no
  tokenizer is needed in the demo.
- `values_match` scales the relative allowance by its second argument. In `answer_agrees`
  the executed value is that argument, as the gold value is in `score_reply`.
- Code is not frozen, config is. Adding a function to `metrics.py` changes no frozen hash.
- `tests/test_package.py` already requires `qlora_finetuning.demo` to import and to name
  its spec group in its docstring.

Gotchas:

- The Windows console is cp1252: print ASCII only
  (`text.encode("ascii", "backslashreplace").decode("ascii")`, `ensure_ascii=True`).
- Nothing under `demo/` may import the training stack or boto3 at module level, and
  `qlora_finetuning.demo` itself must import nothing heavy: `__init__.py` stays a docstring.
- Import `local_qlora.predictor` inside the function that builds the predictor, as
  `scripts.evaluate` does, so `--check` and the tests of `inputs` and `render` stay light.
- `bool` is an `int` in Python: a table cell of `true` must be refused, not printed as 1.
- A float is printed with `repr`, never formatted or rounded (D17).
- **Another session is building v5.02 in this working tree** (`comparison/cost*.py`,
  `config/cost_model.yaml`, `scripts/comparison_cost.py`). Do not touch those files, edit
  shared documents with small targeted edits, and do not count its unfinished tests as
  this version's failures.
- The agent never passes a FinQA row to the demo: invented input only.

## Implementation Blueprint

```
qlora_finetuning/evaluation/metrics.py     + answer_agrees(outcome, config) -> bool | None
qlora_finetuning/evaluation/__init__.py    + export it
qlora_finetuning/local_qlora/predictor.py  split: resolve_tuned_adapter(adapter, context) -> AdapterRef
qlora_finetuning/demo/__init__.py          docstring with usage (no imports)
qlora_finetuning/demo/inputs.py            DemoInput, DemoInputError, load_input, sample_path
qlora_finetuning/demo/answer.py            DemoResult, answer(predictor, demo_input, ...)
qlora_finetuning/demo/render.py            as_text, as_json, ascii_text
qlora_finetuning/demo/cli.py               main(argv, **stand_ins) -> int
qlora_finetuning/demo/__main__.py          sys.exit(main())
qlora_finetuning/demo/sample_input.json    the invented sample, with "expected"
```

```python
# metrics.py
def answer_agrees(outcome, config) -> bool | None:
    if not outcome.ok: return None
    readings = read_answer(outcome.answer_text or "")
    if not readings: return None
    return any(values_match(reading, outcome.value, config.answer) for reading in readings)

# inputs.py
FIELDS = ("question", "table", "context", "expected")
def load_input(path) -> DemoInput:
    # not a file / not JSON / not an object           -> DemoInputError(ConfigError), names the file
    # an unknown field; question not a non-empty str  -> names the file and the field
    # table: a non-empty list of non-empty lists; a cell is str, int, or float (never bool)
    # context: absent or a list of str; expected: absent, a number, "yes", or "no"

# answer.py
def answer(predictor, demo_input, *, prompt=None, budget=None, evaluation_config=None) -> DemoResult:
    prompt = prompt or load_prompt()
    request = PredictionRequest(source name, prompt.system,
                                format_user(passages, table, question, prompt), budget.max_new_tokens)
    prediction = predictor.predict([request])[0]          # a batch of one (D13)
    outcome = run_reply(prediction.text, demo_input.table)  # the only reading of the reply
    agrees = answer_agrees(outcome, evaluation_config)
    over_length = input_tokens is known and input_tokens + max_new_tokens > budget.max_length
    return DemoResult(...)

# cli.py
def main(argv=None, *, tuned=None, untuned=None, resolve=None, out=None) -> int:
    args: --input FILE [FILE ...]  --adapter NAME | --untuned  --json  --check
    inputs = [load_input(p) for p in files or [sample_path()]]      # before the model (D7)
    context = SimpleNamespace(prompt=load_prompt(), split_cfg=None, tracking_cfg=load_tracking_config())
    if args.check: resolve the adapter (unless --untuned), print one line per check, return 0
    predictor = untuned() or tuned(args.adapter, context)            # ConfigError -> exit 2
    for each input: result = answer(...); print as_text or as_json
    return 0 if every result.outcome.ok else 1
```

Errors: `ConfigError` (which `DemoInputError` and `SerializationError` are) and
`PredictorError` print one line, `CONFIG ERROR: ...` or `MODEL ERROR: ...`, and return 2.
A program that does not run is a result with exit code 1. No traceback for either.

Tasks, in order:

1. `answer_agrees` in the metrics, exported; its tests in `tests/evaluation/test_metrics.py`.
2. Split `build_tuned_predictor`; run `tests/local_qlora` unchanged to prove nothing moved.
3. `inputs.py` and the sample (compute `expected` by hand and check it with `run_reply`).
4. `answer.py`, `render.py`, `cli.py`, `__main__.py`, the package docstring.
5. Tests: `tests/demo/test_inputs.py`, `test_answer.py`, `test_render.py`, `test_cli.py`,
   `test_guards.py`, `test_demo_gpu.py` (marked `gpu`).
6. Validation loop below, the GPU part included.
7. Documents: runbook section, `coding.md` rule, README line, architecture arrows, spec
   section 11 (what was built and what the GPU run showed), roadmap, execution plan,
   implementation summary.

## Validation Loop

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest tests/demo tests/evaluation tests/local_qlora -q
venv\Scripts\python -m pytest -q

venv\Scripts\python -m qlora_finetuning.demo --check
venv\Scripts\python -m scripts.comparison_freeze status

# On the GPU: seconds, free, no AWS
venv\Scripts\python -m qlora_finetuning.demo
venv\Scripts\python -m qlora_finetuning.demo --json
venv\Scripts\python -m qlora_finetuning.demo --untuned
venv\Scripts\python -m pytest -m gpu -q tests/demo
```

If the selected adapter gets the sample wrong, replace the sample with another invented
one and say so in the roadmap. Do not touch the adapter, the prompt, or any frozen file.

## Final validation Checklist

- [x] Tests and lint green, with and without the stack
- [x] `--check` passes on the real repository
- [x] The demo answers the sample on the GPU; the output is ASCII and unrounded
- [x] Nothing written under `artifacts/runs/` or `ledger/` by a demo call
- [x] No frozen value changed; no AWS call; no dataset row read
- [x] Documents, spec, roadmap, execution plan, implementation summary

## Time Estimate

| Task | Time |
|---|---|
| `answer_agrees` and the predictor split | 20 min |
| Inputs and the sample | 25 min |
| Answer, rendering, command line | 45 min |
| Tests | 60 min |
| GPU runs and fixes | 15 min |
| Documents | 30 min |
| **Total** | **2.5–3.5 hours** |

Confidence 9/10: every piece the demo joins exists and is tested, and the checkpoint is
on the machine. What can go wrong is the sample (the adapter may answer it differently
than expected) and the working tree shared with the v5.02 build.

## Anti-Patterns to Avoid

- Do not parse reply text in the demo, and do not compare two values outside the metrics.
- Do not add prompt wording, a stop string, or a default for a missing table.
- Do not accept an adapter path, an example id, or a split name.
- Do not start a tracked run or write a file.
- Do not import boto3, `frontier`, `comparison`, or the runner from `demo/`.
- Do not round, format, or rescale a value for display.

## Time Taken for PRP Generation

**~12 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 5 min |
| Research existing PRPs and patterns | 2 min |
| Writing PRP document | 3 min |
