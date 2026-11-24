# PRP: Local development environment and training stack (req_000 v0.03)

**Feature Name**: local_environment (project_scaffold v0.03)  
**Version**: 3  
**Created**: 2026-11-15  
**Status**: ✅ IMPLEMENTED  
**Implemented**: 2026-11-15  
**Implementation Summary**: [impl_000_20261115_005208.md](../impl_summary/impl_000_20261115_005208.md)  
**Confidence Score**: 9/10  
**Estimated Time**: 2.5–3.5 hours  
**Origin**: `reqspecs/reqs/req_000_project_scaffold/req_000_03_spec.md`  
**Template**: `prp_lean.md` (expanded, medium complexity)  

---

## Goal

Pin and install the Path A training stack, and add `scripts/env_check.py`: six fast offline
checks of this machine plus an opt-in smoke step that loads `Qwen/Qwen3-0.6B` in 4-bit,
attaches LoRA, and runs one training step. Passing the smoke on native Windows is milestone
M1 and resolves Q3.

## Why & What

- **User value**: proof, before any paid job, that the exact library versions work on this
  machine, and one command to re-check it.
- **Technical**: `requirements-train.txt`, `config/local_env.yaml`,
  `qlora_finetuning/environment.py`, `scripts/env_check.py`, stand-in unit tests, one `gpu`
  test, an import guard, and reference updates.

## Success Criteria

- [ ] `venv\Scripts\python -m scripts.env_check` prints PASS for E1–E6, SKIP for E7, exit 0.
- [ ] `venv\Scripts\python -m scripts.env_check --smoke` prints PASS for E1–E7 on native
      Windows; or the failure is recorded and the work stops for the owner (spec D2).
- [ ] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass.
- [ ] The unit tests also pass with the training libraries made unimportable (what CI sees).
- [ ] `venv\Scripts\python -m pytest -m gpu -q` passes on this machine.
- [ ] `requirements-train.txt` holds only exact pins and matches what is installed.
- [ ] The gotchas reference marks each entry this version checked as confirmed or corrected.
- [ ] **Owner step**: CI is green on the pull request that carries this version.

---

## Context

### Research method

The library versions are newer than both planning guides and newer than anything that can be
assumed from memory. Per spec D15 the stack was installed during research and every API named
below was read from the installed packages, not recalled.

### References

```yaml
refs:
  - reqspecs/references/common/gotchas.md          # section 1 entries to confirm or correct
  - reqspecs/references/common/troubleshooting.md
  - reqspecs/references/project_patterns/README.md # check-script candidate: this PRP makes it a pattern
  - url: https://huggingface.co/docs/transformers
    why: BitsAndBytesConfig, chat templates
  - url: https://huggingface.co/docs/peft
    why: LoraConfig, prepare_model_for_kbit_training
  - url: https://huggingface.co/docs/bitsandbytes
    why: Linear4bit, the 8-bit optimizers
  - url: https://huggingface.co/Qwen/Qwen3-0.6B
    why: model card; chat template and thinking mode

code:
  - file: scripts/aws_check.py
    why: the shape to mirror: CheckResult, one function per check, guard, exit codes, by-path hint
  - file: tests/scripts/test_aws_check.py
    why: stand-in style, main() tests with capsys, subprocess tests
  - file: qlora_finetuning/settings.py
    why: load_config, ConfigError, REPO_ROOT
```

### Verified facts (2026-11-15, this machine)

| Item | Value |
|---|---|
| Installed | torch 2.14.1+cu126 (CUDA 12.6), transformers 5.23.0, datasets 5.2.0, peft 0.21.3, trl 1.18.0, accelerate 1.15.0, bitsandbytes 0.50.2, tokenizers 0.23.3, safetensors 0.8.0, huggingface_hub 2.4.0 |
| GPU | RTX 4090 Laptop GPU, 15.99 GiB, driver 596.08, capability (8, 9), `torch.cuda.is_bf16_supported()` True |
| Smoke model | `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`, `model.safetensors` 1.5 GB, not gated |
| `bnb.nn.Linear4bit` | `(input_features, output_features, bias=True, compute_dtype=None, compress_statistics=True, quant_type='fp4', quant_storage=torch.uint8, device=None)`. The weight is float32 until `.to("cuda")`, then `uint8` with a `quant_state` |
| `transformers.BitsAndBytesConfig` | `load_in_4bit`, `bnb_4bit_quant_type`, `bnb_4bit_use_double_quant`, `bnb_4bit_compute_dtype` all present |
| `peft.prepare_model_for_kbit_training` | `(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None, auto_clear_cache=True)` |
| `peft.LoraConfig` | `r`, `lora_alpha`, `lora_dropout`, `target_modules` (accepts `"all-linear"`), `task_type` |
| 8-bit optimizer | `bnb.optim.PagedAdamW8bit` exists |
| Qwen3 chat template | Thinking is on by default. `enable_thinking=False` appends an empty `<think>\n\n</think>\n\n` block to the prompt; the model then answers directly. With the default, output starts with `<think>` |
| Tokens | `eos_token` is `<|im_end|>`, `pad_token` is `<|endoftext|>` |
| TRL 1.18.0 | No `DataCollatorForCompletionOnlyLM`. `SFTConfig` has `completion_only_loss` (default `None`), `assistant_only_loss` (default `False`), and `max_length` (default 1024) |
| Measured smoke | Loss 2.071, 10,092,544 trainable parameters, peak VRAM 1.2 GiB, about 35 s on the first run including model load from disk cache, 5 s after |

### Critical Gotchas

- **No training library at module level.** `environment.py`, `env_check.py`, and every test
  import torch and the rest inside functions only. A guard test enforces it.
- **"No thinking block" is a statement about the output.** With thinking disabled the prompt
  itself ends with an empty think block, so check the generated text, not the prompt.
- **Pass `gradient_checkpointing_kwargs={"use_reentrant": False}`** to
  `prepare_model_for_kbit_training`, or torch warns on every checkpointed call.
- **Two `FutureWarning` lines remain** from peft calling transformers'
  `gradient_checkpointing_enable`, which 5.23 deprecates in favour of
  `activation_checkpointing_enable`. They are harmless and are not suppressed.
- **Hugging Face libraries log every HTTP request at INFO** on the `httpx2` logger. Hold the
  noisy loggers at WARNING after `setup_logging()`.
- **`torch` prints "triton not found"** on Windows at import. Harmless.
- **A logger named `__name__` is `__main__` under `-m`.** Name it `scripts.env_check`.
- **`python_min` must be quoted in YAML**; unquoted `3.10` loads as `3.1`.
- **`setup_logging()` in `main()` leaves a handler on the root logger**, so the logging tests
  must remove a stray handler before they start as well as after.
- **Running entry points in tests must not load the real machine.** `main(argv, env=...)`
  takes a prepared `Environment`; only the `gpu` test lets it gather the real one.
- **Shell heredocs on this machine halve doubled backslashes.** Write any Python that contains
  backslashes with the file tool, not through a heredoc.

---

## Implementation

### Files

```yaml
CREATE:
  - requirements-train.txt: ten exact pins and the two install commands in its header
  - config/local_env.yaml: every value the checks compare against
  - qlora_finetuning/environment.py: read_pins, installed_versions, version_matches, environment_report
  - scripts/env_check.py: checks E1-E7, real probes, runner, CLI
  - tests/test_environment.py
  - tests/scripts/test_env_check.py
  - reqspecs/references/tech/hf_training_stack.md
  - reqspecs/references/project_patterns/check_script.md
  - reqspecs/impl_summary/impl_000_<TIMESTAMP>.md

MODIFY:
  - ruff.toml: target-version py311
  - tests/test_process_scaffold.py: guard against module-level imports of the training stack
  - tests/test_logging_setup.py: remove a stray handler before each test
  - docs/runbook.md: section 3 filled in
  - reqspecs/1. PROMPTS/coding.md: the import rule
  - reqspecs/references/common/gotchas.md, troubleshooting.md, project_patterns/README.md
  - roadmap, execution plan (M1, Q3), spec status
```

### Design

```python
# scripts/env_check.py
@dataclass(frozen=True)
class CheckResult: check_id: str; name: str; status: str; detail: str      # PASS | FAIL | SKIP

@dataclass
class Environment:            # everything the checks read; tests build one from stand-ins
    version_info; pins; installed
    torch = None; torch_error = None; driver = None
    bnb_probe = None; bnb_error = None; smoke_run = None

def check_python(version_info, cfg)        # E1  >= python_min
def check_packages(pins, installed)        # E2  every pin installed, version_matches()
def check_cuda_build(torch, cfg)           # E3  __version__ ends with +<tag>; cuda.is_available()
def check_gpu(torch, cfg, driver)          # E4  VRAM >= min; reports name, VRAM, driver
def check_bf16(torch, cfg)                 # E5  capability >= min and is_bf16_supported()
def check_bitsandbytes(probe)              # E6  probe() -> {quantized, finite, version}; exception -> FAIL
def check_smoke(run, cfg)                  # E7  run(cfg) -> dict; finite loss, LoRA grads > 0,
                                           #     base grads == 0, no <think> in output, weights changed

def run_checks(cfg, env, smoke=False):
    # E3 FAIL when torch is None (message + install hint). E4 needs E3; E5 needs E4;
    # E6 needs E4, FAIL when bitsandbytes is missing; E7 SKIP unless smoke, needs E5 and E6.
    # A prerequisite that failed gives SKIP "needs E<n> to pass first".

def main(argv=None, env=None) -> int:      # 0 nothing failed, 1 a check failed, 2 ConfigError
    # --config PATH, --smoke, --report PATH (environment_report() + results as JSON)
```

Real probes, imports inside the functions:

```python
def bnb_probe():
    layer = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4").to("cuda")
    out = layer(torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
    return {"quantized": layer.weight.dtype == torch.uint8, "finite": isfinite(out).all(), ...}

def smoke_step(cfg):
    tokenizer, model = from_pretrained(model_id, revision, quantization_config=BitsAndBytesConfig(4-bit nf4,
                                       double quant, bfloat16), device_map={"": 0})
    prompt = apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    output_text = decode(model.generate(prompt_ids, max_new_tokens, do_sample=False)[new tokens])
    model = get_peft_model(prepare_model_for_kbit_training(model, True, {"use_reentrant": False}), LoraConfig(...))
    labels = input_ids.clone(); labels[:, :prompt_len] = -100        # completion-only by hand
    loss = model(input_ids, labels=labels).loss; loss.backward()
    count LoRA tensors with non-zero grad and frozen tensors with any grad
    bnb.optim.PagedAdamW8bit(trainable, lr).step(); compare one lora_B tensor before and after
```

`environment.py`: `read_pins` accepts only `name==version` lines (comments, blanks, and
option lines skipped; anything else raises `ConfigError` with the line number);
`version_matches("2.14.1+cu126", "2.14.1")` is true; `environment_report()` never raises.

### Tests

| File | Cases |
|---|---|
| `tests/test_environment.py` | Pins parsed, comments and option lines ignored, names normalized, four kinds of non-pin rejected, missing file; the committed file is complete; installed and absent packages; seven `version_matches` cases; report without torch, with a missing pins file, and plain-JSON |
| `tests/scripts/test_env_check.py` | Config loads and rejects a missing key; each of E1–E7 with one passing and at least one failing case; probe and smoke exceptions; the SKIP cascade for missing torch, a small GPU, and missing bitsandbytes; smoke not run unless requested; `main` exit codes 0, 1, 2; `--report`; module run; by-path hint; one `gpu` test of the real E1–E6 |
| `tests/test_process_scaffold.py` | No module-level import of a training library in `qlora_finetuning/`, `scripts/`, `tests/`; the scanner sees imports nested in `try`, `if`, and class bodies but not in functions |

### Tasks (~3 hours total)

1. **Install** (~20 min, mostly download): torch from `cu126`, then the pins.
2. **Config and pins file** (~10 min).
3. **`environment.py` and its tests** (~30 min).
4. **`env_check.py`** (~45 min): checks, probes, runner, CLI.
5. **`env_check` tests and the guards** (~40 min).
6. **Run**: plain, `--smoke`, `-m gpu`, unit tests with the stack blocked (~15 min).
7. **Documents and references** (~30 min).
8. **Status documents and summaries** (~10 min).

### Error handling

| Situation | Behavior | Exit |
|---|---|---|
| Nothing failed (SKIP allowed) | Lines and a summary | 0 |
| A library is missing or at the wrong version | FAIL naming it and the runbook section | 1 |
| A probe or the smoke raises | FAIL with the exception type and first line; no traceback | 1 |
| A prerequisite check failed | SKIP naming the prerequisite | — |
| Config missing or incomplete | `CONFIG ERROR` line | 2 |
| Smoke fails on native Windows | Stop. Record in the roadmap; the owner chooses the fallback (spec D2) | 1 |

---

## Validation

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q
venv\Scripts\python -m pytest -m gpu -q
venv\Scripts\python -m scripts.env_check
venv\Scripts\python -m scripts.env_check --smoke --report artifacts/env_check/report.json
```

**Expected**: ruff clean; all unit tests pass; the `gpu` test passes; six PASS and one SKIP;
then seven PASS and a report under the git-ignored `artifacts/` folder.

CI simulation: run pytest in a process whose import system refuses the ten training
libraries; the unit tests must still pass.

---

## Checklist

- [ ] Files created and modified as listed
- [ ] Each check has passing and failing tests
- [ ] Lint clean, unit tests green with and without the stack, `gpu` test green
- [ ] Smoke passes; M1 and Q3 recorded
- [ ] Gotchas confirmed or corrected; tech reference and pattern written
- [ ] Roadmap, execution plan, spec status updated; implementation summary created

---

## Implementation Time Estimate

**Total: 2.5–3.5 hours**

| Task | Time |
|---|---|
| Install the stack | 20 min |
| Config and pins file | 10 min |
| `environment.py` and tests | 30 min |
| `env_check.py` | 45 min |
| `env_check` tests and guards | 40 min |
| Runs on this machine | 15 min |
| Documents and references | 30 min |
| Status documents and summaries | 10 min |

Risk: +30–60 min had 4-bit loading failed on native Windows. It did not.

## Confidence Score: 9/10

Every API in the design was read from the installed packages and the smoke was seen to pass
during research. The remaining point is the CI run, which cannot be made from here.

## Time Taken for PRP Generation

**~14 minutes** (plus about 5 minutes of download time running in the background)

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 1 min |
| Codebase exploration | 1 min |
| Research existing PRPs and patterns (installing and inspecting the stack, thinking-mode check) | 7 min |
| Writing PRP document | 5 min |
