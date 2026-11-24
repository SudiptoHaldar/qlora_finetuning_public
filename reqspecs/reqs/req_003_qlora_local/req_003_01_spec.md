# QLoRA Local

Spec Group      :   req_003_qlora_local  
Feature Name    :   qlora_local  
Version         :   1  
Status          :   ✅ Built and run 2026-11-19 (`prp_003_qwen_backend_baseline_v1.md`); tests and lint green locally; A0 recorded on quick-dev and dev (section 11); commit pending; one question back to the owner: the batch size for the cloud runs  
Last updated    :   2026-11-19  

## 1. v3.01 — Qwen backend and zero-shot baseline

Path A starts here. This version lets the evaluation runner of v2.03 ask `Qwen/Qwen3-4B`,
loaded in 4-bit on a GPU, for replies, and uses it to score the untuned model on quick-dev on
this machine: arm **A0**.

It adds:

- a **renderer** that wraps the canonical system and user texts in the model's chat template
  with thinking switched off;
- a **backend** that loads the pinned model in 4-bit NF4 and generates greedily in batches,
  with an optional LoRA adapter;
- a **predictor** over that backend, registered in `scripts/evaluate.py` as `qwen3-4b`;
- **`config/local_qlora.yaml`** (quantization and generation settings);
- an optional **`measurements()`** method on a predictor, which the runner stores with the
  run: this is where peak GPU memory comes from;
- the **A0 baseline run** on quick-dev, with peak GPU memory and seconds per example.

It trains nothing and uses no AWS service. Nothing is billed. It downloads the model weights
once (8.04 GB, public). With v4.01 (arms B0-Pro and B0-Lite, done) it reaches milestone M4.

The run is development evidence. The A0 number of the comparison comes from the SageMaker
run of v5.01 (working agreement 11).

**Depends on**: v2.03, v1.04, v0.03 (all built).

## 2. What exists today (verified 2026-11-18)

### 2.1 What earlier versions provide

| Item | State |
|---|---|
| `Predictor` protocol, `Prediction`, `PredictorError` | `evaluation/predictor.py`. A backend implements `predict(requests)` and five attributes. Two optional attributes exist already (`usage_model_id`, `details`) |
| `run_evaluation` | Starts the run, writes the predictions file, and records the usage. Token totals reach the ledger when the predictor names a model and reports every count. **A backend writes nothing to the ledger itself**. The runner has no place for a backend's own measurements |
| `scripts/evaluate.py` | `PREDICTORS` table of `PredictorEntry(factory, kind, path, arm, where, configs)`; a `"module:function"` factory is imported only when named. A `ConfigError` from a factory is exit 2 with no run. A comment marks where the Qwen entry goes |
| `config/token_budget.yaml` | Model `Qwen/Qwen3-4B` at commit `1cfa9a7208912126459214e8b04321603b3df60c`, a fingerprint of its tokenizer, the thinking-off suffix, `max_new_tokens: 192` |
| `data/token_budget.py` | `render_prompt(record, tokenizer)` and `count_split`. Its docstring says the renderer used for models belongs to this version and must give the same token counts |
| Token budget report | Prompt tokens: quick-dev 128,495 in 200 prompts (457 at least, 645 at the median, 851 at most); dev 317,882 in 499 (886 at most) |
| `config/tracking.yaml` | Arm `A0`, path `A`, and place `local` are already allowed |
| `config/local_env.yaml` | The smoke model `Qwen/Qwen3-0.6B` at a pinned commit, with the same quantization values |
| `scripts/env_check.py` | `--smoke` loads the 0.6B model in 4-bit, generates once (one prompt, no batch), and takes one LoRA step. 7 of 7 on this machine |
| `qlora_finetuning/local_qlora/` | An empty package. `tests/local_qlora/` holds an `__init__.py` only |
| Guard tests | No module imports the training stack at module level; nothing under `evaluation/` imports `local_qlora`; nothing under `frontier/` imports `local_qlora` |
| Path B baselines (v4.01) | Execution accuracy on quick-dev: B0-Pro 75 of 200, B0-Lite 47 of 200. On dev: 190 and 123 of 499 |

### 2.2 The model, at the pinned commit

Read from the Hugging Face model repository at that commit.

| Item | Finding |
|---|---|
| Access | Public, not gated, Apache 2.0. The pinned commit is still the newest (last change 2025-07-26) |
| Weights | Three safetensors files, 8,044,982,000 bytes together (8.04 GB), stored as BF16 |
| **In the local cache** | **The tokenizer and `config.json` only (16 MB). The weights have never been downloaded** |
| Size | 4.0B parameters, 3.6B outside the embeddings. 36 layers, hidden size 2,560, 32 attention heads, 8 key-value heads of 128. Input and output embeddings are tied |
| Context | 32,768 tokens natively. Our longest prompt is 1,817 tokens (fit) |
| `generation_config.json` | `do_sample: true`, `temperature: 0.6`, `top_k: 20`, `top_p: 0.95`, `eos_token_id: [151645, 151643]`, `pad_token_id: 151643`. **A call that does not say otherwise samples** |
| Model card on decoding | For thinking mode: "DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions". For non-thinking mode it suggests temperature 0.7 and top-p 0.8, and names no danger. It suggests a presence penalty against repetition |

### 2.3 The tokenizer, from the local cache (no weights, no GPU)

| Item | Finding |
|---|---|
| Fingerprint | Equal to the one pinned in `config/token_budget.yaml` |
| End and padding | `eos_token` is `<\|im_end\|>` (151645); `pad_token` is `<\|endoftext\|>` (151643). Both are the model's two end tokens |
| **Padding side** | **`right` by default.** A batch that is generated from must be padded on the left |
| **`<think>` and `</think>`** | Ids 151667 and 151668. **They are not special tokens**: decoding with `skip_special_tokens=True` leaves them in the text |
| Prompt | With `enable_thinking=False` the prompt ends `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` |
| Two ways to tokenize | Tokenizing the rendered text with `add_special_tokens=False` gives the same ids as `apply_chat_template(tokenize=True)` |

### 2.4 Estimates, none of them measured

| Quantity | Estimate | Basis |
|---|---|---|
| Weights in GPU memory | About 2.6 GB | 3.6B parameters at 4 bits, plus the embeddings (151,936 by 2,560) in BF16. The 0.6B model measured 0.49 GiB against 0.53 by the same sum |
| Cache of one sequence while generating | About 0.16 GB | 147 KB per token (36 layers, keys and values, 8 heads of 128, BF16) times 886 plus 192 tokens |
| A batch of 8 | About 1.3 GB on top of the weights | 8 times the above |
| GPU | 16,376 MiB, of which about 2,000 MiB were in use by other programs when checked | `nvidia-smi` |
| Time per example | Not known | The trial of section 9 measures it |

### 2.5 What could not be checked

No weights are on this machine, so nothing was generated. Four things are therefore settled
by the build, not by this spec:

1. that the 4B model loads in 4-bit on this GPU, and its peak memory;
2. how long an example takes, at batch size 1 and at 8;
3. whether the untuned model ends its reply after the two lines or runs on;
4. whether replies generated in a batch equal those generated one at a time.

## 3. Locked decisions

### The prompt

| # | Decision | Reason |
|---|---|---|
| D1 | The model and its commit are the ones in `config/token_budget.yaml`. `config/local_qlora.yaml` does not repeat them | v1.04 D1: one pin for the tokenizer that counts and the model that answers |
| D2 | `render_chat(system, user, tokenizer)` is the renderer for models: the tokenizer's own chat template over a `system` and a `user` message, with the generation prompt and `enable_thinking=False`. The two texts are the request's, **character for character**. `data/token_budget.py` keeps its own `render_prompt`; a test shows the two give the same text | v1.03: every arm reads the same words. `data/` must not import a path package |
| D3 | The rendered text is tokenized with `add_special_tokens=False` and **never truncated**. There is no length limit on a prompt | v1.04 D8. The context is 32,768 tokens |
| D4 | A prompt that does not end with the thinking-off suffix of `config/token_budget.yaml` is refused before anything is generated | A changed template would silently turn thinking back on |
| D5 | The baseline is zero-shot: the system text and the user text, no examples, no wording for Qwen alone | Plan §4.2 and working agreement 7 |

### Loading

| # | Decision | Reason |
|---|---|---|
| D6 | 4-bit NF4, double quantization, BF16 compute, the whole model on GPU 0. The values come from `config/local_qlora.yaml` | Plan §4.6. They are the values `env_check --smoke` proved on this machine |
| D7 | Model and tokenizer are loaded with `local_files_only=True`. Downloading the weights is one runbook command, run once | A run never touches the network, and a missing file is a clear message, not a silent 8 GB download |
| D8 | The factory checks, **before any run exists**: the training stack is installed, CUDA is available, the tokenizer has the pinned fingerprint, and the weights are in the cache. Then it loads the model. Any failure is a `ConfigError` that names the command to run: exit 2, no run, no ledger line | v4.01 D14: a run that fails before its first reply would leave a failed run in the committed index every time |
| D9 | An adapter is optional. `load_backend(..., adapter_dir=<folder>)` attaches a saved LoRA adapter with PEFT and **does not merge it**. The registered predictor of this version loads none | Merging into 4-bit weights changes the numbers the adapter was trained against. Which adapter a tuned predictor loads is v3.02's to say |
| D10 | In `settings`, an adapter is named by the SHA-256 of its weights file, never by its path | The same adapter has another path in a SageMaker job. The hash says which weights answered |

### Generating

| # | Decision | Reason |
|---|---|---|
| D11 | Greedy decoding: `do_sample=False`, one beam, no repetition or presence penalty. Temperature, top-p, and top-k are explicitly unset, so the model's own sampling defaults (section 2.2) do not apply | Plan §4.6. A baseline must repeat. Path B decodes greedily too (`topK: 1`). The model card's warning is recorded in section 7 |
| D12 | **Generation stops at the model's own end tokens, `<\|im_end\|>` or `<\|endoftext\|>`, or at the reply limit. Nothing else stops it**: no stop string, and no cut after the `ANSWER` line. **Decision for the owner in section 8** | v4.01 D4 sends Nova no stop sequence because it "would be a difference from Path A". A model that runs on has a format fault, and the scorer should see it |
| D13 | Requests are generated in batches of `inference.batch_size` (8), padded on the left with an attention mask. The batch size is part of `settings`. **Decision for the owner in section 8** | v2.03: a backend whose replies can depend on the batch size puts it in `settings`, so a run cannot be resumed under another one |
| D14 | The reply limit is the request's `max_new_tokens` (192) | v2.03 D13: one limit for every arm |
| D15 | A row's generated tokens end at its first end token. `stop_reason` is `end` if it has one and `length` if it ran to the limit. `other` is not used | v2.03 D3 |
| D16 | `output_tokens` counts the generated tokens up to and including the end token. `input_tokens` is the length of the prompt without padding. `usage_source` is `measured` | The same unit as the token budget's "target with end token". The acceptance compares the input total with the budget's |
| D17 | The reply text is the generated tokens before the end token, decoded with special tokens removed and nothing tidied. One more change is allowed: a complete `<think>...</think>` block at the very start is removed, with the blank lines after it, and counted. An unclosed `<think>` is left for the parser to fault | `evaluation_run.md`: the parser knows no model. The two think markers are not special tokens (section 2.3), so decoding does not remove them |
| D18 | A reply the model did give is returned whatever it is: empty, cut at the limit, or in the wrong format | v2.03 D10 |
| D19 | The latency of a prediction is its batch's time divided by the batch's size | v2.03, section 7: the backend's own division |

### Failures and measurements

| # | Decision | Reason |
|---|---|---|
| D20 | An error while generating (out of GPU memory, a CUDA error) is a `PredictorError` that names the error type and the batch size, and holds no prompt text. For out of memory it says to lower `inference.batch_size` and start a new run | v2.03 D9: the run stops and keeps its lines. A resume under another batch size is refused (D13) |
| D21 | A predictor may have a `measurements()` method that returns plain values. After the prediction loop, also when it failed, the runner writes them to `measurements.json` in the run folder and `summarize_run` returns them under `backend`. A predictor without the method runs exactly as before | Peak memory is known only inside the process and only after the last batch, and the predictor does not know its run folder. v3.04 and v3.06 need the same |
| D22 | A `measurements()` that raises, or returns something that is not plain values, is logged and skipped. It never fails a run | A lost measurement must not cost the predictions |
| D23 | The Qwen predictor reports: the GPU's name and total memory; load seconds; the weights' footprint; peak memory allocated and reserved, counted from before the load; batches; generation seconds; tokens generated; think blocks removed | "Peak VRAM and seconds per example" (the execution plan's check for this version) |

### The predictor and the run

| # | Decision | Reason |
|---|---|---|
| D24 | `QwenPredictor(backend, name, settings)` implements `Predictor`. It takes the backend as an argument | Tests hand over a stub; no test outside the `gpu` marker loads a model |
| D25 | `settings` holds everything that can change a reply: the model commit, the three quantization values, `enable_thinking: false`, `do_sample: false`, `num_beams: 1`, `stop: end tokens`, `batch_size`, and `adapter` (`null`, or the hash of D10) | The resume check compares it (v2.03 D19) |
| D26 | It is registered as `qwen3-4b`: path A, arm A0, kind `eval`, place `local`, as `"module:function"`, and the run also snapshots `local_qlora` | The training stack is loaded only when the entry is named. `--where sagemaker` overrides the place in v3.03 |
| D27 | The baseline run is scope `dev` | Development evidence |
| D28 | Only `local_qlora/backend.py` imports `torch`, `transformers`, `peft`, or `bitsandbytes`, and only inside functions | CI has no training stack. One module to read when a library changes |
| D29 | The `gpu` tests use the 0.6B smoke model of `config/local_env.yaml`: same tokenizer and chat template, a few seconds to load. One more loads the 4B model and is skipped when its weights are not cached | The backend takes the model id and commit as arguments |
| D30 | During the build Claude downloads the weights (8.04 GB, into the default Hugging Face cache outside the repository) and runs the local GPU runs of section 9. Neither is billed and neither uses AWS | Stated so the owner can object before the PRP runs (as v0.03 D15) |
| D31 | No report file is added. The A0 figures go into this spec (section 11), `hf_training_stack.md`, the roadmap, and the execution plan; the counts are in `ledger/evaluations.jsonl` | Nothing later reads an A0 report, and the comparison's A0 comes from v5.01 |

## 4. Scope to build

### 4.1 Config

`config/local_qlora.yaml`:

```yaml
version: 1
verified_on: "2026-11-18"
# The model and its commit are tokenizer.model_id and tokenizer.revision in token_budget.yaml.
predictor_name: qwen3-4b
quantization:
  type: nf4
  double_quant: true
  compute_dtype: bfloat16
inference:
  batch_size: 8
  do_sample: false
  num_beams: 1
  end_tokens: ["<|im_end|>", "<|endoftext|>"]
```

### 4.2 `qlora_finetuning/local_qlora/`

| Module | Holds |
|---|---|
| `config.py` | `load_local_qlora_config()`: typed values; a missing key is a `ConfigError` naming it. A batch size below 1, sampling switched on, or more than one beam is refused: this file describes greedy decoding and nothing else |
| `render.py` | `render_chat(system, user, tokenizer)` (D2, D4) and `read_generation(token_ids, tokenizer, end_ids, limit)`, which turns one row of generated ids into text, token count, and stop reason (D15 to D17). No training-stack import: both take the tokenizer as an argument |
| `backend.py` | `check_ready(budget)` (D8), `load_backend(cfg, budget, model_id=None, revision=None, adapter_dir=None)`, and `QwenBackend` with `generate(prompts, max_new_tokens)` and `measurements()`. The only module that imports the training stack (D28) |
| `predictor.py` | `QwenPredictor` (D24, D25) and the factory `qwen3_4b(split, context)` that the script names |

`QwenBackend.generate` returns, per prompt, the reply text, the two token counts, the stop
reason, and the batch's seconds. It sets the padding side to left on its own tokenizer
object and passes the two end-token ids and the padding id itself.

```python
from qlora_finetuning.data.token_budget import load_token_budget_config
from qlora_finetuning.evaluation.runner import run_evaluation
from qlora_finetuning.local_qlora.backend import load_backend
from qlora_finetuning.local_qlora.config import load_local_qlora_config
from qlora_finetuning.local_qlora.predictor import QwenPredictor

cfg, budget = load_local_qlora_config(), load_token_budget_config()
backend = load_backend(cfg, budget)                  # 4-bit, from the local cache, on GPU 0
predictor = QwenPredictor(backend, name=cfg.predictor_name, settings=backend.settings)
summary = run_evaluation(predictor, "quick_dev", label="qwen3-4b quick_dev",
                         path="A", arm="A0", where="local", scope="dev")
summary["all"]["execution_accuracy"]                 # {"count": ..., "rate": ...}
summary["usage"]["input_tokens"]                     # 128,495: the token budget's own count
summary["backend"]["peak_vram_allocated_gib"]        # from measurements()
```

### 4.3 Runner and script

- `evaluation/runner.py`: `measurements.json` (D21, D22), written in the run's `finally`
  beside the usage record; `summarize_run` adds `backend` when the file is there. The
  summary of a run without the file is unchanged.
- `scripts/evaluate.py`: the `qwen3-4b` entry (D26), and one printed line of measurements
  when a summary has them. The module docstring says the entry is local, free, and needs
  the weights.

```powershell
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b --split quick_dev
```

### 4.4 Tests

No test outside the `gpu` marker needs a GPU, the network, or the training stack. Prompts
are invented rows; the tokenizer is the stand-in of `tests/data/synthetic.py`.

- Config: each missing key; a batch size of 0; sampling on; two beams; an unknown
  quantization type.
- Renderer: the system and user texts arrive character for character, including line
  breaks, a trailing space, and non-ASCII characters; the text equals `render_prompt` of
  the token budget for the same record; a template that does not end with the thinking-off
  suffix is refused.
- `read_generation`: cut at the first end token, for either end token; no end token gives
  `length`; the count includes the end token; an empty reply; a think block at the start
  removed and counted; an unclosed `<think>` left in; leading and trailing white space and
  non-ASCII text unchanged.
- Predictor with a stub backend: passes `describe`; settings as D25; one prediction per
  request in order; latency divided over the batch; a backend error becomes a
  `PredictorError` without prompt text. Through the runner on the invented split: token
  totals reach the ledger as `measured` under `Qwen/Qwen3-4B`; a backend that fails after N
  batches is resumed and no example is asked twice; a resume under another batch size or
  another adapter is refused.
- `check_ready`: no training stack, no CUDA, weights not cached, a wrong fingerprint. Each
  is a `ConfigError` with its command, and through the script each is exit 2 with no run
  folder and no ledger line.
- Runner: a predictor with `measurements()` gets the file and the `backend` block, also
  when the run fails; one that raises or returns a non-plain value loses nothing; the gold
  predictor's run and output are as before.
- Script: the entry resolves with a stub factory; the place defaults to `local`; the run
  snapshots `local_qlora`.
- Guards: under `local_qlora/`, only `backend.py` names the training stack, and only
  inside functions; nothing under `local_qlora/` imports `frontier`.
- With the real tokenizer, when it is cached (skipped otherwise): `render_chat` plus the
  tokenizer gives, for invented records, the prompt count of `count_record`; on the real
  quick-dev records, when the snapshot is present, the total is 128,495.
- `gpu`, on the 0.6B model: a batch of three prompts of different lengths generates; each
  row's input count is its unpadded length; no reply contains a think marker; a saved
  LoRA adapter with untrained (zero) weights gives the base model's replies and one with
  random weights does not, and `settings["adapter"]` is the hash of its file.
- `gpu`, on the 4B model, skipped when not cached: it loads, the measurements are filled,
  and one invented prompt comes back with a stop reason of `end` or `length`.

### 4.5 Documentation

- Runbook section 5 (Path A): download the weights, the trial, the baseline, resuming,
  where the model cache is and how to clear it, troubleshooting (out of memory, weights
  missing).
- `hf_training_stack.md`: the batched generation call; left padding; the think markers are
  not special tokens; the model's sampling defaults; a "Measured on this machine
  (Qwen/Qwen3-4B)" table.
- `gotchas.md`: the "small model says nothing about the 4B model's memory" row is closed
  with the measured figure. New rows: the tokenizer pads on the right by default; a call
  that does not set `do_sample` samples; `skip_special_tokens` leaves `<think>`; replies
  can depend on the batch size; the model card's warning about greedy decoding.
- `evaluation_run.md`: `measurements()`, and the Qwen predictor as built in place of the
  sketch.
- `coding.md`: Qwen is asked only through `QwenBackend`; only `backend.py` imports the
  training stack under `local_qlora/`; `config/local_qlora.yaml` in the config list.
- Architecture overview: the backend and the predictor as built.
- Execution plan, section 5 (P3): the first check is reworded as section 8, question 1,
  is decided.

## 5. Out of scope

- Training, the completion-only loss, and rendering a target (v3.02). The renderer of this
  version stops at the generation prompt.
- A registered predictor for a tuned model, and where adapters are kept (v3.02, v3.04).
- Running in a SageMaker job, and A0 in the cloud (v3.03, v5.01).
- Sampling, a repetition or presence penalty, beam search, or a stop string.
- Sorting examples by length to pad less: the runner fixes the order.
- Another runtime (vLLM, llama.cpp, an 8-bit or unquantized load) or another model
  (req_007).
- A baseline report file (D31), and any cost figure: a local run has no bill.
- A second evaluation loop of any kind.

## 6. What this hands to later versions

| For | What |
|---|---|
| v3.02 | `render_chat`, to extend with the target for training, so that a tuned model is asked the way it was taught. `load_backend(adapter_dir=...)`, to reload the smoke run's adapter. The measured memory of the 4B model at inference |
| v3.03 | The same predictor inside a job, with `--where sagemaker`. Seconds per example at batch size 1 and 8, for the first time and cost estimate. `measurements()` for the job's peak memory |
| v3.04 | A0 on dev from this machine, if question 3 of section 8 is accepted, to read A1 against. A predictor for the selected adapter is the same class with `adapter_dir` |
| v3.06 | The predictor for per-checkpoint scoring on quick-dev |
| v5.01 | `--predictor qwen3-4b --split test` in the cloud, with the batch size it then finds in config |
| v5.02 | Ledger records with measured token counts and wall-clock seconds for Path A |
| v6.01 | `load_backend` and `render_chat` for the demo, which needs no runner |

## 7. Known limits and follow-ups

- **The model card advises against greedy decoding** for thinking mode and suggests
  sampling for non-thinking mode (section 2.2). The plan keeps greedy decoding so that an
  arm's replies repeat and both paths decode alike. If the untuned model repeats itself,
  the count of `length` stop reasons shows it. It is a result to report, not a reason to
  sample for Qwen alone.
- **Replies can depend on the batch.** With left padding and a mask the arithmetic is the
  same, but in BF16 a near-tie between two tokens can fall the other way. Section 9
  measures how often. The batch size is in `settings`, so runs are not mixed.
- **A local run is not the cloud run.** Another GPU model can also flip a near-tie. A0 here
  and A0 from SageMaker need not agree to the example.
- **An untuned model may run on after the answer.** Under D12 that lowers format validity
  and costs up to 192 tokens of time per example. The reply is still scored.
- **A baseline fixes the wording.** Nova has already been run on the current prompt, so
  the wording is frozen for Qwen too.
- **Local runs add lines to the committed ledger**: runs, usage records with token counts
  under `Qwen/Qwen3-4B`, and a results line. The Path B cap check counts Path B records
  only, so they change no budget figure.
- **Predictions and measurements stay in a git-ignored folder** on one machine.
- **Other programs use the GPU.** About 2 GB was in use when checked; peak memory is the
  process's own, and the time per example is a laptop's.
- **200 examples is a small sample** from at most 15 companies. Read the baseline as a
  level, not to the percentage point.
- **The estimates of section 2.4 are sums, not measurements.**

## 8. Decisions for the owner

**Taken as recommended on 2026-11-18**: the owner started the PRP and its execution
without changing any of the three. For question 2 the measurement came out against
batch size 8 (16 of 40 replies differed), so by the rule stated there the baseline
was run at batch size 1 and the question returns to the owner for the cloud runs
(section 11). The table is kept as the record of what was asked.

| # | Question | Recommendation |
|---|---|---|
| 1 | **Where does a reply end (D12)?** At the model's end token only, or also cut after the `ANSWER` line? The execution plan's check for this version reads "generation stops after the `ANSWER` line" | End token only. Nova was run with no stop sequence, so cutting Qwen's reply would give Path A a tidier format score than Path B for the same behavior. An untuned model that keeps writing is then visible in format validity, and v3.02 teaches the tuned model to stop by ending every target with the end token. The plan's check becomes "generation ends at the model's end token or the reply limit; the rendered prompt has thinking off and no reply holds a thinking block". The alternative saves time per example and hides the fault |
| 2 | **Batch size 8 or 1 (D13)?** | 8, recorded in `settings`, with the agreement between 8 and 1 measured on 40 examples before the baseline (section 9). It matters later more than now: Path A is billed by the second in the cloud, so the batch size sets its cost per 1,000 questions in the comparison. If the measurement shows replies differing in more than a few examples, the baseline is run at 1 and the question comes back to you with the numbers |
| 3 | **Score A0 on dev (499) on this machine as well, or only on quick-dev?** The plan says quick-dev | Run dev too. It is free, it is the same command, v3.04 reads A1 against A0 on dev, and both B0 baselines already have a dev figure to sit beside. The cost is laptop GPU time, known after the trial |

## 9. Acceptance

Code, by Claude:

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
  and without the training stack blocked.
- A test shows the rendered prompt holds the canonical texts character for character and
  equals the token budget's rendering.
- `--predictor gold` gives the same output as before.

On this machine, by Claude (free; D30):

```powershell
venv\Scripts\hf download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c   # 8.04 GB, once
venv\Scripts\python -m pytest -m gpu -q
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b --split quick_dev --limit 5    # the trial
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b --split quick_dev --limit 40   # once at batch size 1, once at 8
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b --split quick_dev              # arm A0
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b --split dev                    # if question 3 is accepted
```

- Before the download, with the weights absent, the trial command exits 2, prints the
  download command, and leaves no run.
- The `gpu` tests pass.
- The trial completes as a partial run and prints peak memory and latency.
- The two 40-example runs are compared reply by reply: how many differ, and the seconds
  per example of each. `config/local_qlora.yaml` is left at the value question 2 settles.
- The A0 run ends with `RESULT: complete`; checks R1 to R5 pass; it has a line in
  `ledger/evaluations.jsonl` and a ledger record with measured token counts.
- **Its input tokens are 128,495**, the token budget's count for quick-dev (317,882 for
  dev). A difference means the model was not asked what the budget counted, and is
  investigated before any figure is quoted.
- No reply holds a thinking block; the number removed under D17 is reported, and expected
  to be 0.
- Peak GPU memory is under the card's 16 GB, with the figure recorded.
- Section 11, "As built", is added to this spec: the four metrics and the tight
  diagnostic, stop reasons, error codes and format deviations by count, tokens, peak
  memory, seconds per example, and the batch comparison, beside the B0 figures.
- Milestone M4 is recorded in the execution plan, and the roadmap is updated.

## 10. Next step

This version is built and run; milestone M4 is reached. Next is the v3.02 spec (QLoRA
training pipeline and the local smoke run). The owner's open item from this version is
the batch size for the cloud runs (section 11), needed by v3.03.

## 11. As built (2026-11-19)

Built by `prp_003_qwen_backend_baseline_v1.md`. The weights were downloaded once (8.04 GB)
and every run below was made on this machine, at no cost.

What was run:

- `venv\Scripts\python -m pytest -q`: 1,999 passed (72 new). With the training stack
  blocked: 1,995 passed, 4 skipped. `venv\Scripts\python -m ruff check .`: clean.
- `venv\Scripts\python -m pytest -m gpu -q`: 6 passed (5 new: batches, greedy and repeating,
  measurements, an adapter attached and named by its hash, the 4B model).
- `scripts.evaluate --predictor qwen3-4b ... --limit 5` with an empty cache folder: `CONFIG
  ERROR: the weights of Qwen/Qwen3-4B ... are not in the local Hugging Face cache`, the
  download command, exit 2, no run.
- `scripts.evaluate --predictor gold --split quick_dev --limit 3`: as before.
- The trial, the two 40-example runs, and A0 on quick-dev and on dev.

### Arm A0 on this machine (untuned, 4-bit, greedy, batch size 1)

| | quick-dev | dev |
|---|---:|---:|
| Run | `20261119T050056Z-eval-qwen3-4b-quick-dev-d424` | `20261119T051455Z-eval-qwen3-4b-dev-7efd` |
| Examples | 200 | 499 |
| Format validity | 181, 90.50% | 454, 90.98% |
| **Execution accuracy** | **1, 0.50%** | **3, 0.60%** |
| Answer accuracy | 24, 12.00% | 58, 11.62% |
| Program match | 0 | 1, 0.20% |
| Execution accuracy, tight rule (diagnostic) | 1 | 3 |
| Stop reasons | end 180, length 20 | end 454, length 45 |
| Input tokens | 128,495 | 317,882 |
| Output tokens | 16,099 | 38,910 |
| Seconds per example (wall clock over examples) | 4.13 | 4.01 |
| Latency: median / 95th percentile | 3.2 s / 9.7 s | 3.1 s / 9.7 s |
| Peak GPU memory allocated (reserved) | 2.73 GiB (2.85) | 2.74 GiB (2.90) |
| Checks R1 to R5 | pass | pass |

Both runs are results, with a line each in `ledger/evaluations.jsonl`. **The input tokens
equal the token budget's prompt tokens exactly**, so the model was asked what the budget
counted. No reply held a thinking block (0 removed). The weights take 2.42 GiB, as section
2.4 estimated; loading takes 6 to 7 seconds.

For comparison, execution accuracy of the untuned Nova models (v4.01): on quick-dev B0-Pro
75 of 200 and B0-Lite 47 of 200; on dev 190 and 123 of 499.

**Why A0 is near zero.** The untuned model keeps the two-line format in 91% of replies, but
what it writes on the `PROGRAM` line is almost never a program of this language. Replies by
outcome, counted by the error code of `run_reply`:

| Outcome | quick-dev | dev |
|---|---:|---:|
| `bad_reference`: refers to a step that does not exist | 84 | 216 |
| `bad_argument`: an argument that is not a number, a constant, or a reference | 50 | 107 |
| `nested_call`: one operation inside another | 37 | 85 |
| `row_not_found`: a table operation on a row name the table does not have | 11 | 38 |
| `syntax` | 5 | 15 |
| `row_not_numeric`, `wrong_arity`, `unknown_constant` | 5 | 10 |
| Ran without an error (right or wrong) | 8 | 28 |
| Format deviation `answer_missing` (the reply ran to the limit) | 19 | 45 |

The replies were read: they are coherent attempts in the right layout, not noise, and the
figures repeat (two runs of the same 40 examples at batch size 1 gave 40 identical replies).
The answer line is right more often than the program (12%), because the model works the
number out by itself. This is the gap the adapter has to close. It is a result to report
and no reason to change the wording, which is frozen.

### The batch comparison (question 2 of section 8)

The first 40 quick-dev examples, generated one at a time and in batches of 8:

| | Batch size 1 | Batch size 8 |
|---|---:|---:|
| Seconds per example | 4.14 | 1.28 |
| Generated tokens per second | 17.2 | 58.2 |
| Peak GPU memory allocated | 2.72 GiB | 3.83 GiB |
| Execution accuracy | 0 | 0 |
| Answer accuracy | 11 | 9 |
| Format validity | 38 | 37 |

**16 of the 40 replies differed** between the two; 4 answer scores changed and no execution
score. That is more than "a few examples", so by the rule of question 2 the baseline was
run at batch size 1, `config/local_qlora.yaml` is left at 1, and **the question comes back
to the owner** for the cloud runs: batch size 1 (each reply independent of its neighbors,
3.2 times the GPU time) or a larger batch (cheaper per 1,000 questions, replies that depend
on the batch). It is needed by v3.03. A tuned model may be less sensitive, since its
choices are less often near ties; that can be measured in v3.02 on the smoke adapter.

### Differences from sections 3 and 4

- **Batch size 1**, not 8 (above).
- **D11: the model's generation config is replaced, not overridden.** In transformers 5.23
  a config passed to `generate()` has every value that is `None` filled from the model's
  own, so "explicitly unset" would not unset the sampling values. The backend sets
  `model.generation_config` to a greedy one with the two end tokens; a `gpu` test checks
  that decoding is greedy and that two runs give the same replies.
- **The weights-absent check used an empty cache folder** (`HF_HUB_CACHE`), because the
  download was started while the code was being written.
- `render_chat` takes the thinking-off suffix as its fourth argument. `check_ready(budget,
  model_id, revision)` returns the stack and the tokenizer, and checks in this order: stack,
  weights, tokenizer, GPU. The weights check reads the index file and wants every file it
  lists, so a partial download is caught.
- `settings` also holds the `end_tokens`. `QwenPredictor` takes its `model_id` from the
  backend; the factory behind `qwen3_4b` is `build_predictor(adapter_dir=None)`.
- `measurements()` also reports the batch size, the number of prompts, the longest prompt,
  and generated tokens per second. The script's `measurements:` line prints the numbers.
- Requests of one batch with different reply limits are a `PredictorError`.
- No runner entry for a tuned model, and no report file, as planned (D9, D31).

### What stays open

- The batch size for the cloud runs (above): the owner's, by v3.03.
- Memory during training at 2,048 tokens: v3.02's smoke run.
- Whether A0 in the cloud equals A0 here: v5.01.
