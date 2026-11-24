# QLoRA Local

Spec Group      :   req_003_qlora_local  
Feature Name    :   qlora_local  
Version         :   2  
Status          :   ✅ Built and run 2026-11-19 (`prp_003_qlora_training_pipeline_v2.md`); tests and lint green locally; the smoke run passes K1 to K7 (section 11); commit pending. No full training run was made  
Last updated    :   2026-11-19  

## 1. v3.02 — QLoRA training pipeline

v3.01 can ask the untuned model and found that it almost never writes a valid program
(execution accuracy 3 of 499 on dev). This version builds what teaches it: a training
pipeline that fits a LoRA adapter to the 4-bit model on the `fit` records, and a smoke run
on this machine that proves the whole path before any GPU hour is paid for.

It adds:

- an **encoder** that turns a canonical record into the exact token sequence the token
  budget counted, with the loss on the reply only;
- **`config/training.yaml`**: every hyperparameter of plan §4.6;
- a **training entry point**, `scripts/qlora_train.py`, with a dry run, a smoke run,
  checkpoints, and resuming;
- a **worst-batch probe** that runs the longest records through one training step before
  anything else, so a run that cannot fit fails in its first seconds;
- a saved **adapter with its metadata**, and a predictor for it, `qwen3-4b-tuned`;
- the **smoke run**: 64 examples, loss falling, the adapter reloaded and asked.

It runs no full training: that is v3.06 (optional, this machine) and v3.04 (SageMaker). It
uses no AWS service and nothing is billed.

**Depends on**: v3.01 (built).

## 2. What exists today (verified 2026-11-19)

### 2.1 What earlier versions provide

| Item | State |
|---|---|
| `render_chat(system, user, tokenizer, suffix)` | The prompt as the model reads it, thinking off (v3.01). It stops at the generation prompt |
| `load_backend(cfg, budget, adapter_dir=...)` | The 4-bit model for inference, with a saved adapter attached and not merged. `settings["adapter"]` is the SHA-256 of the adapter's weights file |
| `config/local_qlora.yaml` | Quantization values and the inference batch size (1) |
| `config/token_budget.yaml` | `max_length: 2048`. `count_split` gives each record's full-sequence length: prompt, target, end token |
| `serialize_split("fit" \| "dev")` | 4,884 and 499 canonical records. One pass over fit is 3,254,181 tokens |
| `start_run`, `run.record` | Kinds `train` and `smoke`, phase `train`, and a `training_tokens` field exist |
| `scripts/evaluate.py` | `PredictorEntry.build(split, context)`. No way to name an adapter |
| `config/local_env.yaml` | The 0.6B smoke model and the plan's LoRA values, used by `env_check --smoke` for one hand-written step |
| v1.04 D6 | "The peak memory of a batch holding the two longest records is measured in v3.02's smoke run." They are 1,862 and 1,851 tokens, on one page |
| v3.01, section 11 | Replies depend on the batch for the untuned model (16 of 40 differ at batch size 8). Whether a tuned model is as sensitive was left to this version |

### 2.2 The installed trainer (TRL 1.18.0, transformers 5.23.0), read from its source

| Topic | Finding |
|---|---|
| A dataset that already has `input_ids` | Is taken as prepared: nothing is tokenized. A `labels` column is "left as is". The collator pads on the right and pads labels with -100 |
| **`max_length`** | **Defaults to 1,024 and is applied to a prepared dataset too**: every row is cut to it, and a row left with no labelled token is dropped, without an error. `max_length=None` switches it off |
| `completion_only_loss`, `assistant_only_loss` | For datasets the trainer tokenizes itself. Not needed when the labels are ours |
| The trainer's own chat-template path | Renders the conversation with the reply and gets `...<\|im_end\|>\n`: one token more than the token budget's sequence, which ends at the end token |
| `warmup_ratio`, `group_by_length` | **No longer exist.** Warmup is `warmup_steps` |
| Defaults that matter | `gradient_checkpointing: True`, `optim: adamw_torch_fused`, `learning_rate: 2e-5`, `lr_scheduler_type: linear`, `max_grad_norm: 1.0`, `save_steps: 500`, `eval_strategy: no`, `seed: 42`, `report_to: none` |
| Qwen's template with a finished reply | With `enable_thinking=False` it keeps the empty think block before the reply, and the prompt of `render_chat` is a prefix of it. So training and inference prompts agree whichever way they are built |

### 2.3 A trial on this machine (64 fit examples, 3 epochs, throwaway script)

The two longest fit records plus the first 62, the plan's settings (batch 2, accumulation 8,
learning rate 2e-4, cosine, 8-bit paged AdamW, gradient checkpointing, LoRA r=16 on all
linear layers), through `SFTTrainer` with a prepared dataset and `max_length=None`. Nothing
was written to the repository or the ledger.

| Quantity | Value |
|---|---|
| Sequence and target lengths against the token budget | Equal for all 64 |
| The trainer's prepared dataset | 64 rows, longest 1,862, labels untouched |
| Trainable parameters | 33,030,144. Saved adapter: 66 MB |
| GPU memory after load, ready to train | 3.34 GiB |
| One step on two ordinary records (641 tokens) | Peak 5.98 GiB allocated |
| **One step on the two longest records (1,862 tokens)** | **Peak 11.00 GiB allocated, 12.35 reserved; 5.6 s** |
| Peak during the 12 optimizer steps | 8.48 GiB allocated, 9.66 reserved |
| Speed | 12.6 s per optimizer step of 16 sequences: 0.79 s per sequence |
| Loss per step | 2.27, 1.78, 0.95, 0.48, 0.35, 0.35, 0.45, 0.36, 0.23, 0.33, 0.23, 0.29 |

The adapter was then loaded through `load_backend` and asked 24 quick-dev prompts:

| | Untuned (v3.01) | After 12 steps on 64 examples |
|---|---|---|
| Programs that run without an error | 8 of 200 | **24 of 24** |
| Replies identical at batch size 1 and 8 | 24 of 40 | 22 of 24 |
| Replies equal to the gold reply | 0 of 200 | 4 of 24 |
| Generated tokens per second, batch size 1 | 19.5 | 14.1 (the adapter is not merged) |

What this settles and what it suggests:

- **2,048 fits in 16 GB** (v1.04 D6), with little to spare: 12.35 GiB reserved beside the
  2 GiB other programs hold on this card. On a 24 GB cloud GPU there is room.
- **A full run is short.** 4,884 records at 0.79 s are 64 minutes an epoch here: about 3.2
  hours for 3 epochs, 916 optimizer steps. The cloud GPU is a different card; v3.03's smoke
  job measures it.
- **Scoring a checkpoint costs as much as training between two.** 499 dev replies at batch
  size 1 are roughly 20 minutes; 100 optimizer steps are 21.
- The model learns the language of the programs within a dozen steps. Whether the programs
  are right is what the full run and its dev scores are for.

## 3. Locked decisions

### The training sequence

| # | Decision | Reason |
|---|---|---|
| D1 | `encode_example(record, tokenizer, suffix)` builds one sequence: the ids of `render_chat(record.system, record.user)`, then the ids of `record.target` followed by the end token. Prompt and target are tokenized separately | v1.04 D2: this is the sequence the budget counted. Tokenized apart, no token can merge across the boundary |
| D2 | Labels are -100 for every prompt token and the token id for every target token and the end token | Completion-only loss. The end token is learned, so the tuned model stops by itself (v3.01 question 1) |
| D3 | A sequence's length must equal the token budget's count for the record. A test says so on invented rows, and the smoke run on the real ones | One definition of a training example, on both paths' cost estimates |
| D4 | **Nothing is truncated.** A sequence longer than `max_length` of `config/token_budget.yaml` is an error that names the examples, raised before the model is loaded | v1.04 D4 |
| D5 | The pipeline hands the trainer a prepared dataset (`input_ids`, `labels`) and `max_length=None`. After the trainer is built, the pipeline compares the trainer's dataset with its own, row count and every length, and refuses to train on a difference | Section 2.2: the trainer's default would cut 58 fit records to 1,024 tokens and say nothing |
| D6 | `completion_only_loss`, `assistant_only_loss`, and the trainer's chat-template path are not used | The labels are ours and are tested. The template path would add a token the budget does not count |
| D7 | Training reads `fit`: all 4,884 records, repeated ones included. `dev` is used for a loss figure at each checkpoint and for nothing else. No test row is loaded | Both paths train on the same examples (v4.02). The lock |

### The model and the optimizer

| # | Decision | Reason |
|---|---|---|
| D8 | The 4-bit load is one function in `local_qlora/backend.py`, used for inference and for training, with the quantization values of `config/local_qlora.yaml` | The adapter must be trained against the weights it is later asked with |
| D9 | `prepare_model_for_kbit_training` with gradient checkpointing (`use_reentrant=False`), then a LoRA adapter from `config/training.yaml`: r 16, alpha 32, dropout 0.05, all linear layers | Plan §4.6; the calls `env_check --smoke` proved |
| D10 | Every hyperparameter is in `config/training.yaml` and is passed to the trainer explicitly, including the ones whose library default is already right: batch 2, accumulation 8, learning rate 2e-4, cosine schedule, 8-bit paged AdamW, weight decay 0, gradient clipping 1.0, BF16, seed 42 | Plan §4.6 and working agreement 13. A library default can change with a version; section 2.2 lists four that differ from the plan |
| D11 | Warmup is 5% of the optimizer steps, rounded up, computed by the pipeline and passed as a step count | `warmup_ratio` is gone |
| D12 | Batches are drawn at random with the configured seed. There is no sorting or grouping by length | The library offers none now, and the guides do not ask for it |
| D13 | **Before training, the worst batch is probed**: the `batch_size` longest sequences go through one forward and backward pass, with no optimizer step. Its peak memory is recorded. If it fails, the run stops there | The two longest records are on one page; whether they meet in a batch is chance. A run that cannot hold them should fail in seconds, not in hour three |
| D14 | Only `backend.py` and the new `training.py` import the training stack under `local_qlora/`, and only inside functions | v3.01 D28, widened by one module |

### Checkpoints, the run, and resuming

| # | Decision | Reason |
|---|---|---|
| D15 | A checkpoint is saved every `checkpoint.save_steps` optimizer steps (100) and at the end, under the run folder. All are kept. Each holds the adapter and the trainer's state | 916 steps give ten checkpoints of about 0.2 GB. Selection (v3.04) needs every one |
| D16 | At each checkpoint the loss on dev is computed and logged | It is cheap and shows overfitting. It selects nothing: selection is by dev execution accuracy (plan §4.6) |
| D17 | **Checkpoints are scored after training, as ordinary evaluation runs** of the predictor of D23, not inside the training loop. **Decision for the owner in section 8** | Section 2.3: scoring costs as much as the training between checkpoints. Apart, each scoring run is resumable, uses the one runner, and a run can be scored on quick-dev first |
| D18 | One invocation is one run in the registry: kind `train` (kind `smoke` for the smoke run), path A, arm A1 (`none` for the smoke run), scope `dev` unless `measured` is asked for. It snapshots `training`, `local_qlora`, `token_budget`, `prompt`, `split`, and `data` | `run_tracking.md` |
| D19 | The run records one ledger line, phase `train`, also when it fails: wall-clock seconds, the number of fit records, the model id, and `training_tokens`, which is the tokens of one pass times the share of the planned steps that were completed, source `computed` | Path A then has the quantity Path B is billed by, for the comparison |
| D20 | The run folder holds `training.json` (what was asked), `training_log.jsonl` (step, epoch, loss, learning rate, dev loss), `measurements.json` (the probe, peak memory, seconds per step), and `checkpoints/`. None of them holds example text | What v3.04, v3.06, and v5.02 read |
| D21 | When training completes, the last adapter is copied to `artifacts/adapters/<run_id>/` with `adapter_metadata.json`: base model and commit, prompt id, split manifest hash, the hash of the training settings, the adapter's SHA-256, steps, epochs, records, whether it is a smoke adapter, and the library versions | Roadmap v3.06. Git-ignored. The metadata is what lets a predictor refuse an adapter made for another experiment |
| D22 | `--resume <run_id>` starts a **new run** that continues from the last checkpoint of an earlier one. It is refused unless the model commit, the prompt id, the split manifest hash, the training settings, and the selection of records are the same. The earlier log lines are copied, and step numbers continue | A run cannot be reopened (v2.03 D19). A long laptop run can be interrupted (v3.06) |

### The tuned predictor

| # | Decision | Reason |
|---|---|---|
| D23 | `scripts/evaluate.py` gains `--adapter RUN_ID[/checkpoint-N]` and the entry `qwen3-4b-tuned` (path A, arm A1, local). The flag is refused for a predictor that takes no adapter, and the entry is refused without it | v3.01 left this to the version that makes adapters |
| D24 | The factory finds the adapter under `artifacts/adapters/<run_id>/` or the run's `checkpoints/checkpoint-N/`, reads the metadata, and refuses unless the base model commit, the prompt id, and the split manifest hash are the current ones. Every refusal is a `ConfigError` before any run exists | As `build_tuned_predictor` does for Nova: a tuned model is asked only about the experiment it was tuned for |
| D25 | The predictor is `QwenPredictor` unchanged. Its `settings["adapter"]` is the hash, and its `details` name the training run, the checkpoint, and the step. A smoke adapter gives the predictor the name `qwen3-4b-tuned-smoke` | The hash decides a resume; the details say which adapter it was. A smoke adapter must never be read as arm A1's result |

### The smoke run

| # | Decision | Reason |
|---|---|---|
| D26 | `--smoke` trains on `smoke.examples` (64) fit records: the longest ones needed to fill one batch, then the first of `fit` in dataset order. `smoke.epochs` is 3 and `smoke.save_steps` is 4: 12 optimizer steps, 3 checkpoints. Everything else is the full run's | "About 64 examples" (roadmap). The same code path, small |
| D27 | After training, the smoke run loads its adapter through `load_backend`, asks `smoke.generate_examples` (24) quick-dev prompts one at a time and again in batches of 8, and reports how many programs run and how many replies are identical. It writes no predictions file and scores nothing | The adapter must reload and generate (roadmap). The batch figure is what v3.01 handed over |
| D28 | The smoke run prints checks K1 to K7 (section 4.4) and exits 0, 1, or 2 | `check_script.md` |
| D29 | During the build Claude runs the smoke run, an interrupted one with its resume, and a partial evaluation of the smoke adapter: about ten minutes of local GPU time, nothing billed | Stated so the owner can object before the PRP runs |

## 4. Scope to build

### 4.1 Config

`config/training.yaml`:

```yaml
version: 1
verified_on: "2026-11-19"
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: all-linear
optimizer:
  name: paged_adamw_8bit
  learning_rate: 2.0e-4
  weight_decay: 0.0
  max_grad_norm: 1.0
schedule:
  type: cosine
  warmup_fraction: 0.05
batch:
  size: 2
  accumulation: 8
epochs: 3                  # section 8, question 1
seed: 42
precision: bf16
gradient_checkpointing: true
checkpoint:
  save_steps: 100
  dev_loss: true
logging_steps: 10
adapters_dir: artifacts/adapters
smoke:
  examples: 64
  epochs: 3
  save_steps: 4
  logging_steps: 1
  generate_examples: 24
  generate_batch_size: 8
  max_final_loss_ratio: 0.5
```

### 4.2 `qlora_finetuning/local_qlora/`

| Module | Holds |
|---|---|
| `training_config.py` | `load_training_config()`: typed values, each range checked, a missing key named. `settings` is the mapping that a resume compares and whose hash the metadata carries |
| `render.py` | Adds `encode_example` (D1, D2) and `encode_records(records, tokenizer, budget)`, which raises `LengthError` naming every record over `max_length` (D4). No training-stack import |
| `plan.py` | `training_plan(lengths, cfg, smoke)`: records, tokens in one pass, optimizer steps, warmup steps, checkpoint steps, and the worst batch. Pure arithmetic, used by the dry run and the run |
| `backend.py` | `load_quantized_model(...)` taken out of `load_backend` (D8) |
| `training.py` | `run_training(...)`: the probe (D13), the trainer (D5, D10), the log, the checkpoints, the ledger record, the adapter and its metadata, resuming (D22). The second module that imports the training stack |
| `adapters.py` | `write_metadata`, `read_metadata`, `resolve_adapter(name)` (D21, D24). No training-stack import |
| `predictor.py` | `build_tuned_predictor(adapter, context)` and `qwen3_4b_tuned(split, context, adapter)` (D23 to D25) |

```python
from qlora_finetuning.local_qlora.training import run_training

result = run_training(smoke=True)            # or run_training(resume="<run_id>")
result["run_id"], result["steps"], result["adapter_dir"]
result["measurements"]["probe_peak_vram_allocated_gib"]
```

### 4.3 Scripts

| Command | Behavior |
|---|---|
| `venv\Scripts\python -m scripts.qlora_train --dry-run` | No GPU and no model: encodes fit with the tokenizer and prints the plan. Starts no run |
| `... --smoke` | The smoke run (D26 to D28) |
| `... ` (no flag) | The full run on this machine. Built here, first used by v3.06 |
| `... --resume <run_id>` | Continues an interrupted run (D22) |
| `... --stop-after-steps N` | Stops as a failure after N optimizer steps: how an interruption is rehearsed |
| `venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b-tuned --adapter <run_id>[/checkpoint-N] --split quick_dev` | Scores an adapter through the runner (D23) |

Exit codes of `qlora_train`: 0 completed (and, for the smoke run, K1 to K7 pass); 1 training
failed, was stopped, or a check failed; 2 a config problem, a record over the length, or a
resume that is refused.

### 4.4 The smoke checks

| Id | Check | Passes when |
|---|---|---|
| K1 | Sequences | Every sequence has the token budget's length for its record, and none is over `max_length` |
| K2 | Labels | In every row, exactly the target tokens and the end token carry a label |
| K3 | Trainer left the data alone | The trainer's dataset has the same rows and lengths as the pipeline's (D5) |
| K4 | Only the adapter trains | Every trainable parameter is a LoRA weight; the line gives their number |
| K5 | Worst batch fits | The probe of D13 gives a finite loss; the line gives its width and peak memory |
| K6 | Loss falls | The mean of the last quarter of logged losses is under `smoke.max_final_loss_ratio` times the first |
| K7 | Adapter reloads | The saved adapter loads through `load_backend`, its hash equals the metadata's, every prompt gets a reply, and no reply holds a thinking block. The line reports the two figures of D27 |

### 4.5 Tests

No test outside the `gpu` marker needs a GPU, the network, or the training stack.

- Config: each missing key; each value out of range; `settings` and its hash stable.
- Encoder, with the stand-in tokenizer: prompt ids then target ids then the end token; the
  labels of D2; a record over the length is named, and nothing is cut. With the real
  tokenizer when cached: lengths equal `count_split` on invented records, the labelled part
  decodes to the target and the end token, and the rest to `render_chat`'s prompt.
- Plan: steps, warmup, checkpoint steps, and the worst batch on known lengths; the full
  plan for 4,884 records; the smoke selection (the longest first, then dataset order).
- Training with a stand-in engine (no model): the run, its files, its ledger record with
  `training_tokens`, a failure half-way with a partial record, metadata written only on
  completion; no example text in any file.
- Resume: continues from the last checkpoint with step numbers and log lines carried on;
  refused for other settings, another prompt, another selection, or a run without a
  checkpoint; starts no run when refused.
- Adapters: metadata round trip; `resolve_adapter` for a run and for a checkpoint; each
  refusal of D24; a smoke adapter renames the predictor.
- Script: `--dry-run` starts no run and prints the plan; exit codes; `--adapter` refused
  for `gold` and required for `qwen3-4b-tuned`; the existing entries behave as before.
- Guards: only `backend.py` and `training.py` name the training stack; nothing under
  `local_qlora/` imports `frontier`; nothing loads the test split.
- `gpu`, on the 0.6B model with invented records: the trainer's dataset equals ours and a
  collated batch has -100 everywhere but the reply; a few steps give a finite loss and
  change only LoRA weights; a stopped run resumes at the right step; the saved adapter
  loads through `load_backend` with the metadata's hash.

### 4.6 Documentation

- Runbook section 5: the dry run, the smoke run and its checks, resuming, scoring an
  adapter, where adapters and checkpoints are, what to delete, troubleshooting (out of
  memory in the probe, a record over the length, a refused resume or adapter).
- `hf_training_stack.md`: section 2.2 of this spec and the trial's figures.
- `gotchas.md`: the trainer's silent `max_length`; `warmup_ratio` and `group_by_length`
  gone; the two open rows on training memory closed with the measured figures.
- `canonical_record.md`: Path A's training line as built. A new pattern note only if a
  second user appears.
- `coding.md`: train only through `run_training`; adapters are named by run id and checked
  against their metadata; `config/training.yaml`; `qlora_train` in the scripts list.
- Architecture overview: the training pipeline as built.

## 5. Out of scope

- A full training run, on this machine (v3.06) or in the cloud (v3.04).
- Scoring every checkpoint and choosing one (v3.04; v3.06 if it comes first).
- SageMaker: the image, the launcher, S3, billable seconds (v3.03).
- Any change of a hyperparameter from plan §4.6 other than the two questions of section 8.
  Tuning comes after a stable first run, one variable at a time (v3.05).
- Merging the adapter into the base weights, or exporting merged weights (req_007).
- Sweeps, other LoRA ranks, other base models (req_007).
- Experiment trackers: `report_to` stays `none`. The run registry is the record.
- The inference batch size for the cloud runs: still the owner's, by v3.03 (section 7).

## 6. What this hands to later versions

| For | What |
|---|---|
| v3.06 | The full run with one command, resumable; `qwen3-4b-tuned --adapter <run>/checkpoint-N` to score checkpoints on quick-dev; about 3.2 hours for 3 epochs on this machine |
| v3.03 | `run_training` and `scripts.qlora_train`, which take their folders from config and the tracking root, so the same code runs in a job; the ledger record to add billable seconds to; the probe, which makes a wrong instance fail at once |
| v3.04 | Ten checkpoints per run, each scorable through the runner; `training_log.jsonl` with dev loss; the adapter metadata to carry to S3 |
| v3.05 | One file of hyperparameters to change one of |
| v5.02 | `training_tokens` and wall-clock seconds for Path A |
| v6.01, v6.02 | `artifacts/adapters/<run_id>/` with its metadata: the adapter package's contents |

## 7. Known limits and follow-ups

- **Memory at 2,048 is close on this card.** The worst batch reserves 12.35 GiB of 16, and
  other programs hold about 2. If more is in use, the probe fails and says so; nothing is
  truncated to make it fit. The fallback of v1.04 section 8 (1,536 and two examples left
  out on both paths) is still the owner's decision and is not taken here.
- **The trial's speed is a laptop's.** The cloud GPU is another card.
- **A resumed run need not equal an uninterrupted one** to the last digit. The trainer
  restores its random state, but nothing here promises equal weights. A resumed run says
  in its metadata which run it continues.
- **The smoke run proves the path, not the result.** 64 examples and 12 steps teach the
  form of a program. No accuracy is claimed from it.
- **The inference batch size is still open.** The trial's tuned adapter gave 22 of 24
  identical replies at batch size 8, against 24 of 40 untuned. The smoke run measures it
  again; the choice for the cloud runs stays with the owner, by v3.03.
- **An adapter slows inference by about a quarter** because it is not merged (14.1 against
  19.5 tokens per second). Scoring estimates should use the tuned figure.
- **Checkpoints and adapters are on one machine**, in git-ignored folders. Copying them to
  S3 is v3.03's.
- **Dev loss is not the selection rule.** It is logged so that a rising loss is visible.
- **The figures of section 2.3 come from a throwaway script.** The built pipeline must
  reproduce the lengths, the trainable-parameter count, and a falling loss; a difference
  is investigated first.

## 8. Decisions for the owner

**Taken as recommended on 2026-11-19**: the owner started the PRP and its execution
without changing either. `config/training.yaml` says 3 epochs, and checkpoints are
scored after training through `scripts.evaluate --predictor qwen3-4b-tuned`. The table
is kept as the record of what was asked.

| # | Question | Recommendation |
|---|---|---|
| 1 | **How many epochs does the first full run train: 3 or 2?** The plan says "2–3". Path B was set to 2 | 3. Checkpoints are kept every 100 steps and the best is chosen on dev, so a third epoch can only add candidates; an earlier checkpoint is still there if the model starts to overfit. It costs about one more hour of GPU time (64 minutes here), not a multiple of a token price as on Bedrock. The value is one line in `config/training.yaml` and can be changed before v3.04 launches |
| 2 | **Score checkpoints after training, as separate evaluation runs (D17), or inside the training loop every 100 steps as the plan words it?** | After. Scoring 499 dev examples takes about as long as the 100 training steps before it, so inside the loop it would double a job's length and tie two failure modes together. Apart, a scoring run can be resumed, runs through the same runner as every other arm, and can cover quick-dev first and dev only for the best few. Selection is still by dev execution accuracy, as the plan requires |

## 9. Acceptance

Code, by Claude:

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
  and without the training stack blocked.
- A test shows that only the reply carries a label, and that a sequence's length is the
  token budget's.
- `--predictor gold` and `--predictor qwen3-4b` behave as before.

On this machine, by Claude (free; D29):

```powershell
venv\Scripts\python -m pytest -m gpu -q
venv\Scripts\python -m scripts.qlora_train --dry-run
venv\Scripts\python -m scripts.qlora_train --smoke
venv\Scripts\python -m scripts.qlora_train --smoke --stop-after-steps 6
venv\Scripts\python -m scripts.qlora_train --smoke --resume <the stopped run>
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b-tuned --adapter <the smoke run> --split quick_dev --limit 24
```

- The dry run prints, for the full run: 4,884 records, 3,254,181 tokens in one pass, the
  optimizer steps, the warmup steps, and the checkpoint steps, and starts no run.
- The smoke run passes K1 to K7. Its trainable parameters are 33,030,144 and its worst
  batch is 1,862 tokens wide, as in section 2.3.
- The stopped run exits 1 with a checkpoint at step 4 and a ledger record for what it
  consumed; the resumed run completes at step 12 and passes its checks.
- The partial evaluation of the smoke adapter completes under the name
  `qwen3-4b-tuned-smoke` and adds no line to `ledger/evaluations.jsonl`.
- An adapter whose metadata names another prompt id is refused with exit 2 and no run.
- Section 11, "As built", is added: the checks, memory, speed, the loss, and the two
  figures of D27. The roadmap and the execution plan are updated.

## 10. Next step

This version is built and run. Next is either v3.06, the optional full run on this
machine (`python -m scripts.qlora_train`, about 3.6 hours, started by the owner or on
the owner's word), or the v3.03 spec (SageMaker job packaging), which needs the
owner's decision on the inference batch size and questions Q8 and Q9.

## 11. As built (2026-11-19)

Built by `prp_003_qlora_training_pipeline_v2.md`. Every run below was made on this machine,
at no cost. **No full training run was made.**

What was run:

- `venv\Scripts\python -m pytest -q`: 2,130 passed (131 new). With the training stack
  blocked: 2,124 passed, 6 skipped. `venv\Scripts\python -m ruff check .`: clean.
- `venv\Scripts\python -m pytest -m gpu -q`: 8 passed (2 new: the trainer's collator leaves
  labels on the reply only; a run on the 0.6B model is stopped, resumed, and its adapter
  reloaded).
- `scripts.qlora_train --dry-run`: 4,884 records, 3,254,181 tokens in one pass, 918
  optimizer steps (306 an epoch), 46 warmup steps, 10 checkpoints, worst batch 1,862 tokens
  wide; no run started.
- The smoke run, a stopped run, its resume, and two partial evaluations of the smoke adapter.
- `scripts.evaluate --predictor gold --split quick_dev --limit 3`: as before.

### The smoke run

Run `20261119T083236Z-smoke-qwen3-4b-smoke-5477`: 64 fit records, 12 optimizer steps, three checkpoints.

| Check | Result |
|---|---|
| K1 Sequences | PASS: 64 sequences of 45,397 tokens, each of the token budget's length; the longest 1,862 of 2,048 |
| K2 Labels | PASS |
| K3 Trainer left the data alone | PASS: 64 rows, lengths and labels unchanged |
| K4 Only the adapter trains | PASS: 33,030,144 trainable parameters, all LoRA |
| K5 Worst batch fits | PASS: 2 sequences, 1,862 tokens wide; peak 11.00 GiB allocated, 12.35 reserved |
| K6 Loss falls | PASS: 2.272 at step 1; 0.279 mean of the last three |
| K7 Adapter reloads | PASS: 24 prompts answered; **24 programs run without an error**; 22 replies identical alone and in batches of 8 |

| Quantity | Value |
|---|---|
| Training time | 168 s: 14.0 s per optimizer step, 0.88 s per sequence |
| Peak GPU memory while training | 8.48 GiB allocated, 9.66 reserved |
| Dev loss at steps 4, 8, 12 (16 dev records) | 0.329, 0.241, 0.236 |
| A checkpoint on disk | About 140 MB (adapter 66 MB, optimizer state, tokenizer files) |

These reproduce the trial of section 2.3: the same lengths, the same parameter count, the
same first loss.

### Interrupting and resuming

- `--smoke --stop-after-steps 6` (run `20261119T084949Z-smoke-qwen3-4b-smoke-56a1`): exit 1, `TRAINING FAILED: stopped on
  request after 6 of 12 steps`, a checkpoint at step 4, and a ledger record for 6 of 12
  steps (68,096 training tokens).
- `--smoke --resume` of it (run `20261119T085248Z-smoke-qwen3-4b-smoke-4eb7`): started at step 4, completed at step 12, all
  seven checks pass (21 of 24 replies identical alone and in batches). One log covers steps
  1 to 12. Steps 5 and 6 were trained twice, and both runs' ledger records say so.

### Scoring the smoke adapter

`scripts.evaluate --predictor qwen3-4b-tuned --adapter 20261119T083236Z-smoke-qwen3-4b-smoke-5477 --split quick_dev --limit 24`
(run `20261119T085831Z-eval-qwen3-4b-tuned-quick-dev-f771`): a partial run under the name `qwen3-4b-tuned-smoke`, arm A1, no line added
to `ledger/evaluations.jsonl`.

| On the first 24 quick-dev examples | Untuned (v3.01, same 24) | After 12 steps on 64 records |
|---|---:|---:|
| Format validity | 22 | 24 |
| Execution accuracy | 0 | **10** |
| Answer accuracy | 6 | 8 |
| Program match | 0 | 7 |

This is 24 examples and a smoke adapter: it shows the path works and that the model takes
to the program language at once. It is not a result and predicts nothing about arm A1.

Checkpoint 4 of the same run was scored the same way on 8 examples (4 right). The refusals
were tried live: no `--adapter`, `--adapter` with `gold`, a run without a finished adapter,
and a copy of the adapter whose metadata named another prompt id. Each was exit 2 with no
run started.

### Differences from sections 3 and 4

- **918 optimizer steps, not 916.** The trainer counts a last, smaller group of batches as
  a step: 306 an epoch. Ten checkpoints, as section 3 says.
- **`smoke.dev_examples: 16` was added to the config**: a smoke run computes the dev loss on
  the first 16 dev records, not on all 499.
- **Training switches the Hugging Face Hub to offline.** Saving an adapter makes PEFT ask
  the Hub for the base model's config; a run must not use the network. PEFT then prints a
  warning and saves correctly.
- **K1 and K2 are settled before a run exists.** A sequence that does not have the token
  budget's length, or labels outside the reply, is a `ConfigError` with exit 2 and no run.
  In a run they are reported as passed.
- **`--stop-after-steps N` stops before step N+1**, so step N is complete: logged, and
  saved if a checkpoint was due.
- **A checkpoint's metadata comes from its run's `training.json`**, so a checkpoint of an
  unfinished run can be scored. Only a completed run has `adapter_metadata.json`.
- The trainer saves at the last step by itself; the pipeline's own final save is a
  fallback that was not needed.
- `adapter_sha256` moved to the new `adapters.py`. `run_training` takes a `TrainingContext`
  and, for tests, a stand-in engine and reloader.
- The trainer's progress bar and its printing of log entries are switched off; the log is
  `training_log.jsonl`.

### What the runs showed beyond the spec

- **Two runs with the same seed are not identical.** The loss at step 3 was 0.953 in the
  first smoke run and 0.971 in the stopped one, before any interruption. An adapter is
  identified by the hash of its weights, not by its settings.
- **Reserved GPU memory reached 13.6 GiB in the resumed run** (8.6 allocated), against 9.7
  in the uninterrupted one. With about 2 GiB held by other programs, a full local run
  leaves very little room on this card.
- **Speed in these runs was 0.88 to 0.96 s per sequence**, against the trial's 0.79: a full
  run of 3 epochs is nearer 3.6 to 3.9 hours here than 3.2.
- **Tuned inference was slower than estimated**: 8.3 generated tokens per second in the
  partial evaluation (5.0 s per example), against 14.1 in the trial and 19.5 untuned.
  Scoring 499 dev examples is then about 40 minutes a checkpoint, not 20, which
  strengthens the case for scoring after training and on quick-dev first.

### What stays open

- The inference batch size for the cloud runs: the owner's, by v3.03. Tuned replies agreed
  in 21 and 22 of 24 at batch size 8.
- Whether the full run fits beside other GPU use on this machine: v3.06, if it is run. The
  probe answers in the first seconds.
- Speed and memory on the cloud GPU: v3.03's smoke job.
