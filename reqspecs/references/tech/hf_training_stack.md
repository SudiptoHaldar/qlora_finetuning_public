# Hugging Face training stack

**Last verified**: 2026-11-19 on Windows 11, Python 3.13.7, RTX 4090 Laptop GPU (16 GB)  
**Sources**: the installed packages themselves (signatures read with `inspect`), and
`scripts/env_check.py --smoke`, which exercises every call below.

These versions are newer than both planning guides. Where a guide and this file disagree, this
file describes what is installed.

## Versions

Pinned in `requirements-train.txt`:

| Package | Version | Notes |
|---|---|---|
| torch | 2.14.1 (`+cu126` locally, CUDA 12.6) | The CUDA build is installed first from the PyTorch index |
| transformers | 5.23.0 | Needs Python 3.11 or later |
| datasets | 5.2.0 | |
| peft | 0.21.3 | |
| trl | 1.18.0 | Needs Python 3.11 or later |
| accelerate | 1.15.0 | |
| bitsandbytes | 0.50.2 | Ships a `win_amd64` wheel; works on native Windows |
| tokenizers, safetensors, huggingface_hub | 0.23.3, 0.8.0, 2.4.0 | |

## Calls that work

4-bit load:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id, revision=revision, device_map={"": 0},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    ),
)
```

Prompt with thinking disabled:

```python
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
```

LoRA on a 4-bit model:

```python
model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM",
))
```

Optimizer: `bitsandbytes.optim.PagedAdamW8bit(params, lr=...)`.

## Qwen3 chat template

- Thinking is **on by default**: without `enable_thinking=False` the output starts with
  `<think>`.
- `enable_thinking=False` works by appending an empty block, `<think>\n\n</think>\n\n`, to the
  prompt after the assistant marker. The model then answers directly.
- Consequence for training: the target text must follow that empty block, so training prompts
  and inference prompts are built the same way. Check for thinking in the **output**, not in
  the prompt.
- `eos_token` is `<|im_end|>`; `pad_token` is `<|endoftext|>`.

## Tokenizer of the base model

- `Qwen/Qwen3-4B` at commit `1cfa9a7208912126459214e8b04321603b3df60c` (pinned in `config/token_budget.yaml`). Its vocabulary
  (151,669 entries) and chat template are identical to `Qwen/Qwen3-0.6B`'s, so the smoke model
  tokenizes exactly as the real one.
- Load without weights: `AutoTokenizer.from_pretrained(model_id, revision=..., local_files_only=True)`.
- Batch count: `tokenizer(texts, add_special_tokens=False)["input_ids"]`.
- With `enable_thinking=False` the chat template adds 17 tokens around a system and a user
  text, the same for every prompt.
- The project's system text is 372 tokens. Lengths of real prompts are in
  `reports/data/finqa_token_budget_report.md`.

## TRL 1.18.0

- `DataCollatorForCompletionOnlyLM` does not exist.
- **How the project trains (req_003 v3.02)**: it builds each sequence itself and gives
  `SFTTrainer` a dataset that already has `input_ids` and `labels`. The trainer then
  tokenizes nothing and leaves the labels as they are; its collator pads on the right and
  pads labels with -100. `completion_only_loss` and `assistant_only_loss` are not used.
- **`max_length` defaults to 1,024 and is applied to a prepared dataset too**: every row is
  cut to it and a row left without a label is dropped, with no error. Pass
  `max_length=None`. The pipeline also compares the trainer's dataset with its own before
  the first step.
- `warmup_ratio` and `group_by_length` no longer exist; warmup is `warmup_steps`.
- `SFTConfig` defaults that differ from the plan: `optim` `adamw_torch_fused`,
  `learning_rate` 2e-5, `lr_scheduler_type` `linear`, `save_steps` 500. `gradient_checkpointing`
  is `True`, `max_grad_norm` 1.0, `seed` 42, `report_to` `none`, `eval_strategy` `no`.
- Step arithmetic of the trainer: `batches = ceil(records / batch)`, `steps_per_epoch =
  ceil(batches / accumulation)`, `max_steps = ceil(epochs * steps_per_epoch)`. 4,884 records
  at 2 by 8 for 3 epochs are 918 steps.
- With `save_strategy="steps"` the trainer also saves at the last step.
- A checkpoint holds `adapter_model.safetensors`, `adapter_config.json`, `optimizer.pt`,
  `scheduler.pt`, `rng_state.pth`, `trainer_state.json`, `training_args.bin`, and the
  tokenizer files: about 140 MB for the 4B model's adapter.
- `trainer.train(resume_from_checkpoint=<folder>)` works on Windows from a checkpoint in
  another folder than `output_dir`.
- The trainer prints every log entry to standard output and draws a progress bar: remove
  `PrinterCallback` and `ProgressCallback` and pass `disable_tqdm=True`.
- **Saving an adapter makes PEFT ask the Hub** for the base model's config. Set
  `huggingface_hub.constants.HF_HUB_OFFLINE = True` for the run; PEFT then warns and saves.
- Qwen's chat template keeps the empty think block before a finished reply when
  `enable_thinking=False`, so the trainer's own conversational path would also train on the
  inference prompt. It ends the sequence with a line break after the end token, one token
  more than the token budget counts.
- The sequence-length field is `max_length`, not `max_seq_length`. The project's value is
  2,048, from `config/token_budget.yaml`, enforced by the pipeline and never by cutting.

## Measured on this machine (Qwen/Qwen3-4B, training, 2026-11-19)

From the smoke runs of req_003 v3.02 (`scripts.qlora_train --smoke`): LoRA r 16 on all linear
layers, batch 2 by 8, gradient checkpointing, 64 records.

| Quantity | Value |
|---|---|
| Trainable parameters | 33,030,144 (a 66 MB adapter file) |
| GPU memory after load, ready to train | 3.34 GiB |
| One step on the two longest records (1,862 tokens wide) | Peak 11.00 GiB allocated, 12.35 reserved |
| Peak during training | 8.5 GiB allocated; 9.7 to 13.6 reserved |
| Speed | 14.0 to 15.3 s per optimizer step of 16 sequences; 0.88 to 0.96 s per sequence |
| Loss | 2.27 at step 1, about 0.28 at step 12; dev loss 0.33, 0.24, 0.24 at steps 4, 8, 12 |
| Two runs with the same seed | Not identical: the loss at step 3 was 0.953 in one and 0.971 in another |
| Inference with the adapter (not merged) | 8 to 14 generated tokens per second, against 19.5 untuned |
| Replies identical at batch size 1 and 8, tuned | 21 or 22 of 24 (untuned: 24 of 40) |

## Measured on this machine (Qwen/Qwen3-0.6B)

| Quantity | Value |
|---|---|
| Model footprint in 4-bit | 0.49 GiB |
| Trainable parameters, LoRA r=16 on all linear layers | 10,092,544 |
| Peak VRAM for load, one generation, and one training step | 1.2 GiB |
| Smoke time | About 35 s on the first run, 5 s once the model is in the disk cache |

These numbers do not predict the 4B model's memory use; the next section has it.

## Measured on this machine (Qwen/Qwen3-4B, inference, 2026-11-19)

From the A0 runs of req_003 v3.01 (`qlora_finetuning/local_qlora/backend.py`), untuned model,
greedy decoding, reply limit 192, prompts of 457 to 886 tokens.

| Quantity | Value |
|---|---|
| Download | 8.04 GB (three safetensors files, BF16) |
| Load time from the disk cache | 6 to 7 s |
| Model footprint in 4-bit | 2.42 GiB |
| Peak VRAM allocated, batch size 1 | 2.74 GiB (2.90 reserved) |
| Peak VRAM allocated, batch size 8 | 3.83 GiB (4.55 reserved) |
| Speed, batch size 1 | 19.5 generated tokens per second; 4.1 s per example |
| Speed, batch size 8 | 58 generated tokens per second; 1.3 s per example |
| Replies that differ between batch size 8 and 1 | 16 of 40 |
| Replies that differ between two runs at batch size 1 | 0 of 40 |

## Generating in batches

```python
tokenizer.padding_side = "left"          # the default is "right", which is wrong for generating
model.generation_config = GenerationConfig(
    do_sample=False, num_beams=1,
    eos_token_id=[im_end_id, endoftext_id], pad_token_id=tokenizer.pad_token_id,
)
batch = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
with torch.inference_mode():
    output = model.generate(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"], max_new_tokens=192)
new_tokens = output[:, batch["input_ids"].shape[1]:]
```

- **The model's own `generation_config.json` samples**: `do_sample: true`, temperature 0.6,
  top-k 20, top-p 0.95. In transformers 5.23 a `GenerationConfig` passed to `generate()` has
  every value that is `None` filled from the model's own config, so passing
  `temperature=None` does not unset anything. Replace `model.generation_config` whole.
- Passing a `generation_config` and generation keyword arguments together is deprecated.
- **`<think>` and `</think>` are not special tokens** (ids 151667, 151668):
  `decode(..., skip_special_tokens=True)` leaves them in the text.
- `decode` with `skip_special_tokens=True` returns ordinary text exactly as written: spaces
  before punctuation and doubled spaces survive (tested).
- A row that ends early is padded with `<|endoftext|>`, which is also an end token: cut a row
  at its first end token.
- A row's prompt length is `attention_mask.sum()`, not the width of the batch.
- With left padding and a mask the arithmetic is the same as for one prompt, but in BF16 a
  near tie between two tokens can fall the other way: replies depend on the batch.
- The model card says "DO NOT use greedy decoding" for thinking mode and suggests temperature
  0.7 and top-p 0.8 for non-thinking mode. The project decodes greedily (plan 4.6); the
  untuned model then repeats itself to the reply limit in about 9% of replies.
- Is a model in the cache? `huggingface_hub.try_to_load_from_cache(repo, filename,
  revision=...)` returns a path or `None`. A large model has `model.safetensors.index.json`,
  whose `weight_map` lists the files; a small one has a single `model.safetensors`.
- An adapter: `PeftModel.from_pretrained(model, folder, is_trainable=False)`, not merged.
  `LoraConfig(init_lora_weights=False)` gives random, non-zero adapter weights, for a test.

## Warnings that can be ignored

- `triton not found; flop counting will not work`: printed by torch at import on Windows.
- `gradient_checkpointing_enable is deprecated ... Use activation_checkpointing_enable`: peft
  0.21.3 still calls the old name. Transformers removes it in version 6.
- `use_cache=True is incompatible with gradient checkpointing. Setting use_cache=False`:
  transformers turns the KV cache off by itself during training.

## Other notes

- Hugging Face libraries log every HTTP request at INFO on the `httpx2` logger. Entry points
  hold that logger at WARNING.
- The model cache is at the default location, `%USERPROFILE%\.cache\huggingface\hub`. To move
  it set `HF_HOME` in the shell; `.env` is not read by these libraries.
- Pin a model with `revision=<full commit hash>`; the hash is in `config/local_env.yaml`.

## Documentation

- https://huggingface.co/docs/transformers
- https://huggingface.co/docs/peft
- https://huggingface.co/docs/trl
- https://huggingface.co/docs/bitsandbytes
- https://huggingface.co/Qwen/Qwen3-0.6B
