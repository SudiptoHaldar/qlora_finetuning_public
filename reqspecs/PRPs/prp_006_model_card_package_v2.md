# PRP: Model card, adapter package, README (req_006 v6.02)

**Feature Name**: model_card_package (packaging_demo v6.02)  
**Version**: 2  
**Created**: 2026-11-24  
**Status**: ✅ IMPLEMENTED  
**Implemented**: 2026-11-24  
**Implementation Summary**: [impl_006_20261124_051500.md](../impl_summary/impl_006_20261124_051500.md)  
**Confidence Score**: 7/10  
**Estimated Time**: 5–7 hours of code, tests, and documents; about 1.5 hours of this machine (a CPU merge, four dev runs, the cards' code); nothing billed  
**Origin**: `reqspecs/reqs/req_006_packaging_demo/req_006_02_spec.md`  

Template: `prp_base.md` (complex: a new package, a change to the one module that loads the
Qwen model, three predictors, two generated cards, a README, and runs on the GPU).

## Goal

Two folders under git-ignored `artifacts/release/` that can be uploaded as they are: arm
A1's adapter with its card, and a merged copy of the model with its own card. Each is
proved by loading it from its folder and scoring it on dev, and by executing the code its
card prints. The cards and a package record are committed under `reports/packaging/`. The
README is written for a reader who has only the repository.

## Why

- Milestone M9: demo, model card, and README complete.
- v6.03 uploads exactly what this version builds and the owner reviews.
- The owner wants the tuned model to be useful to others: the cards must be runnable and
  must carry the contamination warning.

## What

All eight decisions of spec section 8 are taken (2026-11-24). Those that shape the code:

- The merged copy is withheld when it has 10 or more examples fewer right than R1
  (`noise_examples` of `config/training.yaml`), in 4-bit **or** at full precision.
- The merge runs on this machine's CPU. All dev runs are on this machine.
- Repositories: `sudiptoHaldar/qwen3-4b-finqa-program-qlora` and
  `sudiptoHaldar/qwen3-4b-finqa-program-merged`. Code:
  `https://github.com/SudiptoHaldar/qlora_finetuning_public`.
- The merged copy is scored in 4-bit (R3) **and** at full precision (R4).
- Licence `apache-2.0`; author `sudiptoHaldar`; no email address anywhere.

Success criteria

- [x] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass,
      with and without the training stack blocked.
- [x] `scripts.hf_package adapter` and `merge` write the two folders; check P4 and P5 pass.
- [x] R1, R2, R3, R4 are complete runs on dev (`result: true`), `where: local`, `scope: dev`.
- [x] R2 gives the same reply as R1 on all 499 examples.
- [x] R1 is within 10 examples of the recorded 358 of 499.
- [x] R3 and R4 are inside the allowance, or the merged copy is marked withheld.
- [x] The code of each card runs in an empty directory and its reply executes to the
      sample's `expected`.
- [x] `scripts.hf_package cards` writes both cards and the record; P1 to P10 are reported.
- [x] `comparison_freeze status`: nothing frozen changed. No figure of the cost record or
      of the comparison report changes. No AWS call, no upload.
- [x] README rewritten around the results block, with coloured diagrams; its test passes.

## All Needed Context

| File | Why |
|---|---|
| `reqspecs/reqs/req_006_packaging_demo/req_006_02_spec.md` | D1 to D34, checks P1 to P10, section 8 (decisions), section 8.1 (the public repository the cards may link to) |
| `qlora_finetuning/local_qlora/backend.py` | `load_backend`, `check_ready`, `load_quantized_model`, `QwenBackend`. The only module of Path A that loads the stack for inference. Its tail (padding side, end ids, render check, a whole new `GenerationConfig`) must be shared, not copied |
| `qlora_finetuning/local_qlora/predictor.py` | `QwenPredictor(backend, name, settings, details=)`; `resolve_tuned_adapter("selected", context)` runs every source check (P2, P3) |
| `qlora_finetuning/local_qlora/adapters.py` | `adapter_sha256` hashes the weights file only; `read_selection`; `write_json` (ASCII, LF, no secrets) |
| `qlora_finetuning/local_qlora/config.py` | `LocalQloraConfig.settings(revision, adapter)`: the settings dict a run is matched on. **Do not change its shape**: the freeze compares it |
| `qlora_finetuning/local_qlora/scoring.py` | `evaluation_runs(runs_dir)`: every run with an `evaluation.json` |
| `qlora_finetuning/evaluation/runner.py` | `run_evaluation`, `summarize_run`, `run_predictions(run_id, ctx)` (lines with `example_id`, `reply`, `score["execution_correct"]`), `RunnerContext` |
| `scripts/evaluate.py` | `PREDICTORS`: add three `PredictorEntry` rows (`path="A"`, `arm="none"`, `where="local"`) |
| `qlora_finetuning/comparison/gate.py` | `request_problems`: an arm that is not one of the six is refused on test |
| `qlora_finetuning/comparison/report_build.py`, `report_tables.py` | `fill(template, values)` returns the text and the problems; `table(...)`, `summary_table(record)`, `span`, `CATEGORY_NAMES`, `METRIC_NAMES` |
| `qlora_finetuning/data/serialization.py` | `format_user`, `_one_line`, `_cell`: the card's code must build the same user text, character for character |
| `qlora_finetuning/local_qlora/render.py` | `render_chat`: what the card's code must reproduce with `apply_chat_template(..., enable_thinking=False)` |
| `qlora_finetuning/demo/inputs.py`, `sample_input.json` | The invented sample and its `expected` |
| `reports/comparison/comparison_statistics.json` | `arms[A1].metrics[*].{count,percent,interval}`, `.categories`, `.by_steps`, `.dev`; `costs`; `passed`, `provisional`, `checks` |
| `reports/comparison/cost_model.json` | `making.A1.train` and `.select` (`usd`, `billable_seconds`, `instance_type`), `test_pass.A1`; `passed`, `provisional` |
| `reports/comparison/six_arm_test_runs.json` | `passed`, `checks`, `arms` |
| `reports/local_qlora/adapter_selection.json` | `selected.{adapter, adapter_sha256, step, execution_accuracy, examples}` |
| `artifacts/runs/<run>/training.json`, `artifacts/adapters/<run>/adapter_metadata.json` | Training settings and plan; library versions and trainable parameters |
| `architecture/architecture_overview.md` | Sections 1 and 3: the two diagrams the README carries, recoloured |
| `reqspecs/references/project_patterns/check_script.md`, `evaluation_run.md` | The script and run conventions |
| https://huggingface.co/docs/hub/model-cards, https://huggingface.co/docs/hub/model-card-annotated | The header fields and `model-index` |
| https://huggingface.co/docs/peft/developer_guides/lora#merge-lora-weights-into-the-base-model | `merge_and_unload`; confirm against the installed PEFT 0.21.3 before coding |

### Facts found (2026-11-24)

- **Arm A1 is `artifacts/runs/20261120T182710Z-train-qwen3-4b-fit-4557/checkpoints/checkpoint-500/`**,
  hash `a8c9561b...`. `artifacts/adapters/<run>/` is step 918 (`ed28c7c2...`): never read
  its weights. Its `adapter_metadata.json` is read for `libraries` and
  `trainable_parameters` only, which are the run's.
- The base snapshot holds `tokenizer.json`, `tokenizer_config.json`, `vocab.json`,
  `merges.txt`, three weight files (8.04 GB, bfloat16), `generation_config.json` (which
  samples), and `LICENSE`.
- The recorded dev run of A1 (`...073743Z-eval-qwen3-4b-tuned-dev-fd5b`) is in
  `artifacts/runs/` too, with **the same predictor name and settings** as R1. A run's
  `manifest.json` says `where`; R1 is the newest complete one with `where: local`.
- R1 and R2 have equal settings (same adapter hash). They differ by predictor name.
- `cost.place` puts every `where: local` record in the free category, and `through` is
  taken from records with an amount. So new local runs change no figure. They do raise
  `project.records.no_charge` of the cost record by one each when that record is next
  rebuilt: the spec's "not a byte" is corrected to "no figure" (say so in section 11).
- No test uses the training stack on the CPU today: only `gpu`-marked tests import it.
- Licences read at source on 2026-11-24: Qwen3-4B card and `LICENSE` say `apache-2.0`;
  FinQA (`czyssrs/FinQA`) is MIT, "Copyright (c) 2021 Zhiyu Chen"; the mirror
  `bevaya/FinQA` declares `mit`. **FinTabNet's own page no longer states a licence** (IBM's
  Data Asset eXchange is deprecated); the card of the derived `bsmock/FinTabNet.c` says the
  original is CDLA-Permissive, linking version 1.0. The entry records exactly that, and
  the owner confirms.

### Gotchas

- **Frozen**: nothing in `config/local_qlora.yaml`, `prompt.yaml`, `token_budget.yaml`
  changes. `LocalQloraConfig.settings` keeps its shape. Run `comparison_freeze status`.
- **The training stack is imported inside functions only**, and under `packaging/` only by
  `merge.py`. Tokenizer and cache lookups go through new helpers in `backend.py`.
- **A folder is not a hub id**: `from_pretrained(str(folder), local_files_only=True)`; the
  cache check `weights_missing` does not apply to a folder.
- **The base model's generation settings sample.** The backend replaces them whole. The
  merged copy must be saved with greedy settings, and the card's code for the adapter must
  replace the model's settings whole, as the backend does.
- **Same-settings runs**: find a run by predictor name, settings, split, prompt id,
  manifest hash, reply limit, no limit, `where: local`, and `result: true`.
- **A merged run's settings must name the weights**: the SHA-256 over the weight files,
  and the precision. Otherwise a rebuilt copy would be matched to an old run.
- **The tokenizer files hold arbitrary strings**: the scan for private strings skips them;
  they are checked by bytes against the cache and by fingerprint.
- **Windows console is cp1252**: scripts print ASCII; files are written UTF-8 with LF.
- **`fill` reports a value the template does not use**: build one value set per card.
- **A subprocess for the card's code** gets an empty working directory, `HF_HUB_OFFLINE=1`,
  and the package folder substituted for the repository name. It must not import the
  project.
- **Never commit dataset text.** Cards hold counts, ids of nothing, and the invented sample.
- **GPU memory**: one model at a time. Each dev run is its own process.

## Implementation Blueprint

### Layout

```
config/packaging.yaml
qlora_finetuning/packaging/
  __init__.py          docstring naming req_006
  config.py            load_packaging_config() -> PackagingConfig
  sources.py           the selected checkpoint, the records, the training facts
  files.py             sha256 of a file, a folder's listing, copy by bytes
  adapter_package.py   build_adapter_package(...)
  merge.py             merge_adapter(...): the only module here that loads the stack
  merged_package.py    build_merged_package(...)
  predictors.py        factories: qwen3-4b-package, qwen3-4b-merged, qwen3-4b-merged-full
  scan.py              private strings in a package's text files
  verify.py            find the runs, compare replies, run a card's code
  cards.py             the values and the two builds
  adapter_card.md, merged_card.md, adapter_usage.py.txt, merged_usage.py.txt
  record.py            checks P1 to P10, the record, its Markdown
scripts/hf_package.py  check | adapter | merge | verify | cards | status
```

Deviation from the spec, deliberate: the three factories live in `packaging/predictors.py`,
not in `local_qlora/predictor.py`, so `local_qlora/` does not import `packaging/`.

### backend.py

```python
def cached_files(model_id, revision, names) -> dict[str, Path]      # ConfigError if one is missing
def load_tokenizer_from(folder) -> tokenizer                        # stack inside the function
def load_folder_backend(cfg, budget, model_dir, *, model_id, settings, quantized=True) -> QwenBackend
```

`load_backend` keeps its behaviour. Its tail becomes a private helper both use:

```python
def _prepared(stack, tokenizer, cfg, budget, model_id) -> list[int]:      # end ids; padding side; render check
def _backend(model, tokenizer, stack, *, model_id, settings, budget, end_ids, started, footprint)
```

`load_folder_backend`: stack, CUDA, `config.json` and weights in the folder, tokenizer from
the folder with the pinned fingerprint, then the model 4-bit (`load_in_4bit` as today) or
unquantized in the dtype of `cfg.compute_dtype`.

### Settings and names

```python
PACKAGE, MERGED, MERGED_FULL = "qwen3-4b-package", "qwen3-4b-merged", "qwen3-4b-merged-full"
# package: local_cfg.settings(revision, adapter_sha)            -> equal to R1's
# merged:  the same with adapter=None, plus {"merged": {"weights_sha256", "adapter"}}
# full:    as merged, with "quantization": None and "dtype": "bfloat16"
```

Each factory refuses `split == "test"` with a `ConfigError` before anything loads.

### Merge (CPU)

```python
model = AutoModelForCausalLM.from_pretrained(base, revision=rev, local_files_only=True, dtype=float32)
merged = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False).merge_and_unload(safe_merge=True)
merged.to(bfloat16); merged.generation_config = greedy(end_ids, pad_id)
merged.save_pretrained(out, safe_serialization=True, max_shard_size=cfg.max_shard_size)
```

Confirm each call against the installed versions first (a ten-line trial on a tiny model).

### Verification

```python
find_run(ctx, name, settings, split) -> (run_id, summary) | None        # newest complete local run
compare(reference, other) -> {"differ": n, "right": (a, b), "right_to_wrong": n, "wrong_to_right": n, "ids": [...]}
run_card_code(code, target, workdir) -> {"system", "user", "text", "reply"}   # subprocess, result file
check_card_run(result, sample, prompt, tokenizer, budget, evaluation_cfg) -> list[str]   # problems
```

Card runs are stored in `artifacts/release/card_runs.json` with the SHA-256 of the code
that ran. `cards` embeds the same rendered code; P10 fails if the hashes differ.

### Tasks, in order

1. `config/packaging.yaml`, `packaging/config.py`, `files.py`; tests.
2. `backend.py` helpers and `load_folder_backend`; `pytest -m gpu tests/local_qlora/test_backend_gpu.py` still passes.
3. `sources.py`, `adapter_package.py`, `scan.py`; tests with a stand-in checkpoint.
4. `merge.py` and `merged_package.py`; a CPU test on a tiny model, skipped without the stack.
5. `predictors.py`, the three rows in `scripts/evaluate.py`; gate test.
6. `verify.py`; tests with stub runs.
7. The usage code and the two card templates; `cards.py`; tests.
8. `record.py` and `scripts/hf_package.py`; tests.
9. On the machine: `adapter`, `merge`, R1 to R4, `verify`, the licence entries (the owner
   confirms), `cards`.
10. README with the two recoloured diagrams; its test.
11. Runbook section, `coding.md` rule, architecture, pattern file; spec section 11,
    roadmap, execution plan, implementation summary.

### Error handling

- Every refusal before work is a `ConfigError` printed as one line that names the next
  command. Exit codes of `hf_package`: 0 passed, 1 a check failed, 2 could not start.
- A failed check never stops the record from being written; it says not passed.
- A card is not written while a placeholder is unfilled or a licence entry is unconfirmed.
- A withheld merged copy leaves the adapter package ready.

## Validation Loop

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q
venv\Scripts\python -m pytest -q -p block_stack          # the stack blocked, as in CI
venv\Scripts\python -m scripts.hf_package check
venv\Scripts\python -m scripts.hf_package adapter
venv\Scripts\python -m scripts.hf_package merge
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b-tuned --adapter selected --split dev
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b-package --split dev
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b-merged --split dev
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b-merged-full --split dev
venv\Scripts\python -m scripts.hf_package verify
venv\Scripts\python -m scripts.hf_package cards
venv\Scripts\python -m scripts.hf_package status
venv\Scripts\python -m scripts.comparison_freeze status
venv\Scripts\python -m pytest -m gpu -q tests/packaging
```

Nothing here is billed and nothing calls AWS or uploads. An agent starts no test run.

## Final Checklist

- [x] Tests and lint pass in both modes
- [x] Both folders built; P1 to P10 reported in `reports/packaging/package_record.md`
- [x] Four dev runs complete; R2 equals R1
- [x] Both cards' code executed
- [x] Licence entries confirmed by the owner
- [x] README, runbook, `coding.md`, architecture, pattern file updated
- [x] Spec section 11, roadmap, execution plan, implementation summary

## Anti-Patterns to Avoid

- Building the package from `artifacts/adapters/<run>/`.
- Editing a card, the record, or `adapter_config.json` by hand.
- A second loop over a split, or a score computed outside `run_evaluation`.
- Prompt wording typed into a card: it comes from `config/prompt.yaml`.
- Calling the merged copy arm A1, or running it on test.
- Importing the training stack at module level, or `huggingface_hub`'s upload calls at all.

## Implementation Time Estimate

**5–7 hours** of code, tests, and documents, and about 1.5 hours of machine time.

| Task | Time |
|---|---|
| Config, files, sources, adapter package, scan | 60 min |
| Backend change, merge, merged package, predictors | 75 min |
| Verification and the cards' code | 60 min |
| Card templates, card build, record, script | 75 min |
| Tests | 75 min |
| README with diagrams; runbook and other documents | 60 min |
| Machine: merge 10 min, four dev runs 40 min, cards' code 10 min, suite twice 15 min | 75 min |

## Confidence

**7/10.** The pieces exist and the inputs are on the machine. The risks: the exact
transformers and PEFT calls for a merge and an unquantized load at the pinned versions; a
difference between R2 and R1 from something other than the weights; the full-precision run
fitting beside batches of eight on 16 GB; and the licence entry for FinTabNet, which could
not be re-read at its source.

## Time Taken for PRP Generation

**~25 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 4 min |
| Codebase exploration | 12 min |
| Research existing PRPs and patterns, and the licence sources | 4 min |
| Writing PRP document | 5 min |
