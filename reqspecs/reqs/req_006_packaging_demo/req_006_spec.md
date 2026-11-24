# Packaging Demo

Spec Group      : req_006_packaging_demo  
Feature Name    : packaging_demo  

I will generate multiple versions of PRPs addressing **packaging_demo** items; I would progressively add details on new/existing items in spec files located in this directory and I'll specify which files to read for which PRP version - the different versions of PRPs would be created by the use of '--version' flag.

Read the following files for generating PRPs  
    - [--version 1] | 
        - **req_006_01_spec.md**: Demo CLI
    - [--version 2] | 
        - **req_006_02_spec.md**: Model card, adapter package, README
    - [--version 4] | 
        - **req_006_04_spec.md**: Public repository export

Later versions are listed in `req_006_roadmap.md`; a `[--version N]` entry is added here when its spec file is written.

---

## Requirements collected for v6.02 and v6.03 (Hugging Face publication)

Added 2026-11-23, before either version is specced. These are inputs for `req_006_02_spec.md`
and `req_006_03_spec.md`; nothing here is a locked decision until those files are written.
**`req_006_02_spec.md` was written on 2026-11-24 and replaces the v6.02 parts below**; where
the two differ, the spec file decides.

### What can be published

- Only **arm A1** can go to the Hugging Face Hub: the QLoRA adapter on `Qwen/Qwen3-4B`, at the
  checkpoint the selection record names (`--adapter selected`).
- The tuned Nova Pro and Nova Lite models live inside Bedrock and cannot be exported. They
  appear on the card only as comparison numbers.

### v6.02 — the adapter package

The package is one folder that can be uploaded as it is.

| Item | Requirement |
|---|---|
| Adapter weights | `adapter_model.safetensors` and `adapter_config.json` of the selected checkpoint, pulled from S3 and checked against the recorded hash |
| Left out | Optimizer state, other checkpoints, trainer logs, predictions files, data bundles, anything with an S3 path or an account id |
| Form | **Two packages** (owner's decision, 2026-11-23): the adapter alone, and a merged copy. See "The merged copy" below. A GGUF copy is a possible later addition |
| Split manifest | Published with the adapter (owner's decision, 2026-11-23): the committed manifest file, which holds example ids, the seed, the dataset revision, and the content hash, and no dataset text |
| Tokenizer | The tokenizer files at the commit pinned in `config/token_budget.yaml` |
| Prompt | The system text of `config/prompt.yaml`, word for word, with its prompt id; the note that thinking is off (`enable_thinking=False`) |
| Model card | `README.md` in the folder, as described below |
| Clean-load check | The folder is loaded from an empty directory onto the base model and reproduces the recorded dev score of the selected checkpoint |
| License check | Qwen3 (Apache 2.0), FinQA (MIT), FinTabNet (CDLA-Permissive-1.0) re-read on their cards before the card is written (high-level plan §8) |

### v6.02 — the merged copy

The merged copy is the base model with the adapter folded into its weights, so it loads
without `peft`.

- **Merged into the full-precision base**, at the pinned commit, never into the 4-bit model.
  Saved as `safetensors`, about 8 GB.
- **Its own repository**, recommended: the adapter repository stays small, and each card
  states one thing. The merged card names the adapter repository and the reverse.
- **Its header differs**: `library_name: transformers`, `base_model_relation: finetune`
  (the adapter's is `adapter`). It carries Qwen's weights, so Qwen3's Apache 2.0 license and
  notice go with it.
- **It is not arm A1.** Arm A1 is the adapter on the 4-bit base. The merged copy, loaded in
  4-bit, quantizes different weights and can give a different reply. Before it is published
  it is scored on dev through `run_evaluation`, and its card reports that dev score beside
  the adapter's. The test results on its card are labelled as the adapter's; the merged copy
  is never run on test (the gate allows the six arms only).
- **To decide in the v6.02 spec**: how far the merged copy's dev score may differ from the
  adapter's before it is withheld, and where the merge runs (the laptop's 16 GB GPU, the CPU,
  or a SageMaker job).

### v6.02 — the model card

YAML header: `base_model: Qwen/Qwen3-4B`, `library_name: peft`, `license: apache-2.0`,
`datasets`, `pipeline_tag: text-generation`, tags, and a `model-index` block holding the test
results. The split in `model-index` is named as this project's own, not the official test split.

| Section | Contents |
|---|---|
| Summary | Report excerpt, table, and question in; `PROGRAM` and `ANSWER` out; the program is run by an external executor, the model is never the calculator |
| How to use | A runnable snippet: base model in 4-bit, adapter attached, the system text, greedy decoding, the reply limit of `config/token_budget.yaml` |
| Training data | Dataset id and pinned revision; the 65% / 35% split grouped by company; the seed; fit, dev, and test counts; the manifest hash |
| Training procedure | LoRA rank, alpha, dropout, and target modules; 4-bit NF4 settings; batch size and accumulation; learning rate and schedule; epochs; training length; the selected step; library versions |
| Results | The four metrics on the 2,898 test examples with confidence intervals; all six arms in one table, the four Nova arms included (owner's decision, 2026-11-23), with arm A1 marked as the published model |
| Cost | Instance type, training time, training cost, cost per 1,000 questions |
| Limitations | Oracle-evidence setting, so the numbers are not comparable to published FinQA results; a research artifact, not financial advice; the failure categories of the error analysis |
| Citation and licenses | The FinQA paper, Qwen3, the three licenses |

Rules the card must follow:

- **Contamination warning.** The training pool holds examples from the official FinQA
  validation and test splits. The card says so plainly: a score of this adapter on the official
  test split is inflated. It points to the split manifest.
- **No dataset text.** The card and the package hold counts and example ids only. The usage
  snippet uses an invented example or the demo's bundled sample, never a FinQA row.
- **Numbers come from records, not from memory**: test results from
  `reports/comparison/six_arm_test_runs.json`, the checkpoint from the selection record, costs
  from the ledger and `config/pricing.yaml`.

### v6.03 — publishing

- Needs a Hugging Face account and a **write token**. The token goes in `.env`, never in a
  committed file.
- Upload with `hf upload <user>/<repo> <folder>`, once for the adapter and once for the
  merged copy, **private first**; the owner reviews each rendered card and then makes the
  repository public.
- Publishing is outward-facing and is the owner's step (🟡 operator-gated).
- After upload: load the adapter from the Hub into an empty directory and repeat the
  clean-load check.

### Owner's decisions (2026-11-23)

| Question | Decision |
|---|---|
| Adapter only, or also a merged copy? | A merged copy as well |
| Do the Nova arms appear in the card's results table? | Yes |
| Is the split manifest published with the adapter, or linked from the code repository? | Published with the adapter |

### Questions still open (to settle in the version specs)

1. Repository names, and the account or organization they go under.
2. One repository for the merged copy, or a folder inside the adapter's repository
   (recommended: its own repository).
3. The allowed difference between the merged copy's dev score and the adapter's.
4. Where the merge runs.

### Reference cards

Listed from memory on 2026-11-23 and not opened; confirm each before relying on it.

- `HuggingFaceH4/zephyr-7b-beta` — a fine-tune card with hyperparameters, results, and a `model-index` block.
- `tloen/alpaca-lora-7b` — an adapter-only repository, showing the smallest file set.
- `FinGPT/fingpt-mt_llama2-7b_lora` — a finance LoRA adapter.
- `Qwen/Qwen3-4B` — the base model's card, for the usage snippet and the thinking-mode wording.
- Hugging Face documentation: `https://huggingface.co/docs/hub/model-cards` and
  `https://huggingface.co/docs/hub/model-card-annotated`.
