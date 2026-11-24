# Packaging Demo

Spec Group      :   req_006_packaging_demo  
Feature Name    :   packaging_demo  
Version         :   2  
Status          :   ✅ Built and verified 2026-11-24 (section 11): both packages ready, P1 to P10 pass; committed (M9)  
Last updated    :   2026-11-24  

## 1. v6.02 — Model card, adapter package, README

The comparison is finished (M8). This version turns its Path A result into things another
person can pick up and use without this repository's history:

1. **The adapter package**: one folder holding arm A1's adapter, the tokenizer, the prompt,
   the split manifest, the training facts, and a model card. It can be uploaded as it is.
2. **The merged copy**: the base model with the adapter folded into its weights, in a
   second folder with its own card. It loads without `peft`.
3. **Proof that both load and answer**: each folder is loaded from disk and scored on dev,
   and the code printed on each card is executed.
4. **The README of the code repository**: what the project is, how to set it up, how to
   reproduce it, the results, the licences.

Nothing is uploaded. Publishing is v6.03 and is the owner's step. Nothing is billed and no
AWS call is made: every step runs on the owner's machine.

**Depends on**: v5.03 (the comparison report; complete 2026-11-24, M8) and v6.01 (the demo;
complete 2026-11-23). Reaching this version is milestone **M9**.

## 2. What exists today (verified 2026-11-24)

### 2.1 The adapter

| Fact | Value |
|---|---|
| Arm A1 | `20261120T182710Z-train-qwen3-4b-fit-4557/checkpoint-500`, named by `reports/local_qlora/adapter_selection.json` |
| Its files, on this machine | `artifacts/runs/<run>/checkpoints/checkpoint-500/`: `adapter_model.safetensors` (66,127,776 bytes) and `adapter_config.json` (1,169 bytes) |
| Its hash in the selection record | `a8c9561b4fc6e560a53d5104c0118e5d088fcfbb579b57b8bfb24714c559902b` |
| LoRA | rank 16, alpha 32, dropout 0.05, all linear layers (seven module names in `adapter_config.json`), 33,030,144 trainable parameters |
| Training | 4,884 records, 3 epochs, 918 steps planned, batch 2 with accumulation 8, learning rate 0.0002, cosine schedule, seed 42; step 500 selected on dev |
| Libraries | torch 2.14.1, transformers 5.23.0, peft 0.21.3, trl 1.18.0, bitsandbytes 0.50.2 (`requirements-train.txt`) |

**`artifacts/adapters/<run>/` is not arm A1.** That folder holds the adapter of the last
step, 918 (hash `ed28c7c2...`), and its `adapter_metadata.json` says `steps: 918`. The
package is built from the checkpoint folder, and writes its own metadata.

`adapter_config.json` names the base model as `Qwen/Qwen3-4B` and leaves its `revision`
empty. The commit is pinned in `config/token_budget.yaml`
(`1cfa9a7208912126459214e8b04321603b3df60c`).

### 2.2 The base model and the tokenizer

- The base model is in the local Hugging Face cache at the pinned commit: three weight
  files, 8.04 GB together, stored in bfloat16; `tokenizer.json`, `tokenizer_config.json`,
  `vocab.json`, `merges.txt`; `config.json`; `generation_config.json`; Qwen's `LICENSE`.
- **The base model's own generation settings sample** (temperature 0.6). The project
  replaces them whole with greedy decoding and its two end tokens (`backend.py`).
- The tokenizer's fingerprint is recorded in `config/token_budget.yaml`.
- The machine: 64 GB of memory, a 16 GB GPU, 1,102 GB free on disk.

### 2.3 The records the cards quote

| Record | Holds |
|---|---|
| `reports/local_qlora/adapter_selection.json` | The selected step, its hash, its dev counts: 358 of 499 right on execution, in a SageMaker job at batch size 8 |
| `reports/comparison/six_arm_test_runs.json` | The six test runs and their counts: A1 2,121 of 2,898 |
| `reports/comparison/comparison_statistics.json` | The intervals, the tests between arms, the error categories |
| `reports/comparison/cost_model.json` | Reconciled with the bill: training $5.88, selection $1.94, $0.346 per 1,000 questions as a batch job |
| `splits/finqa_split_manifest.json` | Example ids, the seed, the dataset commit, the content hash; 312,556 bytes; no dataset text |
| `config/prompt.yaml` | The system text and the prompt id |
| `qlora_finetuning/demo/sample_input.json` | The invented sample, with its expected result |

### 2.4 Code that is reused

| Piece | Where | Used for |
|---|---|---|
| `build_predictor(adapter_dir)` | `local_qlora/predictor.py` | Loads the 4-bit base with an adapter **from a folder**: the loader the package check needs already exists |
| `load_backend`, `load_quantized_model` | `local_qlora/backend.py` | The one place the Qwen model is loaded. It takes a model id and a commit today, not a folder of weights |
| `run_evaluation` | `evaluation/runner.py` | Every dev score in this version |
| `adapter_sha256`, `read_selection`, `check_adapter` | `local_qlora/adapters.py` | The source checks |
| `answer(predictor, demo_input)` | `demo/answer.py` | Running a card's code on the sample |
| The report's template-and-placeholder method | `comparison/report_template.md`, `report_tables.py` | The cards are built the same way |

### 2.5 What is missing

- No package folder, no card, no merge code, no record of what a package holds.
- No way to load the merged copy: `load_backend` reads the pinned model from the cache.
- The README has the demo and the generated results block, and nothing else.
- **The recorded dev score cannot be reproduced exactly on this machine.** It was measured
  on another GPU (an A10G, in SageMaker). A reply can change with the GPU and with the
  batch. So "the package reproduces 358 of 499" is not a check that can pass or fail
  cleanly here. Section 3 replaces it with one that can.
- **Account identifiers are in committed files**: the AWS account number is in 43 tracked
  files (`config/aws.yaml`, the ledger, sixteen report files, the runbook, the command log,
  tests) and in git history. Two of them are frozen records. The repository on GitHub is
  private today.

## 3. Locked decisions

### What is packaged

| # | Decision | Reason |
|---|---|---|
| D1 | The adapter package is built from the checkpoint the selection record names, and only after its hash equals the record's `adapter_sha256` | Arm A1 is that checkpoint and nothing else. The folder under `artifacts/adapters/` is another adapter |
| D2 | The two adapter files go into the package byte for byte. Nothing in `adapter_config.json` is edited, the empty `revision` included | An edited file is no longer the adapter that was scored. The card pins the commit |
| D3 | The adapter package holds exactly: the two adapter files; the four tokenizer files; `prompt.json` (system text, prompt id, the reply format, the note that thinking is off); `finqa_split_manifest.json`; `training_metadata.json`; `sample_input.json`; `LICENSE`; `README.md` (the card) | The requirements collected on 2026-11-23, and the owner's decision to publish the manifest |
| D4 | `training_metadata.json` is written by this version from the training run's `training.json`, the selection record, and the library versions. It holds no job name, no S3 path, no account number, no machine path | The run's own metadata file describes step 918 and was never meant to leave the machine |
| D5 | Left out of both packages: optimizer state, other checkpoints, trainer logs, predictions, data bundles, serialized records, the ledger | Size, and dataset text |
| D6 | The packages are written under `artifacts/release/adapter/` and `artifacts/release/merged/`, which git ignores. The two cards and the package record are also committed, under `reports/packaging/` | Weights do not belong in git. The cards must be reviewable, and v6.03 uploads exactly the text that was reviewed |

### The merged copy

| # | Decision | Reason |
|---|---|---|
| D7 | The adapter is merged into the **full-precision** base model at the pinned commit, never into the 4-bit model | A 4-bit model cannot be merged without loss; the collected requirement |
| D8 | The merge is computed in 32-bit floats and saved in the base model's own type (bfloat16), as `safetensors`, in files no larger than the base model's | The sum is formed before it is rounded. The copy is the same size as the base: about 8 GB |
| D9 | The merged copy ships a `generation_config.json` that decodes greedily and stops at the project's two end tokens, in place of the base model's sampling one | Otherwise a plain `generate()` on the copy samples, and gives replies the project never measured |
| D10 | The merged copy ships the tokenizer files, Qwen's `LICENSE` copied from the cached snapshot, a `NOTICE` saying what was changed and by which adapter, `prompt.json`, `sample_input.json`, and its own card | It carries Qwen's weights, so Apache 2.0's terms on notices and stated changes apply to it |
| D11 | **The merged copy is not arm A1** and is never called that. It is never run on test: the gate allows the six arms only. Its card gives its own dev score, and labels every test figure as the adapter's | Loaded in 4-bit, it quantizes other weights than the ones the adapter was trained against |
| D12 | The merged copy is withheld when its dev score falls outside the allowance of owner's decision 1. The adapter package does not depend on it | A copy that answers worse must not be published beside figures it did not earn |

### Proving a package works

| # | Decision | Reason |
|---|---|---|
| D13 | Three runs on dev (499 examples), on this machine, each through `run_evaluation`, at the frozen inference settings (batch size 8): **R1** the selected adapter by name, as the reference; **R2** the adapter package, loaded from its folder; **R3** the merged copy in 4-bit, loaded from its folder | One machine and one batch size, so the three are comparable with each other |
| D14 | **R2 must give the same reply as R1 for every example.** One differing reply fails the check | Same weights, same machine, same batch: any difference is a fault in the package or its loading |
| D15 | R1's score is reported beside the recorded 358 of 499. They must be closer than `noise_examples` of `config/training.yaml` (10). They need not be equal | Two GPUs. The project's own measure of "not told apart" |
| D16 | R3 is compared with R1: examples right in each, the number of replies that differ, and how many of those changed from right to wrong and back | The card states what merging cost or gained |
| D17 | The three runs are tracked runs, recorded `where: local`, `scope: dev`. None is an arm, none enters the comparison, and none is given a price. The cost record and the comparison report must not change by a byte | The rule on local runs. The ledger gains three free lines |
| D18 | The two new predictors, `qwen3-4b-package` and `qwen3-4b-merged`, are refused on test by the gate. A test of the suite proves it | Neither is one of the six arms |
| D19 | **The code on each card is executed**, in a new process started in an empty directory, with the package folder in place of the repository name. It must load, answer the invented sample, and its reply must run in `run_reply` to the sample's `expected` | A usage snippet nobody ran is the commonest fault of a model card. It also proves the package's own tokenizer files |

### The cards

| # | Decision | Reason |
|---|---|---|
| D20 | A card is built, never edited: a template with placeholders, filled from the records of section 2.3 and the package record. A missing figure stops the build | As the comparison report. No figure is typed |
| D21 | The adapter card's header: `base_model: Qwen/Qwen3-4B`, `base_model_relation: adapter`, `library_name: peft`, `pipeline_tag: text-generation`, the licence, the dataset, tags, and a `model-index` with arm A1's four metrics on test. The split is named as this project's own, by company, not the official test split | The Hub reads these fields |
| D22 | The merged card's header differs: `library_name: transformers`, `base_model_relation: finetune`. Its `model-index` holds its own dev figures only | D11 |
| D23 | Sections of both: summary; how to use; training data; training procedure; results; cost; limitations; who holds the other models; citation and licences | The collected requirements |
| D24 | Results: all six arms in one table from the comparison's record, with intervals, arm A1 marked as the published model. The four Nova arms are named as hosted models that cannot be downloaded | Owner's decision of 2026-11-23 |
| D25 | **The contamination warning is on both cards, near the top.** The training pool holds examples from the official FinQA validation and test files. A score of this model on the official test split is inflated and must not be reported. The manifest says which examples were used for what | Without it the published model would quietly poison someone else's result |
| D26 | Limitations state: oracle evidence, so the figures are not comparable with published FinQA results; one training run; the failure categories and the fall in accuracy with the number of steps (81% on one step, 43% on three or more); the reply is a program to be run by an executor, and the model's own answer line is right less often than its program (66.98% against 73.19%); research artifact, not financial advice | The caveats of the comparison report, for a reader who has only the card |
| D27 | The card says how the measured figures were produced and that another setup may answer differently: 4-bit NF4, greedy, thinking off, batches of 8, one GPU type | The demo already answers a few questions differently at batch size 1 |
| D28 | A card holds no dataset text, no AWS account number, ARN, S3 path, bucket name, job name, machine path, or email address. Models are named by arm and base model | Public material. The usage example is the invented sample |
| D29 | The usage code pins the base model's commit, loads in 4-bit with the project's values, passes the system text of `prompt.json`, switches thinking off, decodes greedily with the reply limit of 192 tokens, and shows the reply being run by an executor, not trusted | It is the configuration that was measured |

### Licences

| # | Decision | Reason |
|---|---|---|
| D30 | Before a card is built, the licence of Qwen3-4B, of the FinQA dataset, and of FinTabNet is read again at its source. What each says, where, and on which date go into `config/packaging.yaml`. The owner confirms the entry | The plan's section 8. A licence can change; the card states a date |
| D31 | A card cannot be built on a licence entry that is unconfirmed or older than 30 days | The check is worth nothing if it can go stale unnoticed |

### The README

| # | Decision | Reason |
|---|---|---|
| D32 | The README is written by hand, around the generated results block, which is not touched. Sections: what the project asks and found; results (the block); the demo; the published model; setup; reproducing each stage, with its command and what it costs; the repository's layout; limits; licences and citation | It is prose. The figures in it already come from the report's build |
| D33 | A test reads the README: every `python -m` command it names is a module that exists, every file it links to exists, and the two markers are in place | A README rots first |
| D34 | The README says which steps are billed and that they need the owner's own AWS account. It names no account number | A reader cannot rerun the cloud steps in this account |

## 4. Scope to build

### 4.1 New code

A new package, `qlora_finetuning/packaging/`:

| Module | Holds |
|---|---|
| `config.py` | `load_packaging_config()`: reads `config/packaging.yaml` |
| `sources.py` | Finds and checks what a package is built from: the selected checkpoint and its hash, the tokenizer files and their fingerprint, the records of section 2.3 |
| `adapter_package.py` | Writes the adapter package folder (D1 to D5) |
| `merge.py` | The merge (D7 to D10). The only module here that imports the training stack, and only inside functions |
| `verify.py` | Compares the three runs (D14 to D16); runs a card's code in an empty directory (D19) |
| `scan.py` | Looks through every text file of a package for what D28 forbids |
| `cards.py`, `adapter_card.md`, `merged_card.md` | The two templates and what fills them |
| `record.py` | The package record and checks P1 to P10 |

Changes outside it:

- `local_qlora/backend.py`: `load_backend` can load a model from a folder of weights. The
  quantization, the decoding, and the chat wrapping are the same code as today.
- `local_qlora/predictor.py`: two factories, `qwen3-4b-package` and `qwen3-4b-merged`, each
  reading its folder from `config/packaging.yaml`.
- `scripts/evaluate.py`: the two names are accepted, for dev only.
- `config/packaging.yaml`: the package folders, the allowance of decision 1, the licence
  entries, the repository names, the forbidden patterns, the list of files in each package.

No frozen value changes. `config/local_qlora.yaml` is read, not edited.

### 4.2 The commands

```powershell
venv\Scripts\python -m scripts.hf_package check      # P1 to P3, P7: what a build needs; writes nothing
venv\Scripts\python -m scripts.hf_package adapter    # write artifacts/release/adapter/
venv\Scripts\python -m scripts.hf_package merge      # write artifacts/release/merged/ (minutes, CPU)
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b-tuned --adapter selected --split dev   # R1
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b-package --split dev                    # R2
venv\Scripts\python -m scripts.evaluate --predictor qwen3-4b-merged --split dev                     # R3
venv\Scripts\python -m scripts.hf_package verify     # compare R1, R2, R3; run the cards' code
venv\Scripts\python -m scripts.hf_package cards      # build both cards and the package record
venv\Scripts\python -m scripts.hf_package status     # what is built, what passed, what is left
```

Every command reads only. None calls AWS or the Hugging Face Hub. `adapter`, `merge`, and
`cards` write under `artifacts/release/` and `reports/packaging/` and nowhere else.

### 4.3 Checks

| # | Check | Passes when |
|---|---|---|
| P1 | Records in place | The selection record, the six-arm record (C1 to C8), the cost record (K1 to K8, not provisional), and the comparison's record (R1 to R8, not provisional) are present and passed |
| P2 | The source is arm A1 | The checkpoint's two files are on the machine and their hash is the selection record's |
| P3 | Identity | The adapter's model commit, prompt id, and manifest hash are today's (`check_adapter`) |
| P4 | Package contents | Each package holds the files `config/packaging.yaml` lists and no other; the adapter files are byte for byte the source's |
| P5 | Nothing private | No text file of either package matches a forbidden pattern (D28); the sample is the demo's file, byte for byte |
| P6 | Tokenizer | The package's tokenizer files give the fingerprint of `config/token_budget.yaml` |
| P7 | Licences | Three entries, each confirmed, none older than 30 days |
| P8 | The adapter package answers | R1 and R2 are complete runs over all 499 dev examples; every reply of R2 equals R1's; R1 is within the noise of the recorded score; the adapter card's code ran and gave the sample's expected result |
| P9 | The merged copy answers | R3 is a complete run; its difference from R1 is inside the allowance; the merged card's code ran and gave the expected result |
| P10 | The cards | Each is the template filled from today's records, with no placeholder left; the copy in the package and the committed copy are the same bytes; the repository names are set |

The adapter package is **ready** when P1 to P8 and P10 pass. The merged copy is **ready**
when P9 passes as well, and **withheld** when it does not; a withheld copy gets no card
and is named as withheld in the adapter's card and in the roadmap.

### 4.4 What is written

| File | Committed | Holds |
|---|---|---|
| `artifacts/release/adapter/`, `artifacts/release/merged/` | no | The two packages |
| `reports/packaging/package_record.json`, `.md` | yes | Per package: each file's name, size, and SHA-256; the source hashes; the three run ids and their counts; the differences of D16; P1 to P10; the licence entries. Counts, ids, and hashes only |
| `reports/packaging/adapter_card.md`, `merged_card.md` | yes | The cards, as built |
| `README.md` | yes | D32 |

### 4.5 Tests

No GPU, no network, no credentials, invented rows only; the tiny smoke model and stub
predictors stand in for the real ones.

- Sources: a checkpoint with another hash is refused; the step-918 folder is never read;
  a changed prompt id or manifest hash is refused.
- Adapter package: the file list is exact; the adapter files are the source's bytes;
  `training_metadata.json` holds the selected step and none of the forbidden strings.
- Scan: each forbidden pattern is found in a planted file (an account number, an ARN, an
  S3 path, a home-folder path, an email address, a token); a clean package passes.
- Merge, on a tiny invented model: merged weights equal base plus the adapter's product
  within float tolerance; the saved type is the base's; the generation config is greedy;
  the notice and the licence are present.
- Verify: identical replies pass; one differing reply fails and is named by example id;
  the right-to-wrong and wrong-to-right counts are correct; a merged difference outside
  the allowance marks the copy withheld and leaves the adapter ready.
- Cards: every placeholder is filled; a missing record stops the build; the built card
  holds no forbidden string; the header parses as YAML; the code block of each card is
  extracted and compiles; a rebuild gives the same bytes.
- Licences: an unconfirmed entry and an entry older than 30 days each fail P7.
- Gate: `qwen3-4b-package` and `qwen3-4b-merged` are refused on test.
- Unchanged elsewhere: with three new local dev lines in a ledger, the cost record and the
  comparison report rebuild to the same bytes.
- README: D33.
- Guards: only `merge.py` imports the training stack under `packaging/`, inside functions;
  no module there imports boto3 or `huggingface_hub`'s upload calls; none reads a split
  other than through the runner.
- One test marked `gpu`: the adapter card's code answers the sample.

### 4.6 Documentation

- Runbook: a section on packaging. The order of the commands, how long each takes, how to
  read `status`, what to do when a check fails, how to rebuild after a record changes.
- `reqspecs/1. PROMPTS/coding.md`: one rule for `packaging/` (D1, D2, D11, D17, D20, D28).
- `architecture/architecture_overview.md`: the new package and what it reads.
- A pattern file, `reqspecs/references/project_patterns/release_package.md`.
- `administration/aws_commands_run.md`: nothing. No AWS command is run.

## 5. Out of scope

- Uploading anything, creating a Hugging Face repository, a token. That is v6.03.
- Any test run. The merged copy and the packages are scored on dev only.
- A GGUF copy, an 8-bit copy, a copy for another runtime.
- Scoring the merged copy at full precision, unless decision 6 says otherwise.
- Publishing the Nova models. They cannot leave Bedrock.
- Removing account identifiers from the code repository or its history, and making that
  repository public (decision 4).
- Deleting the Nova deployments or anything else in AWS.
- A new training run, another checkpoint, another prompt.
- Changes to the demo. It keeps naming adapters by run.

## 6. What this hands to later versions

| For | What |
|---|---|
| v6.03 | Two folders ready to upload, with their file hashes in the package record; the reviewed cards; `verify`'s card-code check, to be repeated against the Hub; the repository names in `config/packaging.yaml` |
| The teardown | Nothing here needs the tuned Nova models or their deployments |
| req_007, if started | The merged copy as a starting point for other runtimes |

## 7. Known limits and follow-ups

- **The package check is "same replies as the adapter by name on this machine", not "the
  recorded score again".** The recorded score came from another GPU. A reader on a third
  GPU may see a few other replies.
- **The merged copy's figure is a dev figure on 499 examples.** It has no test result of
  its own and never will under this project's gate.
- **Nobody measured the merged copy at full precision** (decision 6). Its card says so.
- **A merge is not guaranteed to give the same bytes on another machine or library
  version.** The record holds the hashes of the copy that was scored; that copy is the one
  uploaded.
- **The licence check is a reading, not legal advice.** The owner confirms it.
- **The manifest names every example by id.** FinQA ids contain the company's ticker, the
  year, and the page. That is what makes the split checkable, and it is not dataset text.
- **The cards quote list-price costs of one account in one month.**
- **The merge steps follow PEFT's documented `merge_and_unload`.** The PRP confirms the
  exact calls and the saved type against the pinned library versions before coding.

## 8. Decisions for the owner

**All eight are taken (2026-11-24).** The table is kept as the record of what was asked.

| # | Taken as |
|---|---|
| 1 | As recommended: the merged copy is withheld at 10 or more examples fewer right than R1 |
| 2 | As recommended: the merge runs on this machine's CPU |
| 3 | Two repositories |
| 4 | A public GitHub repository, `SudiptoHaldar/qlora_finetuning_public`, made by export, as its own version after this one (section 8.1) |
| 5 | `sudiptoHaldar/qwen3-4b-finqa-program-qlora` and `sudiptoHaldar/qwen3-4b-finqa-program-merged`, both to end public |
| 6 | **The merged copy is scored at full precision as well as in 4-bit**: a fourth dev run, R4, and an unquantized load in the backend. The owner said "as suggested"; the table's recommendation was 4-bit only, and the later suggestion, made because the owner wants the models to be useful to others, was both. Recorded as both; say so if 4-bit only was meant |
| 7 | As recommended: the dev runs are on this machine |
| 8 | As recommended: `apache-2.0` for both packages; the author is the Hugging Face account name, with no email address |

What decision 6 changes above: D13 gains **R4**, the merged copy at full precision
(bfloat16, no quantization), compared with R1 as R3 is (D16); the allowance of decision 1
applies to R3 and to R4, and the copy is withheld if either falls outside it; the merged
card gives both figures and its usage code shows both ways of loading; `load_backend`
gains an unquantized load for the merged copy only; section 5's line on full precision and
section 7's third limit no longer apply.

### 8.1 The public export (decision 4), for the version after this one

| | |
|---|---|
| Destination | `C:\projects\AIML_projects\qlora_finetuning_public`, pushed to `SudiptoHaldar/qlora_finetuning_public` on GitHub; no history. The repository was created by the owner on 2026-11-24, empty and private; it is made public after the export is pushed and reviewed |
| Left out | `.claude/`, `track_progress/`, `docs/old_references/`, everything in `administration/` but the runbook, `reqspecs/0. PLANNING/planning_material/`, `reqspecs/0. PLANNING/execution_plan_reference/`, `tests/test_process_scaffold.py`, the two assistant instruction files under `.github/` |
| Kept | The rest of `reqspecs/`, the CI workflow, `ledger/`, and every other tracked file |
| Runbook | Published as `docs/runbook.md`, a reader's version; pointers to `docs/runbook.md` are rewritten by rule |
| Account number | Replaced by a placeholder everywhere, the bucket name included |
| Freeze checks | Skipped: the public copy is a redacted snapshot, and its README says so. The private repository stays the place where the freeze can be verified |

This version's README and cards may link to `https://github.com/SudiptoHaldar/qlora_finetuning_public`.
The README carries the system view and the one-example walk-through of
`architecture/architecture_overview.md`, with diagrams coloured by kind of component, legible
in GitHub's light and dark themes.

| # | Question | Recommendation |
|---|---|---|
| 1 | **How far may the merged copy's dev score differ from the adapter's before it is withheld?** | Withhold it when it has 10 or more examples fewer right than R1, of 499: `noise_examples`, the project's existing measure. A copy that scores higher is published. Either way the card gives both figures and the number of replies that changed |
| 2 | **Where does the merge run?** | On this machine's CPU. It needs about 16 GB of the 64 GB of memory and some minutes, costs nothing, and makes no AWS call. The GPU has 16 GB, which is too tight for the 32-bit merge; a SageMaker job would be billed for no gain |
| 3 | **The merged copy: its own repository, or a folder inside the adapter's?** | Its own repository. The adapter repository stays at 66 MB, each card describes one thing, and each header can be right (`peft` adapter against `transformers` fine-tune). The two cards name each other. **Decided by the owner, 2026-11-24: two repositories** |
| 4 | **The AWS account number in committed files, and whether the GitHub repository becomes public.** It is in 43 tracked files and in history. Two are frozen records: the freeze record holds the deployment ARNs, and the unlock record names the freeze record's hash | Leave the repository as it is in this version, and keep it private. The two packages hold no account number, ARN, or S3 path, and check P5 enforces that. A card links to the code only when `config/packaging.yaml` names a public address. Making the repository public is a separate decision; rewriting the frozen records would break the freeze chain, so the realistic choices then are to publish as it is (an account number is not a credential) or to publish a fresh copy without history. **Owner's direction, 2026-11-24: a public GitHub repository will be made by exporting the relevant parts of this private one; the owner will name the directories to leave out. In the export the account number is replaced by a placeholder everywhere (owner, 2026-11-24). The public copy is a redacted snapshot that skips the freeze checks, and says so in its README; the private repository stays the place where the freeze can be verified (owner, 2026-11-24). The export is written to `C:\projects\AIML_projects\qlora_finetuning_public` and leaves out `.claude/`, `administration/`, and `track_progress/` (owner, 2026-11-24). `reqspecs/` is kept, because kept files cite it throughout (owner, later the same day); of `reqspecs/0. PLANNING/` and `reqspecs/1. PROMPTS/` only the three cited files are kept, `execution_plan.md`, `high_level_plan.md`, and `coding.md`; `planning_material/` and `execution_plan_reference/` are left out. The README is to carry the system view and the one-example walk-through of `architecture/architecture_overview.md`, with diagrams coloured by kind of component. The runbook is published as `docs/runbook.md`, a reader's version with the account number and bucket replaced by placeholders; the rest of `administration/` and `docs/old_references/` are left out (owner, 2026-11-24). Still to settle: what happens to kept files that point into the folders left out, `.github/` and `ledger/`, the public repository's name on GitHub, and whether the export is part of this version or one of its own** |
| 5 | **The repository names and the Hugging Face account.** Both cards' usage code and their links to each other need them now, not at upload | Decide now and put them in `config/packaging.yaml`. Proposed names: `qwen3-4b-finqa-program-qlora` and `qwen3-4b-finqa-program-merged`, under an account you name. A later change means rebuilding the cards, nothing else. **Decided by the owner, 2026-11-24: `sudiptoHaldar/qwen3-4b-finqa-program-qlora` and `sudiptoHaldar/qwen3-4b-finqa-program-merged`. Both are to end public (v6.03: uploaded private, reviewed, then made public), so that others can use them** |
| 6 | **Is the merged copy scored in 4-bit only, or at full precision too?** | 4-bit only. It is the configuration every other figure of the project uses, and it needs no new loading path. The alternative adds an unquantized load to the backend and a fourth dev run (free, about 10 minutes); the card would then cover the way a merged model is most often served |
| 7 | **The three dev runs: on this machine, or as SageMaker jobs?** | On this machine: free, about 10 minutes each, and the three are compared with each other. SageMaker would repeat arm A1's hardware and could reproduce 358 of 499 exactly, for roughly a dollar a run and three launches by you |
| 8 | **The adapter's licence, and the name shown as its author.** | `apache-2.0` for both packages, the base model's licence; the merged copy must be Apache 2.0 in any case. The author is the Hugging Face account name alone, with no email address. The code repository stays MIT |

## 9. Acceptance

Code, by Claude:

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
  and without the training stack blocked, with no network and no credentials.
- `scripts.hf_package check` passes on the real repository once the licence entries are
  confirmed.
- No frozen file changes (`comparison_freeze status`), and `scripts.comparison_cost report`
  and `scripts.comparison_report build` change no byte after the three dev runs.

On the owner's machine (free, no AWS; by Claude there, or by the owner):

1. `hf_package adapter` and `hf_package merge` write the two folders.
2. R1, R2, and R3 complete on dev, each with `result: true`.
3. `hf_package verify`: R2 equals R1 on all 499 replies; R1 is within the noise of 358;
   R3 is inside the allowance, or the merged copy is marked withheld; both cards' code
   answers the sample.
4. `hf_package cards` writes the cards and the record; P1 to P10 pass (P9 may fail, with
   the merged copy withheld).

By the owner:

5. Confirm the three licence entries.
6. Read both cards as built, and the README.

M9 is the commit of the cards, the package record, and the README with the adapter package
ready. A withheld merged copy does not hold M9 back; the roadmap says so.

## 10. Next step

The eight decisions are taken (section 8) and the PRP was generated and executed on
2026-11-24:

```
/generate-prp reqspecs/reqs/req_006_packaging_demo --version 2
```

## 11. As built (2026-11-24)

Built by `prp_006_model_card_package_v2.md`. Nothing is billed, no AWS call was made,
nothing was uploaded, and no frozen value changed (`comparison_freeze status`: nothing
frozen has changed). Both packages are **ready**: checks P1 to P10 pass.

### 11.1 What was built

| File | Holds |
|---|---|
| `config/packaging.yaml` | The folders, the repository names, the licence entries, the forbidden patterns |
| `qlora_finetuning/packaging/sources.py` | `selected_source`, `read_records`, `records_problems`, `prompt_file`, `training_metadata` |
| `adapter_package.py`, `merged_package.py` | The two builds, and the checks on what a folder holds |
| `merge.py` | `merge_adapter`: the one module of the package that loads the training stack |
| `predictors.py` | `qwen3-4b-package`, `qwen3-4b-merged`, `qwen3-4b-merged-full`; their settings |
| `verify.py` | `find_run`, `compare`, `run_card_code`, `check_card_run` |
| `cards.py`, `adapter_card.md`, `merged_card.md`, `adapter_usage.py.txt`, `merged_usage.py.txt` | The card templates, the usage code, and what fills them |
| `record.py` | `gather`: checks P1 to P10, the record, its Markdown |
| `scan.py`, `files.py`, `config.py`, `workflow.py` | The scan, file helpers, the config loader, one function per command |
| `scripts/hf_package.py` | `check`, `adapter`, `merge`, `verify`, `cards`, `status` |
| `local_qlora/backend.py` | `load_folder_backend`, `cached_files`, `load_tokenizer_from`, `weight_files`, `end_token_ids`, `greedy_generation`. `load_backend` behaves as before; its tail is shared |

Where the build differs from sections 3 and 4:

- **The three predictor factories are in `packaging/predictors.py`**, not in
  `local_qlora/predictor.py`, so `local_qlora/` does not import `packaging/`.
- **Three predictors, not two**: the full-precision run of decision 6 is
  `qwen3-4b-merged-full`.
- **The file lists are in code** (`package_names`, `merged_names`), not in the config: the
  merged copy's weight files are read from its own index.
- **D17 is corrected.** New local runs change no figure of the cost record or of the
  comparison report. They do raise one count: the cost record's "no charge" records go
  from 37 to 41 the next time it is rebuilt (`comparison_cost check` shows 41). The
  committed cost record was left as it is.
- **A card written before the licences are confirmed is a draft** under
  `artifacts/release/drafts/`, so the owner can read it first.
- **The usage code is its own template file**, so the code `verify` executes and the code
  the card prints are one text; the record holds its hash.
- **The package holds `merge_metadata.json`** in the merged copy and nothing for the
  repository names in the adapter's files: only the cards and the notice name them.
- **The failure categories are a table under Results**, not a line under Limitations.
- **The README's dollar figures are checked by a test** against the cost record.

### 11.2 What the runs showed

On the owner's machine (RTX 4090 Laptop GPU), 2026-11-24. All on dev, 499 examples, batch
size 8.

| Run | What | Right on execution | Replies that differ from R1 |
|---|---|---:|---:|
| recorded | Arm A1 in the SageMaker job that scored the checkpoints | 358 | |
| R1 | The selected adapter, by name | 358 | |
| R2 | The adapter package, from its folder | 358 | 0 |
| R3 | The merged copy, in 4-bit | 361 | 157 |
| R4 | The merged copy, at full precision | 357 | 199 |

- **R2 equals R1 on all 499 replies.**
- R1 has the recorded count, though not the recorded replies: answer accuracy is 337 here
  and 338 in the record.
- **The merged copy is inside the allowance both ways** (3 more right, 1 fewer; the limit
  is 10 fewer). Many of its replies differ from the adapter's in wording, few in outcome:
  6 right-to-wrong and 9 wrong-to-right in 4-bit, 15 and 14 at full precision.
- The merged copy is faster: 0.44 s a question in 4-bit and 0.37 s at full precision,
  against 0.72 s with the adapter attached. Full precision needs 9.0 GiB of GPU memory at
  batch size 8.
- The merge took 35 seconds on the CPU: 252 weight matrices, 4,022,468,096 parameters,
  three weight files of 8.04 GB together.
- **All three runs of card code gave the sample's expected result**
  (`0.1256140350877193`): the adapter card's, and the merged card's in 4-bit and at full
  precision.

The licences were read at their sources on 2026-11-24 and confirmed by the owner the same
day. Qwen3-4B is Apache 2.0 and FinQA is MIT, each at its own repository. **FinTabNet's
licence could not be re-read at IBM's page**, which now says its data exchange is
deprecated and states no licence; the entry rests on the card of a derived dataset, which
says the original is CDLA-Permissive 1.0. The cards print that caveat.

### 11.3 Tests

| Run | Result |
|---|---|
| `venv\Scripts\python -m pytest -q` | 3,059 passed, 1 skipped, 12 deselected |
| The same with the training stack blocked | 3,051 passed, 9 skipped, 12 deselected |
| `venv\Scripts\python -m pytest -m gpu -q tests/packaging tests/local_qlora/test_backend_gpu.py` | 7 passed: the adapter card's code, the backend's existing tests, and the demo |
| `venv\Scripts\python -m ruff check .` | clean |

New: `tests/packaging/` (eight files) and `tests/test_readme.py`.

### 11.4 For the owner

- **Read the two cards** (`reports/packaging/adapter_card.md`, `merged_card.md`) and the
  README. The README's two diagrams were not rendered here; look at them in the Markdown
  preview.
- **Committed 2026-11-24**, after the owner read the README and both cards. The ledger gained four free lines for R1 to R4. Milestone M9 is reached.
- **The README links to the public repository and names the two Hugging Face
  repositories**, none of which exists publicly yet. The export is the next version;
  uploading is v6.03.
