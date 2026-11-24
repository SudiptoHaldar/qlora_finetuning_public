# Spec summary: model card, adapter package, README (req_006 v6.02)

**PRP**: [prp_006_model_card_package_v2.md](../PRPs/prp_006_model_card_package_v2.md)  
**Spec**: `reqspecs/reqs/req_006_packaging_demo/req_006_02_spec.md`  
**Created**: 2026-11-24  

## Todo list

- [x] `config/packaging.yaml` and `qlora_finetuning/packaging/` (config, sources, files)
- [x] `backend.py`: load a model from a folder, 4-bit or unquantized; tokenizer and cache helpers
- [x] The adapter package, built from the selected checkpoint (step 500), and the scan for private strings
- [x] The merge on the CPU and the merged package (greedy settings, Qwen's licence, a notice)
- [x] Three predictors: `qwen3-4b-package`, `qwen3-4b-merged`, `qwen3-4b-merged-full`; refused on test
- [x] Verification: find the runs, compare replies, run each card's code in an empty directory
- [x] The two card templates and the card build; the package record with checks P1 to P10
- [x] `scripts/hf_package.py`: `check`, `adapter`, `merge`, `verify`, `cards`, `status`
- [x] Tests under `tests/packaging/`, with and without the training stack
- [x] On the machine: both folders, dev runs R1 to R4, `verify`, `cards`
- [x] The owner confirms the three licence entries
- [x] README with the system view and the one-example walk-through, diagrams coloured by kind
- [x] Runbook, `coding.md`, architecture, pattern file; spec section 11; roadmap; execution plan; implementation summary

## Summary

The version builds two folders that can be uploaded as they are: arm A1's adapter with a
model card, and a merged copy of the model with its own card. Nothing is uploaded, nothing
is billed, and no AWS call is made.

Each folder is proved on the owner's machine. Four dev runs go through the ordinary
runner: the adapter by name (R1), the package from its folder (R2, which must give the
same 499 replies as R1), and the merged copy in 4-bit (R3) and at full precision (R4). The
code printed on each card is executed in an empty directory on the invented sample.

The cards are built from templates and the committed records, never edited. Both carry
the contamination warning: the training pool holds official FinQA validation and test
examples. The README is written by hand around the generated results block.

All eight decisions of spec section 8 are taken. The merged copy is withheld if it has 10
or more examples fewer right than the adapter on dev, in either precision.

## Confidence score

**7/10.** The risks are the exact library calls for the merge and the unquantized load,
an unexpected difference between R2 and R1, GPU memory for the full-precision run, and
the FinTabNet licence, which could not be re-read at its source.

## Implementation Time Estimate

5–7 hours of code, tests, and documents; about 1.5 hours of machine time; nothing billed.

| Task | Time |
|---|---|
| Config, files, sources, adapter package, scan | 60 min |
| Backend change, merge, merged package, predictors | 75 min |
| Verification and the cards' code | 60 min |
| Card templates, card build, record, script | 75 min |
| Tests | 75 min |
| README with diagrams; runbook and other documents | 60 min |
| Machine time | 75 min |

## Time Taken for PRP Generation

**~25 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 4 min |
| Codebase exploration | 12 min |
| Research existing PRPs and patterns, and the licence sources | 4 min |
| Writing PRP document | 5 min |
