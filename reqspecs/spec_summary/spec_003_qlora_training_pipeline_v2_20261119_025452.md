# Spec summary: QLoRA training pipeline (req_003 v3.02)

**PRP**: [prp_003_qlora_training_pipeline_v2.md](../PRPs/prp_003_qlora_training_pipeline_v2.md)  
**Spec**: `reqspecs/reqs/req_003_qlora_local/req_003_02_spec.md`  
**Generated**: 2026-11-19  

## Todo list

- [ ] `config/training.yaml` and its loader, with tests
- [ ] Encoder (`encode_example`, `encode_records`) and the training plan, with tests
- [ ] Adapter metadata and `resolve_adapter`, with tests
- [ ] `load_quantized_model` shared by inference and training
- [ ] `run_training`: the run, resume, log, ledger record, final adapter, checks K1 to K7
- [ ] The TRL engine: LoRA, worst-batch probe, trainer, dataset comparison, checkpoints
- [ ] `qwen3-4b-tuned` and `--adapter` in `scripts/evaluate.py`
- [ ] `scripts/qlora_train.py` (dry run, smoke run, resume, stop after steps)
- [ ] Lint and tests, with and without the training stack; `gpu` tests
- [ ] The smoke run, a stopped run and its resume, a partial evaluation of the smoke adapter
- [ ] Runbook, tech note, gotchas, patterns, `coding.md`, architecture
- [ ] Spec section 11, roadmap, execution plan, implementation summary

## Summary

The pipeline encodes each `fit` record as the exact sequence the token budget counted (the
v3.01 prompt, the target, the end token), with labels on the reply only, and hands the TRL
trainer a prepared dataset with `max_length=None`, because the trainer's default would cut
rows silently. Every hyperparameter comes from `config/training.yaml`. Before training, the
longest records go through one step as a probe, so a run that cannot fit fails at once.

A run is one entry in the registry with a ledger record (wall clock, records, training
tokens), a training log, measurements, and checkpoints every 100 steps. Resuming is a new
run from the last checkpoint of an earlier one. The final adapter is saved with metadata,
and `qwen3-4b-tuned --adapter <run>` scores an adapter through the ordinary runner after
checking that it was trained for the current prompt, split, and model.

The owner started the PRP without changing the spec's two questions, so they are taken as
recommended: 3 epochs, and checkpoints scored after training. The agent runs the smoke run
and its rehearsals on this machine. No full training run is started; nothing is billed.

## Confidence score

7/10. The trial already ran the trainer with these settings on this machine. Not yet
exercised: resuming from a checkpoint, evaluation loss on a prepared dataset, and the
trainer's behaviour at the last step.

## Implementation Time Estimate

6–8 hours of work, plus about 15 minutes of GPU time.

| Task | Time |
|---|---|
| Config and loader | 30 min |
| Encoder and plan | 40 min |
| Adapters and metadata | 35 min |
| Backend refactor | 10 min |
| Training: the run, resume, ledger, checks | 80 min |
| Training: the TRL engine | 60 min |
| Tuned predictor and `--adapter` | 35 min |
| Script | 30 min |
| Tests | 100 min |
| Local runs | 20 min |
| Documents and status files | 45 min |

## Time Taken for PRP Generation

~12 minutes.

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 3 min |
| Research existing PRPs and patterns, installed library source | 3 min |
| Writing PRP document | 4 min |
