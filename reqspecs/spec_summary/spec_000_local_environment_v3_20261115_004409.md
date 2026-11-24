# Spec summary: Local development environment and training stack (req_000 v0.03)

**PRP**: [prp_000_local_environment_v3.md](../PRPs/prp_000_local_environment_v3.md)  
**Origin**: `reqspecs/reqs/req_000_project_scaffold/req_000_03_spec.md`  
**Created**: 2026-11-15  

## Todo list

1. Install torch from the `cu126` index, then the pinned libraries.
2. Write `requirements-train.txt` (ten exact pins) and `config/local_env.yaml`.
3. Write `qlora_finetuning/environment.py` and its tests.
4. Write `scripts/env_check.py`: checks E1–E7, the real probes, runner, CLI.
5. Write the stand-in tests for `env_check`, one `gpu` test, and the guard against
   module-level imports of the training stack.
6. Run `env_check`, `env_check --smoke`, the `gpu` test, and the unit tests with the training
   libraries blocked.
7. Fill runbook section 3; update gotchas, troubleshooting, `coding.md`; add the tech reference
   and the check-script pattern.
8. Record M1 and Q3; update the roadmap, plan, and summaries.

## Summary

The training stack is pinned to the versions that resolve together on Windows with Python
3.13, and `scripts/env_check.py` verifies the machine against them. Six checks are fast and
offline: Python, pinned versions, the CUDA build of torch, GPU and VRAM, BF16, and a 4-bit
quantization probe. An opt-in smoke loads `Qwen/Qwen3-0.6B` in 4-bit, generates with thinking
disabled, attaches LoRA, and takes one training step with the 8-bit optimizer.

Because the library versions are newer than the planning guides, the stack was installed
during research and every API in the design was read from the installed packages. The smoke
was seen to pass on native Windows before the PRP was written, which reaches milestone M1 and
resolves Q3.

## Confidence score

9/10. The design was verified against the installed libraries. The CI run cannot be made from
the development machine.

## Implementation Time Estimate

2.5–3.5 hours.

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

## Time Taken for PRP Generation

~14 minutes, plus about 5 minutes of download time in the background.

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 1 min |
| Codebase exploration | 1 min |
| Research (installing and inspecting the stack, thinking-mode check) | 7 min |
| Writing PRP document | 5 min |
