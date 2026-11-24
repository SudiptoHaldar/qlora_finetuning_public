# Project Scaffold

Spec Group      :   req_000_project_scaffold  
Feature Name    :   project_scaffold  
Version         :   3  
Status          :   ✅ Complete 2026-11-15 (see `reqspecs/PRPs/prp_000_local_environment_v3.md`); `env_check --smoke` passes 7/7 on native Windows (M1, Q3 resolved); CI green on pull request #2 and on `main` (2026-11-15)  
Last updated    :   2026-11-15  

## 1. v0.03 — Local development environment and training stack

Path A is developed and smoke-tested on this machine before any paid SageMaker job. Nothing
about the training stack is proven here yet: no ML library is installed, the versions are not
pinned, and the planning guides warn that 4-bit loading on native Windows may not work.

This version pins the training stack, installs it, and adds `scripts/env_check.py`, which
verifies the machine and, on request, proves the stack end to end by loading a small Qwen3
model in 4-bit and running one LoRA training step. Passing that is milestone M1, and it
answers Q3 (native Windows or WSL2).

**Depends on**: v0.02 (done).

## 2. What exists today (verified 2026-11-15)

| Item | State |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 Laptop GPU, 16,376 MiB, driver 596.08, compute capability 8.9 |
| OS and Python | Windows 11, Python 3.13.7 in `venv/` |
| WSL | WSL2 is enabled but the only distribution is Docker Desktop's. No Linux distribution is installed |
| Disk | 1.1 TB free on `C:` |
| ML libraries | None installed. `requirements.txt` has numpy, pandas, python-dotenv, PyYAML, boto3, pytest, ruff |
| Latest versions on PyPI | torch 2.14.1, transformers 5.23.0, datasets 5.2.0, peft 0.21.3, trl 1.18.0, accelerate 1.15.0, bitsandbytes 0.50.2 |
| Wheels for Windows and Python 3.13 | A dry-run install resolved all seven, 44 packages in total. torch has a `cp313-win_amd64` wheel on the `cu126` index; bitsandbytes ships a `win_amd64` wheel |
| PyTorch CUDA builds of 2.14.1 | `cu126` and `cu130` |
| Python floor of the stack | transformers and trl require Python 3.11 or later |
| Small Qwen3 models | `Qwen/Qwen3-0.6B` (752M parameters) and `Qwen/Qwen3-1.7B`; both Apache 2.0, not gated, so no Hugging Face account or token is needed |

These library versions are newer than the two planning guides. The guides' API details
(for example the completion-only data collator) are treated as unverified until checked against
what is installed.

## 3. Locked decisions

| # | Decision | Reason |
|---|---|---|
| D1 | Native Windows is tried first. `env_check --smoke` passing on native Windows resolves Q3 as "native Windows" | The plan's default for Q3. Wheels exist for every package, so there is a real chance it works |
| D2 | If the smoke fails on native Windows, the work stops there. The failure is recorded in the roadmap and the owner chooses between WSL2 and "CPU-only locally, smoke in the cloud" | Installing a Linux distribution changes the machine and is the owner's call. Measured runs use Linux on SageMaker either way |
| D3 | The seven libraries are pinned with `==` to the versions in section 2, in a new `requirements-train.txt`. `tokenizers`, `safetensors`, and `huggingface_hub` are pinned too, at the versions the resolver picks | One set of versions locally and in the cloud (plan, risks). Exact pins make a run reproducible |
| D4 | `requirements-train.txt` names `torch==2.14.1` with no index URL. The CUDA build is installed first by a runbook command from the `cu126` index; the file then installs on top without replacing it | Keeps the file usable unchanged for the cloud image in v3.03, where the CUDA build comes from the image. A plain `==2.14.1` pin accepts the `+cu126` build |
| D5 | The CUDA build is `cu126` | It is the build the dry run resolved. Driver 596.08 supports it. `cu130` is the alternative if a problem appears |
| D6 | CI does not install `requirements-train.txt`. Unit tests never import torch or any library in that file | The stack is several gigabytes and CI has no GPU. Working agreement 9 |
| D7 | `env_check` follows the same shape as `aws_check`: one function per check, a PASS, FAIL, or SKIP line each, exit code 0 when nothing failed, 1 when a check failed, 2 for a config problem. Run as `python -m scripts.env_check` | One pattern for operator checks. It is recorded as the first project pattern |
| D8 | The smoke test is opt-in: `--smoke`. Without the flag it is reported as SKIP and does not affect the exit code | It downloads a model and takes minutes; the plain check should stay fast and offline |
| D9 | The smoke model is `Qwen/Qwen3-0.6B` at a pinned revision, named in config | Same family, tokenizer, and chat template as the 4B model, at about 1.5 GB. Loading the 4B model is v3.01 |
| D10 | The smoke step is a hand-written training step, not a TRL trainer run: 4-bit NF4 load with double quantization and BF16 compute, LoRA attached with the plan's settings, one forward and backward pass, one optimizer step with the 8-bit paged AdamW | It exercises exactly the parts that can fail on Windows (bitsandbytes kernels, quantized load, the 8-bit optimizer) without depending on trainer APIs that v3.02 owns |
| D11 | Every number the checks compare against lives in `config/local_env.yaml`: minimum VRAM, minimum compute capability, the CUDA build tag, the smoke model and revision, the LoRA settings, the sequence length | Working agreement 13 |
| D12 | The Hugging Face cache stays in its default location outside the repository | Nothing downloaded here belongs in git, and `artifacts/` is for project outputs |
| D13 | ruff's `target-version` moves from `py310` to `py311` | The stack needs Python 3.11 or later, so 3.10 can never run this code. Revises D11 of the v0.02 spec |
| D14 | Reading the pins and collecting installed versions live in the package, in `qlora_finetuning/environment.py`; the checks live in `scripts/env_check.py` | The run manifest (v0.04) and the cloud version check (v3.03) need the same version report |
| D15 | During execution Claude installs the stack into `venv/` (several gigabytes) and downloads the smoke model from Hugging Face (about 1.5 GB). Neither is billed and neither uses AWS | Stated so the owner can object before the PRP runs |

## 4. Scope to build

### 4.1 `requirements-train.txt` and install steps

- The ten pins of D3, with a header comment giving the date, the CUDA build, and the two
  install commands.
- Runbook section 3 ("Local GPU environment") is filled in:

  ```powershell
  venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
  venv\Scripts\python -m pip install -r requirements-train.txt
  ```

- `requirements.txt` is not changed.

### 4.2 `config/local_env.yaml`

`version: 1`, then: minimum Python version; CUDA build tag; minimum VRAM in GiB; minimum
compute capability for BF16; the smoke model id and full revision hash; LoRA rank, alpha,
dropout, and target modules; the smoke sequence length; the path of the pins file.

### 4.3 `qlora_finetuning/environment.py`

| Function | Behavior |
|---|---|
| `read_pins(path)` | Parses `requirements-train.txt` into `{package: version}`. Ignores comments, blank lines, and option lines. A line that is not an exact `==` pin raises `ConfigError` naming it |
| `installed_versions(packages)` | Returns `{package: version or None}` from installed package metadata. Imports none of the packages |
| `environment_report()` | A plain dict: Python version, platform, the installed version of every pinned package, and, when torch is importable, the CUDA build, GPU name, VRAM, and compute capability. Never raises because a library is missing |

### 4.4 `scripts/env_check.py`

| # | Check | Passes when |
|---|---|---|
| E1 | Python | The interpreter is at or above the configured minimum |
| E2 | Pinned packages | Every package in `requirements-train.txt` is installed at exactly its pinned version (a local build suffix such as `+cu126` is accepted) |
| E3 | CUDA build of torch | torch reports a CUDA build matching the configured tag and CUDA is available |
| E4 | GPU | A GPU is present with at least the configured VRAM; the line reports name, VRAM, and driver version |
| E5 | BF16 | Compute capability is at or above the configured minimum and torch reports BF16 as supported |
| E6 | bitsandbytes | The library imports and quantizes a small tensor layer on the GPU in 4-bit. No download |
| E7 | Smoke (`--smoke` only) | The configured model loads in 4-bit NF4; the chat template with thinking disabled produces no thinking block; LoRA attaches; one training step produces a finite loss and non-zero gradients on LoRA weights only; the line reports peak VRAM and elapsed time |

- E3 to E7 are reported as SKIP, not FAIL, when an earlier check they depend on failed, with
  the reason.
- A missing library is a FAIL line that names the install command, not a traceback.
- `--report <path>` writes `environment_report()` plus the check results as JSON.
- Log lines go through `logging_setup`; PASS, FAIL, and SKIP lines go to standard output.

### 4.5 Tests

- Unit tests pass without torch installed, as in CI. Each check takes the modules or values it
  needs as arguments, and tests pass in small stand-ins: one passing and at least one failing
  case per check, the SKIP cascade, exit codes, a missing library, and `--report`.
- `read_pins`, `installed_versions`, and `environment_report` are tested with temporary files
  and without torch.
- One test marked `gpu` runs the real E1 to E6 on this machine.
- A guard test fails if any test module, or any package module outside functions, imports a
  library from `requirements-train.txt` at import time.
- The smoke is run by the script, not by pytest.

### 4.6 Documentation and references

- Runbook section 3: install, `env_check`, `env_check --smoke`, what each check means, how to
  clear the Hugging Face cache, troubleshooting.
- `reqspecs/references/common/gotchas.md`: each "unverified" model-and-training entry that this
  version can check is marked confirmed or corrected (thinking mode, BF16, bitsandbytes on
  Windows, the completion-only API offered by the installed TRL, the 8-bit optimizer).
- `reqspecs/references/tech/`: one short file for the Hugging Face training stack, recording
  the installed versions and the calls the smoke uses.
- `reqspecs/references/project_patterns/check_script.md`: the check-script pattern, now used
  twice, with a reuse checklist.
- `coding.md`: the rule that no test and no module top level imports the training stack.

## 5. Out of scope

- Loading or evaluating `Qwen/Qwen3-4B` (v3.01).
- Downloading FinQA (v1.01).
- The training loop, data collator, and TRL configuration (v3.02).
- The SageMaker image (v3.03).
- Installing or configuring WSL2 (see D2).
- A CPU-only torch in CI.

## 6. Known limits and follow-ups

- **The cloud image may not offer these versions.** AWS's prebuilt training images lag PyPI.
  v3.03 must either find an image with torch 2.14.1 or install the pins inside the job. If
  neither works the pins are revised there and `env_check` is run again here.
- **The pins are today's latest releases**, chosen because they are the versions that resolve
  together on Windows with Python 3.13. They are newer than both guides.
- **A 0.6B model passing does not prove the 4B model fits in 16 GB.** v3.01 measures that.
- **`.env` is not read by Hugging Face libraries**, so a custom cache location must be set in
  the shell, not in `.env`.
- **The `gpu` test and the smoke run only on this machine.** CI covers the logic with stand-ins.

## 7. Acceptance

- `venv\Scripts\python -m scripts.env_check` prints PASS for E1–E6 and SKIP for E7, exit 0.
- `venv\Scripts\python -m scripts.env_check --smoke` prints PASS for E1–E7 on native Windows
  (milestone M1, Q3 resolved), or the failure is recorded and the owner has chosen a fallback.
- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, and CI is
  green without the training stack.
- `venv\Scripts\python -m pytest -m gpu -q` passes on this machine.
- `requirements-train.txt` holds exact pins and installs cleanly into the existing venv.
- The gotchas reference no longer lists as unverified anything this version checked.

## 8. Next step

```
/generate-prp reqspecs/reqs/req_000_project_scaffold --version 3
```
