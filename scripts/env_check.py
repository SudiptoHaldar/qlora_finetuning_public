"""Check that this machine can run the Path A training stack (req_000 v0.03).

Prints one PASS, FAIL, or SKIP line per check and exits non-zero if any check fails. The plain
run is fast and offline. ``--smoke`` adds the end-to-end proof: load a small Qwen3 model in
4-bit, attach LoRA, and run one training step (downloads about 1.5 GB the first time).

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.env_check
    venv\\Scripts\\python -m scripts.env_check --smoke
    venv\\Scripts\\python -m scripts.env_check --smoke --report artifacts/env_check/report.json

Exit codes: 0 nothing failed, 1 at least one check failed, 2 config problem.

Everything the checks compare against comes from ``config/local_env.yaml`` and
``requirements-train.txt``. No training library is imported at module level, so this file and
its unit tests load without the training stack.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from qlora_finetuning.data.serialization import format_target, format_user, load_prompt
    from qlora_finetuning.environment import (
        environment_report,
        installed_versions,
        read_pins,
        version_matches,
    )
    from qlora_finetuning.logging_setup import get_logger, setup_logging
    from qlora_finetuning.settings import CONFIG_DIR, REPO_ROOT, ConfigError, load_config
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.env_check"
    ) from None

log = get_logger("scripts.env_check")

# Libraries that log every HTTP request at INFO; they are held at WARNING during a run.
NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3", "filelock")

DEFAULT_CONFIG = CONFIG_DIR / "local_env.yaml"
REQUIRED_KEYS = (
    "python_min",
    "pins_file",
    "torch.cuda_build",
    "torch.install_index",
    "gpu.min_vram_gib",
    "gpu.bf16_min_compute_capability",
    "smoke.model_id",
    "smoke.revision",
    "smoke.max_seq_length",
    "smoke.max_new_tokens",
    "smoke.learning_rate",
    "smoke.lora.r",
    "smoke.lora.alpha",
    "smoke.lora.dropout",
    "smoke.lora.target_modules",
    "smoke.quantization.type",
    "smoke.quantization.double_quant",
    "smoke.quantization.compute_dtype",
)
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
INSTALL_HINT = "install the training stack: docs/runbook.md section 3"

# One invented example for the smoke step. Its texts are built by the project's serializer
# (req_001 v1.03) from config/prompt.yaml, so the smoke trains on exactly the layout every arm
# uses. It is test data, not a tunable value, and not a dataset example.
SMOKE_PASSAGES = ("Net revenue increased due to higher cloud subscription volume.",)
SMOKE_TABLE = (
    ("Metric ($ in millions)", "2024", "2023"),
    ("Total Revenue", "4850.0", "4474.2"),
)
SMOKE_QUESTION = "What was the growth rate of total revenue from 2023 to 2024?"
SMOKE_PROGRAM = "subtract(4850.0, 4474.2), divide(#0, 4474.2)"
SMOKE_ANSWER = "0.08402"


def smoke_texts() -> tuple[str, str, str]:
    """The system, user, and target texts of the smoke example, in the canonical layout."""
    prompt = load_prompt()
    user = format_user(SMOKE_PASSAGES, SMOKE_TABLE, SMOKE_QUESTION, prompt)
    return prompt.system, user, format_target(SMOKE_PROGRAM, SMOKE_ANSWER, prompt)


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    name: str
    status: str
    detail: str

    @property
    def passed(self) -> bool:
        return self.status == PASS


@dataclass
class Environment:
    """Everything the checks look at, gathered once. Tests build one from stand-ins."""

    version_info: tuple[int, ...]
    pins: dict[str, str]
    installed: dict[str, str | None]
    torch: Any = None
    torch_error: str | None = None
    driver: str | None = None
    bnb_probe: Callable[[], dict[str, Any]] | None = None
    bnb_error: str | None = None
    smoke_run: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- config


def load_env_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path)
    if path.suffix != ".yaml":
        raise ConfigError(f"config file must end in .yaml: {path}")
    return load_config(path.stem, required=REQUIRED_KEYS, config_dir=path.parent)


def _parse_version(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in str(text).split("."))
    except ValueError as exc:
        raise ConfigError(f"python_min must look like 3.11, got {text!r}") from exc


# --------------------------------------------------------------------------- checks


def check_python(version_info: tuple[int, ...], cfg: dict[str, Any]) -> CheckResult:
    """E1: the interpreter is at or above the configured minimum."""
    minimum = _parse_version(cfg["python_min"])
    have = ".".join(str(part) for part in version_info[:3])
    if tuple(version_info[: len(minimum)]) < minimum:
        return CheckResult("E1", "Python", FAIL, f"{have}, need {cfg['python_min']} or later")
    return CheckResult("E1", "Python", PASS, have)


def check_packages(pins: dict[str, str], installed: dict[str, str | None]) -> CheckResult:
    """E2: every pinned package is installed at exactly its pinned version."""
    name = "Pinned packages"
    if not pins:
        return CheckResult("E2", name, FAIL, "the pins file lists no packages")
    missing = [package for package in pins if installed.get(package) is None]
    wrong = [
        f"{package} {installed[package]} (pinned {pinned})"
        for package, pinned in pins.items()
        if installed.get(package) is not None and not version_matches(installed[package], pinned)
    ]
    if missing or wrong:
        parts = []
        if missing:
            parts.append(f"not installed: {', '.join(missing)}")
        if wrong:
            parts.append(f"wrong version: {', '.join(wrong)}")
        return CheckResult("E2", name, FAIL, f"{'; '.join(parts)}; {INSTALL_HINT}")
    return CheckResult("E2", name, PASS, f"{len(pins)} packages at their pinned versions")


def check_cuda_build(torch: Any, cfg: dict[str, Any]) -> CheckResult:
    """E3: torch is the configured CUDA build and CUDA is available."""
    name = "CUDA build of torch"
    tag = cfg["torch"]["cuda_build"]
    version = str(torch.__version__)
    if not version.endswith(f"+{tag}"):
        return CheckResult(
            "E3", name, FAIL, f"torch {version} is not the {tag} build; {INSTALL_HINT}"
        )
    if not torch.cuda.is_available():
        return CheckResult("E3", name, FAIL, f"torch {version} cannot reach a CUDA device")
    return CheckResult("E3", name, PASS, f"torch {version}, CUDA {torch.version.cuda}")


def check_gpu(torch: Any, cfg: dict[str, Any], driver: str | None) -> CheckResult:
    """E4: a GPU is present with at least the configured VRAM."""
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    minimum = float(cfg["gpu"]["min_vram_gib"])
    detail = f"{gpu_name}, {vram:.1f} GiB, driver {driver or 'unknown'}"
    if vram < minimum:
        return CheckResult("E4", "GPU", FAIL, f"{detail}; need at least {minimum:g} GiB")
    return CheckResult("E4", "GPU", PASS, detail)


def check_bf16(torch: Any, cfg: dict[str, Any]) -> CheckResult:
    """E5: compute capability is high enough for BF16 and torch agrees."""
    major, minor = torch.cuda.get_device_capability(0)
    capability = float(f"{major}.{minor}")
    minimum = float(cfg["gpu"]["bf16_min_compute_capability"])
    if capability < minimum:
        return CheckResult(
            "E5", "BF16", FAIL, f"compute capability {capability}, need {minimum}; use FP16"
        )
    if not torch.cuda.is_bf16_supported():
        return CheckResult("E5", "BF16", FAIL, "torch reports BF16 as unsupported on this GPU")
    return CheckResult("E5", "BF16", PASS, f"compute capability {capability}, BF16 supported")


def check_bitsandbytes(probe: Callable[[], dict[str, Any]]) -> CheckResult:
    """E6: bitsandbytes quantizes a small layer to 4-bit on the GPU and runs it."""
    name = "bitsandbytes"
    try:
        result = probe()
    except Exception as exc:  # any failure here is the answer to "does it work on this OS"
        return CheckResult("E6", name, FAIL, f"{type(exc).__name__}: {_first_line(exc)}")
    if not result.get("quantized"):
        return CheckResult("E6", name, FAIL, "the layer was not quantized to 4-bit")
    if not result.get("finite"):
        return CheckResult("E6", name, FAIL, "the 4-bit layer produced non-finite output")
    return CheckResult(
        "E6", name, PASS, f"{result.get('version', '?')}: 4-bit layer quantized and run on the GPU"
    )


def check_smoke(
    run: Callable[[dict[str, Any]], dict[str, Any]], cfg: dict[str, Any]
) -> CheckResult:
    """E7: a small model loads in 4-bit, answers without thinking, and takes one LoRA step."""
    name = "Smoke"
    try:
        result = run(cfg)
    except Exception as exc:
        return CheckResult("E7", name, FAIL, f"{type(exc).__name__}: {_first_line(exc)}")

    problems = []
    if not math.isfinite(result["loss"]):
        problems.append(f"loss is {result['loss']}")
    if result["lora_tensors_with_grad"] == 0:
        problems.append("no gradient reached the LoRA weights")
    if result["base_tensors_with_grad"] > 0:
        problems.append(f"{result['base_tensors_with_grad']} base tensors received gradients")
    if result["thinking_in_output"]:
        problems.append("the model produced a thinking block with thinking disabled")
    if not result["weights_changed"]:
        problems.append("the optimizer step did not change the LoRA weights")
    if problems:
        return CheckResult("E7", name, FAIL, "; ".join(problems))
    return CheckResult(
        "E7",
        name,
        PASS,
        f"{cfg['smoke']['model_id']} in 4-bit, loss {result['loss']:.3f}, "
        f"{result['trainable_parameters']:,} trainable parameters, "
        f"peak VRAM {result['peak_vram_gib']:.1f} GiB, {result['seconds']:.0f} s",
    )


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else "(no message)"


# --------------------------------------------------------------------------- real probes


def nvidia_driver_version() -> str | None:
    """The NVIDIA driver version from nvidia-smi, or None if it cannot be read."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = completed.stdout.strip().splitlines()
    return lines[0].strip() if lines else None


def bnb_probe() -> dict[str, Any]:
    """Quantize one small linear layer to 4-bit NF4 on the GPU and run it. No download."""
    import bitsandbytes as bnb
    import torch

    layer = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4")
    layer = layer.to("cuda")  # moving to the GPU is what quantizes the weight
    with torch.no_grad():
        output = layer(torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
    return {
        "version": f"bitsandbytes {bnb.__version__}",
        "quantized": layer.weight.dtype == torch.uint8,
        "finite": bool(torch.isfinite(output.float()).all()),
    }


def smoke_step(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load the smoke model in 4-bit, generate once, attach LoRA, and take one training step."""
    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    smoke = cfg["smoke"]
    quant = smoke["quantization"]
    lora = smoke["lora"]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)

    log.info("loading %s at %s", smoke["model_id"], smoke["revision"][:10])
    tokenizer = AutoTokenizer.from_pretrained(smoke["model_id"], revision=smoke["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        smoke["model_id"],
        revision=smoke["revision"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant["type"],
            bnb_4bit_use_double_quant=bool(quant["double_quant"]),
            bnb_4bit_compute_dtype=getattr(torch, quant["compute_dtype"]),
        ),
        device_map={"": 0},
    )

    system_text, user_text, target_text = smoke_texts()
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(0)

    log.info("generating %d tokens with thinking disabled", smoke["max_new_tokens"])
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            prompt_ids, max_new_tokens=int(smoke["max_new_tokens"]), do_sample=False
        )
    output_text = tokenizer.decode(generated[0, prompt_ids.shape[1] :], skip_special_tokens=False)

    log.info("attaching LoRA and running one training step")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=lora["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    model.train()

    # Completion-only loss by hand: the prompt tokens are masked out, only the target is scored.
    target_ids = tokenizer(
        target_text + tokenizer.eos_token, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(0)
    input_ids = torch.cat([prompt_ids, target_ids], dim=1)[:, : int(smoke["max_seq_length"])]
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(
        [p for _, p in trainable], lr=float(smoke["learning_rate"])
    )
    probe_name, probe_param = next((n, p) for n, p in trainable if "lora_B" in n)
    before = probe_param.detach().float().clone()

    loss = model(input_ids=input_ids, labels=labels).loss
    loss.backward()
    lora_with_grad = sum(
        1 for _, p in trainable if p.grad is not None and bool(p.grad.abs().sum() > 0)
    )
    base_with_grad = sum(
        1 for _, p in model.named_parameters() if not p.requires_grad and p.grad is not None
    )
    optimizer.step()
    changed = not torch.equal(before, probe_param.detach().float())

    return {
        "loss": float(loss.detach()),
        "trainable_parameters": sum(p.numel() for _, p in trainable),
        "lora_tensors_with_grad": lora_with_grad,
        "base_tensors_with_grad": base_with_grad,
        "weights_changed": changed,
        "weight_checked": probe_name,
        "thinking_in_output": "<think>" in output_text,
        "output_text": output_text,
        "prompt_tokens": int(prompt_ids.shape[1]),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "seconds": time.perf_counter() - started,
    }


# --------------------------------------------------------------------------- runner


def _try_import(name: str) -> tuple[Any, str | None]:
    """Import ``name``; return (module, None) or (None, why it could not be imported)."""
    try:
        return importlib.import_module(name), None
    except ImportError as exc:
        return None, f"{name} is not installed ({_first_line(exc)})"
    except Exception as exc:  # for example a DLL that fails to load on Windows
        return None, f"{name} failed to import: {type(exc).__name__}: {_first_line(exc)}"


def gather_environment(cfg: dict[str, Any]) -> Environment:
    """Read the real machine. Imports torch and bitsandbytes if they are installed."""
    pins = read_pins(REPO_ROOT / cfg["pins_file"])
    torch, torch_error = _try_import("torch")
    _, bnb_error = _try_import("bitsandbytes") if torch is not None else (None, "torch is missing")
    return Environment(
        version_info=tuple(sys.version_info[:3]),
        pins=pins,
        installed=installed_versions(pins),
        torch=torch,
        torch_error=torch_error,
        driver=nvidia_driver_version(),
        bnb_probe=bnb_probe if bnb_error is None else None,
        bnb_error=bnb_error,
        smoke_run=smoke_step,
    )


def run_checks(cfg: dict[str, Any], env: Environment, smoke: bool = False) -> list[CheckResult]:
    """Run E1 to E7 in order. A check whose prerequisite failed is reported as SKIP."""
    results = [check_python(env.version_info, cfg), check_packages(env.pins, env.installed)]

    def skipped(check_id: str, name: str, needs: str) -> CheckResult:
        return CheckResult(check_id, name, SKIP, f"needs {needs} to pass first")

    if env.torch is None:
        e3 = CheckResult("E3", "CUDA build of torch", FAIL, f"{env.torch_error}; {INSTALL_HINT}")
    else:
        e3 = check_cuda_build(env.torch, cfg)
    results.append(e3)

    e4 = check_gpu(env.torch, cfg, env.driver) if e3.passed else skipped("E4", "GPU", "E3")
    results.append(e4)
    e5 = check_bf16(env.torch, cfg) if e4.passed else skipped("E5", "BF16", "E4")
    results.append(e5)

    if not e4.passed:
        e6 = skipped("E6", "bitsandbytes", "E4")
    elif env.bnb_probe is None:
        e6 = CheckResult("E6", "bitsandbytes", FAIL, f"{env.bnb_error}; {INSTALL_HINT}")
    else:
        e6 = check_bitsandbytes(env.bnb_probe)
    results.append(e6)

    if not smoke:
        e7 = CheckResult("E7", "Smoke", SKIP, "not requested; run with --smoke")
    elif not (e5.passed and e6.passed):
        e7 = skipped("E7", "Smoke", "E5 and E6")
    else:
        e7 = check_smoke(env.smoke_run, cfg)
    results.append(e7)
    return results


def write_report(path: Path | str, cfg: dict[str, Any], results: list[CheckResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = environment_report(REPO_ROOT / cfg["pins_file"])
    report["checks"] = [asdict(result) for result in results]
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None, env: Environment | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the local training environment.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to the config YAML")
    parser.add_argument(
        "--smoke", action="store_true", help="also load a small model and run one LoRA step"
    )
    parser.add_argument("--report", default=None, help="write a JSON report to this path")
    args = parser.parse_args(argv)

    try:
        cfg = load_env_config(args.config)
        setup_logging()
        for noisy in NOISY_LOGGERS:
            logging.getLogger(noisy).setLevel(logging.WARNING)
        if env is None:
            env = gather_environment(cfg)
        results = run_checks(cfg, env, smoke=args.smoke)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 2

    for result in results:
        print(f"{result.status}  {result.check_id}  {result.name}: {result.detail}")
    counts = {status: sum(r.status == status for r in results) for status in (PASS, FAIL, SKIP)}
    print(f"{counts[PASS]} passed, {counts[FAIL]} failed, {counts[SKIP]} skipped")
    if args.report:
        write_report(args.report, cfg, results)
        print(f"report written to {args.report}")
    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
