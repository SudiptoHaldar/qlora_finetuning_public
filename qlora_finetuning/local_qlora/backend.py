"""The Qwen backend: the pinned model in 4-bit on a GPU, generating greedily (req_003 v3.01).

This module and ``training.py`` are the only ones of Path A that import the training stack
(``torch``, ``transformers``, ``peft``, ``bitsandbytes``, ``huggingface_hub``), and they do
so inside functions, so the package still loads where the stack is not installed.

Usage::

    from qlora_finetuning.data.token_budget import load_token_budget_config
    from qlora_finetuning.local_qlora.backend import load_backend
    from qlora_finetuning.local_qlora.config import load_local_qlora_config

    cfg, budget = load_local_qlora_config(), load_token_budget_config()
    backend = load_backend(cfg, budget)        # from the local cache, on GPU 0; no network
    replies = backend.generate([(system_text, user_text)], max_new_tokens=192)
    replies[0].text, replies[0].input_tokens, replies[0].output_tokens, replies[0].stop_reason
    backend.measurements()                     # peak GPU memory, seconds, tokens per second

    tuned = load_backend(cfg, budget, adapter_dir="artifacts/adapters/<name>")   # a LoRA adapter

``load_backend`` checks everything that can be wrong before a model is asked (no training
stack, weights not downloaded, a changed tokenizer, no GPU) and raises ``ConfigError`` with
the command to run. The weights are downloaded once, by hand::

    venv\\Scripts\\hf download Qwen/Qwen3-4B --revision <the commit in config/token_budget.yaml>

Nothing here reads a split, writes a file, or scores a reply: that is the evaluation runner.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from qlora_finetuning.data.token_budget import (
    INSTALL_HINT,
    TokenBudgetConfig,
    tokenizer_fingerprint,
)
from qlora_finetuning.local_qlora.adapters import adapter_sha256
from qlora_finetuning.local_qlora.config import LocalQloraConfig
from qlora_finetuning.local_qlora.render import RenderError, read_generation, render_chat
from qlora_finetuning.logging_setup import get_logger
from qlora_finetuning.settings import ConfigError

WEIGHTS_INDEX = "model.safetensors.index.json"  # a model stored in several files
WEIGHTS_SINGLE = "model.safetensors"  # a model stored in one
GIB = 2**30

logger = get_logger(__name__)


@dataclass(frozen=True)
class Generated:
    """One reply of a batch, with what it consumed."""

    text: str
    input_tokens: int  # the prompt's own length, without padding
    output_tokens: int  # up to and including the end token
    stop_reason: str
    batch_seconds: float  # the whole batch's time; the caller divides it
    batch_size: int


# --------------------------------------------------------------------------- readiness


def download_command(model_id: str, revision: str) -> str:
    return f"venv\\Scripts\\hf download {model_id} --revision {revision}"


def _stack() -> SimpleNamespace:
    """The training stack, or a ``ConfigError`` that says how to install it."""
    try:
        import bitsandbytes  # noqa: F401  (4-bit loading needs it; fail here, not mid-load)
        import peft
        import torch
        import transformers
    except ImportError as exc:
        if exc.name:
            raise ConfigError(
                f"the training stack is not installed ({exc.name} is missing); {INSTALL_HINT}"
            ) from exc
        # No module name: a library is installed and broke while loading (for example one
        # built for another torch). The reason is in the error and in what caused it.
        reasons = [f"{type(exc).__name__}: {exc}"]
        if exc.__cause__ is not None:
            reasons.append(f"{type(exc.__cause__).__name__}: {exc.__cause__}")
        raise ConfigError(
            "the training stack is installed and cannot be loaded: " + "; caused by ".join(reasons)
        ) from exc
    return SimpleNamespace(torch=torch, transformers=transformers, peft=peft)


def weights_missing(model_id: str, revision: str) -> list[str]:
    """The model files that are not in the local Hugging Face cache. Nothing is downloaded."""
    from huggingface_hub import try_to_load_from_cache

    def cached(name: str) -> str | None:
        found = try_to_load_from_cache(model_id, name, revision=revision)
        return found if isinstance(found, str) else None

    if cached("config.json") is None:
        return ["config.json"]
    index = cached(WEIGHTS_INDEX)
    if index is None:
        return [] if cached(WEIGHTS_SINGLE) else [WEIGHTS_SINGLE]
    shards = sorted(set(json.loads(Path(index).read_text(encoding="utf-8"))["weight_map"].values()))
    return [name for name in shards if cached(name) is None]


def fetch_weights(
    budget: TokenBudgetConfig, model_id: str | None = None, revision: str | None = None
) -> None:
    """Download the pinned model into the local cache: 8 GB, from the Hugging Face Hub.

    The one place that uses the network. A cloud job calls it before its task
    (``python -m scripts.job_run fetch-model``); a task itself loads from the cache only.
    Raises ``ConfigError`` if the download fails or leaves a file missing.
    """
    model_id, revision = model_id or budget.model_id, revision or budget.revision
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ConfigError(f"huggingface_hub is not installed; {INSTALL_HINT}") from exc
    if weights_missing(model_id, revision):
        logger.info("downloading %s at %s", model_id, revision[:12])
        try:
            snapshot_download(repo_id=model_id, revision=revision)
        except Exception as exc:  # the network, the Hub, a full disk
            raise ConfigError(
                f"{model_id} could not be downloaded ({type(exc).__name__}: {_first_line(exc)})"
            ) from exc
    missing = weights_missing(model_id, revision)
    if missing:
        raise ConfigError(
            f"{model_id} is still incomplete after the download (missing: {missing[0]})"
        )


def _load_tokenizer(stack: SimpleNamespace, model_id: str, revision: str) -> Any:
    return stack.transformers.AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=True
    )


def _first_line(error: BaseException) -> str:
    return (str(error).strip().splitlines() or ["(no message)"])[0]


def check_ready(
    budget: TokenBudgetConfig, model_id: str | None = None, revision: str | None = None
) -> tuple[SimpleNamespace, Any]:
    """Everything that must hold before the model is loaded; returns the stack and tokenizer.

    Raises ``ConfigError`` naming the command to run: the training stack is installed, the
    weights are in the local cache, the tokenizer is the pinned one, and a CUDA GPU is there.
    A caller that builds a predictor runs this before any run exists, so none of these
    leaves a failed run behind.
    """
    model_id, revision = model_id or budget.model_id, revision or budget.revision
    stack = _stack()
    missing = weights_missing(model_id, revision)
    if missing:
        shown = ", ".join(missing[:3]) + (", ..." if len(missing) > 3 else "")
        raise ConfigError(
            f"the weights of {model_id} at {revision[:12]}... are not in the local Hugging "
            f"Face cache (missing: {shown}); download them once with: "
            f"{download_command(model_id, revision)}"
        )
    try:
        tokenizer = _load_tokenizer(stack, model_id, revision)
    except Exception as exc:  # the library raises several kinds for missing or broken files
        raise ConfigError(
            f"the tokenizer of {model_id} could not be loaded from the local cache "
            f"({type(exc).__name__}: {_first_line(exc)}); download it with: "
            f"{download_command(model_id, revision)}"
        ) from exc
    if tokenizer_fingerprint(tokenizer) != budget.fingerprint:
        raise ConfigError(
            f"the tokenizer of {model_id} at {revision[:12]}... is not the one pinned in "
            "config/token_budget.yaml (tokenizer.fingerprint): its vocabulary, chat template, "
            "or end token differs, so prompts would not be the ones the token budget counted"
        )
    if not stack.torch.cuda.is_available():
        raise ConfigError("no CUDA GPU is available; check with: python -m scripts.env_check")
    return stack, tokenizer


def _end_ids(tokenizer: Any, end_tokens: Sequence[str]) -> list[int]:
    """The ids of the end tokens. A name that is not one token of the tokenizer is refused."""
    ids = []
    for token in end_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if not isinstance(token_id, int) or tokenizer.convert_ids_to_tokens(token_id) != token:
            raise ConfigError(
                f"config local_qlora.yaml: inference.end_tokens names {token!r}, which is "
                "not a token of this tokenizer"
            )
        ids.append(token_id)
    return ids


def end_token_ids(tokenizer: Any, end_tokens: Sequence[str]) -> list[int]:
    """The ids of ``end_tokens`` in ``tokenizer``; a ``ConfigError`` if one is not a token."""
    return _end_ids(tokenizer, end_tokens)


# --------------------------------------------------------------------------- the backend


class QwenBackend:
    """A loaded model and its tokenizer. Build one with ``load_backend``."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        torch: Any,
        *,
        model_id: str,
        settings: dict[str, Any],
        thinking_off_suffix: str,
        end_ids: Sequence[int],
        load_seconds: float,
        weights_footprint_bytes: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.settings = settings
        self._torch = torch
        self._suffix = thinking_off_suffix
        self._end_ids = frozenset(end_ids)
        self._load_seconds = load_seconds
        self._weights_footprint_bytes = weights_footprint_bytes
        self._batches = 0
        self._prompts = 0
        self._generate_seconds = 0.0
        self._generated_tokens = 0
        self._think_removed = 0
        self._longest_prompt = 0

    def generate(self, prompts: Sequence[tuple[str, str]], max_new_tokens: int) -> list[Generated]:
        """One reply per ``(system, user)`` pair, in order, generated as one batch.

        The texts are wrapped in the chat template unchanged and are never truncated. The
        batch is padded on the left. Generation is greedy and stops at an end token or at
        ``max_new_tokens``.
        """
        torch = self._torch
        texts = [
            render_chat(system, user, self.tokenizer, self._suffix) for system, user in prompts
        ]
        batch = self.tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)
        lengths = [int(n) for n in batch["attention_mask"].sum(dim=1).tolist()]
        width = int(batch["input_ids"].shape[1])

        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
            )
        rows = output[:, width:].tolist()  # copying to the CPU waits for the GPU to finish
        seconds = time.perf_counter() - started

        replies = []
        for row, length in zip(rows, lengths, strict=True):
            read = read_generation(row, self.tokenizer, self._end_ids, max_new_tokens)
            self._think_removed += read.think_removed
            self._generated_tokens += read.output_tokens
            replies.append(
                Generated(
                    text=read.text,
                    input_tokens=length,
                    output_tokens=read.output_tokens,
                    stop_reason=read.stop_reason,
                    batch_seconds=seconds,
                    batch_size=len(rows),
                )
            )
        self._batches += 1
        self._prompts += len(rows)
        self._generate_seconds += seconds
        self._longest_prompt = max(self._longest_prompt, *lengths)
        return replies

    def measurements(self) -> dict[str, Any]:
        """What the process measured so far, as plain values. No prompt and no reply text.

        Peak memory is counted from before the model was loaded, so it includes the load.
        """
        torch = self._torch
        properties = torch.cuda.get_device_properties(0)
        seconds = self._generate_seconds
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_gib": round(properties.total_memory / GIB, 3),
            "load_seconds": round(self._load_seconds, 3),
            "weights_footprint_gib": round(self._weights_footprint_bytes / GIB, 3),
            "peak_vram_allocated_gib": round(torch.cuda.max_memory_allocated(0) / GIB, 3),
            "peak_vram_reserved_gib": round(torch.cuda.max_memory_reserved(0) / GIB, 3),
            "batch_size": self.settings["batch_size"],
            "batches": self._batches,
            "prompts": self._prompts,
            "longest_prompt_tokens": self._longest_prompt,
            "generate_seconds": round(seconds, 3),
            "generated_tokens": self._generated_tokens,
            "generated_tokens_per_second": (
                round(self._generated_tokens / seconds, 2) if seconds > 0 else None
            ),
            "think_blocks_removed": self._think_removed,
        }


def load_quantized_model(
    stack: SimpleNamespace, cfg: LocalQloraConfig, model_id: str, revision: str
) -> Any:
    """The base model in 4-bit on GPU 0, from the local cache.

    One function for inference and for training (req_003 v3.02): an adapter is trained
    against exactly the weights it is later asked with.
    """
    torch, transformers = stack.torch, stack.transformers
    return transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
        device_map={"": 0},
        quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.quantization_type,
            bnb_4bit_use_double_quant=cfg.double_quant,
            bnb_4bit_compute_dtype=getattr(torch, cfg.compute_dtype),
        ),
    )


def load_backend(
    cfg: LocalQloraConfig,
    budget: TokenBudgetConfig,
    model_id: str | None = None,
    revision: str | None = None,
    adapter_dir: Path | str | None = None,
) -> QwenBackend:
    """Load the model in 4-bit on GPU 0, from the local cache, ready to generate.

    The model and its commit default to the ones pinned in ``config/token_budget.yaml``; a
    test passes the small smoke model. ``adapter_dir`` is a folder saved by PEFT: the LoRA
    adapter is attached and **not merged**, so the 4-bit weights stay the ones it was trained
    against. Every failure is a ``ConfigError``.
    """
    model_id, revision = model_id or budget.model_id, revision or budget.revision
    stack, tokenizer = check_ready(budget, model_id, revision)
    torch = stack.torch
    adapter = adapter_sha256(adapter_dir) if adapter_dir is not None else None
    end_ids = _prepare_tokenizer(tokenizer, cfg, budget, model_id)

    logger.info("loading %s at %s in 4-bit", model_id, revision[:12])
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    try:
        model = load_quantized_model(stack, cfg, model_id, revision)
        footprint = int(model.get_memory_footprint())
        if adapter_dir is not None:
            model = stack.peft.PeftModel.from_pretrained(
                model, str(adapter_dir), is_trainable=False
            )
    except Exception as exc:  # out of memory, a broken file, a library mismatch
        raise ConfigError(
            f"{model_id} could not be loaded in 4-bit ({type(exc).__name__}: {_first_line(exc)})"
        ) from exc
    return _finish(
        model,
        tokenizer,
        stack,
        model_id=model_id,
        settings=cfg.settings(revision, adapter),
        budget=budget,
        end_ids=end_ids,
        started=started,
        footprint=footprint,
    )


def _prepare_tokenizer(
    tokenizer: Any, cfg: LocalQloraConfig, budget: TokenBudgetConfig, model_id: str
) -> list[int]:
    """Set the tokenizer up for generating; returns the ids of the end tokens."""
    tokenizer.padding_side = "left"  # the default is right, which is wrong for generating
    if tokenizer.pad_token_id is None:
        raise ConfigError(f"the tokenizer of {model_id} has no padding token")
    end_ids = _end_ids(tokenizer, cfg.end_tokens)
    try:
        render_chat("system", "user", tokenizer, budget.thinking_off_suffix)
    except RenderError as exc:
        raise ConfigError(str(exc)) from exc
    return end_ids


def greedy_generation(
    transformers: Any, end_ids: Sequence[int], pad_token_id: int
) -> Any:
    """The generation settings of every Qwen run: greedy, stopping at the end tokens.

    The model's own generation config samples (temperature 0.6), and transformers fills
    every value a call leaves unset from it. So it is replaced whole, not overridden.
    """
    return transformers.GenerationConfig(
        do_sample=False,
        num_beams=1,
        eos_token_id=list(end_ids),
        pad_token_id=pad_token_id,
    )


def _finish(
    model: Any,
    tokenizer: Any,
    stack: SimpleNamespace,
    *,
    model_id: str,
    settings: dict[str, Any],
    budget: TokenBudgetConfig,
    end_ids: list[int],
    started: float,
    footprint: int,
) -> QwenBackend:
    """A loaded model made ready to generate, as a backend."""
    model.eval()
    model.generation_config = greedy_generation(
        stack.transformers, end_ids, tokenizer.pad_token_id
    )
    load_seconds = time.perf_counter() - started
    logger.info("loaded in %.1f s, weights %.2f GiB", load_seconds, footprint / GIB)
    return QwenBackend(
        model,
        tokenizer,
        stack.torch,
        model_id=model_id,
        settings=settings,
        thinking_off_suffix=budget.thinking_off_suffix,
        end_ids=end_ids,
        load_seconds=load_seconds,
        weights_footprint_bytes=footprint,
    )


# --------------------------------------------------------------------------- a model in a folder


def cached_files(model_id: str, revision: str, names: Sequence[str]) -> dict[str, Path]:
    """Where files of a model are in the local Hugging Face cache. Nothing is downloaded.

    Raises ``ConfigError`` naming the download command if one is not there.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError as exc:
        raise ConfigError(f"huggingface_hub is not installed; {INSTALL_HINT}") from exc
    found: dict[str, Path] = {}
    for name in names:
        path = try_to_load_from_cache(model_id, name, revision=revision)
        if not isinstance(path, str):
            raise ConfigError(
                f"{name} of {model_id} at {revision[:12]}... is not in the local Hugging Face "
                f"cache; download it once with: {download_command(model_id, revision)}"
            )
        found[name] = Path(path)
    return found


def load_tokenizer_from(folder: Path | str) -> Any:
    """The tokenizer whose files are in ``folder``. No weights are loaded, no GPU is used."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ConfigError(f"transformers is not installed; {INSTALL_HINT}") from exc
    try:
        return AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    except Exception as exc:  # the library raises several kinds for missing or broken files
        raise ConfigError(
            f"the tokenizer in {folder} could not be loaded "
            f"({type(exc).__name__}: {_first_line(exc)})"
        ) from exc


def weight_files(model_dir: Path | str) -> list[Path]:
    """The weight files of a model saved in a folder, in name order.

    Raises ``ConfigError`` if the folder is not a saved model or a file is missing.
    """
    folder = Path(model_dir)
    if not (folder / "config.json").is_file():
        raise ConfigError(f"{folder} is not a saved model: it has no config.json")
    index = folder / WEIGHTS_INDEX
    if index.is_file():
        names = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
    else:
        names = [WEIGHTS_SINGLE]
    missing = [name for name in names if not (folder / name).is_file()]
    if missing:
        raise ConfigError(f"the model in {folder} is incomplete (missing: {missing[0]})")
    return [folder / name for name in names]


def load_folder_backend(
    cfg: LocalQloraConfig,
    budget: TokenBudgetConfig,
    model_dir: Path | str,
    *,
    model_id: str,
    settings: dict[str, Any],
    quantized: bool = True,
) -> QwenBackend:
    """A model saved in a folder, on GPU 0, ready to generate (req_006 v6.02).

    For a copy of the pinned model with an adapter merged into its weights. ``quantized``
    loads it in 4-bit with the values of ``config/local_qlora.yaml``, as every other run of
    the project; otherwise it is loaded as it is stored, in ``compute_dtype``. The tokenizer
    comes from the same folder and must be the pinned one. Prompts, decoding, and the end
    tokens are those of ``load_backend``. ``settings`` is what the run is recorded with:
    the caller names the weights in it. Every failure is a ``ConfigError``.
    """
    folder = Path(model_dir)
    stack = _stack()
    torch, transformers = stack.torch, stack.transformers
    weight_files(folder)
    tokenizer = load_tokenizer_from(folder)
    if tokenizer_fingerprint(tokenizer) != budget.fingerprint:
        raise ConfigError(
            f"the tokenizer in {folder} is not the one pinned in config/token_budget.yaml "
            "(tokenizer.fingerprint)"
        )
    if not torch.cuda.is_available():
        raise ConfigError("no CUDA GPU is available; check with: python -m scripts.env_check")
    end_ids = _prepare_tokenizer(tokenizer, cfg, budget, model_id)

    how = "in 4-bit" if quantized else f"in {cfg.compute_dtype}, not quantized"
    logger.info("loading the model in %s %s", folder.name, how)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    try:
        if quantized:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                str(folder),
                local_files_only=True,
                device_map={"": 0},
                quantization_config=transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=cfg.quantization_type,
                    bnb_4bit_use_double_quant=cfg.double_quant,
                    bnb_4bit_compute_dtype=getattr(torch, cfg.compute_dtype),
                ),
            )
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                str(folder),
                local_files_only=True,
                device_map={"": 0},
                dtype=getattr(torch, cfg.compute_dtype),
            )
        footprint = int(model.get_memory_footprint())
    except Exception as exc:  # out of memory, a broken file, a library mismatch
        raise ConfigError(
            f"the model in {folder} could not be loaded {how} "
            f"({type(exc).__name__}: {_first_line(exc)})"
        ) from exc
    return _finish(
        model,
        tokenizer,
        stack,
        model_id=model_id,
        settings=settings,
        budget=budget,
        end_ids=end_ids,
        started=started,
        footprint=footprint,
    )
