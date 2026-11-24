"""The Path A predictor: the Qwen backend behind the evaluation runner (req_003 v3.01).

``QwenPredictor`` implements the ``Predictor`` protocol over a backend. It has no loop over
a split and writes nothing to the ledger: the runner does both, and takes the token counts
from each prediction.

Usage (local, free; needs the weights in the Hugging Face cache)::

    venv\\Scripts\\python -m scripts.evaluate --predictor qwen3-4b --split quick_dev

or, in code::

    from qlora_finetuning.evaluation.runner import run_evaluation
    from qlora_finetuning.local_qlora.predictor import build_predictor

    predictor = build_predictor()                 # loads the model; ConfigError if it cannot
    summary = run_evaluation(predictor, "quick_dev", label="qwen3-4b quick_dev",
                             path="A", arm="A0", where="local", scope="dev")
    summary["backend"]["peak_vram_allocated_gib"]

A tuned model (req_003 v3.02) is the same predictor over a backend with a LoRA adapter. The
adapter is named by its training run, and is refused unless it was trained for the current
model commit, prompt wording, and split::

    venv\\Scripts\\python -m scripts.evaluate --predictor qwen3-4b-tuned --adapter <run_id> ...

    predictor = build_tuned_predictor("<run_id>/checkpoint-300", context)
    predictor.settings["adapter"]      # the SHA-256 of the adapter's weights
    predictor.details                  # the training run, the checkpoint, the step

A test gives ``QwenPredictor`` a stub backend and needs no GPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qlora_finetuning.data import manifest_hash
from qlora_finetuning.data.token_budget import load_token_budget_config
from qlora_finetuning.evaluation.predictor import Prediction, PredictionRequest, PredictorError
from qlora_finetuning.local_qlora.adapters import (
    SELECTED,
    AdapterRef,
    adapter_sha256,
    check_adapter,
    read_selection,
    resolve_adapter,
    selected_adapter,
)
from qlora_finetuning.local_qlora.config import load_local_qlora_config
from qlora_finetuning.local_qlora.training_config import load_training_config
from qlora_finetuning.logging_setup import get_logger
from qlora_finetuning.settings import ConfigError

TUNED_SUFFIX = "-tuned"
SMOKE_SUFFIX = "-smoke"  # an adapter of a smoke run: never arm A1's result

logger = get_logger(__name__)


class QwenPredictor:
    """The model of one backend. A batch of requests is one batch on the GPU.

    ``settings`` holds the batch size, because a reply can depend on the batch it was
    generated in: a run is resumed only under the same one. For a tuned model it also holds
    the hash of the adapter's weights, and ``details`` says which adapter that is.
    """

    usage_source = "measured"

    def __init__(
        self,
        backend: Any,
        name: str,
        settings: Mapping[str, Any],
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.backend = backend
        self.name = name
        self.model_id = backend.model_id
        self.settings = dict(settings)
        self.batch_size = self.settings["batch_size"]
        self.details = None if details is None else dict(details)

    def predict(self, requests: Sequence[PredictionRequest]) -> list[Prediction]:
        """One prediction per request, in order. A backend failure raises ``PredictorError``."""
        if not requests:
            return []
        limits = {request.max_new_tokens for request in requests}
        if len(limits) != 1:
            raise PredictorError("the requests of one batch have different reply limits")
        try:
            replies = self.backend.generate(
                [(request.system, request.user) for request in requests], limits.pop()
            )
        except Exception as error:
            # The error's own text goes to the log only: nothing a library echoes may reach
            # a run's records.
            logger.error("generation failed: %s: %s", type(error).__name__, error)
            kind = type(error).__name__
            advice = (
                "; lower inference.batch_size in config/local_qlora.yaml and start a new run"
                if "OutOfMemory" in kind
                else ""
            )
            raise PredictorError(
                f"{kind} while generating a batch of {len(requests)}{advice}"
            ) from error
        return [
            Prediction(
                text=reply.text,
                input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens,
                latency_seconds=reply.batch_seconds / reply.batch_size,
                stop_reason=reply.stop_reason,
            )
            for reply in replies
        ]

    def measurements(self) -> dict[str, Any]:
        """What the backend measured: the runner stores it with the run."""
        return self.backend.measurements()


def build_predictor(adapter_dir: Path | str | None = None) -> QwenPredictor:
    """The predictor for the pinned model, loaded on the local GPU.

    Everything that can be wrong is found here, before any run exists: a ``ConfigError``
    names the command to run. ``adapter_dir`` attaches a saved LoRA adapter.
    """
    # The training stack is loaded only when the Qwen model is asked for.
    from qlora_finetuning.local_qlora.backend import load_backend

    cfg, budget = load_local_qlora_config(), load_token_budget_config()
    backend = load_backend(cfg, budget, adapter_dir=adapter_dir)
    return QwenPredictor(backend, cfg.predictor_name, backend.settings)


def qwen3_4b(split: str, context: Any) -> QwenPredictor:
    """Untuned Qwen3-4B in 4-bit, for ``scripts.evaluate --predictor qwen3-4b``."""
    return build_predictor()


def resolve_tuned_adapter(adapter: str | None, context: Any) -> AdapterRef:
    """Find the adapter called ``adapter`` and check it, without loading anything.

    Every check of ``build_tuned_predictor``, and none of its work on the GPU: the demo's
    ``--check`` (req_006 v6.01) calls this alone. Raises ``ConfigError``.
    """
    if not adapter:
        raise ConfigError(
            "a tuned predictor needs an adapter: add --adapter RUN_ID or "
            "--adapter RUN_ID/checkpoint-N (the id of a training run), or --adapter selected"
        )
    budget = load_token_budget_config()
    tracking = context.tracking_cfg
    training_cfg = load_training_config(root=tracking.root)
    scored_sha256 = None
    if adapter == SELECTED:
        adapter = selected_adapter(
            training_cfg.selection_record,
            model_id=budget.model_id,
            revision=budget.revision,
            prompt_id=context.prompt.prompt_id,
            manifest_hash=manifest_hash(context.split_cfg),
        )
        scored_sha256 = read_selection(training_cfg.selection_record)["selected"].get(
            "adapter_sha256"
        )
    ref = resolve_adapter(adapter, training_cfg.adapters_dir, tracking.runs_dir)
    check_adapter(
        ref,
        model_id=budget.model_id,
        revision=budget.revision,
        prompt_id=context.prompt.prompt_id,
        manifest_hash=manifest_hash(context.split_cfg),
    )
    recorded = ref.metadata.get("adapter_sha256")
    if ref.checkpoint is None and recorded != adapter_sha256(ref.path):
        raise ConfigError(
            f"the weights of adapter {ref.name} are not the ones its metadata records: "
            "the folder was changed after training"
        )
    if scored_sha256 is not None and scored_sha256 != adapter_sha256(ref.path):
        raise ConfigError(
            f"the weights of {ref.name} here are not the ones the selection was made on: "
            "pull the checkpoint again (python -m scripts.sagemaker_job pull --checkpoint N)"
        )
    return ref


def build_tuned_predictor(adapter: str | None, context: Any) -> QwenPredictor:
    """The predictor for the pinned model with the adapter called ``adapter`` (req_003 v3.02).

    ``adapter`` is a training run's id, or ``RUN_ID/checkpoint-N``, or ``selected``: the
    checkpoint that the committed selection record names (req_003 v3.04), which is
    refused if its weights here are not the ones that were scored. ``context`` is the
    runner's: it carries the prompt, the split, and where runs are kept. The adapter is
    refused unless it was trained for the current model commit, prompt wording, and split.
    Every refusal is a ``ConfigError`` before the model is loaded and before any run exists.
    """
    ref = resolve_tuned_adapter(adapter, context)
    cfg, budget = load_local_qlora_config(), load_token_budget_config()

    # The training stack is loaded only when the Qwen model is asked for.
    from qlora_finetuning.local_qlora.backend import load_backend

    backend = load_backend(cfg, budget, adapter_dir=ref.path)
    smoke = bool(ref.metadata.get("smoke"))
    return QwenPredictor(
        backend,
        cfg.predictor_name + TUNED_SUFFIX + (SMOKE_SUFFIX if smoke else ""),
        backend.settings,
        details={
            "training_run": ref.run_id,
            "checkpoint": ref.checkpoint,
            "steps": ref.metadata.get("steps"),
            "smoke": smoke,
        },
    )


def qwen3_4b_tuned(split: str, context: Any, adapter: str | None = None) -> QwenPredictor:
    """Qwen3-4B with a trained adapter, for ``scripts.evaluate --predictor qwen3-4b-tuned``."""
    return build_tuned_predictor(adapter, context)
