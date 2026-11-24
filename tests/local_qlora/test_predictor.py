"""Tests for qlora_finetuning.local_qlora.predictor: a stub backend, and the real runner."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from qlora_finetuning.evaluation.predictor import PredictionRequest, PredictorError, describe
from qlora_finetuning.evaluation.runner import MEASUREMENTS_NAME, ResumeError
from qlora_finetuning.local_qlora import predictor as predictor_module
from qlora_finetuning.local_qlora.config import load_local_qlora_config
from qlora_finetuning.local_qlora.predictor import QwenPredictor, qwen3_4b
from tests.evaluation.conftest import (
    evaluate,
    ids_of,
    ledger_records,
    read_lines,
    records_for,
    run_folders,
)

from .conftest import MODEL_ID, REVISION, OutOfMemoryError, StubBackend

PACKAGE = Path(predictor_module.__file__).parent
TRAINING_STACK = {
    "torch", "transformers", "datasets", "peft", "trl", "accelerate", "bitsandbytes",
    "tokenizers", "safetensors", "huggingface_hub",
}  # fmt: skip
A0 = {"path": "A", "arm": "A0", "kind": "eval"}


def make(batch_size=4, adapter=None, **backend_options):
    cfg = load_local_qlora_config()
    settings = cfg.settings(REVISION, adapter) | {"batch_size": batch_size}
    backend = StubBackend(settings=settings, **backend_options)
    return QwenPredictor(backend, cfg.predictor_name, settings), backend


def requests(count, limit=192):
    return [
        PredictionRequest(f"id-{i}", "the system text", f"question {i} of them", limit)
        for i in range(count)
    ]


# --------------------------------------------------------------------------- the predictor


def test_it_describes_itself_with_everything_that_can_change_a_reply():
    predictor, _ = make(batch_size=8)
    described = describe(predictor)
    assert described["name"] == "qwen3-4b" and described["model_id"] == MODEL_ID
    assert described["batch_size"] == 8 and described["usage_source"] == "measured"
    assert described["settings"] == {
        "revision": REVISION,
        "quantization": {"type": "nf4", "double_quant": True, "compute_dtype": "bfloat16"},
        "enable_thinking": False,
        "do_sample": False,
        "num_beams": 1,
        "stop": "end tokens",
        "end_tokens": ["<|im_end|>", "<|endoftext|>"],
        "batch_size": 8,
        "adapter": None,
    }


def test_one_prediction_per_request_in_order_with_the_batch_time_shared():
    predictor, backend = make(batch_seconds=2.0)
    predictions = predictor.predict(requests(4))
    assert backend.calls == [[("the system text", f"question {i} of them") for i in range(4)]]
    assert backend.limits == [192]
    assert [p.latency_seconds for p in predictions] == [0.5] * 4
    assert all(p.stop_reason == "end" and p.input_tokens == 7 for p in predictions)
    assert predictor.predict([]) == [] and len(backend.calls) == 1


def test_a_backend_error_is_a_predictor_error_without_any_text():
    predictor, _ = make(fail_after=0, error=RuntimeError("bad prompt: the system text"))
    with pytest.raises(PredictorError) as caught:
        predictor.predict(requests(3))
    assert str(caught.value) == "RuntimeError while generating a batch of 3"


def test_out_of_memory_says_to_lower_the_batch_size():
    predictor, _ = make(fail_after=0, error=OutOfMemoryError("CUDA out of memory"))
    with pytest.raises(PredictorError, match="lower inference.batch_size .* start a new run"):
        predictor.predict(requests(2))


def test_requests_with_different_reply_limits_are_refused():
    predictor, backend = make()
    with pytest.raises(PredictorError, match="different reply limits"):
        predictor.predict([*requests(1, limit=192), *requests(1, limit=64)])
    assert backend.calls == []


# --------------------------------------------------------------------------- through the runner


def test_a_run_records_measured_tokens_and_the_backends_measurements(context):
    predictor, backend = make(batch_size=3)
    summary = evaluate(predictor, "dev", context, **A0)
    assert summary["result"] and summary["all"]["execution_accuracy"]["count"] == 8
    assert [len(batch) for batch in backend.calls] == [3, 3, 2]

    records = records_for("dev", context)
    assert backend.calls[0][0] == (records[0].system, records[0].user)  # unchanged
    (record,) = ledger_records(context)
    assert (record.model_id, record.source) == (MODEL_ID, "measured")
    assert (record.path, record.arm, record.where, record.examples) == ("A", "A0", "local", 8)
    lines = read_lines(summary["run_id"], context)
    assert record.input_tokens == sum(line["input_tokens"] for line in lines) > 0

    assert summary["backend"] == {"peak_vram_allocated_gib": 3.5, "batches": 3, "gpu_name": "stub"}
    stored = context.tracking_cfg.runs_dir / summary["run_id"] / MEASUREMENTS_NAME
    assert json.loads(stored.read_text(encoding="utf-8")) == summary["backend"]


def test_an_interrupted_run_is_resumed_and_no_example_is_asked_twice(context):
    predictor, backend = make(batch_size=3, fail_after=1)
    with pytest.raises(PredictorError) as caught:
        evaluate(predictor, "dev", context, **A0)
    stored = context.tracking_cfg.runs_dir / caught.value.run_id / MEASUREMENTS_NAME
    assert json.loads(stored.read_text(encoding="utf-8"))["batches"] == 1  # kept on failure

    again, backend_again = make(batch_size=3)
    summary = evaluate(again, "dev", context, resume=caught.value.run_id, **A0)
    assert summary["result"]
    asked = [user for batch in backend.calls + backend_again.calls for _, user in batch]
    assert asked == [record.user for record in records_for("dev", context)]
    assert [line["example_id"] for line in read_lines(summary["run_id"], context)] == ids_of(
        "dev", context
    )
    assert sum(record.examples for record in ledger_records(context)) == 8


@pytest.mark.parametrize("change", [{"batch_size": 1}, {"adapter": "f" * 64}])
def test_a_resume_under_another_batch_size_or_adapter_is_refused(context, change):
    predictor, _ = make(batch_size=3, fail_after=1)
    with pytest.raises(PredictorError) as caught:
        evaluate(predictor, "dev", context, **A0)
    other, backend = make(**({"batch_size": 3} | change))
    before = run_folders(context)
    with pytest.raises(ResumeError, match="another predictor"):
        evaluate(other, "dev", context, resume=caught.value.run_id, **A0)
    assert backend.calls == [] and run_folders(context) == before


# --------------------------------------------------------------------------- the factory


def test_the_factory_builds_the_predictor_over_the_loaded_backend(monkeypatch):
    from qlora_finetuning.local_qlora import backend as backend_module

    seen = {}

    def fake_load(cfg, budget, adapter_dir=None):
        seen.update(model=budget.model_id, adapter_dir=adapter_dir)
        return StubBackend(settings=cfg.settings(budget.revision))

    monkeypatch.setattr(backend_module, "load_backend", fake_load)
    predictor = qwen3_4b("quick_dev", None)
    assert seen == {"model": MODEL_ID, "adapter_dir": None}
    assert predictor.name == "qwen3-4b" and predictor.settings["revision"] == REVISION
    assert predictor.settings["adapter"] is None
    assert predictor.batch_size == load_local_qlora_config().batch_size


# --------------------------------------------------------------------------- guards


def module_level_imports(path: Path) -> set[str]:
    names = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def all_imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


# The modules that load a model: inference (v3.01) and training (v3.02).
STACK_MODULES = {"backend.py", "training.py"}


def test_only_the_backend_and_training_name_the_training_stack_and_nothing_imports_path_b():
    files = sorted(PACKAGE.glob("*.py"))
    others = {"config.py", "predictor.py", "render.py", "plan.py", "adapters.py"}
    assert {path.name for path in files} >= STACK_MODULES | others
    for path in files:
        modules = all_imports(path)
        assert not any(name.startswith("qlora_finetuning.frontier") for name in modules), path.name
        assert not any(name.startswith("scripts") for name in modules), path.name
        assert not {name.split(".")[0] for name in module_level_imports(path)} & TRAINING_STACK
        if path.name not in STACK_MODULES:
            assert not {name.split(".")[0] for name in modules} & TRAINING_STACK, path.name
    # The predictor loads the backend module only when a model is asked for.
    for name in ("predictor.py", "training.py"):
        assert "qlora_finetuning.local_qlora.backend" not in module_level_imports(PACKAGE / name)


def test_loading_the_package_does_not_load_the_training_stack():
    code = (
        "import sys, qlora_finetuning.local_qlora.predictor, qlora_finetuning.local_qlora.backend, "
        "qlora_finetuning.local_qlora.render, qlora_finetuning.local_qlora.config, "
        "qlora_finetuning.local_qlora.training, qlora_finetuning.local_qlora.adapters, "
        "qlora_finetuning.local_qlora.plan, qlora_finetuning.local_qlora.training_config; "
        "print(sorted(m for m in ('torch', 'transformers', 'peft', 'trl', 'datasets', "
        "'bitsandbytes', 'boto3') "
        "if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PACKAGE.parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]"
