"""Stand-ins for the packaging tests: a selected checkpoint, a merged copy, and dev runs.

Everything is written under ``tmp_path``. The checkpoint is a stand-in training run's, the
split is the invented one (8 dev examples), and no model is loaded. The comparison records
are the committed ones of the repository: they hold counts only, and the cards are built
from their real shape.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from qlora_finetuning.data.token_budget import load_token_budget_config
from qlora_finetuning.evaluation.fake import FakePredictor
from qlora_finetuning.local_qlora.adapters import (
    ADAPTER_CONFIG,
    CHECKPOINTS,
    TRAINING_NAME,
    adapter_sha256,
    checkpoint_steps,
    read_json,
    write_json,
)
from qlora_finetuning.local_qlora.config import load_local_qlora_config
from qlora_finetuning.local_qlora.training import run_training
from qlora_finetuning.packaging.config import PackagingConfig, load_packaging_config
from qlora_finetuning.packaging.merge import MergeResult
from qlora_finetuning.packaging.predictors import (
    MERGED,
    MERGED_FULL,
    PACKAGE,
    merged_settings,
    package_settings,
)
from qlora_finetuning.packaging.sources import read_records, selected_source
from qlora_finetuning.packaging.workflow import Inputs
from tests.evaluation.conftest import GOLD_REPLY, evaluate, make_context
from tests.local_qlora.training_helpers import StubEngine, StubReloader, make_training_context

TUNED = "qwen3-4b-tuned"
WRONG_REPLY = "PROGRAM: subtract(920, 90)\nANSWER: 830"
OTHER_RIGHT_REPLY = "PROGRAM: subtract(920, 95)\nANSWER: 825.0"  # another text, still right
TODAY = "2026-11-24"
TOKENIZER_BYTES = {
    "tokenizer.json": b'{"model": "stand-in"}\n',
    "tokenizer_config.json": b'{"tokenizer_class": "StandIn"}\n',
    "vocab.json": b'{"a": 0}\n',
    "merges.txt": b"a b\n",
}
ADAPTER_CONFIG_TEXT = json.dumps(
    {"peft_type": "LORA", "r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"]}
)


def selected_context(root: Path) -> SimpleNamespace:
    """A finished stand-in training run and a selection record naming its first checkpoint."""
    training = make_training_context(root)
    result = run_training(
        smoke=False, context=training, engine=StubEngine(), reloader=StubReloader()
    )
    run_id, runs_dir = result["run_id"], training.tracking_cfg.runs_dir
    key = read_json(runs_dir / run_id / TRAINING_NAME, "the training run's record")["key"]
    step = checkpoint_steps(runs_dir / run_id / CHECKPOINTS)[0]
    folder = runs_dir / run_id / CHECKPOINTS / f"checkpoint-{step}"
    (folder / ADAPTER_CONFIG).write_text(ADAPTER_CONFIG_TEXT, encoding="utf-8")
    training.training_cfg.selection_record.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        training.training_cfg.selection_record,
        {
            "schema": 1,
            "split": "dev",
            "key": key,
            "selected": {
                "adapter": f"{run_id}/checkpoint-{step}",
                "adapter_sha256": adapter_sha256(folder),
                "examples": 8,
                "execution_accuracy": 8,
            },
        },
    )
    return SimpleNamespace(
        prompt=training.prompt,
        split_cfg=training.split_cfg,
        tracking_cfg=training.tracking_cfg,
        selection_record=training.training_cfg.selection_record,
        checkpoint=folder,
    )


def confirmed(cfg: PackagingConfig, day: str | None = TODAY) -> PackagingConfig:
    """The config with every licence entry confirmed on ``day`` (``None``: unconfirmed)."""
    entries = tuple(dataclasses.replace(entry, confirmed_on=day) for entry in cfg.licences)
    return dataclasses.replace(cfg, licences=entries)


@dataclasses.dataclass
class World:
    """One stand-in project under a temporary folder."""

    root: Path
    cfg: PackagingConfig
    inputs: Inputs
    runner: Any  # the RunnerContext the stand-in runs are made in
    base: dict[str, Path]  # the "base model's" tokenizer files and licence

    @property
    def source(self) -> Any:
        return self.inputs.source


def make_world(root: Path) -> World:
    context = selected_context(root)
    cfg = confirmed(load_packaging_config(root=root))
    budget, local_cfg = load_token_budget_config(), load_local_qlora_config()
    base_dir = root / "base"
    base_dir.mkdir()
    base = {}
    for name, content in TOKENIZER_BYTES.items():
        (base_dir / name).write_bytes(content)
        base[name] = base_dir / name
    (base_dir / "LICENSE").write_text("Apache License\nVersion 2.0\n", encoding="utf-8")
    base["LICENSE"] = base_dir / "LICENSE"
    records = read_records(load_packaging_config())  # the repository's committed records
    # The stand-in dev split has 8 examples, all right in the "recorded" run.
    records.statistics["arms"]["A1"]["dev"] = {
        "examples": 8,
        "execution_accuracy": 8,
        "percent": "100.00",
    }
    inputs = Inputs(
        context=context,
        prompt=context.prompt,
        budget=budget,
        local_cfg=local_cfg,
        source=selected_source(context),
        records=records,
    )
    runner = make_context_like(context)
    return World(root=root, cfg=cfg, inputs=inputs, runner=runner, base=base)


def make_context_like(context: SimpleNamespace) -> Any:
    """A runner context over the stand-in split and run folders that already exist.

    The invented split is built the same way each time, so building it again under the
    same root gives the split the training run used.
    """
    return make_context(context.tracking_cfg.root)


def stub_merge(base: Any, adapter_dir: Any, out_dir: Any, **options: Any) -> MergeResult:
    """Writes the three files a merge leaves, without a model."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text('{"model_type": "stand-in"}\n', encoding="utf-8")
    (out / "generation_config.json").write_text('{"do_sample": false}\n', encoding="utf-8")
    (out / "model.safetensors").write_bytes(b"merged weights of " + Path(adapter_dir).name.encode())
    return MergeResult(
        saved_dtype="bfloat16",
        compute_dtype=options.get("compute_dtype", "float32"),
        parameters=1234,
        adapted_tensors=2,
        changed_tensor="model.layers.0.mlp.down_proj.weight",
        libraries={"torch": "0", "transformers": "0", "peft": "0"},
    )


def settings_for(world: World, label: str, weights_sha256: str | None = None) -> dict[str, Any]:
    """The settings the predictor of run ``label`` would describe itself with."""
    budget, local_cfg, sha = world.inputs.budget, world.inputs.local_cfg, world.source.sha256
    if label in ("R1", "R2"):
        return package_settings(local_cfg, budget, sha)
    return merged_settings(
        local_cfg, budget, weights_sha256=weights_sha256, adapter=sha, quantized=label == "R3"
    )


NAMES = {"R1": TUNED, "R2": PACKAGE, "R3": MERGED, "R4": MERGED_FULL}


def make_run(
    world: World,
    label: str,
    *,
    wrong: int = 0,
    reworded: int = 0,
    weights_sha256: str | None = None,
    where: str = "local",
) -> str:
    """A complete stand-in dev run of ``label``. The first ``wrong`` replies are wrong, the
    next ``reworded`` are right in other words. Returns the run id."""
    seen: list[str] = []

    def reply(request: Any) -> str:
        position = len(seen)
        seen.append(request.example_id)
        if position < wrong:
            return WRONG_REPLY
        return OTHER_RIGHT_REPLY if position < wrong + reworded else GOLD_REPLY

    predictor = FakePredictor(
        reply, name=NAMES[label], settings=settings_for(world, label, weights_sha256)
    )
    summary = evaluate(predictor, "dev", world.runner, where=where, kind="eval")
    return summary["run_id"]


def write_card_runs(world: World, codes: dict[str, str], **problems: list[str]) -> None:
    """What ``hf_package verify`` stores: each card's code ran, to the expected result."""
    from qlora_finetuning.packaging.files import sha256_text, write_json_lf

    runs = {
        label: {
            "code_sha256": sha256_text(code),
            "problems": problems.get(label, []),
            "program_ran": True,
            "in_the_format": True,
            "value": 0.1256140350877193,
            "reply": "PROGRAM: subtract(4812, 4275), divide(#0, 4275)\nANSWER: 0.12561",
            "seconds": 1.0,
        }
        for label, code in codes.items()
    }
    world.cfg.release_dir.mkdir(parents=True, exist_ok=True)
    write_json_lf(world.cfg.card_runs_file, {"schema": 1, "runs": runs})
