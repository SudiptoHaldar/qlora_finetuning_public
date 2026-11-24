"""The steps of ``python -m scripts.hf_package``, each as one function (req_006 v6.02).

Usage::

    from qlora_finetuning.packaging import workflow
    from qlora_finetuning.packaging.config import load_packaging_config

    cfg = load_packaging_config()
    workflow.build_adapter(cfg)       # artifacts/release/adapter/, without its card
    workflow.build_merged(cfg)        # artifacts/release/merged/, without its card (CPU)

Each function loads what it needs from ``config/`` and raises ``ConfigError`` when it
cannot start. The training stack is reached only through ``backend.py`` and ``merge.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from qlora_finetuning.data.serialization import PromptDefinition
from qlora_finetuning.data.token_budget import TokenBudgetConfig, load_token_budget_config
from qlora_finetuning.demo.inputs import load_input, sample_path
from qlora_finetuning.evaluation.metrics import load_evaluation_config
from qlora_finetuning.local_qlora.config import LocalQloraConfig, load_local_qlora_config
from qlora_finetuning.packaging.adapter_package import (
    CARD_NAME,
    LICENSE_NAME,
    build_adapter_package,
)
from qlora_finetuning.packaging.cards import ADAPTER, MERGED, usage_code
from qlora_finetuning.packaging.config import PackagingConfig
from qlora_finetuning.packaging.files import sha256_text, write_json_lf, write_text_lf
from qlora_finetuning.packaging.merged_package import build_merged_package
from qlora_finetuning.packaging.record import (
    RUN_NAMES,
    WITHHELD,
    State,
    check_result,
    write_record,
)
from qlora_finetuning.packaging.record import gather as gather_record
from qlora_finetuning.packaging.sources import (
    Records,
    Source,
    packaging_context,
    read_records,
    records_problems,
    selected_source,
)
from qlora_finetuning.packaging.verify import check_card_run, run_card_code
from qlora_finetuning.settings import ConfigError

DRAFTS = "drafts"  # under the release folder: cards as they would be, not yet allowed out


@dataclass(frozen=True)
class Inputs:
    """Everything a build reads from the project."""

    context: Any
    prompt: PromptDefinition
    budget: TokenBudgetConfig
    local_cfg: LocalQloraConfig
    source: Source
    records: Records


def load_inputs(cfg: PackagingConfig, context: Any = None) -> Inputs:
    """The checked checkpoint and the records. Raises ``ConfigError``."""
    context = context or packaging_context()
    return Inputs(
        context=context,
        prompt=context.prompt,
        budget=load_token_budget_config(),
        local_cfg=load_local_qlora_config(),
        source=selected_source(context),
        records=read_records(cfg),
    )


def base_files(cfg: PackagingConfig, budget: TokenBudgetConfig) -> dict[str, Path]:
    """The base model's tokenizer files and licence, in the local cache."""
    from qlora_finetuning.local_qlora.backend import cached_files

    return cached_files(budget.model_id, budget.revision, [*cfg.tokenizer_files, LICENSE_NAME])


def preflight(cfg: PackagingConfig, today: date | None = None) -> list[dict[str, Any]]:
    """What a build needs, as checks: the records, the source, the base files, the licences.

    Reads only, loads no model, and needs no GPU.
    """
    checks = []
    try:
        inputs = load_inputs(cfg)
    except ConfigError as exc:
        return [check_result("P2", "The source is arm A1", [str(exc)], "")]
    source = inputs.source
    checks.append(
        check_result(
            "P1",
            "Records in place",
            records_problems(inputs.records),
            "the six-arm record, the cost record, and the comparison's record passed; none "
            "is provisional",
        )
    )
    checks.append(
        check_result(
            "P2",
            "The source is arm A1",
            [],
            f"{source.name}, weights {source.sha256[:12]}..., as the selection record says",
        )
    )
    checks.append(
        check_result(
            "P3", "Identity", [], "the adapter's model commit, prompt, and split are today's"
        )
    )
    try:
        base_files(cfg, inputs.budget)
        missing: list[str] = []
    except ConfigError as exc:
        missing = [str(exc)]
    checks.append(
        check_result(
            "base",
            "Base model files",
            missing,
            "the tokenizer files and the licence of the base model are in the local cache",
        )
    )
    checks.append(
        check_result(
            "P7",
            "Licences",
            cfg.licence_problems(today or date.today()),
            f"{len(cfg.licences)} entries, each confirmed by the owner",
        )
    )
    return checks


def build_adapter(cfg: PackagingConfig, inputs: Inputs | None = None) -> Path:
    """Write the adapter package, all but its card."""
    inputs = inputs or load_inputs(cfg)
    return build_adapter_package(
        cfg,
        inputs.source,
        inputs.records,
        prompt=inputs.prompt,
        budget=inputs.budget,
        local_cfg=inputs.local_cfg,
        tokenizer_paths=base_files(cfg, inputs.budget),
        split_cfg=inputs.context.split_cfg,
    )


def build_merged(cfg: PackagingConfig, inputs: Inputs | None = None) -> Path:
    """Merge on the CPU and write the merged package, all but its card."""
    from qlora_finetuning.data.token_budget import load_tokenizer
    from qlora_finetuning.local_qlora.backend import end_token_ids
    from qlora_finetuning.packaging.merge import merge_adapter

    inputs = inputs or load_inputs(cfg)
    files = base_files(cfg, inputs.budget)
    tokenizer = load_tokenizer(inputs.budget)
    return build_merged_package(
        cfg,
        inputs.source,
        prompt=inputs.prompt,
        budget=inputs.budget,
        local_cfg=inputs.local_cfg,
        tokenizer_paths=files,
        licence_path=files[LICENSE_NAME],
        end_ids=end_token_ids(tokenizer, inputs.local_cfg.end_tokens),
        pad_token_id=tokenizer.pad_token_id,
        merge=merge_adapter,
    )


def gather(cfg: PackagingConfig, inputs: Inputs | None = None) -> State:
    """The record and the cards as they stand now. Reads only."""
    inputs = inputs or load_inputs(cfg)
    try:
        base: dict[str, Path] | None = base_files(cfg, inputs.budget)
    except ConfigError:
        base = None  # without the cache, the tokenizer is checked by its fingerprint alone
    return gather_record(cfg, inputs, base=base)


def card_targets(cfg: PackagingConfig) -> list[tuple[str, str, dict[str, Any]]]:
    """Each run of a card's code: its label, the card it belongs to, and what is substituted."""
    targets = [(ADAPTER, ADAPTER, {"ADAPTER": str(cfg.adapter_dir)})]
    if (cfg.merged_dir / "config.json").is_file():
        targets += [
            ("merged_4bit", MERGED, {"MODEL": str(cfg.merged_dir), "QUANTIZED": True}),
            ("merged_full", MERGED, {"MODEL": str(cfg.merged_dir), "QUANTIZED": False}),
        ]
    return targets


def verify(cfg: PackagingConfig, inputs: Inputs | None = None, run: Any = None) -> list[str]:
    """Run each card's code on the GPU, store what it gave, and report the dev runs.

    The code is run in a new process in an empty directory, one model at a time. A failure
    is stored as a problem, not raised: the record then says which card's code failed.
    Returns lines to print. ``run`` is a stand-in for ``run_card_code`` in tests.
    """
    from qlora_finetuning.local_qlora.backend import load_tokenizer_from

    inputs = inputs or load_inputs(cfg)
    run = run or run_card_code
    sample = load_input(sample_path())
    evaluation_config = load_evaluation_config()
    folders = {ADAPTER: cfg.adapter_dir, MERGED: cfg.merged_dir}
    runs: dict[str, dict[str, Any]] = {}
    lines = []
    for label, kind, values in card_targets(cfg):
        code, problems = usage_code(kind, cfg, inputs.budget, inputs.local_cfg)
        entry: dict[str, Any] = {"code_sha256": sha256_text(code), "problems": list(problems)}
        if not problems:
            try:
                result = run(code, values, timeout=cfg.card_code_timeout_seconds)
                found, facts = check_card_run(
                    result,
                    sample=sample,
                    prompt=inputs.prompt,
                    tokenizer=load_tokenizer_from(folders[kind]),
                    budget=inputs.budget,
                    evaluation_config=evaluation_config,
                )
                entry.update(facts, reply=result["reply"], seconds=result["seconds"])
                entry["problems"] = found
            except ConfigError as exc:
                entry["problems"] = [str(exc)]
        runs[label] = entry
        outcome = "; ".join(entry["problems"]) or "ran to the sample's expected result"
        lines.append(f"{'FAIL' if entry['problems'] else 'PASS'}  card code ({label}): {outcome}")
    cfg.release_dir.mkdir(parents=True, exist_ok=True)
    write_json_lf(cfg.card_runs_file, {"schema": 1, "runs": runs})

    state = gather(cfg, inputs)
    for label, what in RUN_NAMES.items():
        entry = state.record["runs"].get(label)
        if entry is None:
            lines.append(f"MISSING  {label}  {what}: no complete local run on {cfg.split}")
            continue
        text = (
            f"FOUND  {label}  {what}: {entry['counts']['execution_accuracy']} of "
            f"{entry['examples']} right ({entry['run_id']})"
        )
        difference = state.record["comparisons"].get(label)
        if difference is not None:
            text += f"; {difference['replies_that_differ']} replies differ from R1"
        lines.append(text)
    return lines


def write_cards(cfg: PackagingConfig, inputs: Inputs | None = None) -> tuple[State, list[Path]]:
    """Write the cards and the record. Returns the record as it then stands, and the files.

    A card is written to the package (``README.md``) and to the committed report folder,
    only when every licence entry is confirmed and the card could be built in full. Until
    then it is written as a draft under the git-ignored release folder, for the owner to
    read. The record is always written, and says what did not pass.
    """
    inputs = inputs or load_inputs(cfg)
    state = gather(cfg, inputs)
    allowed = not cfg.licence_problems(date.today())
    targets = {
        ADAPTER: (cfg.adapter_card, cfg.adapter_dir / CARD_NAME),
        MERGED: (cfg.merged_card, cfg.merged_dir / CARD_NAME),
    }
    written: list[Path] = []
    for kind, paths in targets.items():
        text = state.cards.get(kind)
        if text is None or (kind == MERGED and state.facts.merged_status == WITHHELD):
            continue
        if not allowed:
            draft = cfg.release_dir / DRAFTS / paths[0].name
            draft.parent.mkdir(parents=True, exist_ok=True)
            write_text_lf(draft, text)
            written.append(draft)
            continue
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_text_lf(path, text)
            written.append(path)
    state = gather(cfg, inputs)
    written += write_record(cfg, state.record)
    return state, written
