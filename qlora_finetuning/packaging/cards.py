"""The two model cards, built from templates and records (req_006 v6.02, spec D20 to D29).

A card is never edited. Its wording is a template beside this module (``adapter_card.md``,
``merged_card.md``); every figure is a ``{{placeholder}}`` filled from the committed
records, the package, the dev runs, and what running the card's own code gave. The usage
code is a template too (``*_usage.py.txt``), so the code that is executed and the code
that is printed are the same text.

Usage::

    from qlora_finetuning.packaging.cards import build_card, usage_code

    code, problems = usage_code("adapter", cfg, budget, local_cfg)
    text, problems = build_card("adapter", cfg, facts)     # problems: what could not be filled

A card holds counts and figures, the invented sample, and no dataset text. This module
imports no part of the training stack.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from qlora_finetuning.comparison.cost_report import usd
from qlora_finetuning.comparison.report_build import PLACEHOLDER
from qlora_finetuning.comparison.report_tables import (
    CATEGORY_NAMES,
    METRIC_NAMES,
    span,
    summary_table,
    table,
)
from qlora_finetuning.data.serialization import format_user
from qlora_finetuning.data.token_budget import TokenBudgetConfig
from qlora_finetuning.demo.inputs import load_input, sample_path
from qlora_finetuning.evaluation.metrics import METRICS
from qlora_finetuning.local_qlora.adapters import ADAPTER_CONFIG
from qlora_finetuning.local_qlora.config import LocalQloraConfig
from qlora_finetuning.packaging.config import PackagingConfig
from qlora_finetuning.packaging.files import read_text_lf
from qlora_finetuning.packaging.sources import PUBLISHED_ARM

ADAPTER, MERGED = "adapter", "merged"
HERE = Path(__file__).parent
TEMPLATES = {ADAPTER: HERE / "adapter_card.md", MERGED: HERE / "merged_card.md"}
USAGE = {ADAPTER: HERE / "adapter_usage.py.txt", MERGED: HERE / "merged_usage.py.txt"}
LICENSE_NAMES = {
    "apache-2.0": "the Apache License, Version 2.0",
    "mit": "the MIT License",
    "cdla-permissive-1.0": "the Community Data License Agreement - Permissive, Version 1.0",
}
PRIMARY = "execution_accuracy"
# The order a card lists the metrics in: the main one first.
CARD_METRICS = (PRIMARY, *(metric for metric in METRICS if metric != PRIMARY))
SECONDS_PER_HOUR = Decimal(3600)


def fill(template: str, values: dict[str, str]) -> tuple[str, list[str]]:
    """The template with its placeholders filled; a problem for each one without a value."""
    named = sorted(set(PLACEHOLDER.findall(template)))
    problems = [f"the template names {{{{{name}}}}}, which has no value" for name in named
                if values.get(name) is None]
    text = PLACEHOLDER.sub(
        lambda match: (
            match.group(0) if values.get(match.group(1)) is None else str(values[match.group(1)])
        ),
        template,
    )
    return text, problems


def usage_code(
    kind: str, cfg: PackagingConfig, budget: TokenBudgetConfig, local_cfg: LocalQloraConfig
) -> tuple[str, list[str]]:
    """The code a card prints, and what could not be filled. The same text is executed."""
    return fill(
        read_text_lf(USAGE[kind]),
        {
            "adapter_repo": cfg.adapter_repo,
            "merged_repo": cfg.merged_repo,
            "base_model": budget.model_id,
            "base_commit": budget.revision,
            "quantization_type": local_cfg.quantization_type,
            "double_quant": str(bool(local_cfg.double_quant)),
            "compute_dtype": local_cfg.compute_dtype,
        },
    )


def header(card_text: str) -> dict[str, Any] | None:
    """The YAML header of a card, or ``None`` when it has none or it does not parse."""
    text = card_text.replace("\r\n", "\n")
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end == -1:
        return None
    try:
        data = yaml.safe_load(text[4:end])
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------- the values


def _count(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else "—"


def _percent(run: dict[str, Any] | None, metric: str = PRIMARY) -> str | None:
    return None if run is None else run["percent"][metric]


def _a1_table(arm: dict[str, Any]) -> str:
    rows = [
        [
            METRIC_NAMES[metric],
            _count(arm["metrics"][metric]["count"]),
            f"{arm['metrics'][metric]['percent']}%",
            span(arm["metrics"][metric]["interval"]),
        ]
        for metric in CARD_METRICS
    ]
    return table(["Metric", "Right", "Share", "Interval"], rows, right=(1, 2, 3))


def _six_arm_table(statistics: dict[str, Any]) -> str:
    text = summary_table(statistics)
    marked = text.replace(f"| {PUBLISHED_ARM} |", f"| **{PUBLISHED_ARM}** (this adapter) |", 1)
    return marked


def _steps_table(arm: dict[str, Any]) -> str:
    rows = [
        [group["steps"], _count(group["examples"]), _count(group["right"]), f"{group['percent']}%"]
        for group in arm["by_steps"]
    ]
    return table(
        ["Steps in the reference program", "Questions", "Right", "Execution accuracy"],
        rows,
        right=(1, 2, 3),
    )


def _categories_table(arm: dict[str, Any]) -> str:
    """The wrong replies by computed category, in the order the categories are tried."""
    rows = [
        [CATEGORY_NAMES[name].replace("gold", "reference"), _count(arm["categories"][name])]
        for name in CATEGORY_NAMES
        if arm["categories"].get(name)
    ]
    return table(["How the program differs", "Replies"], rows, right=(1,))


def _licence_table(cfg: PackagingConfig) -> str:
    rows = [
        [
            f"{entry.name}: {entry.role}",
            f"`{entry.identifier}`",
            entry.found,
            entry.read_on,
        ]
        for entry in cfg.licences
    ]
    return table(["What", "Licence", "What its source said", "Read on"], rows)


def _run_row(label: str, run: dict[str, Any], comparison: dict[str, Any] | None) -> list[str]:
    cells = [label, f"{run['counts'][PRIMARY]:,} of {run['examples']:,}", f"{_percent(run)}%"]
    if comparison is None:
        return cells + ["—", "—", "—"]
    return cells + [
        _count(comparison["replies_that_differ"]),
        _count(comparison["right_to_wrong"]),
        _count(comparison["wrong_to_right"]),
    ]


def _merged_dev_table(facts: Any) -> str | None:
    runs, comparisons = facts.runs, facts.comparisons
    if any(runs.get(label) is None for label in ("R1", "R3", "R4")):
        return None
    rows = [
        _run_row("The adapter on the 4-bit base model, for reference", runs["R1"], None),
        _run_row("This copy, loaded in 4-bit", runs["R3"], comparisons.get("R3")),
        _run_row("This copy, at full precision", runs["R4"], comparisons.get("R4")),
    ]
    return table(
        ["", "Right", "Execution accuracy", "Replies that differ from the adapter's",
         "Right there, wrong here", "Wrong there, right here"],
        rows,
        right=(1, 2, 3, 4, 5),
    )


def _gap_words(comparison: dict[str, Any] | None) -> str | None:
    if comparison is None:
        return None
    gap = comparison["right_other"] - comparison["right_reference"]
    if gap == 0:
        return "as many right as the adapter"
    return f"{abs(gap)} {'more' if gap > 0 else 'fewer'} right than the adapter"


def _merged_sentence(facts: Any) -> str | None:
    four, full = _gap_words(facts.comparisons.get("R3")), _gap_words(facts.comparisons.get("R4"))
    if four is None or full is None:
        return None
    return (
        f"Loaded in 4-bit, this copy has {four}; at full precision, {full}. This project "
        f"treats two runs that differ by fewer than {facts.allowance} of these questions as "
        "not told apart."
    )


def _package_sentence(facts: Any) -> str | None:
    comparison = facts.comparisons.get("R2")
    if comparison is None:
        return None
    total, differ = comparison["examples"], comparison["replies_that_differ"]
    if differ == 0:
        return (
            "The files of this repository, loaded from a folder, gave the same reply as the "
            f"original checkpoint on every one of the {total:,} development questions."
        )
    return (
        f"The files of this repository, loaded from a folder, gave another reply than the "
        f"original checkpoint on {differ:,} of the {total:,} development questions."
    )


def _merged_line(cfg: PackagingConfig, facts: Any) -> str:
    if facts.merged_status == "withheld":
        return (
            "- **Merged copy:** built and not published; on the development questions it "
            "fell outside the allowed difference from this adapter"
        )
    if facts.merged_status == "not built":
        return "- **Merged copy:** not published"
    return (
        f"- **Merged copy, which loads without `peft`:** "
        f"[`{cfg.merged_repo}`](https://huggingface.co/{cfg.merged_repo})"
    )


def _sample_texts(facts: Any) -> dict[str, Any]:
    sample = load_input(sample_path())
    user = format_user(
        sample.passages, [list(row) for row in sample.table], sample.question, facts.prompt,
        "the sample",
    )
    return {"sample_user": user}


def card_values(kind: str, cfg: PackagingConfig, facts: Any) -> dict[str, Any]:
    """Every value a card's template may name. One that cannot be known yet is ``None``."""
    statistics, cost = facts.records.statistics, facts.records.cost
    arm = statistics["arms"][PUBLISHED_ARM]
    manifest, source = facts.manifest, facts.source
    training = source.training
    settings, plan = training.get("settings") or {}, training.get("plan") or {}
    lora, optimizer = settings.get("lora") or {}, settings.get("optimizer") or {}
    batch, schedule = settings.get("batch") or {}, settings.get("schedule") or {}
    making = (cost.get("making") or {}).get(PUBLISHED_ARM) or {}
    train, select = making.get("train") or {}, making.get("select") or {}
    costs = (statistics.get("costs") or {}).get(PUBLISHED_ARM) or {}
    counts, split = manifest["counts"], manifest["settings"]
    adapter_config = json.loads(
        (source.path / ADAPTER_CONFIG).read_text(encoding="utf-8")
    )
    modules = ", ".join(f"`{name}`" for name in sorted(adapter_config.get("target_modules") or []))
    libraries = source.extras.get("libraries") or {}
    seconds = train.get("billable_seconds")
    code, _ = usage_code(kind, cfg, facts.budget, facts.local_cfg)
    card_run = facts.card_runs.get(ADAPTER if kind == ADAPTER else "merged_4bit") or {}
    merged_meta = (facts.merged_meta or {}).get("merge") or {}
    weights = (facts.merged_weights or {}).get("files") or []
    recorded = arm["dev"]
    r1 = facts.runs.get("R1")
    values: dict[str, Any] = {
        "base_model": facts.budget.model_id,
        "base_commit": facts.budget.revision,
        "license": cfg.license,
        "license_name": LICENSE_NAMES.get(cfg.license),
        "author": cfg.author,
        "code_repository": cfg.code_repository,
        "adapter_repo": cfg.adapter_repo,
        "merged_repo": cfg.merged_repo,
        "adapter_name": cfg.adapter_repo.split("/", 1)[1],
        "merged_name": cfg.merged_repo.split("/", 1)[1],
        "merged_line": _merged_line(cfg, facts),
        "dataset_id": manifest["dataset"]["repo_id"],
        "dataset_commit": manifest["dataset"]["revision"],
        "pooled_examples": _count(counts["examples"]),
        "fit_examples": _count(counts["fit"]["examples"]),
        "fit_companies": _count(counts["fit"]["groups"]),
        "dev_examples": _count(counts["dev"]["examples"]),
        "dev_companies": _count(counts["dev"]["groups"]),
        "test_examples": _count(counts["test"]["examples"]),
        "test_companies": _count(counts["test"]["groups"]),
        "train_percent": f"{(Decimal(str(split['train_fraction'])) * 100).normalize():f}",
        "split_seed": str(split["seed"]),
        "manifest_hash": manifest["content_hash"],
        "a1_execution": arm["metrics"]["execution_accuracy"]["percent"],
        "a1_program": arm["metrics"]["program_match"]["percent"],
        "a1_answer": arm["metrics"]["answer_accuracy"]["percent"],
        "a1_format": arm["metrics"]["format_validity"]["percent"],
        "a1_wrong": _count(arm["examples"] - arm["right"]),
        "a1_table": _a1_table(arm),
        "six_arm_table": _six_arm_table(statistics),
        "steps_table": _steps_table(arm),
        "steps_one": arm["by_steps"][0]["percent"],
        "steps_many": arm["by_steps"][-1]["percent"],
        "categories_table": _categories_table(arm),
        "confidence_percent": str(statistics["settings"]["confidence_percent"]),
        "recorded_dev_right": _count(recorded["execution_accuracy"]),
        "recorded_dev_percent": recorded["percent"],
        "r1_right": None if r1 is None else _count(r1["counts"][PRIMARY]),
        "r1_percent": _percent(r1),
        "r3_percent": _percent(facts.runs.get("R3")),
        "r4_percent": _percent(facts.runs.get("R4")),
        "package_sentence": _package_sentence(facts),
        "merged_dev_table": _merged_dev_table(facts),
        "merged_sentence": _merged_sentence(facts),
        "usage_code": code.rstrip("\n"),
        "sample_reply": (card_run.get("reply") or "").strip("\n") or None,
        "sample_value": None if card_run.get("value") is None else repr(card_run["value"]),
        "max_new_tokens": str(facts.budget.max_new_tokens),
        "max_length": _count(facts.budget.max_length),
        "batch_size": str(facts.local_cfg.batch_size),
        "quantization_type": facts.local_cfg.quantization_type.upper(),
        "compute_dtype": facts.local_cfg.compute_dtype,
        "precision": settings.get("precision"),
        "lora_r": lora.get("r"),
        "lora_alpha": lora.get("alpha"),
        "lora_dropout": lora.get("dropout"),
        "target_modules": modules or None,
        "trainable_parameters": _count(source.extras.get("trainable_parameters")),
        "batch": batch.get("size"),
        "accumulation": batch.get("accumulation"),
        "optimizer": None if not optimizer.get("name") else f"`{optimizer['name']}`",
        "learning_rate": optimizer.get("learning_rate"),
        "max_grad_norm": optimizer.get("max_grad_norm"),
        "schedule": schedule.get("type"),
        "warmup_steps": plan.get("warmup_steps"),
        "epochs": settings.get("epochs"),
        "steps_planned": _count(plan.get("total_steps")),
        "selected_step": str(source.step),
        "seed": settings.get("seed"),
        "instance_type": train.get("instance_type"),
        "training_hours": (
            None if seconds is None else f"{(Decimal(seconds) / SECONDS_PER_HOUR):.2f}"
        ),
        "libraries": ", ".join(f"{name} {version}" for name, version in libraries.items()) or None,
        "train_usd": None if train.get("usd") is None else usd(train["usd"]),
        "select_usd": None if select.get("usd") is None else usd(select["usd"]),
        "per_1000_questions": (
            None if costs.get("per_1000_questions") is None else usd(costs["per_1000_questions"])
        ),
        "per_1000_correct": (
            None if costs.get("per_1000_correct") is None else usd(costs["per_1000_correct"])
        ),
        "adapter_sha256": source.sha256,
        "prompt_id_short": facts.prompt.prompt_id[:12] + "...",
        "licence_table": _licence_table(cfg),
        "adapted_tensors": _count(merged_meta.get("adapted_tensors")),
        "merge_compute_dtype": merged_meta.get("compute_dtype"),
        "saved_dtype": merged_meta.get("saved_dtype"),
        "parameters": _count(merged_meta.get("parameters")),
        "weight_files": str(len(weights)) if weights else None,
        "weight_files_cell": ", ".join(f"`{entry['name']}`" for entry in weights) or None,
    }
    values.update(_sample_texts(facts))
    return values


def build_card(kind: str, cfg: PackagingConfig, facts: Any) -> tuple[str, list[str]]:
    """A card's text and what is wrong with it. Never written while it has a problem."""
    text, problems = fill(read_text_lf(TEMPLATES[kind]), card_values(kind, cfg, facts))
    if not problems and header(text) is None:
        problems.append("the card's header is not a YAML mapping between two --- lines")
    return text, problems
