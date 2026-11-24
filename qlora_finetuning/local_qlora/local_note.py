"""The record of the optional full training run on this machine (req_003 v3.06).

``reports/local_qlora/local_run.json`` and ``.md`` are committed. They say what the local
run was and how it did: hours, speed, peak GPU memory, the per-checkpoint table on
quick-dev, the checkpoint the selection rule picks there, that checkpoint's result on all of
dev, and the categories of its failures. Once the cloud selection exists (v3.04), a section
puts the two side by side.

**A local run is a side note. It is not an arm, it is never arm A1, and it has no price.**
That is why this is a file of its own: the selection record is arm A1, and
``scripts.qlora_select`` refuses local runs by design. Nothing reads this note as a
selection: not ``--adapter selected``, not ``keep``, nothing of the comparison.

=====  ================  ===================================================================
N1     Scored            every checkpoint and the untuned model have one complete result on
                         the split, with the right weights, for a run that completed under
                         the current settings (the selection's checks L1 to L4)
N2     Local             every run was made on this machine
N3     The full run      not a smoke run, and trained on all of ``fit``
N4     On dev            the selected checkpoint has exactly one complete result on all of dev,
                         made with the settings of its result on the split
N5     Scores current    every run passes the runner's checks today
=====  ================  ===================================================================

Usage::

    from qlora_finetuning.local_qlora.local_note import local_note

    outcome = local_note("<training run id>")          # writes the note if N1 to N5 pass
    outcome["passed"], outcome["checks"], outcome["note"]["selected"], outcome["written"]
    local_note(run, check=True)                        # writes nothing

The note holds counts, ids, hashes, and times: no reply, no example text, and no time stamp
of its writing, so the same runs give the same files.

From the command line: ``python -m scripts.qlora_local note --run <training run id>``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qlora_finetuning.data import split_ids
from qlora_finetuning.evaluation.failures import count_failures, run_failures
from qlora_finetuning.evaluation.metrics import EvaluationError
from qlora_finetuning.evaluation.runner import RunnerContext, summarize_run
from qlora_finetuning.local_qlora.adapters import (
    TRAINING_NAME,
    read_json,
    read_selection,
    write_json,
)
from qlora_finetuning.local_qlora.local_run import LABEL, LOCAL
from qlora_finetuning.local_qlora.sagemaker_config import SageMakerConfig
from qlora_finetuning.local_qlora.scoring import evaluation_runs
from qlora_finetuning.local_qlora.selection import METRICS, best_row, gather_scored
from qlora_finetuning.local_qlora.training import MEASUREMENTS_NAME
from qlora_finetuning.local_qlora.training_config import TrainingConfig, load_training_config
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking import TrackingError, load_manifest

NOTE_SCHEMA = 1
QUICK_DEV, DEV, FIT = "quick_dev", "dev", "fit"
SELECTION_CHECKS = ("L1", "L2", "L3", "L4")  # L5 asks for cloud runs and is not asked here
EVALUATE = "python -m scripts.evaluate --predictor qwen3-4b-tuned"


def _check(check_id: str, name: str, problems: list[str], ok: str) -> dict[str, Any]:
    return {
        "id": check_id,
        "name": name,
        "passed": not problems,
        "detail": "; ".join(problems[:3]) + (" ..." if len(problems) > 3 else "")
        if problems
        else ok,
    }


def _where(run_id: str, tracking: Any) -> str | None:
    try:
        return load_manifest(run_id, tracking).get("where")
    except TrackingError:
        return None


def _measurements(chain: list[str], runs_dir: Path) -> dict[str, Any]:
    """Hours, speed, and peak GPU memory over a run and the runs it continues."""
    seconds, known = 0.0, False
    rate, gpu = None, None
    peaks = {"allocated": None, "reserved": None}
    for member in chain:
        try:
            measured = read_json(runs_dir / member / MEASUREMENTS_NAME, "measurements")
        except ConfigError:
            continue
        if isinstance(measured.get("wall_clock_seconds"), (int, float)):
            seconds, known = seconds + measured["wall_clock_seconds"], True
        trainer = measured.get("trainer") or {}
        rate = trainer.get("seconds_per_sequence", rate)
        gpu = trainer.get("gpu_name", gpu)
        for part in (trainer, measured.get("probe") or {}):
            for name in peaks:
                value = part.get(f"peak_vram_{name}_gib")
                if isinstance(value, (int, float)):
                    peaks[name] = max(peaks[name] or 0.0, float(value))
    return {
        "hours": round(seconds / 3600, 2) if known else None,
        "seconds_per_sequence": rate,
        "gpu_name": gpu,
        "peak_vram_allocated_gib": peaks["allocated"],
        "peak_vram_reserved_gib": peaks["reserved"],
    }


def _counts(summary: dict[str, Any]) -> dict[str, Any]:
    every = summary["all"]
    counts = {"eval_run_id": summary["run_id"], "examples": every["examples"]}
    counts.update({name: every[name]["count"] for name in METRICS})
    return counts


def _categories(run_id: str, ctx: RunnerContext) -> dict[str, Any]:
    counts = count_failures(run_failures(run_id, ctx))
    return {"right": counts["right"], **counts["categories"]}


def _on_dev(selected: dict[str, Any], ctx: RunnerContext) -> tuple[list[dict[str, Any]], list[str]]:
    """The complete results on all of dev made with the selected checkpoint's settings."""
    runs_dir, tracking = ctx.tracking_cfg.runs_dir, ctx.tracking_cfg
    runs = evaluation_runs(runs_dir)
    settings = runs[selected["eval_run_id"]]["key"]["predictor"]["settings"]
    found, stale = [], []
    for run_id, evaluation in runs.items():
        key = evaluation["key"]
        if (
            key.get("split") != DEV
            or evaluation.get("limit") is not None
            or (key.get("predictor") or {}).get("settings") != settings
            or _where(run_id, tracking) != LOCAL
        ):
            continue
        try:
            summary = summarize_run(run_id, ctx)
        except (ConfigError, EvaluationError):
            stale.append(run_id)
            continue
        if summary["complete"] and summary["result"]:
            found.append(summary)
        elif summary["examples_predicted"] == summary["examples_asked"]:
            stale.append(run_id)  # all there, and a check of the runner fails today
    return found, stale


def _against_cloud(
    dev: dict[str, Any], local: dict[str, Any], training_cfg: TrainingConfig, runs_dir: Path
) -> dict[str, Any] | None:
    """The cloud's selected checkpoint beside the local one, or ``None`` before it exists."""
    path = training_cfg.selection_record
    if not path.is_file():
        return None
    record = read_selection(path)
    selected = record["selected"]
    cloud = _measurements(record.get("chain") or [record["training_run"]], runs_dir)
    billable = [
        job["billable_seconds"]
        for job in record.get("jobs", [])
        if job.get("task") == "train" and job.get("billable_seconds") is not None
    ]
    comparable = selected["examples"] == dev["examples"]
    gap = dev["execution_accuracy"] - selected["execution_accuracy"] if comparable else None
    return {
        "cloud_training_run": record["training_run"],
        "cloud_selected_step": selected["step"],
        "cloud_examples": selected["examples"],
        "cloud_execution_accuracy": selected["execution_accuracy"],
        "cloud_format_validity": selected["format_validity"],
        "cloud_verdict": record["verdict"],
        "cloud_training_hours": cloud["hours"],
        "cloud_seconds_per_sequence": cloud["seconds_per_sequence"],
        "cloud_gpu_name": cloud["gpu_name"],
        "cloud_training_billable_seconds": sum(billable) if billable else None,
        "local_selected_step": local["step"],
        "execution_gap_examples": gap,  # local minus cloud, on dev
        "noise_examples": training_cfg.noise_examples,
        "within_noise": abs(gap) < training_cfg.noise_examples if gap is not None else None,
    }


def local_note(
    run: str,
    *,
    context: RunnerContext | None = None,
    training_cfg: TrainingConfig | None = None,
    sagemaker_cfg: SageMakerConfig | None = None,
    budget: Any = None,
    split: str = QUICK_DEV,
    check: bool = False,
) -> dict[str, Any]:
    """Check the local run ``run`` and write its note.

    ``split`` is the split its checkpoints were scored on (quick-dev by default). With
    ``check`` nothing is written. Raises ``ConfigError`` for an unknown run.
    """
    ctx = (context or RunnerContext()).resolved()
    tracking, runs_dir = ctx.tracking_cfg, ctx.tracking_cfg.runs_dir
    training_cfg = training_cfg or load_training_config(root=Path(tracking.root))
    scored = gather_scored(
        run,
        context=ctx,
        training_cfg=training_cfg,
        sagemaker_cfg=sagemaker_cfg,
        budget=budget,
        split=split,
        check=True,  # the selection's own writing rules do not apply: this writes no selection
    )
    training = read_json(runs_dir / run / TRAINING_NAME, "the training run's record")
    plan = training.get("plan") or {}
    by_id = {item["id"]: item for item in scored.checks}

    # ---- N1: scored
    problems = [
        f"{item['id']}: {item['detail']}"
        for item in scored.checks
        if item["id"] in SELECTION_CHECKS and not item["passed"]
    ]
    n1 = _check(
        "N1",
        "Scored",
        problems,
        f"{len(scored.steps)} checkpoints and the untuned model on {split}; the right weights; "
        "the run completed under the current settings",
    )

    # ---- N2: local
    run_ids = [*scored.chain, *(row["eval_run_id"] for row in scored.rows)]
    if scored.untuned:
        run_ids.append(scored.untuned["eval_run_id"])
    elsewhere = [f"run {name} is {_where(name, tracking)}" for name in run_ids]
    elsewhere = [text for text, name in zip(elsewhere, run_ids, strict=True)
                 if _where(name, tracking) != LOCAL]  # fmt: skip
    if elsewhere:
        elsewhere.append("a cloud run has the selection record; this note is for local runs")
    n2 = _check("N2", "Local", elsewhere, f"{len(run_ids)} runs, all made on this machine")

    # ---- N3: the full run
    problems = []
    fit = len(split_ids(FIT, ctx.split_cfg))
    if scored.smoke:
        problems.append("a smoke run is a rehearsal, not the full run")
    elif plan.get("records") != fit:
        problems.append(f"the run trained on {plan.get('records')} records; fit has {fit:,}")
    n3 = _check("N3", "The full run", problems, f"trained on all {fit:,} records of fit")

    outcome: dict[str, Any] = {
        "run": run,
        "split": split,
        "label": LABEL,
        "checks": [n1, n2, n3],
        "selection_checks": scored.checks,
        "passed": False,
        "note": None,
        "written": [],
    }
    if not scored.rows:
        return outcome  # nothing is scored: there is no checkpoint to look for on dev

    selected = best_row(scored.rows)
    # ---- N4: on dev
    if split == DEV:
        dev_summary, problems = summarize_run(selected["eval_run_id"], ctx), []
        ok = "the checkpoints were scored on dev: the selected result is the dev result"
    else:
        found, stale = _on_dev(selected, ctx)
        dev_summary = found[0] if len(found) == 1 else None
        problems = []
        if not found:
            problems.append(
                f"checkpoint {selected['step']:,} has no complete result on dev: "
                f"{EVALUATE} --adapter {selected['adapter']} --split dev"
            )
        elif len(found) > 1:
            problems.append(f"checkpoint {selected['step']:,} has {len(found)} results on dev")
        ok = f"checkpoint {selected['step']:,} on dev: run {found[0]['run_id']}" if found else ""
    outcome["checks"].append(_check("N4", "On dev", problems, ok))

    # ---- N5: scores current
    problems = [] if by_id.get("L6", {}).get("passed") else [by_id.get("L6", {}).get("detail", "")]
    if split != DEV:
        problems += [f"run {name} does not pass its own checks now" for name in stale]
    outcome["checks"].append(
        _check("N5", "Scores current", problems, "every run passes R1 to R5 now")
    )

    outcome["passed"] = all(item["passed"] for item in outcome["checks"])
    if dev_summary is None:
        return outcome

    try:
        git = load_manifest(run, tracking).get("git") or {}
    except TrackingError:
        git = {}
    dev = _counts(dev_summary)
    dev["categories"] = _categories(dev["eval_run_id"], ctx)
    note = {
        "schema": NOTE_SCHEMA,
        "label": LABEL,
        "training_run": run,
        "chain": scored.chain,
        "interruptions": len(scored.chain) - 1,
        "git_commit": git.get("commit"),
        "from_a_clean_tree": git.get("dirty") is False,
        "settings_sha256": scored.key.get("settings_sha256"),
        "steps": plan.get("total_steps"),
        "records": plan.get("records"),
        "epochs": plan.get("epochs"),
        **_measurements(scored.chain, runs_dir),
        "selection_split": split,
        "untuned": scored.untuned,
        "checkpoints": scored.rows,
        "selected": selected,
        "selected_categories": _categories(selected["eval_run_id"], ctx),
        "selected_on_dev": dev,
        "against_cloud": _against_cloud(dev, selected, training_cfg, runs_dir),
    }
    outcome["note"] = note
    if outcome["passed"] and not check:
        path = training_cfg.local_note
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, note)
        page = path.with_suffix(".md")
        page.write_text(render_note(note), encoding="utf-8", newline="\n")
        outcome["written"] = [str(path), str(page)]
    return outcome


# --------------------------------------------------------------------------- the page


def _share(count: int, total: int) -> str:
    return f"{100 * count / total:.2f}%" if total else ""


def _figure(value: Any, unit: str = "", digits: int = 2) -> str:
    return "not recorded" if value is None else f"{value:,.{digits}f}{unit}"


def render_note(note: dict[str, Any]) -> str:
    """The local run as a page to read. Counts, ids, hashes, and times only."""
    selected, dev, total = note["selected"], note["selected_on_dev"], note["selected"]["examples"]
    stops = note["interruptions"]
    lines = [
        "# The full training run on this machine",
        "",
        f"**A {note['label']}.** It was made on the laptop's own GPU with the config of the "
        "cloud run, as a learning exercise. It is never arm A1, it has no AWS bill, and it "
        "enters no cost table. Written by `python -m scripts.qlora_local note`; do not edit.",
        "",
        f"Training run `{note['training_run']}`"
        + (f" (continued {stops} time{'s' if stops != 1 else ''} after stopping)" if stops else "")
        + f", commit `{str(note['git_commit'])[:12]}`"
        + ("" if note["from_a_clean_tree"] else " with uncommitted changes")
        + ".",
        "",
        "| | |",
        "|---|---|",
        f"| GPU | {note['gpu_name'] or 'not recorded'} |",
        f"| Records, epochs, optimizer steps | {note['records']:,}, {note['epochs']}, "
        f"{note['steps']:,} |",
        f"| Wall-clock time of training | {_figure(note['hours'], ' hours')} |",
        f"| Seconds per training sequence | {_figure(note['seconds_per_sequence'])} |",
        f"| Peak GPU memory | {_figure(note['peak_vram_allocated_gib'], ' GiB')} allocated, "
        f"{_figure(note['peak_vram_reserved_gib'], ' GiB')} reserved |",
        "",
        f"## The checkpoints on `{note['selection_split']}` ({total:,} examples)",
        "",
        "The rule of the cloud selection, applied here: the most examples right by execution "
        "accuracy, then format validity, then the earlier step. The loss on dev is shown and "
        "is never used.",
        "",
        "| Checkpoint | Execution | Format | Answer | Program | Dev loss | Run |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]

    def cells(row: dict[str, Any], label: str) -> str:
        loss = "" if row.get("dev_loss") is None else f"{row['dev_loss']:.4f}"
        return (
            f"| {label} | {row['execution_accuracy']:,} | {row['format_validity']:,} | "
            f"{row['answer_accuracy']:,} | {row['program_match']:,} | {loss} | "
            f"`{row['eval_run_id']}` |"
        )

    lines.append(cells(note["untuned"], "untuned"))
    for row in note["checkpoints"]:
        mark = " **(selected)**" if row["step"] == selected["step"] else ""
        lines.append(cells(row, f"{row['step']:,}{mark}"))
    lines += [
        "",
        f"**Selected here**: checkpoint {selected['step']:,} (`{selected['adapter']}`), weights "
        f"`{str(selected['adapter_sha256'])[:12]}...`: "
        f"{_share(selected['execution_accuracy'], total)} execution accuracy on "
        f"`{note['selection_split']}`.",
        "",
        "## The selected checkpoint on all of dev",
        "",
        f"{dev['execution_accuracy']:,} of {dev['examples']:,} right on execution "
        f"({_share(dev['execution_accuracy'], dev['examples'])}); {dev['format_validity']:,} in "
        f"the format ({_share(dev['format_validity'], dev['examples'])}). Run "
        f"`{dev['eval_run_id']}`.",
        "",
        "Why its replies were wrong there, by computed category:",
        "",
        "| Category | Replies |",
        "|---|---:|",
    ]
    lines += [
        f"| `{name}` | {count:,} |"
        for name, count in dev["categories"].items()
        if count and name != "right"
    ]
    lines += ["", "## Local against cloud", ""]
    cloud = note["against_cloud"]
    if cloud is None:
        lines.append(
            "Not yet: no cloud checkpoint has been selected. Run this command again after "
            "`scripts.qlora_select` has written the selection record."
        )
    else:
        gap = cloud["execution_gap_examples"]
        lines += [
            "| | This machine | Cloud (arm A1) |",
            "|---|---:|---:|",
            f"| GPU | {note['gpu_name'] or 'not recorded'} | "
            f"{cloud['cloud_gpu_name'] or 'not recorded'} |",
            f"| Training, hours | {_figure(note['hours'])} | "
            f"{_figure(cloud['cloud_training_hours'])} |",
            f"| Seconds per training sequence | {_figure(note['seconds_per_sequence'])} | "
            f"{_figure(cloud['cloud_seconds_per_sequence'])} |",
            f"| Selected checkpoint | {cloud['local_selected_step']:,} | "
            f"{cloud['cloud_selected_step']:,} |",
            f"| Right on execution, dev | {dev['execution_accuracy']:,} of {dev['examples']:,} | "
            f"{cloud['cloud_execution_accuracy']:,} of {cloud['cloud_examples']:,} |",
            f"| In the format, dev | {dev['format_validity']:,} | "
            f"{cloud['cloud_format_validity']:,} |",
            "",
        ]
        if gap is None:
            lines.append(
                "The two were scored on different numbers of examples and are not set "
                "against each other."
            )
        else:
            told = (
                f"within noise (under {cloud['noise_examples']} examples): the two runs are not "
                "told apart, which is what one pipeline in two places should give"
                if cloud["within_noise"]
                else f"more than noise ({cloud['noise_examples']} examples)"
            )
            lines.append(
                f"The local checkpoint has {gap:+,} examples right against the cloud's: {told}."
            )
        lines += [
            "",
            "The speeds compare two GPUs, not two costs: the cloud run is billed and this one "
            "is not, and no price is put on it.",
        ]
    lines += [
        "",
        "## What this is not",
        "",
        f"- The checkpoints were chosen on `{note['selection_split']}` and the cloud's on dev: "
        "the two per-checkpoint tables are not comparable row by row. Only the two selected "
        "checkpoints on dev are.",
        "- Two runs with one config and one seed are two adapters. A small difference says the "
        "pipeline is the same in both places, not that one GPU trains better.",
        "- No test example was used.",
    ]
    return "\n".join(lines) + "\n"
