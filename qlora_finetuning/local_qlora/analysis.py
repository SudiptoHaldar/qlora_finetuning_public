"""Where a trained adapter goes wrong on dev (req_003 v3.05).

Reads the evaluation runs that scored a training run's checkpoints (the ones the selection
was made from) and sorts every wrong reply into a computed category
(``qlora_finetuning.evaluation.failures``). It asks no model and uses no AWS.

The report says, for the selected checkpoint: how many replies fall in each category;
how that compares with the untuned model; how the categories move over the checkpoints;
where the failures sit (by the length of the gold program, by operation, by company, by
prompt length); which examples no checkpoint got right; and which row of the corrective
table the figures fit (the **finding**). The owner decides what to change; this module
only reports.

Usage::

    from qlora_finetuning.local_qlora.analysis import analyze_run

    outcome = analyze_run("<training run id>")          # writes the report for the selected run
    outcome["analysis"]["selected"]["counts"]["categories"]["order"]
    outcome["analysis"]["finding"]["row"]               # "T1" ... "T5", or "none"
    analyze_run(run, split="quick_dev", limit=24, check=True)   # a rehearsal; writes nothing

What it writes, ``reports/local_qlora/error_analysis.json`` and ``.md``, is committed and
holds counts, rates, run ids, and example ids: **no reply and no example text**.

From the command line: ``python -m scripts.qlora_analyze --run <training run id>``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qlora_finetuning.evaluation.failures import (
    BAD_PROGRAM,
    CATEGORIES,
    CUT_OFF,
    EVIDENCE,
    NO_PROGRAM,
    OPERATION,
    STEPS_FEWER,
    STEPS_MORE,
    Failure,
    company_of,
    count_failures,
    format_table,
    run_failures,
)
from qlora_finetuning.evaluation.runner import RunnerContext
from qlora_finetuning.local_qlora.adapters import read_json, read_selection, write_json
from qlora_finetuning.local_qlora.sagemaker_config import SageMakerConfig
from qlora_finetuning.local_qlora.selection import (
    FIRST_SUFFIX,
    Scored,
    best_row,
    gather_scored,
)
from qlora_finetuning.local_qlora.training_config import TrainingConfig, load_training_config
from qlora_finetuning.settings import ConfigError

ANALYSIS_SCHEMA = 1
RISING, FALLING, FLAT = "rising", "falling", "flat"
NONE = "none"
CALCULATION = (STEPS_MORE, STEPS_FEWER, OPERATION, EVIDENCE)  # wrong in the reasoning
LANGUAGE = (CUT_OFF, NO_PROGRAM, BAD_PROGRAM)  # wrong in what training should have taught
LONG_PROGRAM = 4  # gold programs of this many steps or more are one group


# --------------------------------------------------------------------------- the parts


def _trend(rows: list[dict[str, Any]], selected_step: int, noise: int) -> dict[str, Any]:
    """How execution accuracy moves over the checkpoints."""
    rights = [row["execution_accuracy"] for row in rows]
    steps = [row["step"] for row in rows]
    top = max(rights)
    last_three = FLAT
    if len(rights) >= 3:
        change = rights[-1] - rights[-3]
        last_three = RISING if change > noise else FALLING if change < -noise else FLAT
    later = rights[len(rights) // 2 :]
    position = steps.index(selected_step)
    return {
        "best_step": selected_step,
        "best_is_among_the_last_two": position >= len(steps) - 2,
        "best_is_in_the_first_half": selected_step * 2 <= steps[-1],
        "first_step_within_noise_of_best": next(
            step for step, right in zip(steps, rights, strict=True) if top - right <= noise
        ),
        "last_three": last_three,
        "drop_from_best_to_last": rights[position] - rights[-1],
        "range_from_the_middle_on": max(later) - min(later),
        "noise_examples": noise,
    }


def _group(failures: Sequence[Failure], names: Sequence[str], of: Any) -> list[dict[str, Any]]:
    """Right and wrong per group; an example may belong to several groups."""
    total, right = Counter(), Counter()
    for item in failures:
        for name in of(item):
            total[name] += 1
            right[name] += item.right
    order = [name for name in names if total[name]] + sorted(set(total) - set(names))
    return [{"group": name, "examples": total[name], "right": right[name]} for name in order]


def _by_prompt_length(failures: Sequence[Failure]) -> list[dict[str, Any]]:
    """Right and wrong by quarter of prompt length. Empty when token counts are missing."""
    known = sorted(
        (item for item in failures if item.input_tokens is not None),
        key=lambda item: (item.input_tokens, item.example_id),
    )
    if len(known) != len(failures) or len(known) < 4:
        return []
    quarters = []
    for index in range(4):
        part = known[index * len(known) // 4 : (index + 1) * len(known) // 4]
        quarters.append(
            {
                "group": f"quarter {index + 1}",
                "tokens_from": part[0].input_tokens,
                "tokens_to": part[-1].input_tokens,
                "examples": len(part),
                "right": sum(item.right for item in part),
            }
        )
    return quarters


def _steps_label(item: Failure) -> list[str]:
    if item.gold_steps == 0:
        return ["unreadable gold program"]
    return [f"{LONG_PROGRAM} or more" if item.gold_steps >= LONG_PROGRAM else str(item.gold_steps)]


def read_finding(
    trend: dict[str, Any],
    counts: dict[str, Any],
    dev_losses: dict[int, float | None],
    last_step: int,
    execution_met: bool,
    format_met: bool,
    changes: Sequence[tuple[str, str, float]],
) -> dict[str, Any]:
    """Which row of the corrective table the figures fit (spec section 4.3). Reports only."""
    table = {finding: (variable, value) for finding, variable, value in changes}
    categories = counts["categories"]
    wrong = counts["examples"] - counts["right"]

    def share(names: Sequence[str]) -> bool:
        return wrong > 0 and sum(categories[name] for name in names) * 2 > wrong

    best = trend["best_step"]
    best_loss, last_loss = dev_losses.get(best), dev_losses.get(last_step)
    rises = trend["best_is_among_the_last_two"] and trend["last_three"] == RISING
    falls = (
        trend["best_is_in_the_first_half"]
        and trend["drop_from_best_to_last"] > trend["noise_examples"]
        and best_loss is not None
        and last_loss is not None
        and last_loss > best_loss
    )
    flat = trend["range_from_the_middle_on"] <= trend["noise_examples"] and share(CALCULATION)

    def found(row: str, curve: str | None, reason: str) -> dict[str, Any]:
        variable, value = table.get(curve or row, (None, None))
        return {"row": row, "reads_as": curve, "variable": variable, "value": value, "why": reason}

    if share(LANGUAGE):
        return found(
            "T5",
            None,
            "most failures are cut off or are not programs: look for a fault in the pipeline "
            "before changing the recipe",
        )
    if execution_met and not format_met:
        if rises:
            return found("T4", "T1", "format validity missed; the curve is still rising")
        if falls:
            return found("T4", "T2", "format validity missed; accuracy peaked early and fell")
        return found(NONE, None, "format validity missed and the curve fits no row")
    if rises:
        return found("T1", None, "the best checkpoint is at the end and accuracy is still rising")
    if falls:
        return found("T2", None, "accuracy peaked in the first half and fell; the dev loss rose")
    if flat:
        return found(
            "T3", None, "accuracy is flat from the middle on and most failures are calculation"
        )
    return found(NONE, None, "the figures fit no row of the table")


# --------------------------------------------------------------------------- the analysis


def _selected_step(scored: Scored, record: dict[str, Any] | None) -> int:
    """The recorded selection's step when it is of this run; else this run's best."""
    if record is not None and not scored.rehearsal:
        adapter = str(record["selected"].get("adapter") or "")
        if adapter.split("/")[0] in scored.chain:
            return int(record["selected"]["step"])
    return int(best_row(scored.rows)["step"])


def analyze_run(
    run: str,
    *,
    context: RunnerContext | None = None,
    training_cfg: TrainingConfig | None = None,
    sagemaker_cfg: SageMakerConfig | None = None,
    budget: Any = None,
    split: str | None = None,
    limit: int | None = None,
    check: bool = False,
) -> dict[str, Any]:
    """Analyse the failures of training run ``run``'s scored checkpoints.

    Without ``check`` the run must be the one the selection record names, the checks L1 to
    L6 must pass, and the report is written. With ``check`` nothing is written, and any
    scored run may be looked at (a local smoke run, a partial split). Raises
    ``ConfigError`` for an unknown run, a rehearsal without ``check``, or a run that is not
    the selected one.
    """
    ctx = (context or RunnerContext()).resolved()
    training_cfg = training_cfg or load_training_config(root=Path(ctx.tracking_cfg.root))
    scored = gather_scored(
        run,
        context=ctx,
        training_cfg=training_cfg,
        sagemaker_cfg=sagemaker_cfg,
        budget=budget,
        split=split,
        limit=limit,
        check=check,
    )
    path = training_cfg.selection_record
    record = read_selection(path) if path.is_file() else None
    if not check:
        if record is None:
            raise ConfigError(
                "no checkpoint has been selected yet: run python -m scripts.qlora_select "
                "first, or add --check to look at these runs without writing"
            )
        if record["training_run"] not in scored.chain:
            raise ConfigError(
                f"the selected run is {record['training_run']}, not {run}: the report that "
                "is written is of the selected run. Add --check to look at this one"
            )
    outcome: dict[str, Any] = {
        "run": run,
        "split": scored.split,
        "limit": scored.limit,
        "rehearsal": scored.rehearsal,
        "checks": scored.checks,
        "passed": scored.passed,
        "analysis": None,
        "written": [],
    }
    if not scored.rows:
        return outcome

    rows, noise = scored.rows, training_cfg.noise_examples
    step = _selected_step(scored, record)
    by_step = {row["step"]: run_failures(row["eval_run_id"], ctx) for row in rows}
    selected_row = next(row for row in rows if row["step"] == step)
    chosen = by_step[step]
    untuned = run_failures(scored.untuned["eval_run_id"], ctx)
    counts = count_failures(chosen)
    total = counts["examples"]

    untuned_right = {item.example_id for item in untuned if item.right}
    tuned_right = {item.example_id for item in chosen if item.right}
    ever_right = {item.example_id for found in by_step.values() for item in found if item.right}
    trend = _trend(rows, step, noise)
    execution_met = counts["right"] * 100 >= training_cfg.min_execution_accuracy_percent * total
    in_format = sum(item.format_valid for item in chosen)
    format_met = in_format * 100 >= training_cfg.min_format_validity_percent * total
    by_category: dict[str, list[str]] = {name: [] for name in CATEGORIES}
    for item in chosen:
        if not item.right:
            by_category[item.category].append(item.example_id)

    analysis = {
        "schema": ANALYSIS_SCHEMA,
        "training_run": run,
        "split": scored.split,
        "examples": total,
        "selected": {
            "step": step,
            "adapter": selected_row["adapter"],
            "adapter_sha256": selected_row["adapter_sha256"],
            "eval_run_id": selected_row["eval_run_id"],
            "counts": counts,
            "format": format_table(chosen),
            "execution_threshold_met": execution_met,
            "format_threshold_met": format_met,
        },
        "untuned": {
            "eval_run_id": scored.untuned["eval_run_id"],
            "counts": count_failures(untuned),
            "fixed": len(tuned_right - untuned_right),
            "broken": len(untuned_right - tuned_right),
            "right_in_both": len(tuned_right & untuned_right),
            "wrong_in_both": total - len(tuned_right | untuned_right),
        },
        "checkpoints": [
            {
                "step": row["step"],
                "eval_run_id": row["eval_run_id"],
                "dev_loss": row.get("dev_loss"),
                "right": count_failures(by_step[row["step"]])["right"],
                "categories": count_failures(by_step[row["step"]])["categories"],
            }
            for row in rows
        ],
        "trend": trend,
        "where": {
            "by_gold_steps": _group(
                chosen, ("1", "2", "3", f"{LONG_PROGRAM} or more"), _steps_label
            ),
            "by_gold_operation": _group(chosen, (), lambda item: sorted(set(item.gold_operations))),
            "by_company": _group(chosen, (), lambda item: [company_of(item.example_id)]),
            "by_prompt_length": _by_prompt_length(chosen),
        },
        "never_right": {
            "count": total - len(ever_right),
            "example_ids": [
                item.example_id for item in chosen if item.example_id not in ever_right
            ],
        },
        "wrong_example_ids": {name: ids for name, ids in by_category.items() if ids},
        "finding": read_finding(
            trend,
            counts,
            {row["step"]: row.get("dev_loss") for row in rows},
            rows[-1]["step"],
            execution_met,
            format_met,
            training_cfg.iteration_changes,
        ),
    }
    outcome["analysis"] = analysis
    if scored.passed and not check:
        outcome["written"] = _write(analysis, training_cfg.analysis_record)
    return outcome


def _write(analysis: dict[str, Any], path: Path) -> list[str]:
    """Write the report. An earlier report of another run (the first of two) is kept."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report = path.with_suffix(".md")
    if path.is_file():
        earlier = read_json(path, "the error analysis")
        kept = path.with_name(path.stem + FIRST_SUFFIX + path.suffix)
        if earlier.get("training_run") != analysis["training_run"] and not kept.exists():
            write_json(kept, earlier)
            kept.with_suffix(".md").write_text(
                render_analysis(earlier), encoding="utf-8", newline="\n"
            )
    write_json(path, analysis)
    report.write_text(render_analysis(analysis), encoding="utf-8", newline="\n")
    return [str(path), str(report)]


# --------------------------------------------------------------------------- the page

_LABELS = {
    "cut_off": "Cut off at the token limit",
    "no_program": "No program line",
    "bad_program": "Program does not parse",
    "not_executable": "Program cannot be run",
    "kind": "Number where yes or no was due, or the reverse",
    "steps_more": "More steps than the gold program",
    "steps_fewer": "Fewer steps than the gold program",
    "operation": "Another operation",
    "evidence": "Other numbers, same operations",
    "order": "Arguments reversed",
    "scale": "Gold program, result off by a factor",
    "sign": "Gold program, result negated",
    "other": "Other",
}


def _share(count: int, total: int) -> str:
    return f"{100 * count / total:.1f}%" if total else ""


def render_analysis(analysis: dict[str, Any]) -> str:
    """The analysis as a page to read. Counts, rates, run ids, and example ids only."""
    selected, untuned, total = analysis["selected"], analysis["untuned"], analysis["examples"]
    counts, base = selected["counts"], untuned["counts"]
    finding, trend = analysis["finding"], analysis["trend"]
    lines = [
        "# Arm A1 on dev: where it goes wrong",
        "",
        f"Training run `{analysis['training_run']}`, checkpoint {selected['step']:,}, scored on "
        f"`{analysis['split']}` ({total:,} examples). Written by "
        "`python -m scripts.qlora_analyze`; do not edit. Every category is computed from the "
        "stored replies; none was assigned by reading.",
        "",
        f"**Right on execution**: {counts['right']:,} of {total:,} "
        f"({_share(counts['right'], total)}). **In the format**: "
        f"{selected['format']['in_the_format']:,} "
        f"({_share(selected['format']['in_the_format'], total)}).",
        "",
        "## Why replies are wrong",
        "",
        "A reply gets the first category that fits, in this order.",
        "",
        "| Category | Selected checkpoint | Share of all | Untuned model |",
        "|---|---:|---:|---:|",
    ]
    for name in CATEGORIES:
        lines.append(
            f"| {_LABELS[name]} (`{name}`) | {counts['categories'][name]:,} | "
            f"{_share(counts['categories'][name], total)} | {base['categories'][name]:,} |"
        )
    lines += [
        f"| **Right** | **{counts['right']:,}** | {_share(counts['right'], total)} | "
        f"{base['right']:,} |",
        "",
        f"Against the untuned model: {untuned['fixed']:,} examples fixed, {untuned['broken']:,} "
        f"broken, {untuned['right_in_both']:,} right in both, {untuned['wrong_in_both']:,} wrong "
        "in both.",
        "",
    ]
    if counts["error_codes"]:
        codes = ", ".join(f"`{code}` {count:,}" for code, count in counts["error_codes"].items())
        lines += [f"Programs that did not run, by error: {codes}.", ""]
    if counts["value_labels"]:
        labels = ", ".join(f"{name} {count:,}" for name, count in counts["value_labels"].items())
        lines += [
            f"Wrong results that are the gold value negated or off by a factor: {labels}.",
            "",
        ]
    lines += [
        f"Of the wrong replies, {counts['wrong_in_the_format']:,} were in the format, and "
        f"{counts['answer_right_program_wrong']:,} had a right answer line over a wrong program.",
        "",
        "## The format",
        "",
        f"{selected['format']['not_in_the_format']:,} replies were not in the strict format "
        f"({selected['format']['not_in_the_format_and_right']:,} of them right on execution).",
        "",
    ]
    if selected["format"]["deviations"]:
        lines += ["| Deviation | Replies |", "|---|---:|"]
        lines += [
            f"| `{name}` | {count:,} |" for name, count in selected["format"]["deviations"].items()
        ]
        lines.append("")
    reasons = ", ".join(
        f"`{name}` {count:,}" for name, count in selected["format"]["stop_reasons"].items()
    )
    lines += [f"Stop reasons: {reasons}.", "", "## Over the checkpoints", ""]
    shown = [
        name for name in CATEGORIES if any(c["categories"][name] for c in analysis["checkpoints"])
    ]
    lines.append(
        "| Checkpoint | Right | " + " | ".join(f"`{name}`" for name in shown) + " | Dev loss |"
    )
    lines.append("|---:|---:|" + "---:|" * len(shown) + "---:|")
    for point in analysis["checkpoints"]:
        mark = " **(selected)**" if point["step"] == selected["step"] else ""
        loss = "" if point["dev_loss"] is None else f"{point['dev_loss']:.4f}"
        cells = " | ".join(f"{point['categories'][name]:,}" for name in shown)
        lines.append(f"| {point['step']:,}{mark} | {point['right']:,} | {cells} | {loss} |")
    lines += [
        "",
        f"- The last three checkpoints are **{trend['last_three']}** (noise: "
        f"{trend['noise_examples']} examples).",
        f"- Accuracy was first within noise of its best at step "
        f"{trend['first_step_within_noise_of_best']:,}.",
        f"- From the best checkpoint to the last: {-trend['drop_from_best_to_last']:+,} examples.",
        "",
        "## Where the failures sit",
        "",
    ]
    for title, key in (
        ("By the length of the gold program (steps)", "by_gold_steps"),
        ("By operation in the gold program", "by_gold_operation"),
        ("By company", "by_company"),
        ("By prompt length", "by_prompt_length"),
    ):
        groups = analysis["where"][key]
        if not groups:
            continue
        lines += [f"**{title}**", "", "| Group | Examples | Right | Rate |", "|---|---:|---:|---:|"]
        for group in groups:
            label = group["group"]
            if "tokens_from" in group:
                label += f" ({group['tokens_from']:,} to {group['tokens_to']:,} tokens)"
            lines.append(
                f"| {label} | {group['examples']:,} | {group['right']:,} | "
                f"{_share(group['right'], group['examples'])} |"
            )
        lines.append("")
    lines += [
        f"**Never right**: {analysis['never_right']['count']:,} examples were wrong at every "
        "checkpoint. Their ids, and the ids of the wrong examples by category, are in the JSON "
        "file beside this page.",
        "",
        "## The finding",
        "",
    ]
    if finding["row"] == NONE:
        lines.append(f"**None of the table's rows**: {finding['why']}.")
    else:
        reads = f" (read as {finding['reads_as']})" if finding["reads_as"] else ""
        change = (
            f" The table's change: `{finding['variable']}` to {finding['value']:g}."
            if finding["variable"]
            else " The table names no change of recipe for it."
        )
        lines.append(f"**{finding['row']}**{reads}: {finding['why']}.{change}")
    lines += [
        "",
        "The finding is a reading of the table in the v3.05 spec, written before any result "
        "existed. It decides nothing: a corrective iteration is the owner's, and is allowed "
        "only when the selection's verdict is `missed`.",
    ]
    return "\n".join(lines) + "\n"
