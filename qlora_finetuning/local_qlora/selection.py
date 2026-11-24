"""Choosing the checkpoint that becomes arm A1 (req_003 v3.04).

A training run leaves ten checkpoints, and the scoring job leaves one evaluation run for
each on dev. This module reads those runs, recomputes every count from the predictions
files, checks that the runs really are comparable (L1 to L6), and picks the checkpoint with
the best **execution accuracy** on dev. It never looks at the loss to choose.

What it writes, ``reports/local_qlora/adapter_selection.json`` and ``.md``, is committed and
is the one place that says which run is arm A1: ten runs carry that arm, and nothing
downstream may pick one by its arm alone.

Usage::

    from qlora_finetuning.local_qlora.selection import select_checkpoint

    outcome = select_checkpoint("<training run id>")    # writes the record if all is well
    outcome["passed"], outcome["checks"]                # L1 to L6
    outcome["selected"]["adapter"], outcome["verdict"]  # "RUN/checkpoint-N"; "met" | "missed"
    outcome["written"]                                  # the files; [] if none was written

    select_checkpoint(run, check=True)                  # report only; writes nothing

A verdict of ``missed`` starts nothing: it is the trigger of v3.05, which the owner decides
on. The record holds counts, ids, hashes, and amounts: no reply, no example text, and no
time stamp, so the same runs always give the same files.

From the command line: ``python -m scripts.qlora_select --run <training run id>``.

From req_003 v3.05 the reading of a training run's scored runs is a function of its own,
``gather_scored``, which the error analysis uses too. And a second training run, made as
the one registered corrective iteration, is compared with the checkpoint already
selected, by the same rule; a tie keeps the first run's. The record is then schema 2: the
winner's fields, and both runs under ``iterations``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from qlora_finetuning.data import manifest_hash
from qlora_finetuning.data.token_budget import load_token_budget_config
from qlora_finetuning.evaluation.metrics import EvaluationError
from qlora_finetuning.evaluation.runner import RunnerContext, summarize_run
from qlora_finetuning.local_qlora.adapters import (
    CHECKPOINTS,
    ITERATED_SCHEMA,
    SELECTION_SCHEMA,
    TRAINING_NAME,
    adapter_sha256,
    read_json,
    read_selection,
    training_chain,
    write_json,
)
from qlora_finetuning.local_qlora.iteration_record import read_iteration, record_outcome
from qlora_finetuning.local_qlora.job import JOBS_DIR, RESULT_NAME
from qlora_finetuning.local_qlora.job_record import read_job_record
from qlora_finetuning.local_qlora.sagemaker_config import SageMakerConfig, load_sagemaker_config
from qlora_finetuning.local_qlora.scoring import effective_limit, evaluation_runs
from qlora_finetuning.local_qlora.training import LOG_NAME, MEASUREMENTS_NAME
from qlora_finetuning.local_qlora.training_config import TrainingConfig, load_training_config
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking import Ledger, TrackingError, load_manifest

PASS, FAIL = "PASS", "FAIL"
MET, MISSED = "met", "missed"
METRICS = ("format_validity", "execution_accuracy", "answer_accuracy", "program_match")
CLOUD, MEASURED = "sagemaker", "measured"
CENT = Decimal("0.01")
FIRST, SECOND = "first", "second"
FIRST_SUFFIX = "_first"


def _check(check_id: str, name: str, problems: list[str], ok: str) -> dict[str, Any]:
    return {
        "id": check_id,
        "name": name,
        "passed": not problems,
        "detail": "; ".join(problems[:4]) + (" ..." if len(problems) > 4 else "")
        if problems
        else ok,
    }


def _percent(count: int, total: int) -> Decimal:
    return (Decimal(count) * 100 / Decimal(total)) if total else Decimal(0)


def _show(value: Decimal) -> str:
    return f"{value.quantize(CENT, rounding=ROUND_HALF_UP)}%"


def _row(run_id: str, evaluation: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """One model's result: ids and counts, nothing of what it wrote."""
    every = summary["all"]
    details = (evaluation.get("predictor") or {}).get("details") or {}
    settings = evaluation["key"]["predictor"]["settings"]
    backend = summary.get("backend") or {}
    row = {
        "step": details.get("checkpoint"),
        "adapter": (
            f"{details['training_run']}/checkpoint-{details['checkpoint']}"
            if details.get("checkpoint") is not None
            else None
        ),
        "adapter_sha256": settings.get("adapter"),
        "eval_run_id": run_id,
        "examples": every["examples"],
        "execution_accuracy_tight": summary["diagnostic"]["execution_accuracy_tight"]["count"],
        "peak_vram_allocated_gib": backend.get("peak_vram_allocated_gib"),
    }
    row.update({name: every[name]["count"] for name in METRICS})
    return row


def _dev_losses(chain: list[str], runs_dir: Path) -> dict[int, float]:
    """The loss on dev at each checkpoint, from the training log; a later run wins."""
    losses: dict[int, float] = {}
    for member in chain:
        path = runs_dir / member / LOG_NAME
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict) and "eval_loss" in entry and "step" in entry:
                losses[int(entry["step"])] = float(entry["eval_loss"])
    return losses


def _recorded_hashes(root: Path, chain: list[str]) -> dict[int, str]:
    """The hash each scoring job recorded for a checkpoint's weights, from its result file."""
    hashes: dict[int, str] = {}
    folder = root / JOBS_DIR
    if not folder.is_dir():
        return hashes
    for path in sorted(folder.glob(f"*/{RESULT_NAME}")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        for run in result.get("runs", []) if isinstance(result, dict) else []:
            if (
                run.get("role") == "score-checkpoint"
                and run.get("training_run") in chain
                and run.get("step") is not None
                and run.get("adapter_sha256")
            ):
                hashes[int(run["step"])] = str(run["adapter_sha256"])
    return hashes


@dataclass
class Scored:
    """A training run and the evaluation runs that scored its checkpoints, checked (L1 to L6)."""

    run: str
    chain: list[str]
    split: str
    limit: int | None
    rehearsal: bool
    smoke: bool
    steps: list[int]
    key: dict[str, Any]
    checks: list[dict[str, Any]] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)  # one per checkpoint, by step
    untuned: dict[str, Any] | None = None
    current: dict[str, Any] = field(default_factory=dict)  # what the run's key must equal
    measured: dict[str, Any] = field(default_factory=dict)
    jobs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.rows) and all(item["passed"] for item in self.checks)


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The rule: most right on execution, then format validity, then the earlier step."""
    return max(
        rows, key=lambda row: (row["execution_accuracy"], row["format_validity"], -row["step"])
    )


def gather_scored(
    run: str,
    *,
    context: RunnerContext | None = None,
    training_cfg: TrainingConfig | None = None,
    sagemaker_cfg: SageMakerConfig | None = None,
    budget: Any = None,
    split: str | None = None,
    limit: int | None = None,
    check: bool = False,
) -> Scored:
    """Find and check the runs that scored training run ``run``'s checkpoints.

    Another ``split`` than the configured one, a ``limit``, or a smoke run is a rehearsal
    and is allowed only with ``check``. When L1 fails there are no rows and no further
    checks. Raises ``ConfigError`` for an unknown run or a rehearsal without ``check``.
    """
    ctx = (context or RunnerContext()).resolved()
    tracking = ctx.tracking_cfg
    root, runs_dir = Path(tracking.root), tracking.runs_dir
    training_cfg = training_cfg or load_training_config(root=root)
    budget = budget or load_token_budget_config()
    chosen_split = split or training_cfg.selection_split
    rehearsal = limit is not None or chosen_split != training_cfg.selection_split
    if rehearsal and not check:
        raise ConfigError(
            "a selection is made from complete runs of the configured split "
            f"({training_cfg.selection_split}); with --split or --limit it is a rehearsal: "
            "add --check"
        )

    chain = training_chain(run, runs_dir)
    training = read_json(runs_dir / run / TRAINING_NAME, "the training run's record")
    plan, key = training.get("plan") or {}, training.get("key") or {}
    smoke = bool(training.get("smoke"))
    if smoke and not check:
        raise ConfigError(
            f"run {run} is a smoke run: its checkpoints are never arm A1. "
            "Add --check to rehearse the selection on it"
        )
    rehearsal = rehearsal or smoke
    steps = [int(step) for step in plan.get("checkpoint_steps", [])]
    if not steps:
        raise ConfigError(f"run {run} is not a training run with checkpoints")
    try:
        train_manifest = load_manifest(run, tracking)
    except TrackingError as error:
        raise ConfigError(f"training run {run} is not in the registry here: {error}") from error
    where = train_manifest["where"]
    wanted_limit = effective_limit(chosen_split, limit, ctx)

    # ---- every run that scored one of this training's checkpoints, or the untuned model
    by_step: dict[int, list[tuple[str, dict[str, Any]]]] = {step: [] for step in steps}
    untuned: list[tuple[str, dict[str, Any]]] = []
    manifests: dict[str, dict[str, Any]] = {}
    for run_id, evaluation in evaluation_runs(runs_dir).items():
        if (
            evaluation["key"].get("split") != chosen_split
            or evaluation.get("limit") != wanted_limit
        ):
            continue
        try:
            manifest = load_manifest(run_id, tracking)
        except TrackingError:
            continue
        if manifest["where"] != where:
            continue
        manifests[run_id] = manifest
        predictor = evaluation["key"].get("predictor") or {}
        details = (evaluation.get("predictor") or {}).get("details") or {}
        adapter = (predictor.get("settings") or {}).get("adapter")
        if adapter is None and predictor.get("model_id") == key.get("model_id"):
            untuned.append((run_id, evaluation))
        elif details.get("training_run") in chain and details.get("checkpoint") in by_step:
            by_step[int(details["checkpoint"])].append((run_id, evaluation))

    def complete(candidates: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict, dict]]:
        kept = []
        for run_id, evaluation in candidates:
            try:
                summary = summarize_run(run_id, ctx)
            except (ConfigError, EvaluationError):
                continue
            if summary["complete"] and (rehearsal or summary["result"]):
                kept.append((run_id, evaluation, summary))
        return kept

    scored = {step: complete(candidates) for step, candidates in by_step.items()}
    # The untuned run must have been made the way the checkpoints were: same settings but
    # for the adapter. Others (an earlier baseline at another batch size) are not it.
    reference = next((found[0][1] for found in scored.values() if found), None)
    if reference is not None:
        expected = dict(reference["key"]["predictor"]["settings"], adapter=None)
        untuned = [item for item in untuned if item[1]["key"]["predictor"]["settings"] == expected]
    untuned_scored = complete(untuned)

    # ---- L1: every checkpoint scored, once
    problems = [
        f"checkpoint {step:,} has {len(found)} complete results on {chosen_split}"
        for step, found in scored.items()
        if len(found) != 1
    ]
    if len(untuned_scored) != 1:
        problems.append(f"the untuned model has {len(untuned_scored)} complete results")
    checks = [
        _check(
            "L1",
            "Every checkpoint scored",
            problems,
            f"{len(steps)} checkpoints and the untuned model, one complete result each on "
            f"{chosen_split}",
        )
    ]
    scored_run = Scored(
        run=run,
        chain=chain,
        split=chosen_split,
        limit=wanted_limit,
        rehearsal=rehearsal,
        smoke=smoke,
        steps=steps,
        key=key,
        checks=checks,
    )
    if problems:
        return scored_run

    chosen = {step: found[0] for step, found in scored.items()}
    base = untuned_scored[0]
    everything = [*chosen.values(), base]
    losses = _dev_losses(chain, runs_dir)
    rows = []
    for step in steps:
        row = _row(*chosen[step])
        row["dev_loss"] = losses.get(step)
        rows.append(row)
    untuned_row = _row(*base)

    # ---- L2: same question
    current_manifest = manifest_hash(ctx.split_cfg)
    problems = []
    limits = {item[1]["key"].get("max_new_tokens") for item in everything}
    if len(limits) != 1:
        problems.append("the runs have different reply limits")
    plain = {
        json.dumps(dict(item[1]["key"]["predictor"]["settings"], adapter=None), sort_keys=True)
        for item in everything
    }
    if len(plain) != 1:
        problems.append("the runs were made with different settings (the adapter aside)")
    for run_id, evaluation, _ in everything:
        if evaluation["key"].get("prompt_id") != ctx.prompt.prompt_id:
            problems.append(f"run {run_id} was made with another prompt wording")
        if evaluation["key"].get("manifest_hash") != current_manifest:
            problems.append(f"run {run_id} was made with another split manifest")
    checks.append(
        _check(
            "L2", "Same question", problems, "one prompt, one split, one reply limit, one setting"
        )
    )

    # ---- L3: the right weights
    recorded = _recorded_hashes(root, chain)
    problems = []
    hashes = [row["adapter_sha256"] for row in rows]
    if len(set(hashes)) != len(hashes) or not all(hashes):
        problems.append("two checkpoints were scored with the same weights, or one with none")
    for row in rows:
        step = row["step"]
        member = str(row["adapter"]).split("/")[0]
        folder = runs_dir / member / CHECKPOINTS / f"checkpoint-{step}"
        known = recorded.get(step)
        if (folder / "adapter_model.safetensors").is_file():
            known = adapter_sha256(folder)
        if known is None:
            problems.append(f"no recorded hash for checkpoint {step:,}")
        elif known != row["adapter_sha256"]:
            problems.append(f"checkpoint {step:,} was scored with other weights than recorded")
    checks.append(
        _check(
            "L3", "The right weights", problems, "each run's adapter is its checkpoint's, by hash"
        )
    )

    # ---- L4: the run it claims
    problems = []
    if train_manifest.get("status") != "completed":
        problems.append(f"the training run is {train_manifest.get('status')}, not completed")
    try:
        measured = read_json(runs_dir / run / MEASUREMENTS_NAME, "the run's measurements")
    except ConfigError:
        measured = {}
    if measured.get("steps_done") != plan.get("total_steps"):
        problems.append(
            f"the run ended at step {measured.get('steps_done')}, "
            f"of {plan.get('total_steps')} planned"
        )
    current = {
        "model_id": budget.model_id,
        "revision": budget.revision,
        "prompt_id": ctx.prompt.prompt_id,
        "manifest_hash": current_manifest,
        "settings_sha256": training_cfg.settings_sha256(smoke),
    }
    differs = [name for name, value in current.items() if key.get(name) != value]
    if differs:
        problems.append(f"the run was made for another experiment ({', '.join(differs)} differ)")
    checks.append(
        _check(
            "L4",
            "The run it claims",
            problems,
            f"completed at step {plan.get('total_steps'):,}; model, prompt, split, and settings "
            "are the current ones",
        )
    )

    # ---- L5: measured in the cloud
    sagemaker_cfg = sagemaker_cfg or load_sagemaker_config(root=root)
    record = read_job_record(sagemaker_cfg.record_file)
    run_ids = {item[0] for item in everything} | set(chain)
    jobs = [
        job
        for job in record["jobs"]
        if {entry.get("run_id") for entry in job.get("runs", [])} & run_ids
    ]
    covered = {entry.get("run_id") for job in jobs for entry in job.get("runs", [])}
    billed = {
        entry.external_id
        for entry in Ledger(tracking).read()
        if entry.source == "aws_api" and entry.billable_seconds is not None
    }
    problems = []
    for run_id in sorted(run_ids):
        manifest = manifests.get(run_id) or load_manifest(run_id, tracking)
        if manifest["where"] != CLOUD or manifest["scope"] != MEASURED:
            problems.append(
                f"run {run_id} is {manifest['where']} and {manifest['scope']}, "
                f"not {CLOUD} and {MEASURED}"
            )
    for run_id in sorted(run_ids - covered):
        problems.append(f"run {run_id} belongs to no job in the job record")
    for job in jobs:
        if not job.get("pulled") or job["name"] not in billed:
            problems.append(
                f"job {job['name']} is not pulled, or its billable seconds are not recorded"
            )
    checks.append(
        _check(
            "L5",
            "Measured in the cloud",
            problems,
            f"{len(run_ids)} runs in {len(jobs)} pulled jobs, billable seconds recorded",
        )
    )

    # ---- L6: scores current
    problems = [
        f"run {item[0]} does not pass its own checks now"
        for item in everything
        if not item[2]["complete"]
    ]
    if not rehearsal:
        problems += [f"run {item[0]} is partial" for item in everything if not item[2]["result"]]
    checks.append(_check("L6", "Scores current", problems, "every run passes R1 to R5 now"))

    scored_run.rows, scored_run.untuned = rows, untuned_row
    scored_run.current, scored_run.measured = current, measured
    scored_run.jobs = [
        {
            "name": job["name"],
            "task": job.get("task"),
            "instance_type": job.get("instance_type"),
            "billable_seconds": job.get("billable_seconds"),
            "cost_usd": (
                str(
                    sagemaker_cfg.cost_of(
                        job["billable_seconds"],
                        job["instance_type"],
                        job.get("instance_count") or 1,
                    ).quantize(CENT, rounding=ROUND_HALF_UP)
                )
                if job.get("billable_seconds") is not None
                else None
            ),
        }
        for job in jobs
    ]
    return scored_run


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    """One run's part of an iterated record."""
    return {
        "training_run": record["training_run"],
        "chain": record["chain"],
        "settings_sha256": record["key"].get("settings_sha256"),
        "selected": record["selected"],
        "selected_execution_accuracy_percent": record["selected_execution_accuracy_percent"],
        "selected_format_validity_percent": record["selected_format_validity_percent"],
        "verdict": record["verdict"],
        "checkpoints": record["checkpoints"],
    }


def _first_of(existing: dict[str, Any]) -> dict[str, Any]:
    """What an existing record says of the first run: its id, chain, and selected row."""
    if existing["schema"] == ITERATED_SCHEMA:
        return existing["iterations"][0]
    return existing


def _against(
    existing: dict[str, Any] | None,
    scored_run: Scored,
    best: dict[str, Any],
    training_cfg: TrainingConfig,
    check: bool,
) -> dict[str, Any] | None:
    """Compare this run's best checkpoint with the one already selected (req_003 v3.05).

    ``None`` when there is nothing to compare with: no record, or a record of this run.
    A second run is compared only as the registered iteration; anything else is refused,
    or passed over when nothing would be written.
    """
    if existing is None or scored_run.rehearsal:
        return None
    first = _first_of(existing)
    first_chain = first.get("chain") or [first["training_run"]]
    if set(first_chain) & set(scored_run.chain):
        if existing["schema"] == ITERATED_SCHEMA and not check:
            raise ConfigError(
                "a corrective iteration has been made from this run: select with the second "
                f"run's id ({existing['iterations'][1]['training_run']}), which compares both"
            )
        return None
    iteration = read_iteration(training_cfg.iteration_record)
    problem = None
    if iteration is None or iteration.get("first_run") not in first_chain:
        problem = (
            f"another training run is selected ({first['training_run']}). A second run is "
            "selected only as the registered corrective iteration: "
            "python -m scripts.qlora_iterate propose"
        )
    elif iteration.get("settings_sha256") != scored_run.key.get("settings_sha256"):
        problem = (
            f"run {scored_run.run} was not made with the registered change "
            f"({iteration.get('variable')} to {iteration.get('to')})"
        )
    elif first["selected"]["examples"] != best["examples"]:
        problem = "the two runs were scored on different numbers of examples"
    if problem:
        if check:
            return None
        raise ConfigError(problem)
    incumbent = first["selected"]
    gap = best["execution_accuracy"] - incumbent["execution_accuracy"]
    second_wins = (best["execution_accuracy"], best["format_validity"]) > (
        incumbent["execution_accuracy"],
        incumbent["format_validity"],
    )
    return {
        "finding": iteration.get("finding"),
        "variable": iteration.get("variable"),
        "from": iteration.get("from"),
        "to": iteration.get("to"),
        "first_run": first["training_run"],
        "second_run": scored_run.run,
        "winner": SECOND if second_wins else FIRST,
        "gap_examples": gap,
        "noise_examples": training_cfg.noise_examples,
        "within_noise": abs(gap) < training_cfg.noise_examples,
    }


def select_checkpoint(
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
    """Choose the checkpoint of training run ``run`` with the best execution accuracy.

    With ``check`` nothing is written. Another ``split`` than the configured one, a
    ``limit``, or a smoke run is a rehearsal and is allowed only with ``check``. Raises
    ``ConfigError`` for an unknown run, a rehearsal that would write, or a second run that
    is not the registered iteration.
    """
    ctx = (context or RunnerContext()).resolved()
    root = Path(ctx.tracking_cfg.root)
    training_cfg = training_cfg or load_training_config(root=root)
    sagemaker_cfg = sagemaker_cfg or load_sagemaker_config(root=root)
    scored_run = gather_scored(
        run,
        context=ctx,
        training_cfg=training_cfg,
        sagemaker_cfg=sagemaker_cfg,
        budget=budget,
        split=split,
        limit=limit,
        check=check,
    )
    chain, chosen_split, checks = scored_run.chain, scored_run.split, scored_run.checks
    outcome: dict[str, Any] = {
        "run": run,
        "chain": chain,
        "split": chosen_split,
        "limit": scored_run.limit,
        "rehearsal": scored_run.rehearsal,
        "checks": checks,
        "passed": False,
        "rows": [],
        "untuned": None,
        "selected": None,
        "verdict": None,
        "notes": {},
        "written": [],
        "iteration": None,
    }
    if not scored_run.rows:
        return outcome
    rows, untuned_row, steps = scored_run.rows, scored_run.untuned, scored_run.steps
    key, current, measured = scored_run.key, scored_run.current, scored_run.measured
    job_rows = scored_run.jobs

    # ---- the choice, the verdict, the notes
    best = best_row(rows)
    execution = _percent(best["execution_accuracy"], best["examples"])
    validity = _percent(best["format_validity"], best["examples"])
    met = execution >= Decimal(str(training_cfg.min_execution_accuracy_percent)) and (
        validity >= Decimal(str(training_cfg.min_format_validity_percent))
    )
    with_loss = [row for row in rows if row["dev_loss"] is not None]
    lowest = min(with_loss, key=lambda row: (row["dev_loss"], row["step"])) if with_loss else None
    base_execution = _percent(untuned_row["execution_accuracy"], untuned_row["examples"])
    notes = {
        "selected_is_last_checkpoint": best["step"] == max(steps),
        "lowest_dev_loss_step": lowest["step"] if lowest else None,
        "execution_gain_points_over_untuned": str(
            (execution - base_execution).quantize(CENT, rounding=ROUND_HALF_UP)
        ),
    }
    passed = all(item["passed"] for item in checks)
    path = training_cfg.selection_record
    existing = read_selection(path) if path.is_file() else None
    against = _against(existing, scored_run, best, training_cfg, check)
    if against is not None:  # which checkpoint is arm A1 now, and its verdict
        first = _first_of(existing)
        second_wins = against["winner"] == SECOND
        against["verdict"] = (MET if met else MISSED) if second_wins else first["verdict"]
        against["selected"] = best["adapter"] if second_wins else first["selected"]["adapter"]
    outcome.update(
        passed=passed,
        rows=rows,
        untuned=untuned_row,
        selected=best,
        verdict=MET if met else MISSED,
        notes=notes,
        iteration=against,
    )
    if passed and not check:
        record_out = {
            "schema": SELECTION_SCHEMA,
            "training_run": run,
            "chain": chain,
            "split": chosen_split,
            "key": {name: key.get(name) for name in current},
            "selected": best,
            "untuned": untuned_row,
            "checkpoints": rows,
            "thresholds": {
                "min_execution_accuracy_percent": training_cfg.min_execution_accuracy_percent,
                "min_format_validity_percent": training_cfg.min_format_validity_percent,
            },
            "selected_execution_accuracy_percent": str(
                execution.quantize(CENT, rounding=ROUND_HALF_UP)
            ),
            "selected_format_validity_percent": str(
                validity.quantize(CENT, rounding=ROUND_HALF_UP)
            ),
            "verdict": outcome["verdict"],
            "notes": notes,
            "training_peak_vram_allocated_gib": (measured.get("trainer") or {}).get(
                "peak_vram_allocated_gib"
            ),
            "worst_batch_peak_vram_allocated_gib": (measured.get("probe") or {}).get(
                "peak_vram_allocated_gib"
            ),
            "jobs": job_rows,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        if against is not None:
            record_out = _iterated(existing, record_out, against, path)
            record_outcome(training_cfg.iteration_record, run, against)
        write_json(path, record_out)
        report = path.with_suffix(".md")
        report.write_text(render_report(record_out), encoding="utf-8", newline="\n")
        outcome["written"] = [str(path), str(report)]
        outcome["record"] = record_out
    return outcome


def _iterated(
    existing: dict[str, Any], second: dict[str, Any], against: dict[str, Any], path: Path
) -> dict[str, Any]:
    """The record after a corrective iteration: the winner's fields, and both runs.

    The first run's own record is kept beside the record, unchanged, the first time.
    """
    first_path = path.with_name(path.stem + FIRST_SUFFIX + path.suffix)
    if existing["schema"] == SELECTION_SCHEMA:
        first = existing
        write_json(first_path, first)
        first_path.with_suffix(".md").write_text(
            render_report(first), encoding="utf-8", newline="\n"
        )
    else:
        first = read_selection(first_path)
    winner = second if against["winner"] == SECOND else first
    jobs = {job["name"]: job for job in [*first["jobs"], *second["jobs"]]}
    record = dict(winner)
    record.update(
        schema=ITERATED_SCHEMA,
        iteration=dict(against),
        iterations=[_summary(first), _summary(second)],
        jobs=list(jobs.values()),  # what arm A1 cost: both runs
    )
    return record


def render_report(record: dict[str, Any]) -> str:
    """The selection as a page to read. Counts, ids, hashes, and amounts only."""
    selected, untuned = record["selected"], record["untuned"]
    total = selected["examples"]
    lines = [
        "# Arm A1: the selected checkpoint",
        "",
        f"Training run `{record['training_run']}`, scored on `{record['split']}` "
        f"({total:,} examples). Written by `python -m scripts.qlora_select`; do not edit.",
        "",
        f"**Selected**: checkpoint {selected['step']:,} (`{selected['adapter']}`), "
        f"weights `{str(selected['adapter_sha256'])[:12]}...`.",
        "",
        f"**Verdict**: {record['verdict']}. Execution accuracy "
        f"{record['selected_execution_accuracy_percent']}% (at least "
        f"{record['thresholds']['min_execution_accuracy_percent']:g}% asked), format validity "
        f"{record['selected_format_validity_percent']}% (at least "
        f"{record['thresholds']['min_format_validity_percent']:g}%).",
        "",
        "The rule: the most examples right by execution accuracy; a tie goes to the higher "
        "format validity, then to the earlier step. The loss on dev is shown and is never used.",
        "",
        "| Checkpoint | Execution | Format | Answer | Program | Execution, tight | Dev loss "
        "| Run |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def cells(row: dict[str, Any], label: str) -> str:
        loss = "" if row.get("dev_loss") is None else f"{row['dev_loss']:.4f}"
        return (
            f"| {label} | {row['execution_accuracy']:,} | {row['format_validity']:,} | "
            f"{row['answer_accuracy']:,} | {row['program_match']:,} | "
            f"{row['execution_accuracy_tight']:,} | {loss} | `{row['eval_run_id']}` |"
        )

    lines.append(cells(untuned, "untuned"))
    for row in record["checkpoints"]:
        mark = " **(selected)**" if row["step"] == selected["step"] else ""
        lines.append(cells(row, f"{row['step']:,}{mark}"))
    notes = record["notes"]
    lines += [
        "",
        "## Notes",
        "",
        f"- Above the untuned model by {notes['execution_gain_points_over_untuned']} points of "
        "execution accuracy.",
        (
            "- The selected checkpoint is the last one: the model may still have been improving."
            if notes["selected_is_last_checkpoint"]
            else "- The selected checkpoint is not the last one."
        ),
    ]
    if notes["lowest_dev_loss_step"] is not None:
        lines.append(f"- The loss on dev was lowest at step {notes['lowest_dev_loss_step']:,}.")
    lines += [
        "- This figure is the best of the checkpoints on the examples they were chosen on, so it "
        "flatters a little. The test split is what the comparison reports.",
        "",
        "## What it cost",
        "",
        "| Job | Task | Instance | Billable seconds | At the list price |",
        "|---|---|---|---:|---:|",
    ]
    if record.get("iteration"):
        lines[-4:-4] = _iteration_lines(record)  # before the heading "What it cost"
    for job in record["jobs"]:
        seconds = "" if job["billable_seconds"] is None else f"{job['billable_seconds']:,}"
        cost = "" if job["cost_usd"] is None else f"${job['cost_usd']}"
        lines.append(
            f"| `{job['name']}` | {job['task']} | {job['instance_type']} | {seconds} | {cost} |"
        )
    return "\n".join(lines) + "\n"


def _iteration_lines(record: dict[str, Any]) -> list[str]:
    """The section of the report that says what the corrective iteration did."""
    change, (first, second) = record["iteration"], record["iterations"]
    gap = change["gap_examples"]
    lines = [
        "## The corrective iteration",
        "",
        f"The first run missed a threshold, and one training setting was changed "
        f"(finding {change['finding']}): `{change['variable']}` from {change['from']} to "
        f"{change['to']}. The better of the two runs' best checkpoints, by the same rule, is "
        "arm A1; a tie keeps the first run's.",
        "",
        "| Run | Best checkpoint | Execution | Format | Verdict alone |",
        "|---|---:|---:|---:|---|",
    ]
    for label, part in (("first", first), ("second", second)):
        row = part["selected"]
        mark = " **(arm A1)**" if change["winner"] == label else ""
        lines.append(
            f"| {label}: `{part['training_run']}`{mark} | {row['step']:,} | "
            f"{row['execution_accuracy']:,} | {row['format_validity']:,} | {part['verdict']} |"
        )
    lines += [
        "",
        f"- The second run's best checkpoint has {gap:+,} examples right against the first's.",
        (
            f"- That is within noise (under {change['noise_examples']} examples): the change "
            "has not been shown to matter."
            if change["within_noise"]
            else f"- That is more than noise ({change['noise_examples']} examples)."
        ),
        "- Choosing between two runs on the examples they are measured on flatters the winner "
        "again. One change is one data point.",
        "",
    ]
    return lines
