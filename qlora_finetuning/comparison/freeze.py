"""The configuration freeze and the test unlock (req_005 v5.01).

Before the test split is read, the configuration of both paths is written down: for each of
the six arms, the name, model id, and settings of its predictor, taken from the arm's dev
result run; and for all arms together, the prompt, the split, the reply limit, and the
hashes of the shared config files and the gold flags. That is the **freeze record**
(``splits/config_freeze.json``, with a Markdown copy under ``reports/comparison/``).

The owner then unlocks, naming the freeze record by its hash. The unlock record
(``splits/test_unlock.json``) holds that hash and the commit that contains the freeze
record. From then on the gate (``gate.py``) allows a test run only at the frozen
configuration.

Usage::

    from qlora_finetuning.comparison.freeze import build_freeze, load_sources, unlock, write_freeze

    sources = load_sources(cfg, context)            # the committed records and the configs
    record = build_freeze(cfg, context, sources)    # checks F1 to F8; nothing is written
    if record["passed"]:
        digest = write_freeze(record, cfg)          # refused once the split is unlocked
    unlock(cfg, context, sources, yes=True, freeze=digest[:12])   # THE OWNER

Nothing here calls AWS or loads a model. The freeze record holds ids, hashes, counts,
settings, and ARNs: no example text, no person's name, no time stamp (the same state gives
the same bytes). Only ``unlock`` writes the unlock record, and only the owner runs it.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from qlora_finetuning.comparison.config import ComparisonConfig
from qlora_finetuning.comparison.gate import (
    PREDICTOR_FIELDS,
    SCHEMA,
    canonical_json,
    contract_now,
    file_sha256,
    predictor_differences,
    read_freeze,
    read_unlock,
    results_on_test,
    runs_on_test,
    shared_differences,
    shared_state,
    write_canonical,
)
from qlora_finetuning.data.split import load_split_config, read_manifest
from qlora_finetuning.data.split_loader import split_ids, unlock_state
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking import Ledger
from qlora_finetuning.tracking.registry import UNKNOWN_COMMIT, read_git_info

COMMAND = "python -m scripts.comparison_freeze"
REPORT_NAME = "config_freeze.md"
DEV = "dev"
TUNED_SUFFIX = "-tuned"
DEV_COUNTS = ("format_validity", "execution_accuracy", "answer_accuracy", "program_match")
CENT = Decimal("0.01")
MIN_PREFIX = 12  # how much of the freeze record's hash the owner types to unlock
UNLOCKED_BY = "the owner, req_005 v5.01"


def usd(amount: Decimal) -> str:
    """An amount for reading: dollars and cents, half a cent rounded up."""
    return f"${amount.quantize(CENT, rounding=ROUND_HALF_UP)}"


@dataclass
class Sources:
    """What the freeze is built from, besides the dev runs themselves.

    ``arms`` is each arm's predictor as today's configuration and records define it, with
    what identifies its model. ``dev_runs`` names each arm's dev result run. ``problems``
    holds what could not be read, by check id. ``estimator`` turns the dev runs' summaries
    into the estimate of the six test passes. A test builds one by hand.
    """

    arms: dict[str, dict[str, Any]] = field(default_factory=dict)
    dev_runs: dict[str, str] = field(default_factory=dict)
    problems: dict[str, list[str]] = field(default_factory=dict)
    estimator: Callable[[dict[str, dict[str, Any]]], dict[str, Any]] | None = None
    git: Any = None


# --------------------------------------------------------------------------- today's arms


def current_arms(
    cfg: ComparisonConfig, ctx: Any, config_dir: Path | str | None = None
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Each arm's predictor (name, model id, settings) as the configuration defines it now.

    Path A's settings are known without loading the model; arm A1 is the checkpoint of the
    selection record. Path B's tuned arms are the latest attempt of the tuning record.
    Returns the arms that could be worked out, and what went wrong for the others, under
    ``F1`` (Path A) and ``F2`` (Path B). Reads committed records and config; no AWS, no model.
    """
    from qlora_finetuning.data.token_budget import load_token_budget_config
    from qlora_finetuning.frontier.config import load_frontier_config
    from qlora_finetuning.frontier.tuned import tuned_attempt, tuned_details
    from qlora_finetuning.local_qlora.adapters import read_selection, selected_adapter
    from qlora_finetuning.local_qlora.config import load_local_qlora_config
    from qlora_finetuning.local_qlora.iteration_record import read_iteration
    from qlora_finetuning.local_qlora.scoring import expected_settings
    from qlora_finetuning.local_qlora.training_config import load_training_config

    root = cfg.root
    contract = contract_now(ctx)
    arms: dict[str, dict[str, Any]] = {}
    problems: dict[str, list[str]] = {"F1": [], "F2": []}

    path_a = [arm for arm in cfg.arms.values() if arm.path == "A"]
    if path_a:
        try:
            local_cfg = load_local_qlora_config(config_dir)
            budget = load_token_budget_config(root, config_dir)
            training_cfg = load_training_config(root, config_dir)
            for arm in path_a:
                identity: dict[str, Any] = {"revision": budget.revision}
                adapter_hash = None
                if arm.tuned:
                    selection = read_selection(training_cfg.selection_record)
                    adapter = selected_adapter(
                        training_cfg.selection_record,
                        model_id=budget.model_id,
                        revision=budget.revision,
                        prompt_id=contract["prompt_id"],
                        manifest_hash=contract["manifest_hash"],
                    )
                    adapter_hash = selection["selected"].get("adapter_sha256")
                    if not adapter_hash:
                        raise ConfigError("the selection record does not name the adapter's hash")
                    if not selection.get("verdict"):
                        raise ConfigError("the selection record holds no verdict")
                    identity.update(
                        adapter=adapter,
                        adapter_sha256=adapter_hash,
                        training_run=selection.get("training_run"),
                        step=selection["selected"].get("step"),
                        verdict=selection.get("verdict"),
                    )
                    iteration = read_iteration(training_cfg.iteration_record)
                    if iteration is not None and not iteration.get("second_run"):
                        raise ConfigError(
                            "a corrective iteration is registered and not finished "
                            f"({iteration['variable']} from {iteration['from']} to "
                            f"{iteration['to']}): finish it or withdraw it before the freeze"
                        )
                arms[arm.arm] = {
                    "predictor": {
                        "name": local_cfg.predictor_name + (TUNED_SUFFIX if arm.tuned else ""),
                        "model_id": budget.model_id,
                        "settings": expected_settings(local_cfg, budget.revision, adapter_hash),
                    },
                    "identity": identity,
                }
        except ConfigError as error:
            problems["F1"].append(str(error))

    path_b = [arm for arm in cfg.arms.values() if arm.path == "B"]
    if path_b:
        try:
            frontier_cfg = load_frontier_config(root, config_dir)
        except ConfigError as error:
            problems["F2"].append(str(error))
            path_b = []
        for arm in path_b:
            try:
                model = frontier_cfg.model(arm.model)
                identity = {"model": arm.model, "base_model_id": model.model_id}
                model_id = model.model_id
                if arm.tuned:
                    attempt = tuned_attempt(
                        arm.model, frontier_cfg, contract["prompt_id"], contract["manifest_hash"]
                    )
                    model_id = attempt.deployment_arn
                    identity.update(tuned_details(attempt))
                    identity.pop("tuned", None)
                arms[arm.arm] = {
                    "predictor": {
                        "name": arm.model + (TUNED_SUFFIX if arm.tuned else ""),
                        "model_id": model_id,
                        "settings": dict(frontier_cfg.settings),
                    },
                    "identity": identity,
                }
            except ConfigError as error:
                problems["F2"].append(str(error))
    return arms, problems


def dev_run_ids(
    cfg: ComparisonConfig, config_dir: Path | str | None = None
) -> tuple[dict[str, str], list[str]]:
    """Each arm's dev result run, and what could not be found.

    Path A's two runs are named by the selection record (the untuned model and the selected
    checkpoint). Path B's four are the dev pairs of the tuned-model report.
    """
    import json

    from qlora_finetuning.frontier.config import load_frontier_config
    from qlora_finetuning.frontier.tuned_report import report_paths
    from qlora_finetuning.local_qlora.adapters import read_selection
    from qlora_finetuning.local_qlora.training_config import load_training_config

    runs: dict[str, str] = {}
    problems: list[str] = []
    try:
        selection = read_selection(load_training_config(cfg.root, config_dir).selection_record)
        for arm in cfg.arms.values():
            if arm.path == "A":
                entry = selection["selected" if arm.tuned else "untuned"]
                if entry.get("eval_run_id"):
                    runs[arm.arm] = str(entry["eval_run_id"])
    except (ConfigError, KeyError, TypeError) as error:
        problems.append(f"Path A's dev runs are named by the selection record: {error}")

    try:
        frontier_cfg = load_frontier_config(cfg.root, config_dir)
        path = report_paths(frontier_cfg.report_dir)[1]
        report = json.loads(path.read_text(encoding="utf-8"))
        by_model = {entry["model"]: entry for entry in report["models"]}
        for arm in cfg.arms.values():
            if arm.path != "B":
                continue
            pairs = [
                pair
                for pair in by_model.get(arm.model, {}).get("pairs", [])
                if pair.get("split") == frontier_cfg.verdict_split
            ]
            if len(pairs) == 1:
                runs[arm.arm] = str(pairs[0]["tuned" if arm.tuned else "baseline"]["run_id"])
    except (OSError, ValueError, KeyError, TypeError, ConfigError) as error:
        problems.append(
            "Path B's dev runs are named by the tuned-model report "
            f"(python -m scripts.nova_tuned): {type(error).__name__}: {error}"
        )
    missing = [arm for arm in cfg.arms if arm not in runs]
    if missing and not problems:
        problems.append(f"no dev result run is named for: {', '.join(missing)}")
    return runs, problems


# --------------------------------------------------------------------------- the estimate


def estimate_test_passes(
    cfg: ComparisonConfig,
    dev: dict[str, dict[str, Any]],
    *,
    test_examples: int,
    path_a: dict[str, Any],
    path_b: dict[str, Any],
) -> dict[str, Any]:
    """What the six test passes are expected to cost, per arm and per path.

    ``path_a`` holds ``estimate(tuned) -> (seconds, cost, max_cost) | None``, ``spent``, and
    ``cap``. ``path_b`` holds ``cost(base_model_id, input_tokens, output_tokens)``,
    ``spent``, and ``cap``. A Nova arm's tokens are its dev run's, scaled to the test
    split's size: the test prompts are assumed as long as dev's on average, because they
    cannot be counted while the split is locked.
    """
    arms: dict[str, dict[str, Any]] = {}
    totals = {"A": Decimal(0), "B": Decimal(0)}
    most = {"A": Decimal(0), "B": Decimal(0)}
    problems: list[str] = []
    for arm in cfg.arms.values():
        if arm.path == "A":
            found = path_a["estimate"](arm.tuned)
            if found is None:
                problems.append(
                    f"arm {arm.arm}: no pulled job has measured this kind of work, so there "
                    "is no estimate"
                )
                continue
            seconds, cost, max_cost = found
            arms[arm.arm] = {
                "basis": f"{test_examples:,} replies in a SageMaker job",
                "seconds": int(seconds),
                "usd": usd(cost),
                "at_most_usd": usd(max_cost),
            }
        else:
            summary = dev.get(arm.arm)
            usage = (summary or {}).get("usage") or {}
            examples = (summary or {}).get("examples_predicted") or 0
            if not examples or usage.get("input_tokens") is None:
                problems.append(f"arm {arm.arm}: its dev run has no token counts to scale")
                continue
            scale = Decimal(test_examples) / Decimal(examples)
            tokens_in = int((usage["input_tokens"] * scale).to_integral_value(ROUND_HALF_UP))
            tokens_out = int(
                ((usage["output_tokens"] or 0) * scale).to_integral_value(ROUND_HALF_UP)
            )
            base = summary["predictor"].get("usage_model_id") or summary["predictor"]["model_id"]
            cost = max_cost = path_b["cost"](base, tokens_in, tokens_out)
            arms[arm.arm] = {
                "basis": "the dev run's tokens, scaled to the test split's size",
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "usd": usd(cost),
                "at_most_usd": usd(max_cost),
            }
        totals[arm.path] += cost
        most[arm.path] += max_cost

    paths: dict[str, dict[str, Any]] = {}
    for name, source in (("A", path_a), ("B", path_b)):
        spent, cap = Decimal(source["spent"]), Decimal(source["cap"])
        paths[name] = {
            "estimate_usd": usd(totals[name]),
            "at_most_usd": usd(most[name]),
            "spent_usd": usd(spent),
            "cap_usd": usd(cap),
        }
        if spent + most[name] > cap:
            problems.append(
                f"Path {name}: {usd(spent)} spent plus at most {usd(most[name])} for the test "
                f"passes is above the cap of {usd(cap)}"
            )
    return {
        "test_examples": test_examples,
        "arms": arms,
        "paths": paths,
        "total_usd": usd(totals["A"] + totals["B"]),
        "at_most_usd": usd(most["A"] + most["B"]),
        "problems": problems,
    }


def _real_estimator(
    cfg: ComparisonConfig, ctx: Any, config_dir: Path | str | None
) -> Callable[[dict[str, dict[str, Any]]], dict[str, Any]]:
    """The estimator over the real job record, ledger, and prices."""

    def estimate(dev: dict[str, dict[str, Any]]) -> dict[str, Any]:
        from qlora_finetuning.frontier.config import load_frontier_config
        from qlora_finetuning.frontier.pricing import inference_cost, load_pricing
        from qlora_finetuning.frontier.tuning import committed_path_b  # no call is made
        from qlora_finetuning.local_qlora.estimate import (
            committed_path_a,
            estimate_job,
            measured_rates,
        )
        from qlora_finetuning.local_qlora.job_record import read_job_record
        from qlora_finetuning.local_qlora.sagemaker_config import load_sagemaker_config

        split_cfg = ctx.split_cfg or load_split_config()
        test_examples = len(split_ids("test", split_cfg))
        try:
            sagemaker_cfg = load_sagemaker_config(cfg.root, config_dir)
            frontier_cfg = load_frontier_config(cfg.root, config_dir)
            pricing = load_pricing(config_dir)
            job_record = read_job_record(sagemaker_cfg.record_file)
            rates = measured_rates(job_record, cfg.root, sagemaker_cfg.instance_type)
            ledger = Ledger(ctx.tracking_cfg).effective()

            def path_a_estimate(tuned: bool) -> tuple[int, Decimal, Decimal] | None:
                found = estimate_job(
                    "evaluate", sagemaker_cfg, rates, examples=test_examples, tuned=tuned
                )
                if found is None:
                    return None
                limit = sagemaker_cfg.cost_of(
                    found.proposed_max_runtime, count=sagemaker_cfg.instance_count
                )
                return found.seconds, found.cost, limit

            return estimate_test_passes(
                cfg,
                dev,
                test_examples=test_examples,
                path_a={
                    "estimate": path_a_estimate,
                    "spent": committed_path_a(sagemaker_cfg, ctx.tracking_cfg, job_record),
                    "cap": sagemaker_cfg.path_a_cap_usd,
                },
                path_b={
                    "cost": lambda model, tokens_in, tokens_out: inference_cost(
                        model, tokens_in, tokens_out, pricing
                    ),
                    "spent": committed_path_b(ledger, pricing),
                    "cap": frontier_cfg.path_b_cap_usd,
                },
            )
        except ConfigError as error:
            return {"arms": {}, "paths": {}, "problems": [f"no estimate: {error}"]}

    return estimate


def load_sources(cfg: ComparisonConfig, ctx: Any, config_dir: Path | str | None = None) -> Sources:
    """Everything the freeze reads from the repository besides the dev runs. No AWS."""
    arms, problems = current_arms(cfg, ctx, config_dir)
    dev_runs, run_problems = dev_run_ids(cfg, config_dir)
    problems["F3"] = run_problems
    tracking = ctx.tracking_cfg
    git = ctx.git or read_git_info(
        tracking.root, tracking.git_commit_env, (tracking.ledger_prefix,)
    )
    return Sources(
        arms=arms,
        dev_runs=dev_runs,
        problems=problems,
        estimator=_real_estimator(cfg, ctx, config_dir),
        git=git,
    )


# --------------------------------------------------------------------------- the record


def _check(check_id: str, name: str, problems: list[str], ok_detail: str) -> dict[str, Any]:
    return {
        "id": check_id,
        "name": name,
        "passed": not problems,
        "detail": "; ".join(problems) if problems else ok_detail,
    }


def _dev_summaries(
    cfg: ComparisonConfig, ctx: Any, sources: Sources
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Each arm's dev run, summarized again under today's configuration."""
    from qlora_finetuning.evaluation.runner import summarize_run  # the gate may not; this may

    summaries: dict[str, dict[str, Any]] = {}
    problems = list(sources.problems.get("F3", []))
    for arm in cfg.arms:
        run_id = sources.dev_runs.get(arm)
        if not run_id:
            if not problems:
                problems.append(f"arm {arm} has no dev result run")
            continue
        try:
            summary = summarize_run(run_id, ctx)
        except ConfigError as error:
            problems.append(f"arm {arm}: run {run_id} cannot be read here ({error})")
            continue
        if summary["split"] != DEV or not summary["result"]:
            failed = [check["id"] for check in summary["checks"] if not check["passed"]]
            why = f"checks {', '.join(failed)} fail" if failed else "it is partial"
            problems.append(
                f"arm {arm}: run {run_id} is not a result on {DEV} under today's configuration "
                f"({'split ' + summary['split'] if summary['split'] != DEV else why})"
            )
        if summary["arm"] != arm:
            problems.append(f"arm {arm}: run {run_id} was recorded as arm {summary['arm']}")
        summaries[arm] = summary
    return summaries, problems


def build_freeze(cfg: ComparisonConfig, ctx: Any, sources: Sources) -> dict[str, Any]:
    """The freeze record with checks F1 to F8. Writes nothing.

    ``passed`` says whether it may be written. The record holds no time stamp.
    """
    split_cfg = ctx.split_cfg or load_split_config()
    contract = contract_now(ctx)
    shared = shared_state(cfg, ctx.config_dir, ctx.config.flags_file)
    dev, run_problems = _dev_summaries(cfg, ctx, sources)

    contract_problems = []
    for arm, summary in dev.items():
        for key in ("prompt_id", "manifest_hash", "max_new_tokens"):
            if summary.get(key) != contract[key]:
                contract_problems.append(
                    f"arm {arm}: its dev run was made with another {key.replace('_', ' ')}"
                )
    sizes = {summary["examples_predicted"] for summary in dev.values()}
    if len(sizes) > 1:
        contract_problems.append(
            f"the dev runs do not hold the same number of examples: {sorted(sizes)}"
        )
    missing_files = [
        f"config/{name}.yaml" for name, digest in shared["configs"].items() if digest is None
    ]
    if missing_files or shared["gold_flags_sha256"] is None:
        contract_problems.append(f"missing: {', '.join(missing_files) or 'the gold flags file'}")

    predictor_problems = []
    for arm in cfg.arms:
        if arm not in sources.arms or arm not in dev:
            continue
        differs = predictor_differences(sources.arms[arm]["predictor"], dev[arm]["predictor"])
        if differs:
            predictor_problems.append(
                f"arm {arm}: today's configuration differs from its dev run in {', '.join(differs)}"
            )
    undefined = [arm for arm in cfg.arms if arm not in sources.arms]
    if undefined and not (sources.problems.get("F1") or sources.problems.get("F2")):
        predictor_problems.append(f"no predictor is defined for: {', '.join(undefined)}")

    manifest = read_manifest(split_cfg)
    unlocked, reason = unlock_state(split_cfg, manifest)
    lock_problems = []
    if Path(split_cfg.unlock_file).exists():
        lock_problems.append(
            f"an unlock record exists ({reason}): the freeze is written before the unlock "
            "and never after it"
        )
    seen = {line.get("run_id") for line in results_on_test(ctx.config.results_file)} | {
        run["run_id"] for run in runs_on_test(ctx.tracking_cfg)
    }
    if seen:
        lock_problems.append(f"{len(seen)} run(s) have already asked test questions")

    git = sources.git
    git_problems = []
    if git is None or git.commit == UNKNOWN_COMMIT:
        git_problems.append("the commit is unknown: not a git repository")
    elif git.dirty:
        shown = ", ".join(git.changed[:3]) + (" ..." if len(git.changed) > 3 else "")
        git_problems.append(f"uncommitted changes ({shown}): commit first, then freeze")

    estimate = (
        sources.estimator(dev)
        if sources.estimator is not None
        else {"arms": {}, "paths": {}, "problems": ["no estimate was computed"]}
    )
    estimate_problems = list(estimate.pop("problems", []))

    arms: dict[str, Any] = {}
    for arm, arm_cfg in cfg.arms.items():
        if arm not in sources.arms:
            continue
        entry = {
            "path": arm_cfg.path,
            "where": arm_cfg.where,
            "predictor": {
                name: sources.arms[arm]["predictor"].get(name) for name in PREDICTOR_FIELDS
            },
            "identity": sources.arms[arm].get("identity") or {},
            "dev_run": sources.dev_runs.get(arm),
        }
        if arm in dev:
            every = dev[arm]["all"]
            entry["dev"] = {"examples": every["examples"]} | {
                name: every[name]["count"] for name in DEV_COUNTS
            }
        arms[arm] = entry

    path_a = [arm for arm, arm_cfg in cfg.arms.items() if arm_cfg.path == "A" and arm in arms]
    path_b = [arm for arm, arm_cfg in cfg.arms.items() if arm_cfg.path == "B" and arm in arms]
    checks = [
        _check(
            "F1",
            "Path A selected",
            list(sources.problems.get("F1", [])),
            "; ".join(
                f"{arm}: {arms[arm]['identity'].get('adapter') or 'the untuned model'}"
                for arm in path_a
            ),
        ),
        _check(
            "F2",
            "Path B tuned",
            list(sources.problems.get("F2", [])),
            "; ".join(
                f"{arm}: attempt {arms[arm]['identity']['attempt']}, deployment Active"
                for arm in path_b
                if "attempt" in arms[arm]["identity"]
            ),
        ),
        _check(
            "F3",
            "Six dev results",
            run_problems,
            f"{len(dev)} arms, each with one complete result on {DEV} under today's configuration",
        ),
        _check(
            "F4",
            "One contract",
            contract_problems,
            f"prompt {contract['prompt_id'][:12]}..., split manifest "
            f"{contract['manifest_hash'][:12]}..., reply limit {contract['max_new_tokens']}, "
            f"{next(iter(sizes), 0):,} examples each",
        ),
        _check(
            "F5",
            "Nothing changed since",
            predictor_problems,
            "every arm's predictor today has the name, model id, and settings of its dev run",
        ),
        _check("F6", "Still locked", lock_problems, f"locked: {reason}; no test run exists"),
        _check(
            "F7",
            "Committed",
            git_problems,
            f"commit {(git.commit if git else '')[:12]}, working tree clean",
        ),
        _check(
            "F8",
            "Within the caps",
            estimate_problems,
            f"the six test passes: about {estimate.get('total_usd')}, at most "
            f"{estimate.get('at_most_usd')}",
        ),
    ]
    return {
        "schema": SCHEMA,
        "generated_by": f"{COMMAND} freeze",
        "contract": contract | {"dev_examples": next(iter(sizes), None)},
        "shared": shared,
        "arms": arms,
        "estimate": estimate,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }


def freeze_digest(record: dict[str, Any]) -> str:
    """The SHA-256 the record will have on disk."""
    import hashlib

    return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def render_markdown(record: dict[str, Any]) -> str:
    """The freeze record for reading. Every figure is the JSON file's."""
    contract, estimate = record["contract"], record.get("estimate") or {}
    lines = [
        "# Configuration freeze",
        "",
        f"Generated by `{COMMAND} freeze`. Do not edit by hand.",
        "",
        "What each arm is, written down before the test split is read (req_005 v5.01). The",
        "record the code reads is `splits/config_freeze.json`; this file is the same content",
        "for reading. It holds ids, hashes, counts, and settings: no example text.",
        "",
        f"- SHA-256 of the record: `{freeze_digest(record)}`",
        f"- Prompt id: `{contract['prompt_id']}`",
        f"- Split manifest: `{contract['manifest_hash']}`",
        f"- Reply limit: {contract['max_new_tokens']} new tokens",
        "",
        "## Checks",
        "",
        "| Id | Check | Result | Detail |",
        "|---|---|---|---|",
    ]
    for check in record["checks"]:
        result = "PASS" if check["passed"] else "FAIL"
        lines.append(f"| {check['id']} | {check['name']} | {result} | {_cell(check['detail'])} |")
    lines += [
        "",
        "## The six arms",
        "",
        "| Arm | Predictor | Model | Runs in | Dev run | Execution accuracy on dev |",
        "|---|---|---|---|---|---:|",
    ]
    for arm, entry in record["arms"].items():
        dev = entry.get("dev") or {}
        score = f"{dev['execution_accuracy']:,} of {dev['examples']:,}" if dev else "not read"
        lines.append(
            f"| {arm} | `{entry['predictor']['name']}` | `{entry['predictor']['model_id']}` | "
            f"{entry['where']} | `{entry.get('dev_run')}` | {score} |"
        )
    lines += ["", "## Settings", ""]
    for arm, entry in record["arms"].items():
        settings = ", ".join(
            f"{key} {_cell(value)}" for key, value in sorted(entry["predictor"]["settings"].items())
        )
        lines.append(f"- **{arm}**: {settings}")
        identity = ", ".join(
            f"{key} `{_cell(value)}`"
            for key, value in sorted(entry["identity"].items())
            if not isinstance(value, dict)
        )
        if identity:
            lines.append(f"  - {identity}")
    lines += ["", "## Shared files", "", "| File | SHA-256 |", "|---|---|"]
    for name, digest in record["shared"]["configs"].items():
        lines.append(f"| `config/{name}.yaml` | `{digest}` |")
    lines.append(f"| the gold flags | `{record['shared']['gold_flags_sha256']}` |")
    lines += [
        "",
        "Hashes are taken over the content with Unix line ends, so they are the same on",
        "Windows and inside a SageMaker job.",
        "",
        "## What the six test passes are expected to cost",
        "",
        "The test prompts are assumed as long as dev's on average: they cannot be counted",
        "while the split is locked.",
        "",
        "| Arm | Basis | Estimate | At most |",
        "|---|---|---:|---:|",
    ]
    for arm, entry in (estimate.get("arms") or {}).items():
        lines.append(f"| {arm} | {entry['basis']} | {entry['usd']} | {entry['at_most_usd']} |")
    lines.append(
        f"| **All** | | **{estimate.get('total_usd')}** | **{estimate.get('at_most_usd')}** |"
    )
    lines += ["", "| Path | Spent so far | Test passes, at most | Cap |", "|---|---:|---:|---:|"]
    for name, entry in (estimate.get("paths") or {}).items():
        lines.append(
            f"| {name} | {entry['spent_usd']} | {entry['at_most_usd']} | {entry['cap_usd']} |"
        )
    return "\n".join(lines) + "\n"


def write_freeze(record: dict[str, Any], cfg: ComparisonConfig, split_cfg: Any = None) -> str:
    """Write the freeze record and its Markdown copy; return the record's SHA-256.

    Refused when a check failed, and once an unlock record exists: the freeze is written
    before the unlock and never after it.
    """
    split_cfg = split_cfg or load_split_config()
    if Path(split_cfg.unlock_file).exists():
        raise ConfigError(
            "an unlock record exists: the freeze record is written before the unlock and "
            "never rewritten after it"
        )
    if not record.get("passed"):
        raise ConfigError("a freeze check failed: nothing is written")
    digest = write_canonical(cfg.freeze_file, record)
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    (cfg.report_dir / REPORT_NAME).write_text(
        render_markdown(record), encoding="utf-8", newline="\n"
    )
    return digest


# --------------------------------------------------------------------------- does it still hold


def drift(
    stored: dict[str, Any], cfg: ComparisonConfig, ctx: Any, arms_now: dict[str, dict[str, Any]]
) -> list[str]:
    """What differs today from a freeze record: shared files, the contract, each predictor.

    Empty when nothing frozen has changed. Needs no run folder, no AWS, and no model, so a
    test of the suite can run it on the committed record.
    """
    differences = [
        f"{name} changed"
        for name in shared_differences(
            stored["shared"], shared_state(cfg, ctx.config_dir, ctx.config.flags_file)
        )
    ]
    now = contract_now(ctx)
    differences += [
        f"the {key.replace('_', ' ')} changed"
        for key, value in now.items()
        if stored["contract"].get(key) != value
    ]
    for arm, entry in stored["arms"].items():
        if arm not in arms_now:
            differences.append(f"arm {arm} can no longer be worked out from the configuration")
            continue
        differs = predictor_differences(entry["predictor"], arms_now[arm]["predictor"])
        if differs:
            differences.append(f"arm {arm}: {', '.join(differs)} changed")
    return differences


def freeze_status(cfg: ComparisonConfig, ctx: Any, sources: Sources) -> dict[str, Any]:
    """Frozen? Unlocked? Does the freeze still hold? Reads only."""
    split_cfg = ctx.split_cfg or load_split_config()
    stored = read_freeze(cfg)
    unlocked, reason = unlock_state(split_cfg, read_manifest(split_cfg))
    unlock_record = read_unlock(split_cfg) or {}
    digest = file_sha256(cfg.freeze_file)
    problems = [text for key in ("F1", "F2") for text in sources.problems.get(key, [])]
    return {
        "frozen": stored is not None,
        "freeze_sha256": digest,
        "unlocked": unlocked,
        "lock": reason,
        "unlock_names_freeze": bool(digest) and unlock_record.get("freeze_sha256") == digest,
        "unlocked_on": unlock_record.get("unlocked_on"),
        "freeze_commit": unlock_record.get("freeze_commit"),
        "differences": (drift(stored, cfg, ctx, sources.arms) if stored else []) + problems,
    }


# --------------------------------------------------------------------------- the unlock


def is_tracked(root: Path | str, path: Path | str) -> bool:
    """Whether git knows the file: it has been added and is not merely lying in the folder."""
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(path)],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def unlock(
    cfg: ComparisonConfig,
    ctx: Any,
    sources: Sources,
    *,
    yes: bool = False,
    freeze: str | None = None,
    tracked: Callable[[Path], bool] | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Unlock the test split. **Run by the owner, once.** There is no way to lock again.

    ``freeze`` is the beginning of the freeze record's SHA-256, at least twelve characters:
    typing it is the sign-off that this record was read and accepted. Refused unless the
    freeze record is committed, the working tree is clean, and nothing frozen has changed.
    Writes ``splits/test_unlock.json`` and returns its content; the owner commits it.
    """
    split_cfg = ctx.split_cfg or load_split_config()
    manifest = read_manifest(split_cfg)
    if Path(split_cfg.unlock_file).exists():
        unlocked, reason = unlock_state(split_cfg, manifest)
        raise ConfigError(
            f"an unlock record already exists ({reason}): the test split is unlocked once"
        )
    stored = read_freeze(cfg)
    digest = file_sha256(cfg.freeze_file)
    if stored is None or digest is None:
        raise ConfigError(f"there is no freeze record yet: run {COMMAND} freeze, and commit it")
    if not stored.get("passed"):
        raise ConfigError("the freeze record on disk did not pass its checks: freeze again")
    if not yes or not freeze:
        raise ConfigError(
            "unlock opens the test split for good and is run by the owner: "
            f"{COMMAND} unlock --yes --freeze {digest[:MIN_PREFIX]}  (the first "
            f"{MIN_PREFIX} characters of the freeze record's SHA-256, as the sign-off that "
            "the record was read)"
        )
    if len(freeze) < MIN_PREFIX or not digest.startswith(freeze.lower()):
        raise ConfigError(
            f"--freeze {freeze} is not the freeze record on disk (its SHA-256 begins "
            f"{digest[:MIN_PREFIX]}; give at least {MIN_PREFIX} characters)"
        )
    git = sources.git
    if git is None or git.commit == UNKNOWN_COMMIT:
        raise ConfigError("the commit is unknown: the unlock must name the freeze's commit")
    if git.dirty:
        shown = ", ".join(git.changed[:3]) + (" ..." if len(git.changed) > 3 else "")
        raise ConfigError(
            f"uncommitted changes ({shown}): commit the freeze record first, then unlock"
        )
    known = tracked or (lambda path: is_tracked(cfg.root, path))
    if not known(cfg.freeze_file):
        raise ConfigError(
            f"the freeze record {cfg.freeze_file.name} is not committed: git add and commit "
            "it, then unlock"
        )
    differences = drift(stored, cfg, ctx, sources.arms) + [
        text for key in ("F1", "F2") for text in sources.problems.get(key, [])
    ]
    if differences:
        raise ConfigError(
            "the configuration changed since the freeze: "
            + "; ".join(differences)
            + f". Undo the change, or freeze again ({COMMAND} freeze) and commit"
        )
    if results_on_test(ctx.config.results_file) or runs_on_test(ctx.tracking_cfg):
        raise ConfigError("a run has already asked test questions: this cannot be the unlock")

    record = {
        "schema": SCHEMA,
        "manifest_hash": str(manifest["content_hash"]),
        "freeze_sha256": digest,
        "freeze_commit": git.commit,
        "unlocked_on": (today or date.today()).isoformat(),
        "unlocked_by": UNLOCKED_BY,
    }
    write_canonical(Path(split_cfg.unlock_file), record)
    return record
