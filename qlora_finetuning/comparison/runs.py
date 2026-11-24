"""The six test runs: where they stand, the record of them, and voiding one (req_005 v5.01).

After the unlock each arm answers the test split once. This module says which arms are
done, writes the committed **six-arm record** once all six are complete and comparable, and
lets the owner void a run that a fault in the pipeline or the service made worthless.

Usage::

    from qlora_finetuning.comparison.runs import arm_status, build_record, void_run, write_record

    arm_status(cfg, context)                 # per arm: done, unfinished, or missing
    record = build_record(cfg, context, job_record=jobs, arms_now=arms)   # checks C1 to C8
    if record["passed"]:
        write_record(record, cfg)            # reports/comparison/six_arm_test_runs.json, .md

    void_run(cfg, context, run_id, "the deployment was deleted mid-run", yes=True)   # THE OWNER

The six-arm record holds, per arm, the run id and the counts every later figure is computed
from: metrics over all examples and over the unflagged ones, tokens, latency, stop reasons,
and for a Qwen arm its job's billable seconds. It holds no confidence interval, no
comparison between arms, no dollar amount, and no example text: statistics are v5.03's and
costs are v5.02's. Later versions take the six run ids from this record and nowhere else.

Nothing here calls AWS or a model. ``preflight`` reads the test split, so it runs only
after the unlock, through the gate.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from qlora_finetuning.comparison.config import ComparisonConfig
from qlora_finetuning.comparison.gate import (
    MEASURED,
    NO_ARM,
    SCHEMA,
    TEST,
    canonical_json,
    check_test_request,
    file_sha256,
    predictor_differences,
    read_freeze,
    read_unlock,
    read_voids,
    results_on_test,
    run_chain,
    run_evaluation_record,
    runs_on_test,
    shared_differences,
    shared_state,
    voided_ids,
    write_canonical,
)
from qlora_finetuning.data import build_record as build_canonical_record
from qlora_finetuning.data import load_split
from qlora_finetuning.data.split import load_split_config
from qlora_finetuning.data.split_loader import split_ids
from qlora_finetuning.evaluation.fake import gold_predictor
from qlora_finetuning.evaluation.metrics import METRICS
from qlora_finetuning.evaluation.runner import run_evaluation, run_predictions, summarize_run
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking import load_manifest
from qlora_finetuning.tracking.config import check_no_secrets

COMMAND = "python -m scripts.comparison_runs"
REPORT_STEM = "six_arm_test_runs"
DONE, UNFINISHED, MISSING, TWICE = "done", "unfinished", "missing", "more than one result"
HUNDREDTH = Decimal("0.01")
MIN_REASON = 15  # characters: a void needs a reason someone can read later


def percent(count: int, total: int) -> str | None:
    """A share as text with two decimals, from counts: ``71.74``."""
    if not total:
        return None
    return str((Decimal(count) * 100 / Decimal(total)).quantize(HUNDREDTH, ROUND_HALF_UP))


# --------------------------------------------------------------------------- git


class GitQueries:
    """The two questions check C5 asks of git. A test passes a stand-in."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess | None:
        try:
            return subprocess.run(
                ["git", *arguments], cwd=self.root, capture_output=True, text=True,
                check=False, timeout=30,
            )  # fmt: skip
        except (OSError, subprocess.SubprocessError):
            return None

    def added_in(self, path: Path | str) -> str | None:
        """The commit that added ``path``, or ``None`` if it is not committed."""
        done = self._run("log", "--diff-filter=A", "--format=%H", "-1", "--", str(path))
        commit = (done.stdout or "").strip() if done and done.returncode == 0 else ""
        return commit or None

    def is_ancestor(self, earlier: str, later: str) -> bool:
        """Whether ``earlier`` is ``later`` or one of the commits it was built on."""
        done = self._run("merge-base", "--is-ancestor", earlier, later)
        return bool(done) and done.returncode == 0


# --------------------------------------------------------------------------- where the arms stand


def arm_status(cfg: ComparisonConfig, ctx: Any) -> list[dict[str, Any]]:
    """Each arm's state on the test split, from the results file and the run registry.

    ``done``: exactly one complete result that is not voided. ``unfinished``: a run was
    started and none completed; it is continued with ``--resume``. Reads only.
    """
    voided = voided_ids(cfg)
    status = []
    for arm in cfg.arms:
        results = [
            line["run_id"]
            for line in results_on_test(ctx.config.results_file, arm)
            if line.get("run_id") not in voided
        ]
        runs = [
            run["run_id"]
            for run in runs_on_test(ctx.tracking_cfg, arm)
            if run["run_id"] not in voided
        ]
        if len(results) == 1:
            state = DONE
        elif len(results) > 1:
            state = TWICE
        else:
            state = UNFINISHED if runs else MISSING
        status.append(
            {
                "arm": arm,
                "predictor": cfg.arms[arm].predictor,
                "where": cfg.arms[arm].where,
                "state": state,
                "results": results,
                "runs": runs,
                "resume": runs[0] if state == UNFINISHED else None,
            }
        )
    return status


# --------------------------------------------------------------------------- the record


def _check(check_id: str, name: str, problems: list[str], ok_detail: str) -> dict[str, Any]:
    return {
        "id": check_id,
        "name": name,
        "passed": not problems,
        "detail": "; ".join(problems) if problems else ok_detail,
    }


def _counts(block: dict[str, Any]) -> dict[str, Any]:
    total = block["examples"]
    return {"examples": total} | {
        name: {"count": block[name]["count"], "percent": percent(block[name]["count"], total)}
        for name in METRICS
    }


def _jobs_of(chain: list[str], job_record: dict[str, Any]) -> list[dict[str, Any]]:
    """The SageMaker jobs that made the runs of ``chain``, in the order they were launched."""
    wanted = set(chain)
    return [
        job
        for job in job_record.get("jobs", [])
        if wanted & {entry.get("run_id") for entry in job.get("runs", [])}
    ]


def build_record(
    cfg: ComparisonConfig,
    ctx: Any,
    *,
    job_record: dict[str, Any],
    arms_now: dict[str, dict[str, Any]],
    git: Any = None,
) -> dict[str, Any]:
    """The six-arm record with checks C1 to C8. Writes nothing.

    ``job_record`` is Path A's committed record of SageMaker jobs. ``arms_now`` is what
    ``freeze.current_arms`` says each arm is today: the selected adapter, the deployments.
    ``git`` answers check C5 (``GitQueries``). ``passed`` says whether the record may be
    written; with an arm missing it never is.
    """
    split_cfg = ctx.split_cfg or load_split_config()
    git = git or GitQueries(cfg.root)
    freeze = read_freeze(cfg)
    if freeze is None:
        raise ConfigError("there is no freeze record: nothing has been frozen or unlocked")
    unlock = read_unlock(split_cfg) or {}
    contract = freeze["contract"]
    test_ids = split_ids(TEST, split_cfg)
    voided = voided_ids(cfg)
    status = {entry["arm"]: entry for entry in arm_status(cfg, ctx)}

    problems: dict[str, list[str]] = {f"C{n}": [] for n in range(1, 9)}
    arms: dict[str, Any] = {}
    chosen: set[str] = set()
    for arm, arm_cfg in cfg.arms.items():
        entry = status[arm]
        if entry["state"] != DONE:
            hint = f" (continue it: --resume {entry['resume']})" if entry["resume"] else ""
            problems["C1"].append(f"arm {arm}: {entry['state']}{hint}")
            continue
        run_id = entry["results"][0]
        try:
            summary = summarize_run(run_id, ctx)
            lines = run_predictions(run_id, ctx)
        except ConfigError as error:
            problems["C1"].append(f"arm {arm}: run {run_id} cannot be read here ({error})")
            continue
        chain = run_chain(run_id, ctx.tracking_cfg.runs_dir)
        chosen.update(chain)
        if not summary["result"]:
            failed = ", ".join(c["id"] for c in summary["checks"] if not c["passed"])
            problems["C1"].append(
                f"arm {arm}: run {run_id} is not a result ({failed or 'partial'})"
            )

        if [line.example_id for line in lines] != test_ids:
            problems["C2"].append(
                f"arm {arm}: run {run_id} does not hold one prediction per test example in "
                f"the manifest's order ({len(lines):,} of {len(test_ids):,})"
            )
        for key in ("prompt_id", "manifest_hash"):
            if summary.get(key) != contract.get(key):
                problems["C3"].append(f"arm {arm}: made with another {key.replace('_', ' ')}")

        frozen = freeze["arms"].get(arm, {}).get("predictor") or {}
        differs = predictor_differences(frozen, summary["predictor"])
        if differs:
            problems["C4"].append(f"arm {arm}: its {', '.join(differs)} differ from the freeze")
        if summary.get("max_new_tokens") != contract.get("max_new_tokens"):
            problems["C4"].append(f"arm {arm}: made with another reply limit")
        if summary["scope"] != MEASURED or summary["where"] != arm_cfg.where:
            problems["C4"].append(
                f"arm {arm}: recorded as scope {summary['scope']} in {summary['where']}, not "
                f"{MEASURED} in {arm_cfg.where}"
            )

        commit = summary["git_commit"]
        unlock_commit = git.added_in(split_cfg.unlock_file)
        if unlock_commit and not git.is_ancestor(unlock_commit, commit):
            problems["C5"].append(
                f"arm {arm}: run {run_id} was made from commit {commit[:12]}, which does not "
                "contain the unlock record"
            )

        now = arms_now.get(arm)
        if now is None:
            problems["C6"].append(f"arm {arm}: its model can no longer be worked out")
        elif arm_cfg.tuned and predictor_differences(now["predictor"], summary["predictor"]):
            what = "selected adapter" if arm_cfg.path == "A" else "tuning record's deployment"
            problems["C6"].append(f"arm {arm}: run {run_id} did not ask the {what}")

        usage = summary["usage"]
        jobs = _jobs_of(chain, job_record) if arm_cfg.path == "A" else []
        if arm_cfg.path == "B":
            if usage["predictions_without_token_counts"] or (
                summary["predictor"].get("usage_source") != "aws_api"
            ):
                problems["C7"].append(f"arm {arm}: not every prediction has AWS's token counts")
        elif not jobs:
            problems["C7"].append(
                f"arm {arm}: no SageMaker job in the job record made run {run_id}"
            )
        else:
            unbilled = [
                job["name"]
                for job in jobs
                if not job.get("pulled") or job.get("billable_seconds") is None
            ]
            if unbilled:
                problems["C7"].append(
                    f"arm {arm}: job {unbilled[0]} is not pulled or has no billable seconds"
                )

        latency = summary["latency_seconds"]
        arms[arm] = {
            "run_id": run_id,
            "continues": chain[1:],
            "git_commit": commit,
            "path": arm_cfg.path,
            "where": summary["where"],
            "predictor": summary["predictor"]["name"],
            "model_id": summary["predictor"]["model_id"],
            "all": _counts(summary["all"]),
            "unflagged": _counts(summary["unflagged"]),
            "flagged_examples": summary["flagged_examples"],
            "execution_accuracy_tight": summary["diagnostic"]["execution_accuracy_tight"]["count"],
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "latency_seconds": {key: latency[key] for key in ("mean", "median", "p95")},
            "stop_reasons": summary["stop_reasons"],
            "jobs": [
                {
                    "name": job["name"],
                    "instance_type": job.get("instance_type"),
                    "billable_seconds": job.get("billable_seconds"),
                }
                for job in jobs
            ],
        }

    # C3, across arms: the same prompts give the same input tokens to one tokenizer.
    groups: dict[str, list[str]] = {}
    for arm, arm_cfg in cfg.arms.items():
        if arm in arms:
            groups.setdefault(arm_cfg.model or f"path {arm_cfg.path}", []).append(arm)
    for members in groups.values():
        tokens = {arms[arm]["input_tokens"] for arm in members}
        if len(tokens) > 1:
            problems["C3"].append(
                f"{' and '.join(members)} were sent different numbers of input tokens "
                f"({', '.join(str(arms[arm]['input_tokens']) for arm in members)})"
            )

    differs = shared_differences(
        freeze["shared"], shared_state(cfg, ctx.config_dir, ctx.config.flags_file)
    )
    if differs:
        problems["C4"].append(f"changed since the freeze: {', '.join(differs)}")

    digest = file_sha256(cfg.freeze_file)
    unlock_commit = git.added_in(split_cfg.unlock_file)
    if unlock.get("freeze_sha256") != digest:
        problems["C5"].append("the unlock record does not name the freeze record on disk")
    if unlock_commit is None:
        problems["C5"].append("the unlock record is not committed")
    elif unlock.get("freeze_commit") and not git.is_ancestor(
        str(unlock["freeze_commit"]), unlock_commit
    ):
        problems["C5"].append("the freeze record's commit does not come before the unlock's")

    stray = [
        line["run_id"]
        for line in results_on_test(ctx.config.results_file)
        if line.get("run_id") not in voided
        and line.get("run_id") not in chosen
        and not (line.get("arm") == NO_ARM and line.get("predictor") == cfg.gold_predictor)
    ]
    stray += [
        run["run_id"]
        for run in runs_on_test(ctx.tracking_cfg)
        if run["run_id"] not in voided
        and run["run_id"] not in chosen
        and run.get("arm") in cfg.arms
        and run["run_id"] not in stray
    ]
    if stray:
        shown = ", ".join(stray[:3]) + (", ..." if len(stray) > 3 else "")
        problems["C8"].append(
            f"{len(stray)} test run(s) are neither one of the six nor voided: {shown}"
        )

    count = len(arms)
    checks = [
        _check("C1", "Six arms", problems["C1"], f"{count} arms, each with one complete result"),
        _check(
            "C2",
            "Every question",
            problems["C2"],
            f"each run has one prediction for each of the {len(test_ids):,} test examples",
        ),
        _check(
            "C3",
            "Same questions",
            problems["C3"],
            f"prompt {str(contract.get('prompt_id'))[:12]}..., split manifest "
            f"{str(contract.get('manifest_hash'))[:12]}...; equal input tokens within a model",
        ),
        _check("C4", "Frozen", problems["C4"], "every run was made at the frozen configuration"),
        _check(
            "C5",
            "After the unlock",
            problems["C5"],
            f"freeze {str(unlock.get('freeze_commit'))[:12]}, then unlock "
            f"{str(unlock_commit)[:12]}, then every run",
        ),
        _check(
            "C6",
            "The right models",
            problems["C6"],
            "A1 asked the selected adapter; each B1 asked the tuning record's deployment",
        ),
        _check(
            "C7",
            "Usage recorded",
            problems["C7"],
            "AWS's token counts for every Nova prediction; billable seconds for every Qwen job",
        ),
        _check("C8", "No stray run", problems["C8"], "no other test run exists unvoided"),
    ]
    return {
        "schema": SCHEMA,
        "generated_by": f"{COMMAND} record",
        "split": TEST,
        "test_examples": len(test_ids),
        "freeze_sha256": digest,
        "unlock": {
            "unlocked_on": unlock.get("unlocked_on"),
            "freeze_commit": unlock.get("freeze_commit"),
            "commit": unlock_commit,
        },
        "contract": {
            key: contract.get(key) for key in ("prompt_id", "manifest_hash", "max_new_tokens")
        },
        "arms": arms,
        "deviations": read_voids(cfg),
        "checks": checks,
        "passed": all(check["passed"] for check in checks) and count == len(cfg.arms),
    }


def render_markdown(record: dict[str, Any]) -> str:
    """The six-arm record for reading. Every figure is the JSON file's."""
    lines = [
        "# The six test runs",
        "",
        f"Generated by `{COMMAND} record`. Do not edit by hand.",
        "",
        f"Each arm answered the {record['test_examples']:,} test examples once, at the "
        "configuration of the freeze",
        f"record `{str(record['freeze_sha256'])[:12]}...`. This file holds counts and ids "
        "only: no example text,",
        "no comparison between arms, no confidence interval, no dollar amount. Later "
        "versions take the",
        "six run ids from the JSON file beside it.",
        "",
        f"- Unlocked on {record['unlock']['unlocked_on']}",
        f"- Prompt id: `{record['contract']['prompt_id']}`",
        f"- Split manifest: `{record['contract']['manifest_hash']}`",
        "",
        "## Checks",
        "",
        "| Id | Check | Result | Detail |",
        "|---|---|---|---|",
    ]
    for check in record["checks"]:
        result = "PASS" if check["passed"] else "FAIL"
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| {check['id']} | {check['name']} | {result} | {detail} |")
    lines += [
        "",
        "## Results on the test split",
        "",
        "Counts over all examples, with the share in percent.",
        "",
        "| Arm | Execution accuracy | Format validity | Answer accuracy | Program match |",
        "|---|---:|---:|---:|---:|",
    ]

    def cell(block: dict[str, Any], name: str) -> str:
        return f"{block[name]['count']:,} ({block[name]['percent']}%)"

    for arm, entry in record["arms"].items():
        every = entry["all"]
        lines.append(
            f"| {arm} | {cell(every, 'execution_accuracy')} | {cell(every, 'format_validity')} "
            f"| {cell(every, 'answer_accuracy')} | {cell(every, 'program_match')} |"
        )
    lines += [
        "",
        "Without the flagged examples (gold answers that do not reproduce):",
        "",
        "| Arm | Examples | Execution accuracy |",
        "|---|---:|---:|",
    ]
    for arm, entry in record["arms"].items():
        block = entry["unflagged"]
        lines.append(f"| {arm} | {block['examples']:,} | {cell(block, 'execution_accuracy')} |")
    lines += [
        "",
        "## The runs",
        "",
        "| Arm | Run | Model | Input tokens | Output tokens | SageMaker job | Billable seconds |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for arm, entry in record["arms"].items():
        jobs = ", ".join(f"`{job['name']}`" for job in entry["jobs"])
        seconds = sum(job["billable_seconds"] or 0 for job in entry["jobs"])
        tokens_in = "" if entry["input_tokens"] is None else f"{entry['input_tokens']:,}"
        tokens_out = "" if entry["output_tokens"] is None else f"{entry['output_tokens']:,}"
        lines.append(
            f"| {arm} | `{entry['run_id']}` | `{entry['model_id']}` | {tokens_in} | "
            f"{tokens_out} | {jobs} | {f'{seconds:,}' if entry['jobs'] else ''} |"
        )
    lines += ["", "## Deviations", ""]
    if record["deviations"]:
        lines += ["| Arm | Voided runs | Reason | On |", "|---|---|---|---|"]
        for void in record["deviations"]:
            runs = ", ".join(f"`{run_id}`" for run_id in void["runs"])
            reason = str(void["reason"]).replace("|", "\\|")
            lines.append(f"| {void['arm']} | {runs} | {reason} | {void.get('voided_on')} |")
    else:
        lines.append("None: no test run was voided.")
    return "\n".join(lines) + "\n"


def report_paths(cfg: ComparisonConfig) -> tuple[Path, Path]:
    return cfg.report_dir / f"{REPORT_STEM}.md", cfg.report_dir / f"{REPORT_STEM}.json"


def write_record(record: dict[str, Any], cfg: ComparisonConfig) -> tuple[Path, Path]:
    """Write the six-arm record and its Markdown copy. Refused unless every check passed."""
    if not record.get("passed"):
        raise ConfigError("the six-arm record is written only when all six arms pass C1 to C8")
    check_no_secrets(record, REPORT_STEM)
    markdown_path, json_path = report_paths(cfg)
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(canonical_json(record), encoding="utf-8", newline="\n")
    markdown_path.write_text(render_markdown(record), encoding="utf-8", newline="\n")
    return markdown_path, json_path


# --------------------------------------------------------------------------- voiding a run


def void_run(
    cfg: ComparisonConfig,
    ctx: Any,
    run_id: str,
    reason: str,
    *,
    yes: bool = False,
    today: date | None = None,
) -> dict[str, Any]:
    """Void a test run, so that its arm may answer the test split once more. **The owner's.**

    Only for a run that a fault in the pipeline or the service made worthless, never
    because of what the model answered. The reason is committed with the list of voided
    runs and appears in the six-arm record as a deviation. The run and every run it
    continues are voided together.
    """
    reason = " ".join(str(reason or "").split())
    if not yes:
        raise ConfigError(
            "voiding a test run lets its arm read the test split again: it is the owner's "
            "decision, with --yes and a written --reason"
        )
    if len(reason) < MIN_REASON:
        raise ConfigError(
            f"a void needs a reason of at least {MIN_REASON} characters: what went wrong in "
            "the pipeline or the service"
        )
    check_no_secrets(reason, "the reason")
    runs_dir = ctx.tracking_cfg.runs_dir
    evaluation = run_evaluation_record(run_id, runs_dir)
    if evaluation is None or evaluation["key"].get("split") != TEST:
        raise ConfigError(f"run {run_id} is not a test run on this machine")
    arm = load_manifest(run_id, ctx.tracking_cfg)["arm"]
    if arm not in cfg.arms:
        raise ConfigError(f"run {run_id} is arm {arm}: only a run of the six arms is voided")
    voids = read_voids(cfg)
    already = {str(item) for void in voids for item in void["runs"]}
    if run_id in already:
        raise ConfigError(f"run {run_id} is already voided")
    later = [
        run["run_id"]
        for run in runs_on_test(ctx.tracking_cfg, arm)
        if run.get("continues") == run_id and run["run_id"] not in already
    ]
    if later:
        raise ConfigError(
            f"run {later[0]} continues run {run_id}: void the newest run of the chain, and "
            "the runs it continues are voided with it"
        )
    chain = run_chain(run_id, runs_dir)
    results = {line["run_id"] for line in results_on_test(ctx.config.results_file, arm)}
    entry = {
        "arm": arm,
        "runs": chain,
        "reason": reason,
        "had_result": run_id in results,
        "voided_on": (today or date.today()).isoformat(),
    }
    write_canonical(cfg.voids_file, {"schema": SCHEMA, "voids": [*voids, entry]})
    return entry


# --------------------------------------------------------------------------- after the unlock


def prompt_lengths(
    records: list[Any], counter: Callable[[Any], int] | None = None
) -> dict[str, Any]:
    """How long the test prompts are in Qwen tokens: counts only, no text.

    The token budget of v1.04 counted ``fit`` and ``dev``; the test split could not be
    counted while it was locked. A sequence over the training length is reported, never
    refused: inference has no such limit, but the adapter never saw one that long.
    """
    from qlora_finetuning.data.token_budget import (
        count_record,
        load_token_budget_config,
        load_tokenizer,
    )

    budget = load_token_budget_config()
    if counter is None:
        try:
            tokenizer = load_tokenizer(budget)
        except ConfigError as error:  # no training stack here, or the tokenizer is not cached
            return {"counted": False, "reason": str(error)}

        def counter(record: Any) -> int:
            return count_record(record, tokenizer, budget.thinking_off_suffix).total

    lengths = [counter(record) for record in records]
    return {
        "counted": True,
        "records": len(lengths),
        "longest": max(lengths, default=0),
        "max_length": budget.max_length,
        "over_max_length": sum(length > budget.max_length for length in lengths),
    }


def preflight(ctx: Any, counter: Callable[[Any], int] | None = None) -> dict[str, Any]:
    """The free check after the unlock: the gold predictor on test, and the prompt lengths.

    Raises ``TestLockedError`` while the split is locked and ``TestRunRefused`` when the
    unlock does not name the freeze record. Nothing is billed and no model is asked.
    """
    cfg = getattr(ctx, "comparison_cfg", None)
    gold = cfg.gold_predictor if cfg is not None else "gold"
    request = {"arm": NO_ARM, "name": gold, "where": "local", "scope": "dev"}
    check_test_request(**request, limit=None, resume=None, ctx=ctx)
    examples = load_split(TEST, ctx.split_cfg, ctx.data_cfg)
    records = [build_canonical_record(example, ctx.prompt) for example in examples]
    summary = run_evaluation(
        gold_predictor(records, name=gold),
        TEST,
        label=f"{gold} test",
        path="none",
        arm=NO_ARM,
        where="local",
        scope="dev",
        kind="smoke",
        context=ctx,
    )
    return {"gold": summary, "lengths": prompt_lengths(records, counter)}
