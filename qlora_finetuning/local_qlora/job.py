"""One task of Path A, the same here and in a cloud job (req_003 v3.03).

A SageMaker job runs exactly one task, and so can this machine:

- ``smoke``: the smoke training run, then a few quick-dev examples with its adapter and the
  same examples without;
- ``train``: the full training run, or the continuation of one;
- ``evaluate``: one evaluation run of the untuned model or of an adapter;
- ``score`` (v3.04): the untuned model and every checkpoint of a training run on one
  split, each an ordinary evaluation run. A checkpoint that already has a complete result
  is skipped and one that was cut short is resumed, so a stopped scoring job is simply
  run again.

A task calls ``run_training`` and ``run_evaluation`` and nothing else: no training or
evaluation logic exists for the cloud alone. It first checks that the installed libraries
are the pinned ones, and it always leaves ``job_result.json`` behind, also when it fails.

Usage::

    from qlora_finetuning.local_qlora.job import run_task

    result = run_task("smoke", where="sagemaker")
    result["status"]                      # "completed" | "failed" | "config_error"
    result["runs"]                        # each run it made: role, run id, what came of it
    result["libraries"], result["commit"]

    run_task("train", resume="<run_id>")
    run_task("evaluate", predictor="qwen3-4b-tuned", adapter="<run_id>/checkpoint-300", split="dev")
    run_task("score", run="<training run id>", split="dev")

From the command line: ``python -m scripts.job_run <task> ...``.
"""

from __future__ import annotations

import dataclasses
import gc
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qlora_finetuning.comparison.gate import TEST, check_test_request
from qlora_finetuning.data.token_budget import load_token_budget_config
from qlora_finetuning.environment import installed_versions, read_pins
from qlora_finetuning.evaluation.predictor import PredictorError
from qlora_finetuning.evaluation.runner import RunnerContext, run_evaluation
from qlora_finetuning.local_qlora.adapters import (
    CHECKPOINTS,
    adapter_sha256,
    chain_checkpoints,
)
from qlora_finetuning.local_qlora.config import load_local_qlora_config
from qlora_finetuning.local_qlora.sagemaker_config import load_sagemaker_config
from qlora_finetuning.local_qlora.scoring import (
    COMPLETE,
    INCOMPLETE,
    expected_settings,
    find_evaluation,
)
from qlora_finetuning.local_qlora.training import TrainingContext, TrainingError, run_training
from qlora_finetuning.logging_setup import get_logger
from qlora_finetuning.settings import REPO_ROOT, ConfigError, get_str
from qlora_finetuning.tracking.config import check_no_secrets

SCHEMA = 1
TASKS = ("smoke", "train", "evaluate", "score")
RESULT_NAME = "job_result.json"
JOBS_DIR = "artifacts/jobs"  # under the tracking root; in a cloud job, inside the workspace
JOB_NAME_ENV = "QLORA_JOB_NAME"
COMMIT_ENV = "QLORA_GIT_COMMIT"
COMPLETED, FAILED, CONFIG_ERROR = "completed", "failed", "config_error"
UNTUNED, TUNED = "qwen3-4b", "qwen3-4b-tuned"
# How each Qwen predictor is recorded. The same values as the table of scripts/evaluate.py,
# which the package may not import; a test holds the two together.
PREDICTORS: dict[str, dict[str, Any]] = {
    UNTUNED: {"arm": "A0", "configs": ("local_qlora",)},
    TUNED: {"arm": "A1", "configs": ("local_qlora", "training")},
}

logger = get_logger(__name__)


def pin_differences(pins_file: Path | str | None = None) -> list[str]:
    """Pinned libraries that are missing or installed at another version.

    A local build suffix is accepted (``2.14.1+cu126`` satisfies ``2.14.1``), as
    ``env_check`` accepts it.
    """
    pins = read_pins(pins_file) if pins_file is not None else read_pins()
    installed = installed_versions(pins)
    differences = []
    for package, wanted in pins.items():
        found = installed.get(package)
        if found is None:
            differences.append(f"{package} is not installed (pinned {wanted})")
        elif found.split("+")[0] != wanted:
            differences.append(f"{package} is {found}, pinned {wanted}")
    return differences


def _build(name: str, adapter: str | None, context: RunnerContext) -> Any:
    """The real predictors. The model is loaded here, before any run exists."""
    from qlora_finetuning.local_qlora.predictor import build_predictor, build_tuned_predictor

    if name == UNTUNED:
        if adapter:
            raise ConfigError(f"{UNTUNED} takes no adapter; use {TUNED}")
        return build_predictor()
    return build_tuned_predictor(adapter, context)


def _release(predictor: Any) -> None:
    """Let go of a loaded model, so that the next one fits on the GPU."""
    backend = getattr(predictor, "backend", None)
    if backend is not None and hasattr(backend, "model"):
        backend.model = None
    gc.collect()
    torch = getattr(backend, "_torch", None)
    if torch is not None:
        torch.cuda.empty_cache()


def _evaluate(
    name: str,
    split: str,
    *,
    adapter: str | None,
    limit: int | None,
    resume: str | None,
    where: str,
    scope: str,
    role: str,
    context: RunnerContext | None,
    evaluator: Callable[..., dict[str, Any]],
    factory: Callable[[str, str | None, RunnerContext], Any],
) -> dict[str, Any]:
    if name not in PREDICTORS:
        raise ConfigError(f"a job evaluates {UNTUNED} or {TUNED}, not {name!r}")
    entry = PREDICTORS[name]
    ctx = (context or RunnerContext()).resolved()
    extra = tuple(item for item in entry["configs"] if item not in ctx.configs)
    if extra:
        ctx = dataclasses.replace(ctx, configs=(*ctx.configs, *extra))
    if split == TEST:
        # The gate of req_005 v5.01, asked before the model is loaded. The runner asks it
        # again with the predictor. A refusal is a ConfigError: nothing is started.
        check_test_request(
            arm=entry["arm"],
            name=name,
            where=where,
            scope=scope,
            limit=limit,
            resume=resume,
            ctx=ctx,
        )
    predictor = factory(name, adapter, ctx)
    try:
        summary = evaluator(
            predictor,
            split,
            label=f"{name} {split}",
            path="A",
            arm=entry["arm"],
            where=where,
            scope=scope,
            kind="eval",
            limit=limit,
            resume=resume,
            context=ctx,
        )
    finally:
        _release(predictor)
    return _eval_entry(summary, role, adapter, split)


def _eval_entry(
    summary: dict[str, Any], role: str, adapter: str | None, split: str
) -> dict[str, Any]:
    """What the result file says about one evaluation run: ids, counts, measurements."""
    every = summary["all"]
    return {
        "role": role,
        "run_id": summary["run_id"],
        "status": COMPLETED,
        "predictor": summary["predictor"]["name"],
        "adapter": adapter,
        "split": split,
        "examples_predicted": summary["examples_predicted"],
        "complete": summary["complete"],
        "result": summary["result"],
        "format_validity": every["format_validity"]["count"],
        "execution_accuracy": every["execution_accuracy"]["count"],
        "backend": summary.get("backend"),
    }


def _train_entry(result: dict[str, Any]) -> dict[str, Any]:
    measurements = result["measurements"]
    return {
        "role": "train",
        "run_id": result["run_id"],
        "status": COMPLETED,
        "kind": result["kind"],
        "continues": result["continues"],
        "steps": result["steps"],
        "adapter": result["adapter"],
        "adapter_sha256": result["adapter_sha256"],
        "checks_passed": result["passed"],
        "checks_failed": [check["id"] for check in result["checks"] if not check["passed"]],
        "probe": measurements.get("probe"),
        "trainer": measurements.get("trainer"),
        "reload": measurements.get("reload"),
    }


def run_task(
    task: str,
    *,
    where: str = "local",
    scope: str = "dev",
    resume: str | None = None,
    predictor: str | None = None,
    adapter: str | None = None,
    split: str | None = None,
    limit: int | None = None,
    run: str | None = None,
    job_name: str | None = None,
    root: Path | str | None = None,
    check_pins: bool = True,
    runner_context: RunnerContext | None = None,
    training_context: TrainingContext | None = None,
    trainer: Callable[..., dict[str, Any]] = run_training,
    evaluator: Callable[..., dict[str, Any]] = run_evaluation,
    factory: Callable[[str, str | None, RunnerContext], Any] = _build,
    finder: Callable[..., tuple[Any, Any, Any]] = find_evaluation,
) -> dict[str, Any]:
    """Run ``task`` and return the content of ``job_result.json``, which is always written.

    Nothing is raised for a run that fails: the result says ``failed`` (a run failed, or a
    smoke check did) or ``config_error`` (nothing could be started), with what happened.
    """
    root = Path(root) if root is not None else REPO_ROOT
    job_name = job_name or get_str(JOB_NAME_ENV, f"local-{task}")
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "task": task,
        "job_name": job_name,
        "where": where,
        "scope": scope,
        "commit": get_str(COMMIT_ENV, "") or None,
        "libraries": {},
        "runs": [],
        "status": COMPLETED,
        "error": None,
    }
    runs: list[dict[str, Any]] = result["runs"]

    def evaluate(name: str, on: str, role: str, **options: Any) -> None:
        runs.append(
            _evaluate(
                name,
                on,
                where=where,
                scope=scope,
                role=role,
                context=runner_context,
                evaluator=evaluator,
                factory=factory,
                **options,
            )
        )

    try:
        if task not in TASKS:
            raise ConfigError(f"unknown task {task!r}; use one of: {', '.join(TASKS)}")
        if check_pins:
            result["libraries"] = installed_versions(read_pins())
            differences = pin_differences()
            if differences:
                raise ConfigError(
                    "the installed training stack is not the pinned one: " + "; ".join(differences)
                )

        if task == "smoke":
            trained = trainer(smoke=True, where=where, scope=scope, context=training_context)
            runs.append(_train_entry(trained))
            examples = load_sagemaker_config().smoke_examples
            tuned = {"adapter": trained["adapter"], "limit": examples, "resume": None}
            evaluate(TUNED, "quick_dev", "eval-tuned", **tuned)
            evaluate(
                UNTUNED, "quick_dev", "eval-untuned", adapter=None, limit=examples, resume=None
            )
            if not trained["passed"]:
                result["status"] = FAILED
                result["error"] = "a smoke check failed: " + ", ".join(runs[0]["checks_failed"])
        elif task == "train":
            trained = trainer(
                smoke=False, resume=resume, where=where, scope=scope, context=training_context
            )
            runs.append(_train_entry(trained))
            if not trained["passed"]:
                result["status"] = FAILED
                result["error"] = "a check failed: " + ", ".join(runs[0]["checks_failed"])
        elif task == "score":
            if not run:
                raise ConfigError("a score task needs --run, the id of a training run")
            _score(run, split or "dev", limit, runner_context, runs, evaluate, finder)
        else:
            if not predictor or not split:
                raise ConfigError("an evaluate task needs --predictor and --split")
            evaluate(predictor, split, "eval", adapter=adapter, limit=limit, resume=resume)
    except (TrainingError, PredictorError) as error:
        logger.error("the task failed: %s", error)
        role = "train" if isinstance(error, TrainingError) else "eval"
        runs.append({"role": role, "run_id": error.run_id, "status": FAILED})
        result["status"] = FAILED
        result["error"] = f"{type(error).__name__}: {error}"
    except ConfigError as error:  # nothing was started, or a resume or an adapter was refused
        logger.error("the task could not start: %s", error)
        result["status"] = CONFIG_ERROR
        result["error"] = f"{type(error).__name__}: {error}"

    write_result(result, root)
    return result


def _score(
    run: str,
    split: str,
    limit: int | None,
    runner_context: RunnerContext | None,
    runs: list[dict[str, Any]],
    evaluate: Callable[..., None],
    finder: Callable[..., tuple[Any, Any, Any]],
) -> None:
    """Score the untuned model and every checkpoint of ``run`` on ``split`` (req_003 v3.04).

    Each is its own evaluation run. What is already scored is taken as it is; what was cut
    short is resumed. Checkpoints come from the whole chain of a resumed training.
    """
    ctx = (runner_context or RunnerContext()).resolved()
    runs_dir = ctx.tracking_cfg.runs_dir
    checkpoints = chain_checkpoints(run, runs_dir)
    if not checkpoints:
        raise ConfigError(f"training run {run} has no checkpoint here: there is nothing to score")
    local_cfg, budget = load_local_qlora_config(), load_token_budget_config()

    targets: list[tuple[str, str | None, int | None, str | None]] = [(UNTUNED, None, None, None)]
    for step, name in checkpoints.items():
        member = name.split("/")[0]
        folder = runs_dir / member / CHECKPOINTS / f"checkpoint-{step}"
        targets.append((TUNED, name, step, adapter_sha256(folder)))

    for name, adapter, step, sha in targets:
        role = "score-untuned" if adapter is None else "score-checkpoint"
        settings = expected_settings(local_cfg, budget.revision, sha)
        state, existing, summary = finder(ctx, split=split, limit=limit, settings=settings)
        if state == COMPLETE:
            entry = _eval_entry(summary, role, adapter, split)
            logger.info("%s on %s is already scored: run %s", adapter or UNTUNED, split, existing)
        else:
            resume = existing if state == INCOMPLETE else None
            evaluate(name, split, role, adapter=adapter, limit=limit, resume=resume)
            entry = runs.pop()
        entry.update(step=step, adapter_sha256=sha, reused=state == COMPLETE, training_run=run)
        runs.append(entry)


def result_path(job_name: str, root: Path | str = REPO_ROOT) -> Path:
    return Path(root) / JOBS_DIR / job_name / RESULT_NAME


def write_result(result: dict[str, Any], root: Path | str = REPO_ROOT) -> Path:
    """Write ``job_result.json`` for the job. Ids, counts, and measurements only."""
    plain = json.loads(json.dumps(result, sort_keys=True, allow_nan=False, default=str))
    check_no_secrets(plain, RESULT_NAME)
    path = result_path(result["job_name"], root)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(plain, indent=2, ensure_ascii=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
