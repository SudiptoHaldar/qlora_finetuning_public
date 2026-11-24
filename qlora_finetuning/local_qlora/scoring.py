"""Finding the evaluation runs of Path A's models in the run registry (req_003 v3.04).

Scoring ten checkpoints makes ten evaluation runs, and a scoring job that stops must not
repeat the ones it finished. A selection, later, has to find the run of each checkpoint.
Both look runs up by what was asked: the split, the prompt, the reply limit, and the
predictor's settings, which name an adapter by the hash of its weights.

Usage::

    from qlora_finetuning.local_qlora.scoring import expected_settings, find_evaluation

    settings = expected_settings(local_cfg, budget.revision, adapter_sha256)   # None: untuned
    state, run_id, summary = find_evaluation(ctx, split="dev", limit=None, settings=settings)
    state        # "complete": use it; "incomplete": resume run_id; None: nothing yet

A model's settings are known without loading it, so none of this needs a GPU. This module
imports no part of the training stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qlora_finetuning.data import manifest_hash, split_ids
from qlora_finetuning.evaluation.metrics import EvaluationError
from qlora_finetuning.evaluation.runner import EVALUATION_NAME, RunnerContext, summarize_run
from qlora_finetuning.settings import ConfigError

COMPLETE, INCOMPLETE = "complete", "incomplete"


def expected_settings(local_cfg: Any, revision: str, adapter: str | None) -> dict[str, Any]:
    """The settings a Qwen predictor will describe itself with, as plain values.

    ``adapter`` is the SHA-256 of an adapter's weights, or ``None`` for the untuned model.
    """
    return json.loads(json.dumps(local_cfg.settings(revision, adapter), sort_keys=True))


def effective_limit(split: str, limit: int | None, ctx: RunnerContext) -> int | None:
    """``limit`` as the runner records it: a limit that covers the split is no limit."""
    if limit is None:
        return None
    return None if limit >= len(split_ids(split, ctx.split_cfg)) else limit


def evaluation_runs(runs_dir: Path) -> dict[str, dict[str, Any]]:
    """Every run folder that holds an ``evaluation.json``, by run id, in run-id order."""
    found: dict[str, dict[str, Any]] = {}
    if not runs_dir.is_dir():
        return found
    for folder in sorted(runs_dir.iterdir()):
        path = folder / EVALUATION_NAME
        if not path.is_file():
            continue
        try:
            evaluation = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue  # a damaged file is not a run to reuse
        if isinstance(evaluation, dict) and isinstance(evaluation.get("key"), dict):
            found[folder.name] = evaluation
    return found


def matching_runs(
    ctx: RunnerContext, *, split: str, limit: int | None, settings: dict[str, Any]
) -> list[str]:
    """The evaluation runs that asked this model this question, oldest first.

    A run matches when its split, prompt, split manifest, reply limit, example limit, and
    the predictor's settings are all the ones given. Run ids start with a time stamp, so
    their order is the order they were made in.
    """
    limit = effective_limit(split, limit, ctx)
    current = manifest_hash(ctx.split_cfg)
    matches = []
    for run_id, evaluation in evaluation_runs(ctx.tracking_cfg.runs_dir).items():
        key = evaluation["key"]
        if (
            key.get("split") == split
            and key.get("prompt_id") == ctx.prompt.prompt_id
            and key.get("manifest_hash") == current
            and key.get("max_new_tokens") == ctx.max_new_tokens
            and (key.get("predictor") or {}).get("settings") == settings
            and evaluation.get("limit") == limit
        ):
            matches.append(run_id)
    return matches


def find_evaluation(
    ctx: RunnerContext, *, split: str, limit: int | None, settings: dict[str, Any]
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    """Whether this model has already been asked this question, and how far it got.

    Returns ``("complete", run_id, summary)`` for a run whose checks R1 to R5 pass,
    ``("incomplete", run_id, None)`` for the newest run that was cut short (resuming it is
    a new run that continues it), or ``(None, None, None)``.
    """
    newest_incomplete = None
    for run_id in reversed(matching_runs(ctx, split=split, limit=limit, settings=settings)):
        try:
            summary = summarize_run(run_id, ctx)
        except (ConfigError, EvaluationError):
            continue  # no predictions file, or one that cannot be trusted: not reusable
        if summary["complete"]:
            return COMPLETE, run_id, summary
        if newest_incomplete is None and summary["examples_predicted"] > 0:
            newest_incomplete = run_id
    if newest_incomplete is not None:
        return INCOMPLETE, newest_incomplete, None
    return None, None, None
