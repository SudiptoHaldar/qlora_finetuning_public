"""The one corrective iteration of Path A: registering the change (req_003 v3.05).

If arm A1 missed a threshold on dev, one training setting may be changed and the run made
once more. "One" is only checkable if the change is written down before the second run is
paid for. ``propose`` does that: it runs five checks and writes the committed iteration
record.

=====  ====================  ==================================================================
I1     Triggered             a selection record exists for the current experiment, verdict missed
I2     Analysed              the error analysis exists for that run and checkpoint
I3     One variable          ``config/training.yaml`` differs from the first run in exactly one
                             allowed variable (``lora.alpha`` follows ``lora.r`` at twice it)
I4     The first time        no second run has been made
I5     The same experiment   prompt, split, model, reply limit, and inference settings unchanged
=====  ====================  ==================================================================

Usage::

    from qlora_finetuning.local_qlora.iteration import iteration_status, propose

    outcome = propose("T2", "accuracy peaks at step 300 and falls")   # after editing the YAML
    outcome["passed"], outcome["checks"], outcome["change"], outcome["written"]
    iteration_status()["state"]      # "nothing selected" ... "done"

The owner chooses the change; the analysis only reports a finding. Nothing here starts a
job: the second training and scoring jobs are launched by the owner, and check J10 of the
launcher refuses a second full run without this record.

From the command line: ``python -m scripts.qlora_iterate propose --finding T2 --reason "..."``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from qlora_finetuning.data import manifest_hash
from qlora_finetuning.data.token_budget import load_token_budget_config
from qlora_finetuning.evaluation.runner import EVALUATION_NAME, RunnerContext
from qlora_finetuning.local_qlora.adapters import (
    IDENTITY,
    ITERATED_SCHEMA,
    TRAINING_NAME,
    read_json,
    read_selection,
)
from qlora_finetuning.local_qlora.config import load_local_qlora_config
from qlora_finetuning.local_qlora.iteration_record import (
    ITERATION_SCHEMA,
    read_iteration,
    second_run_made,
    write_iteration,
)
from qlora_finetuning.local_qlora.job_record import read_job_record
from qlora_finetuning.local_qlora.sagemaker_config import SageMakerConfig, load_sagemaker_config
from qlora_finetuning.local_qlora.scoring import expected_settings
from qlora_finetuning.local_qlora.training_config import (
    ITERATION_VARIABLES,
    TrainingConfig,
    load_training_config,
)
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking.config import timestamp

MISSED = "missed"
OUTSIDE = "other"  # a change the table does not name: the owner's choice, recorded as such
FINDINGS = ("T1", "T2", "T3", "T4", OUTSIDE)
R, ALPHA = "lora.r", "lora.alpha"
ALPHA_PER_R = 2  # the guides fix alpha at twice r (plan 4.6)

NOTHING_SELECTED, NO_ITERATION, REGISTERED, SECOND_MADE, DONE = (
    "nothing selected",
    "no iteration",
    "registered",
    "second run made, not selected",
    "done",
)


def _check(check_id: str, name: str, problems: list[str], ok: str) -> dict[str, Any]:
    return {
        "id": check_id,
        "name": name,
        "passed": not problems,
        "detail": "; ".join(problems) or ok,
    }


def flatten(settings: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """A settings dict as dotted keys, so that two can be compared key by key."""
    flat: dict[str, Any] = {}
    for name, value in settings.items():
        key = f"{prefix}{name}"
        if isinstance(value, dict):
            flat.update(flatten(value, key + "."))
        else:
            flat[key] = value
    return flat


def _same(first: Any, second: Any) -> bool:
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (first, second))
    if numbers:
        return math.isclose(first, second, rel_tol=1e-12, abs_tol=0.0)
    return json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def changed_variables(
    first: dict[str, Any], current: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """What differs between two runs' settings: the allowed variables, and the problems.

    ``lora.alpha`` is not a variable: it follows ``lora.r`` at twice its value, so a new
    ``r`` with its alpha is one change, and alpha changed any other way is a problem.
    """
    old, new = flatten(first), flatten(current)
    differs = [key for key in sorted(set(old) | set(new)) if not _same(old.get(key), new.get(key))]
    problems = []
    if R in differs or ALPHA in differs:
        if new.get(ALPHA) != ALPHA_PER_R * new.get(R, 0):
            problems.append(
                f"{ALPHA} must stay at {ALPHA_PER_R} times {R} "
                f"(r {new.get(R)}, alpha {new.get(ALPHA)})"
            )
        elif R not in differs:
            problems.append(f"{ALPHA} changed and {R} did not")
        differs = [key for key in differs if key != ALPHA]
    variables = [key for key in differs if key in ITERATION_VARIABLES]
    problems += [
        f"{key} may not change in an iteration" for key in differs if key not in ITERATION_VARIABLES
    ]
    return variables, problems


def _first_of(selection: dict[str, Any]) -> dict[str, Any]:
    return selection["iterations"][0] if selection["schema"] == ITERATED_SCHEMA else selection


def _in_table(
    finding: str, variable: str, value: Any, changes: tuple[tuple[str, str, float], ...]
) -> bool:
    """Whether the change is the one the table names for the finding (T4 reads as T1 or T2)."""
    rows = ("T1", "T2") if finding == "T4" else (finding,)
    return any(
        row in rows and name == variable and _same(float(value), float(wanted))
        for row, name, wanted in changes
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def propose(
    finding: str,
    reason: str,
    *,
    context: RunnerContext | None = None,
    training_cfg: TrainingConfig | None = None,
    sagemaker_cfg: SageMakerConfig | None = None,
    budget: Any = None,
    local_cfg: Any = None,
    check: bool = False,
) -> dict[str, Any]:
    """Check that one change may be made, and register it. Starts nothing.

    The change itself is the edit the owner has made to ``config/training.yaml``. With
    ``check`` nothing is written. Raises ``ConfigError`` for an unknown finding or an empty
    reason.
    """
    if finding not in FINDINGS:
        raise ConfigError(
            f"unknown finding {finding!r}; use one of: {', '.join(FINDINGS)} "
            "(T5 is a fault to look for, not a change)"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise ConfigError("a change needs its reason: add --reason, in your own words")
    ctx = (context or RunnerContext()).resolved()
    root, runs_dir = Path(ctx.tracking_cfg.root), ctx.tracking_cfg.runs_dir
    training_cfg = training_cfg or load_training_config(root=root)
    sagemaker_cfg = sagemaker_cfg or load_sagemaker_config(root=root)
    budget = budget or load_token_budget_config()
    current = {
        "model_id": budget.model_id,
        "revision": budget.revision,
        "prompt_id": ctx.prompt.prompt_id,
        "manifest_hash": manifest_hash(ctx.split_cfg),
    }
    outcome: dict[str, Any] = {"checks": [], "passed": False, "change": None, "written": []}
    checks = outcome["checks"]

    # ---- I1: triggered
    selection, problems = None, []
    try:
        selection = read_selection(training_cfg.selection_record)
    except ConfigError as error:
        problems.append(str(error))
    if selection is not None:
        differs = [name for name in IDENTITY if selection["key"].get(name) != current[name]]
        if differs:
            problems.append(f"the selection is of another experiment ({', '.join(differs)} differ)")
        if _first_of(selection)["verdict"] != MISSED:
            problems.append(
                f"the verdict is {_first_of(selection)['verdict']}: an iteration answers a "
                "missed threshold, and nothing was missed"
            )
    checks.append(_check("I1", "Triggered", problems, "arm A1 is selected and missed a threshold"))
    if selection is None:
        return outcome
    first = _first_of(selection)
    first_run = first["training_run"]

    # ---- I2: analysed
    problems, analysis = [], {}
    path = training_cfg.analysis_record
    if path is None or not path.is_file():
        problems.append("there is no error analysis yet: python -m scripts.qlora_analyze")
    else:
        analysis = read_json(path, "the error analysis")
        analysed = (analysis.get("selected") or {}).get("adapter")
        if analysed != first["selected"]["adapter"]:
            problems.append(
                f"the error analysis is of {analysed}, not of the selected "
                f"{first['selected']['adapter']}: run python -m scripts.qlora_analyze again"
            )
    checks.append(
        _check("I2", "Analysed", problems, f"the analysis of {first['selected']['adapter']} exists")
    )

    # ---- I3: one variable
    problems, change = [], None
    try:
        training = read_json(runs_dir / first_run / TRAINING_NAME, "the first run's record")
    except ConfigError as error:
        training = {}
        problems.append(f"the first run's settings are not on this machine ({error})")
    if training:
        old, new = training.get("settings") or {}, training_cfg.settings(False)
        variables, problems = changed_variables(old, new)
        if not variables and not problems:
            problems.append(
                "config/training.yaml is the first run's: edit one value, then propose again"
            )
        elif len(variables) > 1:
            problems.append(f"{len(variables)} variables differ ({', '.join(variables)}): one may")
        elif len(variables) == 1 and not problems:
            name = variables[0]
            change = {"variable": name, "from": flatten(old).get(name), "to": flatten(new)[name]}
    checks.append(
        _check(
            "I3",
            "One variable",
            problems,
            f"{change['variable']} from {change['from']} to {change['to']}" if change else "",
        )
    )

    # ---- I4: the first time
    made = second_run_made(
        read_iteration(training_cfg.iteration_record), read_job_record(sagemaker_cfg.record_file)
    )
    problems = [f"the one corrective iteration has been made ({made})"] if made else []
    checks.append(_check("I4", "The first time", problems, "no second run has been made"))

    # ---- I5: the same experiment
    problems = []
    run_id = first["selected"].get("eval_run_id")
    evaluation = runs_dir / str(run_id) / EVALUATION_NAME
    if not evaluation.is_file():
        problems.append(f"the selected checkpoint's run {run_id} is not on this machine")
    else:
        key = read_json(evaluation, "the selected checkpoint's run")["key"]
        if key.get("max_new_tokens") != ctx.max_new_tokens:
            problems.append("the reply limit has changed since the first run")
        local_cfg = local_cfg or load_local_qlora_config()
        asked = dict((key.get("predictor") or {}).get("settings") or {}, adapter=None)
        if asked != expected_settings(local_cfg, budget.revision, None):
            problems.append("the inference settings have changed since the first run")
    checks.append(
        _check(
            "I5",
            "The same experiment",
            problems,
            "prompt, split, model, reply limit, and inference settings are the first run's",
        )
    )

    outcome["change"] = change
    outcome["passed"] = all(item["passed"] for item in checks)
    if outcome["passed"] and not check:
        record = {
            "schema": ITERATION_SCHEMA,
            "first_run": first_run,
            "first_selected": first["selected"]["adapter"],
            "finding": finding,
            "analysis_finding": (analysis.get("finding") or {}).get("row"),
            "in_table": _in_table(
                finding, change["variable"], change["to"], training_cfg.iteration_changes
            ),
            "variable": change["variable"],
            "from": change["from"],
            "to": change["to"],
            "reason": reason.strip(),
            "first_settings_sha256": first["key"]["settings_sha256"]
            if "key" in first
            else first.get("settings_sha256"),
            "settings_sha256": training_cfg.settings_sha256(False),
            "proposed_on": timestamp()[:10],
            "second_run": None,
            "outcome": None,
        }
        write_iteration(training_cfg.iteration_record, record)
        outcome["record"] = record
        outcome["written"] = [str(training_cfg.iteration_record)]
    return outcome


def iteration_status(
    *,
    context: RunnerContext | None = None,
    training_cfg: TrainingConfig | None = None,
    sagemaker_cfg: SageMakerConfig | None = None,
) -> dict[str, Any]:
    """Where the iteration stands, and whether the config describes the selected run.

    ``config/training.yaml`` must end as the selected run's settings: the comparison freezes
    it as the recipe of arm A1. That is checked whenever no second run is pending.
    """
    ctx = (context or RunnerContext()).resolved()
    root = Path(ctx.tracking_cfg.root)
    training_cfg = training_cfg or load_training_config(root=root)
    sagemaker_cfg = sagemaker_cfg or load_sagemaker_config(root=root)
    status: dict[str, Any] = {
        "state": NOTHING_SELECTED,
        "selection": None,
        "iteration": None,
        "checks": [],
        "passed": True,
    }
    if not training_cfg.selection_record.is_file():
        return status
    selection = read_selection(training_cfg.selection_record)
    iteration = read_iteration(training_cfg.iteration_record)
    made = second_run_made(iteration, read_job_record(sagemaker_cfg.record_file))
    if iteration is None:
        state = NO_ITERATION
    elif iteration.get("outcome"):
        state = DONE
    else:
        state = SECOND_MADE if made else REGISTERED
    status.update(state=state, selection=selection, iteration=iteration)
    if state in (NO_ITERATION, DONE):
        problems = []
        if selection["key"].get("settings_sha256") != training_cfg.settings_sha256(False):
            problems.append(
                "config/training.yaml is not the selected run's settings: "
                + (
                    "the first run's checkpoint stayed, so revert the change and commit"
                    if state == DONE
                    else "register the change (propose), or revert it"
                )
            )
        status["checks"].append(
            _check(
                "S1",
                "The config is the selected run's",
                problems,
                f"the settings of {selection['training_run']}",
            )
        )
    status["passed"] = all(item["passed"] for item in status["checks"])
    return status
