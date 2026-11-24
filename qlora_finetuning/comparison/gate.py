"""The gate every test run passes (req_005 v5.01).

Until the unlock, the lock protects the test split: ``load_split("test")`` raises. After the
unlock the lock protects nothing, and this gate takes over. A test run is allowed only when
all of this holds:

- the split is unlocked by a record that names the freeze record on disk;
- the arm is one of the six, and its predictor's name, model id, and settings are the
  frozen ones;
- the shared files (prompt, split, token budget, DSL, evaluation, gold flags) are the
  frozen ones, and so are the prompt id, the split manifest, and the reply limit;
- the run has scope ``measured`` and no ``--limit``;
- a Qwen arm runs inside a SageMaker job, a Nova arm against Bedrock;
- the arm has no complete test result yet. An unfinished run is continued, never restarted.

The ``gold`` predictor is not an arm. It may run on test after the unlock: it gives each
example its own gold reply, so it says nothing about any model.

Usage::

    from qlora_finetuning.comparison.gate import TestRunRefused, check_test_request, check_test_run

    check_test_request(arm="B0-Lite", name="nova-lite", where="bedrock", scope="measured",
                       limit=None, resume=None, ctx=context)      # before a model is loaded
    check_test_run(described, arm="B0-Lite", where="bedrock", scope="measured",
                   limit=None, resume=None, ctx=context)          # with the predictor built

Both raise ``TestLockedError`` while the split is locked and ``TestRunRefused`` with the
first reason otherwise. ``run_evaluation`` calls ``check_test_run`` itself whenever the split
is ``test``, so no entry point can go round it.

The gate is a convention enforced by code and tests, not access control. This module must
not import the evaluation runner (the runner imports it), boto3, or the training stack.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from qlora_finetuning.comparison.config import ComparisonConfig, load_comparison_config
from qlora_finetuning.data.bundle import BUNDLE_ENV
from qlora_finetuning.data.split import load_split_config, read_manifest
from qlora_finetuning.data.split_loader import TestLockedError, unlock_state
from qlora_finetuning.settings import CONFIG_DIR, ConfigError, get_str
from qlora_finetuning.tracking import list_runs
from qlora_finetuning.tracking.config import check_no_secrets
from qlora_finetuning.tracking.jsonl import read_lines

SCHEMA = 1
TEST = "test"
NO_ARM = "none"
MEASURED = "measured"
EVALUATION_NAME = "evaluation.json"  # a run's own record; the runner writes it
PREDICTOR_FIELDS = ("name", "model_id", "settings")
FREEZE_SECTIONS = ("contract", "shared", "arms")
VOID_COMMAND = "python -m scripts.comparison_runs void"
FROZEN = "Nothing frozen may change after the freeze (req_005 v5.01)"


class TestRunRefused(ConfigError):
    """A test run was asked for that the freeze, the unlock, or the one-run rule forbids."""

    __test__ = False  # the name starts with "Test"; this keeps pytest from collecting it


# --------------------------------------------------------------------------- files


def canonical_json(data: Any) -> str:
    """The one text form of a record: what its SHA-256 is taken over."""
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"


def write_canonical(path: Path, data: Any) -> str:
    """Write ``data`` in its canonical form and return the SHA-256 of the bytes written."""
    check_no_secrets(data, path.name)
    text = canonical_json(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str | None:
    """The SHA-256 of a text file, or ``None`` when there is no such file.

    Taken over the content with Unix line ends. Git stores these files with LF and may check
    them out with CRLF on Windows, and a SageMaker job always has LF: the same file must
    give the same hash in all three places.
    """
    try:
        content = Path(path).read_bytes()
    except OSError:
        return None
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def _read_json(path: Path, what: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        raise ConfigError(f"{what} {path.name} is not valid JSON") from None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ConfigError(f"{what} {path.name} is damaged")
    return data


def read_freeze(cfg: ComparisonConfig) -> dict[str, Any] | None:
    """The freeze record, or ``None`` when nothing has been frozen."""
    record = _read_json(cfg.freeze_file, "the freeze record")
    if record is None:
        return None
    if not all(isinstance(record.get(name), dict) for name in FREEZE_SECTIONS):
        raise ConfigError(f"the freeze record {cfg.freeze_file.name} is damaged")
    return record


def read_unlock(split_cfg: Any) -> dict[str, Any] | None:
    """The unlock record as it is on disk, or ``None``. Says nothing about its validity."""
    try:
        record = json.loads(Path(split_cfg.unlock_file).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def read_voids(cfg: ComparisonConfig) -> list[dict[str, Any]]:
    """Every voided test run, with its arm, the runs of its chain, and the owner's reason."""
    record = _read_json(cfg.voids_file, "the list of voided runs")
    if record is None:
        return []
    voids = record.get("voids")
    if not isinstance(voids, list) or not all(
        isinstance(item, dict) and isinstance(item.get("runs"), list) for item in voids
    ):
        raise ConfigError(f"the list of voided runs {cfg.voids_file.name} is damaged")
    return voids


def voided_ids(cfg: ComparisonConfig) -> set[str]:
    return {str(run_id) for void in read_voids(cfg) for run_id in void["runs"]}


# --------------------------------------------------------------------------- what is frozen


def shared_state(
    cfg: ComparisonConfig, config_dir: Path | str | None, flags_file: Path | str
) -> dict[str, Any]:
    """The SHA-256 of each shared config file and of the gold flags, as they are now."""
    folder = Path(config_dir) if config_dir is not None else CONFIG_DIR
    return {
        "configs": {name: file_sha256(folder / f"{name}.yaml") for name in cfg.shared_configs},
        "gold_flags_sha256": file_sha256(Path(flags_file)),
    }


def contract_now(ctx: Any) -> dict[str, Any]:
    """What every arm shares besides the files: the prompt, the split, the reply limit."""
    split_cfg = ctx.split_cfg or load_split_config()
    return {
        "prompt_id": ctx.prompt.prompt_id,
        "manifest_hash": str(read_manifest(split_cfg)["content_hash"]),
        "max_new_tokens": ctx.max_new_tokens,
    }


def shared_differences(frozen: dict[str, Any], now: dict[str, Any]) -> list[str]:
    """Which shared files differ from the frozen ones, by name."""
    differs = [
        f"config/{name}.yaml"
        for name, digest in (frozen.get("configs") or {}).items()
        if (now.get("configs") or {}).get(name) != digest
    ]
    if frozen.get("gold_flags_sha256") != now.get("gold_flags_sha256"):
        differs.append("the gold flags")
    return differs


def predictor_differences(frozen: dict[str, Any], described: dict[str, Any]) -> list[str]:
    """Which of a predictor's name, model id, and settings differ from the frozen ones."""
    differs = [
        name.replace("_", " ")
        for name in ("name", "model_id")
        if described.get(name) != frozen.get(name)
    ]
    was, now = frozen.get("settings") or {}, described.get("settings") or {}
    keys = sorted(key for key in set(was) | set(now) if was.get(key) != now.get(key))
    if keys:
        differs.append(f"settings ({', '.join(keys)})")
    return differs


# --------------------------------------------------------------------------- earlier test runs


def run_evaluation_record(run_id: str, runs_dir: Path) -> dict[str, Any] | None:
    """A run's ``evaluation.json``, or ``None`` if it has none here or it cannot be read."""
    try:
        record = json.loads((Path(runs_dir) / run_id / EVALUATION_NAME).read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) and isinstance(record.get("key"), dict) else None


def run_chain(run_id: str, runs_dir: Path) -> list[str]:
    """The run and every run it continues, the newest first."""
    chain: list[str] = []
    current: str | None = run_id
    while current and current not in chain:
        chain.append(current)
        record = run_evaluation_record(current, runs_dir)
        current = record.get("continues") if record else None
    return chain


def runs_on_test(tracking_cfg: Any, arm: str | None = None) -> list[dict[str, Any]]:
    """The runs of the registry that asked test questions, the newest first.

    A run counts when its folder is here and its ``evaluation.json`` names the test split.
    ``arm`` keeps only that arm's runs.
    """
    found = []
    for entry in list_runs(tracking_cfg):
        if arm is not None and entry.get("arm") != arm:
            continue
        record = run_evaluation_record(entry["run_id"], tracking_cfg.runs_dir)
        if record is not None and record["key"].get("split") == TEST:
            found.append(dict(entry, continues=record.get("continues")))
    return found


def results_on_test(results_file: Path | str, arm: str | None = None) -> list[dict[str, Any]]:
    """The lines of the results file for complete test runs, in the order they were written."""
    return [
        line
        for line in read_lines(Path(results_file))
        if line.get("split") == TEST and (arm is None or line.get("arm") == arm)
    ]


# --------------------------------------------------------------------------- the gate


def _comparison_cfg(ctx: Any) -> ComparisonConfig:
    cfg = getattr(ctx, "comparison_cfg", None)
    return cfg if cfg is not None else load_comparison_config(root=ctx.tracking_cfg.root)


def unlock_problems(cfg: ComparisonConfig, split_cfg: Any) -> list[str]:
    """Why the unlock does not allow a test run; empty when it does.

    Raises ``TestLockedError`` while the split is locked: that is the lock's own message,
    and nothing about a freeze is worth saying before it.
    """
    manifest = read_manifest(split_cfg)
    unlocked, reason = unlock_state(split_cfg, manifest)
    if not unlocked:
        raise TestLockedError(
            f"the test split is locked: {reason}. It is read once, by req_005 v5.01, after "
            "every configuration is frozen. Use fit, dev, or quick_dev"
        )
    named = (read_unlock(split_cfg) or {}).get("freeze_sha256")
    digest = file_sha256(cfg.freeze_file)
    if digest is None:
        return [
            f"the test split is unlocked, but there is no freeze record ({cfg.freeze_file.name}): "
            "a test run needs the configuration it is frozen at"
        ]
    if not named:
        return [
            "the unlock record does not name a freeze record: it was not written by "
            "python -m scripts.comparison_freeze unlock"
        ]
    if named != digest:
        return [
            f"the unlock record names freeze record {str(named)[:12]}..., and the one on disk "
            f"is {digest[:12]}...: the freeze record changed after the unlock. {FROZEN}"
        ]
    return []


def request_problems(
    *,
    arm: str,
    name: str,
    where: str,
    scope: str,
    limit: int | None,
    resume: str | None,
    ctx: Any,
    planning: bool = False,
) -> list[str]:
    """Why a test run may not start; empty when it may. Needs no predictor.

    ``planning`` is for the launcher, which asks on this machine about a run that will be
    made inside a job: the check that the process is in a job is left out.
    """
    cfg = _comparison_cfg(ctx)
    split_cfg = ctx.split_cfg or load_split_config()
    problems = unlock_problems(cfg, split_cfg)
    if problems:
        return problems
    if limit is not None:
        return [
            "--limit is refused on the test split: a partial test run is a look at the test "
            "split that is not the measurement. Try a predictor on five questions of quick_dev"
        ]
    freeze = read_freeze(cfg) or {}

    differs = shared_differences(
        freeze["shared"], shared_state(cfg, ctx.config_dir, ctx.config.flags_file)
    )
    if differs:
        return [f"changed since the freeze: {', '.join(differs)}. {FROZEN}"]
    frozen_contract, now = freeze["contract"], contract_now(ctx)
    changed = [
        key.replace("_", " ") for key, value in now.items() if frozen_contract.get(key) != value
    ]
    if changed:
        return [f"changed since the freeze: the {', the '.join(changed)}. {FROZEN}"]

    if arm == NO_ARM and name == cfg.gold_predictor:
        return []  # the gold check: not an arm, nothing of its own to freeze
    if arm not in cfg.arms or arm not in freeze["arms"]:
        return [
            f"only the six arms ({', '.join(cfg.arms)}) and the {cfg.gold_predictor} check may "
            f"answer the test split; this run is arm {arm}, predictor {name}"
        ]
    arm_cfg = cfg.arms[arm]
    if name != arm_cfg.predictor:
        return [f"arm {arm} is the predictor {arm_cfg.predictor}, not {name}"]
    if scope != MEASURED:
        return [
            f"a test run is a measurement: add --scope {MEASURED} (it then also refuses "
            "uncommitted changes)"
        ]
    if where != arm_cfg.where:
        return [
            f"arm {arm} answers the test split in {arm_cfg.where} only, not {where}: reported "
            "numbers come from the cloud"
        ]
    if arm_cfg.path == "A" and not planning and not get_str(BUNDLE_ENV, ""):
        return [
            f"arm {arm} answers the test split inside a SageMaker job only: "
            "python -m scripts.sagemaker_job launch evaluate ... --split test (the owner's)"
        ]

    voided = voided_ids(cfg)
    done = [
        line["run_id"]
        for line in results_on_test(ctx.config.results_file, arm)
        if line.get("run_id") not in voided
    ]
    if done:
        return [
            f"arm {arm} already has its test result (run {done[0]}): the test split is read "
            "once per arm. If that run is void because of a fault in the pipeline or the "
            f"service, the owner runs {VOID_COMMAND} --run {done[0]} --reason ... --yes"
        ]
    earlier = [run for run in runs_on_test(ctx.tracking_cfg, arm) if run["run_id"] not in voided]
    if earlier and resume != earlier[0]["run_id"]:
        return [
            f"arm {arm} has an unfinished test run ({earlier[0]['run_id']}): continue it with "
            f"--resume {earlier[0]['run_id']}, do not start another"
        ]
    return []


def check_test_request(**request: Any) -> None:
    """Raise ``TestRunRefused`` unless the test run asked for may start. See the module."""
    problems = request_problems(**request)
    if problems:
        raise TestRunRefused(problems[0])


def check_test_run(
    described: dict[str, Any],
    *,
    arm: str,
    where: str,
    scope: str,
    limit: int | None,
    resume: str | None,
    ctx: Any,
) -> None:
    """The whole gate, with the predictor that would answer. Raises ``TestRunRefused``."""
    name = str(described.get("name"))
    check_test_request(
        arm=arm, name=name, where=where, scope=scope, limit=limit, resume=resume, ctx=ctx
    )
    cfg = _comparison_cfg(ctx)
    if arm == NO_ARM and name == cfg.gold_predictor:
        return
    frozen = (read_freeze(cfg) or {})["arms"][arm]["predictor"]
    differs = predictor_differences(frozen, described)
    if differs:
        raise TestRunRefused(
            f"the predictor of arm {arm} is not the frozen one: its {', '.join(differs)} "
            f"differ from the freeze record. {FROZEN}"
        )
