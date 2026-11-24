"""The run registry: every training or evaluation run gets an id, a folder, and a manifest.

The manifest says exactly what was run: the git commit, the package version, the environment,
a copy and hash of each config file used, and the split-manifest hash. The bulky run folder
lives under git-ignored ``artifacts/runs/``; a two-line trace of each run is appended to the
committed runs index.

Usage::

    from qlora_finetuning.tracking import start_run

    with start_run("train", "a1 qwen3-4b", path="A", arm="A1", where="local", scope="dev",
                   configs=["local_env"]) as run:
        ...                                   # do the work, write files under run.run_dir
        run.record(phase="train", wall_clock_seconds=812.4, source="measured",
                   source_detail="time.perf_counter around the training loop")
        run.outputs["adapter"] = "artifacts/runs/.../adapter"

Leaving the block finishes the run as ``completed``; an exception finishes it as ``failed`` and
is re-raised. A run with scope ``measured`` (its numbers go into the report) refuses to start
from uncommitted changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import qlora_finetuning
from qlora_finetuning.environment import environment_report
from qlora_finetuning.settings import CONFIG_DIR, get_str, load_config
from qlora_finetuning.tracking.config import (
    TrackingConfig,
    TrackingError,
    check_no_secrets,
    load_tracking_config,
    timestamp,
    utc_now,
)
from qlora_finetuning.tracking.jsonl import append_line, read_lines
from qlora_finetuning.tracking.ledger import Ledger, LedgerRecord

SCHEMA = 1
UNKNOWN_COMMIT = "unknown"
FINAL_STATUSES = ("completed", "failed")
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class GitInfo:
    """The state of the repository when a run started. ``dirty`` is None when git is absent."""

    commit: str
    branch: str | None = None
    dirty: bool | None = None
    changed: tuple[str, ...] = ()


# --------------------------------------------------------------------------- git


def dirty_paths(porcelain: str, ignore_prefixes: tuple[str, ...] = ()) -> list[str]:
    """Paths with uncommitted changes, from ``git status --porcelain`` output.

    Paths under an ignored prefix are left out: appending to the committed ledger must not
    make the next run look like it started from uncommitted code.
    """
    paths: list[str] = []
    for line in porcelain.splitlines():
        if len(line) < 4:
            continue
        for part in line[3:].split(" -> "):
            path = part.strip().strip('"')
            if path and not any(path.startswith(prefix) for prefix in ignore_prefixes if prefix):
                paths.append(path)
    return paths


def _git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=30, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout


def read_git_info(root: Path | str, env_var: str, ignore_prefixes: tuple[str, ...] = ()) -> GitInfo:
    """Read commit, branch, and uncommitted paths from git.

    Where git is not available (a cloud job, a folder that is not a repository), the commit is
    read from the environment variable ``env_var``, else recorded as ``unknown``.
    """
    root = Path(root)
    inside = (_git(root, "rev-parse", "--show-toplevel") or "").strip()
    commit = (_git(root, "rev-parse", "HEAD") or "").strip() if inside else ""
    if not commit or Path(inside).resolve() != root.resolve():
        return GitInfo(commit=get_str(env_var, UNKNOWN_COMMIT))
    branch = (_git(root, "rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None
    status = _git(root, "status", "--porcelain")
    if status is None:
        return GitInfo(commit=commit, branch=branch)
    changed = tuple(dirty_paths(status, ignore_prefixes))
    return GitInfo(commit=commit, branch=branch, dirty=bool(changed), changed=changed)


# --------------------------------------------------------------------------- ids and files


def new_run_id(
    kind: str, label: str, now: datetime | None = None, suffix: str | None = None
) -> str:
    """``<UTC time>-<kind>-<label>-<4 hex>``, for example ``20261115T061500Z-train-a1-3f9a``."""
    moment = now or utc_now()
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:40].strip("-") or "run"
    return f"{moment.strftime('%Y%m%dT%H%M%SZ')}-{kind}-{slug}-{suffix or secrets.token_hex(2)}"


# On Windows another program (a virus scanner, the search indexer) can hold a file for a
# moment after it is written, and replacing it is refused with "Access is denied". Seen in
# about one test run in eight when a manifest was rewritten right after being created.
REPLACE_ATTEMPTS = 8
REPLACE_WAIT_SECONDS = 0.05


def _replace(source: Path, target: Path) -> None:
    """``os.replace``, tried again a few times if the operating system refuses it."""
    for attempt in range(1, REPLACE_ATTEMPTS + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS:
                raise
            time.sleep(REPLACE_WAIT_SECONDS * attempt)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write through a temporary file so a crash never leaves half a manifest."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8", newline="\n")
    _replace(temporary, path)


# --------------------------------------------------------------------------- runs


@dataclass
class Run:
    """A started run. Use it as a context manager, or call ``finish`` yourself."""

    run_id: str
    run_dir: Path
    manifest: dict[str, Any]
    cfg: TrackingConfig
    outputs: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return self.manifest["status"]

    def new_record(self, **fields_: Any) -> LedgerRecord:
        """Build a ledger record for this run without appending it."""
        return LedgerRecord(
            run_id=self.run_id,
            path=self.manifest["path"],
            arm=self.manifest["arm"],
            where=self.manifest["where"],
            **fields_,
        )

    def append(self, record: LedgerRecord) -> LedgerRecord:
        if record.run_id != self.run_id:
            raise TrackingError(f"the record belongs to run {record.run_id}, not {self.run_id}")
        return Ledger(self.cfg).append(record)

    def record(self, **fields_: Any) -> LedgerRecord:
        """Build and append a ledger record for this run."""
        return self.append(self.new_record(**fields_))

    def finish(
        self, status: str, outputs: dict[str, Any] | None = None, error: str | None = None
    ) -> None:
        """Mark the run ``completed`` or ``failed``. Allowed once."""
        if self.status in FINAL_STATUSES:
            raise TrackingError(f"run {self.run_id} is already finished ({self.status})")
        if status not in FINAL_STATUSES:
            raise TrackingError(f"a run finishes as one of: {', '.join(FINAL_STATUSES)}")
        self.outputs.update(outputs or {})
        check_no_secrets(self.outputs, "the run outputs")
        finished_at = timestamp()
        self.manifest.update(
            status=status, finished_at=finished_at, outputs=dict(self.outputs), error=error
        )
        _write_json(self.run_dir / MANIFEST_NAME, self.manifest)
        append_line(
            self.cfg.runs_index_file,
            {
                "schema": SCHEMA,
                "event": "finished",
                "run_id": self.run_id,
                "at": finished_at,
                "status": status,
            },
        )

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self.status not in FINAL_STATUSES:
            if exc is None:
                self.finish("completed")
            else:
                message = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
                self.finish("failed", error=f"{type(exc).__name__}: {message}".rstrip(": "))
        return False


def start_run(
    kind: str,
    label: str,
    *,
    path: str,
    arm: str,
    where: str,
    scope: str,
    configs: tuple[str, ...] | list[str] = (),
    split_manifest_hash: str | None = None,
    allow_dirty: bool = False,
    description: str = "",
    cfg: TrackingConfig | None = None,
    config_dir: Path | str | None = None,
    git: GitInfo | None = None,
    environment: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> Run:
    """Create a run folder and manifest, add the run to the index, and return the ``Run``.

    ``configs`` names the config files the run uses (``"aws"`` for ``config/aws.yaml``); each is
    copied into the run folder and hashed. Everything is checked before anything is written.
    """
    cfg = cfg or load_tracking_config()
    cfg.check("kinds", kind)
    cfg.check("paths", path)
    cfg.check("arms", arm)
    cfg.check("places", where)
    cfg.check("scopes", scope)
    check_no_secrets([label, description], "the run label or description")

    config_dir = Path(config_dir) if config_dir is not None else CONFIG_DIR
    snapshot: dict[str, dict[str, Any]] = {}
    sources: dict[str, bytes] = {}
    for name in configs:
        version = load_config(name, config_dir=config_dir)["version"]  # raises if absent or bad
        content = (config_dir / f"{name}.yaml").read_bytes()
        sources[name] = content
        snapshot[name] = {
            "file": f"config/{name}.yaml",
            "sha256": hashlib.sha256(content).hexdigest(),
            "version": version,
        }

    git = git or read_git_info(cfg.root, cfg.git_commit_env, (cfg.ledger_prefix,))
    if scope == "measured" and not allow_dirty:
        if git.commit == UNKNOWN_COMMIT:
            raise TrackingError(
                "a measured run must trace to a commit, and the commit is unknown; "
                f"set {cfg.git_commit_env} or pass allow_dirty=True"
            )
        if git.dirty:
            shown = ", ".join(git.changed[:5]) + (" ..." if len(git.changed) > 5 else "")
            raise TrackingError(
                f"a measured run cannot start from uncommitted changes ({shown}); "
                "commit them or pass allow_dirty=True"
            )

    moment = now or utc_now()
    run_id = new_run_id(kind, label, moment)
    run_dir = cfg.runs_dir / run_id
    if run_dir.exists():
        raise TrackingError(f"run folder already exists: {run_dir}")
    started_at = timestamp(moment)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "run_id": run_id,
        "kind": kind,
        "label": label,
        "description": description,
        "path": path,
        "arm": arm,
        "where": where,
        "scope": scope,
        "status": "started",
        "started_at": started_at,
        "finished_at": None,
        "git": {
            "commit": git.commit,
            "branch": git.branch,
            "dirty": git.dirty,
            "dirty_override": bool(allow_dirty and (git.dirty or git.commit == UNKNOWN_COMMIT)),
        },
        "package_version": qlora_finetuning.__version__,
        "environment": environment if environment is not None else environment_report(),
        "configs": snapshot,
        "split_manifest_hash": split_manifest_hash,
        "outputs": {},
        "error": None,
    }

    (run_dir / "config").mkdir(parents=True)
    for name, content in sources.items():
        (run_dir / "config" / f"{name}.yaml").write_bytes(content)
    _write_json(run_dir / MANIFEST_NAME, manifest)
    append_line(
        cfg.runs_index_file,
        {
            "schema": SCHEMA,
            "event": "started",
            "run_id": run_id,
            "at": started_at,
            "kind": kind,
            "label": label,
            "path": path,
            "arm": arm,
            "where": where,
            "scope": scope,
            "git_commit": git.commit,
        },
    )
    return Run(run_id=run_id, run_dir=run_dir, manifest=manifest, cfg=cfg)


def load_manifest(run_id: str, cfg: TrackingConfig | None = None) -> dict[str, Any]:
    """Read a run's manifest from its folder."""
    cfg = cfg or load_tracking_config()
    path = cfg.runs_dir / run_id / MANIFEST_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TrackingError(f"no manifest for run {run_id} (looked in {path.parent})") from exc
    except json.JSONDecodeError as exc:
        raise TrackingError(f"the manifest of run {run_id} is not valid JSON") from exc


def list_runs(cfg: TrackingConfig | None = None) -> list[dict[str, Any]]:
    """Every run in the index, newest first, with its final status.

    A run with no ``finished`` line is still ``started``: it is running, or it was killed.
    """
    cfg = cfg or load_tracking_config()
    runs: dict[str, dict[str, Any]] = {}
    name = cfg.runs_index_file.name
    for number, line in enumerate(read_lines(cfg.runs_index_file), start=1):
        run_id, event = line.get("run_id"), line.get("event")
        if line.get("schema") != SCHEMA or not run_id or event not in ("started", "finished"):
            raise TrackingError(f"{name} line {number} is not a run index line")
        if event == "started":
            entry = {key: value for key, value in line.items() if key not in ("schema", "event")}
            entry["started_at"] = entry.pop("at")
            entry.update(status="started", finished_at=None)
            runs[run_id] = entry
        elif run_id in runs:
            runs[run_id].update(status=line.get("status"), finished_at=line.get("at"))
        else:
            raise TrackingError(f"{name} line {number} finishes a run that never started")
    return sorted(runs.values(), key=lambda entry: entry["run_id"], reverse=True)
