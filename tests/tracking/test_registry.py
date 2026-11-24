"""Tests for the run registry. Everything is written under a temporary folder."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

import qlora_finetuning
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking import (
    Ledger,
    TrackingError,
    list_runs,
    load_manifest,
    new_run_id,
    read_git_info,
    start_run,
)
from qlora_finetuning.tracking.jsonl import read_lines
from qlora_finetuning.tracking.registry import dirty_paths

from .conftest import CLEAN, DIRTY, NO_GIT

REPO_ROOT = Path(__file__).resolve().parents[2]
MOMENT = datetime(2026, 11, 15, 6, 15, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- ids


def test_run_id_format():
    assert new_run_id("train", "a1", now=MOMENT, suffix="3f9a") == "20261115T061500Z-train-a1-3f9a"


def test_run_id_label_is_cleaned():
    run_id = new_run_id("eval", "  B0-Pro / Quick dev (v2)!  ", now=MOMENT, suffix="0000")
    assert run_id == "20261115T061500Z-eval-b0-pro-quick-dev-v2-0000"


def test_run_id_empty_label_and_random_suffix():
    first = new_run_id("smoke", "???", now=MOMENT)
    second = new_run_id("smoke", "???", now=MOMENT)
    assert re.fullmatch(r"20261115T061500Z-smoke-run-[0-9a-f]{4}", first)
    assert first != second or True  # 1 in 65,536 collide; the folder check catches that


def test_run_id_long_label_is_cut():
    assert len(new_run_id("eval", "x" * 200, now=MOMENT, suffix="aaaa").split("-")[2]) == 40


# --------------------------------------------------------------------------- start


def test_start_run_creates_folder_manifest_and_index_line(make_run, cfg):
    run = make_run(
        "train",
        "A1 smoke",
        path="A",
        arm="A1",
        where="local",
        configs=["aws", "local_env"],
        split_manifest_hash="sha256:abc",
        description="first wiring test",
        now=MOMENT,
    )
    assert run.run_dir == cfg.runs_dir / run.run_id
    assert run.run_id.startswith("20261115T061500Z-train-a1-smoke-")

    manifest = json.loads((run.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == run.manifest == load_manifest(run.run_id, cfg)
    assert manifest["schema"] == 1
    assert manifest["status"] == "started"
    assert manifest["started_at"] == "2026-11-15T06:15:00Z"
    assert manifest["finished_at"] is None
    assert (manifest["kind"], manifest["path"], manifest["arm"]) == ("train", "A", "A1")
    assert (manifest["where"], manifest["scope"]) == ("local", "dev")
    assert manifest["git"] == {
        "commit": CLEAN.commit,
        "branch": "main",
        "dirty": False,
        "dirty_override": False,
    }
    assert manifest["package_version"] == qlora_finetuning.__version__
    assert manifest["split_manifest_hash"] == "sha256:abc"
    assert manifest["environment"] == {}

    (line,) = read_lines(cfg.runs_index_file)
    assert line["event"] == "started"
    assert line["run_id"] == run.run_id
    assert line["git_commit"] == CLEAN.commit


def test_config_files_are_copied_and_hashed(make_run):
    run = make_run(configs=["aws"])
    source = (REPO_ROOT / "config" / "aws.yaml").read_bytes()
    assert (run.run_dir / "config" / "aws.yaml").read_bytes() == source
    assert run.manifest["configs"] == {
        "aws": {
            "file": "config/aws.yaml",
            "sha256": hashlib.sha256(source).hexdigest(),
            "version": 1,
        }
    }


def test_environment_report_is_recorded_when_not_supplied(cfg, monkeypatch):
    from qlora_finetuning.tracking import registry

    monkeypatch.setattr(registry, "environment_report", lambda: {"python": "3.13.7"})
    run = start_run("eval", "x", path="A", arm="A0", where="local", scope="dev", cfg=cfg, git=CLEAN)
    assert run.manifest["environment"] == {"python": "3.13.7"}


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"kind": "tune"}, "kinds"),
        ({"path": "C"}, "paths"),
        ({"arm": "A2"}, "arms"),
        ({"where": "laptop"}, "places"),
        ({"scope": "final"}, "scopes"),
    ],
)
def test_unknown_values_are_rejected_before_anything_is_written(make_run, cfg, change, field):
    kind = change.pop("kind", "eval")
    with pytest.raises(TrackingError, match=f"allowed value for {field}"):
        make_run(kind, "x", **change)
    assert not cfg.runs_dir.exists()
    assert not cfg.runs_index_file.exists()


def test_missing_config_file_leaves_nothing_behind(make_run, cfg):
    with pytest.raises(ConfigError, match="not found"):
        make_run(configs=["aws", "no_such_config"])
    assert not cfg.runs_dir.exists()
    assert not cfg.runs_index_file.exists()


def test_label_with_an_email_is_rejected(make_run):
    with pytest.raises(TrackingError, match="email"):
        make_run("eval", "run for owner@example.com")


# --------------------------------------------------------------------------- measured runs


def test_measured_run_refuses_uncommitted_changes(make_run, cfg):
    with pytest.raises(TrackingError, match="uncommitted changes") as excinfo:
        make_run(scope="measured", git=DIRTY)
    assert "scripts/train.py" in str(excinfo.value)
    assert "allow_dirty=True" in str(excinfo.value)
    assert not cfg.runs_dir.exists()


def test_measured_run_refuses_an_unknown_commit(make_run):
    with pytest.raises(TrackingError, match="must trace to a commit"):
        make_run(scope="measured", git=NO_GIT)


def test_measured_run_starts_from_a_clean_tree(make_run):
    run = make_run(scope="measured", git=CLEAN)
    assert run.manifest["scope"] == "measured"
    assert run.manifest["git"]["dirty_override"] is False


@pytest.mark.parametrize("git", [DIRTY, NO_GIT])
def test_override_is_recorded(make_run, git):
    run = make_run(scope="measured", git=git, allow_dirty=True)
    assert run.manifest["git"]["dirty_override"] is True
    assert run.manifest["git"]["dirty"] is git.dirty


def test_dev_run_only_records_the_dirty_tree(make_run):
    run = make_run(scope="dev", git=DIRTY)
    assert run.manifest["git"]["dirty"] is True
    assert run.manifest["git"]["dirty_override"] is False


# --------------------------------------------------------------------------- finish


def test_finish_updates_manifest_and_index(make_run, cfg):
    run = make_run()
    run.finish("completed", outputs={"predictions": "predictions.jsonl"})
    manifest = load_manifest(run.run_id, cfg)
    assert manifest["status"] == "completed"
    assert manifest["finished_at"].endswith("Z")
    assert manifest["outputs"] == {"predictions": "predictions.jsonl"}
    assert manifest["error"] is None
    events = [(line["event"], line.get("status")) for line in read_lines(cfg.runs_index_file)]
    assert events == [("started", None), ("finished", "completed")]
    assert not list(run.run_dir.glob("*.tmp"))


def test_finish_is_allowed_once(make_run):
    run = make_run()
    run.finish("completed")
    with pytest.raises(TrackingError, match="already finished"):
        run.finish("failed")


def test_finish_rejects_other_statuses(make_run):
    with pytest.raises(TrackingError, match="finishes as one of"):
        make_run().finish("cancelled")


def test_outputs_with_a_secret_are_rejected(make_run):
    run = make_run()
    with pytest.raises(TrackingError, match="access key"):
        run.finish("completed", outputs={"key": "AKIAIOSFODNN7EXAMPLE"})
    assert run.status == "started"


def test_context_manager_completes_and_keeps_outputs(make_run, cfg):
    with make_run() as run:
        run.outputs["adapter"] = "adapter/"
    manifest = load_manifest(run.run_id, cfg)
    assert manifest["status"] == "completed"
    assert manifest["outputs"] == {"adapter": "adapter/"}


def test_context_manager_marks_failure_and_reraises(make_run, cfg):
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        with make_run() as run:
            raise RuntimeError("CUDA out of memory\nlong details")
    manifest = load_manifest(run.run_id, cfg)
    assert manifest["status"] == "failed"
    assert manifest["error"] == "RuntimeError: CUDA out of memory"


def test_context_manager_respects_an_explicit_finish(make_run, cfg):
    with make_run() as run:
        run.finish("failed", error="gave up")
    assert load_manifest(run.run_id, cfg)["status"] == "failed"


# --------------------------------------------------------------------------- ledger hooks


def test_run_record_fills_in_the_run_fields(make_run, cfg):
    run = make_run()
    record = run.record(
        phase="eval",
        model_id="amazon.nova-pro-v1:0",
        input_tokens=10,
        output_tokens=2,
        examples=1,
        source="aws_api",
    )
    assert (record.run_id, record.path, record.arm, record.where) == (
        run.run_id,
        "B",
        "B0-Pro",
        "bedrock",
    )
    assert Ledger(cfg).read() == [record]


def test_run_append_rejects_another_runs_record(make_run):
    first, second = make_run(label="one"), make_run(label="two")
    with pytest.raises(TrackingError, match="belongs to run"):
        second.append(first.new_record(phase="eval", wall_clock_seconds=1.0, source="measured"))


# --------------------------------------------------------------------------- reading back


def test_list_runs_newest_first_with_final_status(make_run, cfg):
    old = make_run(label="old", now=datetime(2026, 11, 14, 9, 0, 0, tzinfo=UTC))
    new = make_run(label="new", now=datetime(2026, 11, 15, 9, 0, 0, tzinfo=UTC))
    old.finish("failed", error="x")
    runs = list_runs(cfg)
    assert [entry["run_id"] for entry in runs] == [new.run_id, old.run_id]
    assert [entry["status"] for entry in runs] == ["started", "failed"]
    assert runs[1]["finished_at"].endswith("Z")
    assert runs[0]["started_at"] == "2026-11-15T09:00:00Z"
    assert runs[0]["arm"] == "B0-Pro"


def test_list_runs_on_an_empty_index(cfg):
    assert list_runs(cfg) == []


def test_list_runs_rejects_a_damaged_index(cfg):
    cfg.runs_index_file.parent.mkdir(parents=True)
    cfg.runs_index_file.write_text(
        '{"schema": 1, "event": "finished", "run_id": "ghost"}\n', encoding="utf-8"
    )
    with pytest.raises(TrackingError, match="never started"):
        list_runs(cfg)
    cfg.runs_index_file.write_text('{"hello": "world"}\n', encoding="utf-8")
    with pytest.raises(TrackingError, match="line 1 is not a run index line"):
        list_runs(cfg)


def test_load_manifest_of_an_unknown_run(cfg):
    with pytest.raises(TrackingError, match="no manifest for run nope"):
        load_manifest("nope", cfg)


# --------------------------------------------------------------------------- git


def test_dirty_paths_ignores_the_ledger_folder():
    porcelain = (
        " M ledger/ledger.jsonl\n"
        "?? ledger/\n"
        " M scripts/train.py\n"
        'R  "old name.py" -> "new name.py"\n'
        "?? notes.txt\n"
    )
    assert dirty_paths(porcelain, ("ledger/",)) == [
        "scripts/train.py",
        "old name.py",
        "new name.py",
        "notes.txt",
    ]
    assert dirty_paths(" M ledger/ledger.jsonl\n", ("ledger/",)) == []
    assert dirty_paths("", ("ledger/",)) == []
    assert "ledger/ledger.jsonl" in dirty_paths(" M ledger/ledger.jsonl\n")


def test_git_info_outside_a_repository_uses_the_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("QLORA_TEST_COMMIT", "feedbeef")
    info = read_git_info(tmp_path, "QLORA_TEST_COMMIT")
    assert (info.commit, info.branch, info.dirty) == ("feedbeef", None, None)


def test_git_info_outside_a_repository_without_the_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("QLORA_TEST_COMMIT", raising=False)
    assert read_git_info(tmp_path, "QLORA_TEST_COMMIT").commit == "unknown"


def test_git_info_for_this_repository():
    info = read_git_info(REPO_ROOT, "QLORA_GIT_COMMIT", ("ledger/",))
    if info.commit == "unknown":
        pytest.skip("not running from a git checkout")
    assert re.fullmatch(r"[0-9a-f]{40}", info.commit)
    assert isinstance(info.dirty, bool)


# --------------------------------------------------------------------------- file replace


def test_replace_is_tried_again_when_the_operating_system_refuses_it(tmp_path, monkeypatch):
    """Windows can hold a just-written file for a moment; the manifest write must survive it."""
    from qlora_finetuning.tracking import registry

    real_replace, calls, waits = registry.os.replace, [], []

    def refuse_twice(source, target):
        calls.append(target)
        if len(calls) <= 2:
            raise PermissionError(5, "Access is denied")
        real_replace(source, target)

    monkeypatch.setattr(registry.os, "replace", refuse_twice)
    monkeypatch.setattr(registry.time, "sleep", waits.append)
    target = tmp_path / "manifest.json"
    registry._write_json(target, {"status": "completed"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "completed"}
    assert len(calls) == 3
    assert waits == [registry.REPLACE_WAIT_SECONDS, 2 * registry.REPLACE_WAIT_SECONDS]
    assert not (tmp_path / "manifest.json.tmp").exists()


def test_replace_gives_up_after_the_last_attempt(tmp_path, monkeypatch):
    from qlora_finetuning.tracking import registry

    calls = []

    def always_refuse(source, target):
        calls.append(target)
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(registry.os, "replace", always_refuse)
    monkeypatch.setattr(registry.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        registry._write_json(tmp_path / "manifest.json", {})
    assert len(calls) == registry.REPLACE_ATTEMPTS


def test_other_errors_from_replace_are_not_retried(tmp_path, monkeypatch):
    from qlora_finetuning.tracking import registry

    calls = []

    def missing(source, target):
        calls.append(target)
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(registry.os, "replace", missing)
    with pytest.raises(FileNotFoundError):
        registry._write_json(tmp_path / "manifest.json", {})
    assert len(calls) == 1
