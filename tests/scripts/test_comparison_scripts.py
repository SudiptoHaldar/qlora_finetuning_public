"""Tests for scripts/comparison_freeze.py and scripts/comparison_runs.py (req_005 v5.01),
and for what ``scripts.evaluate`` says about a test run.

All on an invented split under ``tmp_path``: the repository's own lock is never touched.
"""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path

import pytest

from qlora_finetuning.comparison.config import load_comparison_config
from qlora_finetuning.comparison.freeze import Sources, estimate_test_passes
from qlora_finetuning.comparison.gate import read_unlock
from qlora_finetuning.data.bundle import BUNDLE_ENV, export_bundle
from qlora_finetuning.settings import REPO_ROOT, ConfigError
from scripts import comparison_freeze, comparison_runs, evaluate
from tests.comparison.helpers import (
    PREDICTORS,
    FakeGit,
    arm_predictor,
    arms_now,
    comparison_cfg,
    freeze_and_unlock,
    job_record_for,
    run_arm,
    six_dev_runs,
)
from tests.evaluation.conftest import GIT, make_context, run_folders

REASON = "the service returned errors for an hour"


def estimator(dev):
    return estimate_test_passes(
        load_comparison_config(),
        dev,
        test_examples=22,
        path_a={
            "estimate": lambda tuned: (3000, Decimal("1.20"), Decimal("2.82")),
            "spent": Decimal("8.24"),
            "cap": Decimal("100"),
        },
        path_b={
            "cost": lambda model, tokens_in, tokens_out: Decimal("0.50"),
            "spent": Decimal("66.84"),
            "cap": Decimal("200"),
        },
    )


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.delenv(BUNDLE_ENV, raising=False)
    return make_context(tmp_path)


@pytest.fixture
def sources(context):
    return Sources(arms=arms_now(), dev_runs=six_dev_runs(context), estimator=estimator, git=GIT)


def freeze(context, sources, *arguments, **stand_ins):
    return comparison_freeze.main(list(arguments), context=context, sources=sources, **stand_ins)


def runs(context, *arguments, **stand_ins):
    return comparison_runs.main(list(arguments), context=context, **stand_ins)


# --------------------------------------------------------------------------- comparison_freeze


def test_check_prints_the_checks_and_the_estimate_and_writes_nothing(context, sources, capsys):
    cfg = comparison_cfg(context)
    assert freeze(context, sources, "check") == 0
    out = capsys.readouterr().out
    assert "PASS  F1  Path A selected" in out and "8 passed, 0 failed" in out
    assert "A1       qwen3-4b-tuned   8 of 8 right on dev" in out
    assert "estimate  all six    $4.40  at most   $7.64" in out
    assert "Path B: $66.84 spent, at most $2.00 more, cap $200.00" in out
    assert "CHECK ONLY: nothing was written" in out
    assert not cfg.freeze_file.exists() and not cfg.report_dir.exists()

    sources.problems["F2"] = ["the deployment of nova-lite (attempt 1) is Failed"]
    assert freeze(context, sources, "check") == 1
    assert "FAIL  F2  Path B tuned: the deployment of nova-lite" in capsys.readouterr().out


def test_freeze_writes_the_record_and_says_how_to_unlock(context, sources, capsys):
    cfg = comparison_cfg(context)
    assert freeze(context, sources, "status") == 0
    assert "frozen:    no" in capsys.readouterr().out
    assert freeze(context, sources, "freeze") == 0
    out = capsys.readouterr().out
    digest = re.search(r"SHA-256: ([0-9a-f]{64})", out).group(1)
    assert "freeze record: splits/config_freeze.json" in out
    assert "for reading:   reports/comparison/config_freeze.md" in out
    assert f"unlock --yes --freeze {digest[:12]}" in out
    assert cfg.freeze_file.is_file() and (cfg.report_dir / "config_freeze.md").is_file()
    assert freeze(context, sources, "status") == 0
    out = capsys.readouterr().out
    assert f"frozen:    yes, SHA-256 {digest}" in out and "nothing frozen has changed" in out


def test_a_failed_check_writes_no_freeze_record(context, sources, capsys):
    del sources.dev_runs["A0"]
    assert freeze(context, sources, "freeze") == 1
    assert "NOT WRITTEN: a check failed" in capsys.readouterr().out
    assert not comparison_cfg(context).freeze_file.exists()


def test_unlock_from_the_command_line(context, sources, capsys):
    assert freeze(context, sources, "freeze") == 0
    digest = re.search(r"SHA-256: ([0-9a-f]{64})", capsys.readouterr().out).group(1)
    known = {"tracked": lambda path: True}

    assert freeze(context, sources, "unlock", **known) == 2
    assert f"unlock --yes --freeze {digest[:12]}" in capsys.readouterr().out
    assert freeze(context, sources, "unlock", "--yes", "--freeze", "0" * 12, **known) == 2
    assert "is not the freeze record on disk" in capsys.readouterr().out
    assert read_unlock(context.split_cfg) is None

    assert freeze(context, sources, "unlock", "--yes", "--freeze", digest[:12], **known) == 0
    out = capsys.readouterr().out
    assert out.startswith("UNLOCKED: the test split, on ")
    assert "written: splits/test_unlock.json" in out and "comparison_runs preflight" in out
    assert read_unlock(context.split_cfg)["freeze_sha256"] == digest

    assert freeze(context, sources, "status") == 0
    out = capsys.readouterr().out
    assert "test split: UNLOCKED" in out and "names this freeze record" in out
    assert freeze(context, sources, "freeze") == 1  # F6: never rewritten after the unlock
    assert "FAIL  F6  Still locked: an unlock record exists" in capsys.readouterr().out

    sources.arms["B0-Pro"]["predictor"]["model_id"] = "amazon.nova-pro-v2:0"
    assert freeze(context, sources, "status") == 1
    assert "CHANGED    arm B0-Pro: model id changed" in capsys.readouterr().out


# --------------------------------------------------------------------------- comparison_runs


def in_a_job(context, monkeypatch, tmp_path):
    folder = tmp_path / "bundle_test"
    export_bundle(folder, ("test",), cfg=context.split_cfg, data_cfg=context.data_cfg)
    monkeypatch.setenv(BUNDLE_ENV, str(folder))


def test_the_run_commands_follow_the_lock(context, capsys):
    assert runs(context, "preflight") == 2
    assert capsys.readouterr().out.startswith("LOCKED: the test split is locked")
    assert runs(context, "status") == 1
    assert "0 of 6 arms have their test result" in capsys.readouterr().out
    assert runs(context, "record") == 2
    assert "CONFIG ERROR: there is no freeze record" in capsys.readouterr().out
    assert run_folders(context) == []


def test_preflight_after_the_unlock(context, capsys):
    freeze_and_unlock(context)
    assert runs(context, "preflight", counter=lambda record: 2100) == 0
    out = capsys.readouterr().out
    assert "PASS  R1  Complete" in out and "gold on test: 22 of 22 right on execution" in out
    assert "22 counted, the longest 2,100; 22 over the training length of 2,048" in out
    assert "NOTE: nothing is cut at inference" in out and "PREFLIGHT PASSED" in out


def test_record_and_void_from_the_command_line(context, monkeypatch, tmp_path, capsys):
    freeze_and_unlock(context)
    in_a_job(context, monkeypatch, tmp_path)
    cfg = comparison_cfg(context)
    made = {arm: run_arm(arm, context)["run_id"] for arm in PREDICTORS if arm != "B1-Pro"}
    stand_ins = {"arms_now": arms_now(), "git": FakeGit()}

    assert runs(context, "record", job_record=job_record_for(made), **stand_ins) == 1
    out = capsys.readouterr().out
    assert "FAIL  C1  Six arms: arm B1-Pro: missing" in out and "NOT WRITTEN" in out
    assert not cfg.report_dir.exists()

    assert runs(context, "void", "--run", made["B0-Pro"], "--reason", REASON) == 2
    assert "the owner's decision, with --yes" in capsys.readouterr().out
    assert runs(context, "void", "--run", made["B0-Pro"], "--reason", REASON, "--yes") == 0
    out = capsys.readouterr().out
    assert f"VOIDED: {made['B0-Pro']}  (arm B0-Pro)" in out
    assert "commit it: git add splits/test_run_voids.json" in out

    made["B0-Pro"] = run_arm("B0-Pro", context)["run_id"]
    made["B1-Pro"] = run_arm("B1-Pro", context)["run_id"]
    assert runs(context, "status") == 0
    assert "6 of 6 arms have their test result" in capsys.readouterr().out
    assert runs(context, "record", job_record=job_record_for(made), **stand_ins) == 0
    out = capsys.readouterr().out
    assert "8 passed, 0 failed" in out and "DEVIATION  B0-Pro: 1 run(s) voided" in out
    assert "B1-Pro   execution accuracy 22 of 22 (100.00%)" in out
    assert "record: reports/comparison/six_arm_test_runs.json" in out
    assert (cfg.report_dir / "six_arm_test_runs.md").is_file()


# --------------------------------------------------------------------------- scripts.evaluate


def test_evaluate_says_refused_before_a_predictor_is_built(context, capsys):
    freeze_and_unlock(context)
    built = []

    def factory(split, runner_context):
        built.append(split)
        return arm_predictor("B0-Lite", runner_context)

    table = {
        "nova-lite": evaluate.PredictorEntry(
            factory=factory, kind="eval", path="B", arm="B0-Lite", where="bedrock"
        )
    }
    asked = ["--predictor", "nova-lite", "--split", "test"]
    for extra, message in (
        ([], "add --scope measured"),
        (["--scope", "measured", "--limit", "5"], "--limit is refused on the test split"),
        (["--scope", "measured", "--where", "local"], "in bedrock only"),
    ):
        assert evaluate.main([*asked, *extra], context=context, predictors=table) == 2
        out = capsys.readouterr().out
        assert out.startswith("REFUSED: ") and message in out and built == []
    assert run_folders(context) == []

    measured = [*asked, "--scope", "measured"]
    assert evaluate.main(measured, context=context, predictors=table) == 0
    assert "RESULT: complete" in capsys.readouterr().out and built == ["test"]
    assert evaluate.main(measured, context=context, predictors=table) == 2
    assert "REFUSED: arm B0-Lite already has its test result" in capsys.readouterr().out


# --------------------------------------------------------------------------- guards


def test_only_listed_modules_ask_for_the_test_split_by_name():
    """Code that trains, exports, selects, or counts never names the test split in a load.

    The runner loads whatever split it is given and puts ``test`` through the gate. Besides
    it, only the bundle (after the unlock) and the comparison package may load test rows.
    """
    allowed = {
        "qlora_finetuning/comparison/runs.py",  # the gold check after the unlock
        # req_005 v5.03: example text for the owner, written under artifacts/ only.
        "qlora_finetuning/comparison/report_examples.py",
        "scripts/sagemaker_job.py",  # the test bundle, after the unlock
    }
    asks = re.compile(
        r"""(load_split|load_split_from_snapshot|export_bundle|serialize_split)\(\s*"""
        r"""[^)]{0,80}?(["']test["']|\bTEST\b|TEST_SPLITS)""",
        re.S,
    )
    found = set()
    for folder in ("qlora_finetuning", "scripts"):
        for path in (REPO_ROOT / folder).rglob("*.py"):
            text = "\n".join(
                line for line in path.read_text(encoding="utf-8").splitlines()
                if not line.lstrip().startswith("#")
            )  # fmt: skip
            code = re.sub(r'""".*?"""', "", text, flags=re.S)
            if asks.search(code):
                found.add(path.relative_to(REPO_ROOT).as_posix())
    assert found <= allowed, sorted(found - allowed)


def test_the_comparison_package_imports_no_aws_library_at_module_level():
    for path in (REPO_ROOT / "qlora_finetuning" / "comparison").glob("*.py"):
        top = [
            line for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith(("import ", "from "))
        ]  # fmt: skip
        assert not any("boto" in line for line in top), path.name
        for module in ("frontier.bedrock", "frontier.staging", "frontier.tuning ", "sagemaker "):
            assert not any(module in line for line in top), (path.name, module)


def test_no_script_is_named_like_a_test():
    assert not list((REPO_ROOT / "scripts").glob("test_*.py"))


@pytest.mark.private_records
def test_the_repository_is_locked_or_unlocked_by_a_record_that_names_the_freeze():
    """The state of the real repository: still locked, or properly unlocked. Never between."""
    from qlora_finetuning.comparison.gate import unlock_problems
    from qlora_finetuning.data import TestLockedError
    from qlora_finetuning.data.split import load_split_config

    try:
        problems = unlock_problems(load_comparison_config(), load_split_config())
    except TestLockedError:
        return  # locked: as it is until the owner unlocks
    except ConfigError as error:
        pytest.fail(str(error))
    assert problems == []
    assert Path(load_comparison_config().freeze_file).is_file()
