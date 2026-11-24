"""Tests for scripts.evaluate, on an invented split in a temporary folder."""

from __future__ import annotations

import dataclasses
import re
import subprocess
import sys
from pathlib import Path

import pytest

from qlora_finetuning.evaluation.fake import gold_predictor
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking import load_manifest
from scripts import evaluate
from scripts.evaluate import PREDICTORS, PredictorEntry
from tests.evaluation.conftest import (
    ids_of,
    ledger_records,
    make_context,
    read_lines,
    records_for,
    results_lines,
    run_folders,
    write_lines,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def context(tmp_path):
    return make_context(tmp_path)


def flaky_gold(split, context):
    """The gold predictor, failing after five requests. Named by a ``module:function`` path."""
    return gold_predictor(records_for(split, context), fail_after=5)


def forbidden(split, context):
    raise AssertionError("a predictor was built")


TABLE = {
    "gold": PREDICTORS["gold"],
    "flaky": PredictorEntry("tests.scripts.test_evaluate:flaky_gold", "smoke", "none", "none"),
}


def run(context, *arguments, table=None):
    return evaluate.main(list(arguments), context=context, predictors=table or TABLE)


def run_id_in(out):
    return re.search(r"^run (\S+)", out, flags=re.MULTILINE).group(1)


def test_gold_on_quick_dev_prints_checks_and_rates(context, tmp_path, capsys):
    assert run(context, "--predictor", "gold", "--split", "quick_dev") == 0
    out = capsys.readouterr().out
    run_id = run_id_in(out)
    assert f"run {run_id}  (smoke; arm none; split quick_dev; predictor gold)" in out
    assert out.count("PASS  ") == 5 and "FAIL" not in out
    assert "PASS  R1  Complete: 3 of 3 examples, in dataset order" in out
    assert "5 passed, 0 failed" in out
    for name in ("format validity", "execution accuracy", "answer accuracy", "program match"):
        assert re.search(rf"^{name} +3 of 3  100\.00%$", out, flags=re.MULTILINE), name
    assert "execution, tight rule 3 of 3  100.00%  (diagnostic)" in out
    assert "stop reasons: end 3" in out
    assert "RESULT: complete; this run counts as a result" in out
    assert f"predictions: artifacts/runs/{run_id}/predictions.jsonl" in out
    assert out.isascii() and str(tmp_path) not in out
    assert "zebrafish" not in out and "subtract" not in out

    assert run_folders(context) == [run_id]
    assert len(results_lines(context)) == 1 and ledger_records(context)[0].examples == 3


def test_the_committed_table_offers_the_gold_predictor_as_a_smoke_run():
    entry = PREDICTORS["gold"]
    assert (entry.kind, entry.path, entry.arm) == ("smoke", "none", "none")
    assert (entry.where, entry.configs) == (None, ())
    assert sorted(PREDICTORS) == [
        "gold", "nova-lite", "nova-lite-tuned", "nova-pro", "nova-pro-tuned", "qwen3-4b",
        "qwen3-4b-merged", "qwen3-4b-merged-full", "qwen3-4b-package", "qwen3-4b-tuned",
    ]  # fmt: skip


def stub_nova(split, context):
    """Stands in for a Nova factory: the gold replies, so no client and no sign-in."""
    return gold_predictor(records_for(split, context))


def no_sign_in(split, context):
    raise ConfigError("the AWS sign-in is missing or has expired: run 'aws sso login'")


NOVA_TABLE = {
    "nova-lite": PredictorEntry(
        "tests.scripts.test_evaluate:stub_nova",
        "eval",
        "B",
        "B0-Lite",
        where="bedrock",
        configs=("frontier", "aws"),
    ),
    "signed-out": PredictorEntry(
        "tests.scripts.test_evaluate:no_sign_in", "eval", "B", "B0-Pro", where="bedrock"
    ),
}


def test_the_nova_entries_are_named_as_text_so_boto3_loads_only_when_asked_for():
    for name, arm in (
        ("nova-pro", "B0-Pro"),
        ("nova-lite", "B0-Lite"),
        ("nova-pro-tuned", "B1-Pro"),
        ("nova-lite-tuned", "B1-Lite"),
    ):
        entry = PREDICTORS[name]
        assert isinstance(entry.factory, str) and entry.factory.startswith(
            "qlora_finetuning.frontier.predictor:"
        )
        assert (entry.kind, entry.path, entry.arm) == ("eval", "B", arm)
        assert (entry.where, entry.configs) == ("bedrock", ("frontier", "aws"))


def test_an_entry_gives_the_run_its_place_and_its_extra_configs(context, capsys):
    assert run(context, "--predictor", "nova-lite", "--split", "quick_dev", table=NOVA_TABLE) == 0
    out = capsys.readouterr().out
    run_id = run_id_in(out)
    assert f"run {run_id}  (eval; arm B0-Lite; split quick_dev; predictor gold)" in out
    manifest = load_manifest(run_id, context.tracking_cfg)
    assert (manifest["where"], manifest["path"], manifest["arm"]) == ("bedrock", "B", "B0-Lite")
    assert set(manifest["configs"]) == {"evaluation", "prompt", "dsl", "token_budget",
                                        "frontier", "aws"}
    folder = context.tracking_cfg.runs_dir / run_id / "config"
    assert (folder / "frontier.yaml").is_file() and (folder / "aws.yaml").is_file()
    assert ledger_records(context)[0].where == "bedrock"


def test_where_on_the_command_line_overrides_the_entry(context, capsys):
    arguments = ("--predictor", "nova-lite", "--split", "quick_dev", "--where", "local")
    assert run(context, *arguments, table=NOVA_TABLE) == 0
    run_id = run_id_in(capsys.readouterr().out)
    assert load_manifest(run_id, context.tracking_cfg)["where"] == "local"


def test_an_entry_without_a_place_runs_locally(context, capsys):
    assert run(context, "--predictor", "gold", "--split", "quick_dev") == 0
    manifest = load_manifest(run_id_in(capsys.readouterr().out), context.tracking_cfg)
    assert manifest["where"] == "local"
    assert set(manifest["configs"]) == {"evaluation", "prompt", "dsl", "token_budget"}


def test_a_factory_without_a_sign_in_exits_2_and_leaves_no_run(context, capsys):
    assert run(context, "--predictor", "signed-out", "--split", "dev", table=NOVA_TABLE) == 2
    out = capsys.readouterr().out
    assert out.startswith("CONFIG ERROR: the AWS sign-in is missing or has expired")
    assert run_folders(context) == [] and ledger_records(context) == []


# --------------------------------------------------------------------------- the Qwen entry


class MeasuringGold:
    """The gold replies, from a predictor that also measured its run, as the Qwen one does."""

    def __init__(self, records):
        self._gold = gold_predictor(records, name="qwen3-4b", model_id="Qwen/Qwen3-4B")
        for name in ("name", "model_id", "settings", "batch_size", "usage_source"):
            setattr(self, name, getattr(self._gold, name))

    def predict(self, requests):
        return self._gold.predict(requests)

    def measurements(self):
        return {"gpu_name": "stand-in", "peak_vram_allocated_gib": 3.25, "batches": 3,
                "tokens_per_second": None, "ok": True}  # fmt: skip


def stub_qwen(split, context):
    return MeasuringGold(records_for(split, context))


def test_the_qwen_entry_is_local_arm_a0_and_loads_the_stack_only_when_asked_for():
    entry = PREDICTORS["qwen3-4b"]
    assert entry.factory == "qlora_finetuning.local_qlora.predictor:qwen3_4b"
    assert (entry.kind, entry.path, entry.arm) == ("eval", "A", "A0")
    assert (entry.where, entry.configs) == ("local", ("local_qlora",))


def test_a_qwen_run_snapshots_its_config_and_prints_its_measurements(context, capsys):
    entry = dataclasses.replace(
        PREDICTORS["qwen3-4b"], factory="tests.scripts.test_evaluate:stub_qwen"
    )
    arguments = ("--predictor", "qwen3-4b", "--split", "quick_dev")
    assert run(context, *arguments, table={"qwen3-4b": entry}) == 0
    out = capsys.readouterr().out
    run_id = run_id_in(out)
    assert f"run {run_id}  (eval; arm A0; split quick_dev; predictor qwen3-4b)" in out
    assert "measurements: batches 3, peak vram allocated gib 3.25" in out  # numbers only
    assert out.isascii() and "stand-in" not in out
    manifest = load_manifest(run_id, context.tracking_cfg)
    assert (manifest["where"], manifest["path"], manifest["arm"]) == ("local", "A", "A0")
    assert "local_qlora" in manifest["configs"]
    assert (context.tracking_cfg.runs_dir / run_id / "config" / "local_qlora.yaml").is_file()
    (record,) = ledger_records(context)
    assert (record.model_id, record.where, record.arm) == ("Qwen/Qwen3-4B", "local", "A0")


def test_without_the_weights_the_qwen_entry_exits_2_and_leaves_no_run(
    context, capsys, monkeypatch
):
    from types import SimpleNamespace

    from qlora_finetuning.local_qlora import backend

    monkeypatch.setattr(backend, "_stack", lambda: SimpleNamespace(torch=None))
    monkeypatch.setattr(
        backend, "weights_missing", lambda model_id, revision: ["model-00001-of-00003.safetensors"]
    )
    arguments = ("--predictor", "qwen3-4b", "--split", "quick_dev", "--limit", "5")
    assert run(context, *arguments, table=PREDICTORS) == 2
    out = capsys.readouterr().out
    assert out.startswith("CONFIG ERROR: the weights of Qwen/Qwen3-4B at 1cfa9a720891...")
    assert "hf download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c" in out
    assert run_folders(context) == [] and ledger_records(context) == []


def test_the_tuned_qwen_entry_is_arm_a1_and_takes_an_adapter():
    entry = PREDICTORS["qwen3-4b-tuned"]
    assert entry.factory == "qlora_finetuning.local_qlora.predictor:qwen3_4b_tuned"
    assert (entry.kind, entry.path, entry.arm, entry.where) == ("eval", "A", "A1", "local")
    assert entry.configs == ("local_qlora", "training") and entry.takes_adapter
    assert not any(e.takes_adapter for name, e in PREDICTORS.items() if name != "qwen3-4b-tuned")


def stub_tuned(split, context, adapter):
    """Stands in for the tuned factory: says which adapter it was given."""
    if not adapter:
        raise ConfigError("a tuned predictor needs an adapter: add --adapter RUN_ID")
    return gold_predictor(records_for(split, context), name=f"tuned with {adapter}")


TUNED_TABLE = {
    "gold": PREDICTORS["gold"],
    "qwen3-4b-tuned": dataclasses.replace(
        PREDICTORS["qwen3-4b-tuned"], factory="tests.scripts.test_evaluate:stub_tuned"
    ),
}


def test_the_adapter_reaches_the_factory_and_the_run_is_arm_a1(context, capsys):
    arguments = ("--predictor", "qwen3-4b-tuned", "--split", "quick_dev")
    name = "20261119T101500Z-train-qwen3-4b-fit-1a2b/checkpoint-300"
    assert run(context, *arguments, "--adapter", name, table=TUNED_TABLE) == 0
    out = capsys.readouterr().out
    assert f"(eval; arm A1; split quick_dev; predictor tuned with {name})" in out
    manifest = load_manifest(run_id_in(out), context.tracking_cfg)
    assert {"local_qlora", "training"} <= set(manifest["configs"])


def test_an_adapter_is_required_by_the_tuned_predictor_and_refused_by_any_other(context, capsys):
    arguments = ("--predictor", "qwen3-4b-tuned", "--split", "quick_dev")
    assert run(context, *arguments, table=TUNED_TABLE) == 2
    assert capsys.readouterr().out.startswith("CONFIG ERROR: a tuned predictor needs an adapter")
    arguments = ("--predictor", "gold", "--split", "quick_dev", "--adapter", "some-run")
    assert run(context, *arguments, table=TUNED_TABLE) == 2
    assert "CONFIG ERROR: --adapter is for a tuned Qwen predictor" in capsys.readouterr().out
    assert run_folders(context) == [] and ledger_records(context) == []
    with pytest.raises(SystemExit):
        run(context, "--summary", "some-run", "--adapter", "x", table=TUNED_TABLE)


def test_limit_gives_a_partial_run_and_exit_0(context, capsys):
    assert run(context, "--predictor", "gold", "--split", "dev", "--limit", "2") == 0
    out = capsys.readouterr().out
    assert "PASS  R1  Complete: 2 of 2 examples, in dataset order (the first 2 of 8" in out
    assert "PARTIAL: the first 2 of 8 examples; not a result" in out
    assert results_lines(context) == []


def test_a_predictor_failure_exits_1_and_the_run_can_be_resumed(context, capsys):
    assert run(context, "--predictor", "flaky", "--split", "dev") == 1
    out = capsys.readouterr().out
    failed_id = run_folders(context)[0]
    assert "PREDICTOR FAILED: the fake predictor fails after 5 requests" in out
    assert f"5 predictions are kept in run {failed_id}" in out
    assert (
        f"resume with: python -m scripts.evaluate --predictor flaky --split dev "
        f"--resume {failed_id}" in out
    )
    assert "PASS" not in out and results_lines(context) == []

    # The same predictor, no longer failing, continues the run.
    assert run(context, "--predictor", "gold", "--split", "dev", "--resume", failed_id) == 0
    out = capsys.readouterr().out
    assert "PASS  R4  Usage adds up: the ledger records of 2 run(s) cover 8 predictions" in out
    assert "RESULT: complete" in out
    resumed = run_id_in(out)
    assert [line["run_id"] for line in read_lines(resumed, context)] == (
        [failed_id] * 5 + [resumed] * 3
    )
    assert [line["example_id"] for line in read_lines(resumed, context)] == ids_of("dev", context)


def test_summary_prints_a_run_again_and_predicts_nothing(context, capsys):
    run(context, "--predictor", "gold", "--split", "dev")
    first = capsys.readouterr().out
    run_id = run_id_in(first)
    nothing = {"gold": PredictorEntry(forbidden, "smoke", "none", "none")}
    assert run(context, "--summary", run_id, table=nothing) == 0
    assert capsys.readouterr().out == first
    assert run_folders(context) == [run_id] and len(ledger_records(context)) == 1


def test_summary_of_a_run_with_a_failed_check_exits_1(context, capsys):
    run(context, "--predictor", "gold", "--split", "dev")
    run_id = run_id_in(capsys.readouterr().out)
    write_lines(run_id, context, read_lines(run_id, context)[:6])
    assert run(context, "--summary", run_id) == 1
    out = capsys.readouterr().out
    assert "FAIL  R1  Complete: 6 of 8 examples have a prediction" in out
    assert "NOT A RESULT: a check failed or the run is unfinished" in out
    assert re.search(r"^execution accuracy +6 of 6  100\.00%$", out, flags=re.MULTILINE)


def test_config_problems_exit_2(context, capsys):
    assert run(context, "--summary", "20261116T000000Z-eval-nothing-0000") == 2
    assert capsys.readouterr().out.startswith("CONFIG ERROR: run 20261116T000000Z")

    assert run(context, "--predictor", "nope", "--split", "dev") == 2
    assert "UNKNOWN PREDICTOR: nope. Known: flaky, gold" in capsys.readouterr().out

    assert run(context, "--predictor", "gold", "--split", "validation") == 2
    assert capsys.readouterr().out.startswith("CONFIG ERROR: ")
    assert run_folders(context) == []


def test_the_locked_test_split_exits_2_and_predicts_nothing(context, capsys):
    assert run(context, "--predictor", "gold", "--split", "test") == 2
    out = capsys.readouterr().out
    assert out.startswith("LOCKED: the test split is locked")
    assert "req_005 v5.01" in out
    assert run_folders(context) == [] and ledger_records(context) == []


def test_a_refused_resume_exits_2(context, capsys):
    run(context, "--predictor", "flaky", "--split", "dev")
    failed_id = run_folders(context)[0]
    capsys.readouterr()
    arguments = ("--predictor", "gold", "--split", "quick_dev", "--resume", failed_id)
    assert run(context, *arguments) == 2
    out = capsys.readouterr().out
    assert out.startswith(f"REFUSED: run {failed_id} cannot be continued")
    assert "another split" in out and run_folders(context) == [failed_id]


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("--predictor", "gold"),
        ("--split", "dev"),
        ("--summary", "x", "--split", "dev"),
        ("--summary", "x", "--limit", "3"),
        ("--predictor", "gold", "--split", "dev", "--scope", "final"),
        ("--predictor", "gold", "--split", "dev", "--limit", "many"),
    ],
)
def test_arguments_that_do_not_go_together(context, arguments):
    with pytest.raises(SystemExit) as caught:
        run(context, *arguments)
    assert caught.value.code == 2
    assert run_folders(context) == []


def test_runs_as_a_module():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.evaluate", "--summary", "20261116T000000Z-eval-none-0000"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "CONFIG ERROR: run 20261116T000000Z-eval-none-0000 has no evaluation.json" in (
        result.stdout
    )
    assert "Traceback" not in result.stderr


def test_running_by_file_path_gives_a_hint_not_a_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "evaluate.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "python -m scripts.evaluate" in result.stderr
    assert "Traceback" not in result.stderr
