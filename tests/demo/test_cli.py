"""Tests for the demo's command line, with stand-ins for the model and for the adapter."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from qlora_finetuning.data.serialization import load_prompt
from qlora_finetuning.demo.cli import main, require_selected_here
from qlora_finetuning.evaluation.predictor import PredictorError
from qlora_finetuning.local_qlora.adapters import (
    ADAPTER_WEIGHTS,
    CHECKPOINTS,
    TRAINING_NAME,
    adapter_sha256,
    checkpoint_steps,
    read_json,
    write_json,
)
from qlora_finetuning.local_qlora.training import run_training
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking.config import load_tracking_config
from tests.local_qlora.training_helpers import StubEngine, StubReloader, make_training_context

from .helpers import RIGHT_REPLY, StubPredictor, tuned_stub, write_input

NAMED = ["--adapter", "some-run"]  # a named adapter: the selection record is not asked


@pytest.fixture
def context(tmp_path):
    """A context whose runs live under the temporary folder: nothing touches the repository."""
    return SimpleNamespace(
        prompt=load_prompt(), split_cfg=None, tracking_cfg=load_tracking_config(root=tmp_path)
    )


def run(argv, context, predictor=None, **stand_ins):
    """``main`` with a stub in place of both models; returns the exit code and what it saw."""
    predictor = predictor or tuned_stub()
    seen = []

    def tuned(adapter, given):
        seen.append(("tuned", adapter))
        assert given is context
        return predictor

    def untuned():
        seen.append(("untuned", None))
        return predictor

    stand_ins.setdefault("resolve", lambda adapter, given: pytest.fail("no check was asked"))
    return main(argv, context=context, tuned=tuned, untuned=untuned, **stand_ins), seen


def files_under(folder):
    return sorted(str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file())


# --------------------------------------------------------------------------- answering


def test_the_sample_is_answered_when_no_input_is_given(context, capsys):
    reply = "PROGRAM: subtract(4812, 4275), divide(#0, 4275)\nANSWER: 0.12561"
    code, seen = run(NAMED, context, tuned_stub(reply=reply))
    out = capsys.readouterr().out
    assert code == 0 and seen == [("tuned", "some-run")]
    assert "Input       sample_input.json" in out
    assert "Executed    0.1256140350877193" in out
    assert "agrees with the executed result" in out and out.isascii()


def test_several_inputs_share_one_model_and_each_is_asked_alone(tmp_path, context, capsys):
    first, second = write_input(tmp_path, "a.json"), write_input(tmp_path, "b.json")
    predictor = tuned_stub()
    code, seen = run(["--input", str(first), str(second), *NAMED], context, predictor)
    out = capsys.readouterr().out
    assert code == 0 and len(seen) == 1  # loaded once
    assert [len(batch) for batch in predictor.requests] == [1, 1]
    assert out.index("Input       a.json") < out.index("\n\nInput       b.json")


def test_json_prints_one_object_per_input(tmp_path, context, capsys):
    paths = [str(write_input(tmp_path, name)) for name in ("a.json", "b.json")]
    code, _ = run(["--input", *paths, "--json", *NAMED], context)
    lines = capsys.readouterr().out.strip().splitlines()
    assert code == 0 and len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert (first["input"], second["input"]) == ("a.json", "b.json")
    assert first["outcome"]["value"] == 0.25 and first["agrees"] is True


def test_show_input_adds_the_context_and_table_and_changes_nothing_else(context, capsys):
    reply = "PROGRAM: subtract(4812, 4275), divide(#0, 4275)\nANSWER: 0.12561"
    assert run(NAMED, context, tuned_stub(reply=reply))[0] == 0
    plain = capsys.readouterr().out.splitlines()
    assert run(["--show-input", *NAMED], context, tuned_stub(reply=reply))[0] == 0
    shown = capsys.readouterr().out.splitlines()
    assert not any(line.startswith(("Context", "Table")) for line in plain)
    assert "Context     Northfield Tools is an invented company; every figure here was made up" in (
        "\n".join(shown)
    )
    assert "            net revenue | $ 4,812 | $ 4,275" in shown
    assert shown[:2] + shown[8:] == plain  # two passages and four table rows were added

    assert run(["--show-input", "--json", *NAMED], context, tuned_stub(reply=reply))[0] == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["table"][1] == ["net revenue", "$ 4,812", "$ 4,275"]
    assert len(printed["context"]) == 2


def test_the_untuned_model_is_asked_without_an_adapter(tmp_path, context, capsys):
    code, seen = run(["--untuned"], context, StubPredictor(name="qwen3-4b"))
    assert seen == [("untuned", None)] and code == 0
    assert "Model       qwen3-4b  (untuned)" in capsys.readouterr().out


def test_the_selected_adapter_is_the_default(tmp_path, capsys):
    context = selected_world(tmp_path)
    code, seen = run(["--input", str(write_input(tmp_path))], context)
    assert code == 0 and seen == [("tuned", "selected")]


# --------------------------------------------------------------------------- exit codes


def test_a_program_that_does_not_run_gives_exit_code_1(tmp_path, context, capsys):
    good, bad = write_input(tmp_path, "good.json"), write_input(tmp_path, "bad.json")

    class Second(StubPredictor):
        def predict(self, requests):
            self.reply = RIGHT_REPLY if not self.requests else "PROGRAM: divide(1, 0)\nANSWER: 1"
            return super().predict(requests)

    code, _ = run(["--input", str(good), str(bad), *NAMED], context, Second())
    out = capsys.readouterr().out
    assert code == 1
    assert "Executed    0.25" in out and "nothing: division_by_zero" in out


def test_a_bad_input_stops_before_any_model_is_loaded(tmp_path, context, capsys):
    good, bad = write_input(tmp_path, "good.json"), write_input(tmp_path, "bad.json", table=[])
    code, seen = run(["--input", str(good), str(bad), *NAMED], context)
    out = capsys.readouterr().out
    assert code == 2 and seen == []
    assert out.strip() == (
        "CONFIG ERROR: input bad.json: table must be a list of rows, the first one the header"
    )


def test_an_adapter_that_cannot_be_used_is_one_line_and_exit_code_2(tmp_path, context, capsys):
    def refuse(adapter, given):
        raise ConfigError("run some-run has no finished adapter")

    code = main(NAMED, context=context, tuned=refuse)
    assert code == 2
    assert capsys.readouterr().out.strip() == "CONFIG ERROR: run some-run has no finished adapter"


def test_a_model_failure_is_one_line_and_exit_code_2(context, capsys):
    failing = tuned_stub(error=PredictorError("OutOfMemoryError while generating a batch of 1"))
    code, _ = run(NAMED, context, failing)
    out = capsys.readouterr().out
    assert code == 2
    assert out.strip() == "MODEL ERROR: OutOfMemoryError while generating a batch of 1"
    assert "Traceback" not in out


def test_an_adapter_and_the_untuned_model_exclude_each_other(context, capsys):
    with pytest.raises(SystemExit) as stopped:
        run(["--untuned", *NAMED], context)
    assert stopped.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_results_go_to_standard_output_and_nothing_is_written(tmp_path, context, capsys):
    path = write_input(tmp_path)
    before = files_under(tmp_path)
    code, _ = run(["--input", str(path), *NAMED], context)
    captured = capsys.readouterr()
    assert code == 0 and "Executed  " in captured.out and "Executed  " not in captured.err
    assert files_under(tmp_path) == before  # no run folder, no registry line, no ledger line
    assert not context.tracking_cfg.runs_dir.exists()


# --------------------------------------------------------------------------- --check


def selected_world(root):
    """A finished stand-in training run and a selection record that names one checkpoint."""
    training = make_training_context(root)
    result = run_training(
        smoke=False, context=training, engine=StubEngine(), reloader=StubReloader()
    )
    run_id, runs_dir = result["run_id"], training.tracking_cfg.runs_dir
    key = read_json(runs_dir / run_id / TRAINING_NAME, "the training run's record")["key"]
    step = checkpoint_steps(runs_dir / run_id / CHECKPOINTS)[0]
    folder = runs_dir / run_id / CHECKPOINTS / f"checkpoint-{step}"
    training.training_cfg.selection_record.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        training.training_cfg.selection_record,
        {
            "schema": 1,
            "key": key,
            "selected": {
                "adapter": f"{run_id}/checkpoint-{step}",
                "adapter_sha256": adapter_sha256(folder),
            },
        },
    )
    context = SimpleNamespace(
        prompt=training.prompt, split_cfg=training.split_cfg, tracking_cfg=training.tracking_cfg
    )
    context.checkpoint = folder
    context.name, context.step = f"{run_id}/checkpoint-{step}", step
    return context


def check(argv, context):
    """``--check`` with the real adapter checks and no model at all."""
    refuse = lambda *args: pytest.fail("a model was loaded")  # noqa: E731
    return main(["--check", *argv], context=context, tuned=refuse, untuned=refuse)


def test_check_passes_on_the_sample_and_the_selected_checkpoint(tmp_path, capsys):
    context = selected_world(tmp_path)
    assert check([], context) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "PASS  input sample_input.json: 4 table rows, 2 passages"
    assert out[1].startswith(f"PASS  adapter selected: {context.name}, weights ")
    assert adapter_sha256(context.checkpoint)[:12] in out[1]
    assert out[2] == "CHECK ONLY: no model was loaded"


def test_check_names_the_pull_command_when_the_checkpoint_is_not_here(tmp_path, capsys):
    context = selected_world(tmp_path)
    (context.checkpoint / ADAPTER_WEIGHTS).unlink()
    assert check([], context) == 2
    out = capsys.readouterr().out
    assert out.startswith(f"CONFIG ERROR: the selected checkpoint ({context.name}) is not on")
    assert f"python -m scripts.sagemaker_job pull --checkpoint {context.step} " in out
    with pytest.raises(ConfigError, match=f"pull --checkpoint {context.step} "):
        require_selected_here(context)


def test_check_refuses_weights_that_are_not_the_ones_selected(tmp_path, capsys):
    context = selected_world(tmp_path)
    (context.checkpoint / ADAPTER_WEIGHTS).write_bytes(b"other weights")
    assert check([], context) == 2
    assert "are not the ones the selection was made on" in capsys.readouterr().out


def test_check_without_a_selection_record_says_so(context, capsys):
    assert check([], context) == 2
    assert "no checkpoint has been selected yet" in capsys.readouterr().out


def test_check_of_a_named_adapter_and_of_the_untuned_model(tmp_path, capsys):
    context = selected_world(tmp_path)
    assert check(["--adapter", context.name], context) == 0
    assert f"PASS  adapter {context.name}: {context.name}, weights " in capsys.readouterr().out
    assert check(["--adapter", "no-such-run"], context) == 2
    assert "has no finished adapter" in capsys.readouterr().out
    assert check(["--adapter", "../elsewhere"], context) == 2
    assert "is not an adapter" in capsys.readouterr().out  # a name, never a path
    assert check(["--untuned"], context) == 0
    assert "no adapter to find" in capsys.readouterr().out


def test_check_refuses_a_bad_input_file(tmp_path, context, capsys):
    bad = write_input(tmp_path, "bad.json", question=None)
    assert check(["--input", str(bad), "--untuned"], context) == 2
    assert "input bad.json: question must be a text" in capsys.readouterr().out
