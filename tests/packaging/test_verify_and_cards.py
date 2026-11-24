"""Finding and comparing the dev runs, running a card's code, and building the cards."""

from __future__ import annotations

import ast
import sys
from types import SimpleNamespace

import pytest

from qlora_finetuning.data.serialization import format_user, load_prompt
from qlora_finetuning.data.token_budget import load_token_budget_config
from qlora_finetuning.demo.inputs import load_input, sample_path
from qlora_finetuning.evaluation.metrics import load_evaluation_config
from qlora_finetuning.evaluation.runner import run_predictions
from qlora_finetuning.local_qlora.config import load_local_qlora_config
from qlora_finetuning.local_qlora.render import render_chat
from qlora_finetuning.packaging.cards import ADAPTER, MERGED, fill, header, usage_code
from qlora_finetuning.packaging.config import load_packaging_config
from qlora_finetuning.packaging.verify import (
    assign,
    check_card_run,
    compare,
    extract_code,
    find_run,
    run_card_code,
)
from qlora_finetuning.settings import ConfigError
from tests.data.synthetic import FakeTokenizer, fake_budget

from .helpers import make_run, make_world, settings_for

NAMES = ("system", "user", "text", "reply")


# --------------------------------------------------------------------------- the dev runs


def test_a_run_is_found_by_name_settings_and_place(tmp_path):
    world = make_world(tmp_path)
    settings = settings_for(world, "R1")
    assert find_run(world.runner, "qwen3-4b-tuned", settings, "dev") is None
    cloud = make_run(world, "R1", where="sagemaker")  # the recorded run: same name and settings
    assert find_run(world.runner, "qwen3-4b-tuned", settings, "dev") is None
    first = make_run(world, "R1")
    package = make_run(world, "R2")
    found = find_run(world.runner, "qwen3-4b-tuned", settings, "dev")
    assert found is not None and found[0] == first and found[0] != cloud
    assert found[1]["result"] is True and found[1]["all"]["examples"] == 8
    assert find_run(world.runner, "qwen3-4b-package", settings, "dev")[0] == package
    assert find_run(world.runner, "qwen3-4b-tuned", settings, "quick_dev") is None
    other = dict(settings, batch_size=settings["batch_size"] + 1)
    assert find_run(world.runner, "qwen3-4b-tuned", other, "dev") is None


def test_the_newest_complete_run_wins(tmp_path):
    world = make_world(tmp_path)
    make_run(world, "R1")
    second = make_run(world, "R1", wrong=1)
    found = find_run(world.runner, "qwen3-4b-tuned", settings_for(world, "R1"), "dev")
    assert found[0] == second and found[1]["all"]["execution_accuracy"]["count"] == 7


def test_two_runs_are_compared_reply_by_reply(tmp_path):
    world = make_world(tmp_path)
    reference = run_predictions(make_run(world, "R1", wrong=1), world.runner)
    same = run_predictions(make_run(world, "R2", wrong=1), world.runner)
    assert compare(reference, same) == {
        "examples": 8, "replies_that_differ": 0, "differing_ids": [], "right_reference": 7,
        "right_other": 7, "right_to_wrong": 0, "wrong_to_right": 0,
    }  # fmt: skip
    # The first becomes right, the second and third wrong, the fourth right in other words.
    lines = list(reference)
    other = run_predictions(make_run(world, "R2", wrong=0, reworded=0), world.runner)
    mixed = [other[0], lines[0], lines[0], *other[3:]]
    mixed = [
        SimpleNamespace(example_id=ref.example_id, reply=line.reply, score=line.score)
        for ref, line in zip(reference, mixed, strict=True)
    ]
    result = compare(reference, mixed)
    assert result["replies_that_differ"] == 3 and result["right_to_wrong"] == 2
    assert result["wrong_to_right"] == 1 and result["right_other"] == 6
    assert result["differing_ids"] == [line.example_id for line in reference[:3]]
    with pytest.raises(ConfigError, match="same examples"):
        compare(reference, reference[::-1])


# --------------------------------------------------------------------------- a card's code


CODE = (
    'NAME = "account/repository"\nFLAG = True\n'
    "system, user = 'S', 'U ' + NAME\ntext = system + user\nreply = 'flag ' + str(FLAG)\n"
)


def test_code_is_taken_from_a_card_and_its_assignments_replaced():
    card = "---\na: 1\n---\n\n# T\n\n```\nnot code\n```\n\n```python\nX = 1\nprint(X)\n```\n"
    assert extract_code(card) == "X = 1\nprint(X)\n"
    with pytest.raises(ConfigError, match="no Python code block"):
        extract_code("# a card without code")
    changed = assign(CODE, {"NAME": "C:\\a folder\\x", "FLAG": False})
    assert "NAME = 'C:\\\\a folder\\\\x'" in changed and "FLAG = False" in changed
    with pytest.raises(ConfigError, match="assigns OTHER 0 times"):
        assign(CODE, {"OTHER": 1})
    with pytest.raises(ConfigError, match="assigns NAME 2 times"):
        assign(CODE + 'NAME = "again"\n', {"NAME": "x"})


def test_a_cards_code_runs_in_an_empty_directory_without_the_project(tmp_path):
    result = run_card_code(CODE, {"NAME": str(tmp_path), "FLAG": False}, timeout=60)
    assert result["user"] == "U " + str(tmp_path) and result["reply"] == "flag False"
    assert set(NAMES) <= set(result) and result["seconds"] >= 0

    probe = (
        "import os, sys\nNAME = 'x'\n"
        "system = os.environ.get('HF_HUB_OFFLINE', '')\n"
        "user = ','.join(sorted(os.listdir('.')))\n"
        "try:\n    import qlora_finetuning\n    text = 'importable'\n"
        "except ImportError:\n    text = 'not importable'\n"
        "reply = os.environ.get('PYTHONPATH', 'none')\n"
    )
    result = run_card_code(probe, {"NAME": "y"}, timeout=60, python=sys.executable)
    assert result["system"] == "1" and result["user"] == "card_code.py"
    assert result["reply"] == "none"


def test_code_that_fails_or_leaves_a_text_out_is_a_config_error():
    with pytest.raises(ConfigError, match="failed .exit code 1.*ZeroDivisionError"):
        run_card_code("NAME = 1\nsystem = user = text = reply = str(1 / 0)\n", {}, timeout=60)
    with pytest.raises(ConfigError, match="exit code"):
        run_card_code("NAME = 1\nsystem = user = text = 'x'\n", {}, timeout=60)  # no reply
    with pytest.raises(ConfigError, match="as text"):
        run_card_code("system = user = text = 'x'\nreply = 5\n", {}, timeout=60)


def card_result(root, **changes):
    prompt, sample = load_prompt(), load_input(sample_path())
    tokenizer = FakeTokenizer()
    budget = fake_budget(root, tokenizer)
    table = [list(row) for row in sample.table]
    user = format_user(sample.passages, table, sample.question, prompt)
    result = {
        "system": prompt.system,
        "user": user,
        "text": render_chat(prompt.system, user, tokenizer, budget.thinking_off_suffix),
        "reply": "PROGRAM: subtract(4812, 4275), divide(#0, 4275)\nANSWER: 0.12561",
    }
    result.update(changes)
    return check_card_run(
        result,
        sample=sample,
        prompt=prompt,
        tokenizer=tokenizer,
        budget=budget,
        evaluation_config=load_evaluation_config(),
    )


def test_a_card_run_is_right_only_with_the_projects_prompt_and_the_expected_result(tmp_path):
    problems, facts = card_result(tmp_path)
    assert problems == [] and facts["program_ran"] and facts["in_the_format"]
    assert facts["value"] == load_input(sample_path()).expected

    assert card_result(tmp_path, system="another")[0][0] == "the system text is not the project's"
    assert "serializer" in card_result(tmp_path, user="Question: x")[0][0]
    assert card_result(tmp_path, text="raw")[0] == [
        "the prompt is not the one the project's backend sends"
    ]
    wrong, facts = card_result(tmp_path, reply="PROGRAM: subtract(4812, 4275)\nANSWER: 537")
    assert wrong == ["the reply's program does not give the sample's expected result"]
    assert facts["value"] == 537.0
    broken, facts = card_result(tmp_path, reply="I think it grew.")
    assert "did not run" in broken[0] and facts["program_ran"] is False


# --------------------------------------------------------------------------- the usage code


@pytest.mark.parametrize(
    ("kind", "switches"), [(ADAPTER, ("ADAPTER",)), (MERGED, ("MODEL", "QUANTIZED"))]
)
def test_the_usage_code_is_complete_python_that_defines_the_four_texts(kind, switches):
    cfg, budget, local_cfg = (
        load_packaging_config(), load_token_budget_config(), load_local_qlora_config()
    )  # fmt: skip
    code, problems = usage_code(kind, cfg, budget, local_cfg)
    assert problems == [] and "{{" not in code
    tree = ast.parse(code)
    assigned = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert set(NAMES) <= assigned and set(switches) <= assigned
    assign(code, {name: "x" for name in switches})  # each is assigned exactly once
    imported = {
        (node.module or "").split(".")[0] if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "qlora_finetuning" not in imported  # a reader has no project to import
    assert budget.revision in code or kind == MERGED
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "enable_thinking=False" in code and not called & {"eval", "exec", "compile"}
    assert (cfg.adapter_repo if kind == ADAPTER else cfg.merged_repo) in code
    assert ("peft" in imported) is (kind == ADAPTER)


@pytest.mark.parametrize("kind", [ADAPTER, MERGED])
def test_the_usage_code_writes_the_user_text_as_the_serializer_does(kind):
    cfg, budget, local_cfg = (
        load_packaging_config(), load_token_budget_config(), load_local_qlora_config()
    )  # fmt: skip
    code, _ = usage_code(kind, cfg, budget, local_cfg)
    functions = [
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name in ("one_line", "user_text")
    ]
    assert len(functions) == 2
    scope: dict = {}
    module = ast.Module(body=[ast.parse("import re").body[0], *functions], type_ignores=[])
    exec(compile(module, "card", "exec"), scope)  # noqa: S102  (two pure functions of the card)
    prompt = load_prompt()
    labels = {
        "context_label": prompt.context_label,
        "list_marker": prompt.list_marker,
        "no_context": prompt.no_context,
        "table_label": prompt.table_label,
        "question_label": prompt.question_label,
    }
    cases = [
        (["One.", "Two\n  lines here."], [["", "2024"], ["a | b", "$ 1,150"]], "what \n is it?"),
        ([], [["name", "value"], ["x", "1"], ["y", "2"]], "how much?"),
        (["  padded  "], [["only"]], "single column?"),
    ]
    for context, rows, question in cases:
        assert scope["user_text"](labels, context, rows, question) == format_user(
            context, rows, question, prompt
        )


def test_fill_reports_what_has_no_value_and_ignores_what_is_not_named():
    text, problems = fill("a {{one}} b {{two}}", {"one": "1", "two": None, "three": "3"})
    assert text == "a 1 b {{two}}"
    assert problems == ["the template names {{two}}, which has no value"]
    assert header("---\na: 1\n---\nbody") == {"a": 1}
    assert header("no header") is None and header("---\n- a\n---\n") is None
