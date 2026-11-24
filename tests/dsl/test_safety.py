"""Safety of the DSL package (req_002 v2.01): hostile replies and a scan of the source."""

from __future__ import annotations

import ast
import builtins
import time
from pathlib import Path

import pytest

import qlora_finetuning.dsl
from qlora_finetuning.dsl import CODES, run_reply

PACKAGE = Path(qlora_finetuning.dsl.__file__).resolve().parent
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "literal_eval"}
# What a module of the package may import at module level: the standard library it uses,
# its own modules, and the settings loader. Never the data package or anything heavier.
ALLOWED_IMPORTS = {
    "__future__",
    "collections.abc",
    "dataclasses",
    "functools",
    "math",
    "pathlib",
    "typing",
    "qlora_finetuning.settings",
}


def _reply(program: str) -> str:
    return f"PROGRAM: {program}\nANSWER: 0"


HOSTILE_PROGRAMS = [
    "__import__('os').system('echo owned')",
    "exec(\"import builtins; builtins.DSL_OWNED = 1\")",
    "eval('setattr(__import__(\"builtins\"), \"DSL_OWNED\", 1)')",
    "add(__import__('builtins').__dict__.__setitem__('DSL_OWNED', 1), 2)",
    "add.__class__.__mro__[1].__subclasses__()",
    "().__class__.__bases__[0]",
    "add(1, 2); import os",
    "add(1, open('dsl_owned.txt', 'w'))",
    "open('dsl_owned.txt', 'w').write('x')",
    "add('1', \"2\")",
    "add(1, 2) if True else 0",
    "lambda: 1",
    "add(1, 2) # comment",
    "add(1, {0})",
    "add(1, [2])",
    "add(1, 2)\x00",
    "add(١, ٢)",
    "add(１, 2)",
    "table_sum(__import__('os'), none)",
    "table_sum(revenue, __class__)",
    "${jndi:ldap://x/a}",
    "{{7*7}}",
    "' OR 1=1 --",
]


@pytest.mark.parametrize("program", HOSTILE_PROGRAMS)
def test_hostile_text_is_rejected_with_a_code_and_does_nothing(program, table, tmp_path,
                                                             monkeypatch):  # fmt: skip
    monkeypatch.chdir(tmp_path)
    outcome = run_reply(_reply(program), table)
    assert not outcome.ok
    assert outcome.error_code in CODES
    assert outcome.value is None
    assert not hasattr(builtins, "DSL_OWNED")
    assert list(tmp_path.iterdir()) == []


def test_a_line_of_code_after_the_program_is_ignored_text(table, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outcome = run_reply(_reply("add(1, 2)\nopen('dsl_owned.txt', 'w').write('x')"), table)
    assert outcome.value == 3.0
    assert outcome.deviations == ("text_between",)
    assert list(tmp_path.iterdir()) == []


def test_hostile_text_in_the_other_fields_changes_nothing(table):
    text = "__import__('os')\nPROGRAM: add(1, 2)\nANSWER: __import__('os').getcwd()\nexec('1')"
    outcome = run_reply(text, table)
    assert outcome.value == 3.0
    assert outcome.answer_text == "__import__('os').getcwd()"  # kept as text, never run


def test_a_huge_reply_is_refused_at_once(table):
    text = "PROGRAM: " + "add(1, 2), " * 10_000 + "add(1, 2)\nANSWER: 3"
    assert len(text) > 100_000
    started = time.perf_counter()
    outcome = run_reply(text, table)
    assert outcome.error_code == "reply_too_long"
    assert time.perf_counter() - started < 1.0


@pytest.mark.parametrize(
    "program",
    [
        "(" * 3900,
        ")" * 3900,
        "add(" * 950,
        "table_sum(" + "(" * 1900 + ")" * 1900 + ", none)",
        "table_sum(" + "a," * 1900 + " none)",
        "add(1, " + "9" * 3900 + ")",
        "add(1, #" + "9" * 3900 + ")",
        "add(1, 2)" + ", add(#0, 1)" * 300,
        "," * 3900,
        "#" * 3900,
    ],
)
def test_awkward_shapes_at_the_length_limit_are_quick(program, table):
    started = time.perf_counter()
    outcome = run_reply(_reply(program), table)
    assert time.perf_counter() - started < 1.0
    assert not outcome.ok and outcome.error_code in CODES


# --------------------------------------------------------------------------- source scan


def _modules() -> list[Path]:
    return sorted(PACKAGE.glob("*.py"))


def _called_names(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def _module_level_imports(tree: ast.Module) -> set[str]:
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.add(node.module or "")
    return found


def test_the_scan_sees_the_whole_package():
    names = {path.name for path in _modules()}
    assert {"reply.py", "program.py", "executor.py", "outcome.py", "language.py"} <= names


@pytest.mark.parametrize("path", _modules(), ids=lambda path: path.name)
def test_no_module_evaluates_text_as_code(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert _called_names(tree) & FORBIDDEN_CALLS == set()


@pytest.mark.parametrize("path", _modules(), ids=lambda path: path.name)
def test_module_level_imports_stay_small(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for name in _module_level_imports(tree):
        assert name in ALLOWED_IMPORTS or name.startswith("qlora_finetuning.dsl"), name
        assert not name.startswith("qlora_finetuning.data")


def test_the_scan_catches_what_it_is_meant_to_catch():
    bad = ast.parse(
        "import ast\nfrom qlora_finetuning.data import load_split\n"
        "def run(text):\n    return eval(text) or ast.literal_eval(text)\n"
    )
    assert _called_names(bad) & FORBIDDEN_CALLS == {"eval", "literal_eval"}
    assert _module_level_imports(bad) == {"ast", "qlora_finetuning.data"}
    inside = ast.parse("def late():\n    from qlora_finetuning.data import load_prompt\n")
    assert _module_level_imports(inside) == set()
