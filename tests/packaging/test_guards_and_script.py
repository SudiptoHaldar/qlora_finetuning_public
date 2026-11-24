"""What the packaging code must never do, and the command line of ``scripts.hf_package``."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import qlora_finetuning.packaging as packaging_package
from qlora_finetuning.comparison.gate import request_problems
from qlora_finetuning.evaluation.runner import RunnerContext
from qlora_finetuning.packaging.config import load_packaging_config
from qlora_finetuning.packaging.predictors import MERGED, MERGED_FULL, PACKAGE
from qlora_finetuning.settings import ConfigError
from scripts.evaluate import PREDICTORS
from scripts.hf_package import COMMANDS, main

PACKAGE_DIR = Path(packaging_package.__file__).parent
MODULES = sorted(PACKAGE_DIR.glob("*.py"))
TRAINING_STACK = {
    "torch", "transformers", "datasets", "peft", "trl", "accelerate", "bitsandbytes",
    "tokenizers", "safetensors", "huggingface_hub",
}  # fmt: skip
STACK_MODULE = "merge.py"
AWS = {"boto3", "botocore"}
# Uploading, reading a split, starting a run, and evaluating text as code.
FORBIDDEN_NAMES = {
    "HfApi", "upload_file", "upload_folder", "create_repo", "push_to_hub", "login",
    "load_split", "serialize_split", "build_record", "load_raw" + "_examples",
    "run_evaluation", "start_run", "literal_eval",
}  # fmt: skip
EVALUATORS = {"eval", "exec", "compile"}  # as bare calls; ``re.compile`` is not one


def imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def top_level_imports(path: Path) -> set[str]:
    names = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def names_used(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name.rsplit(".", 1)[-1])
    return names


def test_the_package_has_the_modules_the_guards_were_written_for():
    assert {path.name for path in MODULES} == {
        "__init__.py", "adapter_package.py", "cards.py", "config.py", "files.py", "merge.py",
        "merged_package.py", "predictors.py", "record.py", "scan.py", "sources.py", "verify.py",
        "workflow.py",
    }  # fmt: skip
    for name in ("adapter_card.md", "merged_card.md", "adapter_usage.py.txt",
                 "merged_usage.py.txt", "apache-2.0.txt"):  # fmt: skip
        assert (PACKAGE_DIR / name).is_file()


@pytest.mark.parametrize("path", MODULES, ids=lambda path: path.name)
def test_only_the_merge_imports_the_training_stack_and_never_at_module_level(path):
    assert not top_level_imports(path) & TRAINING_STACK
    if path.name != STACK_MODULE:
        assert not imports(path) & TRAINING_STACK, path.name
    assert not imports(path) & AWS


@pytest.mark.parametrize("path", MODULES, ids=lambda path: path.name)
def test_no_module_uploads_reads_a_split_starts_a_run_or_evaluates_text(path):
    assert not names_used(path) & FORBIDDEN_NAMES
    called = {
        node.func.id
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not called & EVALUATORS


def test_the_three_predictors_are_registered_as_no_arm_and_local():
    for name in (PACKAGE, MERGED, MERGED_FULL):
        entry = PREDICTORS[name]
        assert (entry.arm, entry.path, entry.where, entry.kind) == ("none", "A", "local", "eval")
        assert entry.factory.startswith("qlora_finetuning.packaging.predictors:")
        assert not entry.takes_adapter and "packaging" in entry.configs


@pytest.mark.private_records
@pytest.mark.parametrize("name", [PACKAGE, MERGED, MERGED_FULL])
def test_the_gate_refuses_a_package_on_the_test_split(name):
    """On the real repository, frozen and unlocked: only the six arms answer test."""
    ctx = RunnerContext().resolved()
    problems = request_problems(
        arm="none", name=name, where="local", scope="measured", limit=None, resume=None, ctx=ctx
    )
    assert len(problems) == 1 and "only the six arms" in problems[0] and name in problems[0]


# --------------------------------------------------------------------------- the script


class Steps:
    """Stand-ins for the workflow: each records that it was called."""

    def __init__(self, tmp_path, failing=(), error=None):
        self.root, self.failing, self.error, self.called = tmp_path, set(failing), error, []

    def _checks(self):
        return [
            {"id": f"P{n}", "name": "a check", "passed": f"P{n}" not in self.failing,
             "detail": "café detail"}
            for n in range(1, 4)
        ]  # fmt: skip

    def _state(self):
        record = {
            "checks": self._checks(),
            "adapter": {"status": "ready"},
            "merged": {"status": "withheld"},
        }
        return SimpleNamespace(record=record)

    def _call(self, name):
        self.called.append(name)
        if self.error:
            raise ConfigError(self.error)

    def build_adapter(self, cfg):
        self._call("adapter")
        return self.root / "artifacts" / "release" / "adapter"

    def build_merged(self, cfg):
        self._call("merge")
        return self.root / "artifacts" / "release" / "merged"

    def verify(self, cfg):
        self._call("verify")
        return ["PASS  card code (adapter): ran"]

    def preflight(self, cfg):
        self._call("check")
        return self._checks()

    def gather(self, cfg):
        self._call("status")
        return self._state()

    def write_cards(self, cfg):
        self._call("cards")
        return self._state(), [self.root / "reports" / "packaging" / "adapter_card.md"]


def run(tmp_path, command, **options):
    steps = Steps(tmp_path, **options)
    cfg = load_packaging_config(root=tmp_path)
    return main([command], cfg=cfg, steps=steps), steps


@pytest.mark.parametrize("command", COMMANDS)
def test_each_command_calls_its_step_and_prints_ascii(tmp_path, capsys, command):
    code, steps = run(tmp_path, command)
    out = capsys.readouterr().out
    assert code == 0 and steps.called == [command] and out.isascii()
    if command in ("adapter", "merge"):
        assert f"WROTE  artifacts/release/{'merged' if command == 'merge' else 'adapter'}/" in out
    if command in ("cards", "status"):
        assert "adapter package: ready" in out and "merged copy: withheld" in out
        assert "3 passed, 0 failed" in out and "caf\\xe9 detail" in out
    if command == "cards":
        assert "WROTE  reports/packaging/adapter_card.md" in out


@pytest.mark.parametrize("command", ["check", "cards", "status"])
def test_a_failed_check_gives_exit_code_1(tmp_path, capsys, command):
    code, _ = run(tmp_path, command, failing={"P2"})
    out = capsys.readouterr().out
    assert code == 1 and "FAIL  P2" in out and "2 passed, 1 failed" in out


@pytest.mark.parametrize("command", COMMANDS)
def test_a_config_error_is_one_line_and_exit_code_2(tmp_path, capsys, command):
    code, _ = run(tmp_path, command, error="the checkpoint is not here")
    assert code == 2
    assert capsys.readouterr().out.strip() == "CONFIG ERROR: the checkpoint is not here"


def test_an_unknown_command_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        main(["upload"], cfg=load_packaging_config(root=tmp_path), steps=Steps(tmp_path))
