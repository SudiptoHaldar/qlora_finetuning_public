"""What the demo package must never do, checked on its source (req_006 v6.01).

The demo answers with Qwen only, reads no dataset row, starts no run, calls no AWS
service, and neither parses nor evaluates reply text.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import qlora_finetuning.demo as demo_package

PACKAGE = Path(demo_package.__file__).parent
REPO_ROOT = PACKAGE.parents[1]
MODULES = sorted(PACKAGE.glob("*.py"))
TRAINING_STACK = {
    "torch", "transformers", "datasets", "peft", "trl", "accelerate", "bitsandbytes",
    "tokenizers", "safetensors", "huggingface_hub",
}  # fmt: skip
AWS = {"boto3", "botocore"}
FORBIDDEN_PACKAGES = (
    "qlora_finetuning.frontier",  # Nova is not asked
    "qlora_finetuning.comparison",  # nothing of the test runs or the gate
    "qlora_finetuning.evaluation.runner",  # no evaluation loop
    "qlora_finetuning.tracking.registry",  # no tracked run
    "qlora_finetuning.tracking.ledger",
    "scripts",
)
# Reading the dataset, starting a run, and evaluating text as code.
FORBIDDEN_NAMES = {
    "load_split", "serialize_split", "build_record", "load_raw" + "_examples", "export_bundle",
    "start_run", "run_evaluation", "eval", "exec", "compile", "literal_eval", "__import__",
}  # fmt: skip


def imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
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
        "__init__.py", "__main__.py", "answer.py", "cli.py", "inputs.py", "render.py",
    }  # fmt: skip
    assert (PACKAGE / "sample_input.json").is_file()


@pytest.mark.parametrize("path", MODULES, ids=lambda path: path.name)
def test_no_module_names_the_training_stack_aws_or_a_forbidden_package(path):
    found = imports(path)
    assert not {name.split(".")[0] for name in found} & (TRAINING_STACK | AWS)
    for package in FORBIDDEN_PACKAGES:
        assert not any(name == package or name.startswith(package + ".") for name in found)


@pytest.mark.parametrize("path", MODULES, ids=lambda path: path.name)
def test_no_module_reads_the_dataset_starts_a_run_or_evaluates_text(path):
    assert not names_used(path) & FORBIDDEN_NAMES


def test_only_the_answer_step_reads_a_reply_and_only_through_run_reply():
    readers = [path.name for path in MODULES if "run_reply" in names_used(path)]
    assert readers == ["answer.py"]
    for path in MODULES:  # no pattern over reply text anywhere in the package
        assert "re" not in {name.split(".")[0] for name in imports(path)}, path.name
        assert not names_used(path) & {"parse_reply", "parse_program", "execute"}, path.name


def test_only_the_metrics_compare_a_value():
    for path in MODULES:
        assert not names_used(path) & {"values_match", "numbers_match", "isclose", "round"}
    assert "answer_agrees" in names_used(PACKAGE / "answer.py")


def test_importing_the_demo_loads_no_model_library_and_no_aws_library():
    code = (
        "import sys, qlora_finetuning.demo.cli, qlora_finetuning.demo.__main__; "
        "print(sorted(m for m in ('torch', 'transformers', 'peft', 'bitsandbytes', 'datasets', "
        "'boto3', 'botocore') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_the_package_itself_imports_nothing():
    body = ast.parse((PACKAGE / "__init__.py").read_text(encoding="utf-8")).body
    assert len(body) == 1 and isinstance(body[0], ast.Expr)  # a docstring and nothing else


def test_the_module_runs_from_the_command_line_and_check_loads_nothing(tmp_path):
    """``python -m qlora_finetuning.demo --check --untuned``: the real entry point, no model."""
    result = subprocess.run(
        [sys.executable, "-m", "qlora_finetuning.demo", "--check", "--untuned"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS  input sample_input.json" in result.stdout
    assert result.stdout.strip().endswith("CHECK ONLY: no model was loaded")
