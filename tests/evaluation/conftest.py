"""Shared pieces for the runner tests: an invented split and a context under ``tmp_path``.

The invented split has 60 examples: fit 30, dev 8, quick-dev 3 (part of dev), test 22. Every
gold reply is ``PROGRAM: subtract(920, 95)`` with ``ANSWER: 825``. Nothing is written to the
repository: the split, the run folders, the ledger, and the results file are all temporary.
"""

from __future__ import annotations

import json

import pytest

from qlora_finetuning.data import build_record, load_prompt, load_split
from qlora_finetuning.data.gold_check import build_flags, write_flags
from qlora_finetuning.evaluation.metrics import load_evaluation_config
from qlora_finetuning.evaluation.predictions import PREDICTIONS_NAME
from qlora_finetuning.evaluation.runner import RunnerContext, run_evaluation
from qlora_finetuning.tracking import GitInfo, Ledger, load_tracking_config
from tests.data.synthetic import build_frozen_split

GIT = GitInfo(commit="0123456789abcdef0123456789abcdef01234567", branch="main", dirty=False)
GOLD_REPLY = "PROGRAM: subtract(920, 95)\nANSWER: 825"
# The config files of the real repository; the invented split's own two are left out.
CONFIGS = ("evaluation", "prompt", "dsl", "token_budget")


def make_context(root, rows=None) -> RunnerContext:
    """A frozen invented split and everything the runner reads, all under ``root``."""
    split_cfg, data_cfg, manifest = build_frozen_split(root, rows=rows)
    config = load_evaluation_config(root=root)
    write_flags(build_flags({"flagged": [], "settings": {}}), config)  # no example is flagged
    context = RunnerContext(
        config=config,
        split_cfg=split_cfg,
        data_cfg=data_cfg,
        prompt=load_prompt(root=root),
        tracking_cfg=load_tracking_config(root=root),
        max_new_tokens=192,
        git=GIT,
        environment={},
        configs=CONFIGS,
    ).resolved()
    context.manifest = manifest  # for tests that need the split's hash or an unlock record
    return context


@pytest.fixture
def context(tmp_path) -> RunnerContext:
    return make_context(tmp_path)


def records_for(split: str, context: RunnerContext) -> list:
    examples = load_split(split, context.split_cfg, context.data_cfg)
    return [build_record(example, context.prompt) for example in examples]


def ids_of(split: str, context: RunnerContext) -> list[str]:
    return [record.example_id for record in records_for(split, context)]


def evaluate(predictor, split: str, context: RunnerContext, **options) -> dict:
    """``run_evaluation`` with the labels a fake run carries."""
    settings = {
        "label": "fake run",
        "path": "none",
        "arm": "none",
        "where": "local",
        "scope": "dev",
        "kind": "smoke",
    }
    settings.update(options)
    return run_evaluation(predictor, split, context=context, **settings)


def predictions_path(run_id: str, context: RunnerContext):
    return context.tracking_cfg.runs_dir / run_id / PREDICTIONS_NAME


def read_lines(run_id: str, context: RunnerContext) -> list[dict]:
    path = predictions_path(run_id, context)
    if not path.is_file():  # a run that failed before its first batch wrote nothing
        return []
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def write_lines(run_id: str, context: RunnerContext, lines: list[dict]) -> None:
    text = "".join(json.dumps(line) + "\n" for line in lines)
    predictions_path(run_id, context).write_text(text, encoding="utf-8", newline="\n")


def ledger_records(context: RunnerContext) -> list:
    return Ledger(context.tracking_cfg).read()


def run_folders(context: RunnerContext) -> list[str]:
    folder = context.tracking_cfg.runs_dir
    return sorted(path.name for path in folder.iterdir()) if folder.is_dir() else []


def results_lines(context: RunnerContext) -> list[dict]:
    path = context.config.results_file
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
