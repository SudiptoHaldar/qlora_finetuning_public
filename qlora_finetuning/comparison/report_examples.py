"""The local file of example text for the comparison report (req_005 v5.03).

The committed statistics record names a few example ids per arm and error category. This
module writes what those examples say: the question, the gold reply, and the arm's reply.
**That is dataset text, and every one of these is a test example**, so the file is written
only under the git-ignored ``artifacts/`` folder and nothing here returns text to a caller
that writes a committed file.

Usage::

    from qlora_finetuning.comparison.report_examples import write_examples

    path = write_examples(record, cfg)        # artifacts/comparison/error_examples.md

It reads test rows on this machine (the execution plan's principle 6, as amended for this
version). No model is asked and nothing is uploaded. This module imports neither boto3 nor
the training stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qlora_finetuning.comparison.report_config import ARTIFACTS, ReportConfig
from qlora_finetuning.settings import ConfigError

NOTICE = (
    "**Not for the repository.** This file holds FinQA test questions, gold programs, and "
    "model replies. It is written under `artifacts/`, which git ignores; do not copy it "
    "into a committed file."
)


def _fence(text: Any) -> list[str]:
    return ["```text", *str(text).replace("```", "'''").splitlines(), "```"]


def render_examples(record: dict[str, Any], context: Any = None) -> str:
    """The examples the record names, with their text. Holds dataset text."""
    from qlora_finetuning.data import build_record, load_split
    from qlora_finetuning.evaluation.runner import RunnerContext, run_predictions

    ctx = (context or RunnerContext()).resolved()
    wanted = {
        example_id
        for entry in record["arms"].values()
        for ids in entry["example_ids"].values()
        for example_id in ids
    }
    questions, golds = {}, {}
    for example in load_split("test", ctx.split_cfg, ctx.data_cfg):
        if example.id in wanted:
            questions[example.id] = example.question
            golds[example.id] = build_record(example, ctx.prompt).target

    lines = ["# Examples of wrong replies, by arm and category", "", NOTICE]
    for arm, entry in record["arms"].items():
        replies = {
            line.example_id: line.reply
            for line in run_predictions(entry["run_id"], ctx)
            if line.example_id in wanted
        }
        lines += ["", f"## {arm}: {entry['label']}"]
        for category, ids in entry["example_ids"].items():
            lines += ["", f"### {arm}, `{category}` ({entry['categories'][category]:,} replies)"]
            for example_id in ids:
                lines += ["", f"**{example_id}**", "", "Question:", ""]
                lines += _fence(questions.get(example_id, "(not found)"))
                lines += ["", "Gold reply:", ""] + _fence(golds.get(example_id, "(not found)"))
                lines += ["", "The arm's reply:", ""]
                lines += _fence(replies.get(example_id, "(not found)"))
    return "\n".join(lines) + "\n"


def write_examples(record: dict[str, Any], cfg: ReportConfig, context: Any = None) -> Path:
    """Write the examples file. Refused anywhere but under ``artifacts/``."""
    path = cfg.examples_file
    try:
        inside = path.resolve().relative_to((cfg.root / ARTIFACTS).resolve())
    except ValueError:
        inside = None
    if inside is None:
        raise ConfigError(
            f"the examples file holds dataset text and is written only under {ARTIFACTS}/"
        )
    if not record.get("arms"):
        raise ConfigError("the statistics record names no examples: build the report first")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_examples(record, context), encoding="utf-8", newline="\n")
    return path
