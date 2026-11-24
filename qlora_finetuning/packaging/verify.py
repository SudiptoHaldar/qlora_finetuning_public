"""Proving that a package loads and answers (req_006 v6.02, spec D13 to D19).

Two kinds of proof, and neither scores anything itself:

- **Dev runs.** Each package is scored on dev by the ordinary runner
  (``scripts.evaluate``). This module finds those runs again and compares their replies
  with the reference run, the selected adapter loaded by name on the same machine.
- **The card's code.** The code a model card prints is executed in a new process, in an
  empty directory, with the package folder in place of the repository name. It must build
  the project's own prompt for the invented sample, and its reply must run to the
  sample's expected result.

Usage::

    from qlora_finetuning.packaging.verify import compare, find_run, run_card_code

    found = find_run(ctx, "qwen3-4b-package", settings, "dev")       # (run_id, summary) or None
    difference = compare(run_predictions(r1, ctx), run_predictions(r2, ctx))
    result = run_card_code(code, {"ADAPTER": str(folder)}, timeout=1200)
    check_card_run(result, sample=sample, prompt=prompt, tokenizer=tokenizer, budget=budget,
                   evaluation_config=evaluation_config)               # problems; [] when right

This module imports no part of the training stack; the card's code does, in its own process.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qlora_finetuning.data import manifest_hash
from qlora_finetuning.data.serialization import PromptDefinition, format_user
from qlora_finetuning.data.token_budget import TokenBudgetConfig
from qlora_finetuning.demo.inputs import DemoInput
from qlora_finetuning.dsl import ReplyFormat, run_reply
from qlora_finetuning.evaluation.metrics import EvaluationConfig, EvaluationError, values_match
from qlora_finetuning.evaluation.runner import EVALUATION_NAME, RunnerContext, summarize_run
from qlora_finetuning.local_qlora.render import RenderError, render_chat
from qlora_finetuning.local_qlora.scoring import evaluation_runs
from qlora_finetuning.settings import ConfigError

LOCAL = "local"
MANIFEST_NAME = "manifest.json"
RESULT_NAME = "card_result.json"
SCRIPT_NAME = "card_code.py"
RESULT_FIELDS = ("system", "user", "text", "reply")
IDS_LISTED = 10  # how many differing example ids a record lists
_CODE_BLOCK = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
# Appended to a card's code: it hands the four texts back through a file, so nothing a
# library prints can be mistaken for the reply.
TRAILER = (
    "\n\nimport json as _json\n"
    f"with open({RESULT_NAME!r}, 'w', encoding='utf-8') as _handle:\n"
    "    _json.dump({'system': system, 'user': user, 'text': text, 'reply': reply}, _handle)\n"
)


# --------------------------------------------------------------------------- the dev runs


def _where(run_id: str, runs_dir: Path) -> str | None:
    try:
        manifest = json.loads((runs_dir / run_id / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return manifest.get("where") if isinstance(manifest, dict) else None


def find_run(
    ctx: RunnerContext, name: str, settings: Mapping[str, Any], split: str
) -> tuple[str, dict[str, Any]] | None:
    """The newest complete run of a predictor on this machine, with its summary.

    A run matches by the predictor's name and settings, the split, the prompt, the split
    manifest, and the reply limit, with no example limit, made here (``where: local``), and
    counting as a result. The recorded run of arm A1 has the same name and settings as the
    reference run and was made in the cloud: the place tells them apart.
    """
    current = manifest_hash(ctx.split_cfg)
    runs_dir = ctx.tracking_cfg.runs_dir
    runs = evaluation_runs(runs_dir)

    def age(run_id: str) -> tuple[str, int]:
        # A run id starts with the second it was made in; two runs of one second are told
        # apart by when their record was written.
        try:
            written = (runs_dir / run_id / EVALUATION_NAME).stat().st_mtime_ns
        except OSError:
            written = 0
        return run_id.split("-", 1)[0], written

    for run_id in sorted(runs, key=age, reverse=True):
        evaluation = runs[run_id]
        key = evaluation["key"]
        predictor = key.get("predictor") or {}
        if (
            key.get("split") != split
            or key.get("prompt_id") != ctx.prompt.prompt_id
            or key.get("manifest_hash") != current
            or key.get("max_new_tokens") != ctx.max_new_tokens
            or predictor.get("name") != name
            or predictor.get("settings") != dict(settings)
            or evaluation.get("limit") is not None
            or _where(run_id, ctx.tracking_cfg.runs_dir) != LOCAL
        ):
            continue
        try:
            summary = summarize_run(run_id, ctx)
        except (ConfigError, EvaluationError):
            continue  # no predictions file, or one that cannot be trusted
        if summary.get("result"):
            return run_id, summary
    return None


def compare(reference: Sequence[Any], other: Sequence[Any]) -> dict[str, Any]:
    """Two runs over the same examples, reply by reply. ``reference`` comes first.

    Each item is a prediction line (``example_id``, ``reply``, ``score``). Returns counts
    and example ids only: no reply text.
    """
    if [line.example_id for line in reference] != [line.example_id for line in other]:
        raise ConfigError("the two runs did not answer the same examples in the same order")
    differing, right_to_wrong, wrong_to_right = [], 0, 0
    right_reference = right_other = 0
    for first, second in zip(reference, other, strict=True):
        was, now = bool(first.score["execution_correct"]), bool(second.score["execution_correct"])
        right_reference += was
        right_other += now
        right_to_wrong += was and not now
        wrong_to_right += now and not was
        if first.reply != second.reply:
            differing.append(first.example_id)
    return {
        "examples": len(reference),
        "replies_that_differ": len(differing),
        "differing_ids": differing[:IDS_LISTED],
        "right_reference": right_reference,
        "right_other": right_other,
        "right_to_wrong": right_to_wrong,
        "wrong_to_right": wrong_to_right,
    }


# --------------------------------------------------------------------------- the card's code


def extract_code(card_text: str) -> str:
    """The first Python block of a card: its usage code."""
    match = _CODE_BLOCK.search(card_text.replace("\r\n", "\n"))
    if match is None:
        raise ConfigError("the card has no Python code block")
    return match.group(1) + "\n"


def assign(code: str, values: Mapping[str, Any]) -> str:
    """``code`` with each ``NAME = ...`` line at the left margin given a new value.

    A name that the code does not assign exactly once is a ``ConfigError``: the card's code
    changed and the check must be changed with it.
    """
    for name, value in values.items():
        pattern = re.compile(rf"^{re.escape(name)} = .*$", re.MULTILINE)
        found = pattern.findall(code)
        if len(found) != 1:
            raise ConfigError(
                f"the card's code assigns {name} {len(found)} times at the left margin; "
                "the check expects exactly one"
            )
        line = f"{name} = {value!r}"
        code = pattern.sub(lambda _match, line=line: line, code)
    return code


def run_card_code(
    code: str,
    values: Mapping[str, Any],
    *,
    timeout: int,
    python: str | None = None,
) -> dict[str, Any]:
    """Run a card's code in an empty directory and return the texts it made.

    ``values`` replaces assignments of the code: the package folder for the repository
    name, and a switch where the card has one. The process may not reach the network
    (``HF_HUB_OFFLINE``) and does not see the project's code. Returns ``system``, ``user``,
    ``text``, ``reply``, and ``seconds``. Raises ``ConfigError`` with the end of the
    process's output when it fails.
    """
    script = assign(code, values) + TRAILER
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8")
    with tempfile.TemporaryDirectory(prefix="card-code-") as folder:
        path = Path(folder) / SCRIPT_NAME
        path.write_text(script, encoding="utf-8", newline="\n")
        started = time.perf_counter()
        try:
            done = subprocess.run(
                [python or sys.executable, SCRIPT_NAME],
                cwd=folder,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise ConfigError(f"the card's code did not finish in {timeout} seconds") from None
        seconds = time.perf_counter() - started
        if done.returncode != 0:
            tail = " | ".join((done.stderr or done.stdout).strip().splitlines()[-3:])
            raise ConfigError(f"the card's code failed (exit code {done.returncode}): {tail}")
        try:
            result = json.loads((Path(folder) / RESULT_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ConfigError("the card's code ended without its four texts") from None
    if not isinstance(result, dict) or any(
        not isinstance(result.get(name), str) for name in RESULT_FIELDS
    ):
        raise ConfigError("the card's code must define system, user, text, and reply as text")
    result["seconds"] = round(seconds, 1)
    return result


def check_card_run(
    result: Mapping[str, Any],
    *,
    sample: DemoInput,
    prompt: PromptDefinition,
    tokenizer: Any,
    budget: TokenBudgetConfig,
    evaluation_config: EvaluationConfig,
) -> tuple[list[str], dict[str, Any]]:
    """What is wrong with what a card's code made, and what its reply gave.

    The system and user texts must be the project's own for the sample, the prompt must be
    the one the backend sends, and the reply must run to the sample's expected result.
    """
    table = [list(row) for row in sample.table]
    problems = []
    if result["system"] != prompt.system:
        problems.append("the system text is not the project's")
    if result["user"] != format_user(sample.passages, table, sample.question, prompt, "the sample"):
        problems.append("the user text is not what the serializer writes for the sample")
    try:
        expected_text = render_chat(
            result["system"], result["user"], tokenizer, budget.thinking_off_suffix
        )
    except RenderError as exc:
        expected_text = None
        problems.append(str(exc))
    if expected_text is not None and result["text"] != expected_text:
        problems.append("the prompt is not the one the project's backend sends")

    outcome = run_reply(result["reply"], table, ReplyFormat.from_prompt(prompt))
    facts = {
        "program_ran": bool(outcome.ok),
        "in_the_format": bool(outcome.strict),
        "value": outcome.value,
        "error_code": outcome.error_code,
    }
    if not outcome.ok:
        problems.append(f"the reply's program did not run ({outcome.error_code})")
    elif sample.expected is None:
        problems.append("the sample has no expected result to compare with")
    elif not values_match(outcome.value, sample.expected, evaluation_config.execution):
        problems.append("the reply's program does not give the sample's expected result")
    return problems, facts
