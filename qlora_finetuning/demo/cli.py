"""The demo's command line: ``python -m qlora_finetuning.demo`` (req_006 v6.01).

Usage (from the repo root)::

    venv\\Scripts\\python -m qlora_finetuning.demo                         # the sample
    venv\\Scripts\\python -m qlora_finetuning.demo --input my.json
    venv\\Scripts\\python -m qlora_finetuning.demo --input a.json b.json --json
    venv\\Scripts\\python -m qlora_finetuning.demo --show-input            # context and table too
    venv\\Scripts\\python -m qlora_finetuning.demo --untuned               # the base model
    venv\\Scripts\\python -m qlora_finetuning.demo --adapter RUN_ID/checkpoint-N
    venv\\Scripts\\python -m qlora_finetuning.demo --check                 # no model is loaded

The model is Qwen3-4B in 4-bit on the local GPU, with the adapter the selection record
names unless told otherwise. An adapter is named by its training run, never by a path.
Every input is checked before the model is loaded, and each is answered alone.

It is a demonstration: it scores nothing, starts no run, writes no file, and calls no
AWS service. A selected checkpoint that is not on this machine is an error that names the
command to pull it.

Results go to standard output and log lines to standard error.

Exit codes: 0 every input's program ran; 1 at least one did not; 2 the demo stopped
before a reply (a bad input file, no adapter, no GPU, no weights, a model failure).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from qlora_finetuning.data.serialization import load_prompt
from qlora_finetuning.demo.answer import answer
from qlora_finetuning.demo.inputs import DemoInput, load_input, sample_path
from qlora_finetuning.demo.render import as_json, as_text
from qlora_finetuning.evaluation.predictor import PredictorError
from qlora_finetuning.local_qlora.adapters import (
    ADAPTER_WEIGHTS,
    CHECKPOINTS,
    SELECTED,
    AdapterRef,
    adapter_sha256,
    read_selection,
)
from qlora_finetuning.local_qlora.training_config import load_training_config
from qlora_finetuning.logging_setup import get_logger, setup_logging
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking.config import load_tracking_config

logger = get_logger("qlora_finetuning.demo")

# Libraries that log every HTTP request or warn about a missing token; held quiet.
NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3", "filelock")
PULL = "python -m scripts.sagemaker_job pull --checkpoint"
NOTICE = "a demonstration, not a measured result: nothing here is scored or recorded"


def demo_context() -> SimpleNamespace:
    """What the tuned predictor reads from a runner, without a runner.

    The prompt, the split (``None``: the committed one), and where runs are kept.
    """
    return SimpleNamespace(
        prompt=load_prompt(), split_cfg=None, tracking_cfg=load_tracking_config()
    )


def require_selected_here(context: Any) -> None:
    """Stop with the pull command when the selected checkpoint is not on this machine."""
    tracking = context.tracking_cfg
    record = read_selection(load_training_config(root=tracking.root).selection_record)
    name = str(record["selected"]["adapter"])
    run_id, _, checkpoint = name.partition("/")
    if not checkpoint:
        return
    weights = tracking.runs_dir / run_id / CHECKPOINTS / checkpoint / ADAPTER_WEIGHTS
    if not weights.is_file():
        step = checkpoint.rsplit("-", 1)[-1]
        raise ConfigError(
            f"the selected checkpoint ({name}) is not on this machine. Bring it here with: "
            f"{PULL} {step}  (it reads from S3 and needs an AWS sign-in), then run the demo again"
        )


def _resolve(adapter: str, context: Any) -> AdapterRef:
    from qlora_finetuning.local_qlora.predictor import resolve_tuned_adapter

    return resolve_tuned_adapter(adapter, context)


def _tuned(adapter: str, context: Any) -> Any:
    from qlora_finetuning.local_qlora.predictor import build_tuned_predictor

    return build_tuned_predictor(adapter, context)


def _untuned() -> Any:
    from qlora_finetuning.local_qlora.predictor import build_predictor

    return build_predictor()


def _check(inputs: list[DemoInput], adapter: str | None, context: Any, resolve: Callable) -> None:
    for demo_input in inputs:
        print(
            f"PASS  input {demo_input.source}: {len(demo_input.table)} table rows, "
            f"{len(demo_input.passages)} passages"
        )
    if adapter is None:
        print("PASS  the untuned model is asked: no adapter to find")
    else:
        ref = resolve(adapter, context)
        print(f"PASS  adapter {adapter}: {ref.name}, weights {adapter_sha256(ref.path)[:12]}...")
    print("CHECK ONLY: no model was loaded")


def main(
    argv: list[str] | None = None,
    *,
    context: Any = None,
    resolve: Callable[[str, Any], AdapterRef] = _resolve,
    tuned: Callable[[str, Any], Any] = _tuned,
    untuned: Callable[[], Any] = _untuned,
    **answer_settings: Any,
) -> int:
    """Run the demo. The keyword arguments are stand-ins for tests."""
    parser = argparse.ArgumentParser(
        prog="python -m qlora_finetuning.demo",
        description="Ask the tuned Qwen model a question about a table and run its program.",
    )
    parser.add_argument(
        "--input", nargs="+", metavar="FILE", help="JSON input files (default: the sample)"
    )
    which = parser.add_mutually_exclusive_group()
    which.add_argument(
        "--adapter",
        metavar="NAME",
        help=f"{SELECTED} (the default), RUN_ID, or RUN_ID/checkpoint-N; never a path",
    )
    which.add_argument("--untuned", action="store_true", help="ask the base model")
    parser.add_argument("--json", action="store_true", help="one JSON object per input")
    parser.add_argument(
        "--show-input",
        action="store_true",
        help="also print the context and the table the model was given",
    )
    parser.add_argument(
        "--check", action="store_true", help="check the inputs and the adapter; load no model"
    )
    args = parser.parse_args(argv)

    setup_logging()
    for noisy in NOISY_LOGGERS:
        get_logger(noisy).setLevel("WARNING")
    adapter = None if args.untuned else (args.adapter or SELECTED)
    try:
        inputs = [load_input(path) for path in (args.input or [sample_path()])]
        context = context or demo_context()
        if adapter == SELECTED:
            require_selected_here(context)
        if args.check:
            _check(inputs, adapter, context, resolve)
            return 0
        logger.info(
            "loading the model%s on the local GPU; %s",
            "" if adapter is None else f" with adapter {adapter}",
            NOTICE,
        )
        predictor = untuned() if adapter is None else tuned(adapter, context)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 2

    every_program_ran = True
    for number, demo_input in enumerate(inputs):
        try:
            result = answer(predictor, demo_input, prompt=context.prompt, **answer_settings)
        except PredictorError as exc:
            print(f"MODEL ERROR: {exc}")
            return 2
        except ConfigError as exc:
            print(f"CONFIG ERROR: {exc}")
            return 2
        if args.json:
            print(as_json(result, show_input=args.show_input))
        else:
            print(("\n" if number else "") + as_text(result, show_input=args.show_input))
        every_program_ran = every_program_ran and result.outcome.ok
    return 0 if every_program_ran else 1
