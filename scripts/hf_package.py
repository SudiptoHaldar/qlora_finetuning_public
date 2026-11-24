"""Build and check the two packages that can be published (req_006 v6.02).

The adapter package is arm A1's adapter with its tokenizer, prompt, split manifest,
training facts, and model card. The merged copy is the base model with that adapter in its
weights, with its own card. Both are written under git-ignored ``artifacts/release/``.

Usage (from the repo root)::

    venv\\Scripts\\python -m scripts.hf_package check      # what a build needs; writes nothing
    venv\\Scripts\\python -m scripts.hf_package adapter    # write artifacts/release/adapter/
    venv\\Scripts\\python -m scripts.hf_package merge      # write artifacts/release/merged/ (CPU)
    venv\\Scripts\\python -m scripts.hf_package verify     # run the cards' code (GPU)
    venv\\Scripts\\python -m scripts.hf_package cards      # write both cards and the record
    venv\\Scripts\\python -m scripts.hf_package status     # checks P1 to P10; writes nothing

Between ``merge`` and ``verify`` come four dev runs, each through the ordinary runner
(``python -m scripts.evaluate --split dev --predictor ...``): ``qwen3-4b-tuned --adapter
selected``, then ``qwen3-4b-package``, ``qwen3-4b-merged``, and ``qwen3-4b-merged-full``.

``cards`` writes a card to its package and to ``reports/packaging/`` only when every
licence entry of ``config/packaging.yaml`` is confirmed by the owner. Until then it writes
drafts under ``artifacts/release/drafts/`` for the owner to read.

Nothing here is billed, calls AWS, uploads anything, or reads the test split. Uploading is
the owner's step and belongs to req_006 v6.03.

Exit codes: 0 done, and for ``check``, ``cards``, and ``status`` every check passed;
1 a check failed; 2 the command could not start (a config problem, a missing file).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

try:
    from qlora_finetuning.logging_setup import setup_logging
    from qlora_finetuning.packaging import workflow
    from qlora_finetuning.packaging.config import load_packaging_config
    from qlora_finetuning.settings import ConfigError
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.hf_package"
    ) from None

PASS, FAIL = "PASS", "FAIL"
COMMANDS = ("check", "adapter", "merge", "verify", "cards", "status")


def _show(text: str) -> None:
    print(text.encode("ascii", "backslashreplace").decode("ascii"))


def print_checks(checks: list[dict[str, Any]]) -> int:
    """Print each check; returns how many failed."""
    for check in checks:
        status = PASS if check["passed"] else FAIL
        _show(f"{status}  {check['id']}  {check['name']}: {check['detail']}")
    failed = sum(not check["passed"] for check in checks)
    _show(f"{len(checks) - failed} passed, {failed} failed")
    return failed


def main(argv: list[str] | None = None, cfg: Any = None, steps: Any = workflow) -> int:
    """Run one command. ``cfg`` and ``steps`` are stand-ins for tests."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.hf_package",
        description="Build and check the adapter package and the merged copy.",
    )
    parser.add_argument("command", choices=COMMANDS)
    args = parser.parse_args(argv)

    setup_logging()
    try:
        cfg = cfg or load_packaging_config()
        if args.command == "adapter":
            folder = steps.build_adapter(cfg)
            _show(f"WROTE  {folder.relative_to(cfg.root).as_posix()}/ (without its card)")
            _show("next: python -m scripts.hf_package merge")
            return 0
        if args.command == "merge":
            folder = steps.build_merged(cfg)
            _show(f"WROTE  {folder.relative_to(cfg.root).as_posix()}/ (without its card)")
            _show("next: the four dev runs, then python -m scripts.hf_package verify")
            return 0
        if args.command == "verify":
            for line in steps.verify(cfg):
                _show(line)
            _show("next: python -m scripts.hf_package cards")
            return 0
        if args.command == "check":
            return 1 if print_checks(steps.preflight(cfg)) else 0
        if args.command == "cards":
            state, written = steps.write_cards(cfg)
            for path in written:
                _show(f"WROTE  {path.relative_to(cfg.root).as_posix()}")
        else:
            state = steps.gather(cfg)
        failed = print_checks(state.record["checks"])
        _show(f"adapter package: {state.record['adapter']['status']}")
        _show(f"merged copy: {state.record['merged']['status']}")
        return 1 if failed else 0
    except ConfigError as exc:
        _show(f"CONFIG ERROR: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
