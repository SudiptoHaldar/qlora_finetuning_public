"""Count the tokens of the fit and dev records and report on them (req_001 v1.04).

Counts every canonical record with the tokenizer pinned in ``config/token_budget.yaml``, as
the full sequence Path A trains on, runs checks T1 to T6, and writes the committed report
``reports/data/finqa_token_budget_report.md`` and ``.json``: lengths, how many records exceed
each threshold, the longest records by id, and the token volume of one pass over the data.
The report holds no example text and no prices.

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.finqa_tokens
    venv\\Scripts\\python -m scripts.finqa_tokens --check
    venv\\Scripts\\python -m scripts.finqa_tokens --download

``--check`` writes nothing and fails if the report differs from a fresh one. ``--download``
lets the tokenizer files be fetched first (a few megabytes, once); otherwise they are read
from the local Hugging Face cache and no network is used.

Exit codes: 0 every check passed and the files are current, 1 a check failed or the report is
out of date, 2 config problem, no training stack, no tokenizer, no snapshot, or no manifest.

Needs the training stack for the tokenizer library. Loads no model weights and uses no GPU.
The test split is never read.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

try:
    from qlora_finetuning.data import (
        DataConfig,
        load_data_config,
        load_split,
        manifest_hash,
    )
    from qlora_finetuning.data.serialization import (
        PromptDefinition,
        SerializationError,
        build_record,
        load_prompt,
    )
    from qlora_finetuning.data.serialization_report import read_committed_digests
    from qlora_finetuning.data.split import SplitConfig, load_split_config
    from qlora_finetuning.data.token_budget import (
        TokenBudgetConfig,
        count_split,
        load_token_budget_config,
        load_tokenizer,
        tokenizer_fingerprint,
    )
    from qlora_finetuning.data.token_budget_report import build_report, stale_files, write_report
    from qlora_finetuning.logging_setup import setup_logging
    from qlora_finetuning.settings import ConfigError
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.finqa_tokens"
    ) from None

PASS, FAIL = "PASS", "FAIL"
SPLITS = ("fit", "dev", "quick_dev")
# Libraries that log every HTTP request or warn about a missing token; held quiet during a run.
NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3", "filelock")


def _relative(path, cfg: TokenBudgetConfig) -> str:
    try:
        return path.relative_to(cfg.root).as_posix()
    except ValueError:
        return str(path)


def main(
    argv: list[str] | None = None,
    cfg: SplitConfig | None = None,
    data_cfg: DataConfig | None = None,
    prompt: PromptDefinition | None = None,
    budget: TokenBudgetConfig | None = None,
    tokenizer: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Count the tokens of the fit and dev records.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; fail if the report files differ from a fresh report",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="allow the tokenizer files to be fetched (a few megabytes, once)",
    )
    args = parser.parse_args(argv)

    try:
        budget = budget or load_token_budget_config()
        prompt = prompt or load_prompt()
        cfg = cfg or load_split_config()
        data_cfg = data_cfg or load_data_config()
        if tokenizer is None:
            setup_logging()
            for noisy in NOISY_LOGGERS:
                logging.getLogger(noisy).setLevel(logging.ERROR)
            tokenizer = load_tokenizer(budget, download=args.download)
        by_split = {}
        for name in SPLITS:
            records = [build_record(example, prompt) for example in load_split(name, cfg, data_cfg)]
            by_split[name] = (records, count_split(records, tokenizer, budget.thinking_off_suffix))
        inputs = {
            "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
            "prompt_id": prompt.prompt_id,
            "split_content_hash": manifest_hash(cfg),
        }
    except SerializationError as exc:
        print(f"SERIALIZATION ERROR: {exc}")
        return 1
    except ConfigError as exc:  # config, tokenizer, snapshot, or manifest
        print(f"CONFIG ERROR: {exc}")
        return 2

    report = build_report(by_split, budget, inputs, read_committed_digests(prompt.report_dir))
    for check in report["checks"]:
        status = PASS if check["passed"] else FAIL
        print(f"{status}  {check['id']}  {check['name']}: {check['detail']}")
    failed = sum(not check["passed"] for check in report["checks"])
    print(f"{len(report['checks']) - failed} passed, {failed} failed")
    for name in SPLITS:
        stats = report["splits"][name]
        full = stats["full_sequence_tokens"]
        print(
            f"{name}: {stats['records']:,} records, full sequence p50 {full['p50']:,} "
            f"p99 {full['p99']:,} max {full['max']:,}, {stats['tokens_in_one_pass']:,} tokens "
            "in one pass"
        )

    stale: list[str] = []
    if args.check:
        stale = stale_files(report, budget.report_dir)
        if stale:
            print(
                f"OUT OF DATE: {', '.join(stale)} in {_relative(budget.report_dir, budget)}. The "
                "wording, the split, the data, or the tokenizer changed; rebuild without --check"
            )
        else:
            print("report files are current")
    else:
        for path in write_report(report, budget.report_dir):
            print(f"written {_relative(path, budget)}")
    return 1 if failed or stale else 0


if __name__ == "__main__":
    sys.exit(main())
