"""Score every gold reply with the evaluation harness and flag the ones that fail (req_002 v2.02).

Before any model is scored, the scorer is tried on the replies whose score is known: the
dataset's own. This script builds each example's gold reply, runs it, scores it against
itself, runs checks K1 to K7, and writes

- the committed report ``reports/data/finqa_gold_check_report.md`` and ``.json``, and
- the committed flags file ``splits/finqa_gold_flags.json``: the examples whose gold program
  does not reproduce its gold answer. They are never dropped; scoring reports its totals
  with and without them.

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.finqa_gold_check
    venv\\Scripts\\python -m scripts.finqa_gold_check --check

``--check`` writes nothing and fails if a file differs from a fresh one.

Exit codes: 0 every check passed and the files are current, 1 a check failed or a file is
out of date, 2 config problem, no verified snapshot, or no split manifest.

Needs the snapshot and the split manifest; no network, no model. It covers all 8,281
examples, test rows included: it reads their gold programs and gold answers and prints
counts and example ids only.
"""

from __future__ import annotations

import argparse
import sys

try:
    from qlora_finetuning.data import DataConfig, load_data_config, load_raw_examples
    from qlora_finetuning.data.gold_check import (
        REPORTED,
        build_flags,
        build_report,
        read_flags,
        stale_files,
        write_flags,
        write_report,
    )
    from qlora_finetuning.data.serialization import PromptDefinition, load_prompt
    from qlora_finetuning.data.split import (
        SplitConfig,
        SplitError,
        load_split_config,
        read_manifest,
    )
    from qlora_finetuning.data.split_loader import check_manifest_against_data
    from qlora_finetuning.evaluation.metrics import EvaluationConfig, load_evaluation_config
    from qlora_finetuning.settings import ConfigError
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.finqa_gold_check"
    ) from None

PASS, FAIL = "PASS", "FAIL"


def _relative(path, config: EvaluationConfig) -> str:
    try:
        return path.relative_to(config.root).as_posix()
    except ValueError:
        return str(path)


def main(
    argv: list[str] | None = None,
    cfg: SplitConfig | None = None,
    data_cfg: DataConfig | None = None,
    prompt: PromptDefinition | None = None,
    config: EvaluationConfig | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Score every gold reply with the evaluation harness."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; fail if the report or the flags file differs from a fresh one",
    )
    args = parser.parse_args(argv)

    try:
        prompt = prompt or load_prompt()
        cfg = cfg or load_split_config()
        data_cfg = data_cfg or load_data_config()
        config = config or load_evaluation_config()
        manifest = read_manifest(cfg)
        examples = load_raw_examples(data_cfg)
        previous = None if args.check else read_flags(config)
    except ConfigError as exc:  # missing snapshot, missing or altered manifest, bad config
        print(f"CONFIG ERROR: {exc}")
        return 2

    try:
        check_manifest_against_data(manifest, examples, data_cfg)
        data_problem = None
    except SplitError as exc:
        data_problem = str(exc)

    report = build_report(examples, manifest, prompt, config, previous, data_problem)
    for check in report["checks"]:
        status = PASS if check["passed"] else FAIL
        print(f"{status}  {check['id']}  {check['name']}: {check['detail']}")
    failed = sum(not check["passed"] for check in report["checks"])
    print(f"{len(report['checks']) - failed} passed, {failed} failed")
    for name in REPORTED:
        stats = report["splits"][name]
        print(
            f"{name}: {stats['gold_programs_that_reproduce_the_gold_answer']:,} of "
            f"{stats['examples']:,} gold programs reproduce the gold answer, "
            f"{stats['flagged_examples']:,} flagged"
        )

    stale: list[str] = []
    if args.check:
        stale = stale_files(report, config)
        if stale:
            print(
                f"OUT OF DATE: {', '.join(stale)}. The data, the split, the wording, the "
                "executor, or config/evaluation.yaml changed; rebuild without --check and review"
            )
        else:
            print("report and flags files are current")
    else:
        for path in write_report(report, config):
            print(f"written {_relative(path, config)}")
        print(f"written {_relative(write_flags(build_flags(report), config), config)}")
    return 1 if failed or stale else 0


if __name__ == "__main__":
    sys.exit(main())
