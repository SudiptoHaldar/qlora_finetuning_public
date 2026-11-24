"""Build the canonical records for fit and dev and report on them (req_001 v1.03).

Each FinQA example becomes one record of three texts (system, user, target) from the wording
in ``config/prompt.yaml``. This script serializes the fit and dev splits, runs checks C1 to
C10, and writes the committed report ``reports/data/finqa_serialization_report.md`` and
``.json``: lengths in characters, what was skipped, and a digest of all records. The report
holds no example text.

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.finqa_serialize
    venv\\Scripts\\python -m scripts.finqa_serialize --check
    venv\\Scripts\\python -m scripts.finqa_serialize --dump
    venv\\Scripts\\python -m scripts.finqa_serialize --show "<example id>"

``--check`` writes nothing and fails if the report differs from a fresh one. ``--dump`` also
writes every fit and dev record to ``artifacts/serialized/`` (git-ignored) for reading.
``--show`` prints one fit or dev record and does nothing else; a test id is refused.

Exit codes: 0 every check passed and the files are current, 1 a check failed, the report is
out of date, or the id cannot be shown, 2 config problem, no verified snapshot, or no split
manifest.

Needs the snapshot and the split manifest; no network. The test split is never read.
"""

from __future__ import annotations

import argparse
import json
import sys

try:
    from qlora_finetuning.data import DataConfig, load_data_config, load_split, split_ids
    from qlora_finetuning.data.serialization import (
        PromptDefinition,
        SerializationError,
        build_record,
        load_prompt,
    )
    from qlora_finetuning.data.serialization_report import (
        build_report,
        read_committed_digests,
        stale_files,
        write_report,
    )
    from qlora_finetuning.data.split import SplitConfig, load_split_config
    from qlora_finetuning.settings import ConfigError
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.finqa_serialize"
    ) from None

PASS, FAIL = "PASS", "FAIL"
SPLITS = ("fit", "dev", "quick_dev")
DUMPED = ("fit", "dev")
RULE = "-" * 72


def _relative(path, prompt: PromptDefinition) -> str:
    try:
        return path.relative_to(prompt.root).as_posix()
    except ValueError:
        return str(path)


def _show(example_id: str, cfg: SplitConfig, data_cfg: DataConfig, prompt: PromptDefinition) -> int:
    """Print one fit or dev record. Test ids are refused before any data is read."""
    if example_id in set(split_ids("test", cfg)):
        print(
            f"REFUSED: {example_id} is in the test split, which is locked until req_005 v5.01. "
            "Only fit and dev records can be shown"
        )
        return 1
    for name in DUMPED:
        if example_id in set(split_ids(name, cfg)):
            example = next(e for e in load_split(name, cfg, data_cfg) if e.id == example_id)
            record = build_record(example, prompt)
            print(f"{example_id}  ({name}; prompt id {record.prompt_id[:12]}...)")
            for title, text in (
                ("SYSTEM", record.system),
                ("USER", record.user),
                ("TARGET", record.target),
            ):
                print(f"{RULE}\n{title}\n{RULE}\n{text}")
            return 0
    print(f"NOT FOUND: no example has the id {example_id}")
    return 1


def main(
    argv: list[str] | None = None,
    cfg: SplitConfig | None = None,
    data_cfg: DataConfig | None = None,
    prompt: PromptDefinition | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Serialize fit and dev into canonical records.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; fail if the report files differ from a fresh report",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="also write every fit and dev record under artifacts/serialized/ (git-ignored)",
    )
    parser.add_argument("--show", metavar="ID", help="print one fit or dev record and stop")
    args = parser.parse_args(argv)
    if args.check and args.dump:
        parser.error("--check writes nothing, so it cannot be combined with --dump")

    try:
        prompt = prompt or load_prompt()
        cfg = cfg or load_split_config()
        data_cfg = data_cfg or load_data_config()
        if args.show:
            return _show(args.show, cfg, data_cfg, prompt)
        by_split = {}
        for name in SPLITS:
            examples = load_split(name, cfg, data_cfg)
            by_split[name] = (examples, [build_record(example, prompt) for example in examples])
        allowed = set(split_ids("fit", cfg)) | set(split_ids("dev", cfg))
        test_ids = set(split_ids("test", cfg))
    except SerializationError as exc:
        print(f"SERIALIZATION ERROR: {exc}")
        return 1
    except ConfigError as exc:  # missing snapshot, missing or altered manifest, bad config
        print(f"CONFIG ERROR: {exc}")
        return 2

    previous = None if args.check else read_committed_digests(prompt.report_dir)
    report = build_report(by_split, prompt, previous, allowed, test_ids)
    for check in report["checks"]:
        status = PASS if check["passed"] else FAIL
        print(f"{status}  {check['id']}  {check['name']}: {check['detail']}")
    failed = sum(not check["passed"] for check in report["checks"])
    print(f"{len(report['checks']) - failed} passed, {failed} failed")
    print(
        "  ".join(f"{name} {len(records):,}" for name, (_, records) in by_split.items())
        + f"  (prompt id {prompt.prompt_id})"
    )

    stale: list[str] = []
    if args.check:
        stale = stale_files(report, prompt.report_dir)
        if stale:
            print(
                f"OUT OF DATE: {', '.join(stale)} in {_relative(prompt.report_dir, prompt)}. The "
                "wording, the split, or the data changed; rebuild without --check and review"
            )
        else:
            print("report files are current")
    else:
        for path in write_report(report, prompt.report_dir):
            print(f"written {_relative(path, prompt)}")
    if args.dump:
        prompt.dump_dir.mkdir(parents=True, exist_ok=True)
        for name in DUMPED:
            path = prompt.dump_dir / f"{name}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for record in by_split[name][1]:
                    handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            print(f"dumped {_relative(path, prompt)} (not for committing)")
    return 1 if failed or stale else 0


if __name__ == "__main__":
    sys.exit(main())
