"""Build the FinQA schema report from the pinned snapshot (req_001 v1.01).

Reads the verified snapshot, compares it with the expected facts in ``config/data.yaml``
(checks S1 to S10), and writes ``reports/data/finqa_schema_report.md`` and ``.json``. The
report holds counts and example ids only, and no timestamp, so running it again on the same
snapshot gives the same files.

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.finqa_report
    venv\\Scripts\\python -m scripts.finqa_report --check

``--check`` writes nothing and fails if the committed report differs from a fresh one.

Exit codes: 0 every check passed (and, with ``--check``, the files are current), 1 a check
failed or a report file is out of date, 2 config problem or no verified snapshot.

Needs no network. Fetch the snapshot first with ``python -m scripts.finqa_fetch``.
"""

from __future__ import annotations

import argparse
import sys

try:
    from qlora_finetuning.data import DataConfig, load_data_config, load_raw_examples, read_columns
    from qlora_finetuning.data.schema_report import build_report, stale_files, write_report
    from qlora_finetuning.settings import ConfigError
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.finqa_report"
    ) from None

PASS, FAIL = "PASS", "FAIL"


def _relative(path, cfg: DataConfig) -> str:
    try:
        return path.relative_to(cfg.root).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None, cfg: DataConfig | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the FinQA schema report.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; fail if the report files differ from a fresh report",
    )
    args = parser.parse_args(argv)

    try:
        cfg = cfg or load_data_config()
        report = build_report(load_raw_examples(cfg), cfg, read_columns(cfg))
    except ConfigError as exc:  # includes SnapshotError: missing or altered snapshot
        print(f"CONFIG ERROR: {exc}")
        return 2

    for check in report["checks"]:
        status = PASS if check["passed"] else FAIL
        print(f"{status}  {check['id']}  {check['name']}: {check['detail']}")
    failed = sum(not check["passed"] for check in report["checks"])
    print(f"{len(report['checks']) - failed} passed, {failed} failed")

    stale: list[str] = []
    if args.check:
        stale = stale_files(report, cfg.report_dir)
        if stale:
            print(
                f"OUT OF DATE: {', '.join(stale)} in {_relative(cfg.report_dir, cfg)}; "
                "rebuild with: python -m scripts.finqa_report"
            )
        else:
            print("report files are current")
    else:
        for path in write_report(report, cfg.report_dir):
            print(f"written {_relative(path, cfg)}")
    return 1 if failed or stale else 0


if __name__ == "__main__":
    sys.exit(main())
