"""Build or check the frozen FinQA split (req_001 v1.02).

Assigns every example to fit, dev, or test with the settings in ``config/split.yaml``, keeps
the result in the committed manifest ``splits/finqa_split_manifest.json``, runs checks P1 to
P10, and writes the split report under ``reports/data/``.

The manifest is written **once**. If it already exists, this script only confirms that the
settings and the data still produce it; it never replaces it. There is no force option.

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.finqa_split
    venv\\Scripts\\python -m scripts.finqa_split --check

``--check`` writes nothing and fails if the manifest or the report differs from a fresh
build.

Exit codes: 0 every check passed and the files are current, 1 a check failed or a committed
file differs, 2 config problem or no verified snapshot.

Needs the snapshot (``python -m scripts.finqa_fetch``); no network.
"""

from __future__ import annotations

import argparse
import sys

try:
    from qlora_finetuning.data import DataConfig, load_data_config, load_raw_examples
    from qlora_finetuning.data.split import (
        SplitConfig,
        SplitError,
        build_split,
        load_split_config,
        read_manifest,
        render_manifest,
        write_manifest,
    )
    from qlora_finetuning.data.split_report import build_report, stale_files, write_report
    from qlora_finetuning.settings import ConfigError
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.finqa_split"
    ) from None

PASS, FAIL = "PASS", "FAIL"


def _relative(path, cfg: SplitConfig) -> str:
    try:
        return path.relative_to(cfg.root).as_posix()
    except ValueError:
        return str(path)


def main(
    argv: list[str] | None = None,
    cfg: SplitConfig | None = None,
    data_cfg: DataConfig | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Build or check the frozen FinQA split.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; fail if the manifest or the report differs from a fresh build",
    )
    args = parser.parse_args(argv)

    try:
        cfg = cfg or load_split_config()
        data_cfg = data_cfg or load_data_config()
        examples = load_raw_examples(data_cfg)
        rebuilt = build_split(examples, cfg, data_cfg)
    except SplitError as exc:
        print(f"SPLIT ERROR: {exc}")
        return 1
    except ConfigError as exc:  # includes SnapshotError: missing or altered snapshot
        print(f"CONFIG ERROR: {exc}")
        return 2

    manifest_name = _relative(cfg.manifest_file, cfg)
    try:
        if args.check:
            manifest = read_manifest(cfg)
        else:
            written = write_manifest(rebuilt, cfg)
            manifest = read_manifest(cfg)
            print(f"{'written' if written else 'unchanged'} {manifest_name}")
    except SplitError as exc:
        print(f"SPLIT ERROR: {exc}")
        return 1

    report = build_report(manifest, rebuilt, examples, cfg, data_cfg)
    for check in report["checks"]:
        status = PASS if check["passed"] else FAIL
        print(f"{status}  {check['id']}  {check['name']}: {check['detail']}")
    failed = sum(not check["passed"] for check in report["checks"])
    print(f"{len(report['checks']) - failed} passed, {failed} failed")

    counts = manifest["counts"]
    print(
        f"fit {counts['fit']['examples']:,}  dev {counts['dev']['examples']:,}  "
        f"test {counts['test']['examples']:,}  quick-dev {counts['quick_dev']:,}  "
        f"(groups by {manifest['settings']['group_by']}: {counts['fit']['groups']}, "
        f"{counts['dev']['groups']}, {counts['test']['groups']})"
    )
    print(f"content hash {manifest['content_hash']}")

    stale: list[str] = []
    if args.check:
        if render_manifest(rebuilt) != render_manifest(manifest):
            stale.append(cfg.manifest_file.name)
        stale += stale_files(report, cfg.report_dir)
        if stale:
            print(
                f"OUT OF DATE: {', '.join(stale)}. The split is frozen: find what changed in "
                "the settings, the data, or the files before rebuilding anything"
            )
        else:
            print("manifest and report files are current")
    else:
        for path in write_report(report, cfg.report_dir):
            print(f"written {_relative(path, cfg)}")
    return 1 if failed or stale else 0


if __name__ == "__main__":
    sys.exit(main())
