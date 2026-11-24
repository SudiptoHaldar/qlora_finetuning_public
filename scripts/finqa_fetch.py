"""Download the pinned FinQA snapshot and check it (req_001 v1.01).

Fetches the three dataset files named in ``config/data.yaml`` at the pinned commit into
``artifacts/data/finqa/<commit>/`` and checks each one's size, SHA-256, and row count. Files
that are already present and correct are not downloaded again. Prints one PASS or FAIL line
per file.

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.finqa_fetch
    venv\\Scripts\\python -m scripts.finqa_fetch --verify-only
    venv\\Scripts\\python -m scripts.finqa_fetch --force

Exit codes: 0 every file verified, 1 a file is missing or wrong or the download failed,
2 config problem.

The download needs ``huggingface_hub`` (part of the training stack) and network access. The
dataset is public; no token is used. ``--verify-only`` needs neither.
"""

from __future__ import annotations

import argparse
import logging
import sys

try:
    from qlora_finetuning.data import (
        DataConfig,
        SnapshotError,
        fetch_snapshot,
        load_data_config,
        snapshot_path,
        verify_snapshot,
    )
    from qlora_finetuning.data.snapshot import Downloader
    from qlora_finetuning.logging_setup import get_logger, setup_logging
    from qlora_finetuning.settings import ConfigError
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.finqa_fetch"
    ) from None

log = get_logger("scripts.finqa_fetch")

# Libraries that log every HTTP request at INFO; they are held at WARNING during a run.
NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3", "filelock")
PASS, FAIL = "PASS", "FAIL"


def main(
    argv: list[str] | None = None,
    cfg: DataConfig | None = None,
    download: Downloader | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Download and verify the pinned FinQA snapshot.")
    parser.add_argument(
        "--verify-only", action="store_true", help="check the files on disk; download nothing"
    )
    parser.add_argument("--force", action="store_true", help="download every file again")
    args = parser.parse_args(argv)
    if args.verify_only and args.force:
        parser.error("--verify-only and --force cannot be combined")

    try:
        cfg = cfg or load_data_config()
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 2

    setup_logging()
    for noisy in NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # The Hub library warns that no token is set. The dataset is public and none is wanted.
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    print(f"Dataset {cfg.repo_id} at commit {cfg.revision}")
    print(f"Folder  {snapshot_path(cfg)}")
    error = None
    if not args.verify_only:
        try:
            downloaded = fetch_snapshot(cfg, download=download, force=args.force)
        except SnapshotError as exc:
            error = str(exc)
        else:
            print(f"Downloaded {len(downloaded)} of {len(cfg.files)} files")

    checks = verify_snapshot(cfg)
    for check in checks:
        status = PASS if check.ok else FAIL
        print(f"{status}  {check.official_split}  {check.path}: {check.detail}")
    failed = sum(not check.ok for check in checks)
    print(f"{len(checks) - failed} verified, {failed} failed")
    if error:
        print(f"SNAPSHOT ERROR: {error}")
    return 1 if failed or error else 0


if __name__ == "__main__":
    sys.exit(main())
