"""Make and check the public copy of this repository (req_006 v6.04).

The copy is a commit's files, without history: the folders the owner left out are not
copied, the AWS account number and a few other private strings are replaced, the runbook
goes to ``docs/runbook.md``, and a notice says that the copy is a redacted snapshot. What
is left out and what is replaced is in ``config/public_export.yaml``.

Usage (from the repo root)::

    venv\\Scripts\\python -m scripts.public_export plan
    venv\\Scripts\\python -m scripts.public_export build --to C:\\path\\to\\public_folder
    venv\\Scripts\\python -m scripts.public_export check --to C:\\path\\to\\public_folder

``plan`` reads the commit and writes nothing. ``build`` writes the folder: it must be empty
or hold a previous export, and tracked files here must have no uncommitted change.
``check`` runs checks E1 to E10, among them lint and the test suite in a temporary copy of
the folder (several minutes), and writes ``reports/public_export/export_record.json`` and
``.md`` here. Without ``--to`` the folder is the one named by ``QLORA_PUBLIC_EXPORT_DIR`` in
``.env``.

**The tool makes no commit, no push, and no network call.** Committing in the folder,
pushing it, and making the repository public are the owner's steps (runbook section 12).

Exit codes: 0 done, and for ``check`` every check passed; 1 a check failed; 2 the command
could not start (a config problem, uncommitted changes, a folder that is refused).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    from qlora_finetuning.logging_setup import setup_logging
    from qlora_finetuning.public_export.build import write_copy
    from qlora_finetuning.public_export.checks import run_checks, write_record
    from qlora_finetuning.public_export.config import load_export_config
    from qlora_finetuning.public_export.source import (
        commit_files,
        commit_info,
        tracked_changes,
        untracked,
    )
    from qlora_finetuning.public_export.transform import expected_files
    from qlora_finetuning.settings import ConfigError
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.public_export"
    ) from None

PASS, FAIL = "PASS", "FAIL"
SHOWN = 5


def _show(text: str) -> None:
    print(text.encode("ascii", "backslashreplace").decode("ascii"))


def _expected(cfg: Any) -> Any:
    info = commit_info(cfg.root)
    return expected_files(commit_files(cfg.root, info.sha), cfg, info)


def _some(names: list[str]) -> str:
    more = f", and {len(names) - SHOWN} more" if len(names) > SHOWN else ""
    return ", ".join(names[:SHOWN]) + more


def plan(cfg: Any) -> int:
    expected = _expected(cfg)
    kept, left = expected.plan.kept, expected.plan.left_out
    _show(f"commit {expected.info.short} of {expected.info.day}")
    _show(f"COPY      {len(kept):,} files")
    for target, source in expected.plan.moved.items():
        _show(f"MOVE      {source} -> {target}")
    _show(f"GENERATE  {', '.join(expected.generated)}")
    _show(f"LEAVE OUT {len(left):,} files under: {', '.join(cfg.left_out)}")
    for name, count in expected.counts.items():
        _show(f"REPLACE   {name}: {count['occurrences']:,} in {count['files']:,} files")
    if expected.binary:
        names = _some(list(expected.binary))
        _show(f"AS IT IS  {len(expected.binary)} files that are not text: {names}")
    changed, new = tracked_changes(cfg.root), untracked(cfg.root)
    if changed:
        _show(f"NOT CLEAN {len(changed)} tracked files have uncommitted changes; build will refuse")
    if new:
        _show(f"NOTE      {len(new)} untracked files are in no commit and will not be copied")
    _show("PLAN ONLY: nothing was written")
    return 0


def build(cfg: Any, dest: Path) -> int:
    changed = tracked_changes(cfg.root)
    if changed:
        raise ConfigError(
            f"tracked files have uncommitted changes ({_some(changed)}): an export is of a "
            "commit. Commit or put them aside, then build again"
        )
    expected = _expected(cfg)
    result = write_copy(dest, expected, cfg, source_root=cfg.root, clean=True)
    _show(f"WROTE     {result.written:,} files to {dest} (commit {expected.info.short})")
    if result.removed:
        names = _some(list(result.removed))
        _show(f"REMOVED   {len(result.removed)} files of the previous export: {names}")
    _show(f"next: python -m scripts.public_export check --to {dest}")
    return 0


def check(cfg: Any, dest: Path, checks: Any = run_checks) -> int:
    record = checks(cfg, dest)
    for item in record["checks"]:
        _show(f"{PASS if item['passed'] else FAIL}  {item['id']}  {item['name']}: {item['detail']}")
    failed = sum(not item["passed"] for item in record["checks"])
    _show(f"{len(record['checks']) - failed} passed, {failed} failed")
    for path in write_record(cfg, record):
        _show(f"WROTE  {path.relative_to(cfg.root).as_posix()}")
    if not failed:
        _show("the folder is ready for the owner to commit and push (runbook section 12)")
    return 1 if failed else 0


def main(argv: list[str] | None = None, cfg: Any = None, checks: Any = run_checks) -> int:
    """Run one command. ``cfg`` and ``checks`` are stand-ins for tests."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.public_export",
        description="Make and check the public copy of this repository.",
    )
    parser.add_argument("command", choices=("plan", "build", "check"))
    parser.add_argument("--to", metavar="FOLDER", help="the folder of the public copy")
    args = parser.parse_args(argv)

    setup_logging()
    try:
        cfg = cfg or load_export_config()
        if args.command == "plan":
            return plan(cfg)
        dest = cfg.destination(args.to)
        if args.command == "build":
            return build(cfg, dest)
        return check(cfg, dest, checks)
    except ConfigError as exc:
        _show(f"CONFIG ERROR: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
