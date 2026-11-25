"""Publish the two packages on the Hugging Face Hub, and check what is there (req_006 v6.03).

The packages are the folders that ``scripts.hf_package`` built and checked: arm A1's
adapter, and the merged copy. Each goes to a **private** repository. No command here makes
a repository public: the owner does that on the Hub, after reading the rendered card.

Usage (from the repo root)::

    venv\\Scripts\\python -m scripts.hf_publish plan                   # no network, no token
    venv\\Scripts\\python -m scripts.hf_publish upload adapter --yes   # THE OWNER
    venv\\Scripts\\python -m scripts.hf_publish check adapter
    venv\\Scripts\\python -m scripts.hf_publish upload merged --yes    # THE OWNER: 8 GB
    venv\\Scripts\\python -m scripts.hf_publish check merged
    venv\\Scripts\\python -m scripts.hf_publish check                  # both

``upload`` creates the repository as private if it is not there, and uploads the folder
only after every file on disk has the size and hash the package record holds. A
repository that holds anything else is refused. ``check`` compares the Hub with the
package record by name, size, and hash, runs the card's code by repository name (it needs
the GPU; the merged card's code downloads 8 GB the first time), and writes
``reports/packaging/publish_record.json`` and ``.md``.

**The token.** ``upload``, and ``check`` of a private repository, read ``HF_TOKEN`` from the
environment or from ``.env``. It is never taken from the command line and never printed.
A public repository is checked without any token, so the check sees what the world sees.

**Uploading is the owner's step.** An agent runs ``plan``, and ``check`` once the
repositories are public.

Exit codes: 0 done, and for ``plan`` and ``check`` every check passed; 1 a check failed;
2 the command could not start or was refused (no ``--yes``, no token, a config problem).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from typing import Any

try:
    from qlora_finetuning.logging_setup import setup_logging
    from qlora_finetuning.packaging import publish
    from qlora_finetuning.packaging.config import load_packaging_config
    from qlora_finetuning.packaging.files import read_json
    from qlora_finetuning.packaging.hub import HubClient
    from qlora_finetuning.settings import ConfigError
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.hf_publish"
    ) from None

PASS, FAIL = "PASS", "FAIL"


class Output:
    """Prints ASCII lines with the token's value taken out, whatever a line holds."""

    def __init__(self) -> None:
        self.token: str | None = None

    def __call__(self, text: str) -> None:
        text = publish.redact(text, self.token)
        print(text.encode("ascii", "backslashreplace").decode("ascii"))


def _megabytes(count: int) -> str:
    return f"{count / 1_000_000:,.1f} MB"


def plan(cfg: Any, show: Output) -> int:
    failed = 0
    for package in publish.read_packages(cfg).values():
        problems = publish.local_problems(package)
        failed += bool(problems)
        detail = "; ".join(problems) or (
            f"{len(package.files)} files, {_megabytes(package.total_bytes)}, each with the size "
            "and hash of the package record"
        )
        show(f"{FAIL if problems else PASS}  U1  The {package.kind} package: {detail}")
        show(
            f"WOULD UPLOAD  {package.folder.relative_to(cfg.root).as_posix()}/ -> "
            f"{package.repo_id} (created private)"
        )
    show("PLAN ONLY: nothing was sent and no token was read")
    return 1 if failed else 0


def upload(cfg: Any, kind: str, yes: bool, show: Output, make_client: Any) -> int:
    if not yes:
        show(
            "REFUSED: an upload publishes to the Hugging Face Hub under the owner's name. It is "
            "the owner's step: add --yes"
        )
        return 2
    token = publish.read_token(cfg)
    if not token:
        raise ConfigError(
            f"no token: put {cfg.token_env}=<your write token> in .env (see .env.example). "
            "It is never given on the command line"
        )
    show.token = token
    package = publish.read_packages(cfg)[kind]
    show(f"uploading the {kind} package ({_megabytes(package.total_bytes)}) to {package.repo_id}")
    commit = publish.upload_package(cfg, package, make_client(token))
    short = commit[:12] or "unknown"
    show(f"UPLOADED  {package.repo_id} at commit {short}; the repository is private")
    show(f"next: python -m scripts.hf_publish check {kind}")
    return 0


def check(
    cfg: Any, kinds: tuple[str, ...], show: Output, make_client: Any, card_check: Any, today: date
) -> int:
    packages = publish.read_packages(cfg)
    entries: dict[str, Any] = {}
    if cfg.publish_json.is_file():  # keep what an earlier check found for the other package
        entries = dict(read_json(cfg.publish_json, "the publish record").get("packages") or {})
        for entry in entries.values():
            entry["checks"] = [item for item in entry["checks"] if item["id"] != "U7"]
    token: str | None = None
    for kind in kinds:
        package = packages[kind]
        # Without a token first. The token is read only if the repository cannot be seen.
        if make_client(None).state(package.repo_id) is None:
            token = token or publish.read_token(cfg)
            show.token = token
        entries[kind] = publish.check_package(
            cfg, package, make_client=make_client, token=token, card_check=card_check
        )
    record = publish.build_record(cfg, entries, today, token)
    failed = 0
    for kind in kinds:
        entry = record["packages"][kind]
        show(f"{kind}: {entry['repository']} ({entry['visibility']})")
        for item in entry["checks"]:
            failed += not item["passed"]
            status = PASS if item["passed"] else FAIL
            show(f"{status}  {item['id']}  {item['name']}: {item['detail']}")
    for path in publish.write_record(cfg, record):
        show(f"WROTE  {path.relative_to(cfg.root).as_posix()}")
    if record["published"]:
        show("PUBLISHED: both repositories are public and every check passed")
    elif not failed:
        show("checked; a repository is still private. Making it public is the owner's, on the Hub")
    return 1 if failed else 0


def main(
    argv: list[str] | None = None,
    cfg: Any = None,
    make_client: Any = HubClient,
    card_check: Any = None,
    today: date | None = None,
) -> int:
    """Run one command. The keyword arguments are stand-ins for tests."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.hf_publish",
        description="Publish the two packages on the Hugging Face Hub, and check what is there.",
    )
    parser.add_argument("command", choices=("plan", "upload", "check"))
    parser.add_argument("package", nargs="?", choices=publish.KINDS)
    parser.add_argument("--yes", action="store_true", help="the owner's go-ahead for an upload")
    args = parser.parse_args(argv)
    if args.command == "upload" and args.package is None:
        parser.error("upload needs a package: adapter or merged")

    setup_logging()
    show = Output()
    try:
        cfg = cfg or load_packaging_config()
        if args.command == "plan":
            return plan(cfg, show)
        if args.command == "upload":
            return upload(cfg, args.package, args.yes, show, make_client)
        kinds = (args.package,) if args.package else publish.KINDS
        runner = card_check or (
            lambda kind, online, token: publish.card_runs(cfg, kind, online, token)
        )
        return check(cfg, kinds, show, make_client, runner, today or date.today())
    except ConfigError as exc:
        show(f"CONFIG ERROR: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
