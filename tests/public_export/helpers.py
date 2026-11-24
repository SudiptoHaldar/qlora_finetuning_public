"""A stand-in repository for the export tests: a few files, one commit, its own config.

Everything is under ``tmp_path``. The stand-in account number and the private strings are
invented. Values that a scan must find are written in pieces, so that this file can itself
be published without being mistaken for something private.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from qlora_finetuning.public_export.config import ExportConfig, load_export_config
from qlora_finetuning.public_export.source import commit_files, commit_info
from qlora_finetuning.public_export.transform import Expected, expected_files

ACCOUNT = "4444" + "5555" + "6666"  # the stand-in repository's "real" account number
PLACEHOLDER = "123456789012"
PRIVATE_NAME = "private-org/the-project"
PUBLIC_NAME = "someone/the-project-public"
# In pieces: an export of this repository rewrites the old path wherever it is written out.
OLD_RUNBOOK, NEW_RUNBOOK = "administration" + "/runbook.md", "docs/runbook.md"
IDENTITY = ("-c", "user.name=Tester", "-c", "user.email=tester@example.com")

FILES = {
    "README.md": f"# The project\n\nSee [{OLD_RUNBOOK}]({OLD_RUNBOOK}) and [LICENSE](LICENSE).\n",
    "LICENSE": "A licence.\n",
    ".gitignore": "__pycache__/\n.public_export_manifest.json\n",
    "config/aws.yaml": f'version: 1\naccount_id: "{ACCOUNT}"\nbucket: data-{ACCOUNT}\n',
    "src/code.py": (
        f'HINT = "see {OLD_RUNBOOK} section 3"\n'
        f'ROLE = "arn:aws:iam::{ACCOUNT}:role/x"\n'
    ),
    "tests/test_code.py": f'def test_hint():\n    assert "{OLD_RUNBOOK}" in "see {OLD_RUNBOOK}"\n',
    "reports/record.json": (
        f'{{"arn": "arn:aws:bedrock:us-east-1:{ACCOUNT}:custom-model/m", '
        f'"n": {ACCOUNT}9}}\n'
    ),
    "notes/plan.md": f"The repository is `{PRIVATE_NAME}`, private.\n",
    OLD_RUNBOOK: f"# Runbook\n\n## 1. AWS\n\nAccount {ACCOUNT}.\n",
    "administration/commands.md": f"aws s3 ls s3://data-{ACCOUNT}\n",
    ".claude/settings.json": "{}\n",
    "track_progress/wip.md": "work in progress\n",
    "image.bin": b"\x00\x01\x02 not text " + ACCOUNT.encode(),
}
CONFIG = {
    "version": 1,
    "report_dir": "reports/public_export",
    "public_repository": PUBLIC_NAME,
    "left_out": [".claude/", "track_progress/", "administration/", "reports/public_export/"],
    "moved": [{"from": OLD_RUNBOOK, "to": NEW_RUNBOOK}],
    "replacements": [
        {"name": "the AWS account number", "from_config": "aws.account_id", "to": PLACEHOLDER},
        {"name": "the private repository's name", "from": PRIVATE_NAME, "to": PUBLIC_NAME},
    ],
    "path_rewrites": [{"from": OLD_RUNBOOK, "to": NEW_RUNBOOK}],
    "may_not_remain": [
        {"name": "an AWS access key", "pattern": "(?<![0-9A-Z])A[KS]IA[0-9A-Z]{16}(?![0-9A-Z])"},
        {
            "name": "a path on someone's machine",
            "pattern": r"(?i)(?<![a-z])[a-z]:\\+users(?![a-z])",
        },
    ],
    "invented_values": ["AKIA" + "INVENTEDVALUE000"],
    "allowed_email_domains": ["example.com"],
    "ignored_at_destination": [".git", "venv", "__pycache__", ".pytest_cache"],
    "limits": {"max_file_mb": 1, "forbidden_suffixes": [".pdf", ".safetensors"]},
    "suite_timeout_minutes": 5,
}


def git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), *IDENTITY, *args], capture_output=True, text=True, check=True
    )
    return done.stdout


def write_files(root: Path, files: dict[str, str | bytes]) -> None:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8", newline="\n")


def make_repo(root: Path, **changes: object) -> ExportConfig:
    """A committed stand-in repository under ``root`` and its export config."""
    root.mkdir(parents=True, exist_ok=True)
    config = {**CONFIG, **changes}
    write_files(root, FILES)
    write_files(root, {"config/public_export.yaml": yaml.safe_dump(config, sort_keys=False)})
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "core.autocrlf", "false")
    commit(root, "first")
    return load_export_config(root=root, config_dir=root / "config")


def commit(root: Path, message: str) -> None:
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)


def expected_of(cfg: ExportConfig) -> Expected:
    info = commit_info(cfg.root)
    return expected_files(commit_files(cfg.root, info.sha), cfg, info)
