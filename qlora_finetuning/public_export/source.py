"""The files of a commit, read with ``git archive`` (req_006 v6.04, spec D1).

An export is of a commit, never of the working folder: what is published can then be
named, and nothing that git ignores can be copied by accident.

Usage::

    from qlora_finetuning.public_export.source import commit_files, commit_info, tracked_changes

    info = commit_info(root)                 # the commit HEAD names: hash, short hash, day
    files = commit_files(root, info.sha)     # {"path/in/repo": bytes}
    tracked_changes(root)                    # tracked files with uncommitted changes

Only ``git`` commands that read are run here: ``rev-parse``, ``log``, ``status``, ``archive``.
"""

from __future__ import annotations

import io
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

from qlora_finetuning.settings import ConfigError

HEAD = "HEAD"
UNTRACKED = "??"


@dataclass(frozen=True)
class CommitInfo:
    """Which commit is exported."""

    sha: str
    short: str
    day: str  # the day it was committed, YYYY-MM-DD


def git(root: Path | str, *args: str) -> bytes:
    """Run a reading ``git`` command in ``root``. A failure is a ``ConfigError``."""
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, check=False
        )
    except FileNotFoundError:
        raise ConfigError("git is not installed or not on the path") from None
    if done.returncode != 0:
        first = (done.stderr.decode("utf-8", "replace").strip().splitlines() or ["no message"])[0]
        raise ConfigError(f"git {args[0]} failed in {root}: {first}")
    return done.stdout


def commit_info(root: Path | str, commit: str = HEAD) -> CommitInfo:
    """The hash and the day of ``commit``. Raises ``ConfigError`` if it is not a commit."""
    lines = git(root, "log", "-1", "--format=%H%n%h%n%cs", commit).decode().split()
    if len(lines) != 3:
        raise ConfigError(f"{commit!r} is not a commit of {root}")
    return CommitInfo(sha=lines[0], short=lines[1], day=lines[2])


def _status(root: Path | str) -> list[tuple[str, str]]:
    entries = git(root, "status", "--porcelain", "-z").decode("utf-8", "replace").split("\0")
    found, skip = [], False
    for entry in entries:
        if skip or len(entry) < 4:  # the second path of a rename, or the empty end
            skip = False
            continue
        code, path = entry[:2], entry[3:]
        skip = code[0] in "RC"
        found.append((code, path))
    return found


def tracked_changes(root: Path | str) -> list[str]:
    """Tracked files with a change that is not committed, staged or not."""
    return [path for code, path in _status(root) if code != UNTRACKED]


def untracked(root: Path | str) -> list[str]:
    """Files git does not track and does not ignore. They are in no commit, so in no export."""
    return [path for code, path in _status(root) if code == UNTRACKED]


def commit_files(root: Path | str, treeish: str) -> dict[str, bytes]:
    """Every file of a commit (or a tree), by its path with forward slashes."""
    # Text files are stored with Unix line ends; ask for them so, whatever this machine's
    # git would check out. Files the attributes mark as CRLF (Windows scripts) keep theirs.
    archive = git(
        root, "-c", "core.autocrlf=false", "-c", "core.eol=lf", "archive", "--format=tar", treeish
    )
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            if not member.isfile():
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            files[member.name] = handle.read()
    return files
