"""Proving a public copy: checks E1 to E10 and the export record (req_006 v6.04).

``run_checks`` reads the copy's folder and this repository, and writes nothing. It
rebuilds, in memory, what the commit named in the copy's notice exports, and compares the
folder with it byte for byte. It scans every text file for what may not remain. It runs
lint and the test suite **in a temporary copy of the folder**, with the training stack
blocked as in CI, so nothing is written into the folder that will be pushed.

Usage::

    from qlora_finetuning.public_export.checks import run_checks, write_record

    record = run_checks(cfg, dest)
    record["passed"], record["checks"]          # E1 to E10
    write_record(cfg, record)                   # reports/public_export/export_record.json, .md

The record holds counts, names of rules, and the exported commit. It holds none of the
values that were replaced.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qlora_finetuning.public_export.build import MANIFEST_NAME, compare, files_at, read_manifest
from qlora_finetuning.public_export.config import ExportConfig, Rule
from qlora_finetuning.public_export.snapshot import MARKER, NOTICE_NAME, REASON
from qlora_finetuning.public_export.source import commit_files, commit_info, tracked_changes
from qlora_finetuning.public_export.transform import (
    COMMAND,
    README,
    Expected,
    as_text,
    expected_files,
    rule_pattern,
)
from qlora_finetuning.settings import ConfigError

SCHEMA = 1
SHOWN = 5
STACK = (
    "torch", "transformers", "datasets", "peft", "trl", "accelerate", "bitsandbytes",
    "tokenizers", "safetensors", "huggingface_hub",
)  # fmt: skip
_LOCAL_PART = re.compile(r"[A-Za-z0-9._%+-]+\Z")  # \Z: "$" would also match before a newline
_DOMAIN = re.compile(r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_LINK = re.compile(r"\]\((?!https?://|#)([^)#\s]+)(?:#[^)]*)?\)")
_COMMIT = re.compile(r"`([0-9a-f]{40})`")
_TOTALS = re.compile(r"(\d+) (passed|failed|errors?|skipped|deselected)")
_SKIPPED = re.compile(r"^SKIPPED \[(\d+)\] (.+?):\d+: (.*)$")


@dataclass(frozen=True)
class SuiteResult:
    """Lint and the test suite, run in a copy of the copy."""

    lint_passed: bool
    lint_detail: str
    passed: int
    failed: int
    skipped_by_notice: int
    skipped_tests: tuple[str, ...]  # the files whose tests the notice skipped
    marked: int  # tests that carry the marker
    summary: str


def check_result(check_id: str, name: str, problems: list[str], ok_detail: str) -> dict[str, Any]:
    return {
        "id": check_id,
        "name": name,
        "passed": not problems,
        "detail": "; ".join(problems) if problems else ok_detail,
    }


# --------------------------------------------------------------------------- the scan


def private_strings(text: str, cfg: ExportConfig, rules: tuple[Rule, ...]) -> list[str]:
    """The kinds of private string in ``text``: a replaced value, a key, a token, an address."""
    for value in cfg.invented_values:  # planted by tests on purpose, and known to be invented
        text = text.replace(value, "")
    found = [rule.name for rule in rules if rule_pattern(rule).search(text)]
    found += [name for name, pattern in cfg.may_not_remain if pattern.search(text)]
    if email_domains(text) - set(cfg.allowed_email_domains):
        found.append("an email address")
    return found


def email_domains(text: str) -> set[str]:
    """The domains of the email addresses in ``text``.

    Each ``@`` is looked at with a little text on either side, so a long file without one
    costs nothing and a long run of letters cannot stall the scan.
    """
    domains = set()
    position = text.find("@")
    while position != -1:
        before = _LOCAL_PART.search(text[max(0, position - 64) : position])
        after = _DOMAIN.match(text[position + 1 : position + 256])
        if before and after:
            domains.add(after.group(0).lower())
        position = text.find("@", position + 1)
    return domains


def scan_copy(dest: Path, cfg: ExportConfig) -> list[str]:
    """Every text file of the copy that holds something private (check E3)."""
    rules = cfg.rules()
    problems = []
    for relative, path in sorted(files_at(dest, cfg).items()):
        text = as_text(path.read_bytes())
        if text is None:
            continue
        problems += [f"{relative} holds {name}" for name in private_strings(text, cfg, rules)]
    return problems


# --------------------------------------------------------------------------- the suite


def _blocked_stack(folder: Path) -> Path:
    """A folder of stub packages that make the training stack unimportable, as in CI."""
    for name in STACK:
        package = folder / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            f"raise ModuleNotFoundError(\"No module named '{name}' (blocked for the check)\", "
            f"name='{name}')\n",
            encoding="utf-8",
        )
    return folder


def _as_repository(work: Path) -> None:
    """Make the scratch copy a git repository with one commit, as a checkout in CI is.

    Some tests read "this repository's commit". This is a throwaway repository in a
    temporary folder: nothing is committed in the copy's own folder or in this repository.
    """
    identity = ["-c", "user.name=check", "-c", "user.email=check@example.com"]
    for args in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "-m", "check"]):
        subprocess.run(
            ["git", "-C", str(work), *identity, *args], capture_output=True, check=False
        )


def _run(args: list[str], cwd: Path, env: dict[str, str], timeout: int) -> tuple[int, str]:
    try:
        done = subprocess.run(
            [sys.executable, "-m", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 1, f"did not finish in {timeout} seconds"
    return done.returncode, (done.stdout + done.stderr).strip()


def run_suite(dest: Path, cfg: ExportConfig) -> SuiteResult:
    """Lint and the test suite in a temporary copy of ``dest``, the training stack blocked."""
    with tempfile.TemporaryDirectory(prefix="public-export-check-") as scratch:
        work, stubs = Path(scratch) / "copy", _blocked_stack(Path(scratch) / "blocked")
        shutil.copytree(
            dest, work, ignore=shutil.ignore_patterns(*cfg.ignored_at_destination, MANIFEST_NAME)
        )
        _as_repository(work)
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        env.update(PYTHONPATH=str(stubs), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
        timeout = cfg.suite_timeout_seconds

        code, output = _run(["ruff", "check", ".", "--no-cache"], work, env, timeout)
        lint_detail = (output.splitlines() or ["no output"])[-1]
        _, collected = _run(
            ["pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "-m",
             f"{MARKER} and not network and not gpu and not paid"],
            work, env, timeout,
        )  # fmt: skip
        marked = sum(1 for line in collected.splitlines() if "::" in line)
        _, output = _run(["pytest", "-q", "-rs", "-p", "no:cacheprovider"], work, env, timeout)

    lines = output.splitlines()
    summary = next((line for line in reversed(lines) if _TOTALS.search(line)), "no totals line")
    totals = {kind.rstrip("s"): int(number) for number, kind in _TOTALS.findall(summary)}
    by_notice, files = 0, set()
    for line in lines:
        match = _SKIPPED.match(line)
        if match and match.group(3).startswith(REASON[:40]):
            by_notice += int(match.group(1))
            files.add(match.group(2).replace("\\", "/"))
    return SuiteResult(
        lint_passed=code == 0,
        lint_detail=lint_detail,
        passed=totals.get("passed", 0),
        failed=totals.get("failed", 0) + totals.get("error", 0),
        skipped_by_notice=by_notice,
        skipped_tests=tuple(sorted(files)),
        marked=marked,
        summary=summary.strip("= "),
    )


# --------------------------------------------------------------------------- the checks


def _expected_for(cfg: ExportConfig, dest: Path) -> tuple[Expected | None, list[str]]:
    """What the commit named in the copy's notice exports, or why that cannot be said."""
    notice = dest / NOTICE_NAME
    if not notice.is_file():
        return None, [f"{dest} has no {NOTICE_NAME}: it is not an export; run {COMMAND} build"]
    match = _COMMIT.search(notice.read_text(encoding="utf-8"))
    if match is None:
        return None, [f"the {NOTICE_NAME} of the copy names no commit"]
    try:
        info = commit_info(cfg.root, match.group(1))
        return expected_files(commit_files(cfg.root, info.sha), cfg, info), []
    except ConfigError as exc:
        return None, [f"the commit the copy names is not in this repository ({exc})"]


def run_checks(
    cfg: ExportConfig,
    dest: Path,
    *,
    suite: Callable[[Path, ExportConfig], SuiteResult] = run_suite,
) -> dict[str, Any]:
    """Checks E1 to E10 on the copy in ``dest``, as a record. Writes nothing.

    ``suite`` is a stand-in for the lint-and-test run in tests.
    """
    expected, problems = _expected_for(cfg, dest)
    manifest = read_manifest(dest) if dest.is_dir() else None
    present = files_at(dest, cfg)
    checks = []

    e1 = list(problems)
    if expected is not None:
        if manifest is None or manifest.get("commit") != expected.info.sha:
            e1.append("the copy's manifest does not name the commit of its notice")
        elif not manifest.get("clean"):
            e1.append("the copy was built while tracked files had uncommitted changes")
    checks.append(
        check_result(
            "E1",
            "Source",
            e1,
            "" if expected is None else f"commit {expected.info.short} of {expected.info.day}, "
            "exported from a clean tree",
        )
    )
    if expected is None:
        return _record(cfg, None, checks, None)

    plan = expected.plan
    checks.append(
        check_result(
            "E2",
            "File set",
            compare(dest, expected, cfg),
            f"{len(plan.kept):,} files copied, {len(plan.left_out):,} left out, "
            f"{len(expected.generated)} generated; the folder is exactly the export of its commit",
        )
    )
    checks.append(
        check_result(
            "E3",
            "Nothing private",
            scan_copy(dest, cfg),
            "no file holds a replaced value, an access key, a token, an address outside the "
            "allowed domains, or a path on someone's machine",
        )
    )
    unused = [
        f"{name} was replaced nowhere" for name, count in expected.counts.items()
        if not count["occurrences"]
    ]
    made = "; ".join(
        f"{name}: {count['occurrences']:,} in {count['files']:,} files"
        for name, count in expected.counts.items()
    )
    checks.append(check_result("E4", "Replacements", unused, made))

    e5 = []
    for _, target in cfg.moved:
        if target not in present:
            e5.append(f"{target} is missing")
    readme = present.get(README)
    if readme is not None:
        text = readme.read_text(encoding="utf-8")
        e5 += [
            f"the README links to {link}, which is not in the copy"
            for link in sorted(set(_LINK.findall(text)))
            if not (dest / link).exists()
        ]
    checks.append(
        check_result(
            "E5",
            "Pointers",
            e5,
            "the moved files are in place and every link of the README leads to a file of the copy",
        )
    )

    e6 = []
    if readme is None or NOTICE_NAME not in readme.read_text(encoding="utf-8"):
        e6.append(f"the README does not point to {NOTICE_NAME}")
    checks.append(
        check_result(
            "E6",
            "The notice",
            e6,
            f"{NOTICE_NAME} names commit {expected.info.short}; the README points to it",
        )
    )

    result = suite(dest, cfg)
    e7 = []
    if not result.lint_passed:
        e7.append(f"lint failed in the copy: {result.lint_detail}")
    if result.failed or not result.passed:
        e7.append(f"the suite did not pass in the copy: {result.summary}")
    checks.append(
        check_result("E7", "Lint and tests", e7, f"lint clean; {result.summary}, in the copy")
    )
    e8 = []
    if result.skipped_by_notice != result.marked:
        e8.append(
            f"{result.marked} tests carry the marker {MARKER} and {result.skipped_by_notice} "
            "were skipped by the notice"
        )
    checks.append(
        check_result(
            "E8",
            "Skipped tests",
            e8,
            f"{result.skipped_by_notice} tests skipped by the notice, all of them marked {MARKER}",
        )
    )

    changed = [
        path for path in tracked_changes(cfg.root)
        if not path.startswith(cfg.report_dir.relative_to(cfg.root).as_posix() + "/")
    ]
    checks.append(
        check_result(
            "E9",
            "The private repository is untouched",
            [f"tracked files here have uncommitted changes: {', '.join(changed[:SHOWN])}"]
            if changed
            else [],
            "no tracked file of this repository has an uncommitted change, the export record aside",
        )
    )

    e10 = []
    for relative, path in sorted(present.items()):
        if path.suffix.lower() in cfg.forbidden_suffixes:
            e10.append(f"{relative} is a kind of file that is not published")
        elif path.stat().st_size > cfg.max_file_bytes:
            e10.append(f"{relative} is larger than {cfg.max_file_bytes // (1024 * 1024)} MB")
    total = sum(path.stat().st_size for path in present.values())
    checks.append(
        check_result(
            "E10",
            "Size and kinds",
            e10,
            f"{len(present):,} files, {total / 1_000_000:.1f} MB; no file over the limit, no "
            "archive, document scan, or weight file",
        )
    )
    return _record(cfg, expected, checks, result)


def _record(
    cfg: ExportConfig,
    expected: Expected | None,
    checks: list[dict[str, Any]],
    result: SuiteResult | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_by": f"{COMMAND} check",
        "passed": all(check["passed"] for check in checks),
        "public_repository": cfg.public_repository,
        "checks": checks,
    }
    if expected is not None:
        record.update(
            commit={
                "sha": expected.info.sha,
                "short": expected.info.short,
                "day": expected.info.day,
            },
            files={
                "copied": len(expected.plan.kept),
                "left_out": len(expected.plan.left_out),
                "generated": list(expected.generated),
                "moved": expected.plan.moved,
                "not_text": list(expected.binary),
            },
            left_out=list(cfg.left_out),
            replacements=[
                {"name": name, "files": count["files"], "occurrences": count["occurrences"]}
                for name, count in expected.counts.items()
            ],
        )
    if result is not None:
        record["suite"] = {
            "summary": result.summary,
            "passed": result.passed,
            "failed": result.failed,
            "marked": result.marked,
            "skipped_by_notice": result.skipped_by_notice,
            "skipped_files": list(result.skipped_tests),
        }
    return record


# --------------------------------------------------------------------------- the record


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def render_markdown(record: dict[str, Any]) -> str:
    """The record for reading."""
    lines = [
        "# The public export",
        "",
        f"Written by `{record['generated_by']}`. Do not edit by hand: check again. This file "
        "names rules and counts; it holds none of the values that were replaced.",
        "",
        f"- **Public repository**: `{record['public_repository']}`",
    ]
    commit = record.get("commit")
    if commit:
        lines.append(f"- **Exported commit**: `{commit['sha']}`, committed {commit['day']}")
    lines += [
        f"- **Checks**: {'all passed' if record['passed'] else '**NOT PASSED**'}",
        "",
        "The export tool writes a folder and nothing else. Committing in that folder, pushing, "
        "and making the repository public are the owner's steps.",
        "",
        "## Checks",
        "",
        _table(
            ["Check", "", "Result", "Detail"],
            [
                [check["id"], check["name"], "passed" if check["passed"] else "**FAILED**",
                 check["detail"]]
                for check in record["checks"]
            ],
        ),
        "",
    ]
    if "files" in record:
        files = record["files"]
        lines += [
            "## What was copied",
            "",
            f"{files['copied']:,} files copied, {files['left_out']:,} left out, "
            f"{len(files['generated'])} generated ({', '.join(files['generated'])}).",
            "",
            _table(
                ["Moved from", "To"],
                [[source, target] for target, source in files["moved"].items()],
            ),
            "",
            "Left out:",
            "",
            *[f"- `{entry}`" for entry in record["left_out"]],
            "",
            "## What was replaced",
            "",
            _table(
                ["Rule", "Files", "Occurrences"],
                [[rule["name"], f"{rule['files']:,}", f"{rule['occurrences']:,}"]
                 for rule in record["replacements"]],
            ),
            "",
        ]
    if "suite" in record:
        suite = record["suite"]
        lines += [
            "## The suite in the copy",
            "",
            f"{suite['summary']}. {suite['skipped_by_notice']} tests were skipped by the "
            f"snapshot notice; {suite['marked']} carry the marker `{MARKER}`.",
            "",
            *[f"- `{name}`" for name in suite["skipped_files"]],
            "",
        ]
    return "\n".join(lines)


def write_record(cfg: ExportConfig, record: dict[str, Any]) -> list[Path]:
    """Write the record's two files in this repository. Returns their paths."""
    import json

    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    cfg.record_json.write_text(
        json.dumps(record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n"
    )
    cfg.record_markdown.write_text(render_markdown(record), encoding="utf-8", newline="\n")
    return [cfg.record_json, cfg.record_markdown]
