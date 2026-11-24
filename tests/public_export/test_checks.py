"""Checks E1 to E10, the snapshot marker, the script, and what the export code may not do."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import qlora_finetuning.public_export as export_package
from qlora_finetuning.public_export.build import MANIFEST_NAME, write_copy
from qlora_finetuning.public_export.checks import (
    SuiteResult,
    email_domains,
    private_strings,
    render_markdown,
    run_checks,
    scan_copy,
    write_record,
)
from qlora_finetuning.public_export.snapshot import (
    MARKER,
    NOTICE_NAME,
    REASON,
    is_snapshot,
    skip_private_records,
)
from scripts.public_export import main

from .helpers import ACCOUNT, NEW_RUNBOOK, PRIVATE_NAME, commit, expected_of, make_repo, write_files

PACKAGE_DIR = Path(export_package.__file__).parent
MODULES = sorted(PACKAGE_DIR.glob("*.py"))


def good_suite(dest, cfg):
    return SuiteResult(True, "All checks passed!", 120, 0, 3, ("tests/test_freeze.py",), 3,
                       "120 passed, 3 skipped")  # fmt: skip


def exported(tmp_path, **changes):
    cfg = make_repo(tmp_path / "repo", **changes)
    dest = tmp_path / "public"
    write_copy(dest, expected_of(cfg), cfg, source_root=cfg.root, clean=True)
    return cfg, dest


def by_id(record):
    return {check["id"]: check for check in record["checks"]}


# --------------------------------------------------------------------------- the checks


def test_a_fresh_export_passes_every_check(tmp_path):
    cfg, dest = exported(tmp_path)
    record = run_checks(cfg, dest, suite=good_suite)
    assert [check["id"] for check in record["checks"]] == [f"E{n}" for n in range(1, 11)]
    assert [check["passed"] for check in record["checks"]] == [True] * 10 and record["passed"]
    assert record["commit"]["sha"] == expected_of(cfg).info.sha
    assert record["files"]["moved"] == {NEW_RUNBOOK: "administration" + "/runbook.md"}
    assert record["files"]["not_text"] == ["image.bin"]
    assert record["replacements"][0] == {
        "name": "the AWS account number", "files": 4, "occurrences": 5,
    }  # fmt: skip
    assert record["suite"]["skipped_by_notice"] == 3
    assert ACCOUNT not in json.dumps(record) and PRIVATE_NAME not in json.dumps(record)


def test_a_folder_that_is_not_an_export_fails_the_first_check_only(tmp_path):
    cfg = make_repo(tmp_path / "repo")
    record = run_checks(cfg, tmp_path / "empty", suite=good_suite)
    assert [check["id"] for check in record["checks"]] == ["E1"]
    assert not record["passed"] and "is not an export" in record["checks"][0]["detail"]
    assert "commit" not in record


def test_a_build_from_a_tree_with_changes_or_another_commit_fails_e1(tmp_path):
    cfg = make_repo(tmp_path / "repo")
    dest = tmp_path / "public"
    write_copy(dest, expected_of(cfg), cfg, source_root=cfg.root, clean=False)
    assert "uncommitted changes" in by_id(run_checks(cfg, dest, suite=good_suite))["E1"]["detail"]
    manifest = json.loads((dest / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest.update(commit="0" * 40, clean=True)
    (dest / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    detail = by_id(run_checks(cfg, dest, suite=good_suite))["E1"]["detail"]
    assert "does not name the commit" in detail


def test_private_strings_are_found_and_invented_ones_are_set_aside(tmp_path):
    cfg, dest = exported(tmp_path)
    key = "AKIA" + "REALLOOKINGKEY00"
    planted = {
        "a.txt": f"account {ACCOUNT} again",
        "b.txt": f"key {key}",
        "c.txt": "write to owner@" + "mail.test",
        "d.txt": "saved under C:" + "\\Users\\me",
        "e.txt": f"the repository {PRIVATE_NAME}",
        "fine.txt": "someone@example.com, AKIA" + "INVENTEDVALUE000, " + f"{ACCOUNT}9",
    }
    write_files(dest, planted)
    found = scan_copy(dest, cfg)
    assert found == [
        "a.txt holds the AWS account number",
        "b.txt holds an AWS access key",
        "c.txt holds an email address",
        "d.txt holds a path on someone's machine",
        "e.txt holds the private repository's name",
    ]
    record = run_checks(cfg, dest, suite=good_suite)
    assert not by_id(record)["E3"]["passed"] and not by_id(record)["E2"]["passed"]
    assert private_strings("nothing private", cfg, cfg.rules()) == []
    # A decorator on the line after another is not an address, and a long run of letters
    # with no "@" costs nothing.
    decorators = "\n".join(["@pytest.mark.one", "@pytest.mark.two(x)", ""])
    assert email_domains(decorators + "x" * 2_000_000) == set()
    assert email_domains("to a.b@Example.org, and c@d") == {"example.org"}


def test_pointers_the_notice_and_file_kinds_are_checked(tmp_path):
    cfg, dest = exported(tmp_path)
    (dest / NEW_RUNBOOK).unlink()
    readme = dest / "README.md"
    text = readme.read_text(encoding="utf-8").replace(NOTICE_NAME, "the notice")
    readme.write_text(text + "\nSee [gone](docs/gone.md).\n", encoding="utf-8", newline="\n")
    (dest / "scan.pdf").write_bytes(b"%PDF-")
    (dest / "big.txt").write_bytes(b"x" * (1024 * 1024 + 1))
    checks = by_id(run_checks(cfg, dest, suite=good_suite))
    assert f"{NEW_RUNBOOK} is missing" in checks["E5"]["detail"]
    assert "links to docs/gone.md" in checks["E5"]["detail"]
    assert "does not point to" in checks["E6"]["detail"]
    assert "scan.pdf is a kind of file that is not published" in checks["E10"]["detail"]
    assert "big.txt is larger than 1 MB" in checks["E10"]["detail"]


def test_a_failing_suite_or_an_unexplained_skip_fails_its_check(tmp_path):
    cfg, dest = exported(tmp_path)

    def failing(dest, cfg):
        return SuiteResult(False, "Found 2 errors.", 118, 2, 3, (), 3, "2 failed, 118 passed")

    checks = by_id(run_checks(cfg, dest, suite=failing))
    assert "lint failed in the copy: Found 2 errors." in checks["E7"]["detail"]
    assert "2 failed, 118 passed" in checks["E7"]["detail"] and checks["E8"]["passed"]

    def unmarked(dest, cfg):
        return SuiteResult(True, "ok", 120, 0, 2, (), 3, "120 passed")

    detail = by_id(run_checks(cfg, dest, suite=unmarked))["E8"]["detail"]
    assert "3 tests carry the marker" in detail and "2 were skipped" in detail

    def nothing_ran(dest, cfg):
        return SuiteResult(True, "ok", 0, 0, 0, (), 0, "no totals line")

    assert not by_id(run_checks(cfg, dest, suite=nothing_ran))["E7"]["passed"]


def test_uncommitted_changes_here_fail_e9_but_the_record_does_not(tmp_path):
    cfg, dest = exported(tmp_path)
    write_record(cfg, run_checks(cfg, dest, suite=good_suite))
    commit(cfg.root, "the record")  # reports/public_export/ is left out of a copy
    record = run_checks(cfg, dest, suite=good_suite)
    write_record(cfg, {**record, "passed": False})
    assert by_id(run_checks(cfg, dest, suite=good_suite))["E9"]["passed"]
    write_files(cfg.root, {"LICENSE": "changed\n"})
    assert "LICENSE" in by_id(run_checks(cfg, dest, suite=good_suite))["E9"]["detail"]


def test_the_record_is_written_and_reads_as_a_page(tmp_path):
    cfg, dest = exported(tmp_path)
    record = run_checks(cfg, dest, suite=good_suite)
    paths = write_record(cfg, record)
    assert [path.name for path in paths] == ["export_record.json", "export_record.md"]
    assert json.loads(paths[0].read_text(encoding="utf-8")) == record
    page = render_markdown(record)
    assert "| E7 | Lint and tests | passed |" in page and "all passed" in page
    assert "Do not edit by hand" in page and ACCOUNT not in page and "`.claude/`" in page
    assert paths[1].read_text(encoding="utf-8") == page and b"\r\n" not in paths[1].read_bytes()


# --------------------------------------------------------------------------- the marker


class Item:
    def __init__(self, marked):
        self.marked, self.added = marked, []

    def get_closest_marker(self, name):
        return object() if self.marked and name == MARKER else None

    def add_marker(self, marker):
        self.added.append(marker)


def test_marked_tests_are_skipped_only_in_a_snapshot(tmp_path):
    items = [Item(True), Item(False), Item(True)]
    assert not is_snapshot(tmp_path) and skip_private_records(items, tmp_path, "skip") == 0
    assert all(item.added == [] for item in items)
    (tmp_path / NOTICE_NAME).write_text("# Public snapshot\n", encoding="utf-8")
    assert is_snapshot(tmp_path) and skip_private_records(items, tmp_path, "skip") == 2
    assert [item.added for item in items] == [["skip"], [], ["skip"]]
    assert NOTICE_NAME in REASON and "redacted" in REASON


def test_the_marker_is_registered_and_used_by_the_suite():
    root = Path(__file__).resolve().parents[2]
    assert f"{MARKER}:" in (root / "pytest.ini").read_text(encoding="utf-8")
    conftest = (root / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "skip_private_records" in conftest and "pytest_collection_modifyitems" in conftest


# --------------------------------------------------------------------------- the script


def run(tmp_path, argv, **options):
    cfg = options.pop("cfg", None) or make_repo(tmp_path / "repo")
    return main(argv, cfg=cfg, **options), cfg


def test_plan_prints_what_would_be_copied_and_writes_nothing(tmp_path, capsys):
    code, cfg = run(tmp_path, ["plan"])
    out = capsys.readouterr().out
    assert code == 0 and out.isascii() and "PLAN ONLY: nothing was written" in out
    assert "COPY      11 files" in out and "LEAVE OUT 3 files" in out
    assert f"MOVE      {'administration'}/runbook.md -> {NEW_RUNBOOK}" in out
    assert "REPLACE   the AWS account number: 5 in 4 files" in out
    assert "AS IT IS  1 files that are not text: image.bin" in out
    assert ACCOUNT not in out and not (tmp_path / "public").exists()


def test_build_refuses_uncommitted_changes_and_then_writes(tmp_path, capsys):
    cfg = make_repo(tmp_path / "repo")
    dest = tmp_path / "public"
    write_files(cfg.root, {"LICENSE": "changed\n", "scratch.txt": "untracked\n"})
    assert main(["build", "--to", str(dest)], cfg=cfg) == 2
    out = capsys.readouterr().out
    assert "CONFIG ERROR: tracked files have uncommitted changes (LICENSE)" in out
    assert not dest.exists()
    write_files(cfg.root, {"LICENSE": "A licence.\n"})  # the untracked file alone does not stop it
    assert main(["build", "--to", str(dest)], cfg=cfg) == 0
    out = capsys.readouterr().out
    assert "WROTE     12 files" in out and "next: python -m scripts.public_export check" in out
    assert not (dest / "scratch.txt").exists() and (dest / NOTICE_NAME).is_file()
    assert main(["plan"], cfg=cfg) == 0
    assert "NOTE      1 untracked files" in capsys.readouterr().out


def test_check_prints_the_checks_writes_the_record_and_says_so(tmp_path, capsys):
    cfg, dest = exported(tmp_path)
    checks = lambda cfg, dest: run_checks(cfg, dest, suite=good_suite)  # noqa: E731
    assert main(["check", "--to", str(dest)], cfg=cfg, checks=checks) == 0
    out = capsys.readouterr().out
    assert "PASS  E10  Size and kinds" in out and "10 passed, 0 failed" in out
    assert "WROTE  reports/public_export/export_record.json" in out
    assert "ready for the owner to commit and push" in out and out.isascii()
    (dest / "LICENSE").write_text("edited\n", encoding="utf-8")
    assert main(["check", "--to", str(dest)], cfg=cfg, checks=checks) == 1
    out = capsys.readouterr().out
    assert "FAIL  E2  File set: 1 file(s) changed since the export: LICENSE" in out
    assert "ready for the owner" not in out and cfg.record_json.is_file()


def test_a_missing_destination_or_a_refused_folder_is_exit_code_2(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("QLORA_PUBLIC_EXPORT_DIR", "")
    code, cfg = run(tmp_path, ["build"])
    assert code == 2 and "CONFIG ERROR: no destination" in capsys.readouterr().out
    assert main(["build", "--to", str(cfg.root / "inside")], cfg=cfg) == 2
    assert "outside this repository" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["push"], cfg=cfg)


# --------------------------------------------------------------------------- guards


def strings_and_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports, strings = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.add(node.value)
    return imports, strings


SOURCES = [*MODULES, PACKAGE_DIR.parents[1] / "scripts" / "public_export.py"]


@pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
def test_the_export_code_never_commits_pushes_or_reaches_the_network(path):
    imports, strings = strings_and_imports(path)
    assert not imports & {
        "boto3", "botocore", "requests", "urllib", "http", "socket", "httpx", "huggingface_hub",
        "torch", "transformers",
    }  # fmt: skip
    assert not strings & {"push", "gh", "remote", "clone", "fetch", "pull"}
    if path.name == "source.py":  # the one module that runs git on a real repository
        assert not strings & {"commit", "init", "add"}


def test_git_is_only_asked_to_read():
    _, strings = strings_and_imports(PACKAGE_DIR / "source.py")
    asked = {"log", "status", "archive"} & strings
    assert asked == {"log", "status", "archive"}
    other = [
        path.name
        for path in MODULES
        if path.name not in ("source.py", "checks.py")
        and "subprocess" in strings_and_imports(path)[0]
    ]
    assert other == []  # only source.py runs git; checks.py runs lint and the suite
