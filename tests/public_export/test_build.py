"""The plan, the changed texts, and writing the copy (req_006 v6.04)."""

from __future__ import annotations

import json

import pytest

from qlora_finetuning.public_export.build import (
    MANIFEST_NAME,
    compare,
    destination_problems,
    write_copy,
)
from qlora_finetuning.public_export.config import Rule, load_export_config
from qlora_finetuning.public_export.snapshot import NOTICE_NAME
from qlora_finetuning.public_export.source import (
    commit_files,
    commit_info,
    tracked_changes,
    untracked,
)
from qlora_finetuning.public_export.transform import apply_rules, make_plan
from qlora_finetuning.settings import ConfigError

from .helpers import (
    ACCOUNT,
    NEW_RUNBOOK,
    OLD_RUNBOOK,
    PLACEHOLDER,
    PRIVATE_NAME,
    PUBLIC_NAME,
    commit,
    expected_of,
    make_repo,
    write_files,
)


def build(cfg, dest, clean=True):
    return write_copy(dest, expected_of(cfg), cfg, source_root=cfg.root, clean=clean)


# --------------------------------------------------------------------------- config and source


def test_the_committed_config_loads():
    cfg = load_export_config()
    assert cfg.public_repository.count("/") == 1
    assert "reports/public_export/" in cfg.left_out and ".claude/" in cfg.left_out
    assert dict(cfg.moved) == {OLD_RUNBOOK: NEW_RUNBOOK} or dict(cfg.moved) == {
        NEW_RUNBOOK: NEW_RUNBOOK
    }  # in a public copy of this repository the old path was itself rewritten
    assert cfg.record_json.name == "export_record.json"
    assert "users.noreply.github.com" in cfg.allowed_email_domains


def test_the_value_to_replace_is_read_from_its_own_config(tmp_path):
    cfg = make_repo(tmp_path / "repo")
    rules = cfg.rules()
    assert rules[0] == Rule("the AWS account number", ACCOUNT, PLACEHOLDER)
    assert rules[1].old == PRIVATE_NAME and rules[-1].old == OLD_RUNBOOK
    same = make_repo(
        tmp_path / "second",
        replacements=[{"name": "a", "from_config": "aws.account_id", "to": ACCOUNT}],
    )
    with pytest.raises(ConfigError, match="changes nothing"):
        same.rules()


def test_the_destination_comes_from_the_command_line_or_the_environment(tmp_path, monkeypatch):
    cfg = make_repo(tmp_path / "repo")
    assert cfg.destination(str(tmp_path / "out")) == tmp_path / "out"
    monkeypatch.setenv("QLORA_PUBLIC_EXPORT_DIR", str(tmp_path / "named"))
    assert cfg.destination(None) == tmp_path / "named"
    monkeypatch.setenv("QLORA_PUBLIC_EXPORT_DIR", "")
    with pytest.raises(ConfigError, match="no destination"):
        cfg.destination(None)
    with pytest.raises(ConfigError, match="full path"):
        cfg.destination("relative/folder")


def test_a_commits_files_are_read_and_changes_in_the_folder_are_not(tmp_path):
    cfg = make_repo(tmp_path / "repo")
    info = commit_info(cfg.root)
    assert len(info.sha) == 40 and info.sha.startswith(info.short) and len(info.day) == 10
    write_files(cfg.root, {"LICENSE": "changed, not committed\n", "new.txt": "untracked\n"})
    files = commit_files(cfg.root, info.sha)
    assert files["LICENSE"] == b"A licence.\n" and "new.txt" not in files
    assert files["image.bin"].startswith(b"\x00\x01")
    assert tracked_changes(cfg.root) == ["LICENSE"] and untracked(cfg.root) == ["new.txt"]
    with pytest.raises(ConfigError, match="not a commit|failed"):
        commit_info(cfg.root, "no-such-branch")


# --------------------------------------------------------------------------- the plan and the texts


def test_the_plan_keeps_moves_and_leaves_out(tmp_path):
    cfg = make_repo(tmp_path / "repo")
    plan = make_plan(commit_files(cfg.root, "HEAD"), cfg)
    assert plan.moved == {NEW_RUNBOOK: OLD_RUNBOOK}
    assert set(plan.left_out) == {
        ".claude/settings.json", "track_progress/wip.md", "administration/commands.md",
    }  # fmt: skip
    assert "src/code.py" in plan.kept and OLD_RUNBOOK not in plan.kept
    assert not any(path.startswith("administration/") for path in plan.kept)


def test_a_commit_that_is_itself_a_public_copy_is_refused(tmp_path):
    cfg = make_repo(tmp_path / "repo")
    write_files(cfg.root, {NOTICE_NAME: "# Public snapshot\n"})
    commit(cfg.root, "a notice")
    with pytest.raises(ConfigError, match="a public copy itself"):
        expected_of(cfg)


def test_the_account_number_is_replaced_wherever_it_stands_as_a_whole_number():
    rule = Rule("the account", ACCOUNT, PLACEHOLDER)
    text = (
        f"arn:aws:iam::{ACCOUNT}:role/x\nbucket: data-{ACCOUNT}\n"
        f'{{"id": "{ACCOUNT}", "n": {ACCOUNT}}}\n'
        f"longer {ACCOUNT}9 and 9{ACCOUNT} stay\n"
    )
    changed, counts = apply_rules(text, (rule,))
    assert counts == {"the account": 4}
    assert f"arn:aws:iam::{PLACEHOLDER}:role/x" in changed and f"data-{PLACEHOLDER}" in changed
    assert f"{ACCOUNT}9" in changed and f"9{ACCOUNT}" in changed
    assert apply_rules("nothing here", (rule,)) == ("nothing here", {})


def test_every_text_file_is_changed_and_a_file_that_is_not_text_is_copied(tmp_path):
    cfg = make_repo(tmp_path / "repo")
    expected = expected_of(cfg)
    files = {name: data for name, data in expected.files.items()}
    assert f'account_id: "{PLACEHOLDER}"' in files["config/aws.yaml"].decode()
    assert f"arn:aws:iam::{PLACEHOLDER}:role/x" in files["src/code.py"].decode()
    assert f"see {NEW_RUNBOOK} section 3" in files["src/code.py"].decode()  # a message in code
    assert NEW_RUNBOOK in files["tests/test_code.py"].decode()  # and the test that asserts it
    assert f"{ACCOUNT}9" in files["reports/record.json"].decode()  # only resembles the number
    assert PUBLIC_NAME in files["notes/plan.md"].decode()
    assert files["image.bin"].endswith(ACCOUNT.encode()) and expected.binary == ("image.bin",)
    assert expected.counts["the AWS account number"] == {"files": 4, "occurrences": 5}
    # Five in the project's files and two in the export's own config, which names the path.
    assert expected.counts[f"the path {OLD_RUNBOOK}"]["occurrences"] == 7
    text = "".join(data.decode() for name, data in files.items() if name != "image.bin")
    assert OLD_RUNBOOK not in text and PRIVATE_NAME not in text


def test_the_notice_the_readme_line_and_the_runbook_note_are_generated(tmp_path):
    cfg = make_repo(tmp_path / "repo")
    expected = expected_of(cfg)
    notice = expected.files[NOTICE_NAME].decode()
    assert expected.info.sha in notice and expected.info.day in notice
    assert PLACEHOLDER in notice and "placeholder" in notice and "No history" in notice
    assert "private_records" in notice and "`.claude/`" in notice and NEW_RUNBOOK in notice
    assert OLD_RUNBOOK not in notice
    readme = expected.files["README.md"].decode().splitlines()
    assert readme[0] == "# The project" and readme[1] == ""
    assert readme[2].startswith("> **This is a public snapshot") and NOTICE_NAME in readme[2]
    assert not any(character.isdigit() for character in readme[2])  # no number in the README line
    assert readme[3] == "" and readme[4].startswith("See [")
    runbook = expected.files[NEW_RUNBOOK].decode()
    assert runbook.startswith("# Runbook\n\n> **Written for the project's owner.**")
    assert f"Account {PLACEHOLDER}." in runbook and expected.generated == (NOTICE_NAME,)


def test_the_same_commit_gives_the_same_bytes(tmp_path):
    cfg = make_repo(tmp_path / "repo")
    assert expected_of(cfg).files == expected_of(cfg).files


# --------------------------------------------------------------------------- writing the copy


def test_an_empty_folder_is_written_and_is_exactly_the_export(tmp_path):
    cfg = make_repo(tmp_path / "repo")
    dest = tmp_path / "public"
    (dest / ".git").mkdir(parents=True)
    (dest / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    result = build(cfg, dest)
    expected = expected_of(cfg)
    assert result.written == len(expected.files) and result.removed == ()
    assert compare(dest, expected, cfg) == []
    assert (dest / NEW_RUNBOOK).is_file() and not (dest / "administration").exists()
    assert not (dest / ".claude").exists() and (dest / NOTICE_NAME).is_file()
    assert (dest / ".git" / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"
    manifest = json.loads((dest / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["commit"] == expected.info.sha and manifest["clean"] is True
    assert b"\r\n" not in (dest / "README.md").read_bytes()


def test_a_second_export_removes_only_what_the_first_one_wrote(tmp_path):
    cfg = make_repo(tmp_path / "repo")
    dest = tmp_path / "public"
    build(cfg, dest)
    (dest / "venv").mkdir()
    (dest / "venv" / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
    (dest / "notes").mkdir(exist_ok=True)
    (dest / "notes" / "mine.txt").write_text("the owner's own file\n", encoding="utf-8")
    (cfg.root / "notes" / "plan.md").unlink()
    (cfg.root / "src" / "code.py").unlink()
    write_files(cfg.root, {"src/other.py": "X = 1\n"})
    commit(cfg.root, "second")
    result = build(cfg, dest)
    assert set(result.removed) == {"notes/plan.md", "src/code.py"}
    assert (dest / "src" / "other.py").is_file() and not (dest / "src" / "code.py").exists()
    assert (dest / "venv" / "pyvenv.cfg").is_file()  # ignored at the destination
    assert (dest / "notes" / "mine.txt").is_file()  # not written by an export: left alone
    problems = compare(dest, expected_of(cfg), cfg)
    assert problems == ["1 file(s) not part of the export: notes/mine.txt"]


def test_a_folder_that_is_not_an_export_is_refused_and_left_as_it_was(tmp_path):
    cfg = make_repo(tmp_path / "repo")
    dest = tmp_path / "documents"
    dest.mkdir()
    (dest / "thesis.txt").write_text("years of work\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not empty and is not a previous export"):
        build(cfg, dest)
    assert sorted(path.name for path in dest.iterdir()) == ["thesis.txt"]
    for bad in (cfg.root / "inside", cfg.root, tmp_path, tmp_path / "absent" / "deeper"):
        assert destination_problems(bad, cfg, cfg.root), bad
    with pytest.raises(ConfigError, match="outside this repository"):
        build(cfg, cfg.root / "inside")


def test_a_changed_missing_or_extra_file_is_found(tmp_path):
    cfg = make_repo(tmp_path / "repo")
    dest = tmp_path / "public"
    build(cfg, dest)
    (dest / "LICENSE").write_text("edited by hand\n", encoding="utf-8")
    (dest / "src" / "code.py").unlink()
    (dest / "extra.txt").write_text("x\n", encoding="utf-8")
    assert compare(dest, expected_of(cfg), cfg) == [
        "1 file(s) missing: src/code.py",
        "1 file(s) not part of the export: extra.txt",
        "1 file(s) changed since the export: LICENSE",
    ]
