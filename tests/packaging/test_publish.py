"""Publishing to the Hugging Face Hub, against a stand-in Hub that keeps files in memory.

No network, no token of anyone's, no GPU. The "token" used here is an invented word.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date

import pytest

from qlora_finetuning.packaging import publish
from qlora_finetuning.packaging.config import load_packaging_config
from qlora_finetuning.packaging.files import write_json_lf
from qlora_finetuning.packaging.hub import RemoteFile, RepoState
from qlora_finetuning.packaging.verify import TOKEN_VARIABLES, run_card_code
from qlora_finetuning.settings import ConfigError
from scripts.hf_publish import main

TOKEN = "invented-" + "token-value-0001"
DAY = date(2026, 11, 24)
LARGE = 64  # the stand-in Hub reports a hash for a file of this many bytes or more
FILES = {
    "adapter": {
        "weights.safetensors": b"w" * 100,
        "prompt.json": b'{"a": 1}\n',
        "README.md": b"# card\n",
    },
    "merged": {"model.safetensors": b"m" * 200, "README.md": b"# merged card\n"},
}


class FakeHub:
    """Repositories in memory. A caller without the token sees only public ones."""

    def __init__(self):
        self.repos, self.calls, self.failing = {}, [], None

    def client(self, token):
        return FakeClient(self, token)


class FakeClient:
    def __init__(self, hub, token):
        self.hub, self.token = hub, token

    def state(self, repo_id):
        self.hub.calls.append(("state", repo_id, self.token))
        repo = self.hub.repos.get(repo_id)
        if repo is None or (repo["private"] and self.token != TOKEN):
            return None
        files = {
            name: RemoteFile(
                size=len(data),
                sha256=hashlib.sha256(data).hexdigest() if len(data) >= LARGE else None,
            )
            for name, data in repo["files"].items()
        }
        return RepoState(private=repo["private"], commit=repo["commit"], files=files)

    def create_private(self, repo_id):
        self.hub.calls.append(("create", repo_id, self.token))
        self.hub.repos[repo_id] = {
            "private": True,
            "files": {".gitattributes": b"*\n"},
            "commit": "c0",
        }

    def upload(self, repo_id, folder, message):
        self.hub.calls.append(("upload", repo_id, self.token))
        if self.hub.failing:
            raise ConfigError(self.hub.failing)
        repo = self.hub.repos[repo_id]
        for path in folder.iterdir():
            repo["files"][path.name] = path.read_bytes()
        repo["commit"] = "c" + str(int(repo["commit"][1:]) + 1) + "0" * 38
        return repo["commit"]

    def fetch(self, repo_id, name, revision):
        self.hub.calls.append(("fetch", name, self.token))
        return self.hub.repos[repo_id]["files"][name]


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Two package folders, their record, a stand-in Hub, and a token in the environment."""
    cfg = load_packaging_config(root=tmp_path)
    record = {"schema": 1, "passed": True}
    for kind, folder in (("adapter", cfg.adapter_dir), ("merged", cfg.merged_dir)):
        folder.mkdir(parents=True)
        entries = []
        for name, data in FILES[kind].items():
            (folder / name).write_bytes(data)
            entries.append(
                {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            )
        record[kind] = {"status": "ready", "files": entries}
    cfg.report_dir.mkdir(parents=True)
    write_json_lf(cfg.record_json, record)
    monkeypatch.setenv(cfg.token_env, TOKEN)
    return cfg, FakeHub(), record


def run(world, argv, capsys, **options):
    cfg, hub, _ = world
    options.setdefault("card_check", good_cards)
    code = main(argv, cfg=cfg, make_client=hub.client, today=DAY, **options)
    return code, capsys.readouterr().out


def good_cards(kind, online, token):
    labels = ["adapter"] if kind == "adapter" else ["merged_4bit", "merged_full"]
    return [
        {"label": label, "code_sha256": "0" * 64, "problems": [], "value": 0.125, "seconds": 1.0}
        for label in labels
    ]


# --------------------------------------------------------------------------- plan and U1


def test_plan_lists_both_packages_and_reads_no_token(world, capsys, monkeypatch):
    cfg, hub, _ = world
    monkeypatch.setattr(publish, "read_token", lambda cfg: pytest.fail("the token was read"))
    code, out = run(world, ["plan"], capsys)
    assert code == 0 and hub.calls == [] and out.isascii()
    assert "PASS  U1  The adapter package: 3 files" in out
    assert f"WOULD UPLOAD  artifacts/release/merged/ -> {cfg.merged_repo} (created private)" in out
    assert "PLAN ONLY: nothing was sent and no token was read" in out


def test_a_folder_that_is_not_the_checked_package_fails_u1(world, capsys):
    cfg, hub, record = world
    (cfg.adapter_dir / "prompt.json").write_bytes(b'{"a": 2}\n')
    (cfg.adapter_dir / "notes.txt").write_bytes(b"x")
    (cfg.merged_dir / "README.md").unlink()
    packages = publish.read_packages(cfg)
    assert publish.local_problems(packages["adapter"]) == [
        "not in the package record: notes.txt",
        "changed since the package record was written: prompt.json",
    ]
    assert publish.local_problems(packages["merged"]) == ["missing on disk: README.md"]
    code, out = run(world, ["plan"], capsys)
    assert code == 1 and out.count("FAIL  U1") == 2
    assert run(world, ["upload", "adapter", "--yes"], capsys)[0] == 2
    assert not [call for call in hub.calls if call[0] in ("create", "upload")]

    record["adapter"]["status"] = "not ready"
    write_json_lf(cfg.record_json, record)
    problem = publish.local_problems(publish.read_packages(cfg)["adapter"])[0]
    assert "'not ready', not 'ready'" in problem and "hf_package status" in problem


# --------------------------------------------------------------------------- upload


def test_an_upload_needs_the_owners_yes_and_a_token(world, capsys, monkeypatch):
    cfg, hub, _ = world
    code, out = run(world, ["upload", "adapter"], capsys)
    assert code == 2 and "REFUSED" in out and "add --yes" in out and hub.calls == []
    monkeypatch.setenv(cfg.token_env, "")
    monkeypatch.setattr(publish, "read_token", lambda cfg: None)
    code, out = run(world, ["upload", "adapter", "--yes"], capsys)
    assert code == 2 and "CONFIG ERROR: no token: put HF_TOKEN=" in out and ".env" in out
    assert hub.calls == []
    with pytest.raises(SystemExit):
        main(["upload", "--yes"], cfg=cfg, make_client=hub.client)


def test_an_upload_creates_a_private_repository_and_sends_the_folder(world, capsys):
    cfg, hub, _ = world
    code, out = run(world, ["upload", "adapter", "--yes"], capsys)
    repo = hub.repos[cfg.adapter_repo]
    assert code == 0 and repo["private"] is True
    assert {
        name: data for name, data in repo["files"].items() if name != ".gitattributes"
    } == FILES["adapter"]
    assert [call[0] for call in hub.calls] == ["state", "create", "upload"]
    assert all(call[2] == TOKEN for call in hub.calls)  # handed to the client, not the command
    assert "UPLOADED" in out and "the repository is private" in out and TOKEN not in out
    assert cfg.merged_repo not in hub.repos  # one package per command


def test_a_repository_with_other_content_is_refused_and_one_with_part_is_resumed(world, capsys):
    cfg, hub, _ = world
    hub.repos[cfg.adapter_repo] = {
        "private": True, "commit": "c5", "files": {"someone_elses.bin": b"z" * 10},
    }  # fmt: skip
    code, out = run(world, ["upload", "adapter", "--yes"], capsys)
    assert code == 2 and "already holds other content" in out and "someone_elses.bin" in out
    assert hub.repos[cfg.adapter_repo]["files"] == {"someone_elses.bin": b"z" * 10}

    hub.repos[cfg.adapter_repo]["files"] = {"prompt.json": b"another prompt\n"}
    code, out = run(world, ["upload", "adapter", "--yes"], capsys)
    assert code == 2 and "not the package's file: prompt.json" in out

    hub.repos[cfg.adapter_repo]["files"] = {
        ".gitattributes": b"*\n", "prompt.json": FILES["adapter"]["prompt.json"],
    }  # fmt: skip
    hub.calls.clear()
    code, _ = run(world, ["upload", "adapter", "--yes"], capsys)
    assert code == 0 and "create" not in [call[0] for call in hub.calls]
    assert set(hub.repos[cfg.adapter_repo]["files"]) == {".gitattributes", *FILES["adapter"]}


def test_the_token_is_never_printed_even_when_an_error_holds_it(world, capsys):
    cfg, hub, _ = world
    hub.failing = f"the upload failed: Authorization: Bearer {TOKEN} was rejected"
    code, out = run(world, ["upload", "adapter", "--yes"], capsys)
    assert code == 2 and TOKEN not in out and publish.HIDDEN in out
    assert publish.redact(f"a {TOKEN} b", TOKEN) == f"a {publish.HIDDEN} b"
    assert publish.redact("nothing", None) == "nothing"


# --------------------------------------------------------------------------- check


def uploaded(world, capsys, *kinds):
    for kind in kinds:
        assert run(world, ["upload", kind, "--yes"], capsys)[0] == 0


def test_a_private_upload_is_checked_with_the_token_and_recorded(world, capsys):
    cfg, hub, _ = world
    uploaded(world, capsys, "adapter")
    seen = []

    def cards(kind, online, token):
        seen.append((kind, online, token))
        return good_cards(kind, online, token)

    code, out = run(world, ["check", "adapter"], capsys, card_check=cards)
    assert code == 0 and seen == [("adapter", True, TOKEN)]
    assert "adapter: " + cfg.adapter_repo + " (private)" in out
    for check in ("U1", "U4", "U5", "U6", "U7"):
        assert f"PASS  {check}  " in out
    assert "read with the owner's token" in out and "still private" in out
    assert TOKEN not in out and "WROTE  reports/packaging/publish_record.json" in out
    text = cfg.publish_json.read_text(encoding="utf-8") + cfg.publish_markdown.read_text("utf-8")
    assert TOKEN not in text
    record = json.loads(cfg.publish_json.read_text(encoding="utf-8"))
    entry = record["packages"]["adapter"]
    assert record["published"] is False and record["passed"] is True
    assert entry["visibility"] == "private" and entry["hub_commit"].startswith("c1")
    assert record["checked_on"] == "2026-11-24" and "merged" not in record["packages"]
    # A small file was fetched and hashed; the large one was compared by the Hub's own hash.
    fetched = [call[1] for call in hub.calls if call[0] == "fetch"]
    assert "prompt.json" in fetched and "weights.safetensors" not in fetched


def test_public_repositories_are_checked_without_a_token_and_called_published(
    world, capsys, monkeypatch
):
    cfg, hub, _ = world
    uploaded(world, capsys, "adapter", "merged")
    for repo in hub.repos.values():
        repo["private"] = False
    hub.calls.clear()
    monkeypatch.setattr(publish, "read_token", lambda cfg: pytest.fail("the token was read"))
    seen = []

    def cards(kind, online, token):
        seen.append((kind, token))
        return good_cards(kind, online, token)

    code, out = run(world, ["check"], capsys, card_check=cards)
    assert code == 0 and seen == [("adapter", None), ("merged", None)]
    assert all(call[2] is None for call in hub.calls)  # nothing was asked with a token
    assert "PUBLISHED: both repositories are public and every check passed" in out
    assert out.count("read without a token") == 2
    record = json.loads(cfg.publish_json.read_text(encoding="utf-8"))
    assert record["published"] is True and set(record["packages"]) == {"adapter", "merged"}
    assert len(record["packages"]["merged"]["card_code"]) == 2
    page = cfg.publish_markdown.read_text(encoding="utf-8")
    assert "**published**" in page and "| U4 | The Hub holds the package | passed |" in page


def test_a_missing_extra_or_changed_file_on_the_hub_fails_u4(world, capsys):
    cfg, hub, _ = world
    uploaded(world, capsys, "adapter")
    files = hub.repos[cfg.adapter_repo]["files"]
    del files["README.md"]
    files["stray.txt"] = b"x"
    files["prompt.json"] = b"edited on the website\n"
    files["weights.safetensors"] = b"v" * 100
    code, out = run(world, ["check", "adapter"], capsys)
    assert code == 1 and "FAIL  U4" in out
    assert "on the Hub and not in the package: stray.txt" in out
    assert "in the package and not on the Hub: README.md" in out
    assert "not the package's file: prompt.json, weights.safetensors" in out
    assert json.loads(cfg.publish_json.read_text(encoding="utf-8"))["passed"] is False


def test_a_repository_that_cannot_be_seen_and_failing_card_code_fail_their_checks(world, capsys):
    cfg, hub, _ = world
    code, out = run(world, ["check", "adapter"], capsys)
    assert code == 1 and "does not exist, or it is private and no token was given" in out
    uploaded(world, capsys, "adapter")

    def bad_cards(kind, online, token):
        return [{"label": "adapter", "problems": ["the reply's program did not run (x)"]}]

    code, out = run(world, ["check", "adapter"], capsys, card_check=bad_cards)
    assert code == 1 and "FAIL  U5" in out and "adapter: the reply's program did not run" in out


def test_a_second_check_keeps_the_other_packages_entry(world, capsys):
    cfg, hub, _ = world
    uploaded(world, capsys, "adapter", "merged")
    assert run(world, ["check", "adapter"], capsys)[0] == 0
    assert run(world, ["check", "merged"], capsys)[0] == 0
    record = json.loads(cfg.publish_json.read_text(encoding="utf-8"))
    assert set(record["packages"]) == {"adapter", "merged"}
    for entry in record["packages"].values():
        assert [check["id"] for check in entry["checks"]].count("U7") == 1


# --------------------------------------------------------------------------- the card's code


PROBE = (
    "import os\nNAME = 'x'\n"
    "system = os.environ.get('HF_HUB_OFFLINE', 'unset')\n"
    "user = os.environ.get('HF_TOKEN', 'unset')\n"
    "text = os.environ.get('HF_HUB_DISABLE_IMPLICIT_TOKEN', 'unset')\n"
    "reply = os.environ.get('HUGGING_FACE_HUB_TOKEN', 'unset')\n"
)


def test_the_cards_code_gets_the_hub_and_the_token_only_when_asked(monkeypatch):
    for name in TOKEN_VARIABLES:
        monkeypatch.setenv(name, "inherited-from-the-shell")
    offline = run_card_code(PROBE, {}, timeout=60)
    assert (offline["system"], offline["user"], offline["reply"]) == ("1", "unset", "unset")
    public = run_card_code(PROBE, {}, timeout=60, online=True)
    assert (public["system"], public["user"], public["text"]) == ("unset", "unset", "1")
    private = run_card_code(PROBE, {}, timeout=60, online=True, token=TOKEN)
    assert (private["system"], private["user"], private["text"]) == ("unset", TOKEN, "unset")
    assert private["reply"] == "unset"  # another variable of the shell is not passed on


def test_a_large_file_without_a_hub_hash_is_compared_by_size_and_never_fetched(monkeypatch):
    monkeypatch.setattr(publish, "FETCH_LIMIT_BYTES", 10)
    entry = {"name": "big", "bytes": 100, "sha256": "0" * 64}
    never = lambda name: pytest.fail("a large file was fetched")  # noqa: E731
    assert publish._same(RemoteFile(size=100, sha256=None), entry, never, "big") is True
    assert publish._same(RemoteFile(size=99, sha256=None), entry, never, "big") is False
    assert publish._same(RemoteFile(size=None, sha256=None), entry, never, "big") is False
