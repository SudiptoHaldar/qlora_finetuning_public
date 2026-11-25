"""Publishing a package to the Hugging Face Hub, and proving what is there (req_006 v6.03).

Only the folders that v6.02 built and checked are uploaded, each to a private repository,
and only after every file on disk has the size and hash the package record holds. After
an upload the Hub is compared with that record, file by file, and the card's code is run
by repository name. No function here makes a repository public: the owner does that on
the Hub, after reading the rendered card.

Usage::

    from qlora_finetuning.packaging.hub import HubClient
    from qlora_finetuning.packaging.publish import check_package, read_packages, upload_package

    package = read_packages(cfg)["adapter"]
    local_problems(package)                               # U1: [] when it is the checked folder
    upload_package(cfg, package, HubClient(token))        # the owner's; returns the Hub's commit
    entry = check_package(cfg, package, make_client=HubClient, token=None)
    entry["checks"], entry["visibility"]

**The token** is the owner's. It is held in memory, handed to the Hub client and to the
environment of the process that runs a card's code, and nowhere else: never an argument,
never printed, never recorded. ``redact`` takes it out of any text before it is shown.

This module imports nothing of the Hub's library; ``hub.py`` does.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from qlora_finetuning.comparison.report_tables import table
from qlora_finetuning.packaging.cards import ADAPTER, MERGED
from qlora_finetuning.packaging.config import PackagingConfig
from qlora_finetuning.packaging.files import read_json, sha256_file, write_json_lf, write_text_lf
from qlora_finetuning.packaging.record import READY, check_result
from qlora_finetuning.settings import ConfigError, get_str

SCHEMA = 1
COMMAND = "python -m scripts.hf_publish"
KINDS = (ADAPTER, MERGED)
PRIVATE, PUBLIC, NOT_THERE = "private", "public", "not there"
HIDDEN = "<the token>"
FETCH_LIMIT_BYTES = 50 * 1024 * 1024  # a file up to this size is fetched to be hashed
SHOWN = 5
# Which runs of a card's code belong to a package, and what each one sets.
CARD_RUNS = {
    ADAPTER: (("adapter", {}),),
    MERGED: (("merged_4bit", {"QUANTIZED": True}), ("merged_full", {"QUANTIZED": False})),
}


@dataclass(frozen=True)
class Package:
    """One package as v6.02 recorded it."""

    kind: str
    repo_id: str
    folder: Path
    status: str
    files: tuple[dict[str, Any], ...]  # name, bytes, sha256

    @property
    def total_bytes(self) -> int:
        return sum(int(entry["bytes"]) for entry in self.files)


def redact(text: str, token: str | None) -> str:
    """``text`` with the token's value taken out."""
    return text.replace(token, HIDDEN) if token else text


def read_token(cfg: PackagingConfig) -> str | None:
    """The token from the environment or ``.env``, or ``None``. It is never shown."""
    return get_str(cfg.token_env, "") or None


def read_packages(cfg: PackagingConfig) -> dict[str, Package]:
    """The two packages, from the committed package record. Raises ``ConfigError``."""
    record = read_json(cfg.record_json, "the package record")
    folders = {ADAPTER: cfg.adapter_dir, MERGED: cfg.merged_dir}
    repos = {ADAPTER: cfg.adapter_repo, MERGED: cfg.merged_repo}
    packages = {}
    for kind in KINDS:
        entry = record.get(kind) or {}
        packages[kind] = Package(
            kind=kind,
            repo_id=repos[kind],
            folder=folders[kind],
            status=str(entry.get("status")),
            files=tuple(entry.get("files") or ()),
        )
    return packages


def _some(names: list[str]) -> str:
    more = f", and {len(names) - SHOWN} more" if len(names) > SHOWN else ""
    return ", ".join(names[:SHOWN]) + more


def local_problems(package: Package) -> list[str]:
    """Why the folder on disk is not the package that was checked (U1); empty when it is."""
    if package.status != READY:
        return [
            f"the package record says the {package.kind} package is {package.status!r}, not "
            f"{READY!r}: python -m scripts.hf_package status"
        ]
    if not package.folder.is_dir():
        return [
            f"{package.folder.name}/ does not exist: python -m scripts.hf_package {package.kind}"
        ]
    recorded = {entry["name"]: entry for entry in package.files}
    present = {path.name for path in package.folder.iterdir() if path.is_file()}
    problems = []
    if recorded.keys() - present:
        problems.append(f"missing on disk: {_some(sorted(recorded.keys() - present))}")
    if present - recorded.keys():
        problems.append(f"not in the package record: {_some(sorted(present - recorded.keys()))}")
    changed = [
        name
        for name in sorted(recorded.keys() & present)
        if (package.folder / name).stat().st_size != recorded[name]["bytes"]
        or sha256_file(package.folder / name) != recorded[name]["sha256"]
    ]
    if changed:
        problems.append(f"changed since the package record was written: {_some(changed)}")
    return problems


def _same(remote: Any, entry: dict[str, Any], fetch: Callable[[str], bytes], name: str) -> bool:
    """Whether a file on the Hub is the package's file: by size, then by hash.

    The Hub reports a hash for a large file. A small file is fetched and hashed. A large
    file for which the Hub reports no hash is compared by size alone: fetching gigabytes
    into memory to hash them is not done here, and U5 loads those weights anyway.
    """
    if remote.size is not None and remote.size != entry["bytes"]:
        return False
    if remote.sha256 is not None:
        return remote.sha256 == entry["sha256"]
    if entry["bytes"] > FETCH_LIMIT_BYTES:
        return remote.size == entry["bytes"]
    return hashlib.sha256(fetch(name)).hexdigest() == entry["sha256"]


def remote_problems(
    state: Any,
    package: Package,
    fetch: Callable[[str], bytes],
    hub_files: tuple[str, ...],
    *,
    complete: bool,
) -> list[str]:
    """How a repository differs from the package.

    With ``complete`` every file of the package must be there (U4). Without it, only what
    is there is looked at: nothing foreign, nothing changed (U2, before an upload).
    """
    recorded = {entry["name"]: entry for entry in package.files}
    remote = {name: item for name, item in state.files.items() if name not in hub_files}
    problems = []
    if remote.keys() - recorded.keys():
        problems.append(
            f"on the Hub and not in the package: {_some(sorted(remote.keys() - recorded.keys()))}"
        )
    if complete and recorded.keys() - remote.keys():
        problems.append(
            f"in the package and not on the Hub: {_some(sorted(recorded.keys() - remote.keys()))}"
        )
    changed = [
        name
        for name in sorted(remote.keys() & recorded.keys())
        if not _same(remote[name], recorded[name], fetch, name)
    ]
    if changed:
        problems.append(f"not the package's file: {_some(changed)}")
    return problems


def upload_package(cfg: PackagingConfig, package: Package, client: Any) -> str:
    """Upload a package to its private repository (the owner's step). Returns the commit.

    Refuses, with a ``ConfigError`` and without touching the Hub's files, when the folder is
    not the checked package (U1) or the repository holds something else (U2).
    """
    problems = local_problems(package)
    if problems:
        raise ConfigError(
            f"the {package.kind} package is not the one that was checked: {problems[0]}"
        )
    state = client.state(package.repo_id)
    if state is not None:
        fetch = lambda name: client.fetch(package.repo_id, name, state.commit)  # noqa: E731
        foreign = remote_problems(state, package, fetch, cfg.hub_files, complete=False)
        if foreign:
            raise ConfigError(
                f"{package.repo_id} already holds other content ({foreign[0]}): nothing was "
                "uploaded. Look at the repository on the Hub"
            )
    else:
        client.create_private(package.repo_id)
    return client.upload(package.repo_id, package.folder, f"Upload the {package.kind} package")


def find_state(
    package: Package, make_client: Callable[[str | None], Any], token: str | None
) -> tuple[Any, Any, bool]:
    """The repository's state, the client that could see it, and whether the token was used.

    It is asked for without a token first: what that sees is what the world sees.
    """
    client = make_client(None)
    state = client.state(package.repo_id)
    if state is None and token:
        client = make_client(token)
        return client.state(package.repo_id), client, True
    return state, client, False


def check_package(
    cfg: PackagingConfig,
    package: Package,
    *,
    make_client: Callable[[str | None], Any],
    token: str | None,
    card_check: Callable[[str, bool, str | None], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Checks U1 and U4 to U6 on one package, as an entry of the record. Writes nothing."""
    checks = [
        check_result(
            "U1",
            "The package is the one that was checked",
            local_problems(package),
            f"{len(package.files)} files, {package.total_bytes:,} bytes, each with the size and "
            "hash of the package record",
        )
    ]
    state, client, used_token = find_state(package, make_client, token)
    entry: dict[str, Any] = {
        "repository": package.repo_id,
        "visibility": NOT_THERE,
        "hub_commit": None,
        "files": len(package.files),
        "bytes": package.total_bytes,
        "card_code": [],
    }
    if state is None:
        why = "it does not exist, or it is private and no token was given"
        checks.append(
            check_result("U4", "The Hub holds the package", [f"{package.repo_id}: {why}"], "")
        )
        entry["checks"] = checks
        return entry

    entry.update(visibility=PRIVATE if state.private else PUBLIC, hub_commit=state.commit)
    fetch = lambda name: client.fetch(package.repo_id, name, state.commit)  # noqa: E731
    checks.append(
        check_result(
            "U4",
            "The Hub holds the package",
            remote_problems(state, package, fetch, cfg.hub_files, complete=True),
            f"the same {len(package.files)} files by name, size, and hash; nothing else but "
            f"{', '.join(cfg.hub_files)}",
        )
    )
    runs = card_check(package.kind, True, token if used_token else None)
    entry["card_code"] = runs
    failed = [f"{run['label']}: {problem}" for run in runs for problem in run["problems"]]
    checks.append(
        check_result(
            "U5",
            "The card's code runs from the Hub",
            failed or ([] if runs else ["the card's code was not run"]),
            f"{len(runs)} run(s) by repository name, each to the sample's expected result",
        )
    )
    checks.append(
        check_result(
            "U6",
            "Visibility",
            [],
            f"{entry['visibility']}, at commit {state.commit[:12]}"
            + ("; read with the owner's token" if used_token else "; read without a token"),
        )
    )
    entry["checks"] = checks
    return entry


def card_runs(
    cfg: PackagingConfig, kind: str, online: bool, token: str | None
) -> list[dict[str, Any]]:
    """Run a card's code by repository name and say what each run gave (U5).

    The code is the card's own: the same template that v6.02 ran against the folder, with
    nothing substituted but the precision switch of the merged card. No reply is kept.
    """
    from qlora_finetuning.data.serialization import load_prompt
    from qlora_finetuning.data.token_budget import load_token_budget_config, load_tokenizer
    from qlora_finetuning.demo.inputs import load_input, sample_path
    from qlora_finetuning.evaluation.metrics import load_evaluation_config
    from qlora_finetuning.local_qlora.config import load_local_qlora_config
    from qlora_finetuning.packaging.cards import usage_code
    from qlora_finetuning.packaging.files import sha256_text
    from qlora_finetuning.packaging.verify import check_card_run, run_card_code

    budget, prompt = load_token_budget_config(), load_prompt()
    code, problems = usage_code(kind, cfg, budget, load_local_qlora_config())
    runs = []
    for label, values in CARD_RUNS[kind]:
        entry: dict[str, Any] = {
            "label": label,
            "code_sha256": sha256_text(code),
            "problems": list(problems),
        }
        if not problems:
            try:
                result = run_card_code(
                    code, values, timeout=cfg.publish_timeout_seconds, online=online, token=token
                )
                found, facts = check_card_run(
                    result,
                    sample=load_input(sample_path()),
                    prompt=prompt,
                    tokenizer=load_tokenizer(budget),
                    budget=budget,
                    evaluation_config=load_evaluation_config(),
                )
                entry.update(facts, seconds=result["seconds"])
                entry["problems"] = found
            except ConfigError as exc:
                entry["problems"] = [redact(str(exc), token)]
        runs.append(entry)
    return runs


def build_record(
    cfg: PackagingConfig, entries: dict[str, dict[str, Any]], today: date, token: str | None
) -> dict[str, Any]:
    """The publish record, with the check that the token is nowhere in it (U7)."""
    record = {
        "schema": SCHEMA,
        "generated_by": f"{COMMAND} check",
        "checked_on": today.isoformat(),
        "packages": entries,
    }
    if token and token in json.dumps(record):  # never write it down
        raise ConfigError("the record would hold the token's value; nothing was written")
    for entry in entries.values():
        entry["checks"].append(
            check_result(
                "U7",
                "Nothing leaked",
                [],
                "the token's value is in no line of this record",
            )
        )
    record["passed"] = all(
        check["passed"] for entry in entries.values() for check in entry["checks"]
    )
    record["published"] = (
        bool(entries)
        and record["passed"]
        and all(entry["visibility"] == PUBLIC for entry in entries.values())
    )
    return record


def render_markdown(record: dict[str, Any]) -> str:
    lines = [
        "# The published packages",
        "",
        f"Written by `{record['generated_by']}` on {record['checked_on']}. Do not edit by hand: "
        "check again. This file holds repository names, hashes of nothing private, and no token.",
        "",
    ]
    for kind, entry in record["packages"].items():
        lines += [
            f"## The {kind} package: `{entry['repository']}`",
            "",
            f"- **Visibility when checked**: {entry['visibility']}",
            f"- **Commit on the Hub**: `{entry['hub_commit'] or 'none'}`",
            f"- **Package**: {entry['files']} files, {entry['bytes']:,} bytes",
            "",
            table(
                ["Check", "", "Result", "Detail"],
                [
                    [
                        check["id"],
                        check["name"],
                        "passed" if check["passed"] else "**FAILED**",
                        check["detail"],
                    ]
                    for check in entry["checks"]
                ],
            ),
            "",
        ]
        if entry["card_code"]:
            lines += [
                table(
                    ["Card code", "Result", "Executed value", "Seconds"],
                    [
                        [
                            run["label"],
                            "; ".join(run["problems"]) or "ran to the expected result",
                            "—" if run.get("value") is None else repr(run["value"]),
                            run.get("seconds", "—"),
                        ]
                        for run in entry["card_code"]
                    ],
                ),
                "",
            ]
    state = (
        "**published**: both are public and every check passed"
        if record["published"]
        else ("not published yet: a repository is not public, or a check did not pass")
    )
    lines += [f"Status: {state}.", ""]
    return "\n".join(lines)


def write_record(cfg: PackagingConfig, record: dict[str, Any]) -> list[Path]:
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    write_json_lf(cfg.publish_json, record)
    write_text_lf(cfg.publish_markdown, render_markdown(record))
    return [cfg.publish_json, cfg.publish_markdown]
