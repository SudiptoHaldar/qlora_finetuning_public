"""What a public copy holds: the plan, the changed texts, and the generated notice.

Everything here is a pure function of a commit's files and the config, so the same commit
gives the same bytes every time (spec D22).

Usage::

    from qlora_finetuning.public_export.transform import expected_files, make_plan

    plan = make_plan(files, cfg)                              # kept, moved, left out
    expected = expected_files(files, cfg, info)               # what the copy must hold
    expected.files["docs/runbook.md"], expected.counts, expected.binary
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from qlora_finetuning.public_export.config import ExportConfig, Rule
from qlora_finetuning.public_export.snapshot import MARKER, NOTICE_NAME
from qlora_finetuning.public_export.source import CommitInfo
from qlora_finetuning.settings import ConfigError

README = "README.md"
COMMAND = "python -m scripts.public_export"
SNAPSHOT_LINE = (
    "> **This is a public snapshot of a private repository.** The AWS account number in "
    "every file is a placeholder, the committed reports and records were redacted and not "
    f"rebuilt, and some folders were left out. See [{NOTICE_NAME}]({NOTICE_NAME}).\n"
)
RUNBOOK_NOTE = (
    "> **Written for the project's owner.** This is the project's own runbook, published as "
    "it is. The steps that are billed or that change AWS need an AWS account of your own; "
    "the account number, the bucket, and the roles shown here are placeholders. See "
    f"[{NOTICE_NAME}](../{NOTICE_NAME}).\n"
)


@dataclass(frozen=True)
class Plan:
    """Which files of a commit go into the copy, and where."""

    kept: dict[str, str]  # path in the copy -> path in the commit
    left_out: tuple[str, ...]

    @property
    def moved(self) -> dict[str, str]:
        return {target: source for target, source in self.kept.items() if target != source}


@dataclass
class Expected:
    """The copy as it must be: every file's bytes, and what was done to make them."""

    info: CommitInfo
    plan: Plan
    files: dict[str, bytes]
    generated: tuple[str, ...]
    counts: dict[str, dict[str, int]] = field(default_factory=dict)  # rule -> files, occurrences
    binary: tuple[str, ...] = ()


def is_left_out(path: str, left_out: tuple[str, ...]) -> bool:
    return any(
        path == entry or (entry.endswith("/") and path.startswith(entry)) for entry in left_out
    )


def make_plan(files: Mapping[str, bytes], cfg: ExportConfig) -> Plan:
    """Sort a commit's paths into kept and left out. A moved file is kept under its new name."""
    moved = dict(cfg.moved)
    kept: dict[str, str] = {}
    left: list[str] = []
    for source in sorted(files):
        target = moved.get(source)
        if target is None:
            if is_left_out(source, cfg.left_out):
                left.append(source)
                continue
            target = source
        if target in kept:
            raise ConfigError(f"two files of the commit would both be written to {target}")
        kept[target] = source
    missing = sorted(set(moved) - set(files))
    if missing:
        raise ConfigError(f"the commit has no {missing[0]} to move")
    if NOTICE_NAME in kept:
        raise ConfigError(f"the commit already holds a {NOTICE_NAME}: it is a public copy itself")
    return Plan(kept=kept, left_out=tuple(left))


def as_text(data: bytes) -> str | None:
    """The file's text, or ``None`` for a file that is not UTF-8 text."""
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def rule_pattern(rule: Rule) -> re.Pattern[str]:
    """What a rule matches: its text exactly. A number matches only as a whole number, so a
    longer number that merely contains the account number is left alone."""
    literal = re.escape(rule.old)
    return re.compile(rf"(?<!\d){literal}(?!\d)" if rule.old.isdigit() else literal)


def apply_rules(text: str, rules: tuple[Rule, ...]) -> tuple[str, dict[str, int]]:
    """``text`` with every rule applied in order, and how often each one matched."""
    counts: dict[str, int] = {}
    for rule in rules:
        text, found = rule_pattern(rule).subn(lambda _match, new=rule.new: new, text)
        if found:
            counts[rule.name] = found
    return text, counts


def _after_title(text: str, note: str) -> str:
    """``text`` with ``note`` as a paragraph of its own under the first line."""
    first, _, rest = text.partition("\n")
    return f"{first}\n\n{note}\n{rest.lstrip(chr(10))}"


def notice_text(cfg: ExportConfig, info: CommitInfo, plan: Plan, rules: tuple[Rule, ...]) -> str:
    """``PUBLIC_SNAPSHOT.md``: what this copy is and how it differs from its source."""
    account = rules[0]
    others = "".join(
        f"- **{rule.name[0].upper()}{rule.name[1:]}** was replaced by `{rule.new}`.\n"
        for rule in rules[1 : len(cfg.replacements)]
    )
    left_out = "".join(f"- `{entry}`\n" for entry in cfg.left_out)
    # The old path is given in two pieces: its full form is what the rewrites remove.
    moved = "".join(
        f"- `{target}` (in the private repository it is `{source.rsplit('/', 1)[-1]}` under "
        f"`{source.rsplit('/', 1)[0]}/`)\n"
        for source, target in cfg.moved
    )
    return (
        "# Public snapshot\n\n"
        f"This repository is a copy of a private one, written by `{COMMAND}`. It is not\n"
        "edited by hand: a change is made in the private repository and exported again.\n\n"
        "| | |\n|---|---|\n"
        f"| Exported commit | `{info.sha}` of the private repository, committed {info.day} |\n"
        f"| Files | {len(plan.kept):,} copied, {len(plan.left_out):,} left out, 1 generated "
        "(this one) |\n\n"
        "## How it differs from the private repository\n\n"
        "- **No history.** Each commit here is one snapshot.\n"
        f"- **{account.name[0].upper()}{account.name[1:]} is a placeholder**, `{account.new}`, "
        "in every file: configs, reports, records, the ledger, tests, and documents. It is "
        "the account number AWS uses in its own documentation. The bucket and the roles "
        "whose names are built on it are placeholders too.\n"
        f"{others}"
        "- **Reports and records were redacted, not rebuilt.** A file that says \"do not "
        "edit by hand\" was changed by the replacements above and by nothing else.\n"
        "- **The checks of the configuration freeze are skipped here.** The freeze record "
        "names AWS resources, so its content changed with the placeholder and its hash is "
        "no longer the one the unlock record names. Tests marked "
        f"`{MARKER}` are skipped in this copy, and say so in the test output. The gate on "
        "the test split refuses a test run here. The freeze can be verified in the private "
        "repository only.\n"
        "- **Some paths changed.**\n"
        f"{''.join('  ' + line for line in moved.splitlines(keepends=True))}"
        "\n## Left out\n\n"
        "A document here may name a file under one of these. It is not in this copy.\n\n"
        f"{left_out}"
        "\n## Using it\n\n"
        "The tests need no GPU, no network, and no credentials. The steps that are billed "
        "or that change AWS need an AWS account of your own, with its own bucket, roles, "
        "and budget in `config/aws.yaml`. The dataset, the runs, the replies, and the model "
        "weights are not in the repository.\n"
    )


def expected_files(
    files: Mapping[str, bytes], cfg: ExportConfig, info: CommitInfo
) -> Expected:
    """Everything the copy must hold, as bytes. Raises ``ConfigError``."""
    plan = make_plan(files, cfg)
    rules = cfg.rules()
    notes = {README: SNAPSHOT_LINE, **{target: RUNBOOK_NOTE for _, target in cfg.moved}}
    result: dict[str, bytes] = {}
    counts: dict[str, dict[str, int]] = {
        rule.name: {"files": 0, "occurrences": 0} for rule in rules
    }
    binary = []
    for target, source in plan.kept.items():
        text = as_text(files[source])
        if text is None:
            result[target] = files[source]
            binary.append(target)
            continue
        text, found = apply_rules(text, rules)
        for name, number in found.items():
            counts[name]["files"] += 1
            counts[name]["occurrences"] += number
        if target in notes:
            text = _after_title(text, notes[target])
        result[target] = text.encode("utf-8")
    result[NOTICE_NAME] = notice_text(cfg, info, plan, rules).encode("utf-8")
    return Expected(
        info=info,
        plan=plan,
        files=result,
        generated=(NOTICE_NAME,),
        counts=counts,
        binary=tuple(binary),
    )
