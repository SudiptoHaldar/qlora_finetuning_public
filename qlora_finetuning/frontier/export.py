"""The Bedrock training files: fit and dev in conversation format (req_004 v4.02).

One line per record, holding the canonical record's three texts unchanged: the system text,
the user text as the one ``user`` turn, and the target as the ``assistant`` turn. That is
also exactly what ``BedrockProvider`` sends at inference, so a tuned model is asked the way
it was taught. Every record of a split is exported, repeated ones included, so both paths
train on the same examples.

Only ``fit`` (the training file) and ``dev`` (the validation file) can be exported: the
exporter takes no split name. The files hold dataset text and are written under the
git-ignored ``artifacts/bedrock/<export_id>/``; the export id is derived from their content.

Usage::

    from qlora_finetuning.evaluation.runner import RunnerContext
    from qlora_finetuning.frontier.config import load_frontier_config
    from qlora_finetuning.frontier.export import export_files

    export, newly_written = export_files(load_frontier_config(), RunnerContext().resolved())
    export.files["training"].path       # .../artifacts/bedrock/<export_id>/finqa_fit.jsonl
    export.files["training"].sha256     # what the gate record and S3 are compared with

Check the files with ``qlora_finetuning.frontier.export_validation.validate_export`` before
anything is done with them. ``python -m scripts.nova_export`` does both.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qlora_finetuning.data import manifest_hash, serialize_split
from qlora_finetuning.frontier.config import FrontierConfig

TRAINING, VALIDATION = "training", "validation"
# The only splits that are ever exported, and the name of each one's file.
SPLITS = {TRAINING: "fit", VALIDATION: "dev"}
FILE_NAMES = {TRAINING: "finqa_fit.jsonl", VALIDATION: "finqa_dev.jsonl"}
IDS_SUFFIX = ".ids.txt"
EXPORT_ID_LENGTH = 12

# Words that make a Nova fine-tuning job fail. AWS's guide names seven for the user and
# system prompts; AWS's validation script adds two and compares without regard to case in
# every text. This is the stricter reading: a file that passes it passes both.
RESERVED_WORDS = (
    "system:",
    "user:",
    "bot:",
    "assistant:",
    "thought:",
    "[eos]",
    "<image>",
    "<video>",
    "<unk>",
)


@dataclass(frozen=True)
class ExportFile:
    """One exported file and what identifies it."""

    role: str  # training | validation
    split: str  # fit | dev
    path: Path
    ids_path: Path  # one example id per line, in file order; stays local
    sha256: str
    size: int  # bytes
    records: int
    repeated_lines: int  # lines identical to an earlier line; kept on purpose


@dataclass(frozen=True)
class Export:
    """The two files of one export."""

    export_id: str
    folder: Path
    files: dict[str, ExportFile]
    prompt_id: str
    split_hash: str
    schema_version: str


def conversation_line(record: Any, schema_version: str) -> str:
    """One record as a Bedrock conversation line, without the line break.

    Fixed key order, compact separators, characters outside ASCII written as themselves:
    the same record always gives the same bytes.
    """
    return json.dumps(
        {
            "schemaVersion": schema_version,
            "system": [{"text": record.system}],
            "messages": [
                {"role": "user", "content": [{"text": record.user}]},
                {"role": "assistant", "content": [{"text": record.target}]},
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def render_file(records: Iterable[Any], schema_version: str) -> bytes:
    """The bytes of a training file: one line per record, each ended by a line break."""
    return "".join(
        conversation_line(record, schema_version) + "\n" for record in records
    ).encode("utf-8")


def render_ids(records: Iterable[Any]) -> bytes:
    return "".join(f"{record.example_id}\n" for record in records).encode("utf-8")


def sha256_of(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def render_split(role: str, cfg: FrontierConfig, context: Any) -> tuple[bytes, bytes, list[Any]]:
    """The file bytes, the id-list bytes, and the records of one role's split."""
    records = serialize_split(SPLITS[role], context.prompt, context.split_cfg, context.data_cfg)
    return render_file(records, cfg.schema_version), render_ids(records), records


def _repeated(content: bytes) -> int:
    counts = Counter(content.splitlines())
    return sum(count - 1 for count in counts.values())


def export_files(cfg: FrontierConfig, context: Any) -> tuple[Export, bool]:
    """Render fit and dev and write them under ``cfg.export_dir``; return the export.

    ``context`` is a resolved ``RunnerContext``: it carries the prompt and the split. The
    second value is true when files were written, false when the folder already held
    exactly these bytes. Asking twice gives the same export id and the same files.
    """
    rendered = {role: render_split(role, cfg, context) for role in SPLITS}
    digests = {role: sha256_of(content) for role, (content, _, _) in rendered.items()}
    export_id = sha256_of("\n".join(digests[role] for role in SPLITS).encode())[:EXPORT_ID_LENGTH]
    folder = cfg.export_dir / export_id

    files: dict[str, ExportFile] = {}
    written = False
    for role, (content, ids, records) in rendered.items():
        path = folder / FILE_NAMES[role]
        ids_path = path.with_name(path.name + IDS_SUFFIX)
        for target, wanted in ((path, content), (ids_path, ids)):
            if not target.is_file() or target.read_bytes() != wanted:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(wanted)
                written = True
        files[role] = ExportFile(
            role=role,
            split=SPLITS[role],
            path=path,
            ids_path=ids_path,
            sha256=digests[role],
            size=len(content),
            records=len(records),
            repeated_lines=_repeated(content),
        )
    prompt_id = context.prompt.prompt_id
    return (
        Export(
            export_id=export_id,
            folder=folder,
            files=files,
            prompt_id=prompt_id,
            split_hash=manifest_hash(context.split_cfg),
            schema_version=cfg.schema_version,
        ),
        written,
    )
