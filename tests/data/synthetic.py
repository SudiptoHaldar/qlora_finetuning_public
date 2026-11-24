"""Invented FinQA-shaped rows for tests. No real dataset row is ever used in a test.

``build_snapshot`` writes the rows to Parquet files with the real column types under a
temporary folder and returns a config whose checksums, sizes, and row counts describe those
files, so the snapshot code runs exactly as it does on the real data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from qlora_finetuning.data import DataConfig, load_data_config
from qlora_finetuning.data.snapshot import sha256_of, snapshot_path
from qlora_finetuning.settings import CONFIG_DIR

REVISION = "0123456789abcdef0123456789abcdef01234567"

# Words that appear only in the invented text below. The report must contain none of them.
SECRET_WORDS = ("zebrafish", "marmalade", "quokka", "xylophone", "parsnip")


def row(example_id: str = "ACME/2015/page_10.pdf-1", **overrides: Any) -> dict[str, Any]:
    """One well-formed row. Keyword arguments replace fields; ``gold_inds`` may be a dict."""
    filename = example_id.rsplit("-", 1)[0]
    values: dict[str, Any] = {
        "id": example_id,
        "filename": filename,
        "pre_text": ["zebrafish revenue rose in the year .", "the marmalade segment grew ."],
        "post_text": ["see the quokka note for details ."],
        "table": [["", "2015", "2014"], ["xylophone sales", "920", "95"], ["total", "1000", "900"]],
        "table_ori": [["", "2015", "2014"], ["xylophone sales", "$ 920", "$ 95"]],
        "question": f"what was the change in xylophone sales for {example_id}?",
        "answer": "825",
        "exe_ans": "825",
        "explanation": "",
        "ann_table_rows": [1],
        "ann_text_rows": [],
        "steps": {"op": ["minus1-1"], "arg1": ["920"], "arg2": ["95"], "res": ["825"]},
        "program": "subtract(920, 95)",
        "program_re": "subtract(920, 95)",
        "gold_inds": {"table_1": "the xylophone sales of 2015 is 920 ; parsnip"},
    }
    values.update(overrides)
    if not isinstance(values["gold_inds"], str):
        values["gold_inds"] = json.dumps(values["gold_inds"])
    return values


def clean_rows() -> dict[str, list[dict[str, Any]]]:
    """A small dataset with nothing odd in it: every check passes."""
    return {
        "train": [
            row("ACME/2015/page_10.pdf-1"),
            row(
                "ACME/2015/page_10.pdf-2",
                program="divide(920, const_100), add(#0, 1.5)",
                program_re="add(divide(920, const_100), 1.5)",
                answer="10.7",
                exe_ans="10.7",
            ),
            row(
                "BOLT/2016/page_3.pdf-1",
                program="table_sum(xylophone sales, none)",
                program_re="table_sum(xylophone sales, none)",
                answer="1015",
                exe_ans="1015",
            ),
        ],
        "validation": [
            row(
                "ACME/2016/page_4.pdf-1",
                program="subtract(920, 95), divide(#0, 95)",
                program_re="divide(subtract(920, 95), 95)",
                answer="868.4%",
                exe_ans="8.68421",
                gold_inds={"table_1": "xylophone", "text_0": "zebrafish"},
                ann_text_rows=[0],
            )
        ],
        "test": [
            row(
                "CRUX/2017/page_8.pdf-1",
                program="greater(920, 95)",
                program_re="greater(920, 95)",
                answer="yes",
                exe_ans="yes",
            )
        ],
    }


# What ``clean_rows`` contains. The real config expects the full dataset's inventory, which a
# handful of rows cannot cover, so synthetic snapshots are checked against this instead.
CLEAN_EXPECTED = {
    "operations": ["add", "subtract", "divide", "greater", "table_sum"],
    "constants": ["const_100"],
    "argument_kinds": ["integer", "decimal", "reference", "constant", "row_name", "none"],
    "max_program_steps": 2,
}


def schema() -> Any:
    """The real files' column types (read from the dataset on 2026-11-15)."""
    import pyarrow as pa

    strings = pa.list_(pa.string())
    return pa.schema(
        [
            ("id", pa.string()),
            ("filename", pa.string()),
            ("pre_text", strings),
            ("post_text", strings),
            ("table", pa.list_(strings)),
            ("table_ori", pa.list_(strings)),
            ("question", pa.string()),
            ("answer", pa.string()),
            ("exe_ans", pa.string()),
            ("explanation", pa.string()),
            ("ann_table_rows", pa.list_(pa.int32())),
            ("ann_text_rows", pa.list_(pa.int32())),
            (
                "steps",
                pa.struct([(name, strings) for name in ("op", "arg1", "arg2", "res")]),
            ),
            ("program", pa.string()),
            ("program_re", pa.string()),
            ("gold_inds", pa.string()),
        ]
    )


def write_parquet(path: Path, rows: list[dict[str, Any]], file_schema: Any = None) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=file_schema or schema()), path)


def build_snapshot(
    root: Path,
    rows_by_split: dict[str, list[dict[str, Any]]] | None = None,
    place_files: bool = True,
    expected: dict[str, Any] | None = None,
) -> DataConfig:
    """Write a synthetic dataset and a matching ``data.yaml`` under ``root``; return the config.

    The files are written to ``<root>/source/``. With ``place_files`` they are also copied into
    the snapshot folder, as if already downloaded. ``expected`` replaces keys of the real
    config's ``expected`` block; by default it is ``CLEAN_EXPECTED``.
    """
    rows_by_split = clean_rows() if rows_by_split is None else rows_by_split
    real = yaml.safe_load((CONFIG_DIR / "data.yaml").read_text(encoding="utf-8"))
    finqa = real["finqa"]
    finqa["repo_id"] = "example/finqa-synthetic"
    finqa["former_repo_id"] = ""
    finqa["revision"] = REVISION
    finqa["files"] = []
    finqa["expected"].update(CLEAN_EXPECTED if expected is None else expected)
    source = root / "source"
    for split, rows in rows_by_split.items():
        relative = f"data/{split}-00000-of-00001.parquet"
        write_parquet(source / relative, rows)
        finqa["files"].append(
            {
                "official_split": split,
                "path": relative,
                "sha256": sha256_of(source / relative),
                "bytes": (source / relative).stat().st_size,
                "rows": len(rows),
            }
        )
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "data.yaml").write_text(yaml.safe_dump(real, sort_keys=False), encoding="utf-8")
    cfg = load_data_config(root=root, config_dir=config_dir)
    if place_files:
        for item in cfg.files:
            target = snapshot_path(cfg) / item.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source / item.path).read_bytes())
    return cfg


class FakeDownloader:
    """Stands in for the Hub: copies files from ``<root>/source/`` and records each call."""

    def __init__(self, source: Path, corrupt: tuple[str, ...] = (), fail: bool = False):
        self.source = Path(source)
        self.corrupt = corrupt
        self.fail = fail
        self.calls: list[tuple[str, str, str, bool]] = []

    def __call__(
        self, repo_id: str, filename: str, revision: str, local_dir: Path, force: bool
    ) -> None:
        self.calls.append((repo_id, filename, revision, force))
        if self.fail:
            raise ConnectionError("no route to host\nsecond line that must not be shown")
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        data = (self.source / filename).read_bytes()
        target.write_bytes(data + b"x" if filename in self.corrupt else data)

    @property
    def filenames(self) -> list[str]:
        return [call[1] for call in self.calls]


# --------------------------------------------------------------------------- split helpers

# Eleven invented companies of different sizes: 60 examples, so 65% is 39.
COMPANY_SIZES = {
    "AAA": 12, "BBB": 9, "CCC": 8, "DDD": 7, "EEE": 6, "FFF": 5,
    "GGG": 4, "HHH": 3, "III": 3, "JJJ": 2, "KKK": 1,
}  # fmt: skip

# Split settings sized for COMPANY_SIZES; everything else comes from the real config.
# With eleven groups the training side lands on 38 of 60, so the ratio tolerance is wider.
SMALL_SPLIT = {
    "dev_size": 8,
    "dev_size_tolerance": 8,
    "quick_dev_size": 3,
    "train_fraction_tolerance": 0.03,
}


def company_rows(sizes: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """Rows for several companies: two examples per page, two years per company."""
    rows = []
    for company, count in (COMPANY_SIZES if sizes is None else sizes).items():
        for number in range(count):
            page, index = number // 2 + 1, number % 2 + 1
            rows.append(row(f"{company}/{2015 + page % 2}/page_{page}.pdf-{index}"))
    return rows


def split_rows(sizes: dict[str, int] | None = None) -> dict[str, list[dict[str, Any]]]:
    """``company_rows`` dealt over the three official files, as the real data is."""
    rows = company_rows(sizes)
    return {"train": rows[0::3], "validation": rows[1::3], "test": rows[2::3]}


def to_examples(rows: list[dict[str, Any]], official_split: str = "train") -> list[Any]:
    """Records built directly from rows, for tests that need no files."""
    import re

    from qlora_finetuning.data.snapshot import to_example

    real = yaml.safe_load((CONFIG_DIR / "data.yaml").read_text(encoding="utf-8"))
    pattern = re.compile(real["finqa"]["expected"]["id_pattern"])
    return [to_example(item, official_split, pattern) for item in rows]


def build_split_config(root: Path, **overrides: Any) -> Any:
    """Write a ``split.yaml`` under ``<root>/config`` and load it with paths under ``root``.

    Starts from the real settings with ``SMALL_SPLIT`` applied; keyword arguments replace keys.
    """
    from qlora_finetuning.data.split import load_split_config

    raw = yaml.safe_load((CONFIG_DIR / "split.yaml").read_text(encoding="utf-8"))
    raw.update(SMALL_SPLIT)
    raw.update(overrides)
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "split.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_split_config(root=root, config_dir=config_dir)


def build_frozen_split(
    root: Path,
    write: bool = True,
    rows: dict[str, list[dict[str, Any]]] | None = None,
    **overrides: Any,
) -> tuple[Any, Any, Any]:
    """A synthetic snapshot, a split config, and the manifest built from them.

    Returns ``(split_cfg, data_cfg, manifest)``. With ``write`` the manifest is also written to
    its file under ``root``, as the build script would. ``rows`` replaces ``split_rows()``:
    the rows of each official file, for a test that needs some of them changed.
    """
    from qlora_finetuning.data import load_raw_examples
    from qlora_finetuning.data.split import build_split, write_manifest

    data_cfg = build_snapshot(root, split_rows() if rows is None else rows)
    cfg = build_split_config(root, **overrides)
    manifest = build_split(load_raw_examples(data_cfg), cfg, data_cfg)
    if write:
        write_manifest(manifest, cfg)
    return cfg, data_cfg, manifest


# --------------------------------------------------------------------------- golden records

# Four invented examples whose canonical records are frozen in tests/data/golden/.
GOLDEN_ROWS = {
    "text_and_table": row(
        "ACME/2015/page_10.pdf-1",
        gold_inds={"text_2": "ignored", "table_1": "ignored", "text_0": "ignored"},
        program="subtract(920, 95), divide(#0, 95)",
        exe_ans="8.68421",
    ),
    "table_only": row("ACME/2015/page_10.pdf-2"),
    "table_operation": row(
        "BOLT/2016/page_3.pdf-1",
        program="table_max(xylophone sales, none), multiply(#0, const_1000000)",
        exe_ans="9.2e+08",
    ),
    "yes_no": row(
        "CRUX/2017/page_8.pdf-1",
        gold_inds={"text_1": "ignored"},
        program="divide(6.9%, const_100), greater(#0, 95)",
        exe_ans="no",
    ),
}
GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_SEPARATOR = "\n=====\n"


def golden_text(record: Any) -> str:
    """How a record is stored in a golden file: the user text, a separator, the target."""
    return f"{record.user}{GOLDEN_SEPARATOR}{record.target}\n"


# --------------------------------------------------------------------------- stand-in tokenizer


class FakeTokenizer:
    """Stands in for a Hugging Face tokenizer: one token per space-separated word.

    Its chat template wraps each message in three marker words and ends with a two-word reply
    marker and, with thinking off, a two-word empty think block, so it adds ten tokens to
    every prompt.
    """

    eos_token = " <end>"  # with its space, so it is counted as one more word
    chat_template = "fake template v1"
    name_or_path = "fake/tokenizer"
    THINKING_OFF_SUFFIX = "<think> </think> "

    def __init__(self, vocabulary: dict[str, int] | None = None):
        self._vocabulary = vocabulary or {"alpha": 0, "beta": 1}

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocabulary)

    def __call__(self, texts: Any, add_special_tokens: bool = False) -> dict[str, Any]:
        if isinstance(texts, str):
            return {"input_ids": [0] * len(texts.split())}
        return {"input_ids": [[0] * len(text.split()) for text in texts]}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> str:
        text = "".join(f"<start> {m['role']} {m['content']} <stop> " for m in messages)
        if add_generation_prompt:
            text += "<start> assistant "
        if not enable_thinking:
            text += self.THINKING_OFF_SUFFIX
        return text


def fake_budget(root: Path, tokenizer: Any = None, **changes: Any) -> Any:
    """The real token budget settings, pointed at ``root`` and at the stand-in tokenizer."""
    import dataclasses

    from qlora_finetuning.data.token_budget import load_token_budget_config, tokenizer_fingerprint

    values = {
        "fingerprint": tokenizer_fingerprint(tokenizer or FakeTokenizer()),
        "thinking_off_suffix": FakeTokenizer.THINKING_OFF_SUFFIX,
    }
    values.update(changes)
    return dataclasses.replace(load_token_budget_config(root=root), **values)
