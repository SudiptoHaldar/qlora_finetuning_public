"""The pinned FinQA snapshot: download, verification, and the raw reader (req_001 v1.01).

The dataset is three Parquet files at one commit of one Hugging Face repository. Their
checksums, sizes, and row counts are in ``config/data.yaml`` and are checked after download
and again on every load, so a changed file can never be read by accident.

Usage::

    from qlora_finetuning.data import load_raw_examples

    examples = load_raw_examples()          # 8,281 records: train, validation, test order
    first = examples[0]
    first.id, first.question, first.program, first.gold_inds

The reader is **raw**: it returns every row of every official split, including rows the frozen
split (v1.02) assigns to test. Code outside the data package uses the split-aware loaders that
v1.02 puts in front of it.

Nothing is dropped, deduplicated, or repaired. Fetch the files first with
``python -m scripts.finqa_fetch``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from qlora_finetuning.settings import REPO_ROOT, ConfigError, load_config

REQUIRED_KEYS = (
    "finqa.repo_id",
    "finqa.revision",
    "finqa.snapshot_dir",
    "finqa.report_dir",
    "finqa.report_max_ids_listed",
    "finqa.files",
    "finqa.expected.columns",
    "finqa.expected.id_pattern",
    "finqa.expected.operations",
    "finqa.expected.constants",
    "finqa.expected.argument_kinds",
    "finqa.expected.arguments_per_step",
    "finqa.expected.max_program_steps",
)
SNAPSHOT_RECORD = "snapshot.json"
FETCH_COMMAND = "python -m scripts.finqa_fetch"
INSTALL_HINT = "install the training stack: docs/runbook.md section 3.1"

_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHUNK = 1024 * 1024

# download(repo_id, filename, revision, local_dir, force): put the file at local_dir/filename.
Downloader = Callable[[str, str, str, Path, bool], Any]


class SnapshotError(ConfigError):
    """The snapshot is missing, incomplete, or does not match the recorded checksums."""


@dataclass(frozen=True)
class SnapshotFile:
    """One dataset file and what it must look like."""

    official_split: str
    path: str  # relative to the repository on the Hub and to the snapshot folder
    sha256: str
    size: int
    rows: int


@dataclass(frozen=True)
class DataConfig:
    """``config/data.yaml`` with its paths resolved."""

    root: Path
    repo_id: str
    former_repo_id: str
    revision: str
    snapshot_dir: Path
    report_dir: Path
    report_max_ids_listed: int
    files: tuple[SnapshotFile, ...]
    expected: dict[str, Any]

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(item.official_split for item in self.files)


@dataclass(frozen=True)
class FileCheck:
    """The result of checking one snapshot file."""

    official_split: str
    path: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class RawExample:
    """One FinQA row, exactly as stored, with plain Python values.

    ``gold_inds`` is the stored JSON text parsed into a dict in its original key order, or
    ``None`` if the text is not a JSON object. ``ticker``, ``year``, ``page``, and ``index``
    come from the id and are ``None`` when the id does not match the expected pattern.
    """

    id: str
    filename: str
    official_split: str
    ticker: str | None
    year: int | None
    page: int | None
    index: int | None
    pre_text: list[str]
    post_text: list[str]
    table: list[list[str]]
    table_ori: list[list[str]]
    question: str
    answer: str
    exe_ans: str
    explanation: str
    ann_table_rows: list[int]
    ann_text_rows: list[int]
    steps: dict[str, list[str]]
    program: str
    program_re: str
    gold_inds: dict[str, Any] | None

    @property
    def report(self) -> str | None:
        """The annual report the example comes from: ``<ticker>/<year>``."""
        if self.ticker is None or self.year is None:
            return None
        return f"{self.ticker}/{self.year}"


# --------------------------------------------------------------------------- config


def _positive_int(value: Any, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"config data.yaml: {key} must be a positive integer, got {value!r}")
    return value


def _snapshot_file(entry: Any, position: int) -> SnapshotFile:
    key = f"finqa.files[{position}]"
    if not isinstance(entry, dict):
        raise ConfigError(f"config data.yaml: {key} must be a mapping")
    missing = [
        name for name in ("official_split", "path", "sha256", "bytes", "rows") if name not in entry
    ]
    if missing:
        raise ConfigError(f"config data.yaml: {key} is missing: {', '.join(missing)}")
    path = str(entry["path"])
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or "\\" in path or ":" in path or not path:
        raise ConfigError(f"config data.yaml: {key}.path must be a relative path, got {path!r}")
    checksum = str(entry["sha256"])
    if not _SHA256.fullmatch(checksum):
        raise ConfigError(f"config data.yaml: {key}.sha256 must be 64 lower-case hex characters")
    return SnapshotFile(
        official_split=str(entry["official_split"]),
        path=path,
        sha256=checksum,
        size=_positive_int(entry["bytes"], f"{key}.bytes"),
        rows=_positive_int(entry["rows"], f"{key}.rows"),
    )


def load_data_config(
    root: Path | str = REPO_ROOT, config_dir: Path | str | None = None
) -> DataConfig:
    """Read ``config/data.yaml``. Relative paths in it resolve under ``root``.

    Tests pass a temporary folder as ``root`` and their own ``config_dir``.
    """
    raw = load_config("data", required=REQUIRED_KEYS, config_dir=config_dir)["finqa"]
    revision = str(raw["revision"])
    if not _COMMIT.fullmatch(revision):
        raise ConfigError(
            "config data.yaml: finqa.revision must be a full 40-character commit hash, "
            f"not a branch name or a short hash: {revision!r}"
        )
    if not isinstance(raw["files"], list):
        raise ConfigError("config data.yaml: finqa.files must be a list")
    files = tuple(_snapshot_file(entry, i) for i, entry in enumerate(raw["files"]))
    for field_name in ("official_split", "path"):
        values = [getattr(item, field_name) for item in files]
        repeated = sorted({value for value in values if values.count(value) > 1})
        if repeated:
            raise ConfigError(
                f"config data.yaml: finqa.files repeats {field_name}: {', '.join(repeated)}"
            )
    expected = raw["expected"]
    try:
        re.compile(str(expected["id_pattern"]))
    except re.error as exc:
        raise ConfigError(
            f"config data.yaml: finqa.expected.id_pattern is not valid: {exc}"
        ) from exc
    root = Path(root)
    return DataConfig(
        root=root,
        repo_id=str(raw["repo_id"]),
        former_repo_id=str(raw.get("former_repo_id") or ""),
        revision=revision,
        snapshot_dir=root / str(raw["snapshot_dir"]),
        report_dir=root / str(raw["report_dir"]),
        report_max_ids_listed=_positive_int(
            raw["report_max_ids_listed"], "finqa.report_max_ids_listed"
        ),
        files=files,
        expected=expected,
    )


# --------------------------------------------------------------------------- files


def snapshot_path(cfg: DataConfig) -> Path:
    """The folder holding this revision's files: ``<snapshot_dir>/<revision>/``."""
    return cfg.snapshot_dir / cfg.revision


def sha256_of(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _check_file(folder: Path, item: SnapshotFile) -> FileCheck:
    path = folder / item.path

    def failed(detail: str) -> FileCheck:
        return FileCheck(item.official_split, item.path, False, detail)

    if not path.is_file():
        return failed("file is missing")
    size = path.stat().st_size
    if size != item.size:
        return failed(f"size is {size:,} bytes, expected {item.size:,}")
    actual = sha256_of(path)
    if actual != item.sha256:
        return failed(f"sha256 is {actual}, expected {item.sha256}")
    try:
        rows = _row_count(path)
    except Exception as exc:  # a file with the right bytes that is not Parquet: config is wrong
        return failed(f"cannot be read as Parquet: {type(exc).__name__}")
    if rows != item.rows:
        return failed(f"has {rows:,} rows, expected {item.rows:,}")
    return FileCheck(
        item.official_split,
        item.path,
        True,
        f"{size:,} bytes, {rows:,} rows, sha256 {actual[:12]}...",
    )


def verify_snapshot(cfg: DataConfig) -> list[FileCheck]:
    """Check every file's presence, size, checksum, and row count. Reads only; never raises."""
    folder = snapshot_path(cfg)
    return [_check_file(folder, item) for item in cfg.files]


def _require_verified(cfg: DataConfig, hint: str) -> None:
    failures = [check for check in verify_snapshot(cfg) if not check.ok]
    if failures:
        lines = "; ".join(f"{check.path}: {check.detail}" for check in failures)
        raise SnapshotError(
            f"the FinQA snapshot in {snapshot_path(cfg)} is not valid ({lines}). {hint}"
        )


def _hub_download(repo_id: str, filename: str, revision: str, local_dir: Path, force: bool) -> None:
    """The real downloader. ``huggingface_hub`` is training stack, so it is imported here."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SnapshotError(f"huggingface_hub is not installed; {INSTALL_HINT}") from exc
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        local_dir=str(local_dir),
        force_download=force,
    )


def _library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("huggingface_hub", "pyarrow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _write_snapshot_record(cfg: DataConfig, downloaded: list[str], now: datetime | None) -> None:
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    record = {
        "schema": 1,
        "repo_id": cfg.repo_id,
        "former_repo_id": cfg.former_repo_id,
        "revision": cfg.revision,
        "fetched_at": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "downloaded": downloaded,
        "files": [
            {
                "official_split": item.official_split,
                "path": item.path,
                "sha256": item.sha256,
                "bytes": item.size,
                "rows": item.rows,
            }
            for item in cfg.files
        ],
        "libraries": _library_versions(),
    }
    path = snapshot_path(cfg) / SNAPSHOT_RECORD
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")


def fetch_snapshot(
    cfg: DataConfig,
    download: Downloader | None = None,
    force: bool = False,
    now: datetime | None = None,
) -> list[str]:
    """Download the files that are missing, verify all of them, and return what was downloaded.

    A file that is present and correct is not downloaded again. A file that is present but
    wrong is left alone and reported, unless ``force`` is set, which downloads every file
    again. Nothing is ever deleted. ``download`` is the function that fetches one file; the
    default uses ``huggingface_hub``.
    """
    download = download or _hub_download
    folder = snapshot_path(cfg)
    before = {check.path: check for check in verify_snapshot(cfg)}
    if not force:
        wrong = [c for c in before.values() if not c.ok and (folder / c.path).exists()]
        if wrong:
            lines = "; ".join(f"{check.path}: {check.detail}" for check in wrong)
            raise SnapshotError(
                f"a file in {folder} does not match config/data.yaml ({lines}). Nothing was "
                "changed. If the file is damaged, run the fetch again with --force"
            )
    wanted = [item for item in cfg.files if force or not before[item.path].ok]
    folder.mkdir(parents=True, exist_ok=True)
    for item in wanted:
        try:
            download(cfg.repo_id, item.path, cfg.revision, folder, force)
        except SnapshotError:
            raise
        except Exception as exc:
            first_line = (str(exc).strip().splitlines() or ["(no message)"])[0]
            raise SnapshotError(
                f"download of {item.path} from {cfg.repo_id} failed: "
                f"{type(exc).__name__}: {first_line}"
            ) from exc
    _require_verified(
        cfg, "Nothing was deleted. Compare config/data.yaml with the repository on the Hub"
    )
    downloaded = [item.path for item in wanted]
    if downloaded or not (folder / SNAPSHOT_RECORD).is_file():
        _write_snapshot_record(cfg, downloaded, now)
    return downloaded


# --------------------------------------------------------------------------- schema


def type_name(data_type: Any) -> str:
    """A short, stable name for a pyarrow type: ``list[string]``, ``struct{op: list[string]}``."""
    import pyarrow.types as pat

    if pat.is_string(data_type) or pat.is_large_string(data_type):
        return "string"
    if pat.is_list(data_type) or pat.is_large_list(data_type):
        return f"list[{type_name(data_type.value_type)}]"
    if pat.is_struct(data_type):
        inner = ", ".join(f"{field.name}: {type_name(field.type)}" for field in data_type)
        return f"struct{{{inner}}}"
    return str(data_type)


def read_columns(cfg: DataConfig) -> dict[str, dict[str, str]]:
    """Column names and types of each file, keyed by official split, in file order."""
    import pyarrow.parquet as pq

    folder = snapshot_path(cfg)
    return {
        item.official_split: {
            field.name: type_name(field.type) for field in pq.read_schema(folder / item.path)
        }
        for item in cfg.files
    }


# --------------------------------------------------------------------------- reader


def _parse_gold_inds(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def to_example(row: dict[str, Any], official_split: str, id_pattern: re.Pattern[str]) -> RawExample:
    """Build a record from one row as pyarrow returns it. Missing columns become ``None``."""
    match = id_pattern.fullmatch(row.get("id") or "")
    ticker, year, page, index = (
        (match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4)))
        if match
        else (None, None, None, None)
    )
    return RawExample(
        id=row.get("id"),
        filename=row.get("filename"),
        official_split=official_split,
        ticker=ticker,
        year=year,
        page=page,
        index=index,
        pre_text=row.get("pre_text"),
        post_text=row.get("post_text"),
        table=row.get("table"),
        table_ori=row.get("table_ori"),
        question=row.get("question"),
        answer=row.get("answer"),
        exe_ans=row.get("exe_ans"),
        explanation=row.get("explanation"),
        ann_table_rows=row.get("ann_table_rows"),
        ann_text_rows=row.get("ann_text_rows"),
        steps=row.get("steps"),
        program=row.get("program"),
        program_re=row.get("program_re"),
        gold_inds=_parse_gold_inds(row.get("gold_inds")),
    )


def load_raw_examples(cfg: DataConfig | None = None) -> list[RawExample]:
    """Verify the snapshot, then return every example of every official split.

    Order: the files in config order (train, validation, test), each in file order. Raises
    ``SnapshotError`` if a file is missing or does not match its recorded checksum.
    """
    import pyarrow.parquet as pq

    cfg = cfg or load_data_config()
    _require_verified(cfg, f"Fetch it with: {FETCH_COMMAND}")
    id_pattern = re.compile(str(cfg.expected["id_pattern"]))
    folder = snapshot_path(cfg)
    examples: list[RawExample] = []
    for item in cfg.files:
        rows = pq.read_table(folder / item.path).to_pylist()
        examples.extend(to_example(row, item.official_split, id_pattern) for row in rows)
    return examples
