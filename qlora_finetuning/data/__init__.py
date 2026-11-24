"""FinQA snapshot, split manifest, serialization, and token budget (req_001).

Built so far: the pinned snapshot with its raw reader (``snapshot``, v1.01), the schema
report (``schema_report``, v1.01), and the frozen split with its loaders and test lock
(``split``, ``split_loader``, ``split_report``, v1.02), and canonical serialization
(``serialization``, ``serialization_report``, v1.03), and the token budget (``token_budget``,
``token_budget_report``, v1.04), which also holds the two length limits the paths read.

Outside this package, get examples with ``load_split``. ``load_raw_examples`` returns every
row, including test rows, and is for this package only; a guard test enforces that.
"""

from qlora_finetuning.data.serialization import (
    CanonicalRecord,
    PromptDefinition,
    SerializationError,
    build_record,
    load_prompt,
    serialize_split,
)
from qlora_finetuning.data.snapshot import (
    DataConfig,
    FileCheck,
    RawExample,
    SnapshotError,
    SnapshotFile,
    fetch_snapshot,
    load_data_config,
    load_raw_examples,
    read_columns,
    snapshot_path,
    verify_snapshot,
)
from qlora_finetuning.data.split import SplitConfig, SplitError, load_split_config
from qlora_finetuning.data.split_loader import (
    TestLockedError,
    is_test_unlocked,
    load_split,
    manifest_hash,
    split_ids,
)

__all__ = [
    "CanonicalRecord",
    "DataConfig",
    "FileCheck",
    "PromptDefinition",
    "RawExample",
    "SnapshotError",
    "SerializationError",
    "SnapshotFile",
    "SplitConfig",
    "SplitError",
    "TestLockedError",
    "build_record",
    "fetch_snapshot",
    "is_test_unlocked",
    "load_data_config",
    "load_prompt",
    "load_raw_examples",
    "load_split",
    "load_split_config",
    "manifest_hash",
    "read_columns",
    "serialize_split",
    "snapshot_path",
    "split_ids",
    "verify_snapshot",
]
