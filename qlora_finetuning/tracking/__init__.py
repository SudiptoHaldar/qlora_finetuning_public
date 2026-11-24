"""Run registry and cost ledger (req_000 v0.04).

Every training or evaluation entry point starts a run and records what it consumed::

    from qlora_finetuning.tracking import Ledger, start_run

    with start_run("eval", "a0-quick-dev", path="A", arm="A0", where="local", scope="dev",
                   configs=["local_env"]) as run:
        run.record(phase="eval", wall_clock_seconds=412.0, examples=200, source="measured",
                   source_detail="time.perf_counter around the evaluation loop")

    rows = Ledger().summarize(by=("arm", "phase"))

See ``registry`` for runs and manifests, ``ledger`` for records, rules, and the functions that
turn an AWS response into a record.
"""

from qlora_finetuning.tracking.config import (
    TrackingConfig,
    TrackingError,
    load_tracking_config,
)
from qlora_finetuning.tracking.ledger import (
    SOURCES,
    Ledger,
    LedgerRecord,
    record_from_converse_usage,
    record_from_sagemaker_job,
    sum_converse_usage,
)
from qlora_finetuning.tracking.registry import (
    GitInfo,
    Run,
    list_runs,
    load_manifest,
    new_run_id,
    read_git_info,
    start_run,
)

__all__ = [
    "SOURCES",
    "GitInfo",
    "Ledger",
    "LedgerRecord",
    "Run",
    "TrackingConfig",
    "TrackingError",
    "list_runs",
    "load_manifest",
    "load_tracking_config",
    "new_run_id",
    "read_git_info",
    "record_from_converse_usage",
    "record_from_sagemaker_job",
    "start_run",
    "sum_converse_usage",
]
