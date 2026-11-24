"""The cost ledger: append-only records of what each run consumed, in the units AWS bills.

The ledger stores quantities (seconds, instance counts, tokens, questions). It never multiplies
by a price; the cost model does that from ``config/`` rates. Every record says where its
numbers came from, because not everything AWS bills can be read back from an API.

Usage::

    from qlora_finetuning.tracking import Ledger, start_run

    with start_run("eval", "b0-pro-dev", path="B", arm="B0-Pro", where="bedrock",
                   scope="dev", configs=["aws"]) as run:
        run.record(phase="eval", model_id="amazon.nova-pro-v1:0", input_tokens=812_000,
                   output_tokens=41_000, examples=500, wall_clock_seconds=930.0,
                   source="aws_api", source_detail="bedrock-runtime:Converse usage, summed")

    for row in Ledger().summarize(by=("arm", "phase")):
        print(row)

A wrong record is never edited. Append a new one with ``replaces=<record_id>``; readers then
skip the old one.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from qlora_finetuning.tracking.config import (
    TrackingConfig,
    TrackingError,
    check_no_secrets,
    load_tracking_config,
    timestamp,
)
from qlora_finetuning.tracking.jsonl import append_line, read_lines

if TYPE_CHECKING:
    from qlora_finetuning.tracking.registry import Run

SCHEMA = 1

# Where a record's numbers came from.
#   measured  our own clock or counter
#   aws_api   a field AWS returned (billable seconds, Converse usage)
#   computed  derived, for example training tokens = tokens in the file x epochs
#   bill      read from the AWS bill by the owner
SOURCES = ("measured", "aws_api", "computed", "bill")

INTEGER_QUANTITIES = (
    "billable_seconds",
    "input_tokens",
    "output_tokens",
    "training_tokens",
    "examples",
)
TOKEN_QUANTITIES = ("input_tokens", "output_tokens", "training_tokens")
QUANTITIES = ("wall_clock_seconds", *INTEGER_QUANTITIES, "billed_amount_usd")
GROUP_FIELDS = ("run_id", "phase", "path", "arm", "where", "source", "model_id", "instance_type")
FINISHED_JOB_STATES = ("Completed", "Failed", "Stopped")


@dataclass(frozen=True)
class LedgerRecord:
    """One line of the ledger. Fields are described in the v0.04 spec, section 4.3."""

    run_id: str
    phase: str
    path: str
    arm: str
    where: str
    source: str
    source_detail: str = ""
    instance_type: str | None = None
    instance_count: int | None = None
    model_id: str | None = None
    wall_clock_seconds: float | None = None
    billable_seconds: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    training_tokens: int | None = None
    examples: int | None = None
    billed_amount_usd: str | None = None
    external_id: str | None = None
    replaces: str | None = None
    note: str = ""
    record_id: str = field(default_factory=lambda: uuid4().hex)
    recorded_at: str = field(default_factory=timestamp)
    schema: int = SCHEMA

    def to_dict(self) -> dict[str, Any]:
        """A plain dict without empty fields, so ledger lines stay short."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        return {key: value for key, value in data.items() if value not in (None, "")}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LedgerRecord:
        if data.get("schema") != SCHEMA:
            raise TrackingError(f"unknown ledger schema {data.get('schema')!r}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise TrackingError(f"unknown ledger field(s): {', '.join(unknown)}")
        try:
            return cls(**data)
        except TypeError as exc:
            raise TrackingError(f"incomplete ledger record: {exc}") from exc


def _amount(value: Any) -> Decimal:
    """Parse a dollar amount given as text, Decimal, or int. Floats are refused."""
    if isinstance(value, bool) or isinstance(value, float):
        raise TrackingError("billed_amount_usd must be text such as '12.34', not a float")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise TrackingError(f"billed_amount_usd is not a number: {value!r}") from exc
    if not amount.is_finite():
        raise TrackingError(f"billed_amount_usd is not a number: {value!r}")
    return amount


def validate(record: LedgerRecord, cfg: TrackingConfig, known_ids: Iterable[str] = ()) -> None:
    """Check ``record`` against every ledger rule. Raises ``TrackingError`` naming the rule."""
    known = set(known_ids)
    if record.schema != SCHEMA:
        raise TrackingError(f"unknown ledger schema {record.schema!r}")
    if not record.run_id:
        raise TrackingError("a ledger record needs a run_id")
    cfg.check("phases", record.phase)
    cfg.check("paths", record.path)
    cfg.check("arms", record.arm)
    cfg.check("places", record.where)
    if record.source not in SOURCES:
        raise TrackingError(f"{record.source!r} is not a source; use one of: {', '.join(SOURCES)}")

    for name in INTEGER_QUANTITIES + ("instance_count",):
        value = getattr(record, name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise TrackingError(f"{name} must be a whole number, got {value!r}")
    seconds = record.wall_clock_seconds
    if seconds is not None and (isinstance(seconds, bool) or not isinstance(seconds, (int, float))):
        raise TrackingError(f"wall_clock_seconds must be a number, got {seconds!r}")

    present = [name for name in QUANTITIES if getattr(record, name) is not None]
    if not present:
        raise TrackingError(f"a ledger record needs at least one quantity: {', '.join(QUANTITIES)}")
    for name in present:
        value = getattr(record, name)
        number = _amount(value) if name == "billed_amount_usd" else value
        if number < 0:
            raise TrackingError(f"{name} cannot be negative, got {value!r}")

    if record.instance_count is not None and record.instance_count < 1:
        raise TrackingError(f"instance_count must be at least 1, got {record.instance_count}")
    if record.billable_seconds is not None and not (record.instance_type and record.instance_count):
        raise TrackingError("billable_seconds needs instance_type and instance_count")
    if any(getattr(record, name) is not None for name in TOKEN_QUANTITIES) and not record.model_id:
        raise TrackingError("a token count needs model_id")
    if record.billed_amount_usd is not None and record.source != "bill":
        raise TrackingError("billed_amount_usd is only allowed with source 'bill'")

    if record.record_id in known:
        raise TrackingError(f"record_id {record.record_id} is already in the ledger")
    if record.replaces is not None and record.replaces not in known:
        raise TrackingError(f"replaces names a record that is not in the ledger: {record.replaces}")
    check_no_secrets(record.to_dict(), "the ledger record")


class Ledger:
    """Reads and appends the ledger file named in the tracking config."""

    def __init__(self, cfg: TrackingConfig | None = None) -> None:
        self.cfg = cfg or load_tracking_config()

    def read(self) -> list[LedgerRecord]:
        """Every record, in the order written."""
        records = []
        name = self.cfg.ledger_file.name
        for number, data in enumerate(read_lines(self.cfg.ledger_file), start=1):
            try:
                records.append(LedgerRecord.from_dict(data))
            except TrackingError as exc:
                raise TrackingError(f"{name} record {number}: {exc}") from exc
        return records

    def append(self, record: LedgerRecord) -> LedgerRecord:
        """Validate ``record`` and append it. Nothing is written if it breaks a rule."""
        if isinstance(record.billed_amount_usd, (Decimal, int)) and not isinstance(
            record.billed_amount_usd, bool
        ):
            record = _with_amount_as_text(record)
        validate(record, self.cfg, (existing.record_id for existing in self.read()))
        append_line(self.cfg.ledger_file, record.to_dict())
        return record

    def effective(self) -> list[LedgerRecord]:
        """The records that count: those no later record replaces."""
        records = self.read()
        replaced = {record.replaces for record in records if record.replaces}
        return [record for record in records if record.record_id not in replaced]

    def summarize(self, by: Iterable[str] = ("arm", "phase")) -> list[dict[str, Any]]:
        """Totals of every quantity per group, over the effective records.

        ``billable_instance_seconds`` is ``billable_seconds`` times ``instance_count``, which is
        what AWS charges for. ``sources`` counts the records by where their numbers came from.
        """
        by = tuple(by)
        bad = [name for name in by if name not in GROUP_FIELDS]
        if bad or not by:
            raise TrackingError(
                f"cannot group by {', '.join(bad) or 'nothing'}; use: {', '.join(GROUP_FIELDS)}"
            )

        groups: dict[tuple, dict[str, Any]] = {}
        for record in self.effective():
            key = tuple(getattr(record, name) or "-" for name in by)
            row = groups.get(key)
            if row is None:
                row = {name: value for name, value in zip(by, key, strict=True)}
                row.update(
                    records=0,
                    wall_clock_seconds=0.0,
                    billable_instance_seconds=0,
                    input_tokens=0,
                    output_tokens=0,
                    training_tokens=0,
                    examples=0,
                    billed_amount_usd=Decimal("0"),
                    sources={},
                )
                groups[key] = row
            row["records"] += 1
            row["wall_clock_seconds"] += record.wall_clock_seconds or 0.0
            row["billable_instance_seconds"] += (record.billable_seconds or 0) * (
                record.instance_count or 0
            )
            for name in ("input_tokens", "output_tokens", "training_tokens", "examples"):
                row[name] += getattr(record, name) or 0
            if record.billed_amount_usd is not None:
                row["billed_amount_usd"] += _amount(record.billed_amount_usd)
            row["sources"][record.source] = row["sources"].get(record.source, 0) + 1

        rows = [groups[key] for key in sorted(groups)]
        for row in rows:
            row["billed_amount_usd"] = str(row["billed_amount_usd"])
        return rows


def _with_amount_as_text(record: LedgerRecord) -> LedgerRecord:
    data = {f.name: getattr(record, f.name) for f in fields(record)}
    data["billed_amount_usd"] = str(record.billed_amount_usd)
    return LedgerRecord(**data)


# --------------------------------------------------------------------------- AWS responses


def sum_converse_usage(usages: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Add up the ``usage`` blocks of many ``bedrock-runtime:Converse`` responses."""
    totals = {"inputTokens": 0, "outputTokens": 0}
    for usage in usages:
        for key in totals:
            value = usage.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TrackingError(f"Converse usage needs a whole number for {key}, got {value!r}")
            totals[key] += value
    return totals


def record_from_sagemaker_job(description: dict[str, Any], run: Run, phase: str) -> LedgerRecord:
    """Build (not append) the ledger record for a finished SageMaker training job.

    ``description`` is the response of ``sagemaker:DescribeTrainingJob``. ``BillableTimeInSeconds``
    is per instance, as AWS reports it; the summary multiplies by the instance count. A stopped
    or failed job is still billed, so it is recorded too.
    """
    name = description.get("TrainingJobName", "the job")
    status = description.get("TrainingJobStatus")
    if status not in FINISHED_JOB_STATES:
        raise TrackingError(f"{name} has not finished (status {status!r}); nothing to record yet")
    billable = description.get("BillableTimeInSeconds")
    if billable is None:
        raise TrackingError(f"{name} reports no BillableTimeInSeconds")
    resources = description.get("ResourceConfig") or {}
    if not resources.get("InstanceType") or not resources.get("InstanceCount"):
        raise TrackingError(f"{name} reports no instance type or count")
    training_time = description.get("TrainingTimeInSeconds")
    return run.new_record(
        phase=phase,
        instance_type=str(resources["InstanceType"]),
        instance_count=int(resources["InstanceCount"]),
        billable_seconds=int(billable),
        wall_clock_seconds=float(training_time) if training_time is not None else None,
        source="aws_api",
        source_detail="sagemaker:DescribeTrainingJob BillableTimeInSeconds (per instance)",
        external_id=description.get("TrainingJobName"),
        note=f"job status {status}",
    )


def record_from_converse_usage(
    totals: dict[str, Any],
    run: Run,
    model_id: str,
    examples: int,
    wall_clock_seconds: float | None = None,
    phase: str = "eval",
) -> LedgerRecord:
    """Build (not append) the ledger record for a run's Bedrock inference tokens.

    ``totals`` is the sum of the ``usage`` blocks of every ``Converse`` call in the run; see
    ``sum_converse_usage``. One record per run, not per question.
    """
    checked = sum_converse_usage([totals])
    return run.new_record(
        phase=phase,
        model_id=model_id,
        input_tokens=checked["inputTokens"],
        output_tokens=checked["outputTokens"],
        examples=examples,
        wall_clock_seconds=wall_clock_seconds,
        source="aws_api",
        source_detail="bedrock-runtime:Converse usage, summed over the run",
    )
