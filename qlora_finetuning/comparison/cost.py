"""The cost model: what each arm cost, and what it would cost to use (req_005 v5.02).

Every dollar figure is a quantity from the ledger times a price from ``config/pricing.yaml``,
as a ``Decimal``. The one exception is a ledger record read from the AWS bill, whose amount
is used as it stands. Nothing here calls AWS, starts a run, or reads a split.

What it works out:

- **Categories.** Each effective ledger record goes into exactly one of ``train``,
  ``select``, ``test``, ``development``, ``no_charge``. Which run is which comes from the
  committed records (the six-arm record, the job record, the selection record, the tuning
  record), never from an arm name alone.
- **Per arm.** The test pass, the cost per 1,000 questions (for a Qwen arm both as run and
  steady, without the job's start-up), and the cost per 1,000 correct answers.
- **Serving Qwen.** A batch job at several sizes; an always-on endpoint at several shares
  of the throughput the test job measured.
- **Break-evens.** The monthly volume at which a Qwen arm and a Nova arm cost the same,
  each marked reachable or not against the instance's capacity and the account's limits.
- **Against the bill.** Per service, the model's total beside what the owner read.

Usage::

    from qlora_finetuning.comparison.cost import CostInputs, build_record

    record = build_record(inputs, cfg)            # checks K1 to K8 and every table
    record["passed"], record["provisional"]

``cost_scenarios.py`` holds the serving scenarios and the break-evens, ``cost_bill.py`` the
owner's bill entries, and ``cost_report.py`` renders and writes the record.
``scripts/comparison_cost.py`` gathers the inputs. This module imports neither boto3 nor
the training stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from qlora_finetuning.comparison.cost_config import CostConfig, InstancePrices
from qlora_finetuning.comparison.cost_scenarios import break_evens, serving
from qlora_finetuning.frontier.pricing import MILLION, Pricing, inference_cost

SCHEMA = 1
COMMAND = "python -m scripts.comparison_cost"

TRAIN, SELECT, TEST, DEVELOPMENT, NO_CHARGE = "train", "select", "test", "development", "no_charge"
CATEGORIES = (TRAIN, SELECT, TEST, DEVELOPMENT, NO_CHARGE)
LOCAL, SAGEMAKER, BEDROCK = "local", "sagemaker", "bedrock"
BILLED, LIST_PRICE, ESTIMATE, THROUGH_JOB, FREE = (
    "billed",
    "list price",
    "estimate",
    "priced through its job",
    "no AWS charge",
)
PATHS = ("A", "B")

SECONDS_PER_HOUR = Decimal(3600)
THOUSAND = Decimal(1000)
ZERO = Decimal(0)
SIX_PLACES = Decimal("0.000001")
RUN_DATE = re.compile(r"^(\d{4})(\d{2})(\d{2})T\d{6}Z-")


@dataclass
class CostInputs:
    """Everything the model reads. Passed in, so a test needs no file."""

    ledger: list[Any]  # the ledger's effective records
    six_arm: dict[str, Any] | None  # the six-arm record of the test runs
    jobs: dict[str, Any]  # the SageMaker job record
    selection: dict[str, Any] | None  # the adapter selection record
    tuning: list[dict[str, Any]]  # the tuning record's attempts
    pricing: Pricing  # Bedrock prices
    instances: InstancePrices  # SageMaker training and hosting prices
    bills: list[dict[str, Any]] = field(default_factory=list)  # the owner's entries
    committed: dict[str, Decimal] = field(default_factory=dict)  # what the cap checks count
    caps: dict[str, Decimal] = field(default_factory=dict)


@dataclass(frozen=True)
class Placed:
    """One ledger record with its category and its price."""

    record: Any
    category: str
    amount: Decimal
    basis: str
    test_arm: str | None = None
    problem: str | None = None

    @property
    def date(self) -> str | None:
        return run_date(self.record.run_id)


@dataclass
class Index:
    """Which run is which, from the committed records."""

    test_runs: dict[str, str] = field(default_factory=dict)  # run id -> arm
    test_jobs: dict[str, str] = field(default_factory=dict)  # job name -> arm
    jobs: dict[str, dict[str, Any]] = field(default_factory=dict)  # by name
    run_job: dict[str, str] = field(default_factory=dict)  # run id -> its job's name
    train_job: str | None = None
    select_job: str | None = None
    tuning: dict[str, str] = field(default_factory=dict)  # job ARN -> arm, current attempts


def money(amount: Decimal) -> str:
    """An amount as the record stores it: text, six decimal places."""
    return str(amount.quantize(SIX_PLACES))


def run_date(run_id: str) -> str | None:
    """The day a run started, from its id. A bill record is written days after its job."""
    match = RUN_DATE.match(str(run_id))
    return "-".join(match.groups()) if match else None


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


# --------------------------------------------------------------------------- categories


def current_attempts(tuning: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The latest attempt of each model: the one its tuned arm is."""
    latest: dict[str, dict[str, Any]] = {}
    for attempt in tuning:
        model = str(attempt.get("model"))
        if model not in latest or int(attempt.get("attempt", 0)) > int(
            latest[model].get("attempt", 0)
        ):
            latest[model] = attempt
    return list(latest.values())


def build_index(inputs: CostInputs) -> Index:
    index = Index()
    for arm, entry in ((inputs.six_arm or {}).get("arms") or {}).items():
        for run_id in [entry.get("run_id"), *(entry.get("continues") or [])]:
            if run_id:
                index.test_runs[str(run_id)] = arm
        for job in entry.get("jobs") or []:
            index.test_jobs[str(job["name"])] = arm
    for job in inputs.jobs.get("jobs") or []:
        name = str(job["name"])
        index.jobs[name] = job
        if job.get("launch_run_id"):
            index.run_job[str(job["launch_run_id"])] = name
        for run in job.get("runs") or []:
            index.run_job[str(run["run_id"])] = name
    selection = inputs.selection or {}
    training_run = selection.get("training_run")
    scoring_run = (selection.get("selected") or {}).get("eval_run_id")
    for name, job in index.jobs.items():
        inside = {str(run["run_id"]) for run in job.get("runs") or []}
        if training_run and training_run in inside:
            index.train_job = name
        if scoring_run and scoring_run in inside:
            index.select_job = name
    for attempt in current_attempts(inputs.tuning):
        if attempt.get("job_arn"):
            index.tuning[str(attempt["job_arn"])] = str(attempt.get("arm"))
    return index


def _job_category(name: str | None, index: Index) -> str:
    if name is None:
        return DEVELOPMENT
    if name in index.test_jobs:
        return TEST
    if name == index.train_job:
        return TRAIN
    if name == index.select_job:
        return SELECT
    return DEVELOPMENT


def place(record: Any, index: Index, inputs: CostInputs) -> Placed:
    """The category and the price of one effective ledger record."""

    def unplaced(problem: str) -> Placed:
        return Placed(record, DEVELOPMENT, ZERO, FREE, problem=f"{record.run_id}: {problem}")

    if record.where == LOCAL:
        return Placed(record, NO_CHARGE, ZERO, FREE)

    if record.where == SAGEMAKER:
        if record.billable_seconds is None:
            name = index.run_job.get(record.run_id)
            return Placed(record, _job_category(name, index), ZERO, THROUGH_JOB)
        name = record.external_id
        if name not in index.jobs:
            return unplaced(f"its job {name!r} is not in the job record")
        price = inputs.instances.training.get(record.instance_type)
        if price is None:
            return unplaced(f"no training price for {record.instance_type}")
        seconds = Decimal(record.billable_seconds) * (record.instance_count or 1)
        return Placed(
            record,
            _job_category(name, index),
            seconds * price / SECONDS_PER_HOUR,
            LIST_PRICE,
            test_arm=index.test_jobs.get(name),
        )

    if record.where == BEDROCK:
        test_arm = index.test_runs.get(record.run_id)
        if record.phase == TRAIN:
            category = TRAIN if record.external_id in index.tuning else DEVELOPMENT
        else:
            category = TEST if test_arm else DEVELOPMENT
        if record.billed_amount_usd is not None:
            return Placed(record, category, _decimal(record.billed_amount_usd), BILLED, test_arm)
        if record.model_id not in inputs.pricing.models:
            return unplaced(f"no price for model {record.model_id!r}")
        if record.phase == TRAIN:
            if record.training_tokens is None:
                return unplaced("a training record with neither a billed amount nor tokens")
            price = inputs.pricing.model(record.model_id).training_per_million
            basis = ESTIMATE if record.source == "computed" else LIST_PRICE
            return Placed(record, category, record.training_tokens * price / MILLION, basis)
        if record.input_tokens is None and record.output_tokens is None:
            return unplaced("a Bedrock record with nothing to price")
        amount = inference_cost(
            record.model_id, record.input_tokens or 0, record.output_tokens or 0, inputs.pricing
        )
        return Placed(record, category, amount, LIST_PRICE, test_arm)

    return unplaced(f"unknown place {record.where!r}")


def place_all(inputs: CostInputs) -> tuple[list[Placed], Index]:
    index = build_index(inputs)
    return [place(record, index, inputs) for record in inputs.ledger], index


def path_totals(placed: list[Placed]) -> dict[str, dict[str, Decimal]]:
    """Dollars per path and category. ``total`` is the sum over the categories."""
    totals: dict[str, dict[str, Decimal]] = {}
    for item in placed:
        row = totals.setdefault(
            str(item.record.path), {category: ZERO for category in CATEGORIES} | {"total": ZERO}
        )
        row[item.category] += item.amount
        row["total"] += item.amount
    return totals


# --------------------------------------------------------------------------- per arm


def making_costs(placed: list[Placed]) -> dict[str, dict[str, Any]]:
    """Per tuned arm: its training run, and for Path A the selection of its checkpoint."""
    making: dict[str, dict[str, Any]] = {}
    for item in placed:
        if item.category not in (TRAIN, SELECT) or item.basis == THROUGH_JOB:
            continue
        entry = making.setdefault(str(item.record.arm), {})
        line = entry.setdefault(item.category, {"usd": ZERO, "basis": item.basis, "records": 0})
        line["usd"] += item.amount
        line["records"] += 1
        if item.record.billable_seconds is not None:
            line["billable_seconds"] = (
                line.get("billable_seconds", 0) + item.record.billable_seconds
            )
            line["instance_type"] = item.record.instance_type
        if item.basis == ESTIMATE:
            line["basis"] = ESTIMATE
            line["training_tokens"] = item.record.training_tokens
    return making


def making_total(entry: dict[str, Any] | None) -> Decimal:
    return sum((line["usd"] for line in (entry or {}).values()), ZERO)


def storage_lines(inputs: CostInputs, cfg: CostConfig) -> dict[str, dict[str, Any]]:
    """Per tuned Nova arm: what keeping its model costs a month, and for stated months."""
    lines: dict[str, dict[str, Any]] = {}
    for attempt in current_attempts(inputs.tuning):
        model_id = attempt.get("base_model_id")
        if attempt.get("model_deleted") or model_id not in inputs.pricing.models:
            continue
        monthly = inputs.pricing.model(model_id).storage_per_model_month
        lines[str(attempt.get("arm"))] = {
            "per_month": monthly,
            "months": {months: monthly * months for months in cfg.storage_months},
            "since": str(attempt.get("ended_at") or "")[:10] or None,
        }
    return lines


def storage_so_far(storage: dict[str, dict[str, Any]], through: str | None) -> dict[str, Decimal]:
    """One model-month for each calendar month a model has existed, up to ``through``.

    That is how the bill charges it: both models were made in November 2026 and each line
    read "1 Model/month", whatever the day.
    """
    owed: dict[str, Decimal] = {}
    for arm, line in storage.items():
        since = line.get("since")
        if not since or not through or since[:10] > through:
            continue
        months = (int(through[:4]) - int(since[:4])) * 12 + int(through[5:7]) - int(since[5:7]) + 1
        owed[arm] = line["per_month"] * months
    return owed


def _wall_seconds(run_ids: list[str], inputs: CostInputs) -> Decimal | None:
    seconds = [
        _decimal(record.wall_clock_seconds)
        for record in inputs.ledger
        if record.run_id in run_ids
        and record.billable_seconds is None
        and record.wall_clock_seconds is not None
    ]
    return sum(seconds, ZERO) if seconds else None


def arm_figures(
    placed: list[Placed], inputs: CostInputs, index: Index
) -> dict[str, dict[str, Any]]:
    """Per arm of the six-arm record: the test pass and what a question costs."""
    figures: dict[str, dict[str, Any]] = {}
    for arm, entry in ((inputs.six_arm or {}).get("arms") or {}).items():
        every = entry.get("all") or {}
        questions = int(every.get("examples") or 0)
        correct = int((every.get("execution_accuracy") or {}).get("count") or 0)
        cost = sum(
            (item.amount for item in placed if item.category == TEST and item.test_arm == arm),
            ZERO,
        )
        figure: dict[str, Any] = {
            "path": entry.get("path"),
            "run_id": entry.get("run_id"),
            "questions": questions,
            "correct": correct,
            "usd": cost,
            "per_1000_questions": cost / questions * THOUSAND if questions else None,
            "per_1000_correct": cost / correct * THOUSAND if correct else None,
        }
        jobs = entry.get("jobs") or []
        if jobs:
            run_ids = [run for run, owner in index.test_runs.items() if owner == arm]
            instance_type = jobs[0].get("instance_type")
            rate = inputs.instances.training.get(instance_type)
            billable = sum(int(job.get("billable_seconds") or 0) for job in jobs)
            wall = _wall_seconds(run_ids, inputs)
            figure |= {"instance_type": instance_type, "billable_seconds": billable}
            if wall is not None and rate is not None and questions:
                steady = wall * rate / SECONDS_PER_HOUR
                figure |= {
                    "run_seconds": wall,
                    "startup_seconds": max(Decimal(billable) - wall, ZERO),
                    "seconds_per_question": wall / questions,
                    "steady_usd": steady,
                    "per_1000_questions_steady": steady / questions * THOUSAND,
                    "per_1000_correct_steady": steady / correct * THOUSAND if correct else None,
                }
        else:
            figure |= {
                "model_id": entry.get("model_id"),
                "input_tokens": entry.get("input_tokens"),
                "output_tokens": entry.get("output_tokens"),
            }
        figures[arm] = figure
    return figures


# --------------------------------------------------------------------------- against the bill


def model_total(
    placed: list[Placed],
    place_name: str,
    through: str,
    storage: dict[str, dict[str, Any]],
) -> Decimal:
    """What the model says a service cost through a day, storage of tuned models included."""
    total = sum(
        (
            item.amount
            for item in placed
            if item.record.where == place_name and item.date is not None and item.date <= through
        ),
        ZERO,
    )
    if place_name == BEDROCK:
        total += sum(storage_so_far(storage, through).values(), ZERO)
    return total


def reconcile(
    placed: list[Placed],
    inputs: CostInputs,
    cfg: CostConfig,
    storage: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per service: each entered amount beside the model's total for the same days."""
    services = []
    for service, place_name in cfg.services.items():
        dates = [
            item.date
            for item in placed
            if item.record.where == place_name and item.amount > 0 and item.date
        ]
        last = max(dates) if dates else None
        entries = []
        for bill in inputs.bills:
            if bill.get("service") != service:
                continue
            billed = _decimal(bill["usd"])
            model = model_total(placed, place_name, str(bill["through"]), storage)
            tolerance = max(cfg.tolerance_usd, billed * cfg.tolerance_percent / 100)
            entries.append(
                {
                    "through": str(bill["through"]),
                    "entered_on": bill.get("entered_on"),
                    "billed_usd": billed,
                    "model_usd": model,
                    "difference_usd": billed - model,
                    "tolerance_usd": tolerance,
                    "agrees": abs(billed - model) <= tolerance,
                    "covers_every_run": last is None or str(bill["through"]) >= last,
                }
            )
        services.append(
            {
                "service": service,
                "last_run_date": last,
                "model_usd_to_date": model_total(placed, place_name, last, storage)
                if last
                else ZERO,
                "entries": entries,
                "covered": any(entry["covers_every_run"] for entry in entries),
            }
        )
    return services


# --------------------------------------------------------------------------- the checks


def _check(
    check_id: str,
    name: str,
    problems: list[str],
    ok_detail: str,
    reservations: list[str] | None = None,
) -> dict[str, Any]:
    reservations = reservations or []
    if problems:
        detail = "; ".join(problems)
    elif reservations:
        detail = f"{ok_detail}; RESERVATION: {'; '.join(reservations)}"
    else:
        detail = ok_detail
    return {
        "id": check_id,
        "name": name,
        "passed": not problems,
        "reservation": bool(reservations) and not problems,
        "detail": detail,
    }


def _quantities(
    inputs: CostInputs, index: Index, placed: list[Placed]
) -> tuple[list[str], list[str]]:
    """K2: the priced quantities are in the ledger and equal the committed records."""
    problems: list[str] = []
    for arm, entry in ((inputs.six_arm or {}).get("arms") or {}).items():
        run_ids = [run for run, owner in index.test_runs.items() if owner == arm]
        records = [r for r in inputs.ledger if r.run_id in run_ids and r.input_tokens is not None]
        if not records:
            problems.append(f"{arm}: no ledger record for run {entry.get('run_id')}")
            continue
        for name in ("input_tokens", "output_tokens"):
            ledger = sum(getattr(r, name) or 0 for r in records)
            if ledger != entry.get(name):
                problems.append(
                    f"{arm}: {name} {ledger:,} in the ledger, {entry.get(name)} in the "
                    "six-arm record"
                )
        for job in entry.get("jobs") or []:
            name = job["name"]
            billed = [
                r.billable_seconds
                for r in inputs.ledger
                if r.external_id == name and r.billable_seconds is not None
            ]
            recorded = (index.jobs.get(name) or {}).get("billable_seconds")
            if billed != [job.get("billable_seconds")] or recorded != job.get("billable_seconds"):
                problems.append(
                    f"{arm}: job {name} has billable seconds {billed} in the ledger, "
                    f"{recorded} in the job record, {job.get('billable_seconds')} in the "
                    "six-arm record"
                )
    estimates = [
        f"{item.record.arm} {item.category} of run {item.record.run_id} is an estimate"
        for item in placed
        if item.basis == ESTIMATE and item.amount > 0
    ]
    return problems, estimates


def _prices_found(inputs: CostInputs, figures: dict[str, dict[str, Any]]) -> list[str]:
    problems = []
    for record in inputs.ledger:
        if record.where == BEDROCK and record.model_id not in inputs.pricing.models:
            problems.append(f"no price for model {record.model_id!r}")
        if (
            record.where == SAGEMAKER
            and record.billable_seconds is not None
            and record.instance_type not in inputs.instances.training
        ):
            problems.append(f"no training price for {record.instance_type}")
    for arm, figure in figures.items():
        instance_type = figure.get("instance_type")
        if instance_type and instance_type not in inputs.instances.hosting:
            problems.append(f"no hosting price for {instance_type}, which {arm} ran on")
    return sorted(set(problems))


def _same_questions(inputs: CostInputs) -> list[str]:
    six = inputs.six_arm or {}
    problems = []
    tokens: dict[str, set[Any]] = {}
    for arm, entry in (six.get("arms") or {}).items():
        examples = (entry.get("all") or {}).get("examples")
        if examples != six.get("test_examples"):
            problems.append(f"{arm} has {examples} examples, not {six.get('test_examples')}")
        tokens.setdefault(str(entry.get("path")), set()).add(entry.get("input_tokens"))
    for path, counts in sorted(tokens.items()):
        if len(counts) > 1:
            problems.append(f"the arms of Path {path} have unequal input tokens: {sorted(counts)}")
    return problems


def build_record(inputs: CostInputs, cfg: CostConfig) -> dict[str, Any]:
    """Checks K1 to K8 and every table of the cost record."""
    six = inputs.six_arm or {}
    arms = six.get("arms") or {}
    placed, index = place_all(inputs)
    totals = path_totals(placed)
    making = making_costs(placed)
    storage = storage_lines(inputs, cfg)
    figures = arm_figures(placed, inputs, index)
    scenarios = serving(figures, inputs.instances, cfg)
    making_totals = {arm: making_total(entry) for arm, entry in making.items()}
    crossings = break_evens(figures, scenarios, making_totals, storage, cfg)
    services = reconcile(placed, inputs, cfg, storage)

    checks = []
    problems = []
    if not arms:
        problems.append(f"there is no six-arm record: run {COMMAND.replace('cost', 'runs')} record")
    elif not six.get("passed"):
        problems.append("the six-arm record did not pass its checks")
    problems += [f"{arm} names no run" for arm, entry in arms.items() if not entry.get("run_id")]
    checks.append(
        _check("K1", "Six test runs", problems, f"{len(arms)} arms, one run each, C1 to C8 passed")
    )

    problems, estimates = _quantities(inputs, index, placed)
    checks.append(
        _check(
            "K2",
            "Quantities found",
            problems,
            "tokens and billable seconds in the ledger equal the six-arm record and the job record",
            estimates,
        )
    )
    checks.append(
        _check(
            "K3",
            "Prices found",
            _prices_found(inputs, figures),
            f"Bedrock price list of {inputs.pricing.published}; SageMaker training "
            f"{inputs.instances.training_published}, hosting {inputs.instances.hosting_published}",
        )
    )
    counts = {
        category: sum(item.category == category for item in placed) for category in CATEGORIES
    }
    checks.append(
        _check(
            "K4",
            "Every record placed",
            [item.problem for item in placed if item.problem],
            f"{len(placed)} records: "
            + ", ".join(f"{counts[c]} {c.replace('_', ' ')}" for c in CATEGORIES),
        )
    )
    problems = []
    for path in PATHS:
        mine = totals.get(path, {}).get("total", ZERO)
        theirs = inputs.committed.get(path)
        if theirs is None:
            problems.append(f"the spend of Path {path} was not computed")
        elif money(mine) != money(theirs):
            problems.append(
                f"Path {path}: the categories sum to ${mine:.2f}, the cap check counts "
                f"${theirs:.2f} (a job launched and not pulled counts at its maximum there)"
            )
    checks.append(
        _check(
            "K5",
            "Totals agree",
            problems,
            ", ".join(f"Path {p} ${totals.get(p, {}).get('total', ZERO):.2f}" for p in PATHS)
            + ": the same as the cap checks count",
        )
    )
    checks.append(
        _check(
            "K6",
            "Same questions",
            _same_questions(inputs),
            f"{six.get('test_examples') or 0:,} examples per arm; equal input tokens within a path",
        )
    )
    unmarked = [
        f"{entry['pair']} {entry['scenario']}" for entry in crossings if "reachable" not in entry
    ]
    out_of_reach = sum(not entry["reachable"] for entry in crossings)
    checks.append(
        _check(
            "K7",
            "Reachable",
            unmarked,
            f"{len(crossings)} break-evens, each marked; {out_of_reach} not reachable here",
        )
    )
    problems, reservations = [], []
    for service in services:
        for entry in service["entries"]:
            if not entry["agrees"]:
                problems.append(
                    f"{service['service']} through {entry['through']}: billed "
                    f"${entry['billed_usd']:.2f}, the model ${entry['model_usd']:.2f}"
                )
        if not service["covered"] and service["last_run_date"]:
            reservations.append(
                f"no bill amount for {service['service']} covers {service['last_run_date']}, "
                f"its last run; the model says ${service['model_usd_to_date']:.2f}"
            )
    entered = sum(len(service["entries"]) for service in services)
    checks.append(
        _check(
            "K8",
            "Agrees with the bill",
            problems,
            f"{entered} entered amount(s), each within the tolerance",
            reservations,
        )
    )

    dates = [item.date for item in placed if item.date and item.amount > 0]
    through = max(dates) if dates else None
    project = {
        path: {
            **{category: money(row[category]) for category in CATEGORIES},
            "total": money(row["total"]),
            "cap": money(inputs.caps[path]) if path in inputs.caps else None,
        }
        for path, row in sorted(totals.items())
    }
    record = {
        "schema": SCHEMA,
        "generated_by": f"{COMMAND} report",
        "passed": all(check["passed"] for check in checks),
        "provisional": any(check["reservation"] for check in checks),
        "checks": checks,
        "through": through,
        "test_examples": six.get("test_examples"),
        "six_arm_freeze_sha256": six.get("freeze_sha256"),
        "prices": _prices_block(inputs),
        "making": _making_block(making, storage),
        "test_pass": {arm: _plain(figure) for arm, figure in figures.items()},
        "serving": {arm: _plain(scenario) for arm, scenario in scenarios.items()},
        "serving_assumptions": {
            "hours_per_month": cfg.hours_per_month,
            "batch_jobs_per_month": cfg.batch_jobs_per_month,
            "requests_per_minute": dict(cfg.requests_per_minute),
            "limits_read_on": cfg.limits_read_on,
        },
        "break_evens": crossings,
        "project": {
            "paths": project,
            "storage_so_far": {
                arm: money(amount) for arm, amount in storage_so_far(storage, through).items()
            },
            "records": counts,
        },
        "reconciliation": [_plain(service) for service in services],
        "estimates": estimates,
        "left_out": list(cfg.left_out),
    }
    return record


def _plain(value: Any) -> Any:
    """Decimals as text, so the record is the same bytes for the same state."""
    if isinstance(value, Decimal):
        return money(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _prices_block(inputs: CostInputs) -> dict[str, Any]:
    pricing, instances = inputs.pricing, inputs.instances
    return {
        "bedrock": {
            "offer": pricing.offer,
            "region": pricing.region,
            "published": pricing.published,
            "read_on": pricing.read_on,
            "models": {
                model_id: {
                    "input_per_million": str(price.input_per_million),
                    "output_per_million": str(price.output_per_million),
                    "training_per_million": str(price.training_per_million),
                    "storage_per_model_month": str(price.storage_per_model_month),
                }
                for model_id, price in sorted(pricing.models.items())
            },
        },
        "sagemaker": {
            "training_published": instances.training_published,
            "training_per_hour": {k: str(v) for k, v in sorted(instances.training.items())},
            "hosting_published": instances.hosting_published,
            "hosting_per_hour": {k: str(v) for k, v in sorted(instances.hosting.items())},
        },
    }


def _making_block(
    making: dict[str, dict[str, Any]], storage: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for arm in sorted(set(making) | set(storage)):
        entry = making.get(arm) or {}
        block[arm] = {
            "train": _plain(entry.get(TRAIN)),
            "select": _plain(entry.get(SELECT)),
            "making_usd": money(making_total(entry)),
            "storage": _plain(storage.get(arm)),
        }
    return block
