"""What the owner read from the AWS bill (req_005 v5.02).

AWS bills by service and day, not by run, so only a service's total can be compared with
the cost model. The owner reads the total from the console (Billing and Cost Management,
Bills, Charges by service) and enters it; ``cost.reconcile`` puts it beside the model's
total for the same days.

Usage::

    from qlora_finetuning.comparison.cost_bill import add_bill, read_bills

    add_bill(cfg, "bedrock", "81.98", "2026-11-23")     # THE OWNER; appends, never rewrites
    read_bills(cfg.bill_file)                            # oldest first; [] if no file

The file is committed: amounts and dates only. This module imports neither boto3 nor the
training stack.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from qlora_finetuning.comparison.cost_config import CostConfig
from qlora_finetuning.comparison.gate import canonical_json
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking.config import check_no_secrets

SCHEMA = 1


def read_bills(path: Path) -> list[dict[str, Any]]:
    """The amounts the owner read from the AWS bill, oldest first; none if no file."""
    if not Path(path).is_file():
        return []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = data["entries"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ConfigError(f"{Path(path).name} cannot be read: {exc}") from None
    if not isinstance(entries, list):
        raise ConfigError(f"{Path(path).name}: entries must be a list")
    return entries


def add_bill(
    cfg: CostConfig, service: str, usd: str, through: str, today: date | None = None
) -> dict[str, Any]:
    """Append what the owner read from the bill. **Run by the owner.** Earlier entries stay.

    ``service`` is one of the reconciliation's services, ``usd`` the service's total on the
    bill, and ``through`` the last day (UTC) whose usage that total covers.
    """
    if service not in cfg.services:
        raise ConfigError(f"unknown service {service!r}; known: {', '.join(sorted(cfg.services))}")
    try:
        amount = Decimal(str(usd))
    except InvalidOperation:
        raise ConfigError(f"--usd is not an amount: {usd!r}") from None
    if not amount.is_finite() or amount < 0:
        raise ConfigError(f"--usd must be zero or more, got {usd!r}")
    today = today or date.today()
    try:
        last_day = date.fromisoformat(str(through))
    except ValueError:
        raise ConfigError(f"--through must be a date such as 2026-11-23, got {through!r}") from None
    if last_day > today:
        raise ConfigError(f"--through {through} is in the future")
    entries = read_bills(cfg.bill_file)
    entry = {
        "service": service,
        "usd": str(amount),
        "through": last_day.isoformat(),
        "entered_on": today.isoformat(),
    }
    data = {"schema": SCHEMA, "entries": [*entries, entry]}
    check_no_secrets(data, cfg.bill_file.name)
    cfg.bill_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.bill_file.write_text(canonical_json(data), encoding="utf-8", newline="\n")
    return entry
