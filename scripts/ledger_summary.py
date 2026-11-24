"""Print the runs and the cost-ledger totals (req_000 v0.04).

Reads the committed ``ledger/`` files. It shows quantities (seconds, tokens, questions), not
dollars: prices are applied by the cost model.

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.ledger_summary
    venv\\Scripts\\python -m scripts.ledger_summary --by arm,phase,where
    venv\\Scripts\\python -m scripts.ledger_summary --json

Exit codes: 0 summary printed (also when there are no records), 2 a ledger or config problem.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

try:
    from qlora_finetuning.settings import ConfigError
    from qlora_finetuning.tracking import Ledger, TrackingConfig, list_runs, load_tracking_config
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.ledger_summary"
    ) from None

RUN_COLUMNS = ("run_id", "kind", "arm", "where", "scope", "status")
TOTAL_COLUMNS = (
    "records",
    "wall_clock_seconds",
    "billable_instance_seconds",
    "input_tokens",
    "output_tokens",
    "training_tokens",
    "examples",
    "billed_amount_usd",
)


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.1f}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value)


def format_table(columns: tuple[str, ...], rows: list[dict[str, Any]]) -> str:
    """A plain text table: left-aligned text, right-aligned numbers."""
    cells = [[_cell(row.get(column, "")) for column in columns] for row in rows]
    widths = [
        max(len(column), *(len(line[i]) for line in cells)) for i, column in enumerate(columns)
    ]
    numeric = [
        bool(rows) and all(isinstance(row.get(column), (int, float)) for row in rows)
        for column in columns
    ]

    def render(values: list[str]) -> str:
        parts = [
            value.rjust(width) if right else value.ljust(width)
            for value, width, right in zip(values, widths, numeric, strict=True)
        ]
        return "  ".join(parts).rstrip()

    lines = [render(list(columns)), render(["-" * width for width in widths])]
    lines.extend(render(line) for line in cells)
    return "\n".join(lines)


def build_summary(cfg: TrackingConfig, by: tuple[str, ...]) -> dict[str, Any]:
    ledger = Ledger(cfg)
    effective = ledger.effective()
    return {
        "runs": list_runs(cfg),
        "by": list(by),
        "totals": ledger.summarize(by=by),
        "records": len(effective),
        "replaced_records": len(ledger.read()) - len(effective),
        "estimates": sum(record.source == "computed" for record in effective),
    }


def main(argv: list[str] | None = None, cfg: TrackingConfig | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the runs and the cost-ledger totals.")
    parser.add_argument(
        "--by", default="arm,phase", help="comma-separated fields to group by (default arm,phase)"
    )
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)
    by = tuple(part.strip() for part in args.by.split(",") if part.strip())

    try:
        summary = build_summary(cfg or load_tracking_config(), by)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 2

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0
    if not summary["runs"] and not summary["records"]:
        print("no records")
        return 0

    print(f"Runs ({len(summary['runs'])}, newest first)")
    print(format_table(RUN_COLUMNS, summary["runs"]))
    print()
    print(f"Ledger totals by {', '.join(by)} ({summary['records']} records)")
    if summary["totals"]:
        print(format_table((*by, *TOTAL_COLUMNS), summary["totals"]))
    else:
        print("no ledger records")
    print()
    print(
        f"{summary['estimates']} of {summary['records']} records are estimates (computed); "
        f"{summary['replaced_records']} replaced records are not counted"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
