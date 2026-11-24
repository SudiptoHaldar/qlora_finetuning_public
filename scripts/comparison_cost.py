"""The cost model: what each arm cost and what it would cost to use (req_005 v5.02).

Run from the repository root::

    venv\\Scripts\\python -m scripts.comparison_cost check
    venv\\Scripts\\python -m scripts.comparison_cost report
    venv\\Scripts\\python -m scripts.comparison_cost bill --service bedrock --usd 81.98 ^
        --through 2026-11-23

``check`` prices every effective ledger record, runs checks K1 to K8, and prints the main
figures. It writes nothing. ``report`` does the same and writes
``reports/comparison/cost_model.json`` and ``.md``; it writes only when every check passes.
A check may pass **with a reservation** (a bill amount not yet entered, a figure that is an
estimate): the record is then written with "provisional" in its title.

**``bill`` is run by the owner.** It appends one amount, as read from the AWS bill
(Billing and Cost Management, Bills, Charges by service), to
``reports/comparison/aws_bill.json``: the service's total, and the last day (UTC) whose
usage that total covers. Earlier entries are never changed.

None of the three calls AWS or needs a sign-in. Every figure is a quantity from the ledger
times a price from ``config/pricing.yaml``.

Exit codes: 0 done (a reservation is not a failure); 1 a check failed, so nothing was
written; 2 a config problem, a missing record, or a refused bill entry.
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

try:
    from qlora_finetuning.comparison.config import load_comparison_config
    from qlora_finetuning.comparison.cost import CostInputs, build_record
    from qlora_finetuning.comparison.cost_bill import add_bill, read_bills
    from qlora_finetuning.comparison.cost_config import (
        CostConfig,
        load_cost_config,
        load_instance_prices,
    )
    from qlora_finetuning.comparison.cost_report import usd, write_record
    from qlora_finetuning.frontier.config import load_frontier_config
    from qlora_finetuning.frontier.pricing import load_pricing
    from qlora_finetuning.frontier.tuning import committed_path_b
    from qlora_finetuning.local_qlora.estimate import committed_path_a
    from qlora_finetuning.local_qlora.job_record import read_job_record
    from qlora_finetuning.local_qlora.sagemaker_config import load_sagemaker_config
    from qlora_finetuning.logging_setup import setup_logging
    from qlora_finetuning.settings import REPO_ROOT, ConfigError
    from qlora_finetuning.tracking import Ledger
    from qlora_finetuning.tracking.config import load_tracking_config
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.comparison_cost"
    ) from None

SELF = "python -m scripts.comparison_cost"
PASS, FAIL = "PASS", "FAIL"


def _ascii(text: Any) -> str:
    return str(text).encode("ascii", "replace").decode()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _read_json(path: Path, what: str, required: bool = False) -> Any:
    if not path.is_file():
        if required:
            raise ConfigError(f"{what} is missing: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ConfigError(f"{what} cannot be read: {exc}") from None


def gather(cfg: CostConfig, root: Path) -> CostInputs:
    """Read the ledger, the committed records, the prices, and the owner's bill entries."""
    tracking_cfg = load_tracking_config(root)
    ledger = Ledger(tracking_cfg).effective()
    pricing = load_pricing()
    sagemaker_cfg = load_sagemaker_config(root)
    jobs = read_job_record(cfg.record_files["jobs"])
    tuning = _read_json(cfg.record_files["tuning"], "the tuning record") or {}
    return CostInputs(
        ledger=ledger,
        six_arm=_read_json(cfg.record_files["six_arm"], "the six-arm record"),
        jobs=jobs,
        selection=_read_json(cfg.record_files["selection"], "the selection record"),
        tuning=list(tuning.get("attempts") or []),
        pricing=pricing,
        instances=load_instance_prices(),
        bills=read_bills(cfg.bill_file),
        committed={
            "A": committed_path_a(sagemaker_cfg, tracking_cfg, jobs),
            "B": committed_path_b(ledger, pricing),
        },
        caps={
            "A": Decimal(str(sagemaker_cfg.path_a_cap_usd)),
            "B": Decimal(str(load_frontier_config(root).path_b_cap_usd)),
        },
    )


def print_record(record: dict[str, Any]) -> None:
    for check in record["checks"]:
        status = PASS if check["passed"] else FAIL
        print(f"{status}  {check['id']}  {check['name']}: {_ascii(check['detail'])}")
    failed = sum(not check["passed"] for check in record["checks"])
    reserved = sum(check["reservation"] for check in record["checks"])
    print(
        f"{len(record['checks']) - failed} passed, {failed} failed, {reserved} with a reservation"
    )
    for arm, entry in record["making"].items():
        train, select = entry["train"], entry["select"]
        parts = [f"training run {usd(train['usd'])} ({train['basis']})"] if train else []
        if select:
            parts.append(f"selection {usd(select['usd'])}")
        if entry["storage"]:
            parts.append(f"storage {usd(entry['storage']['per_month'])} a month")
        print(f"making    {arm:<8} " + "; ".join(parts))
    for arm, figure in record["test_pass"].items():
        steady = figure.get("per_1000_questions_steady")
        print(
            f"test pass {arm:<8} {usd(figure['usd']):>7}  per 1,000 questions "
            f"{usd(figure['per_1000_questions'])}"
            + (f" (steady {usd(steady)})" if steady else "")
            + f"; per 1,000 correct {usd(figure['per_1000_correct'])}"
        )
    for entry in record["break_evens"]:
        if entry["kind"] == "volume":
            what = f"{entry['questions']:,} questions; above it {entry['cheaper_above']} is cheaper"
        elif entry["kind"] == "always":
            what = f"{entry['cheaper_always']} cheaper at every volume"
        else:
            what = "the same at every volume"
        reach = "" if entry["reachable"] else "  NOT REACHABLE HERE"
        print(
            f"break-even {entry['pair'][0]} and {entry['pair'][1]:<8} "
            f"{entry['scenario']:<9} {what}{reach}"
        )
    for path, row in record["project"]["paths"].items():
        print(
            f"Path {path}: {usd(row['total'])} so far"
            + (f" of {usd(row['cap'])}" if row["cap"] else "")
        )


def main(argv: list[str] | None = None, root: Path | str | None = None, **stand_ins: Any) -> int:
    parser = argparse.ArgumentParser(description="The cost model of the comparison.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="K1 to K8 and the figures; writes nothing")
    commands.add_parser("report", help="K1 to K8; write the cost record")
    bill = commands.add_parser("bill", help="THE OWNER: enter a service's total from the AWS bill")
    bill.add_argument("--service", required=True, help="bedrock or sagemaker")
    bill.add_argument("--usd", required=True, help="the service's total on the bill")
    bill.add_argument(
        "--through", required=True, metavar="DATE", help="the last day (UTC) the total covers"
    )
    args = parser.parse_args(argv)
    setup_logging()
    root = Path(root or REPO_ROOT)
    try:
        cfg = stand_ins.get("cfg") or load_cost_config(root)
        if args.command == "bill":
            entry = add_bill(cfg, args.service, args.usd, args.through, stand_ins.get("today"))
            print(f"ENTERED: {entry['service']} {usd(entry['usd'])} through {entry['through']}")
            print(f"written: {_relative(cfg.bill_file, root)}")
            print(f"then: {SELF} report")
            return 0
        inputs = stand_ins.get("inputs") or gather(cfg, root)
        record = build_record(inputs, cfg)
        report_dir = stand_ins.get("report_dir") or load_comparison_config(root=root).report_dir
    except ConfigError as exc:
        print(f"CONFIG ERROR: {_ascii(exc)}")
        return 2

    print_record(record)
    if not record["passed"]:
        print("NOT WRITTEN: the cost record is written when K1 to K8 pass")
        return 1
    if record["provisional"]:
        print("PROVISIONAL: a check passed with a reservation; see the RESERVATION lines above")
    if args.command == "check":
        print("CHECK ONLY: nothing was written")
        return 0
    markdown_path, json_path = write_record(record, cfg, Path(report_dir))
    print(f"record: {_relative(json_path, root)}")
    print(f"record: {_relative(markdown_path, root)}")
    print("commit both" + (", with the bill file" if inputs.bills else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
