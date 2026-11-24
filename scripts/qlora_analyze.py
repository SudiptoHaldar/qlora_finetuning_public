"""Where a trained adapter goes wrong on dev (req_003 v3.05).

Sorts every wrong reply of a training run's scored checkpoints into a computed category and
writes the committed report::

    reports/local_qlora/error_analysis.json
    reports/local_qlora/error_analysis.md

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.qlora_analyze --run <training run id>
    venv\\Scripts\\python -m scripts.qlora_analyze --run <training run id> --check
    venv\\Scripts\\python -m scripts.qlora_analyze --run <training run id> --show order --n 5
    venv\\Scripts\\python -m scripts.qlora_analyze --run <run> --split quick_dev --limit 24 --check

It costs nothing and uses no GPU and no AWS: it reads the predictions files of the pulled
runs. Without ``--check`` the run must be the one ``scripts.qlora_select`` selected, and
checks L1 to L6 must pass. ``--check`` writes nothing and looks at any scored run.

The report holds counts, rates, run ids, and example ids. **It never holds a reply.**
``--show CATEGORY`` prints replies of that category beside their gold programs, **to the
terminal only**: they can repeat dataset text, so do not paste them into a committed file.

The last line is the finding: which row of the corrective table (T1 to T5) the figures fit.
It decides nothing. A corrective iteration is the owner's, through ``scripts.qlora_iterate``,
and only when the selection's verdict is ``missed``.

Exit codes: 0 analysed (and written, unless ``--check``); 1 a check failed; 2 a config
problem, an unknown run, or a run that is not the selected one.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

try:
    from qlora_finetuning.evaluation.failures import CATEGORIES, run_failures
    from qlora_finetuning.evaluation.runner import run_gold, run_predictions
    from qlora_finetuning.local_qlora.analysis import NONE, analyze_run
    from qlora_finetuning.logging_setup import setup_logging
    from qlora_finetuning.settings import ConfigError
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.qlora_analyze"
    ) from None

PASS, FAIL = "PASS", "FAIL"


def _ascii(text: str) -> str:
    """A reply as the console can print it, whatever its code page."""
    return text.encode("ascii", "backslashreplace").decode("ascii")


def print_outcome(outcome: dict[str, Any]) -> None:
    kind = "rehearsal on" if outcome["rehearsal"] else "on"
    limit = f", the first {outcome['limit']:,} examples" if outcome["limit"] else ""
    print(f"training run {outcome['run']}  ({kind} {outcome['split']}{limit})")
    for check in outcome["checks"]:
        status = PASS if check["passed"] else FAIL
        print(f"{status}  {check['id']}  {check['name']}: {check['detail']}")
    failed = sum(not check["passed"] for check in outcome["checks"])
    print(f"{len(outcome['checks']) - failed} passed, {failed} failed")
    analysis = outcome["analysis"]
    if analysis is None:
        return

    selected, untuned = analysis["selected"], analysis["untuned"]
    counts, base, total = selected["counts"], untuned["counts"], analysis["examples"]
    print(f"checkpoint {selected['step']:,}: {counts['right']:,} of {total:,} right on execution")
    print(f"{'category':<16}{'selected':>9}{'untuned':>9}")
    for name in CATEGORIES:
        if counts["categories"][name] or base["categories"][name]:
            print(f"{name:<16}{counts['categories'][name]:>9,}{base['categories'][name]:>9,}")
    print(f"{'right':<16}{counts['right']:>9,}{base['right']:>9,}")
    print(f"against the untuned model: {untuned['fixed']:,} fixed, {untuned['broken']:,} broken")
    shown = [n for n in CATEGORIES if any(c["categories"][n] for c in analysis["checkpoints"])]
    print("over the checkpoints:")
    print(f"{'step':>8}{'right':>7}  " + "  ".join(shown))
    for point in analysis["checkpoints"]:
        cells = "  ".join(f"{point['categories'][n]:>{len(n)},}" for n in shown)
        mark = "  <- selected" if point["step"] == selected["step"] else ""
        print(f"{point['step']:>8,}{point['right']:>7,}  {cells}{mark}")
    trend = analysis["trend"]
    print(f"the last three checkpoints are {trend['last_three']}")
    by_steps = ", ".join(
        f"{g['group']} step(s): {g['right']:,} of {g['examples']:,}"
        for g in analysis["where"]["by_gold_steps"]
    )
    print(f"right by gold program length: {by_steps}")
    print(f"never right at any checkpoint: {analysis['never_right']['count']:,}")
    finding = analysis["finding"]
    if finding["row"] == NONE:
        print(f"FINDING: none ({finding['why']})")
    else:
        change = (
            f"; the table's change: {finding['variable']} to {finding['value']:g}"
            if (finding["variable"])
            else ""
        )
        print(f"FINDING: {finding['row']} ({finding['why']}){change}")


def show(outcome: dict[str, Any], category: str, count: int, stand_ins: dict[str, Any]) -> None:
    """Print replies of one category beside their gold programs. Writes nothing."""
    run_id = outcome["analysis"]["selected"]["eval_run_id"]
    context = stand_ins.get("context")
    failures = run_failures(run_id, context)
    lines, golds = run_predictions(run_id, context), run_gold(run_id, context)
    print(f"--- replies in category {category} (terminal only; do not commit) ---")
    printed = 0
    for failure, line, gold in zip(failures, lines, golds, strict=False):
        if failure.category != category:
            continue
        print(f"{failure.example_id}")
        print(f"  gold : {_ascii(gold.program_text)}  ->  {_ascii(gold.answer_text)}")
        for part in _ascii(line.reply).splitlines() or [""]:
            print(f"  reply: {part}")
        printed += 1
        if printed >= count:
            break
    if not printed:
        print("(none)")


def main(argv: list[str] | None = None, **stand_ins: Any) -> int:
    parser = argparse.ArgumentParser(description="Where a trained adapter goes wrong on dev.")
    parser.add_argument("--run", required=True, metavar="RUN_ID", help="the training run")
    parser.add_argument("--check", action="store_true", help="report only; write nothing")
    parser.add_argument("--split", help="rehearsal only: another split than the configured one")
    parser.add_argument("--limit", type=int, help="rehearsal only: runs of the first N examples")
    parser.add_argument(
        "--show", choices=CATEGORIES, metavar="CATEGORY", help="print replies of one category"
    )
    parser.add_argument("--n", type=int, default=5, help="how many replies --show prints")
    args = parser.parse_args(argv)

    setup_logging()
    try:
        outcome = analyze_run(
            args.run, split=args.split, limit=args.limit, check=args.check, **stand_ins
        )
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 2

    print_outcome(outcome)
    if args.show and outcome["analysis"] is not None:
        show(outcome, args.show, max(args.n, 1), stand_ins)
    if outcome["written"]:
        print("ANALYSED: the report is written; commit it: git add reports/local_qlora")
    elif outcome["passed"]:
        print("CHECK ONLY: all checks pass; nothing was written")
    else:
        print("NOT WRITTEN: a check failed, so this is a look and not arm A1's analysis")
    return 0 if outcome["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
