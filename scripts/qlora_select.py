"""Choose the checkpoint that becomes arm A1 (req_003 v3.04).

After a training run's checkpoints have been scored on dev (``sagemaker_job launch score``,
then ``pull``), this picks the one with the best execution accuracy, checks that the runs
are comparable, and writes the committed selection record::

    reports/local_qlora/adapter_selection.json
    reports/local_qlora/adapter_selection.md

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.qlora_select --run <training run id>
    venv\\Scripts\\python -m scripts.qlora_select --run <training run id> --check
    venv\\Scripts\\python -m scripts.qlora_select --run <run> --split quick_dev --limit 24 --check

It costs nothing and uses no GPU and no AWS: every count is recomputed from the predictions
files of the pulled runs. ``--check`` writes nothing. ``--split`` and ``--limit`` rehearse
the selection on partial runs and are allowed only with ``--check``.

Checks L1 to L6: every checkpoint has one complete result; all runs asked the same
question; each was scored with its own weights; the training run completed and is the
current experiment's; everything was measured in the cloud and its jobs are pulled; every
score is current. The record is written only if all six pass.

The verdict is ``met`` or ``missed`` against the two thresholds of ``config/training.yaml``.
``missed`` starts nothing: it offers the one corrective iteration of v3.05.

From req_003 v3.05: given the second run of a registered corrective iteration
(``scripts.qlora_iterate propose``), it compares that run's best checkpoint with the one
already selected, by the same rule; a tie keeps the first run's. The record then names
the winner and lists both runs, and the verdict is the winner's.

Exit codes: 0 selected and the verdict is met (or a check run in which all checks pass);
1 a check failed, or the verdict is missed; 2 a config problem or an unknown run.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

try:
    from qlora_finetuning.local_qlora.selection import MISSED, select_checkpoint
    from qlora_finetuning.logging_setup import setup_logging
    from qlora_finetuning.settings import ConfigError
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.qlora_select"
    ) from None

PASS, FAIL = "PASS", "FAIL"


def _share(count: int, total: int) -> str:
    return f"{count:>5,}  {100 * count / total:6.2f}%" if total else f"{count:>5,}"


def print_outcome(outcome: dict[str, Any]) -> None:
    kind = "rehearsal on" if outcome["rehearsal"] else "on"
    limit = f", the first {outcome['limit']:,} examples" if outcome["limit"] else ""
    print(f"training run {outcome['run']}  ({kind} {outcome['split']}{limit})")
    for check in outcome["checks"]:
        status = PASS if check["passed"] else FAIL
        print(f"{status}  {check['id']}  {check['name']}: {check['detail']}")
    failed = sum(not check["passed"] for check in outcome["checks"])
    print(f"{len(outcome['checks']) - failed} passed, {failed} failed")
    if not outcome["rows"]:
        return

    print(f"{'checkpoint':>10}  {'execution':>14}  {'format':>14}  {'dev loss':>8}")
    rows = [("untuned", outcome["untuned"]), *((f"{r['step']:,}", r) for r in outcome["rows"])]
    for label, row in rows:
        loss = "" if row.get("dev_loss") is None else f"{row['dev_loss']:.4f}"
        mark = "  <- best" if row is outcome["selected"] else ""
        print(
            f"{label:>10}  {_share(row['execution_accuracy'], row['examples'])}  "
            f"{_share(row['format_validity'], row['examples'])}  {loss:>8}{mark}"
        )
    selected, notes = outcome["selected"], outcome["notes"]
    print(f"best: checkpoint {selected['step']:,}  ({selected['adapter']})")
    print(f"above the untuned model by {notes['execution_gain_points_over_untuned']} points")
    if notes["selected_is_last_checkpoint"]:
        print("note: it is the last checkpoint; the model may still have been improving")
    iteration = outcome.get("iteration")
    if not iteration:
        print(f"VERDICT: {outcome['verdict']}")
        return
    # A second run, compared with the checkpoint that was already selected (v3.05).
    print(f"this run alone: {outcome['verdict']}")
    gap = f"{iteration['gap_examples']:+,} examples"
    noise = "within noise" if iteration["within_noise"] else "more than noise"
    if iteration["winner"] == "second":
        print(f"ITERATION: the second run's checkpoint is arm A1 ({gap}, {noise})")
    else:
        print(f"ITERATION: the first run's checkpoint stays arm A1 ({gap}, {noise})")
        print("then: revert the change in config/training.yaml and commit")
    print(f"arm A1: {iteration['selected']}")
    print(f"VERDICT: {iteration['verdict']}")


def main(argv: list[str] | None = None, **stand_ins: Any) -> int:
    parser = argparse.ArgumentParser(description="Choose the checkpoint that becomes arm A1.")
    parser.add_argument("--run", required=True, metavar="RUN_ID", help="the training run")
    parser.add_argument("--check", action="store_true", help="report only; write nothing")
    parser.add_argument("--split", help="rehearsal only: another split than the configured one")
    parser.add_argument("--limit", type=int, help="rehearsal only: runs of the first N examples")
    args = parser.parse_args(argv)

    setup_logging()
    try:
        outcome = select_checkpoint(
            args.run, split=args.split, limit=args.limit, check=args.check, **stand_ins
        )
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 2

    print_outcome(outcome)
    if outcome["written"]:
        print("SELECTED: the selection record is written; commit it: git add reports/local_qlora")
        final = (outcome.get("iteration") or {}).get("selected") or outcome["selected"]["adapter"]
        print(f"pull the adapter: python -m scripts.sagemaker_job pull --checkpoint "
              f"{final.rsplit('-', 1)[-1]}")  # fmt: skip
    elif outcome["passed"]:
        print("CHECK ONLY: all checks pass; nothing was written")
    else:
        print("NOT SELECTED: a check failed; nothing was written")
    verdict = (outcome.get("iteration") or {}).get("verdict") or outcome["verdict"]
    if not outcome["passed"] or verdict == MISSED:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
