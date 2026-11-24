"""Make the Bedrock training files, check them, and price Path B (req_004 v4.02).

It renders the fit split as the training file and the dev split as the validation file
(Bedrock's conversation format, the canonical texts unchanged) under the git-ignored
``artifacts/bedrock/<export_id>/``, reads them back and runs checks X1 to X7, estimates the
cost of Path B, runs the two gate checks, and writes the committed gate record
``reports/frontier/nova_training_gate.md`` and ``.json``. A fine-tuning job (v4.03) starts
only with a record that says passed.

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.nova_export
    venv\\Scripts\\python -m scripts.nova_export --check
    venv\\Scripts\\python -m scripts.nova_export --upload

Without an option it calls no AWS service and costs nothing. ``--check`` writes no
committed file and fails if the gate record differs from a fresh one. ``--upload`` also
stages the two files in the project's S3 bucket and verifies them: **it creates objects in
AWS, so the owner runs it**, after ``aws sso login --profile qlora``. It is refused if a
check X1 to X7 failed; it does not need the gate to be open.

| Check | Passes when |
|---|---|
| X1 Same records | One line per record of the split, in order, with the record's texts |
| X2 No test rows | No exported id is a test id |
| X3 Format | Bedrock's conversation format; UTF-8; allowed file names |
| X4 Reserved words | No text holds one of AWS's nine |
| X5 Limits | Record counts and the longest record are inside AWS's limits |
| X6 Same question | The prompt and split are the current ones, and the token report's |
| X7 Reproducible | Rendering again gives the bytes on disk |
| G1 Ratio measured | The v4.01 baseline report gives the Nova to Qwen token ratio |
| G2 Within the cap | The Path B estimate is at most ``budget.path_b_cap_usd`` |

Exit codes: 0 every check passed (the gate is open); 1 a check failed, or with ``--check``
the record is out of date; 2 a config problem, no snapshot, or with ``--upload`` no sign-in.

Needs the snapshot, the split manifest, and the committed token budget report.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from typing import Any

try:
    from qlora_finetuning.data import split_ids
    from qlora_finetuning.data.token_budget import load_token_budget_config
    from qlora_finetuning.data.token_budget_report import report_paths as token_report_paths
    from qlora_finetuning.evaluation.runner import RunnerContext
    from qlora_finetuning.frontier.config import (
        FrontierConfig,
        load_aws_settings,
        load_bucket,
        load_frontier_config,
    )
    from qlora_finetuning.frontier.export import Export, export_files
    from qlora_finetuning.frontier.export_validation import (
        TokenVolumes,
        load_token_volumes,
        validate_export,
    )
    from qlora_finetuning.frontier.pricing import Pricing, load_pricing
    from qlora_finetuning.frontier.training_estimate import (
        build_gate_record,
        estimate_path_b,
        gate_checks,
        load_token_ratio,
        read_gate_record,
        staged_before,
        stale_files,
        usd,
        write_gate_record,
    )
    from qlora_finetuning.logging_setup import setup_logging
    from qlora_finetuning.settings import ConfigError
    from qlora_finetuning.tracking import Ledger, start_run
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.nova_export"
    ) from None

PASS, FAIL = "PASS", "FAIL"
EXTRA_CONFIGS = ("frontier", "pricing")


def _s3_client(cfg: FrontierConfig) -> tuple[Any, str, str]:
    """The real S3 client, the bucket, and the profile, after a free sign-in check.

    boto3 is loaded only here, so a run without ``--upload`` creates no client.
    """
    from qlora_finetuning.frontier.bedrock import aws_session, check_identity, client_config

    aws = load_aws_settings()
    session = aws_session(aws)
    check_identity(session.client("sts", config=client_config(cfg)), aws)
    return session.client("s3", config=client_config(cfg)), load_bucket(), aws["profile"]


def _relative(path: Any, context: RunnerContext) -> str:
    try:
        return path.relative_to(context.tracking_cfg.root).as_posix()
    except ValueError:
        return str(path)


def _record_run(
    export: Export, context: RunnerContext, seconds: float, staged: dict[str, Any] | None
) -> str:
    """One run in the registry for an export that wrote or staged files."""
    configs = tuple(dict.fromkeys((*context.configs, *EXTRA_CONFIGS)))
    with start_run(
        "export",
        f"nova export {export.export_id}",
        path="B",
        arm="none",
        where="local",
        scope="dev",
        configs=configs,
        split_manifest_hash=export.split_hash,
        description="Bedrock training and validation files" + (", staged in S3" if staged else ""),
        cfg=context.tracking_cfg,
        config_dir=context.config_dir,
        git=context.git,
        environment=context.environment,
    ) as run:
        run.record(
            phase="other",
            examples=sum(file.records for file in export.files.values()),
            wall_clock_seconds=round(seconds, 3),
            source="measured",
            source_detail="records written to the Bedrock training and validation files",
            external_id=export.export_id,
        )
        for role, file in export.files.items():
            run.outputs[f"{role}_sha256"] = file.sha256
            if staged:
                run.outputs[f"{role}_s3_uri"] = staged[role]["s3_uri"]
        return run.run_id


def _print_estimate(estimate: dict[str, Any]) -> None:
    measured = "measured" if estimate["ratio_measured"] else "NOT MEASURED, assumed"
    print(f"Nova tokens per Qwen token: {estimate['token_ratio']} ({measured})")
    print(
        f"training, {estimate['epoch_count']} epochs, both models: "
        f"{usd(estimate['training_total_usd'])} "
        f"({usd(estimate['training_with_margin_usd'])} with the "
        f"{estimate['margin_percent']}% margin)"
    )
    by_epochs = ", ".join(
        f"{epochs}: {usd(amount)}"
        for epochs, amount in estimate["training_total_by_epochs_usd"].items()
    )
    print(f"training by epoch count: {by_epochs}")
    print(
        f"already in the ledger: {usd(estimate['spent_usd'])}; evaluation to come: "
        f"{usd(estimate['evaluation_to_come_total_usd'])}; storage: "
        f"{usd(estimate['storage_total_usd'])}"
    )
    print(f"Path B total: {usd(estimate['total_usd'])} of a cap of {usd(estimate['cap_usd'])}")


def main(
    argv: list[str] | None = None,
    context: RunnerContext | None = None,
    cfg: FrontierConfig | None = None,
    pricing: Pricing | None = None,
    volumes: TokenVolumes | None = None,
    s3_factory: Callable[[FrontierConfig], tuple[Any, str, str]] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Export and price the Bedrock training files.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--upload", action="store_true", help="also stage the files in S3 (owner)")
    group.add_argument(
        "--check", action="store_true", help="write no record; fail if it is out of date"
    )
    args = parser.parse_args(argv)

    setup_logging()
    started = time.perf_counter()
    try:
        context = (context or RunnerContext()).resolved()
        cfg = cfg or load_frontier_config()
        pricing = pricing or load_pricing()
        if volumes is None:
            _, token_json = token_report_paths(load_token_budget_config().report_dir)
            volumes = load_token_volumes(token_json)

        export, _ = export_files(cfg, context)
        ratio = load_token_ratio(cfg, export.prompt_id, export.split_hash)
        checks = validate_export(export, cfg, context, volumes, ratio.value)
        files_ok = all(check["passed"] for check in checks)
        estimate = estimate_path_b(
            cfg,
            pricing,
            volumes,
            ratio,
            Ledger(context.tracking_cfg).effective(),
            test_examples=len(split_ids("test", context.split_cfg)),
            max_new_tokens=context.max_new_tokens,
        )
        checks += gate_checks(estimate, cfg)

        # An export is recorded as a run the first time its gate record is written, and
        # again when it is staged: not on every repeat, which would flood the ledger.
        try:
            first_record = read_gate_record(cfg).get("export_id") != export.export_id
        except ConfigError:
            first_record = True
        staged = staged_before(cfg, export)
        newly_staged = False
        refused = args.upload and not files_ok
        if args.upload and files_ok:
            from qlora_finetuning.frontier.staging import stage_export

            s3, bucket, profile = (s3_factory or _s3_client)(cfg)
            staged = stage_export(s3, export, bucket, cfg, profile=profile)
            newly_staged = True
        record = build_gate_record(export, checks, estimate, cfg, pricing, staged)
    except ConfigError as exc:  # bad config, no snapshot, no sign-in, a staging failure
        print(f"CONFIG ERROR: {exc}")
        return 2

    for role, file in export.files.items():
        print(
            f"{role}: {_relative(file.path, context)}  {file.records:,} records, "
            f"{file.size:,} bytes, {file.repeated_lines:,} repeated lines"
        )
    for check in checks:
        status = PASS if check["passed"] else FAIL
        print(f"{status}  {check['id']}  {check['name']}: {check['detail']}")
    failed = sum(not check["passed"] for check in checks)
    print(f"{len(checks) - failed} passed, {failed} failed")
    _print_estimate(estimate)
    if refused:
        print("REFUSED: nothing was uploaded, because a check X1 to X7 failed")
    if staged:
        for role, item in staged.items():
            print(f"staged {role}: {item['s3_uri']}")

    if args.check:
        stale = stale_files(record, cfg.report_dir)
        if stale:
            print(f"STALE: {', '.join(stale)}; run python -m scripts.nova_export")
            return 1
        print("the gate record is current")
    else:
        markdown_path, json_path = write_gate_record(record, cfg.report_dir)
        if first_record or newly_staged:
            run_id = _record_run(export, context, time.perf_counter() - started, staged)
            print(f"run {run_id}")
        print(f"gate record: {_relative(markdown_path, context)}")
        print(f"gate record: {_relative(json_path, context)}")
    if record["passed"]:
        print("GATE OPEN: every check passed")
    else:
        print("GATE CLOSED: a check failed; no fine-tuning job may be started")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
