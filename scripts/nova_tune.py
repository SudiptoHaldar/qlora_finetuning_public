"""Plan, launch, and follow the Bedrock fine-tuning jobs of Path B (req_004 v4.03).

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.nova_tune plan
    venv\\Scripts\\python -m scripts.nova_tune launch --model nova-lite --yes
    venv\\Scripts\\python -m scripts.nova_tune status --wait
    venv\\Scripts\\python -m scripts.nova_tune bill --model nova-lite --usd 14.20
    venv\\Scripts\\python -m scripts.nova_tune deploy --model nova-lite --yes
    venv\\Scripts\\python -m scripts.nova_tune stop --model nova-lite --yes
    venv\\Scripts\\python -m scripts.nova_tune teardown --model nova-lite --all --yes

| Command | Does | Who |
|---|---|---|
| ``plan`` | Shows what a launch would send and cost, and checks T1 to T9. Starts nothing | Anyone |
| ``launch`` | Creates one fine-tuning job. **Billed** | The owner |
| ``status`` | Reads the jobs and deployments; fetches the loss curves of a finished job | Anyone |
| ``bill`` | Enters the amount AWS billed for a job; it replaces the ledger's estimate | The owner |
| ``deploy`` | Creates the on-demand deployment of a finished model | The owner |
| ``stop`` | Stops a running job. Its tokens so far are billed; no model is made | The owner |
| ``teardown`` | Deletes a deployment, and with ``--all`` the tuned model too | The owner |

``launch``, ``deploy``, ``stop``, and ``teardown`` change AWS. Without ``--yes`` each prints
what it would do and stops. Nothing here creates Provisioned Throughput, and no command
launches more than the one model it is told to. Nova Pro is refused until Nova Lite's job
has completed and its billed amount has been entered (check T9).

``plan`` works without a sign-in: the checks that need AWS are then shown as SKIP. Every
other command needs ``aws sso login --profile qlora`` first.

Exit codes: 0 done, or for ``plan`` every check passed, or for ``status`` every job has
completed; 1 a check failed, a command was refused, a job failed or is still running; 2 a
config problem or no sign-in.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:
    from qlora_finetuning.evaluation.runner import RunnerContext
    from qlora_finetuning.frontier.config import (
        FrontierConfig,
        load_aws_settings,
        load_bucket,
        load_customization_role,
        load_frontier_config,
    )
    from qlora_finetuning.frontier.pricing import Pricing, load_pricing
    from qlora_finetuning.frontier.training_estimate import usd
    from qlora_finetuning.frontier.tuning import BedrockTuning, JobPlan, Refused, is_active
    from qlora_finetuning.frontier.tuning_record import (
        COMPLETED,
        Attempt,
        TuningRecord,
        read_tuning_record,
        report_paths,
    )
    from qlora_finetuning.logging_setup import setup_logging
    from qlora_finetuning.settings import ConfigError
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.nova_tune"
    ) from None

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
CHANGES_AWS = ("launch", "deploy", "stop", "teardown")


@dataclass
class Clients:
    """The two AWS clients and the profile they were made for."""

    bedrock: Any
    s3: Any
    profile: str = "qlora"


def _clients(cfg: FrontierConfig) -> Clients:
    """The real clients, after a free sign-in check. Raises ``ConfigError`` without one."""
    from qlora_finetuning.frontier.bedrock import (
        aws_session,
        bedrock_control_client,
        check_identity,
        client_config,
    )

    aws = load_aws_settings()
    session = aws_session(aws)
    check_identity(session.client("sts", config=client_config(cfg)), aws)
    return Clients(
        bedrock=bedrock_control_client(aws, cfg, session=session),
        s3=session.client("s3", config=client_config(cfg)),
        profile=aws["profile"],
    )


def _relative(path: Any, context: RunnerContext) -> str:
    try:
        return path.relative_to(context.tracking_cfg.root).as_posix()
    except ValueError:
        return str(path)


def _print_plan(plan: JobPlan) -> None:
    print(f"{plan.model.name}, attempt {plan.attempt}  (arm {plan.model.tuned_arm})")
    print(f"  job, tuned model, deployment: {plan.job_name}")
    print(f"  base model: {plan.model.customization_model_id}")
    print(f"  training file:   {plan.training_s3_uri or 'not staged'}")
    print(f"  validation file: {plan.validation_s3_uri or 'not staged'}")
    print(f"  output:          {plan.output_s3_uri}")
    settings = ", ".join(f"{key} {value}" for key, value in plan.hyperparameters.items())
    print(f"  hyperparameters: {settings}")
    print(
        f"  estimated: {plan.estimated_training_tokens:,} training tokens, "
        f"{usd(plan.estimated_usd)} ({usd(plan.estimated_with_margin_usd)} with the margin)"
    )


def _print_checks(checks: list[dict[str, Any]]) -> bool:
    for item in checks:
        status = SKIP if item["passed"] is None else (PASS if item["passed"] else FAIL)
        print(f"  {status}  {item['id']}  {item['name']}: {item['detail']}")
    passed = sum(item["passed"] is True for item in checks)
    skipped = sum(item["passed"] is None for item in checks)
    failed = len(checks) - passed - skipped
    print(f"  {passed} passed, {failed} failed, {skipped} skipped")
    return passed == len(checks)


def _line(attempt: Attempt) -> str:
    parts = [f"{attempt.model} a{attempt.attempt}: job {attempt.job_status}"]
    if attempt.failed_in:
        parts.append(f"failed in {attempt.failed_in}")
    if attempt.training_loss is not None:
        parts.append(f"training loss {attempt.training_loss:.4f}")
    if attempt.validation_loss is not None:
        parts.append(f"validation loss {attempt.validation_loss:.4f}")
    if attempt.model_deleted:
        parts.append("tuned model deleted")
    elif attempt.deployment_arn:
        parts.append(f"deployment {attempt.deployment_status}")
    elif attempt.job_status == COMPLETED:
        parts.append("not deployed")
    if attempt.billed_usd:
        parts.append(f"billed {usd(attempt.billed_usd)}")
    else:
        parts.append(f"estimate {usd(attempt.estimated_usd)}")
    return "; ".join(parts)


def _print_record(record: TuningRecord) -> None:
    if not record.attempts:
        print("no job has been launched")
    for attempt in record.attempts:
        print(_line(attempt))
        for note in attempt.notes:
            print(f"  note: {note}")


def _needs_yes(action: str, detail: str) -> int:
    print(f"NOT DONE: this would {detail}.")
    print(f"Run it again with --yes to {action}. It changes AWS, so the owner runs it.")
    return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bedrock fine-tuning jobs for Path B.")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="show what a launch would do; start nothing")
    plan.add_argument("--model", help="one model (default: every model, in launch order)")
    plan.add_argument("--attempt", type=int, default=1)

    launch = commands.add_parser("launch", help="create one fine-tuning job (billed)")
    launch.add_argument("--model", required=True)
    launch.add_argument("--attempt", type=int, default=1)
    launch.add_argument("--yes", action="store_true")

    status = commands.add_parser("status", help="read the jobs and deployments")
    status.add_argument("--wait", action="store_true", help="look again until all have ended")

    bill = commands.add_parser("bill", help="enter the amount AWS billed for a job")
    bill.add_argument("--model", required=True)
    bill.add_argument("--usd", required=True)
    bill.add_argument("--attempt", type=int)

    for name, text in (
        ("deploy", "create the on-demand deployment of a finished model"),
        ("stop", "stop a running job"),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument("--model", required=True)
        command.add_argument("--yes", action="store_true")

    teardown = commands.add_parser("teardown", help="delete a deployment or a tuned model")
    teardown.add_argument("--model", required=True)
    what = teardown.add_mutually_exclusive_group(required=True)
    what.add_argument("--deployment", action="store_true", help="the deployment only")
    what.add_argument("--all", action="store_true", help="the deployment and the tuned model")
    teardown.add_argument("--attempt", type=int)
    teardown.add_argument("--yes", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    context: RunnerContext | None = None,
    cfg: FrontierConfig | None = None,
    pricing: Pricing | None = None,
    client_factory: Callable[[FrontierConfig], Clients] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    setup_logging()
    try:
        context = (context or RunnerContext()).resolved()
        cfg = cfg or load_frontier_config()
        pricing = pricing or load_pricing()
        factory = client_factory or _clients
        offline = args.command == "bill"
        clients: Clients | None = None
        if not offline:
            try:
                clients = factory(cfg)
            except ConfigError as error:
                if args.command == "status" and not read_tuning_record(cfg).attempts:
                    print("no job has been launched")  # nothing to ask AWS about
                    return 0
                if args.command != "plan":
                    raise
                print(f"no sign-in: {error}")
                print("the checks that need AWS are skipped")
        options = {"sleep": sleep} if sleep is not None else {}
        tuning = BedrockTuning(
            cfg,
            pricing,
            context,
            bucket=load_bucket(),
            role_arn=load_customization_role(),
            bedrock=clients.bedrock if clients else None,
            s3=clients.s3 if clients else None,
            profile=clients.profile if clients else "qlora",
            **options,
        )
        return _run(args, tuning, cfg, context)
    except Refused as refusal:
        print(f"REFUSED: {refusal}")
        if refusal.checks:
            _print_checks(refusal.checks)
        return 1
    except ConfigError as error:  # bad config, no sign-in, an AWS failure
        print(f"CONFIG ERROR: {error}")
        return 2


def _run(args: Any, tuning: BedrockTuning, cfg: FrontierConfig, context: RunnerContext) -> int:
    record_path = _relative(report_paths(cfg.report_dir)[0], context)

    if args.command == "plan":
        names = [args.model] if args.model else list(cfg.launch_order)
        all_passed = True
        for name in names:
            plan = tuning.plan(name, args.attempt)
            _print_plan(plan)
            all_passed &= _print_checks(tuning.preflight(plan))
        print("nothing was started")
        return 0 if all_passed else 1

    if args.command == "launch":
        plan = tuning.plan(args.model, args.attempt)
        _print_plan(plan)
        if not args.yes:
            return _needs_yes(
                "launch",
                f"create the fine-tuning job {plan.job_name}, billed at about "
                f"{usd(plan.estimated_usd)}",
            )
        attempt = tuning.launch(plan)
        print(f"LAUNCHED: {attempt.job_arn}")
        print(f"run {attempt.run_id}")
        print(f"tuning record: {record_path}  (commit it)")
        print("follow it with: python -m scripts.nova_tune status --wait")
        return 0

    if args.command == "status":
        record = tuning.wait(on_look=_print_record) if args.wait else tuning.refresh()
        if not args.wait:
            _print_record(record)
        done = all(a.job_status == COMPLETED for a in record.attempts if a.job_arn)
        waiting = [a.model for a in record.attempts if a.deployment_arn and not is_active(a)
                   and not a.deployment_deleted]
        if waiting:
            print(f"deployment not Active yet: {', '.join(waiting)}")
        return 0 if done and not waiting else 1

    if args.command == "bill":
        attempt = tuning.bill(args.model, args.usd, args.attempt)
        print(
            f"BILLED: {attempt.model} a{attempt.attempt} is ${attempt.billed_usd} in the "
            f"ledger; the estimate was ${attempt.estimated_usd}"
        )
        print(f"tuning record: {record_path}  (commit it with the ledger)")
        return 0

    if args.command == "deploy":
        if not args.yes:
            return _needs_yes("deploy", f"create an on-demand deployment for {args.model}")
        attempt = tuning.deploy(args.model)
        print(f"DEPLOYING: {attempt.deployment_arn} ({attempt.deployment_status})")
        print("it can be asked questions once status shows it Active")
        return 0

    if args.command == "stop":
        if not args.yes:
            return _needs_yes(
                "stop",
                f"stop the job of {args.model}; its tokens so far are billed and no model is made",
            )
        attempt = tuning.stop(args.model)
        print(f"STOPPING: {attempt.job_arn}")
        return 0

    everything = bool(args.all)
    what = "the deployment and the tuned model" if everything else "the deployment"
    if not args.yes:
        return _needs_yes("delete", f"delete {what} of {args.model}; it cannot be undone")
    attempt = tuning.teardown(args.model, everything, args.attempt)
    print(f"DELETED: {what} of {attempt.model} a{attempt.attempt}")
    print(f"tuning record: {record_path}  (commit it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
