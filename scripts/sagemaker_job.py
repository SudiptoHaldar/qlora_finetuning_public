"""Run Path A in AWS: plan, launch, follow, and bring home a SageMaker job (req_003 v3.03).

Usage (from the repo root; the script imports the package, so it runs as a module)::

    venv\\Scripts\\python -m scripts.sagemaker_job bundle
    venv\\Scripts\\python -m scripts.sagemaker_job rehearse smoke
    venv\\Scripts\\python -m scripts.sagemaker_job plan smoke
    venv\\Scripts\\python -m scripts.sagemaker_job launch smoke --yes
    venv\\Scripts\\python -m scripts.sagemaker_job status
    venv\\Scripts\\python -m scripts.sagemaker_job logs --tail 50
    venv\\Scripts\\python -m scripts.sagemaker_job pull
    venv\\Scripts\\python -m scripts.sagemaker_job stop --yes

``bundle`` writes the data bundle, the ``fit`` and ``dev`` examples and nothing of the test
split, under git-ignored ``artifacts/bundle/``. ``rehearse`` runs a task on this machine the
way a job would, from an archive of the code and the bundle, in a temporary folder. Neither
touches AWS.

``plan`` prints the job a launch would create, checks J1 to J11, and the most the job can
cost. It creates nothing and bills nothing.

From req_005 v5.01: ``bundle --split test`` writes the test bundle, the test examples and
nothing else, under git-ignored ``artifacts/bundle_test/``; it answers ``LOCKED`` until the
owner has unlocked the test split. An ``evaluate`` job with ``--split test --scope measured``
is given that bundle (the launch uploads it: the one upload of test rows to S3) and must
pass check J11: the arm is at its frozen configuration and has no test result yet.
``rehearse`` refuses ``--split test``.

**``launch`` and ``stop`` are run by the owner**, with ``--yes``: ``launch`` uploads the
code and the bundle to S3 and creates one billed SageMaker training job; ``stop`` ends one.
A launch is refused unless every check passes. A job is billed by the second while it runs,
up to its maximum runtime.

``status`` asks AWS how the job is doing; ``logs`` prints the end of its log; ``pull`` brings
a finished job's runs, ledger lines, and billable seconds here. All three read from AWS and
bill nothing. Without ``--job`` they mean the last job launched.

From req_003 v3.04: the plan of a ``train``, ``score``, or ``evaluate`` job prints an
estimate, computed from what the last pulled job measured, and a proposed
``--max-runtime``; ``score --run RUN_ID --workspace JOB`` scores the untuned model and
every checkpoint of a training run in one job; ``pull --checkpoint N`` downloads one
checkpoint's adapter; ``keep --yes`` (the owner's) copies the selected adapter to the
bucket's ``adapters/`` folder; ``--adapter selected`` means the checkpoint that
``scripts.qlora_select`` chose.

Tasks: ``smoke`` (the smoke training run and a few scored examples: about 25 minutes),
``train`` (the full training run, or ``--resume RUN_ID`` on the same ``--workspace``), and
``evaluate`` (``--predictor``, ``--split``, and for a tuned model ``--adapter`` and the
``--workspace`` of the job that trained it).

Exit codes: 0 done, or a plan whose checks pass; 1 a check failed, or the job or the
rehearsal failed; 2 a config problem, no sign-in, or ``--yes`` missing.

Needs the snapshot and the split manifest for ``bundle``; a valid sign-in
(``aws sso login --profile qlora``) for everything from ``plan`` on.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

try:
    from qlora_finetuning.comparison.config import load_comparison_config
    from qlora_finetuning.comparison.gate import unlock_problems
    from qlora_finetuning.data import TestLockedError
    from qlora_finetuning.data.bundle import export_bundle
    from qlora_finetuning.data.split import load_split_config
    from qlora_finetuning.local_qlora.job import TASKS
    from qlora_finetuning.local_qlora.job_record import is_final
    from qlora_finetuning.local_qlora.sagemaker import (
        FAIL,
        JobError,
        JobOptions,
        Plan,
        SageMakerJobs,
        make_clients,
        rehearse,
        usd,
    )
    from qlora_finetuning.local_qlora.sagemaker_config import (
        load_aws_names,
        load_sagemaker_config,
    )
    from qlora_finetuning.logging_setup import setup_logging
    from qlora_finetuning.settings import ConfigError
    from qlora_finetuning.tracking import load_tracking_config
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.sagemaker_job"
    ) from None

SELF = "python -m scripts.sagemaker_job"


def _ascii(text: Any) -> str:
    return str(text).encode("ascii", "replace").decode()


def _relative(path: Any, root: Any) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def print_plan(plan: Plan, jobs: SageMakerJobs) -> None:
    definition, cfg = plan.definition, jobs.cfg
    resources, minutes = definition["ResourceConfig"], plan.max_runtime // 60
    print(f"job {plan.name}  (task {plan.task}; arm {plan.arm}; commit {plan.commit[:12]})")
    print(f"image       {definition['AlgorithmSpecification']['TrainingImage']}")
    print(
        f"instance    {resources['InstanceCount']} x {resources['InstanceType']}, "
        f"{resources['VolumeSizeInGB']} GB volume"
    )
    arguments = definition["AlgorithmSpecification"]["ContainerArguments"][2:]
    print(f"runs        python -m scripts.job_run {' '.join(arguments)}")
    for channel in definition["InputDataConfig"]:
        uri = channel["DataSource"]["S3DataSource"]["S3Uri"]
        print(f"input       {channel['ChannelName']}: {uri}")
    print(f"workspace   {definition['CheckpointConfig']['S3Uri']}")
    print(f"role        {definition['RoleArn']}")
    print(
        f"runtime     at most {minutes} minutes: at most {usd(plan.max_cost)} "
        f"(${cfg.hourly()} an hour)"
    )
    if plan.estimate is not None:
        estimate = plan.estimate
        print(f"estimate    {_ascii(estimate.detail)}: about {usd(estimate.cost)}")
        print(f"proposed    --max-runtime {estimate.proposed_max_runtime}")
    for check in plan.checks:
        print(f"{check['status']}  {check['id']}  {check['name']}: {_ascii(check['detail'])}")
    counts = {
        status: sum(c["status"] == status for c in plan.checks) for status in ("PASS", FAIL, "SKIP")
    }
    print(f"{counts['PASS']} passed, {counts[FAIL]} failed, {counts['SKIP']} skipped")


def print_job(job: dict[str, Any], jobs: SageMakerJobs) -> None:
    print(f"job {job['name']}  (task {job['task']}; {job['instance_type']})")
    print(f"status      {job.get('status')} / {job.get('secondary_status')}")
    if job.get("failure_reason"):
        print(f"reason      {_ascii(job['failure_reason'])}")
    if job.get("training_started_at"):
        print(f"started     {job['training_started_at']}")
    if job.get("training_ended_at"):
        print(f"ended       {job['training_ended_at']}")
    seconds = job.get("billable_seconds")
    if seconds is not None:
        cost = jobs.cfg.cost_of(seconds, job["instance_type"], job["instance_count"])
        print(f"billable    {seconds:,} seconds: {usd(cost)} at the list price")
    print(f"workspace   {job['workspace_uri']}")
    print(f"pulled      {'yes' if job.get('pulled') else 'no'}")


def _options(args: argparse.Namespace) -> JobOptions:
    return JobOptions(
        task=args.task,
        attempt=args.attempt,
        workspace=args.workspace,
        resume=args.resume,
        predictor=args.predictor,
        adapter=args.adapter,
        split=args.split,
        limit=args.limit,
        scope=args.scope,
        run=args.run,
        max_runtime=args.max_runtime,
    )


def _add_task_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--attempt", type=int, default=1, help="a new job for the same commit")
    parser.add_argument("--workspace", help="an earlier job's workspace, to continue from it")
    parser.add_argument("--resume", metavar="RUN_ID", help="train or evaluate: continue a run")
    parser.add_argument("--predictor", help="evaluate: qwen3-4b or qwen3-4b-tuned")
    parser.add_argument(
        "--adapter",
        metavar="RUN_ID[/checkpoint-N]",
        help="evaluate: the adapter; 'selected' is the checkpoint of the selection record",
    )
    parser.add_argument("--split", help="evaluate: quick_dev or dev")
    parser.add_argument("--limit", type=int, help="evaluate: the first N examples only")
    parser.add_argument("--scope", default="dev", choices=("dev", "measured"))
    parser.add_argument("--run", metavar="RUN_ID", help="score: the training run to score")
    parser.add_argument(
        "--max-runtime", type=int, metavar="SECONDS", help="this job's limit (default: the task's)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Path A as a SageMaker training job.")
    commands = parser.add_subparsers(dest="command", required=True)
    bundle = commands.add_parser("bundle", help="write the data bundle: fit and dev; no AWS")
    bundle.add_argument(
        "--split",
        choices=("test",),
        help="write the test bundle instead: the test examples alone, after the unlock",
    )
    rehearsal = commands.add_parser("rehearse", help="run a task here as a job would; no AWS")
    _add_task_arguments(rehearsal)
    rehearsal.add_argument("--keep", action="store_true", help="keep the temporary folder")
    _add_task_arguments(commands.add_parser("plan", help="the job and its checks; creates nothing"))
    launch = commands.add_parser("launch", help="THE OWNER: upload and create one billed job")
    _add_task_arguments(launch)
    launch.add_argument("--yes", action="store_true", help="required: this creates a billed job")
    for name, text in (
        ("status", "how the job is doing"),
        ("logs", "the end of the job's log"),
        ("pull", "bring a finished job's runs and billable seconds here"),
        ("stop", "THE OWNER: stop a running job"),
        ("keep", "THE OWNER: copy the selected adapter to the bucket's adapters/ folder"),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument("--job", help="the job's name (default: the last one launched)")
    commands.choices["logs"].add_argument("--tail", type=int, default=50)
    commands.choices["pull"].add_argument(
        "--checkpoints", action="store_true", help="also download every checkpoint"
    )
    commands.choices["pull"].add_argument(
        "--checkpoint", type=int, metavar="N", help="download only the adapter of checkpoint N"
    )
    commands.choices["stop"].add_argument("--yes", action="store_true", help="required")
    commands.choices["keep"].add_argument("--yes", action="store_true", help="required")
    return parser


def main(argv: list[str] | None = None, jobs: SageMakerJobs | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    try:
        if jobs is None:
            cfg, names = load_sagemaker_config(), load_aws_names()
            if args.command in ("bundle", "rehearse"):
                clients = dict.fromkeys(("sagemaker", "s3", "sts", "iam", "quotas", "logs"))
            else:
                clients = make_clients(names)
            jobs = SageMakerJobs(cfg, names, load_tracking_config(), **clients)
        cfg = jobs.cfg

        if args.command == "bundle":
            test = args.split == "test"
            folder = cfg.test_bundle_dir if test else cfg.bundle_dir
            if test:
                # Only after the unlock, and only an unlock that names the freeze record.
                split_cfg = jobs._split_cfg or load_split_config()
                problems = unlock_problems(
                    load_comparison_config(root=jobs.tracking.root), split_cfg
                )
                if problems:
                    raise ConfigError(problems[0])
            record = export_bundle(
                folder,
                ("test",) if test else ("fit", "dev"),
                cfg=jobs._split_cfg,
                data_cfg=jobs._data_cfg,
            )
            counts = ", ".join(f"{name} {count:,}" for name, count in record["splits"].items())
            print(f"bundle: {counts}; {record['examples']:,} examples, {record['bytes']:,} bytes")
            print(f"sha256: {record['sha256']}")
            if test:
                print("the test examples and nothing else: for the two test evaluation jobs")
                print("it holds dataset text and stays under artifacts/; a launch uploads it")
            else:
                print("no test row is in it; it holds dataset text and stays under artifacts/")
            print(f"written to {_relative(folder, cfg.root)}/")
            return 0

        if args.command == "rehearse":
            outcome = rehearse(
                args.task,
                cfg,
                cfg.root,
                jobs.git(),
                arguments=tuple(_task_flags(args)),
                keep=args.keep,
            )
            result = outcome["result"]
            print(f"rehearsal of {args.task}, from {outcome['source']}")
            print(f"bundle sha256 {outcome['bundle_sha256'][:12]}...; a snapshot in the folder: "
                  f"{'yes' if outcome['snapshot_present'] else 'no'}")  # fmt: skip
            print(f"task status: {result.get('status', 'no result file')}")
            if outcome["folder"]:
                print(f"kept: {outcome['folder']}")
            print("REHEARSAL ONLY: nothing was sent to AWS and nothing was added to the ledger")
            return 0 if outcome["exit_code"] == 0 else 1

        if args.command == "plan":
            plan = jobs.plan(_options(args))
            print_plan(plan, jobs)
            print("PLAN ONLY: nothing was created")
            return 0 if plan.passed else 1

        if args.command == "launch":
            plan, job = jobs.launch(_options(args), yes=args.yes)
            print_plan(plan, jobs)
            if job is None:
                print("NOT LAUNCHED: a check failed; nothing was created")
                return 1
            word = "ADOPTED (it already existed)" if job["adopted"] else "LAUNCHED"
            print(f"{word}: {job['name']}")
            print(f"arn: {job['arn']}")
            print(f"follow it with: {SELF} status   |   {SELF} logs")
            print(f"when it has ended: {SELF} pull")
            print("commit the job record: git add reports/local_qlora ledger")
            return 0

        if args.command == "status":
            print_job(jobs.status(args.job), jobs)
            return 0

        if args.command == "logs":
            lines = jobs.logs(args.job, tail=args.tail)
            for line in lines:
                print(_ascii(line))
            if not lines:
                print("no log yet: the job has not started its container")
            return 0

        if args.command == "stop":
            print_job(jobs.stop(args.job, yes=args.yes), jobs)
            return 0

        if args.command == "keep":
            kept = jobs.keep(yes=args.yes)
            print(f"kept: {kept['adapter']}  (sha256 {kept['adapter_sha256'][:12]}...)")
            print(f"at:   {kept['uri']}")
            print("commit the job record: git add reports/local_qlora")
            return 0

        if args.checkpoint is not None:
            fetched = jobs.pull_checkpoint(args.checkpoint, args.job)
            print(f"job {fetched['job']}: {fetched['files']} files downloaded")
            for adapter in fetched["adapters"]:
                print(f"adapter: {adapter}")
                print(f"use it with: python -m scripts.evaluate --predictor qwen3-4b-tuned "
                      f"--adapter {adapter} --split quick_dev")  # fmt: skip
            return 0

        outcome = jobs.pull(args.job, checkpoints=args.checkpoints)
        print_job(outcome["job"], jobs)
        if not outcome["pulled"]:
            print(f"NOT PULLED: {outcome['reason']}; try again when it has ended")
            return 0
        lines = outcome["ledger_lines"]
        print(f"files downloaded: {outcome['files']:,}")
        print(
            f"lines added: runs {lines['runs']}, ledger {lines['ledger']}, "
            f"results {lines['results']}; billable seconds "
            f"{'recorded' if outcome['billable_recorded'] else 'already recorded'}"
        )
        for run in outcome["job"]["runs"]:
            print(f"run {run.get('run_id')}  ({run.get('role', '?')}; {run.get('status', '?')})")
        status = outcome["result"].get("status")
        print(f"task status: {status or 'no result file: the job ended before its task did'}")
        print("commit the job record and the ledger: git add reports/local_qlora ledger")
        ok = is_final(outcome["job"]) and outcome["job"]["status"] == "Completed"
        return 0 if ok and status == "completed" else 1
    except JobError as exc:
        print(f"AWS ERROR: {_ascii(exc)}")
        return 1
    except TestLockedError as exc:  # the test bundle, asked for before the unlock
        print(f"LOCKED: {_ascii(exc)}")
        return 2
    except ConfigError as exc:  # config, no sign-in, --yes missing, no job in the record
        print(f"CONFIG ERROR: {_ascii(exc)}")
        return 2


def _task_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    for flag in ("resume", "predictor", "adapter", "split", "limit", "run"):
        value = getattr(args, flag)
        if value is not None:
            flags += [f"--{flag}", str(value)]
    return flags


if __name__ == "__main__":
    sys.exit(main())
