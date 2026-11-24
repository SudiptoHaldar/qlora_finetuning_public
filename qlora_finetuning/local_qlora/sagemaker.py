"""Path A in AWS: plan, launch, follow, and bring home a SageMaker training job (req_003 v3.03).

A job is given three things: the code as ``git archive`` of a commit, a data bundle of
``fit`` and ``dev`` (never the snapshot), and an S3 workspace that SageMaker keeps in step
with the job's ``artifacts/`` and ``ledger/`` folders. In the job, ``scripts/job_bootstrap.sh``
installs the pinned libraries and runs ``scripts.job_run`` with a task.

Usage::

    from qlora_finetuning.local_qlora.sagemaker import JobOptions, SageMakerJobs, make_clients

    jobs = SageMakerJobs(cfg, names, tracking_cfg, **make_clients(names))
    plan = jobs.plan(JobOptions("smoke"))       # the definition, checks J1 to J8; creates nothing
    plan.passed, plan.max_cost, plan.definition["TrainingJobName"]

    jobs.launch(JobOptions("smoke"), yes=True)  # THE OWNER: uploads, creates one billed job
    jobs.status()                               # reads; updates the job record
    jobs.logs(tail=50)
    jobs.pull()                                 # runs, ledger lines, billable seconds; reads only
    jobs.stop(yes=True)                         # THE OWNER

``launch`` and ``stop`` change AWS and are run by the owner, with ``--yes``. Everything else
reads. A launch refuses unless every check passes, and a job that already exists under the
same name is adopted, never created twice. The ledger gets quantities (billable seconds per
instance type), never dollars; a price is read from ``config/pricing.yaml``.

Every client is passed in, so a test hands over stand-ins. Only this module and
``staging.py`` use botocore under ``local_qlora/``.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from qlora_finetuning.aws_common import (
    AUTH_ERROR_CODES,
    AUTH_EXCEPTIONS,
    aws_session,
    check_identity,
    error_code,
    sign_in_message,
)
from qlora_finetuning.comparison.config import load_comparison_config
from qlora_finetuning.comparison.gate import (
    TEST,
    predictor_differences,
    read_freeze,
    request_problems,
)
from qlora_finetuning.data import load_prompt, manifest_hash, split_ids
from qlora_finetuning.data.bundle import (
    BUNDLE_SPLITS,
    EXAMPLES_NAME,
    RECORD_NAME,
    TEST_SPLITS,
    read_bundle,
    read_record,
)
from qlora_finetuning.data.token_budget import load_token_budget_config
from qlora_finetuning.evaluation.metrics import load_evaluation_config
from qlora_finetuning.evaluation.runner import RunnerContext
from qlora_finetuning.local_qlora import staging
from qlora_finetuning.local_qlora.adapters import (
    ADAPTER_FILES,
    CHECKPOINTS,
    ITERATED_SCHEMA,
    SELECTED,
    TRAINING_NAME,
    adapter_sha256,
    read_selection,
    resolve_adapter,
    selected_adapter,
)
from qlora_finetuning.local_qlora.config import load_local_qlora_config
from qlora_finetuning.local_qlora.estimate import (
    Estimate,
    committed_path_a,
    estimate_job,
    measured_rates,
)
from qlora_finetuning.local_qlora.iteration_record import read_iteration, second_run_made
from qlora_finetuning.local_qlora.job import JOBS_DIR, PREDICTORS, RESULT_NAME, TASKS
from qlora_finetuning.local_qlora.job_record import (
    is_final,
    latest_job,
    read_job_record,
    upsert_job,
    write_job_record,
)
from qlora_finetuning.local_qlora.plan import training_plan
from qlora_finetuning.local_qlora.sagemaker_config import AwsNames, SageMakerConfig
from qlora_finetuning.local_qlora.scoring import expected_settings
from qlora_finetuning.local_qlora.training_config import load_training_config
from qlora_finetuning.logging_setup import get_logger
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking import Ledger, LedgerRecord, start_run
from qlora_finetuning.tracking.config import timestamp
from qlora_finetuning.tracking.jsonl import append_line, read_lines
from qlora_finetuning.tracking.ledger import record_from_sagemaker_job
from qlora_finetuning.tracking.registry import UNKNOWN_COMMIT, read_git_info

# What a job's code is: these paths of a commit, and nothing else.
SOURCE_PATHS = (
    "qlora_finetuning",
    "scripts",
    "config",
    "splits",
    "requirements.txt",
    "requirements-train.txt",
)
SOURCE_NAME = "source.tar.gz"
BOOTSTRAP = "scripts/job_bootstrap.sh"
CODE_DIR = "/opt/ml/code"
CHECKPOINT_DIR = "/opt/ml/checkpoints"
CODE_CHANNEL, BUNDLE_CHANNEL = "code", "bundle"
INPUT_DIR = "/opt/ml/input/data"
# The container's entry point. An entry-point string may hold at most 256 characters, so
# everything else is in the bootstrap script, inside the archive.
ENTRY_LINE = (
    f"mkdir -p {CODE_DIR} && tar -xzf {INPUT_DIR}/{CODE_CHANNEL}/{SOURCE_NAME} -C {CODE_DIR} "
    f'&& exec bash {CODE_DIR}/{BOOTSTRAP} "$@"'
)
MAX_ARGUMENT = 256
MAX_JOB_NAME = 63
LOG_GROUP = "/aws/sagemaker/TrainingJobs"
IN_PROGRESS = "InProgress"
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
RUN_CONFIGS = ("sagemaker", "pricing", "aws", "training", "local_qlora", "token_budget")
CENT = Decimal("0.01")

logger = get_logger(__name__)


class JobError(Exception):
    """AWS refused a call about a job, or a job is not in a state for what was asked."""


@dataclass(frozen=True)
class JobOptions:
    """What to run. ``workspace`` defaults to the job's own name: a fresh workspace."""

    task: str
    attempt: int = 1
    workspace: str | None = None
    resume: str | None = None
    predictor: str | None = None
    adapter: str | None = None
    split: str | None = None
    limit: int | None = None
    scope: str = "dev"
    run: str | None = None  # score: the training run whose checkpoints are scored
    max_runtime: int | None = None  # seconds; the task's own limit if not given


@dataclass
class Plan:
    """What a launch would do, and whether it may."""

    name: str
    task: str
    commit: str
    workspace: str
    arm: str
    definition: dict[str, Any]
    max_cost: Decimal
    committed: Decimal
    cap: Decimal
    checks: list[dict[str, str]] = field(default_factory=list)
    options: JobOptions | None = None  # as launched: an adapter called "selected" is resolved
    max_runtime: int = 0
    estimate: Estimate | None = None

    @property
    def passed(self) -> bool:
        return all(check["status"] == PASS for check in self.checks)


def usd(amount: Decimal) -> str:
    """An amount for reading: dollars and cents, half a cent rounded up."""
    return f"${amount.quantize(CENT, rounding=ROUND_HALF_UP)}"


# --------------------------------------------------------------------------- the code


def _git(root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(["git", *arguments], cwd=root, capture_output=True, check=False)
    except OSError as error:
        raise ConfigError(f"git could not be run: {type(error).__name__}") from None
    if completed.returncode != 0:
        first = (completed.stderr.decode("utf-8", "replace").strip().splitlines() or [""])[0]
        command = next((a for a in arguments if not a.startswith("-") and "=" not in a), "")
        raise ConfigError(f"git {command} failed: {first}")
    return completed.stdout


def build_source(root: Path | str, commit: str) -> bytes:
    """The code of ``commit`` as a compressed archive: the paths of ``SOURCE_PATHS`` only.

    Line endings are the repository's own (``core.autocrlf`` is switched off for the call):
    with it on, git would hand a Windows machine a shell script that Linux cannot run.
    """
    return _git(
        Path(root),
        "-c",
        "core.autocrlf=false",
        "archive",
        "--format=tar.gz",
        commit,
        "--",
        *SOURCE_PATHS,
    )


def working_tree_files(root: Path | str) -> list[str]:
    """The files of ``SOURCE_PATHS`` on disk now, committed or not; ignored ones left out."""
    listing = _git(
        Path(root),
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        *SOURCE_PATHS,
    )
    names = [name for name in listing.decode("utf-8").split("\0") if name]
    return sorted(name for name in names if (Path(root) / name).is_file())


# --------------------------------------------------------------------------- the job definition


def job_name(cfg: SageMakerConfig, task: str, commit: str, attempt: int = 1) -> str:
    name = f"{cfg.name_prefix}-{task}-{commit[:8]}-a{attempt}"
    if len(name) > MAX_JOB_NAME or attempt < 1:
        raise ConfigError(f"{name!r} is not a job name (at most {MAX_JOB_NAME} characters)")
    return name


def task_arguments(options: JobOptions) -> list[str]:
    """The arguments of ``scripts.job_run`` for these options, after the task and ``--where``."""
    arguments: list[str] = []
    if options.scope != "dev":
        arguments += ["--scope", options.scope]
    for flag, value in (
        ("--resume", options.resume),
        ("--predictor", options.predictor),
        ("--adapter", options.adapter),
        ("--split", options.split),
        ("--limit", options.limit),
        ("--run", options.run),
    ):
        if value is not None:
            arguments += [flag, str(value)]
    return arguments


def is_test_job(options: JobOptions) -> bool:
    """An evaluation job on the test split: one of the two test jobs of req_005 v5.01."""
    return options.task == "evaluate" and options.split == TEST


def _is_test_arguments(arguments: list[str] | tuple[str, ...]) -> bool:
    """Whether a job's recorded arguments ask for the test split."""
    arguments = list(arguments)
    return any(
        flag == "--split" and value == TEST
        for flag, value in zip(arguments, arguments[1:], strict=False)
    )


def job_arm(cfg: SageMakerConfig, options: JobOptions) -> str:
    """The arm a job's billable time is recorded under."""
    arm = cfg.task(options.task).arm
    if arm is not None:
        return arm
    if options.predictor not in PREDICTORS:
        raise ConfigError(f"an evaluate job needs --predictor, one of: {', '.join(PREDICTORS)}")
    return str(PREDICTORS[options.predictor]["arm"])


def job_definition(
    cfg: SageMakerConfig,
    names: AwsNames,
    options: JobOptions,
    *,
    name: str,
    workspace: str,
    commit: str,
    code_uri: str,
    bundle_uri: str,
) -> dict[str, Any]:
    """The ``CreateTrainingJob`` request. Pure: no client, no clock, no file."""
    task = cfg.task(options.task)
    arguments = [ENTRY_LINE, "job", task.name, "--where", "sagemaker", *task_arguments(options)]
    too_long = [item for item in arguments if len(item) > MAX_ARGUMENT]
    if too_long:
        raise ConfigError(
            f"an entry-point string is over {MAX_ARGUMENT} characters: {too_long[0][:40]}..."
        )

    def channel(channel_name: str, uri: str) -> dict[str, Any]:
        return {
            "ChannelName": channel_name,
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": uri,
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "InputMode": "File",
        }

    bucket = f"s3://{names.bucket}"
    return {
        "TrainingJobName": name,
        "RoleArn": names.role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": cfg.image,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["bash", "-c"],
            "ContainerArguments": arguments,
        },
        "InputDataConfig": [channel(CODE_CHANNEL, code_uri), channel(BUNDLE_CHANNEL, bundle_uri)],
        "OutputDataConfig": {"S3OutputPath": f"{bucket}/{cfg.output_prefix}/"},
        "ResourceConfig": {
            "InstanceType": cfg.instance_type,
            "InstanceCount": cfg.instance_count,
            "VolumeSizeInGB": cfg.volume_gb,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": options.max_runtime or task.max_runtime_seconds
        },
        "CheckpointConfig": {
            "S3Uri": f"{bucket}/{cfg.workspace_prefix}/{workspace}/",
            "LocalPath": CHECKPOINT_DIR,
        },
        "Environment": {
            "QLORA_GIT_COMMIT": commit,
            "QLORA_JOB_NAME": name,
            "QLORA_DATA_BUNDLE": f"{INPUT_DIR}/{BUNDLE_CHANNEL}",
            "PYTHONUNBUFFERED": "1",
            "PIP_NO_CACHE_DIR": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "Tags": [{"Key": key, "Value": value} for key, value in sorted(cfg.tags.items())]
        + [{"Key": "task", "Value": task.name}, {"Key": "commit", "Value": commit[:12]}],
    }


# --------------------------------------------------------------------------- clients


def make_clients(names: AwsNames) -> dict[str, Any]:
    """The real clients, for the configured profile and region. Makes no network call."""
    session = aws_session(names.settings)
    return {
        "sagemaker": session.client("sagemaker"),
        "s3": session.client("s3"),
        "sts": session.client("sts"),
        "iam": session.client("iam"),
        "quotas": session.client("service-quotas"),
        "logs": session.client("logs"),
    }


class _LaunchRun:
    """Enough of a ``Run`` for ``record_from_sagemaker_job``: the launch run of a job."""

    def __init__(self, job: dict[str, Any]) -> None:
        self.job = job

    def new_record(self, **fields_: Any) -> LedgerRecord:
        return LedgerRecord(
            run_id=self.job["launch_run_id"],
            path="A",
            arm=self.job["arm"],
            where="sagemaker",
            **fields_,
        )


class SageMakerJobs:
    """Plan, launch, follow, and pull the jobs of Path A."""

    def __init__(
        self,
        cfg: SageMakerConfig,
        names: AwsNames,
        tracking_cfg: Any,
        *,
        sagemaker: Any,
        s3: Any,
        sts: Any,
        iam: Any,
        quotas: Any,
        logs: Any,
        git: Any = None,
        environment: dict[str, Any] | None = None,
        split_cfg: Any = None,
        data_cfg: Any = None,
        config_dir: Path | str | None = None,
    ) -> None:
        self.cfg, self.names, self.tracking = cfg, names, tracking_cfg
        self.sagemaker, self.s3, self.sts = sagemaker, s3, sts
        self.iam, self.quotas, self.logs_client = iam, quotas, logs
        self._git, self._environment = git, environment
        self._split_cfg, self._data_cfg, self._config_dir = split_cfg, data_cfg, config_dir

    # ------------------------------------------------------------------ helpers

    def _call(
        self, operation: Any, what: str, allow: tuple[str, ...] = (), **arguments: Any
    ) -> Any:
        """Call AWS. A code in ``allow`` gives ``None``; no sign-in is a ``ConfigError``."""
        try:
            return operation(**arguments)
        except AUTH_EXCEPTIONS:
            raise ConfigError(sign_in_message(self.names.profile)) from None
        except ClientError as error:
            code = error_code(error)
            if code in AUTH_ERROR_CODES:
                raise ConfigError(sign_in_message(self.names.profile)) from None
            if code in allow:
                return None
            raise JobError(f"AWS answered {code} when asked to {what}") from None
        except BotoCoreError as error:
            raise JobError(f"AWS could not be reached to {what}: {type(error).__name__}") from None

    def git(self) -> Any:
        if self._git is not None:
            return self._git
        return read_git_info(
            self.tracking.root, self.tracking.git_commit_env, (self.tracking.ledger_prefix,)
        )

    def _record(self) -> dict[str, Any]:
        return read_job_record(self.cfg.record_file)

    def _save(self, job: dict[str, Any]) -> None:
        write_job_record(self.cfg.record_file, upsert_job(self._record(), job))

    def bundle_key_prefix(self, record: dict[str, Any]) -> str:
        return (
            f"{self.cfg.bundle_prefix}/"
            f"{str(record['manifest_hash'])[:12]}-{str(record['sha256'])[:12]}/"
        )

    # ------------------------------------------------------------------ cost

    def committed(self, excluding: str | None = None) -> Decimal:
        """What Path A has cost so far, and may yet cost.

        Billable seconds in the ledger at the price of their instance type, plus the
        maximum of every job that was launched and not yet pulled.
        """
        return committed_path_a(self.cfg, self.tracking, self._record(), excluding)

    # ------------------------------------------------------------------ checks

    def _check_signed_in(self) -> dict[str, str]:
        try:
            check_identity(self.sts, self.names.settings)
        except ConfigError as error:
            return _result("J1", "Signed in", FAIL, str(error))
        return _result("J1", "Signed in", PASS, f"account {self.names.account_id}")

    def _check_commit(self, git: Any) -> dict[str, str]:
        if git.commit == UNKNOWN_COMMIT:
            return _result("J2", "A commit", FAIL, "the commit is unknown: not a git repository")
        if git.dirty:
            shown = ", ".join(git.changed[:3]) + (" ..." if len(git.changed) > 3 else "")
            return _result(
                "J2",
                "A commit",
                FAIL,
                f"uncommitted changes ({shown}): a job runs a commit; commit first",
            )
        return _result("J2", "A commit", PASS, f"commit {git.commit[:12]}, working tree clean")

    def _bundle_dir(self, options: JobOptions) -> Path:
        """The bundle a job is given: the test examples alone for a test evaluation job
        (req_005 v5.01), ``fit`` and ``dev`` for every other job."""
        if is_test_job(options):
            return self.cfg.test_bundle_dir
        return self.cfg.bundle_dir

    def _check_bundle(self, options: JobOptions) -> dict[str, str]:
        test_job = is_test_job(options)
        try:
            record, examples = read_bundle(
                self._bundle_dir(options), self._split_cfg, self._data_cfg
            )
            current = manifest_hash(self._split_cfg)
        except ConfigError as error:
            hint = " (write it with: bundle --split test)" if test_job else ""
            return _result("J3", "Bundle", FAIL, f"{error}{hint}")
        wanted = TEST_SPLITS if test_job else BUNDLE_SPLITS
        if not set(record["splits"]) <= set(wanted):
            return _result(
                "J3",
                "Bundle",
                FAIL,
                f"this job is given {' and '.join(wanted)}, and the bundle holds "
                f"{' and '.join(record['splits'])}",
            )
        counts = ", ".join(f"{name} {count:,}" for name, count in record["splits"].items())
        holds = "test rows only" if test_job else "no test id"
        return _result(
            "J3",
            "Bundle",
            PASS,
            f"{counts}; {len(examples):,} examples, {holds}; manifest {current[:12]}...; "
            f"sha256 {record['sha256'][:12]}...",
        )

    def _runner_context(self) -> RunnerContext:
        """What the gate on a test run reads, for this launcher's repository."""
        return RunnerContext(
            config=load_evaluation_config(root=self.tracking.root),
            split_cfg=self._split_cfg,
            data_cfg=self._data_cfg,
            tracking_cfg=self.tracking,
            git=self._git,
            config_dir=self._config_dir,
        ).resolved()

    def _check_test_job(self, options: JobOptions) -> dict[str, str]:
        """J11: a job on the test split is one the freeze and the unlock allow (req_005 v5.01).

        The gate runs again inside the job, where the predictor is real. Here it is asked
        from this machine, with the local ledger and the job record, which the job cannot
        see: an arm that already has its result, or a test job that was never pulled.
        """
        name = "Test run"
        if options.split != TEST:
            return _result("J11", name, PASS, "not a job on the test split")
        if options.task != "evaluate":
            return _result(
                "J11",
                name,
                FAIL,
                f"only an evaluate job may read the test split, not a {options.task} job",
            )
        try:
            arm = job_arm(self.cfg, options)
            ctx = self._runner_context()
            problems = request_problems(
                arm=arm,
                name=str(options.predictor),
                where="sagemaker",
                scope=options.scope,
                limit=options.limit,
                resume=options.resume,
                ctx=ctx,
                planning=True,
            )
            if problems:
                return _result("J11", name, FAIL, problems[0])
            budget = load_token_budget_config()
            adapter_hash = None
            if options.adapter:
                training_cfg = load_training_config(root=self.tracking.root)
                selection = read_selection(training_cfg.selection_record)["selected"]
                if options.adapter != selection["adapter"]:
                    return _result(
                        "J11",
                        name,
                        FAIL,
                        f"arm {arm} is the selected checkpoint ({selection['adapter']}), not "
                        f"{options.adapter}: use --adapter {SELECTED}",
                    )
                adapter_hash = selection.get("adapter_sha256")
            expected = {
                "name": options.predictor,
                "model_id": budget.model_id,
                "settings": expected_settings(
                    load_local_qlora_config(self._config_dir), budget.revision, adapter_hash
                ),
            }
            frozen = (read_freeze(load_comparison_config(root=self.tracking.root)) or {})["arms"]
            differs = predictor_differences(frozen[arm]["predictor"], expected)
        except ConfigError as error:  # locked, no freeze record, no selection, a bad config
            return _result("J11", name, FAIL, str(error))
        if differs:
            return _result(
                "J11",
                name,
                FAIL,
                f"the predictor of arm {arm} is not the frozen one: its {', '.join(differs)} "
                "differ from the freeze record",
            )
        waiting = [
            job["name"]
            for job in self._record()["jobs"]
            if job.get("arm") == arm
            and not job.get("pulled")
            and _is_test_arguments(job.get("arguments") or [])
        ]
        if waiting:
            return _result(
                "J11",
                name,
                FAIL,
                f"an earlier test job of arm {arm} has not been pulled ({waiting[0]}): pull it "
                "first, then resume its run or see its result",
            )
        return _result(
            "J11", name, PASS, f"arm {arm} at the frozen configuration; no test result yet"
        )

    def _check_role(self) -> dict[str, str]:
        role_name = self.names.role_arn.rsplit("/", 1)[-1]
        response = self._call(
            self.iam.get_role,
            "read the execution role",
            allow=("NoSuchEntity",),
            RoleName=role_name,
        )
        if response is None:
            return _result("J4", "Role", FAIL, f"the role {role_name} does not exist")
        document = response.get("Role", {}).get("AssumeRolePolicyDocument") or {}
        services: list[str] = []
        for statement in document.get("Statement", []):
            service = (statement.get("Principal") or {}).get("Service", [])
            services += [service] if isinstance(service, str) else list(service)
        if self.names.role_trusts not in services:
            return _result(
                "J4", "Role", FAIL, f"{role_name} does not trust {self.names.role_trusts}"
            )
        return _result("J4", "Role", PASS, f"{role_name} trusts {self.names.role_trusts}")

    def _check_quota(self) -> dict[str, str]:
        wanted = f"{self.cfg.instance_type} for {self.names.quota_usage}"
        pages = self._call(
            lambda: list(
                self.quotas.get_paginator("list_service_quotas").paginate(
                    ServiceCode=self.names.quota_service
                )
            ),
            "read the service quotas",
        )
        for page in pages:
            for quota in page.get("Quotas", []):
                if quota.get("QuotaName") == wanted:
                    value = int(quota.get("Value", 0))
                    status = PASS if value >= self.cfg.instance_count else FAIL
                    return _result("J5", "Quota", status, f"{wanted}: {value}")
        return _result("J5", "Quota", FAIL, f"no quota called {wanted!r}")

    def _check_idle(self, name: str) -> dict[str, str]:
        response = self._call(
            self.sagemaker.list_training_jobs,
            "list the training jobs",
            NameContains=f"{self.cfg.name_prefix}-",
            StatusEquals=IN_PROGRESS,
            MaxResults=20,
        )
        running = [
            item["TrainingJobName"]
            for item in response.get("TrainingJobSummaries", [])
            if item["TrainingJobName"] != name
        ]
        if running:
            return _result("J6", "Nothing running", FAIL, f"in progress: {', '.join(running)}")
        return _result("J6", "Nothing running", PASS, "no other job of this project is in progress")

    def _check_cap(self, name: str, max_cost: Decimal) -> tuple[dict[str, str], Decimal]:
        try:
            committed = self.committed(excluding=name)
        except ConfigError as error:
            return _result("J7", "Within the cap", FAIL, str(error)), Decimal(0)
        total, cap = committed + max_cost, self.cfg.path_a_cap_usd
        detail = (
            f"Path A so far {usd(committed)} + at most {usd(max_cost)} for this job = "
            f"{usd(total)} of the {usd(cap)} cap"
        )
        return _result("J7", "Within the cap", PASS if total <= cap else FAIL, detail), committed

    # ------------------------------------------------------------------ the estimate (v3.04)

    def _identity(self) -> dict[str, str]:
        """What an adapter must have been trained for, to be arm A1 of this experiment."""
        budget = load_token_budget_config()
        return {
            "model_id": budget.model_id,
            "revision": budget.revision,
            "prompt_id": load_prompt().prompt_id,
            "manifest_hash": manifest_hash(self._split_cfg),
        }

    def _prepared(self, options: JobOptions) -> JobOptions:
        """The options as they will be launched; refuses ones that cannot work.

        ``--adapter selected`` becomes the checkpoint of the selection record. A job that
        continues another's work (a resumed run, a tuned evaluation, a scoring) must name
        that job's workspace: on a fresh one there would be nothing to continue.
        """
        if options.max_runtime is not None and (
            isinstance(options.max_runtime, bool) or options.max_runtime < 600
        ):
            raise ConfigError("--max-runtime is in seconds and must be at least 600")
        if options.task == "score" and not options.run:
            raise ConfigError("a score job needs --run, the id of the training run to score")
        if options.adapter == SELECTED:
            training_cfg = load_training_config(root=self.tracking.root)
            name = selected_adapter(training_cfg.selection_record, **self._identity())
            options = dataclasses.replace(options, adapter=name)
        continues = options.task == "score" or options.resume or options.adapter
        if continues and not options.workspace:
            raise ConfigError(
                "this job continues an earlier job's work (a training run, its checkpoints): "
                "add --workspace, the name of the job that made them"
            )
        return options

    def estimate(self, options: JobOptions) -> Estimate | None:
        """What the job is expected to take, from what pulled jobs measured; ``None`` if nothing.

        The smoke job has no estimate: it is the job that measures.
        """
        if options.task == "smoke":
            return None
        rates = measured_rates(self._record(), self.tracking.root, self.cfg.instance_type)
        try:
            counts = read_record(self._bundle_dir(options))["splits"]
        except ConfigError:
            return None  # J3 says why
        training_cfg = load_training_config(root=self.tracking.root)
        fit, dev = int(counts.get("fit", 0)), int(counts.get("dev", 0))
        if options.task == "train":
            return estimate_job("train", self.cfg, rates, sequences=fit * training_cfg.epochs)
        split = options.split or "dev"
        examples = dev if split == "dev" else len(split_ids(split, self._split_cfg))
        if options.limit is not None:
            examples = min(examples, options.limit)
        if options.task == "score":
            plan = training_plan([1] * max(fit, training_cfg.batch_size), training_cfg, smoke=False)
            return estimate_job(
                "score", self.cfg, rates, examples=examples, checkpoints=len(plan.checkpoint_steps)
            )
        return estimate_job(
            "evaluate", self.cfg, rates, examples=examples, tuned=bool(options.adapter)
        )

    def _check_estimate(
        self, options: JobOptions, estimate: Estimate | None, max_runtime: int
    ) -> dict[str, str]:
        name = "Estimated"
        if options.task == "smoke":
            return _result("J9", name, PASS, "the smoke job is what measures; it has no estimate")
        if estimate is None:
            return _result(
                "J9",
                name,
                FAIL,
                f"no pulled job on {self.cfg.instance_type} has measured this kind of work: "
                "run and pull the smoke job first (launch smoke --yes, then pull)",
            )
        detail = (
            f"{estimate.detail}; about {usd(estimate.cost)}; measured by "
            f"{', '.join(estimate.sources)}"
        )
        if estimate.seconds > max_runtime:
            return _result(
                "J9",
                name,
                FAIL,
                f"{detail}. That is over this job's limit of {max_runtime:,} s: launch with "
                f"--max-runtime {estimate.proposed_max_runtime}",
            )
        return _result("J9", name, PASS, detail)

    def _second_full_run(self, options: JobOptions) -> bool:
        """A full training run from its start, while a checkpoint is already selected."""
        if options.task != "train" or options.resume:
            return False
        return load_training_config(root=self.tracking.root).selection_record.is_file()

    def _check_registered(self, options: JobOptions) -> dict[str, str]:
        """J10: a second full training run is the one registered corrective iteration."""
        name = "Registered"
        if options.task != "train" or options.resume:
            return _result("J10", name, PASS, "not a full training run from its start")
        training_cfg = load_training_config(root=self.tracking.root)
        if not training_cfg.selection_record.is_file():
            return _result("J10", name, PASS, "no run is selected yet: this is the first full run")
        try:
            selection = read_selection(training_cfg.selection_record)
            iteration = read_iteration(training_cfg.iteration_record)
        except ConfigError as error:
            return _result("J10", name, FAIL, str(error))
        record = self._record()
        made = second_run_made(iteration, record)
        if made:
            return _result(
                "J10",
                name,
                FAIL,
                f"the one corrective iteration has been made ({made}); a further training "
                "run is outside this project's plan",
            )
        if iteration is None or iteration["settings_sha256"] != training_cfg.settings_sha256(False):
            return _result(
                "J10",
                name,
                FAIL,
                f"a checkpoint is already selected ({selection['training_run']}). A second "
                "full run is the one corrective iteration and is registered first: edit one "
                "value in config/training.yaml, then python -m scripts.qlora_iterate propose",
            )
        first = selection["iterations"][0] if selection["schema"] == ITERATED_SCHEMA else selection
        chain = set(first.get("chain") or [first["training_run"]])
        first_job = next(
            (
                job
                for job in record["jobs"]
                if job.get("task") == "train"
                and chain & {entry.get("run_id") for entry in job.get("runs", [])}
            ),
            None,
        )
        wanted = first_job.get("workspace") if first_job else None
        if (wanted and options.workspace != wanted) or not options.workspace:
            return _result(
                "J10",
                name,
                FAIL,
                "the second run is made on the first training job's workspace, where the "
                f"untuned result is: add --workspace {wanted or '<first training job>'}",
            )
        return _result(
            "J10",
            name,
            PASS,
            f"the registered change: {iteration['variable']} from {iteration['from']} to "
            f"{iteration['to']} (finding {iteration.get('finding')})",
        )

    def _check_priced(self) -> dict[str, str]:
        price = self.cfg.hourly()  # the loader has already refused an instance type without one
        return _result(
            "J8",
            "Priced and pinned",
            PASS,
            f"{self.cfg.instance_type} at ${price} an hour (price list of "
            f"{self.cfg.prices_published}); image tag {self.cfg.image.rsplit(':', 1)[-1]}",
        )

    # ------------------------------------------------------------------ plan and launch

    def plan(self, options: JobOptions) -> Plan:
        """The job definition, the estimate, and checks J1 to J11. Creates and bills nothing."""
        if options.task not in TASKS:
            raise ConfigError(f"unknown task {options.task!r}; use one of: {', '.join(TASKS)}")
        options = self._prepared(options)
        git = self.git()
        commit = git.commit
        name = job_name(self.cfg, options.task, commit, options.attempt)
        workspace = options.workspace or name
        arm = job_arm(self.cfg, options)
        max_runtime = options.max_runtime or self.cfg.task(options.task).max_runtime_seconds
        max_cost = self.cfg.max_cost(options.task, max_runtime)

        try:
            bundle_prefix = self.bundle_key_prefix(read_record(self._bundle_dir(options)))
        except ConfigError:
            bundle_prefix = f"{self.cfg.bundle_prefix}/<no bundle yet>/"
        bucket = f"s3://{self.names.bucket}"
        definition = job_definition(
            self.cfg,
            self.names,
            options,
            name=name,
            workspace=workspace,
            commit=commit,
            code_uri=f"{bucket}/{self.cfg.code_prefix}/{commit}/",
            bundle_uri=f"{bucket}/{bundle_prefix}",
        )

        signed_in = self._check_signed_in()
        checks = [signed_in, self._check_commit(git), self._check_bundle(options)]
        if signed_in["status"] == PASS:
            checks += [self._check_role(), self._check_quota(), self._check_idle(name)]
        else:
            checks += [
                _result(check_id, title, SKIP, "needs a sign-in")
                for check_id, title in (("J4", "Role"), ("J5", "Quota"), ("J6", "Nothing running"))
            ]
        cap_check, committed = self._check_cap(name, max_cost)
        estimate = self.estimate(options)
        checks += [
            cap_check,
            self._check_priced(),
            self._check_estimate(options, estimate, max_runtime),
            self._check_registered(options),
            self._check_test_job(options),
        ]
        return Plan(
            name=name,
            task=options.task,
            commit=commit,
            workspace=workspace,
            arm=arm,
            definition=definition,
            max_cost=max_cost,
            committed=committed,
            cap=self.cfg.path_a_cap_usd,
            checks=checks,
            options=options,
            max_runtime=max_runtime,
            estimate=estimate,
        )

    def launch(self, options: JobOptions, yes: bool = False) -> tuple[Plan, dict[str, Any] | None]:
        """Upload the code and the bundle and create one job. **Run by the owner.**

        Returns the plan and the job's entry in the job record, or ``None`` for the job when
        a check failed and nothing was created.
        """
        if not yes:
            raise ConfigError(
                "launch creates a billed SageMaker job and S3 objects: it is run by the owner, "
                "with --yes. Use plan to see what it would do"
            )
        plan = self.plan(options)
        if not plan.passed:
            return plan, None
        options = plan.options or options  # an adapter called "selected" is resolved here

        profile, bucket = self.names.profile, self.names.bucket
        source = build_source(self.tracking.root, plan.commit)
        code = staging.put_checked(
            self.s3,
            bucket,
            f"{self.cfg.code_prefix}/{plan.commit}/{SOURCE_NAME}",
            source,
            profile=profile,
            content_type="application/gzip",
        )
        # For a test job this is the test bundle: the one upload of test rows to S3.
        bundle_dir = self._bundle_dir(options)
        record = read_record(bundle_dir)
        prefix = self.bundle_key_prefix(record)
        for file_name, kind in (
            (EXAMPLES_NAME, "application/jsonl"),
            (RECORD_NAME, "application/json"),
        ):
            content = (bundle_dir / file_name).read_bytes()
            staging.put_checked(
                self.s3, bucket, prefix + file_name, content, profile=profile, content_type=kind
            )

        with start_run(
            "other",
            f"sagemaker {plan.task} {plan.name}",
            path="A",
            arm=plan.arm,
            where="sagemaker",
            scope=options.scope,
            configs=RUN_CONFIGS,
            split_manifest_hash=manifest_hash(self._split_cfg),
            description=f"launches the SageMaker job {plan.name}",
            cfg=self.tracking,
            config_dir=self._config_dir,
            git=self._git,
            environment=self._environment,
        ) as run:
            response = self._call(
                self.sagemaker.create_training_job,
                "create the job",
                allow=("ResourceInUse",),
                **plan.definition,
            )
            adopted = response is None
            description = self._describe(plan.name)  # also when adopted: the job that exists
            job = {
                "name": plan.name,
                "arn": description["TrainingJobArn"],
                "task": plan.task,
                "arguments": task_arguments(options),
                "arm": plan.arm,
                "phase": self.cfg.task(plan.task).phase,
                "commit": plan.commit,
                "image": self.cfg.image,
                "instance_type": self.cfg.instance_type,
                "instance_count": self.cfg.instance_count,
                "max_runtime_seconds": plan.max_runtime,
                "max_cost_usd": str(plan.max_cost),
                "workspace": plan.workspace,
                "workspace_uri": plan.definition["CheckpointConfig"]["S3Uri"],
                "code_uri": code["s3_uri"],
                "code_sha256": code["sha256"],
                "bundle_uri": f"s3://{bucket}/{prefix}",
                "bundle_sha256": record["sha256"],
                "launch_run_id": run.run_id,
                "launched_at": timestamp(),
                "adopted": adopted,
                "pulled": False,
                "runs": [],
            }
            if self._second_full_run(options):
                job["iteration"] = True  # the second run of the corrective iteration
            job.update(_job_state(description))
            self._save(job)  # the moment the ARN is known
            run.outputs.update(job=plan.name, arn=job["arn"], workspace=plan.workspace)
        return plan, job

    # ------------------------------------------------------------------ following a job

    def _describe(self, name: str) -> dict[str, Any]:
        return self._call(
            self.sagemaker.describe_training_job, f"describe the job {name}", TrainingJobName=name
        )

    def status(self, name: str | None = None) -> dict[str, Any]:
        """Ask AWS how the job is doing and update the job record. Reads only."""
        job = dict(latest_job(self._record(), name))
        job.update(_job_state(self._describe(job["name"])))
        self._save(job)
        return job

    def logs(self, name: str | None = None, tail: int = 50) -> list[str]:
        """The last ``tail`` lines of the job's log. Reads only."""
        job = latest_job(self._record(), name)
        streams = self._call(
            self.logs_client.describe_log_streams,
            "find the job's log",
            allow=("ResourceNotFoundException",),
            logGroupName=LOG_GROUP,
            logStreamNamePrefix=f"{job['name']}/",
        )
        found = (streams or {}).get("logStreams", [])
        if not found:
            return []
        events = self._call(
            self.logs_client.get_log_events,
            "read the job's log",
            logGroupName=LOG_GROUP,
            logStreamName=found[0]["logStreamName"],
            limit=tail,
            startFromHead=False,
        )
        return [str(event.get("message", "")).rstrip() for event in events.get("events", [])]

    def stop(self, name: str | None = None, yes: bool = False) -> dict[str, Any]:
        """Stop the job. **Run by the owner.** It is billed for the time it ran."""
        if not yes:
            raise ConfigError("stop ends a running job: it is run by the owner, with --yes")
        job = latest_job(self._record(), name)
        self._call(
            self.sagemaker.stop_training_job,
            f"stop the job {job['name']}",
            TrainingJobName=job["name"],
        )
        return self.status(job["name"])

    # ------------------------------------------------------------------ bringing it home

    def pull(self, name: str | None = None, checkpoints: bool = False) -> dict[str, Any]:
        """Bring a finished job's runs, ledger lines, and billable seconds here.

        Downloads the run folders and adapters of the job's runs (checkpoints only when
        asked), appends the lines the job wrote to the ledger and the results file once,
        and records the billable seconds. Changes nothing in AWS. Pulling again adds nothing.
        """
        job = dict(latest_job(self._record(), name))
        description = self._describe(job["name"])
        job.update(_job_state(description))
        if not is_final(job):
            self._save(job)
            return {"job": job, "pulled": False, "reason": f"the job is {job['status']}"}

        profile, bucket = self.names.profile, self.names.bucket
        prefix = f"{self.cfg.workspace_prefix}/{job['workspace']}/"
        keys = staging.list_keys(self.s3, bucket, prefix, profile=profile)
        relative = {key[len(prefix) :]: size for key, size in keys.items()}

        result_name = f"{JOBS_DIR}/{job['name']}/{RESULT_NAME}"
        result: dict[str, Any] = {}
        if result_name in relative:
            content = staging.read_object(self.s3, bucket, prefix + result_name, profile=profile)
            result = json.loads(content.decode("utf-8")) if content else {}
        runs = [run for run in result.get("runs", []) if run.get("run_id")]
        run_ids = [run["run_id"] for run in runs]
        if not result:  # the job died before it wrote a result: take every run it left
            run_ids = sorted(
                {name.split("/")[2] for name in relative if name.startswith("artifacts/runs/")}
            )

        root, files = Path(self.tracking.root), 0
        for name_, size in sorted(relative.items()):
            parts = name_.split("/")
            if ".." in parts or name_.startswith("/"):
                continue
            wanted = name_ == result_name or (
                len(parts) > 3
                and parts[0] == "artifacts"
                and parts[1] in ("runs", "adapters")
                and parts[2] in run_ids
            )
            if not wanted or (not checkpoints and "checkpoints" in parts):
                continue
            files += staging.download_object(
                self.s3, bucket, prefix + name_, root.joinpath(*parts), profile=profile, size=size
            )

        lines = self._merge_ledger(prefix, relative)
        billed = self._record_billable(job, description)
        job.update(
            pulled=True, pulled_at=timestamp(), runs=runs or [{"run_id": i} for i in run_ids]
        )
        job["result_status"] = result.get("status")
        self._save(job)
        return {
            "job": job,
            "pulled": True,
            "files": files,
            "ledger_lines": lines,
            "billable_recorded": billed,
            "result": result,
        }

    def pull_checkpoint(self, step: int, name: str | None = None) -> dict[str, Any]:
        """Download one checkpoint's adapter from a job's workspace (req_003 v3.04).

        Only the adapter's two files and the training run's record: enough for
        ``--adapter RUN_ID/checkpoint-N`` to work here. Reads only.
        """
        job = latest_job(self._record(), name)
        profile, bucket = self.names.profile, self.names.bucket
        prefix = f"{self.cfg.workspace_prefix}/{job['workspace']}/"
        keys = staging.list_keys(self.s3, bucket, prefix, profile=profile)
        relative = {key[len(prefix) :]: size for key, size in keys.items()}
        runs = sorted(
            parts[2]
            for parts in (item.split("/") for item in relative)
            if len(parts) == 6
            and parts[:2] == ["artifacts", "runs"]
            and parts[3:5] == [CHECKPOINTS, f"checkpoint-{step}"]
            and parts[5] == ADAPTER_FILES[0]
        )
        if not runs:
            raise ConfigError(
                f"the workspace of job {job['name']} has no checkpoint at step {step:,}"
            )
        root, files = Path(self.tracking.root), 0
        for run_id in runs:
            folder = f"artifacts/runs/{run_id}"
            wanted = [f"{folder}/{CHECKPOINTS}/checkpoint-{step}/{item}" for item in ADAPTER_FILES]
            for item in [*wanted, f"{folder}/{TRAINING_NAME}"]:
                if item in relative:
                    files += staging.download_object(
                        self.s3,
                        bucket,
                        prefix + item,
                        root.joinpath(*item.split("/")),
                        profile=profile,
                        size=relative[item],
                    )
        return {
            "job": job["name"],
            "files": files,
            "adapters": [f"{r}/checkpoint-{step}" for r in runs],
        }

    def keep(self, yes: bool = False) -> dict[str, Any]:
        """Copy the selected adapter to the bucket's ``adapters/`` folder. **Run by the owner.**

        A workspace is working storage; this is where a chosen adapter stays. The adapter
        must have been pulled, and must be the weights the selection record names.
        """
        if not yes:
            raise ConfigError(
                "keep writes the selected adapter to S3: it is run by the owner, with --yes"
            )
        training_cfg = load_training_config(root=self.tracking.root)
        selection = read_selection(training_cfg.selection_record)
        selected = selection["selected"]
        try:
            ref = resolve_adapter(
                selected["adapter"], training_cfg.adapters_dir, self.tracking.runs_dir
            )
        except ConfigError as error:
            raise ConfigError(
                f"the selected adapter {selected['adapter']} is not on this machine ({error}); "
                f"pull it first: pull --checkpoint {selected.get('step')}"
            ) from error
        sha256 = adapter_sha256(ref.path)
        if sha256 != selected.get("adapter_sha256"):
            raise ConfigError(
                f"the weights of {selected['adapter']} here are not the ones the selection "
                "record names; pull the checkpoint again"
            )
        profile, bucket = self.names.profile, self.names.bucket
        prefix = f"{self.names.adapters_prefix}{selected['adapter']}/"
        for item in ADAPTER_FILES:
            staging.put_checked(
                self.s3, bucket, prefix + item, (ref.path / item).read_bytes(), profile=profile
            )
        staging.put_checked(
            self.s3,
            bucket,
            prefix + training_cfg.selection_record.name,
            training_cfg.selection_record.read_bytes(),
            profile=profile,
            content_type="application/json",
        )
        kept = {
            "adapter": selected["adapter"],
            "adapter_sha256": sha256,
            "uri": f"s3://{bucket}/{prefix}",
            "kept_at": timestamp(),
        }
        record = self._record()
        record["kept"] = [k for k in record.get("kept", []) if k.get("adapter") != kept["adapter"]]
        record["kept"].append(kept)
        write_job_record(self.cfg.record_file, record)
        return kept

    def _merge_ledger(self, prefix: str, relative: dict[str, int]) -> dict[str, int]:
        """Append the lines a job wrote that are not here yet. Returns how many, per file."""
        profile, bucket = self.names.profile, self.names.bucket

        def remote(name: str) -> list[dict[str, Any]]:
            if name not in relative:
                return []
            content = staging.read_object(self.s3, bucket, prefix + name, profile=profile) or b""
            return [
                json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()
            ]

        tracking = self.tracking
        added = {"runs": 0, "ledger": 0, "results": 0}
        index = {
            (line.get("run_id"), line.get("event")) for line in read_lines(tracking.runs_index_file)
        }
        for line in remote(f"ledger/{tracking.runs_index_file.name}"):
            if (line.get("run_id"), line.get("event")) not in index:
                append_line(tracking.runs_index_file, line)
                added["runs"] += 1

        ledger = Ledger(tracking)
        known = {record.record_id for record in ledger.read()}
        for line in remote(f"ledger/{tracking.ledger_file.name}"):
            if line.get("record_id") not in known:
                ledger.append(LedgerRecord.from_dict(line))
                added["ledger"] += 1

        results_file = load_evaluation_config(root=tracking.root).results_file
        have = {line.get("run_id") for line in read_lines(results_file)}
        for line in remote(f"ledger/{results_file.name}"):
            if line.get("run_id") not in have:
                append_line(results_file, line)
                added["results"] += 1
        return added

    def _record_billable(self, job: dict[str, Any], description: dict[str, Any]) -> bool:
        """Add the job's billable seconds to the ledger, once. Quantities, no dollars."""
        ledger = Ledger(self.tracking)
        if any(
            record.external_id == job["name"] and record.source == "aws_api"
            for record in ledger.read()
        ):
            return False
        ledger.append(record_from_sagemaker_job(description, _LaunchRun(job), job["phase"]))
        return True


def _result(check_id: str, name: str, status: str, detail: str) -> dict[str, str]:
    return {"id": check_id, "name": name, "status": status, "detail": detail}


def _iso(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else (str(value) if value else None)


def _job_state(description: dict[str, Any]) -> dict[str, Any]:
    """What ``DescribeTrainingJob`` says about a job, as plain values for the job record."""
    reason = str(description.get("FailureReason") or "").strip().splitlines()
    return {
        "status": description.get("TrainingJobStatus"),
        "secondary_status": description.get("SecondaryStatus"),
        "failure_reason": reason[0][:300] if reason else None,
        "training_started_at": _iso(description.get("TrainingStartTime")),
        "training_ended_at": _iso(description.get("TrainingEndTime")),
        "training_seconds": description.get("TrainingTimeInSeconds"),
        "billable_seconds": description.get("BillableTimeInSeconds"),
        "checked_at": timestamp(),
    }


# --------------------------------------------------------------------------- rehearsing locally


def rehearse(
    task: str,
    cfg: SageMakerConfig,
    root: Path | str,
    git: Any,
    *,
    arguments: tuple[str, ...] = (),
    keep: bool = False,
    python: str = sys.executable,
) -> dict[str, Any]:
    """Run a task on this machine the way a job would: from an archive and the bundle.

    The code is unpacked into a temporary folder that has no snapshot, ``QLORA_DATA_BUNDLE``
    names the bundle, and ``scripts.job_run`` runs there. No AWS call is made. With
    uncommitted changes the working tree's files are used instead of the commit's, and the
    result says so. Its runs and ledger lines stay in the temporary folder: a rehearsal
    leaves nothing in the repository.
    """
    root = Path(root)
    if _is_test_arguments(arguments):
        raise ConfigError(
            "a rehearsal never reads the test split: a Qwen model answers test questions "
            "inside a SageMaker job only, once (req_005 v5.01). Rehearse on dev with --limit"
        )
    record = read_record(cfg.bundle_dir)  # a rehearsal needs the bundle, as a job does
    folder = Path(tempfile.mkdtemp(prefix="qlora-rehearsal-"))
    from_commit = git.commit != UNKNOWN_COMMIT and not git.dirty
    try:
        if from_commit:
            with tarfile.open(
                fileobj=io.BytesIO(build_source(root, git.commit)), mode="r:gz"
            ) as archive:
                archive.extractall(folder, filter="data")
        else:
            for name in working_tree_files(root):
                target = folder / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / name, target)
        job = f"rehearsal-{task}"
        environment = dict(os.environ)
        environment.update(
            QLORA_DATA_BUNDLE=str(cfg.bundle_dir.resolve()),
            QLORA_JOB_NAME=job,
            QLORA_GIT_COMMIT=git.commit,
        )
        completed = subprocess.run(
            [python, "-m", "scripts.job_run", task, "--where", "local", *arguments],
            cwd=folder,
            env=environment,
            check=False,
        )
        path = folder / JOBS_DIR / job / RESULT_NAME
        result = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        return {
            "exit_code": completed.returncode,
            "source": f"commit {git.commit[:12]}"
            if from_commit
            else "the working tree (uncommitted)",
            "bundle_sha256": record["sha256"],
            "snapshot_present": (folder / "artifacts" / "data").exists(),
            "folder": str(folder) if keep else None,
            "result": result,
        }
    finally:
        if not keep:
            shutil.rmtree(folder, ignore_errors=True)
