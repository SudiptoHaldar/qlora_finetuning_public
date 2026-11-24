"""Fine-tuning jobs on Amazon Bedrock: plan, launch, follow, deploy, tear down (req_004 v4.03).

A job is planned from the gate record of v4.02 (which files, what they cost) and
``config/frontier.yaml`` (which model, which hyperparameters). Before anything is created,
checks T1 to T9 must pass. A job is created once: its request token is derived from the
model, the files, the hyperparameters, and the attempt number, and a job that already
exists under the planned name with that token is adopted, not created again.

``launch``, ``deploy``, ``stop``, and ``teardown`` create, change, or delete AWS resources,
and ``launch`` is billed. **They are run by the owner** (``python -m scripts.nova_tune``).
``preflight`` and ``refresh`` only read. Nothing here creates Provisioned Throughput.

Usage::

    from qlora_finetuning.frontier.tuning import BedrockTuning

    tuning = BedrockTuning(cfg, pricing, context, bucket, role_arn, bedrock=client, s3=s3)
    plan = tuning.plan("nova-lite")
    checks = tuning.preflight(plan)           # T1 .. T9; reads only
    attempt = tuning.launch(plan)             # the owner: creates the job (billed)
    record = tuning.refresh()                 # reads the jobs; fetches the loss curves

| Id | Check | Passes when |
|---|---|---|
| T1 | Gate open | The gate record says passed and names both S3 objects |
| T2 | Gate current | Its prompt, split, epochs, cap, and price-list date are today's |
| T3 | Files in place | Both objects have the recorded size and SHA-256 |
| T4 | Model ready | The model is fine-tunable and Active |
| T5 | No Provisioned Throughput | The account has none |
| T6 | Not launched yet | No such attempt in the record, and no other job of that name |
| T7 | Within the cap | Path B so far plus this job, with the margin, is at most the cap |
| T8 | Committed | The working tree is clean |
| T9 | Earlier bill seen | Every earlier model in the launch order has completed and been billed |

Besides ``bedrock.py`` and ``staging.py`` this is the only module that imports botocore.
Tests pass in-memory stand-ins for both clients.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from qlora_finetuning.data import manifest_hash
from qlora_finetuning.frontier.bedrock import AUTH_ERROR_CODES, AUTH_EXCEPTIONS, sign_in_message
from qlora_finetuning.frontier.config import FrontierConfig, FrontierModel
from qlora_finetuning.frontier.export_validation import check
from qlora_finetuning.frontier.pricing import MILLION, Pricing, inference_cost
from qlora_finetuning.frontier.training_estimate import read_gate_record, usd
from qlora_finetuning.frontier.tuning_record import (
    ACTIVE,
    COMPLETED,
    CREATING,
    FAILED,
    IN_PROGRESS,
    STOPPING,
    Attempt,
    TuningRecord,
    as_text,
    read_tuning_record,
    write_tuning_record,
)
from qlora_finetuning.logging_setup import get_logger
from qlora_finetuning.settings import ConfigError
from qlora_finetuning.tracking import Ledger, LedgerRecord, read_git_info, start_run

PATH = "B"
PHASE = "train"
ROLES = ("training", "validation")
EXTRA_CONFIGS = ("frontier", "pricing", "aws")
# What Bedrock may answer when asked for a job or deployment by a name it does not know.
# ResourceNotFoundException is the declared one; a name can also be refused as invalid.
UNKNOWN_NAME = frozenset({"ResourceNotFoundException", "ValidationException"})
ALREADY_GONE = frozenset({"ResourceNotFoundException"})
JOB_PHASES = (
    ("validationDetails", "validation"),
    ("dataProcessingDetails", "data processing"),
    ("trainingDetails", "training"),
)
METRICS_FILES = {
    "training": ("step_wise_training_metrics.csv", "training_loss"),
    "validation": ("validation_metrics.csv", "validation_loss"),
}
NEEDS_SIGN_IN = "skipped: it needs a sign-in (aws sso login)"

logger = get_logger(__name__)


class TuningError(ConfigError):
    """Bedrock or S3 could not do what was asked, or a job is not in the state it needs."""


class Refused(TuningError):  # noqa: N818 (it reads as a verdict, not as an error class)
    """A command was refused before anything was changed. ``checks`` holds the reasons."""

    def __init__(self, message: str, checks: Sequence[dict[str, Any]] = ()) -> None:
        super().__init__(message)
        self.checks = list(checks)


# --------------------------------------------------------------------------- the plan


@dataclass(frozen=True)
class JobPlan:
    """Everything one launch would send, and what it is estimated to cost."""

    model: FrontierModel
    attempt: int
    job_name: str
    custom_model_name: str
    deployment_name: str
    token: str
    hyperparameters: dict[str, str]
    export_id: str
    training_s3_uri: str | None  # None while the files are not staged
    validation_s3_uri: str | None
    output_s3_uri: str
    role_arn: str
    tags: dict[str, str]
    estimated_training_tokens: int
    estimated_usd: Decimal
    estimated_with_margin_usd: Decimal

    def request(self) -> dict[str, Any]:
        """The arguments of ``create_model_customization_job``."""
        if not (self.training_s3_uri and self.validation_s3_uri):
            raise TuningError("the training files are not staged in S3")
        tags = [{"key": key, "value": value} for key, value in sorted(self.tags.items())]
        return {
            "jobName": self.job_name,
            "customModelName": self.custom_model_name,
            "roleArn": self.role_arn,
            "clientRequestToken": self.token,
            "baseModelIdentifier": self.model.customization_model_id,
            "customizationType": "FINE_TUNING",
            "trainingDataConfig": {"s3Uri": self.training_s3_uri},
            "validationDataConfig": {"validators": [{"s3Uri": self.validation_s3_uri}]},
            "outputDataConfig": {"s3Uri": self.output_s3_uri},
            "hyperParameters": dict(self.hyperparameters),
            "jobTags": tags,
            "customModelTags": tags,
        }


def request_token(
    model_name: str, export_id: str, hyperparameters: dict[str, str], attempt: int
) -> str:
    """The same launch always gives the same token, so Bedrock creates the job once."""
    identity = json.dumps([model_name, export_id, hyperparameters, attempt], sort_keys=True)
    return "qlora-" + hashlib.sha256(identity.encode()).hexdigest()[:48]


def build_plan(
    model_name: str,
    attempt: int,
    cfg: FrontierConfig,
    gate: dict[str, Any],
    pricing: Pricing,
    bucket: str,
    role_arn: str,
) -> JobPlan:
    """The launch of ``model_name`` on the files of the gate record. Calls nothing."""
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ConfigError(f"an attempt is a whole number of 1 or more, got {attempt!r}")
    model = cfg.model(model_name)
    try:
        export_id = str(gate["export_id"])
        one_epoch = int(gate["estimate"]["training_tokens_one_epoch"])
    except (KeyError, TypeError, ValueError):
        raise ConfigError("the gate record is damaged; run python -m scripts.nova_export") from None
    staged = gate.get("staged") or {}
    name = f"{cfg.name_prefix}-{model.name}-{export_id}-a{attempt}"
    tokens = one_epoch * cfg.epoch_count
    dollars = tokens * pricing.model(model.model_id).training_per_million / MILLION
    hyperparameters = cfg.hyperparameters
    return JobPlan(
        model=model,
        attempt=attempt,
        job_name=name,
        custom_model_name=name,
        deployment_name=name,
        token=request_token(model.name, export_id, hyperparameters, attempt),
        hyperparameters=hyperparameters,
        export_id=export_id,
        training_s3_uri=(staged.get("training") or {}).get("s3_uri"),
        validation_s3_uri=(staged.get("validation") or {}).get("s3_uri"),
        output_s3_uri=f"s3://{bucket}/{cfg.output_prefix}{export_id}/{model.name}-a{attempt}/",
        role_arn=role_arn,
        tags=dict(cfg.tags),
        estimated_training_tokens=tokens,
        estimated_usd=dollars,
        estimated_with_margin_usd=dollars * (1 + cfg.margin_percent / 100),
    )


def committed_path_b(ledger_records: Sequence[Any], pricing: Pricing) -> Decimal:
    """What Path B has cost so far, from the ledger's effective records.

    A record with a billed amount counts as that amount. Any other counts its inference and
    training tokens at list price. A record whose model has no price counts as nothing.
    """
    total = Decimal(0)
    for record in ledger_records:
        if record.path != PATH:
            continue
        if record.billed_amount_usd is not None:
            total += Decimal(str(record.billed_amount_usd))
            continue
        if record.model_id not in pricing.models:
            continue
        if record.input_tokens is not None or record.output_tokens is not None:
            total += inference_cost(
                record.model_id, record.input_tokens or 0, record.output_tokens or 0, pricing
            )
        if record.training_tokens:
            price = pricing.model(record.model_id).training_per_million
            total += record.training_tokens * price / MILLION
    return total


def summarize_metrics(content: bytes, column: str) -> dict[str, Any]:
    """First, last, and lowest loss of a metrics file from a job's output."""
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
    try:
        losses = [float(row[column]) for row in rows]
        steps = [int(float(row["step_number"])) for row in rows]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"no usable {column} column") from None
    if not losses:
        raise ValueError("the file has no rows")
    lowest = min(range(len(losses)), key=losses.__getitem__)
    return {
        "steps": len(losses),
        "first": losses[0],
        "last": losses[-1],
        "lowest": losses[lowest],
        "lowest_step": steps[lowest],
    }


def _skipped(check_id: str, name: str) -> dict[str, Any]:
    return {"id": check_id, "name": name, "passed": None, "detail": NEEDS_SIGN_IN}


def _split_uri(uri: str) -> tuple[str, str]:
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key


def _s3_checksum(sha256_hex: str) -> str:
    return base64.b64encode(bytes.fromhex(sha256_hex)).decode("ascii")


# --------------------------------------------------------------------------- job control


class BedrockTuning:
    """Job control over a Bedrock control-plane client and an S3 client, both passed in.

    ``context`` is a resolved ``RunnerContext``: it carries the prompt, the split, and the
    tracking settings. Without clients only ``plan`` and the local checks work.
    """

    def __init__(
        self,
        cfg: FrontierConfig,
        pricing: Pricing,
        context: Any,
        bucket: str,
        role_arn: str,
        bedrock: Any = None,
        s3: Any = None,
        profile: str = "qlora",
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self.pricing = pricing
        self.context = context
        self.bucket = bucket
        self.role_arn = role_arn
        self.bedrock = bedrock
        self.s3 = s3
        self.profile = profile
        self._sleep = sleep

    # ------------------------------------------------------------------ small helpers

    def gate(self) -> dict[str, Any] | None:
        try:
            return read_gate_record(self.cfg)
        except ConfigError:
            return None

    def record(self) -> TuningRecord:
        return read_tuning_record(self.cfg)

    def ledger(self) -> list[Any]:
        return Ledger(self.context.tracking_cfg).effective()

    def plan(self, model_name: str, attempt: int = 1) -> JobPlan:
        gate = self.gate()
        if gate is None:
            raise ConfigError("no gate record; run python -m scripts.nova_export")
        return build_plan(
            model_name, attempt, self.cfg, gate, self.pricing, self.bucket, self.role_arn
        )

    def _need(self, *clients: str) -> None:
        if any(getattr(self, name) is None for name in clients):
            raise ConfigError(sign_in_message(self.profile))

    def _aws(
        self, operation: Any, what: str, missing: frozenset[str] = frozenset(), **arguments: Any
    ) -> dict[str, Any] | None:
        """Call AWS. ``None`` means "not there", for the codes in ``missing``."""
        try:
            return operation(**arguments)
        except AUTH_EXCEPTIONS:
            raise ConfigError(sign_in_message(self.profile)) from None
        except ClientError as error:
            details = error.response.get("Error", {})
            code = str(details.get("Code", "ClientError"))
            if code in AUTH_ERROR_CODES:
                raise ConfigError(sign_in_message(self.profile)) from None
            if code in missing:
                return None
            logger.error("AWS %s when asked to %s: %s", code, what, details.get("Message", ""))
            raise TuningError(f"AWS answered {code} when asked to {what}") from None
        except BotoCoreError as error:
            raise TuningError(
                f"AWS could not be reached to {what}: {type(error).__name__}"
            ) from None

    def _job_by_name(self, name: str) -> dict[str, Any] | None:
        return self._aws(
            self.bedrock.get_model_customization_job,
            f"look for the job {name}",
            missing=UNKNOWN_NAME,
            jobIdentifier=name,
        )

    def _billed(self, attempt: Attempt) -> Decimal | None:
        """The amount entered from the bill for an attempt's job, if any."""
        for record in self.ledger():
            if (
                record.external_id == attempt.job_arn
                and record.source == "bill"
                and record.billed_amount_usd is not None
            ):
                return Decimal(str(record.billed_amount_usd))
        return None

    # ------------------------------------------------------------------ the checks

    def preflight(self, plan: JobPlan) -> list[dict[str, Any]]:
        """Checks T1 to T9 for ``plan``. Reads only. A check that needs a client that was
        not given has ``passed: None`` and counts as not passed."""
        gate = self.gate() or {}
        return [
            self._gate_open(gate),
            self._gate_current(gate),
            self._files_in_place(gate),
            self._model_ready(plan),
            self._no_provisioned_throughput(),
            self._not_launched(plan),
            self._within_cap(plan),
            self._committed(),
            self._earlier_bill_seen(plan),
        ]

    def _gate_open(self, gate: dict[str, Any]) -> dict[str, Any]:
        problems = []
        if not gate:
            problems.append("there is no gate record; run python -m scripts.nova_export")
        elif gate.get("passed") is not True:
            failed = [c["id"] for c in gate.get("checks", []) if not c.get("passed")]
            problems.append(
                f"the gate is closed ({', '.join(failed) or 'a check'} failed); see "
                "python -m scripts.nova_export"
            )
        staged = gate.get("staged") or {}
        if gate and not all((staged.get(role) or {}).get("s3_uri") for role in ROLES):
            problems.append(
                "the training files are not staged; run python -m scripts.nova_export --upload"
            )
        return check("T1", "Gate open", problems, "the gate record says passed and is staged")

    def _gate_current(self, gate: dict[str, Any]) -> dict[str, Any]:
        inputs = gate.get("inputs") or {}
        expected = {
            "prompt_id": ("the prompt wording", self.context.prompt.prompt_id),
            "split_content_hash": ("the split manifest", manifest_hash(self.context.split_cfg)),
            "epoch_count": ("the epoch count", self.cfg.epoch_count),
            "cap_usd": ("the cap", format(self.cfg.path_b_cap_usd.normalize(), "f")),
        }
        problems = [
            f"{what} changed since the gate record was written"
            for key, (what, value) in expected.items()
            if inputs.get(key) != value
        ]
        if (inputs.get("price_list") or {}).get("published") != self.pricing.published:
            problems.append("the price list changed since the gate record was written")
        if problems:
            problems.append("run python -m scripts.nova_export")
        return check(
            "T2",
            "Gate current",
            problems if gate else ["there is no gate record"],
            f"prompt, split, {self.cfg.epoch_count} epochs, cap, and prices are as the gate "
            "saw them",
        )

    def _files_in_place(self, gate: dict[str, Any]) -> dict[str, Any]:
        name = "Files in place"
        staged, files = gate.get("staged") or {}, gate.get("files") or {}
        if not all((staged.get(role) or {}).get("s3_uri") for role in ROLES):
            return check("T3", name, ["the training files are not staged"], "")
        if self.s3 is None:
            return _skipped("T3", name)
        problems = []
        for role in ROLES:
            uri = staged[role]["s3_uri"]
            bucket, key = _split_uri(uri)
            head = self._aws(
                self.s3.head_object,
                f"look at {uri}",
                missing=frozenset({"404", "NoSuchKey", "NotFound"}),
                Bucket=bucket,
                Key=key,
                ChecksumMode="ENABLED",
            )
            wanted = files.get(role) or {}
            if head is None:
                problems.append(f"{uri} is not there")
            elif head.get("ContentLength") != wanted.get("bytes") or head.get(
                "ChecksumSHA256"
            ) != _s3_checksum(str(wanted.get("sha256", ""))):
                problems.append(f"{uri} is not the file the gate record names")
        return check("T3", name, problems, "both objects have the recorded size and SHA-256")

    def _model_ready(self, plan: JobPlan) -> dict[str, Any]:
        name = "Model ready"
        if self.bedrock is None:
            return _skipped("T4", name)
        model = plan.model
        listed = self._aws(
            self.bedrock.list_foundation_models,
            "list the fine-tunable models",
            byCustomizationType="FINE_TUNING",
        )
        ids = {item.get("modelId") for item in (listed or {}).get("modelSummaries", [])}
        details = self._aws(
            self.bedrock.get_foundation_model,
            f"read the lifecycle of {model.model_id}",
            modelIdentifier=model.model_id,
        )
        status = ((details or {}).get("modelDetails", {}).get("modelLifecycle") or {}).get(
            "status"
        )
        problems = []
        if model.customization_model_id not in ids:
            problems.append(f"{model.customization_model_id} is not listed as fine-tunable")
        if status != "ACTIVE":
            problems.append(f"{model.model_id} is {status}; a Legacy model cannot be fine-tuned")
        return check(
            "T4", name, problems, f"{model.customization_model_id} is fine-tunable and Active"
        )

    def _provisioned(self) -> list[str]:
        listed = self._aws(
            self.bedrock.list_provisioned_model_throughputs, "list Provisioned Throughput"
        )
        return [
            str(item.get("provisionedModelName") or item.get("provisionedModelArn"))
            for item in (listed or {}).get("provisionedModelSummaries", [])
        ]

    def _no_provisioned_throughput(self) -> dict[str, Any]:
        name = "No Provisioned Throughput"
        if self.bedrock is None:
            return _skipped("T5", name)
        found = self._provisioned()
        problems = (
            [
                f"the account has Provisioned Throughput ({', '.join(found)}), which bills by "
                "the hour; delete it first"
            ]
            if found
            else []
        )
        return check("T5", name, problems, "the account has none")

    def _not_launched(self, plan: JobPlan) -> dict[str, Any]:
        name = "Not launched yet"
        known = self.record().find(plan.model.name, plan.attempt)
        if known is not None and known.job_arn:
            return check(
                "T6",
                name,
                [
                    f"attempt {plan.attempt} of {plan.model.name} was already launched (job "
                    f"status {known.job_status}); another launch needs --attempt "
                    f"{plan.attempt + 1}"
                ],
                "",
            )
        if self.bedrock is None:
            return _skipped("T6", name)
        existing = self._job_by_name(plan.job_name)
        if existing is None:
            return check("T6", name, [], f"Bedrock has no job named {plan.job_name}")
        if existing.get("clientRequestToken") == plan.token:
            return check(
                "T6",
                name,
                [],
                f"Bedrock already has {plan.job_name} from this very request; a launch "
                "adopts it and creates nothing",
            )
        return check(
            "T6",
            name,
            [f"Bedrock has a job named {plan.job_name} that this record does not know"],
            "",
        )

    def _within_cap(self, plan: JobPlan) -> dict[str, Any]:
        committed = committed_path_b(self.ledger(), self.pricing)
        total, cap = committed + plan.estimated_with_margin_usd, self.cfg.path_b_cap_usd
        problems = (
            []
            if total <= cap
            else [
                f"Path B so far is {usd(committed)}; with this job "
                f"({usd(plan.estimated_with_margin_usd)} with the margin) it would be "
                f"{usd(total)}, above the cap of {usd(cap)}"
            ]
        )
        return check(
            "T7",
            "Within the cap",
            problems,
            f"Path B so far is {usd(committed)}; with this job "
            f"({usd(plan.estimated_with_margin_usd)} with the margin) it is {usd(total)} of "
            f"{usd(cap)}",
        )

    def _committed(self) -> dict[str, Any]:
        tracking = self.context.tracking_cfg
        git = self.context.git or read_git_info(
            tracking.root, tracking.git_commit_env, (tracking.ledger_prefix,)
        )
        problems = []
        if git.dirty is None:
            problems.append("the commit of this code is not known")
        elif git.dirty:
            shown = ", ".join(git.changed[:3]) + (", ..." if len(git.changed) > 3 else "")
            problems.append(f"there are uncommitted changes ({shown}); commit them first")
        return check("T8", "Committed", problems, f"the tree is clean at {git.commit[:12]}")

    def _earlier_bill_seen(self, plan: JobPlan) -> dict[str, Any]:
        name = "Earlier bill seen"
        order = list(self.cfg.launch_order)
        earlier = order[: order.index(plan.model.name)]
        if not earlier:
            return check("T9", name, [], f"{plan.model.name} is first in the launch order")
        record, problems, seen = self.record(), [], []
        for model_name in earlier:
            attempt = record.latest(model_name)
            if attempt is None or not attempt.job_arn:
                problems.append(f"{model_name} has not been launched")
            elif attempt.job_status != COMPLETED:
                problems.append(f"the job of {model_name} is {attempt.job_status}, not Completed")
            else:
                billed = self._billed(attempt)
                if billed is None:
                    problems.append(
                        f"the amount AWS billed for {model_name} has not been entered; read "
                        f"it from the bill, then: nova_tune bill --model {model_name} --usd N"
                    )
                else:
                    estimate = Decimal(attempt.estimated_usd)
                    proportion = billed / estimate if estimate else Decimal(1)
                    seen.append(
                        f"{model_name} was billed {usd(billed)} against an estimate of "
                        f"{usd(estimate)} ({proportion:.2f} times); at that proportion this "
                        f"job is about {usd(plan.estimated_usd * proportion)}"
                    )
        return check("T9", name, problems, "; ".join(seen))

    # ------------------------------------------------------------------ launch

    def launch(self, plan: JobPlan) -> Attempt:
        """Create the job of ``plan`` (billed), or adopt it if it was already created.

        Refuses unless T1 to T9 all pass. The tuning record is written as soon as the job's
        ARN is known; then the estimated training tokens go into the ledger.
        """
        self._need("bedrock", "s3")
        checks = self.preflight(plan)
        if not all(item["passed"] is True for item in checks):
            raise Refused("a check failed, so nothing was launched", checks)
        gate = self.gate() or {}
        existing = self._job_by_name(plan.job_name)
        configs = tuple(dict.fromkeys((*self.context.configs, *EXTRA_CONFIGS)))
        with start_run(
            "train",
            plan.job_name,
            path=PATH,
            arm=plan.model.tuned_arm,
            where="bedrock",
            scope="measured",
            configs=configs,
            split_manifest_hash=(gate.get("inputs") or {}).get("split_content_hash"),
            description=f"Bedrock fine-tuning job for {plan.model.name}, attempt {plan.attempt}",
            cfg=self.context.tracking_cfg,
            config_dir=self.context.config_dir,
            git=self.context.git,
            environment=self.context.environment,
        ) as run:
            if existing is None:
                created = self._aws(
                    self.bedrock.create_model_customization_job,
                    f"create the job {plan.job_name}",
                    **plan.request(),
                )
                job_arn, status, created_at = created["jobArn"], IN_PROGRESS, None
            else:
                job_arn = existing["jobArn"]
                status = existing.get("status") or IN_PROGRESS
                created_at = as_text(existing.get("creationTime"))
            attempt = Attempt(
                model=plan.model.name,
                attempt=plan.attempt,
                arm=plan.model.tuned_arm,
                base_model_id=plan.model.model_id,
                customization_model_id=plan.model.customization_model_id,
                job_name=plan.job_name,
                custom_model_name=plan.custom_model_name,
                deployment_name=plan.deployment_name,
                client_request_token=plan.token,
                hyperparameters=dict(plan.hyperparameters),
                export_id=plan.export_id,
                training_s3_uri=str(plan.training_s3_uri),
                validation_s3_uri=str(plan.validation_s3_uri),
                output_s3_uri=plan.output_s3_uri,
                estimated_training_tokens=plan.estimated_training_tokens,
                estimated_usd=format(plan.estimated_usd.normalize(), "f"),
                run_id=run.run_id,
                job_arn=job_arn,
                job_status=status,
                created_at=created_at,
            )
            record = self.record()
            record.put(attempt)
            write_tuning_record(record, self.cfg)  # first: the only memory of the job
            run.record(
                phase=PHASE,
                model_id=plan.model.model_id,
                training_tokens=plan.estimated_training_tokens,
                source="computed",
                source_detail=(
                    "estimated: Qwen tokens of one pass over fit times the measured Nova to "
                    "Qwen ratio (gate record) times the epochs; Bedrock reports no count"
                ),
                external_id=job_arn,
            )
            run.outputs.update(job_arn=job_arn, job_name=plan.job_name)
        return attempt

    # ------------------------------------------------------------------ status

    def _refresh_job(self, attempt: Attempt) -> None:
        job = self._aws(
            self.bedrock.get_model_customization_job,
            f"read the job {attempt.job_name}",
            jobIdentifier=attempt.job_arn,
        ) or {}
        attempt.job_status = job.get("status") or attempt.job_status
        attempt.created_at = as_text(job.get("creationTime")) or attempt.created_at
        attempt.ended_at = as_text(job.get("endTime")) or attempt.ended_at
        attempt.failure_message = job.get("failureMessage") or attempt.failure_message
        details = job.get("statusDetails") or {}
        attempt.failed_in = next(
            (name for key, name in JOB_PHASES if (details.get(key) or {}).get("status") == FAILED),
            attempt.failed_in,
        )
        if attempt.job_status == COMPLETED:
            attempt.model_arn = job.get("outputModelArn") or attempt.model_arn
            training = (job.get("trainingMetrics") or {}).get("trainingLoss")
            validation = [
                item.get("validationLoss") for item in job.get("validationMetrics") or []
            ]
            attempt.training_loss = training if training is not None else attempt.training_loss
            if validation and validation[0] is not None:
                attempt.validation_loss = validation[0]

    @staticmethod
    def _settle_losses(attempt: Attempt) -> None:
        """Replace a loss Bedrock did not report with the last value of its loss curve.

        For a Nova job Bedrock answers -1 for both losses (seen 2026-11-18). A loss is never
        negative, so a negative figure means "not reported".
        """
        for field, kind in (("training_loss", "training"), ("validation_loss", "validation")):
            value = getattr(attempt, field)
            if value is not None and value < 0:
                value = None
            if value is None and attempt.metrics and kind in attempt.metrics:
                value = attempt.metrics[kind]["last"]
            setattr(attempt, field, value)

    def _fetch_metrics(self, attempt: Attempt) -> dict[str, Any]:
        bucket, prefix = _split_uri(attempt.output_s3_uri)
        listed = self._aws(
            self.s3.list_objects_v2,
            f"list {attempt.output_s3_uri}",
            Bucket=bucket,
            Prefix=prefix,
        )
        keys = [item["Key"] for item in (listed or {}).get("Contents", [])]
        folder = self.cfg.export_dir / attempt.export_id / f"{attempt.model}-a{attempt.attempt}"
        metrics = {}
        for kind, (file_name, column) in METRICS_FILES.items():
            key = next((key for key in keys if key.endswith("/" + file_name)), None)
            if key is None:
                raise ValueError(f"{file_name} is not under the job's output location")
            body = self._aws(
                self.s3.get_object, f"download {file_name}", Bucket=bucket, Key=key
            )["Body"].read()
            folder.mkdir(parents=True, exist_ok=True)
            (folder / file_name).write_bytes(body)
            metrics[kind] = summarize_metrics(body, column)
        return metrics

    def refresh(self) -> TuningRecord:
        """Read every job and deployment of the record from Bedrock and write the record.

        Changes nothing in AWS. For a completed job whose loss curves are not yet in the
        record, the two metrics files are downloaded to the git-ignored export folder and
        summarized; a problem there is noted and does not lose the job's status.
        """
        self._need("bedrock")
        record = self.record()
        for attempt in record.attempts:
            if not attempt.job_arn:
                continue
            if not attempt.ended or (attempt.job_status == COMPLETED and not attempt.model_arn):
                self._refresh_job(attempt)
            if (
                attempt.job_status == COMPLETED
                and attempt.metrics is None
                and self.s3 is not None
            ):
                try:
                    attempt.metrics = self._fetch_metrics(attempt)
                except (TuningError, ValueError, UnicodeDecodeError) as error:
                    note = f"the loss curves could not be read: {error}"
                    if note not in attempt.notes:
                        attempt.notes.append(note)
            if attempt.job_status == COMPLETED:
                self._settle_losses(attempt)
            if attempt.deployment_arn and not attempt.deployment_deleted:
                deployment = self._aws(
                    self.bedrock.get_custom_model_deployment,
                    f"read the deployment {attempt.deployment_name}",
                    customModelDeploymentIdentifier=attempt.deployment_arn,
                ) or {}
                attempt.deployment_status = deployment.get("status") or attempt.deployment_status
                attempt.deployment_failure = deployment.get("failureMessage")
        if record.attempts:
            write_tuning_record(record, self.cfg)
        return record

    @staticmethod
    def pending(record: TuningRecord) -> list[Attempt]:
        """Attempts whose job has not ended or whose deployment is still being created."""
        return [
            attempt
            for attempt in record.attempts
            if attempt.job_arn
            and (
                not attempt.ended
                or (attempt.deployment_status == CREATING and not attempt.deployment_deleted)
            )
        ]

    def wait(self, on_look: Callable[[TuningRecord], None] | None = None) -> TuningRecord:
        """``refresh`` at the configured interval until nothing is pending."""
        while True:
            record = self.refresh()
            if on_look is not None:
                on_look(record)
            if not self.pending(record):
                return record
            self._sleep(self.cfg.poll_seconds)

    # ------------------------------------------------------------------ deploy, stop

    def _latest(self, model_name: str, attempt: int | None = None) -> tuple[TuningRecord, Attempt]:
        self.cfg.model(model_name)  # an unknown name is a config error
        record = self.record()
        found = record.latest(model_name) if attempt is None else record.find(model_name, attempt)
        if found is None or not found.job_arn:
            raise Refused(f"{model_name} has no launched job in the tuning record")
        return record, found

    def deploy(self, model_name: str) -> Attempt:
        """Create the on-demand deployment of a completed model, or adopt an existing one."""
        self._need("bedrock")
        record, attempt = self._latest(model_name)
        if attempt.job_status != COMPLETED or not attempt.model_arn or attempt.model_deleted:
            raise Refused(
                f"{model_name} has no completed, existing tuned model (job status "
                f"{attempt.job_status}); run nova_tune status"
            )
        if attempt.deployment_arn and not attempt.deployment_deleted:
            raise Refused(f"{model_name} is already deployed ({attempt.deployment_status})")
        provisioned = self._provisioned()
        if provisioned:
            raise Refused(
                f"the account has Provisioned Throughput ({', '.join(provisioned)}); "
                "delete it first"
            )
        existing = self._aws(
            self.bedrock.get_custom_model_deployment,
            f"look for the deployment {attempt.deployment_name}",
            missing=UNKNOWN_NAME,
            customModelDeploymentIdentifier=attempt.deployment_name,
        )
        if existing is None:
            created = self._aws(
                self.bedrock.create_custom_model_deployment,
                f"deploy {attempt.custom_model_name}",
                modelDeploymentName=attempt.deployment_name,
                modelArn=attempt.model_arn,
                description=f"On-demand deployment of {attempt.custom_model_name}",
                clientRequestToken=f"{attempt.client_request_token}-deploy",
                tags=[{"key": k, "value": v} for k, v in sorted(self.cfg.tags.items())],
            )
            attempt.deployment_arn = created["customModelDeploymentArn"]
            attempt.deployment_status = CREATING
        else:
            attempt.deployment_arn = existing["customModelDeploymentArn"]
            attempt.deployment_status = existing.get("status") or CREATING
        attempt.deployment_deleted, attempt.deployment_failure = False, None
        write_tuning_record(record, self.cfg)
        return attempt

    def stop(self, model_name: str) -> Attempt:
        """Stop a running job. Its tokens so far are billed and no model is made."""
        self._need("bedrock")
        record, attempt = self._latest(model_name)
        self._refresh_job(attempt)
        if attempt.job_status != IN_PROGRESS:
            write_tuning_record(record, self.cfg)
            raise Refused(f"the job of {model_name} is {attempt.job_status}, not running")
        self._aws(
            self.bedrock.stop_model_customization_job,
            f"stop the job {attempt.job_name}",
            jobIdentifier=attempt.job_arn,
        )
        attempt.job_status = STOPPING
        write_tuning_record(record, self.cfg)
        return attempt

    # ------------------------------------------------------------------ bill, teardown

    def bill(self, model_name: str, amount: str, attempt: int | None = None) -> Attempt:
        """Enter the amount AWS billed for a job. It replaces the estimate in the ledger."""
        try:
            dollars = Decimal(str(amount))
        except InvalidOperation:
            raise ConfigError(f"the billed amount is not a number: {amount!r}") from None
        if not dollars.is_finite() or dollars < 0:
            raise ConfigError(f"the billed amount must be zero or more, got {amount!r}")
        record, found = self._latest(model_name, attempt)
        replaced = next(
            (
                item
                for item in self.ledger()
                if item.external_id == found.job_arn and item.phase == PHASE
            ),
            None,
        )
        if replaced is None:
            raise TuningError(f"the ledger has no record for the job of {model_name}")
        text = format(dollars.normalize(), "f")
        Ledger(self.context.tracking_cfg).append(
            LedgerRecord(
                run_id=replaced.run_id,
                phase=PHASE,
                path=PATH,
                arm=found.arm,
                where="bedrock",
                source="bill",
                source_detail="read from the AWS bill by the owner",
                model_id=found.base_model_id,
                billed_amount_usd=text,
                external_id=found.job_arn,
                replaces=replaced.record_id,
            )
        )
        found.billed_usd = text
        write_tuning_record(record, self.cfg)
        return found

    def teardown(self, model_name: str, everything: bool, attempt: int | None = None) -> Attempt:
        """Delete an attempt's deployment and, with ``everything``, its tuned model too.

        Only what the tuning record names is touched. S3 objects are never deleted here.
        """
        self._need("bedrock")
        record, found = self._latest(model_name, attempt)
        has_deployment = bool(found.deployment_arn) and not found.deployment_deleted
        has_model = bool(found.model_arn) and not found.model_deleted
        if not has_deployment and not (everything and has_model):
            raise Refused(f"{model_name} has nothing left to delete of what was asked")
        if has_deployment:
            self._aws(
                self.bedrock.delete_custom_model_deployment,
                f"delete the deployment {found.deployment_name}",
                missing=ALREADY_GONE,
                customModelDeploymentIdentifier=found.deployment_arn,
            )
            found.deployment_deleted, found.deployment_status = True, "Deleted"
            write_tuning_record(record, self.cfg)
        if everything and has_model:
            self._aws(
                self.bedrock.delete_custom_model,
                f"delete the tuned model {found.custom_model_name}",
                missing=ALREADY_GONE,
                modelIdentifier=found.model_arn,
            )
            found.model_deleted = True
            write_tuning_record(record, self.cfg)
        return found


def is_active(attempt: Attempt) -> bool:
    """True when the attempt's tuned model can be asked questions."""
    return attempt.deployment_status == ACTIVE and not attempt.deployment_deleted
