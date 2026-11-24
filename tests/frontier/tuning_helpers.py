"""Shared pieces for the v4.03 tests: an in-memory Bedrock, and a staged, open gate.

Nothing here reaches AWS. ``FakeBedrock`` keeps jobs, tuned models, and deployments in
memory and answers the calls ``BedrockTuning`` makes; a test moves a job on with
``finish``. The gate record is the real one of v4.02, made on the invented split and staged
into ``FakeS3``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from botocore.exceptions import ClientError

from qlora_finetuning.evaluation.runner import RunnerContext
from qlora_finetuning.frontier.export import export_files
from qlora_finetuning.frontier.export_validation import validate_export
from qlora_finetuning.frontier.pricing import load_pricing
from qlora_finetuning.frontier.staging import stage_export
from qlora_finetuning.frontier.training_estimate import (
    build_gate_record,
    estimate_path_b,
    gate_checks,
    load_token_ratio,
    write_gate_record,
)
from qlora_finetuning.frontier.tuning import BedrockTuning
from tests.evaluation.conftest import ledger_records
from tests.frontier.export_helpers import FakeS3, volumes_for, write_baseline_report

BUCKET = "qlora-sagemaker-123456789012"
ROLE = "arn:aws:iam::123456789012:role/qlora-bedrock-customization"
ACCOUNT = "arn:aws:bedrock:us-east-1:123456789012"
CREATED = datetime(2026, 11, 17, 4, 5, 6, tzinfo=UTC)
ENDED = datetime(2026, 11, 17, 7, 30, 0, tzinfo=UTC)
TRAINING_CSV = (
    b"step_number,epoch_number,training_loss,perplexity\n"
    b"0,0,2.5,12.2\n1,0,1.2,3.3\n2,1,0.4,1.5\n3,1,0.6,1.8\n"
)
VALIDATION_CSV = b"step_number,epoch_number,validation_loss,perplexity\n1,0,1.1,3.0\n3,1,0.7,2.0\n"


def client_error(code: str, operation: str = "Op") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "stubbed"}}, operation)


class FakeBedrock:
    """The Bedrock control plane, as far as job control uses it."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}  # by job name
        self.deployments: dict[str, dict] = {}  # by deployment name
        self.deleted_models: list[str] = []
        self.provisioned: list[dict] = []
        self.fine_tunable = ["amazon.nova-pro-v1:0:300k", "amazon.nova-lite-v1:0:300k"]
        self.lifecycle = "ACTIVE"
        self.calls: list[str] = []
        self.fail_create: ClientError | None = None
        self.lose_reply = False  # the job is created, but the caller gets an error

    # ---- reads
    def list_foundation_models(self, byCustomizationType=None):  # noqa: N803 (boto3's names)
        self.calls.append("list_foundation_models")
        assert byCustomizationType == "FINE_TUNING"
        return {"modelSummaries": [{"modelId": model_id} for model_id in self.fine_tunable]}

    def get_foundation_model(self, modelIdentifier):  # noqa: N803
        self.calls.append("get_foundation_model")
        return {
            "modelDetails": {
                "modelId": modelIdentifier,
                "modelLifecycle": {"status": self.lifecycle},
            }
        }

    def list_provisioned_model_throughputs(self):
        self.calls.append("list_provisioned_model_throughputs")
        return {"provisionedModelSummaries": list(self.provisioned)}

    def _job(self, identifier: str) -> dict:
        for job in self.jobs.values():
            if identifier in (job["jobName"], job["jobArn"]):
                return job
        raise client_error("ResourceNotFoundException", "GetModelCustomizationJob")

    def get_model_customization_job(self, jobIdentifier):  # noqa: N803
        self.calls.append("get_model_customization_job")
        return dict(self._job(jobIdentifier))

    def _deployment(self, identifier: str) -> dict:
        for deployment in self.deployments.values():
            names = (deployment["modelDeploymentName"], deployment["customModelDeploymentArn"])
            if identifier in names:
                return deployment
        raise client_error("ResourceNotFoundException", "GetCustomModelDeployment")

    def get_custom_model_deployment(self, customModelDeploymentIdentifier):  # noqa: N803
        self.calls.append("get_custom_model_deployment")
        return dict(self._deployment(customModelDeploymentIdentifier))

    # ---- changes
    def create_model_customization_job(self, **request):
        self.calls.append("create_model_customization_job")
        if self.fail_create is not None:
            raise self.fail_create
        name = request["jobName"]
        existing = self.jobs.get(name)
        if existing is not None:
            if existing["clientRequestToken"] != request["clientRequestToken"]:
                raise client_error("ConflictException", "CreateModelCustomizationJob")
            return {"jobArn": existing["jobArn"]}
        base = request["baseModelIdentifier"]
        self.jobs[name] = {
            "jobArn": f"{ACCOUNT}:model-customization-job/{base}/{len(self.jobs) + 1:012d}",
            "jobName": name,
            "outputModelName": request["customModelName"],
            "clientRequestToken": request["clientRequestToken"],
            "status": "InProgress",
            "creationTime": CREATED,
            "statusDetails": {
                "validationDetails": {"status": "InProgress"},
                "dataProcessingDetails": {"status": "NotStarted"},
                "trainingDetails": {"status": "NotStarted"},
            },
            "request": request,
        }
        if self.lose_reply:
            self.lose_reply = False
            raise client_error("InternalServerException", "CreateModelCustomizationJob")
        return {"jobArn": self.jobs[name]["jobArn"]}

    def stop_model_customization_job(self, jobIdentifier):  # noqa: N803
        self.calls.append("stop_model_customization_job")
        self._job(jobIdentifier)["status"] = "Stopping"
        return {}

    def create_custom_model_deployment(self, **request):
        self.calls.append("create_custom_model_deployment")
        name = request["modelDeploymentName"]
        self.deployments[name] = {
            "customModelDeploymentArn": f"{ACCOUNT}:custom-model-deployment/{name}",
            "modelDeploymentName": name,
            "modelArn": request["modelArn"],
            "status": "Creating",
            "request": request,
        }
        return {"customModelDeploymentArn": self.deployments[name]["customModelDeploymentArn"]}

    def delete_custom_model_deployment(self, customModelDeploymentIdentifier):  # noqa: N803
        self.calls.append("delete_custom_model_deployment")
        deployment = self._deployment(customModelDeploymentIdentifier)
        del self.deployments[deployment["modelDeploymentName"]]
        return {}

    def delete_custom_model(self, modelIdentifier):  # noqa: N803
        self.calls.append("delete_custom_model")
        if modelIdentifier in self.deleted_models:
            raise client_error("ResourceNotFoundException", "DeleteCustomModel")
        self.deleted_models.append(modelIdentifier)
        return {}

    # ---- what a test does to move things on
    def finish(self, name: str, status: str = "Completed", failed_in: str | None = None, **more):
        """End a job: completed with a tuned model and losses, or failed in a phase."""
        job = self.jobs[name]
        job["status"] = status
        job["endTime"] = ENDED
        phases = ["validationDetails", "dataProcessingDetails", "trainingDetails"]
        for phase in phases:
            job["statusDetails"][phase] = {"status": "Completed"}
        if status == "Completed":
            job["outputModelArn"] = f"{ACCOUNT}:custom-model/{name}"
            job["trainingMetrics"] = {"trainingLoss": 0.6}
            job["validationMetrics"] = [{"validationLoss": 0.7}]
        elif failed_in is not None:
            index = phases.index(failed_in)
            job["statusDetails"][failed_in] = {"status": "Failed"}
            for phase in phases[index + 1 :]:
                job["statusDetails"][phase] = {"status": "NotStarted"}
            job["failureMessage"] = "stubbed failure"
        job.update(more)

    def count(self, name: str) -> int:
        return self.calls.count(name)

    def changes(self) -> list[str]:
        """Every call made so far that creates, stops, or deletes something."""
        return [call for call in self.calls if call.startswith(("create_", "stop_", "delete_"))]


def open_gate(cfg, context, s3: FakeS3 | None = None, staged: bool = True) -> dict:
    """Write a passing gate record for the invented split; stage the files into ``s3``."""
    write_baseline_report(cfg, context)
    volumes, pricing = volumes_for(context), load_pricing()
    export, _ = export_files(cfg, context)
    ratio = load_token_ratio(cfg, export.prompt_id, export.split_hash)
    checks = validate_export(export, cfg, context, volumes, ratio.value)
    estimate = estimate_path_b(
        cfg, pricing, volumes, ratio, ledger_records(context), test_examples=22, max_new_tokens=192
    )
    checks += gate_checks(estimate, cfg)
    where = stage_export(s3, export, BUCKET, cfg) if staged and s3 is not None else None
    record = build_gate_record(export, checks, estimate, cfg, pricing, where)
    write_gate_record(record, cfg.report_dir)
    return record


def put_metrics(s3: FakeS3, attempt, training=TRAINING_CSV, validation=VALIDATION_CSV) -> None:
    """Put the two metrics files where a finished job writes them."""
    prefix = attempt.output_s3_uri.removeprefix(f"s3://{BUCKET}/")
    folder = f"{prefix}model-customization-job-abc123/"
    files = {
        "training_artifacts/step_wise_training_metrics.csv": training,
        "validation_artifacts/post_fine_tuning_validation/validation_metrics.csv": validation,
    }
    for name, content in files.items():
        if content is not None:
            s3.objects[(BUCKET, folder + name)] = {"Body": content, "ChecksumSHA256": "-"}


def make_tuning(cfg, context: RunnerContext, bedrock=None, s3=None, **options) -> BedrockTuning:
    return BedrockTuning(
        cfg, load_pricing(), context, BUCKET, ROLE, bedrock=bedrock, s3=s3, **options
    )
