"""Stand-in AWS clients for the SageMaker tests, and a launcher wired to them under ``tmp_path``.

Nothing here reaches AWS. The invented split is frozen under the temporary root, its bundle
is written there, and the tracking root is the same folder.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError, NoCredentialsError

from qlora_finetuning.data.bundle import export_bundle
from qlora_finetuning.local_qlora import sagemaker as sagemaker_module
from qlora_finetuning.local_qlora.sagemaker import SageMakerJobs
from qlora_finetuning.local_qlora.sagemaker_config import load_aws_names, load_sagemaker_config
from qlora_finetuning.tracking import GitInfo, load_tracking_config
from tests.data.synthetic import build_frozen_split

COMMIT = "0123456789abcdef0123456789abcdef01234567"
GIT = GitInfo(commit=COMMIT, branch="main", dirty=False)
DIRTY = GitInfo(commit=COMMIT, branch="main", dirty=True, changed=("config/training.yaml",))
SOURCE = b"the archive of a commit"
ACCOUNT = "123456789012"


def client_error(code: str, operation: str = "Op") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "stubbed"}}, operation)


class FakeS3:
    def __init__(self, page_size: int = 1000) -> None:
        self.objects: dict[str, bytes] = {}
        self.puts: list[str] = []
        self.gets: list[str] = []
        self.page_size = page_size

    def head_object(self, Bucket, Key, ChecksumMode=None):  # noqa: N803 (boto3's names)
        if Key not in self.objects:
            raise client_error("404", "HeadObject")
        content = self.objects[Key]
        digest = base64.b64encode(hashlib.sha256(content).digest()).decode("ascii")
        return {"ContentLength": len(content), "ChecksumSHA256": digest}

    def put_object(self, Bucket, Key, Body, ChecksumSHA256=None, ContentType=None):  # noqa: N803
        self.objects[Key] = bytes(Body)
        self.puts.append(Key)
        return {}

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise client_error("NoSuchKey", "GetObject")
        self.gets.append(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):  # noqa: N803
        keys = sorted(key for key in self.objects if key.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start : start + self.page_size]
        response: dict[str, Any] = {
            "Contents": [{"Key": key, "Size": len(self.objects[key])} for key in page]
        }
        if start + self.page_size < len(keys):
            response["NextContinuationToken"] = str(start + self.page_size)
        return response


class FakeSageMaker:
    """Training jobs kept in memory. ``lose_reply`` creates the job and then fails the call."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.created: list[str] = []
        self.stopped: list[str] = []
        self.calls: list[str] = []

    def create_training_job(self, **request):
        self.calls.append("create")
        name = request["TrainingJobName"]
        if name in self.jobs:
            raise client_error("ResourceInUse", "CreateTrainingJob")
        self.created.append(name)
        self.jobs[name] = {
            "TrainingJobName": name,
            "TrainingJobArn": f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:training-job/{name}",
            "TrainingJobStatus": "InProgress",
            "SecondaryStatus": "Starting",
            "ResourceConfig": dict(request["ResourceConfig"]),
            "request": request,
        }
        return {"TrainingJobArn": self.jobs[name]["TrainingJobArn"]}

    def describe_training_job(self, TrainingJobName):  # noqa: N803
        self.calls.append("describe")
        if TrainingJobName not in self.jobs:
            raise client_error("ValidationException", "DescribeTrainingJob")
        return {key: value for key, value in self.jobs[TrainingJobName].items() if key != "request"}

    def list_training_jobs(self, NameContains=None, StatusEquals=None, MaxResults=None):  # noqa: N803
        self.calls.append("list")
        return {
            "TrainingJobSummaries": [
                {"TrainingJobName": name}
                for name, job in self.jobs.items()
                if (NameContains or "") in name and job["TrainingJobStatus"] == StatusEquals
            ]
        }

    def stop_training_job(self, TrainingJobName):  # noqa: N803
        self.calls.append("stop")
        self.stopped.append(TrainingJobName)
        self.finish(TrainingJobName, "Stopped", billable=120)
        return {}

    def finish(self, name: str, status: str = "Completed", billable: int = 1500, **more) -> None:
        self.jobs[name].update(
            TrainingJobStatus=status,
            SecondaryStatus=status,
            TrainingStartTime=datetime(2026, 11, 20, 10, 0, tzinfo=UTC),
            TrainingEndTime=datetime(2026, 11, 20, 10, 25, tzinfo=UTC),
            TrainingTimeInSeconds=billable,
            BillableTimeInSeconds=billable,
            **more,
        )


class FakeSts:
    def __init__(self, account: str = ACCOUNT, signed_in: bool = True) -> None:
        self.account, self.signed_in = account, signed_in

    def get_caller_identity(self):
        if not self.signed_in:
            raise NoCredentialsError()
        return {"Account": self.account}


class FakeIam:
    def __init__(self, service: Any = "sagemaker.amazonaws.com", exists: bool = True) -> None:
        self.service, self.exists = service, exists

    def get_role(self, RoleName):  # noqa: N803
        if not self.exists:
            raise client_error("NoSuchEntity", "GetRole")
        statement = {"Effect": "Allow", "Principal": {"Service": self.service}}
        return {"Role": {"AssumeRolePolicyDocument": {"Statement": [statement]}}}


class FakeQuotas:
    def __init__(self, value: float = 1.0, name: str = "ml.g5.xlarge for training job usage"):
        self.value, self.name = value, name

    def get_paginator(self, operation):
        quotas = self

        class Paginator:
            def paginate(self, ServiceCode):  # noqa: N803
                other = {"QuotaName": "ml.g5.xlarge for spot training job usage", "Value": 0.0}
                return [
                    {"Quotas": [other]},
                    {"Quotas": [{"QuotaName": quotas.name, "Value": quotas.value}]},
                ]

        return Paginator()


class FakeLogs:
    def __init__(self, lines: list[str] | None = None) -> None:
        self.lines = lines

    def describe_log_streams(self, logGroupName, logStreamNamePrefix):  # noqa: N803
        if self.lines is None:
            raise client_error("ResourceNotFoundException", "DescribeLogStreams")
        return {"logStreams": [{"logStreamName": f"{logStreamNamePrefix}algo-1-1700000000"}]}

    def get_log_events(self, logGroupName, logStreamName, limit, startFromHead):  # noqa: N803
        return {"events": [{"message": line} for line in (self.lines or [])[-limit:]]}


def make_jobs(root: Path, monkeypatch=None, **clients: Any) -> SageMakerJobs:
    """A launcher over stand-in clients, an invented split, and its bundle, all under ``root``."""
    split_cfg, data_cfg, _ = build_frozen_split(root)
    cfg = load_sagemaker_config(root=root)
    export_bundle(cfg.bundle_dir, cfg=split_cfg, data_cfg=data_cfg)
    if monkeypatch is not None:  # the temporary folder is not a git repository
        monkeypatch.setattr(sagemaker_module, "build_source", lambda root_, commit: SOURCE)
    wired = {
        "sagemaker": FakeSageMaker(),
        "s3": FakeS3(),
        "sts": FakeSts(),
        "iam": FakeIam(),
        "quotas": FakeQuotas(),
        "logs": FakeLogs(["line one", "line two"]),
        "git": GIT,
    }
    wired.update(clients)
    return SageMakerJobs(
        cfg,
        load_aws_names(),
        load_tracking_config(root=root),
        environment={},
        split_cfg=split_cfg,
        data_cfg=data_cfg,
        **wired,
    )


def with_cfg(jobs: SageMakerJobs, **changes: Any) -> SageMakerJobs:
    jobs.cfg = dataclasses.replace(jobs.cfg, **changes)
    return jobs


def workspace_key(jobs: SageMakerJobs, job: dict[str, Any], relative: str) -> str:
    return f"{jobs.cfg.workspace_prefix}/{job['workspace']}/{relative}"


def put_json_lines(s3: FakeS3, key: str, lines: list[dict[str, Any]]) -> None:
    s3.objects[key] = "".join(json.dumps(line) + "\n" for line in lines).encode()
