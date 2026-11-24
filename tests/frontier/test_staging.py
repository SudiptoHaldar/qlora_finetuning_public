"""Tests for qlora_finetuning.frontier.staging, with an in-memory stand-in for S3."""

from __future__ import annotations

import base64
import hashlib
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, UnauthorizedSSOTokenError

from qlora_finetuning.frontier.export import export_files
from qlora_finetuning.frontier.staging import StagingError, object_key, stage_export
from qlora_finetuning.settings import ConfigError
from tests.frontier.export_helpers import FakeS3

BUCKET = "qlora-sagemaker-123456789012"


@pytest.fixture
def export(context, cfg):
    return export_files(cfg, context)[0]


def client_error(code: str, operation: str = "HeadObject") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "stubbed"}}, operation)


def test_both_files_are_uploaded_under_the_export_id_and_verified(export, cfg):
    s3 = FakeS3()
    staged = stage_export(s3, export, BUCKET, cfg)
    prefix = f"data/bedrock/{export.export_id}/"
    assert staged == {
        role: {
            "s3_uri": f"s3://{BUCKET}/{prefix}{file.path.name}",
            "bytes": file.size,
            "sha256": file.sha256,
        }
        for role, file in export.files.items()
    }
    assert set(s3.objects) == {
        (BUCKET, prefix + "finqa_fit.jsonl"),
        (BUCKET, prefix + "finqa_dev.jsonl"),
    }
    for role, file in export.files.items():
        stored = s3.objects[(BUCKET, object_key(cfg, export, file))]
        content = file.path.read_bytes()
        assert stored["Body"] == content, role
        # The checksum sent is the file's SHA-256 as S3 writes it: the digest in base64.
        assert stored["ChecksumSHA256"] == base64.b64encode(
            hashlib.sha256(content).digest()
        ).decode()
    # Per file: look, upload, look again. The id lists stay local.
    assert s3.count("put_object") == 2 and s3.count("head_object") == 4
    assert not any("ids" in key for _, key in s3.objects)


def test_files_already_staged_and_equal_are_not_sent_again(export, cfg):
    s3 = FakeS3()
    first = stage_export(s3, export, BUCKET, cfg)
    second = stage_export(s3, export, BUCKET, cfg)
    assert second == first
    assert s3.count("put_object") == 2  # only the first time


def test_an_object_with_other_content_under_the_key_is_an_error(export, cfg):
    s3 = FakeS3()
    key = object_key(cfg, export, export.files["training"])
    s3.objects[(BUCKET, key)] = {"Body": b"something else", "ChecksumSHA256": "AAAA"}
    with pytest.raises(StagingError, match="already exists with other content; delete it"):
        stage_export(s3, export, BUCKET, cfg)
    assert s3.count("put_object") == 0


def test_an_object_that_is_not_the_file_after_the_upload_is_an_error(export, cfg):
    s3 = FakeS3()
    s3.corrupt_on_put = True
    with pytest.raises(StagingError, match="is not the local file after the upload"):
        stage_export(s3, export, BUCKET, cfg)


def test_a_file_changed_after_the_export_is_refused_before_any_call(export, cfg):
    path = export.files["validation"].path
    path.write_bytes(path.read_bytes() + b"{}\n")
    s3 = FakeS3()
    with pytest.raises(StagingError, match="the validation file changed on disk"):
        stage_export(s3, export, BUCKET, cfg)
    assert not any(key.endswith("finqa_dev.jsonl") for _, key in s3.objects)


@pytest.mark.parametrize("code", ["404", "NoSuchKey", "NotFound"])
def test_each_way_s3_says_missing_leads_to_an_upload(export, cfg, code):
    s3 = FakeS3()
    real_head = s3.head_object
    seen = []

    def head(**arguments):
        if arguments["Key"] not in seen:
            seen.append(arguments["Key"])
            raise client_error(code)
        return real_head(**arguments)

    s3.head_object = head
    assert set(stage_export(s3, export, BUCKET, cfg)) == {"training", "validation"}
    assert s3.count("put_object") == 2


def test_an_s3_error_is_a_staging_error_with_the_code(export, cfg):
    s3 = MagicMock()
    s3.head_object.side_effect = client_error("AccessDenied")
    with pytest.raises(StagingError, match="S3 answered AccessDenied when asked to look for"):
        stage_export(s3, export, BUCKET, cfg)
    s3.put_object.assert_not_called()

    s3 = FakeS3()
    s3.put_object = MagicMock(side_effect=client_error("InvalidRequest", "PutObject"))
    with pytest.raises(StagingError, match="S3 answered InvalidRequest when asked to upload"):
        stage_export(s3, export, BUCKET, cfg)


def test_no_network_is_a_staging_error_naming_the_kind(export, cfg):
    s3 = MagicMock()
    s3.head_object.side_effect = EndpointConnectionError(endpoint_url="https://stub.invalid")
    with pytest.raises(StagingError, match="could not be reached .* EndpointConnectionError"):
        stage_export(s3, export, BUCKET, cfg)


@pytest.mark.parametrize("error", [UnauthorizedSSOTokenError(), client_error("ExpiredToken")])
def test_a_missing_sign_in_names_the_command(export, cfg, error):
    s3 = MagicMock()
    s3.head_object.side_effect = error
    with pytest.raises(ConfigError, match="aws sso login --profile owner-profile") as caught:
        stage_export(s3, export, BUCKET, cfg, profile="owner-profile")
    assert not isinstance(caught.value, StagingError)


def test_the_key_uses_the_configured_prefix_and_only_allowed_characters(export, cfg):
    key = object_key(cfg, export, export.files["training"])
    assert key == f"data/bedrock/{export.export_id}/finqa_fit.jsonl"
    assert all(ch.isalnum() or ch in "_-./" for ch in key)
