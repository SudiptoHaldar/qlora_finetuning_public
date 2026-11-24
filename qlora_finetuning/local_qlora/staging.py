"""S3 for Path A: stage a job's inputs, and bring a workspace's files home (req_003 v3.03).

Uploads carry a SHA-256 checksum and are read back before they count. An object that
already exists with the same content is left alone; one with other content is an error.
Uploading creates objects in the project's bucket, so it happens inside ``launch``, which
the owner runs. Listing and downloading read only.

Usage::

    from qlora_finetuning.local_qlora.staging import download_object, list_keys, put_checked

    put_checked(s3, bucket, "sagemaker/code/<commit>/source.tar.gz", content, profile="qlora")
    keys = list_keys(s3, bucket, "sagemaker/workspaces/<name>/", profile="qlora")
    download_object(s3, bucket, key, local_path, profile="qlora")

Every client is passed in; a test hands over a stand-in.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from qlora_finetuning.aws_common import (
    AUTH_ERROR_CODES,
    AUTH_EXCEPTIONS,
    error_code,
    sign_in_message,
)
from qlora_finetuning.logging_setup import get_logger
from qlora_finetuning.settings import ConfigError

MISSING_CODES = frozenset({"404", "NoSuchKey", "NotFound"})

logger = get_logger(__name__)


class StagingError(ConfigError):
    """An object could not be staged or read, or the object in S3 is not the local content."""


def checksum(content: bytes) -> str:
    """SHA-256 as S3 reports it: the digest in base64."""
    return base64.b64encode(hashlib.sha256(content).digest()).decode("ascii")


def call(operation: Any, profile: str, what: str, **arguments: Any) -> Any:
    """Call S3. ``None`` means the object does not exist; other failures raise."""
    try:
        return operation(**arguments)
    except AUTH_EXCEPTIONS:
        raise ConfigError(sign_in_message(profile)) from None
    except ClientError as error:
        code = error_code(error)
        if code in AUTH_ERROR_CODES:
            raise ConfigError(sign_in_message(profile)) from None
        if code in MISSING_CODES:
            return None
        raise StagingError(f"S3 answered {code} when asked to {what}") from None
    except BotoCoreError as error:
        raise StagingError(f"S3 could not be reached to {what}: {type(error).__name__}") from None


def _matches(head: dict[str, Any] | None, size: int, digest: str) -> bool:
    return (
        head is not None
        and head.get("ContentLength") == size
        and head.get("ChecksumSHA256") == digest
    )


def put_checked(
    s3: Any,
    bucket: str,
    key: str,
    content: bytes,
    *,
    profile: str,
    content_type: str = "application/octet-stream",
) -> dict[str, Any]:
    """Upload ``content`` unless it is already there, and verify it.

    Returns the S3 URI, the size, and the SHA-256 that ``head_object`` confirmed.
    """
    digest, where = checksum(content), {"Bucket": bucket, "Key": key}
    uri = f"s3://{bucket}/{key}"
    head = call(s3.head_object, profile, f"look for {uri}", ChecksumMode="ENABLED", **where)
    if head is not None and not _matches(head, len(content), digest):
        raise StagingError(f"{uri} already exists with other content; delete it and try again")
    if head is None:
        call(
            s3.put_object,
            profile,
            f"upload {uri}",
            Body=content,
            ChecksumSHA256=digest,
            ContentType=content_type,
            **where,
        )
        head = call(s3.head_object, profile, f"verify {uri}", ChecksumMode="ENABLED", **where)
        if not _matches(head, len(content), digest):
            raise StagingError(f"{uri} is not the local content after the upload")
        logger.info("staged %s (%d bytes)", uri, len(content))
    else:
        logger.info("%s is already staged and equal", uri)
    return {"s3_uri": uri, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def list_keys(s3: Any, bucket: str, prefix: str, *, profile: str) -> dict[str, int]:
    """Every key under ``prefix`` with its size. Reads only."""
    keys: dict[str, int] = {}
    token: str | None = None
    while True:
        arguments: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            arguments["ContinuationToken"] = token
        page = call(s3.list_objects_v2, profile, f"list s3://{bucket}/{prefix}", **arguments)
        for item in (page or {}).get("Contents", []):
            keys[str(item["Key"])] = int(item["Size"])
        token = (page or {}).get("NextContinuationToken")
        if not token:
            return keys


def read_object(s3: Any, bucket: str, key: str, *, profile: str) -> bytes | None:
    """The object's content, or ``None`` if it does not exist. Reads only."""
    response = call(s3.get_object, profile, f"read s3://{bucket}/{key}", Bucket=bucket, Key=key)
    return None if response is None else response["Body"].read()


def download_object(
    s3: Any, bucket: str, key: str, path: Path, *, profile: str, size: int | None = None
) -> bool:
    """Write the object to ``path`` unless a file of that size is already there.

    Returns whether anything was written. Reads only.
    """
    path = Path(path)
    if size is not None and path.is_file() and path.stat().st_size == size:
        return False
    content = read_object(s3, bucket, key, profile=profile)
    if content is None:
        raise StagingError(f"s3://{bucket}/{key} is not there any more")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return True
