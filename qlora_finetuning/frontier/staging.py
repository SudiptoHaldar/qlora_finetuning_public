"""Stage the training files in S3 and prove that what arrived is what was made (req_004 v4.02).

Each file goes to ``s3://<bucket>/data/bedrock/<export_id>/<name>`` with its SHA-256 as the
object's checksum, so S3 itself refuses a damaged upload, and is then read back with
``head_object``: size and checksum must equal the local file's. The key carries the export
id, which is derived from the content, so an object that is already there and equal is left
alone, and one that differs is an error.

Staging creates objects in the project's bucket. It is run by the owner
(``python -m scripts.nova_export --upload``), only after the files passed validation. The
files hold dataset text from fit and dev, never from test.

Usage::

    from qlora_finetuning.frontier.staging import stage_export

    staged = stage_export(s3_client, export, bucket="qlora-sagemaker-123456789012", cfg=cfg)
    staged["training"]["s3_uri"]    # s3://.../data/bedrock/<export_id>/finqa_fit.jsonl

A test passes a stub client. Besides ``bedrock.py`` this is the only module of the package
that imports botocore.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from qlora_finetuning.frontier.bedrock import AUTH_ERROR_CODES, AUTH_EXCEPTIONS, sign_in_message
from qlora_finetuning.frontier.config import FrontierConfig
from qlora_finetuning.frontier.export import Export, ExportFile
from qlora_finetuning.logging_setup import get_logger
from qlora_finetuning.settings import ConfigError

MISSING_CODES = frozenset({"404", "NoSuchKey", "NotFound"})

logger = get_logger(__name__)


class StagingError(ConfigError):
    """A file could not be staged, or the object in S3 is not the local file."""


def object_key(cfg: FrontierConfig, export: Export, file: ExportFile) -> str:
    return f"{cfg.s3_prefix}{export.export_id}/{file.path.name}"


def _checksum(content: bytes) -> str:
    """SHA-256 as S3 reports it: the digest in base64."""
    return base64.b64encode(hashlib.sha256(content).digest()).decode("ascii")


def _call(operation: Any, profile: str, what: str, **arguments: Any) -> dict[str, Any] | None:
    """Call S3. ``None`` means the object does not exist; other failures raise."""
    try:
        return operation(**arguments)
    except AUTH_EXCEPTIONS:
        raise ConfigError(sign_in_message(profile)) from None
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", "ClientError"))
        if code in AUTH_ERROR_CODES:
            raise ConfigError(sign_in_message(profile)) from None
        if code in MISSING_CODES:
            return None
        raise StagingError(f"S3 answered {code} when asked to {what}") from None
    except BotoCoreError as error:
        raise StagingError(f"S3 could not be reached to {what}: {type(error).__name__}") from None


def _matches(head: dict[str, Any] | None, size: int, checksum: str) -> bool:
    return (
        head is not None
        and head.get("ContentLength") == size
        and head.get("ChecksumSHA256") == checksum
    )


def stage_export(
    s3: Any, export: Export, bucket: str, cfg: FrontierConfig, profile: str = "qlora"
) -> dict[str, dict[str, Any]]:
    """Upload both files of ``export`` unless they are already there, and verify them.

    Returns, per role, the S3 URI, the size, and the SHA-256 that ``head_object`` confirmed.
    Raises ``StagingError`` if a local file changed since the export, if an object with the
    same key holds other content, or if the object read back is not the local file.
    """
    staged: dict[str, dict[str, Any]] = {}
    for role, file in export.files.items():
        content = file.path.read_bytes()
        if hashlib.sha256(content).hexdigest() != file.sha256:
            raise StagingError(f"the {role} file changed on disk after it was exported")
        key, checksum = object_key(cfg, export, file), _checksum(content)
        where = {"Bucket": bucket, "Key": key}
        uri = f"s3://{bucket}/{key}"

        head = _call(s3.head_object, profile, f"look for {uri}", ChecksumMode="ENABLED", **where)
        if head is not None and not _matches(head, len(content), checksum):
            raise StagingError(
                f"{uri} already exists with other content; delete it and stage again"
            )
        if head is None:
            _call(
                s3.put_object,
                profile,
                f"upload {uri}",
                Body=content,
                ChecksumSHA256=checksum,
                ContentType="application/jsonl",
                **where,
            )
            head = _call(s3.head_object, profile, f"verify {uri}", ChecksumMode="ENABLED", **where)
            if not _matches(head, len(content), checksum):
                raise StagingError(f"{uri} is not the local file after the upload")
            logger.info("staged %s (%d bytes)", uri, len(content))
        else:
            logger.info("%s is already staged and equal", uri)
        staged[role] = {"s3_uri": uri, "bytes": len(content), "sha256": file.sha256}
    return staged
