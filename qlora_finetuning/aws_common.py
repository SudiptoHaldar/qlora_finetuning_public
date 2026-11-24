"""The AWS sign-in, shared by both paths (req_003 v3.03; taken out of ``frontier/bedrock.py``).

Path A (SageMaker) and Path B (Bedrock) must not import each other, and both need the same
three things: a session for the configured profile, one clear message when the sign-in is
missing or expired, and a free check that the sign-in is the configured account.

Usage::

    from qlora_finetuning.aws_common import aws_session, check_identity, load_aws_settings

    aws = load_aws_settings()                       # profile, account_id, region
    session = aws_session(aws)                      # no network call
    check_identity(session.client("sts"), aws)      # ConfigError with the sign-in command

    try:
        client.some_call()
    except AUTH_EXCEPTIONS:
        raise ConfigError(sign_in_message(aws["profile"])) from None

Creating a session or a client costs nothing and makes no call. Anything that creates,
changes, or deletes an AWS resource, or that is billed, is run by the owner.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    CredentialRetrievalError,
    NoCredentialsError,
    ProfileNotFound,
    SSOTokenLoadError,
    TokenRetrievalError,
    UnauthorizedSSOTokenError,
)

from qlora_finetuning.settings import ConfigError, load_config

AWS_REQUIRED_KEYS = ("profile", "account_id", "region")
# An expired or absent sign-in shows up as one of these exceptions or error codes.
AUTH_EXCEPTIONS = (
    UnauthorizedSSOTokenError,
    SSOTokenLoadError,
    TokenRetrievalError,
    NoCredentialsError,
    CredentialRetrievalError,
    ProfileNotFound,
)
AUTH_ERROR_CODES = frozenset(
    {
        "ExpiredToken",
        "ExpiredTokenException",
        "RequestExpired",
        "InvalidClientTokenId",
        "UnrecognizedClientException",
    }
)


def sign_in_message(profile: str) -> str:
    return f"the AWS sign-in is missing or has expired: run 'aws sso login --profile {profile}'"


def load_aws_settings(config_dir: Path | str | None = None) -> dict[str, str]:
    """The profile, account id, and region of ``config/aws.yaml``, each as text."""
    raw = load_config("aws", required=AWS_REQUIRED_KEYS, config_dir=config_dir)
    return {key: str(raw[key]) for key in AWS_REQUIRED_KEYS}


def aws_session(aws: dict[str, str]) -> Any:
    """A boto3 session for the configured profile and region. Makes no network call."""
    # botocore logs a full traceback when an SSO sign-in has expired. The operator gets one
    # line from us instead, telling them to sign in.
    for noisy in ("botocore.tokens", "botocore.credentials"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    try:
        return boto3.Session(profile_name=aws["profile"], region_name=aws["region"])
    except ProfileNotFound:
        raise ConfigError(sign_in_message(aws["profile"])) from None


def error_code(error: ClientError) -> str:
    return str(error.response.get("Error", {}).get("Code", "ClientError"))


def check_identity(sts: Any, aws: dict[str, str]) -> None:
    """Raise ``ConfigError`` unless the sign-in is valid and is the configured account.

    Called before a run exists, so a missing sign-in leaves no failed run behind. The call
    is free.
    """
    try:
        account = str(sts.get_caller_identity().get("Account"))
    except AUTH_EXCEPTIONS:
        raise ConfigError(sign_in_message(aws["profile"])) from None
    except ClientError as error:
        code = error_code(error)
        if code in AUTH_ERROR_CODES:
            raise ConfigError(sign_in_message(aws["profile"])) from None
        raise ConfigError(f"AWS did not confirm the sign-in: {code}") from None
    except BotoCoreError as error:
        raise ConfigError(f"AWS could not be reached: {type(error).__name__}") from None
    if account != aws["account_id"]:
        raise ConfigError(
            f"profile {aws['profile']} signs in to account {account}, "
            f"expected {aws['account_id']} (config/aws.yaml)"
        )
