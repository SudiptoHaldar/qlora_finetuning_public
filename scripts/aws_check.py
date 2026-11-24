"""Read-only check of the project's AWS foundation (req_000 v0.05).

Prints one PASS or FAIL line per check and exits non-zero if any check fails. It creates,
changes, and deletes nothing, and makes no billed call. Everything it compares against comes
from ``config/aws.yaml``, and the model ids of check C8 from ``config/frontier.yaml``.

Usage (from the repo root, after ``aws sso login --profile qlora``). The script imports the
``qlora_finetuning`` package, so it is run as a module, not by file path::

    venv\\Scripts\\python -m scripts.aws_check
    venv\\Scripts\\python -m scripts.aws_check --config config/aws.yaml --profile qlora

Exit codes: 0 all checks passed, 1 at least one check failed, 2 sign-in or config problem.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fnmatch import fnmatch, fnmatchcase
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
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

try:
    from qlora_finetuning.frontier.config import load_frontier_config
    from qlora_finetuning.settings import CONFIG_DIR, ConfigError, load_config
except ModuleNotFoundError as exc:  # run by file path, so the repo root is not importable
    if exc.name != "qlora_finetuning":
        raise
    raise SystemExit(
        "Run this script as a module from the repository root: python -m scripts.aws_check"
    ) from None

DEFAULT_CONFIG = CONFIG_DIR / "aws.yaml"

REQUIRED_KEYS = (
    "profile",
    "account_id",
    "region",
    "bucket.name",
    "bucket.prefixes.data",
    "bucket.prefixes.bedrock",
    "roles.sagemaker_execution.arn",
    "roles.sagemaker_execution.trusted_service",
    "roles.bedrock_customization.arn",
    "roles.bedrock_customization.trusted_service",
    "roles.bedrock_customization.inline_policy",
    "quota.service_code",
    "quota.usage",
    "quota.minimum",
    "quota.instance_types",
    "bedrock.fine_tunable_models",
    "budget.name",
    "budget.cap_usd",
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
AUTH_EXCEPTIONS = (
    UnauthorizedSSOTokenError,
    SSOTokenLoadError,
    TokenRetrievalError,
    NoCredentialsError,
    CredentialRetrievalError,
    ProfileNotFound,
)
PUBLIC_ACCESS_BLOCKS = (
    "BlockPublicAcls",
    "IgnorePublicAcls",
    "BlockPublicPolicy",
    "RestrictPublicBuckets",
)
CLIENT_CONFIG = Config(
    retries={"max_attempts": 3, "mode": "standard"}, connect_timeout=10, read_timeout=30
)


class SignInError(Exception):
    """The profile has no valid sign-in. The message is the profile name."""


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    name: str
    passed: bool
    detail: str


# --------------------------------------------------------------------------- config


def load_aws_config(
    path: Path | str = DEFAULT_CONFIG, frontier_dir: Path | str | None = None
) -> dict[str, Any]:
    """Load the AWS config through ``settings.load_config`` and check its required keys.

    ``path`` may point at any ``*.yaml`` file, so ``--config`` can name a copy elsewhere.
    The model ids that check C8 asks about come from ``config/frontier.yaml`` (in
    ``frontier_dir``, by default the repository's ``config/``) and are added as
    ``bedrock.inference_models``.
    """
    path = Path(path)
    if path.suffix != ".yaml":
        raise ConfigError(f"config file must end in .yaml: {path}")
    cfg = load_config(path.stem, required=REQUIRED_KEYS, config_dir=path.parent)
    cfg["account_id"] = str(cfg["account_id"])
    frontier = load_frontier_config(config_dir=frontier_dir)
    cfg["bedrock"]["inference_models"] = [model.model_id for model in frontier.models.values()]
    return cfg


# --------------------------------------------------------------------------- helpers


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _role_name(arn: str) -> str:
    return arn.rsplit("/", 1)[-1]


def _trusts(document: dict[str, Any], service: str) -> bool:
    """True when the trust policy lets ``service`` assume the role."""
    for statement in _as_list(document.get("Statement")):
        principal = statement.get("Principal")
        services = _as_list(principal.get("Service")) if isinstance(principal, dict) else []
        if (
            statement.get("Effect") == "Allow"
            and "sts:AssumeRole" in _as_list(statement.get("Action"))
            and service in services
        ):
            return True
    return False


def _allows(document: dict[str, Any], action: str, resource_arn: str) -> bool:
    """True when some Allow statement covers ``action`` on ``resource_arn``."""
    for statement in _as_list(document.get("Statement")):
        if statement.get("Effect") != "Allow":
            continue
        actions = _as_list(statement.get("Action"))
        resources = _as_list(statement.get("Resource"))
        if any(fnmatch(action.lower(), str(a).lower()) for a in actions) and any(
            fnmatchcase(resource_arn, str(r)) for r in resources
        ):
            return True
    return False


# --------------------------------------------------------------------------- checks


def check_identity(sts: Any, cfg: dict[str, Any]) -> CheckResult:
    """C1: the profile signs in and the account id matches config."""
    account = str(sts.get_caller_identity().get("Account"))
    expected = cfg["account_id"]
    if account != expected:
        return CheckResult(
            "C1", "Caller identity", False, f"account {account}, expected {expected}"
        )
    return CheckResult("C1", "Caller identity", True, f"account {account}")


def check_bucket(s3: Any, cfg: dict[str, Any]) -> CheckResult:
    """C2: bucket exists, right region, public access blocked, encrypted by default."""
    name = "Bucket"
    bucket = cfg["bucket"]["name"]
    region = s3.get_bucket_location(Bucket=bucket).get("LocationConstraint") or "us-east-1"
    if region != cfg["region"]:
        return CheckResult("C2", name, False, f"{bucket} is in {region}, expected {cfg['region']}")

    blocks = s3.get_public_access_block(Bucket=bucket).get("PublicAccessBlockConfiguration", {})
    off = [key for key in PUBLIC_ACCESS_BLOCKS if blocks.get(key) is not True]
    if off:
        return CheckResult("C2", name, False, f"public-access blocks off: {', '.join(off)}")

    encryption = s3.get_bucket_encryption(Bucket=bucket).get(
        "ServerSideEncryptionConfiguration", {}
    )
    algorithms = [
        rule.get("ApplyServerSideEncryptionByDefault", {}).get("SSEAlgorithm")
        for rule in encryption.get("Rules", [])
    ]
    algorithms = [a for a in algorithms if a]
    if not algorithms:
        return CheckResult("C2", name, False, "no default encryption")
    return CheckResult(
        "C2", name, True, f"{bucket} in {region}, private, encrypted ({', '.join(algorithms)})"
    )


def check_sagemaker_role(iam: Any, cfg: dict[str, Any]) -> CheckResult:
    """C3: the SageMaker role exists and trusts the SageMaker service."""
    name = "SageMaker role"
    role = cfg["roles"]["sagemaker_execution"]
    role_name = _role_name(role["arn"])
    document = iam.get_role(RoleName=role_name)["Role"]["AssumeRolePolicyDocument"]
    if not _trusts(document, role["trusted_service"]):
        return CheckResult(
            "C3", name, False, f"{role_name} does not trust {role['trusted_service']}"
        )
    return CheckResult("C3", name, True, f"{role_name} trusts {role['trusted_service']}")


def check_bedrock_role(iam: Any, cfg: dict[str, Any]) -> CheckResult:
    """C4: the Bedrock role exists, trusts Bedrock, and has the expected S3 permissions."""
    name = "Bedrock role"
    role = cfg["roles"]["bedrock_customization"]
    role_name = _role_name(role["arn"])
    trust = iam.get_role(RoleName=role_name)["Role"]["AssumeRolePolicyDocument"]
    if not _trusts(trust, role["trusted_service"]):
        return CheckResult(
            "C4", name, False, f"{role_name} does not trust {role['trusted_service']}"
        )

    policy = iam.get_role_policy(RoleName=role_name, PolicyName=role["inline_policy"])[
        "PolicyDocument"
    ]
    bucket_arn = f"arn:aws:s3:::{cfg['bucket']['name']}"
    prefixes = cfg["bucket"]["prefixes"]
    needed = (
        ("s3:GetObject", prefixes["data"], "read on"),
        ("s3:GetObject", prefixes["bedrock"], "read on"),
        ("s3:PutObject", prefixes["bedrock"], "write on"),
    )
    lacking = [
        f"{label} {prefix}*"
        for action, prefix, label in needed
        if not _allows(policy, action, f"{bucket_arn}/{prefix}aws-check-probe")
    ]
    if lacking:
        return CheckResult("C4", name, False, f"{role['inline_policy']} lacks {', '.join(lacking)}")
    return CheckResult(
        "C4",
        name,
        True,
        f"{role_name} trusts {role['trusted_service']}; {role['inline_policy']} grants "
        f"read on {prefixes['data']}*, read and write on {prefixes['bedrock']}*",
    )


def check_gpu_quota(quotas: Any, cfg: dict[str, Any]) -> CheckResult:
    """C5: the training-job quota is at least the minimum for every configured instance type."""
    name = "GPU quota"
    quota_cfg = cfg["quota"]
    wanted = {f"{itype} for {quota_cfg['usage']}": itype for itype in quota_cfg["instance_types"]}
    found: dict[str, float] = {}
    paginator = quotas.get_paginator("list_service_quotas")
    for page in paginator.paginate(ServiceCode=quota_cfg["service_code"]):
        for quota in page.get("Quotas", []):
            if quota.get("QuotaName") in wanted:
                found[wanted[quota["QuotaName"]]] = float(quota.get("Value", 0))

    minimum = float(quota_cfg["minimum"])
    problems = []
    for itype in quota_cfg["instance_types"]:
        if itype not in found:
            problems.append(f"{itype} quota not found")
        elif found[itype] < minimum:
            problems.append(f"{itype} = {found[itype]:g}, need {minimum:g}")
    if problems:
        return CheckResult("C5", name, False, "; ".join(problems))
    summary = ", ".join(f"{itype} = {found[itype]:g}" for itype in quota_cfg["instance_types"])
    return CheckResult("C5", name, True, f"{quota_cfg['usage']}: {summary}")


def check_nova_models(bedrock: Any, cfg: dict[str, Any]) -> CheckResult:
    """C6: every configured model id is listed as fine-tunable."""
    name = "Nova models"
    response = bedrock.list_foundation_models(byCustomizationType="FINE_TUNING")
    listed = {summary.get("modelId") for summary in response.get("modelSummaries", [])}
    configured = cfg["bedrock"]["fine_tunable_models"]
    missing = [model for model in configured if model not in listed]
    if missing:
        return CheckResult("C6", name, False, f"not fine-tunable: {', '.join(missing)}")
    return CheckResult("C6", name, True, f"fine-tunable: {', '.join(configured)}")


def check_nova_inference(bedrock: Any, cfg: dict[str, Any]) -> CheckResult:
    """C8: every model Path B calls is active and can be called on demand (req_004 v4.01).

    A model that has moved to Legacy can still answer, but no new fine-tuning job can be
    created on it, so it fails here.
    """
    name = "Nova inference"
    problems, details = [], []
    for model_id in cfg["bedrock"]["inference_models"]:
        model = bedrock.get_foundation_model(modelIdentifier=model_id).get("modelDetails", {})
        lifecycle = model.get("modelLifecycle") or {}
        status = lifecycle.get("status")
        end = lifecycle.get("endOfLifeTime")
        ending = f", end of life {end:%Y-%m-%d}" if hasattr(end, "strftime") else ""
        if "ON_DEMAND" not in _as_list(model.get("inferenceTypesSupported")):
            problems.append(f"{model_id} cannot be called on demand")
        elif status != "ACTIVE":
            problems.append(f"{model_id} is {status}{ending}")
        else:
            details.append(f"{model_id} {status}, on demand{ending}")
    if problems:
        return CheckResult("C8", name, False, "; ".join(problems))
    return CheckResult("C8", name, True, "; ".join(details))


def check_budget(budgets: Any, cfg: dict[str, Any]) -> CheckResult:
    """C7: the budget exists with the configured cap and an alert that has a subscriber.

    Subscriber addresses are counted, never printed.
    """
    name = "Budget"
    account = cfg["account_id"]
    budget_name = cfg["budget"]["name"]
    limit = budgets.describe_budget(AccountId=account, BudgetName=budget_name)["Budget"].get(
        "BudgetLimit", {}
    )
    try:
        amount = Decimal(str(limit.get("Amount")))
        cap = Decimal(str(cfg["budget"]["cap_usd"]))
    except InvalidOperation:
        return CheckResult("C7", name, False, f"{budget_name} has no readable limit")
    if amount != cap or limit.get("Unit") != "USD":
        return CheckResult(
            "C7",
            name,
            False,
            f"{budget_name} cap is {amount} {limit.get('Unit')}, expected {cap} USD",
        )

    notifications = budgets.describe_notifications_for_budget(
        AccountId=account, BudgetName=budget_name
    ).get("Notifications", [])
    with_subscriber = 0
    for notification in notifications:
        subscribers = budgets.describe_subscribers_for_notification(
            AccountId=account, BudgetName=budget_name, Notification=notification
        ).get("Subscribers", [])
        if subscribers:
            with_subscriber += 1
    if with_subscriber == 0:
        return CheckResult(
            "C7",
            name,
            False,
            f"{budget_name} has {len(notifications)} alert(s), none with a subscriber",
        )
    return CheckResult(
        "C7",
        name,
        True,
        f"{budget_name} cap {cap} USD, {with_subscriber} of {len(notifications)} alert(s) "
        "have a subscriber",
    )


# --------------------------------------------------------------------------- runner

CheckFn = Callable[[Any, dict[str, Any]], CheckResult]

# (check id, name, boto3 service name, function), in report order.
CHECKS: tuple[tuple[str, str, str, CheckFn], ...] = (
    ("C1", "Caller identity", "sts", check_identity),
    ("C2", "Bucket", "s3", check_bucket),
    ("C3", "SageMaker role", "iam", check_sagemaker_role),
    ("C4", "Bedrock role", "iam", check_bedrock_role),
    ("C5", "GPU quota", "service-quotas", check_gpu_quota),
    ("C6", "Nova models", "bedrock", check_nova_models),
    ("C7", "Budget", "budgets", check_budget),
    ("C8", "Nova inference", "bedrock", check_nova_inference),
)


def _guard(check_id: str, name: str, fn: CheckFn, client: Any, cfg: dict[str, Any]) -> CheckResult:
    """Run one check. AWS errors become FAIL; a sign-in problem raises SignInError."""
    try:
        return fn(client, cfg)
    except AUTH_EXCEPTIONS as exc:
        raise SignInError(cfg["profile"]) from exc
    except ClientError as exc:
        error = exc.response.get("Error", {})
        code = error.get("Code", "ClientError")
        if code in AUTH_ERROR_CODES:
            raise SignInError(cfg["profile"]) from exc
        return CheckResult(
            check_id, name, False, f"{code}: {error.get('Message', '')}".rstrip(": ")
        )
    except BotoCoreError as exc:
        return CheckResult(check_id, name, False, str(exc))


def run_checks(session: Any, cfg: dict[str, Any]) -> list[CheckResult]:
    """Run C1 to C8 in order. One failing check does not stop the others."""
    clients: dict[str, Any] = {}
    results = []
    for check_id, name, service, fn in CHECKS:
        try:
            if service not in clients:
                clients[service] = session.client(service, config=CLIENT_CONFIG)
        except AUTH_EXCEPTIONS as exc:
            raise SignInError(cfg["profile"]) from exc
        results.append(_guard(check_id, name, fn, clients[service], cfg))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only check of the AWS foundation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to the AWS config YAML")
    parser.add_argument("--profile", default=None, help="AWS CLI profile (default: from config)")
    args = parser.parse_args(argv)

    try:
        cfg = load_aws_config(args.config)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 2
    if args.profile:
        cfg["profile"] = args.profile

    try:
        session = boto3.Session(profile_name=cfg["profile"], region_name=cfg["region"])
        results = run_checks(session, cfg)
    except (SignInError, *AUTH_EXCEPTIONS):
        print(f"SIGN-IN REQUIRED: run 'aws sso login --profile {cfg['profile']}' and try again")
        return 2

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status}  {result.check_id}  {result.name}: {result.detail}")
    passed = sum(result.passed for result in results)
    print(f"{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
