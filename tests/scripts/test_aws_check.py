"""Unit tests for scripts.aws_check.

Every check gets a stubbed client, so nothing here touches the network or needs credentials.
The one live run is marked ``network`` and is deselected by default.
"""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError, UnauthorizedSSOTokenError

from scripts import aws_check

BUCKET = "qlora-sagemaker-123456789012"
INFERENCE_MODELS = ("amazon.nova-pro-v1:0", "amazon.nova-lite-v1:0")
SUBSCRIBER_ADDRESS = "owner-at-example"  # stands in for an email; must never be printed


def client_error(code: str, operation: str = "Op") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "stubbed"}}, operation)


@pytest.fixture
def cfg() -> dict:
    return {
        "version": 1,
        "profile": "qlora",
        "account_id": "123456789012",
        "region": "us-east-1",
        "bucket": {
            "name": BUCKET,
            "prefixes": {
                "data": "data/",
                "sagemaker": "sagemaker/",
                "bedrock": "bedrock/",
                "adapters": "adapters/",
            },
        },
        "roles": {
            "sagemaker_execution": {
                "arn": "arn:aws:iam::123456789012:role/qlora-sagemaker-execution",
                "trusted_service": "sagemaker.amazonaws.com",
            },
            "bedrock_customization": {
                "arn": "arn:aws:iam::123456789012:role/qlora-bedrock-customization",
                "trusted_service": "bedrock.amazonaws.com",
                "inline_policy": "qlora-bedrock-s3",
            },
        },
        "quota": {
            "service_code": "sagemaker",
            "usage": "training job usage",
            "minimum": 1,
            "instance_types": ["ml.g5.xlarge", "ml.g6.xlarge"],
        },
        "bedrock": {
            "fine_tunable_models": ["amazon.nova-pro-v1:0:300k", "amazon.nova-lite-v1:0:300k"],
            "inference_models": list(INFERENCE_MODELS),
        },
        "budget": {"name": "qlora-finetuning", "cap_usd": 300},
    }


def trust(service) -> dict:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}
        ],
    }


BEDROCK_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": f"arn:aws:s3:::{BUCKET}"},
        {"Effect": "Allow", "Action": "s3:GetObject", "Resource": f"arn:aws:s3:::{BUCKET}/data/*"},
        {
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject"],
            "Resource": f"arn:aws:s3:::{BUCKET}/bedrock/*",
        },
    ],
}


# --------------------------------------------------------------------------- stub builders


def make_sts(account="123456789012"):
    sts = MagicMock()
    sts.get_caller_identity.return_value = {"Account": account}
    return sts


def make_s3(location=None, blocks_off=(), algorithm="AES256"):
    s3 = MagicMock()
    s3.get_bucket_location.return_value = {"LocationConstraint": location}
    s3.get_public_access_block.return_value = {
        "PublicAccessBlockConfiguration": {
            key: key not in blocks_off for key in aws_check.PUBLIC_ACCESS_BLOCKS
        }
    }
    rules = (
        [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": algorithm}}] if algorithm else []
    )
    s3.get_bucket_encryption.return_value = {"ServerSideEncryptionConfiguration": {"Rules": rules}}
    return s3


def make_iam(trust_doc, policy_doc=None):
    iam = MagicMock()
    iam.get_role.return_value = {"Role": {"AssumeRolePolicyDocument": trust_doc}}
    iam.get_role_policy.return_value = {"PolicyDocument": policy_doc or BEDROCK_POLICY}
    return iam


def make_quotas(pages):
    quotas = MagicMock()
    quotas.get_paginator.return_value.paginate.return_value = [
        {"Quotas": [{"QuotaName": name, "Value": value} for name, value in page]} for page in pages
    ]
    return quotas


GOOD_QUOTA_PAGES = [
    [
        ("ml.g5.xlarge for spot training job usage", 0.0),
        ("ml.g5.xlarge for training job usage", 1.0),
    ],
    [("ml.g6.xlarge for training job usage", 1.0)],
]


def make_bedrock(model_ids=("amazon.nova-pro-v1:0:300k", "amazon.nova-lite-v1:0:300k")):
    bedrock = MagicMock()
    bedrock.list_foundation_models.return_value = {
        "modelSummaries": [{"modelId": model_id} for model_id in model_ids]
    }
    bedrock.get_foundation_model.side_effect = lambda modelIdentifier: foundation_model(
        modelIdentifier
    )
    return bedrock


def foundation_model(model_id, status="ACTIVE", types=("ON_DEMAND", "PROVISIONED"), end=None):
    lifecycle = {"status": status}
    if end is not None:
        lifecycle["endOfLifeTime"] = end
    return {
        "modelDetails": {
            "modelId": model_id,
            "inferenceTypesSupported": list(types),
            "modelLifecycle": lifecycle,
        }
    }


def make_budgets(amount="300.0", thresholds=(100.0, 80.0), subscribers=1):
    budgets = MagicMock()
    budgets.describe_budget.return_value = {
        "Budget": {"BudgetLimit": {"Amount": amount, "Unit": "USD"}, "TimeUnit": "ANNUALLY"}
    }
    budgets.describe_notifications_for_budget.return_value = {
        "Notifications": [
            {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN", "Threshold": t}
            for t in thresholds
        ]
    }
    budgets.describe_subscribers_for_notification.return_value = {
        "Subscribers": [{"SubscriptionType": "EMAIL", "Address": SUBSCRIBER_ADDRESS}] * subscribers
    }
    return budgets


# --------------------------------------------------------------------------- config


def test_committed_config_loads():
    cfg = aws_check.load_aws_config()
    assert cfg["account_id"] == "123456789012"
    assert isinstance(cfg["account_id"], str)
    assert cfg["quota"]["instance_types"] == ["ml.g5.xlarge", "ml.g6.xlarge"]
    assert set(cfg["bucket"]["prefixes"]) == {"data", "sagemaker", "bedrock", "adapters"}
    # Check C8's model ids come from config/frontier.yaml, not from config/aws.yaml.
    assert cfg["bedrock"]["inference_models"] == list(INFERENCE_MODELS)
    assert "inference_models" not in aws_check.DEFAULT_CONFIG.read_text(encoding="utf-8")


def test_config_fails_when_the_frontier_config_is_absent(tmp_path):
    with pytest.raises(aws_check.ConfigError, match="frontier.yaml"):
        aws_check.load_aws_config(frontier_dir=tmp_path)


def test_committed_config_has_no_secret_or_email():
    text = aws_check.DEFAULT_CONFIG.read_text(encoding="utf-8")
    assert "@" not in text
    assert "AKIA" not in text
    assert "secret" not in text.replace("non-secret", "").lower()


def test_config_coerces_numeric_account_id(tmp_path, cfg):
    import yaml

    cfg["account_id"] = 123456789012
    path = tmp_path / "aws.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert aws_check.load_aws_config(path)["account_id"] == "123456789012"


def test_config_missing_file(tmp_path):
    with pytest.raises(aws_check.ConfigError, match="not found"):
        aws_check.load_aws_config(tmp_path / "absent.yaml")


def test_config_missing_key(tmp_path, cfg):
    import yaml

    del cfg["budget"]["cap_usd"]
    path = tmp_path / "aws.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(aws_check.ConfigError, match="budget.cap_usd"):
        aws_check.load_aws_config(path)


def test_config_not_a_mapping(tmp_path):
    path = tmp_path / "aws.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(aws_check.ConfigError, match="mapping"):
        aws_check.load_aws_config(path)


# --------------------------------------------------------------------------- C1


def test_identity_pass(cfg):
    assert aws_check.check_identity(make_sts(), cfg).passed


def test_identity_wrong_account(cfg):
    result = aws_check.check_identity(make_sts("111111111111"), cfg)
    assert not result.passed
    assert "111111111111" in result.detail


# --------------------------------------------------------------------------- C2


def test_bucket_pass(cfg):
    result = aws_check.check_bucket(make_s3(), cfg)
    assert result.passed
    assert "AES256" in result.detail


def test_bucket_wrong_region(cfg):
    result = aws_check.check_bucket(make_s3(location="eu-west-1"), cfg)
    assert not result.passed
    assert "eu-west-1" in result.detail


def test_bucket_public_block_off(cfg):
    result = aws_check.check_bucket(make_s3(blocks_off=("BlockPublicPolicy",)), cfg)
    assert not result.passed
    assert "BlockPublicPolicy" in result.detail


def test_bucket_no_encryption_rules(cfg):
    assert not aws_check.check_bucket(make_s3(algorithm=None), cfg).passed


@pytest.mark.parametrize(
    ("method", "code"),
    [
        ("get_bucket_location", "NoSuchBucket"),
        ("get_public_access_block", "NoSuchPublicAccessBlockConfiguration"),
        ("get_bucket_encryption", "ServerSideEncryptionConfigurationNotFoundError"),
    ],
)
def test_bucket_aws_error_becomes_fail(cfg, method, code):
    s3 = make_s3()
    getattr(s3, method).side_effect = client_error(code)
    result = aws_check._guard("C2", "Bucket", aws_check.check_bucket, s3, cfg)
    assert not result.passed
    assert code in result.detail


# --------------------------------------------------------------------------- C3


@pytest.mark.parametrize(
    "service", ["sagemaker.amazonaws.com", ["ec2.amazonaws.com", "sagemaker.amazonaws.com"]]
)
def test_sagemaker_role_pass(cfg, service):
    iam = make_iam(trust(service))
    assert aws_check.check_sagemaker_role(iam, cfg).passed
    iam.get_role.assert_called_once_with(RoleName="qlora-sagemaker-execution")


def test_sagemaker_role_wrong_service(cfg):
    assert not aws_check.check_sagemaker_role(make_iam(trust("ec2.amazonaws.com")), cfg).passed


def test_sagemaker_role_deny_statement_does_not_count(cfg):
    doc = trust("sagemaker.amazonaws.com")
    doc["Statement"][0]["Effect"] = "Deny"
    assert not aws_check.check_sagemaker_role(make_iam(doc), cfg).passed


def test_sagemaker_role_missing(cfg):
    iam = make_iam(trust("sagemaker.amazonaws.com"))
    iam.get_role.side_effect = client_error("NoSuchEntity")
    result = aws_check._guard("C3", "SageMaker role", aws_check.check_sagemaker_role, iam, cfg)
    assert not result.passed
    assert "NoSuchEntity" in result.detail


# --------------------------------------------------------------------------- C4


def test_bedrock_role_pass(cfg):
    iam = make_iam(trust("bedrock.amazonaws.com"))
    assert aws_check.check_bedrock_role(iam, cfg).passed
    iam.get_role_policy.assert_called_once_with(
        RoleName="qlora-bedrock-customization", PolicyName="qlora-bedrock-s3"
    )


def test_bedrock_role_wrong_trust(cfg):
    assert not aws_check.check_bedrock_role(make_iam(trust("sagemaker.amazonaws.com")), cfg).passed


def test_bedrock_role_no_write_on_bedrock_prefix(cfg):
    policy = copy.deepcopy(BEDROCK_POLICY)
    policy["Statement"][2]["Action"] = ["s3:GetObject"]
    result = aws_check.check_bedrock_role(make_iam(trust("bedrock.amazonaws.com"), policy), cfg)
    assert not result.passed
    assert "write on bedrock/*" in result.detail


def test_bedrock_role_no_read_on_data_prefix(cfg):
    policy = copy.deepcopy(BEDROCK_POLICY)
    del policy["Statement"][1]
    result = aws_check.check_bedrock_role(make_iam(trust("bedrock.amazonaws.com"), policy), cfg)
    assert not result.passed
    assert "read on data/*" in result.detail


def test_bedrock_role_policy_for_other_bucket(cfg):
    policy = copy.deepcopy(BEDROCK_POLICY)
    for statement in policy["Statement"]:
        statement["Resource"] = statement["Resource"].replace(BUCKET, "some-other-bucket")
    assert not aws_check.check_bedrock_role(
        make_iam(trust("bedrock.amazonaws.com"), policy), cfg
    ).passed


def test_bedrock_role_inline_policy_missing(cfg):
    iam = make_iam(trust("bedrock.amazonaws.com"))
    iam.get_role_policy.side_effect = client_error("NoSuchEntity")
    result = aws_check._guard("C4", "Bedrock role", aws_check.check_bedrock_role, iam, cfg)
    assert not result.passed


# --------------------------------------------------------------------------- C5


def test_gpu_quota_pass_across_pages(cfg):
    quotas = make_quotas(GOOD_QUOTA_PAGES)
    result = aws_check.check_gpu_quota(quotas, cfg)
    assert result.passed
    quotas.get_paginator.assert_called_once_with("list_service_quotas")
    quotas.get_paginator.return_value.paginate.assert_called_once_with(ServiceCode="sagemaker")


def test_gpu_quota_zero(cfg):
    pages = [
        [("ml.g5.xlarge for training job usage", 1.0), ("ml.g6.xlarge for training job usage", 0.0)]
    ]
    result = aws_check.check_gpu_quota(make_quotas(pages), cfg)
    assert not result.passed
    assert "ml.g6.xlarge" in result.detail


def test_gpu_quota_missing(cfg):
    pages = [[("ml.g5.xlarge for training job usage", 1.0)]]
    result = aws_check.check_gpu_quota(make_quotas(pages), cfg)
    assert not result.passed
    assert "ml.g6.xlarge quota not found" in result.detail


def test_gpu_quota_spot_quota_is_not_the_training_quota(cfg):
    pages = [
        [
            ("ml.g5.xlarge for spot training job usage", 1.0),
            ("ml.g6.xlarge for spot training job usage", 1.0),
        ]
    ]
    assert not aws_check.check_gpu_quota(make_quotas(pages), cfg).passed


# --------------------------------------------------------------------------- C6


def test_nova_models_pass(cfg):
    bedrock = make_bedrock()
    assert aws_check.check_nova_models(bedrock, cfg).passed
    bedrock.list_foundation_models.assert_called_once_with(byCustomizationType="FINE_TUNING")


def test_nova_models_one_missing(cfg):
    result = aws_check.check_nova_models(make_bedrock(("amazon.nova-lite-v1:0:300k",)), cfg)
    assert not result.passed
    assert "amazon.nova-pro-v1:0:300k" in result.detail


# --------------------------------------------------------------------------- C8


def test_nova_inference_pass(cfg):
    bedrock = make_bedrock()
    result = aws_check.check_nova_inference(bedrock, cfg)
    assert result.passed and result.check_id == "C8"
    assert result.detail == (
        "amazon.nova-pro-v1:0 ACTIVE, on demand; amazon.nova-lite-v1:0 ACTIVE, on demand"
    )
    assert [call.kwargs for call in bedrock.get_foundation_model.call_args_list] == [
        {"modelIdentifier": model_id} for model_id in INFERENCE_MODELS
    ]


def test_nova_inference_prints_an_end_of_life_date(cfg):
    from datetime import UTC, datetime

    bedrock = make_bedrock()
    bedrock.get_foundation_model.side_effect = lambda modelIdentifier: foundation_model(
        modelIdentifier, end=datetime(2027, 6, 30, tzinfo=UTC)
    )
    result = aws_check.check_nova_inference(bedrock, cfg)
    assert result.passed and "ACTIVE, on demand, end of life 2027-06-30" in result.detail


def test_nova_inference_a_legacy_model_fails(cfg):
    bedrock = make_bedrock()
    bedrock.get_foundation_model.side_effect = lambda modelIdentifier: foundation_model(
        modelIdentifier, status="LEGACY" if "pro" in modelIdentifier else "ACTIVE"
    )
    result = aws_check.check_nova_inference(bedrock, cfg)
    assert not result.passed
    assert result.detail == "amazon.nova-pro-v1:0 is LEGACY"


def test_nova_inference_without_on_demand_support_fails(cfg):
    bedrock = make_bedrock()
    bedrock.get_foundation_model.side_effect = lambda modelIdentifier: foundation_model(
        modelIdentifier, types=("PROVISIONED",)
    )
    result = aws_check.check_nova_inference(bedrock, cfg)
    assert not result.passed
    assert "amazon.nova-pro-v1:0 cannot be called on demand" in result.detail
    assert "amazon.nova-lite-v1:0 cannot be called on demand" in result.detail


def test_nova_inference_model_not_found_becomes_fail(cfg):
    bedrock = make_bedrock()
    bedrock.get_foundation_model.side_effect = client_error("ResourceNotFoundException")
    result = aws_check._guard(
        "C8", "Nova inference", aws_check.check_nova_inference, bedrock, cfg
    )
    assert not result.passed and result.detail.startswith("ResourceNotFoundException")


# --------------------------------------------------------------------------- C7


def test_budget_pass_and_address_not_printed(cfg):
    budgets = make_budgets()
    result = aws_check.check_budget(budgets, cfg)
    assert result.passed
    assert SUBSCRIBER_ADDRESS not in result.detail
    sent = budgets.describe_subscribers_for_notification.call_args.kwargs["Notification"]
    assert None not in sent.values()


def test_budget_cap_differs(cfg):
    result = aws_check.check_budget(make_budgets(amount="500.0"), cfg)
    assert not result.passed
    assert "500" in result.detail


def test_budget_no_notifications(cfg):
    assert not aws_check.check_budget(make_budgets(thresholds=()), cfg).passed


def test_budget_notification_without_subscriber(cfg):
    assert not aws_check.check_budget(make_budgets(subscribers=0), cfg).passed


def test_budget_missing(cfg):
    budgets = make_budgets()
    budgets.describe_budget.side_effect = client_error("NotFoundException")
    result = aws_check._guard("C7", "Budget", aws_check.check_budget, budgets, cfg)
    assert not result.passed
    assert "NotFoundException" in result.detail


# --------------------------------------------------------------------------- guard and main


class FakeSession:
    """Hands out the stub client for each service name."""

    def __init__(self, clients):
        self.clients = clients

    def client(self, service, config=None):
        return self.clients[service]


def good_clients() -> dict:
    iam = MagicMock()
    iam.get_role.side_effect = lambda RoleName: {
        "Role": {
            "AssumeRolePolicyDocument": trust(
                "bedrock.amazonaws.com" if "bedrock" in RoleName else "sagemaker.amazonaws.com"
            )
        }
    }
    iam.get_role_policy.return_value = {"PolicyDocument": BEDROCK_POLICY}
    return {
        "sts": make_sts(),
        "s3": make_s3(),
        "iam": iam,
        "service-quotas": make_quotas(GOOD_QUOTA_PAGES),
        "bedrock": make_bedrock(),
        "budgets": make_budgets(),
    }


@pytest.fixture
def patch_session(monkeypatch):
    def apply(clients):
        seen = {}

        def factory(profile_name=None, region_name=None):
            seen.update(profile=profile_name, region=region_name)
            return FakeSession(clients)

        monkeypatch.setattr(aws_check.boto3, "Session", factory)
        return seen

    return apply


def test_main_all_pass(patch_session, capsys):
    seen = patch_session(good_clients())
    assert aws_check.main([]) == 0
    out = capsys.readouterr().out
    assert out.count("PASS") == 8
    assert "8/8 checks passed" in out
    assert SUBSCRIBER_ADDRESS not in out
    assert seen == {"profile": "qlora", "region": "us-east-1"}
    assert out.isascii()


def test_main_one_failure_still_runs_the_rest(patch_session, capsys):
    clients = good_clients()
    clients["s3"].get_bucket_location.side_effect = client_error("NoSuchBucket")
    patch_session(clients)
    assert aws_check.main([]) == 1
    out = capsys.readouterr().out
    assert "FAIL  C2" in out
    assert out.count("PASS") == 7
    assert "7/8 checks passed" in out


def test_main_profile_override(patch_session):
    seen = patch_session(good_clients())
    assert aws_check.main(["--profile", "other"]) == 0
    assert seen["profile"] == "other"


@pytest.mark.parametrize(
    "error", [UnauthorizedSSOTokenError(), client_error("ExpiredToken", "GetCallerIdentity")]
)
def test_main_expired_sign_in(patch_session, capsys, error):
    clients = good_clients()
    clients["sts"].get_caller_identity.side_effect = error
    patch_session(clients)
    assert aws_check.main([]) == 2
    out = capsys.readouterr().out
    assert "aws sso login --profile qlora" in out
    assert "Traceback" not in out
    assert "PASS" not in out


def test_main_bad_config_path(tmp_path, capsys):
    assert aws_check.main(["--config", str(tmp_path / "absent.yaml")]) == 2
    assert "CONFIG ERROR" in capsys.readouterr().out


def test_checks_only_call_read_operations():
    """The script must stay read-only: every stubbed call is a get, list, or describe."""
    clients = good_clients()
    cfg = aws_check.load_aws_config()
    results = aws_check.run_checks(FakeSession(clients), cfg)
    assert all(result.passed for result in results)
    for client in clients.values():
        for call in client.method_calls:
            method = call[0].split(".")[0]
            assert method.startswith(("get_", "list_", "describe_")), method


# --------------------------------------------------------------------------- how it is run

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_runs_as_a_module(tmp_path):
    """The supported form: python -m scripts.aws_check. A missing config needs no network."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.aws_check", "--config", str(tmp_path / "absent.yaml")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "CONFIG ERROR" in result.stdout
    assert "Traceback" not in result.stderr


def test_running_by_file_path_gives_a_hint_not_a_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "aws_check.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "python -m scripts.aws_check" in result.stderr
    assert "Traceback" not in result.stderr


def test_config_path_must_be_a_yaml_file(tmp_path, capsys):
    assert aws_check.main(["--config", str(tmp_path / "aws.yml")]) == 2
    assert "must end in .yaml" in capsys.readouterr().out


# --------------------------------------------------------------------------- live


@pytest.mark.network
def test_live_account_passes_every_check():
    """Needs a valid sign-in: aws sso login --profile qlora. Read-only and not billed."""
    assert aws_check.main([]) == 0
