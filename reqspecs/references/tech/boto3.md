# boto3

**Last verified**: 2026-11-14 against boto3 1.44.3 and the live account  
**Sources**: `reqspecs/PRPs/prp_000_aws_foundation_v5.md`, `scripts/aws_check.py`, and the
documentation pages listed at the end.

The Bedrock Converse call, Nova's parameters, and its errors are in `bedrock_nova.md`.

Pinned in `requirements.txt` as `boto3==1.44.3`. Working code that uses everything below:
`scripts/aws_check.py`.

## Session and clients

```python
session = boto3.Session(profile_name="qlora", region_name="us-east-1")
client = session.client("s3", config=Config(retries={"max_attempts": 3, "mode": "standard"},
                                            connect_timeout=10, read_timeout=30))
```

- Profile and region come from `config/aws.yaml`, never from code.
- `iam` and `budgets` are global services; a client created in us-east-1 works for both.
- Creating a session or client makes no network call. `ProfileNotFound` is raised at session
  creation; credential errors are raised on the first request.

## Response shapes (read-only calls used so far)

```text
sts.get_caller_identity()                  -> {"Account": "123456789012", ...}
s3.get_bucket_location(Bucket)             -> {"LocationConstraint": None}      # None means us-east-1
s3.get_public_access_block(Bucket)         -> {"PublicAccessBlockConfiguration": {BlockPublicAcls, IgnorePublicAcls, BlockPublicPolicy, RestrictPublicBuckets: bool}}
s3.get_bucket_encryption(Bucket)           -> {"ServerSideEncryptionConfiguration": {"Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]}}
iam.get_role(RoleName)                     -> {"Role": {"AssumeRolePolicyDocument": {...already a dict...}}}
iam.get_role_policy(RoleName, PolicyName)  -> {"PolicyDocument": {...already a dict...}}
service-quotas paginator list_service_quotas(ServiceCode="sagemaker")
                                           -> pages of {"Quotas": [{"QuotaName": "ml.g5.xlarge for training job usage", "Value": 1.0}, ...]}
bedrock.list_foundation_models(byCustomizationType="FINE_TUNING")
                                           -> {"modelSummaries": [{"modelId": "amazon.nova-pro-v1:0:300k"}, ...]}
budgets.describe_budget(AccountId, BudgetName)
                                           -> {"Budget": {"BudgetLimit": {"Amount": "300.0", "Unit": "USD"}, "TimeUnit": "ANNUALLY"}}
budgets.describe_notifications_for_budget(AccountId, BudgetName)
                                           -> {"Notifications": [{"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN", "Threshold": 100.0}, ...]}
budgets.describe_subscribers_for_notification(AccountId, BudgetName, Notification)
                                           -> {"Subscribers": [{"SubscriptionType": "EMAIL", "Address": "..."}]}
```

## Quirks

- **Null region.** `get_bucket_location` returns `None` for us-east-1.
- **Pagination.** SageMaker has thousands of quotas. Use
  `client.get_paginator("list_service_quotas")`; a single call returns one page.
- **Quota names.** Match the full name, `"<instance type> for training job usage"`. The spot,
  endpoint, and notebook quotas share the same prefix.
- **String or list.** In IAM policy documents `Action`, `Resource`, and `Principal.Service` are
  a string for one value and a list for several.
- **Budget amounts are text.** Compare with `Decimal`.
- **Notifications round-trip.** Pass each notification dict from
  `describe_notifications_for_budget` straight back to
  `describe_subscribers_for_notification`. Do not add keys whose value is `None`.
- **Subscriber addresses are personal data.** Count them; never print or store them.
- **Cost Explorer calls are billed** per request, so read-only checks avoid that service.

## Errors

| Situation | What is raised |
|---|---|
| Profile missing | `botocore.exceptions.ProfileNotFound` at session creation |
| SSO session expired or absent | `UnauthorizedSSOTokenError`, `SSOTokenLoadError`, `TokenRetrievalError`, `NoCredentialsError`, or `CredentialRetrievalError` on the first request |
| Expired temporary credentials | `ClientError` with code `ExpiredToken`, `ExpiredTokenException`, `RequestExpired`, `InvalidClientTokenId`, or `UnrecognizedClientException` |
| Resource missing | `ClientError` with a service code: `NoSuchBucket`, `NoSuchEntity`, `NotFoundException`, `NoSuchPublicAccessBlockConfiguration`, `ServerSideEncryptionConfigurationNotFoundError` |
| Network or endpoint failure | a subclass of `BotoCoreError` |

The error code is at `exc.response["Error"]["Code"]`.

## Testing without AWS

Write each function to take the client as an argument. In tests pass a
`unittest.mock.MagicMock` with return values set, and raise errors with:

```python
ClientError({"Error": {"Code": "NoSuchEntity", "Message": "stubbed"}}, "GetRole")
```

A paginator stub: `client.get_paginator.return_value.paginate.return_value = [page1, page2]`.
No test creates a real client. See `tests/scripts/test_aws_check.py`.

## Documentation

- Budgets: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/budgets.html
- Service Quotas paginator: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/service-quotas/paginator/ListServiceQuotas.html
- Bedrock `list_foundation_models`: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/list_foundation_models.html
- Error handling: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/error-handling.html
