# Amazon Nova on Bedrock

**Last verified**: 2026-11-16 against boto3 1.44.3 and the AWS documentation. **Not yet
verified against the live account**: the sign-in had expired, so the first real call is the
owner's trial run of req_004 v4.01.  
**Sources**: `reqspecs/reqs/req_004_frontier_tuning/req_004_01_spec.md` sections 2.2 to 2.4,
`qlora_finetuning/frontier/bedrock.py`, and the pages listed at the end.

Working code: `qlora_finetuning/frontier/bedrock.py` (the only module that calls Bedrock).
General boto3 notes are in `boto3.md`.

## Model ids

| Use | Nova Pro | Nova Lite |
|---|---|---|
| Inference, in us-east-1 (what the project uses) | `amazon.nova-pro-v1:0` | `amazon.nova-lite-v1:0` |
| Inference through the cross-region profile (not used) | `us.amazon.nova-pro-v1:0` | `us.amazon.nova-lite-v1:0` |
| Customization (fine-tuning) | `amazon.nova-pro-v1:0:300k` | `amazon.nova-lite-v1:0:300k` |
| A tuned model, on demand | The ARN of its custom model deployment | same |

The inference ids are in `config/frontier.yaml`; the customization ids in `config/aws.yaml`.

## The Converse call

```python
response = client.converse(
    modelId="amazon.nova-lite-v1:0",
    system=[{"text": system}],
    messages=[{"role": "user", "content": [{"text": user}]}],
    inferenceConfig={"maxTokens": 192, "temperature": 0},
    additionalModelRequestFields={"inferenceConfig": {"topK": 1}},
)
response["output"]["message"]["content"]   # [{"text": "..."}], possibly other block kinds
response["stopReason"]                     # see below
response["usage"]                          # {"inputTokens", "outputTokens", "totalTokens", ...}
response["metrics"]["latencyMs"]           # the service's own latency
```

- `topK` is not part of `inferenceConfig` in the Converse definition. For Nova it goes in
  `additionalModelRequestFields`.
- `stopReason` is one of `end_turn`, `tool_use`, `max_tokens`, `stop_sequence`,
  `guardrail_intervened`, `content_filtered`, `malformed_model_output`, `malformed_tool_use`,
  `model_context_window_exceeded`.
- `usage` may also hold `cacheReadInputTokens`, `cacheWriteInputTokens`, and `cacheDetails`.

## Decoding

| Parameter | Range | Default |
|---|---|---|
| `temperature` | 0.00001 to 1 | 0.7 |
| `topP` | 0 to 1 | 0.9 |
| `topK` | 0 to 128 | not used |
| `maxTokens` | up to 5K (request schema) or 10K (model cards): the pages disagree | dynamic |

AWS recommends `temperature=0` for greedy decoding, which contradicts the stated minimum.
The project sends 0 with `topK` 1; if 0 is refused, 0.00001 gives the same reply because
`topK` 1 leaves a single token. AWS also says output "is not deterministic".

Reasoning is off unless a `reasoningConfig` is sent. Whether the two v1 models accept one is
not confirmed; the project sends none.

## Caching

- With no cache point in the request, Nova may still reuse a prompt prefix by itself. That
  reuse "isn't reported in the cache usage fields and isn't billed as cache reads".
- With explicit caching, `inputTokens` leaves the cached tokens out: total input is
  `inputTokens + cacheReadInputTokens + cacheWriteInputTokens`. A cache point needs at least
  1K tokens. The project sends none, and treats a non-zero cache count as a failure.

## Errors

| Error | HTTP | Retried by botocore's `standard` mode |
|---|---|---|
| `ThrottlingException` | 429 | Yes |
| `ModelNotReadyException` | 429 | Yes (the SDK retries it up to 5 times by itself) |
| `ServiceUnavailableException` | 503 | Yes |
| `InternalServerException` | 500 | Yes |
| `ModelTimeoutException` | 408 | No guidance from AWS |
| `ModelErrorException` | 424 | No guidance from AWS |
| `ValidationException` | 400 | No |
| `AccessDeniedException` | 403 | No |
| `ResourceNotFoundException` | 404 | No |

In `botocore.config.Config(retries=...)`, `total_max_attempts` counts the first try and
`max_attempts` counts retries only. Whether a failed request is billed is not confirmed.

## Quotas (us-east-1, documented defaults; an account's own values can be lower)

| Model | Requests a minute | Tokens a minute |
|---|---:|---:|
| Nova Pro, plain id | 250 | 1,000,000 |
| Nova Lite, plain id | 2,000 | 4,000,000 |
| Nova Pro, cross-region profile | 500 | 2,000,000 |
| Nova Lite, cross-region profile | 4,000 | 8,000,000 |
| Custom Nova Pro deployment | 200 | 800,000 |
| Custom Nova Lite deployment | 2,000 | 4,000,000 |

Tokens a minute counts input plus output. One call at a time is about 40 requests a minute.

**This account's own limits are a tenth of those for requests** (Service Quotas, read
2026-11-16; none is adjustable):

| Model | Requests a minute | Tokens a minute |
|---|---:|---:|
| Nova Pro, plain id | 25 | 1,000,000 |
| Nova Lite, plain id | 200 | 4,000,000 |
| Custom Nova Pro deployment | 20 | 800,000 |
| Custom Nova Lite deployment | 200 | 4,000,000 |

Untuned Nova Pro answers in about 2 seconds, so a run goes at the limit and is sustained by
retries; with `standard` retries and 8 attempts one run was throttled out at 182 of 200.
`config/frontier.yaml` now uses `adaptive` mode with 12 attempts. Read the limits with
`aws service-quotas list-service-quotas --service-code bedrock`.

## Service tiers and prices

Requests use the standard tier unless `serviceTier` is sent. Nova Pro also has priority and
flex tiers at other prices; Nova Lite has the standard tier only. The project never sends
`serviceTier`. Prices are in `config/pricing.yaml` with the date of the price list; a tuned
model's tokens cost the same as its base model's.

## Lifecycle

Both models are Active (checked on the model cards, 2026-11-16). Their guaranteed date ("EOL
no sooner than Dec 05, 2025") has passed; what remains is at least six months' notice. Once a
model is Legacy, no new fine-tuning job can be created on it. `aws_check` C8 reads the
status from `bedrock:GetFoundationModel`.

## Fine-tuning limits (for req_004 v4.02 and v4.03)

- 8 to 20,000 samples per model; 32k tokens of context per sample; training plus validation
  records at most 20,000. File size limits are not stated for Nova (other Bedrock models:
  1 GB training, 100 MB validation); the project's files are 10.7 MB and 1.1 MB.
- Format: JSONL, one conversation per line: `schemaVersion` (`bedrock-conversation-2024` in
  every AWS example; the guide says it "can be any string value"), an optional `system`
  list of `{"text"}`, and `messages` with `role` and a `content` list of `{"text"}`. Roles
  are `user` and `assistant`, alternating, first `user`, last `assistant`. No empty text.
- **Reserved words.** The guide: `User:`, `Bot:`, `Assistant:`, `System:`, `<image>`,
  `<video>`, `[EOS]` anywhere in a user or system prompt make the job "fail due to data
  issues". AWS's validation script (`nova_ft_dataset_validator.py`, read 2026-11-16, not
  run) also refuses `Thought:` and `<unk>`, compares without regard to case, and checks
  every text, the assistant's too. The project checks the stricter reading
  (`RESERVED_WORDS` in `qlora_finetuning/frontier/export.py`); fit and dev hold none.
- File names: "only alphanumeric characters, underscores, hyphens, slashes, and dots".
- AWS advises "the same system prompt during both training and inference" and to
  "deduplicate"; the project keeps repeated records so both paths train on the same data.
- A job writes `step_wise_training_metrics.csv` and, for some jobs, `validation_metrics.csv`
  to its output location. Neither holds a billed token count. Whether validation tokens
  are billed is not confirmed.
- Bedrock has a free `CountTokens` call; whether it supports Nova Pro and Lite is not
  confirmed (the page refers to each model's card).
- Staging in S3 (boto3 1.44.3): `put_object(Bucket, Key, Body, ChecksumSHA256=<base64 of the
  digest>)` makes S3 refuse a damaged upload; `head_object(Bucket, Key,
  ChecksumMode="ENABLED")` returns `ContentLength` and `ChecksumSHA256`; a missing key is a
  `ClientError` with code `404` (or `NoSuchKey`). Working code:
  `qlora_finetuning/frontier/staging.py`. Not yet tried against the live bucket.
- Hyperparameters: `epochCount` 1 to 5 (default 2), `learningRate` 1e-6 to 1e-4 (default
  1e-5), `learningRateWarmupSteps` 0 to 100 (default 10). There is no batch size.
- A tuned model is served with `CreateCustomModelDeployment` (us-east-1 only); the
  deployment's ARN is the `modelId` for Converse.

## Fine-tuning jobs and deployments (req_004 v4.03)

From the definitions in boto3 1.44.3 and the AWS pages below. **Not yet tried against the live
account**: the first real job is Nova Lite's.

```python
bedrock.create_model_customization_job(
    jobName=name, customModelName=name,          # 1 to 63: letters, digits, hyphens
    roleArn=role_arn, clientRequestToken=token,  # the same token: the same job
    baseModelIdentifier="amazon.nova-lite-v1:0:300k", customizationType="FINE_TUNING",
    trainingDataConfig={"s3Uri": training_uri},
    validationDataConfig={"validators": [{"s3Uri": validation_uri}]},
    outputDataConfig={"s3Uri": output_uri},
    hyperParameters={"epochCount": "2", "learningRate": "0.00001",
                     "learningRateWarmupSteps": "10"},   # text to text
    jobTags=[{"key": "project", "value": "qlora-finetuning"}], customModelTags=[...],
)                                                # -> {"jobArn": ...}
bedrock.get_model_customization_job(jobIdentifier=arn_or_name)
```

- Job `status`: `InProgress`, `Completed`, `Failed`, `Stopping`, `Stopped`. `statusDetails` has
  `validationDetails`, `dataProcessingDetails`, and `trainingDetails`, each with a `status`; the
  one that says `Failed` is where a job failed. `failureMessage` says why.
- A completed job gives `outputModelArn`, `trainingMetrics.trainingLoss`, and
  `validationMetrics[].validationLoss`. **It gives no token count and no amount.**
- Hyperparameter defaults for Nova Pro and Lite: `epochCount` 2, `learningRate` 1.00E-5 (at
  most 1.00E-4), `learningRateWarmupSteps` 10 (at most 100). AWS recommends both "starting
  with the default hyperparameters" and warmup steps of dataset size over 160 (Lite) or 320
  (Pro), which for 4,884 records is 31 and 15. The project uses the defaults.
- "You can't resume a stopped job. Amazon Bedrock charges for the tokens that it used to train
  the model before you stopped the job", and it makes no model.
- "The customization job can take several hours."
- On-demand deployment: `create_custom_model_deployment(modelDeploymentName, modelArn,
  clientRequestToken, tags)` returns `customModelDeploymentArn`; `get_custom_model_deployment`
  gives `status` `Creating`, `Active`, or `Failed`. The ARN of an Active deployment is the
  `modelId` for Converse. Nova Pro and Lite: us-east-1 only; the model must have been
  customized on or after 7/16/2025. "You only pay for what you use."
- Deleting: `delete_custom_model_deployment(customModelDeploymentIdentifier=)`, then
  `delete_custom_model(modelIdentifier=)`. Storage is billed per tuned model per month from the
  moment it exists, deployed or not.
- `list_provisioned_model_throughputs()` lists Provisioned Throughput. The project never
  creates any and refuses to launch or deploy while the account has some.
- Not confirmed: whether a job that fails by itself is billed; how many jobs may run at once;
  whether an idle deployment costs anything; what `get_model_customization_job` answers for a
  name that does not exist (`ResourceNotFoundException` is declared; the code also accepts
  `ValidationException`).

Working code: `qlora_finetuning/frontier/tuning.py`. An in-memory stand-in for these calls:
`tests/frontier/tuning_helpers.py` (`FakeBedrock`).

## Asking a tuned model (req_004 v4.04)

```python
from qlora_finetuning.frontier.predictor import build_tuned_predictor

predictor = build_tuned_predictor("nova-lite", context=context)  # refuses unless Active
predictor.model_id          # the deployment's ARN: the modelId of every call
predictor.usage_model_id    # "amazon.nova-lite-v1:0": what the ledger records it under
```

- The Converse request is the untuned one with the deployment's ARN as `modelId`. Nothing
  else changes.
- `config/pricing.yaml` prices a tuned model at its base model's token prices (the price list
  carries custom-model rows with the same amounts). Usage is therefore recorded under the
  base id; an ARN in the ledger would have no price and would count as nothing in the cap check.
- Before a run, `tuned_attempt()` (`frontier/tuned.py`) checks the tuning record and the gate
  record, and `deployment_status()` (`frontier/bedrock.py`) asks Bedrock once: one
  `get_custom_model_deployment`, a free read.
- Not verified until the first tuned run: whether the first call to an idle deployment is slow
  or refused; which `ModelId` CloudWatch lists a deployment's calls under; that the bill
  charges a tuned model at the base rate on this account.

## Testing without AWS

`BedrockProvider` takes the client as an argument. Tests pass a `MagicMock` whose `converse`
returns a response dict or raises a `ClientError`; see `tests/frontier/test_bedrock.py`. A
test of anything above the provider uses a stub provider (`tests/frontier/conftest.py`).

## Documentation

- Converse: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
- Nova request schema: https://docs.aws.amazon.com/nova/latest/userguide/complete-request-schema.html
- Nova with Converse: https://docs.aws.amazon.com/nova/latest/userguide/using-converse-api.html
- Structured output and greedy decoding: https://docs.aws.amazon.com/nova/latest/userguide/prompting-structured-output.html
- Prompt caching: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
- Error codes: https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html
- Quotas: https://docs.aws.amazon.com/general/latest/gr/bedrock.html
- Service tiers: https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
- Model lifecycle: https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle-legacy.html
- Model cards: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-pro.html and https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-lite.html
- Fine-tuning data: https://docs.aws.amazon.com/nova/latest/userguide/fine-tune-prepare-data-understanding.html
- Hyperparameters: https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models-hp.html
- On-demand deployment of a custom model: https://docs.aws.amazon.com/bedrock/latest/userguide/deploy-custom-model-on-demand.html
- Preparing data, all models: https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-prepare.html
- Job results: https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-analyze.html
- AWS's dataset validation script: https://github.com/aws-samples/amazon-nova-samples/tree/main/customization/bedrock-finetuning/understanding/dataset_validation
- Counting tokens: https://docs.aws.amazon.com/bedrock/latest/userguide/count-tokens.html
- Submitting a job: https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-submit.html
- Monitoring a job: https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-monitor.html
- Stopping a job: https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-stop.html
- `CreateModelCustomizationJob`: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_CreateModelCustomizationJob.html
