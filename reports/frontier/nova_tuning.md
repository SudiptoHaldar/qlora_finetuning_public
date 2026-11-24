# Nova tuning record

Written by `python -m scripts.nova_tune`. Do not edit by hand.

What was launched on Bedrock for Path B and what became of it: jobs, tuned models,
and on-demand deployments. Names, ARNs, numbers, and dates only. Dollar amounts are
estimates at list price until the owner has entered what the AWS bill says.

## Jobs

| Model | Attempt | Arm | Job status | Failed in | Created | Ended | Estimate | Billed |
|---|---:|---|---|---|---|---|---:|---:|
| nova-lite | 1 | B1-Lite | Completed | - | 2026-11-19T02:10:06Z | 2026-11-19T04:29:30Z | $12.822776 | $14.6 |
| nova-pro | 1 | B1-Pro | Completed | - | 2026-11-21T09:55:42Z | 2026-11-22T04:35:35Z | $51.291104 | $58.75 |

## Tuned models and deployments

| Model | Attempt | Training loss | Validation loss | Deployment | Deleted |
|---|---:|---:|---:|---|---|
| nova-lite | 1 | 0.0589 | 0.1035 | Active | - |
| nova-pro | 1 | 0.0411 | 0.0821 | Active | - |

## nova-lite, attempt 1

- Base model for the job: `amazon.nova-lite-v1:0:300k`
- Hyperparameters: `{"epochCount": "2", "learningRate": "0.00001", "learningRateWarmupSteps": "10"}`
- Export id: `c2d67fe65c7f`
- Training file: `s3://qlora-sagemaker-123456789012/data/bedrock/c2d67fe65c7f/finqa_fit.jsonl`
- Validation file: `s3://qlora-sagemaker-123456789012/data/bedrock/c2d67fe65c7f/finqa_dev.jsonl`
- Output: `s3://qlora-sagemaker-123456789012/bedrock/c2d67fe65c7f/nova-lite-a1/`
- Estimated training tokens: 6,411,388
- Job: `arn:aws:bedrock:us-east-1:123456789012:model-customization-job/amazon.nova-lite-v1:0:300k/9u5awtwqxk7j` (name `qlora-finqa-nova-lite-c2d67fe65c7f-a1`)
- Tuned model: `arn:aws:bedrock:us-east-1:123456789012:custom-model/amazon.nova-lite-v1:0:300k/gmwz1fhs0vbt`
- Deployment: `arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/308jz60uh4a5`
- Launch run: `20261119T021002Z-train-qlora-finqa-nova-lite-c2d67fe65c7f-a1-ca6d`
- Training loss over 152 steps: first 0.4394, last 0.0589, lowest 0.0426 at step 145
- Validation loss over 2 steps: first 0.1083, last 0.1035, lowest 0.1035 at step 152

## nova-pro, attempt 1

- Base model for the job: `amazon.nova-pro-v1:0:300k`
- Hyperparameters: `{"epochCount": "2", "learningRate": "0.00001", "learningRateWarmupSteps": "10"}`
- Export id: `c2d67fe65c7f`
- Training file: `s3://qlora-sagemaker-123456789012/data/bedrock/c2d67fe65c7f/finqa_fit.jsonl`
- Validation file: `s3://qlora-sagemaker-123456789012/data/bedrock/c2d67fe65c7f/finqa_dev.jsonl`
- Output: `s3://qlora-sagemaker-123456789012/bedrock/c2d67fe65c7f/nova-pro-a1/`
- Estimated training tokens: 6,411,388
- Job: `arn:aws:bedrock:us-east-1:123456789012:model-customization-job/amazon.nova-pro-v1:0:300k/ibzwgwgvqxue` (name `qlora-finqa-nova-pro-c2d67fe65c7f-a1`)
- Tuned model: `arn:aws:bedrock:us-east-1:123456789012:custom-model/amazon.nova-pro-v1:0:300k/z86zmzl5wkz8`
- Deployment: `arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/62v0vii1ge23`
- Launch run: `20261121T095540Z-train-qlora-finqa-nova-pro-c2d67fe65c7f-a1-752d`
- Training loss over 304 steps: first 0.3580, last 0.0411, lowest 0.0204 at step 290
- Validation loss over 2 steps: first 0.0794, last 0.0821, lowest 0.0794 at step 152
