# Frontier Tuning

Spec Group      :   req_004_frontier_tuning  
Feature Name    :   frontier_tuning  
Version         :   1  
Status          :   ✅ Complete: code built 2026-11-16 (`prp_004_bedrock_nova_adapter_v1.md`); the owner ran the four baseline runs on 2026-11-16 and the baseline report on 2026-11-18 (section 11)  
Last updated    :   2026-11-18  

## 1. v4.01 — Bedrock Nova adapter and zero-shot baselines

Path B starts here. This version lets the evaluation runner of v2.03 ask Amazon Nova Pro and
Amazon Nova Lite for replies through Amazon Bedrock, and uses it to score both untuned
models on quick-dev and on dev: arms **B0-Pro** and **B0-Lite**.

It adds:

- a **`FrontierProvider`** interface and its Bedrock implementation (the Converse API);
- a **predictor** over that provider, registered in `scripts/evaluate.py` as `nova-pro` and
  `nova-lite`;
- **`config/frontier.yaml`** (model ids, decoding, client settings) and
  **`config/pricing.yaml`** (the Bedrock prices, with the date of the price list);
- an **availability check** in `scripts/aws_check.py` (C8);
- the four **baseline runs** (two models, two splits) and a committed **baseline report**: scores, tokens, cost at
  list price, and how Nova's token counts compare with the Qwen counts that every estimate
  has used so far.

It trains nothing, uploads nothing, and creates no AWS resource. Its only billed calls are
the inference calls of the baseline runs, estimated at under $1.10 in total (section 2.4).
With v3.01 (arm A0) it reaches milestone M4.

**Depends on**: v2.03, v1.03, v0.05 (all built). It also reads the reply limit of v1.04.

## 2. What exists today (verified 2026-11-16)

### 2.1 What earlier versions provide

| Item | State |
|---|---|
| `Predictor` protocol, `Prediction`, `PredictorError` | `evaluation/predictor.py`. A backend implements `predict(requests)` and five attributes; nothing of the loop |
| `run_evaluation` | Starts the run, writes the predictions file, and records the usage. When the predictor names a model and reports every token count, the ledger record carries the token totals with the predictor's `usage_source`. **So a backend writes nothing to the ledger itself** |
| `scripts/evaluate.py` | `PREDICTORS` table of `PredictorEntry(factory, kind, path, arm)`; a factory may be `"module:function"`, imported only when named. `--where` defaults to `local`. The run snapshots six config files (`RUN_CONFIGS`) |
| `config/tracking.yaml` | Arms `B0-Pro`, `B0-Lite`, path `B`, and place `bedrock` are already allowed |
| `config/aws.yaml` | Profile `qlora`, region us-east-1, account id. Its two model ids end in `:300k`: those are the ids for **customization**, not for inference |
| `config/token_budget.yaml` | `max_new_tokens: 192` for every arm |
| `boto3` | 1.44.3, pinned in `requirements.txt`, installed in CI |
| `qlora_finetuning/frontier/` | An empty package. `tests/frontier/` holds an `__init__.py` only |
| Token budget report (v1.04) | Prompt tokens, Qwen tokenizer: quick-dev 128,495 in 200 prompts; dev 317,882 in 499. Amazon publishes no Nova tokenizer |
| Guard tests | Nothing under `evaluation/` may import `frontier`; no module imports the training stack at module level |

### 2.2 The Converse API, from the installed boto3 (no network)

Read from the service definition that boto3 1.44.3 ships.

| Part | Shape |
|---|---|
| Request | `modelId` (required), `system: [{"text"}]`, `messages: [{"role", "content": [{"text"}]}]`, `inferenceConfig: {maxTokens, temperature, topP, stopSequences}`, `additionalModelRequestFields` (free form), `serviceTier`, `performanceConfig`, `guardrailConfig`, `toolConfig`, `outputConfig` |
| Response | `output.message.content` (a list of blocks: `text`, `reasoningContent`, and others), `stopReason`, `usage`, `metrics.latencyMs` |
| `stopReason` | `end_turn`, `tool_use`, `max_tokens`, `stop_sequence`, `guardrail_intervened`, `content_filtered`, `malformed_model_output`, `malformed_tool_use`, `model_context_window_exceeded` |
| `usage` | `inputTokens`, `outputTokens`, `totalTokens` always; `cacheReadInputTokens`, `cacheWriteInputTokens`, `cacheDetails` sometimes |
| Errors | `ThrottlingException` 429, `ModelNotReadyException` 429, `ModelTimeoutException` 408, `ServiceUnavailableException` 503, `InternalServerException` 500, `ModelErrorException` 424, `ValidationException` 400, `AccessDeniedException` 403, `ResourceNotFoundException` 404 |
| `bedrock` client | Has `get_foundation_model` (returns `inferenceTypesSupported` and `modelLifecycle.status`: `ACTIVE` or `LEGACY`) and `create_custom_model_deployment` (for v4.03) |

### 2.3 What AWS documents (read 2026-11-16)

| Topic | Finding | Source |
|---|---|---|
| Model ids | `amazon.nova-pro-v1:0` and `amazon.nova-lite-v1:0` can be called on demand in us-east-1 by the plain id. The cross-region profiles are `us.amazon.nova-pro-v1:0` and `us.amazon.nova-lite-v1:0`; from us-east-1 they may route a request to us-east-2 or us-west-2 | Bedrock model cards for Nova Pro and Nova Lite |
| Decoding | `temperature` 0.00001 to 1, default 0.7. `topP` 0 to 1, default 0.9. `topK` 0 to 128, passed as `additionalModelRequestFields={"inferenceConfig": {"topK": n}}`. For structured output AWS recommends "greedy decoding parameters. That is, `temperature=0`", which **contradicts the stated minimum**. The same page says output "is not deterministic" | Nova user guide: complete request schema; using the Converse API; structured output |
| Reply length | The request schema says at most 5K new tokens; the model cards say 10K. Either is far above 192 | same |
| Reasoning | Off unless a `reasoningConfig` is sent. Whether the two v1 models accept one at all is **not confirmed**: the pages disagree | Nova user guide: request schema; extended thinking |
| Caching | With no cache point in the request, Nova may still reuse a prompt prefix by itself; that reuse "isn't reported in the cache usage fields and isn't billed as cache reads". With explicit caching, `inputTokens` leaves the cached tokens out. A cache point needs at least 1K tokens | Bedrock user guide: prompt caching |
| Errors | Throttling, service-unavailable, and internal errors: retry with exponential backoff and jitter. The SDK's `standard` retry mode does that, 3 attempts by default. Validation, access, and not-found errors are not retried. Whether a failed request is billed is **not confirmed** | Bedrock troubleshooting: API error codes; AWS SDK retry behavior |
| Quotas, us-east-1, plain id | Nova Pro: 250 requests and 1,000,000 tokens a minute. Nova Lite: 2,000 and 4,000,000. Not adjustable. The cross-region profiles have twice that. An account's own values can be lower | AWS general reference: Bedrock quotas |
| Service tier | Requests use the standard tier unless `serviceTier` is sent. Nova Pro also has priority ($1.40 / $5.60) and flex ($0.40 / $1.60); Nova Lite has the standard tier only | Bedrock user guide: service tiers; model cards |
| Lifecycle | Both models are **Active**. Their guaranteed date ("EOL no sooner than Dec 05, 2025") has passed; what remains is at least six months' notice. Once a model is Legacy, **no new fine-tuning job can be created on it** | Model cards; Bedrock model lifecycle |
| Fine-tuning limits (resolves the open part of Q1) | Pro and Lite: 8 to 20,000 samples, 32k tokens of context per sample; training plus validation records at most 20,000. Hyperparameters: `epochCount` 1 to 5 (default 2), `learningRate` 1e-6 to 1e-4 (default 1e-5), `learningRateWarmupSteps` 0 to 100 (default 10). There is no batch size. File size limits are **not confirmed** | Nova user guide: preparing data; Bedrock user guide: custom model hyperparameters |
| Custom model on demand (for v4.03, v4.04) | `CreateCustomModelDeployment`; the deployment's ARN is then the `modelId` for Converse. Nova Pro and Lite, us-east-1 only. Priced "the same as base Nova inference". Deployment quotas: Pro 200 requests and 800,000 tokens a minute | Bedrock user guide: deploy a custom model for on-demand inference; pricing page |

Our numbers against those limits: fit 4,884 plus dev 499 is 5,383 records of at most 1,862
Qwen tokens, well inside 20,000 and 32k. A sequential run makes about 40 requests a minute,
well inside 250.

### 2.4 Prices and the cost of this version

From the public AWS price list for Bedrock in us-east-1, **published 2026-11-11** and
downloaded 2026-11-16. Nothing has changed since the plan was written.

| Per 1M tokens | Nova Pro | Nova Lite |
|---|---:|---:|
| Input | $0.80 | $0.06 |
| Output | $3.20 | $0.24 |
| Input and output of a custom model | $0.80 / $3.20 | $0.06 / $0.24 |
| Fine-tuning, per training token | $8.00 | $2.00 |
| Custom model storage, per model per month | $1.95 | $1.95 |

The price list has one set of prices per model: none separate for the cross-region profile.

| Run | Input tokens (Qwen count) | Cost if Nova counts 1.5 times as many tokens and every reply runs to 192 |
|---|---:|---:|
| B0-Pro, quick-dev | 128,495 | $0.28 |
| B0-Lite, quick-dev | 128,495 | $0.02 |
| B0-Pro, dev | 317,882 | $0.69 |
| B0-Lite, dev | 317,882 | $0.05 |

A run cannot cost more than its split allows: one call per example, a fixed prompt, and at
most 192 reply tokens. The two quick-dev runs with a five-example trial come to under $0.35,
and the two dev runs to at most $0.75: under $1.10 for the version.

### 2.5 What could not be checked

The AWS sign-in had expired, so nothing was asked of the live account. Three things are
therefore settled by the build, not by this spec:

1. that the account may call both plain model ids on demand (check C8, free);
2. that `temperature: 0` is accepted (the five-example trial, section 9);
3. how Nova's token counts compare with Qwen's (the baseline report).

## 3. Locked decisions

### The provider

| # | Decision | Reason |
|---|---|---|
| D1 | `FrontierProvider` is a protocol with **one method in this version**: `converse(model_id, system, user, max_tokens) -> ProviderReply`. The other steps of plan §4.7 (stage a file, start a job, poll, deploy) are added by the versions that build them | An interface declared before its second method exists is a guess |
| D2 | `BedrockProvider` takes a `bedrock-runtime` client as an argument. One function builds the real client from `config/aws.yaml` (profile, region) and `config/frontier.yaml` | Tests hand over a stub; no test creates a real client |
| D3 | The request is: `system=[{"text": system}]`, one user message `[{"text": user}]`, `inferenceConfig={"maxTokens": max_tokens, "temperature": 0}`, and `additionalModelRequestFields={"inferenceConfig": {"topK": 1}}`. The two texts are the canonical record's, **character for character** | v1.03: every arm reads the same words. `topK: 1` leaves only the most likely token, which is greedy decoding whatever the temperature; Path A decodes greedily too |
| D4 | Nothing else is sent: no `topP`, no stop sequence, no tool, no guardrail, no cache point, no `reasoningConfig`, no `serviceTier`, no `performanceConfig` | Each would be a difference from Path A or a different price. Unsent, the tier is standard and reasoning is off |
| D5 | If Bedrock refuses `temperature: 0`, the config value becomes `0.00001`, the documented minimum, and the report says so. With `topK: 1` the reply is the same | The documentation contradicts itself (section 2.3) |
| D6 | The model id is the **plain in-region id**, from config: `amazon.nova-pro-v1:0`, `amazon.nova-lite-v1:0` | The plan says us-east-1. A tuned model can only be deployed there, so B0 and B1 are then served the same way. The owner's decision, 2026-11-16 (section 8) |
| D7 | The reply text is the `text` blocks of `output.message.content`, joined in order, unchanged. Any other block is left out, and a `reasoningContent` block is counted in a log line | v2.03 D3: the parser sees the reply without what is specific to a model |
| D8 | Stop reason: `end_turn` and `stop_sequence` are `end`; `max_tokens` is `length`; every other value is `other`. Such a reply is still scored, even if empty | Those values are the service's answer to that prompt and would be the same on a retry. Stopping the run for one would make it impossible to finish (v2.03 D10) |
| D9 | Token counts are `usage.inputTokens` and `usage.outputTokens`, source `aws_api`. A response that reports cached tokens (`cacheReadInputTokens` or `cacheWriteInputTokens` above zero) is a failure | We send no cache point, so `inputTokens` is every input token. If that ever stops being true the counts are wrong, and a wrong count must not reach the ledger |
| D10 | Latency is our own clock around the call, including any retry and wait | It is what a caller experiences. Local and hosted latency are not comparable anyway (v2.03, section 7) |

### Failures

| # | Decision | Reason |
|---|---|---|
| D11 | Retrying is botocore's: `standard` mode, `max_attempts` from config (8). The adapter adds no loop of its own | Tested code that already backs off with jitter on exactly the errors AWS names. Eight attempts wait up to about two minutes |
| D12 | When botocore gives up, or the error is one it does not retry, the provider raises `PredictorError` with the error code and no request text. An expired sign-in says: run `aws sso login --profile qlora` | v2.03 D9: the run stops, keeps its lines, and is resumed |
| D13 | A response without a message, with a role other than `assistant`, or without whole-number token counts is a `PredictorError` | A broken response must not produce a score |
| D14 | Before any run exists, the factory asks `sts:GetCallerIdentity` and compares the account with `config/aws.yaml`. A failure is a config error: exit 2, no run, no ledger line | A run that fails on its first call for want of a sign-in would leave a failed run in the committed index every time |

### The predictor

| # | Decision | Reason |
|---|---|---|
| D15 | `FrontierPredictor(provider, model_id, name, settings)` implements `Predictor`. `usage_source` is `aws_api` | One class for both models, and for the tuned ones in v4.04 |
| D16 | `settings` holds everything that can change a reply: `api: converse`, `temperature`, `top_k`, `top_p: null` (not sent), `stop_sequences: []`, `service_tier: default`, `reasoning: not sent` | The resume check compares it (v2.03 D19) |
| D17 | `batch_size` is 1 and calls are sequential | v2.03: a batch is all or nothing, so with paid calls a failed batch is paid twice. 200 examples take a few minutes. Batch size is not part of the resume check, so v5.01 can raise it without invalidating anything |
| D18 | Predictors are registered as `nova-pro` (path B, arm B0-Pro) and `nova-lite` (arm B0-Lite), both as `"module:function"` | boto3 is loaded only when one of them is named |
| D19 | `PredictorEntry` gains two optional fields: `where` (used when `--where` is not given; `bedrock` for the Nova entries) and `configs` (extra config files for the run to snapshot: `frontier`, `aws`). `gold` is unchanged | Today the operator would have to remember `--where bedrock`, and the run would not hold the settings that decided its replies |
| D20 | The baselines are zero-shot: the system text and the user text, no examples, no wording for Nova alone | Plan §4.2 and working agreement 7 |

### Config, prices, and the report

| # | Decision | Reason |
|---|---|---|
| D21 | `config/frontier.yaml` holds the two models (id, arm), the decoding values, and the client settings. `aws_check` reads the model ids from it | One place for what Path B sends |
| D22 | `config/pricing.yaml` holds the Bedrock prices of section 2.4 per model id, as text per 1M tokens, with the price list's publication date and the day it was read. This version reads the two inference prices only | The roadmap asks that prices be recorded on the day. v4.02 and v5.02 read the same file |
| D23 | No dollar amount goes into the ledger. The report computes cost as tokens times price, with `Decimal` | v0.04: the ledger holds quantities |
| D24 | The baseline report is committed under `reports/frontier/` as Markdown and JSON: counts, tokens, and dollars only. No reply and no example text | Coding rule: never commit dataset text. v4.02 reads the token ratio from the JSON |
| D25 | The ratio of Nova to Qwen tokens is a run's input tokens over the prompt tokens of its split in the committed token budget report (quick-dev 128,495; dev 317,882) | No tokenizer needs loading, and the Qwen figure is already pinned |
| D26 | The baseline runs are scope `dev` | They are development evidence. The numbers of the comparison come from v5.01 |

## 4. Scope to build

### 4.1 Config

`config/frontier.yaml`:

```yaml
version: 1
verified_on: "2026-11-16"
models:
  nova-pro:
    model_id: "amazon.nova-pro-v1:0"
    untuned_arm: B0-Pro
  nova-lite:
    model_id: "amazon.nova-lite-v1:0"
    untuned_arm: B0-Lite
inference:
  temperature: 0        # 0.00001 if Bedrock refuses 0 (D5)
  top_k: 1
client:
  retry_mode: standard
  max_attempts: 8
  connect_timeout: 10
  read_timeout: 120
report_dir: reports/frontier
```

`config/pricing.yaml`: `version`, `price_list` (`offer: AmazonBedrock`, `region`,
`published: "2026-11-11"`, `read_on: "2026-11-16"`), and under `bedrock`, per model id:
`input_per_million`, `output_per_million`, `training_per_million`,
`storage_per_model_month`, each as quoted text.

### 4.2 Provider: `qlora_finetuning/frontier/provider.py` and `bedrock.py`

| Item | Behavior |
|---|---|
| `ProviderReply` | `text`, `input_tokens`, `output_tokens`, `stop_reason` (already mapped, D8), `latency_seconds` |
| `FrontierProvider` | The protocol of D1 |
| `load_frontier_config()` | Typed values of `config/frontier.yaml`; a missing key is a `ConfigError` naming it |
| `client_config(cfg)` | The botocore `Config` of D11 |
| `bedrock_runtime_client(aws_cfg, cfg)` | The real client. The only place that creates one |
| `check_identity(sts, aws_cfg)` | D14 |
| `BedrockProvider(client, cfg)` | `converse(...)`: builds the request of D3 and D4, reads the response by D7 to D10 and D13, and turns errors into `PredictorError` by D12 |

Only `bedrock.py` imports boto3 or botocore.

### 4.3 Predictor: `qlora_finetuning/frontier/predictor.py`

`FrontierPredictor` (D15 to D17), and the two factories the script names, `nova_pro(split,
context)` and `nova_lite(split, context)`: load both configs, check the identity, build the
client, and return the predictor.

```python
from qlora_finetuning.evaluation.runner import run_evaluation
from qlora_finetuning.frontier.bedrock import BedrockProvider
from qlora_finetuning.frontier.predictor import FrontierPredictor

predictor = FrontierPredictor(BedrockProvider(client, cfg), model_id="amazon.nova-lite-v1:0",
                              name="nova-lite", settings=cfg.settings)
summary = run_evaluation(predictor, "quick_dev", label="nova-lite quick_dev",
                         path="B", arm="B0-Lite", where="bedrock", scope="dev")
summary["usage"]        # {"input_tokens": ..., "output_tokens": ..., ...}, as Bedrock reported
```

### 4.4 `scripts/evaluate.py` and `scripts/aws_check.py`

- `PredictorEntry.where` and `.configs` (D19); the two Nova entries (D18).
- `aws_check` gains **C8, Nova inference**: for each model in `config/frontier.yaml`,
  `bedrock:GetFoundationModel` says it supports `ON_DEMAND` and its lifecycle status is
  `ACTIVE`. The detail line prints the status and any end-of-life date. Read-only and free.

### 4.5 Prices and the baseline report

`qlora_finetuning/frontier/pricing.py`: `load_pricing()` and
`inference_cost(model_id, input_tokens, output_tokens) -> Decimal`.

`qlora_finetuning/frontier/baseline_report.py` and `scripts/nova_baseline.py`:

```powershell
venv\Scripts\python -m scripts.nova_baseline --runs <run_id> <run_id> <run_id> <run_id>
```

It takes the runs of both models on quick-dev and on dev, in any order. It reads each run's summary through `summarize_run`, writes
`reports/frontier/nova_baseline_report.md` and `.json`, and prints the checks. Per arm and split: the
four counts and the tight diagnostic, format deviations and error codes by count, stop
reasons, input and output tokens, the Nova to Qwen token ratio (D25), cost at list price in
total and per 1,000 questions, and latency.

| Id | Check | Passes when |
|---|---|---|
| B1 | Four results | Exactly one run per configured model and per split (`quick_dev`, `dev`), each with `result: true` |
| B2 | Same question | All runs have the current prompt id, the current manifest hash, the same reply limit, and the same settings |
| B3 | Usage from AWS | Every prediction has both token counts, and each run's ledger record has source `aws_api` |
| B4 | As configured | Each run's model id and decoding values are those of `config/frontier.yaml` |

Exit codes: 0 all pass; 1 a check failed; 2 a config problem or an unknown run.

### 4.6 Tests

No test needs the network, credentials, or a paid call; the client is a stub. Prompts are
invented rows.

- Request: the system and user texts arrive character for character, including line breaks,
  a trailing space, and non-ASCII characters; exactly the keys of D3 are sent and none of D4.
- Response: text blocks joined; a reasoning block left out; each stop reason mapped; an
  empty reply returned as a reply, not a failure.
- Failures: each error code of section 2.2 becomes a `PredictorError` that holds the code
  and no prompt text; each sign-in error of `boto3.md` gives the sign-in message; a
  malformed response and cached tokens are failures (D9, D13).
- `client_config`: retry mode, attempts, and timeouts from config.
- `check_identity`: right account, wrong account, expired sign-in.
- Predictor: passes `describe`; settings as D16; one prediction per request in order;
  through the runner with a stub provider on the invented split, token totals reach the
  ledger with source `aws_api`; a provider that fails after N calls is resumed and no
  example is asked twice.
- Script: the Nova entries resolve with a stub factory; `where` defaults to `bedrock` and
  `--where` overrides it; the run snapshots `frontier` and `aws`; `gold` behaves as before.
- `aws_check` C8: active and on demand; legacy; no on-demand support; model not found.
- Pricing: cost of known token counts as an exact `Decimal`; an unknown model id is an error.
- Report: built from four stub runs; a missing or doubled run fails B1; each of B1 to B4 passes and fails; the files hold no
  reply text.
- Guards: nothing under `frontier/` imports `local_qlora` or the training stack; only
  `bedrock.py` imports boto3.

### 4.7 Documentation

- Runbook section 6: sign in, run C8, the trial, the two baselines, the report, resuming
  after a failure, and what each step costs.
- A new tech note, `reqspecs/references/tech/bedrock_nova.md`: sections 2.2 and 2.3 of this
  spec with their source links.
- `gotchas.md`: the `:300k` ids are for customization only; the temperature contradiction;
  Nova reuses prompt prefixes without reporting it; a Legacy model cannot be fine-tuned; the
  Qwen-count row is closed with the measured ratio.
- `evaluation_run.md`: the two new `PredictorEntry` fields. `coding.md`: Bedrock is called
  only through `BedrockProvider`; prices are read only through `pricing.py`.
- `administration/aws_commands_run.md`: the live commands of section 9.
- Architecture overview: the provider and the predictor as built.

## 5. Out of scope

- The training file, its upload, and the budget gate (v4.02). Fine-tuning jobs and
  deployments (v4.03). Tuned-model evaluation (v4.04).
- Calls in parallel, batch inference, the flex tier, and prompt caching.
- Few-shot prompts, or any wording for Nova alone.
- Any other model: Nova Micro, Nova 2 Lite, Nova Premier (req_007).
- The cost model and the reconciliation with the AWS bill (v5.02). This version multiplies
  tokens by list price for two runs and nothing more.
- A second evaluation loop of any kind.

## 6. What this hands to later versions

| For | What |
|---|---|
| v4.02 | `config/pricing.yaml` with the training prices; the measured Nova to Qwen token ratio in the report's JSON; the fine-tuning limits of section 2.3 |
| v4.03 | `FrontierProvider`, to extend with the job and deployment steps; the hyperparameter ranges; the lifecycle check C8, to run again before launching |
| v4.04 | B0-Pro and B0-Lite on dev, to compare each tuned model with. The same predictor with a deployment ARN as `model_id`. Pricing is keyed by base model id, so v4.04 adds the mapping from a deployment to its base model |
| v5.01 | Four Nova runs on the test split with the same command. 2,898 sequential calls are roughly one to two hours per arm; raising the batch size is allowed by the resume check |
| v5.02 | `inference_cost`, and ledger records whose tokens are Bedrock's own counts |

## 7. Known limits and follow-ups

- **Replies may not repeat.** AWS says Nova's output is not deterministic even with greedy
  settings. Each arm is run once; nobody has measured how much a second run would differ.
- **A baseline fixes the wording.** Once B0 has run, changing `config/prompt.yaml` means
  rerunning every arm. If the untuned models break the two-line format, that is a result
  to report, not a reason to adjust the prompt for them.
- **The reply limit is in Nova's tokens.** 192 was sized with the Qwen tokenizer (the
  longest correct reply is 80). The report's count of `length` stop reasons shows whether
  it binds.
- **A timed-out call may be billed and not counted.** If Bedrock answers after the client
  has given up, the tokens are on the bill and not in the ledger. v5.02's reconciliation is
  where it would show.
- **An SSO sign-in can expire mid-run.** The run stops with the sign-in message and is
  resumed.
- **Quotas are from the documentation.** The account's own values were not read.
- **The models can be retired.** Both are Active with no date, but their guaranteed period
  is over. A move to Legacy would block v4.03.
- **Latency includes waiting.** A throttled call's backoff is in its latency.
- **200 examples is a small sample**, and quick-dev comes from at most 15 companies. Read the
  baseline as a level, not to the percentage point.

## 8. Decisions for the owner

**Decided by the owner on 2026-11-16**, each as recommended: (1) the owner runs the billed
calls; (2) B0 is scored on dev in this version as well as on quick-dev; (3) the plain
in-region model ids are used. The table is kept as the record of what was asked.

| # | Question | Recommendation |
|---|---|---|
| 1 | **Who runs the billed baseline calls?** The rule today: anything billed is run by the owner. Here that is three commands costing under $0.35 together | Keep the rule: you run them (section 9). The alternative is to let Claude run inference whose worst-case cost, printed first, is under a figure you name (say $1), which would also cover v4.04. Training jobs stay yours either way |
| 2 | **Score B0 on dev (499) now as well, or only on quick-dev?** The plan says quick-dev for this version, but v4.04 compares each tuned model with its B0 on dev, so the dev run is needed by then. It costs at most $0.75 for both models | Run dev now too. It is the same command, the comparison in v4.04 then waits on nothing, and a second, larger run shows early whether the quick-dev figure holds. Cost aside, there is no reason to wait |
| 3 | **Plain in-region model id, or the cross-region profile (D6)?** The profile has twice the quota and may send a prompt to us-east-2 or us-west-2. The price is the same | In-region. Sequential runs use a sixth of the in-region quota, the plan keeps everything in us-east-1, and tuned models can only be served there |

## 9. Acceptance

Code, by Claude:

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
  and without the training stack blocked, with no network and no credentials.
- A test shows the prompt sent to Bedrock equals the canonical record's texts.
- `--predictor gold` gives the same output as before.

Live, by the owner (🟡), each command logged in `administration/aws_commands_run.md`:

```powershell
aws sso login --profile qlora
venv\Scripts\python -m scripts.aws_check                                             # 8 of 8, free
venv\Scripts\python -m scripts.evaluate --predictor nova-lite --split quick_dev --limit 5   # the trial, under $0.01
venv\Scripts\python -m scripts.evaluate --predictor nova-lite --split quick_dev
venv\Scripts\python -m scripts.evaluate --predictor nova-pro --split quick_dev
venv\Scripts\python -m scripts.evaluate --predictor nova-lite --split dev
venv\Scripts\python -m scripts.evaluate --predictor nova-pro --split dev
venv\Scripts\python -m scripts.nova_baseline --runs <the four run ids>
```

- C8 passes for both models.
- The trial completes as a partial run. If it fails on the temperature, D5 applies.
- All four baseline runs end with `RESULT: complete`, checks R1 to R5 pass, and each has a line
  in `ledger/evaluations.jsonl` and a ledger record with Bedrock's token counts.
- The report's checks B1 to B4 pass, and the report is committed.
- The baseline figures are written into this spec as section 11, and the B0 part of
  milestone M4 is recorded in the execution plan.

## 10. Next step

This version is complete: the owner ran the live commands of section 9 and the report is
written. The B0 part of milestone M4 is reached. Next is v4.02's live part: `python -m
scripts.nova_export`, which reprices Path B with the measured ratio and should open the
gate, then `--upload` (run sheet, stage C).

## 11. As built (2026-11-16)

No call was made to Bedrock during the build: the sign-in was expired throughout. The
figures of the owner's runs are at the end of this section.

What was run:

- `venv\Scripts\python -m pytest -q`: 1,560 passed (168 new). With the training stack
  blocked: 1,559 passed, 1 skipped. `venv\Scripts\python -m ruff check .`: clean.
- `scripts.evaluate --predictor gold --split quick_dev --limit 3`: as before (5 PASS lines,
  3 of 3). This partial smoke run added one run to the committed ledger.
- `scripts.evaluate --predictor nova-lite --split quick_dev --limit 5`, signed out: `CONFIG
  ERROR: the AWS sign-in is missing or has expired: run 'aws sso login --profile qlora'`,
  exit 2, no run, nothing reached Bedrock.

Differences from sections 3 and 4:

- **`max_attempts` is sent to botocore as `total_max_attempts`**, which counts the first
  try. botocore's own `max_attempts` counts retries only, so 8 would have meant 9 calls.
- **The AWS error message is logged, not stored.** A `PredictorError` carries the error code
  and the model id; AWS's own text goes to the log, so nothing a service echoes can reach a
  run's manifest.
- **C8's model ids reach `aws_check` through `load_aws_config`**, which adds them from
  `config/frontier.yaml` as `bedrock.inference_models`. `config/aws.yaml` is unchanged.
- **B1 also checks the arm** (a run recorded as B1-Lite is not a B0 baseline) and names any
  run that is not a baseline of a configured model on quick-dev or dev.
- **The report script writes nothing when a check fails**, and refuses a run id given twice.
- **A run with a model id that has no price** (a cross-region profile, later a deployment
  ARN) gets no cost in the report: it is left empty, never guessed from the base model.
- **One factory serves both models**: `build_predictor(model_name)`; `nova_pro` and
  `nova_lite` are the two names the script table points at.
- No test is marked `paid`: the owner's five-example trial is the first live call.

### Live results (the owner's runs, 2026-11-16; the report, 2026-11-18)

The first call to Bedrock was the five-example trial on 2026-11-16. Bedrock accepted
temperature 0, so the config stays as written. The four baseline runs followed the same
day, and `python -m scripts.nova_baseline` wrote the report on 2026-11-18: B1 to B4 passed.

| Arm | Split | Run | Execution accuracy | Input tokens | Output tokens | Nova / Qwen | Cost |
|---|---|---|---:|---:|---:|---:|---:|
| B0-Pro | quick-dev | `20261116T214132Z-eval-nova-pro-quick-dev-156c` | 75 of 200, 37.50% | 126,540 | 9,305 | 0.9848 | $0.131008 |
| B0-Lite | quick-dev | `20261116T201734Z-eval-nova-lite-quick-dev-700e` | 47 of 200, 23.50% | 126,540 | 9,311 | 0.9848 | $0.009827 |
| B0-Pro | dev | `20261116T203032Z-eval-nova-pro-dev-d704` | 190 of 499, 38.08% | 313,156 | 22,149 | 0.9851 | $0.321402 |
| B0-Lite | dev | `20261116T202551Z-eval-nova-lite-dev-d37e` | 123 of 499, 24.65% | 313,156 | 23,118 | 0.9851 | $0.024338 |

All four cost $0.49 at list price, under the $1.10 ceiling. The full figures are in
`reports/frontier/nova_baseline_report.md`.

What the runs settled, of the things section 7 left open:

- **Temperature 0 is accepted** by both models.
- **Nova counts slightly fewer tokens than Qwen for these prompts**: 0.985 Nova tokens per
  Qwen token, not the 1.5 the estimates assumed. Every Path B estimate made at 1.5 was
  about a third too high; v4.02's gate reprices with the measured ratio.
- **The account's limit for Nova Pro is 25 requests a minute**, a tenth of the documented
  default. Nova Pro's quick-dev run was throttled out at 182 of 200 after botocore's 8
  attempts (run `20261116T201927Z-eval-nova-pro-quick-dev-3a1b`) and was resumed; the run
  above is the resume. `config/frontier.yaml` now retries in `adaptive` mode with 12
  attempts. Nova Pro's dev run completed under the old settings.
- Mean latency: about 0.5 seconds for Nova Lite and 2 seconds for Nova Pro.

Still open: whether the charge on the bill equals the report's cost (v5.02 reconciles).
