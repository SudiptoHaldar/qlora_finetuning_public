# PRP: Bedrock Nova adapter and zero-shot baselines (req_004 v4.01)

**Feature Name**: bedrock_nova_adapter (frontier_tuning v4.01)  
**Version**: 1  
**Created**: 2026-11-16  
**Status**: ✅ IMPLEMENTED (code; the owner's billed baseline runs remain)  
**Implemented**: 2026-11-16  
**Implementation Summary**: [impl_004_20261116_021029.md](../impl_summary/impl_004_20261116_021029.md)  
**Confidence Score**: 8/10  
**Estimated Time**: 4–5 hours of code, plus the owner's live steps  
**Origin**: `reqspecs/reqs/req_004_frontier_tuning/req_004_01_spec.md`  
**Template**: `prp_lean.md` (expanded, medium complexity)  

---

## Goal

Let the evaluation runner ask Amazon Nova Pro and Amazon Nova Lite for replies through the
Bedrock Converse API, so that arms B0-Pro and B0-Lite can be scored on quick-dev and dev
with `python -m scripts.evaluate --predictor nova-pro|nova-lite`.

## Why & What

- **User value**: Path B gets its untuned reference points, the first real token counts
  from Bedrock, and a measured Nova to Qwen token ratio for the v4.02 cost estimate.
- **Technical**: two config files, a provider protocol with a Bedrock implementation, one
  predictor class, a price reader, a baseline report with four checks, one new `aws_check`
  check, two new fields on `PredictorEntry`, tests with stub clients.

The owner decided the three questions of spec section 8 on 2026-11-16: **the owner runs
every billed call**; B0 is scored on **dev as well as quick-dev**; the **plain in-region
model ids** are used. So this PRP's agent builds and tests the code and stops before any
call to Bedrock. The version ends 🟡 operator-gated.

## Success Criteria

Code (the agent):

- [x] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass.
- [x] A test shows the texts sent to Bedrock equal the request's, character for character.
- [x] `--predictor gold` prints what it printed before.
- [x] No test creates a real client, needs a sign-in, or makes a billed call.
- [x] Without a sign-in, `scripts.evaluate --predictor nova-lite ...` exits 2 with the
      sign-in message and leaves no run behind.

Live (the owner, spec section 9): `aws_check` 8 of 8; the five-example trial; four baseline
runs that are results; `scripts.nova_baseline` with B1 to B4 passing; report committed.

---

## Context

### References

```yaml
refs:
  - reqspecs/references/project_patterns/evaluation_run.md   # what a backend implements
  - reqspecs/references/project_patterns/run_tracking.md     # the runner records the usage
  - reqspecs/references/project_patterns/canonical_record.md # texts unchanged
  - reqspecs/references/project_patterns/check_script.md     # C8; the report script's exit codes
  - reqspecs/references/tech/boto3.md                        # stubs, sign-in errors
  - reqspecs/references/common/gotchas.md

code:
  - file: qlora_finetuning/evaluation/predictor.py
    why: Predictor, Prediction, PredictorError, describe(); stop reasons end | length | other
  - file: qlora_finetuning/evaluation/runner.py
    why: run_evaluation records tokens when model_id is set and every count is known;
         RunnerContext.configs; summarize_run gives the summary the report reads
  - file: scripts/evaluate.py
    why: PredictorEntry and PREDICTORS; ConfigError from a factory already exits 2
  - file: scripts/aws_check.py
    why: CheckResult, _guard, CHECKS, AUTH_EXCEPTIONS, AUTH_ERROR_CODES, load_aws_config
  - file: qlora_finetuning/evaluation/metrics.py
    why: load_evaluation_config(root=, config_dir=): the shape of a typed config loader
  - file: qlora_finetuning/data/token_budget_report.py
    why: build_report / render_markdown / write_report conventions for a committed report
  - file: tests/evaluation/conftest.py
    why: make_context(tmp_path): invented split (dev 8, quick_dev 3), temporary ledger
  - file: tests/scripts/test_aws_check.py, tests/scripts/test_evaluate.py
    why: MagicMock clients, FakeSession; both hold assertions that change (7/7, ["gold"])

docs (read 2026-11-16; summarized in spec sections 2.2 to 2.4):
  - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
  - https://docs.aws.amazon.com/nova/latest/userguide/complete-request-schema.html
  - https://docs.aws.amazon.com/nova/latest/userguide/using-converse-api.html
  - https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
  - https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html
  - https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle-legacy.html
  - https://docs.aws.amazon.com/bedrock/latest/userguide/deploy-custom-model-on-demand.html
```

### Facts verified 2026-11-16

| Fact | Value |
|---|---|
| Converse request (boto3 1.44.3) | `modelId`, `system=[{"text"}]`, `messages=[{"role","content":[{"text"}]}]`, `inferenceConfig={maxTokens, temperature, topP, stopSequences}`, `additionalModelRequestFields` |
| Converse response | `output.message.{role, content[]}`, `stopReason`, `usage.{inputTokens, outputTokens, totalTokens, cacheReadInputTokens?, cacheWriteInputTokens?}`, `metrics.latencyMs` |
| `bedrock.get_foundation_model(modelIdentifier=)` | `modelDetails.inferenceTypesSupported` (list), `modelDetails.modelLifecycle.{status: ACTIVE or LEGACY, endOfLifeTime?}` |
| botocore `Config(retries=...)` | `total_max_attempts` counts the first try; `max_attempts` counts retries only. Use `total_max_attempts` |
| Sign-in errors | The exception classes and codes in `scripts/aws_check.py`; all the classes are `BotoCoreError`, so catch them first |
| `settings.load_config` | `0` is not "missing"; an empty list is. A `ConfigError` names every missing key |
| `start_run(configs=...)` | Copies `config/<name>.yaml` from `config_dir` (the real folder when `None`), so `frontier` and `aws` can be added to a test context |
| Token budget report JSON | `splits.<name>.prompt_tokens_in_one_pass`: quick_dev 128,495, dev 317,882 |
| Summary totals | `summary["all"]` already holds `error_codes`, `deviations`, `mismatches` by count |
| AWS sign-in | Expired on 2026-11-16: nothing here was tried against the live account |

### Critical Gotchas

- **`qlora_finetuning/frontier/__init__.py` must not import `bedrock.py`**:
  `tests/test_package.py` checks that a bare package import does not load boto3. Only
  `bedrock.py` imports boto3 or botocore; the predictor's factories import it inside the
  function.
- **Nothing under `evaluation/` may import `frontier`** (existing guard). The script names
  the factories as `"module:function"` text.
- **`temperature: 0` is a valid config value.** Check its type with `bool` excluded, never
  with truthiness.
- **Never put request text in an error.** `PredictorError` carries the AWS error code; the
  AWS message goes to the log only.
- **An empty reply is a reply.** Only a missing or malformed response is a failure.
- **A model id is not a secret**, but run `describe()` output through nothing new: the
  runner already checks it for credentials.
- **Prices are text in YAML** and become `Decimal`; a float is a config error.
- **`tests/scripts/test_evaluate.py` asserts `sorted(PREDICTORS) == ["gold"]`** and
  `test_aws_check.py` asserts `7/7`: both change. `make_bedrock()` there needs
  `get_foundation_model`.
- **The Windows console is cp1252**: scripts print ASCII only.
- **The agent makes no Bedrock call** and runs no command that needs the expired sign-in.

---

## Implementation

### Files

```yaml
CREATE:
  - config/frontier.yaml, config/pricing.yaml
  - qlora_finetuning/frontier/config.py: FrontierModel, FrontierConfig, load_frontier_config,
      load_aws_settings
  - qlora_finetuning/frontier/provider.py: ProviderReply, FrontierProvider
  - qlora_finetuning/frontier/bedrock.py: client_config, aws_session, bedrock_runtime_client,
      check_identity, build_request, read_response, BedrockProvider
  - qlora_finetuning/frontier/predictor.py: FrontierPredictor, build_predictor, nova_pro,
      nova_lite
  - qlora_finetuning/frontier/pricing.py: ModelPrice, Pricing, load_pricing, inference_cost
  - qlora_finetuning/frontier/baseline_report.py: build_report, render_markdown,
      write_report, load_qwen_prompt_tokens
  - scripts/nova_baseline.py
  - tests/frontier/test_config.py, test_bedrock.py, test_predictor.py, test_pricing.py,
      test_baseline_report.py; tests/scripts/test_nova_baseline.py
  - reqspecs/references/tech/bedrock_nova.md
  - reqspecs/impl_summary/impl_004_<TIMESTAMP>.md

MODIFY:
  - scripts/evaluate.py: PredictorEntry.where and .configs; entries nova-pro, nova-lite
  - scripts/aws_check.py: C8; load_aws_config adds bedrock.inference_models from frontier.yaml
  - tests/scripts/test_evaluate.py, tests/scripts/test_aws_check.py
  - qlora_finetuning/frontier/__init__.py: docstring only
  - docs/runbook.md (section 6), administration/aws_commands_run.md (the owner's
    commands, prepared), coding.md, gotchas.md, evaluation_run.md, boto3.md (pointer),
    architecture/architecture_overview.md
  - the spec, req_004_roadmap.md, execution_plan.md
```

### Design

```python
# config.py
@dataclass(frozen=True)
class FrontierConfig:
    models: dict[str, FrontierModel]      # name -> (name, model_id, untuned_arm)
    temperature: int | float; top_k: int
    retry_mode: str; max_attempts: int; connect_timeout: int; read_timeout: int
    report_dir: Path
    @property
    def settings(self) -> dict:           # spec D16; goes into the resume check
        return {"api": "converse", "temperature": ..., "top_k": ..., "top_p": None,
                "stop_sequences": [], "service_tier": "default", "reasoning": "not sent"}
    def model(self, name) -> FrontierModel   # ConfigError naming the known ones

# provider.py
@dataclass(frozen=True)
class ProviderReply: text: str; input_tokens: int; output_tokens: int
                     stop_reason: str; latency_seconds: float
class FrontierProvider(Protocol):
    def converse(self, model_id, system, user, max_tokens) -> ProviderReply: ...

# bedrock.py
def build_request(model_id, system, user, max_tokens, cfg) -> dict:
    return {"modelId": model_id,
            "system": [{"text": system}],
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": cfg.temperature},
            "additionalModelRequestFields": {"inferenceConfig": {"topK": cfg.top_k}}}

def read_response(response, latency_seconds) -> ProviderReply:
    # message with role assistant and a content list, else PredictorError
    # text = "".join(text blocks); other blocks left out
    # stopReason: end_turn, stop_sequence -> end; max_tokens -> length; any other text -> other
    # usage.inputTokens / outputTokens whole and >= 0, else PredictorError
    # cacheReadInputTokens or cacheWriteInputTokens > 0 -> PredictorError

class BedrockProvider:
    def __init__(self, client, cfg, profile="qlora", clock=time.perf_counter)
    def converse(...):
        try: response = client.converse(**build_request(...))
        except AUTH_EXCEPTIONS            -> PredictorError(sign-in message)
        except ClientError: code in AUTH_ERROR_CODES -> sign-in message
                            else PredictorError(f"Bedrock answered {code} for {model_id}")
        except BotoCoreError              -> PredictorError(type name)   # timeouts, no network

def check_identity(sts, aws) -> None      # ConfigError: wrong account, or sign-in message

# predictor.py
class FrontierPredictor:                  # batch_size = 1, usage_source = "aws_api"
    def predict(self, requests): one provider.converse per request, in order
def build_predictor(model_name, *, cfg=None, aws=None, session=None) -> FrontierPredictor
def nova_pro(split, context); def nova_lite(split, context)

# pricing.py
def inference_cost(model_id, input_tokens, output_tokens, pricing=None) -> Decimal

# baseline_report.py
def build_report(summaries, ledger_records, cfg, pricing, qwen_prompt_tokens,
                 prompt_id, manifest_hash) -> dict     # runs, checks B1..B4, passed
```

`PredictorEntry(factory, kind, path, arm, where=None, configs=())`. In `main`: the place is
`--where`, else the entry's, else `local`; the entry's configs are added to the context's.

C8 in `aws_check`: for each id in `cfg["bedrock"]["inference_models"]`,
`get_foundation_model` must list `ON_DEMAND` and report `ACTIVE`.

### Tasks (~4.5 hours)

1. **Config files and loaders, with tests** (~30 min).
2. **`provider.py` and `bedrock.py`, with tests** (~60 min).
3. **`predictor.py`, with tests through the runner** (~35 min).
4. **`scripts/evaluate.py` and its tests** (~20 min).
5. **`aws_check` C8 and its tests** (~20 min).
6. **`pricing.py`, `baseline_report.py`, `scripts/nova_baseline.py`, with tests** (~60 min).
7. **Documents: runbook, tech note, gotchas, patterns, coding rules, architecture** (~30 min).
8. **Validation; status documents; implementation summary** (~20 min).

### Error handling

| Situation | Behavior |
|---|---|
| No sign-in, wrong account, missing profile, bad config | `ConfigError` from the factory: `CONFIG ERROR`, exit 2, no run |
| Bedrock error after botocore's retries; timeout; no network | `PredictorError` with the code or the error type; the run stops and can be resumed |
| Sign-in expires mid-run | `PredictorError` with the sign-in command |
| Malformed response; cached tokens reported | `PredictorError` |
| Empty reply, reply cut at the limit, content filtered | Returned and scored; stop reason `end`, `length`, or `other` |
| Unknown model id in the price file | `ConfigError` |
| Report asked for missing, doubled, partial, or foreign runs | Check B1 fails; nothing is written; exit 1 |

---

## Validation

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q
venv\Scripts\python -m scripts.evaluate --predictor gold --split quick_dev --limit 3
venv\Scripts\python -m scripts.evaluate --predictor nova-lite --split quick_dev --limit 5
```

**Expected**: ruff clean; all tests pass; the gold run as before; the last command, with the
sign-in expired, prints `CONFIG ERROR` with the `aws sso login` hint and exits 2 without
reaching Bedrock. It is run only while the sign-in is expired; signed in, it is the owner's.

---

## Checklist

- [x] Files created and modified as listed
- [x] Every function and every check has passing and failing tests
- [x] Lint clean; tests green
- [x] Runbook, references, `coding.md`, architecture updated
- [x] Roadmap, execution plan, spec status updated (🟡); implementation summary created
- [ ] Owner: sign in, `aws_check` 8 of 8, the trial, four baseline runs, the report

---

## Implementation Time Estimate

**Total: 4–5 hours**, then the owner's six commands (about 30 minutes, mostly waiting).

| Task | Time |
|---|---|
| Config files and loaders | 30 min |
| Provider and Bedrock implementation | 60 min |
| Predictor | 35 min |
| Evaluate script | 20 min |
| `aws_check` C8 | 20 min |
| Prices, report, report script | 60 min |
| Documents | 30 min |
| Validation and status documents | 20–40 min |

## Confidence Score: 8/10

The interfaces on both sides were read: the runner's protocol in this repository and the
Converse definition in the installed boto3. Two points are held back because nothing could
be tried against the live account: whether Bedrock accepts `temperature: 0`, and whether a
real response holds anything the stub responses do not.

## Time Taken for PRP Generation

**~20 minutes** (the research was done while writing the spec the same day)

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 3 min |
| Codebase exploration | 9 min |
| Research existing PRPs and patterns | 3 min |
| Writing PRP document | 5 min |
