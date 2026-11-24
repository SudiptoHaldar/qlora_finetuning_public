# Convention: how an entry point uses the run registry and cost ledger

**Last verified**: 2026-11-16  
**Defined by**: req_000 v0.04 (`qlora_finetuning/tracking/`). First users: the evaluation
runner (v2.03), the training pipeline (v3.02), the SageMaker launcher (v3.03), the Bedrock
adapter (v4.01).

Every training or evaluation entry point starts a run, does its work inside it, and records
what it consumed. Nothing else in the project invents run ids, run folders, or usage records.

## Shape

```python
from qlora_finetuning.tracking import start_run

with start_run(
    "eval", "a0 quick-dev",
    path="A", arm="A0", where="local",
    scope="dev",                        # "measured" only for runs whose numbers are reported
    configs=["local_env", "prompt"],    # every config file the run reads
    split_manifest_hash=manifest_hash,  # from the split manifest once req_001 provides it
) as run:
    started = time.perf_counter()
    ...                                 # write outputs under run.run_dir
    run.record(
        phase="eval",
        wall_clock_seconds=time.perf_counter() - started,
        examples=len(examples),
        source="measured",
        source_detail="time.perf_counter around the evaluation loop",
    )
    run.outputs["predictions"] = "predictions.jsonl"
```

Leaving the block marks the run `completed`; an exception marks it `failed` and is re-raised.

## What to record, and from where

| Usage | Record | Source |
|---|---|---|
| Local run time | `wall_clock_seconds`, `examples` | `measured` |
| SageMaker job | `record_from_sagemaker_job(describe_response, run, phase)`, then `run.append(...)` | `aws_api` |
| Bedrock inference | Sum the `usage` block of every `Converse` response with `sum_converse_usage`, then `record_from_converse_usage(totals, run, model_id, examples, wall_clock_seconds)` | `aws_api` |
| Bedrock fine-tuning | `training_tokens` = tokens in the training file times epochs, at launch | `computed` |
| The same, once billed | A new record with `billed_amount_usd`, `replaces` naming the estimate | `bill` |

One record per run and phase, with totals. Per-question detail belongs in the predictions file.

An evaluation does not call `start_run` itself: `run_evaluation` starts the run, records its usage
(also when the run fails), and names its outputs (`evaluation_run.md`). The shape above is for
every other entry point.

## A SageMaker job (req_003 v3.03)

A job's runs are ordinary runs made inside the job with `where="sagemaker"`: they write to
the job's own `artifacts/` and `ledger/`, which are copied to its S3 workspace. `pull`
brings the run folders here and appends the job's ledger lines to the committed ledger,
each once.

The launch itself is a run (kind `other`, place `sagemaker`) with no usage at first. When
the job has ended, `pull` adds that run's record from `DescribeTrainingJob`, through
`record_from_sagemaker_job`: instance type, instance count, billable seconds, source
`aws_api`, the job's name as `external_id`. One record per job, whatever runs it held; the
runs' own records carry wall-clock seconds and tokens, so nothing is counted twice.

## Rules

- Pass `scope="measured"` only for runs that feed the report. Such a run refuses to start from
  uncommitted changes; commit first. Changes inside `ledger/` do not count.
- Name every config file the run reads in `configs`, so the manifest holds a copy and a hash.
- In a cloud job there is no git: the launcher sets `QLORA_GIT_COMMIT`.
- Never put a credential, token, or email address in a label, note, or output.
- Do not compute dollars. Record quantities; the cost model applies `config/` rates.
- In tests, pass `cfg=load_tracking_config(root=tmp_path)`, a fixed `GitInfo`, and
  `environment={}`, so nothing is written to the repository and torch is not imported.

## Checklist

- [ ] The entry point's work happens inside `with start_run(...)`
- [ ] Every billed quantity the run caused has a ledger record with a truthful source
- [ ] Outputs worth finding later are named in `run.outputs`
- [ ] Tests use a temporary root and stand-in git state
