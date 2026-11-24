# Convention: evaluating a model

**Last verified**: 2026-11-19  
**Defined in**: `qlora_finetuning/evaluation/` (`predictor.py`, `runner.py`, `predictions.py`)
and `scripts/evaluate.py` (req_002 v2.03). Recorded before its second use so that v3.01,
v3.03, v3.04, v4.01, v4.04, and v5.01 all follow it.

Every arm is evaluated by one loop. A model backend supplies a **predictor**: prompts in,
replies out. The runner does everything else: the split, the order, the predictions file,
the scoring, the run in the registry, the usage in the ledger, resuming, and the summary.

## A backend implements this and nothing else

```python
from qlora_finetuning.evaluation import Prediction, PredictorError

class MyPredictor:
    name = "my-model greedy"                # part of the resume check
    model_id = "vendor/my-model"            # None only for a fake; needed for token counts
    settings = {"do_sample": False, "adapter": None}   # everything that can change a reply
    batch_size = 8
    usage_source = "measured"               # measured | aws_api | computed

    def predict(self, requests):
        # request.system, request.user: the canonical record's texts. Wrap them in the
        # model's format; add no word. request.max_new_tokens: the reply limit.
        ...
        return [
            Prediction(text=reply, input_tokens=n_in, output_tokens=n_out,
                       latency_seconds=seconds, stop_reason="end")      # end | length | other
            for ...
        ]                                   # one per request, in the same order
```

- `text` is the reply with what is specific to the model already removed (a thinking block,
  special tokens). The parser knows no model.
- Raise `PredictorError` when the model could not be asked, after the backend's own retries.
  **Never return an empty text in place of a failure**: an empty reply is scored as the
  model's answer.
- A token count is `None` when unknown. Do not estimate one.
- `request.example_id` is for logs. It must not change the reply.

## Running it

```python
from qlora_finetuning.evaluation.runner import run_evaluation, summarize_run

summary = run_evaluation(predictor, "quick_dev", label="a0 quick-dev",
                         path="A", arm="A0", where="local", scope="dev")
summary["result"]                                  # True: complete, and not partial
summary["all"]["execution_accuracy"]               # {"count": ..., "rate": ...}: the headline
summary["unflagged"]["execution_accuracy"]         # beside it
summary["diagnostic"]["execution_accuracy_tight"]  # the same under the tight rule
summary["usage"], summary["latency_seconds"], summary["stop_reasons"]

summarize_run(summary["run_id"])                   # again, from the file; no model
```

A local backend is shown in `qlora_finetuning/local_qlora/predictor.py` (req_003 v3.01):
`QwenPredictor` over a `QwenBackend` that is passed in, one GPU batch per batch of requests,
`usage_source = "measured"`, the batch size inside `settings` because a reply can depend
on its batch, and a factory that loads the model before any run exists (no weights, no GPU,
or a changed tokenizer is a `ConfigError` there and leaves no failed run behind).

A predictor may have one more, optional method, **`measurements()`**, returning plain
values that are known only inside its process (peak GPU memory, load time, tokens per
second). After the prediction loop, also when it failed, the runner writes them to
`measurements.json` in the run folder, and the summary carries them under `backend`. A
method that raises, or returns something that is not plain values, is logged and skipped:
it never costs a run. Put no prompt and no reply text in a measurement.

A hosted backend is shown in `qlora_finetuning/frontier/predictor.py` (req_004 v4.01): one
call per request, `batch_size = 1` because a batch is all or nothing and its calls are paid
for, `usage_source = "aws_api"`, and a factory that checks the sign-in before any run exists
(a `ConfigError` there leaves no failed run behind).

A `PredictorEntry` may also carry `where` (the place recorded when `--where` is not given)
and `configs` (config files the run snapshots beside the runner's own, the ones that decide
this backend's replies):

```python
"nova-lite": PredictorEntry(factory="qlora_finetuning.frontier.predictor:nova_lite",
                            kind="eval", path="B", arm="B0-Lite",
                            where="bedrock", configs=("frontier", "aws")),
```

From the command line, register the predictor in `PREDICTORS` in `scripts/evaluate.py` as
`"package.module:function"` (imported only when named), then:

```powershell
venv\Scripts\python -m scripts.evaluate --predictor <name> --split quick_dev
venv\Scripts\python -m scripts.evaluate --predictor <name> --split dev --resume <run_id>
venv\Scripts\python -m scripts.evaluate --summary <run_id>
```

## What a run leaves behind

| Where | What | Committed |
|---|---|---|
| `artifacts/runs/<run_id>/predictions.jsonl` | One line per example: the reply, the outcome with its unrounded value, the score, token counts, latency, stop reason. No prompt and no gold text | No |
| `artifacts/runs/<run_id>/evaluation.json` | What was asked: split, manifest hash, prompt id, reply limit, the predictor's description | No |
| `artifacts/runs/<run_id>/summary.json` | The summary | No |
| `artifacts/runs/<run_id>/measurements.json` | What the predictor's `measurements()` returned, if it has one | No |
| `ledger/runs.jsonl`, `ledger/ledger.jsonl` | The run, and one record of what it consumed | Yes |
| `ledger/evaluations.jsonl` | One line of counts, for a run that is a result | Yes |

## Why replies were wrong (req_003 v3.05)

`qlora_finetuning/evaluation/failures.py` gives every wrong reply of a stored run one
category, computed from its predictions line and the gold program: `run_failures(run_id)`
returns one `Failure` per example (right ones included), `count_failures` the counts,
`format_table` the replies outside the strict format. The categories, in the order the
rules are tried: `cut_off`, `no_program`, `bad_program`, `not_executable`, `kind`,
`steps_more`, `steps_fewer`, `operation`, `evidence`, `order`, `scale`, `sign`, `other`.
It works on any arm's run and imports neither path. Import it by its own name, as the
runner. A `Failure` holds ids, labels, and counts and may go into a committed report;
a reply or a gold program may not. Do not add a category that needs reading to assign.

## Finding a run that already answered (req_003 v3.04)

`qlora_finetuning/local_qlora/scoring.py` looks evaluation runs up by what was asked:
`find_evaluation(ctx, split=, limit=, settings=)` returns `("complete", run_id, summary)`,
`("incomplete", run_id, None)` for the newest run that was cut short (resume it), or
`(None, None, None)`. `settings` is what the predictor describes itself with;
`expected_settings` gives a Qwen predictor's without loading the model. Use it before
starting a run that a stopped job may already have made. It reads `evaluation.json` in
the run folders and trusts a run only if `summarize_run` passes its checks today.

When several runs carry one arm (ten checkpoints are all arm A1), no code may choose by
arm. The selection record (`read_selection`, `selected_adapter` in
`local_qlora/adapters.py`) names the run.

## Rules

- **Evaluate only through `run_evaluation`.** No backend, script, or notebook writes its own
  loop over a split, its own predictions file, or its own accuracy.
- **Import the runner by its own name**, `qlora_finetuning.evaluation.runner`. It is not
  exported from the package: it imports the data package, whose gold self-check imports the
  metrics.
- **A failure stops the run; it is resumed, never scored.** `PredictorError` carries the
  run id. Resuming is a new run that continues the earlier one, and it is refused unless the
  split, the prompt, the reply limit, and the predictor's name, model, and settings are the
  same. So put every setting that can change a reply into `settings`.
- **Only a run with `result: true` is a number to quote.** A limited run (`limit=N`) is
  partial; an interrupted run is unfinished. Neither gets a line in the results file.
- **Scores are recomputed from the file.** A changed allowance needs `summarize_run`, not a
  model. The scores stored in the lines are then out of date and check R3 says so.
- **Predictions are never committed** and never copied into a report: a reply can repeat
  dataset text. Counts are committed.
- **The test split** is asked for by name like any other, and the loader refuses it until
  req_005 v5.01. The runner adds no second way in.
- **`scope="measured"`** only for runs whose numbers go into the report; such a run refuses
  to start from uncommitted changes.

## Tests

`tests/evaluation/conftest.py` has `make_context(tmp_path)`: an invented split of 60
examples and a `RunnerContext` whose split, run folders, ledger, and results file are all
under the temporary folder. Pass it as `context=`. Use `FakePredictor` with `fail_after=N`
for an interruption. A backend's own tests give its predictor a stub client or model and
check the request it builds; they do not need the runner.

## Reuse checklist

- [ ] The backend implements `predict` and the five attributes; it has no loop over a split
- [ ] Every setting that changes a reply is in `settings`
- [ ] Model-specific text is removed before `Prediction.text`, and a test shows it
- [ ] A failure raises `PredictorError`; an empty reply is returned only when the model gave one
- [ ] The entry point calls `run_evaluation`; results are read from its summary
- [ ] Anything that can fail before the first reply fails in the factory, as a `ConfigError`
- [ ] Nothing from `artifacts/runs/` is committed
