# Project patterns

**Last verified**: 2026-11-24

Solutions specific to this project that settled through iteration. A pattern is recorded here
once the same approach has been used twice; each gets its own file with a short description,
the files that show it, and a reuse checklist.

## Patterns

| Pattern | File | Use it for |
|---|---|---|
| Operator check script | `check_script.md` | A read-only script that reports PASS or FAIL per check on something outside the code. Shown in `scripts/aws_check.py` and `scripts/env_check.py` |

## Conventions

Defined by one version for later versions to follow, so recorded before their second use.

| Convention | File | Use it for |
|---|---|---|
| Run tracking | `run_tracking.md` | Any training or evaluation entry point: start a run, record what it consumed |
| Canonical record | `canonical_record.md` | Any code that shows a model a prompt or a target: take the three texts from the serializer and wrap them unchanged |
| Split loading | `split_loading.md` | Any code that needs FinQA examples: ask for `fit`, `dev`, or `quick_dev`; the test split is locked |
| Reply scoring | `reply_scoring.md` | Any code that receives a model's reply: call `run_reply`, store its code; never parse or evaluate reply text elsewhere |
| Evaluation run | `evaluation_run.md` | Any code that evaluates a model: implement `Predictor`, call `run_evaluation`, quote only a run that is a result |
| Pricing a ledger record | `pricing_a_ledger_record.md` | Any code that needs a dollar figure: take it from the cost model or its record; never multiply a quantity by a price of your own |
| A package for publication | `release_package.md` | Any folder or document that leaves the project: copy from a checked source, generate from records, prove it by the ordinary runner, execute the code a card prints |
| Comparing two runs | `paired_comparison.md` | Any statement that one run is better than another or how sure a figure is: paired bootstrap, McNemar's exact test, Holm; never an interval or a p-value computed elsewhere |

## Candidates

None at present.
