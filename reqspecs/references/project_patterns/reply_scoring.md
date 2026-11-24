# Convention: reading and running a model's reply

**Last verified**: 2026-11-16  
**Defined in**: `qlora_finetuning/dsl/` and `config/dsl.yaml` (req_002 v2.01). Recorded before
its second use so that v2.02, v2.03, v3.01, v4.01, and v6.01 all follow it.

A model replies with a program and an answer. One package reads the reply and runs the
program. Nothing else in the project parses reply text, and nothing anywhere evaluates it as
Python.

## The call

```python
from qlora_finetuning.dsl import run_reply

outcome = run_reply(reply_text, example.table)

outcome.ok             # the program executed
outcome.value          # a float, not rounded, or "yes" / "no"; None on an error
outcome.program        # the program in canonical form: "op(a, b), op(a, b)"
outcome.answer_text    # what followed ANSWER:, as text; "" when there was none
outcome.strict         # the reply kept the two-line format exactly
outcome.deviations     # which format rules were broken, e.g. ("text_before",)
outcome.error_code     # e.g. "division_by_zero"; None when ok
outcome.error_stage    # "reply", "program", or "execution"
outcome.notes          # e.g. ("duplicate_row",)
outcome.to_dict()      # plain values for a JSON Lines record
```

`run_reply` returns an outcome for every string and never raises on one. The labels come from
`config/prompt.yaml` and the limits from `config/dsl.yaml`, each read once per process.

## Scoring the outcome (req_002 v2.02)

```python
from qlora_finetuning.data.gold_check import load_gold_flags
from qlora_finetuning.dsl import ReplyFormat, run_reply
from qlora_finetuning.evaluation import Gold, load_evaluation_config, score_reply, summarize

config = load_evaluation_config()                 # config/evaluation.yaml
labels = ReplyFormat.from_prompt(prompt)
gold = Gold.from_target(record.target, labels)    # the canonical record's target text

score = score_reply(run_reply(reply_text, example.table), gold, config, example.id)
score.format_valid, score.execution_correct, score.answer_correct, score.program_match
score.mismatch         # "scale", "sign", "kind", "other"; None when right or not run
score.to_dict()        # plain values for a JSON Lines record

totals = summarize(scores, flagged=load_gold_flags())
totals["all"]["execution_accuracy"]         # {"count": ..., "rate": ...}: the headline
totals["unflagged"]["execution_accuracy"]   # beside it, without the flagged examples
```

`answer_agrees(outcome, config)` (req_006 v6.01) compares a reply's answer line with its own
program's result, for showing one reply in the demo. It needs no gold value and is not a
metric: no run summary reports it.

`score_reply`, `summarize`, and `answer_agrees` are the only code that compares a value or
computes a rate.
A model is not scored by calling them in a loop of your own: `run_evaluation` does that for
every arm (`evaluation_run.md`).
Every rate is a count over all the examples given: a reply with an error is wrong on what
could not be judged.

## What the outcome means

| Field | Use it for |
|---|---|
| `strict` | The format-validity metric (plan §4.4) |
| `value` | Execution accuracy: within absolute 0.00001 or relative 0.5% of the gold result |
| `program` | Program match: the gold program, numbers compared by value, order kept |
| `answer_text` | Answer accuracy: read by `read_answer`; a `%` answer may be on either scale |
| `error_code`, `deviations`, `notes` | Error analysis. Store them with every prediction |

A reply that breaks the format is still read when it has a program line: `strict` is false and
`value` is set. Format and calculation are judged separately.

## Rules

- **Never parse a reply anywhere else**: no pattern over reply text in a runner, a backend, or
  a report. If something is missing from the outcome, add it to the package.
- **Never pass reply text to `eval`, `exec`, `compile`, `ast.literal_eval`, or an import.** A
  guard test fails the build if one appears in the package.
- **A backend removes what is specific to its model** (a thinking block, special tokens)
  before calling `run_reply`. The parser knows no model.
- **Do not round or compare in a caller.** The executor returns the unrounded value; the
  tolerance rule lives in one place, `qlora_finetuning/evaluation/metrics.py`, with its
  numbers in `config/evaluation.yaml`.
- **Store the unrounded value and the `ExampleScore` with every prediction.** If an
  allowance ever changes, every arm is rescored from the stored values without running a
  model again.
- **Never leave an example out of a denominator**, and never compute a rate by hand: pass
  every score to `summarize`.
- **Error codes are stable.** Once a model has been scored, a code is never renamed, because
  stored predictions carry it.
- **The language is fixed in code** (`dsl/language.py`: ten operations, seventeen constants).
  Changing it changes what "correct" means for every arm, and a test ties it to
  `config/data.yaml` and to the system text.
- **Pass the table as given**: the example's `table`, a list of rows of text cells. The
  executor does its own cell reading; do not clean cells first.
- **Tests use invented programs and tables** (`tests/dsl/conftest.py`), never dataset text.

## Reuse checklist

- [ ] Reply text reaches the project only through `run_reply` (or the three parts it joins)
- [ ] The outcome's code, deviations, and notes are stored with the prediction
- [ ] No rounding, comparison, or cell cleaning outside the package and the metrics
- [ ] Scores come from `score_reply`, totals from `summarize` with `load_gold_flags()`
- [ ] Model-specific text is removed by the backend, and a test shows it
