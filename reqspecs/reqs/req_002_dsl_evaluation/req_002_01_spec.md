# DSL Evaluation

Spec Group      :   req_002_dsl_evaluation  
Feature Name    :   dsl_evaluation  
Version         :   1  
Status          :   ✅ Complete 2026-11-15 (see `reqspecs/PRPs/prp_002_dsl_parser_executor_v1.md`); tests and lint green locally, confirmed by the owner  
Last updated    :   2026-11-15  

## 1. v2.01 — DSL parser and safe executor

Every model in the comparison replies with a short program and an answer. The program is
never trusted as arithmetic done by the model: code runs it. That code is also the scorer, so
it defines what "correct" means for all six arms, and it reads text written by a model, so it
must be safe on any input.

This version adds three things to `qlora_finetuning/dsl/`:

- a **reply parser**: a model's reply in, the program text and the answer text out, with a
  flag saying whether the reply kept the two-line format exactly;
- a **program parser**: program text in, a checked list of steps out;
- an **executor**: steps and the example's table in, a number or `yes` / `no` out.

Everything that can go wrong with a reply has a named error. Nothing in a reply is ever
evaluated as Python.

It decides nothing about rounding, tolerance, or which metric is reported. Those are v2.02.

**Depends on**: v0.02 (built). It also uses the operation inventory of v1.01 and the target
format of v1.03, both built.

## 2. What exists today (verified 2026-11-15)

Measured on fit and dev together (5,383 examples, 8,283 program steps), read through
`load_split`. The test split was not read. The figures come from a throwaway script that
included a trial executor; the built code must reproduce them (section 9).

### 2.1 What earlier versions provide

| Item | State |
|---|---|
| `qlora_finetuning/dsl/`, `tests/dsl/` | Empty packages |
| `config/prompt.yaml` | The reply labels `PROGRAM:` and `ANSWER:`; a system text that lists the ten operations, the argument kinds, and the seventeen constants |
| `config/data.yaml`, `finqa.expected` | The ten operations, the seventeen constants, seven argument kinds, two arguments per step, at most six steps; checked against all 8,281 examples by the schema report |
| `reports/data/finqa_split_report.json` | Operation counts, program lengths, and argument kinds for fit and for dev |
| `serialize_split(name)` | Targets of exactly two lines: 5,383 of 5,383. The longest is 145 characters |
| `schema_report.split_program` | A counting pattern, private to that report. Not a parser (v1.01 D14) |
| v1.03 D16 | The system text's statements about percent literals and operations wait for confirmation by execution |

### 2.2 Gold programs

- Every program has the form `op(a, b), op(a, b)`: steps joined by a comma and one space,
  two arguments each, no other spacing. The longest is 122 characters.
- Characters used: lowercase letters, digits, the space, and `# % & ( ) , - . / _`.

| Steps in a program | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Examples | 3,185 | 1,743 | 298 | 71 | 82 | 4 |

| Argument of an arithmetic step | Count | Forms seen |
|---|---|---|
| Whole number | 8,270 | `920`, `-121` |
| Decimal | 3,641 | `4474.2`, `-3.5`, `4850.0`; 19 start with the point (`.35`) |
| Step reference | 2,907 | `#0` |
| Constant | 1,117 | `const_100` |
| Percent literal | 235 | `6.9%`, `7%` |

- 161 literals are negative. None has a plus sign, an exponent, a thousands separator, a
  currency sign, or brackets. No percent literal is negative.
- Every reference points to an earlier step. 18 point to a table step.
- `greater` is always the last step (v1.01).

### 2.3 Table operations and tables

Table operations: 198 steps in 191 examples.

| Fact | Count |
|---|---|
| Second argument is `none` | 198 of 198 (two steps in the whole dataset have a number there, per the schema report) |
| Named row is in the example's table | 198 |
| Named row is the header row | 0 |
| Named row has an empty or non-numeric cell | 0 |
| Named row has percent cells | 29 |
| Named row has cells with a bracketed part, such as `-9 ( 9 )` | 47 |
| Named row's name occurs twice in the table | 1 (`PNC/2013/page_158.pdf-1`; both rows give the same result) |

What a model could name, over all 34,177 table rows:

| Fact | Count |
|---|---|
| Row names containing a bracket | 4,650 (14%), in 2,126 tables. All are balanced; 11 are nested |
| Row names containing a comma | 191, in 177 tables |
| Row names with leading or trailing spaces, starting with `#`, or equal to `none` | 0 |
| Empty row names | 1,794, all in header rows |
| Tables with a repeated row name | 241. Of 511 repeated names, the rows differ in 473 |
| Data rows whose cells are all numbers under the cell rule of D17 | 26,595 of 28,794 (92.4%) |

No gold program names a row with a bracket or a comma in it, but a model may: one row in
seven has one.

### 2.4 Trial execution of the gold programs

With binary floating point and the rules of section 3:

| Result | Examples |
|---|---|
| Runs without an error | 5,383 of 5,383 |
| Equals `exe_ans` after rounding to five decimal places | 4,945 |
| Differs from `exe_ans` by a relative 0.0005% at most | 438 |

Of the 438: in 83 the gold value is in scientific notation, and in 355 it carries only five
or six significant digits (`41932.2` for 41,932.20339). So the dataset's results are not
uniformly rounded to five decimal places. No gold program divides by zero or overflows.

Two rules were tested against the data:

- **Percent cells in a table row are divided by 100.** Leaving them as printed makes 27 of
  the 191 table examples wrong.
- **A percent literal is its number divided by 100**, as the system text says. All 235 agree.

## 3. Locked decisions

### Safety

| # | Decision | Reason |
|---|---|---|
| D1 | No reply text is ever passed to `eval`, `exec`, `compile`, `ast.literal_eval`, or an import. Operations are looked up in a fixed table of ten names. A guard test reads the package source and fails if one of those calls appears | Roadmap standing decision. A model's reply is untrusted input |
| D2 | The scanner is written by hand and reads the text once, left to right. No pattern with nested repetition is applied to reply text | A crafted reply must not be able to make scoring slow |
| D3 | Limits are checked before any other work: reply length and number of steps, from `config/dsl.yaml`. Over the limit is an error, never a cut | The longest correct reply is 145 characters and 6 steps; the limits are for safety and sit far above that |
| D4 | The package uses the standard library only. It does not import `qlora_finetuning.data` at module level and never reads the dataset; the table is passed in as a list of rows | The demo (v6.01) runs it on input that is not a dataset example. The test lock is not involved |
| D5 | Every failure caused by reply text is a `DslError` with a stable `code`. `run_reply` returns an outcome and never raises on any string | Plan: typed errors, never exceptions leaking. The runner (v2.03) stores the code per example |

### The reply

| # | Decision | Reason |
|---|---|---|
| D6 | The labels come from `config/prompt.yaml` through `load_prompt()`; they are not written in the code | Working rule: no prompt wording in code |
| D7 | A reply is **strict** when, apart from whitespace at its two ends, it is exactly two lines: the program label and a program, then the answer label and an answer. Labels are in their exact case. Blank lines, spaces or tabs at either end of a line or after a label, and `\r\n` line ends are allowed. (As built: blank lines and leading spaces were added to the allowances, since no deviation names them and such a reply holds nothing but the two fields) | This is the "format validity" metric of plan §4.4 |
| D8 | When a reply is not strict, the parser still **recovers** what it can: the program is what follows the label on the first line that starts with the program label; the answer is what follows the label on the first later line that starts with the answer label. Labels match in any case here. Other lines are ignored. The result says which rules were broken | Plan §4.4 lists format validity as a diagnostic beside execution accuracy. With one flag, v2.02 can report accuracy both ways. **Decision for the owner in section 8** |
| D9 | A reply with no program line is an error. A reply with a program line and no answer line is recovered with an empty answer. (As built: an answer line with nothing after its label also counts as `answer_missing`) | The program is what is executed; the answer is a secondary metric |
| D10 | The parser knows nothing about any model: no handling of thinking blocks, Markdown, or code fences as such. A fence on its own line is simply an ignored line under D8 | One rule for six arms. A backend that must strip something does it before the parser (v3.01, v4.01) |

### The program

| # | Decision | Reason |
|---|---|---|
| D11 | Grammar: steps separated by commas; a step is `name(first, second)`; spaces and tabs are allowed around every name, bracket, comma, and argument, and nowhere inside one | The gold form with free spacing. A model that writes `add( 1,2 )` has written the same program |
| D12 | Names are the ten operations, lowercase only. Anything else before a bracket is `unknown_operation`, whatever it looks like | Allowlist. `Add`, `sum`, `os.system` are all the same error |
| D13 | A step has exactly two arguments; otherwise `wrong_arity`. A bracket inside an arithmetic step is `nested_call` | Section 2.2. The nested form (`program_re`) is not the contract |
| D14 | An arithmetic argument is a number, a percent literal, a reference, or a constant. A number is an optional minus sign, digits, and an optional fraction, or a fraction starting with the point. No plus sign, exponent, thousands separator, currency sign, or space before `%` | The forms of section 2.2 and of the system text. `1,234` reads as two arguments and fails on arity |
| D15 | A constant is one of the seventeen names. Another `const_` name is `unknown_constant` | The system text lists them all, and any plain number is accepted anyway. **Decision for the owner in section 8** |
| D16 | A reference is `#` and a step number lower than the step it is in. Anything else is `bad_reference` | Forward and self references have no value to read |
| D17 | For a table operation, the first argument is the **row name**: the text between the opening bracket and the **last** comma that is outside inner brackets, trimmed. Brackets inside a name must balance. The second argument is `none` or any valid arithmetic argument, and is ignored | Section 2.3: 14% of row names hold brackets and 191 hold commas; all are balanced. Two gold steps carry a number as second argument |
| D18 | A parsed program keeps each argument's text as written. `format_program` writes the canonical form `op(a, b), op(a, b)`; every gold program is its own canonical form | Plan P1 check "the target round-trips through the parser". v2.02 builds its exact-match rule on the parsed form |

### Execution

| # | Decision | Reason |
|---|---|---|
| D19 | Numbers are Python floats. The executor does not round | Section 2.4: this reproduces every gold result. Rounding and tolerance are v2.02's |
| D20 | A percent literal is its number divided by 100 | Section 2.4; confirms the system text (v1.03 D16) |
| D21 | `add`, `subtract`, `multiply`, `divide` are the four operations on two numbers. `exp` is the first raised to the power of the second. `greater` gives `yes` when the first is strictly larger, otherwise `no` | The system text |
| D22 | Division by zero is `division_by_zero`. A result that is not a finite real number (overflow, a negative number to a fractional power) is `numeric_error` | Named errors, never `inf`, `nan`, or a complex number in a score |
| D23 | A `yes` / `no` value used as an operand is `yes_no_operand`. `greater` may appear anywhere; the program's result is the value of its last step | Gold programs end with `greater`; a model may not |
| D24 | A table operation looks for a row whose first cell, trimmed, equals the row name exactly, in every row of the table including the first. No such row is `row_not_found`. Without a table it is `table_missing` | Exact names are what the system text asks for |
| D25 | When two rows have the name, **the last one is used**, and the outcome carries the note `duplicate_row` | The published FinQA evaluator builds a name-to-row map, so a later row replaces an earlier one, and the gold results came from it. Fit and dev cannot confirm this: the one gold case gives the same result either way. **Decision for the owner in section 8** |
| D26 | The cell rule: remove `$`, keep the text before the first `(`, remove thousands separators; a number is taken as it is, and a number followed by `%` is divided by 100. A cell may carry a plus sign (as built: 18 cells, all in one table, have the form `+5%`; a program literal still may not). All cells after the first are used. A row with any cell that is not a number under this rule, or with no cells, is `row_not_numeric` | Section 2.4: it reproduces all 191 table examples, and the percent rule is needed for 27 of them. A row of mixed content has no defined sum |
| D27 | The four table operations are the sum, the mean, the largest, and the smallest of the row's numbers | The system text |

## 4. Scope to build

### 4.1 `config/dsl.yaml`

`version: 1`; `max_reply_chars: 4000`; `max_steps: 32`. Read by `load_dsl_config()`, which
rejects values that are not positive whole numbers.

The operations, the constants, and their meanings are a fixed table in the code, not
configuration: changing one changes what "correct" means. A test checks that table against
`config/data.yaml` and against the names in the system text.

### 4.2 Errors: `qlora_finetuning/dsl/errors.py`

`DslError` carries `code`, `stage` (`reply`, `program`, or `execution`), a message, and where
known the position in the text. A message quotes at most 40 characters of the reply.

| Stage | Codes |
|---|---|
| Reply | `reply_too_long`, `reply_empty`, `no_program` |
| Program | `empty_program`, `too_many_steps`, `syntax`, `unknown_operation`, `nested_call`, `wrong_arity`, `bad_argument`, `unknown_constant`, `bad_reference`, `bad_row_name` |
| Execution | `division_by_zero`, `numeric_error`, `yes_no_operand`, `table_missing`, `row_not_found`, `row_not_numeric` |

### 4.3 Reply parser: `qlora_finetuning/dsl/reply.py`

| Item | Behavior |
|---|---|
| `ReplyFormat` | The two labels. `ReplyFormat.from_prompt(prompt)` takes them from a `PromptDefinition` |
| `parse_reply(text, reply_format, config)` | D7 to D10. Returns a `ParsedReply`: `program_text`, `answer_text`, `strict`, `deviations` |

Deviations, each a fixed word: `text_before`, `text_between`, `text_after`, `label_case`,
`answer_missing`. A strict reply has none.

### 4.4 Program parser: `qlora_finetuning/dsl/program.py`

| Item | Behavior |
|---|---|
| `Argument` | `kind` (`integer`, `decimal`, `percent`, `reference`, `constant`, `row_name`, `none`: the seven kinds of `config/data.yaml`), `text` as written, and `value` (a float, a step number, or nothing) |
| `Step`, `Program` | A step is an operation and two arguments; a program is its steps. All frozen |
| `parse_program(text, config)` | D11 to D17 |
| `format_program(program)` | D18 |

### 4.5 Executor: `qlora_finetuning/dsl/executor.py`

| Item | Behavior |
|---|---|
| `execute(program, table=None)` | D19 to D27. Returns an `Execution`: `value` (a float, or `yes` / `no`), `step_values`, `notes` |
| `row_numbers(table, row_name)` | D24 to D26: the numbers of the named row, and whether the name was repeated |

### 4.6 One call for the runner and the demo

`run_reply(text, table, reply_format=None, config=None)` in `qlora_finetuning/dsl/__init__.py`
parses the reply, parses the program, and executes it. It returns a `ReplyOutcome` and never
raises on any string:

`strict`, `deviations`, `program_text`, `answer_text`, `program` (canonical form, when it
parsed), `value`, `step_values`, `notes`, `error_code`, `error_stage`, `error_message`.

Usage, on an invented table:

```python
from qlora_finetuning.dsl import run_reply

table = [["", "2024", "2023"], ["revenue", "$ 920", "$ 95"], ["margin", "12.5%", "10%"]]

outcome = run_reply("PROGRAM: subtract(920, 95), divide(#0, 95)\nANSWER: 8.68421", table)
outcome.strict          # True
outcome.value           # 8.684210526..., not rounded
outcome.program         # "subtract(920, 95), divide(#0, 95)"

run_reply("PROGRAM: table_average(margin, none)\nANSWER: 0.1125", table).value   # 0.1125
run_reply("PROGRAM: divide(5, 0)\nANSWER: 0", table).error_code       # "division_by_zero"
run_reply("Sure!\nprogram: add(1, 2)", table).deviations
# ("text_before", "label_case", "answer_missing"); value 3.0
```

### 4.7 Tests

On invented programs and tables. No test needs the network, a GPU, or the training stack.

- Reply: a strict reply; each allowance of D7; each deviation alone and combined; a fenced
  reply; two program lines (the first wins); no program line; an empty reply; a reply over the
  length limit.
- Program: each operation; free spacing; every number form of D14 and each rejected form;
  each of the seventeen constants and an unknown one; references that are forward, to
  themselves, and with thousands of digits; one and three arguments; a nested call; an
  unknown, capitalized, and dotted operation name; a missing bracket; a trailing comma; text
  after the last step; an empty program; more steps than the limit.
- Row names: with spaces, a comma, brackets, nested brackets, an unbalanced bracket, an empty
  name; a number as second argument.
- Round trip: `format_program(parse_program(text))` gives the canonical form, and parsing
  that again gives the same program.
- Executor: each arithmetic operation with a hand-computed value; chaining through `#k`; a
  percent literal; `const_m1`; `greater` both ways and with equal numbers; division by zero;
  overflow; a negative number to a fractional power; a `yes` / `no` operand; `greater` in the
  middle of a program.
- Table operations: each of the four on a hand-computed row; cells with `$`, a thousands
  separator, a bracketed part, a percent sign; a text cell, an empty cell, and a bracketed
  negative `( 9 )` each give `row_not_numeric`; a missing row; no table; a repeated row name
  takes the last row and adds the note; the header row can be named.
- Hostile input: attribute access, dunder names, quotes, semicolons, line breaks inside a
  program, non-ASCII digits, a 100,000-character reply. Each gives a `DslError` code, and a
  test confirms nothing was evaluated.
- No leaks: 2,000 replies built by a seeded generator that mutates valid replies (deleting,
  repeating, and swapping characters) all return an outcome from `run_reply`.
- Guard: no `eval`, `exec`, `compile`, `literal_eval`, or `__import__` in the package source;
  no import of `qlora_finetuning.data` at module level.
- Language table: the ten operations and seventeen constants equal those of
  `config/data.yaml`; each name appears in the system text of `config/prompt.yaml`.
- On the real data when the snapshot is present (skipped otherwise, so CI stays green):
  every fit and dev target is strict; every program parses and equals its canonical form; the
  operation counts, program lengths, and argument kinds equal those of the split report for
  fit and for dev (v1.01 D14); every program executes without an error.

### 4.8 Documentation

- `reqspecs/references/project_patterns/`: a note on scoring a reply: call `run_reply`, store
  the code, never parse a reply anywhere else.
- `gotchas.md`: the repeated-row-name row becomes decided; add that 14% of row names hold
  brackets, that percent cells are divided by 100, and that `exe_ans` is not uniformly rounded
  to five decimal places.
- `coding.md`: reply text is parsed and executed only through `qlora_finetuning.dsl`.
- `high_level_plan.md` §4.4: format validity is the strict flag; execution uses the recovered
  program (if the owner accepts item 1 of section 8).
- `architecture/architecture_overview.md`: no change expected; checked when built.

## 5. Out of scope

- Comparing a value with the gold answer: rounding, tolerance, percent and scale
  normalization, reading the answer line as a number (v2.02).
- The four metrics and the gold self-check report over the whole dataset (v2.02).
- The `Predictor` protocol, the runner, the predictions file (v2.03).
- Repairing a program: no spelling correction, no guessing a row name, no reading a nested
  program.
- A command-line entry point. The demo is v6.01.

## 6. What this hands to later versions

| For | What |
|---|---|
| v2.02 | `run_reply` and `ReplyOutcome`; the `strict` flag for format validity; the parsed program for exact match; unrounded values. Section 2.4: gold results are not uniformly rounded (five or six significant digits in 355 examples, scientific notation in 83), so the tolerance needs an absolute part as well as a relative one. (Corrected by v2.02: the first wording asked only for "a relative 0.0005%", which holds for the 438 results of section 2.4 but not for the 4,945 that equal the gold one only after rounding to five decimal places) |
| v2.02 | The gold self-check over all 8,281 examples needs the test rows' programs, which only code inside `qlora_finetuning/data/` may read. v2.02 decides how |
| v1.03 D16 | Confirmed on fit and dev: the percent literal and every operation mean what the system text says |
| v2.03 | Error codes and deviations to store per example; `run_reply` never raises |
| v3.01, v4.01 | Anything model-specific in a reply is removed by the backend before `run_reply` (D10) |
| v3.05 | Error codes, deviations, and the `duplicate_row` note as the first cut of an error analysis |
| v6.01 | `run_reply` on a table that is not from the dataset |

## 7. Known limits and follow-ups

- **The duplicate-row rule rests on the published evaluator, not on a measurement** (D25). It
  matters for at most the 241 tables with a repeated name, and only when a program names
  that row.
- **7.6% of data rows cannot be used by a table operation** (D26): a text cell, an empty
  cell, or a bracketed negative makes the whole row an error. No gold program is affected. A
  model that names such a row is scored wrong.
- **The cell rule is tested on 198 gold steps only.** It is right for every one of them and
  unexamined beyond.
- **Recovery can read a program the model did not mean as its answer**, for example a first
  attempt followed by a corrected one (D8 takes the first). The strict flag shows how often
  recovery was used.
- **Floats, not decimals.** A result can differ from exact arithmetic in its last digits.
  v2.02's tolerance covers it.
- **The figures in section 2 come from a throwaway script.** Any difference in the built
  code is investigated before either is accepted.
- **Test-split programs are not parsed here.** v2.02's self-check is the first full pass.

## 8. Decisions for the owner

All three were accepted on 2026-11-15, when the owner proceeded to the PRP.

| # | Question | Recommendation |
|---|---|---|
| 1 | **Reply leniency**: recover the program from a reply that breaks the two-line format (D8), or score every such reply as wrong? | Recover, and report format validity beside it. Execution accuracy then measures the calculation, and the strict flag measures obedience to the layout. Strict-only is the purer reading of "exactly two lines" and would mostly lower the two untuned arms. The code supports both; v2.02 reports the one chosen as the headline |
| 2 | **A repeated row name**: take the last row (D25), take the first, or make it an error? | The last, because the gold answers were computed that way. An error is the cautious choice but would mark the gold program of one fit or dev example wrong |
| 3 | **A constant outside the seventeen**, such as `const_12`: an error (D15), or read as the number 12? | An error. The prompt lists the constants and accepts any plain number, so `12` is always available to the model |

## 9. Acceptance

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with and
  without the training stack installed.
- With the snapshot present, the real-data test passes: 5,383 strict targets; 5,383 programs
  equal to their canonical form; operation counts, program lengths, and argument kinds equal
  to the split report; 5,383 programs executed without an error.
- The trial figures of section 2.4 are reproduced by the built executor: 4,945 results equal
  to `exe_ans` at five decimal places and 438 within a relative 0.0005%. Checked once by hand
  when built and recorded in the implementation summary; the standing check is v2.02's.
- The guard test finds no `eval`, `exec`, `compile`, `literal_eval`, or `__import__` in
  `qlora_finetuning/dsl/`.
- No committed file contains real example text; tests use invented programs and tables.

## 10. Next step

Built 2026-11-15. The trial figures of section 2.4 were reproduced exactly by the built code:
5,383 strict targets, 5,383 canonical programs, no execution error, 4,945 results equal to
`exe_ans` at five decimal places and 438 within a relative 0.0005%. The built cell rule reads
every one of the 98,247 fit and dev value cells as the trial rule did.

Complete: the owner ran the tests (1,045 passed) and the lint locally on 2026-11-15. A pull
request is no longer needed to mark a version complete (`coding.md`, Task Completion). Next:
the v2.02 spec (metrics and gold self-check).
