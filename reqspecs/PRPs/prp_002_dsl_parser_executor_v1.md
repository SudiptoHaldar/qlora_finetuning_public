# PRP: DSL parser and safe executor (req_002 v2.01)

**Feature Name**: dsl_parser_executor (dsl_evaluation v2.01)  
**Version**: 1  
**Created**: 2026-11-15  
**Status**: ✅ IMPLEMENTED  
**Implemented**: 2026-11-15  
**Implementation Summary**: [impl_002_20261115_225903.md](../impl_summary/impl_002_20261115_225903.md)  
**Confidence Score**: 9/10  
**Estimated Time**: 3–4 hours  
**Origin**: `reqspecs/reqs/req_002_dsl_evaluation/req_002_01_spec.md`  
**Template**: `prp_lean.md` (expanded, medium complexity)  

---

## Goal

Fill `qlora_finetuning/dsl/` with a reply parser, a program parser, and an executor, and one
call, `run_reply(text, table)`, that turns any model reply into a value or a named error
without ever evaluating reply text as code.

## Why & What

- **User value**: one definition of "correct" for all six arms, safe on any model output,
  with a reason recorded for every reply that fails.
- **Technical**: one config file, six small modules, tests on invented programs and tables,
  one real-data test that skips without the snapshot. No script, no report, no new dependency.

The owner proceeded to this PRP, which accepts the three items of spec section 8 as
recommended: recover the program from a reply that breaks the format; a repeated row name
means the last row; a constant outside the seventeen is an error.

## Success Criteria

- [ ] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, also
      with the training libraries blocked.
- [ ] Real-data test (snapshot present): 5,383 strict targets; 5,383 programs equal to their
      canonical form; operation counts, program lengths, and argument kinds equal to
      `reports/data/finqa_split_report.json` for fit and for dev; 5,383 run without an error.
- [ ] Checked once by hand and recorded in the implementation summary: 4,945 results equal
      `exe_ans` at five decimal places, 438 are within a relative 0.0005%.
- [ ] Guard test: no `eval`, `exec`, `compile`, `literal_eval`, `__import__` call in the
      package; no module-level import of `qlora_finetuning.data`.
- [ ] No committed file holds real example text.

---

## Context

### References

```yaml
refs:
  - reqspecs/references/project_patterns/canonical_record.md  # the reply format; labels in config/prompt.yaml
  - reqspecs/references/project_patterns/split_loading.md     # load_split for the real-data test
  - reqspecs/references/tech/finqa_dataset.md                 # programs, constants, table operations
  - reqspecs/references/common/gotchas.md                     # sections 4 and 5

code:
  - file: qlora_finetuning/data/serialization.py
    why: load_prompt() and PromptDefinition.program_label / answer_label; frozen dataclass style;
         an error class with a clear message
  - file: qlora_finetuning/data/token_budget.py
    why: a load_*_config function over settings.load_config with value checks
  - file: qlora_finetuning/data/schema_report.py
    why: split_program and argument_kind, the counting pattern the parser must agree with
  - file: tests/data/test_serialization.py
    why: the real-data test that skips without the snapshot (verify_snapshot)
  - file: tests/test_process_scaffold.py
    why: an ast-based scan of package source, the shape of the guard test
```

No web research was needed: the package uses the standard library only.

### Facts verified 2026-11-15

| Fact | Value |
|---|---|
| Spec section 2 | Every figure the real-data test asserts; measured with a trial executor |
| Split report | `columns.fit` and `columns.dev` hold `operations`, `program_length` (keys are strings), `argument_kinds` |
| Argument kinds in the split report | For a table step: `row_name` for the first argument and the kind of the second (`none`); fit has 188 of each |
| `config/data.yaml` | `finqa.expected.operations`, `.constants`, `.argument_kinds` |
| `tests/test_package.py` | Each sub-package docstring must contain `req_` |
| Runbook | Sections 5 to 7 are placeholders for req_003 to req_005; req_002 has none yet |

### Critical Gotchas

- **`float()` accepts more than a number**: `inf`, `nan`, `1e5`, `1_000`, `+5`, Unicode
  digits. Check the text by hand against ASCII digits first, then convert. A very long digit
  string converts to `inf`: reject a literal that is not finite.
- **`int()` refuses strings over 4,300 digits.** A reference with more than six significant
  digits is `bad_reference` without converting.
- **`str.isdigit()` is true for non-ASCII digits.** Compare against `"0123456789"`.
- **`**` returns a complex number** for a negative base and a fractional exponent. Use
  `math.pow`, which raises `ValueError` and `OverflowError`; map both to `numeric_error`.
- **Float multiplication overflows silently to `inf`.** Check every result with
  `math.isfinite`.
- **`str.splitlines()` splits on more than line breaks.** Normalize `\r\n` and `\r` to `\n`
  and split on `\n`.
- **Error messages quote reply text**: at most 40 characters, with non-ASCII escaped, because
  the console is cp1252.
- **Table steps cannot be over-arity**: every comma before the last belongs to the row name.
- **The real-data test must skip in CI**: `verify_snapshot(load_data_config())`, as in
  `tests/data/test_serialization.py`. Import `qlora_finetuning.data` in tests freely; in the
  package only inside a function.
- **Shell heredocs halve backslashes** here: write files with the file tool.

---

## Implementation

### Files

```yaml
CREATE:
  - config/dsl.yaml: version, max_reply_chars 4000, max_steps 32
  - qlora_finetuning/dsl/errors.py: DslError, ReplyError, ProgramError, ExecutionError, CODES, excerpt
  - qlora_finetuning/dsl/config.py: DslConfig, load_dsl_config
  - qlora_finetuning/dsl/language.py: OPERATIONS, TABLE_OPERATIONS, CONSTANTS, ARGUMENT_KINDS, read_number
  - qlora_finetuning/dsl/reply.py: ReplyFormat, ParsedReply, DEVIATIONS, parse_reply
  - qlora_finetuning/dsl/program.py: Argument, Step, Program, parse_program, format_program
  - qlora_finetuning/dsl/executor.py: Execution, execute, row_numbers, cell_number
  - qlora_finetuning/dsl/outcome.py: ReplyOutcome, run_reply
  - tests/dsl/test_reply.py, test_program.py, test_executor.py, test_outcome.py,
      test_safety.py, test_language.py, test_real_data.py
  - reqspecs/references/project_patterns/reply_scoring.md
  - reqspecs/impl_summary/impl_002_<TIMESTAMP>.md

MODIFY:
  - qlora_finetuning/dsl/__init__.py: docstring, public names
  - docs/runbook.md: a section for the evaluation harness
  - reqspecs/1. PROMPTS/coding.md, gotchas.md, project_patterns/README.md, high_level_plan.md (4.4)
  - req_002_01_spec.md (status, section 8, any difference), req_002_roadmap.md, execution_plan.md
```

### Design

```python
# errors.py
CODES = {code: stage}       # the nineteen codes of spec 4.2, each with its stage
class DslError(Exception):  # .code .stage .position ; message quotes <= 40 characters
class ReplyError(DslError); class ProgramError(DslError); class ExecutionError(DslError)

# language.py
OPERATIONS = ("add", "subtract", "multiply", "divide", "exp", "greater",
              "table_sum", "table_average", "table_max", "table_min")
CONSTANTS = {"const_1": 1.0, ..., "const_1000000000": 1e9, "const_m1": -1.0}
def read_number(text) -> float | None   # -?(digits(.digits)? | .digits), ASCII, finite

# reply.py
def parse_reply(text, reply_format, config) -> ParsedReply
#  1. longer than max_reply_chars -> reply_too_long; blank -> reply_empty
#  2. split into lines; trim spaces and tabs; drop blank lines
#  3. p = first line starting with the program label (any case); none -> no_program
#  4. a = first later line starting with the answer label (any case)
#  5. deviations: text_before, text_between, text_after, label_case, answer_missing
#  6. strict = no deviation

# program.py
def parse_program(text, config) -> Program
#  loop: skip blanks; name = text up to "("; not one of ten -> unknown_operation
#        (empty, or holding "," or ")" -> syntax); over max_steps -> too_many_steps
#    arithmetic: read to the first ")"; a "(" before it -> nested_call; split on ",";
#                not two parts -> wrong_arity; each part -> _argument(part, step_index)
#    table:      track bracket depth to the matching ")"; remember the last comma at depth 1;
#                none -> wrong_arity; name before it (empty -> bad_row_name); second after it
#                ("none", or _argument)
#    then: end of text -> done; "," -> another step must follow; anything else -> syntax
def _argument(text, step_index) -> Argument
#  "#digits" -> reference (k < step_index, else bad_reference); "const_..." -> constant or
#  unknown_constant; "...%" -> percent, value number / 100; number -> integer | decimal;
#  anything else, including "none" and "" -> bad_argument
def format_program(program) -> str      # "op(a, b), op(a, b)" from the kept texts

# executor.py
def cell_number(cell) -> float | None   # D26
def row_numbers(table, row_name) -> tuple[list[float], bool]   # last matching row; repeated?
def execute(program, table=None) -> Execution   # value, step_values, notes

# outcome.py
def run_reply(text, table=None, reply_format=None, config=None) -> ReplyOutcome
#  defaults load config/dsl.yaml and the labels of config/prompt.yaml once per process;
#  catches DslError only and returns it as error_code / error_stage / error_message
```

Two refinements of the spec, to be written back into it:

- **Blank lines and spaces or tabs at the start of a line do not break strictness.** The spec
  names five deviations and none of them is "a blank line"; such a reply holds nothing but the
  two fields.
- **An answer line with nothing after the label counts as `answer_missing`.**

### Tests

| File | Cases |
|---|---|
| `test_reply.py` | Strict reply; each allowance; each deviation alone and combined; fenced reply; two program lines; no program line; empty; over the limit; `ReplyFormat.from_prompt` |
| `test_program.py` | Each operation; spacing; number forms accepted and rejected; all constants and an unknown one; references forward, self, huge; arity; nested call; unknown, capitalized, dotted names; missing bracket; trailing comma; trailing text; empty; too many steps; row names with spaces, comma, brackets, nested brackets, unbalanced, empty; number as second argument; round trip |
| `test_executor.py` | Each operation by hand; chaining; percent; `const_m1`; `greater` three ways and mid-program; division by zero; overflow; negative to a fractional power; `yes`/`no` operand; each table operation; cell forms; `row_not_numeric` three ways; missing row; no table; repeated name; header row |
| `test_outcome.py` | The spec's usage example; an error at each stage; defaults load; `to_dict`; 2,000 seeded mutations never raise |
| `test_safety.py` | Hostile inputs give a code and leave no trace; 100,000-character reply; non-ASCII digits; ast guard over the package source |
| `test_language.py` | Operations, constants, kinds equal `config/data.yaml`; names appear in the system text; config validation |
| `test_real_data.py` | Spec 4.7, last item; skips without the snapshot |

### Tasks (~3.5 hours total)

1. **Config, errors, language** (~25 min).
2. **Reply parser and tests** (~30 min).
3. **Program parser and tests** (~50 min).
4. **Executor and tests** (~40 min).
5. **`run_reply`, safety and fuzz tests, language test** (~30 min).
6. **Real-data test; hand check of the 4,945 / 438 figures and of the cell rule against the
   trial rule on every fit and dev cell** (~15 min).
7. **Documents** (~25 min).
8. **Validation with the stack blocked, status documents, summaries** (~15 min).

### Error handling

| Situation | Behavior |
|---|---|
| Anything wrong in reply text | A `DslError` subclass with one of the nineteen codes; `run_reply` returns it in the outcome |
| `text` is not a string | `TypeError`: a caller's mistake, not a reply |
| `config/dsl.yaml` value not a positive whole number | `ConfigError` naming the key |
| A bug in the package (any other exception) | Not caught. The fuzz test exists to find these |

---

## Validation

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q
venv\Scripts\python -m pytest tests\dsl -q
```

**Expected**: ruff clean; all tests pass; the real-data test runs (not skipped) on this
machine. Also the unit tests with the training libraries blocked.

---

## Checklist

- [ ] Files created and modified as listed
- [ ] Every error code is raised by at least one test
- [ ] Lint clean; tests green with and without the training stack
- [ ] Real-data figures reconciled with spec section 2
- [ ] Runbook, references, plan, `coding.md` updated
- [ ] Roadmap, execution plan, spec status updated; implementation summary created

---

## Implementation Time Estimate

**Total: 3–4 hours**

| Task | Time |
|---|---|
| Config, errors, language | 25 min |
| Reply parser and tests | 30 min |
| Program parser and tests | 50 min |
| Executor and tests | 40 min |
| `run_reply`, safety, fuzz, language tests | 30 min |
| Real-data test and hand check | 15 min |
| Documents | 25 min |
| Validation and status documents | 15–30 min |

## Confidence Score: 9/10

The rules were tried on all 5,383 fit and dev programs before the spec was written and
reproduced every gold result. The point held back is for the hand-written scanner's edge
cases on malformed input, which the fuzz test is there to shake out.

## Time Taken for PRP Generation

**~8 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 2 min |
| Research existing PRPs and patterns | 1 min |
| Writing PRP document | 3 min |
