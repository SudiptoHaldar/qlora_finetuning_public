# Packaging Demo

Spec Group      :   req_006_packaging_demo  
Feature Name    :   packaging_demo  
Version         :   1  
Status          :   ✅ Built and verified 2026-11-23 (section 11); the five decisions of section 8 taken as recommended; committed  
Last updated    :   2026-11-23  

## 1. v6.01 — Demo CLI

The project has a tuned adapter (arm A1) and an executor that runs a model's program
safely. This version puts the two behind one command, so that a person can hand the model
a report excerpt, a table, and a question, and see what comes back:

1. the program the model proposes,
2. the result of running that program in the project's executor,
3. the answer the model wrote itself, and whether it agrees with the executed result.

```powershell
venv\Scripts\python -m qlora_finetuning.demo
```

With no argument it answers one bundled sample input, with the selected adapter, on the
local GPU.

The demo is a way to look at the model. It is not a measurement: it scores nothing, has no
gold answer, writes no record, and nothing it prints goes into a report.

**Depends on**: v3.04 (the selection record and the selected checkpoint; complete
2026-11-21) and v2.01 (`run_reply`; complete). It does not wait for v5.02 or v5.03.

## 2. What exists today (verified 2026-11-23)

### 2.1 The pieces the demo joins

| Piece | Where | What the demo uses it for |
|---|---|---|
| `load_prompt()`, `format_user(passages, table, question, prompt)` | `data/serialization.py` | The system text and the user text for input that is not a dataset example. The pattern file `canonical_record.md` already names the demo as a user |
| `build_tuned_predictor(adapter, context)` | `local_qlora/predictor.py` | The pinned model in 4-bit with an adapter named `selected`, `RUN_ID`, or `RUN_ID/checkpoint-N`. It checks the adapter's model commit, prompt wording, and split, and for `selected` the hash the selection was made on. It reads three things from `context`: `prompt`, `split_cfg`, `tracking_cfg` |
| `build_predictor()` | the same | The untuned model |
| `QwenPredictor.predict([PredictionRequest])` | the same | One reply, with token counts, seconds, and the stop reason |
| `load_token_budget_config()` | `data/token_budget.py` | The reply limit (`max_new_tokens`, 192) and the training length (`max_length`, 2,048) |
| `run_reply(text, table)` | `dsl/outcome.py` | The program, each step's value, the final value, or a named error. Never raises on any string |
| `read_answer`, `values_match`, `load_evaluation_config()` | `evaluation/metrics.py` | Reading the number on the answer line and comparing two values under the allowances of `config/evaluation.yaml` |

No new model code, no new parsing, and no new comparison rule is needed.

### 2.2 The adapter

- The selection record names `20261120T182710Z-train-qwen3-4b-fit-4557/checkpoint-500`.
- That checkpoint's two files are on the owner's machine today, under
  `artifacts/runs/<run>/checkpoints/checkpoint-500/`, from `sagemaker_job pull --checkpoint 500`.
- On a machine without them, `build_tuned_predictor` already stops with a message that
  names what is missing. The pull needs an AWS sign-in and reads only.
- The base model is loaded from the local Hugging Face cache and is never downloaded by a
  run (`local_files_only=True`). `env_check` names the download command.

### 2.3 What is missing

- `qlora_finetuning/demo/` holds one line, a docstring. `tests/demo/` holds an empty
  `__init__.py`.
- `build_tuned_predictor` takes a runner's context. The demo has no runner.
- Nothing compares a reply's answer line with its own program's result. `score_reply`
  compares each of them with a gold value, which the demo does not have.
- There is no sample input that is safe to commit. Every FinQA row may be a test example.

## 3. Locked decisions

### What the demo is

| # | Decision | Reason |
|---|---|---|
| D1 | The entry point is `python -m qlora_finetuning.demo` (`demo/__main__.py`), not a file under `scripts/` | The plan names it. It is the one command a reader of the model card is given, and it belongs to the package |
| D2 | The demo answers with Qwen only: the selected adapter by default, another adapter by name, or the untuned model. It asks no Nova model (owner's decision 1) | A Nova call is billed and needs the deployments kept |
| D3 | The demo is not an evaluation. It has no gold answer, computes no metric, takes no split name and no example id, and never reads the dataset | The test split is reachable only through the gate. A demo that could load an example would be a way round it |
| D4 | The demo starts no tracked run and writes no file: no registry entry, no ledger line, no predictions file (owner's decision 3) | It is local, free, and evidence of nothing. The rule on tracking covers training and evaluation entry points |
| D5 | The demo calls no AWS service and imports no boto3 module. A missing adapter is an error that names the pull command (owner's decision 5) | A demo that needs a sign-in is not a demo. The pull already exists |

### Input

| # | Decision | Reason |
|---|---|---|
| D6 | Input is a JSON file: `question` (text), `table` (a list of rows, each a list of cells, the first row the header), and `context` (a list of passages, which may be empty or left out). `--input` takes one or more files (owner's decision 4) | A table does not fit on a command line. The shape is the one `format_user` and `run_reply` already take |
| D7 | A file with no table, an empty table, a missing question, or a value of the wrong type is refused before the model loads, with the file and the field named. Unknown fields are refused too | The prompt has no wording for "no table", and none may be written in code. A misspelt field must not be dropped silently |
| D8 | The sample input is `qlora_finetuning/demo/sample_input.json`: an invented company, invented figures, written for this project. It is not a FinQA row and is not adapted from one | No dataset text is committed. v6.02's model card uses the same sample in its usage snippet |
| D9 | The sample asks for a percentage change read from two table cells: two steps, one reference to an earlier step. The file also holds the expected result, under `expected`, which the demo ignores and the acceptance check reads | A sample the reader can check by hand. FinQA's most common question type |

### Asking the model

| # | Decision | Reason |
|---|---|---|
| D10 | The system text is `load_prompt().system` and the user text is `format_user(...)`, unchanged. The reply limit is `max_new_tokens` from `config/token_budget.yaml`. Thinking is off, decoding is greedy: whatever `QwenPredictor` does | One prompt for every use of the model. The demo adds no wording, no example, no stop string |
| D11 | The adapter is named, never given as a path: `--adapter selected` (the default), `--adapter RUN_ID`, `--adapter RUN_ID/checkpoint-N`. `--untuned` asks the base model. The two flags exclude each other | The project's rule. `selected` is the only name that is arm A1 |
| D12 | The demo gives `build_tuned_predictor` a small context of its own (prompt, split configuration, tracking configuration) and does not import the runner | The v2.03 hand-off: the demo needs a predictor, not the loop |
| D13 | Each input is answered alone, in a batch of one. The model is loaded once for all the files of one command | A reply can depend on its batch. One at a time is what a person asking one question gets |
| D14 | The prompt is never truncated. A prompt that, with the reply limit, is longer than `max_length` (2,048 tokens) is still answered, with a warning line saying that the adapter saw no training sequence that long (owner's decision 2) | Nothing in the project cuts a prompt. The test run refused no example for length either |

### Running the reply

| # | Decision | Reason |
|---|---|---|
| D15 | The reply goes to `run_reply(text, table)` with the table of the input file, and nowhere else. The demo parses no reply text | The one place where reply text is read or executed |
| D16 | The model's own answer is compared with the executed result by one new function in `evaluation/metrics.py`, `answer_agrees(outcome, config)`: `read_answer` on the answer line, `values_match` against the executed value, under the `answer` allowance. It returns yes, no, or "not comparable" (the program did not run, or there is no answer line) | No comparison is written outside the metrics. The function is not a fifth metric: no run summary reports it |
| D17 | Values are printed as they are, unrounded. A percentage is not multiplied by 100 and no unit is added | The executor does not round, and nothing outside the metrics may |
| D18 | A program that does not run is a result, not a crash: the output shows the error code, its stage, and its message, and the model's own answer marked as unchecked | Showing a refusal is half the point of a safe executor |

### Output

| # | Decision | Reason |
|---|---|---|
| D19 | The result goes to standard output and log lines to standard error. Text is the default; `--json` prints one JSON object per input, one per line, with the whole `ReplyOutcome` | The project's rule for scripts. JSON lets v6.02's clean-load check read the result |
| D20 | Output is ASCII: any other character is written as a backslash escape, in text and in JSON | The Windows console uses cp1252 (`gotchas.md`); a currency sign in a row name must not end the demo |
| D21 | The exit code is 0 when every input's program ran, 1 when at least one did not, and 2 when the demo stopped before asking the model (bad input, no adapter, no GPU, no weights) | A caller can tell "the model was wrong" from "the demo could not start" |
| D22 | Every refusal before the model is a `ConfigError` printed as one line that names the command to run next. No traceback | As in every other entry point |
| D23 | `--check` validates the input files and resolves the adapter, then stops without loading the training stack | It runs in the test suite and on a machine with no GPU |

## 4. Scope to build

New code under `qlora_finetuning/demo/`:

| Module | Holds |
|---|---|
| `inputs.py` | `DemoInput` (question, table, passages, source file); `load_input(path)` with the checks of D7; `sample_path()` |
| `answer.py` | `DemoResult`; `answer(predictor, demo_input, ...)`: build the two texts, ask, run the reply, compare. Takes any `Predictor`, so tests pass a stub |
| `render.py` | `as_text(result)` and `as_json(result)`: ASCII only |
| `__main__.py` | The arguments, the predictor, the loop over inputs, the exit code |
| `sample_input.json` | The sample of D8 and D9 |

One function added to `qlora_finetuning/evaluation/metrics.py`: `answer_agrees` (D16).
No new config file: the demo has no tunable number of its own.

### 4.1 The commands

```powershell
venv\Scripts\python -m qlora_finetuning.demo                                  # the sample, the selected adapter
venv\Scripts\python -m qlora_finetuning.demo --input my_question.json
venv\Scripts\python -m qlora_finetuning.demo --input a.json b.json --json
venv\Scripts\python -m qlora_finetuning.demo --untuned                        # the base model, for contrast
venv\Scripts\python -m qlora_finetuning.demo --adapter RUN_ID/checkpoint-N    # another adapter, by name
venv\Scripts\python -m qlora_finetuning.demo --check --input my_question.json # no model is loaded
```

### 4.2 An input file

```json
{
  "context": ["Northfield Tools reported higher revenue in 2024 after opening two plants."],
  "table": [
    ["", "2024", "2023"],
    ["revenue", "$ 1,150", "$ 920"],
    ["operating income", "$ 138", "$ 101"]
  ],
  "question": "What was the percentage change in revenue from 2023 to 2024?"
}
```

The figures above are invented and show the shape only. The committed sample is written in
the PRP and also carries `expected`.

### 4.3 What is printed

```
Input       sample_input.json
Model       qwen3-4b-tuned  20261120T182710Z-train-qwen3-4b-fit-4557/checkpoint-500
Question    What was the percentage change in revenue from 2023 to 2024?
Program     subtract(1150, 920), divide(#0, 920)
Steps       #0 = 230.0   #1 = 0.25
Executed    0.25
Model said  0.25   agrees with the executed result
Format      exact
Tokens      prompt 431, reply 24, stopped at the end token
Time        1.8 s
```

The values in this block are illustrative. When the program does not run:

```
Program     divide(138, 0)
Executed    nothing: division_by_zero (execution) - step 0 divides by zero
Model said  0.12   not checked: the program did not run
```

Lines shown only when they apply: `Format` lists the deviations when the two-line format
was not kept; a warning when the reply stopped at the limit; the length warning of D14;
the executor's notes.

Before the first result, one line on standard error says which model and adapter are being
loaded and that the output is a demonstration, not a measured result.

### 4.4 Tests

No GPU, no network, no credentials, invented input only, a stub predictor.

- Input: the sample loads; a file without a table, with an empty table, without a
  question, with a cell that is a list, with an unknown field, or that is not JSON is
  refused with the file and field named; passages may be absent.
- Prompt: the system text is `load_prompt().system`; the user text equals
  `format_user(...)` character for character; the request carries the reply limit of the
  token budget.
- Answer: a stub reply that runs gives the value, the step values, and "agrees"; an answer
  line outside the allowance gives "does not agree"; `yes` and `no` are compared as words;
  a reply with no answer line and a program that does not run each give "not comparable".
- `answer_agrees`: uses the `answer` allowance of the configuration it is given; no run
  summary gains a field.
- Errors: every error code of `dsl.CODES` renders without an exception; a predictor that
  raises `PredictorError` ends with exit code 2 and one line.
- Output: text and JSON hold the same values; a non-ASCII row name is escaped; values are
  not rounded; the same result gives the same bytes.
- Exit codes: 0, 1, and 2 as in D21. `--adapter` with `--untuned` is refused.
- `--check`: passes on the sample with an adapter folder made in a temporary directory;
  fails when the checkpoint is absent, and the message names
  `sagemaker_job pull --checkpoint N`.
- The sample: a hand-written correct reply executes to the file's `expected`.
- Guards: no module under `demo/` imports the training stack or boto3 at module level, or
  imports `frontier`, `comparison`, `tracking.registry`, or the runner; none calls
  `load_split`, `serialize_split`, or the raw reader; none calls `eval`, `exec`, or
  `compile`.
- One test marked `gpu`: the sample, with `--adapter selected`, ends with exit code 0 and
  the executed value matches `expected`.

### 4.5 Documentation

- Runbook: a section on the demo. What it needs (the GPU, the cached base model, the
  pulled checkpoint), the commands, how to write an input file, what each line means.
- `reqspecs/1. PROMPTS/coding.md`: one rule for `demo/` (D3, D4, D5, D15, D16).
- `README.md`: the one command and a pointer to the runbook. The full README is v6.02.
- `architecture/architecture_overview.md`: `demo` also reads `data` (the serializer) and
  `evaluation` (the comparison); the drawing shows `local` and `dsl` only.
- `administration/aws_commands_run.md`: nothing. No AWS command is run.

## 5. Out of scope

- The Nova arms, tuned or untuned (owner's decision 1).
- Scoring, gold answers, accuracy over several inputs. That is `scripts.evaluate`.
- Loading a FinQA example by id, from any split.
- An adapter given as a folder path, loading from the Hugging Face Hub, and the merged
  copy. v6.02 builds its clean-load check on `answer(predictor, ...)` with its own loader.
- A prompt-and-wait loop, a web page, a notebook, an API server.
- Two models side by side in one command. Run the demo twice, once with `--untuned`.
- Reading a table from CSV, a PDF, or free text.
- Running on the CPU, on another GPU setup, or at another quantization.
- Downloading the base model or pulling the checkpoint.
- Saving what was asked or answered.

## 6. What this hands to later versions

| For | What |
|---|---|
| v6.02 | `sample_input.json` for the model card's usage snippet; `answer(predictor, demo_input)` and `--json` for the clean-load check of the adapter package and of the merged copy; the demo command for the README |
| v6.03 | The same check, repeated on the adapter loaded from the Hub |
| The teardown | With decision 1 taken as recommended, the demo needs neither tuned Nova model nor its deployment |

## 7. Known limits and follow-ups

- **The demo's reply is not always the test run's reply.** Arm A1 was scored at batch size
  8; the demo asks at batch size 1. In v3.03's measurement 2 or 3 of 24 tuned replies
  differed between the two. Both are greedy; neither is wrong.
- **One sample shows one success.** It was chosen to be answerable. The measured figure is
  the test result in the six-arm record: 2,121 of 2,898.
- **Input unlike FinQA is outside what the adapter learned**: a table with merged headers,
  a question that needs an operation the language lacks, a prompt longer than the training
  length. The demo answers anyway and the executor says what did not run.
- **"Agrees" is about the model's two lines, not about the truth.** A wrong program with a
  matching answer line agrees.
- **A user can type in any text, a test example included.** The demo cannot know. It keeps
  nothing, and an agent does not feed it dataset rows.
- **It needs this machine's setup**: a CUDA GPU with room for the 4-bit model, the training
  stack of `requirements-train.txt`, the cached base model at the pinned commit.
- **Loading takes far longer than answering.** Several files in one command share one load.

## 8. Decisions for the owner

**All five taken as recommended on 2026-11-23**: the owner started the PRP and its
execution without changing one. The table is kept as the record of what was asked.

| # | Question | Recommendation |
|---|---|---|
| 1 | **Does the demo also ask the Nova models?** `aws_commands_run.md` section 19.6 left this open: each tuned Nova model costs $1.95 a month while it exists, and every call is billed | No. Qwen only. The adapter is the artifact this project hands over; the Nova models cannot leave Bedrock. The deployments are then kept or deleted on v5.03's needs alone |
| 2 | **What happens to an input longer than the training length (2,048 tokens)?** | Answer it, with a warning line (D14). The alternative is to refuse it. Either way nothing is truncated |
| 3 | **Is a demo call recorded** in the run registry or the ledger? | No (D4). It costs nothing and proves nothing, and a registry full of demo calls hides the real runs |
| 4 | **How is input given?** | A JSON file only (D6). The alternative adds `--question`, `--context`, and a table file as flags, which is more to test for the same result |
| 5 | **What does the demo do when the selected checkpoint is not on the machine?** The roadmap's scope says "pulled from S3" | Stop and name the pull command (D5). The demo itself never signs in to AWS. On the owner's machine the checkpoint is already there |

## 9. Acceptance

Code, by Claude:

- `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, with
  and without the training stack blocked, with no network and no credentials.
- `python -m qlora_finetuning.demo --check` passes on the real repository: the sample is
  valid and the selected checkpoint is found with the hash of the selection record.
- No frozen file changes: `comparison_freeze status` reports nothing new.

On the GPU (seconds, free, no AWS; by Claude on the owner's machine, or by the owner):

1. `venv\Scripts\python -m qlora_finetuning.demo`: exit code 0; the program runs; the
   executed value matches the sample's `expected`; the model's answer agrees.
2. `venv\Scripts\python -m qlora_finetuning.demo --untuned`: ends without an exception,
   whatever the untuned model replies.
3. `venv\Scripts\python -m pytest -m gpu -q tests/demo`.

If the selected adapter gets the sample wrong, the sample is replaced by another invented
one and the roadmap says so; the adapter and the prompt are not touched.

## 10. Next step

The decisions are taken (section 8) and the PRP was generated and executed on 2026-11-23:

```
/generate-prp reqspecs/reqs/req_006_packaging_demo --version 1
```

## 11. As built (2026-11-23)

Built by `prp_006_demo_cli_v1.md`. Nothing is billed, no AWS call was made, and no frozen
value changed (`comparison_freeze status`: nothing frozen has changed).

### 11.1 What was built

| File | Holds |
|---|---|
| `qlora_finetuning/demo/inputs.py` | `DemoInput`, `load_input`, `read_input`, `sample_path`, `DemoInputError` |
| `qlora_finetuning/demo/answer.py` | `DemoResult`, `answer(predictor, demo_input)`, `adapter_name` |
| `qlora_finetuning/demo/render.py` | `as_text`, `as_json`, `as_dict`, `warnings`, `ascii_text` |
| `qlora_finetuning/demo/cli.py` | `main`, `demo_context`, `require_selected_here`; `__main__.py` calls `main` |
| `qlora_finetuning/demo/sample_input.json` | The invented sample, with `expected` |
| `qlora_finetuning/evaluation/metrics.py` | `answer_agrees(outcome, config)`, exported from the package |
| `qlora_finetuning/local_qlora/predictor.py` | `resolve_tuned_adapter`: the checks of `build_tuned_predictor` without the load. What either refuses or returns is unchanged |

Where the build differs from sections 3 and 4:

- **The command line is `demo/cli.py`**, and `__main__.py` only calls it, so that tests can
  call `main` with stand-ins.
- **A table cell that is a number becomes text when the file is read** (`1150` is `"1150"`),
  so the prompt and the executor see the same cell.
- **`expected` is a known field** of every input file, ignored by the demo (D9 asked for it
  in the sample only).
- **The text output has a `Reply` line** with the raw reply when the program did not run or
  the format was not exact, and a `Notes` line for the executor's notes.
- **The length warning is known after the reply**, from the prompt's token count that the
  backend reports. The demo loads no tokenizer of its own.
- **The pull command is named by the demo**, not by the adapter check: for `selected`, the
  demo looks for the checkpoint first and stops with
  `python -m scripts.sagemaker_job pull --checkpoint N`.

### 11.2 What the GPU runs showed

On the owner's machine (RTX 4090 Laptop GPU), 2026-11-23. The input is the invented sample.

| Command | Result |
|---|---|
| `demo --check` | The sample is valid; the selected checkpoint is found, weights `a8c9561b4fc6...`, as the selection record says |
| `demo` | Exit code 0. Program `subtract(4812, 4275), divide(#0, 4275)`; executed 0.1256140350877193, the sample's `expected`; the model said 0.12556, which agrees; format exact. 532 prompt tokens, 37 reply tokens; model loaded in 10.3 s, answer in 4.6 s |
| `demo --json` | The same reply, as one JSON object |
| `demo --untuned` | Exit code 1, no exception: the untuned model wrote a nested call, which the executor refused as `nested_call`; the reply is shown |
| `demo --show-input` | The same result, with the sample's two passages and four table rows printed before the question |
| `pytest -m gpu -q tests/demo` | 1 passed |

The sample did not have to be replaced.

`--show-input` was added after the first run, at the owner's request: the output named
the question but not the company, which is only in the context. With the flag, the text
has `Context` and `Table` lines before the question and the JSON has `context` and
`table`. Without it the output is as before. Three tests cover it.

### 11.3 Tests

| Run | Result |
|---|---|
| `venv\Scripts\python -m pytest -q` | 2,763 passed, 1 failed, 1 skipped, 11 deselected. The failure was this version's own (a guard of `tests/data` read a name in the demo's guard test as a use of the raw reader); fixed, and the two files pass (129 passed, 1 skipped). The whole suite was not run a third time |
| The same with the training stack blocked | 2,757 passed, 1 failed (the same test), 7 skipped; after the fix the two files pass |
| `venv\Scripts\python -m pytest -q`, before the commit (with `--show-input`) | 2,767 passed, 1 skipped, 11 deselected |
| `venv\Scripts\python -m ruff check .` | clean |

102 tests are new: 99 under `tests/demo/` (one of them GPU-marked) and 3 for
`answer_agrees`. The totals include the tests of v5.02, which was being built in the same
working tree.
