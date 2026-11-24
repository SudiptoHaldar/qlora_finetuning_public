# req_002 DSL Evaluation — Roadmap

**Spec group**: req_002_dsl_evaluation  
**Last updated**: 2026-11-16 — v2.01 complete (✅); v2.02 complete (✅); v2.03 complete (✅); the group is complete; M3 reached; 3 envisioned versions.  

Version log:

- **2026-11-13** — group created from `reqspecs/0. PLANNING/execution_plan.md` §2; `req_002_01_spec.md` is a placeholder.
- **2026-11-14** — plan revised before any version was specced: measured Qwen runs move to AWS SageMaker; the hosted path is Amazon Nova Pro and Nova Lite on Bedrock (OpenAI dropped); Q5 and Q6 resolved. Version scopes updated in place.
- **2026-11-15** — `req_002_01_spec.md` written for v2.01 (📝) after profiling the gold programs and tables of fit and dev: a trial executor ran all 5,383 gold programs without an error and matched every result. Found on the way: 14% of table row names hold brackets, so the row-name rule had to be defined; gold results are not uniformly rounded to five decimal places, which v2.02's tolerance must allow for.
- **2026-11-15** — the owner proceeded to the PRP, accepting the three decisions as recommended. `prp_002_dsl_parser_executor_v1.md` written and executed the same day (🟡): `qlora_finetuning/dsl/` holds the reply parser, the program parser, the executor, and `run_reply`; 406 new tests, 1,045 in total, lint clean, also with the training stack blocked. The built code reproduces the trial figures exactly. One finding while building: 18 table cells carry a plus sign, which the cell rule now accepts. One item remains: the CI run on the pull request.
- **2026-11-15** — v2.01 marked ✅. The owner ran the tests (1,045 passed) and the lint locally and decided that a pull request and its CI run are no longer needed to mark a version complete: local tests and lint, plus a commit, are enough (`coding.md`, Task Completion). Pull requests are opened when convenient.
- **2026-11-15** — `req_002_02_spec.md` written for v2.02 (📝) after measuring fit and dev with the built executor. Gold answers are rounded to five decimal places and to about six significant digits, so a relative tolerance alone fails on small results; with "absolute 0.00001 or relative 0.001%" all 5,270 numeric gold programs match. At the Gemini guide's 0.5%, ten times as many examples have a wrong operand that scores as right. This corrects v2.01's wording that every result was "within a relative 0.0005%" of the gold one (true for 438; the other 4,945 are equal at five decimal places); the v2.01 spec, the plan, and the gotchas file were corrected.
- **2026-11-15** — `prp_002_metrics_gold_check_v2.md` written and its execution started; the session crashed after the config, the metrics module, the self-check module, and the script were on disk, before any test or the real run.
- **2026-11-16** — the owner decided the three questions: **0.5% for execution accuracy** (the Gemini guide's figure, not the recommended tight rule), flagged examples kept in every total, and the self-check may read test rows. The build was resumed and finished (🟡): `config/evaluation.yaml`, `evaluation/metrics.py`, `data/gold_check.py`, `scripts/finqa_gold_check.py`; 192 new tests, 1,237 in total, lint clean, also with the training stack blocked. **Gold self-check over all 8,281 examples: every gold program reproduces its gold answer, in fit (4,884), dev (499), and test (2,898); no example is flagged**, so `splits/finqa_gold_flags.json` is empty. The same check under the tight rule flags nothing either. One defect of the half-built code was found and fixed on the way: the wording of check K5 depended on whether a flags file already existed, so a second run would have rewritten the report. One step remains: the commit.
- **2026-11-16** — v2.02 marked ✅: tests and lint green locally, and the owner asked for the commit. v2.03 is unblocked.
- **2026-11-16** — `req_002_03_spec.md` written for v2.03 (📝) after reading the tracking, serializer, and split code and trying the loop on dev: running and scoring 499 replies takes 0.03 seconds and a prediction line is about 825 bytes, so the predictions file holds everything about each reply and a summary is always recomputed from it. Found on the way: a run in the registry cannot be reopened, so a resumed evaluation is a new run that continues an earlier one; the ledger holds no record yet, so the first lines will come from this version's fake run; the runner must not be exported from the `evaluation` package, or the gold self-check's import of the metrics becomes a cycle.
- **2026-11-16** — the owner started the PRP and its execution without changing the three decisions, so they are taken as recommended. `prp_002_evaluation_runner_v3.md` written and executed the same day (🟡): `evaluation/predictor.py`, `fake.py`, `predictions.py`, `runner.py`, and `scripts/evaluate.py`; 155 new tests, 1,392 in total, lint clean, also with the training stack blocked. **The gold predictor scores 200 of 200 on quick-dev and 499 of 499 on dev on all four metrics, with checks R1 to R5 passing: milestone M3 (harness trusted) is reached.** Those two runs wrote the first lines of the committed ledger. Found while building: a process that is killed writes no ledger record, so the resuming run now rebuilds the earlier run's usage from its lines; a tampered line crashed the summary and now fails check R5 instead. One step remains: the commit.
- **2026-11-16** — v2.03 marked ✅ on the owner's instruction: tests and lint green locally. All three versions of this group are complete; v3.01 and v4.01 are unblocked.

This file is the **single source of truth** for the req_002 version sequence. The group owns
**the definition of "correct": the arithmetic DSL parser and safe executor, the metrics, and the model-agnostic evaluation runner. It extends `req_001_finqa_data`; it does NOT import either training path**.

## Why this group exists

Both guides insist that the model proposes a program and deterministic Python performs the
arithmetic. That executor is also the scorer, so it must exist and be trusted before any model
is measured. Four models (two paths, tuned and untuned) are scored by the same code through one
`Predictor` protocol.

Standing decisions: allowlisted operations only, no `eval`; program execution accuracy is the
primary metric; tolerance and normalization rules are fixed before the first model is scored;
the harness proves itself by executing every gold program against the gold answer.

**Status legend** (in lifecycle order):

- ⏸ **Conditional** — only starts when a specific trigger fires
- 📋 **Planned** — next-up; depends on a prior version landing
- 📝 **Spec ready** — req_002_NN_spec.md written + decisions locked; PRP pending
- 🛠 **PRP ready** — PRP authored + validated; execution pending
- 🚧 **In flight** — PRP being executed or impl in progress
- 🟡 **Operator-gated** — prepared; waiting on a step only the user performs
- ✅ **Complete** — spec + impl shipped
- 🗄 **Shelved** — parked indefinitely; revival trigger noted

---

## Quick-reference table (version sequence)

| # | Title | Status | Depends on | Trigger to start |
|---|---|---|---|---|
| **v2.01** | **DSL parser and safe executor** — Strict two-field output parser; program split on top-level commas; allowlisted operations (`add`, `subtract`, `multiply`, `divide`, `exp`, `greater`, table helpers), `#k` references, `const_*` literals; typed errors for malformed syntax, bad references, wrong arity, division by zero | ✅ Complete 2026-11-15 | v0.02 (+ operation inventory from v1.01) | — |
| **v2.02** | **Metrics and gold self-check** — Numeric normalization (commas, percent, scale, rounding), tolerance rule, yes/no for `greater` | ✅ Complete 2026-11-16 | v2.01, v1.02 | — |
| **v2.03** | **Evaluation runner** — `Predictor` protocol (prompt in; text, token counts, latency out) | ✅ Complete 2026-11-16 | v2.02, v1.03, v0.04 | — |

---

## Detail per version

### v2.01 — DSL parser and safe executor  ✅

**Status**: ✅ Complete 2026-11-15 (`req_002_01_spec.md`, `prp_002_dsl_parser_executor_v1.md`, `impl_002_20261115_225903.md`); tests and lint green locally, confirmed by the owner  
**Depends on**: v0.02 (+ operation inventory from v1.01)  

**Scope**: Strict two-field output parser; program split on top-level commas; allowlisted operations (`add`, `subtract`, `multiply`, `divide`, `exp`, `greater`, table helpers), `#k` references, `const_*` literals; typed errors for malformed syntax, bad references, wrong arity, division by zero. As specced: a reply parser with a strict flag, a program parser with a canonical form, an executor on floats that does not round, and one call, `run_reply`, that never raises.

**Decisions and open questions**: Three decisions, accepted by the owner on 2026-11-15 (spec section 8): a reply that breaks the two-line format is still read, with format validity reported beside it; a repeated row name means the last row, as in the published evaluator; a constant outside the seventeen is an error. Handed to v2.02: gold results are not uniformly rounded, and the self-check over all 8,281 examples needs a way to read test-row programs from inside the data package.

### v2.02 — Metrics and gold self-check  ✅

**Status**: ✅ Complete 2026-11-16 (`req_002_02_spec.md`, `prp_002_metrics_gold_check_v2.md`, `impl_002_20261116_004319.md`); tests and lint green locally; committed  
**Depends on**: v2.01, v1.02  

**Scope**: Numeric normalization (commas, percent, scale, rounding), tolerance rule, yes/no for `greater`. Execution accuracy, answer accuracy, program exact match, format validity. Gold self-check over all examples; mismatches reported and flagged.

**Decisions and open questions**: Decided by the owner on 2026-11-16 (spec section 8): execution accuracy allows absolute 0.00001 or relative 0.5% (the guide's figure; the spec had recommended relative 0.001%), and the answer line the same; flagged examples stay in every total with the unflagged total beside it; the self-check reads test rows, for counts and ids only. As built: flags go in `splits/finqa_gold_flags.json`, never in the manifest; program match compares numbers by value and keeps argument order; all 8,281 gold programs reproduce their gold answer and none is flagged. Known cost of the 0.5% rule, measured on fit and dev: 8.65% of examples have a wrong operand that still scores as right (0.87% under the tight rule); every prediction keeps its unrounded value, so v2.03 or v5.02 can report the tight figure beside the headline without another model run.

### v2.03 — Evaluation runner  ✅

**Status**: ✅ Complete 2026-11-16 (`req_002_03_spec.md`, `prp_002_evaluation_runner_v3.md`, `impl_002_20261116_011831.md`); tests and lint green locally; marked complete by the owner  
**Depends on**: v2.02, v1.03, v0.04  

**Scope**: `Predictor` protocol (prompt in; text, token counts, latency out). Runner over a named split; resumable predictions JSONL with raw output, parsed program, executed value, per-example scores; writes to the run registry and cost ledger; fake predictor for tests.

**Decisions and open questions**: Reaching it is milestone M3 (harness trusted). M3 was reached on 2026-11-16: the gold predictor scores every example of quick-dev and dev right through the runner. Three decisions, taken as recommended when the owner started the build (spec section 8): when a predictor cannot answer, the run stops and is resumed, and the example is never scored as wrong; every summary shows execution accuracy under the tight rule beside the 0.5% headline; each run that is a result adds one line of counts to `ledger/evaluations.jsonl`. As built: a predictor takes a list of requests and returns a list of predictions; the predictions file holds the reply, the outcome with its unrounded value, and the score, and no prompt or gold text; a resumed evaluation is a new run that continues an earlier one; only a run that passes checks R1 to R5 is a result.
