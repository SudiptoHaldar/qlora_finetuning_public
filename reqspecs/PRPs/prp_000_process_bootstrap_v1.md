# PRP: Process bootstrap (req_000 v0.01)

**Feature Name**: process_bootstrap (project_scaffold v0.01)  
**Version**: 1  
**Created**: 2026-11-14  
**Status**: ✅ IMPLEMENTED  
**Implemented**: 2026-11-14  
**Implementation Summary**: [impl_000_20261114_203322.md](../impl_summary/impl_000_20261114_203322.md)  
**Confidence Score**: 9/10  
**Estimated Time**: 1.5–2.5 hours  
**Origin**: `reqspecs/reqs/req_000_project_scaffold/req_000_01_spec.md`  
**Template**: `prp_lean.md` (expanded, medium complexity)  

---

## Goal

Make the process files describe this project. Rewrite the stale parts of five command files,
three PRP templates, and `coding.md`; delete the inherited database stack; create the
`reqspecs/references/` and `track_progress/` folders with seeded content. No application code.

## Why & What

- **User value**: every later `/generate-prp` and `/execute-prp` run follows instructions that
  are true for this repository, instead of working around `uv`, an absent EDGAR guide, and an
  MCP server that does not exist. Reaches milestone M0.
- **Technical**: text edits to 12 existing files, 2 deletions, 9 new files, 1 guard test.

## Success Criteria

- [ ] The stale-reference scan (Validation) returns nothing.
- [ ] `alembic.ini` and `alembic/` are gone.
- [ ] `venv\Scripts\python -m pip install -r requirements.txt` succeeds.
- [ ] `venv\Scripts\python -m pytest -q` passes (the 44 existing tests plus the new guard
      tests) and `venv\Scripts\python -m ruff check .` is clean.
- [ ] Every path in spec section 4.5 exists.
- [ ] No command or template names a file, tool, or command absent from the repository.
- [ ] `scripts\aws_check.py` still passes 7/7 (nothing it depends on was disturbed).

---

## Context

### References

```yaml
refs:   # reqspecs/references/ is created by this PRP; nothing local exists yet
  - file: reqspecs/reqs/req_000_project_scaffold/req_000_01_spec.md
    why: decisions D1-D11 (section 3) and the scope tables (section 4)
  - file: docs/old_references/stale_project_name_references.md
    why: section 3 lists the stale lines file by file
  - file: reqspecs/PRPs/prp_000_aws_foundation_v5.md
    why: source for reqspecs/references/tech/boto3.md (response shapes, gotchas)
  - file: reqspecs/0. PLANNING/planning_material/*.pdf
    why: source for common/gotchas.md; findings are summarised below, no need to re-read

code:
  - file: docs/runbook.md
    why: section 1 (AWS) is the style exemplar for the new runbook example in execute-prp.md
  - file: scripts/aws_check.py
    why: the "run the changed script once" manual gate refers to scripts like this
```

### Decisions that shape every edit (from the spec)

- **D1** commands are `venv\Scripts\python -m <tool>`; POSIX form given once in `coding.md`.
- **D3** file-size guideline is 1000 lines.
- **D4** planning material is the two guides, read when a PRP touches training, data
  preparation, or evaluation.
- **D5** manual gate: when a script's command line or output changed, run it once and show the
  output. Network or credential checks sit behind `-m network`. Billed steps are the owner's.
- **D6** `release_number.md` is the only release number; no package-version text.
- **D7** `wip-roll` "Release status" becomes "Project status" (milestones, AWS spend against
  the cap in `config/aws.yaml`, operator-gated steps). `roadmap_view-roll` examples follow.
- **D8** `pytest.ini` keeps only `network`. **D11** historical documents are not rewritten.

### Findings from the two guides (for `common/gotchas.md`)

| Gotcha | Source |
|---|---|
| Qwen3 thinking is on by default; `<think>` blocks cost 400–800 output tokens. Pass `enable_thinking=False` in the chat template | Gemini guide §1 |
| Full FinQA contexts often exceed 2,000 tokens; truncating at 1,024 cuts the question or table off the end. Use `gold_inds` plus the table at 1,536 and measure first | Gemini §1; OpenAI guide §5 |
| The guide names `DataCollatorForCompletionOnlyLM`; recent TRL releases removed it | Gemini §1; plan §3 |
| BF16 needs compute capability 8.0 or higher; older cards use FP16 | Gemini Phase 1 |
| Native Windows lacks reliable bitsandbytes CUDA support according to the guide | Gemini Phase 1; Q3 |
| `max_new_tokens` too low truncates programs and shows up as format failures; the guide uses 192 | Gemini Phase 5 |
| Select checkpoints on dev execution accuracy, not lowest loss | Gemini Phase 4; OpenAI §4 |
| Disable the KV cache during training | OpenAI §3 |
| Off-by-100x scale errors and reversed operand order are the common model errors | Gemini Phase 5 |
| The OpenAI guide's example uses `;` and `result`; the project uses FinQA's native `#k` references and commas | plan §4.2 |
| The guides' accuracy ranges assume the official split | plan §3 |
| Never tune against the test split | both guides |

### Critical Gotchas

- **Edit by line range, verify before replacing.** The files have trailing spaces and mixed
  content that make exact-string matches brittle. For each replacement, assert that the old
  lines contain an expected phrase, then replace the range. Apply ranges bottom-up so earlier
  line numbers stay valid. Preserve each file's line-ending style.
- **Legitimate words that the spec's scan pattern would hit.** "run registry" is this project's
  term and "PyPI" correctly names the package index in `coding.md` and `requirements.txt`. The
  spec allows narrowing; use the pattern in Validation.
- **`pytest -m` in `addopts`** must stay; only the `db` marker line goes.
- **Do not touch** `.claude/CLAUDE.md`, `.claude/settings.json`, `.claude/key_value.md`,
  `README.md`, or anything under `docs/old_references/` except appending the closing section.
- **`alembic/` is an empty untracked folder**; `alembic.ini` is tracked. Delete both from the
  working tree; the owner commits.
- **Inherited example names** in `generate-prp.md` (`req_026_paypal_integration` and similar)
  are retargeted to real names from this repository (`req_000_project_scaffold`,
  `prp_000_aws_foundation_v5.md`). Formats and naming rules do not change.

---

## Implementation

### Files

```yaml
MODIFY:
  - .claude/commands/generate-prp.md: reading list, examples, validation gates, limit
  - .claude/commands/execute-prp.md: reading list, runbook step, next-step examples, runbook example, AWS log rule
  - .claude/commands/release-tag.md: remove package-version text; test-plan examples
  - .claude/commands/wip-roll.md: "Project status"; project examples
  - .claude/commands/roadmap_view-roll.md: operator-gated and milestone examples
  - reqspecs/PRPs/templates/prp_base.md: commands, paths, examples, integration points
  - reqspecs/PRPs/templates/prp_lean.md: validation block
  - reqspecs/PRPs/templates/GUIDELINES.md: web and chart examples become project examples
  - reqspecs/1. PROMPTS/coding.md: full rewrite per spec 4.3
  - requirements.txt: remove six database and API pins
  - .env.example: remove DATABASE_URL
  - pytest.ini: remove the db marker
  - .gitignore: add artifacts/
  - docs/runbook.md: 1.1 installs from requirements.txt; stub sections 2-6
  - docs/old_references/stale_project_name_references.md: append the resolution section
  - roadmap, execution plan, spec status: v0.01 complete, M0 reached

DELETE:
  - alembic.ini
  - alembic/

CREATE:
  - reqspecs/references/README.md
  - reqspecs/references/common/gotchas.md
  - reqspecs/references/common/troubleshooting.md
  - reqspecs/references/tech/boto3.md
  - reqspecs/references/patterns/.gitkeep
  - reqspecs/references/project_patterns/README.md
  - track_progress/wip/.gitkeep
  - track_progress/roadmap_views/.gitkeep
  - tests/test_process_scaffold.py: guard tests (stale-term scan, required paths, removed files)
  - reqspecs/impl_summary/impl_000_<TIMESTAMP>.md
```

### Edits per file (line numbers as of 2026-11-14)

**`generate-prp.md`**

| Lines | Becomes |
|---|---|
| 7, 11, 12, 23, 27, 28 | Examples use `req_000_project_scaffold`, `req_000_05_spec.md`, `000`, `prp_000_aws_foundation_v5.md` |
| 43 | Read the two guides in `planning_material/` when the PRP touches training, data preparation, or evaluation |
| 44 | Files under 1000 lines; tests need no GPU, network, credentials, or paid call |
| 47, 51 | Technologies: boto3, `datasets`, `transformers`, `peft`, `trl`, `bitsandbytes`, Bedrock, SageMaker. Patterns: split manifest reader, DSL executor, cost ledger record, read-only AWS check |
| 96–105 | Gates: `venv\Scripts\python -m ruff check .`; `venv\Scripts\python -m pytest -q` (live checks via `-m network`); manual gate per D5; billed steps are the owner's |
| 160, 183, 185 | Example file names from this repository |

**`execute-prp.md`**

| Lines | Becomes |
|---|---|
| 12 | The two guides, same condition as above |
| 14 | 1000-line guideline; venv commands; no GPU, network, credentials, or paid call in unit tests; billed steps are the owner's; log every AWS command in `administration/aws_commands_run.md` |
| 82–87 | Runbook entry contents: setup steps; `pip install` plus the `requirements.txt` line; config files touched; how to run each new script and its exit codes; operator-gated or billed steps; troubleshooting |
| 124, 126 | "Scripts to run by hand"; "Operator-gated or billed steps waiting on the user" |
| 155–157 | Tests, lint, and `scripts\aws_check.py` as the run-the-script example |
| 165–173 | Example runbook entry modelled on runbook section 1 (sign in, run the check) |

**`release-tag.md`**: line 11 states that `release_number.md` is the only release number and no
other file carries a version; lines 57–59 become "Nothing else needs editing."; line 122
example becomes a paid training run the owner starts after merge; lines 125–127 use venv
commands and `scripts\aws_check.py`.

**`wip-roll.md`**: lines 41–42, 64, 83–88 per D7 (milestones M0–M7 from `execution_plan.md`,
spend against `budget.cap_usd` in `config/aws.yaml`, "not measured" when unknown, operator-gated
steps); line 70 example "Path A smoke job gate"; lines 104–105 examples become library version
quirks, AWS quota or throttling limits, dataset oddities.

**`roadmap_view-roll.md`**: lines 71–72 "a paid SageMaker or Bedrock job the user launches";
line 117 "milestones reached (M0 to M7)"; line 122 "critical-path item"; line 127 trigger
example "(→ after the first full run)".

**`prp_base.md`**: lines 62–66 project gotcha examples; 73–80 data-structure examples
(dataclasses, YAML keys, JSONL records); 84–100 paths under `qlora_finetuning/`; 104–126 a
synchronous pseudocode example (validate, read config, injected client, typed result); 128–141
integration points (config, dependencies, markers, runbook, AWS log); 145–152 ruff only;
175–193 venv pytest and a run-the-script Level 3; 196–199 checklist; 276 replaced by "no unit
test may need GPU, network, credentials, or a paid call".

**`prp_lean.md`**: lines 66–75 venv ruff, venv pytest, run-the-script manual check.

**`GUIDELINES.md`**: lines 16, 29, 30, 59–61, 70–72, 113: web, chart, and UI examples become
config, script, split-manifest, DSL-executor, and cost-ledger examples.

**`coding.md`**: rewrite. Keep the four section headings and the task-completion rules. Fix the
plan path twice; state D1 with both forms; list the eight sub-packages from plan §5 plus
`scripts/` and `config/`, noting the package arrives in v0.02; keep the `settings.py` rule with
a note that a script loads its own YAML until v0.02; add the secrets rule and the AWS-log rule;
replace the database test rule with the GPU, network, credentials, paid-call rule, noting
`network` is registered today and `gpu` and `paid` come in v0.02.

### New file contents

- **`references/README.md`**: what each folder holds; the reference-first rule (check here
  before the web; refresh entries older than six months; add a reference after any web fetch);
  the entry format (title, last-verified date, source).
- **`common/gotchas.md`**: four groups, each row a gotcha, what to do, source, and status.
  Model and training (the table above, status "from the guides; confirm in v0.03"); AWS and
  cost (Provisioned Throughput $55–$60.50 per hour never used; GPU quota lead time and the
  zero quota at `24xlarge` and `48xlarge`; $1.95 per custom Nova model per month; an instance
  bills per hour whether busy or idle; the budget alarm only emails and resets 2027-01-01);
  boto3 (one line each, linking to `tech/boto3.md`); local environment (cp1252 console, venv
  commands, `-m` override, Python 3.13 untested with the training stack).
- **`common/troubleshooting.md`**: symptom, cause, fix. Empty venv copied from another project;
  stale launchers in `venv/Scripts/`; `SIGN-IN REQUIRED` and SSO token errors; profile not
  found; null bucket region; budget creation rejected for the placeholder email; C5 quota not
  found; C7 after a cap change; live test "deselected"; ruff UP035.
- **`tech/boto3.md`**: version 1.44.3; session and client creation; the response-shape block
  and gotchas from the v0.05 PRP; error classes for sign-in; stubbing clients in tests; the four
  documentation URLs; last verified 2026-11-14.
- **`project_patterns/README.md`**: no pattern recorded yet; names the read-only check script
  (`scripts/aws_check.py`) as the first candidate.

### Tasks (~2 hours total)

1. **Command files** (~35 min): five files, line-range edits with pre-checks.
2. **Templates and `coding.md`** (~25 min).
3. **Database stack** (~5 min): two deletions, `requirements.txt`, `.env.example`, `pytest.ini`.
4. **New folders and files** (~25 min).
5. **Runbook, `.gitignore`, stale-reference report** (~10 min).
6. **Guard test** (~10 min).
7. **Validation** (~10 min): scan, install, tests, lint, `aws_check`.
8. **Roadmap, execution plan, spec status, summaries** (~10 min).

### Pseudocode (edit helper and guard test)

```python
def replace_lines(path, edits):            # edits: (start, end, must_contain, new_text)
    text = read_bytes(path).decode("utf-8"); eol = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(eol)
    for start, end, must_contain, new in sorted(edits, reverse=True):   # bottom-up
        old = "\n".join(lines[start - 1:end])
        assert must_contain in old, (path, start, must_contain)        # stop, do not guess
        lines[start - 1:end] = new.split("\n")
    write_bytes(path, eol.join(lines).encode("utf-8"))

# tests/test_process_scaffold.py
STALE = re.compile(<the Validation pattern>, re.IGNORECASE)
def test_no_stale_terms(): scan .claude/commands, reqspecs/PRPs/templates, "reqspecs/1. PROMPTS",
                           requirements.txt, pytest.ini, .env.example; assert no hits (report file:line)
def test_required_paths_exist(): the eight paths from spec 4.5
def test_database_stack_removed(): alembic.ini and alembic/ absent
def test_artifacts_ignored(): "artifacts/" is a line in .gitignore
```

**Error handling**: a failed pre-check assertion means the file changed since this PRP was
written. Read the file, find the new location, and edit by hand; never force a replacement.

---

## Validation

```powershell
# Stale-reference scan (Git Bash); expected: no output
grep -rniE -- 'uv run|uv sync|uv add|uvx|edgar|\bmcp\b|pypi (release|publish|version)|pypi ?/|server\.json|registry (publish|entry|release|validation)|/ ?registry|pyproject|postgres|psycopg|sqlalchemy|fastapi|uvicorn|alembic|DATABASE_URL|mark\.db|500-line|reqspecs/planning/|date picker|tooltip|\bchart|\bticker|\bmypy\b|\bcurl\b|\bsrc/|paypal' .claude/commands reqspecs/PRPs/templates "reqspecs/1. PROMPTS" requirements.txt pytest.ini .env.example

# Install, lint, tests
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q

# Nothing AWS-facing was disturbed (read-only; needs a valid sign-in)
venv\Scripts\python scripts\aws_check.py
```

**Expected**: the scan prints nothing; install succeeds; ruff clean; all tests pass; 7/7.

The scan pattern is narrower than the one in the spec in two places, as the spec allows:
`registry` only in its publication senses, and `pypi` only in its release senses.

---

## Checklist

- [ ] Files created, modified, and deleted as listed
- [ ] Scan clean; guard tests pass
- [ ] Lint clean, all tests green, `aws_check` 7/7
- [ ] Roadmap, execution plan, and spec status updated in the same turn; M0 recorded
- [ ] Implementation summary created; PRP status set to implemented

---

## Implementation Time Estimate

**Total: 1.5–2.5 hours**

| Task | Time |
|---|---|
| Command files | 35 min |
| Templates and `coding.md` | 25 min |
| Database stack removal | 5 min |
| References and tracking folders | 25 min |
| Runbook, `.gitignore`, stale-reference report | 10 min |
| Guard test | 10 min |
| Validation and fixes | 10–30 min |
| Roadmap, plan, summaries | 10 min |

Risk: +20 min if scan hits remain after the first pass and need rewording.

## Confidence Score: 9/10

All work is text editing against files read in full today, with an executable scan as the
gate. The point held back is for the number of separate edits: a missed stale phrase is likely
on the first pass and is caught by the scan.

## Time Taken for PRP Generation

**~12 minutes**

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration (five command files, three templates) | 3 min |
| Research existing PRPs and patterns (the two guides, v0.05 PRP) | 2 min |
| Writing PRP document | 5 min |
