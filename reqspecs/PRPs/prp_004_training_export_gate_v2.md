# PRP: Training file export and budget gate (req_004 v4.02)

**Feature Name**: training_export_gate (frontier_tuning v4.02)  
**Version**: 2  
**Created**: 2026-11-16  
**Status**: ✅ IMPLEMENTED (code; the gate opens after the owner's baselines, and the upload is the owner's)  
**Implemented**: 2026-11-16  
**Implementation Summary**: [impl_004_20261116_030558.md](../impl_summary/impl_004_20261116_030558.md)  
**Confidence Score**: 8/10  
**Estimated Time**: 4–5 hours of code, plus the owner's upload  
**Origin**: `reqspecs/reqs/req_004_frontier_tuning/req_004_02_spec.md`  
**Template**: `prp_lean.md` (expanded, medium complexity)  

---

## Goal

One command, `python -m scripts.nova_export`, that writes the two files Bedrock will train
on (fit as training, dev as validation), checks them on disk, estimates what Path B will
cost, and writes a committed gate record that says whether the estimate is inside the cap.
`--upload` stages the files in S3 and verifies them.

## Why & What

- **User value**: no fine-tuning job can start on a file nobody checked or on a cost nobody
  saw. v4.03 reads the gate record and refuses without a passing one for these very files.
- **Technical**: new config keys, four modules under `qlora_finetuning/frontier/` (export,
  validation, estimate and gate, staging), a script, tests on invented rows and stub clients.

The owner started this PRP and its execution without changing spec section 8, so its three
items are taken **as recommended**: the gate prices **2 epochs**; the Path B cap is
**$200**; **dev is the validation file**.

The v4.01 baselines have not been run, so no token ratio has been measured. **The gate is
therefore expected to fail on check G1 when this PRP is executed.** That is the correct
result, not a defect. The agent does not upload: `--upload` is the owner's.

## Success Criteria

- [x] `venv\Scripts\python -m pytest -q` and `venv\Scripts\python -m ruff check .` pass, also
      with the training libraries blocked.
- [x] `venv\Scripts\python -m scripts.nova_export` writes files of 4,884 and 499 lines; X1
      to X7 pass; the estimate is printed; the gate record is written with `passed: false`
      (G1), and the script exits 1.
- [x] A test shows every line's three texts equal the canonical record's.
- [x] No committed file holds a line of either training file.
- [x] Without `--upload` no AWS client is created.

Live (the owner, spec section 9): after the v4.01 baseline report exists, the plain command
passes G1 and G2; then `--upload`; the gate record is committed.

---

## Context

### References

```yaml
refs:
  - reqspecs/references/project_patterns/canonical_record.md  # texts unchanged; Path B's use
  - reqspecs/references/project_patterns/split_loading.md     # fit and dev only; the test lock
  - reqspecs/references/project_patterns/run_tracking.md      # start_run for an export
  - reqspecs/references/project_patterns/check_script.md      # PASS/FAIL lines, exit codes
  - reqspecs/references/tech/bedrock_nova.md, boto3.md
  - reqspecs/references/common/gotchas.md

code:
  - file: qlora_finetuning/data/serialization.py
    why: serialize_split(name, prompt, cfg, data_cfg) -> CanonicalRecord(example_id,
         prompt_id, system, user, target), dataset order
  - file: qlora_finetuning/data/split_loader.py
    why: split_ids("test", cfg) works while locked; manifest_hash(cfg)
  - file: qlora_finetuning/frontier/config.py, pricing.py, baseline_report.py, bedrock.py
    why: FrontierConfig to extend; Decimal prices; report_paths and the report's JSON
         (`passed`, `inputs`, `runs[].nova_to_qwen_input_ratio`); aws_session,
         client_config, check_identity, AUTH_EXCEPTIONS
  - file: qlora_finetuning/data/token_budget_report.py
    why: report_paths; JSON `inputs.prompt_id`, `inputs.split_content_hash`,
         `splits.<name>.{records, tokens_in_one_pass, prompt_tokens_in_one_pass,
         full_sequence_tokens.max}`; stale_files for --check
  - file: qlora_finetuning/evaluation/runner.py
    why: RunnerContext carries prompt, split_cfg, data_cfg, tracking_cfg, max_new_tokens,
         git, environment, configs
  - file: tests/evaluation/conftest.py, tests/frontier/conftest.py
    why: make_context(tmp_path): invented split (fit 30, dev 8, test 22);
         baseline_run(...) puts Path B token records into a temporary ledger
  - file: tests/frontier/test_config.py
    why: its GOOD dict must equal the committed frontier.yaml; it changes with the config

docs (read 2026-11-16; summarized in spec section 2.2):
  - https://docs.aws.amazon.com/nova/latest/userguide/fine-tune-prepare-data-understanding.html
  - https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-prepare.html
  - https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-analyze.html
  - https://github.com/aws-samples/amazon-nova-samples/tree/main/customization/bedrock-finetuning/understanding/dataset_validation
```

### Facts verified 2026-11-16

| Fact | Value |
|---|---|
| Trial export | fit 4,884 lines, 10.7 MB; dev 499 lines, 1.1 MB; no test id; no reserved word; all ASCII; 52 and 2 repeated lines |
| Reserved words (AWS's script) | `system:`, `user:`, `bot:`, `assistant:`, `thought:`, `[eos]`, `<image>`, `<video>`, `<unk>`: substring, case-insensitive, every text |
| Limits | 8 to 20,000 samples; training plus validation at most 20,000; 32,000 tokens a sample |
| File names | Letters, digits, `_`, `-`, `/`, `.` only |
| S3 (boto3 1.44.3) | `put_object(Bucket, Key, Body, ChecksumSHA256=<base64>)`; `head_object(Bucket, Key, ChecksumMode="ENABLED")` returns `ContentLength` and `ChecksumSHA256`; a missing key is a `ClientError` with code `404` or `NoSuchKey` |
| Baseline report | Not on disk. `reports/frontier/` does not exist yet |
| Token budget report | fit 3,254,181 tokens in one pass, longest 1,862; dev 333,677, longest 916 |
| `settings.load_config` | `0` is not missing; an empty list is |

### Critical Gotchas

- **Only `bedrock.py` may import boto3 today** (`tests/frontier/test_predictor.py` guard).
  `staging.py` needs `botocore.exceptions.ClientError`: extend the guard to allow exactly
  those two modules, and say so in `coding.md`.
- **The exporter never takes a split name.** It renders `fit` and `dev` and nothing else.
- **Validate from disk.** Read the bytes back; do not check the strings that were written.
- **Money is `Decimal`.** The cap and the ratio come from YAML numbers: convert with
  `Decimal(str(value))`, never `Decimal(float)`.
- **The gate record must be reproducible**: no time stamp, no absolute path, no run id, so
  that `--check` can compare. Carry the `staged` block over from the existing record when
  its file hashes are unchanged.
- **Do not flood the committed ledger.** Record an export run only when files were newly
  written or staged; never for `--check`.
- **`config/frontier.yaml` gains keys; `settings` must not change**, or v4.01 runs could no
  longer be resumed.
- **JSON**: `ensure_ascii=False` for the training lines (compact, fixed key order);
  `ensure_ascii=True`, `allow_nan=False`, `indent=2` for the record. Always `newline="\n"`
  or `write_bytes`.
- **Never print or commit a line of a training file**: any dataset text may be test text
  elsewhere; errors name example ids and words only.

---

## Implementation

### Files

```yaml
CREATE:
  - qlora_finetuning/frontier/export.py: RESERVED_WORDS, conversation_line, render_file,
      ExportFile, Export, export_files
  - qlora_finetuning/frontier/export_validation.py: check, TokenVolumes,
      load_token_volumes, validate_export (X1 to X7)
  - qlora_finetuning/frontier/training_estimate.py: TokenRatio, load_token_ratio,
      estimate_path_b, gate_checks (G1, G2), build_gate_record, render_markdown,
      write_gate_record, read_gate_record, stale_files
  - qlora_finetuning/frontier/staging.py: StagingError, object_key, stage_export
  - scripts/nova_export.py
  - tests/frontier/test_export.py, test_export_validation.py, test_training_estimate.py,
      test_staging.py; tests/scripts/test_nova_export.py
  - reports/frontier/nova_training_gate.json and .md (written by the real run)
  - reqspecs/impl_summary/impl_004_<TIMESTAMP>.md

MODIFY:
  - config/frontier.yaml: version 2; customization, export, budget, estimate
  - qlora_finetuning/frontier/config.py: the new fields; load_bucket
  - tests/frontier/test_config.py, tests/frontier/test_predictor.py (the guard)
  - docs/runbook.md (section 6), coding.md, gotchas.md, canonical_record.md,
    bedrock_nova.md, architecture overview, aws_commands_run.md (the owner's upload)
  - the spec, req_004_roadmap.md, execution_plan.md
```

### Design

```python
# export.py
def conversation_line(record, schema_version) -> str:
    return json.dumps({"schemaVersion": schema_version,
                       "system": [{"text": record.system}],
                       "messages": [{"role": "user", "content": [{"text": record.user}]},
                                    {"role": "assistant", "content": [{"text": record.target}]}]},
                      ensure_ascii=False, separators=(",", ":"))

def export_files(cfg, context) -> tuple[Export, bool]:
    # render fit and dev -> bytes; sha256 each; export_id = sha256(both digests)[:12]
    # folder = cfg.export_dir / export_id; write the two files and two id lists unless the
    # folder already holds exactly these bytes; return (export, newly_written)

# export_validation.py
def validate_export(export, cfg, context, volumes, ratio) -> list[dict]:   # X1 .. X7
#   reads each file's bytes; parses each line; compares with serialize_split again

# training_estimate.py
def load_token_ratio(cfg, prompt_id, manifest_hash) -> TokenRatio   # measured or assumed
def estimate_path_b(cfg, pricing, volumes, ratio, ledger_records, *, test_examples,
                    max_new_tokens) -> dict
#   training tokens = ceil(fit Qwen tokens * ratio) * epochs, per model
#   total = spent (ledger, Path B inference at list price)
#         + training * (1 + margin)
#         + evaluation to come (dev once, test twice, per model; replies at the limit)
#         + storage (models * months)
def gate_checks(estimate, ratio, cfg) -> list[dict]                  # G1, G2
def build_gate_record(export, checks, estimate, ratio, cfg, pricing, staged) -> dict

# staging.py
def stage_export(s3, export, bucket, cfg) -> dict
#   per file: head_object; equal checksum -> keep; missing -> put_object with
#   ChecksumSHA256; then head_object again and compare size and checksum
```

Script flow: resolve context and configs, export, validate, ratio, estimate, gate; with
`--upload` (refused if any X check failed) check the sign-in and stage; build the record;
`--check` compares with the committed record and writes no committed file; otherwise write
the record and, if files were new or staged, record one `export` run. Exit 0 passed, 1 not
passed, 2 config or sign-in.

### Tasks (~4.5 hours)

1. **Config keys and `FrontierConfig` fields; update the config tests** (~25 min).
2. **`export.py` and tests** (~40 min).
3. **`export_validation.py` and tests, each check passing and failing** (~60 min).
4. **`training_estimate.py` and tests** (~60 min).
5. **`staging.py` and tests; the import guard** (~25 min).
6. **Script and tests** (~35 min).
7. **The real run; commit-ready gate record** (~5 min).
8. **Documents** (~25 min).
9. **Validation with the stack blocked, status documents, summaries** (~20 min).

### Error handling

| Situation | Behavior |
|---|---|
| A check X1 to X7 fails | FAIL line naming example ids or words, never text; record written with `passed: false`; exit 1 |
| No baseline report, a failed one, or one for another prompt or split | G1 fails; the estimate uses the assumed ratio and says "not measured"; exit 1 |
| Estimate above the cap | G2 fails; exit 1 |
| `--upload` when an X check failed | `REFUSED` line; nothing is sent; exit 1 |
| `--upload` without a sign-in, or the wrong account | `CONFIG ERROR` with the `aws sso login` command; exit 2 |
| An object in S3 with another checksum, or a mismatch after upload | `StagingError` (a `ConfigError`); exit 2 |
| Token budget report missing or made from another prompt | Missing: exit 2. Another prompt: X6 fails |
| No snapshot or no manifest | `CONFIG ERROR`; exit 2 |

---

## Validation

```powershell
venv\Scripts\python -m ruff check .
venv\Scripts\python -m pytest -q
venv\Scripts\python -m scripts.nova_export
venv\Scripts\python -m scripts.nova_export --check
```

**Expected**: ruff clean; all tests pass; the export prints X1 to X7 PASS, G1 FAIL
("not measured"), an estimate at the assumed ratio 1.5, and exits 1; `--check` then
reports the record current (and exits 1 for the same reason: the gate is not open).

---

## Checklist

- [x] Files created and modified as listed
- [x] Every function and every check has passing and failing tests
- [x] Lint clean; tests green with and without the training stack
- [x] The real run shown; gate record written
- [x] Runbook, references, `coding.md`, architecture updated
- [x] Roadmap, execution plan, spec status updated; implementation summary created
- [ ] Owner: the v4.01 baselines, then the export again (gate open), then `--upload`

---

## Implementation Time Estimate

**Total: 4–5 hours**

| Task | Time |
|---|---|
| Config | 25 min |
| Export | 40 min |
| Validation | 60 min |
| Estimate and gate | 60 min |
| Staging and the guard | 25 min |
| Script | 35 min |
| Real run | 5 min |
| Documents | 25 min |
| Validation and status documents | 20–40 min |

## Confidence Score: 8/10

Everything the export reads exists and a trial of it ran clean. Held back: the staging
code cannot be tried against S3 by the agent, and the gate's passing path cannot be seen
on real numbers until the baselines exist.

## Time Taken for PRP Generation

**~10 minutes** (the research was done while writing the spec the same day)

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 2 min |
| Codebase exploration | 4 min |
| Research existing PRPs and patterns | 1 min |
| Writing PRP document | 3 min |
