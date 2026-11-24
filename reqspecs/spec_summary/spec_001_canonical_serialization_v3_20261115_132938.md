# Spec summary: Canonical serialization (req_001 v1.03)

**PRP**: [prp_001_canonical_serialization_v3.md](../PRPs/prp_001_canonical_serialization_v3.md)  
**Origin**: `reqspecs/reqs/req_001_finqa_data/req_001_03_spec.md`  
**Created**: 2026-11-15  

## Todo list

1. Write `config/prompt.yaml` (system text, layout, version) and `data/serialization.py`:
   evidence selection, table formatting, answer formatting, the record, the prompt id.
2. Write four golden records from invented rows and the serializer tests.
3. Write `data/serialization_report.py` (lengths, counts, digests, checks C1–C10) and its
   tests.
4. Write `scripts/finqa_serialize.py` (`--check`, `--dump`, `--show`) and its tests.
5. Serialize the real fit and dev splits, reconcile with spec section 2, and read a few real
   prompts.
6. Build the `env_check` smoke example with the serializer and prove the smoke step still
   passes.
7. Write the runbook part, the `coding.md` rule, the canonical-record note, and update the
   plan.
8. Validate with and without the training stack; update the roadmap, plan, and summaries.

## Summary

Each example becomes one canonical record of three texts. The system text is the same for
every example and describes the whole output language, so the untuned arms are told
everything the tuned ones learn. The user text is the text evidence (or `(none)`), the whole
table as Markdown with cells copied word for word, and the question. The target is two lines:
the dataset's program unchanged, and the executed result with scientific notation written
out.

The wording lives in `config/prompt.yaml`. A prompt id, a hash of the wording and layout,
travels with every record. Records are served by split through the existing loaders, so the
test lock applies. A committed report holds lengths in characters, what was skipped, and a
digest of all fit and dev records; it holds no example text. Golden tests use invented rows.

## Confidence score

8/10. Small pure functions over data already measured. The risk is in the checks that parse
rendered text and in the changed smoke step.

## Implementation Time Estimate

4–5 hours.

| Task | Time |
|---|---|
| Prompt definition and `serialization.py` | 50 min |
| Golden records and serializer tests | 50 min |
| `serialization_report.py` and tests | 50 min |
| Script and tests | 30 min |
| Real run | 15 min |
| Smoke example | 20 min |
| Documents | 30 min |
| Validation and status documents | 15–30 min |

## Time Taken for PRP Generation

~8 minutes.

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 1 min |
| Codebase exploration | 1 min |
| Research (existing patterns) | 1 min |
| Writing PRP document | 5 min |
