# Convention: a package for publication

**Last verified**: 2026-11-24
**Shown in**: `qlora_finetuning/packaging/` (req_006 v6.02), `scripts/hf_package.py`,
`config/packaging.yaml`, and `tests/packaging/`; and `qlora_finetuning/public_export/`
(req_006 v6.04), `scripts/public_export.py`, `config/public_export.yaml`, for a whole
repository: the source is a commit, the proof is the suite run inside the copy.

A folder that leaves the project: model weights with what a stranger needs to use them,
and a model card. Everything in it is either copied byte for byte from a checked source or
generated from a committed record. Nothing in it is typed.

## Shape

- **The source is checked before anything is copied.** `selected_source(context)` runs every
  check of `--adapter selected` and returns the checkpoint with its hash. A package is never
  built from `artifacts/adapters/<run>/`: that folder is a run's last step.
- **Weights and tokenizer files are copied by bytes**; text files are written with Unix line
  ends, so a package has the same bytes on every machine.
- **A changed model is a new thing with a new name.** The merged copy has its own
  predictors, its own dev runs, and its own card; it is never arm A1 and never runs on test.
  Its runs name the weights by hash in their settings, so a rebuilt copy does not inherit an
  old run.
- **Proof goes through the ordinary runner.** A package is scored by `run_evaluation` on dev,
  and compared with a reference run made on the same machine at the same batch size. "The
  same replies as the reference" can pass or fail cleanly; "the recorded score again" on
  another GPU cannot.
- **The card's code is the code that ran.** Usage code is a template file; `verify` renders
  it, runs it in a new process in an empty directory with the folder in place of the
  repository name, and stores the hash of what ran. The card embeds the same rendering, and
  a check fails if the hashes differ.
- **A card is a template and placeholders**, filled from records; a placeholder without a
  value stops the card. A share is the runner's own rate; a dollar figure is the cost
  model's.
- **One scan for what must not leave**: account numbers, ARNs, S3 paths, job names, keys,
  tokens, email addresses, machine paths. Patterns are in the config. Tokenizer files are
  exempt and are checked by bytes instead.
- **One record, written by one command**, with numbered checks, a status per package, and
  no reply or dataset text. `status` computes the same record and writes nothing.
- **The owner's confirmation is a field in the config** (`confirmed_on`), checked with an
  age limit. Until it is set, the cards are written as drafts outside git.
- **Nothing uploads.** No module imports an upload call; a guard test scans for them.

## Tests

- A stand-in project under `tmp_path`: a stub training run, a selection record, stub dev
  runs made with `FakePredictor` through the real runner (`tests/packaging/helpers.py`).
- Each check passes and fails; a withheld package leaves the other ready.
- The usage code is parsed, its two pure functions are executed against the serializer,
  and the whole code runs only in the `gpu`-marked test.
- The merge is tested on a tiny random model on the CPU, skipped where the stack is absent.

## Reuse checklist

- [ ] The source's hash is compared with a committed record before the first copy
- [ ] Every file of the package is in an exact list; an extra file fails a check
- [ ] Anything that changes the weights gets its own predictor name and its own runs
- [ ] The comparison is against a reference run on the same machine and batch size
- [ ] Usage code lives in one template, is executed, and its hash is recorded
- [ ] The card has no figure that is not a placeholder
- [ ] The scan runs over the package, the card, and the record
- [ ] The step that needs the owner is a dated field, not a command-line flag
