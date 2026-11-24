# Convention: the canonical record

**Last verified**: 2026-11-19  
**Defined in**: `qlora_finetuning/data/serialization.py` and `config/prompt.yaml` (req_001
v1.03). Recorded before its second use so that v1.04, v2.01, v3.01, v3.02, v4.01, v4.02, and
v6.01 all follow it.

Every model in the comparison reads the same words and is asked for the same reply. Those
words are produced in one place. A path renderer wraps them; it never writes or edits them.

## The record

```python
from qlora_finetuning.data import serialize_split

records = serialize_split("fit")       # or "dev", "quick_dev"; "test" is locked
record = records[0]
record.example_id    # the FinQA id
record.prompt_id     # which wording produced it
record.system        # the same for every example
record.user          # text evidence, the table, the question
record.target        # two lines: PROGRAM: ... and ANSWER: ...
```

```
Context:
- <evidence passage>            or the single line (none)

Table:
| <header> | <header> |
|---|---|
| <row name> | <cell> |

Question: <question>
```

```
PROGRAM: subtract(920, 95), divide(#0, 95)
ANSWER: 8.68421
```

- **Context** is the text evidence only, in document order. Table evidence is not marked.
- **Table** is the whole table, cells word for word. The first row is the header.
- **Program** is the dataset's program, unchanged.
- **Answer** is the executed result (`exe_ans`), with scientific notation written out.

## How a path uses it

| Path | Wraps the record as |
|---|---|
| A, Qwen (v3.01, v3.02) | Chat-template messages: `system` and `user` roles, with `target` as the assistant's reply when training. `enable_thinking=False`. As built: `render_chat` gives the prompt (v3.01), and `encode_example` in `qlora_finetuning/local_qlora/render.py` makes the training sequence from it: the prompt's tokens, then the tokens of `target` and the end token, with the loss on that reply only (v3.02). So a tuned model is asked the way it was taught, and the sequence has the length the token budget counted |
| B, Nova (v4.01, v4.02) | Bedrock conversation: the `system` field, one `user` turn, and `target` as the assistant turn in a training file. As built: `BedrockProvider.build_request` for a question (v4.01) and `conversation_line` in `qlora_finetuning/frontier/export.py` for a training line (v4.02) put the same two texts in the same two places, so a tuned model is asked the way it was taught. A training line is `{"schemaVersion": "bedrock-conversation-2024", "system": [{"text": system}], "messages": [{"role": "user", "content": [{"text": user}]}, {"role": "assistant", "content": [{"text": target}]}]}` |

## Rules

- **No prompt text in code.** Wording lives in `config/prompt.yaml`. A renderer adds only
  what its model's format requires (role markers, JSON structure).
- **Never edit a record's texts**: no trimming, no re-wrapping, no added instruction, no
  example appended. If a model needs something different, that is a change to the prompt
  definition, for every arm.
- **Record the prompt id** with every run: `"prompt"` in the run's `configs`, and the
  record's `prompt_id` in the predictions.
- **The wording is frozen once any arm has been run on it.** After that, a change means a new
  `version` in `config/prompt.yaml` and a rerun of every arm. The test in
  `tests/data/test_serialization.py` that pins the prompt id fails on any change, on purpose.
- **Input that is not a dataset example** (the demo, a smoke test) uses the same functions:
  `format_user(passages, table, question, prompt)` with `load_prompt().system`.
- **Nothing is truncated by the serializer.** What happens to an over-length record is the
  token budget's decision (v1.04).
- **Real records are never committed.** Read them with
  `python -m scripts.finqa_serialize --show "<example id>"` or `--dump`, which writes under
  git-ignored `artifacts/serialized/`. Tests use invented rows.

## Reuse checklist

- [ ] Texts come from `serialize_split` or `build_record`, unedited
- [ ] The renderer adds structure only
- [ ] The run records the prompt id
- [ ] A test shows the rendered text contains `record.system`, `record.user`, and (when
      training) `record.target` character for character
- [ ] Tests use `build_frozen_split(tmp_path)` and invented rows
