---
base_model: Qwen/Qwen3-4B
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
license: apache-2.0
language:
- en
datasets:
- bevaya/FinQA
tags:
- finqa
- finance
- question-answering
- program-synthesis
- merged-lora
model-index:
- name: qwen3-4b-finqa-program-merged
  results:
  - task:
      type: text-generation
      name: Writing a program that answers a question about a financial report
    dataset:
      name: "FinQA, this project's own split by company (not an official split)"
      type: bevaya/FinQA
      revision: 3d6a736bc67e06bc15fbf3618d88204a57c5b25e
      split: "dev (this project's split: 499 questions of 15 companies)"
    metrics:
    - type: execution_accuracy
      name: Execution accuracy, loaded in 4-bit
      value: 72.34
    - type: execution_accuracy
      name: Execution accuracy, at full precision
      value: 71.54
---

# qwen3-4b-finqa-program-merged

[`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) with the LoRA adapter
[`sudiptoHaldar/qwen3-4b-finqa-program-qlora`](https://huggingface.co/sudiptoHaldar/qwen3-4b-finqa-program-qlora) merged into its weights. It
loads with `transformers` alone, without `peft`. Given text from a company's financial
report, a table from the same report, and a question, it writes a short program that
computes the answer, and then the result. **The program is meant to be run by an
executor: the model is not the calculator.**

> **Do not report this model's score on the official FinQA test split.** The questions it
> was trained on were drawn from all of FinQA: the official training, validation, and test
> files were pooled and split again by company. A score on the official validation or test
> split is therefore inflated. The adapter's repository holds the split manifest, which
> says by example id which questions were used for what.

> **This copy is not the model that was measured on the test split.** The test figures
> below belong to the adapter on the 4-bit base model. A merged copy holds other weights
> once it is quantized, and it was scored on the development questions only.

- **Author:** sudiptoHaldar
- **Licence:** apache-2.0 (it carries Qwen's weights; see `LICENSE` and `NOTICE`)
- **Base model:** `Qwen/Qwen3-4B` at commit `1cfa9a7208912126459214e8b04321603b3df60c`
- **Adapter that was merged:** `sudiptoHaldar/qwen3-4b-finqa-program-qlora`, weights SHA-256 `a8c9561b4fc6e560a53d5104c0118e5d088fcfbb579b57b8bfb24714c559902b`
- **Code, with the executor and the full comparison:** https://github.com/SudiptoHaldar/qlora_finetuning_public

## How to use

The code below is executed before this card is published, with `QUANTIZED` set both ways:
it loads the files of this repository, asks an invented sample, and its reply must run to
the expected result. It needs a CUDA GPU: about 6 GB of memory in 4-bit, about 10 GB at
full precision.

```python
import json
import os
import re

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "sudiptoHaldar/qwen3-4b-finqa-program-merged"  # or a folder that holds this repository's files
QUANTIZED = True  # False: as stored (bfloat16): 8 GB of weights on the GPU

folder = MODEL if os.path.isdir(MODEL) else snapshot_download(MODEL)


def read(name):
    with open(os.path.join(folder, name), encoding="utf-8") as handle:
        return json.load(handle)


def one_line(value):
    return re.sub(r"\s*[\r\n]+\s*", " ", str(value)).strip()


def user_text(labels, context, table, question):
    """The user text the model was trained on: the context, the table, the question."""
    passages = "\n".join(labels["list_marker"] + one_line(passage) for passage in context)
    rows = [
        "| " + " | ".join(one_line(cell).replace("|", "\\|") for cell in row) + " |"
        for row in table
    ]
    rows.insert(1, "|" + "---|" * len(table[0]))
    return (
        labels["context_label"] + "\n" + (passages or labels["no_context"]) + "\n\n"
        + labels["table_label"] + "\n" + "\n".join(rows) + "\n\n"
        + labels["question_label"] + " " + one_line(question)
    )


spec = read("prompt.json")
sample = read("sample_input.json")  # an invented company; put your own input here
system = spec["system"]
user = user_text(spec["user"], sample["context"], sample["table"], sample["question"])

tokenizer = AutoTokenizer.from_pretrained(folder)
if QUANTIZED:
    model = AutoModelForCausalLM.from_pretrained(
        folder,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        folder, device_map={"": 0}, dtype=torch.bfloat16
    )
model.eval()

text = tokenizer.apply_chat_template(
    [{"role": "system", "content": system}, {"role": "user", "content": user}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
with torch.inference_mode():
    # This copy's own generation settings are greedy and stop at the end tokens.
    output = model.generate(**inputs, max_new_tokens=spec["max_new_tokens"])
reply = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
print(reply)  # two lines: PROGRAM and ANSWER. Run the program; do not trust the answer line.
```

- **The prompt.** The system text and the layout of the user text come from
  `prompt.json`, word for word. The model saw no other wording.
- **Thinking is off** (`enable_thinking=False`).
- **Greedy decoding.** `generation_config.json` in this repository decodes greedily and
  stops at the end tokens. The base model's own settings sample; they were replaced.
- At most 192 new tokens.

For the invented sample the reply is:

```
PROGRAM: subtract(4812, 4275), divide(#0, 4275)
ANSWER: 0.12567
```

The reply is text. Parse the `PROGRAM:` line and execute it with code you trust; never
pass it to `eval`. The project's executor is in the code repository
(`qlora_finetuning/dsl/`).

## How it was made

| | |
|---|---|
| Base weights | `Qwen/Qwen3-4B` at commit `1cfa9a7208912126459214e8b04321603b3df60c`, at full precision, never the 4-bit model |
| Adapter | `sudiptoHaldar/qwen3-4b-finqa-program-qlora`: LoRA rank 16, alpha 32, on 252 weight matrices |
| Merge | Each adapted weight becomes W + (alpha / r) x B x A, computed in float32 and saved as bfloat16 |
| Size | 4,022,468,096 parameters; 3 weight files |
| Generation settings | Replaced by greedy decoding |
| Tokenizer | The base model's files, unchanged |

How the adapter itself was trained is on its own card.

## Results

### This copy, on the development questions

499 questions of 15 companies, on one laptop GPU, in batches
of 8:

|  | Right | Execution accuracy | Replies that differ from the adapter's | Right there, wrong here | Wrong there, right here |
|---|---:|---:|---:|---:|---:|
| The adapter on the 4-bit base model, for reference | 358 of 499 | 71.74% | — | — | — |
| This copy, loaded in 4-bit | 361 of 499 | 72.34% | 157 | 6 | 9 |
| This copy, at full precision | 357 of 499 | 71.54% | 199 | 15 | 14 |

Loaded in 4-bit, this copy has 3 more right than the adapter; at full precision, 1 fewer right than the adapter. This project treats two runs that differ by fewer than 10 of these questions as not told apart.

### The adapter, on the test split

These figures are **the adapter's**, measured on the 4-bit base model with the adapter
attached: 2,898 questions from 43 companies that the model
never saw. They are given here because this copy has no test result of its own.

| Metric | Right | Share | Interval |
|---|---:|---:|---:|
| Execution accuracy | 2,121 | 73.19% | 71.60 to 74.78 |
| In the format | 2,898 | 100.00% | 100.00 to 100.00 |
| Answer accuracy | 1,941 | 66.98% | 65.29 to 68.67 |
| Program match | 1,972 | 68.05% | 66.36 to 69.74 |

Intervals are 95% confidence intervals, in percent. The other models
of the comparison, and its cost figures, are on the adapter's card.

## Limitations

- **The evidence is given.** At training and at test time the model received the passages
  the dataset marks as the evidence, not the whole report. Finding the evidence is not
  measured, so these figures are **not comparable with published FinQA results**.
- **No test result of its own**, as said above.
- **Trust the program, not the answer line.** The adapter's own answer was right for
  66.98% of the test questions; its executed program for 73.19%.
- **Longer calculations are harder.** The adapter's execution accuracy is 81.04%
  when the reference program has one step and 43.05% when it has three or more.
- **One training run, one prompt, one task.** Other wording, tables with merged headers,
  or questions that need an operation the program language lacks are outside what it
  learned.
- **A reply can change with the batch size and with the GPU.**
- **A research artifact, not financial advice.** Check every result before relying on it.

## Files

| File | Holds |
|---|---|
| `model-00001-of-00003.safetensors`, `model-00002-of-00003.safetensors`, `model-00003-of-00003.safetensors`, `model.safetensors.index.json` | The merged weights |
| `config.json`, `generation_config.json` | The model's configuration, and greedy generation settings |
| `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` | The base model's tokenizer, unchanged |
| `prompt.json` | The system text, the labels of the user text, the reply limit |
| `merge_metadata.json` | What was merged into what, and how |
| `sample_input.json` | The invented sample used above |
| `LICENSE`, `NOTICE` | Qwen's licence, and what was changed |

## Licences and citation

This model is a modified copy of `Qwen/Qwen3-4B` and is released under the Apache License, Version 2.0,
as the original is. What it is built on:

| What | Licence | What its source said | Read on |
|---|---|---|---|
| Qwen3-4B: the base model | `apache-2.0` | The model card shows "License: apache-2.0" and the repository's LICENSE file is the Apache License, Version 2.0, with "Copyright 2024 Alibaba Cloud". | 2026-11-24 |
| FinQA: the questions, programs, and answers | `mit` | The repository's LICENSE file is the MIT License, "Copyright (c) 2021 Zhiyu Chen". The Hugging Face copy this project pins, bevaya/FinQA, declares "License: mit"; it is a re-upload by a third party, so that tag is the uploader's statement. | 2026-11-24 |
| FinTabNet: the report tables that FinQA's examples are built on | `cdla-permissive-1.0` | Not re-read at its own source: IBM's Data Asset eXchange page for FinTabNet now says the exchange is deprecated and states no licence. The card of the derived dataset bsmock/FinTabNet.c says "the license of the original FinTabNet dataset is CDLA-Permissive" and links the Community Data License Agreement - Permissive, Version 1.0 (cdla.dev/permissive-1-0). One source, and not IBM's. | 2026-11-24 |

FinQA:

```bibtex
@article{chen2021finqa,
  title={FinQA: A Dataset of Numerical Reasoning over Financial Data},
  author={Chen, Zhiyu and Chen, Wenhu and Smiley, Charese and Shah, Sameena and Borova, Iana and Langdon, Dylan and Moussa, Reema and Beane, Matt and Huang, Ting-Hao and Routledge, Bryan and Wang, William Yang},
  journal={Proceedings of EMNLP 2021},
  year={2021}
}
```

Qwen3:

```bibtex
@misc{qwen3technicalreport,
  title={Qwen3 Technical Report},
  author={Qwen Team},
  year={2025},
  eprint={2505.09388},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2505.09388}
}
```
