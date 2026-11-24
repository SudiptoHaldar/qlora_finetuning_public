---
base_model: Qwen/Qwen3-4B
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
language:
- en
datasets:
- bevaya/FinQA
tags:
- lora
- qlora
- finqa
- finance
- question-answering
- program-synthesis
model-index:
- name: qwen3-4b-finqa-program-qlora
  results:
  - task:
      type: text-generation
      name: Writing a program that answers a question about a financial report
    dataset:
      name: "FinQA, this project's own split by company (not the official test split)"
      type: bevaya/FinQA
      revision: 3d6a736bc67e06bc15fbf3618d88204a57c5b25e
      split: "test (this project's split: 2,898 questions of 43 companies)"
    metrics:
    - type: execution_accuracy
      name: Execution accuracy
      value: 73.19
    - type: program_match
      name: Program match
      value: 68.05
    - type: answer_accuracy
      name: Answer accuracy
      value: 66.98
    - type: format_validity
      name: Replies in the format
      value: 100.00
---

# qwen3-4b-finqa-program-qlora

A LoRA adapter for [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B), trained with
QLoRA on FinQA. Given text from a company's financial report, a table from the same
report, and a question, it writes a short program that computes the answer, and then the
result. **The program is meant to be run by an executor: the model is not the calculator.**

> **Do not report this model's score on the official FinQA test split.** The questions it
> was trained on were drawn from all of FinQA: the official training, validation, and test
> files were pooled and split again by company. A score on the official validation or test
> split is therefore inflated. `finqa_split_manifest.json` in this repository says, by
> example id, which questions were used for training, for development, and for testing.

- **Author:** sudiptoHaldar
- **Licence:** apache-2.0
- **Base model:** `Qwen/Qwen3-4B` at commit `1cfa9a7208912126459214e8b04321603b3df60c`
- **Code, with the executor and the full comparison:** https://github.com/SudiptoHaldar/qlora_finetuning_public
- **Merged copy, which loads without `peft`:** [`sudiptoHaldar/qwen3-4b-finqa-program-merged`](https://huggingface.co/sudiptoHaldar/qwen3-4b-finqa-program-merged)

## What it does

Input, as the model sees it (an invented company; every figure was made up):

```
Context:
- Northfield Tools is an invented company; every figure here was made up for this demo.
- Net revenue rose in 2024 after the company opened two plants. Amounts are in millions.

Table:
|  | 2024 | 2023 |
|---|---|---|
| net revenue | $ 4,812 | $ 4,275 |
| operating income | $ 655 | $ 590 |
| capital expenditures | $ 310 | $ 284 |

Question: what was the percentage change in net revenue from 2023 to 2024?
```

Reply:

```
PROGRAM: subtract(4812, 4275), divide(#0, 4275)
ANSWER: 0.12556
```

The program language has ten operations (`add`, `subtract`, `multiply`, `divide`, `exp`,
`greater`, and four that work on a table row) and refers to an earlier step as `#0`, `#1`.
The system text in `prompt.json` describes it completely. Running the program above gives
`0.1256140350877193`.

## How to use

The code below is executed before this card is published: it loads the files of this
repository, asks the invented sample, and its reply must run to the expected result.
It needs a CUDA GPU with about 6 GB of memory free.

```python
import json
import os
import re

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

ADAPTER = "sudiptoHaldar/qwen3-4b-finqa-program-qlora"  # or a folder that holds this repository's files
BASE = "Qwen/Qwen3-4B"
BASE_COMMIT = "1cfa9a7208912126459214e8b04321603b3df60c"

folder = ADAPTER if os.path.isdir(ADAPTER) else snapshot_download(ADAPTER)


def read(name):
    with open(os.path.join(folder, name), encoding="utf-8") as handle:
        return json.load(handle)


def one_line(value):
    return re.sub(r"\s*[\r\n]+\s*", " ", str(value)).strip()


def user_text(labels, context, table, question):
    """The user text the adapter was trained on: the context, the table, the question."""
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
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    revision=BASE_COMMIT,
    device_map={"": 0},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
)
model = PeftModel.from_pretrained(model, folder).eval()
# The base model's own generation settings sample. Replace them whole: decode greedily.
model.generation_config = GenerationConfig(
    do_sample=False,
    num_beams=1,
    eos_token_id=[tokenizer.convert_tokens_to_ids(token) for token in spec["end_tokens"]],
    pad_token_id=tokenizer.pad_token_id,
)

text = tokenizer.apply_chat_template(
    [{"role": "system", "content": system}, {"role": "user", "content": user}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=spec["max_new_tokens"])
reply = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
print(reply)  # two lines: PROGRAM and ANSWER. Run the program; do not trust the answer line.
```

Four things in it matter, because they are the configuration the figures below were
measured with:

- **The prompt.** The system text and the layout of the user text come from
  `prompt.json`, word for word. The adapter saw no other wording.
- **Thinking is off** (`enable_thinking=False`).
- **4-bit NF4 quantization** of the base model, with the adapter attached and not merged.
- **Greedy decoding**, stopping at the end tokens, with at most 192 new tokens.

The measured runs asked questions in batches of 8 on one GPU. A reply can change
with the batch size and with the GPU, so another setup may answer a few questions
differently.

The reply is text. Parse the `PROGRAM:` line and execute it with code you trust; never
pass it to `eval`. The project's executor is in the code repository
(`qlora_finetuning/dsl/`).

## Training data

| | |
|---|---|
| Dataset | [`bevaya/FinQA`](https://huggingface.co/datasets/bevaya/FinQA) at commit `3d6a736bc67e06bc15fbf3618d88204a57c5b25e`, a copy of [FinQA](https://github.com/czyssrs/FinQA) |
| Questions | 8,281: the official training, validation, and test files, pooled |
| Split | By company, so no company has questions on both sides: 65% for training and development, the rest for testing; seed 20261115 |
| Used for training (`fit`) | 4,884 questions of 79 companies |
| Used to choose the checkpoint (`dev`) | 499 questions of 15 companies |
| Held out (`test`) | 2,898 questions of 43 companies, answered once |
| Split manifest | `finqa_split_manifest.json`, content hash `d34fb22e37b78a31ac22287cc8e199ab942d93125f9a80cb0b323862e599a07a` |

Each training example gives the model the passages FinQA marks as the evidence for the
question, the whole table, and the question. See "Limitations".

## Training procedure

| | |
|---|---|
| Method | QLoRA: the base model in 4-bit (NF4, double quantization, bfloat16 compute), LoRA adapters trained in bf16 |
| LoRA | rank 16, alpha 32, dropout 0.05, on the linear layers of the attention and feed-forward blocks (`down_proj`, `gate_proj`, `k_proj`, `o_proj`, `q_proj`, `up_proj`, `v_proj`) |
| Trainable parameters | 33,030,144 |
| Loss | On the reply only, not on the prompt |
| Batch | 2 sequences, gradients accumulated over 8 batches |
| Optimizer | `paged_adamw_8bit`, learning rate 0.0002, cosine schedule, 46 warm-up steps, gradient norm clipped at 1.0 |
| Length | 3 epochs planned (918 steps); longest sequence 2,048 tokens, nothing truncated |
| **Published checkpoint** | **step 500**, the one with the most development questions right by execution accuracy (358 of 499) |
| Seed | 42 |
| Hardware | one `ml.g5.xlarge` instance (an NVIDIA A10G, 24 GB), 4.18 hours for the whole run |
| Libraries | torch 2.14.1, transformers 5.23.0, peft 0.21.3, trl 1.18.0, datasets 5.2.0, accelerate 1.15.0, bitsandbytes 0.50.2 |

`training_metadata.json` holds the same facts for a program to read.

## Results

All figures are on this project's test split: 2,898 questions from
43 companies that the model never saw. Each model answered once, after
every setting was fixed on the development questions.

This adapter (arm A1 of the comparison):

| Metric | Right | Share | Interval |
|---|---:|---:|---:|
| Execution accuracy | 2,121 | 73.19% | 71.60 to 74.78 |
| In the format | 2,898 | 100.00% | 100.00 to 100.00 |
| Answer accuracy | 1,941 | 66.98% | 65.29 to 68.67 |
| Program match | 1,972 | 68.05% | 66.36 to 69.74 |

*Execution accuracy* is the main figure: the program runs and gives the expected result.
*Program match* asks for the same program as the reference. *Answer accuracy* looks at
the result the model wrote itself, without running anything. Intervals are
95% confidence intervals, in percent.

Beside the other models of the comparison:

| Arm | Model | Execution accuracy | Interval | Per 1,000 questions | Per 1,000 correct answers |
|---|---|---:|---:|---:|---:|
| A0 | Qwen3-4B, untuned | 0.93% | 0.59 to 1.31 | $0.562 | $60.32 |
| **A1** (this adapter) | Qwen3-4B with QLoRA | 73.19% | 71.60 to 74.78 | $0.346 | $0.473 |
| B0-Lite | Nova Lite, untuned | 16.70% | 15.36 to 18.08 | $0.047 | $0.283 |
| B0-Pro | Nova Pro, untuned | 35.37% | 33.64 to 37.09 | $0.624 | $1.76 |
| B1-Lite | Nova Lite, tuned | 71.39% | 69.74 to 73.02 | $0.044 | $0.062 |
| B1-Pro | Nova Pro, tuned | 75.19% | 73.60 to 76.74 | $0.590 | $0.784 |

The four Nova arms are hosted models on Amazon Bedrock. Their tuned versions cannot be
downloaded; they are here for comparison only. Costs are at list price.

By the length of the reference program:

| Steps in the reference program | Questions | Right | Execution accuracy |
|---|---:|---:|---:|
| 1 | 1,709 | 1,385 | 81.04% |
| 2 | 966 | 640 | 66.25% |
| 3 or more | 223 | 96 | 43.05% |

The 777 wrong replies, by how the program differs from the reference one. Each
reply is counted once, under the first row that fits. A category says how a program
differs, not why the model wrote it:

| How the program differs | Replies |
|---|---:|
| Program does not parse | 6 |
| Program cannot be run | 3 |
| Number where yes or no was due, or the reverse | 2 |
| More steps than the reference program | 101 |
| Fewer steps than the reference program | 274 |
| Another operation | 87 |
| Other numbers, same operations | 247 |
| Arguments reversed | 57 |

On the development questions, this adapter got 358 of 499
right (71.74%) on the hardware it was trained on, and 358 of
499 (71.74%) on a laptop GPU. The files of this repository, loaded from a folder, gave the same reply as the original checkpoint on every one of the 499 development questions.

## Cost

| | |
|---|---|
| Training run | $5.88 |
| Scoring the checkpoints to choose one | $1.94 |
| Answering 1,000 questions, as a batch job on one `ml.g5.xlarge` | $0.346 |
| The same, per 1,000 questions answered correctly | $0.473 |

List prices of Amazon SageMaker in us-east-1 as read in November 2026; a job's start-up
is included.

## Limitations

- **The evidence is given.** At training and at test time the model received the passages
  the dataset marks as the evidence, not the whole report. Finding the evidence is not
  measured here, so these figures are **not comparable with published FinQA results**.
- **Trust the program, not the answer line.** The model's own answer was right for
  66.98% of the questions; the executed program for 73.19%.
- **Longer calculations are harder.** Execution accuracy is 81.04% when the
  reference program has one step and 43.05% when it has three or more.
- **One training run.** The intervals describe the test questions. They do not say what
  another run with another seed would give.
- **One prompt, one task.** The adapter was trained on one wording and one output format.
  Other wording, tables with merged headers, or questions that need an operation the
  program language lacks are outside what it learned.
- **English-language reports of large listed companies**, as in FinQA.
- **A research artifact, not financial advice.** Check every result before relying on it.

## Files

| File | Holds |
|---|---|
| `adapter_model.safetensors`, `adapter_config.json` | The adapter. SHA-256 of the weights: `a8c9561b4fc6e560a53d5104c0118e5d088fcfbb579b57b8bfb24714c559902b` |
| `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` | The base model's tokenizer, unchanged |
| `prompt.json` | The system text, the labels of the user text, the reply limit; prompt id `d63742a62b76...` |
| `finqa_split_manifest.json` | Which example was used for what; ids only, no dataset text |
| `training_metadata.json` | How the adapter was trained and chosen |
| `sample_input.json` | The invented sample used above |

## Licences and citation

This adapter is released under the Apache License, Version 2.0. What it is built on:

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
