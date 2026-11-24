---
base_model: {{base_model}}
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
license: {{license}}
language:
- en
datasets:
- {{dataset_id}}
tags:
- finqa
- finance
- question-answering
- program-synthesis
- merged-lora
model-index:
- name: {{merged_name}}
  results:
  - task:
      type: text-generation
      name: Writing a program that answers a question about a financial report
    dataset:
      name: "FinQA, this project's own split by company (not an official split)"
      type: {{dataset_id}}
      revision: {{dataset_commit}}
      split: "dev (this project's split: {{dev_examples}} questions of {{dev_companies}} companies)"
    metrics:
    - type: execution_accuracy
      name: Execution accuracy, loaded in 4-bit
      value: {{r3_percent}}
    - type: execution_accuracy
      name: Execution accuracy, at full precision
      value: {{r4_percent}}
---

# {{merged_name}}

[`{{base_model}}`](https://huggingface.co/{{base_model}}) with the LoRA adapter
[`{{adapter_repo}}`](https://huggingface.co/{{adapter_repo}}) merged into its weights. It
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

- **Author:** {{author}}
- **Licence:** {{license}} (it carries Qwen's weights; see `LICENSE` and `NOTICE`)
- **Base model:** `{{base_model}}` at commit `{{base_commit}}`
- **Adapter that was merged:** `{{adapter_repo}}`, weights SHA-256 `{{adapter_sha256}}`
- **Code, with the executor and the full comparison:** {{code_repository}}

## How to use

The code below is executed before this card is published, with `QUANTIZED` set both ways:
it loads the files of this repository, asks an invented sample, and its reply must run to
the expected result. It needs a CUDA GPU: about 6 GB of memory in 4-bit, about 10 GB at
full precision.

```python
{{usage_code}}
```

- **The prompt.** The system text and the layout of the user text come from
  `prompt.json`, word for word. The model saw no other wording.
- **Thinking is off** (`enable_thinking=False`).
- **Greedy decoding.** `generation_config.json` in this repository decodes greedily and
  stops at the end tokens. The base model's own settings sample; they were replaced.
- At most {{max_new_tokens}} new tokens.

For the invented sample the reply is:

```
{{sample_reply}}
```

The reply is text. Parse the `PROGRAM:` line and execute it with code you trust; never
pass it to `eval`. The project's executor is in the code repository
(`qlora_finetuning/dsl/`).

## How it was made

| | |
|---|---|
| Base weights | `{{base_model}}` at commit `{{base_commit}}`, at full precision, never the 4-bit model |
| Adapter | `{{adapter_repo}}`: LoRA rank {{lora_r}}, alpha {{lora_alpha}}, on {{adapted_tensors}} weight matrices |
| Merge | Each adapted weight becomes W + (alpha / r) x B x A, computed in {{merge_compute_dtype}} and saved as {{saved_dtype}} |
| Size | {{parameters}} parameters; {{weight_files}} weight files |
| Generation settings | Replaced by greedy decoding |
| Tokenizer | The base model's files, unchanged |

How the adapter itself was trained is on its own card.

## Results

### This copy, on the development questions

{{dev_examples}} questions of {{dev_companies}} companies, on one laptop GPU, in batches
of {{batch_size}}:

{{merged_dev_table}}

{{merged_sentence}}

### The adapter, on the test split

These figures are **the adapter's**, measured on the 4-bit base model with the adapter
attached: {{test_examples}} questions from {{test_companies}} companies that the model
never saw. They are given here because this copy has no test result of its own.

{{a1_table}}

Intervals are {{confidence_percent}}% confidence intervals, in percent. The other models
of the comparison, and its cost figures, are on the adapter's card.

## Limitations

- **The evidence is given.** At training and at test time the model received the passages
  the dataset marks as the evidence, not the whole report. Finding the evidence is not
  measured, so these figures are **not comparable with published FinQA results**.
- **No test result of its own**, as said above.
- **Trust the program, not the answer line.** The adapter's own answer was right for
  {{a1_answer}}% of the test questions; its executed program for {{a1_execution}}%.
- **Longer calculations are harder.** The adapter's execution accuracy is {{steps_one}}%
  when the reference program has one step and {{steps_many}}% when it has three or more.
- **One training run, one prompt, one task.** Other wording, tables with merged headers,
  or questions that need an operation the program language lacks are outside what it
  learned.
- **A reply can change with the batch size and with the GPU.**
- **A research artifact, not financial advice.** Check every result before relying on it.

## Files

| File | Holds |
|---|---|
| {{weight_files_cell}}, `model.safetensors.index.json` | The merged weights |
| `config.json`, `generation_config.json` | The model's configuration, and greedy generation settings |
| `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` | The base model's tokenizer, unchanged |
| `prompt.json` | The system text, the labels of the user text, the reply limit |
| `merge_metadata.json` | What was merged into what, and how |
| `sample_input.json` | The invented sample used above |
| `LICENSE`, `NOTICE` | Qwen's licence, and what was changed |

## Licences and citation

This model is a modified copy of `{{base_model}}` and is released under {{license_name}},
as the original is. What it is built on:

{{licence_table}}

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
