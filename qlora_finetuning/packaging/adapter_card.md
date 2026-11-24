---
base_model: {{base_model}}
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
license: {{license}}
language:
- en
datasets:
- {{dataset_id}}
tags:
- lora
- qlora
- finqa
- finance
- question-answering
- program-synthesis
model-index:
- name: {{adapter_name}}
  results:
  - task:
      type: text-generation
      name: Writing a program that answers a question about a financial report
    dataset:
      name: "FinQA, this project's own split by company (not the official test split)"
      type: {{dataset_id}}
      revision: {{dataset_commit}}
      split: "test (this project's split: {{test_examples}} questions of {{test_companies}} companies)"
    metrics:
    - type: execution_accuracy
      name: Execution accuracy
      value: {{a1_execution}}
    - type: program_match
      name: Program match
      value: {{a1_program}}
    - type: answer_accuracy
      name: Answer accuracy
      value: {{a1_answer}}
    - type: format_validity
      name: Replies in the format
      value: {{a1_format}}
---

# {{adapter_name}}

A LoRA adapter for [`{{base_model}}`](https://huggingface.co/{{base_model}}), trained with
QLoRA on FinQA. Given text from a company's financial report, a table from the same
report, and a question, it writes a short program that computes the answer, and then the
result. **The program is meant to be run by an executor: the model is not the calculator.**

> **Do not report this model's score on the official FinQA test split.** The questions it
> was trained on were drawn from all of FinQA: the official training, validation, and test
> files were pooled and split again by company. A score on the official validation or test
> split is therefore inflated. `finqa_split_manifest.json` in this repository says, by
> example id, which questions were used for training, for development, and for testing.

- **Author:** {{author}}
- **Licence:** {{license}}
- **Base model:** `{{base_model}}` at commit `{{base_commit}}`
- **Code, with the executor and the full comparison:** {{code_repository}}
{{merged_line}}

## What it does

Input, as the model sees it (an invented company; every figure was made up):

```
{{sample_user}}
```

Reply:

```
{{sample_reply}}
```

The program language has ten operations (`add`, `subtract`, `multiply`, `divide`, `exp`,
`greater`, and four that work on a table row) and refers to an earlier step as `#0`, `#1`.
The system text in `prompt.json` describes it completely. Running the program above gives
`{{sample_value}}`.

## How to use

The code below is executed before this card is published: it loads the files of this
repository, asks the invented sample, and its reply must run to the expected result.
It needs a CUDA GPU with about 6 GB of memory free.

```python
{{usage_code}}
```

Four things in it matter, because they are the configuration the figures below were
measured with:

- **The prompt.** The system text and the layout of the user text come from
  `prompt.json`, word for word. The adapter saw no other wording.
- **Thinking is off** (`enable_thinking=False`).
- **4-bit NF4 quantization** of the base model, with the adapter attached and not merged.
- **Greedy decoding**, stopping at the end tokens, with at most {{max_new_tokens}} new tokens.

The measured runs asked questions in batches of {{batch_size}} on one GPU. A reply can change
with the batch size and with the GPU, so another setup may answer a few questions
differently.

The reply is text. Parse the `PROGRAM:` line and execute it with code you trust; never
pass it to `eval`. The project's executor is in the code repository
(`qlora_finetuning/dsl/`).

## Training data

| | |
|---|---|
| Dataset | [`{{dataset_id}}`](https://huggingface.co/datasets/{{dataset_id}}) at commit `{{dataset_commit}}`, a copy of [FinQA](https://github.com/czyssrs/FinQA) |
| Questions | {{pooled_examples}}: the official training, validation, and test files, pooled |
| Split | By company, so no company has questions on both sides: {{train_percent}}% for training and development, the rest for testing; seed {{split_seed}} |
| Used for training (`fit`) | {{fit_examples}} questions of {{fit_companies}} companies |
| Used to choose the checkpoint (`dev`) | {{dev_examples}} questions of {{dev_companies}} companies |
| Held out (`test`) | {{test_examples}} questions of {{test_companies}} companies, answered once |
| Split manifest | `finqa_split_manifest.json`, content hash `{{manifest_hash}}` |

Each training example gives the model the passages FinQA marks as the evidence for the
question, the whole table, and the question. See "Limitations".

## Training procedure

| | |
|---|---|
| Method | QLoRA: the base model in 4-bit ({{quantization_type}}, double quantization, {{compute_dtype}} compute), LoRA adapters trained in {{precision}} |
| LoRA | rank {{lora_r}}, alpha {{lora_alpha}}, dropout {{lora_dropout}}, on the linear layers of the attention and feed-forward blocks ({{target_modules}}) |
| Trainable parameters | {{trainable_parameters}} |
| Loss | On the reply only, not on the prompt |
| Batch | {{batch}} sequences, gradients accumulated over {{accumulation}} batches |
| Optimizer | {{optimizer}}, learning rate {{learning_rate}}, {{schedule}} schedule, {{warmup_steps}} warm-up steps, gradient norm clipped at {{max_grad_norm}} |
| Length | {{epochs}} epochs planned ({{steps_planned}} steps); longest sequence {{max_length}} tokens, nothing truncated |
| **Published checkpoint** | **step {{selected_step}}**, the one with the most development questions right by execution accuracy ({{recorded_dev_right}} of {{dev_examples}}) |
| Seed | {{seed}} |
| Hardware | one `{{instance_type}}` instance (an NVIDIA A10G, 24 GB), {{training_hours}} hours for the whole run |
| Libraries | {{libraries}} |

`training_metadata.json` holds the same facts for a program to read.

## Results

All figures are on this project's test split: {{test_examples}} questions from
{{test_companies}} companies that the model never saw. Each model answered once, after
every setting was fixed on the development questions.

This adapter (arm A1 of the comparison):

{{a1_table}}

*Execution accuracy* is the main figure: the program runs and gives the expected result.
*Program match* asks for the same program as the reference. *Answer accuracy* looks at
the result the model wrote itself, without running anything. Intervals are
{{confidence_percent}}% confidence intervals, in percent.

Beside the other models of the comparison:

{{six_arm_table}}

The four Nova arms are hosted models on Amazon Bedrock. Their tuned versions cannot be
downloaded; they are here for comparison only. Costs are at list price.

By the length of the reference program:

{{steps_table}}

The {{a1_wrong}} wrong replies, by how the program differs from the reference one. Each
reply is counted once, under the first row that fits. A category says how a program
differs, not why the model wrote it:

{{categories_table}}

On the development questions, this adapter got {{recorded_dev_right}} of {{dev_examples}}
right ({{recorded_dev_percent}}%) on the hardware it was trained on, and {{r1_right}} of
{{dev_examples}} ({{r1_percent}}%) on a laptop GPU. {{package_sentence}}

## Cost

| | |
|---|---|
| Training run | {{train_usd}} |
| Scoring the checkpoints to choose one | {{select_usd}} |
| Answering 1,000 questions, as a batch job on one `{{instance_type}}` | {{per_1000_questions}} |
| The same, per 1,000 questions answered correctly | {{per_1000_correct}} |

List prices of Amazon SageMaker in us-east-1 as read in November 2026; a job's start-up
is included.

## Limitations

- **The evidence is given.** At training and at test time the model received the passages
  the dataset marks as the evidence, not the whole report. Finding the evidence is not
  measured here, so these figures are **not comparable with published FinQA results**.
- **Trust the program, not the answer line.** The model's own answer was right for
  {{a1_answer}}% of the questions; the executed program for {{a1_execution}}%.
- **Longer calculations are harder.** Execution accuracy is {{steps_one}}% when the
  reference program has one step and {{steps_many}}% when it has three or more.
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
| `adapter_model.safetensors`, `adapter_config.json` | The adapter. SHA-256 of the weights: `{{adapter_sha256}}` |
| `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` | The base model's tokenizer, unchanged |
| `prompt.json` | The system text, the labels of the user text, the reply limit; prompt id `{{prompt_id_short}}` |
| `finqa_split_manifest.json` | Which example was used for what; ids only, no dataset text |
| `training_metadata.json` | How the adapter was trained and chosen |
| `sample_input.json` | The invented sample used above |

## Licences and citation

This adapter is released under {{license_name}}. What it is built on:

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
