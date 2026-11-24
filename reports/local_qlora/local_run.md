# The full training run on this machine

**A local run: not an arm of the comparison.** It was made on the laptop's own GPU with the config of the cloud run, as a learning exercise. It is never arm A1, it has no AWS bill, and it enters no cost table. Written by `python -m scripts.qlora_local note`; do not edit.

Training run `20261120T223038Z-train-qwen3-4b-fit-db4b`, commit `56dd25c89511`.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 Laptop GPU |
| Records, epochs, optimizer steps | 4,884, 3, 918 |
| Wall-clock time of training | 2.87 hours |
| Seconds per training sequence | 0.70 |
| Peak GPU memory | 11.00 GiB allocated, 15.41 GiB reserved |

## The checkpoints on `quick_dev` (200 examples)

The rule of the cloud selection, applied here: the most examples right by execution accuracy, then format validity, then the earlier step. The loss on dev is shown and is never used.

| Checkpoint | Execution | Format | Answer | Program | Dev loss | Run |
|---:|---:|---:|---:|---:|---:|---|
| untuned | 2 | 186 | 23 | 0 |  | `20261121T030738Z-eval-qwen3-4b-quick-dev-09a7` |
| 100 | 127 | 200 | 112 | 118 | 0.1655 | `20261121T031110Z-eval-qwen3-4b-tuned-quick-dev-a309` |
| 200 | 133 | 200 | 121 | 125 | 0.1506 | `20261121T031329Z-eval-qwen3-4b-tuned-quick-dev-62ae` |
| 300 | 136 | 200 | 121 | 130 | 0.1385 | `20261121T031552Z-eval-qwen3-4b-tuned-quick-dev-34df` |
| 400 | 139 | 200 | 133 | 132 | 0.1383 | `20261121T031816Z-eval-qwen3-4b-tuned-quick-dev-746d` |
| 500 | 142 | 200 | 135 | 135 | 0.1331 | `20261121T032035Z-eval-qwen3-4b-tuned-quick-dev-3313` |
| 600 | 140 | 200 | 129 | 132 | 0.1279 | `20261121T032256Z-eval-qwen3-4b-tuned-quick-dev-d502` |
| 700 | 142 | 200 | 131 | 135 | 0.1310 | `20261121T032512Z-eval-qwen3-4b-tuned-quick-dev-fe58` |
| 800 | 145 | 200 | 135 | 137 | 0.1303 | `20261121T032732Z-eval-qwen3-4b-tuned-quick-dev-7bcb` |
| 900 **(selected)** | 146 | 200 | 133 | 137 | 0.1300 | `20261121T032956Z-eval-qwen3-4b-tuned-quick-dev-87f1` |
| 918 | 146 | 200 | 136 | 138 | 0.1299 | `20261121T033218Z-eval-qwen3-4b-tuned-quick-dev-3e2f` |

**Selected here**: checkpoint 900 (`20261120T223038Z-train-qwen3-4b-fit-db4b/checkpoint-900`), weights `d68591103a8f...`: 73.00% execution accuracy on `quick_dev`.

## The selected checkpoint on all of dev

360 of 499 right on execution (72.14%); 499 in the format (100.00%). Run `20261121T050005Z-eval-qwen3-4b-tuned-dev-abf8`.

Why its replies were wrong there, by computed category:

| Category | Replies |
|---|---:|
| `bad_program` | 3 |
| `not_executable` | 1 |
| `steps_more` | 20 |
| `steps_fewer` | 54 |
| `operation` | 16 |
| `evidence` | 37 |
| `order` | 8 |

## Local against cloud

| | This machine | Cloud (arm A1) |
|---|---:|---:|
| GPU | NVIDIA GeForce RTX 4090 Laptop GPU | NVIDIA A10G |
| Training, hours | 2.87 | 4.07 |
| Seconds per training sequence | 0.70 | 0.99 |
| Selected checkpoint | 900 | 500 |
| Right on execution, dev | 360 of 499 | 358 of 499 |
| In the format, dev | 499 | 499 |

The local checkpoint has +2 examples right against the cloud's: within noise (under 10 examples): the two runs are not told apart, which is what one pipeline in two places should give.

The speeds compare two GPUs, not two costs: the cloud run is billed and this one is not, and no price is put on it.

## What this is not

- The checkpoints were chosen on `quick_dev` and the cloud's on dev: the two per-checkpoint tables are not comparable row by row. Only the two selected checkpoints on dev are.
- Two runs with one config and one seed are two adapters. A small difference says the pipeline is the same in both places, not that one GPU trains better.
- No test example was used.
