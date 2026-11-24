# The packages

Written by `python -m scripts.hf_package cards`. Do not edit by hand: build again. This file holds counts, ids, and hashes only: no reply and no dataset text.

- **Adapter package** (`sudiptoHaldar/qwen3-4b-finqa-program-qlora`): **ready**
- **Merged copy** (`sudiptoHaldar/qwen3-4b-finqa-program-merged`): **ready**

Nothing is uploaded by this version. Publishing is the owner's step (req_006 v6.03).

## Checks

| Check |  | Result | Detail |
|---|---|---|---|
| P1 | Records in place | passed | the six-arm record passed C1 to C8, the cost record K1 to K8, the comparison's record R1 to R8; none is provisional |
| P2 | The source is arm A1 | passed | 20261120T182710Z-train-qwen3-4b-fit-4557/checkpoint-500, weights a8c9561b4fc6..., as the selection record says |
| P3 | Identity | passed | the adapter's model commit, prompt, and split are today's |
| P4 | Package contents | passed | each package holds the files it should and no other; the adapter files are the selected checkpoint's bytes |
| P5 | Nothing private | passed | no text file of either package matches a forbidden pattern; the sample is the demo's |
| P6 | Tokenizer | passed | the tokenizer files are the base model's and give the pinned fingerprint |
| P7 | Licences | passed | 3 entries, each confirmed by the owner within 30 days |
| P8 | The adapter package answers | passed | R2 gives the same reply as R1 on all 499 examples; R1 has 358 right, the recorded run 358; the adapter card's code ran to the sample's expected result |
| P9 | The merged copy answers | passed | R3 has 361 right against R1's 358; R4 has 357 right against R1's 358; the merged card's code ran both ways to the expected result |
| P10 | The cards | passed | each card is its template filled from today's records; the copy in the package and the committed copy are the same |

## The runs on `dev`

Each is an ordinary evaluation run on this machine. None is an arm of the comparison and none has a price.

|  | What | Execution | Answer | Program | Format | Run |
|---|---|---:|---:|---:|---:|---|
| R1 | the selected adapter, by name | 358 of 499 | 337 | 332 | 499 | `20261124T091907Z-eval-qwen3-4b-tuned-dev-7666` |
| R2 | the adapter package, from its folder | 358 of 499 | 337 | 332 | 499 | `20261124T092628Z-eval-qwen3-4b-package-dev-6009` |
| R3 | the merged copy, in 4-bit | 361 of 499 | 335 | 331 | 499 | `20261124T093421Z-eval-qwen3-4b-merged-dev-5494` |
| R4 | the merged copy, at full precision | 357 of 499 | 336 | 328 | 499 | `20261124T094038Z-eval-qwen3-4b-merged-full-dev-9df6` |

The recorded run of arm A1 on `dev`: 358 of 499 right on execution. Two runs closer than 10 examples are not told apart.

Against R1, reply by reply:

|  | Replies that differ | Right, against R1 | Right in R1, wrong here | Wrong in R1, right here |
|---|---:|---:|---:|---:|
| R2 | 0 | 0 | 0 | 0 |
| R3 | 157 | +3 | 6 | 9 |
| R4 | 199 | -1 | 15 | 14 |

The merged copy is withheld when R3 or R4 has 10 or more examples fewer right than R1.

## The cards' code

The code each card prints, run in an empty directory on the invented sample:

|  | Result | Executed value | Code |
|---|---|---|---|
| adapter | ran to the expected result | 0.1256140350877193 | `4951432bd6ab...` |
| merged_4bit | ran to the expected result | 0.1256140350877193 | `5436b7195101...` |
| merged_full | ran to the expected result | 0.1256140350877193 | `5436b7195101...` |

## The adapter package: files

| File | Bytes | SHA-256 |
|---|---:|---|
| `adapter_config.json` | 1,169 | `eb45c6b4aefcf52bb2eeb8db944385e1f09898c1b47d667ade924af7fc8c37c2` |
| `adapter_model.safetensors` | 66,127,776 | `a8c9561b4fc6e560a53d5104c0118e5d088fcfbb579b57b8bfb24714c559902b` |
| `finqa_split_manifest.json` | 303,891 | `a031241256656f7e94830afdfa34576a735c457396b048b7526ece7869c2edac` |
| `LICENSE` | 11,358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `merges.txt` | 1,671,853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `prompt.json` | 2,449 | `e8ed49e7d7b26911525c9ee7c2eb36bba04a78351d4fe0225fe3702365395df6` |
| `README.md` | 15,750 | `66f3c0ce3a2aacaaaf70f7e545d58cb29eca91943d101eba2a309128c9f9773c` |
| `sample_input.json` | 503 | `5b02531733ce51fd2816f0ed4cb6b5ff814d6f2080870ec10d12279369eef92e` |
| `tokenizer.json` | 11,422,654 | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `tokenizer_config.json` | 9,732 | `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` |
| `training_metadata.json` | 2,086 | `8bb28dde8d3653aa942bca4c616521eff4ca2007c259c2c5b3ab30d235195810` |
| `vocab.json` | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |

## The merged copy: files

| File | Bytes | SHA-256 |
|---|---:|---|
| `config.json` | 1,664 | `9ded6bb06e5546765cd26883f3e4267d858cea59e43bcc8b33645f2b6f4c8890` |
| `generation_config.json` | 163 | `49e91b76eaa763f95a5d6d18889739d49d47d0eaa6c1c7acff24e430f43a8fa3` |
| `LICENSE` | 11,343 | `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e` |
| `merge_metadata.json` | 862 | `1f29cfacd9d7dce92652edc52f5b1dcc5713d3124a99d8ec5a686cb974703335` |
| `merges.txt` | 1,671,853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `model-00001-of-00003.safetensors` | 3,981,504,152 | `e4a38c5f0469ef33b2dc5b32390a5179bd94a4f486fab1311cb5855cadbd07f3` |
| `model-00002-of-00003.safetensors` | 3,961,229,912 | `0125e46938361388087c95be90bf7613d327faf9498da1c2934444f4fde9eac4` |
| `model-00003-of-00003.safetensors` | 102,247,936 | `6ec8fe2b20b8823501324713ba379bd65c90e1b960f79fd8822481a570d70b43` |
| `model.safetensors.index.json` | 33,261 | `646973887378a236d3cf81292de5cd631218b940ec0aecf03a16f57890572efc` |
| `NOTICE` | 614 | `70a18c9a7415730b1757fbfe95da53e59b177383f4681ff83ad680feae6485b8` |
| `prompt.json` | 2,449 | `e8ed49e7d7b26911525c9ee7c2eb36bba04a78351d4fe0225fe3702365395df6` |
| `README.md` | 11,804 | `c22a67e10dfce18132f6371f07ed1177acd0e836471b040dd2e990813705d6b9` |
| `sample_input.json` | 503 | `5b02531733ce51fd2816f0ed4cb6b5ff814d6f2080870ec10d12279369eef92e` |
| `tokenizer.json` | 11,422,654 | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `tokenizer_config.json` | 9,732 | `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` |
| `vocab.json` | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |

## Licences

| What | Licence | Source | Read on | Confirmed by the owner |
|---|---|---|---|---|
| Qwen3-4B: the base model | `apache-2.0` | https://huggingface.co/Qwen/Qwen3-4B | 2026-11-24 | 2026-11-24 |
| FinQA: the questions, programs, and answers | `mit` | https://github.com/czyssrs/FinQA | 2026-11-24 | 2026-11-24 |
| FinTabNet: the report tables that FinQA's examples are built on | `cdla-permissive-1.0` | https://huggingface.co/datasets/bsmock/FinTabNet.c | 2026-11-24 | 2026-11-24 |
