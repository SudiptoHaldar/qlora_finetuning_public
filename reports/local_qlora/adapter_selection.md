# Arm A1: the selected checkpoint

Training run `20261120T182710Z-train-qwen3-4b-fit-4557`, scored on `dev` (499 examples). Written by `python -m scripts.qlora_select`; do not edit.

**Selected**: checkpoint 500 (`20261120T182710Z-train-qwen3-4b-fit-4557/checkpoint-500`), weights `a8c9561b4fc6...`.

**Verdict**: met. Execution accuracy 71.74% (at least 40% asked), format validity 100.00% (at least 98%).

The rule: the most examples right by execution accuracy; a tie goes to the higher format validity, then to the earlier step. The loss on dev is shown and is never used.

| Checkpoint | Execution | Format | Answer | Program | Execution, tight | Dev loss | Run |
|---:|---:|---:|---:|---:|---:|---:|---|
| untuned | 6 | 457 | 60 | 2 | 6 |  | `20261121T070030Z-eval-qwen3-4b-dev-2416` |
| 100 | 321 | 499 | 287 | 291 | 321 | 0.1643 | `20261121T071054Z-eval-qwen3-4b-tuned-dev-1bf0` |
| 200 | 322 | 498 | 290 | 299 | 321 | 0.1531 | `20261121T071743Z-eval-qwen3-4b-tuned-dev-a173` |
| 300 | 338 | 499 | 317 | 308 | 337 | 0.1414 | `20261121T072435Z-eval-qwen3-4b-tuned-dev-5c2d` |
| 400 | 350 | 499 | 321 | 320 | 349 | 0.1397 | `20261121T073119Z-eval-qwen3-4b-tuned-dev-2711` |
| 500 **(selected)** | 358 | 499 | 338 | 333 | 358 | 0.1321 | `20261121T073743Z-eval-qwen3-4b-tuned-dev-fd5b` |
| 600 | 356 | 499 | 341 | 328 | 355 | 0.1286 | `20261121T074423Z-eval-qwen3-4b-tuned-dev-78d8` |
| 700 | 356 | 499 | 340 | 326 | 355 | 0.1330 | `20261121T075115Z-eval-qwen3-4b-tuned-dev-86e9` |
| 800 | 358 | 499 | 343 | 329 | 357 | 0.1352 | `20261121T075759Z-eval-qwen3-4b-tuned-dev-a994` |
| 900 | 356 | 499 | 342 | 327 | 356 | 0.1351 | `20261121T080439Z-eval-qwen3-4b-tuned-dev-d886` |
| 918 | 356 | 499 | 342 | 329 | 356 | 0.1348 | `20261121T081121Z-eval-qwen3-4b-tuned-dev-19c3` |

## Notes

- Above the untuned model by 70.54 points of execution accuracy.
- The selected checkpoint is not the last one.
- The loss on dev was lowest at step 600.
- This figure is the best of the checkpoints on the examples they were chosen on, so it flatters a little. The test split is what the comparison reports.

## What it cost

| Job | Task | Instance | Billable seconds | At the list price |
|---|---|---|---:|---:|
| `qlora-train-958bed10-a1` | train | ml.g5.xlarge | 15,037 | $5.88 |
| `qlora-score-63b89247-a1` | score | ml.g5.xlarge | 4,965 | $1.94 |
