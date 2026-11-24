# Arm A1 on dev: where it goes wrong

Training run `20261120T182710Z-train-qwen3-4b-fit-4557`, checkpoint 500, scored on `dev` (499 examples). Written by `python -m scripts.qlora_analyze`; do not edit. Every category is computed from the stored replies; none was assigned by reading.

**Right on execution**: 358 of 499 (71.7%). **In the format**: 499 (100.0%).

## Why replies are wrong

A reply gets the first category that fits, in this order.

| Category | Selected checkpoint | Share of all | Untuned model |
|---|---:|---:|---:|
| Cut off at the token limit (`cut_off`) | 0 | 0.0% | 45 |
| No program line (`no_program`) | 0 | 0.0% | 0 |
| Program does not parse (`bad_program`) | 0 | 0.0% | 383 |
| Program cannot be run (`not_executable`) | 2 | 0.4% | 39 |
| Number where yes or no was due, or the reverse (`kind`) | 0 | 0.0% | 0 |
| More steps than the gold program (`steps_more`) | 21 | 4.2% | 24 |
| Fewer steps than the gold program (`steps_fewer`) | 65 | 13.0% | 2 |
| Another operation (`operation`) | 12 | 2.4% | 0 |
| Other numbers, same operations (`evidence`) | 34 | 6.8% | 0 |
| Arguments reversed (`order`) | 7 | 1.4% | 0 |
| Gold program, result off by a factor (`scale`) | 0 | 0.0% | 0 |
| Gold program, result negated (`sign`) | 0 | 0.0% | 0 |
| Other (`other`) | 0 | 0.0% | 0 |
| **Right** | **358** | 71.7% | 6 |

Against the untuned model: 354 examples fixed, 2 broken, 4 right in both, 139 wrong in both.

Programs that did not run, by error: `row_not_numeric` 2.

Wrong results that are the gold value negated or off by a factor: scale 31, sign 10.

Of the wrong replies, 141 were in the format, and 3 had a right answer line over a wrong program.

## The format

0 replies were not in the strict format (0 of them right on execution).

Stop reasons: `end` 499.

## Over the checkpoints

| Checkpoint | Right | `bad_program` | `not_executable` | `kind` | `steps_more` | `steps_fewer` | `operation` | `evidence` | `order` | Dev loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 321 | 0 | 5 | 0 | 23 | 70 | 13 | 47 | 20 | 0.1643 |
| 200 | 322 | 1 | 2 | 0 | 34 | 59 | 16 | 50 | 15 | 0.1531 |
| 300 | 338 | 1 | 1 | 1 | 29 | 58 | 14 | 51 | 6 | 0.1414 |
| 400 | 350 | 0 | 4 | 1 | 12 | 66 | 11 | 44 | 11 | 0.1397 |
| 500 **(selected)** | 358 | 0 | 2 | 0 | 21 | 65 | 12 | 34 | 7 | 0.1321 |
| 600 | 356 | 0 | 1 | 0 | 23 | 61 | 12 | 37 | 9 | 0.1286 |
| 700 | 356 | 2 | 2 | 1 | 25 | 57 | 13 | 35 | 8 | 0.1330 |
| 800 | 358 | 1 | 3 | 0 | 25 | 58 | 14 | 32 | 8 | 0.1352 |
| 900 | 356 | 1 | 3 | 0 | 25 | 59 | 14 | 33 | 8 | 0.1351 |
| 918 | 356 | 1 | 3 | 0 | 27 | 57 | 15 | 32 | 8 | 0.1348 |

- The last three checkpoints are **flat** (noise: 10 examples).
- Accuracy was first within noise of its best at step 400.
- From the best checkpoint to the last: -2 examples.

## Where the failures sit

**By the length of the gold program (steps)**

| Group | Examples | Right | Rate |
|---|---:|---:|---:|
| 1 | 286 | 232 | 81.1% |
| 2 | 160 | 112 | 70.0% |
| 3 | 35 | 7 | 20.0% |
| 4 or more | 18 | 7 | 38.9% |

**By operation in the gold program**

| Group | Examples | Right | Rate |
|---|---:|---:|---:|
| add | 84 | 53 | 63.1% |
| divide | 330 | 234 | 70.9% |
| greater | 13 | 12 | 92.3% |
| multiply | 59 | 18 | 30.5% |
| subtract | 218 | 148 | 67.9% |
| table_average | 6 | 4 | 66.7% |
| table_max | 2 | 2 | 100.0% |
| table_sum | 2 | 2 | 100.0% |

**By company**

| Group | Examples | Right | Rate |
|---|---:|---:|---:|
| ADBE | 100 | 87 | 87.0% |
| CNP | 2 | 2 | 100.0% |
| EMR | 17 | 11 | 64.7% |
| FITB | 15 | 11 | 73.3% |
| FTV | 4 | 4 | 100.0% |
| HIG | 56 | 39 | 69.6% |
| IPG | 149 | 87 | 58.4% |
| MA | 39 | 28 | 71.8% |
| MAS | 41 | 35 | 85.4% |
| MRK | 10 | 6 | 60.0% |
| NTAP | 2 | 1 | 50.0% |
| NWS | 9 | 8 | 88.9% |
| ORLY | 23 | 17 | 73.9% |
| TXN | 3 | 1 | 33.3% |
| VLO | 29 | 21 | 72.4% |

**By prompt length**

| Group | Examples | Right | Rate |
|---|---:|---:|---:|
| quarter 1 (457 to 561 tokens) | 124 | 101 | 81.5% |
| quarter 2 (563 to 635 tokens) | 125 | 90 | 72.0% |
| quarter 3 (636 to 712 tokens) | 125 | 88 | 70.4% |
| quarter 4 (713 to 886 tokens) | 125 | 79 | 63.2% |

**Never right**: 100 examples were wrong at every checkpoint. Their ids, and the ids of the wrong examples by category, are in the JSON file beside this page.

## The finding

**T3**: accuracy is flat from the middle on and most failures are calculation. The table's change: `lora.r` to 32.

The finding is a reading of the table in the v3.05 spec, written before any result existed. It decides nothing: a corrective iteration is the owner's, and is allowed only when the selection's verdict is `missed`.
