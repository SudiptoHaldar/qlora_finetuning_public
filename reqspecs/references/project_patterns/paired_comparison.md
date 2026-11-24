# Comparing two runs

**Defined by**: req_005 v5.03 (the comparison report)
**Code**: `qlora_finetuning/comparison/statistics.py`, `outcomes.py`, `report_record.py`

Any statement that one run is better than another, or how sure an accuracy figure is, comes
from these modules. No other code computes a confidence interval or a p-value.

## The rule

| Question | Answer with |
|---|---|
| How sure is this rate? | `bootstrap_rates` and `interval`: a percentile bootstrap over the examples |
| How sure is the difference between two runs? | The same resamples for both columns, then `interval` of the difference: the difference stays paired |
| Could the difference be chance? | `mcnemar_exact` on the two counts of examples that exactly one run got right |
| Several pairs tested at once? | `holm` over the whole family; show the raw and the adjusted value |
| Are the examples independent? | No: the split is by company. Give the interval by company (`groups=`) beside the one by example |

A difference is **settled** only when its adjusted p-value is below one minus the
confidence and neither interval includes zero. Otherwise it is reported as "not settled",
whatever its p-value.

## How the runs are read

`outcomes.read_arms({arm: run_id}, ctx)` gives, per example and in dataset order, true or
false for each of the four metrics, the error category, and the gold step count. It rescores
the stored replies with `score_reply` and `classify`; the totals come from `summarize`. An
`ArmOutcomes` holds ids, labels, and counts: no reply and no gold text.

## Reuse checklist

- [ ] Take the run ids from a committed record (the six-arm record for test), never by arm name.
- [ ] Check that both runs hold the same example ids in the same order before pairing them.
- [ ] Check that the rescored counts equal the recorded ones before quoting anything.
- [ ] Resample with `bootstrap_rates`, which uses numpy's legacy `RandomState` with a fixed
      integer type: the same seed gives the same bytes on every machine. Do not use
      `default_rng` for a committed figure.
- [ ] A company resample divides by the examples it drew, not by the split's size.
- [ ] Keep p-values as `Fraction` until they are shown: a float underflows.
- [ ] Name the pairs in config before computing a test, and say when they were named after
      the counts were known.
- [ ] A committed file gets counts and example ids only. Text goes to `artifacts/`.
