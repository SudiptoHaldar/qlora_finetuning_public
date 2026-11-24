"""The statistics of the comparison: intervals and paired tests (req_005 v5.03).

Pure functions on lists of right and wrong. They know nothing of runs, arms, or files.

* ``bootstrap_rates`` resamples the examples with replacement and gives, for each column,
  its rate in every resample. **One set of resampled examples serves all columns**, so the
  difference between two columns in a resample is a paired difference. With ``groups`` it
  resamples whole groups instead (the companies of the test split), and a resample's rate
  is the right answers over the examples of the groups drawn.
* ``interval`` is the percentile interval of those rates.
* ``mcnemar_exact`` is McNemar's exact test: the two-sided binomial test on the examples
  that exactly one of two arms got right. It is computed with whole numbers.
* ``holm`` adjusts a family of p-values with Holm's step-down method.

Usage::

    from qlora_finetuning.comparison.statistics import (
        bootstrap_rates, holm, interval, mcnemar_exact,
    )

    rates = bootstrap_rates({"A1": a1, "B1-Pro": pro}, resamples=10000, seed=20261115)
    low, high = interval(rates["A1"], 95)
    low, high = interval(rates["A1"] - rates["B1-Pro"], 95)      # the paired difference
    by_company = bootstrap_rates(columns, resamples=10000, seed=20261116, groups=companies)
    p = mcnemar_exact(163, 221)                                   # a Fraction
    adjusted = holm([p, other])

The same seed and the same lists give the same numbers on every machine: the resampling
uses numpy's legacy ``RandomState``, whose stream is frozen, with a fixed integer type.
Import this module by its full name; it is not the standard library's ``statistics``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import comb
from typing import Any

import numpy

CHUNK = 500  # resamples drawn at a time: 10,000 at once would need hundreds of megabytes


def _matrix(columns: Mapping[str, Sequence[bool]]) -> tuple[list[str], Any]:
    names = list(columns)
    if not names:
        raise ValueError("there is no column to resample")
    lengths = {len(columns[name]) for name in names}
    if len(lengths) != 1:
        raise ValueError("the columns do not have the same number of examples")
    if lengths == {0}:
        raise ValueError("there is no example to resample")
    matrix = numpy.array([[bool(v) for v in columns[name]] for name in names], dtype=numpy.int64)
    return names, matrix.T  # one row per example, one column per name


def bootstrap_rates(
    columns: Mapping[str, Sequence[bool]],
    *,
    resamples: int,
    seed: int,
    groups: Sequence[Any] | None = None,
    chunk: int = CHUNK,
) -> dict[str, Any]:
    """Each column's rate in every resample: ``{name: array of `resamples` rates}``.

    ``groups`` gives one label per example; whole groups are then resampled.
    """
    if resamples < 1:
        raise ValueError("resamples must be 1 or more")
    names, matrix = _matrix(columns)
    examples = matrix.shape[0]
    if groups is None:
        sums = matrix
        sizes = numpy.ones(examples, dtype=numpy.int64)
    else:
        if len(groups) != examples:
            raise ValueError("there must be one group label per example")
        labels = sorted({str(label) for label in groups})
        position = {label: index for index, label in enumerate(labels)}
        member = numpy.array([position[str(label)] for label in groups], dtype=numpy.int64)
        sums = numpy.zeros((len(labels), len(names)), dtype=numpy.int64)
        numpy.add.at(sums, member, matrix)
        sizes = numpy.bincount(member, minlength=len(labels)).astype(numpy.int64)

    units = sums.shape[0]
    state = numpy.random.RandomState(seed)
    parts = []
    done = 0
    while done < resamples:
        count = min(chunk, resamples - done)
        drawn = state.randint(0, units, size=(count, units), dtype=numpy.int64)
        # How often each unit was drawn in each resample: one row per resample.
        rows = numpy.arange(count, dtype=numpy.int64)[:, None] * units
        times = numpy.bincount((drawn + rows).ravel(), minlength=count * units)
        times = times.reshape(count, units)
        right = times @ sums
        asked = times @ sizes
        parts.append(right / asked[:, None])
        done += count
    rates = numpy.concatenate(parts, axis=0)
    return {name: rates[:, index] for index, name in enumerate(names)}


def interval(values: Any, confidence_percent: int) -> tuple[float, float]:
    """The percentile interval of resampled values at the given confidence."""
    if not 0 < confidence_percent < 100:
        raise ValueError("the confidence must be between 0 and 100 percent")
    tail = (100 - confidence_percent) / 2
    low, high = numpy.percentile(numpy.asarray(values, dtype=float), [tail, 100 - tail])
    return float(low), float(high)


def mcnemar_exact(only_first: int, only_second: int) -> Fraction:
    """McNemar's exact two-sided p-value from the two counts of discordant examples.

    Under "no difference" each discordant example is as likely to favour either arm, so the
    smaller count is binomial with one half. With no discordant example the value is 1.
    """
    if only_first < 0 or only_second < 0:
        raise ValueError("a count cannot be negative")
    total = only_first + only_second
    if total == 0:
        return Fraction(1)
    smaller = min(only_first, only_second)
    tail = sum(comb(total, k) for k in range(smaller + 1))
    return min(Fraction(1), Fraction(2 * tail, 2**total))


def holm(p_values: Sequence[Fraction]) -> list[Fraction]:
    """Holm's adjusted p-values, in the order given.

    The smallest of ``m`` values is multiplied by ``m``, the next by ``m - 1``, and so on;
    an adjusted value is never below the one before it and never above 1.
    """
    values = [Fraction(p) for p in p_values]
    if any(p < 0 or p > 1 for p in values):
        raise ValueError("a p-value must be from 0 to 1")
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    adjusted: list[Fraction] = [Fraction(0)] * len(values)
    highest = Fraction(0)
    for rank, index in enumerate(order):
        highest = max(highest, min(Fraction(1), (len(values) - rank) * values[index]))
        adjusted[index] = highest
    return adjusted
