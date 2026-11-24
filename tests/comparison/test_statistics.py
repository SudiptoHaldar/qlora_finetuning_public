"""Tests for the comparison's statistics: bootstrap, exact test, Holm (req_005 v5.03)."""

from __future__ import annotations

from fractions import Fraction

import pytest

from qlora_finetuning.comparison.statistics import (
    bootstrap_rates,
    holm,
    interval,
    mcnemar_exact,
)

FIRST = [index % 4 != 0 for index in range(200)]  # 150 of 200 right
SECOND = [index % 3 != 0 for index in range(200)]


def rates(columns, **options):
    settings = {"resamples": 400, "seed": 7}
    return bootstrap_rates(columns, **(settings | options))


# --------------------------------------------------------------------------- bootstrap


def test_the_same_seed_gives_the_same_rates_and_another_seed_others():
    once, again = rates({"a": FIRST}), rates({"a": FIRST})
    assert list(once["a"]) == list(again["a"])
    assert list(once["a"]) != list(rates({"a": FIRST}, seed=8)["a"])


def test_the_number_of_resamples_is_what_was_asked_whatever_the_chunk():
    assert len(rates({"a": FIRST}, resamples=1234)["a"]) == 1234
    small = rates({"a": FIRST}, resamples=30, chunk=7)
    assert list(small["a"]) == list(rates({"a": FIRST}, resamples=30, chunk=30)["a"])


def test_the_interval_is_around_the_rate():
    low, high = interval(rates({"a": FIRST})["a"], 95)
    assert low < 0.75 < high and 0.6 < low and high < 0.9


def test_an_arm_right_everywhere_has_an_interval_of_one_point():
    always = rates({"a": [True] * 50, "b": [False] * 50})
    assert interval(always["a"], 95) == (1.0, 1.0)
    assert interval(always["b"], 95) == (0.0, 0.0)


def test_one_resample_serves_all_columns_so_a_difference_is_paired():
    both = rates({"a": FIRST, "same": FIRST, "b": SECOND})
    assert not (both["a"] - both["same"]).any()
    alone = rates({"b": SECOND})
    assert list(alone["b"]) == list(both["b"])


def test_groups_of_one_example_give_the_interval_by_example():
    labels = [f"g{index:03d}" for index in range(len(FIRST))]
    by_example = rates({"a": FIRST})["a"]
    by_group = rates({"a": FIRST}, groups=labels)["a"]
    assert list(by_example) == list(by_group)


def test_groups_that_are_right_or_wrong_together_widen_the_interval():
    right = [(index // 20) % 2 == 0 for index in range(200)]  # ten groups, all or nothing
    labels = [index // 20 for index in range(200)]
    low, high = interval(rates({"a": right})["a"], 95)
    wide_low, wide_high = interval(rates({"a": right}, groups=labels)["a"], 95)
    assert wide_high - wide_low > 2 * (high - low)


def test_a_group_resample_divides_by_the_examples_it_drew():
    labels = ["small"] * 2 + ["large"] * 98
    always = rates({"a": [True] * 100, "b": [True] * 2 + [False] * 98}, groups=labels)
    assert set(always["a"]) == {1.0}
    assert all(0.0 <= value <= 1.0 for value in always["b"])
    assert 1.0 in set(always["b"]) and 0.0 in set(always["b"])


@pytest.mark.parametrize(
    ("columns", "options", "message"),
    [
        ({}, {}, "no column"),
        ({"a": []}, {}, "no example"),
        ({"a": [True], "b": [True, False]}, {}, "same number"),
        ({"a": [True, False]}, {"groups": ["x"]}, "one group label per example"),
        ({"a": [True]}, {"resamples": 0}, "1 or more"),
    ],
)
def test_bad_input_is_refused(columns, options, message):
    with pytest.raises(ValueError, match=message):
        rates(columns, **options)


def test_interval_percentiles_and_a_bad_confidence():
    assert interval(range(101), 95) == pytest.approx((2.5, 97.5))
    assert interval(range(101), 50) == pytest.approx((25.0, 75.0))
    with pytest.raises(ValueError, match="between 0 and 100"):
        interval([1.0], 100)


# --------------------------------------------------------------------------- McNemar


def test_exact_test_by_hand():
    assert mcnemar_exact(0, 0) == 1
    assert mcnemar_exact(5, 5) == 1
    assert mcnemar_exact(0, 10) == Fraction(2, 1024)
    assert mcnemar_exact(1, 3) == Fraction(5, 8)  # (1 + 4) of 16, twice


def test_exact_test_is_symmetric_and_exact_for_large_counts():
    assert mcnemar_exact(163, 221) == mcnemar_exact(221, 163)
    assert float(mcnemar_exact(163, 221)) == pytest.approx(0.00357, abs=0.00001)
    tiny = mcnemar_exact(2, 2096)
    assert isinstance(tiny, Fraction) and 0 < tiny < Fraction(1, 10**600)


def test_exact_test_refuses_a_negative_count():
    with pytest.raises(ValueError, match="negative"):
        mcnemar_exact(-1, 3)


# --------------------------------------------------------------------------- Holm


def test_holm_by_hand():
    values = [Fraction(1, 100), Fraction(4, 100), Fraction(3, 100)]
    assert holm(values) == [Fraction(3, 100), Fraction(6, 100), Fraction(6, 100)]


def test_holm_never_lowers_a_value_and_never_passes_one():
    values = [Fraction(1, 2), Fraction(9, 10), Fraction(1, 1000), Fraction(1, 5)]
    adjusted = holm(values)
    assert all(after >= before for before, after in zip(values, adjusted, strict=True))
    assert max(adjusted) == 1 and adjusted[2] == Fraction(4, 1000)


def test_holm_does_not_depend_on_the_order_given():
    values = [Fraction(1, 50), Fraction(1, 200), Fraction(1, 20)]
    turned = list(reversed(values))
    assert holm(turned) == list(reversed(holm(values)))


def test_holm_of_nothing_and_of_a_bad_value():
    assert holm([]) == []
    with pytest.raises(ValueError, match="from 0 to 1"):
        holm([Fraction(3, 2)])
