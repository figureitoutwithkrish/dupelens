"""Unit tests for src/metrics.py on small hand-checkable examples."""
import math

import numpy as np
import pytest

from src.metrics import group_bootstrap_ci, mean_f1, pairwise_precision_recall, per_listing_f1


def truth_from_groups(groups):
    """Build a truth dict from {group: [ids]}; each id maps to its whole group."""
    return {pid: set(ids) for ids in groups.values() for pid in ids}


@pytest.mark.parametrize("n", [1, 2, 3, 5, 50])
def test_self_only_scores_2_over_n_plus_1(n):
    # Self-only predicts {a}; |P ∩ T| = 1, |P| = 1, |T| = n -> F1 = 2/(n+1).
    truth = truth_from_groups({"g": [f"p{i}" for i in range(n)]})
    pred = {pid: {pid} for pid in truth}
    assert np.allclose(per_listing_f1(pred, truth), 2 / (n + 1))


def test_self_only_mean_over_mixed_groups():
    # Groups of size 1, 2, 3: listings score 1, 2/3, 2/3, 1/2, 1/2, 1/2.
    truth = truth_from_groups({"a": ["a1"], "b": ["b1", "b2"], "c": ["c1", "c2", "c3"]})
    pred = {pid: set() for pid in truth}  # empty prediction; self is added by the metric
    assert mean_f1(pred, truth) == pytest.approx((1 + 2 / 3 + 2 / 3 + 0.5 * 3) / 6)


def test_perfect_prediction_scores_1():
    truth = truth_from_groups({"a": ["a1", "a2"], "b": ["b1", "b2", "b3"]})
    assert mean_f1(truth, truth) == 1.0


def test_f1_hand_example():
    # True group {x, y, z}; predict {x, y, w} for x -> overlap {x, y} = 2, F1 = 4/6.
    truth = {"x": {"x", "y", "z"}}
    pred = {"x": {"y", "w"}}
    assert per_listing_f1(pred, truth)[0] == pytest.approx(4 / 6)


def test_pairwise_excludes_self_and_counts_each_pair_once():
    truth = truth_from_groups({"a": ["a1", "a2", "a3"], "b": ["b1", "b2"]})
    # True pairs: (a1,a2) (a1,a3) (a2,a3) (b1,b2) = 4.
    # Predicted: (a1,a2) given from both sides, plus wrong (a3,b1) = 2 unique pairs.
    pred = {"a1": {"a1", "a2"}, "a2": {"a1"}, "a3": {"b1"}}
    precision, recall = pairwise_precision_recall(pred, truth)
    assert precision == pytest.approx(1 / 2)
    assert recall == pytest.approx(1 / 4)


def test_pairwise_self_only_has_undefined_precision_and_zero_recall():
    truth = truth_from_groups({"a": ["a1", "a2"]})
    precision, recall = pairwise_precision_recall({p: {p} for p in truth}, truth)
    assert math.isnan(precision)
    assert recall == 0.0


def test_bootstrap_constant_values_give_point_ci():
    low, high = group_bootstrap_ci([0.7] * 10, list("aabbccddee"), n_boot=200)
    assert low == pytest.approx(0.7) and high == pytest.approx(0.7)


def test_bootstrap_is_deterministic_and_contains_mean():
    rng = np.random.default_rng(0)
    values, groups = rng.random(300), rng.integers(0, 60, 300)
    ci1 = group_bootstrap_ci(values, groups)
    ci2 = group_bootstrap_ci(values, groups)
    assert ci1 == ci2
    assert ci1[0] <= values.mean() <= ci1[1]


def test_bootstrap_resamples_groups_not_listings():
    # One group of 1 listing scoring 0, one group of 99 listings scoring 1.
    # Listing-level mean is 0.99. Resampling GROUPS draws only the small group
    # in about 1 of 4 resamples, so the lower bound must reach 0; resampling
    # listings would almost never do that.
    values = [0.0] + [1.0] * 99
    groups = ["small"] + ["big"] * 99
    low, high = group_bootstrap_ci(values, groups)
    assert low == 0.0
    assert high == 1.0


def test_fast_pair_f1_and_pr_match_set_based_metrics():
    from src.metrics import f1_from_pairs, pair_pr_from_pairs, predictions_from_pairs
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 15, 60)
    ids = [f"p{i}" for i in range(60)]
    truth = {ids[i]: {ids[j] for j in range(60) if labels[j] == labels[i]} for i in range(60)}
    a, b = np.triu_indices(60, k=1)
    keep = rng.random(len(a)) < 0.2  # a random candidate set of unique pairs
    a, b = a[keep], b[keep]
    y = labels[a] == labels[b]
    accept = rng.random(len(a)) < 0.5
    pred = predictions_from_pairs(ids, a[accept], b[accept])
    assert np.allclose(f1_from_pairs(labels, a, b, y, accept), per_listing_f1(pred, truth))
    assert np.allclose(pair_pr_from_pairs(labels, y, accept), pairwise_precision_recall(pred, truth))
