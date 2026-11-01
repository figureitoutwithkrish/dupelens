"""Evaluation metrics: per-listing F1, pairwise precision/recall, group bootstrap CI.

Predictions and truth are dicts: posting_id -> set of posting_ids.
"""
import numpy as np

from src import config


def per_listing_f1(pred, truth):
    """F1 for each listing, Kaggle convention: both sets include the listing itself.

    F1 = 2|P ∩ T| / (|P| + |T|). Self is added to P even if the model left it out,
    so a model that predicts nothing still gets credit for the self-match.
    Returns an array aligned with truth's key order.
    """
    scores = []
    for pid, true_set in truth.items():
        pred_set = set(pred.get(pid, ())) | {pid}
        scores.append(2 * len(pred_set & true_set) / (len(pred_set) + len(true_set)))
    return np.array(scores)


def mean_f1(pred, truth):
    """Mean per-listing F1, the headline Shopee score."""
    return float(per_listing_f1(pred, truth).mean())


def predictions_from_pairs(ids, a, b):
    """Predicted sets from accepted pairs (positions a, b into ids); symmetric by construction.

    If (a, b) is accepted, b goes into a's set and a into b's. Self is added by per_listing_f1.
    """
    pred = {pid: set() for pid in ids}
    for i, j in zip(a, b):
        pred[ids[i]].add(ids[j])
        pred[ids[j]].add(ids[i])
    return pred


def f1_from_pairs(labels, a, b, y, accept):
    """Per-listing F1 computed directly from unique candidate pairs (fast, for threshold search).

    For listing i: |P| = 1 + accepted pairs touching i, |P ∩ T| = 1 + accepted TRUE pairs
    touching i, |T| = its group size. Same result as per_listing_f1, without building sets;
    tests/test_metrics.py checks the two agree.
    """
    n = len(labels)
    acc_a, acc_b, hit = a[accept], b[accept], y[accept]
    n_acc = np.bincount(acc_a, minlength=n) + np.bincount(acc_b, minlength=n)
    n_hit = np.bincount(acc_a[hit], minlength=n) + np.bincount(acc_b[hit], minlength=n)
    _, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return 2 * (1 + n_hit) / (1 + n_acc + counts[inv])


def pair_pr_from_pairs(labels, y, accept):
    """Pairwise precision/recall from candidate pairs; recall counts ALL true pairs in the split,
    including those that never reached the candidate set."""
    _, counts = np.unique(labels, return_counts=True)
    n_true = (counts * (counts - 1) // 2).sum()
    hits = int((y & accept).sum())
    precision = hits / accept.sum() if accept.any() else float("nan")
    return precision, hits / n_true


def _pairs(sets):
    """Unordered pairs (a, b) with a < b, self excluded. (a, b) and (b, a) count once."""
    return {(min(a, b), max(a, b)) for a, others in sets.items() for b in others if b != a}


def pairwise_precision_recall(pred, truth):
    """Precision and recall over matching pairs, self-matches excluded.

    Unlike per-listing F1, this gives no free credit for self, so it shows how
    good the actual matches are. Precision is nan when no pairs are predicted
    (e.g. the self-only baseline), because it is undefined, not zero.
    """
    pred_pairs, true_pairs = _pairs(pred), _pairs(truth)
    hits = len(pred_pairs & true_pairs)
    precision = hits / len(pred_pairs) if pred_pairs else float("nan")
    recall = hits / len(true_pairs) if true_pairs else float("nan")
    return precision, recall


def group_bootstrap_ci(values, groups, n_boot=config.N_BOOT, seed=config.SEED, alpha=0.05):
    """95% CI of the mean of `values` by resampling product groups with replacement.

    Listings in one group are not independent (they share one product), so we
    resample whole groups, keep predictions fixed, and re-average all listings
    in the drawn groups. Resampling listings would make the CI too narrow.
    """
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    _, group_idx = np.unique(groups, return_inverse=True)
    sums = np.bincount(group_idx, weights=values)   # per-group total
    counts = np.bincount(group_idx)                  # per-group listing count
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)
