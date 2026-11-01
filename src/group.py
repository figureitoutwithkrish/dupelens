"""Grouping rule used on Shopee (to validate it) and on India: connected components
over accepted edges, after dropping any edge whose titles have a number-token conflict.

Components make groups transitive (if a~b and b~c, then a, b, c are one product),
which a price table needs. The risk is chaining: one wrong edge merges two
products, so the edge threshold is tuned separately on validation for this rule,
and on India any group larger than 15 is flagged for manual review.
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

MAX_GROUP = 15  # India groups above this size are flagged as possible chaining


def components(n, a, b, keep):
    """Component id for each of n listings, using only the kept edges (a[keep], b[keep])."""
    graph = coo_matrix((np.ones(int(keep.sum())), (a[keep], b[keep])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def group_edges(prob, threshold, conflict):
    """Edges used for grouping: probability >= threshold and no number-token conflict."""
    return (prob >= threshold) & (conflict == 0)


def component_scores(labels, comp):
    """Per-listing F1, pairwise precision and recall when each predicted set is a whole component."""
    df = pd.DataFrame({"c": comp, "l": labels})
    inter = df.groupby(["c", "l"])["c"].transform("size").to_numpy()
    c_size = df.groupby("c")["c"].transform("size").to_numpy()
    g_size = df.groupby("l")["l"].transform("size").to_numpy()
    f1 = 2 * inter / (c_size + g_size)
    pred_pairs = (np.bincount(comp) * (np.bincount(comp) - 1) // 2).sum()
    cell = df.groupby(["c", "l"]).size().to_numpy()
    hits = (cell * (cell - 1) // 2).sum()
    _, g_counts = np.unique(labels, return_counts=True)
    n_true = (g_counts * (g_counts - 1) // 2).sum()
    precision = hits / pred_pairs if pred_pairs else float("nan")
    return f1, precision, hits / n_true
