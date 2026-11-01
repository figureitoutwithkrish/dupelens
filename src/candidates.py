"""FAISS top-k neighbours within one split, candidate pairs, and candidate recall.

Only candidate pairs are ever scored by the classifier, so candidate recall
(share of true matching pairs that reach the candidate set) caps its recall.
Run: python -m src.candidates   (PCA sweep + k check on validation)
"""
import faiss  # import before anything that can load torch: two OpenMP runtimes crash on this Mac
import joblib
import numpy as np
import pandas as pd

from src import config
from src.data import load_cached, load_split
from src.export import log_result
from src.metrics import group_bootstrap_ci
from src.reduce import VARIANTS, fit_pca, project


def retrieval_vectors(img, txt):
    """Concatenate the unit image and text vectors, scaled by 1/sqrt(2) to stay unit length.

    The inner product is then (image cosine + text cosine) / 2: equal weight per modality.
    """
    return (np.hstack([img, txt]) / np.sqrt(2)).astype(np.float32)


def top_k(vecs, k):
    """Each row's k nearest neighbours by inner product (exact search), self excluded.

    We ask FAISS for k+1 and drop the row itself. Identical listings can tie
    with self, so self is removed by index, not by assuming it comes first.
    """
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(np.ascontiguousarray(vecs, dtype=np.float32))
    scores, idx = index.search(np.ascontiguousarray(vecs, dtype=np.float32), k + 1)
    not_self = idx != np.arange(len(vecs))[:, None]
    order = np.argsort(~not_self, axis=1, kind="stable")[:, :k]  # keep first k non-self, in rank order
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(scores, order, axis=1)


def candidate_pairs(nbr, k):
    """Unique unordered pairs (a, b), a < b, from the neighbour lists, with both ranks.

    rank_ab = position (1-based) of b in a's list, k+1 if b is not there; same
    for rank_ba. These feed the mutual-neighbour and rank features later.
    """
    n = len(nbr)
    row, col = np.repeat(np.arange(n), k), nbr.ravel()
    rank = np.tile(np.arange(1, k + 1), n)
    a, b = np.minimum(row, col), np.maximum(row, col)
    df = pd.DataFrame({"a": a, "b": b,
                       "rank_ab": np.where(row == a, rank, k + 1), "rank_ba": np.where(row == b, rank, k + 1)})
    return df.groupby(["a", "b"], as_index=False).min()


def true_pairs(labels):
    """All unordered pairs (a, b), a < b, of positions sharing a label (self excluded)."""
    pairs = []
    for members in pd.Series(np.arange(len(labels))).groupby(labels).agg(list):
        m = np.array(members)
        i, j = np.triu_indices(len(m), k=1)
        pairs.append(np.stack([m[i], m[j]], axis=1))
    return np.vstack(pairs)


def candidate_recall(pairs, labels):
    """Share of true matching pairs present among the candidates, with a group bootstrap CI.

    Each true pair belongs to one product group, so we resample groups (as for F1).
    """
    n = len(labels)
    truth = true_pairs(labels)
    found = np.isin(truth[:, 0] * n + truth[:, 1], pairs["a"].to_numpy() * n + pairs["b"].to_numpy())
    low, high = group_bootstrap_ci(found, labels[truth[:, 0]])
    return float(found.mean()), low, high


def fit_and_save_pcas(emb, train_mask):
    """Fit one PCA per modality on train rows only, and cache them (frozen for India)."""
    pcas = {m: fit_pca(emb[m][train_mask].astype(np.float64)) for m in ("img", "txt")}
    joblib.dump(pcas, config.CACHE_DIR / "pca.joblib")
    return pcas


def sweep(emb, pcas, labels, rows, k):
    """Candidate recall on one split for every PCA variant x dim, per modality and combined."""
    out = []
    for dim in config.DIMS:
        for variant in (VARIANTS if dim else ["none"]):
            img = project(emb["img"][rows], pcas["img"], dim, variant)
            txt = project(emb["txt"][rows], pcas["txt"], dim, variant)
            for name, vecs in [("image", img), ("text", txt), ("image+text", retrieval_vectors(img, txt))]:
                nbr, _ = top_k(vecs, k)
                recall, low, high = candidate_recall(candidate_pairs(nbr, k), labels)
                out.append(dict(features=name, pca_variant=variant, dim=dim or "full", k=k,
                                cand_recall=recall, low=low, high=high))
    return pd.DataFrame(out)


if __name__ == "__main__":
    df = load_split()
    emb = load_cached("shopee", df["posting_id"])
    pcas = fit_and_save_pcas(emb, df["split"].eq("train").to_numpy())
    val = df["split"].eq("val").to_numpy()
    labels = df.loc[val, "label_group"].to_numpy()

    res = sweep(emb, pcas, labels, val, config.K)
    for r in res.itertuples():
        log_result(step="sweep", model="faiss_flat_ip", features=r.features, pca_variant=r.pca_variant,
                   dim=r.dim, k=r.k, split="val", cand_recall=r.cand_recall,
                   notes=f"cand_recall 95% CI [{r.low:.4f}, {r.high:.4f}]")
    pd.set_option("display.width", 200)
    print(res.pivot_table(index=["pca_variant", "dim"], columns="features", values="cand_recall", sort=False).round(4))

    combined = res[res.features == "image+text"]
    best = combined.loc[combined.cand_recall.idxmax()]
    full = combined[combined.dim == "full"].iloc[0]
    print("\nbest combined:", best.pca_variant, best.dim, f"{best.cand_recall:.4f} [{best.low:.4f}, {best.high:.4f}]")
    print("full combined:", f"{full.cand_recall:.4f} [{full.low:.4f}, {full.high:.4f}]")
    within = combined[(combined.cand_recall >= full.low) & (combined.dim != "full")]
    print("settings inside the full-dim CI:", within[["pca_variant", "dim", "cand_recall"]].to_dict("records"))

    # k check, done once on validation (k=20 vs 50); skipped once K is already 50.
    for cfg in ([best, full] if config.K < config.K_CHECK else []):
        dim = None if cfg.dim == "full" else int(cfg.dim)
        img = project(emb["img"][val], pcas["img"], dim, cfg.pca_variant)
        txt = project(emb["txt"][val], pcas["txt"], dim, cfg.pca_variant)
        nbr, _ = top_k(retrieval_vectors(img, txt), config.K_CHECK)
        recall, low, high = candidate_recall(candidate_pairs(nbr, config.K_CHECK), labels)
        log_result(step="k_check", model="faiss_flat_ip", features="image+text", pca_variant=cfg.pca_variant,
                   dim=cfg.dim, k=config.K_CHECK, split="val", cand_recall=recall,
                   notes=f"cand_recall 95% CI [{low:.4f}, {high:.4f}]; k={config.K} recall {cfg.cand_recall:.4f}")
        print(f"k check {cfg.pca_variant} {cfg.dim}: k={config.K} {cfg.cand_recall:.4f} -> k={config.K_CHECK} "
              f"{recall:.4f} [{low:.4f}, {high:.4f}], gain {100 * (recall - cfg.cand_recall):.2f} points")
