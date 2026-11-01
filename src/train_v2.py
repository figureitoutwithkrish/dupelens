"""M1_v2: the M1 logistic regression plus two image features (step 2).

New pair features: dino_cos (DINOv2 cosine, exact-item similarity) and
colour_dist (1 - hue/saturation histogram intersection). Same split, same
candidate pairs (shrink-whitened PCA, 32 dims, k=50) and same threshold rule
(tuned on VALIDATION) as M1, so the comparison is like-for-like.
The Shopee test split is already used once, so v2 is compared on validation;
its real check is a fresh Indian held-out set.
Run: python -m src.train_v2
"""
import faiss  # noqa: F401  import first (OpenMP clash with torch, see config.py)
import json

import joblib
import numpy as np

from src import config, features
from src.data import load_cached, load_split
from src.export import log_result
from src.extra import colour_dist
from src.metrics import f1_from_pairs, group_bootstrap_ci
from src.train import THRESHOLDS, fit_m1, prepare, score, tune

FEATURES_V2 = features.FEATURES + ["dino_cos", "colour_dist"]
CONFIG_V2 = config.CACHE_DIR / "final_config_v2.json"


def add_image_features(d, dino, colour):
    """Append dino_cos and colour_dist to a split's pair table (rows = that split's listings)."""
    a, b = d["a"], d["b"]
    d["X"] = d["X"].assign(dino_cos=np.einsum("ij,ij->i", dino[a], dino[b]), colour_dist=colour_dist(colour[a], colour[b]))
    return d


def main():
    cfg = json.loads((config.CACHE_DIR / "final_config.json").read_text())
    s = (cfg["pca_variant"], cfg["dim"])
    df = load_split()
    emb = load_cached("shopee", df["posting_id"])
    dino_all = np.load(config.CACHE_DIR / "shopee_dino.npy")
    colour_all = np.load(config.CACHE_DIR / "shopee_colour.npy")
    pcas = joblib.load(config.CACHE_DIR / "pca.joblib")
    vec = features.load_tfidf()
    splits = {}
    for split in ("train", "val"):
        rows = df["split"].eq(split).to_numpy()
        splits[split] = add_image_features(prepare(df, emb, pcas, vec, split, s, s), dino_all[rows], colour_all[rows])
    tr, va = splits["train"], splits["val"]

    # Old M1 on the same pairs, refitted from the same features, as the paired baseline.
    base = fit_m1(tr["X"][features.FEATURES], tr["y"])
    p_base = base.predict_proba(va["X"][features.FEATURES])[:, 1]
    f1_base = f1_from_pairs(va["labels"], va["a"], va["b"], va["y"], p_base >= cfg["m1_threshold"])

    out = {}
    for name, cols in [("M1_v2", FEATURES_V2),
                       ("M1_v2_no_dino", [c for c in FEATURES_V2 if c != "dino_cos"]),
                       ("M1_v2_no_colour", [c for c in FEATURES_V2 if c != "colour_dist"])]:
        model = fit_m1(tr["X"][cols], tr["y"])
        prob = model.predict_proba(va["X"][cols])[:, 1]
        thr = tune(va, prob, THRESHOLDS)
        m = score(va, prob >= thr)
        f1 = f1_from_pairs(va["labels"], va["a"], va["b"], va["y"], prob >= thr)
        low, high = group_bootstrap_ci(f1 - f1_base, va["labels"])  # paired: same listings, same groups
        log_result(step=name, model="logreg", features=",".join(cols[len(features.FEATURES):]) or "base",
                   pca_variant=s[0], dim=s[1], k=cfg["k"], split="val", **m, threshold=thr,
                   notes=f"vs M1 on same pairs: diff {f1.mean() - f1_base.mean():+.4f}, 95% CI [{low:+.4f}, {high:+.4f}]")
        print(f"{name}: val F1 {m['f1']:.4f} [{m['f1_ci_low']:.4f}, {m['f1_ci_high']:.4f}] thr {thr} "
              f"| vs M1 {f1.mean() - f1_base.mean():+.4f} [{low:+.4f}, {high:+.4f}]")
        out[name] = (model, thr, cols)
    print(f"M1 (refit, same pairs, thr {cfg['m1_threshold']}): val F1 {f1_base.mean():.4f}")

    model, thr, cols = out["M1_v2"]
    coefs = dict(zip(cols, model[-1].coef_[0].round(4).tolist()))
    (config.RESULTS_DIR / "json" / "m1_v2_coefficients.json").write_text(json.dumps(coefs, indent=2))
    joblib.dump({"m1_v2": model}, config.CACHE_DIR / "models_v2.joblib")
    CONFIG_V2.write_text(json.dumps({**cfg, "m1_v2_threshold": thr, "features_v2": cols}, indent=2))
    print("coefficients:", coefs)


if __name__ == "__main__":
    main()
