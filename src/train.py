"""Baselines (B0-B2), logistic regression (M1), MLP (M2) and ablation, all on the
same split and the same candidate pairs. Thresholds are tuned on VALIDATION.
Test is not touched here.
Run: python -m src.train
"""
import faiss  # noqa: F401  import first (OpenMP clash with torch, see config.py)
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config, features, group
from src.candidates import candidate_pairs, candidate_recall, retrieval_vectors, top_k
from src.data import build_ground_truth, load_cached, load_split
from src.export import log_result
from src.metrics import (f1_from_pairs, group_bootstrap_ci, pair_pr_from_pairs, pairwise_precision_recall,
                         per_listing_f1)
from src.reduce import project

RAW = ("none", None)
SETTINGS = [RAW, ("shrink", 32), ("shrink", 64), ("shrink", 256)]  # shortlist from the recall sweep
THRESHOLDS = np.round(np.arange(0.02, 0.99, 0.02), 2)
FINAL_CONFIG = config.CACHE_DIR / "final_config.json"


def prepare(df, emb, pcas, vec, split, retrieval, cosine):
    """Candidate pairs and their feature table for one split.

    retrieval: (variant, dim) used to build the FAISS vectors (which pairs are candidates).
    cosine: (variant, dim) used for the image/text cosine features. Passing a different
    cosine setting on the same candidates gives the raw-vs-PCA feature comparison.
    """
    rows = df["split"].eq(split).to_numpy()
    sub = df[rows].reset_index(drop=True)
    proj = lambda m, s: project(emb[m][rows], pcas[m], s[1], s[0])
    img_r, txt_r = proj("img", retrieval), proj("txt", retrieval)
    pairs = candidate_pairs(top_k(retrieval_vectors(img_r, txt_r), config.K)[0], config.K)
    img_c, txt_c = (img_r, txt_r) if cosine == retrieval else (proj("img", cosine), proj("txt", cosine))
    X = features.build(pairs, img_c, txt_c, vec.transform(sub["title"]), emb["phash"][rows],
                       sub["title"].tolist(), config.K)
    labels = sub["label_group"].to_numpy()
    a, b = pairs["a"].to_numpy(), pairs["b"].to_numpy()
    return dict(sub=sub, labels=labels, pairs=pairs, a=a, b=b, X=X, y=labels[a] == labels[b])


def score(d, accept):
    """Mean per-listing F1 with group bootstrap CI, plus pairwise precision/recall."""
    f1 = f1_from_pairs(d["labels"], d["a"], d["b"], d["y"], accept)
    low, high = group_bootstrap_ci(f1, d["labels"])
    precision, recall = pair_pr_from_pairs(d["labels"], d["y"], accept)
    return dict(f1=float(f1.mean()), f1_ci_low=low, f1_ci_high=high, pair_precision=precision, pair_recall=recall)


def tune(d, s, thresholds):
    """Cutoff on scores s (higher = more likely a match) that maximises mean validation F1."""
    f1s = [f1_from_pairs(d["labels"], d["a"], d["b"], d["y"], s >= t).mean() for t in thresholds]
    return float(thresholds[int(np.argmax(f1s))])


def fit_m1(X, y):
    """Logistic regression on standardised features, classes reweighted.

    Standardising (scaler fitted on train pairs only) puts features on one scale,
    so the penalty treats them equally and coefficients are comparable in size.
    class_weight='balanced' weights each class by 1/frequency, because most
    candidate pairs are non-matches and an unweighted loss would favour 'no match'.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)).fit(X, y)


def fit_m2(X, y):
    """Small MLP (two hidden layers) on standardised features.

    MLPClassifier has no class_weight, so we balance by randomly keeping as many
    train negatives as there are positives (undersampling, seed 42).
    """
    rng = np.random.default_rng(config.SEED)
    pos, neg = np.where(y)[0], np.where(~y)[0]
    keep = np.concatenate([pos, rng.choice(neg, size=len(pos), replace=False)])
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True, random_state=config.SEED, max_iter=200)
    return make_pipeline(StandardScaler(), mlp).fit(X.iloc[keep], y[keep])


def log(step, model, feats, setting, split, metrics, **extra):
    log_result(step=step, model=model, features=feats, pca_variant=setting[0], dim=setting[1] or "full",
               k=config.K, split=split, **metrics, **extra)


def baselines_b0_b1(df, split):
    """B0 self-only and B1 exact match on Shopee's image_phash column (whole split, no candidates)."""
    sub = df[df["split"] == split]
    truth = build_ground_truth(sub)
    groups = sub.set_index("posting_id").loc[list(truth), "label_group"]
    by_hash = sub.groupby("image_phash")["posting_id"].agg(set)
    for step, pred in [("B0", {p: {p} for p in truth}),
                       ("B1", dict(zip(sub["posting_id"], sub["image_phash"].map(by_hash))))]:
        f1 = per_listing_f1(pred, truth)
        low, high = group_bootstrap_ci(f1, groups)
        precision, recall = pairwise_precision_recall(pred, truth)
        log_result(step=step, model="self_only" if step == "B0" else "exact_image_phash", features="none",
                   pca_variant="none", dim="none", k="none", split=split, f1=float(f1.mean()), f1_ci_low=low,
                   f1_ci_high=high, pair_precision=precision, pair_recall=recall)
        print(f"{step} {split}: F1 {f1.mean():.4f} [{low:.4f}, {high:.4f}] P {precision:.4f} R {recall:.4f}")


def run_validation():
    df = load_split()
    emb = load_cached("shopee", df["posting_id"])
    pcas = joblib.load(config.CACHE_DIR / "pca.joblib")
    vec = features.fit_tfidf(df.loc[df["split"] == "train", "title"])
    baselines_b0_b1(df, "val")

    # M1 for each shortlisted retrieval setting; cosine features from the same projection.
    m1 = {}
    for s in SETTINGS:
        tr, va = (prepare(df, emb, pcas, vec, split, s, s) for split in ("train", "val"))
        model = fit_m1(tr["X"], tr["y"])
        prob = model.predict_proba(va["X"])[:, 1]
        thr = tune(va, prob, THRESHOLDS)
        rec = candidate_recall(va["pairs"], va["labels"])[0]
        m = score(va, prob >= thr)
        log("M1", "logreg", "all", s, "val", m, cand_recall=rec, threshold=thr,
            notes=f"train pairs {len(tr['y'])}, positive share {tr['y'].mean():.4f}; val pairs {len(va['y'])}")
        print(f"M1 {s}: val F1 {m['f1']:.4f} [{m['f1_ci_low']:.4f}, {m['f1_ci_high']:.4f}] thr {thr} cand_recall {rec:.4f}")
        m1[s] = dict(tr=tr, va=va, model=model, prob=prob, thr=thr, metrics=m)
        if s != RAW:  # same candidates, raw cosine features instead of PCA ones
            tr_r, va_r = (prepare(df, emb, pcas, vec, split, s, RAW) for split in ("train", "val"))
            model_r = fit_m1(tr_r["X"], tr_r["y"])
            prob_r = model_r.predict_proba(va_r["X"])[:, 1]
            thr_r = tune(va_r, prob_r, THRESHOLDS)
            m_r = score(va_r, prob_r >= thr_r)
            log("M1_rawcos", "logreg", "all, raw cosines", s, "val", m_r, cand_recall=rec, threshold=thr_r)
            print(f"   same candidates, raw cosines: val F1 {m_r['f1']:.4f} thr {thr_r}")

    # Choose the smallest dim whose val F1 is inside the full-dim CI (blueprint 7.3).
    full = m1[RAW]["metrics"]
    ok = [s for s in SETTINGS[1:] if m1[s]["metrics"]["f1"] >= full["f1_ci_low"]]
    chosen = min(ok, key=lambda s: s[1]) if ok else RAW
    print("chosen setting:", chosen)
    c = m1[chosen]
    tr, va, X_cols = c["tr"], c["va"], features.FEATURES

    # B2: best single feature with a tuned cutoff (phash distance: lower = closer, so negate).
    b2 = {}
    for f in ["img_cos", "txt_cos", "tfidf_cos", "phash_dist"]:
        s = -va["X"][f].to_numpy() if f == "phash_dist" else va["X"][f].to_numpy()
        cut = tune(va, s, np.unique(np.quantile(s, np.linspace(0, 1, 201))))
        b2[f] = (score(va, s >= cut), -cut if f == "phash_dist" else cut)
        print(f"B2 {f}: val F1 {b2[f][0]['f1']:.4f} cutoff {b2[f][1]:.4f}")
    best_f = max(b2, key=lambda f: b2[f][0]["f1"])
    for f, (m, cut) in b2.items():
        log("B2", "single_feature", f, chosen, "val", m, threshold=cut, notes="best single feature" if f == best_f else "")

    # M2 and the ablation (M1 without number-token features), same candidates.
    m2 = fit_m2(tr["X"], tr["y"])
    p2 = m2.predict_proba(va["X"])[:, 1]
    t2 = tune(va, p2, THRESHOLDS)
    log("M2", "mlp_32_16", "all", chosen, "val", score(va, p2 >= t2), threshold=t2)
    no_num = [f for f in X_cols if f not in features.NUM_FEATURES]
    abl = fit_m1(tr["X"][no_num], tr["y"])
    pa = abl.predict_proba(va["X"][no_num])[:, 1]
    ta = tune(va, pa, THRESHOLDS)
    log("ablation", "logreg", "all minus number tokens", chosen, "val", score(va, pa >= ta), threshold=ta)

    # Component rule (India procedure), its edge threshold fixed on validation.
    conflict = va["X"]["num_conflict"].to_numpy()
    comp_f1 = {}
    for t in THRESHOLDS:
        comp = group.components(len(va["labels"]), va["a"], va["b"], group.group_edges(c["prob"], t, conflict))
        comp_f1[t] = group.component_scores(va["labels"], comp)
    t_comp = float(max(comp_f1, key=lambda t: comp_f1[t][0].mean()))
    f1c, pc, rc = comp_f1[t_comp]
    low, high = group_bootstrap_ci(f1c, va["labels"])
    log("M1_components", "logreg + components", "all", chosen, "val",
        dict(f1=float(f1c.mean()), f1_ci_low=low, f1_ci_high=high, pair_precision=pc, pair_recall=rc),
        threshold=t_comp, notes="connected components, number-conflict edges dropped")

    coefs = dict(zip(X_cols, c["model"][-1].coef_[0].round(4).tolist()))
    (config.RESULTS_DIR / "json").mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "json" / "m1_coefficients.json").write_text(json.dumps(coefs, indent=2))
    joblib.dump({"m1": c["model"], "m2": m2, "ablation": abl}, config.CACHE_DIR / "models.joblib")
    FINAL_CONFIG.write_text(json.dumps({"pca_variant": chosen[0], "dim": chosen[1], "k": config.K,
                                        "m1_threshold": c["thr"], "m2_threshold": t2, "ablation_threshold": ta,
                                        "component_threshold": t_comp, "b2_feature": best_f,
                                        "b2_cutoff": b2[best_f][1]}, indent=2))
    print("M1 coefficients (standardised):", coefs)
    print("final config:", FINAL_CONFIG.read_text())


TEST_DONE = config.CACHE_DIR / "test_done.json"


def run_test(n_errors=15):
    """The single test run with the frozen validation config (blueprint 7.7).

    Refuses to run twice: test_done.json is written at the end, so the test split
    can never be used to tune anything. Writes test rows to results.csv, the test
    PR curve, and a random sample of errors for manual tagging.
    """
    assert not TEST_DONE.exists(), "test split already used once; see cache/test_done.json"
    cfg = json.loads(FINAL_CONFIG.read_text())
    s = (cfg["pca_variant"], cfg["dim"])
    df = load_split()
    emb = load_cached("shopee", df["posting_id"])
    pcas = joblib.load(config.CACHE_DIR / "pca.joblib")
    models = joblib.load(config.CACHE_DIR / "models.joblib")
    te = prepare(df, emb, pcas, features.load_tfidf(), "test", s, s)
    X = te["X"]
    baselines_b0_b1(df, "test")

    f, cut = cfg["b2_feature"], cfg["b2_cutoff"]
    accept_b2 = X[f].to_numpy() <= cut if f == "phash_dist" else X[f].to_numpy() >= cut
    log("B2", "single_feature", f, s, "test", score(te, accept_b2), threshold=cut, notes="best single feature")
    rec, rlow, rhigh = candidate_recall(te["pairs"], te["labels"])
    prob = models["m1"].predict_proba(X)[:, 1]
    log("M1", "logreg", "all", s, "test", score(te, prob >= cfg["m1_threshold"]), cand_recall=rec,
        threshold=cfg["m1_threshold"], notes=f"cand_recall 95% CI [{rlow:.4f}, {rhigh:.4f}]")
    p2 = models["m2"].predict_proba(X)[:, 1]
    log("M2", "mlp_32_16", "all", s, "test", score(te, p2 >= cfg["m2_threshold"]), threshold=cfg["m2_threshold"])
    no_num = [c for c in features.FEATURES if c not in features.NUM_FEATURES]
    pa = models["ablation"].predict_proba(X[no_num])[:, 1]
    log("ablation", "logreg", "all minus number tokens", s, "test", score(te, pa >= cfg["ablation_threshold"]),
        threshold=cfg["ablation_threshold"])
    keep = group.group_edges(prob, cfg["component_threshold"], X["num_conflict"].to_numpy())
    f1c, pc, rc = group.component_scores(te["labels"], group.components(len(te["labels"]), te["a"], te["b"], keep))
    low, high = group_bootstrap_ci(f1c, te["labels"])
    log("M1_components", "logreg + components", "all", s, "test",
        dict(f1=float(f1c.mean()), f1_ci_low=low, f1_ci_high=high, pair_precision=pc, pair_recall=rc),
        threshold=cfg["component_threshold"], notes="connected components, number-conflict edges dropped")

    # Pairwise precision-recall curve of M1 on test (recall over ALL true pairs in the split).
    curve = [dict(threshold=float(t), precision=p, recall=r)
             for t in THRESHOLDS for p, r in [pair_pr_from_pairs(te["labels"], te["y"], prob >= t)]]
    (config.RESULTS_DIR / "json" / "pr_curve.json").write_text(json.dumps(curve, indent=2))

    # Random false matches and misses (among candidates) for manual error tagging.
    rng = np.random.default_rng(config.SEED)
    accept = prob >= cfg["m1_threshold"]
    sub = te["sub"]
    errors = []
    for kind, mask in [("false_match", accept & ~te["y"]), ("miss", ~accept & te["y"])]:
        for i in rng.choice(np.where(mask)[0], size=min(n_errors, int(mask.sum())), replace=False):
            a, b = te["a"][i], te["b"][i]
            errors.append(dict(kind=kind, prob=round(float(prob[i]), 4), title_a=sub.at[a, "title"],
                               title_b=sub.at[b, "title"], image_a=sub.at[a, "image"], image_b=sub.at[b, "image"],
                               **{k: round(float(X.at[i, k]), 4) for k in features.FEATURES}, tag=None))
    (config.RESULTS_DIR / "json" / "error_sample.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False))
    TEST_DONE.write_text(json.dumps({"config": cfg}, indent=2))
    print("test run done; rows logged to results/results.csv")


if __name__ == "__main__":
    import sys
    run_test() if "--test" in sys.argv else run_validation()
