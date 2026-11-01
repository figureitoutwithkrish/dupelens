"""Step 1: three-way verdict for a pair: same product / variant / different.

same    = same model AND same colour and pack   (what price comparison needs)
variant = same model, but another colour, size, pack or combo
different = not the same model
The M1_v2 probability says "same model"; colour_dist, a number-token conflict or
a combo/pack mismatch then splits same vs variant. Cut-offs are chosen on the
210 Indian pairs labelled earlier (calibration only); the reported numbers come
from a FRESH labelled set built by `fresh_sheet()`.
Run: python -m src.variant --calibrate | --sheet | --stats
"""
import faiss  # noqa: F401  import first (OpenMP, see config.py)
import json
import re
import sys

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data import load_cached
from src.extra import colour_dist
from src.features import load_tfidf
from src.india import make_grids, score_site

COMBO = re.compile(r"\bcombo\b|\bkit\b|\bset of \d+|\bpack of \d+|\+|\b\d+\s*in\s*1\b", re.IGNORECASE)
PAIRS_CSV = config.CACHE_DIR / "india_pairs_v2.csv"
CONFIG_V2 = config.CACHE_DIR / "final_config_v2.json"


def combo_mismatch(title_a, title_b):
    """True when exactly one title describes a combo, kit, set/pack or N-in-1 product."""
    return bool(COMBO.search(title_a)) != bool(COMBO.search(title_b))


def classify(prob, conflict, colour_d, combo, cfg):
    """Vectorised verdict: 'same', 'variant' or 'different' for each pair."""
    same_model = prob >= cfg["india_same_model_threshold"]
    differs = conflict.astype(bool) | combo.astype(bool) | (colour_d > cfg["colour_cut"])
    return np.where(same_model & ~differs, "same", np.where(same_model, "variant", "different"))


def pair_table():
    """Every India candidate pair scored by M1_v2, with the variant clues. Cached to csv."""
    cfg = json.loads(CONFIG_V2.read_text())
    model = joblib.load(config.CACHE_DIR / "models_v2.joblib")["m1_v2"]
    india = pd.read_csv(config.INDIA_LISTINGS)
    emb = load_cached("india", india["listing_id"])
    dino, colour = (np.load(config.CACHE_DIR / f"india_{m}.npy") for m in ("dino", "colour"))
    pairs, X, _ = score_site(india, emb, joblib.load(config.CACHE_DIR / "pca.joblib"), load_tfidf(),
                             joblib.load(config.CACHE_DIR / "models.joblib")["m1"], cfg)
    a, b = pairs["a"].to_numpy(), pairs["b"].to_numpy()
    X = X.assign(dino_cos=np.einsum("ij,ij->i", dino[a], dino[b]), colour_dist=colour_dist(colour[a], colour[b]))
    t = india["title"].to_numpy()
    out = pd.DataFrame({"id_a": india["listing_id"].to_numpy()[a], "id_b": india["listing_id"].to_numpy()[b],
                        "prob_v2": model.predict_proba(X[cfg["features_v2"]])[:, 1], "conflict": X["num_conflict"],
                        "colour_dist": X["colour_dist"], "combo": [combo_mismatch(x, y) for x, y in zip(t[a], t[b])],
                        "title_a": t[a], "title_b": t[b],
                        "image_url_a": india["image_url"].to_numpy()[a], "image_url_b": india["image_url"].to_numpy()[b]})
    out.to_csv(PAIRS_CSV, index=False)
    return out


def labelled_pairs():
    """The 210 earlier labels (strict: 1 = same model and same colour/pack), as unordered id pairs."""
    rows = []
    for sheet, key in [("labelling_sheet.csv", "labelling_key.csv"), ("holdout_sheet.csv", "holdout_key.csv")]:
        k = pd.read_csv(config.CACHE_DIR / key)[["pair_id", "id_a", "id_b"]]
        s = pd.read_csv(config.RESULTS_DIR / sheet)[["pair_id", "same_product"]]
        rows.append(k.merge(s, on="pair_id"))
    lab = pd.concat(rows, ignore_index=True)
    lab["key"] = [tuple(sorted(p)) for p in zip(lab["id_a"], lab["id_b"])]
    return lab


def calibrate():
    """Pick the same-model threshold and colour cut that maximise F1 of 'same' on the 210 labels."""
    pairs = pair_table()
    pairs["key"] = [tuple(sorted(p)) for p in zip(pairs["id_a"], pairs["id_b"])]
    lab = labelled_pairs().merge(pairs.drop(columns=["id_a", "id_b"]), on="key", how="left")
    lab["prob_v2"] = lab["prob_v2"].fillna(0.0)  # labelled pairs outside the top-k are "different"
    lab = lab.fillna({"conflict": 0, "colour_dist": 1.0, "combo": False})
    y = lab["same_product"].astype(int).to_numpy()
    best = None
    for t in [0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999]:
        for c in np.round(np.arange(0.1, 1.01, 0.05), 2):
            pred = classify(lab["prob_v2"].to_numpy(), lab["conflict"].to_numpy(), lab["colour_dist"].to_numpy(),
                            lab["combo"].to_numpy(), {"india_same_model_threshold": t, "colour_cut": c}) == "same"
            tp = int((pred & (y == 1)).sum())
            f1 = 2 * tp / max(int(pred.sum()) + int(y.sum()), 1)
            if best is None or f1 > best[0]:
                best = (f1, t, float(c), tp, int(pred.sum()))
    f1, t, c, tp, npred = best
    cfg = json.loads(CONFIG_V2.read_text())
    cfg.update(india_same_model_threshold=t, colour_cut=c)
    CONFIG_V2.write_text(json.dumps(cfg, indent=2))
    print(f"calibration (210 labels, {int(y.sum())} positive): best F1 {f1:.3f} at threshold {t}, colour cut {c}; "
          f"predicted same {npred}, of which labelled same {tp} (precision {tp / max(npred, 1):.3f})")


def fresh_sheet(n_same=40, n_variant=20):
    """Fresh blind sheet for the v2 verdict: pairs never labelled before, scores and verdicts hidden."""
    cfg = json.loads(CONFIG_V2.read_text())
    pairs = pd.read_csv(PAIRS_CSV)
    pairs["key"] = [tuple(sorted(p)) for p in zip(pairs["id_a"], pairs["id_b"])]
    pairs = pairs[~pairs["key"].isin(set(labelled_pairs()["key"]))]
    pairs["verdict"] = classify(pairs["prob_v2"].to_numpy(), pairs["conflict"].to_numpy(),
                                pairs["colour_dist"].to_numpy(), pairs["combo"].to_numpy(), cfg)
    picks = [pairs[pairs.verdict == v].sample(min(n, int((pairs.verdict == v).sum())), random_state=config.SEED + 2)
             for v, n in [("same", n_same), ("variant", n_variant)]]
    key = pd.concat(picks).sample(frac=1, random_state=config.SEED + 2).reset_index(drop=True)
    key["pair_id"] = [f"V{i:03d}" for i in range(len(key))]
    key.to_csv(config.CACHE_DIR / "v2_key.csv", index=False)
    cols = ["pair_id", "title_a", "title_b", "image_url_a", "image_url_b", "id_a", "id_b"]
    key[cols].assign(label="").to_csv(config.RESULTS_DIR / "v2_sheet.csv", index=False)
    make_grids(sheet_name="v2_sheet.csv", prefix="v2")
    counts = pairs["verdict"].value_counts().to_dict()
    print("verdict counts over all candidate pairs:", counts, "| sheet:", key["verdict"].value_counts().to_dict())


def stats(n_boot=config.N_BOOT):
    """Precision of 'same' and accuracy of 'variant' on the fresh sheet (labels: same/variant/different)."""
    key = pd.read_csv(config.CACHE_DIR / "v2_key.csv")[["pair_id", "verdict"]]
    lab = key.merge(pd.read_csv(config.RESULTS_DIR / "v2_sheet.csv")[["pair_id", "label"]], on="pair_id")
    rng = np.random.default_rng(config.SEED)
    out = {"labeller": "Claude (AI assistant), blind, from titles and photos; no human labels"}
    for v in ["same", "variant"]:
        g = (lab[lab.verdict == v]["label"] == v).to_numpy().astype(float)
        boots = [rng.choice(g, len(g)).mean() for _ in range(n_boot)]
        out[v] = dict(n=int(len(g)), correct=float(g.mean()), ci=[float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                      labels=lab[lab.verdict == v]["label"].value_counts().to_dict())
    same_any_colour = lab[lab.verdict == "same"]["label"].isin(["same", "variant"]).mean()
    out["same_verdict_same_model_any_variant"] = float(same_any_colour)
    path = config.RESULTS_DIR / "json" / "india_summary.json"
    summary = json.loads(path.read_text())
    summary["v2"] = out
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    {"--calibrate": calibrate, "--sheet": fresh_sheet, "--stats": stats}[sys.argv[1]]()
