"""results.csv (one row per experiment); dashboard JSON and plots are added later."""
import math

import pandas as pd

from src import config

RESULTS_CSV = config.RESULTS_DIR / "results.csv"
COLUMNS = ["step", "model", "features", "pca_variant", "dim", "k", "split", "f1", "f1_ci_low", "f1_ci_high",
           "pair_precision", "pair_recall", "cand_recall", "threshold", "notes"]
KEY = ["step", "model", "features", "pca_variant", "dim", "k", "split"]


def log_result(**row):
    """Add one experiment row to results/results.csv, replacing an older row with the same key.

    Re-running a stage therefore updates its rows instead of duplicating them.
    Floats are rounded to 4 decimals; missing columns stay empty.
    """
    row = {c: row.get(c) for c in COLUMNS}
    row = {c: round(v, 4) if isinstance(v, float) and not math.isnan(v) else v for c, v in row.items()}
    new = pd.DataFrame([row], columns=COLUMNS).astype({c: str for c in KEY})
    if RESULTS_CSV.exists():
        old = pd.read_csv(RESULTS_CSV, dtype={c: str for c in KEY})
        old = old[~old[KEY].apply(tuple, axis=1).isin(new[KEY].apply(tuple, axis=1))]
        new = pd.concat([old, new], ignore_index=True)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(RESULTS_CSV, index=False)


def plots_and_json():
    """Three README plots + the JSON files the static dashboard reads (dashboard/data/)."""
    import json

    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from src.reduce import explained_variance

    plots, data = config.RESULTS_DIR / "plots", config.ROOT / "dashboard" / "data"
    plots.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(RESULTS_CSV)
    js = config.RESULTS_DIR / "json"

    # 1. Explained variance of the train PCA, per modality.
    pcas = joblib.load(config.CACHE_DIR / "pca.joblib")
    ev = {m: explained_variance(pcas[m]).tolist() for m in ("img", "txt")}
    plt.figure(figsize=(6, 4))
    for m, label in [("img", "image (CLIP, 512-d)"), ("txt", "text (MPNet, 768-d)")]:
        plt.plot(np.arange(1, len(ev[m]) + 1), ev[m], label=label)
    plt.axvline(32, color="grey", ls="--", lw=1)
    plt.xscale("log"); plt.xlabel("PCA components (log scale)"); plt.ylabel("cumulative explained variance")
    plt.title("PCA fitted on Shopee train"); plt.legend(); plt.tight_layout(); plt.savefig(plots / "explained_variance.png", dpi=150)

    # 2. Validation candidate recall vs dims (image+text, k=50), 3 PCA variants + raw.
    sw = res[(res.step == "sweep") & (res.features == "image+text") & (res.k.astype(str) == "50")]
    plt.figure(figsize=(6, 4))
    for v in ["plain", "whiten", "shrink"]:
        d = sw[sw.pca_variant == v].assign(dim=lambda x: x.dim.astype(int)).sort_values("dim")
        plt.plot(d.dim, d.cand_recall, marker="o", label=v)
    raw = float(sw[sw.pca_variant == "none"].cand_recall.iloc[0])
    plt.axhline(raw, color="black", ls="--", lw=1, label="raw full-dim")
    plt.xscale("log", base=2); plt.xlabel("dims per modality"); plt.ylabel("candidate recall (val, k=50)")
    plt.legend(); plt.tight_layout(); plt.savefig(plots / "recall_vs_dims.png", dpi=150)

    # 3. Test precision-recall curve of M1 (pairwise).
    pr = json.loads((js / "pr_curve.json").read_text())
    plt.figure(figsize=(5, 4))
    plt.plot([p["recall"] for p in pr], [p["precision"] for p in pr], marker=".")
    plt.xlabel("pairwise recall"); plt.ylabel("pairwise precision"); plt.title("M1 on Shopee test")
    plt.tight_layout(); plt.savefig(plots / "pr_curve_test.png", dpi=150)

    # Dashboard JSON.
    test = res[res.split == "test"].astype(object).where(res[res.split == "test"].notna(), None)  # NaN -> null
    order = ["B0", "B1", "B2", "ablation", "M1", "M2", "M1_components"]
    ladder = [r for s in order for r in test[test.step == s].to_dict("records")]
    (data / "ladder.json").write_text(json.dumps(ladder, indent=2, default=str))
    (data / "pca.json").write_text(json.dumps(dict(
        sweep=sw[["pca_variant", "dim", "cand_recall"]].to_dict("records"),
        explained_variance={m: [round(x, 4) for x in v[:256]] for m, v in ev.items()}), indent=2))
    (data / "pr_curve.json").write_text(json.dumps(pr))
    (data / "errors.json").write_text((js / "error_sample.json").read_text())
    (data / "india_matches.json").write_text((js / "india_matches.json").read_text())
    (data / "india_summary.json").write_text((js / "india_summary.json").read_text())
    # v2 model (DINOv2 + colour), compared on Shopee VALIDATION (test is used up).
    val = res[(res.split == "val") & res.step.isin(["M1", "M1_v2", "M1_v2_no_dino", "M1_v2_no_colour"])]
    val = val[(val.step != "M1") | (val.dim.astype(str) == "32")]
    v2 = {r.step: dict(f1=r.f1, f1_ci_low=r.f1_ci_low, f1_ci_high=r.f1_ci_high, notes=None if pd.isna(r.notes) else r.notes)
          for r in val.itertuples()}
    (data / "model_v2.json").write_text(json.dumps(v2, indent=2))
    print("wrote plots to", plots, "and dashboard data to", data)


if __name__ == "__main__":
    plots_and_json()
