"""Checks on the cached embeddings (Phase 2). Prints facts only; trains nothing.

1. Shape, dtype, row alignment with the source table, unit length, no NaNs.
2. Our recomputed phash vs Shopee's image_phash column (same algorithm?).
3. Sanity on VALIDATION only: mean cosine of same-product pairs vs random pairs.
Run: python results/phase2/check_embeddings.py   (from the repo root)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config
from src.data import load_split

for name, src_ids in [("shopee", load_split()["posting_id"]),
                      ("india", pd.read_csv(config.INDIA_LISTINGS)["listing_id"])]:
    img = np.load(config.CACHE_DIR / f"{name}_img.npy")
    txt = np.load(config.CACHE_DIR / f"{name}_txt.npy")
    ph = np.load(config.CACHE_DIR / f"{name}_phash.npy")
    ids = pd.read_csv(config.CACHE_DIR / f"{name}_ids.csv")["id"]
    aligned = ids.tolist() == src_ids.tolist()
    norms = [np.abs(np.linalg.norm(x, axis=1) - 1).max() for x in (img, txt)]
    print(f"{name}: img {img.shape} {img.dtype}, txt {txt.shape} {txt.dtype}, phash {ph.shape} {ph.dtype} | "
          f"rows aligned: {aligned} | max |norm-1|: img {norms[0]:.1e}, txt {norms[1]:.1e} | "
          f"NaNs: {int(np.isnan(img).sum() + np.isnan(txt).sum())}")

df = load_split()
ph = np.load(config.CACHE_DIR / "shopee_phash.npy")
given = np.array([int(h, 16) for h in df["image_phash"]], dtype=np.uint64)
ham = np.bitwise_count(ph ^ given)
print(f"\nphash vs Shopee column over {len(ham)} rows: exact match {np.mean(ham == 0):.3f}, "
      f"<= 2 bits {np.mean(ham <= 2):.3f}, median {np.median(ham):.0f} bits, max {ham.max()} bits")

val = df[df.split == "val"].index.to_numpy()
img = np.load(config.CACHE_DIR / "shopee_img.npy")[val]
txt = np.load(config.CACHE_DIR / "shopee_txt.npy")[val]
groups = df.loc[val, "label_group"].to_numpy()
same = [(i, j) for g in np.unique(groups) for k, i in enumerate(np.where(groups == g)[0])
        for j in np.where(groups == g)[0][k + 1:]]
rng = np.random.default_rng(config.SEED)
rand = rng.integers(0, len(val), size=(len(same), 2))
rand = rand[groups[rand[:, 0]] != groups[rand[:, 1]]]
for label, x in [("image", img), ("text", txt)]:
    s = np.mean([x[i] @ x[j] for i, j in same])
    r = np.mean([x[i] @ x[j] for i, j in rand])
    print(f"val {label} cosine: same-product pairs {s:.3f} (n={len(same)}), random different-product pairs {r:.3f} (n={len(rand)})")
