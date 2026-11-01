"""Second category probe: which Amazon categories contain TRUE duplicate listings?

The first check (TF-IDF pairs) could not tell templated titles from real duplicates;
jewellery passed it but had 1/150 true matches. Here a pair counts as a likely true
duplicate only if both the photo (phash Hamming <= 6) and the title (TF-IDF cosine
>= 0.5) are near-identical and the two rows have different product IDs (ASINs).
Usage: python duplicate_probe.py <india_raw dir> <image cache dir>
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer

root, img_dir = sys.argv[1], Path(sys.argv[2])
img_dir.mkdir(parents=True, exist_ok=True)
CATS = ["Make-up", "Beauty & Grooming", "Health & Personal Care", "Luxury Beauty", "Personal Care Appliances",
        "Coffee, Tea & Beverages", "Snack Foods", "All Grocery & Gourmet Foods", "Diet & Nutrition",
        "Household Supplies", "Baby Bath, Skin & Grooming", "Kitchen & Dining", "Home Storage"]


def phash(row):
    path = img_dir / f"{row.asin}.jpg"
    try:
        if not path.exists():
            r = requests.get(row.image, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                return None
            path.write_bytes(r.content)
        with Image.open(path) as im:
            return int(str(imagehash.phash(im.convert("RGB"))), 16)
    except Exception:
        return None


a = pd.read_csv(f"{root}/amazon/Amazon-Products.csv")
a["asin"] = a.link.str.extract(r"/dp/([A-Z0-9]{10})")[0]
a["price"] = pd.to_numeric(a.discount_price.astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
a = a[a.price.notna() & a.asin.notna() & ~a.image.str.contains("/images/W/IMAGERENDERING_")]
rows = []
for cat in CATS:
    d = a[a.sub_category == cat].drop_duplicates("asin").reset_index(drop=True)
    with ThreadPoolExecutor(32) as ex:
        d["ph"] = list(ex.map(phash, d.itertuples()))
    d = d[d.ph.notna()].reset_index(drop=True)
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform(d.name)
    S = (X @ X.T).tocoo()
    m = (S.row < S.col) & (S.data >= 0.5)
    i, j = S.row[m], S.col[m]
    ph = d.ph.to_numpy(dtype=np.uint64)
    close = np.bitwise_count(ph[i] ^ ph[j]) <= 6
    i, j = i[close], j[close]
    diff_price = (d.price.to_numpy()[i] != d.price.to_numpy()[j]).mean() if len(i) else float("nan")
    rows.append(dict(category=cat, listings=len(d), likely_dup_pairs=len(i),
                     listings_in_dup_pairs=len(set(i) | set(j)), share_with_diff_price=round(diff_price, 2)))
    print(rows[-1], flush=True)
    for x, y in list(zip(i, j))[:3]:
        print("   ", d.name[x][:60], "|", d.name[y][:60], "|", d.price[x], d.price[y])
pd.DataFrame(rows).to_csv(Path(__file__).with_name("duplicate_probe.csv"), index=False)
