"""Second pass: '&' category names, live image URL share in 50-row samples, 20 sample rows."""
import json, sys
import numpy as np, pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer

root = sys.argv[1]
pd.set_option("display.width", 250, "display.max_colwidth", 60, "display.max_columns", 20)
price = lambda s: pd.to_numeric(s.astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")

def tfidf_pairs(titles, thr=0.6):
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform(titles)
    S = (X @ X.T).tocsr(); S.setdiag(0); S.eliminate_zeros()
    S = S.multiply(S > thr).tocoo()
    return int((S.row < S.col).sum())

def live(url):
    # live = HTTP 200 and an image content type
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        return r.status_code == 200 and r.headers.get("content-type", "").startswith("image")
    except Exception:
        return False

def live_share(urls):
    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(live, urls))
    return sum(ok), len(ok)

a = pd.read_csv(f"{root}/amazon/Amazon-Products.csv")
a["sell"] = price(a["discount_price"])
f = pd.read_csv(f"{root}/flipkart/flipkart_com-ecommerce_sample.csv")
f["top"] = f["product_category_tree"].str.strip('["]').str.split(" >> ").str[0]
f["first_img"] = f["image"].apply(lambda s: json.loads(s)[0] if isinstance(s, str) and s.startswith("[") else None)

print("Amazon, generic sub_categories:")
for name in ["Kitchen & Dining", "Kitchen Storage & Containers", "Car Accessories", "Backpacks", "Wallets",
             "Travel Accessories", "Bags & Luggage", "Home Storage", "Fitness Accessories"]:
    d = a[(a.sub_category == name) & a.sell.notna()]
    if not len(d):
        print(f"  {name}: 0 rows"); continue
    s = d.sample(50, random_state=42)
    ok, n = live_share(s.image.tolist())
    print(f"  {name}: priced_rows={len(d)} pairs>0.6={tfidf_pairs(d.name)} live_images={ok}/{n}")

print("Flipkart:")
for name in ["Kitchen & Dining", "Mobiles & Accessories", "Automotive"]:
    d = f[(f.top == name) & f.discounted_price.notna() & f.first_img.notna()]
    s = d.sample(50, random_state=42)
    ok, n = live_share(s.first_img.tolist())
    print(f"  {name}: rows_with_price_and_image={len(d)} live_images={ok}/{n}")

for name in ["Wallets", "Car Accessories"]:
    print(f"\n20 sample rows, Amazon {name} (seed 42):")
    print(a[(a.sub_category == name) & a.sell.notna()].sample(20, random_state=42)[
        ["name", "discount_price", "actual_price", "ratings", "no_of_ratings", "image"]].to_string(index=False))
