"""Section 3 duplicate check for every candidate Indian category, on both sites.

For each category: priced rows, rows with live images, title pairs with TF-IDF
char 3-5-gram cosine > 0.6 among the live rows, and the share of live listings
that have at least one such partner (a size-independent version of the count).

Candidates = generic, non-electronics categories with >= 1,000 priced rows on
Amazon, and every non-electronics top-level Flipkart category. Excluded:
electronics and appliances (blueprint rule), umbrella or store pages that mix
many product types ("All ...", "stores", "Bags & Luggage"), and media.

Live image = HTTP 200 with an image content type. Categories over 5,000 rows are
too slow to check in full, so there we count rows whose URL lacks the dead
/W/IMAGERENDERING_ pattern and verify that rule on a 100-row sample.
Usage: python category_check.py <india_raw dir> <output csv>
"""
import hashlib, json, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd, requests
from sklearn.feature_extraction.text import TfidfVectorizer

root, out_csv = sys.argv[1], sys.argv[2]
FULL_CHECK_MAX = 5000
DEAD = "/images/W/IMAGERENDERING_"

AMAZON = ["Fashion & Silver Jewellery", "Jewellery", "Handbags & Clutches", "Watches", "Gold & Diamond Jewellery",
          "Sunglasses", "Rucksacks", "Backpacks", "Travel Accessories", "Wallets", "Make-up", "Beauty & Grooming",
          "Diet & Nutrition", "Car Accessories", "Motorbike Accessories & Parts", "Car Parts", "Home Furnishing",
          "Furniture", "Home Décor", "Indoor Lighting", "Bedroom Linen", "Home Storage", "Kitchen & Dining",
          "Sewing & Craft Supplies", "Test, Measure & Inspect", "Lab & Scientific", "Kids' Fashion", "Baby Fashion",
          "Kids' Watches", "School Bags", "Kids' Clothing", "Kids' Shoes", "T-shirts & Polos", "Shirts", "Jeans",
          "Innerwear", "Casual Shoes", "Sports Shoes", "Formal Shoes", "Fitness Accessories", "Football",
          "Badminton", "Cricket", "Strength Training", "Yoga", "Baby Bath, Skin & Grooming", "Diapers", "Clothing",
          "Ethnic Wear", "Western Wear", "Lingerie & Nightwear", "Fashion Sandals", "Shoes"]
FLIPKART = ["Clothing", "Jewellery", "Footwear", "Mobiles & Accessories", "Automotive", "Home Decor & Festive Needs",
            "Beauty and Personal Care", "Home Furnishing", "Kitchen & Dining", "Watches", "Baby Care",
            "Tools & Hardware", "Toys & School Supplies", "Pens & Stationery", "Bags, Wallets & Belts", "Furniture",
            "Sports & Fitness"]


def fetch(url):
    """Return md5 of the image bytes if the URL is live, else None."""
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
            return hashlib.md5(r.content).hexdigest()
    except Exception:
        pass
    return None


def fetch_all(urls):
    with ThreadPoolExecutor(32) as ex:
        return list(ex.map(fetch, urls))


def pair_stats(titles, thr=0.6, chunk=500):
    """Unordered pairs with cosine > thr, and share of listings with >= 1 such partner."""
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform(titles)
    n, pairs, has_partner = X.shape[0], 0, np.zeros(X.shape[0], bool)
    for start in range(0, n, chunk):
        S = (X[start:start + chunk] @ X.T).tocoo()
        rows = S.row + start
        keep = (S.data > thr) & (S.col != rows)
        pairs += int((keep & (S.col > rows)).sum())
        has_partner[rows[keep]] = True
    return pairs, float(has_partner.mean())


def check(site, category, d, title_col, url_col):
    """Live-image count, then duplicate stats on the live rows only."""
    d = d.dropna(subset=[url_col]).copy()
    if len(d) <= FULL_CHECK_MAX or site == "flipkart":
        d["md5"] = fetch_all(d[url_col].tolist())
        live, method, verify = d[d.md5.notna()], "full", ""
        top_same = Counter(live.md5).most_common(1)[0][1] if len(live) else 0
    else:
        s = d.sample(100, random_state=42)
        ok = np.array([m is not None for m in fetch_all(s[url_col].tolist())])
        verify = f"{int((ok == ~s[url_col].str.contains(DEAD).values).sum())}/100"
        live, method, top_same = d[~d[url_col].str.contains(DEAD)], "url-rule", ""
    pairs, share = pair_stats(live[title_col]) if len(live) > 1 else (0, 0.0)
    row = dict(site=site, category=category, priced_rows=len(d), live_rows=len(live), live_method=method,
               rule_agreement=verify, pairs_gt_0_6=pairs, share_with_partner=round(share, 3),
               identical_titles=int(live[title_col].duplicated().sum()), max_rows_same_image=top_same)
    print(row, flush=True)
    return row


a = pd.read_csv(f"{root}/amazon/Amazon-Products.csv")
a = a[pd.to_numeric(a.discount_price.astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce").notna()]
f = pd.read_csv(f"{root}/flipkart/flipkart_com-ecommerce_sample.csv")
f = f[f.discounted_price.notna()]
f["top"] = f.product_category_tree.str.strip('["]').str.split(" >> ").str[0]
f["first_img"] = f.image.apply(lambda s: json.loads(s)[0] if isinstance(s, str) and s.startswith("[") else None)

rows = [check("amazon", c, a[a.sub_category == c], "name", "image") for c in AMAZON]
rows += [check("flipkart", c, f[f.top == c].drop_duplicates("pid"), "product_name", "first_img") for c in FLIPKART]
pd.DataFrame(rows).to_csv(out_csv, index=False)
print("wrote", out_csv)
