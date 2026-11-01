"""Inspect exact repeated titles in shortlisted categories (live-image rows only).

Shows whether high TF-IDF pair counts come from true duplicates or from generic
template titles: a title repeated hundreds of times with a 5x price range is
many different items, not one product.
Usage: python repeated_titles.py <india_raw dir>
"""
import sys

import pandas as pd

r = sys.argv[1]
pd.set_option("display.width", 220, "display.max_colwidth", 90)
num = lambda s: pd.to_numeric(s.astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")

a = pd.read_csv(f"{r}/amazon/Amazon-Products.csv")
a["price"] = num(a.discount_price)
a = a[a.price.notna() & ~a.image.str.contains("/images/W/IMAGERENDERING_")]
f = pd.read_csv(f"{r}/flipkart/flipkart_com-ecommerce_sample.csv")
f = f[f.discounted_price.notna()].drop_duplicates("pid")
f["top"] = f.product_category_tree.str.strip('["]').str.split(" >> ").str[0]
f["price"] = f.discounted_price


def show(site, cat, d, col):
    g = d.groupby(col).price.agg(["size", "min", "max"])
    g = g[g["size"] > 1]
    print(f"\n{site}/{cat}: titles repeated on >1 listing = {len(g)}; "
          f"share of those with different prices = {(g['max'] > g['min']).mean():.2f}")
    print(g.sort_values("size", ascending=False).head(5).to_string())


for c in ["Gold & Diamond Jewellery", "Fashion & Silver Jewellery", "Shirts", "Backpacks", "Kitchen & Dining"]:
    show("amazon", c, a[a.sub_category == c], "name")
for c in ["Jewellery", "Clothing", "Mobiles & Accessories", "Kitchen & Dining"]:
    show("flipkart", c, f[f.top == c], "product_name")
