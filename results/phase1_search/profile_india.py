"""Profile the two candidate Indian datasets (search task 1). Prints facts only."""
import json, sys, re
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

root = sys.argv[1]
pd.set_option("display.width", 250, "display.max_colwidth", 70, "display.max_columns", 20)

def price(s):
    # "₹1,299" -> 1299.0
    return pd.to_numeric(s.astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")

def tfidf_pairs(titles, thr=0.6):
    # count unordered title pairs with char 3-5 gram TF-IDF cosine > thr
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform(titles)
    S = (X @ X.T).tocsr()
    S.setdiag(0); S.eliminate_zeros()
    S = S.multiply(S > thr).tocoo()
    return int((S.row < S.col).sum())

# ---------- Amazon ----------
a = pd.read_csv(f"{root}/amazon/Amazon-Products.csv")
print("AMAZON columns:", list(a.columns), "rows:", len(a))
a["sell"] = price(a["discount_price"]); a["mrp"] = price(a["actual_price"])
print("rows with selling price:", a["sell"].notna().sum(), "| unique links:", a["link"].nunique(),
      "| sample price strings:", a["discount_price"].dropna().head(3).tolist())
print("main_category counts:\n", a["main_category"].value_counts().to_string())
sub = a.groupby("sub_category").agg(rows=("name", "size"), with_price=("sell", "count"), unique_links=("link", "nunique"))
print("sub_category counts (top 60 by rows):\n", sub.sort_values("rows", ascending=False).head(60).to_string())

# ---------- Flipkart ----------
f = pd.read_csv(f"{root}/flipkart/flipkart_com-ecommerce_sample.csv")
print("\nFLIPKART columns:", list(f.columns), "rows:", len(f))
print("crawl years:", pd.to_datetime(f["crawl_timestamp"], errors="coerce").dt.year.value_counts().to_dict())
f["top"] = f["product_category_tree"].str.strip('["]').str.split(" >> ").str[0]
f["lvl2"] = f["product_category_tree"].str.strip('["]').str.split(" >> ").str[:2].str.join(" > ")
print("rows with discounted_price:", f["discounted_price"].notna().sum(), "| unique pid:", f["pid"].nunique())
print("top category counts (top 20):\n", f["top"].value_counts().head(20).to_string())
print("level-2 counts (top 20):\n", f["lvl2"].value_counts().head(20).to_string())

# ---------- duplicate signal in generic candidate categories ----------
print("\nTF-IDF char3-5 cosine > 0.6 pairs, after dropping repeated links/pids and missing prices:")
for name in ["Kitchen and Dining", "Kitchen Storage and Containers", "Car Accessories", "Home Storage",
             "Backpacks", "Wallets", "Household Supplies", "Fitness Accessories", "Bags and Luggage"]:
    d = a[(a.sub_category == name) & a.sell.notna()].drop_duplicates("link")
    if len(d):
        print(f"  amazon/{name}: rows={len(d)} identical_title_dupes={int(d.name.duplicated().sum())} pairs>0.6={tfidf_pairs(d.name)}")
for name in ["Kitchen & Dining", "Mobiles & Accessories", "Automotive", "Home Decor & Festive Needs"]:
    d = f[(f.top == name) & f.discounted_price.notna()].drop_duplicates("pid")
    if len(d):
        print(f"  flipkart/{name}: rows={len(d)} pairs>0.6={tfidf_pairs(d.product_name)}")
