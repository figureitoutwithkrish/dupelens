"""India price step, part 1: load the chosen category from each site, clean
prices, download images, and drop listings whose image cannot be downloaded.

The frozen-model scoring, grouping, price stats and labelling sheet are added
after the Shopee model is final (blueprint section 8).
Run: python -m src.india
"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from PIL import Image

from src import config

COLUMNS = ["listing_id", "site", "title", "price", "mrp", "image_url", "url"]


def _rupees(s):
    """'₹1,299' -> 1299.0; anything unparseable -> NaN."""
    return pd.to_numeric(s.astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")


def load_amazon(category):
    """Amazon rows of one sub_category in the shared column format.

    price = discount_price, the selling price; actual_price is the MRP, which is
    not what buyers pay. The id is a hash of the product link, which is unique.
    """
    a = pd.read_csv(config.AMAZON_CSV)
    a = a[a["sub_category"] == category]
    # one row per product page: the same ASIN scraped twice is not a duplicate listing
    a = a.drop_duplicates(a["link"].str.extract(r"/dp/([A-Z0-9]{10})", expand=False).fillna(a["link"]))
    return pd.DataFrame({
        "listing_id": "amazon_" + a["link"].map(lambda u: hashlib.md5(u.encode()).hexdigest()[:12]),
        "site": "amazon", "title": a["name"], "price": _rupees(a["discount_price"]),
        "mrp": _rupees(a["actual_price"]), "image_url": a["image"], "url": a["link"],
    })


def load_flipkart(category):
    """Flipkart rows whose top-level category matches, in the shared column format.

    price = discounted_price (selling price); retail_price is the MRP. The image
    column holds a JSON list of URLs; we use the first, as the listing photo.
    """
    f = pd.read_csv(config.FLIPKART_CSV).drop_duplicates("pid")  # 2 pids appear twice
    top = f["product_category_tree"].str.strip('["]').str.split(" >> ").str[0]
    f = f[top == category]
    first_image = f["image"].map(lambda s: json.loads(s)[0] if isinstance(s, str) and s.startswith("[") else None)
    return pd.DataFrame({
        "listing_id": "flipkart_" + f["pid"], "site": "flipkart", "title": f["product_name"],
        "price": pd.to_numeric(f["discounted_price"], errors="coerce"),
        "mrp": pd.to_numeric(f["retail_price"], errors="coerce"), "image_url": first_image, "url": f["product_url"],
    })


def download_image(row):
    """Save one listing photo; return True only if a decodable image is on disk.

    Already-downloaded files are re-checked, not re-fetched, so a crash mid-run
    loses nothing. A file that fails to decode counts as not downloadable.
    """
    path = config.INDIA_IMAGES / f"{row.listing_id}.jpg"
    if not path.exists():
        try:
            r = requests.get(row.image_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image"):
                return False
            path.write_bytes(r.content)
        except requests.RequestException:
            return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        path.unlink(missing_ok=True)
        return False


def prepare(categories=config.INDIA_CATEGORY):
    """Build cache/india_listings.csv and return it with a per-site count table.

    Listings without a selling price or image URL are dropped first, then those
    whose image does not download. Image features are never imputed (blueprint
    section 3), so a listing without its photo cannot be scored fairly.
    """
    loaders = {"amazon": load_amazon, "flipkart": load_flipkart}
    frames = [loaders[site](cat) for site, cat in categories.items() if cat]
    df = pd.concat(frames, ignore_index=True)[COLUMNS]
    counts = df.groupby("site").size().rename("in_category").to_frame()

    df = df[(df["price"] > 0) & df["image_url"].notna()]
    counts["with_price_and_url"] = df.groupby("site").size()

    config.INDIA_IMAGES.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(32) as ex:
        ok = list(ex.map(download_image, df.itertuples(index=False)))
    df = df[ok]
    counts["image_downloaded"] = df.groupby("site").size()
    counts["dropped_no_image"] = counts["with_price_and_url"] - counts["image_downloaded"]

    config.INDIA_LISTINGS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.INDIA_LISTINGS, index=False)
    return df, counts


def score_site(sub, emb_rows, pcas, vec, model, cfg):
    """Frozen Shopee pipeline on one site's listings: candidates, features, M1 probabilities.

    Nothing is refitted: PCA, TF-IDF, model and thresholds all come from Shopee train/val.
    """
    import faiss  # noqa: F401  import before sklearn/torch users (OpenMP clash, see config.py)
    from src import features
    from src.candidates import candidate_pairs, retrieval_vectors, top_k
    from src.reduce import project

    s = (cfg["pca_variant"], cfg["dim"])
    img = project(emb_rows["img"], pcas["img"], s[1], s[0])
    txt = project(emb_rows["txt"], pcas["txt"], s[1], s[0])
    k = min(cfg["k"], len(sub) - 1)
    pairs = candidate_pairs(top_k(retrieval_vectors(img, txt), k)[0], k)
    X = features.build(pairs, img, txt, vec.transform(sub["title"]), emb_rows["phash"], sub["title"].tolist(), k)
    return pairs, X, model.predict_proba(X)[:, 1]


def price_groups(sub, comp):
    """Per predicted group: listings with price and % vs the group's typical (median) price.

    Groups of 3+ get a typical price; a listing more than 20% above it is flagged.
    Groups of 2 only list both prices (a median of two says little).
    """
    out = []
    for c, g in sub.assign(group=comp).groupby("group"):
        if len(g) < 2:
            continue
        typical = float(g["price"].median()) if len(g) >= 3 else None
        rows = [dict(listing_id=r.listing_id, title=r.title, price=float(r.price), url=r.url,
                     pct_vs_typical=round(100 * (r.price / typical - 1), 1) if typical else None,
                     above_20pct=bool(typical and r.price > 1.2 * typical)) for r in g.itertuples()]
        out.append(dict(site=g["site"].iloc[0], group_id=f"{g['site'].iloc[0]}_{c}", size=len(g),
                        min_price=float(g["price"].min()), max_price=float(g["price"].max()), typical_price=typical,
                        spread_pct=round(100 * (g["price"].max() / g["price"].min() - 1), 1),
                        review_chaining=len(g) > 15, listings=rows))
    return out


def labelling_sheets(pairs_all, threshold, n_per_stratum=50):
    """Blind labelling sheet (blueprint 8.4): strata H, B, R; titles and image links only.

    H: predicted matches with prob >= cut (cut = 0.9, or midway between threshold and 1
       if 0.9 is below the threshold); B: predicted matches below cut; R: rejected pairs
       with TF-IDF cosine >= 0.5 (to estimate misses). 50 per stratum, 150 unique pairs,
       shuffled. The team chose not to label by hand, so ONE labeller (Claude, an AI
       assistant) labels all 150 from titles and photos; there is no second labeller,
       so no agreement (kappa) can be measured. The key (stratum, probability) stays
       hidden in cache/labelling_key.csv and is only joined after labelling.
    """
    cut = 0.9 if 0.9 >= threshold else (threshold + 1) / 2
    strata = {"H": pairs_all["prob"] >= cut,
              "B": (pairs_all["prob"] >= threshold) & (pairs_all["prob"] < cut),
              "R": (pairs_all["prob"] < threshold) & (pairs_all["tfidf_cos"] >= 0.5)}
    picks, sizes = [], {}
    for name, mask in strata.items():
        pool = pairs_all[mask]
        sizes[name] = int(len(pool))
        picks.append(pool.sample(min(n_per_stratum, len(pool)), random_state=config.SEED).assign(stratum=name))
    key = pd.concat(picks).sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    key["pair_id"] = [f"P{i:03d}" for i in range(len(key))]
    cols = ["pair_id", "site", "title_a", "title_b", "image_url_a", "image_url_b", "id_a", "id_b"]
    key[cols].assign(same_product="").to_csv(config.RESULTS_DIR / "labelling_sheet.csv", index=False)
    key.to_csv(config.CACHE_DIR / "labelling_key.csv", index=False)
    return sizes, cut


def make_grids(per_grid=10, thumb=200, sheet_name="labelling_sheet.csv", prefix="grid"):
    """Contact sheets of the labelling pairs: each row = pair id, photo A, photo B.

    Lets the labeller see both photos side by side; titles are read from the sheet.
    """
    from PIL import ImageDraw
    sheet = pd.read_csv(config.RESULTS_DIR / sheet_name)
    out = config.RESULTS_DIR / "labelling"
    out.mkdir(parents=True, exist_ok=True)
    for g, start in enumerate(range(0, len(sheet), per_grid)):
        rows = sheet.iloc[start:start + per_grid]
        canvas = Image.new("RGB", (60 + 2 * thumb, len(rows) * thumb), "white")
        draw = ImageDraw.Draw(canvas)
        for r, row in enumerate(rows.itertuples()):
            draw.text((5, r * thumb + thumb // 2), row.pair_id, fill="black")
            for c, lid in enumerate([row.id_a, row.id_b]):
                with Image.open(config.INDIA_IMAGES / f"{lid}.jpg") as im:
                    im = im.convert("RGB")
                    im.thumbnail((thumb - 4, thumb - 4))
                    canvas.paste(im, (60 + c * thumb, r * thumb))
        canvas.save(out / f"{prefix}_{g:02d}.jpg", quality=80)


def label_stats(n_boot=config.N_BOOT):
    """Blueprint 8.5 from the filled sheet: stratum-weighted precision, miss-rate proxy.

    precision = (N_H * prec_H + N_B * prec_B) / (N_H + N_B), where N is the number of
    predicted matches in each stratum (from india_summary.json), so the 50+50 sample
    is re-weighted to the real mix of high- and borderline-confidence matches.
    CI: bootstrap pairs within each stratum (1,000 resamples), recomputing the formula.
    """
    labels = pd.read_csv(config.RESULTS_DIR / "labelling_sheet.csv")[["pair_id", "same_product"]]
    key = pd.read_csv(config.CACHE_DIR / "labelling_key.csv").merge(labels, on="pair_id", validate="one_to_one")
    summary_path = config.RESULTS_DIR / "json" / "india_summary.json"
    summary = json.loads(summary_path.read_text())
    n = summary["strata_sizes"]
    y = {s: key.loc[key.stratum == s, "same_product"].astype(int).to_numpy() for s in ("H", "B", "R")}
    weighted = lambda h, b: (n["H"] * h.mean() + n["B"] * b.mean()) / (n["H"] + n["B"])
    rng = np.random.default_rng(config.SEED)
    boots = [weighted(rng.choice(y["H"], len(y["H"])), rng.choice(y["B"], len(y["B"]))) for _ in range(n_boot)]
    stats = dict(labeller="Claude (AI assistant), from titles and photos; no human labels",
                 n_labelled={s: int(len(v)) for s, v in y.items()},
                 precision_H=float(y["H"].mean()), precision_B=float(y["B"].mean()),
                 weighted_precision=float(weighted(y["H"], y["B"])),
                 weighted_precision_ci=[float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                 r_share_true_matches=float(y["R"].mean()),
                 by_site={site: float(g["same_product"].mean()) for site, g in key[key.stratum != "R"].groupby("site")})
    summary["labelling"] = stats
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return stats


def run_frozen():
    """India step (blueprint 8): frozen model per site, groups, price stats, shift check, sheets."""
    import faiss  # noqa: F401  first, before sklearn models load (OpenMP, see config.py)
    import joblib
    from src import group
    from src.data import load_cached, load_split
    from src.features import load_tfidf
    from src.reduce import explained_variance

    cfg = json.loads((config.CACHE_DIR / "final_config.json").read_text())
    pcas = joblib.load(config.CACHE_DIR / "pca.joblib")
    model = joblib.load(config.CACHE_DIR / "models.joblib")["m1"]
    vec = load_tfidf()
    india = pd.read_csv(config.INDIA_LISTINGS)
    emb = load_cached("india", india["listing_id"])

    groups, all_pairs, stats = [], [], {}
    for site, sub in india.groupby("site", sort=False):  # search and group within one site only
        rows = sub.index.to_numpy()
        sub = sub.reset_index(drop=True)
        pairs, X, prob = score_site(sub, {m: emb[m][rows] for m in emb}, pcas, vec, model, cfg)
        a, b = pairs["a"].to_numpy(), pairs["b"].to_numpy()
        keep = group.group_edges(prob, cfg["component_threshold"], X["num_conflict"].to_numpy())
        comp = group.components(len(sub), a, b, keep)
        site_groups = price_groups(sub, comp)
        groups += site_groups
        all_pairs.append(pd.DataFrame({
            "site": site, "prob": prob, "tfidf_cos": X["tfidf_cos"],
            "title_a": sub["title"].to_numpy()[a], "title_b": sub["title"].to_numpy()[b],
            "image_url_a": sub["image_url"].to_numpy()[a], "image_url_b": sub["image_url"].to_numpy()[b],
            "id_a": sub["listing_id"].to_numpy()[a], "id_b": sub["listing_id"].to_numpy()[b]}))
        sizes = [g["size"] for g in site_groups]
        stats[site] = dict(listings=len(sub), candidate_pairs=len(pairs),
                           predicted_match_pairs=int((prob >= cfg["m1_threshold"]).sum()),
                           groups_2plus=len(sizes), groups_3plus=sum(s >= 3 for s in sizes),
                           listings_in_groups=int(sum(sizes)), groups_over_15=sum(s > 15 for s in sizes),
                           largest_group=max(sizes, default=0),
                           listings_above_20pct=sum(r["above_20pct"] for g in site_groups for r in g["listings"]))

    # Shift check: frozen Shopee PCA on Indian vectors vs on Shopee test, per modality.
    df = load_split()
    shop = load_cached("shopee", df["posting_id"])
    test = df["split"].eq("test").to_numpy()
    d = cfg["dim"]
    shift = {m: dict(shopee_test=float(explained_variance(pcas[m], shop[m][test])[d - 1]),
                     india=float(explained_variance(pcas[m], emb[m])[d - 1]), dim=d) for m in ("img", "txt")} \
        if d else {}

    sizes, cut = labelling_sheets(pd.concat(all_pairs, ignore_index=True), cfg["m1_threshold"])
    out = config.RESULTS_DIR / "json"
    out.mkdir(parents=True, exist_ok=True)
    (out / "india_groups.json").write_text(json.dumps(groups, indent=2, ensure_ascii=False))
    (out / "india_summary.json").write_text(json.dumps(
        dict(category=config.INDIA_CATEGORY, sites=stats, shift_check=shift, strata_sizes=sizes, h_cut=cut),
        indent=2, ensure_ascii=False))
    print(json.dumps(dict(sites=stats, shift_check=shift, strata_sizes=sizes, h_cut=cut), indent=2))


def run_matches(n_holdout=60):
    """Price view from DIRECT matches (no chaining) + a held-out labelling sample.

    Each listing's 'same product' set = itself + listings it matches directly with
    prob >= INDIA_MATCH_THRESHOLD and no number-token conflict. Connected components
    chained even at 0.999 on India, and on Shopee test direct matches scored higher
    (pairwise F1) than components, so direct matches are used for prices.
    Held-out sample: matches NOT in the calibration sheet, to measure precision honestly.
    """
    import faiss  # noqa: F401
    import joblib
    from src.data import load_cached
    from src.features import load_tfidf

    cfg = json.loads((config.CACHE_DIR / "final_config.json").read_text())
    pcas = joblib.load(config.CACHE_DIR / "pca.joblib")
    model = joblib.load(config.CACHE_DIR / "models.joblib")["m1"]
    india = pd.read_csv(config.INDIA_LISTINGS)
    emb = load_cached("india", india["listing_id"])
    out, all_matches = [], []
    for site, sub in india.groupby("site", sort=False):
        rows = sub.index.to_numpy()
        sub = sub.reset_index(drop=True)
        pairs, X, prob = score_site(sub, {m: emb[m][rows] for m in emb}, pcas, load_tfidf(), model, cfg)
        keep = (prob >= config.INDIA_MATCH_THRESHOLD) & (X["num_conflict"].to_numpy() == 0)
        a, b, p = pairs["a"].to_numpy()[keep], pairs["b"].to_numpy()[keep], prob[keep]
        partners = {i: [] for i in range(len(sub))}
        for i, j, s in zip(a, b, p):
            partners[i].append((j, s))
            partners[j].append((i, s))
        for i, plist in partners.items():
            if not plist:
                continue
            members = [i] + [j for j, _ in plist]
            prices = sub.loc[members, "price"]
            typical = float(prices.median()) if len(members) >= 3 else None
            price = float(sub.at[i, "price"])
            out.append(dict(site=site, listing_id=sub.at[i, "listing_id"], title=sub.at[i, "title"], price=price,
                            url=sub.at[i, "url"], typical_price=typical,
                            pct_vs_typical=round(100 * (price / typical - 1), 1) if typical else None,
                            above_20pct=bool(typical and price > 1.2 * typical),
                            min_price=float(prices.min()), max_price=float(prices.max()),
                            matches=[dict(listing_id=sub.at[j, "listing_id"], title=sub.at[j, "title"],
                                          price=float(sub.at[j, "price"]), url=sub.at[j, "url"], prob=round(float(s), 4))
                                     for j, s in sorted(plist, key=lambda x: -x[1])]))
        all_matches.append(pd.DataFrame({
            "site": site, "prob": p, "id_a": sub["listing_id"].to_numpy()[a], "id_b": sub["listing_id"].to_numpy()[b],
            "title_a": sub["title"].to_numpy()[a], "title_b": sub["title"].to_numpy()[b],
            "image_url_a": sub["image_url"].to_numpy()[a], "image_url_b": sub["image_url"].to_numpy()[b]}))
    (config.RESULTS_DIR / "json" / "india_matches.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # Held-out sample: matches not used for calibration, shuffled, scores hidden.
    matches = pd.concat(all_matches, ignore_index=True)
    calib = pd.read_csv(config.CACHE_DIR / "labelling_key.csv")
    used = set(zip(calib["id_a"], calib["id_b"])) | set(zip(calib["id_b"], calib["id_a"]))
    fresh = matches[[(x, y) not in used for x, y in zip(matches["id_a"], matches["id_b"])]]
    key = fresh.sample(min(n_holdout, len(fresh)), random_state=config.SEED + 1).reset_index(drop=True)
    key["pair_id"] = [f"H{i:03d}" for i in range(len(key))]
    key.to_csv(config.CACHE_DIR / "holdout_key.csv", index=False)
    cols = ["pair_id", "site", "title_a", "title_b", "image_url_a", "image_url_b", "id_a", "id_b"]
    key[cols].assign(same_product="").to_csv(config.RESULTS_DIR / "holdout_sheet.csv", index=False)
    summary = dict(threshold=config.INDIA_MATCH_THRESHOLD, listings=int(len(india)),
                   listings_with_match=len(out), listings_with_typical_price=sum(o["typical_price"] is not None for o in out),
                   listings_above_20pct=sum(o["above_20pct"] for o in out), match_pairs=int(len(matches)),
                   holdout_pairs=int(len(key)))
    print(json.dumps(summary, indent=2))
    return summary


def holdout_stats(n_boot=config.N_BOOT):
    """Precision of India matches on the held-out sample, with a bootstrap 95% CI."""
    y = pd.read_csv(config.RESULTS_DIR / "holdout_sheet.csv")["same_product"].astype(int).to_numpy()
    rng = np.random.default_rng(config.SEED)
    boots = [rng.choice(y, len(y)).mean() for _ in range(n_boot)]
    stats = dict(labeller="Claude (AI assistant), blind, from titles and photos; no human labels",
                 threshold=config.INDIA_MATCH_THRESHOLD, n=int(len(y)), precision=float(y.mean()),
                 precision_ci=[float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))])
    path = config.RESULTS_DIR / "json" / "india_summary.json"
    summary = json.loads(path.read_text())
    summary["holdout"] = stats
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return stats


if __name__ == "__main__":
    import sys
    if "--matches" in sys.argv:
        run_matches()
        make_grids(sheet_name="holdout_sheet.csv", prefix="holdout")
        sys.exit()
    if "--photos" in sys.argv:
        # Fresh machine: download photos for the COMMITTED catalog without rebuilding it, so the
        # catalog rows stay aligned with the committed cache/india_*.npy vectors.
        cat = pd.read_csv(config.INDIA_LISTINGS)
        config.INDIA_IMAGES.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(32) as ex:
            ok = list(ex.map(download_image, cat.itertuples(index=False)))
        print(f"photos on disk: {sum(ok)} of {len(cat)}")
        sys.exit()
    if "--holdout-stats" in sys.argv:
        print(json.dumps(holdout_stats(), indent=2))
        sys.exit()
    if "--frozen" in sys.argv:
        run_frozen()
        make_grids()
        sys.exit()
    if "--label-stats" in sys.argv:
        print(json.dumps(label_stats(), indent=2))
        sys.exit()
    df, counts = prepare()
    print("categories:", config.INDIA_CATEGORY)
    print(counts.to_string())
    print("price summary (selling price, ₹):")
    print(df.groupby("site")["price"].describe().round(0).to_string())
    print("wrote", config.INDIA_LISTINGS)
