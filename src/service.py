"""Check ONE listing against the catalog: "is this product already listed?"

Uses exactly the trained pipeline (frozen PCA, TF-IDF, logistic regression M1)
and the same pair features as training, so a new listing is scored the way the
model was tested. The catalog is the Indian listings in cache/india_listings.csv.

Search uses a numpy dot product instead of FAISS: the catalog is small (hundreds
to a few thousand rows), exact search gives the same neighbours as IndexFlatIP,
and it avoids loading faiss next to torch (two OpenMP runtimes crash on this Mac).
"""
import io
import json

import joblib
import numpy as np
import pandas as pd
import requests

from src import config, features
from src.numtokens import compare, extract
from src.reduce import project

SAME = config.INDIA_MATCH_THRESHOLD  # "same product" on Indian data (calibrated, see config.py)
SIMILAR = 0.82                       # the Shopee threshold: likely a variant (colour, pack) if below SAME


class Matcher:
    def __init__(self):
        cfg = json.loads((config.CACHE_DIR / "final_config.json").read_text())
        self.setting, self.k = (cfg["pca_variant"], cfg["dim"]), cfg["k"]
        self.pcas = joblib.load(config.CACHE_DIR / "pca.joblib")
        self.model = joblib.load(config.CACHE_DIR / "models.joblib")["m1"]
        self.tfidf = features.load_tfidf()
        self.catalog = pd.read_csv(config.INDIA_LISTINGS)
        ids = pd.read_csv(config.CACHE_DIR / "india_ids.csv")["id"]
        assert ids.tolist() == self.catalog["listing_id"].tolist(), "catalog and embeddings out of sync"
        raw = {m: np.load(config.CACHE_DIR / f"india_{m}.npy") for m in ("img", "txt", "phash")}
        self.phash = raw["phash"]
        self.img = self._project(raw["img"], "img")
        self.txt = self._project(raw["txt"], "txt")
        self.ret = self._retrieval(self.img, self.txt)
        self.sims = self.ret @ self.ret.T          # catalog-to-catalog, for the neighbour-rank features
        np.fill_diagonal(self.sims, -np.inf)       # a listing is never its own neighbour
        self.tfidf_rows = self.tfidf.transform(self.catalog["title"])
        self.tokens = [extract(t) for t in self.catalog["title"]]
        self.lengths = self.catalog["title"].str.len().to_numpy(dtype=float)
        self._encoders = None

    def _project(self, X, m):
        return project(np.atleast_2d(X), self.pcas[m], self.setting[1], self.setting[0])

    @staticmethod
    def _retrieval(img, txt):
        """Same as candidates.retrieval_vectors: concatenated, scaled to unit length."""
        return (np.hstack([img, txt]) / np.sqrt(2)).astype(np.float32)

    def encode(self, title, image_bytes):
        """Raw CLIP and MPNet vectors + phash for a brand-new listing (loads models once)."""
        from src.embed import IMAGE_MODEL, TEXT_MODEL, encode, get_device, load_image  # torch only here
        if self._encoders is None:
            from sentence_transformers import SentenceTransformer
            device = get_device()
            self._encoders = (SentenceTransformer(IMAGE_MODEL, device=device),
                              SentenceTransformer(TEXT_MODEL, device=device))
        im, ph = load_image(io.BytesIO(image_bytes))
        img = encode(self._encoders[0], [im], batch_size=1)[0]
        txt = encode(self._encoders[1], [title], batch_size=1)[0]
        return img, txt, ph

    def match(self, img_raw, txt_raw, phash, title, exclude=None, top=12):
        """Score the listing against its k nearest catalog neighbours.

        exclude: a catalog row to leave out, used to replay a catalog listing as if
        it were new (leave-one-out), so the demo shows real model output.
        Rank features follow training: rank_ab = where the catalog item sits in the
        new listing's list, rank_ba = where the new listing would sit in the item's
        list (k + 1 if outside the top k).
        """
        q_img, q_txt = self._project(img_raw, "img"), self._project(txt_raw, "txt")
        s = self.ret @ self._retrieval(q_img, q_txt)[0]
        if exclude is not None:
            s[exclude] = -np.inf
        k = min(self.k, len(s) - (exclude is not None))
        cand = np.argsort(-s)[:k]
        others = self.sims[cand] > s[cand][:, None]
        if exclude is not None:
            others[:, exclude] = False
        rank_ab = np.arange(1, k + 1)
        rank_ba = np.minimum(1 + others.sum(axis=1), k + 1)
        q_tokens, q_len = extract(title), float(len(title))
        num = np.array([compare(q_tokens, self.tokens[j]) for j in cand], dtype=float).reshape(-1, 2)
        X = pd.DataFrame({
            "img_cos": self.img[cand] @ q_img[0],
            "txt_cos": self.txt[cand] @ q_txt[0],
            "tfidf_cos": np.asarray(self.tfidf_rows[cand] @ self.tfidf.transform([title]).T.toarray()).ravel(),
            "phash_dist": np.bitwise_count(self.phash[cand] ^ np.uint64(phash)).astype(float),
            "num_match": num[:, 0], "num_conflict": num[:, 1],
            "mutual": (rank_ba <= k).astype(float),
            "rank_min": np.minimum(rank_ab, rank_ba).astype(float),
            "rank_max": np.maximum(rank_ab, rank_ba).astype(float),
            "len_ratio": np.minimum(q_len, self.lengths[cand]) / np.maximum(q_len, self.lengths[cand]),
        })[features.FEATURES]
        prob = self.model.predict_proba(X)[:, 1]
        order = np.argsort(-prob)[:top]
        return [dict(row=int(cand[i]), prob=float(prob[i]), conflict=bool(num[i, 1]),
                     img_cos=float(X["img_cos"].iloc[i]), title_sim=float(X["tfidf_cos"].iloc[i]))
                for i in order]

    def explain(self, matches, price=None):
        """Verdict + price comparison for a match list, in plain words."""
        same = [m for m in matches if m["prob"] >= SAME and not m["conflict"]]
        similar = [m for m in matches if SIMILAR <= m["prob"] < SAME or (m["prob"] >= SAME and m["conflict"])]
        prices = self.catalog["price"].to_numpy()[[m["row"] for m in same]]
        # typical price = median of at least 2 OTHER listings of the same product (one other price is too thin)
        typical = float(np.median(prices)) if len(prices) >= 2 else None
        verdict = ("already_listed" if same else "possible_variant" if similar else "looks_new")
        out = dict(verdict=verdict, same=len(same), similar=len(similar), typical_price=typical,
                   min_price=float(prices.min()) if len(prices) else None,
                   max_price=float(prices.max()) if len(prices) else None)
        if price and typical:
            out["your_price_vs_typical_pct"] = round(100 * (price / typical - 1), 1)
        rows = self.catalog
        out["matches"] = [dict(listing_id=rows.at[m["row"], "listing_id"], title=rows.at[m["row"], "title"],
                               price=float(rows.at[m["row"], "price"]), url=rows.at[m["row"], "url"],
                               image_url=rows.at[m["row"], "image_url"], prob=round(m["prob"], 4),
                               label="same product" if m in same else "similar (variant?)" if m in similar else "different",
                               pct_vs_typical=round(100 * (rows.at[m["row"], "price"] / typical - 1), 1) if typical else None)
                          for m in matches if m["prob"] >= SIMILAR]
        return out

    def check_new(self, title, image_bytes=None, image_url=None, price=None):
        """Full check for a listing the model has never seen: photo (bytes or URL) + title."""
        if image_bytes is None:
            r = requests.get(image_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            image_bytes = r.content
        img, txt, ph = self.encode(title, image_bytes)
        return self.explain(self.match(img, txt, ph, title), price)

    def replay(self, row, raw):
        """Leave-one-out check of catalog row `row`, using its cached raw vectors."""
        title = self.catalog.at[row, "title"]
        res = self.explain(self.match(raw["img"][row], raw["txt"][row], raw["phash"][row], title, exclude=row),
                           float(self.catalog.at[row, "price"]))
        return res


def build_demo_json():
    """Leave-one-out results for every catalog listing -> dashboard/data/catalog.json.

    The hosted (static) dashboard cannot run the models, so it shows these real
    model outputs: each listing checked as if it were new, against all the others.
    """
    m = Matcher()
    raw = {k: np.load(config.CACHE_DIR / f"india_{k}.npy") for k in ("img", "txt", "phash")}
    cat = m.catalog
    listings = [dict(id=r.listing_id, title=r.title, price=float(r.price), url=r.url, image=r.image_url)
                for r in cat.itertuples()]
    pos = {lid: i for i, lid in enumerate(cat["listing_id"])}
    results = []
    for i in range(len(cat)):
        res = m.replay(i, raw)
        results.append(dict(verdict=res["verdict"], same=res["same"], similar=res["similar"],
                            typical=res["typical_price"], pct=res.get("your_price_vs_typical_pct"),
                            matches=[[pos[x["listing_id"]], x["prob"], "same" if x["label"] == "same product" else "similar"] for x in res["matches"]]))
    out = config.ROOT / "dashboard" / "data" / "catalog.json"
    out.write_text(json.dumps(dict(category=config.INDIA_CATEGORY["amazon"], same_threshold=SAME,
                                   similar_threshold=SIMILAR, listings=listings, results=results)))
    print("wrote", out, f"({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build_demo_json()
