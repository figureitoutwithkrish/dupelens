"""Check ONE listing against the catalog: "is this product already listed?"

Uses the trained pipeline with the v2 model (M1_v2 = M1 features + DINOv2 cosine
+ colour distance) and the three-way verdict from src/verdict.py:
  same     = same model, same colour and pack  -> used for the typical price
  variant  = same model, other colour/size/pack/combo
  different
Every pair feature is computed the way it was in training, so a new listing is
scored like the validation pairs were.

Search uses a numpy dot product instead of FAISS: the catalog is small, exact
search gives the same neighbours as IndexFlatIP, and it avoids loading faiss next
to torch (two OpenMP runtimes crash on this Mac).
"""
import io
import json

import joblib
import numpy as np
import pandas as pd
import requests

from src import config
from src.extra import colour_dist
from src.numtokens import compare, extract
from src.reduce import project
from src.verdict import classify, combo_mismatch

CONFIG_V2 = config.CACHE_DIR / "final_config_v2.json"


class Matcher:
    def __init__(self):
        self.cfg = json.loads(CONFIG_V2.read_text())
        self.setting, self.k = (self.cfg["pca_variant"], self.cfg["dim"]), self.cfg["k"]
        self.pcas = joblib.load(config.CACHE_DIR / "pca.joblib")
        self.model = joblib.load(config.CACHE_DIR / "models_v2.joblib")["m1_v2"]
        self.tfidf = joblib.load(config.CACHE_DIR / "tfidf.joblib")
        self.catalog = pd.read_csv(config.INDIA_LISTINGS)
        ids = pd.read_csv(config.CACHE_DIR / "india_ids.csv")["id"]
        assert ids.tolist() == self.catalog["listing_id"].tolist(), "catalog and embeddings out of sync"
        self.raw = {m: np.load(config.CACHE_DIR / f"india_{m}.npy") for m in ("img", "txt", "phash", "dino", "colour")}
        self.img = self._project(self.raw["img"], "img")
        self.txt = self._project(self.raw["txt"], "txt")
        self.ret = self._retrieval(self.img, self.txt)
        self.sims = self.ret @ self.ret.T          # catalog-to-catalog, for the neighbour-rank features
        np.fill_diagonal(self.sims, -np.inf)       # a listing is never its own neighbour
        self.tfidf_rows = self.tfidf.transform(self.catalog["title"])
        self.titles = self.catalog["title"].tolist()
        self.tokens = [extract(t) for t in self.titles]
        self.lengths = self.catalog["title"].str.len().to_numpy(dtype=float)
        self._encoders = None

    def _project(self, X, m):
        return project(np.atleast_2d(X), self.pcas[m], self.setting[1], self.setting[0])

    @staticmethod
    def _retrieval(img, txt):
        """Same as candidates.retrieval_vectors: concatenated, scaled to unit length."""
        return (np.hstack([img, txt]) / np.sqrt(2)).astype(np.float32)

    def encode(self, title, image_bytes):
        """All signals for a brand-new listing: CLIP, MPNet, phash, DINOv2, colour (models load once)."""
        from src.embed import IMAGE_MODEL, TEXT_MODEL, encode, get_device, load_image  # torch only here
        from src.extra import Dino, colour_hist
        if self._encoders is None:
            from sentence_transformers import SentenceTransformer
            device = get_device()
            self._encoders = (SentenceTransformer(IMAGE_MODEL, device=device),
                              SentenceTransformer(TEXT_MODEL, device=device), Dino(device))
        im, ph = load_image(io.BytesIO(image_bytes))
        return dict(img=encode(self._encoders[0], [im], batch_size=1)[0],
                    txt=encode(self._encoders[1], [title], batch_size=1)[0],
                    phash=ph, dino=self._encoders[2]([im])[0], colour=colour_hist(im))

    def match(self, q, title, exclude=None, top=12):
        """Score the listing against its k nearest catalog neighbours; returns matches with verdicts.

        exclude: a catalog row to leave out, to replay a catalog listing as if it were new.
        Rank features follow training: rank_ab = where the catalog item sits in the new
        listing's list, rank_ba = where the new listing would sit in the item's list
        (k + 1 if outside the top k).
        """
        q_img, q_txt = self._project(q["img"], "img"), self._project(q["txt"], "txt")
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
            "phash_dist": np.bitwise_count(self.raw["phash"][cand] ^ np.uint64(q["phash"])).astype(float),
            "num_match": num[:, 0], "num_conflict": num[:, 1],
            "mutual": (rank_ba <= k).astype(float),
            "rank_min": np.minimum(rank_ab, rank_ba).astype(float),
            "rank_max": np.maximum(rank_ab, rank_ba).astype(float),
            "len_ratio": np.minimum(q_len, self.lengths[cand]) / np.maximum(q_len, self.lengths[cand]),
            "dino_cos": self.raw["dino"][cand] @ q["dino"],
            "colour_dist": colour_dist(self.raw["colour"][cand], q["colour"]),
        })[self.cfg["features_v2"]]
        prob = self.model.predict_proba(X)[:, 1]
        combo = np.array([combo_mismatch(title, self.titles[j]) for j in cand])
        verdict = classify(prob, num[:, 1], X["colour_dist"].to_numpy(), combo, self.cfg)
        order = np.argsort(-prob)[:top]
        return [dict(row=int(cand[i]), prob=float(prob[i]), verdict=str(verdict[i])) for i in order]

    def explain(self, matches, price=None):
        """Verdict + price comparison in plain words. Typical price uses 'same' matches only."""
        same = [m for m in matches if m["verdict"] == "same"]
        variants = [m for m in matches if m["verdict"] == "variant"]
        p = self.catalog["price"].to_numpy()
        prices = p[[m["row"] for m in same]]
        # typical price = median of at least 2 OTHER listings of the same product (one price is too thin)
        typical = float(np.median(prices)) if len(prices) >= 2 else None
        out = dict(verdict="already_listed" if same else "variant_listed" if variants else "looks_new",
                   same=len(same), variants=len(variants), typical_price=typical,
                   min_price=float(prices.min()) if len(prices) else None,
                   max_price=float(prices.max()) if len(prices) else None,
                   cheapest_variant=float(p[[m["row"] for m in variants]].min()) if variants else None)
        if price and typical:
            out["your_price_vs_typical_pct"] = round(100 * (price / typical - 1), 1)
        rows = self.catalog
        out["matches"] = [dict(listing_id=rows.at[m["row"], "listing_id"], title=rows.at[m["row"], "title"],
                               price=float(rows.at[m["row"], "price"]), url=rows.at[m["row"], "url"],
                               image_url=rows.at[m["row"], "image_url"], prob=round(m["prob"], 4), verdict=m["verdict"],
                               pct_vs_typical=round(100 * (rows.at[m["row"], "price"] / typical - 1), 1) if typical else None)
                          for m in matches if m["verdict"] != "different"]
        return out

    def check_new(self, title, image_bytes=None, image_url=None, price=None):
        """Full check for a listing the model has never seen: photo (bytes or URL) + title."""
        if image_bytes is None:
            r = requests.get(image_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            image_bytes = r.content
        return self.explain(self.match(self.encode(title, image_bytes), title), price)

    def replay(self, row):
        """Leave-one-out check of catalog row `row`, using its cached signals."""
        q = {m: self.raw[m][row] for m in self.raw}
        return self.explain(self.match(q, self.titles[row], exclude=row), float(self.catalog.at[row, "price"]))


def build_demo_json():
    """Leave-one-out results for every catalog listing -> dashboard/data/catalog.json.

    The hosted (static) dashboard cannot run the models, so it shows these real
    model outputs: each listing checked as if it were new, against all the others.
    """
    m = Matcher()
    cat = m.catalog
    listings = [dict(id=r.listing_id, title=r.title, price=float(r.price), url=r.url, image=r.image_url)
                for r in cat.itertuples()]
    pos = {lid: i for i, lid in enumerate(cat["listing_id"])}
    results = []
    for i in range(len(cat)):
        r = m.replay(i)
        results.append(dict(verdict=r["verdict"], same=r["same"], variants=r["variants"], typical=r["typical_price"],
                            pct=r.get("your_price_vs_typical_pct"), cheapest_variant=r["cheapest_variant"],
                            matches=[[pos[x["listing_id"]], x["prob"], x["verdict"]] for x in r["matches"]]))
    out = config.ROOT / "dashboard" / "data" / "catalog.json"
    out.write_text(json.dumps(dict(category=config.INDIA_CATEGORY["amazon"], model="M1_v2",
                                   same_model_threshold=m.cfg["india_same_model_threshold"],
                                   colour_cut=m.cfg["colour_cut"], listings=listings, results=results)))
    v = [r["verdict"] for r in results]
    print("wrote", out, f"({out.stat().st_size // 1024} KB) | verdicts:", {x: v.count(x) for x in sorted(set(v))},
          "| with typical price:", sum(r["typical"] is not None for r in results),
          "| >20% above typical:", sum(r["pct"] is not None and r["pct"] > 20 for r in results))


if __name__ == "__main__":
    build_demo_json()
