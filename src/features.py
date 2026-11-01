"""Pair features for candidate pairs. Every feature is symmetric in (a, b).

Price is never a feature: the India step compares prices within predicted
groups, so using price to form the groups would make that comparison circular.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config
from src.numtokens import compare, extract

TFIDF_PATH = config.CACHE_DIR / "tfidf.joblib"
FEATURES = ["img_cos", "txt_cos", "tfidf_cos", "phash_dist", "num_match", "num_conflict",
            "mutual", "rank_min", "rank_max", "len_ratio"]
NUM_FEATURES = ["num_match", "num_conflict"]


def fit_tfidf(train_titles):
    """Char 3-5-gram TF-IDF fitted on Shopee TRAIN titles only, then frozen (reused for India).

    Character n-grams within word boundaries ('char_wb') catch spelling variants,
    joined words and mixed Indonesian/English without a tokenizer. min_df=2 drops
    n-grams seen in only one title, which cannot help match two titles.
    """
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit(train_titles)
    joblib.dump(vec, TFIDF_PATH)
    return vec


def load_tfidf():
    return joblib.load(TFIDF_PATH)


def build(pairs, img, txt, tfidf_rows, phash, titles, k):
    """Feature table for candidate pairs (positions a, b within one split).

    img, txt: unit vectors of that split (from the chosen projection), so a row-wise
    dot product is the cosine. tfidf_rows: TF-IDF matrix of the split's titles
    (rows are L2-normalised by sklearn, so their dot product is also a cosine).
    """
    a, b = pairs["a"].to_numpy(), pairs["b"].to_numpy()
    tokens = [extract(t) for t in titles]
    num = np.array([compare(tokens[i], tokens[j]) for i, j in zip(a, b)], dtype=float).reshape(-1, 2)
    lengths = np.array([len(t) for t in titles], dtype=float)
    return pd.DataFrame({
        "img_cos": np.einsum("ij,ij->i", img[a], img[b]),
        "txt_cos": np.einsum("ij,ij->i", txt[a], txt[b]),
        "tfidf_cos": np.asarray(tfidf_rows[a].multiply(tfidf_rows[b]).sum(axis=1)).ravel(),
        "phash_dist": np.bitwise_count(phash[a] ^ phash[b]).astype(float),  # Hamming distance, 0..64
        "num_match": num[:, 0],
        "num_conflict": num[:, 1],
        "mutual": ((pairs["rank_ab"] <= k) & (pairs["rank_ba"] <= k)).to_numpy(dtype=float),
        "rank_min": np.minimum(pairs["rank_ab"], pairs["rank_ba"]).to_numpy(dtype=float),
        "rank_max": np.maximum(pairs["rank_ab"], pairs["rank_ba"]).to_numpy(dtype=float),
        "len_ratio": np.minimum(lengths[a], lengths[b]) / np.maximum(lengths[a], lengths[b]),
    })
