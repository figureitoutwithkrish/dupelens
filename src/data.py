"""Load Shopee train.csv, build the answer key, and split by product group."""
import numpy as np
import pandas as pd

from src import config


def load_train(path=config.TRAIN_CSV):
    """Read train.csv. Columns: posting_id, image, image_phash, title, label_group."""
    df = pd.read_csv(path)
    assert df["posting_id"].is_unique, "posting_id must identify one listing"
    return df


def build_ground_truth(df):
    """Map each posting_id to the set of posting_ids in its label_group.

    The set includes the listing itself, following the Kaggle F1 convention.
    Call this on one split at a time, because search only happens within a split.
    """
    groups = df.groupby("label_group")["posting_id"].agg(set)
    return dict(zip(df["posting_id"], df["label_group"].map(groups)))


def group_split(df, fracs=config.SPLIT_FRACS, seed=config.SEED):
    """Assign each label_group to train, val or test; returns df with a 'split' column.

    We split groups, not listings, so no product appears in two splits. A random
    listing split would put copies of one product in train and test, and the test
    score would reward memorising products instead of matching new ones.
    Fractions are of groups, so listing shares come out only roughly 70/15/15.
    """
    groups = np.sort(df["label_group"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_train = round(fracs["train"] * len(groups))
    n_val = round(fracs["val"] * len(groups))
    split_of = {g: "train" for g in groups[:n_train]}
    split_of.update({g: "val" for g in groups[n_train:n_train + n_val]})
    split_of.update({g: "test" for g in groups[n_train + n_val:]})
    return df.assign(split=df["label_group"].map(split_of))


def load_split(path=config.SPLIT_CSV):
    """Load train.csv with its split, creating and saving the split on first use."""
    df = load_train()
    if path.exists():
        saved = pd.read_csv(path)[["posting_id", "split"]]
        return df.merge(saved, on="posting_id", how="left", validate="one_to_one")
    df = group_split(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["posting_id", "label_group", "split"]].to_csv(path, index=False)
    return df


def load_cached(name, ids):
    """Load cached image/text vectors and phashes for `name`, checking row order matches `ids`.

    Lives here, not in embed.py, so loading never imports torch (see config.py).
    """
    cached_ids = pd.read_csv(config.CACHE_DIR / f"{name}_ids.csv")["id"]
    assert cached_ids.tolist() == list(ids), f"{name} cache rows do not match the table"
    return {m: np.load(config.CACHE_DIR / f"{name}_{m}.npy") for m in ("img", "txt", "phash")}


if __name__ == "__main__":
    # Print split sizes and check that no group leaks across splits.
    df = load_split()
    sizes = df.groupby("split").agg(listings=("posting_id", "size"), groups=("label_group", "nunique"))
    print(sizes.loc[["train", "val", "test"]].assign(listing_share=lambda s: (s.listings / s.listings.sum()).round(3)))
    assert df.groupby("label_group")["split"].nunique().max() == 1, "a group appears in two splits"
    print("total listings:", len(df), "| total groups:", df["label_group"].nunique(), "| no group in two splits")
