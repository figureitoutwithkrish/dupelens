"""Image and text embeddings plus perceptual hashes, computed once and cached.

Outputs in cache/, row-aligned with the source table (train.csv order for
Shopee, cache/india_listings.csv order for India):
  <name>_img.npy    CLIP ViT-B/32 image vectors, (n, 512), unit length
  <name>_txt.npy    multilingual MPNet title vectors, (n, 768), unit length
  <name>_phash.npy  imagehash.phash as uint64, (n,)
  <name>_ids.csv    row ids, to check alignment when loading
Run: python -m src.embed   (skips any output that already exists)
"""
import time
from concurrent.futures import ThreadPoolExecutor

import imagehash
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from src import config
from src.data import load_train

IMAGE_MODEL = "clip-ViT-B-32"
TEXT_MODEL = "paraphrase-multilingual-mpnet-base-v2"
CHUNK = 256  # images decoded per step; bounds memory, since Shopee photos are up to 1024 px


def get_device():
    """Apple GPU ('mps') when available, else CPU."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def encode(model, items, batch_size):
    """Encode with unit-length output; on an mps error, retry on CPU with a smaller batch.

    Unit length means a dot product equals cosine similarity, which is what
    FAISS IndexFlatIP computes later.
    """
    try:
        return model.encode(items, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)
    except RuntimeError as err:
        if model.device.type != "mps":
            raise
        print(f"mps failed ({err}); falling back to CPU", flush=True)
        model.to("cpu")
        return model.encode(items, batch_size=max(8, batch_size // 4), normalize_embeddings=True, convert_to_numpy=True)


def load_image(path):
    """Open one photo as RGB and compute its phash; returns (image, phash as int).

    draft() lets the JPEG decoder skip straight to a smaller size (at least
    448 px), which is much faster to read; CLIP resizes to 224 px and phash to
    32 px anyway. The same function runs on Shopee and Indian photos, so their
    hashes stay comparable.
    """
    im = Image.open(path)
    im.draft("RGB", (448, 448))
    im = im.convert("RGB")
    return im, int(str(imagehash.phash(im)), 16)


def embed_images(paths, model):
    """CLIP vectors and phashes for each path, decoding CHUNK images at a time."""
    vecs, hashes = [], []
    with ThreadPoolExecutor(8) as pool:  # decode images in parallel while the GPU encodes
        for start in range(0, len(paths), CHUNK):
            images, chunk_hashes = zip(*pool.map(load_image, paths[start:start + CHUNK]))
            vecs.append(encode(model, list(images), batch_size=64))
            hashes.extend(chunk_hashes)
            if (start // CHUNK) % 20 == 0:
                print(f"  images {start + len(images)}/{len(paths)}", flush=True)
    return np.vstack(vecs).astype(np.float32), np.array(hashes, dtype=np.uint64)


def run(name, ids, image_paths, titles, device):
    """Embed one dataset and save its four cache files; skips if already done."""
    out = {k: config.CACHE_DIR / f"{name}_{k}" for k in ["img.npy", "txt.npy", "phash.npy", "ids.csv"]}
    if all(p.exists() for p in out.values()):
        print(f"{name}: cached, skipping")
        return
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Embed each distinct photo once (Shopee sellers reuse photos), then map back to rows.
    unique_paths, row_to_unique = np.unique(np.asarray(image_paths, dtype=str), return_inverse=True)
    t0 = time.time()
    img_model = SentenceTransformer(IMAGE_MODEL, device=device)
    img, phash = embed_images(list(unique_paths), img_model)
    print(f"{name}: {len(unique_paths)} images in {time.time() - t0:.0f}s on {img_model.device}", flush=True)
    del img_model

    t0 = time.time()
    txt_model = SentenceTransformer(TEXT_MODEL, device=device)
    txt = encode(txt_model, list(titles), batch_size=128).astype(np.float32)
    print(f"{name}: {len(titles)} titles in {time.time() - t0:.0f}s on {txt_model.device}", flush=True)

    np.save(out["img.npy"], img[row_to_unique])
    np.save(out["txt.npy"], txt)
    np.save(out["phash.npy"], phash[row_to_unique])
    pd.DataFrame({"id": ids}).to_csv(out["ids.csv"], index=False)


if __name__ == "__main__":
    device = get_device()
    print("device:", device, flush=True)
    shopee = load_train()
    run("shopee", shopee["posting_id"], [str(config.TRAIN_IMAGES / f) for f in shopee["image"]],
        shopee["title"], device)
    india = pd.read_csv(config.INDIA_LISTINGS)
    run("india", india["listing_id"], [str(config.INDIA_IMAGES / f"{i}.jpg") for i in india["listing_id"]],
        india["title"], device)
