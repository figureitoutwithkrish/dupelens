"""Two extra image signals (step 2), cached like the CLIP vectors.

- DINOv2 small (384-d CLS vector, unit length): trained to recognise the SAME
  object across photos (self-supervised on images only), unlike CLIP, which is
  trained on captions and mostly knows "what kind of thing" it is.
- Colour histogram: hue x saturation of the product's pixels, ignoring the near-
  white background most product photos use, so it describes the product's colour.
Outputs: cache/<name>_dino.npy (n, 384) and cache/<name>_colour.npy (n, 32).
Run: python -m src.extra   (skips outputs that exist)
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from src import config
from src.data import load_train
from src.embed import get_device, load_image

DINO = "facebook/dinov2-small"
HUE_BINS, SAT_BINS = 8, 4


def colour_hist(im):
    """32-bin hue x saturation histogram of non-background pixels, summing to 1.

    Pixels that are near-white (bright, unsaturated) are treated as background.
    Very dark or grey pixels have no reliable hue, so they fall in the lowest
    saturation bins. All-background images return a uniform histogram.
    """
    hsv = np.asarray(im.convert("RGB").resize((64, 64)).convert("HSV"), dtype=np.float32) / 255.0
    h, s, v = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()
    keep = ~((v > 0.9) & (s < 0.12))
    if keep.sum() < 20:
        return np.full(HUE_BINS * SAT_BINS, 1 / (HUE_BINS * SAT_BINS), dtype=np.float32)
    hist, _, _ = np.histogram2d(h[keep], s[keep], bins=[HUE_BINS, SAT_BINS], range=[[0, 1], [0, 1]])
    return (hist.ravel() / hist.sum()).astype(np.float32)


def colour_dist(a, b):
    """1 - histogram intersection: 0 = same colour mix, 1 = no overlap."""
    return 1.0 - np.minimum(a, b).sum(axis=-1)


class Dino:
    def __init__(self, device=None):
        self.device = device or get_device()
        self.proc = AutoImageProcessor.from_pretrained(DINO)
        self.model = AutoModel.from_pretrained(DINO).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, images):
        x = self.proc(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        cls = self.model(pixel_values=x).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls, dim=-1).cpu().numpy().astype(np.float32)


def run(name, image_paths, chunk=128):
    out_d, out_c = config.CACHE_DIR / f"{name}_dino.npy", config.CACHE_DIR / f"{name}_colour.npy"
    if out_d.exists() and out_c.exists():
        print(f"{name}: cached, skipping")
        return
    unique, inv = np.unique(np.asarray(image_paths, dtype=str), return_inverse=True)
    dino, vecs, hists, t0 = Dino(), [], [], time.time()
    with ThreadPoolExecutor(8) as pool:
        for start in range(0, len(unique), chunk):
            ims = [im for im, _ in pool.map(load_image, unique[start:start + chunk])]
            vecs.append(dino(ims))
            hists.extend(pool.map(colour_hist, ims))
            if (start // chunk) % 40 == 0:
                print(f"  {name} {start + len(ims)}/{len(unique)}", flush=True)
    np.save(out_d, np.vstack(vecs)[inv])
    np.save(out_c, np.vstack(hists)[inv])
    print(f"{name}: {len(unique)} images in {time.time() - t0:.0f}s on {dino.device}", flush=True)


if __name__ == "__main__":
    shopee = load_train()
    run("shopee", [str(config.TRAIN_IMAGES / f) for f in shopee["image"]])
    india = pd.read_csv(config.INDIA_LISTINGS)
    run("india", [str(config.INDIA_IMAGES / f"{i}.jpg") for i in india["listing_id"]])
