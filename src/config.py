"""Paths and fixed settings shared by every stage.

Kept free of heavy imports (no torch, no faiss) so any module can import it
cheaply. Importing torch and faiss in one process can crash on this Mac
(two OpenMP runtimes), so each stage imports only what it needs.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_IMAGES = DATA_DIR / "train_images"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
SPLIT_CSV = CACHE_DIR / "split.csv"  # saved once so every stage uses the same split

SEED = 42
SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}  # fractions of label_groups

DIMS = [32, 64, 128, 256, None]  # PCA dims per modality; None = full dim, no PCA
# FAISS neighbours per listing. Started at 20; the one-off validation check found
# k=50 adds 3.08 points of candidate recall (shrink, 256-d) and 4.45 (raw, full),
# both above the 2-point rule (blueprint 7.4), so k=50 is used everywhere.
K = 50
K_CHECK = 50

N_BOOT = 1000  # group bootstrap resamples for 95% CIs

# India price demo: the one category, named as each site names it. Changing these
# two values and re-running src/india.py is all it takes to switch category.
# A site set to None is skipped. Prices are only compared within a site, because
# the Amazon data is from 2023 and the Flipkart data from 2015-16.
# Jewellery (Amazon Fashion & Silver Jewellery + Flipkart Jewellery) was tried first: its titles
# are templated, blind labels found 1/150 true matches. results/phase1_search/duplicate_probe.py
# found Personal Care Appliances has by far the most true duplicate listings (599 pairs with
# near-identical photo and title under different ASINs), so it is used; Flipkart has no match.
INDIA_CATEGORY = {"amazon": "Personal Care Appliances", "flipkart": None}
# Out of domain, the Shopee threshold (0.82) over-matches. On the 150 labelled India
# calibration pairs, precision was 0.04 at prob <= 0.82 but 0.74 above 0.99, so India
# matches use 0.99. Precision is then measured on a SEPARATE held-out labelled sample.
INDIA_MATCH_THRESHOLD = 0.99
INDIA_RAW = DATA_DIR / "india_raw"
AMAZON_CSV = INDIA_RAW / "amazon" / "Amazon-Products.csv"
FLIPKART_CSV = INDIA_RAW / "flipkart" / "flipkart_com-ecommerce_sample.csv"
INDIA_IMAGES = DATA_DIR / "india_images"
INDIA_LISTINGS = CACHE_DIR / "india_listings.csv"  # cleaned listings with a downloaded image
