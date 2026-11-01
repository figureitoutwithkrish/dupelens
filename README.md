# DupeLens

**Give it an Amazon listing (title + photo, optionally your price). It tells you whether the same product is already listed by other sellers, shows those listings with their photos and prices, and says whether the price is fair.**

A logistic-regression matcher over image, text and variant-aware features reaches **per-listing F1 0.838 (95% CI 0.825–0.851)** on a held-out set of Shopee product groups, against 0.461 for a self-only floor and 0.794 for the best single feature. An upgraded model (v2: adds DINOv2 exact-item photo matching and a colour check) raises Shopee validation F1 from 0.843 to 0.864 and, on fresh Amazon India pairs, is right **82.5% of the time (95% CI 70–92.5%)** when it says two listings are the same product, up from 50%. It also tells exact duplicates apart from colour/pack variants.

**Live dashboard: https://dupelens.vercel.app**

Every number below comes from code in this repo; the raw rows are in `results/results.csv`.

## Use it

- **Online demo:** https://dupelens.vercel.app — pick any of the 698 catalog listings; it is hidden from the catalog and checked as if new (real model output, precomputed).
- **Check a brand-new listing (runs the models on your Mac):**

```bash
cd "/Volumes/T7 Shield/dupelens" && .venv/bin/python -m src.app
```

Open http://127.0.0.1:8000, choose **Check a new listing**, paste the Amazon title, paste the photo address (right-click the product photo → Copy image address) or drop the photo file, optionally add your price, and press **Check listing**.

Example (a real Amazon listing that is *not* in the catalog): "FIT & SHAPE Hair Straightener Comb…", ₹390 → **Already listed by 12 other sellers**, same product sells for ₹395–₹1,449, typical ₹498, so ₹390 is 21.7% below typical. The first check takes ~10 s while the models load; later checks took 0.13–0.48 s (measured, photo download included).

The app works best for products in the trained category (personal-care appliances: hair straighteners, dryers, curlers, trimmers). To use another category, change `INDIA_CATEGORY` in `src/config.py` and re-run the India steps.

## The problem

On Amazon and Flipkart one product often appears as many separate listings from different sellers, with different titles, photos and prices (for example the same hair-straightener comb at ₹299, ₹399 and ₹499). Buyers cannot tell which listings are the same item, or whether a cheaper one is a different variant or pack size.

## How it works

1. **Data.** Shopee Price Match Guarantee (Kaggle): 34,250 listings in 11,014 product groups (`label_group` = same product). Split **by product group** 70/15/15 (seed 42): train 24,065 listings / 7,710 groups, validation 5,045 / 1,652, test 5,140 / 1,652. No product appears in two splits, so the test score measures matching *new* products.
2. **Embeddings.** CLIP ViT-B/32 image vectors (512-d) and multilingual MPNet title vectors (768-d), unit length; perceptual hash (`imagehash.phash`) recomputed for every photo.
3. **PCA.** One PCA per modality, fitted on train only, with shrinkage whitening (Λ + εI)^-1/2, ε = 10% of the mean eigenvalue, then re-normalised. Kept **32 dims per modality**, the smallest setting whose validation F1 fell inside the full-dimension confidence interval.
4. **Candidates.** FAISS exact inner-product search, top k = 50 neighbours within a split (k = 50 beat k = 20 by more than 2 recall points on validation).
5. **Pair features (symmetric):** image cosine, text cosine, TF-IDF char 3–5-gram cosine, phash Hamming distance, number-token match/conflict (e.g. "500 ml" vs "1 l"), mutual-neighbour flag, min/max rank, title length ratio. **Price is never a feature**, otherwise the price comparison would be circular.
6. **Model.** Logistic regression on standardised features with balanced class weights (positive share of train candidate pairs: 6.0%), probability threshold tuned on validation (0.82). **Test was used exactly once.**

## Results (Shopee test, single run)

| Step | Model | Test F1 | 95% CI | Pair precision | Pair recall |
|---|---|---|---|---|---|
| B0 | Self-only floor | 0.461 | 0.442–0.481 | – | 0.000 |
| B1 | Exact Shopee image hash | 0.547 | 0.527–0.569 | 0.994 | 0.107 |
| B2 | Best single feature (TF-IDF, tuned cutoff) | 0.794 | 0.780–0.807 | 0.782 | 0.694 |
| Ablation | M1 without number tokens | 0.833 | 0.821–0.846 | 0.827 | 0.708 |
| **M1** | **Logistic regression** | **0.838** | **0.825–0.851** | 0.833 | 0.717 |
| M2 | Small MLP (32-16) | 0.843 | 0.830–0.855 | 0.846 | 0.715 |
| M1 + components | Grouping rule (connected components, conflict edges dropped) | 0.816 | 0.802–0.830 | 0.760 | 0.817 |

F1 is per listing with the self-match counted (Kaggle convention); CIs bootstrap product groups (1,000 resamples). Candidate recall on test was 0.906 (CI 0.882–0.934), which caps pairwise recall. M2 is within M1's CI, so the simpler, interpretable M1 is the main model.

**Compression (validation).** Candidate recall with raw full-dimension vectors was 0.954; with shrinkage-whitened PCA it was 0.935 at 32 dims and 0.982 at 256 dims. Whitening raised recall over plain PCA at every size (e.g. 0.970 vs 0.940 at 64 dims), so compression with whitening is not just smaller but better at 64+ dims. M1 validation F1: raw 0.846, 32-d 0.843 (inside the raw CI 0.836–0.856).

**What the model weighs (standardised coefficients):** TF-IDF cosine +2.01, phash distance −0.54, number conflict −0.53, image cosine +0.42, text cosine −0.19. Text cosine is negative because it is strongly correlated with TF-IDF cosine, so the two split the weight; it does not mean similar meaning hurts.

**Where it fails.** Of 15 sampled false matches, 14 were variants (same brand or product line, different item); of 15 misses, 5 were variants (labelled as one product despite different size/shade), 3 garbled or abbreviated titles, 3 stock photo vs product photo, 2 bundles, 2 label noise.

![Recall vs dims](results/plots/recall_vs_dims.png) ![PR curve](results/plots/pr_curve_test.png) ![Explained variance](results/plots/explained_variance.png)

## Model v2: exact-item photos, colour, and a three-way verdict

The audit found that most wrong Indian matches were the same model in another colour or pack, and that CLIP recognises *what kind of thing* is in a photo better than *which exact item*. v2 adds two pair features and a three-way verdict:

- **dino_cos:** cosine of DINOv2-small image vectors (384-d), a model trained to recognise the same object across photos.
- **colour_dist:** 1 − histogram intersection of hue × saturation of the product pixels (near-white background ignored).
- **Verdict:** *same* (same model, same colour/size/pack) · *variant* (same model, other colour, size, pack or combo) · *different*. The model probability decides "same model" (≥ 0.90); colour distance > 0.45, a size/pack conflict, or a combo mismatch turns it into a variant. These two cut-offs were chosen on the 210 earlier Indian labels only.

| Shopee validation (same candidate pairs) | F1 | 95% CI | vs M1 (paired) |
|---|---|---|---|
| M1 (original) | 0.843 | 0.833–0.852 | – |
| **M1_v2** | **0.864** | 0.855–0.874 | +0.022 (+0.017 to +0.026) |
| v2 without DINOv2 | 0.850 | 0.841–0.860 | +0.007 |
| v2 without colour | 0.861 | 0.852–0.870 | +0.018 |

The Shopee test split was already used once, so v2 is compared on validation and checked on a **fresh** Indian held-out set (60 pairs never labelled before, blind): **"same product" verdicts were correct for 33 of 40 (82.5%, CI 70–92.5%)** and at least the same model for 37 of 40; "variant" verdicts were correct for 11 of 20 (55%, CI 30–75%; 5 were actually identical, 4 different products). With v2, 434 of 698 catalog listings are already listed as the same product by another seller, 88 only as a colour/pack variant, 176 look new; 343 have a typical price and 62 are priced more than 20% above it.

## India price demo (v1 model) (Amazon India, Personal Care Appliances, 2023 snapshot)

- **Category choice.** Jewellery was tried first; its titles are templated and only 1 of 150 sampled predicted matches was truly the same product. A duplicate probe (`results/phase1_search/duplicate_probe.py`) then found Personal Care Appliances has by far the most true duplicate listings (599 pairs with near-identical photo and title under different ASINs). 698 listings kept after dropping 372 with dead images.
- **Distribution shift.** At 32 dims the frozen Shopee PCA explains 0.503 of Indian image variance (Shopee test 0.479) and 0.570 of text variance (0.563): little shift in the embeddings, but the Shopee threshold over-matches inside one narrow category.
- **Calibration.** 150 blind-labelled pairs: stratum-weighted precision at the Shopee threshold was 0.39 (CI 0.27–0.50), rising to 0.74 above probability 0.99. The India match threshold is therefore 0.99.
- **Held-out precision.** On 60 *new* labelled matches above 0.99: **precision 0.50 (CI 0.37–0.63)** under a strict rule (same model *and* same colour and pack). Of the 30 false matches, 11 were the same model in another colour, 7 combo packs with different contents, 7 a single item vs a multi-colour/multi-pack set, 5 other.
- **Price output.** Each listing is checked against all the others (leave-one-out). 417 of 698 are already listed by at least one other seller; 336 have a typical price (median of at least 2 other listings of the same product); 61 are priced more than 20% above it and 61 at least 10% below. Prices use each listing's direct matches; connected components chained into one 100+ listing group on Indian data.

**Labelling disclosure:** the Indian match labels (150 calibration + 60 held-out pairs) and the 30 Shopee error tags were made by **Claude, an AI assistant**, blind to model scores, from titles and photos. No human labels were collected, so no inter-annotator agreement could be measured.

## Limits

- Shopee has no prices; the price comparison runs only on the Indian snapshot, which is a 2023 scrape, not live prices.
- India precision rests on 60 AI-labelled pairs; its CI is wide.
- Colour variants are the main India error: the image features at 32 dims do not separate shades of the same model well.
- Number tokens miss karat ("22 K") and model codes; the brand "3M" is read as 3 metres.

## Reproduce

```bash
python3.11 -m venv --copies .venv && .venv/bin/pip install torch sentence-transformers faiss-cpu scikit-learn pandas numpy matplotlib imagehash pillow requests kaggle six pytest fastapi uvicorn python-multipart
kaggle competitions download -c shopee-product-matching -p data/ && unzip data/shopee-product-matching.zip -d data/
.venv/bin/python -m pytest tests            # 25 unit tests
.venv/bin/python -m src.data                # group split
.venv/bin/python -m src.india               # Indian listings + photos (category in src/config.py)
.venv/bin/python -m src.embed               # embeddings + phash (Apple GPU 'mps' if available)
.venv/bin/python -m src.candidates          # PCA sweep, candidate recall
.venv/bin/python -m src.train               # baselines, M1, M2, ablation (validation)
.venv/bin/python -m src.train --test        # the single test run (refuses a second run)
.venv/bin/python -m src.india --frozen      # India calibration sample
.venv/bin/python -m src.india --matches     # India price view + held-out sample
.venv/bin/python -m src.export              # plots + dashboard data
.venv/bin/python -m src.extra               # v2: DINOv2 + colour features
.venv/bin/python -m src.train_v2            # v2 model, compared on validation
.venv/bin/python -m src.variant --calibrate  # v2 verdict cut-offs on earlier labels (then --sheet, --stats)
.venv/bin/python -m src.service             # demo results for every catalog listing
.venv/bin/python -m src.app                 # live app at http://127.0.0.1:8000
.venv/bin/python -m http.server 8765 --directory dashboard   # open http://localhost:8765 (live: https://dupelens.vercel.app)
```

`notebooks/dupelens.ipynb` runs the same steps and shows the results. On an exFAT drive, delete macOS `._*` sidecar files after big writes (`find . -name '._*' -delete`). Do not import torch and faiss in one process unless faiss is imported first.

## Future work (not in this build)

- More Indian categories (one setting in `src/config.py`, then re-run the India steps).
- A colour feature to separate colour variants; karat and model-code tokens.
- Human labels with a second annotator for agreement.
- Live scraping, CLIP fine-tuning or ArcFace, approximate search (IVF/HNSW), price prediction, listing photos on the dashboard.

## Resume bullet

Built DupeLens, a duplicate-listing classifier trained on 34k Shopee listings (product-group split) using image, text, and variant-aware features; logistic regression reached test F1 0.838 (95% CI 0.825–0.851) vs 0.794 for the best single-feature baseline, with number-token features adding 0.45 points. Compressed image and text embeddings from 512/768 to 32 dimensions each with PCA and shrinkage whitening, keeping validation F1 at 0.843 vs 0.846 at full dimension. Added DINOv2 and colour features (validation F1 0.843 → 0.864) and a same/variant/different verdict that is 82.5% precise on fresh blind-labelled Amazon India pairs (up from 50%), flagging 62 of 698 listings priced more than 20% above the typical price for the same product; shipped as an app that checks any new listing (title + photo) in under half a second once loaded.
