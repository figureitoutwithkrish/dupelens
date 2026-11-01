# DupeLens: Final Build Blueprint v2 (7 hours, 2 people, MacBook M4)

## Instructions for the assistant reading this

You are helping a two-person student team build this project for a Fundamentals of Machine Learning (FOML) course in 7 hours. Follow this blueprint. Rules:
1. Never invent results, dataset properties, or scores. Every number in the README, dashboard, or resume bullet must come from code we actually ran.
2. If a dataset, library, or step fails or differs from this plan, stop and tell us before changing scope.
3. Explain each piece of code briefly as you write it. We must be able to defend every decision in a viva or interview.
4. Keep it simple. Do not add features, models, or tools that are not in this plan.
5. You have web search. Use it for the "Search tasks" section first and report findings before we write India-specific code.

## 1. Project in one line

A trained classifier that decides whether two marketplace listings are the same product (and catches near-duplicates that are different variants), used to group Indian listings and show each listing's price against the typical (median) price for that product.

Use the phrase "typical price", never "accurate price".

## 2. Problem

On Amazon and Flipkart, near-identical listings from different sellers appear at different prices (for example ₹300, ₹400, ₹500). Buyers cannot tell which listings are the same product, or whether a cheaper one is actually a different variant or pack size. DupeLens identifies same-product listings and reports each listing's price relative to the typical price.

## 3. Data

**Training and evaluation:** Shopee Price Match Guarantee (Kaggle competition `shopee-product-matching`), train.csv plus train_images.
- About 34k listings, about 11k product groups.
- Columns: posting_id, image, image_phash, title, label_group.
- Two listings match if they share label_group.
- Shopee has no prices, so it is used only for matching.

**Price demo:** one generic category (phone cases, kitchen tools, or similar; NOT branded electronics, where sellers share one product page) from an Indian Amazon or Flipkart Kaggle dataset. Needs title, selling price, and image URL or file.
- Drop listings whose image cannot be downloaded. Do not impute image features. Report how many were dropped.
- Risk: a scraped snapshot may contain few true duplicates in one category. In the go/no-go check, count title pairs with TF-IDF cosine above 0.6 as a rough sign that duplicates exist. If there are fewer than about 30 such pairs, use the fallback.

**Fallback if no usable Indian dataset by hour 0:45:** manually collect about 100 listings for about 20 products in one generic category from Amazon and Flipkart (title, price, image URL, seller, date). State clearly that it is a small, manually collected sample.

## 4. Search tasks (do these first, report back by 0:45)

1. Indian dataset. Search Kaggle for "amazon india products dataset", "flipkart products dataset image", "meesho products dataset", "indian ecommerce listings". Requirements: title, selling price (not only MRP), image URL or file, at least 1,000 rows in one generic category. Report: link, columns, row counts per category, scrape year, licence, 20 sample rows, share of live image URLs in a 50-row sample. Recommend one category.
2. Shopee rules. Confirm the data can still be downloaded and what the rules say about use outside the competition.
3. Score context. Find public Shopee notebooks reporting F1 for TF-IDF and image-embedding approaches, for context only (our gallery size differs, so not directly comparable).
4. Install check. Confirm faiss-cpu, sentence-transformers, and imagehash install on Apple Silicon.

## 5. Setup

```
pip install torch sentence-transformers faiss-cpu scikit-learn pandas numpy matplotlib imagehash pillow requests kaggle
kaggle competitions download -c shopee-product-matching -p data/
unzip data/shopee-product-matching.zip -d data/
```
Accept the Shopee competition rules in the browser first. Use PyTorch device "mps"; fall back to CPU with smaller batches if a library errors.

## 6. Repo structure

```
dupelens/
├── data/            # gitignored
├── cache/           # gitignored, *.npy embeddings, phash cache
├── src/
│   ├── config.py      # paths, seed 42, dims to sweep, k=20
│   ├── data.py        # load, ground truth, group split 70/15/15
│   ├── metrics.py     # per-listing F1, pairwise P/R, group bootstrap CI
│   ├── numtokens.py   # number-token extraction, unit normalisation, conflict check
│   ├── embed.py       # CLIP image + multilingual text, cached; imagehash phash
│   ├── reduce.py      # PCA, whitening, shrinkage per modality (fit on train only)
│   ├── candidates.py  # FAISS top-k neighbours, candidate recall
│   ├── features.py    # pair features (symmetric)
│   ├── train.py       # baselines, logistic regression, MLP, threshold
│   ├── group.py       # connected components + conflict rule (used on Shopee and India)
│   ├── india.py       # frozen model on Indian data, groups, price stats, labelling sheet
│   └── export.py      # results.csv, JSON, plots
├── tests/             # test_metrics.py, test_numtokens.py
├── notebooks/dupelens.ipynb
├── results/           # results.csv, plots/, json/
├── dashboard/         # static HTML + JS reading JSON
└── README.md
```

## 7. Method

### 7.1 Split
Split by label_group, 70/15/15, seed 42. No product group appears in two splits. All search happens within a split.

### 7.2 Embeddings and hashes
- Image: sentence-transformers `clip-ViT-B-32` (512-d).
- Text: `paraphrase-multilingual-mpnet-base-v2` (768-d), because Shopee titles mix Indonesian and English.
- L2-normalise each. Cache to .npy. Run once in the background early.
- Perceptual hash: recompute phash with `imagehash.phash` for ALL Shopee images and all Indian images. Do not use Shopee's provided image_phash column as a feature, because it may come from a different algorithm than imagehash, and the Hamming distances would not be comparable on Indian data. (Keep the provided column only for baseline B1.)

### 7.3 PCA (integrated, measured, not decorative)
- Fit PCA separately per modality on Shopee TRAIN only: one for image (512-d), one for text (768-d). Centre, do not standardise.
- Reason: a PCA fitted on the concatenated vector mixes image and text in every component, so image-only and text-only cosines cannot be read off it.
- Retrieval vector = concatenation of the two reduced, re-normalised vectors (equal weight per modality).
- Variants: plain PCA; whitening z = Λ^(-1/2) Uᵀ(x − μ); shrinkage whitening (Λ + εI)^(-1/2) with ε = 10% of the mean eigenvalue over all components of that modality.
- Always re-normalise to unit length after projecting.
- Sweep dims per modality: 32, 64, 128, 256, full (no PCA).
- Step 1 (hour 2): choose candidates for the sweep by VALIDATION candidate recall. Step 2 (hour 3): confirm the top 2 or 3 settings with validation classifier F1. Not by explained variance. Pick the smallest dim whose validation result is within the full-dim confidence interval (group bootstrap).
- Image-only and text-only cosine features come from the chosen per-modality projections. Compare classifier F1 using raw vs PCA cosine features on the same candidate pairs. Never put raw and PCA versions of the same cosine in one model (collinear).
- India shift check: apply the frozen Shopee PCA to Indian embeddings and compare explained variance with Shopee test, per modality. A large drop indicates distribution shift.
- Do NOT apply PCA to the pair features. Do NOT headline memory savings (index size is just rows × dims × 4 bytes).
- If PCA does not help, report that honestly.

### 7.4 Candidate pairs
- For each listing, top k = 20 neighbours within its split using FAISS IndexFlatIP on the retrieval vector (excluding itself).
- Check once on validation: candidate recall at k = 50. Shopee groups go up to about 50 listings, so k = 20 caps recall for big groups. If k = 50 gains more than 2 points, use 50 everywhere.
- Deduplicate unordered pairs (a, b) with a < b. Label = same label_group.
- Report candidate recall: the share of true matching pairs that appear in the candidate set, with group bootstrap CI. This caps the classifier.
- Note the class imbalance (positive share of pairs) in the README.

### 7.5 Pair features (all symmetric in a and b)

| Feature | Captures |
|---|---|
| Image cosine (CLIP, chosen projection) | Same photo or look |
| Text cosine (multilingual, chosen projection) | Same meaning |
| TF-IDF char 3 to 5-gram cosine (fit on Shopee train titles, frozen for India) | Same wording |
| phash Hamming distance (imagehash, both datasets) | Near-identical photo |
| Number-token match and conflict flags | Variants and pack sizes |
| Mutual neighbour flag; min and max of the two ranks (rank = k + 1 if absent) | Mutual similarity |
| Title length ratio = shorter / longer | Bundles and accessories |

Number tokens: regex for a number plus a unit (gb, tb, mb, ml, l, ltr, litre, g, gm, kg, mm, cm, m, inch, in, ", w, mah, pcs, pc, pieces) and phrases like "pack of N", "set of N", "N pcs", and Indonesian "isi N" (Shopee titles use it for pack contents).
- Normalise units before comparing: ml/l/ltr/litre to ml; g/gm/kg to g; mb/gb/tb to gb; mm/cm/m/inch to mm; pcs/pieces/pack of/set of/isi to count.
- Handle decimals and commas ("1.5l", "1,5 kg").
- Match = both titles share a normalised token with the same value. Conflict = both titles have a token of the same unit type and no value of that type is shared.
- Write unit tests with real examples, including "1 l" vs "1000 ml" (no conflict) and "500 ml" vs "1 l" (conflict).

RULE: price is never a feature. Otherwise the price comparison is circular.

### 7.6 Models (same split, same candidate pairs)

| Step | Model | Purpose |
|---|---|---|
| B0 | Self-only | Floor. Shopee F1 counts the self-match, so for a group of size n this scores 2/(n+1) |
| B1 | Exact image_phash match (Shopee's column, whole split, not limited to candidates) | Trivial rule |
| B2 | Best single feature with tuned cutoff | Is one score enough? |
| M1 | Logistic regression, StandardScaler fit on train, class_weight="balanced" | Main model, interpretable |
| M2 | Small MLP (sklearn MLPClassifier); balance by randomly undersampling train negatives, since MLPClassifier has no class_weight | Optional, ties to forward propagation |
| Ablation | M1 without number-token features | Does the variant guard help? |

Report M1 coefficients and interpret them. Coefficients are on standardised features, so they are comparable in size, but correlated features (image cosine and phash distance) can split weight between them, so say so.

### 7.7 Evaluation
- Tune the probability threshold on validation. Use test exactly once with the final config.
- Predictions are symmetric: if pair (a, b) is above threshold, b is in a's predicted set and a is in b's.
- Per-listing F1 (Kaggle convention): predicted set = self + matched listings; true set includes self; average over listings.
- Also report per-listing F1 of the India grouping procedure (connected components plus conflict rule, section 8) on Shopee test, from the same single test run. This validates the exact procedure used on Indian data. Fix the component rule on validation first.
- Pairwise precision and recall, excluding self.
- Precision-recall curve on test.
- 95% CI by bootstrapping product GROUPS (1,000 resamples), not listings. Keep predictions fixed; resample groups and re-average.
- Error analysis: tag about 30 test errors as variant, stock photo, bundle, language, or label noise.

### 7.8 results.csv (one row per experiment)
Columns: step, model, features, pca_variant, dim, k, split, f1, f1_ci_low, f1_ci_high, pair_precision, pair_recall, cand_recall, threshold, notes

## 8. India price step (frozen model, no retraining)
1. Embed Indian listings, compute imagehash phash, apply the frozen PCA and TF-IDF, generate top-k candidates, score with M1.
2. Groups: connected components on edges above threshold, dropping any edge with a number-token conflict (same rule as validated on Shopee). Flag any group larger than 15 for manual review (chaining).
3. Per group with at least 3 listings: listing count, min, median, max price; each listing's % above or below the median; flag listings more than 20% above the median. Groups of 2 show both prices without a "typical price".
4. Blind hand-labelling of 150 unique pairs (sheet shows titles and image links, no scores, shuffled):
   - Stratum H: 50 pairs sampled from predicted matches with probability ≥ 0.9
   - Stratum B: 50 pairs sampled from predicted matches with threshold ≤ probability < 0.9
   - Stratum R: 50 rejected pairs (below threshold) with TF-IDF cosine ≥ 0.5
   - H and B together cover every predicted match. If 0.9 is below the threshold, set the cut at the midpoint between the threshold and 1.
   - Each person labels 85 pairs: 65 of their own plus 20 shared pairs (150 unique, 170 labels). Use the 20 shared pairs for agreement (Cohen's kappa). Settle disagreements by discussion and record the final label.
5. Report:
   - Precision on predicted matches = (N_H × prec_H + N_B × prec_B) / (N_H + N_B), where N is the number of predicted matches in each stratum, with a bootstrap CI.
   - Share of stratum R that were true matches (a miss-rate proxy, not true recall).
   - Kappa on 20 pairs, noting the CI is wide with so few pairs.
6. If two different Kaggle datasets come from different years, compare prices within a site only.

## 9. Schedule

| Time | Person A (Shopee model) | Person B (India + dashboard) |
|---|---|---|
| 0:00 to 0:45 | Env, Shopee download, data.py split, metrics.py + tests | Search tasks 1 to 4; go/no-go on Indian data |
| 0:45 to 1:15 | Start embeddings + imagehash in background; TF-IDF | Download category, download images, drop dead links, clean prices |
| 1:15 to 2:15 | PCA variants + dim sweep by candidate recall; k check; B0 to B2 | numtokens.py + tests |
| 2:15 to 3:15 | Pair features; train M1 for top 2 or 3 PCA settings (confirm choice), raw vs PCA cosines, M2, ablation; tune threshold; fix component rule on validation | Embed Indian listings; dashboard skeleton |
| 3:15 to 3:45 | Single test run (pairwise + component F1), bootstrap CIs, PR curve | Run frozen model on India; groups; price stats |
| 3:45 to 4:30 | Error analysis; India PCA shift check | Build blind labelling sheet from India scores |
| 4:30 to 5:15 | Both label 85 pairs blind | Same |
| 5:15 to 6:15 | export.py: results.csv, JSON, plots | Labelling stats; wire dashboard, deploy to Vercel |
| 6:15 to 7:00 | README + FOML report | Resume bullet, checks, push repo |

If more than 20 minutes behind: cut M2 first, then error analysis to 15 errors, then the dashboard to the price lookup only.

## 10. Dashboard (one static page, plain HTML + JS, Vercel)
- Price lookup: search an Indian product, see grouped listings, typical price, each listing's % vs typical, flags.
- Model evidence: results ladder (B0 to M2), PR curve, PCA recall vs dims plot, ablation result, 5 real error examples.
- Footer: "Model trained and tested on Shopee Kaggle data. Prices are Kaggle snapshots [or manually collected on DATE], not live. No listing photos shown."
- Data files: ladder.json, pca.json, india_groups.json, errors.json.

## 11. Not in scope (list as future work)
Live scraping, CLIP fine-tuning or ArcFace, FAISS speed benchmarks, approximate search, price prediction, multi-page dashboard, listing photos on the dashboard.

## 12. Done means
- results.csv has every ladder row with CIs, plus the PCA sweep.
- Test split used exactly once.
- Ablation reported honestly.
- Three plots: explained variance, candidate recall vs dims (3 PCA variants), PR curve.
- Component-grouping F1 on Shopee test reported next to pairwise F1.
- 150 blind-labelled Indian pairs with weighted precision, miss-rate proxy, and kappa.
- Dashboard live, README written (headline, ladder table, plots, limits, how to reproduce), repo public.

## 13. Must be able to explain without notes
Group split vs random split; logistic regression loss, standardisation, coefficients, class weights; threshold tuning and single test use; why price is not a feature; candidate recall and why k caps it; group bootstrap; PCA fit on train only and per modality; choosing dims by validation not explained variance; what whitening does and why shrinkage; why re-normalise; why phash is recomputed; what the India shift check means; why rejected pairs are labelled; how stratum-weighted precision works.

## 14. Resume bullet (fill only with measured numbers)
Built DupeLens, a duplicate-listing classifier trained on 34k Shopee listings (product-group split) using image, text, and variant-aware features; logistic regression reached test F1 [b] (95% CI [..]) vs [a] for the best single-feature baseline, with number-token features adding [x] points. Compressed image and text embeddings from 512/768 to [k] dimensions each with PCA and whitening, keeping candidate recall at [r1] vs [r0] at full dimension. Applied unchanged to [n] Indian [category] listings at [p]% precision on blind-labelled pairs, flagging listings priced [m]% above the typical price for the same product.
