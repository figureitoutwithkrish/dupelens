# DupeLens: handoff brief for a new assistant session

Paste this whole file into the new session, or tell it: "Read /Volumes/T7 Shield/dupelens/HANDOFF.md
and BLUEPRINT.md, then continue from section 7 'Current task'." Check section 9 (progress log) first.

## 0. Fresh machine setup (friend's laptop; repo already pushed)
```bash
git clone https://github.com/figureitoutwithkrish/dupelens.git && cd dupelens
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt   # pinned: saved models need sklearn 1.9.1
.venv/bin/python -m pytest tests -q                                       # 25 should pass
```
- Already in git (no rebuild needed): trained models (cache/models.joblib, models_v2.joblib), PCA, TF-IDF,
  configs, India catalog + its vectors (cache/india_*.npy), label keys, results/, dashboard/data/.
  cache/test_done.json is committed: NEVER run `python -m src.train --test` (the test split is used up).
- Run the live app: `.venv/bin/python -m src.india --photos` only downloads photos for the committed catalog
  from Amazon's image URLs (no Kaggle needed). Then `.venv/bin/python -m src.app` -> http://127.0.0.1:8000.
- Only to RETRAIN on Shopee (not needed to run the app): log in to Kaggle (`kaggle auth login`), accept the
  rules at kaggle.com/competitions/shopee-product-matching, then
  `kaggle competitions download -c shopee-product-matching -p data/ && unzip data/shopee-product-matching.zip -d data/`,
  `.venv/bin/python -m src.data` (rebuilds the identical split, seed 42), `-m src.embed` (~12 min on Apple GPU),
  `-m src.extra` (~8 min). Amazon India raw data (only to change category):
  `curl -L -o data/india_raw/amazon/amazon.zip https://www.kaggle.com/api/v1/datasets/download/lokeshparab/amazon-products-dataset`
  then unzip there. Changing INDIA_CATEGORY means re-running the India steps and new labels.
- Not on this machine: the exFAT `._*` problem, the Vercel login and the T7 SSD paths are specific to the
  original Mac. Redeploying the site needs the repo owner's Vercel login (`npx vercel login` then
  `cd dashboard && npx vercel deploy --prod`), or connect the repo in the Vercel dashboard.

## 1. What the project is
DupeLens checks a marketplace listing (title + photo, optional price) and answers: is the same product
already listed by other sellers, which listings are they (with photos and prices), and is the price fair
(vs the typical = median price of >= 2 other listings of the same product).
- Model trained/tested on Shopee Price Match Guarantee (Kaggle), applied to Amazon India
  "Personal Care Appliances" (Kaggle 2023 snapshot, 698 listings with live photos).
- Team: two FOML (Fundamentals of ML) students. They want a working, real-world, accurate project and
  do NOT want to do manual work: the assistant does everything (including labelling, always disclosed
  as "labelled by Claude, an AI assistant, not humans").
- Published: GitHub (public) https://github.com/figureitoutwithkrish/dupelens ;
  site https://dupelens.vercel.app (static demo mode).

## 2. Binding rules (from BLUEPRINT.md + the user)
1. Never invent numbers. Every number in README/dashboard/resume bullet must come from code that ran.
   (Earlier slip: "checks take ~2 s" was written before measuring; measured 0.13-0.48 s. Measure first.)
2. Stop and ask before changing scope; explain code briefly; keep it simple.
3. Shopee TEST split is USED UP (cache/test_done.json; `python -m src.train --test` refuses to rerun).
   New models: compare on Shopee VALIDATION and evaluate on a FRESH Indian held-out labelled set.
   Never re-tune on a set you report.
4. Price is never a matching feature (the price comparison would be circular).
5. The user approved publishing to this GitHub repo and Vercel project; still say what you push.

## 3. Environment (read before running anything)
- Repo: `/Volumes/T7 Shield/dupelens` on an exFAT SSD (Mac M4). venv `.venv` (Python 3.11, created with
  `--copies`). Always run `.venv/bin/python -m src.<module>` from the repo root.
- exFAT creates `._*` sidecar files next to every written file; they broke `transformers` once. After any
  write or pip install: `find . -name '._*' -delete` (inside `.venv` too after pip).
- torch and faiss in ONE process crash (two OpenMP runtimes) unless faiss is imported first. Only
  src/embed.py and src/service.py (lazily) load torch; src/service.py uses numpy search, no faiss.
- HF models cache on the internal disk (~/.cache/huggingface; symlinks do not work on exFAT).
- Kaggle CLI logged in (~/.kaggle/credentials.json). gh CLI logged in as the repo owner.
- Vercel CLI: `~/.local/vercel-cli/node_modules/.bin/vercel` (logged in). Redeploy the site:
  `cd dashboard && ~/.local/vercel-cli/node_modules/.bin/vercel deploy --prod --yes`
  (Vercel's GitHub integration is NOT connected; pushing does not redeploy.)
- Live app (checks NEW listings, runs models locally): `.venv/bin/python -m src.app` ->
  http://127.0.0.1:8000. Restart it after changing src/service.py.

## 4. Code map (src/)
config.py (paths, seed 42, K=50, INDIA_CATEGORY, INDIA_MATCH_THRESHOLD=0.99) · data.py (load, group split
70/15/15, load_cached) · metrics.py (per-listing F1, pairwise P/R, group bootstrap CI, fast pair F1) ·
numtokens.py (sizes/packs, conflict) · embed.py (CLIP ViT-B/32 512-d + multilingual MPNet 768-d + phash,
cached to cache/*.npy) · reduce.py (PCA per modality, whiten/shrink) · candidates.py (FAISS top-k, recall,
sweep) · features.py (10 pair features) · train.py (B0-B2, M1 logreg, M2 MLP, ablation; `--test` once) ·
group.py (components) · india.py (load category, download photos, frozen run, labelling sheets + photo
grids, stats) · service.py (Matcher: check one listing vs catalog; `python -m src.service` builds
dashboard/data/catalog.json = leave-one-out results for the static site) · app.py (FastAPI: / and
POST /api/check) · export.py (results.csv logger, plots, dashboard JSON). Tests: tests/ (25 pass).
Dashboard: dashboard/index.html (one file; dark navy design from the user's dashboard-skill.md:
bg #07111F, surfaces #0D1B2A/#12243A, accents #2589FF/#22D3EE). Live mode if /api/health answers.

## 5. Results so far (all measured; rows in results/results.csv)
- Split: train 24,065 listings / 7,710 groups; val 5,045 / 1,652; test 5,140 / 1,652.
- Final config (cache/final_config.json): shrink-whitened PCA, 32 dims per modality, k=50, M1 thr 0.82.
- Shopee TEST F1: B0 0.461, B1 0.547, B2 (TF-IDF) 0.794, ablation 0.833, M1 0.838 [0.825, 0.851],
  M2 0.843, components 0.816. Shopee VAL M1 F1 (shrink 32): 0.8429.
- India: jewellery failed first (precision 0.015; archived in cache/jewellery_run, results/jewellery_run).
  Personal Care Appliances: calibration 150 pairs (results/labelling_sheet.csv, key cache/labelling_key.csv)
  -> threshold 0.99; held-out 60 pairs (results/holdout_sheet.csv, key cache/holdout_key.csv):
  precision 0.50 [0.37, 0.63] under a STRICT rule (same model AND same colour/pack).
  Held-out false matches (results/labelling/holdout_error_types.json): 11 colour variants, 7 combos with
  different contents, 7 single vs multi-pack/colour set, 5 other. Counting colour variants as same: 41/60.
- Leave-one-out catalog: 417/698 already listed elsewhere, 336 with a typical price, 61 > 20% above.
- Live app check of an unseen listing (FIT & SHAPE straightener comb, Rs 390): already listed by 12,
  typical Rs 498 (-21.7%). Warm check time 0.13-0.48 s.

## 6. Audit findings (why India is weak)
1. Definition mismatch: Shopee groups often treat size/shade variants as the same product; India labels
   were strict. 2. CLIP knows "what kind of thing", not "which exact item"; 32 dims loses colour detail.
3. Rank/mutual features (top-50 of a diverse 5k gallery) say "match" too easily inside a narrow 698-item
   catalog (hypothesis, not tested). 4. No human labels; small India samples. 5. Small frozen 2023 catalog.

## 7. Current task (user approved): steps 1 and 2 (+ database research, done: section 8)
Step 2 (do first, it feeds step 1): better image signals, retrain.
- Add a per-image colour descriptor (HSV hue x saturation histogram, normalised) and pair feature
  `colour_dist` (histogram distance).
- Add an instance-level image embedding: DINOv2 small (`facebook/dinov2-small`, 384-d CLS token) via
  transformers, cached like CLIP (cache/{shopee,india}_dino.npy). Pair feature `dino_cos`.
- Retrain M1 with the new features on the SAME Shopee train candidates (shrink 32, k=50), tune threshold
  on VAL, compare val F1 vs old M1 (0.8429) with group bootstrap CI; log rows to results.csv with
  step "M1_v2" (+ ablation without each new feature). Do NOT run Shopee test again.
Step 1: three-way verdict "same / variant / different".
- same = same model AND same colour/pack; variant = same model but different colour, size, pack or combo;
  different = not the same model. In src/service.py: M1_v2 prob decides "same model"; colour_dist, a
  number-token conflict, or a combo/pack keyword mismatch ("combo", "set of", "pack of", "+") marks
  "variant". Choose cut-offs on the existing India calibration labels (150) only.
- Fresh India held-out: sample NEW pairs (not in labelling_sheet or holdout_sheet), label 3-way blind
  (titles + photo grids via india.make_grids), report precision of "same" and accuracy of "variant".
- Update app/dashboard (variant section: "same product in other colours/packs from Rs X"), README, push,
  redeploy.

## 8. New data sources (searched 2026-11-01; verify before use)
| Source | What it has | Use for DupeLens | Catch |
|---|---|---|---|
| Amazon Reviews 2023, McAuley Lab (huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) | item metadata per category: title, price (often missing), images, `parent_asin` (colour/size variants share one parent) | variant labels (same parent = variant) to train/test step 1; bigger catalog | Amazon US, USD prices; not cross-seller duplicates; check if variants appear as separate rows |
| WDC Products / Large-Scale Product Matching (Web Data Commons) | offers from many shops clustered as the same product | true cross-seller "same product" labels for a text matcher | text only (no images); English web shops, not India |
| Amazon Berkeley Objects (ABO), CC BY-NC 4.0 | ~147,700 listings, ~398k catalogue images, metadata | more images/catalog for pretraining image similarity | catalog only, no duplicate labels; non-commercial |
| Flipkart Affiliate API (affiliate.flipkart.com/api-docs) | official product feed + product lookup | fresh, legal Indian catalog with prices and photos | needs a Flipkart affiliate account (tracking ID + token); fields not verified |
| Amazon Product Advertising API | official Amazon product data | fresh Amazon India catalog | needs Amazon Associates approval (from memory, NOT verified in this search) |
| Kaggle Flipkart 2023-24 | none confirmed (only a smartphone dataset) | - | - |
| Apify / scraping services | live Flipkart/Amazon pages | - | terms-of-service risk; not recommended |
Recommendation: Amazon Reviews 2023 metadata (Beauty / personal care) for variant training data; Flipkart
Affiliate API for a fresh Indian catalog once the user has an affiliate account.

## 9. Progress log (update after every milestone)
- 2026-11-01: brief written; database research done (section 8). Next: Step 2.
- Step 2 code: src/extra.py (DINOv2 small + colour histogram -> cache/{shopee,india}_{dino,colour}.npy;
  `python -m src.extra`, ~8 min, log cache/extra.log) and src/train_v2.py (`python -m src.train_v2`:
  M1_v2 + ablations on Shopee VAL, paired diff vs M1; writes cache/models_v2.joblib,
  cache/final_config_v2.json, results/json/m1_v2_coefficients.json). If extra.npy files are missing,
  rerun src.extra first.
- Step 1 code: src/variant.py. Order: `python -m src.variant --calibrate` (scores all India pairs with
  M1_v2 -> cache/india_pairs_v2.csv; picks same-model threshold + colour cut on the 210 earlier labels)
  -> `--sheet` (fresh blind sheet results/v2_sheet.csv, 40 predicted-same + 20 predicted-variant, photo
  grids results/labelling/v2_*.jpg, hidden key cache/v2_key.csv) -> label column `label` with
  same / variant / different -> `--stats` (writes india_summary.json "v2"). Then wire classify() into
  src/service.py (Matcher must also compute dino + colour for new listings via src.extra) and the dashboard.
- DONE step 2: M1_v2 val F1 0.8643 [0.8549, 0.8737], paired +0.0215 [+0.0168, +0.0262] vs M1 (0.8429);
  no-DINO +0.0073, no-colour +0.0181. Coefs: tfidf 1.74, dino_cos 1.37, colour_dist -0.43.
- DONE step 1: calibration on 210 labels -> same-model threshold 0.90, colour cut 0.45 (cache/final_config_v2.json).
  Fresh blind sheet results/v2_sheet.csv (60): "same" correct 33/40 = 0.825 [0.70, 0.925] (same model 37/40);
  "variant" correct 11/20 = 0.55 [0.30, 0.75]. Stored in results/json/india_summary.json "v2".
- DONE wiring: src/verdict.py (rules, no faiss/torch), src/service.py now uses M1_v2 + verdicts; app restarted;
  catalog.json rebuilt (434 same / 88 variant / 176 new; 343 typical price; 62 > 20% above); dashboard shows
  verdict, "other colour/pack" pills, v2 accuracy (dashboard/data/model_v2.json from src.export). README updated.
- NEXT ideas (ask user): improve variant precision (55%); more Indian labels / human check of AI labels; grow
  catalog (Flipkart Affiliate API needs an account; Amazon Reviews 2023 for variant training data); host the
  live checker online (models ~1.6 GB: needs a Python host, e.g. Hugging Face Spaces).
