# DupeLens handoff brief (for a new assistant session)

Status: **project complete locally.** Not a git repo yet; nothing pushed to GitHub, nothing deployed to Vercel
(both need the user's explicit OK). README.md has every result; this file has the working context.

Rules (BLUEPRINT.md "Instructions for the assistant"): never invent numbers; stop and ask before scope changes;
explain code; keep it simple. User's latest wishes: the team does NO manual work (Claude does everything,
including labelling, disclosed as AI labels); goal is a working, accurate project; report only problems.

## Environment
- Repo `/Volumes/T7 Shield/dupelens` on an exFAT SSD. venv `.venv` (Python 3.11, `--copies`).
  Run `.venv/bin/python -m src.<module>` from repo root.
- exFAT writes `._*` sidecar files; after big writes `find . -name '._*' -delete` (also in .venv after pip).
- torch + faiss in one process crash unless faiss is imported first; only src/embed.py imports torch.
- Dashboard: `.venv/bin/python -m http.server 8765 --directory dashboard` -> http://localhost:8765 (static, reads dashboard/data/*.json).

## Final results (all in results/results.csv, results/json/)
- Split: train 24,065 / 7,710 groups; val 5,045 / 1,652; test 5,140 / 1,652.
- Config (cache/final_config.json): shrink-whitened PCA 32 dims/modality, k=50, M1 thr 0.82.
- Shopee TEST (used once; cache/test_done.json blocks reruns): B0 0.461, B1 0.547, B2 (tfidf) 0.794,
  ablation 0.833, M1 0.838 [0.825, 0.851], M2 0.843, components 0.816. Test cand recall 0.906.
- India: category = Amazon "Personal Care Appliances" (config.INDIA_CATEGORY; jewellery tried first,
  archived in cache/jewellery_run and results/jewellery_run, precision 0.015). 698 listings.
  Calibration (150 AI-labelled pairs, results/labelling_sheet.csv): weighted precision 0.39 at thr 0.82,
  0.74 above 0.99 -> config.INDIA_MATCH_THRESHOLD = 0.99. Held-out (60 new pairs, results/holdout_sheet.csv):
  precision 0.50 [0.37, 0.63]; errors: 11 colour variants, 7 combo contents, 7 set vs single, 5 other.
  Price view (results/json/india_matches.json): 417 listings matched, 339 with typical price, 52 > 20% above.
- 30 Shopee test errors tagged (results/json/error_sample.json).

## Possible next steps (ask the user first)
- `git init`, commit, push to GitHub; deploy `dashboard/` to Vercel as a static site.
- Optional improvement: a colour feature for colour variants (would need retraining; the test split is
  already used, so any new model needs a fresh evaluation plan agreed with the user).
