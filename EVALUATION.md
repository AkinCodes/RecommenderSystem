# Evaluation & Ablation Studies

Quantitative evaluation of the RecommenderSystem across model comparison, feature ablation, and serving architecture.

---

## Model Comparison

Head-to-head comparison of the DLRM against classical baselines on the MovieLens 100K dataset (943 users, 1,682 items, 100K ratings).

| Model | Approach | NDCG@10 | Precision@10 | Recall@10 | HitRate@10 | AUC |
|-------|----------|---------|--------------|-----------|------------|-----|
| DLRM | Learned user/item embeddings (128-dim) + 3-layer MLP | Pending | Pending | Pending | Pending | Pending |
| XGBoost | 200 trees, max_depth=6, lr=0.1, subsample=0.8 on 8 features | Pending | Pending | Pending | Pending | Pending |
| LightGBM | 200 trees, max_depth=6, lr=0.1, subsample=0.8 on 8 features | Pending | Pending | Pending | Pending | Pending |
| LogReg | StandardScaler + Logistic Regression (C=1.0, lbfgs) on 8 features | Pending | Pending | Pending | Pending | Pending |
| Most Popular | Rank by training-set popularity count (non-personalized) | Pending | Pending | Pending | Pending | Pending |
| Random | Uniformly random scores (lower bound) | Pending | Pending | Pending | Pending | Pending |

> Run `python scripts/retrain_and_compare.py` to reproduce.

**What to expect:** On MovieLens 100K, tree-based models (XGBoost, LightGBM) often match or beat simple neural approaches because the dataset is small enough that hand-crafted features capture most of the signal. The DLRM's advantage should appear in its ability to learn user-item interactions directly from embeddings, but at 100K ratings, that advantage may be marginal. If DLRM does not beat XGBoost here, that is an honest result — the value of learned embeddings shows more clearly at scale.

---

## Feature Ablation

How does the number of engineered features affect ranking quality? Each row uses the same DLRM architecture but trains on a different feature subset.

| Feature Set | Features Included | NDCG@10 | Precision@10 | HitRate@10 |
|-------------|-------------------|---------|--------------|------------|
| 2-feature (minimal) | `user_mean_rating`, `item_mean_rating` | Pending | Pending | Pending |
| 4-feature | Above + `user_rating_count`, `item_rating_count` | Pending | Pending | Pending |
| 8-feature (full) | All 8 features (see README for full list) | Pending | Pending | Pending |

> Requires retraining with feature subsets. Modify the feature list in `data/preprocessing.py` and re-run `scripts/retrain_and_compare.py`.

**What to look for:** The jump from 2 to 4 features should be significant (adding count features captures user/item activity level). The jump from 4 to 8 should be smaller — `user_rating_var`, `user_days_active`, `item_popularity_rank`, and `user_item_deviation` add nuance but may not move NDCG dramatically on a dataset this size.

---

## Serving Architecture

Comparison of brute-force scoring vs. two-stage retrieval (Faiss ANN + DLRM rerank).

| Strategy | Items Scored | Expected Latency (ms) | Notes |
|----------|-------------|----------------------|-------|
| Brute-force (`/recommend/{user_id}`) | All 1,682 | Pending | Single forward pass over entire item catalog |
| Two-stage: Faiss ANN + rerank (`/recommend_v2/{user_id}`) | 100 (from 1,682) | Pending | Faiss retrieves top-100 candidates (~1ms), DLRM reranks to top-K |

> Run both endpoints with timing enabled to measure. At 1,682 items the difference is small; the two-stage architecture is designed for catalogs of 100K+ items.

**Why both exist:** On MovieLens 100K (1,682 items), brute-force is fast enough that the two-stage pipeline adds complexity without meaningful latency savings. The two-stage endpoint exists to demonstrate the architecture that becomes necessary at scale. At 100K+ items, brute-force scoring becomes a bottleneck and ANN retrieval is essential.

### Cold-Start Behavior

Unknown users (user IDs not in the training set) receive popularity-ranked recommendations. This is a simple but honest fallback — no bandits, no content-based features for new users. The system returns results with a `"note": "cold-start"` flag so the client can surface this to the user.

---

## Limitations & Known Issues

- **MovieLens 100K is small.** 943 users, 1,682 items, 100K ratings. Results on this dataset do not directly generalize to production-scale systems with millions of users and items. The architecture is designed for scale, but it has not been validated at scale.
- **Cold-start is popularity-based only.** No exploration/exploitation trade-off (bandits), no content-based features for new users, no session-based recommendations. A new user gets the same popular movies as every other new user.
- **No online learning.** The model is trained offline and served statically. User ratings after deployment do not update the model. Retraining requires a manual pipeline run.
- **Drift detection is threshold-based only.** Feature distribution drift is detected by comparing current feature statistics against training-time baselines using fixed thresholds. There is no statistical test (KS, PSI), no automated alerting, and no automated retraining trigger.
- **BCELoss for ordinal data.** Ratings are binarized (>= 4 is positive, < 4 is negative), discarding the ordinal signal. A 5-star rating and a 4-star rating are treated identically. A ranking loss (BPR, margin loss) or regression loss (MSE on raw ratings) would preserve more information.
- **No diversity or fairness metrics.** Evaluation is purely accuracy-focused (NDCG, Precision, HitRate). There is no measurement of catalog coverage, intra-list diversity, novelty, or demographic fairness.
- **TMDB integration couples recommendations to external availability.** The `/predict` endpoint fetches movies from TMDB at request time. If TMDB is down, recommendations fail entirely (502). There is no caching layer.
- **Feature store is pickle-based.** The serving context (user stats, item stats, embeddings) is loaded from pickle files at startup. There is no feature store (Feast, Redis) for fresh features, and no versioning of feature artifacts.
- **Embedding tables dominate parameters.** ~93% of model parameters are in the user and item embedding tables. For a 100K-rating dataset, this is likely overparameterized. Smaller embedding dimensions or regularization may improve generalization.
- **No A/B testing infrastructure.** There is no mechanism to compare model versions in production or measure the impact of model changes on user behavior.
