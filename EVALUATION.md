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
| Two-stage + bandit (`/recommend_v3/{user_id}`) | 100 (from 1,682) | Pending | v2 retrieval, reranked by the common-scale blend `(1-w)·norm(DLRM) + w·norm(bonus)` (default `w=0.2`); adds a sub-millisecond bandit stage |

> Run both endpoints with timing enabled to measure. At 1,682 items the difference is small; the two-stage architecture is designed for catalogs of 100K+ items.

**Why both exist:** On MovieLens 100K (1,682 items), brute-force is fast enough that the two-stage pipeline adds complexity without meaningful latency savings. The two-stage endpoint exists to demonstrate the architecture that becomes necessary at scale. At 100K+ items, brute-force scoring becomes a bottleneck and ANN retrieval is essential.

### Cold-Start Behavior

Unknown users (user IDs not in the training set) receive popularity-based recommendations, returned with a `"note": "cold-start"` flag so the client can surface this to the user. `/recommend` and `/recommend_v2` return the **static** popularity list (identical for every new user). `/recommend_v3` instead reranks the popularity candidates with the same common-scale blend used for warm users — popularity as the exploit base, a LinUCB exploration bonus over a **mean-user-embedding context**, both min-max normalized and convex-combined by `w` (default `0.2`). Because the raw popularity counts (~hundreds) and the raw bonus (~0.1) are on wildly different scales, a plain sum would leave the order identical to the static list; normalizing both to `[0, 1]` before combining is what actually perturbs the ranking (verified by test: the served cold-start order differs from the raw popularity list). See [Exploration & Online Learning](#exploration--online-learning).

### Exploration & Online Learning

`/recommend_v3` adds a **LinUCB contextual bandit** (`models/bandit.py`) on top of the two-stage pipeline. The context `x` is the elementwise `user × item` embedding product (the same interaction signal the DLRM uses), and the bonus is `alpha·√(xᵀ A⁻¹ x)`.

**Common-scale combination (the core fix).** The DLRM exploit score is a **sigmoid** and the LinUCB bonus is an **unbounded quadratic form** — on this candidate set the DLRM scores span roughly `[0, 1]` while the bonus has mean ~3.3. A raw `dlrm_score + bonus` sum does **not** blend these; it ranks by whichever term is numerically larger (the bonus), so *every* warm item was flagged exploratory and the ranking was effectively pure exploration. The fix (`combine_explore`) min-max normalizes **both** the exploit scores and the bonuses across the candidate set to `[0, 1]` and convex-combines them:

```
final = (1 - w) · norm(dlrm_score) + w · norm(bonus)
```

`w` (the explore weight, default **`0.2`**) is the single interpretable knob: `w = 0` is pure exploit (identical to DLRM order), `w = 1` is pure exploration. `alpha` still shapes *which* contexts look uncertain, but it no longer competes with the exploit term on an accidental scale. `POST /api/v1/feedback` applies rank-1 online updates from observed rewards (under a lock, atomic persist) so post-deployment feedback moves the policy.

| Property | Value |
|----------|-------|
| Policy | LinUCB (shared-weight, single linear model over the interaction context) |
| Combination | `combine_explore`: `(1-w)·norm(exploit) + w·norm(bonus)`, both min-max normalized per candidate set |
| Explore weight `w` | Default `0.2` (chosen from the frontier sweep below) |
| Baseline / control | Epsilon-greedy (`EpsilonGreedyBandit`) |
| Context | `user_emb × item_emb` (elementwise), dim = 64 |
| Update | Sherman-Morrison rank-1 (no full re-inversion) |
| Offline evaluation | `scripts/offline_bandit_eval.py` — explore-weight coverage/reward frontier; exploit = **real served DLRM sigmoid** |

**Explore-weight frontier (replay on the 20K test-split events).** The exploit signal is the real served DLRM sigmoid; the bandit ranks by the exact `combine_explore` blend served, learning `A` online over retained events. **SNIPS is the estimator to trust** (see propensity note below); coverage = distinct items shown.

| Policy | SNIPS reward | Distinct items (coverage) | Mean pop-rank |
|--------|-------------|---------------------------|---------------|
| DLRM (pure exploit, `w=0`) | **0.731** | 107 | 0.847 |
| Bandit `w=0.1` | 0.700 | 164 | 0.894 |
| Bandit `w=0.2` (**served default**) | 0.696 | 180 | 0.905 |
| Bandit `w=0.3` | 0.663 | 190 | 0.909 |
| Bandit `w=0.5` | 0.654 | 195 | 0.914 |
| Bandit `w=1.0` (pure explore) | 0.585 | 196 | 0.917 |
| Most-Popular (reference) | 0.754 | 58 | 0.997 |

**Owning the tradeoff:** exploration buys catalog coverage at a small, tunable reward cost. At the served `w = 0.2` the bandit keeps ~95% of the pure-DLRM SNIPS reward (0.696 vs 0.731) while lifting distinct-item coverage from 107 → 180 (+68%). Lower `w` (0.1) keeps ~96% of reward for +57 items. `w = 0` reproduces pure DLRM exactly (a sanity check that the blend is correct). The frontier plot is `experiments/results/bandit_explore_frontier.png` and the raw numbers are in `experiments/results/bandit_offline_eval.json`.

**Propensity & popularity honesty.** MovieLens was not logged under a known randomized policy, so we use a *documented stand-in* propensity (softmax over training popularity), **renormalized within each event's candidate pool**. Because of that pool-local renormalization the **absolute IPS scale is uninterpretable** — use **SNIPS** (invariant to a constant scaling of the weights) for all cross-policy comparison. Most-Popular's higher SNIPS/replay reward is a **self-selection / replay artifact**: it retains a tiny, favorably-biased subset of events (only where the single most-popular candidate happened to be the logged, liked item), and its IPS collapses (0.40) and coverage is the lowest (58 distinct items). It is **not** evidence that popularity beats DLRM.

**Honest caveats / limitations of this measurement:**

- **Exploit/bonus were previously mis-scaled.** The original `dlrm_score + bonus` sum ranked by the larger term, not a blend; the offline eval also measured a *different* policy than what served (a `θ·x + bonus` LinUCB with `θ≈0`, i.e. pure exploration, and a raw embedding dot-product "DLRM" rather than the served sigmoid). Both are fixed: serving and eval now use the identical `combine_explore(dlrm_sigmoid, bonus, w)` logic.
- **Replay-simulated exploration.** MovieLens 100K is *static* — there is no live feedback loop. Rewards come from recorded ratings, not from users reacting to what the bandit showed. This measures whether the mechanism *works* (blend is scale-free, coverage rises with `w`, the update math is correct), not true online lift, which needs a live A/B test.
- **Served policy is DLRM-exploit + uncertainty-bonus.** The exploration bonus reflects contextual *uncertainty* (`√(xᵀA⁻¹x)`), not a learned reward model that beats the DLRM; the DLRM sigmoid remains the exploit signal.
- **Single-worker state.** The bandit lives in memory and persists to `bandit_state.pkl` via an atomic `os.replace` under a `threading.Lock`, which protects concurrent `/feedback` requests **within one worker**. Multiple uvicorn workers each hold their own in-memory bandit and would race the pickle; a shared store (Redis / a param server) is future work.

---

## Limitations & Known Issues

- **MovieLens 100K is small.** 943 users, 1,682 items, 100K ratings. Results on this dataset do not directly generalize to production-scale systems with millions of users and items. The architecture is designed for scale, but it has not been validated at scale.
- **Cold-start exploration is now addressed (with a caveat).** `/recommend_v3` applies the LinUCB contextual bandit to cold-start users over popularity candidates (mean-user-embedding context), combined with popularity on a **common `[0, 1]` scale** (`combine_explore`). This is the fix for the earlier bug where the raw popularity counts swamped the tiny bonus and the served order was *identical* to the static popularity list — the served cold-start order now genuinely differs from the raw list (verified by test). Content-based features for new items and session-based recommendations are still future work. See [Exploration & Online Learning](#exploration--online-learning).
- **Online learning is now addressed for the exploration layer (DLRM weights still static).** `POST /api/v1/feedback` feeds post-deployment rewards into the bandit via rank-1 updates, and the state is persisted (`bandit_state.pkl`) — so user ratings after deployment *do* now move the served policy. The underlying **DLRM** weights are still trained offline and require a manual retraining pipeline; only the exploration layer learns online.
- **Drift detection is threshold-based only.** Feature distribution drift is detected by comparing current feature statistics against training-time baselines using fixed thresholds. There is no statistical test (KS, PSI), no automated alerting, and no automated retraining trigger.
- **BCELoss for ordinal data.** Ratings are binarized (>= 4 is positive, < 4 is negative), discarding the ordinal signal. A 5-star rating and a 4-star rating are treated identically. A ranking loss (BPR, margin loss) or regression loss (MSE on raw ratings) would preserve more information.
- **Diversity via exploration (bandit), measured as catalog coverage on a frontier.** The `/recommend_v3` blend surfaces under-explored items; `scripts/offline_bandit_eval.py` reports an **explore-weight coverage/reward frontier** (distinct-item coverage vs. SNIPS reward per `w`), which quantifies the coverage gain and its reward cost. There is still no direct measurement of intra-list diversity or demographic fairness, and the exploration is **replay-simulated** on static MovieLens data rather than validated against live traffic.
- **TMDB integration couples recommendations to external availability.** The `/predict` endpoint fetches movies from TMDB at request time. If TMDB is down, recommendations fail entirely (502). There is no caching layer.
- **Feature store is pickle-based.** The serving context (user stats, item stats, embeddings) is loaded from pickle files at startup. There is no feature store (Feast, Redis) for fresh features, and no versioning of feature artifacts.
- **Embedding tables dominate parameters.** ~93% of model parameters are in the user and item embedding tables. For a 100K-rating dataset, this is likely overparameterized. Smaller embedding dimensions or regularization may improve generalization.
- **No A/B testing infrastructure.** There is no mechanism to compare model versions in production or measure the impact of model changes on user behavior.
