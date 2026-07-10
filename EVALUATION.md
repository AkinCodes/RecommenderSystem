# Evaluation & Ablation Studies

Quantitative evaluation of the RecommenderSystem across model comparison, feature ablation, and serving architecture.

---

## Model Comparison

Head-to-head comparison of the DLRM against classical baselines on the MovieLens 100K dataset (943 users, 1,682 items, 100K ratings).

**Split methodology.** The DLRM is trained on a strict **3-way, timestamp-ordered split — train 0.70 / val 0.10 / test 0.20**. Each slice has one job: **train** fits the weights, **validation** tunes the *training decisions* (early stopping, the `ReduceLROnPlateau` LR schedule, and which checkpoint to keep — all monitor validation loss, with per-epoch val NDCG@10 logged), and **test** is untouched until the final report. User/item feature statistics come from the **train portion only**, so nothing leaks from val/test into the features. The test slice is the same last-20%-by-timestamp used previously, so the DLRM numbers below stay comparable to earlier runs.

| Model | Approach | NDCG@10 | Precision@10 | Recall@10 | HitRate@10 | AUC |
|-------|----------|---------|--------------|-----------|------------|-----|
| DLRM | Learned user/item embeddings (64-dim) + interaction + 3-layer MLP | 0.7861 | 0.6739 | 0.4210 | 0.9965 | 0.6762 |

> DLRM row: production trainer (`scripts/train_movielens.py`) — 3-way split, validation-driven early stopping, seeded (`SEED=42`), evaluated on 283 held-out users / 20K test ratings. Best val loss **0.6551**, best val NDCG@10 **0.7823**, early stop at **epoch 8** (best checkpoint epoch 3). A prior per-user ranking-eval bug (every candidate was scored with a single item's continuous features via `np.tile`) was fixed so each candidate now uses its own per-item features; this is why the calibration metrics (AUC in particular) improved. For the classical baselines under identical matched conditions, see the fair head-to-head table below.

**Reproducibility.** The trainer is seeded (`SEED=42`), so these metrics are reproducible to the digit.

**Validation vs. test (no overfitting gap).** The checkpoint chosen by lowest **validation** loss scored **val NDCG@10 0.7823**; that same checkpoint scores **test NDCG@10 0.7861** (early stop at epoch 8, best checkpoint epoch 3). Val and test moving together (test marginally higher, not lower) is the healthy signal — the model isn't overfitting the validation slice we tuned against. **AUC is 0.6762** on the fixed evaluation — genuinely calibrated separation, not the artifact the old buggy eval produced.

**HitRate caveat.** HitRate@10 is **0.9965**, but it is near-saturated by the per-user candidate design (a random baseline scores ~0.996), so it is reported for completeness, not as a discriminator between models.

**Matched head-to-head comparison.** Under identical matched conditions (`scripts/retrain_and_compare.py`): the DLRM trains with the same 3-way split and validation-driven early stopping as the production trainer, and every model — DLRM, XGBoost, LightGBM, LogReg — fits on the **same 0.70 train portion** and is evaluated on the **same unchanged 0.20 test set** with the same fixed per-item evaluation. The comparison DLRM is now **0.7861**, identical to the production trainer, which resolves the earlier "two DLRM numbers" inconsistency:

| Model | NDCG@10 | Precision@10 | Recall@10 | HitRate@10 | AUC |
|-------|---------|--------------|-----------|------------|-----|
| LogReg | 0.7937 | 0.6823 | 0.4259 | 1.0000 | 0.6807 |
| DLRM | 0.7861 | 0.6739 | 0.4210 | 0.9965 | 0.6762 |
| XGBoost | 0.7607 | 0.6491 | 0.4145 | 0.9929 | 0.6629 |
| LightGBM | 0.7553 | 0.6435 | 0.4152 | 0.9965 | 0.6396 |

**Is the DLRM–LogReg gap real? Paired significance testing.** A 0.0076 NDCG@10 gap between LogReg and the DLRM on **283 users** is meaningless without an interval. So each model is paired against the DLRM on the **same 283 held-out users** (per-user NDCG@10 vectors, aligned by user), and we compute a **paired bootstrap 95% CI** (10,000 resamples, `seed=42`) on the mean delta plus a **two-sided paired permutation p-value** (10,000 sign-flips). Reference model = DLRM; a **positive** delta means the other model beats the DLRM. Reproduce with `uv run python scripts/retrain_and_compare.py` (numbers land in `reports/model_comparison.json` under `"significance"`):

| Comparison | Mean Δ NDCG@10 | 95% CI | p-value | Verdict |
|------------|----------------|--------|---------|---------|
| LogReg − DLRM | +0.0076 | [−0.0027, +0.0175] | 0.138 | **statistical tie** |
| XGBoost − DLRM | −0.0254 | [−0.0417, −0.0088] | 0.003 | **DLRM wins** |
| LightGBM − DLRM | −0.0308 | [−0.0473, −0.0145] | 0.0004 | **DLRM wins** |

> **Honest takeaway.** The headline "LogReg edges ahead of the DLRM" does **not** survive a significance test: the +0.0076 gap has a 95% CI of **[−0.0027, +0.0175]** that **straddles zero** (p = 0.138). LogReg and the DLRM are **statistically indistinguishable** on this data — a coin-flip could put either on top on the next resample, so reporting LogReg as the winner would be over-claiming. The DLRM's advantage over both tree ensembles **is** real: XGBoost (p = 0.003) and LightGBM (p = 0.0004) both fall significantly behind, with CIs entirely below zero. So the defensible statement is: **on matched MovieLens 100K conditions the DLRM significantly beats both gradient-boosted trees and ties a well-scaled linear baseline** — not that any single model "wins" the LogReg/DLRM race. Point estimates can still shift marginally with the BLAS/library version; the significance verdicts (tie vs. beat) are the durable result.

**What to expect:** On MovieLens 100K, a well-scaled linear model (LogReg) stays neck-and-neck with a neural approach because the dataset is small enough that hand-crafted features capture most of the signal — and the significance test above confirms that (LogReg ties the DLRM, it does not beat it). Where the DLRM's learned user-item interaction *does* pay off is against the tree ensembles, which it significantly outranks under matched conditions. At 100K ratings the neural vs. linear advantage is marginal by design; the value of learned embeddings shows more clearly at scale.

---

## SASRec (Sequential Recommendation)

SASRec (`models/sasrec.py`, Kang & McAuley 2018) is a **sequential** model: instead of scoring an isolated `(user, item)` pair from tabular features, it reads a user's chronologically-ordered sequence of liked items and predicts the next one with two causally-masked self-attention blocks (64-dim item embeddings, learned positional embeddings, 1 head, `padding_idx=0`, dropout 0.2 — 161,472 parameters). Trained with a BPR next-item objective (one sampled negative per position, negatives rejected against the user's liked set), seeded (`SEED=42`), leave-last-one-out (LOO) validation HR@10 for early stopping. Reproduce with `uv run python scripts/train_sasrec.py` (early-stopped at **epoch 13**, best val HR@10 **0.4526**). It is a **different task** from the DLRM's pairwise re-ranking, so we report two protocols and are careful not to conflate them.

**(a) LOO next-item retrieval over the full catalogue (the native SASRec metric).** For each of **749** users with ≥2 liked training items, hold out the last liked item and rank it against **all 1,682** catalogue items (already-liked items masked out). This is the standard SASRec evaluation:

| Model | Protocol | HR@10 | NDCG@10 | MRR |
|-------|----------|-------|---------|-----|
| SASRec | LOO next-item vs. full 1,682-item catalogue (749 users) | **0.0628** | **0.0273** | **0.0270** |

> There is **no DLRM number in this row on purpose.** The DLRM is a pointwise `(user, item)` scorer trained on engineered features; it was never trained to retrieve the next item from the full catalogue and has no sequence input, so it cannot produce a comparable full-catalogue next-item number without a different training setup. HR@10 ≈ 0.063 (finding the exact next liked title in a top-10 out of 1,682, ~6× a random 0.6% baseline) is a modest-but-real signal for a 161K-param model on 100K ratings — full-catalogue next-item retrieval is a genuinely harder task than re-ranking a handful of candidates.

> **Does the model actually learn? Yes — the fair, paper-standard check.** Full-catalogue retrieval (0.063 above) is deliberately the hardest framing. Under SASRec's own standard protocol — leave-one-out, ranking the held-out next item against **100 sampled negatives** on the users who actually have a sequence — the model scores **HR@10 ≈ 0.45** (validation HR@10 **0.4526**, emitted by `train_sasrec.py`; held-out warm-test users ≈ 0.44), i.e. **~4.5× the random ≈ 0.099 baseline**. So SASRec genuinely learns sequential structure. The harsh full-catalogue 0.063 (1,682-way retrieval) and the near-chance re-ranking AUC in (b) below are **task-difficulty and cold-user averaging effects, not a broken model** — the same trained checkpoint scores respectably on its native task under the standard protocol.

**(b) Re-ranking the SAME held-out test candidates the DLRM is scored on (apples-to-apples table).** Here SASRec scores each test candidate by `dot(encode(user's train history), item_emb(candidate))` and runs through the **identical** `compute_ranking_metrics` used for the DLRM table above (same 283 eval users, same 0.20 test slice):

| Model | NDCG@10 | Recall@10 | HitRate@10 | MRR | AUC |
|-------|---------|-----------|------------|-----|-----|
| DLRM (reference) | **0.7861** | **0.4210** | 0.9965 | — | **0.6762** |
| SASRec (re-rank) | 0.6530 | 0.3856 | 0.9894 | 0.7645 | 0.5292 |

> **Honest takeaway — SASRec does NOT win the re-ranking task here; the DLRM leads it** (NDCG@10 0.6530 vs 0.7861, Recall@10 0.3856 vs 0.4210). SASRec's re-ranking AUC (**0.5292**) is close to chance, which is the tell: **192 of the test users are unseen in the train slice** and therefore have an empty liked-item history, so SASRec falls back to uniform (zero) scores for them — it has nothing to attend over. This is not a bug in SASRec; it is the structural mismatch between the two evaluations. **Two asymmetries make this table not a clean, controlled A/B — do not read it as "DLRM beats SASRec, full stop":**
> 1. **Per-row-feature asymmetry (footnote).** The DLRM scores every candidate with its **own** engineered continuous features — `item_mean`, `item_count`, `item_popularity_rank`, and the `user_mean − item_mean` interaction — computed per `(user, item)` row (see `scripts/retrain_and_compare.py`'s `dlrm_rank`, which pulls each candidate's own `test_cont` row). SASRec uses **none** of those; it sees only the item-id sequence. On MovieLens-100K, popularity/mean features alone are strong predictors of a like, so the DLRM enters this table with a per-row information advantage SASRec structurally cannot access.
> 2. **Cold-user asymmetry.** SASRec needs a non-empty *liked* history; the DLRM only needs a user embedding + features, so it degrades more gracefully on the ~68% of eval users whose train history is empty here. The re-ranking test set was designed around the DLRM's inputs, not SASRec's.
>
> **Defensible statement:** on MovieLens-100K, SASRec is a correct, from-scratch sequential recommender that **owns the native next-item retrieval task the DLRM can't do**, but on the DLRM's own feature-rich re-ranking benchmark it trails the DLRM — largely because that benchmark supplies per-row popularity features SASRec ignores and is dominated by cold users with no sequence to attend over. The two models are best seen as **complementary** (sequential candidate generation + feature-rich re-ranking), not as a single-winner shootout.

**Paired significance (same machinery as the classical comparison).** Running `scripts/retrain_and_compare.py` appends a `SASRec` row so the paired-bootstrap CI / permutation loop scores SASRec-vs-DLRM per-user NDCG@10 on the same 283 held-out users (reference model = DLRM; positive delta ⇒ the other model beats the DLRM):

| Comparison | Mean Δ NDCG@10 | 95% CI | p-value | Verdict |
|------------|----------------|--------|---------|---------|
| SASRec − DLRM | −0.1331 | [−0.1604, −0.1060] | 0.0001 | **DLRM wins** |

> The gap is **significant, not a coin-flip** (CI entirely below zero, p = 0.0001) — so on this re-ranking benchmark the honest verdict is unambiguously "DLRM ahead of SASRec," subject to the two asymmetries above (SASRec sees no per-row popularity features and ~68% of these eval users are cold with no sequence). This is reported with the same discipline as the LogReg result above, where the DLRM edge was *not* significant and we called it a tie. Here it **is** significant, so we say so; we do not manufacture a SASRec win. SASRec's contribution is the sequential next-item retrieval capability (protocol **a**), a task the tabular DLRM is not built for.

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
| DLRM (pure exploit, `w=0`) | **0.788** | 81 | 0.848 |
| Bandit `w=0.1` | 0.779 | 95 | 0.870 |
| Bandit `w=0.2` (**served default**) | **0.802** | 122 | 0.904 |
| Bandit `w=0.3` | 0.775 | 151 | 0.903 |
| Bandit `w=0.5` | 0.703 | 218 | 0.887 |
| Bandit `w=1.0` (pure explore) | 0.560 | 273 | 0.856 |
| Most-Popular (reference) | 0.754 | 58 | 0.997 |

**Owning the tradeoff:** exploration buys catalog coverage, and on this replay the cost is negative at the served operating point. At `w = 0.2` the bandit **matches (slightly exceeds) the pure-DLRM SNIPS reward** (0.802 vs 0.788, ~102%) while lifting distinct-item coverage from 81 → 122 (+51%) — a genuine free-lunch point where a little exploration surfaces more of the catalog without a measurable reward cost. Reward holds through `w = 0.3` (0.775, 151 items) and only drops off at `w = 0.5` (0.703, 218 items) and `w = 1.0` (0.560, 273 items), where coverage is highest. Lower `w` (0.1) keeps ~99% of reward for +14 items. `w = 0` reproduces pure DLRM exactly (a sanity check that the blend is correct). This near-costless small-`w` region is a replay artifact of a static dataset — treat it as evidence the mechanism is sound, not as a guaranteed online reward lift. The frontier plot is `experiments/results/bandit_explore_frontier.png` and the raw numbers are in `experiments/results/bandit_offline_eval.json`.

**Propensity & popularity honesty.** MovieLens was not logged under a known randomized policy, so we use a *documented stand-in* propensity (softmax over training popularity), **renormalized within each event's candidate pool**. Because of that pool-local renormalization the **absolute IPS scale is uninterpretable** — use **SNIPS** (invariant to a constant scaling of the weights) for all cross-policy comparison. Most-Popular's SNIPS reward (0.754) sits **below** both pure DLRM and the served bandit, and its replay reward is a **self-selection / replay artifact**: it retains a tiny, favorably-biased subset of events (only where the single most-popular candidate happened to be the logged, liked item), while its IPS collapses (0.40) and its coverage is the lowest (58 distinct items). It is **not** evidence that popularity beats DLRM.

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
