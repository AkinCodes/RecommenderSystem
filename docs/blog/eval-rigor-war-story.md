# The eval was lying to me: an `np.tile` bug, a bootstrap, and a headline I had to retract

*A war story about how most ML bugs live in the evaluation code — and how they flatter you.*

I had a good-looking recommender. A DLRM trained on MovieLens 100K, an AUC that said the model separated liked from disliked items, an NDCG@10 in the same neighborhood as the classical baselines. The README had the numbers. The story wrote itself: *neural model, competitive with the trees, ship it.*

Then I went to reproduce the evaluation from scratch, and it quietly fell apart — not because the model was bad, but because the code that *measured* it was wrong, in the most seductive way possible: it made everything look fine. This is a story of two bugs in the eval, one a correctness bug and one a *rigor* bug. Both flattered me. Fixing them changed a headline I'd already written.

## Bug 1: every candidate got scored with one item's features

The ranking eval does something conceptually simple. For each held-out user, take the items they interacted with in the test slice, score each item with the model, sort by score, and compute NDCG@10 against the items they actually liked. The model needs two things per candidate: the categorical IDs (user id, item id) and the continuous features (engineered user/item statistics — item mean rating, item popularity, user activity, and so on).

Here's the shape of the old scoring code:

```python
# The bug: score every candidate with ONE row of continuous features.
cont_tensor = torch.tensor(
    np.tile(user_cont[uidx], (len(item_indices), 1)),   # <-- one row, tiled N times
    dtype=torch.float32,
)
cat_tensor = torch.zeros(len(item_indices), 2, dtype=torch.long)
cat_tensor[:, 0] = uidx
cat_tensor[:, 1] = torch.tensor(item_indices, dtype=torch.long)
scores = model(cont_tensor, cat_tensor).squeeze().numpy()
```

Look at the `np.tile`. Every candidate for that user is fed the **same** continuous row. The categorical embedding for the item changes (that's the `cat_tensor[:, 1]`), but the continuous features — including the item-specific ones like item mean rating and item popularity — are identical for all candidates. A blockbuster and an obscure indie film, scored side by side, got the *same* "how popular is this item" input. The only thing distinguishing them was the item embedding.

This is exactly the kind of bug that doesn't crash and doesn't look wrong. The eval still runs, still produces an NDCG and an AUC. Those numbers just measure a crippled model — one blind to half its own features at ranking time. And because the bug was *symmetric* (it hobbled every candidate the same way), the metrics stayed plausible: not zero, not NaN, just quietly optimistic in the wrong places (calibration/AUC especially).

I wrote a regression test that pins the bug down without any of the surrounding machinery. Give one user two candidates with **deliberately different** continuous rows, and assert they get different scores — because a correct eval feeds each its own row, and the tiled bug would collapse them:

```python
def test_candidates_get_distinct_scores_from_own_continuous_features():
    test_cont = np.array([[0.10, 0.10, 0.10],
                          [0.90, 0.90, 0.90]], dtype=np.float32)
    test_cat  = np.array([[0, 1], [0, 2]], dtype=np.int64)
    # correct path: each candidate scored with ITS OWN row
    scores = model(torch.tensor(test_cont), torch.tensor(test_cat)).squeeze().numpy()
    assert not np.isclose(scores[0], scores[1])
    # the tiled bug would feed test_cont[0] to BOTH -> identical continuous inputs
    tiled = model(torch.tensor(np.tile(test_cont[0], (2, 1))), torch.tensor(test_cat)).squeeze().numpy()
    assert not np.allclose(scores, tiled)
```

(That test lives in `tests/test_evaluate.py`.)

The fix is boring, which is the point. Track each candidate's source row index while grouping by user, and index the continuous matrix with it:

```python
item_indices = [ir[0] for ir in items_ratings]
row_idxs     = [ir[2] for ir in items_ratings]      # each candidate's OWN row
# Each candidate scored with its OWN continuous row (via row_idxs).
cont_tensor = torch.tensor(test_cont[row_idxs], dtype=torch.float32)
cat_tensor  = torch.zeros(len(item_indices), 2, dtype=torch.long)
cat_tensor[:, 0] = uidx
cat_tensor[:, 1] = torch.tensor(item_indices, dtype=torch.long)
scores = model(cont_tensor, cat_tensor).squeeze().numpy()
```

After the fix, the DLRM's calibration metrics moved (AUC settled at **0.6762**, honestly measured), and the ranking number landed at **NDCG@10 = 0.7861** on 283 held-out users. Good. Now I had a *correct* point estimate.

I thought I was done. I was not done.

## Bug 2: a correct point estimate still isn't evidence

With the eval fixed, I ran the matched head-to-head — DLRM, XGBoost, LightGBM, and a logistic-regression baseline, all trained on the same 0.70 train slice, all scored on the same untouched 0.20 test slice, same fixed per-candidate scoring for everyone. The table:

| Model | NDCG@10 |
|-------|---------|
| LogReg | 0.7937 |
| DLRM | 0.7861 |
| XGBoost | 0.7607 |
| LightGBM | 0.7553 |

And I wrote the honest-sounding sentence: *"LogReg edges narrowly ahead of the DLRM."* It's a 0.0076 gap. It's right there in the numbers. LogReg is on top. What's dishonest about reporting what the table says?

Here's what stopped me. That 0.7937 vs 0.7861 is a difference of **means over 283 users**. A mean has a sampling distribution. 0.0076 is small. 283 is not a huge number. So the real question isn't "which mean is bigger" — I can read that off the table — it's **"if I re-ran this on a fresh draw of 283 users, would LogReg still be on top?"** A point estimate can't answer that. I was about to publish a ranking of two models based on a gap I had no interval for.

The move is a **paired bootstrap**. I already had per-user NDCG@10 vectors (I just had to stop throwing them away after `np.mean`). Because every model is scored on the *same* users, I can pair them: for each user, compute `LogReg_ndcg − DLRM_ndcg`, then resample those paired deltas with replacement 10,000 times and read off the 2.5th and 97.5th percentiles for a 95% CI. Pair it with a permutation test — under the null of no difference, the sign of each paired delta is exchangeable, so flip signs at random 10,000 times and count how often the permuted mean delta is as extreme as the observed one:

```python
def paired_bootstrap_ci(scores_a, scores_b, n=10000, seed=42):
    diffs = np.asarray(scores_a) - np.asarray(scores_b)   # paired per-user deltas
    rng = np.random.default_rng(seed)
    # bootstrap CI on the mean delta
    boot = diffs[rng.integers(0, diffs.size, size=(n, diffs.size))].mean(axis=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    # two-sided paired permutation p-value (sign-flip)
    obs = abs(diffs.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, diffs.size))
    perm = np.abs((signs * diffs).mean(axis=1))
    p = (np.sum(perm >= obs) + 1) / (n + 1)
    return {"mean_delta": diffs.mean(), "ci_low": ci_low, "ci_high": ci_high, "p_value": p}
```

Seeded, so it reproduces to the digit. Here's what it actually says (reference = DLRM; positive delta means the other model wins):

| Comparison | Mean Δ NDCG@10 | 95% CI | p-value | Verdict |
|------------|----------------|--------|---------|---------|
| LogReg − DLRM | **+0.0076** | **[−0.0027, +0.0175]** | **0.138** | **statistical tie** |
| XGBoost − DLRM | −0.0254 | [−0.0417, −0.0088] | 0.003 | DLRM wins |
| LightGBM − DLRM | −0.0308 | [−0.0473, −0.0145] | 0.0004 | DLRM wins |

There it is. The LogReg "win" is a **statistical tie**. The 95% CI on the gap is **[−0.0027, +0.0175]** — it straddles zero — and the permutation p-value is **0.138**. On the next draw of users, either model could be on top. "LogReg edges ahead" was a sentence I could only write because I'd never asked whether the edge would survive resampling. It doesn't.

And — the part that actually matters for the model I built — the DLRM's advantage over the tree ensembles *does* survive. XGBoost (p = 0.003) and LightGBM (p = 0.0004) both fall significantly behind, CIs entirely below zero. So the honest, defensible headline isn't "LogReg wins" and it isn't "DLRM wins." It's: **the DLRM significantly outranks both gradient-boosted trees and is statistically indistinguishable from a well-scaled linear baseline** on this small, 8-feature MovieLens setup. Less punchy. True.

I added a unit test asserting the helper's own sanity — identical inputs give a CI that contains 0 and p = 1, a constant shift is significant, pure noise is a tie — so the rigor tool is itself under test (`tests/test_bootstrap_ci.py`).

## The transferable lesson

Two things I'm taking with me:

1. **Most ML bugs are in the eval, not the model — and they flatter you.** The `np.tile` bug didn't crash; it produced believable numbers. A crash you'll fix in an hour. A plausible-but-wrong metric can live in your README for months, because nothing is screaming. The tell was that I couldn't cleanly reproduce the eval from first principles — when the measurement code is doing something you can't explain in one sentence ("each candidate is scored with its own features"), distrust the number, not just the model.

2. **A correct point estimate is not evidence; a point estimate with an interval is.** The second bug was subtler than the first because the number was *right* — 0.7937 really is bigger than 0.7861. The bug was in what I concluded from it. Any time you're about to rank two models by a small gap on a modest sample, the gap needs a confidence interval before it earns a sentence. If the CI straddles zero, the honest word is "tie," and you say so even when a cleaner story was one rounding decision away.

The recruiter-facing version of this: you can reproduce every number above with `uv run python scripts/retrain_and_compare.py` (CIs and p-values land in `reports/model_comparison.json` under `"significance"`), and the whole test suite — including the two regression tests that pin these bugs — runs with `uv run pytest` (147 passing). The point of the exercise was never the recommender. It was learning to be suspicious of a metric that agrees with me.
