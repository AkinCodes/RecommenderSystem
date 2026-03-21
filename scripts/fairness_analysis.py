"""Fairness analysis for the DLRM recommender on MovieLens 100K."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.dlrm import DLRMModel

# ---------------------------------------------------------------------------
# Config (mirrors train_movielens.py)
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ml-100k", "u.data")
ITEM_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ml-100k", "u.item")
GENRE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ml-100k", "u.genre")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_model_movielens.pth")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "FAIRNESS_REPORT.md")

NUM_FEATURES = 2
EMBEDDING_SIZES = [943, 1682]
MLP_LAYERS = [128, 64, 32]
K = 10

# ---------------------------------------------------------------------------
# Data loading (same as train_movielens.py)
# ---------------------------------------------------------------------------
def load_and_split():
    raw = np.loadtxt(DATA_PATH, dtype=np.int64)
    sorted_idx = np.argsort(raw[:, 3])
    raw = raw[sorted_idx]

    split = int(len(raw) * 0.8)
    train_raw = raw[:split]
    test_raw = raw[split:]

    all_users = np.unique(raw[:, 0])
    all_items = np.unique(raw[:, 1])
    user2idx = {uid: i for i, uid in enumerate(all_users)}
    item2idx = {iid: i for i, iid in enumerate(all_items)}

    user_ratings = {}
    for row in train_raw:
        uid = row[0]
        user_ratings.setdefault(uid, []).append(row[2])

    max_count = max(len(v) for v in user_ratings.values()) if user_ratings else 1

    def make_features(data):
        cont = np.zeros((len(data), NUM_FEATURES), dtype=np.float32)
        cat = np.zeros((len(data), 2), dtype=np.int64)
        targets = np.zeros(len(data), dtype=np.float32)
        for i, row in enumerate(data):
            uid, iid, rating, _ = row
            cat[i, 0] = user2idx[uid]
            cat[i, 1] = item2idx[iid]
            targets[i] = rating / 5.0
            uratings = user_ratings.get(uid, [3])
            cont[i, 0] = np.mean(uratings) / 5.0
            cont[i, 1] = len(uratings) / max_count
        return cont, cat, targets

    train_cont, train_cat, train_targets = make_features(train_raw)
    test_cont, test_cat, test_targets = make_features(test_raw)

    return (train_raw, test_raw,
            train_cont, train_cat, train_targets,
            test_cont, test_cat, test_targets,
            user2idx, item2idx, user_ratings, max_count)


def load_genres():
    """Load genre names and item-genre mapping."""
    genre_names = []
    with open(GENRE_PATH, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            genre_names.append(parts[0])

    # item_id -> list of genre indices
    item_genres = {}
    with open(ITEM_PATH, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            iid = int(parts[0])
            genre_flags = [int(x) for x in parts[5:]]  # 19 genre flags
            genres = [i for i, g in enumerate(genre_flags) if g == 1]
            item_genres[iid] = genres

    return genre_names, item_genres


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------
def dcg(relevances, k):
    rel = np.array(relevances[:k], dtype=np.float64)
    if len(rel) == 0:
        return 0.0
    discounts = np.log2(np.arange(len(rel)) + 2)
    return np.sum(rel / discounts)


def ndcg_at_k(ranked_relevances, k):
    actual = dcg(ranked_relevances, k)
    ideal = dcg(sorted(ranked_relevances, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("FAIRNESS ANALYSIS — DLRM on MovieLens 100K")
    print("=" * 60)

    # Load data
    (train_raw, test_raw,
     train_cont, train_cat, train_targets,
     test_cont, test_cat, test_targets,
     user2idx, item2idx, user_ratings_dict, max_count) = load_and_split()

    idx2item = {v: k for k, v in item2idx.items()}
    idx2user = {v: k for k, v in user2idx.items()}

    # Load model
    model = DLRMModel(num_features=NUM_FEATURES, embedding_sizes=EMBEDDING_SIZES, mlp_layers=MLP_LAYERS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    print("Model loaded.\n")

    # Load genres
    genre_names, item_genres = load_genres()

    # -----------------------------------------------------------------------
    # Group test interactions by user (same logic as train_movielens.py)
    # -----------------------------------------------------------------------
    user_test = {}  # uidx -> [(iidx, norm_rating)]
    for i in range(len(test_raw)):
        uid = test_raw[i, 0]
        uidx = user2idx[uid]
        iidx = int(test_cat[i, 1])
        rating = test_targets[i]
        user_test.setdefault(uidx, []).append((iidx, rating))

    user_cont_map = {}
    for i in range(len(test_raw)):
        uidx = int(test_cat[i, 0])
        if uidx not in user_cont_map:
            user_cont_map[uidx] = test_cont[i]

    # -----------------------------------------------------------------------
    # Generate top-K recommendations per user
    # -----------------------------------------------------------------------
    user_topk = {}  # uidx -> [iidx, ...]
    user_ndcg = {}  # uidx -> ndcg value

    with torch.no_grad():
        for uidx, items_ratings in user_test.items():
            if len(items_ratings) < 2:
                continue
            relevant = {iidx for iidx, r in items_ratings if r >= 0.8}
            if len(relevant) == 0:
                continue

            item_indices = [ir[0] for ir in items_ratings]
            n = len(item_indices)

            cont_tensor = torch.tensor(np.tile(user_cont_map[uidx], (n, 1)), dtype=torch.float32)
            cat_tensor = torch.zeros(n, 2, dtype=torch.long)
            cat_tensor[:, 0] = uidx
            cat_tensor[:, 1] = torch.tensor(item_indices, dtype=torch.long)

            scores = model(cont_tensor, cat_tensor).squeeze().numpy()
            if scores.ndim == 0:
                scores = np.array([scores.item()])

            ranked_idx = np.argsort(-scores)
            top_k_items = [item_indices[j] for j in ranked_idx[:K]]
            user_topk[uidx] = top_k_items

            ranked_rels = [1.0 if item_indices[j] in relevant else 0.0 for j in ranked_idx]
            user_ndcg[uidx] = ndcg_at_k(ranked_rels, K)

    print(f"Evaluated {len(user_topk)} users with top-{K} recommendations.\n")

    # ===================================================================
    # 1. POPULARITY BIAS
    # ===================================================================
    print("=" * 60)
    print("1. POPULARITY BIAS")
    print("=" * 60)

    # Count training appearances per item (original IDs)
    item_train_count = {}
    for row in train_raw:
        iid = row[1]
        item_train_count[iid] = item_train_count.get(iid, 0) + 1

    # Top 20% threshold
    counts = sorted(item_train_count.values(), reverse=True)
    n_popular = max(1, int(len(counts) * 0.2))
    popularity_threshold = counts[n_popular - 1]  # count at the 20th percentile boundary

    popular_items = {iid for iid, c in item_train_count.items() if c >= popularity_threshold}
    popular_item_idx = {item2idx[iid] for iid in popular_items if iid in item2idx}

    print(f"  Popularity threshold: >= {popularity_threshold} ratings in training")
    print(f"  Popular items: {len(popular_items)} ({100*len(popular_items)/len(item2idx):.1f}% of catalog)")

    # What % of recommendations come from popular items?
    total_recs = 0
    popular_recs = 0
    for uidx, topk in user_topk.items():
        for iidx in topk:
            total_recs += 1
            if iidx in popular_item_idx:
                popular_recs += 1

    rec_popular_pct = 100.0 * popular_recs / total_recs if total_recs > 0 else 0

    # What % of test set interactions involve popular items?
    test_total = len(test_raw)
    test_popular = sum(1 for row in test_raw if row[1] in popular_items)
    test_popular_pct = 100.0 * test_popular / test_total

    print(f"\n  {rec_popular_pct:.1f}% of recommendations are popular movies "
          f"vs {test_popular_pct:.1f}% in the actual test data")
    popularity_bias_ratio = rec_popular_pct / test_popular_pct if test_popular_pct > 0 else float('inf')
    print(f"  Popularity amplification factor: {popularity_bias_ratio:.2f}x")

    # ===================================================================
    # 2. USER ACTIVITY BIAS
    # ===================================================================
    print("\n" + "=" * 60)
    print("2. USER ACTIVITY BIAS")
    print("=" * 60)

    # Count training ratings per user (original IDs)
    user_train_count = {}
    for row in train_raw:
        uid = row[0]
        user_train_count[uid] = user_train_count.get(uid, 0) + 1

    user_counts_sorted = sorted(user_train_count.values(), reverse=True)
    n_active = max(1, int(len(user_counts_sorted) * 0.2))
    activity_threshold = user_counts_sorted[n_active - 1]

    active_users = {uid for uid, c in user_train_count.items() if c >= activity_threshold}
    active_user_idx = {user2idx[uid] for uid in active_users if uid in user2idx}

    active_ndcgs = [v for k, v in user_ndcg.items() if k in active_user_idx]
    casual_ndcgs = [v for k, v in user_ndcg.items() if k not in active_user_idx]

    active_mean = np.mean(active_ndcgs) if active_ndcgs else 0
    casual_mean = np.mean(casual_ndcgs) if casual_ndcgs else 0

    print(f"  Activity threshold: >= {activity_threshold} ratings in training")
    print(f"  Active users: {len(active_users)}, Casual users: {len(user_train_count) - len(active_users)}")
    print(f"  Active users evaluated: {len(active_ndcgs)}, Casual users evaluated: {len(casual_ndcgs)}")
    print(f"\n  Active user NDCG@10: {active_mean:.4f}")
    print(f"  Casual user NDCG@10: {casual_mean:.4f}")
    gap = active_mean - casual_mean
    print(f"  Gap: {gap:+.4f} ({'Active users get better recs' if gap > 0 else 'Casual users get better recs'})")

    # ===================================================================
    # 3. ITEM COVERAGE
    # ===================================================================
    print("\n" + "=" * 60)
    print("3. ITEM COVERAGE")
    print("=" * 60)

    all_recommended = set()
    for uidx, topk in user_topk.items():
        all_recommended.update(topk)

    total_items = len(item2idx)
    coverage = len(all_recommended)
    coverage_pct = 100.0 * coverage / total_items

    print(f"  {coverage} out of {total_items} movies recommended ({coverage_pct:.1f}% coverage)")

    # ===================================================================
    # 4. GENRE DIVERSITY
    # ===================================================================
    print("\n" + "=" * 60)
    print("4. GENRE DIVERSITY")
    print("=" * 60)

    per_user_genre_counts = []
    for uidx, topk in user_topk.items():
        genres_in_topk = set()
        for iidx in topk:
            original_iid = idx2item.get(iidx)
            if original_iid and original_iid in item_genres:
                genres_in_topk.update(item_genres[original_iid])
        per_user_genre_counts.append(len(genres_in_topk))

    avg_diversity = np.mean(per_user_genre_counts) if per_user_genre_counts else 0
    min_diversity = np.min(per_user_genre_counts) if per_user_genre_counts else 0
    max_diversity = np.max(per_user_genre_counts) if per_user_genre_counts else 0

    print(f"  Total genres: {len(genre_names)}")
    print(f"  Average unique genres in top-{K}: {avg_diversity:.2f}")
    print(f"  Min: {min_diversity}, Max: {max_diversity}")

    # ===================================================================
    # Save report
    # ===================================================================
    report = f"""# Fairness Analysis Report — DLRM on MovieLens 100K

## 1. Popularity Bias

| Metric | Value |
|--------|-------|
| Popularity threshold | >= {popularity_threshold} ratings in training |
| Popular items (top 20%) | {len(popular_items)} ({100*len(popular_items)/len(item2idx):.1f}% of catalog) |
| % of recommendations from popular items | {rec_popular_pct:.1f}% |
| % of test interactions with popular items | {test_popular_pct:.1f}% |
| Popularity amplification factor | {popularity_bias_ratio:.2f}x |

**Interpretation:** {"The model over-recommends popular items relative to their natural frequency in the test set." if rec_popular_pct > test_popular_pct else "The model does not significantly amplify popularity bias."}

## 2. User Activity Bias

| Metric | Value |
|--------|-------|
| Activity threshold | >= {activity_threshold} ratings in training |
| Active users evaluated | {len(active_ndcgs)} |
| Casual users evaluated | {len(casual_ndcgs)} |
| Active user NDCG@10 | {active_mean:.4f} |
| Casual user NDCG@10 | {casual_mean:.4f} |
| Gap | {gap:+.4f} |

**Interpretation:** {"Active users receive better recommendations, which is expected since the model has more data to learn their preferences." if gap > 0 else "Casual users receive comparable or better recommendations."}

## 3. Item Coverage

| Metric | Value |
|--------|-------|
| Total catalog size | {total_items} |
| Unique items recommended | {coverage} |
| Catalog coverage | {coverage_pct:.1f}% |

**Interpretation:** {"Low coverage indicates the model concentrates recommendations on a small subset of items, potentially creating a filter bubble." if coverage_pct < 50 else "Reasonable coverage — the model recommends a broad range of items."}

## 4. Genre Diversity

| Metric | Value |
|--------|-------|
| Total genres | {len(genre_names)} |
| Avg unique genres in top-{K} | {avg_diversity:.2f} |
| Min genres in any user's top-{K} | {min_diversity} |
| Max genres in any user's top-{K} | {max_diversity} |

**Interpretation:** On average, each user's top-{K} recommendations span {avg_diversity:.1f} out of {len(genre_names)} genres.
"""

    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport saved to {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
