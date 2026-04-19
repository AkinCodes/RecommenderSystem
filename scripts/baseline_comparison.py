"""Baseline comparison for the DLRM recommender on MovieLens 100K."""

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
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_model_movielens.pth")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "BASELINE_COMPARISON.md")

NUM_FEATURES = 2
EMBEDDING_SIZES = [943, 1682]
MLP_LAYERS = [128, 64, 32]
K = 10

np.random.seed(42)


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


# ---------------------------------------------------------------------------
# Ranking metrics
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


def compute_metrics(user_test, rank_fn, k=K):
    """Compute NDCG@K, Precision@K, HitRate@K for a ranking function.

    rank_fn(uidx, item_indices) -> scores array (higher = better)
    """
    ndcgs, precisions, hit_rates = [], [], []

    for uidx, items_ratings in user_test.items():
        if len(items_ratings) < 2:
            continue
        relevant = {iidx for iidx, r in items_ratings if r >= 0.8}
        if len(relevant) == 0:
            continue

        item_indices = [ir[0] for ir in items_ratings]
        scores = rank_fn(uidx, item_indices)

        ranked_idx = np.argsort(-scores)
        top_k_items = [item_indices[j] for j in ranked_idx[:k]]

        # NDCG
        ranked_rels = [1.0 if item_indices[j] in relevant else 0.0 for j in ranked_idx]
        ndcgs.append(ndcg_at_k(ranked_rels, k))

        # Precision@K
        hits_in_k = sum(1 for it in top_k_items if it in relevant)
        precisions.append(hits_in_k / k)

        # HitRate@K
        hit_rates.append(1.0 if hits_in_k > 0 else 0.0)

    return {
        "NDCG@10": np.mean(ndcgs) if ndcgs else 0.0,
        "Precision@10": np.mean(precisions) if precisions else 0.0,
        "HitRate@10": np.mean(hit_rates) if hit_rates else 0.0,
        "num_eval_users": len(ndcgs),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("BASELINE COMPARISON — MovieLens 100K")
    print("=" * 60)

    # Load data
    (train_raw, test_raw,
     train_cont, train_cat, train_targets,
     test_cont, test_cat, test_targets,
     user2idx, item2idx, user_ratings_dict, max_count) = load_and_split()

    print(f"Train: {len(train_raw)}, Test: {len(test_raw)}")
    print(f"Users: {len(user2idx)}, Items: {len(item2idx)}\n")

    # Group test interactions by user
    user_test = {}
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

    # -------------------------------------------------------------------
    # Baseline 1: Random
    # -------------------------------------------------------------------
    print("Running Baseline 1: Random...")

    def random_rank(uidx, item_indices):
        return np.random.rand(len(item_indices))

    random_metrics = compute_metrics(user_test, random_rank)
    print(f"  NDCG@10: {random_metrics['NDCG@10']:.4f}, "
          f"Precision@10: {random_metrics['Precision@10']:.4f}, "
          f"HitRate@10: {random_metrics['HitRate@10']:.4f}")

    # -------------------------------------------------------------------
    # Baseline 2: Most Popular
    # -------------------------------------------------------------------
    print("Running Baseline 2: Most Popular...")

    # Count training appearances per item index
    item_popularity = np.zeros(len(item2idx), dtype=np.float64)
    for row in train_raw:
        iid = row[1]
        if iid in item2idx:
            item_popularity[item2idx[iid]] += 1

    def popularity_rank(uidx, item_indices):
        return np.array([item_popularity[i] for i in item_indices])

    popular_metrics = compute_metrics(user_test, popularity_rank)
    print(f"  NDCG@10: {popular_metrics['NDCG@10']:.4f}, "
          f"Precision@10: {popular_metrics['Precision@10']:.4f}, "
          f"HitRate@10: {popular_metrics['HitRate@10']:.4f}")

    # -------------------------------------------------------------------
    # Baseline 3: User Mean (item average rating)
    # -------------------------------------------------------------------
    print("Running Baseline 3: User Mean (item avg rating)...")

    # Average rating per item from training data
    item_rating_sum = np.zeros(len(item2idx), dtype=np.float64)
    item_rating_count = np.zeros(len(item2idx), dtype=np.float64)
    for row in train_raw:
        iid = row[1]
        if iid in item2idx:
            idx = item2idx[iid]
            item_rating_sum[idx] += row[2]
            item_rating_count[idx] += 1

    item_avg_rating = np.where(
        item_rating_count > 0,
        item_rating_sum / item_rating_count,
        3.0  # default for unseen items
    )

    def user_mean_rank(uidx, item_indices):
        return np.array([item_avg_rating[i] for i in item_indices])

    user_mean_metrics = compute_metrics(user_test, user_mean_rank)
    print(f"  NDCG@10: {user_mean_metrics['NDCG@10']:.4f}, "
          f"Precision@10: {user_mean_metrics['Precision@10']:.4f}, "
          f"HitRate@10: {user_mean_metrics['HitRate@10']:.4f}")

    # -------------------------------------------------------------------
    # DLRM
    # -------------------------------------------------------------------
    print("Running DLRM model...")

    model = DLRMModel(num_features=NUM_FEATURES, embedding_sizes=EMBEDDING_SIZES, mlp_layers=MLP_LAYERS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    def dlrm_rank(uidx, item_indices):
        n = len(item_indices)
        cont_tensor = torch.tensor(np.tile(user_cont_map[uidx], (n, 1)), dtype=torch.float32)
        cat_tensor = torch.zeros(n, 2, dtype=torch.long)
        cat_tensor[:, 0] = uidx
        cat_tensor[:, 1] = torch.tensor(item_indices, dtype=torch.long)
        with torch.no_grad():
            scores = model(cont_tensor, cat_tensor).squeeze().numpy()
        if scores.ndim == 0:
            scores = np.array([scores.item()])
        return scores

    dlrm_metrics = compute_metrics(user_test, dlrm_rank)
    print(f"  NDCG@10: {dlrm_metrics['NDCG@10']:.4f}, "
          f"Precision@10: {dlrm_metrics['Precision@10']:.4f}, "
          f"HitRate@10: {dlrm_metrics['HitRate@10']:.4f}")

    # -------------------------------------------------------------------
    # Print comparison table
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)

    results = [
        ("Random", random_metrics),
        ("Most Popular", popular_metrics),
        ("User Mean", user_mean_metrics),
        ("DLRM", dlrm_metrics),
    ]

    header = f"| {'Model':<15} | {'NDCG@10':>8} | {'Precision@10':>13} | {'HitRate@10':>11} |"
    sep = f"|{'-'*17}|{'-'*10}|{'-'*15}|{'-'*13}|"
    print(header)
    print(sep)
    for name, m in results:
        print(f"| {name:<15} | {m['NDCG@10']:>8.4f} | {m['Precision@10']:>13.4f} | {m['HitRate@10']:>11.4f} |")

    # -------------------------------------------------------------------
    # Save report
    # -------------------------------------------------------------------
    report_lines = [
        "# Baseline Comparison — DLRM vs Simple Baselines on MovieLens 100K\n",
        f"Evaluated on {dlrm_metrics['num_eval_users']} users with top-{K} recommendations.\n",
        "## Results\n",
        f"| Model           | NDCG@10 | Precision@10 | HitRate@10 |",
        f"|-----------------|---------|--------------|------------|",
    ]
    for name, m in results:
        report_lines.append(
            f"| {name:<15} | {m['NDCG@10']:.4f}  | {m['Precision@10']:.4f}       | {m['HitRate@10']:.4f}     |"
        )

    report_lines.append("")
    report_lines.append("## Baseline Descriptions\n")
    report_lines.append("- **Random**: Assign random scores to each candidate item.")
    report_lines.append("- **Most Popular**: Rank items by number of ratings in the training set.")
    report_lines.append("- **User Mean**: Rank items by their average rating in the training set (a proxy for item quality).")
    report_lines.append("- **DLRM**: Deep Learning Recommendation Model with user/item embeddings and dense features.")
    report_lines.append("")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
