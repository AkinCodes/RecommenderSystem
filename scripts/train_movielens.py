"""Train DLRM on MovieLens 100K and evaluate with ranking metrics."""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import wandb

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.dlrm import DLRMModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ml-100k", "u.data")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_model_movielens.pth")

NUM_FEATURES = 2          # mean_rating, normalised_count
EMBEDDING_SIZES = [943, 1682]  # users, items
MLP_LAYERS = [128, 64, 32]
EPOCHS = 20
BATCH_SIZE = 256
LR = 0.001
DEVICE = "cpu"
K = 10  # top-K for ranking metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    """Load MovieLens 100K u.data and return structured arrays."""
    raw = np.loadtxt(DATA_PATH, dtype=np.int64)  # user_id, item_id, rating, timestamp
    return raw


def prepare_splits(raw):
    """Timestamp-based 80/20 split and feature engineering."""
    # Sort by timestamp for chronological split
    sorted_idx = np.argsort(raw[:, 3])
    raw = raw[sorted_idx]

    split = int(len(raw) * 0.8)
    train_raw = raw[:split]
    test_raw = raw[split:]

    # Build contiguous ID mappings
    all_users = np.unique(raw[:, 0])
    all_items = np.unique(raw[:, 1])
    user2idx = {uid: i for i, uid in enumerate(all_users)}
    item2idx = {iid: i for i, iid in enumerate(all_items)}

    # Compute user-level stats from TRAINING data only (avoid leakage)
    user_ratings = {}
    for row in train_raw:
        uid = row[0]
        user_ratings.setdefault(uid, []).append(row[2])

    max_count = max(len(v) for v in user_ratings.values()) if user_ratings else 1

    def make_features(data, user_ratings_dict, max_count):
        cont = np.zeros((len(data), NUM_FEATURES), dtype=np.float32)
        cat = np.zeros((len(data), 2), dtype=np.int64)
        targets = np.zeros(len(data), dtype=np.float32)

        for i, row in enumerate(data):
            uid, iid, rating, _ = row
            cat[i, 0] = user2idx[uid]
            cat[i, 1] = item2idx[iid]
            targets[i] = rating / 5.0  # normalise to [0, 1]

            # User continuous features
            uratings = user_ratings_dict.get(uid, [3])  # default mean=3 if unseen
            cont[i, 0] = np.mean(uratings) / 5.0       # normalised mean rating
            cont[i, 1] = len(uratings) / max_count      # normalised count

        return cont, cat, targets

    train_cont, train_cat, train_targets = make_features(train_raw, user_ratings, max_count)
    # For test, we still use training stats (no leakage)
    test_cont, test_cat, test_targets = make_features(test_raw, user_ratings, max_count)

    return (train_cont, train_cat, train_targets,
            test_cont, test_cat, test_targets,
            user2idx, item2idx, test_raw)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MovieLensDataset(Dataset):
    def __init__(self, cont, cat, targets):
        self.cont = torch.tensor(cont, dtype=torch.float32)
        self.cat = torch.tensor(cat, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.cont[idx], self.cat[idx], self.targets[idx]


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


def evaluate(model, test_cont, test_cat, test_targets, user2idx, item2idx, test_raw, k=K):
    """Compute ranking metrics per user, then average."""
    model.eval()
    num_items = len(item2idx)

    # Group test interactions by user
    user_test = {}
    for i in range(len(test_raw)):
        uid = test_raw[i, 0]
        uidx = user2idx[uid]
        iidx = int(test_cat[i, 1])
        rating = test_targets[i]
        user_test.setdefault(uidx, []).append((iidx, rating))

    # We also need per-user continuous features
    user_cont = {}
    for i in range(len(test_raw)):
        uidx = int(test_cat[i, 0])
        if uidx not in user_cont:
            user_cont[uidx] = test_cont[i]

    ndcgs, precisions, recalls, hit_rates = [], [], [], []
    all_labels, all_scores = [], []

    with torch.no_grad():
        for uidx, items_ratings in user_test.items():
            if len(items_ratings) < 2:
                continue

            # Relevant items: rating >= 4 (i.e., normalised >= 0.8)
            relevant = {iidx for iidx, r in items_ratings if r >= 0.8}
            if len(relevant) == 0:
                continue

            # Score all test items for this user
            item_indices = [ir[0] for ir in items_ratings]
            n = len(item_indices)

            cont_tensor = torch.tensor(
                np.tile(user_cont[uidx], (n, 1)), dtype=torch.float32
            )
            cat_tensor = torch.zeros(n, 2, dtype=torch.long)
            cat_tensor[:, 0] = uidx
            cat_tensor[:, 1] = torch.tensor(item_indices, dtype=torch.long)

            scores = model(cont_tensor, cat_tensor).squeeze().numpy()
            if scores.ndim == 0:
                scores = np.array([scores.item()])

            # For AUC
            for idx_j, (iidx, r) in enumerate(items_ratings):
                all_labels.append(1.0 if r >= 0.8 else 0.0)
                all_scores.append(float(scores[idx_j]))

            # Rank by score descending
            ranked_idx = np.argsort(-scores)
            top_k_items = [item_indices[j] for j in ranked_idx[:k]]

            # NDCG
            ranked_rels = [1.0 if item_indices[j] in relevant else 0.0 for j in ranked_idx]
            ndcgs.append(ndcg_at_k(ranked_rels, k))

            # Precision@K
            hits_in_k = sum(1 for it in top_k_items if it in relevant)
            precisions.append(hits_in_k / k)

            # Recall@K
            recalls.append(hits_in_k / len(relevant) if relevant else 0.0)

            # Hit Rate@K (1 if at least one hit)
            hit_rates.append(1.0 if hits_in_k > 0 else 0.0)

    # AUC
    auc = 0.0
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_scores)

    metrics = {
        "NDCG@10": np.mean(ndcgs),
        "Precision@10": np.mean(precisions),
        "Recall@10": np.mean(recalls),
        "HitRate@10": np.mean(hit_rates),
        "AUC": auc,
        "num_eval_users": len(ndcgs),
    }
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("DLRM Training on MovieLens 100K")
    print("=" * 60)

    # Initialise Weights & Biases
    wandb.init(
        project="recommender-system",
        entity="workwithakin-akin-olusanya",
        config={
            "model": "DLRM",
            "dataset": "MovieLens-100K",
            "num_features": NUM_FEATURES,
            "embedding_sizes": EMBEDDING_SIZES,
            "mlp_layers": MLP_LAYERS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "device": DEVICE,
            "top_k": K,
        },
    )

    # Load and prepare data
    print("\n[1/4] Loading data...")
    raw = load_data()
    print(f"  Total ratings: {len(raw)}")
    (train_cont, train_cat, train_targets,
     test_cont, test_cat, test_targets,
     user2idx, item2idx, test_raw) = prepare_splits(raw)
    print(f"  Train samples: {len(train_targets)}")
    print(f"  Test samples:  {len(test_targets)}")
    print(f"  Users: {len(user2idx)}, Items: {len(item2idx)}")

    # Build model
    print("\n[2/4] Building model...")
    model = DLRMModel(
        num_features=NUM_FEATURES,
        embedding_sizes=EMBEDDING_SIZES,
        mlp_layers=MLP_LAYERS,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture: num_features={NUM_FEATURES}, "
          f"embedding_sizes={EMBEDDING_SIZES}, mlp_layers={MLP_LAYERS}")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Dataloaders
    train_ds = MovieLensDataset(train_cont, train_cat, train_targets)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Training
    print(f"\n[3/4] Training for {EPOCHS} epochs (batch_size={BATCH_SIZE}, lr={LR})...")
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for cont_b, cat_b, target_b in train_loader:
            optimizer.zero_grad()
            preds = model(cont_b, cat_b).squeeze()
            loss = criterion(preds, target_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / n_batches
        wandb.log({"train_loss": avg_loss, "epoch": epoch})
        print(f"  Epoch {epoch:2d}/{EPOCHS} — train loss: {avg_loss:.6f}")
    train_time = time.time() - train_start
    print(f"  Training completed in {train_time:.1f}s")

    # Save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"  Model saved to {MODEL_SAVE_PATH}")

    # Evaluation
    print(f"\n[4/4] Evaluating on test set...")
    metrics = evaluate(model, test_cont, test_cat, test_targets, user2idx, item2idx, test_raw)

    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print(f"  NDCG@10:      {metrics['NDCG@10']:.4f}")
    print(f"  Precision@10: {metrics['Precision@10']:.4f}")
    print(f"  Recall@10:    {metrics['Recall@10']:.4f}")
    print(f"  HitRate@10:   {metrics['HitRate@10']:.4f}")
    print(f"  AUC:          {metrics['AUC']:.4f}")
    print(f"  Eval users:   {metrics['num_eval_users']}")

    # Inference latency
    print("\n" + "=" * 60)
    print("INFERENCE LATENCY (1000 runs, single sample)")
    print("=" * 60)
    model.eval()
    dummy_cont = torch.randn(1, NUM_FEATURES)
    dummy_cat = torch.randint(0, 100, (1, 2))

    # Warmup
    for _ in range(100):
        with torch.no_grad():
            model(dummy_cont, dummy_cat)

    latencies = []
    with torch.no_grad():
        for _ in range(1000):
            t0 = time.perf_counter()
            model(dummy_cont, dummy_cat)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies = np.array(latencies)
    print(f"  Mean:   {np.mean(latencies):.3f} ms")
    print(f"  Median: {np.median(latencies):.3f} ms")
    print(f"  P95:    {np.percentile(latencies, 95):.3f} ms")
    print(f"  P99:    {np.percentile(latencies, 99):.3f} ms")
    print(f"  Min:    {np.min(latencies):.3f} ms")
    print(f"  Max:    {np.max(latencies):.3f} ms")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Model:        DLRM")
    print(f"  Dataset:      MovieLens 100K ({len(raw)} ratings)")
    print(f"  Parameters:   {total_params:,}")
    print(f"  Embeddings:   user({EMBEDDING_SIZES[0]}x{MLP_LAYERS[0]}), item({EMBEDDING_SIZES[1]}x{MLP_LAYERS[0]})")
    print(f"  MLP layers:   {MLP_LAYERS}")
    print(f"  Training:     {EPOCHS} epochs in {train_time:.1f}s")
    print(f"  NDCG@10:      {metrics['NDCG@10']:.4f}")
    print(f"  Precision@10: {metrics['Precision@10']:.4f}")
    print(f"  Recall@10:    {metrics['Recall@10']:.4f}")
    print(f"  HitRate@10:   {metrics['HitRate@10']:.4f}")
    print(f"  AUC:          {metrics['AUC']:.4f}")
    print(f"  Latency:      {np.mean(latencies):.3f} ms (mean), {np.percentile(latencies, 95):.3f} ms (p95)")
    print("=" * 60)

    # Log evaluation metrics to W&B
    wandb.log({
        "ndcg_at_10": metrics["NDCG@10"],
        "precision_at_10": metrics["Precision@10"],
        "recall_at_10": metrics["Recall@10"],
        "hit_rate_at_10": metrics["HitRate@10"],
        "auc": metrics["AUC"],
        "num_eval_users": metrics["num_eval_users"],
    })

    # Log inference latency to W&B
    wandb.log({
        "inference_ms_mean": float(np.mean(latencies)),
        "inference_ms_median": float(np.median(latencies)),
        "inference_ms_p95": float(np.percentile(latencies, 95)),
        "inference_ms_p99": float(np.percentile(latencies, 99)),
    })

    # Log the model as a W&B artifact
    artifact = wandb.Artifact("dlrm-movielens", type="model")
    artifact.add_file(MODEL_SAVE_PATH)
    wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()
