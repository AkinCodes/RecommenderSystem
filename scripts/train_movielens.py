"""Train DLRM on MovieLens 100K and evaluate with ranking metrics."""

import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

import wandb

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.preprocessing import (
    MAX_RATING_VAR,
    NUM_FEATURES,
    RATING_MAX,
    compute_item_stats,
    compute_max_user_days,
    compute_user_stats,
    load_movielens_data,
    prepare_splits,
)
from models.dlrm import DLRMModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ml-100k", "u.data")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_model_movielens.pth")

EMBEDDING_SIZES = [943, 1682]  # users, items
MLP_LAYERS = [128, 64, 32]
EPOCHS = 20
BATCH_SIZE = 256
LR = 0.001
WEIGHT_DECAY = 1e-5
DROPOUT = 0.2
EARLY_STOP_PATIENCE = 5
DEVICE = "cpu"
K = 10  # top-K for ranking metrics


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

            # Relevant items: binary label 1.0 (liked)
            relevant = {iidx for iidx, r in items_ratings if r >= 1.0}
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
                all_labels.append(1.0 if r >= 1.0 else 0.0)
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
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "device": DEVICE,
            "top_k": K,
        },
    )

    # Load and prepare data
    print("\n[1/4] Loading data...")
    raw = load_movielens_data(DATA_PATH)
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
        dropout=DROPOUT,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

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

        # Learning rate scheduling
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        wandb.log({
            "train_loss": avg_loss,
            "epoch": epoch,
            "learning_rate": current_lr,
            "epochs_no_improve": epochs_no_improve,
        })
        print(f"  Epoch {epoch:2d}/{EPOCHS} — train loss: {avg_loss:.6f}  lr: {current_lr:.6f}  no_improve: {epochs_no_improve}")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping triggered after {epoch} epochs (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            wandb.log({"early_stopped_epoch": epoch})
            break

    train_time = time.time() - train_start
    print(f"  Training completed in {train_time:.1f}s")

    # Restore best model weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print("  Restored best model weights")

    # Save best model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"  Model saved to {MODEL_SAVE_PATH}")

    # Save serving context for the recommendation API
    idx2item = {v: k for k, v in item2idx.items()}
    train_raw_serve = raw[:int(len(raw) * 0.8)]
    user_ratings_serve, max_count_serve, user_timestamps_serve = compute_user_stats(train_raw_serve)
    item_ratings_serve, item_max_count_serve, item_popularity_rank_serve = compute_item_stats(train_raw_serve)
    max_user_days_serve = compute_max_user_days(user_timestamps_serve)

    # Pre-compute per-user features (4 values: mean, count, var, days_active)
    user_features = {}
    for uid, uidx in user2idx.items():
        uratings = user_ratings_serve.get(uid, [3])
        user_mean = np.mean(uratings) / RATING_MAX
        user_count = len(uratings) / max_count_serve
        user_var = np.var(uratings) / MAX_RATING_VAR if len(uratings) > 1 else 0.0
        utimes = user_timestamps_serve.get(uid, [])
        if len(utimes) > 1:
            days = (max(utimes) - min(utimes)) / 86400.0
            user_days = days / max_user_days_serve if max_user_days_serve > 0 else 0.0
        else:
            user_days = 0.0
        user_features[uidx] = [user_mean, user_count, user_var, user_days]

    # Pre-compute per-item features (3 values: mean, count, popularity_rank)
    item_features = {}
    for iid, iidx in item2idx.items():
        iratings = item_ratings_serve.get(iid, [3])
        item_mean = np.mean(iratings) / RATING_MAX
        item_count = len(iratings) / item_max_count_serve
        item_pop = item_popularity_rank_serve.get(iid, 0.0)
        item_features[iidx] = [item_mean, item_count, item_pop]

    serving_ctx = {
        "user2idx": user2idx, "item2idx": item2idx, "idx2item": idx2item,
        "user_features": user_features, "item_features": item_features,
        "max_count": max_count_serve,
    }
    ctx_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), "serving_context.pkl")
    with open(ctx_path, "wb") as f:
        pickle.dump(serving_ctx, f)
    print(f"  Serving context saved to {ctx_path}")

    # Evaluation
    print("\n[4/4] Evaluating on test set...")
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
    print("  Model:        DLRM")
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
