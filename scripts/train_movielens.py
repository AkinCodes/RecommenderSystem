"""Train DLRM on MovieLens 100K and evaluate with ranking metrics."""

import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.preprocessing import (
    MAX_RATING_VAR,
    NUM_FEATURES,
    RATING_MAX,
    PrepConfig,
    compute_item_stats,
    compute_max_user_days,
    compute_user_stats,
    load_movielens_data,
    prepare_train_val_test,
)
from models.dlrm import DLRMModel

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ml-100k", "u.data")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_model_movielens.pth")

CARDINALITIES = [943, 1682]
EMBEDDING_DIM = 64
MLP_LAYERS = [128, 64, 32]
EPOCHS = 20
BATCH_SIZE = 256
LR = 0.001
WEIGHT_DECAY = 1e-5
DROPOUT = 0.2
EARLY_STOP_PATIENCE = 5
DEVICE = "cpu"
K = 10
SEED = 42


class MovieLensDataset(Dataset):
    def __init__(self, cont, cat, targets):
        self.cont = torch.tensor(cont, dtype=torch.float32)
        self.cat = torch.tensor(cat, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.cont[idx], self.cat[idx], self.targets[idx]


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
    model.eval()

    # Keep each candidate's source ROW index so we can score it with ITS OWN
    # continuous features (item_mean/count/popularity, user×item interaction),
    # not one tiled row that would be wrong for every candidate but one.
    user_test = {}
    for i in range(len(test_raw)):
        uid = test_raw[i, 0]
        uidx = user2idx[uid]
        iidx = int(test_cat[i, 1])
        rating = test_targets[i]
        user_test.setdefault(uidx, []).append((iidx, rating, i))

    ndcgs, precisions, recalls, hit_rates = [], [], [], []
    all_labels, all_scores = [], []

    with torch.no_grad():
        for uidx, items_ratings in user_test.items():
            if len(items_ratings) < 2:
                continue

            relevant = {iidx for iidx, r, _ in items_ratings if r >= 1.0}
            if len(relevant) == 0:
                continue

            item_indices = [ir[0] for ir in items_ratings]
            row_idxs = [ir[2] for ir in items_ratings]

            # Each candidate scored with its OWN continuous row (via row_idxs);
            # the categorical tensor (user id + item id) is unchanged.
            cont_tensor = torch.tensor(test_cont[row_idxs], dtype=torch.float32)
            cat_tensor = torch.zeros(len(item_indices), 2, dtype=torch.long)
            cat_tensor[:, 0] = uidx
            cat_tensor[:, 1] = torch.tensor(item_indices, dtype=torch.long)

            scores = model(cont_tensor, cat_tensor).squeeze().numpy()
            if scores.ndim == 0:
                scores = np.array([scores.item()])

            for idx_j, (iidx, r, _) in enumerate(items_ratings):
                all_labels.append(1.0 if r >= 1.0 else 0.0)
                all_scores.append(float(scores[idx_j]))

            ranked_idx = np.argsort(-scores)
            top_k_items = [item_indices[j] for j in ranked_idx[:k]]

            ranked_rels = [1.0 if item_indices[j] in relevant else 0.0 for j in ranked_idx]
            ndcgs.append(ndcg_at_k(ranked_rels, k))

            hits_in_k = sum(1 for it in top_k_items if it in relevant)
            precisions.append(hits_in_k / k)

            recalls.append(hits_in_k / len(relevant) if relevant else 0.0)

            hit_rates.append(1.0 if hits_in_k > 0 else 0.0)

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


def main():
    # Seed everything so runs are reproducible to the digit.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 60)
    print("DLRM Training on MovieLens 100K")
    print("=" * 60)

    wandb.init(
        project="recommender-system",
        entity="workwithakin-akin-olusanya",
        config={
            "model": "DLRM",
            "dataset": "MovieLens-100K",
            "num_features": NUM_FEATURES,
            "cardinalities": CARDINALITIES,
            "embedding_dim": EMBEDDING_DIM,
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

    print("\n[1/4] Loading data...")
    raw = load_movielens_data(DATA_PATH)
    print(f"  Total ratings: {len(raw)}")
    splits = prepare_train_val_test(raw)
    train_cont, train_cat, train_targets = splits.train_cont, splits.train_cat, splits.train_targets
    val_cont, val_cat, val_targets = splits.val_cont, splits.val_cat, splits.val_targets
    test_cont, test_cat, test_targets = splits.test_cont, splits.test_cat, splits.test_targets
    user2idx, item2idx = splits.user2idx, splits.item2idx
    val_raw, test_raw = splits.val_raw, splits.test_raw
    print(f"  Train samples: {len(train_targets)}")
    print(f"  Val samples:   {len(val_targets)}")
    print(f"  Test samples:  {len(test_targets)}")
    print(f"  Users: {len(user2idx)}, Items: {len(item2idx)}")

    print("\n[2/4] Building model...")
    model = DLRMModel(
        num_features=NUM_FEATURES,
        cardinalities=CARDINALITIES,
        embedding_dim=EMBEDDING_DIM,
        mlp_layers=MLP_LAYERS,
        dropout=DROPOUT,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture: num_features={NUM_FEATURES}, "
          f"cardinalities={CARDINALITIES}, embedding_dim={EMBEDDING_DIM}, mlp_layers={MLP_LAYERS}")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    train_ds = MovieLensDataset(train_cont, train_cat, train_targets)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    print(f"\n[3/4] Training for {EPOCHS} epochs (batch_size={BATCH_SIZE}, lr={LR})...")
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Validation tensors built once; validation drives all training decisions.
    val_cont_t = torch.tensor(val_cont, dtype=torch.float32)
    val_cat_t = torch.tensor(val_cat, dtype=torch.long)
    val_targets_t = torch.tensor(val_targets, dtype=torch.float32)

    best_loss = float("inf")
    best_state = None
    best_val_ndcg = 0.0
    epochs_no_improve = 0
    early_stopped_epoch = None

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
        train_loss = epoch_loss / n_batches

        # Validation pass: loss drives scheduler / best-model / early stopping.
        model.eval()
        with torch.no_grad():
            val_preds = model(val_cont_t, val_cat_t).squeeze()
            val_loss = criterion(val_preds, val_targets_t).item()
        val_metrics = evaluate(model, val_cont, val_cat, val_targets, user2idx, item2idx, val_raw)
        val_ndcg = val_metrics["NDCG@10"]

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_loss < best_loss:
            best_loss = val_loss
            best_val_ndcg = val_ndcg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ndcg": val_ndcg,
            "epoch": epoch,
            "learning_rate": current_lr,
            "epochs_no_improve": epochs_no_improve,
        })
        print(f"  Epoch {epoch:2d}/{EPOCHS} — train loss: {train_loss:.6f}  val loss: {val_loss:.6f}  "
              f"val NDCG@10: {val_ndcg:.4f}  lr: {current_lr:.6f}  no_improve: {epochs_no_improve}")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            early_stopped_epoch = epoch
            print(f"  Early stopping triggered after {epoch} epochs (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            wandb.log({"early_stopped_epoch": epoch})
            break

    train_time = time.time() - train_start
    print(f"  Training completed in {train_time:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
        print("  Restored best model weights")
    print(f"  Best val loss: {best_loss:.6f}  (val NDCG@10 at best: {best_val_ndcg:.4f})")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"  Model saved to {MODEL_SAVE_PATH}")

    idx2item = {v: k for k, v in item2idx.items()}
    # Serving stats must come from the SAME train portion the model trained on
    # (first `train_ratio - val_ratio` of timestamp-sorted rows) to avoid
    # train/serve skew. This reconstructs prepare_train_val_test's train slice.
    prep_cfg = PrepConfig()
    raw_sorted = raw[np.argsort(raw[:, 3])]
    train_split_serve = int(len(raw_sorted) * (prep_cfg.train_ratio - prep_cfg.val_ratio))
    train_raw_serve = raw_sorted[:train_split_serve]
    user_ratings_serve, max_count_serve, user_timestamps_serve = compute_user_stats(train_raw_serve)
    item_ratings_serve, item_max_count_serve, item_popularity_rank_serve = compute_item_stats(train_raw_serve)
    max_user_days_serve = compute_max_user_days(user_timestamps_serve)

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

    print("\n" + "=" * 60)
    print("INFERENCE LATENCY (1000 runs, single sample)")
    print("=" * 60)
    model.eval()
    dummy_cont = torch.randn(1, NUM_FEATURES)
    dummy_cat = torch.randint(0, 100, (1, 2))

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

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  Model:        DLRM")
    print(f"  Dataset:      MovieLens 100K ({len(raw)} ratings)")
    print(f"  Parameters:   {total_params:,}")
    print(f"  Embeddings:   user({CARDINALITIES[0]}x{EMBEDDING_DIM}), item({CARDINALITIES[1]}x{EMBEDDING_DIM})")
    print(f"  MLP layers:   {MLP_LAYERS}")
    print(f"  Training:     {EPOCHS} epochs in {train_time:.1f}s")
    if early_stopped_epoch is not None:
        print(f"  Early stop:   epoch {early_stopped_epoch}")
    print(f"  Best val loss:{best_loss:.6f}  (val NDCG@10: {best_val_ndcg:.4f})")
    print("  --- TEST (final exam) ---")
    print(f"  NDCG@10:      {metrics['NDCG@10']:.4f}")
    print(f"  Precision@10: {metrics['Precision@10']:.4f}")
    print(f"  Recall@10:    {metrics['Recall@10']:.4f}")
    print(f"  HitRate@10:   {metrics['HitRate@10']:.4f}")
    print(f"  AUC:          {metrics['AUC']:.4f}")
    print(f"  Latency:      {np.mean(latencies):.3f} ms (mean), {np.percentile(latencies, 95):.3f} ms (p95)")
    print("=" * 60)

    wandb.log({
        "ndcg_at_10": metrics["NDCG@10"],
        "precision_at_10": metrics["Precision@10"],
        "recall_at_10": metrics["Recall@10"],
        "hit_rate_at_10": metrics["HitRate@10"],
        "auc": metrics["AUC"],
        "num_eval_users": metrics["num_eval_users"],
    })

    wandb.log({
        "inference_ms_mean": float(np.mean(latencies)),
        "inference_ms_median": float(np.median(latencies)),
        "inference_ms_p95": float(np.percentile(latencies, 95)),
        "inference_ms_p99": float(np.percentile(latencies, 99)),
    })

    artifact = wandb.Artifact("dlrm-movielens", type="model")
    artifact.add_file(MODEL_SAVE_PATH)
    wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()
