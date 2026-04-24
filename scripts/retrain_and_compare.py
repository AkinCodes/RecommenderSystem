"""Retrain DLRM with 8-feature architecture and compare against classical baselines.

Trains the DLRM locally (no wandb), evaluates it alongside XGBoost, LightGBM,
and Logistic Regression using ranking metrics, and outputs a markdown comparison
table plus a JSON report.

Usage:
    python scripts/retrain_and_compare.py
"""

import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

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
from models.classical import ClassicalRanker, FeatureBuilder
from models.dlrm import DLRMModel

# ---------------------------------------------------------------------------
# Config (same hyperparameters as train_movielens.py)
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "ml-100k", "u.data")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "trained_model_movielens.pth")
REPORT_JSON_PATH = os.path.join(PROJECT_ROOT, "reports", "model_comparison.json")

EMBEDDING_SIZES = [943, 1682]
MLP_LAYERS = [128, 64, 32]
EPOCHS = 20
BATCH_SIZE = 256
LR = 0.001
WEIGHT_DECAY = 1e-5
DROPOUT = 0.2
EARLY_STOP_PATIENCE = 5
DEVICE = "cpu"
K = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
np.random.seed(42)
torch.manual_seed(42)


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


def compute_ranking_metrics(user_test, rank_fn, k=K):
    """Compute NDCG@K, Precision@K, Recall@K, HitRate@K, and AUC."""
    ndcgs, precisions, recalls, hit_rates = [], [], [], []
    all_labels, all_scores = [], []

    for uidx, items_ratings in user_test.items():
        if len(items_ratings) < 2:
            continue
        relevant = {iidx for iidx, r in items_ratings if r >= 1.0}
        if len(relevant) == 0:
            continue

        item_indices = [ir[0] for ir in items_ratings]
        scores = rank_fn(uidx, item_indices)
        if hasattr(scores, 'numpy'):
            scores = scores.numpy()
        if scores.ndim == 0:
            scores = np.array([scores.item()])

        # For AUC
        for idx_j, (iidx, r) in enumerate(items_ratings):
            all_labels.append(1.0 if r >= 1.0 else 0.0)
            all_scores.append(float(scores[idx_j]))

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

        # HitRate@K
        hit_rates.append(1.0 if hits_in_k > 0 else 0.0)

    auc = 0.0
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_scores)

    return {
        "NDCG@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "Precision@10": float(np.mean(precisions)) if precisions else 0.0,
        "Recall@10": float(np.mean(recalls)) if recalls else 0.0,
        "HitRate@10": float(np.mean(hit_rates)) if hit_rates else 0.0,
        "AUC": float(auc),
        "num_eval_users": len(ndcgs),
    }


def measure_inference_latency(rank_fn, user_test, n_runs=200):
    """Measure average per-user inference time in milliseconds."""
    users = list(user_test.keys())[:n_runs]
    if not users:
        return 0.0

    # Warmup
    for uidx in users[:10]:
        items = [ir[0] for ir in user_test[uidx]]
        rank_fn(uidx, items)

    total_ms = 0.0
    count = 0
    for uidx in users:
        items = [ir[0] for ir in user_test[uidx]]
        t0 = time.perf_counter()
        rank_fn(uidx, items)
        total_ms += (time.perf_counter() - t0) * 1000.0
        count += 1

    return total_ms / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# DLRM Training
# ---------------------------------------------------------------------------
def train_dlrm(train_cont, train_cat, train_targets,
               test_cont, test_cat, test_targets,
               user2idx, item2idx, test_raw, raw):
    """Train the DLRM model and return (model, metrics, train_time, param_count)."""
    print("\n" + "=" * 60)
    print("TRAINING DLRM (8-feature architecture, no wandb)")
    print("=" * 60)

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

    train_ds = MovieLensDataset(train_cont, train_cat, train_targets)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    print(f"\n  Training for {EPOCHS} epochs (batch_size={BATCH_SIZE}, lr={LR})...")
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

        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"  Epoch {epoch:2d}/{EPOCHS} -- loss: {avg_loss:.6f}  "
              f"lr: {current_lr:.6f}  no_improve: {epochs_no_improve}")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping after {epoch} epochs")
            break

    train_time = time.time() - train_start
    print(f"  Training completed in {train_time:.1f}s")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print("  Restored best model weights")

    # Save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"  Model saved to {MODEL_SAVE_PATH}")

    # Save serving context
    idx2item = {v: k for k, v in item2idx.items()}
    sorted_idx = np.argsort(raw[:, 3])
    raw_sorted = raw[sorted_idx]
    split = int(len(raw_sorted) * 0.8)
    train_raw_serve = raw_sorted[:split]

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

    return model, train_time, total_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("RETRAIN & COMPARE -- DLRM vs Classical Baselines")
    print(f"NUM_FEATURES = {NUM_FEATURES}, targets = binary (BCELoss)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading MovieLens 100K data...")
    raw = load_movielens_data(DATA_PATH)
    (train_cont, train_cat, train_targets,
     test_cont, test_cat, test_targets,
     user2idx, item2idx, test_raw) = prepare_splits(raw)

    sorted_idx = np.argsort(raw[:, 3])
    raw_sorted = raw[sorted_idx]
    split = int(len(raw_sorted) * 0.8)
    train_raw = raw_sorted[:split]

    print(f"  Total ratings: {len(raw)}")
    print(f"  Train: {len(train_raw)}, Test: {len(test_raw)}")
    print(f"  Users: {len(user2idx)}, Items: {len(item2idx)}")
    print(f"  Label balance: {train_targets.mean():.1%} positive (rating >= 4)")

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

    # ------------------------------------------------------------------
    # 2. Train DLRM
    # ------------------------------------------------------------------
    print("\n[2/5] Training DLRM...")
    model, dlrm_train_time, dlrm_param_count = train_dlrm(
        train_cont, train_cat, train_targets,
        test_cont, test_cat, test_targets,
        user2idx, item2idx, test_raw, raw,
    )

    # DLRM ranking function
    def dlrm_rank(uidx, item_indices):
        n = len(item_indices)
        cont_tensor = torch.tensor(
            np.tile(user_cont_map[uidx], (n, 1)), dtype=torch.float32
        )
        cat_tensor = torch.zeros(n, 2, dtype=torch.long)
        cat_tensor[:, 0] = uidx
        cat_tensor[:, 1] = torch.tensor(item_indices, dtype=torch.long)
        with torch.no_grad():
            scores = model(cont_tensor, cat_tensor).squeeze().numpy()
        if scores.ndim == 0:
            scores = np.array([scores.item()])
        return scores

    print("\n[3/5] Evaluating DLRM...")
    dlrm_metrics = compute_ranking_metrics(user_test, dlrm_rank, k=K)
    dlrm_latency = measure_inference_latency(dlrm_rank, user_test)
    dlrm_metrics["train_time"] = dlrm_train_time
    dlrm_metrics["latency_ms"] = dlrm_latency
    dlrm_metrics["param_count"] = dlrm_param_count

    print(f"  NDCG@10:    {dlrm_metrics['NDCG@10']:.4f}")
    print(f"  Prec@10:    {dlrm_metrics['Precision@10']:.4f}")
    print(f"  Recall@10:  {dlrm_metrics['Recall@10']:.4f}")
    print(f"  HR@10:      {dlrm_metrics['HitRate@10']:.4f}")
    print(f"  AUC:        {dlrm_metrics['AUC']:.4f}")

    # ------------------------------------------------------------------
    # 3. Train classical baselines
    # ------------------------------------------------------------------
    print("\n[4/5] Training classical baselines...")
    fb = FeatureBuilder(train_raw, user2idx, item2idx)
    X_train, y_train = fb.build_matrix(train_raw)
    X_test, y_test = fb.build_matrix(test_raw)
    print(f"  Feature matrix: {X_train.shape[1]} features, "
          f"{X_train.shape[0]} train, {X_test.shape[0]} test")

    results = [("DLRM", dlrm_metrics)]

    # --- XGBoost ---
    try:
        import xgboost as xgb

        print("\n  Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        t0 = time.time()
        xgb_model.fit(X_train, y_train)
        xgb_time = time.time() - t0
        print(f"  XGBoost trained in {xgb_time:.1f}s")

        xgb_ranker = ClassicalRanker(xgb_model, fb, name="XGBoost")
        xgb_metrics = compute_ranking_metrics(user_test, xgb_ranker.rank, k=K)
        xgb_metrics["train_time"] = xgb_time
        xgb_metrics["latency_ms"] = measure_inference_latency(xgb_ranker.rank, user_test)
        results.append(("XGBoost", xgb_metrics))
        print(f"  XGBoost -- NDCG@10: {xgb_metrics['NDCG@10']:.4f}, "
              f"AUC: {xgb_metrics['AUC']:.4f}")
    except ImportError:
        print("  [SKIP] xgboost not installed")

    # --- LightGBM ---
    try:
        import lightgbm as lgb

        print("\n  Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        t0 = time.time()
        lgb_model.fit(X_train, y_train)
        lgb_time = time.time() - t0
        print(f"  LightGBM trained in {lgb_time:.1f}s")

        lgb_ranker = ClassicalRanker(lgb_model, fb, name="LightGBM")
        lgb_metrics = compute_ranking_metrics(user_test, lgb_ranker.rank, k=K)
        lgb_metrics["train_time"] = lgb_time
        lgb_metrics["latency_ms"] = measure_inference_latency(lgb_ranker.rank, user_test)
        results.append(("LightGBM", lgb_metrics))
        print(f"  LightGBM -- NDCG@10: {lgb_metrics['NDCG@10']:.4f}, "
              f"AUC: {lgb_metrics['AUC']:.4f}")
    except ImportError:
        print("  [SKIP] lightgbm not installed")

    # --- Logistic Regression ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    print("\n  Training Logistic Regression...")
    logreg_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])
    t0 = time.time()
    logreg_pipeline.fit(X_train, y_train)
    logreg_time = time.time() - t0
    print(f"  LogReg trained in {logreg_time:.1f}s")

    logreg_ranker = ClassicalRanker(logreg_pipeline, fb, name="LogReg")
    logreg_metrics = compute_ranking_metrics(user_test, logreg_ranker.rank, k=K)
    logreg_metrics["train_time"] = logreg_time
    logreg_metrics["latency_ms"] = measure_inference_latency(logreg_ranker.rank, user_test)
    results.append(("LogReg", logreg_metrics))
    print(f"  LogReg -- NDCG@10: {logreg_metrics['NDCG@10']:.4f}, "
          f"AUC: {logreg_metrics['AUC']:.4f}")

    # ------------------------------------------------------------------
    # 4. Print comparison table
    # ------------------------------------------------------------------
    print("\n\n[5/5] Results")
    print("=" * 70)
    print()

    # Markdown table
    header = "| Model | NDCG@10 | Prec@10 | Recall@10 | HR@10 | AUC | Train Time | Latency (ms) |"
    sep =    "|-------|---------|---------|-----------|-------|-----|------------|--------------|"
    print(header)
    print(sep)

    for name, m in results:
        tt = f"{m['train_time']:.1f}s" if m.get("train_time") is not None else "-"
        lat = f"{m['latency_ms']:.2f}" if m.get("latency_ms") is not None else "-"
        print(
            f"| {name:<9} | {m['NDCG@10']:.4f}  | {m['Precision@10']:.4f}  "
            f"| {m['Recall@10']:.4f}    | {m['HitRate@10']:.4f} "
            f"| {m['AUC']:.4f} | {tt:>10} | {lat:>12} |"
        )

    print()
    print(f"DLRM parameter count: {dlrm_param_count:,}")
    print(f"Evaluated on {dlrm_metrics['num_eval_users']} users with top-{K} recommendations.")
    print()

    # ------------------------------------------------------------------
    # 5. Save JSON report
    # ------------------------------------------------------------------
    report = {
        "dataset": "MovieLens-100K",
        "num_ratings": int(len(raw)),
        "num_users": len(user2idx),
        "num_items": len(item2idx),
        "train_size": int(len(train_raw)),
        "test_size": int(len(test_raw)),
        "num_features": NUM_FEATURES,
        "dlrm_config": {
            "embedding_sizes": EMBEDDING_SIZES,
            "mlp_layers": MLP_LAYERS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "param_count": dlrm_param_count,
        },
        "models": {},
    }

    for name, m in results:
        report["models"][name] = {
            "NDCG@10": m["NDCG@10"],
            "Precision@10": m["Precision@10"],
            "Recall@10": m["Recall@10"],
            "HitRate@10": m["HitRate@10"],
            "AUC": m["AUC"],
            "num_eval_users": m["num_eval_users"],
            "train_time_s": m.get("train_time"),
            "latency_ms": m.get("latency_ms"),
        }

    os.makedirs(os.path.dirname(REPORT_JSON_PATH), exist_ok=True)
    with open(REPORT_JSON_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Results saved to {REPORT_JSON_PATH}")
    print(f"Trained model saved to {MODEL_SAVE_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
