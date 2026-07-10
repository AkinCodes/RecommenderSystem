"""Retrain DLRM (no wandb) and compare against classical baselines."""

import json
import logging
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from data.preprocessing import (
    MAX_RATING_VAR,
    NUM_FEATURES,
    RATING_MAX,
    PrepConfig,
    Splits,
    compute_item_stats,
    compute_max_user_days,
    compute_user_stats,
    load_movielens_data,
    prepare_train_val_test,
)
from models.classical import ClassicalRanker, FeatureBuilder
from models.dlrm import DLRMModel

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "ml-100k", "u.data")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "trained_model_movielens.pth")
REPORT_JSON_PATH = os.path.join(PROJECT_ROOT, "reports", "model_comparison.json")

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class MovieLensDataset(Dataset):
    def __init__(self, cont, cat, targets):
        self.cont = torch.tensor(cont, dtype=torch.float32)
        self.cat = torch.tensor(cat, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.cont[idx], self.cat[idx], self.targets[idx]


def paired_bootstrap_ci(scores_a, scores_b, n=10000, seed=SEED):
    """Paired bootstrap 95% CI on the mean delta (a - b) + permutation p-value.

    ``scores_a`` and ``scores_b`` are per-user metric vectors ALIGNED by user
    (element i is the same user under model A and model B), so we resample and
    permute the paired deltas — this respects the fact that both models are
    scored on the same held-out users.

    Returns a dict with:
      - ``mean_delta``: observed mean(a) - mean(b)
      - ``ci_low`` / ``ci_high``: 95% bootstrap CI on that mean delta
      - ``p_value``: two-sided paired permutation test p-value
      - ``n_users``: number of paired observations
    Seeded (default ``SEED=42``) so results are reproducible to the digit.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"scores must be aligned/same length: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("empty score vectors")

    diffs = a - b
    n_users = diffs.size
    mean_delta = float(diffs.mean())

    rng = np.random.default_rng(seed)

    # Bootstrap: resample paired deltas with replacement, recompute mean delta.
    boot_idx = rng.integers(0, n_users, size=(n, n_users))
    boot_means = diffs[boot_idx].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    # Paired permutation test: under H0 the sign of each paired delta is
    # exchangeable, so flip signs at random and count how often the permuted
    # |mean delta| meets or exceeds the observed |mean delta|.
    obs = abs(mean_delta)
    signs = rng.choice([-1.0, 1.0], size=(n, n_users))
    perm_means = np.abs((signs * diffs).mean(axis=1))
    # +1 numerator/denominator: the observed arrangement is itself one valid
    # permutation, which keeps the p-value from ever being exactly 0.
    p_value = float((np.sum(perm_means >= obs) + 1) / (n + 1))

    return {
        "mean_delta": mean_delta,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
        "n_users": int(n_users),
    }


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
    ndcgs, precisions, recalls, hit_rates = [], [], [], []
    all_labels, all_scores = [], []
    # Retain the per-user NDCG vector keyed by uidx so the same users can be
    # paired ACROSS models (bootstrap CIs / significance tests downstream).
    per_user_ndcg = {}

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

        for idx_j, (iidx, r) in enumerate(items_ratings):
            all_labels.append(1.0 if r >= 1.0 else 0.0)
            all_scores.append(float(scores[idx_j]))

        ranked_idx = np.argsort(-scores)
        top_k_items = [item_indices[j] for j in ranked_idx[:k]]

        ranked_rels = [1.0 if item_indices[j] in relevant else 0.0 for j in ranked_idx]
        user_ndcg = ndcg_at_k(ranked_rels, k)
        ndcgs.append(user_ndcg)
        per_user_ndcg[uidx] = user_ndcg

        hits_in_k = sum(1 for it in top_k_items if it in relevant)
        precisions.append(hits_in_k / k)
        recalls.append(hits_in_k / len(relevant) if relevant else 0.0)
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
        "per_user_ndcg": per_user_ndcg,
    }


def measure_inference_latency(rank_fn, user_test, n_runs=200):
    users = list(user_test.keys())[:n_runs]
    if not users:
        return 0.0

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


def train_dlrm(train_cont, train_cat, train_targets,
               val_cont, val_cat, val_targets,
               user2idx, item2idx, raw):
    # Validation-driven training regime, mirroring scripts/train_movielens.py:
    # the DLRM trains on the 0.70 TRAIN slice only; the 0.10 VAL slice drives the
    # scheduler / best-model selection / early stopping (never used as training
    # data). The 0.20 TEST slice is untouched here and reserved for evaluation,
    # so every model (DLRM + classicals) is scored on the SAME held-out test set.
    print("\n" + "=" * 60)
    print("TRAINING DLRM (8-feature architecture, validation-driven, no wandb)")
    print("=" * 60)

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

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Validation tensors built once; VAL loss drives every training decision
    # (scheduler step, best-model selection, early stopping) — same recipe and
    # settings as scripts/train_movielens.py.
    val_cont_t = torch.tensor(val_cont, dtype=torch.float32)
    val_cat_t = torch.tensor(val_cat, dtype=torch.long)
    val_targets_t = torch.tensor(val_targets, dtype=torch.float32)

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
        train_loss = epoch_loss / n_batches

        # Validation pass (eval() + no_grad): val loss drives scheduler /
        # best-model / early stopping, exactly as in train_movielens.py.
        model.eval()
        with torch.no_grad():
            val_preds = model(val_cont_t, val_cat_t).squeeze()
            val_loss = criterion(val_preds, val_targets_t).item()

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"  Epoch {epoch:2d}/{EPOCHS} -- train loss: {train_loss:.6f}  "
              f"val loss: {val_loss:.6f}  lr: {current_lr:.6f}  no_improve: {epochs_no_improve}")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping after {epoch} epochs")
            break

    train_time = time.time() - train_start
    print(f"  Training completed in {train_time:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best model weights (best val loss: {best_loss:.6f})")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"  Model saved to {MODEL_SAVE_PATH}")

    idx2item = {v: k for k, v in item2idx.items()}
    # Serving stats must come from the SAME 0.70 train portion the model trained
    # on (first `train_ratio - val_ratio` of timestamp-sorted rows), matching
    # prepare_train_val_test's train slice, to avoid train/serve skew.
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

    return model, train_time, total_params


def main():
    # Re-seed at the start of main (in addition to module-level seeding) so
    # results are reproducible to the digit, mirroring train_movielens.py.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("RETRAIN & COMPARE -- DLRM vs Classical Baselines")
    print(f"NUM_FEATURES = {NUM_FEATURES}, targets = binary (BCELoss)")
    print("=" * 70)

    print("\n[1/5] Loading MovieLens 100K data...")
    raw = load_movielens_data(DATA_PATH)
    # Strict 3-way split (0.70 train / 0.10 val / 0.20 test), same as the
    # production trainer scripts/train_movielens.py. The DLRM trains on TRAIN
    # and early-stops on VAL; classicals train on the SAME TRAIN slice; ALL
    # models are evaluated on the SAME UNCHANGED TEST slice (0.20).
    splits: Splits = prepare_train_val_test(raw)
    train_cont, train_cat, train_targets = splits.train_cont, splits.train_cat, splits.train_targets
    val_cont, val_cat, val_targets = splits.val_cont, splits.val_cat, splits.val_targets
    test_cont, test_cat, test_targets = splits.test_cont, splits.test_cat, splits.test_targets
    user2idx, item2idx = splits.user2idx, splits.item2idx
    test_raw = splits.test_raw

    # Reconstruct the 0.70 TRAIN raw rows (matching prepare_train_val_test) so the
    # classical models train on the EXACT same interactions the DLRM trains on.
    prep_cfg = PrepConfig()
    raw_sorted = raw[np.argsort(raw[:, 3])]
    train_split = int(len(raw_sorted) * (prep_cfg.train_ratio - prep_cfg.val_ratio))
    train_raw = raw_sorted[:train_split]

    print(f"  Total ratings: {len(raw)}")
    print(f"  Train: {len(train_raw)} (0.70), Val: {len(val_targets)} (0.10), "
          f"Test: {len(test_raw)} (0.20)")
    print(f"  Users: {len(user2idx)}, Items: {len(item2idx)}")
    print(f"  Label balance: {train_targets.mean():.1%} positive (rating >= 4)")

    # Track each candidate's source ROW index (parallel to item_indices order)
    # so DLRM can score every candidate with ITS OWN continuous features.
    user_test = {}
    user_rows = {}
    for i in range(len(test_raw)):
        uid = test_raw[i, 0]
        uidx = user2idx[uid]
        iidx = int(test_cat[i, 1])
        rating = test_targets[i]
        user_test.setdefault(uidx, []).append((iidx, rating))
        user_rows.setdefault(uidx, []).append(i)

    print("\n[2/5] Training DLRM...")
    model, dlrm_train_time, dlrm_param_count = train_dlrm(
        train_cont, train_cat, train_targets,
        val_cont, val_cat, val_targets,
        user2idx, item2idx, raw,
    )

    def dlrm_rank(uidx, item_indices):
        n = len(item_indices)
        # Score each candidate with its OWN continuous row (item_indices and
        # user_rows[uidx] are built in the same order), not one tiled row.
        cont_tensor = torch.tensor(
            test_cont[user_rows[uidx]], dtype=torch.float32
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

    print("\n[4/5] Training classical baselines...")
    fb = FeatureBuilder(train_raw, user2idx, item2idx)
    X_train, y_train = fb.build_matrix(train_raw)
    X_test, y_test = fb.build_matrix(test_raw)
    print(f"  Feature matrix: {X_train.shape[1]} features, "
          f"{X_train.shape[0]} train, {X_test.shape[0]} test")

    results = [("DLRM", dlrm_metrics)]

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
            n_jobs=1,  # single-thread => deterministic, environment-independent
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
            n_jobs=1,  # single-thread => deterministic, environment-independent
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

    print("\n\n[5/5] Results")
    print("=" * 70)
    print()

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

    # ------------------------------------------------------------------ #
    # Paired bootstrap CIs + permutation p-values on per-user NDCG@10.
    # A 0.0076 point-estimate gap on ~283 users means nothing without an
    # interval / significance test — so pair every model against the DLRM
    # (and rank the models) on the SAME users and report the truth.
    # ------------------------------------------------------------------ #
    print("Paired bootstrap 95% CI + permutation p-value on per-user NDCG@10")
    print("(reference model = DLRM; positive delta => the OTHER model is better)")
    print("=" * 70)
    ci_header = "| Comparison        | Mean Δ NDCG@10 | 95% CI            | p-value | Verdict |"
    ci_sep =    "|-------------------|----------------|-------------------|---------|---------|"
    print(ci_header)
    print(ci_sep)

    dlrm_ndcg = dlrm_metrics["per_user_ndcg"]
    comparisons = {}
    for name, m in results:
        if name == "DLRM":
            continue
        other_ndcg = m["per_user_ndcg"]
        # Pair on the intersection of eval users (all models share the same
        # user_test, so this is the full 283-user set in practice).
        shared = sorted(set(dlrm_ndcg) & set(other_ndcg))
        a = np.array([other_ndcg[u] for u in shared])  # candidate model
        b = np.array([dlrm_ndcg[u] for u in shared])   # DLRM reference
        stat = paired_bootstrap_ci(a, b)
        tie = stat["ci_low"] <= 0.0 <= stat["ci_high"]
        verdict = "tie" if tie else (f"{name} wins" if stat["mean_delta"] > 0 else "DLRM wins")
        comparisons[f"{name}_vs_DLRM"] = stat
        label = f"{name} - DLRM"
        print(
            f"| {label:<17} | {stat['mean_delta']:+.4f}        "
            f"| [{stat['ci_low']:+.4f}, {stat['ci_high']:+.4f}] "
            f"| {stat['p_value']:.4f}  | {verdict:<7} |"
        )
    print()

    report = {
        "dataset": "MovieLens-100K",
        "num_ratings": int(len(raw)),
        "num_users": len(user2idx),
        "num_items": len(item2idx),
        "train_size": int(len(train_raw)),
        "val_size": int(len(val_targets)),
        "test_size": int(len(test_raw)),
        "num_features": NUM_FEATURES,
        "dlrm_config": {
            "cardinalities": CARDINALITIES,
            "embedding_dim": EMBEDDING_DIM,
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

    # Persist the paired significance results (keyed <model>_vs_DLRM) so the
    # CI / p-value numbers in EVALUATION.md are traceable to a real run.
    report["significance"] = {
        "metric": "NDCG@10",
        "method": "paired bootstrap 95% CI + two-sided paired permutation test",
        "n_bootstrap": 10000,
        "n_permutations": 10000,
        "seed": SEED,
        "reference_model": "DLRM",
        "comparisons": comparisons,
    }

    os.makedirs(os.path.dirname(REPORT_JSON_PATH), exist_ok=True)
    with open(REPORT_JSON_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Results saved to {REPORT_JSON_PATH}")
    print(f"Trained model saved to {MODEL_SAVE_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
