"""Drift detection stub/proof-of-concept for the DLRM recommender.

This script demonstrates how distribution drift in model predictions
could be detected in a production setting. It compares the distribution
of prediction scores on the training set vs a simulated "new" batch
using statistical tests.
"""

import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.dlrm import DLRMModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ml-100k", "u.data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_model_movielens.pth")

NUM_FEATURES = 2
EMBEDDING_SIZES = [943, 1682]
MLP_LAYERS = [128, 64, 32]


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
            user2idx, item2idx)


def get_prediction_scores(model, cont, cat, batch_size=1024):
    """Get model prediction scores for a dataset."""
    model.eval()
    all_scores = []
    n = len(cont)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            cont_t = torch.tensor(cont[start:end], dtype=torch.float32)
            cat_t = torch.tensor(cat[start:end], dtype=torch.long)
            scores = model(cont_t, cat_t).squeeze().numpy()
            if scores.ndim == 0:
                scores = np.array([scores.item()])
            all_scores.append(scores)
    return np.concatenate(all_scores)


def kl_divergence(p, q, n_bins=50):
    """Compute KL divergence between two distributions using histograms."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    p_hist, _ = np.histogram(p, bins=bin_edges, density=True)
    q_hist, _ = np.histogram(q, bins=bin_edges, density=True)

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps

    # Normalize to proper distributions
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    return np.sum(p_hist * np.log(p_hist / q_hist))


def main():
    print("=" * 60)
    print("DRIFT DETECTION — Proof of Concept")
    print("=" * 60)

    print("\n[1] Loading data and model...")
    (train_raw, test_raw,
     train_cont, train_cat, train_targets,
     test_cont, test_cat, test_targets,
     user2idx, item2idx) = load_and_split()

    model = DLRMModel(num_features=NUM_FEATURES, embedding_sizes=EMBEDDING_SIZES, mlp_layers=MLP_LAYERS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    print("  Model loaded.")

    # -------------------------------------------------------------------
    # Scenario A: Train vs Test (natural temporal drift)
    # -------------------------------------------------------------------
    print("\n[2] Computing prediction scores on train and test sets...")
    train_scores = get_prediction_scores(model, train_cont, train_cat)
    test_scores = get_prediction_scores(model, test_cont, test_cat)

    print(f"  Train scores: mean={np.mean(train_scores):.4f}, "
          f"std={np.std(train_scores):.4f}, "
          f"min={np.min(train_scores):.4f}, max={np.max(train_scores):.4f}")
    print(f"  Test scores:  mean={np.mean(test_scores):.4f}, "
          f"std={np.std(test_scores):.4f}, "
          f"min={np.min(test_scores):.4f}, max={np.max(test_scores):.4f}")

    # KS test
    ks_stat, ks_pval = stats.ks_2samp(train_scores, test_scores)
    print(f"\n  Kolmogorov-Smirnov test (train vs test):")
    print(f"    KS statistic: {ks_stat:.4f}")
    print(f"    p-value:      {ks_pval:.6f}")

    # KL divergence
    kl_div = kl_divergence(train_scores, test_scores)
    print(f"\n  KL divergence (train || test): {kl_div:.4f}")

    if ks_pval < 0.05:
        print("\n  >> DRIFT DETECTED: The prediction score distributions differ "
              "significantly between train and test (p < 0.05).")
        print("     This is expected with a temporal split — user behavior and ")
        print("     item popularity shift over time.")
    else:
        print("\n  >> NO SIGNIFICANT DRIFT: Prediction distributions are similar "
              "between train and test.")

    # -------------------------------------------------------------------
    # Scenario B: Test vs Simulated "corrupted" batch (synthetic drift)
    # -------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[3] Simulating drift with a corrupted input batch...")

    # Create a synthetic batch with perturbed continuous features
    np.random.seed(42)
    corrupted_cont = test_cont.copy()
    # Add noise to simulate feature drift (e.g., data pipeline issue)
    corrupted_cont += np.random.normal(0, 0.3, corrupted_cont.shape).astype(np.float32)
    corrupted_cont = np.clip(corrupted_cont, 0, 1)

    corrupted_scores = get_prediction_scores(model, corrupted_cont, test_cat)

    print(f"  Corrupted scores: mean={np.mean(corrupted_scores):.4f}, "
          f"std={np.std(corrupted_scores):.4f}, "
          f"min={np.min(corrupted_scores):.4f}, max={np.max(corrupted_scores):.4f}")

    ks_stat2, ks_pval2 = stats.ks_2samp(test_scores, corrupted_scores)
    kl_div2 = kl_divergence(test_scores, corrupted_scores)

    print(f"\n  Kolmogorov-Smirnov test (test vs corrupted):")
    print(f"    KS statistic: {ks_stat2:.4f}")
    print(f"    p-value:      {ks_pval2:.6f}")
    print(f"\n  KL divergence (test || corrupted): {kl_div2:.4f}")

    if ks_pval2 < 0.05:
        print("\n  >> DRIFT DETECTED: The corrupted batch shows significantly "
              "different prediction distributions.")
        print("     In production, this would trigger an alert for investigation.")
    else:
        print("\n  >> NO SIGNIFICANT DRIFT detected with the corrupted batch.")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
