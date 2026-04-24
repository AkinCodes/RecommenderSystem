"""Drift detection for the DLRM recommender.

Compares prediction-score distributions between training and test splits
using the Kolmogorov-Smirnov test and KL divergence.  Outputs structured
JSON results and returns a non-zero exit code when drift is detected, so
CI pipelines can gate on it.

Usage:
    python scripts/drift_detection.py
    python scripts/drift_detection.py --output drift_report.json
    python scripts/drift_detection.py --threshold-ks 0.15 --threshold-kl 0.1
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.dlrm import DLRMModel

# ---------------------------------------------------------------------------
# Default thresholds — override via CLI flags
# ---------------------------------------------------------------------------
KS_THRESHOLD = 0.1
KL_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ml-100k", "u.data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_model_movielens.pth")

NUM_FEATURES = 2
EMBEDDING_SIZES = [943, 1682]
MLP_LAYERS = [128, 64, 32]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_and_split():
    """Load MovieLens 100k data, split 80/20 by timestamp, and featurise."""
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

    user_ratings: dict[int, list[int]] = {}
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

    return (
        train_cont, train_cat, train_targets,
        test_cont, test_cat, test_targets,
    )


# ---------------------------------------------------------------------------
# Prediction & metrics
# ---------------------------------------------------------------------------

def get_prediction_scores(model, cont, cat, batch_size=1024):
    """Run model inference in batches and return a 1-D numpy array of scores."""
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
    """Compute KL divergence between two distributions via histograms."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    p_hist, _ = np.histogram(p, bins=bin_edges, density=True)
    q_hist, _ = np.histogram(q, bins=bin_edges, density=True)

    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


def build_report(train_scores, test_scores, ks_threshold, kl_threshold):
    """Return a structured dict with per-metric pass/fail results."""
    ks_stat, ks_pval = stats.ks_2samp(train_scores, test_scores)
    kl_div = kl_divergence(train_scores, test_scores)

    ks_pass = float(ks_stat) <= ks_threshold
    kl_pass = float(kl_div) <= kl_threshold
    overall_pass = ks_pass and kl_pass

    return {
        "overall": "pass" if overall_pass else "fail",
        "thresholds": {
            "ks": ks_threshold,
            "kl": kl_threshold,
        },
        "metrics": {
            "ks_statistic": {
                "value": round(float(ks_stat), 6),
                "p_value": round(float(ks_pval), 6),
                "threshold": ks_threshold,
                "pass": ks_pass,
            },
            "kl_divergence": {
                "value": round(float(kl_div), 6),
                "threshold": kl_threshold,
                "pass": kl_pass,
            },
        },
        "summary": {
            "train_mean": round(float(np.mean(train_scores)), 6),
            "train_std": round(float(np.std(train_scores)), 6),
            "test_mean": round(float(np.mean(test_scores)), 6),
            "test_std": round(float(np.std(test_scores)), 6),
            "train_n": len(train_scores),
            "test_n": len(test_scores),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Detect prediction-distribution drift for the DLRM model."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to write the JSON report (default: stdout only).",
    )
    parser.add_argument(
        "--threshold-ks",
        type=float,
        default=KS_THRESHOLD,
        help=f"KS-statistic threshold for drift (default: {KS_THRESHOLD}).",
    )
    parser.add_argument(
        "--threshold-kl",
        type=float,
        default=KL_THRESHOLD,
        help=f"KL-divergence threshold for drift (default: {KL_THRESHOLD}).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Load data
    (
        train_cont, train_cat, _train_targets,
        test_cont, test_cat, _test_targets,
    ) = load_and_split()

    # Load model
    model = DLRMModel(
        num_features=NUM_FEATURES,
        embedding_sizes=EMBEDDING_SIZES,
        mlp_layers=MLP_LAYERS,
    )
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()

    # Score
    train_scores = get_prediction_scores(model, train_cont, train_cat)
    test_scores = get_prediction_scores(model, test_cont, test_cat)

    # Build report
    report = build_report(
        train_scores, test_scores,
        ks_threshold=args.threshold_ks,
        kl_threshold=args.threshold_kl,
    )

    # Output
    report_json = json.dumps(report, indent=2)
    print(report_json)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report_json + "\n")
        print(f"\nReport written to {args.output}")

    # Exit code: 0 = no drift, 1 = drift detected
    return 0 if report["overall"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
