"""Classical ML baselines vs DLRM on MovieLens 100K.

Trains XGBoost, LightGBM, and Logistic Regression classifiers with rich
hand-crafted features, then evaluates all models (including the saved DLRM)
using the same ranking metrics (NDCG@10, Precision@10, HitRate@10).

Usage:
    python scripts/classical_baselines.py
"""

import logging
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from data.preprocessing import NUM_FEATURES, load_movielens_data, prepare_splits  # noqa: E402
from scripts.train_movielens import EMBEDDING_SIZES, MLP_LAYERS  # noqa: E402

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "ml-100k", "u.data")
from models.classical import ClassicalRanker, FeatureBuilder  # noqa: E402
from scripts.baseline_comparison import compute_metrics  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "trained_model_movielens.pth")
REPORT_PATH = os.path.join(PROJECT_ROOT, "CLASSICAL_COMPARISON.md")
K = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

np.random.seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _measure_inference_latency(rank_fn, user_test, n_runs=200):
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
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("CLASSICAL ML BASELINES — MovieLens 100K")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data (reuse the same pipeline as DLRM)
    # ------------------------------------------------------------------
    logger.info("Loading MovieLens 100K data...")
    raw = load_movielens_data(DATA_PATH)
    (train_cont, train_cat, train_targets,
     test_cont, test_cat, test_targets,
     user2idx, item2idx, test_raw) = prepare_splits(raw)

    # We also need train_raw for feature engineering
    sorted_idx = np.argsort(raw[:, 3])
    raw_sorted = raw[sorted_idx]
    split = int(len(raw_sorted) * 0.8)
    train_raw = raw_sorted[:split]

    logger.info("Train: %d, Test: %d, Users: %d, Items: %d",
                len(train_raw), len(test_raw), len(user2idx), len(item2idx))

    # ------------------------------------------------------------------
    # 2. Build rich features
    # ------------------------------------------------------------------
    logger.info("Building feature matrices with %d features...", FeatureBuilder.NUM_FEATURES)
    fb = FeatureBuilder(train_raw, user2idx, item2idx)
    X_train, y_train = fb.build_matrix(train_raw)
    X_test, y_test = fb.build_matrix(test_raw)

    pos_rate = y_train.mean()
    logger.info("Training label balance: %.1f%% positive (rating >= 4)", pos_rate * 100)
    logger.info("Feature matrix shape: train=%s, test=%s", X_train.shape, X_test.shape)

    # ------------------------------------------------------------------
    # 3. Group test interactions by user (for ranking evaluation)
    # ------------------------------------------------------------------
    user_test = {}
    for i in range(len(test_raw)):
        uid = test_raw[i, 0]
        uidx = user2idx[uid]
        iidx = int(test_cat[i, 1])
        rating = test_targets[i]  # normalised to [0,1]
        user_test.setdefault(uidx, []).append((iidx, rating))

    user_cont_map = {}
    for i in range(len(test_raw)):
        uidx = int(test_cat[i, 0])
        if uidx not in user_cont_map:
            user_cont_map[uidx] = test_cont[i]

    # ------------------------------------------------------------------
    # 4. Train classical models
    # ------------------------------------------------------------------
    results = []

    # --- XGBoost ---
    logger.info("Training XGBoost...")
    try:
        import xgboost as xgb

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
        xgb_train_time = time.time() - t0
        logger.info("XGBoost trained in %.1fs", xgb_train_time)

        xgb_ranker = ClassicalRanker(xgb_model, fb, name="XGBoost")
        xgb_metrics = compute_metrics(user_test, xgb_ranker.rank, k=K)
        xgb_latency = _measure_inference_latency(xgb_ranker.rank, user_test)
        xgb_metrics["train_time"] = xgb_train_time
        xgb_metrics["latency_ms"] = xgb_latency
        results.append(("XGBoost", xgb_metrics))
        logger.info("XGBoost — NDCG@10: %.4f, Prec@10: %.4f, Hit@10: %.4f",
                     xgb_metrics["NDCG@10"], xgb_metrics["Precision@10"], xgb_metrics["HitRate@10"])
    except ImportError:
        logger.warning("xgboost not installed — skipping XGBoost baseline")

    # --- LightGBM ---
    logger.info("Training LightGBM...")
    try:
        import lightgbm as lgb

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
        lgb_train_time = time.time() - t0
        logger.info("LightGBM trained in %.1fs", lgb_train_time)

        lgb_ranker = ClassicalRanker(lgb_model, fb, name="LightGBM")
        lgb_metrics = compute_metrics(user_test, lgb_ranker.rank, k=K)
        lgb_latency = _measure_inference_latency(lgb_ranker.rank, user_test)
        lgb_metrics["train_time"] = lgb_train_time
        lgb_metrics["latency_ms"] = lgb_latency
        results.append(("LightGBM", lgb_metrics))
        logger.info("LightGBM — NDCG@10: %.4f, Prec@10: %.4f, Hit@10: %.4f",
                     lgb_metrics["NDCG@10"], lgb_metrics["Precision@10"], lgb_metrics["HitRate@10"])
    except ImportError:
        logger.warning("lightgbm not installed — skipping LightGBM baseline")

    # --- Logistic Regression (with StandardScaler pipeline) ---
    logger.info("Training Logistic Regression...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

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
    logreg_train_time = time.time() - t0
    logger.info("LogReg trained in %.1fs", logreg_train_time)

    logreg_ranker = ClassicalRanker(logreg_pipeline, fb, name="LogReg")
    logreg_metrics = compute_metrics(user_test, logreg_ranker.rank, k=K)
    logreg_latency = _measure_inference_latency(logreg_ranker.rank, user_test)
    logreg_metrics["train_time"] = logreg_train_time
    logreg_metrics["latency_ms"] = logreg_latency
    results.append(("LogReg", logreg_metrics))
    logger.info("LogReg — NDCG@10: %.4f, Prec@10: %.4f, Hit@10: %.4f",
                 logreg_metrics["NDCG@10"], logreg_metrics["Precision@10"], logreg_metrics["HitRate@10"])

    # ------------------------------------------------------------------
    # 5. Evaluate DLRM
    # ------------------------------------------------------------------
    logger.info("Loading saved DLRM model from %s...", MODEL_SAVE_PATH)
    import torch

    from models.dlrm import DLRMModel

    dlrm_metrics = {"NDCG@10": 0.0, "Precision@10": 0.0, "HitRate@10": 0.0,
                    "train_time": None, "latency_ms": None, "num_eval_users": 0}

    if os.path.exists(MODEL_SAVE_PATH):
        model = DLRMModel(
            num_features=NUM_FEATURES,
            embedding_sizes=EMBEDDING_SIZES,
            mlp_layers=MLP_LAYERS,
        )
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu", weights_only=True))
        model.eval()

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

        dlrm_metrics = compute_metrics(user_test, dlrm_rank, k=K)
        dlrm_latency = _measure_inference_latency(dlrm_rank, user_test)
        dlrm_metrics["train_time"] = None  # was trained separately
        dlrm_metrics["latency_ms"] = dlrm_latency
        logger.info("DLRM — NDCG@10: %.4f, Prec@10: %.4f, Hit@10: %.4f",
                     dlrm_metrics["NDCG@10"], dlrm_metrics["Precision@10"], dlrm_metrics["HitRate@10"])
    else:
        logger.warning("DLRM model not found at %s — skipping", MODEL_SAVE_PATH)

    # ------------------------------------------------------------------
    # 6. Naive baselines (for the full table)
    # ------------------------------------------------------------------
    logger.info("Running naive baselines...")

    def random_rank(uidx, item_indices):
        return np.random.rand(len(item_indices))

    random_metrics = compute_metrics(user_test, random_rank, k=K)
    random_metrics["train_time"] = None
    random_metrics["latency_ms"] = None

    # Most Popular
    item_popularity = np.zeros(len(item2idx), dtype=np.float64)
    for row in train_raw:
        iid = row[1]
        if iid in item2idx:
            item_popularity[item2idx[iid]] += 1

    def popularity_rank(uidx, item_indices):
        return np.array([item_popularity[i] for i in item_indices])

    popular_metrics = compute_metrics(user_test, popularity_rank, k=K)
    popular_metrics["train_time"] = None
    popular_metrics["latency_ms"] = None

    # ------------------------------------------------------------------
    # 7. Print comparison table
    # ------------------------------------------------------------------
    all_results = [
        ("DLRM", dlrm_metrics),
    ] + results + [
        ("Random", random_metrics),
        ("Most Popular", popular_metrics),
    ]

    print("\n" + "=" * 90)
    print("COMPARISON TABLE")
    print("=" * 90)

    header = (
        f"{'Model':<16} | {'NDCG@10':>8} | {'Prec@10':>8} | {'Hit@10':>8} "
        f"| {'Train Time':>11} | {'Inference':>13}"
    )
    sep = (
        f"{'-'*16}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-"
        f"+-{'-'*11}-+-{'-'*13}"
    )
    print(header)
    print(sep)

    for name, m in all_results:
        tt = f"{m['train_time']:.1f}s" if m.get("train_time") is not None else "-"
        lat = f"{m['latency_ms']:.2f}ms/user" if m.get("latency_ms") is not None else "-"
        print(
            f"{name:<16} | {m['NDCG@10']:>8.4f} | {m['Precision@10']:>8.4f} | "
            f"{m['HitRate@10']:>8.4f} | {tt:>11} | {lat:>13}"
        )

    print("=" * 90)
    print(f"Evaluated on {dlrm_metrics.get('num_eval_users', '?')} users with top-{K} recommendations.\n")

    # ------------------------------------------------------------------
    # 8. Save as CLASSICAL_COMPARISON.md
    # ------------------------------------------------------------------
    lines = [
        "# Classical ML Baselines vs DLRM — MovieLens 100K\n",
        f"Evaluated on {dlrm_metrics.get('num_eval_users', '?')} users with top-{K} recommendations.\n",
        "## Features for Classical Models\n",
        "| # | Feature | Description |",
        "|---|---------|-------------|",
        "| 0 | user_mean_rating | Average rating the user gave (normalised) |",
        "| 1 | user_rating_count | Number of ratings by user (normalised) |",
        "| 2 | user_rating_var | Variance of user's ratings (normalised) |",
        "| 3 | user_days_active | Days between first and last rating (normalised) |",
        "| 4 | item_mean_rating | Average rating the item received (normalised) |",
        "| 5 | item_rating_count | Number of ratings for item (normalised) |",
        "| 6 | item_popularity_rank | Popularity rank normalised to [0, 1] |",
        "| 7 | deviation | user_mean - item_mean (preference signal) |",
        "",
        "## Results\n",
        f"| {'Model':<16} | {'NDCG@10':>8} | {'Prec@10':>8} | {'Hit@10':>8} | {'Train Time':>11} | {'Inference':>13} |",
        f"|{'-'*18}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*13}|{'-'*15}|",
    ]

    for name, m in all_results:
        tt = f"{m['train_time']:.1f}s" if m.get("train_time") is not None else "-"
        lat = f"{m['latency_ms']:.2f}ms/user" if m.get("latency_ms") is not None else "-"
        lines.append(
            f"| {name:<16} | {m['NDCG@10']:>8.4f} | {m['Precision@10']:>8.4f} | "
            f"{m['HitRate@10']:>8.4f} | {tt:>11} | {lat:>13} |"
        )

    lines.append("")
    lines.append("## Model Details\n")
    lines.append("- **DLRM**: Deep Learning Recommendation Model with user/item embeddings (128-dim) and 3-layer MLP.")
    lines.append("- **XGBoost**: Gradient-boosted trees (200 estimators, max_depth=6) on 8 hand-crafted features.")
    lines.append("- **LightGBM**: Gradient-boosted trees (200 estimators, max_depth=6) on the same features.")
    lines.append("- **LogReg**: Logistic Regression with StandardScaler preprocessing on the same features.")
    lines.append("- **Random**: Uniformly random scores.")
    lines.append("- **Most Popular**: Rank by training-set popularity count.")
    lines.append("")
    lines.append("## Key Takeaway\n")
    lines.append(
        "Classical models with good feature engineering are competitive baselines. "
        "The DLRM's advantage comes from learning user/item embeddings that capture "
        "latent interaction patterns beyond hand-crafted features."
    )
    lines.append("")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))

    logger.info("Report saved to %s", REPORT_PATH)
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
