"""Shared preprocessing functions for MovieLens data.

Used by both training (scripts/train_movielens.py) and serving (api/app.py)
to guarantee identical feature engineering and prevent train/serve skew.
"""

import numpy as np

NUM_FEATURES = 2  # mean_rating, normalised_count
RATING_MAX = 5.0  # ratings are 1-5, normalise by dividing by 5


def load_movielens_data(data_path: str) -> np.ndarray:
    """Load MovieLens u.data and return (N, 4) int64 array."""
    return np.loadtxt(data_path, dtype=np.int64)


def build_id_mappings(raw: np.ndarray):
    """Return contiguous user2idx and item2idx dicts from raw data."""
    all_users = np.unique(raw[:, 0])
    all_items = np.unique(raw[:, 1])
    user2idx = {uid: i for i, uid in enumerate(all_users)}
    item2idx = {iid: i for i, iid in enumerate(all_items)}
    return user2idx, item2idx


def compute_user_stats(train_raw: np.ndarray):
    """Compute per-user rating lists and max count from training data only."""
    user_ratings = {}
    for row in train_raw:
        uid = row[0]
        user_ratings.setdefault(uid, []).append(row[2])
    max_count = max(len(v) for v in user_ratings.values()) if user_ratings else 1
    return user_ratings, max_count


def normalise_rating(rating):
    """Normalise a raw rating to [0, 1]."""
    return rating / RATING_MAX


def make_features(data, user2idx, item2idx, user_ratings_dict, max_count):
    """Build continuous features, categorical indices, and normalised targets.

    Continuous features per sample:
        0: normalised mean rating for the user (default 3/5 if unseen)
        1: normalised interaction count for the user

    Returns (cont, cat, targets) numpy arrays.
    """
    cont = np.zeros((len(data), NUM_FEATURES), dtype=np.float32)
    cat = np.zeros((len(data), 2), dtype=np.int64)
    targets = np.zeros(len(data), dtype=np.float32)

    for i, row in enumerate(data):
        uid, iid, rating, _ = row
        cat[i, 0] = user2idx[uid]
        cat[i, 1] = item2idx[iid]
        targets[i] = rating / RATING_MAX

        uratings = user_ratings_dict.get(uid, [3])
        cont[i, 0] = np.mean(uratings) / RATING_MAX
        cont[i, 1] = len(uratings) / max_count

    return cont, cat, targets


def prepare_splits(raw: np.ndarray):
    """Timestamp-based 80/20 split with feature engineering.

    Returns (train_cont, train_cat, train_targets,
             test_cont, test_cat, test_targets,
             user2idx, item2idx, test_raw).
    """
    sorted_idx = np.argsort(raw[:, 3])
    raw = raw[sorted_idx]

    split = int(len(raw) * 0.8)
    train_raw = raw[:split]
    test_raw = raw[split:]

    user2idx, item2idx = build_id_mappings(raw)
    user_ratings, max_count = compute_user_stats(train_raw)

    train_cont, train_cat, train_targets = make_features(
        train_raw, user2idx, item2idx, user_ratings, max_count
    )
    test_cont, test_cat, test_targets = make_features(
        test_raw, user2idx, item2idx, user_ratings, max_count
    )

    return (train_cont, train_cat, train_targets,
            test_cont, test_cat, test_targets,
            user2idx, item2idx, test_raw)
