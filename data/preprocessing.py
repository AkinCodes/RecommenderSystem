"""Shared preprocessing functions for MovieLens data.

Used by both training (scripts/train_movielens.py) and serving (api/app.py)
to guarantee identical feature engineering and prevent train/serve skew.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

NUM_FEATURES = 8  # user: mean_rating, rating_count, rating_var, days_active
                   # item: mean_rating, rating_count, popularity_rank
                   # interaction: user_item_deviation
RATING_MAX = 5.0  # used to normalise continuous features (mean rating)
LIKE_THRESHOLD = 4  # ratings >= 4 are "liked" (label 1.0), else 0.0
MIN_RATINGS = 1000  # minimum number of ratings required
MIN_INTERACTIONS_WARN = 5  # warn if a user/item has fewer than this many ratings
MAX_RATING_VAR = 4.0  # max possible variance for integer ratings in [1, 5]


def validate_raw_data(raw: np.ndarray, min_ratings: int = MIN_RATINGS) -> None:
    """Validate the raw ratings array. Raises ValueError on any issue."""
    if raw.size == 0:
        raise ValueError("Data is empty — received a 0-length array.")

    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError(
            f"Expected (N, 4) array with columns [user_id, item_id, rating, timestamp], "
            f"got shape {raw.shape}."
        )

    # --- null / NaN check (int64 arrays can't hold NaN, but guard against
    #     future dtype changes or masked arrays) ---
    if hasattr(raw, "mask") or np.isnan(raw.astype(float)).any():
        nan_rows = np.where(np.isnan(raw.astype(float)).any(axis=1))[0]
        raise ValueError(
            f"Data contains NaN/null values in {len(nan_rows)} row(s). "
            f"First offending row index: {nan_rows[0]}."
        )

    # --- rating range ---
    ratings = raw[:, 2]
    out_of_range = (ratings < 1) | (ratings > 5)
    if out_of_range.any():
        bad_count = int(out_of_range.sum())
        bad_vals = np.unique(ratings[out_of_range])
        raise ValueError(
            f"{bad_count} rating(s) outside valid range [1, 5]. "
            f"Invalid values found: {bad_vals.tolist()}."
        )

    # --- positive IDs ---
    user_ids = raw[:, 0]
    item_ids = raw[:, 1]
    if (user_ids <= 0).any():
        bad = np.unique(user_ids[user_ids <= 0])
        raise ValueError(f"Non-positive user IDs found: {bad.tolist()}.")
    if (item_ids <= 0).any():
        bad = np.unique(item_ids[item_ids <= 0])
        raise ValueError(f"Non-positive item IDs found: {bad.tolist()}.")

    # --- minimum volume ---
    n = len(raw)
    if n < min_ratings:
        raise ValueError(
            f"Dataset has only {n} ratings, need at least {min_ratings}."
        )

    # --- advisory warnings (don't block, just log) ---
    unique_users, user_counts = np.unique(user_ids, return_counts=True)
    sparse_users = unique_users[user_counts < MIN_INTERACTIONS_WARN]
    if len(sparse_users) > 0:
        logger.warning(
            "%d user(s) have fewer than %d ratings (e.g. user_ids %s).",
            len(sparse_users),
            MIN_INTERACTIONS_WARN,
            sparse_users[:5].tolist(),
        )

    unique_items, item_counts = np.unique(item_ids, return_counts=True)
    sparse_items = unique_items[item_counts < MIN_INTERACTIONS_WARN]
    if len(sparse_items) > 0:
        logger.warning(
            "%d item(s) have fewer than %d ratings (e.g. item_ids %s).",
            len(sparse_items),
            MIN_INTERACTIONS_WARN,
            sparse_items[:5].tolist(),
        )


def load_movielens_data(data_path: str) -> np.ndarray:
    """Load MovieLens u.data, validate, and return (N, 4) int64 array."""
    raw = np.loadtxt(data_path, dtype=np.int64)
    validate_raw_data(raw)
    return raw


def build_id_mappings(raw: np.ndarray):
    """Return contiguous user2idx and item2idx dicts from raw data."""
    all_users = np.unique(raw[:, 0])
    all_items = np.unique(raw[:, 1])
    user2idx = {uid: i for i, uid in enumerate(all_users)}
    item2idx = {iid: i for i, iid in enumerate(all_items)}
    return user2idx, item2idx


def compute_user_stats(train_raw: np.ndarray):
    """Compute per-user rating lists, max count, and timestamps from training data only."""
    user_ratings = {}
    user_timestamps = {}
    for row in train_raw:
        uid = row[0]
        user_ratings.setdefault(uid, []).append(row[2])
        user_timestamps.setdefault(uid, []).append(row[3])
    max_count = (
        max(len(v) for v in user_ratings.values()) if user_ratings else 1
    )
    return user_ratings, max_count, user_timestamps


def compute_item_stats(train_raw: np.ndarray):
    """Compute per-item rating stats from training data only.

    Returns:
        item_ratings: dict mapping item_id -> list of ratings
        item_max_count: max number of ratings any single item received
        item_popularity_rank: dict mapping item_id -> normalised rank in [0, 1]
            where 1.0 = most popular item
    """
    item_ratings = {}
    for row in train_raw:
        iid = row[1]
        item_ratings.setdefault(iid, []).append(row[2])

    item_max_count = (
        max(len(v) for v in item_ratings.values()) if item_ratings else 1
    )

    sorted_items = sorted(item_ratings.keys(), key=lambda x: len(item_ratings[x]))
    num_items = len(sorted_items)
    item_popularity_rank = {}
    for rank_pos, iid in enumerate(sorted_items):
        item_popularity_rank[iid] = (rank_pos + 1) / num_items if num_items > 0 else 0.0

    return item_ratings, item_max_count, item_popularity_rank


def compute_max_user_days(user_timestamps: dict) -> float:
    """Return the maximum days-active span across all users."""
    max_days = 0.0
    for times in user_timestamps.values():
        if len(times) > 1:
            days = (max(times) - min(times)) / 86400.0
            if days > max_days:
                max_days = days
    return max_days


def make_features(data, user2idx, item2idx, user_ratings_dict, max_count,
                  user_timestamps, item_ratings_dict, item_max_count,
                  item_popularity_rank, max_user_days):
    """Build continuous features, categorical indices, and binary labels.

    Continuous features per sample (8 total):
        0: user_mean_rating -- normalised mean rating for the user
        1: user_rating_count -- normalised interaction count for the user
        2: user_rating_var -- variance of user's ratings, normalised
        3: user_days_active -- days between first and last rating, normalised
        4: item_mean_rating -- average rating the item received, normalised
        5: item_rating_count -- number of ratings the item received, normalised
        6: item_popularity_rank -- rank-ordered popularity normalised to [0, 1]
        7: user_item_deviation -- user_mean - item_mean (shifted to [0, 1])

    Targets are binary labels: 1.0 if rating >= LIKE_THRESHOLD, else 0.0.

    Returns (cont, cat, targets) numpy arrays.
    """
    cont = np.zeros((len(data), NUM_FEATURES), dtype=np.float32)
    cat = np.zeros((len(data), 2), dtype=np.int64)
    targets = np.zeros(len(data), dtype=np.float32)

    for i, row in enumerate(data):
        uid, iid, rating, _ = row
        cat[i, 0] = user2idx[uid]
        cat[i, 1] = item2idx[iid]
        targets[i] = 1.0 if rating >= LIKE_THRESHOLD else 0.0

        # --- User features ---
        uratings = user_ratings_dict.get(uid, [3])
        user_mean = np.mean(uratings) / RATING_MAX
        cont[i, 0] = user_mean
        cont[i, 1] = len(uratings) / max_count
        cont[i, 2] = np.var(uratings) / MAX_RATING_VAR if len(uratings) > 1 else 0.0

        utimes = user_timestamps.get(uid, [])
        if len(utimes) > 1:
            days = (max(utimes) - min(utimes)) / 86400.0
            cont[i, 3] = days / max_user_days if max_user_days > 0 else 0.0
        else:
            cont[i, 3] = 0.0

        # --- Item features ---
        iratings = item_ratings_dict.get(iid, [3])
        item_mean = np.mean(iratings) / RATING_MAX
        cont[i, 4] = item_mean
        cont[i, 5] = len(iratings) / item_max_count
        cont[i, 6] = item_popularity_rank.get(iid, 0.0)

        # --- Interaction feature ---
        # user_mean - item_mean is in [-1, 1]; shift to [0, 1]
        cont[i, 7] = (user_mean - item_mean + 1.0) / 2.0

    return cont, cat, targets


def validate_splits(train_raw: np.ndarray, test_raw: np.ndarray,
                    user2idx: dict, item2idx: dict) -> None:
    """Validate train/test splits. Raises ValueError on issues."""
    if len(train_raw) <= len(test_raw):
        raise ValueError(
            f"Train set ({len(train_raw)}) should be larger than test set "
            f"({len(test_raw)})."
        )

    train_users = set(np.unique(train_raw[:, 0]))
    train_items = set(np.unique(train_raw[:, 1]))
    test_users = set(np.unique(test_raw[:, 0]))
    test_items = set(np.unique(test_raw[:, 1]))

    unseen_users = test_users - train_users
    unseen_items = test_items - train_items

    if unseen_users:
        logger.warning(
            "%d user(s) in test set not seen in train (e.g. %s). "
            "Embeddings will be untrained for these.",
            len(unseen_users),
            sorted(unseen_users)[:5],
        )
    if unseen_items:
        logger.warning(
            "%d item(s) in test set not seen in train (e.g. %s). "
            "Embeddings will be untrained for these.",
            len(unseen_items),
            sorted(unseen_items)[:5],
        )

    logger.info(
        "Split sizes — train: %d, test: %d (%.1f%% / %.1f%%).",
        len(train_raw),
        len(test_raw),
        100 * len(train_raw) / (len(train_raw) + len(test_raw)),
        100 * len(test_raw) / (len(train_raw) + len(test_raw)),
    )


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
    validate_splits(train_raw, test_raw, user2idx, item2idx)
    user_ratings, max_count, user_timestamps = compute_user_stats(train_raw)
    item_ratings, item_max_count, item_popularity_rank = compute_item_stats(train_raw)
    max_user_days = compute_max_user_days(user_timestamps)

    train_cont, train_cat, train_targets = make_features(
        train_raw, user2idx, item2idx, user_ratings, max_count,
        user_timestamps, item_ratings, item_max_count,
        item_popularity_rank, max_user_days,
    )
    test_cont, test_cat, test_targets = make_features(
        test_raw, user2idx, item2idx, user_ratings, max_count,
        user_timestamps, item_ratings, item_max_count,
        item_popularity_rank, max_user_days,
    )

    return (train_cont, train_cat, train_targets,
            test_cont, test_cat, test_targets,
            user2idx, item2idx, test_raw)


def load_item_metadata(data_dir: str) -> dict:
    """Load movie titles and genres from MovieLens u.item file."""
    items = {}
    item_path = os.path.join(data_dir, "u.item")
    genres = [
        "Unknown", "Action", "Adventure", "Animation", "Children",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western",
    ]
    with open(item_path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            item_id = int(parts[0])
            title = parts[1]
            genre_flags = [int(g) for g in parts[5:24]]
            item_genres = [genres[i] for i, flag in enumerate(genre_flags) if flag]
            items[item_id] = {"title": title, "genres": item_genres}
    return items
