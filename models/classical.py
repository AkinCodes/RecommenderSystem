"""Thin wrappers giving classical ML models the same ranking interface as DLRM.

Each wrapper exposes a ``rank(user_idx, item_indices) -> scores`` method so
that ``compute_metrics`` from baseline_comparison.py can evaluate them
identically to the deep learning model.
"""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Builds feature vectors for (user, item) pairs from pre-computed stats.

    All statistics are derived from the training set only to prevent leakage.

    Features produced (per user-item pair):
        0  user_mean_rating       — average rating the user gave (normalised)
        1  user_rating_count      — how many ratings the user gave (normalised)
        2  user_rating_var        — variance of the user's ratings (normalised)
        3  user_days_active       — days between first and last training rating (normalised)
        4  item_mean_rating       — average rating the item received (normalised)
        5  item_rating_count      — how many ratings the item received (normalised)
        6  item_popularity_rank   — rank by popularity, normalised to [0, 1]
        7  deviation              — user_mean - item_mean (how much user deviates)
    """

    NUM_FEATURES = 8

    def __init__(self, train_raw, user2idx, item2idx):
        """Pre-compute all user and item statistics from training data.

        Args:
            train_raw: numpy array with columns [user_id, item_id, rating, timestamp].
            user2idx: dict mapping raw user id to contiguous index.
            item2idx: dict mapping raw item id to contiguous index.
        """
        self.user2idx = user2idx
        self.item2idx = item2idx
        num_users = len(user2idx)
        num_items = len(item2idx)

        # --- User stats ---
        self.user_mean = np.full(num_users, 3.0 / 5.0, dtype=np.float32)
        self.user_count = np.zeros(num_users, dtype=np.float32)
        self.user_var = np.zeros(num_users, dtype=np.float32)
        self.user_days_active = np.zeros(num_users, dtype=np.float32)

        user_ratings = {}   # uidx -> list of ratings
        user_timestamps = {}  # uidx -> list of timestamps

        for row in train_raw:
            uid, iid, rating, ts = row
            uidx = user2idx[uid]
            user_ratings.setdefault(uidx, []).append(rating)
            user_timestamps.setdefault(uidx, []).append(ts)

        max_user_count = max((len(v) for v in user_ratings.values()), default=1)
        max_var = 0.0

        for uidx, ratings in user_ratings.items():
            arr = np.array(ratings, dtype=np.float64)
            self.user_mean[uidx] = arr.mean() / 5.0
            self.user_count[uidx] = len(arr) / max_user_count
            self.user_var[uidx] = arr.var()
            if max_var < self.user_var[uidx]:
                max_var = self.user_var[uidx]

            ts_arr = np.array(user_timestamps[uidx], dtype=np.float64)
            days = (ts_arr.max() - ts_arr.min()) / 86400.0
            self.user_days_active[uidx] = days

        # Normalise variance and days_active
        if max_var > 0:
            self.user_var /= max_var
        max_days = self.user_days_active.max()
        if max_days > 0:
            self.user_days_active /= max_days

        # --- Item stats ---
        self.item_mean = np.full(num_items, 3.0 / 5.0, dtype=np.float32)
        self.item_count = np.zeros(num_items, dtype=np.float32)
        self.item_pop_rank = np.zeros(num_items, dtype=np.float32)

        item_ratings = {}  # iidx -> list of ratings

        for row in train_raw:
            uid, iid, rating, ts = row
            if iid in item2idx:
                iidx = item2idx[iid]
                item_ratings.setdefault(iidx, []).append(rating)

        max_item_count = max((len(v) for v in item_ratings.values()), default=1)

        for iidx, ratings in item_ratings.items():
            arr = np.array(ratings, dtype=np.float64)
            self.item_mean[iidx] = arr.mean() / 5.0
            self.item_count[iidx] = len(arr) / max_item_count

        # Popularity rank: most popular item = 1.0, least = 0.0
        raw_counts = np.array([len(item_ratings.get(i, [])) for i in range(num_items)])
        ranks = raw_counts.argsort().argsort()  # rank order
        self.item_pop_rank = ranks.astype(np.float32) / max(num_items - 1, 1)

        logger.info(
            "FeatureBuilder initialised: %d users, %d items, %d features",
            num_users, num_items, self.NUM_FEATURES,
        )

    def __call__(self, user_idx, item_indices):
        """Build feature matrix for one user scored against multiple items.

        Args:
            user_idx: integer index of the user.
            item_indices: list/array of integer item indices.

        Returns:
            numpy array of shape (len(item_indices), NUM_FEATURES).
        """
        n = len(item_indices)
        features = np.zeros((n, self.NUM_FEATURES), dtype=np.float32)

        u_mean = self.user_mean[user_idx]
        u_count = self.user_count[user_idx]
        u_var = self.user_var[user_idx]
        u_days = self.user_days_active[user_idx]

        for j, iidx in enumerate(item_indices):
            i_mean = self.item_mean[iidx]
            features[j, 0] = u_mean
            features[j, 1] = u_count
            features[j, 2] = u_var
            features[j, 3] = u_days
            features[j, 4] = i_mean
            features[j, 5] = self.item_count[iidx]
            features[j, 6] = self.item_pop_rank[iidx]
            features[j, 7] = u_mean - i_mean  # deviation signal

        return features

    def build_matrix(self, data_raw):
        """Build feature matrix and binary labels for an entire dataset.

        Args:
            data_raw: numpy array with columns [user_id, item_id, rating, timestamp].

        Returns:
            Tuple of (X, y) where X is (n_samples, NUM_FEATURES) and y is binary.
        """
        n = len(data_raw)
        X = np.zeros((n, self.NUM_FEATURES), dtype=np.float32)
        y = np.zeros(n, dtype=np.int32)

        for i, row in enumerate(data_raw):
            uid, iid, rating, _ = row
            uidx = self.user2idx.get(uid)
            iidx = self.item2idx.get(iid)
            if uidx is None or iidx is None:
                continue

            feats = self(uidx, [iidx])
            X[i] = feats[0]
            y[i] = 1 if rating >= 4 else 0

        return X, y


class ClassicalRanker:
    """Wraps a trained sklearn-compatible classifier for the ranking interface.

    Args:
        model: a fitted model with ``predict_proba(X) -> array``.
        feature_builder: a ``FeatureBuilder`` instance.
        name: human-readable name for logging.
    """

    def __init__(self, model, feature_builder, name="ClassicalRanker"):
        self.model = model
        self.feature_builder = feature_builder
        self.name = name

    def rank(self, user_idx, item_indices):
        """Score items for a single user.

        Args:
            user_idx: contiguous user index.
            item_indices: list of contiguous item indices to rank.

        Returns:
            numpy array of scores (higher = more likely positive).
        """
        features = self.feature_builder(user_idx, item_indices)
        probas = self.model.predict_proba(features)
        # predict_proba returns (n, 2) for binary — take the positive class
        if probas.ndim == 2 and probas.shape[1] == 2:
            return probas[:, 1]
        return probas.ravel()
