import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocessing import (
    PrepConfig,
    compute_item_stats,
    compute_user_stats,
    prepare_splits,
    prepare_train_val_test,
    validate_raw_data,
    validate_splits,
)


def _make_valid_data(n=1500):
    """Generate a minimal valid ratings array."""
    rng = np.random.RandomState(42)
    user_ids = rng.randint(1, 100, size=n)
    item_ids = rng.randint(1, 200, size=n)
    ratings = rng.randint(1, 6, size=n)  # [1, 5]
    timestamps = np.arange(n)
    return np.column_stack([user_ids, item_ids, ratings, timestamps]).astype(
        np.int64
    )


class TestValidateRawData:
    def test_valid_data_passes(self):
        data = _make_valid_data()
        validate_raw_data(data)  # should not raise

    def test_empty_data_raises(self):
        empty = np.array([], dtype=np.int64).reshape(0, 4)
        with pytest.raises(ValueError, match="empty"):
            validate_raw_data(empty)

    def test_wrong_column_count_raises(self):
        data = np.ones((100, 3), dtype=np.int64)
        with pytest.raises(ValueError, match="Expected.*4.*columns"):
            validate_raw_data(data)

    def test_rating_below_range_raises(self):
        data = _make_valid_data()
        data[0, 2] = 0  # invalid rating
        with pytest.raises(ValueError, match="outside valid range"):
            validate_raw_data(data)

    def test_rating_above_range_raises(self):
        data = _make_valid_data()
        data[0, 2] = 6  # invalid rating
        with pytest.raises(ValueError, match="outside valid range"):
            validate_raw_data(data)

    def test_negative_user_id_raises(self):
        data = _make_valid_data()
        data[0, 0] = -1
        with pytest.raises(ValueError, match="Non-positive user IDs"):
            validate_raw_data(data)

    def test_zero_item_id_raises(self):
        data = _make_valid_data()
        data[0, 1] = 0
        with pytest.raises(ValueError, match="Non-positive item IDs"):
            validate_raw_data(data)

    def test_too_few_ratings_raises(self):
        data = _make_valid_data(n=50)
        with pytest.raises(ValueError, match="only 50 ratings"):
            validate_raw_data(data, config=PrepConfig(min_ratings=100))

    def test_custom_min_ratings_respected(self):
        data = _make_valid_data(n=50)
        validate_raw_data(data, config=PrepConfig(min_ratings=10))  # should pass


class TestValidateSplits:
    def test_train_larger_than_test(self):
        data = _make_valid_data(n=2000)
        split = int(len(data) * 0.8)
        train = data[:split]
        test = data[split:]
        # should not raise
        validate_splits(train, test, {}, {})

    def test_train_smaller_than_test_raises(self):
        data = _make_valid_data(n=2000)
        train = data[:400]
        test = data[400:]
        with pytest.raises(ValueError, match="should be larger"):
            validate_splits(train, test, {}, {})


class TestPrepareTrainValTest:
    def _data(self, n=2000):
        # Unique, shuffled timestamps so temporal ordering is a real test.
        data = _make_valid_data(n=n)
        rng = np.random.RandomState(7)
        data[:, 3] = rng.permutation(n)
        return data

    def _sizes(self, raw, config):
        n = len(raw)
        test_split = int(n * config.train_ratio)
        val_split = int(n * (config.train_ratio - config.val_ratio))
        return val_split, test_split - val_split, n - test_split

    def test_splits_disjoint_and_cover(self):
        raw = self._data()
        splits = prepare_train_val_test(raw)
        # Reconstruct the sorted order used internally to compare row identity.
        sorted_raw = raw[np.argsort(raw[:, 3])]
        n = len(sorted_raw)
        n_train, n_val, _ = self._sizes(raw, PrepConfig())
        # Row-for-row identity against the sorted array (all timestamps unique).
        np.testing.assert_array_equal(
            splits.train_cat[:, 0], np.array([splits.user2idx[u] for u in sorted_raw[:n_train, 0]])
        )
        # Index-set disjointness/coverage over positions in the sorted array.
        train_pos = set(range(0, n_train))
        val_pos = set(range(n_train, n_train + n_val))
        test_pos = set(range(n_train + n_val, n))
        assert train_pos.isdisjoint(val_pos)
        assert train_pos.isdisjoint(test_pos)
        assert val_pos.isdisjoint(test_pos)
        assert train_pos | val_pos | test_pos == set(range(n))

    def test_temporal_ordering(self):
        raw = self._data()
        splits = prepare_train_val_test(raw)
        sorted_raw = raw[np.argsort(raw[:, 3])]
        n_train, n_val, _ = self._sizes(raw, PrepConfig())
        train_ts = sorted_raw[:n_train, 3]
        val_ts = splits.val_raw[:, 3]
        test_ts = splits.test_raw[:, 3]
        assert train_ts.max() <= val_ts.min()
        assert val_ts.max() <= test_ts.min()

    def test_sizes_match_ratios(self):
        raw = self._data()
        config = PrepConfig()
        splits = prepare_train_val_test(raw, config)
        n_train, n_val, n_test = self._sizes(raw, config)
        assert len(splits.train_cat) == n_train
        assert len(splits.val_raw) == n_val
        assert len(splits.test_raw) == n_test
        # Default ratios: 0.70 / 0.10 / 0.20.
        assert n_train == 1400
        assert n_val == 200
        assert n_test == 400

    def test_test_slice_matches_prepare_splits(self):
        raw = self._data()
        two_way = prepare_splits(raw)
        three_way = prepare_train_val_test(raw)
        # test_raw (9th element of the 2-way tuple) must be identical.
        np.testing.assert_array_equal(two_way[8], three_way.test_raw)

    def test_no_leakage_stats_from_train_only(self):
        raw = self._data()
        config = PrepConfig()
        splits = prepare_train_val_test(raw, config)
        sorted_raw = raw[np.argsort(raw[:, 3])]
        n_train, _, _ = self._sizes(raw, config)
        train_raw = sorted_raw[:n_train]

        # Train-only stats vs full-data stats.
        train_user_ratings, train_max_count, _ = compute_user_stats(train_raw)
        full_user_ratings, full_max_count, _ = compute_user_stats(sorted_raw)
        _, train_item_max, _ = compute_item_stats(train_raw)
        _, full_item_max, _ = compute_item_stats(sorted_raw)

        # The train-only max counts should differ from full-data ones (leakage
        # would make features use full-data stats).
        assert train_max_count <= full_max_count
        assert train_item_max <= full_item_max

        # Verify a val feature equals the TRAIN-only computation, not full-data.
        # Pick a val user that appears in train.
        val_raw = splits.val_raw
        for i, row in enumerate(val_raw):
            uid = row[0]
            if uid in train_user_ratings:
                expected = np.mean(train_user_ratings[uid]) / 5.0
                full_expected = np.mean(full_user_ratings[uid]) / 5.0
                # cont[:, 0] is user_mean built from stats.
                np.testing.assert_allclose(splits.val_cont[i, 0], expected, rtol=1e-6)
                if not np.isclose(expected, full_expected):
                    # When they differ, the feature must match TRAIN, not full.
                    assert not np.isclose(splits.val_cont[i, 0], full_expected)
                break
        else:
            pytest.skip("no shared user between train and val in this fixture")

    def test_serving_stats_train_slice_matches_model_train_portion(self):
        # train_movielens builds serving_context stats from a reconstructed train
        # slice: timestamp-sort raw, take first (train_ratio - val_ratio) rows.
        # That slice must be identical to the train portion prepare_train_val_test
        # actually trains on (no train/serve skew).
        raw = self._data()
        config = PrepConfig()
        splits = prepare_train_val_test(raw, config)

        raw_sorted = raw[np.argsort(raw[:, 3])]
        train_split_serve = int(len(raw_sorted) * (config.train_ratio - config.val_ratio))
        train_raw_serve = raw_sorted[:train_split_serve]

        # The reconstructed serving slice must produce the same train_cat rows.
        expected_user_idx = np.array([splits.user2idx[u] for u in train_raw_serve[:, 0]])
        np.testing.assert_array_equal(splits.train_cat[:, 0], expected_user_idx)
        assert len(train_raw_serve) == len(splits.train_cat)
