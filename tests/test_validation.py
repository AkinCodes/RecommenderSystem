import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocessing import validate_raw_data, validate_splits


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
            validate_raw_data(data, min_ratings=100)

    def test_custom_min_ratings_respected(self):
        data = _make_valid_data(n=50)
        validate_raw_data(data, min_ratings=10)  # should pass


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
