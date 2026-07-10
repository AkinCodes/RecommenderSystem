"""Tests for models/sasrec.py.

These are real behavioural tests, not smoke tests. The headline one is
``test_causal_mask_blocks_the_future``: it asserts that the hidden state at
position t does not change when we mutate inputs at positions > t. If the causal
mask were dropped (or built with the wrong triangle), that test fails.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocessing import build_user_sequences  # noqa: E402
from models.sasrec import PAD_IDX, SASRec  # noqa: E402

DIM = 64
MAX_LEN = 50
NUM_ITEMS = 100


def make_model(seed: int = 42) -> SASRec:
    torch.manual_seed(seed)
    model = SASRec(num_items=NUM_ITEMS, embedding_dim=DIM, max_len=MAX_LEN)
    model.eval()  # disable dropout so forward passes are deterministic
    return model


class TestShapes:
    def test_forward_returns_b_l_d(self):
        model = make_model()
        seq = torch.randint(1, NUM_ITEMS + 1, (4, 10))
        out = model.forward(seq)
        assert out.shape == (4, 10, DIM)

    def test_encode_returns_b_d(self):
        model = make_model()
        seq = torch.randint(1, NUM_ITEMS + 1, (4, 10))
        enc = model.encode(seq)
        assert enc.shape == (4, DIM)
        assert enc.shape[1] == 64  # dim contract for downstream dot-product scoring

    def test_single_position_sequence(self):
        model = make_model()
        seq = torch.randint(1, NUM_ITEMS + 1, (2, 1))
        assert model.forward(seq).shape == (2, 1, DIM)
        assert model.encode(seq).shape == (2, DIM)


class TestCausalMask:
    def test_causal_mask_blocks_the_future(self):
        """Output at position t must be invariant to inputs at positions > t."""
        model = make_model()
        seq_a = torch.randint(1, NUM_ITEMS + 1, (1, 8))
        seq_b = seq_a.clone()
        # Mutate everything strictly AFTER position t = 3.
        t = 3
        seq_b[0, t + 1:] = torch.randint(1, NUM_ITEMS + 1, (8 - (t + 1),))
        assert not torch.equal(seq_a, seq_b)  # inputs really differ in the future

        with torch.no_grad():
            out_a = model.forward(seq_a)
            out_b = model.forward(seq_b)

        # Positions 0..t identical (they only see the shared past); positions > t differ.
        assert torch.allclose(out_a[:, : t + 1, :], out_b[:, : t + 1, :], atol=1e-6)
        assert not torch.allclose(out_a[:, t + 1:, :], out_b[:, t + 1:, :], atol=1e-6)

    def test_first_position_invariant_to_all_later_inputs(self):
        model = make_model()
        seq_a = torch.randint(1, NUM_ITEMS + 1, (1, 6))
        seq_b = seq_a.clone()
        seq_b[0, 1:] = torch.randint(1, NUM_ITEMS + 1, (5,))
        with torch.no_grad():
            out_a = model.forward(seq_a)
            out_b = model.forward(seq_b)
        # Position 0 attends only to itself, so it cannot change.
        assert torch.allclose(out_a[:, 0, :], out_b[:, 0, :], atol=1e-6)


class TestPadding:
    def test_pad_row_embedding_is_zero(self):
        model = make_model()
        pad_vec = model.item_emb(torch.tensor([PAD_IDX]))
        assert torch.allclose(pad_vec, torch.zeros_like(pad_vec))

    def test_left_padding_does_not_change_real_encoding(self):
        """A left-padded sequence must encode to the SAME summary as the unpadded one.

        encode() reads the LAST column (most recent real item). Prepending PAD
        on the left should not alter that summary, because padded keys are masked.
        """
        model = make_model()
        real = torch.tensor([[5, 9, 2, 7]])
        padded = torch.tensor([[PAD_IDX, PAD_IDX, 5, 9, 2, 7]])
        with torch.no_grad():
            enc_real = model.encode(real)
            enc_padded = model.encode(padded)
        assert torch.allclose(enc_real, enc_padded, atol=1e-5)


class TestDeterminism:
    def test_same_seed_same_weights(self):
        m1 = make_model(seed=123)
        m2 = make_model(seed=123)
        seq = torch.randint(1, NUM_ITEMS + 1, (3, 12))
        with torch.no_grad():
            assert torch.allclose(m1.forward(seq), m2.forward(seq), atol=1e-6)

    def test_different_seed_different_weights(self):
        m1 = make_model(seed=1)
        m2 = make_model(seed=2)
        assert not torch.allclose(m1.item_emb.weight, m2.item_emb.weight)


class TestBuildUserSequences:
    def _raw(self):
        # columns: user_id, item_id, rating, timestamp
        return np.array(
            [
                [1, 10, 5, 100],   # like
                [1, 20, 3, 200],   # not a like (rating < 4) -> dropped
                [1, 30, 4, 300],   # like
                [2, 10, 5, 150],   # like
                [1, 40, 5, 900],   # TEST slice (latest ts) -> dropped by train_ratio
            ],
            dtype=np.int64,
        )

    def test_filters_and_orders_chronologically(self):
        raw = self._raw()
        item2idx = {10: 0, 20: 1, 30: 2, 40: 3}
        seqs = build_user_sequences(raw, item2idx, like_threshold=4, train_ratio=0.8)
        # With train_ratio 0.8 the latest-timestamp row (user 1, item 40) is held out.
        assert seqs[1] == [item2idx[10], item2idx[30]]  # chronological, likes only
        assert seqs[2] == [item2idx[10]]

    def test_unknown_items_skipped(self):
        raw = self._raw()
        item2idx = {10: 0}  # item 30 not mapped
        seqs = build_user_sequences(raw, item2idx, like_threshold=4, train_ratio=1.0)
        assert seqs[1] == [0]  # only the mapped, liked item survives

    def test_pure_function_does_not_mutate_raw(self):
        raw = self._raw()
        snapshot = raw.copy()
        build_user_sequences(raw, {10: 0, 30: 2}, train_ratio=0.8)
        assert np.array_equal(raw, snapshot)


class TestEncodeContract:
    def test_encode_matches_last_position_of_forward(self):
        model = make_model()
        seq = torch.randint(1, NUM_ITEMS + 1, (2, 9))
        with torch.no_grad():
            full = model.forward(seq)
            enc = model.encode(seq)
        assert torch.allclose(enc, full[:, -1, :], atol=1e-6)

    def test_item_vectors_shape(self):
        model = make_model()
        ids = torch.tensor([1, 2, 3, 4, 5])
        assert model.item_vectors(ids).shape == (5, DIM)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
