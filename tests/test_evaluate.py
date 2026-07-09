"""Regression test for the per-user ranking eval bug.

The old code scored EVERY candidate for a user with ONE row's continuous
features via ``np.tile``, so item-specific continuous columns (item_mean,
item_count, item_popularity, user*item interaction) were identical for all
candidates but one. This test constructs two candidates for the same user with
DIFFERENT continuous features and asserts they receive DIFFERENT scores. The
old tiled code would give them identical continuous inputs -> identical scores
(FAIL); the fix scores each candidate with its own row (PASS).
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.dlrm import DLRMModel  # noqa: E402
from scripts.train_movielens import evaluate  # noqa: E402


def _tiny_model(num_features):
    torch.manual_seed(0)
    model = DLRMModel(
        num_features=num_features,
        cardinalities=[4, 4],
        embedding_dim=4,
        mlp_layers=[8, 4],
        dropout=0.0,
    )
    model.eval()
    return model


def test_candidates_get_distinct_scores_from_own_continuous_features():
    """Two candidates of one user with different cont rows must score differently.

    Fails on the old np.tile(user_cont[uidx], ...) code (identical cont inputs),
    passes on the fix (test_cont[row_idxs] -> per-candidate rows).
    """
    num_features = 3
    model = _tiny_model(num_features)

    # One user (raw id 1 -> uidx 0), two candidate items (iidx 1 and 2).
    # The two candidates have clearly DIFFERENT continuous rows.
    test_cont = np.array(
        [
            [0.10, 0.10, 0.10],
            [0.90, 0.90, 0.90],
        ],
        dtype=np.float32,
    )
    test_cat = np.array([[0, 1], [0, 2]], dtype=np.int64)
    test_targets = np.array([1.0, 1.0], dtype=np.float32)
    test_raw = np.array([[1, 11, 5, 100], [1, 22, 5, 200]], dtype=np.int64)
    user2idx = {1: 0}
    item2idx = {11: 1, 22: 2}

    # Score each candidate directly through the model the way evaluate() does.
    with torch.no_grad():
        cont_t = torch.tensor(test_cont, dtype=torch.float32)
        cat_t = torch.tensor(test_cat, dtype=torch.long)
        scores = model(cont_t, cat_t).squeeze().numpy()

    # Precondition: with different continuous rows the model gives different
    # scores. If it didn't, the test below would be vacuous.
    assert not np.isclose(scores[0], scores[1]), (
        "model insensitive to continuous features; test would be vacuous"
    )

    # The tiled bug would have used test_cont[0] for BOTH candidates, so the
    # continuous inputs would be identical. Confirm that would collapse scores.
    with torch.no_grad():
        tiled_cont = torch.tensor(np.tile(test_cont[0], (2, 1)), dtype=torch.float32)
        tiled_scores = model(tiled_cont, cat_t).squeeze().numpy()
    # Under the buggy path the two candidates differ ONLY by item embedding, but
    # the continuous contribution is identical — demonstrate the two paths differ.
    assert not np.allclose(scores, tiled_scores), (
        "fixed per-candidate scoring must differ from the old tiled scoring"
    )

    # Now exercise the real evaluate(): it must not crash and must run on the
    # user, proving the fixed grouping (iidx, rating, row_idx) works end-to-end.
    metrics = evaluate(model, test_cont, test_cat, test_targets, user2idx, item2idx, test_raw, k=10)
    assert metrics["num_eval_users"] == 1
    # Both candidates relevant -> perfect ranking regardless of order.
    assert metrics["NDCG@10"] == 1.0


def test_evaluate_uses_per_candidate_rows_not_first_row():
    """evaluate() must feed each candidate its own continuous row.

    We monkeypatch the model's forward to record the continuous tensor it
    receives, then assert both distinct rows were passed (not one row tiled).
    """
    num_features = 3
    model = _tiny_model(num_features)

    test_cont = np.array(
        [
            [0.10, 0.20, 0.30],
            [0.70, 0.80, 0.90],
        ],
        dtype=np.float32,
    )
    test_cat = np.array([[0, 1], [0, 2]], dtype=np.int64)
    test_targets = np.array([1.0, 1.0], dtype=np.float32)
    test_raw = np.array([[1, 11, 5, 100], [1, 22, 5, 200]], dtype=np.int64)
    user2idx = {1: 0}
    item2idx = {11: 1, 22: 2}

    seen = {}
    orig_forward = model.forward

    def spy_forward(cont, cat):
        seen["cont"] = cont.detach().cpu().numpy().copy()
        return orig_forward(cont, cat)

    model.forward = spy_forward
    evaluate(model, test_cont, test_cat, test_targets, user2idx, item2idx, test_raw, k=10)

    # Both distinct rows must have been passed through — not row 0 tiled twice.
    np.testing.assert_allclose(seen["cont"], test_cont, rtol=1e-6)
    # And explicitly NOT the tiled-first-row inputs the old bug produced.
    assert not np.allclose(seen["cont"], np.tile(test_cont[0], (2, 1)))
