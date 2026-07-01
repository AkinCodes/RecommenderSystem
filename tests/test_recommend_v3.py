"""Endpoint tests for the bandit-backed /recommend_v3 and /feedback routes.

These require the trained DLRM artifacts (trained_model_movielens.pth,
serving_context.pkl) and a built ANN index. If any of those are missing in the
test environment, the tests skip gracefully rather than fail.
"""

import pytest
from fastapi.testclient import TestClient

import api.app as app_module

client = TestClient(app_module.app)

ARTIFACTS_READY = (
    app_module.model_loaded
    and app_module.serving_context is not None
    and app_module.faiss_index is not None
    and app_module.bandit is not None
)

pytestmark = pytest.mark.skipif(
    not ARTIFACTS_READY,
    reason="trained model / serving context / ANN index / bandit not available in test env",
)


def _known_user_id() -> int:
    # Cast from numpy int64 so the value is JSON-serializable in request bodies.
    return int(next(iter(app_module.serving_context["user2idx"].keys())))


def _cold_user_id() -> int:
    # An id guaranteed to be outside the trained user vocabulary.
    return int(max(app_module.serving_context["user2idx"].keys())) + 10_000


def _known_item_id() -> int:
    return int(next(iter(app_module.serving_context["item2idx"].keys())))


def _assert_exploration_fields(rec: dict) -> None:
    assert "item_id" in rec
    assert "score" in rec
    assert "dlrm_score" in rec
    assert "exploration_bonus" in rec
    assert "is_exploratory" in rec
    assert isinstance(rec["is_exploratory"], bool)
    assert rec["exploration_bonus"] >= 0.0


def test_recommend_v3_warm_user_shape():
    resp = client.get(f"/api/v1/recommend_v3/{_known_user_id()}?top_k=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == _known_user_id()
    assert "policy" in body
    assert "latency_ms" in body
    assert len(body["recommendations"]) <= 5
    assert body["recommendations"], "expected at least one recommendation"
    for rec in body["recommendations"]:
        _assert_exploration_fields(rec)


def test_recommend_v3_cold_user_uses_bandit_path():
    resp = client.get(f"/api/v1/recommend_v3/{_cold_user_id()}?top_k=5")
    assert resp.status_code == 200
    body = resp.json()
    assert "cold-start" in body["note"]
    assert body["pipeline"] == "bandit-reranked (cold-start)"
    assert "cold_start_context" in body
    for rec in body["recommendations"]:
        _assert_exploration_fields(rec)


def test_feedback_updates_and_persists():
    user_id = _known_user_id()
    item_id = _known_item_id()
    resp = client.post(
        "/api/v1/feedback",
        json={"user_id": user_id, "item_id": item_id, "reward": 1.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["updates_applied"] == 1
    assert "exploration_bonus" in body


def test_feedback_rejects_bad_reward():
    user_id = _known_user_id()
    item_id = _known_item_id()
    resp = client.post(
        "/api/v1/feedback",
        json={"user_id": user_id, "item_id": item_id, "reward": 2.0},
    )
    assert resp.status_code == 400


# ------------------------------------------------------------------------------------------------
# Ordering-effect tests: these guard the exploit/bonus scale bug that shipped originally. Before the
# common-scale blend, the bonus swamped the DLRM sigmoid (warm: ALL items flagged exploratory) and
# the popularity counts swamped the bonus (cold: order IDENTICAL to the static popularity list).
# ------------------------------------------------------------------------------------------------


def test_warm_user_blend_reorders_but_keeps_exploit_signal():
    """After the common-scale fix the blend genuinely mixes exploit and explore, so the served
    order must differ from pure-DLRM order (at least one item flagged exploratory) WITHOUT the
    bonus swamping the score to a pure-exploration ranking.

    The original scale bug ranked warm users by the raw bonus alone: the combined order equalled
    the bonus order and was totally decoupled from the DLRM score. We assert the opposite -- the
    served order is neither identical to pure-DLRM nor identical to pure-bonus -- by comparing the
    default-w list against the w=0 (pure exploit) and w=1 (pure explore) endpoints via the same
    combine helper on the same candidates.
    """
    import numpy as np
    import torch

    from models.bandit import build_interaction_context, combine_explore

    uid = _known_user_id()
    ui = app_module.serving_context["user2idx"][uid]
    uf = app_module.serving_context["user_features"][ui]
    itf = app_module.serving_context.get("item_features", {})

    liked = app_module.user_history.get(ui, [])
    if liked and app_module.item_embeddings_np is not None:
        qv = app_module.item_embeddings_np[liked].mean(axis=0, keepdims=True).astype(np.float32)
    else:
        qv = app_module.model.embeddings[0](torch.tensor([ui])).detach().numpy().astype(np.float32)
    qv = (qv / (np.linalg.norm(qv) or 1.0)).astype(np.float32)
    if app_module.FAISS_AVAILABLE:
        _, ci = app_module.faiss_index.search(qv, 100)
        ci = ci[0].tolist()
    else:
        _, idx = app_module.faiss_index.kneighbors(qv, n_neighbors=100)
        ci = idx[0].tolist()

    cont = app_module._build_user_item_features(uf, ci, itf, app_module.num_continuous_features)
    cat = torch.tensor([[ui, i] for i in ci], dtype=torch.int64)
    with torch.no_grad():
        dlrm = app_module.model(cont, cat).squeeze().numpy().astype(np.float64)
    ctx = build_interaction_context(app_module._raw_user_emb(ui), app_module._raw_item_embs(ci))
    bonus = app_module.bandit.exploration_bonus(ctx)

    w = app_module.BANDIT_EXPLORE_WEIGHT
    default_order = np.argsort(combine_explore(dlrm, bonus, w))[::-1]
    exploit_order = np.argsort(combine_explore(dlrm, bonus, 0.0))[::-1]
    explore_order = np.argsort(combine_explore(dlrm, bonus, 1.0))[::-1]

    # w=0 must reproduce the pure DLRM order exactly (combine_explore is scale-free, monotone).
    np.testing.assert_array_equal(exploit_order, np.argsort(dlrm)[::-1])
    # The served (default-w) order must reorder vs pure exploit -- the blend actually blends...
    assert not np.array_equal(default_order, exploit_order), "blend did not perturb pure-DLRM order"
    # ...but must NOT collapse to the pure-bonus order (the original swamping bug).
    assert not np.array_equal(default_order, explore_order), (
        "served order equals the pure-exploration order => bonus is swamping the DLRM score"
    )


def test_warm_user_exploratory_flag_is_meaningful():
    """is_exploratory must be a real per-item signal, not hardcoded: across a sample of warm users
    both True and False must appear (the old bug flagged EVERY item exploratory)."""
    seen = set()
    for uid in list(app_module.serving_context["user2idx"].keys())[:25]:
        resp = client.get(f"/api/v1/recommend_v3/{int(uid)}?top_k=10")
        if resp.status_code != 200:
            continue
        for rec in resp.json()["recommendations"]:
            seen.add(bool(rec["is_exploratory"]))
        if seen == {True, False}:
            break
    assert seen == {True, False}, f"is_exploratory not a meaningful mix across warm users: saw {seen}"


def test_cold_start_order_differs_from_raw_popularity():
    """A cold user's blended order must DIFFER from the raw static popularity list. Originally the
    popularity counts (~hundreds) swamped the bonus (~0.1), so the order was identical."""
    resp = client.get(f"/api/v1/recommend_v3/{_cold_user_id()}?top_k=20")
    assert resp.status_code == 200
    served_ids = [rec["item_id"] for rec in resp.json()["recommendations"]]

    # Raw static popularity list (what /recommend and /recommend_v2 return), same top_k.
    raw_popular_ids = [it["item_id"] for it in app_module.popular_items[: len(served_ids)]]

    assert served_ids != raw_popular_ids, (
        "cold-start order identical to the static popularity list => bonus not affecting rank"
    )
    # Still the same SET of candidates -- exploration reorders, it does not invent new items.
    assert set(served_ids) <= {it["item_id"] for it in app_module.popular_items}


def test_cold_start_flags_are_not_all_false():
    """If the blend reorders at all, some items must be flagged exploratory (rank changed)."""
    resp = client.get(f"/api/v1/recommend_v3/{_cold_user_id()}?top_k=20")
    assert resp.status_code == 200
    flags = [rec["is_exploratory"] for rec in resp.json()["recommendations"]]
    assert any(flags), "no item flagged exploratory => blend did not perturb the popularity order"
