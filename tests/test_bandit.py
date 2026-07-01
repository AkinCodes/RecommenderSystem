"""Tests for the contextual bandit exploration layer (models/bandit.py)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.bandit import (
    EpsilonGreedyBandit,
    LinUCBBandit,
    build_interaction_context,
    combine_explore,
    load_state,
    save_state,
)

DIM = 64
SEED = 42


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


class TestBuildInteractionContext:
    def test_single_shape_and_value(self, rng):
        user = rng.standard_normal(DIM)
        item = rng.standard_normal(DIM)
        ctx = build_interaction_context(user, item)
        assert ctx.shape == (DIM,)
        np.testing.assert_allclose(ctx, user * item)

    def test_batched_shape_and_value(self, rng):
        user = rng.standard_normal(DIM)
        items = rng.standard_normal((5, DIM))
        ctx = build_interaction_context(user, items)
        assert ctx.shape == (5, DIM)
        # Each row must equal the elementwise product with the (broadcast) user vec.
        np.testing.assert_allclose(ctx, user[None, :] * items)

    def test_batched_row_matches_single(self, rng):
        user = rng.standard_normal(DIM)
        items = rng.standard_normal((3, DIM))
        batched = build_interaction_context(user, items)
        for i in range(3):
            np.testing.assert_allclose(batched[i], build_interaction_context(user, items[i]))

    def test_dim_mismatch_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            build_interaction_context(np.ones(4), np.ones(5))

    def test_bad_user_ndim_raises(self):
        with pytest.raises(ValueError, match="user_emb must be 1D"):
            build_interaction_context(np.ones((2, 4)), np.ones(4))


class TestLinUCBShermanMorrison:
    def test_ainv_matches_true_inverse(self, rng):
        """After many updates, A_inv must equal np.linalg.inv(A) reconstructed from updates."""
        l2 = 1.0
        bandit = LinUCBBandit(dim=DIM, alpha=1.0, l2=l2)

        A_ref = l2 * np.eye(DIM)
        for _ in range(50):
            x = rng.standard_normal(DIM)
            reward = float(rng.random())
            bandit.update(x, reward)
            A_ref += np.outer(x, x)

        # The bandit's own A must match the independently reconstructed A.
        np.testing.assert_allclose(bandit.A, A_ref, atol=1e-8)
        # And the incrementally maintained A_inv must match a fresh inversion.
        np.testing.assert_allclose(bandit.A_inv, np.linalg.inv(A_ref), atol=1e-8)
        # A_inv is a genuine inverse of A.
        np.testing.assert_allclose(bandit.A @ bandit.A_inv, np.eye(DIM), atol=1e-8)

    def test_theta_solves_ridge(self, rng):
        bandit = LinUCBBandit(dim=DIM, alpha=1.0, l2=1.0)
        for _ in range(20):
            bandit.update(rng.standard_normal(DIM), float(rng.random()))
        # theta must satisfy A theta = b.
        np.testing.assert_allclose(bandit.A @ bandit.theta, bandit.b, atol=1e-8)

    def test_update_batch_equals_sequential(self, rng):
        contexts = rng.standard_normal((10, DIM))
        rewards = rng.random(10)

        b1 = LinUCBBandit(dim=DIM)
        for x, r in zip(contexts, rewards):
            b1.update(x, r)

        b2 = LinUCBBandit(dim=DIM)
        b2.update_batch(contexts, rewards)

        np.testing.assert_allclose(b1.A, b2.A, atol=1e-10)
        np.testing.assert_allclose(b1.A_inv, b2.A_inv, atol=1e-10)
        np.testing.assert_allclose(b1.theta, b2.theta, atol=1e-10)


class TestLinUCBLearning:
    def test_repeated_positive_reward_increases_mean(self):
        """Seeing a fixed context with reward=1 pushes exploit_mean toward the reward."""
        bandit = LinUCBBandit(dim=DIM, alpha=1.0, l2=1.0)
        x = np.ones(DIM) / np.sqrt(DIM)  # unit-norm context

        means = []
        for _ in range(30):
            bandit.update(x, 1.0)
            means.append(float(bandit.exploit_mean(x)[0]))

        # Monotonically increasing toward the reward, and bounded above by 1.
        assert means[-1] > means[0]
        for a, b in zip(means, means[1:]):
            assert b >= a - 1e-9
        assert means[-1] < 1.0
        assert means[-1] > 0.9  # after 30 updates it should be close to the reward

    def test_uncertainty_shrinks_with_observations(self):
        """exploration_bonus for a context DECREASES as that context is seen."""
        bandit = LinUCBBandit(dim=DIM, alpha=1.0, l2=1.0)
        x = np.ones(DIM) / np.sqrt(DIM)

        bonus_before = float(bandit.exploration_bonus(x)[0])
        for _ in range(30):
            bandit.update(x, 1.0)
        bonus_after = float(bandit.exploration_bonus(x)[0])

        assert bonus_after < bonus_before

    def test_bonus_nonnegative(self, rng):
        bandit = LinUCBBandit(dim=DIM, alpha=1.0, l2=1.0)
        for _ in range(10):
            bandit.update(rng.standard_normal(DIM), float(rng.random()))
        contexts = rng.standard_normal((20, DIM))
        bonuses = bandit.exploration_bonus(contexts)
        assert np.all(bonuses >= 0.0)

    def test_unseen_context_has_higher_bonus(self):
        """A well-seen direction has a smaller bonus than an orthogonal unseen one."""
        bandit = LinUCBBandit(dim=DIM, alpha=1.0, l2=1.0)
        seen = np.zeros(DIM)
        seen[0] = 1.0
        unseen = np.zeros(DIM)
        unseen[1] = 1.0  # orthogonal direction, never observed

        for _ in range(50):
            bandit.update(seen, 1.0)

        assert float(bandit.exploration_bonus(unseen)[0]) > float(bandit.exploration_bonus(seen)[0])

    def test_score_components_add_up(self, rng):
        bandit = LinUCBBandit(dim=DIM, alpha=1.0, l2=1.0)
        for _ in range(10):
            bandit.update(rng.standard_normal(DIM), float(rng.random()))
        contexts = rng.standard_normal((7, DIM))
        total, mean, bonus = bandit.score(contexts, return_components=True)
        np.testing.assert_allclose(total, mean + bonus)
        np.testing.assert_allclose(total, bandit.score(contexts))

    def test_rank_orders_by_score_desc(self, rng):
        bandit = LinUCBBandit(dim=DIM, alpha=1.0, l2=1.0)
        for _ in range(10):
            bandit.update(rng.standard_normal(DIM), float(rng.random()))
        contexts = rng.standard_normal((15, DIM))
        order = bandit.rank(contexts)
        ranked_scores = bandit.score(contexts)[order]
        assert np.all(np.diff(ranked_scores) <= 1e-12)


class TestCombineExplore:
    """The common-scale blend that fixes the exploit/bonus scale mismatch."""

    def test_w_zero_matches_exploit_order(self, rng):
        base = rng.standard_normal(30)
        bonus = rng.random(30) * 100.0  # wildly larger scale than base
        combined = combine_explore(base, bonus, w=0.0)
        # w=0 => pure exploit: the blend order must equal the base order.
        np.testing.assert_array_equal(np.argsort(combined)[::-1], np.argsort(base)[::-1])

    def test_w_one_matches_bonus_order(self, rng):
        base = rng.standard_normal(30) * 1000.0  # base far larger than bonus
        bonus = rng.random(30)
        combined = combine_explore(base, bonus, w=1.0)
        np.testing.assert_array_equal(np.argsort(combined)[::-1], np.argsort(bonus)[::-1])

    def test_scale_invariance_in_base(self, rng):
        base = rng.standard_normal(25)
        bonus = rng.random(25)
        a = combine_explore(base, bonus, w=0.3)
        # Multiplying base by a positive constant (and shifting) must not change the blend,
        # because base is min-max normalized inside combine_explore.
        b = combine_explore(base * 17.0 + 5.0, bonus, w=0.3)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_scale_invariance_in_bonus(self, rng):
        base = rng.standard_normal(25)
        bonus = rng.random(25)
        a = combine_explore(base, bonus, w=0.3)
        b = combine_explore(base, bonus * 250.0 + 9.0, w=0.3)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_output_in_unit_interval(self, rng):
        base = rng.standard_normal(40) * 50.0
        bonus = rng.random(40) * 3.0
        out = combine_explore(base, bonus, w=0.4)
        assert out.min() >= -1e-12
        assert out.max() <= 1.0 + 1e-12

    def test_degenerate_all_equal_base(self):
        base = np.full(10, 7.0)  # no spread -> normalizes to zeros
        bonus = np.arange(10, dtype=float)
        out = combine_explore(base, bonus, w=0.5)
        # base contributes nothing; order is driven entirely by the bonus.
        np.testing.assert_array_equal(np.argsort(out)[::-1], np.argsort(bonus)[::-1])

    def test_blend_can_reorder_vs_pure_exploit(self, rng):
        # A middling w must be able to change the top-1 vs pure exploit when a low-exploit item has
        # a high bonus. Construct exactly that.
        base = np.array([1.0, 0.9, 0.2])
        bonus = np.array([0.0, 0.0, 10.0])  # last item is very uncertain
        # w=0.0: norm(base)=[1, 0.875, 0] -> top is item 0.
        exploit_top = int(np.argmax(combine_explore(base, bonus, w=0.0)))
        # w=0.6: 0.4*[1,0.875,0] + 0.6*[0,0,1] = [0.4, 0.35, 0.6] -> the high-bonus item wins.
        blend_top = int(np.argmax(combine_explore(base, bonus, w=0.6)))
        assert exploit_top == 0
        assert blend_top == 2  # the high-bonus item wins the blend

    def test_invalid_w_raises(self):
        with pytest.raises(ValueError, match="w must be in"):
            combine_explore(np.ones(3), np.ones(3), w=1.5)


class TestEpsilonGreedy:
    def test_epsilon_zero_is_pure_exploit(self, rng):
        scores = rng.standard_normal(20)
        eg = EpsilonGreedyBandit(epsilon=0.0, seed=SEED)
        np.testing.assert_array_equal(eg.rank(scores), np.argsort(scores)[::-1])

    def test_epsilon_one_is_not_exploit(self, rng):
        scores = rng.standard_normal(20)
        eg = EpsilonGreedyBandit(epsilon=1.0, seed=SEED)
        exploit = np.argsort(scores)[::-1]
        order = eg.rank(scores)
        # Fully random -> a permutation of all indices, but not the exploit order.
        assert sorted(order.tolist()) == list(range(20))
        assert not np.array_equal(order, exploit)

    def test_determinism_same_seed(self, rng):
        scores = rng.standard_normal(20)
        a = EpsilonGreedyBandit(epsilon=0.3, seed=SEED)
        b = EpsilonGreedyBandit(epsilon=0.3, seed=SEED)
        np.testing.assert_array_equal(a.rank(scores), b.rank(scores))

    def test_different_seed_differs(self, rng):
        scores = rng.standard_normal(20)
        a = EpsilonGreedyBandit(epsilon=0.5, seed=1)
        b = EpsilonGreedyBandit(epsilon=0.5, seed=2)
        # Overwhelmingly likely to differ for a partly-random policy over 20 items.
        assert not np.array_equal(a.rank(scores), b.rank(scores))

    def test_rank_is_permutation(self, rng):
        scores = rng.standard_normal(20)
        eg = EpsilonGreedyBandit(epsilon=0.4, seed=SEED)
        order = eg.rank(scores)
        assert sorted(order.tolist()) == list(range(20))

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            EpsilonGreedyBandit(epsilon=1.5)


class TestPersistence:
    def test_save_load_roundtrip_preserves_ranking(self, rng, tmp_path):
        bandit = LinUCBBandit(dim=DIM, alpha=1.3, l2=2.0)
        for _ in range(25):
            bandit.update(rng.standard_normal(DIM), float(rng.random()))

        path = str(tmp_path / "bandit_state.pkl")
        save_state(bandit, path)
        loaded = load_state(path)

        assert loaded.dim == bandit.dim
        assert loaded.alpha == bandit.alpha
        assert loaded.l2 == bandit.l2
        np.testing.assert_allclose(loaded.A, bandit.A)
        np.testing.assert_allclose(loaded.A_inv, bandit.A_inv)
        np.testing.assert_allclose(loaded.b, bandit.b)
        np.testing.assert_allclose(loaded.theta, bandit.theta)

        contexts = rng.standard_normal((12, DIM))
        np.testing.assert_array_equal(loaded.rank(contexts), bandit.rank(contexts))
        np.testing.assert_allclose(loaded.score(contexts), bandit.score(contexts))

    def test_loaded_bandit_can_continue_updating(self, rng, tmp_path):
        bandit = LinUCBBandit(dim=DIM)
        for _ in range(5):
            bandit.update(rng.standard_normal(DIM), 1.0)
        path = str(tmp_path / "b.pkl")
        save_state(bandit, path)
        loaded = load_state(path)
        # Should update without error and keep A_inv consistent.
        loaded.update(rng.standard_normal(DIM), 1.0)
        np.testing.assert_allclose(loaded.A @ loaded.A_inv, np.eye(DIM), atol=1e-8)

    def test_save_is_atomic_no_tmp_left_behind(self, rng, tmp_path):
        bandit = LinUCBBandit(dim=DIM)
        bandit.update(rng.standard_normal(DIM), 1.0)
        path = str(tmp_path / "atomic.pkl")
        save_state(bandit, path)
        # The temp file must have been renamed away (atomic os.replace), not left dangling.
        assert not os.path.exists(path + ".tmp")
        assert os.path.exists(path)
        # And re-saving over an existing file must still succeed.
        save_state(bandit, path)
        assert load_state(path).dim == DIM
