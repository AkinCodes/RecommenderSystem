"""Contextual bandit exploration layer for the DLRM recommender.

Pure NumPy — no torch dependency. Embeddings (user/item) are produced elsewhere
(e.g. by ``DLRMModel.embeddings``) and passed in here as ``np.ndarray``.

The bandit operates on an *interaction context*: the elementwise product of the
user embedding and an item embedding. This mirrors the DLRM's own interaction
term (``cat_embeds[0] * cat_embeds[1]`` in ``models/dlrm.py``), so the bandit
"sees" the same user x item signal the ranking model does. Because arm identity
is carried by the item embedding *inside* the context (rather than by a separate
per-arm weight matrix), a single shared-weight LinUCB model generalizes to cold
items that were never explicitly seen during updates.

Context dimensionality equals the shared embedding dim (64 in configs/config.yaml).
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


def build_interaction_context(user_emb: np.ndarray, item_emb: np.ndarray) -> np.ndarray:
    """Build the interaction context = elementwise(user_emb, item_emb).

    Mirrors the DLRM interaction term (``user_embed * item_embed``, elementwise).

    Shape contract:
      - Single item:   user_emb (dim,)  x  item_emb (dim,)      -> context (dim,)
      - Batched items: user_emb (dim,)  x  item_emb (n_items, dim) -> context (n_items, dim)
        (the 1D user_emb is broadcast against every row of the 2D item matrix)

    Args:
        user_emb: 1D user embedding of shape (dim,).
        item_emb: 1D item embedding (dim,) or 2D item matrix (n_items, dim).

    Returns:
        Interaction context: (dim,) for a single item, or (n_items, dim) batched.
    """
    user_emb = np.asarray(user_emb, dtype=np.float64)
    item_emb = np.asarray(item_emb, dtype=np.float64)

    if user_emb.ndim != 1:
        raise ValueError(f"user_emb must be 1D (dim,), got shape {user_emb.shape}")

    if item_emb.ndim == 1:
        if item_emb.shape[0] != user_emb.shape[0]:
            raise ValueError(
                f"Dimension mismatch: user_emb dim {user_emb.shape[0]} != item_emb dim {item_emb.shape[0]}"
            )
        return user_emb * item_emb

    if item_emb.ndim == 2:
        if item_emb.shape[1] != user_emb.shape[0]:
            raise ValueError(
                f"Dimension mismatch: user_emb dim {user_emb.shape[0]} != item_emb dim {item_emb.shape[1]}"
            )
        # Broadcast user_emb (dim,) against each row of item_emb (n_items, dim).
        return user_emb[None, :] * item_emb

    raise ValueError(f"item_emb must be 1D (dim,) or 2D (n_items, dim), got shape {item_emb.shape}")


def _as_2d(contexts: np.ndarray) -> np.ndarray:
    """Coerce a context input into a 2D (n, dim) array without copying when possible."""
    contexts = np.asarray(contexts, dtype=np.float64)
    if contexts.ndim == 1:
        return contexts[None, :]
    if contexts.ndim == 2:
        return contexts
    raise ValueError(f"contexts must be 1D (dim,) or 2D (n, dim), got shape {contexts.shape}")


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Min-max normalize a 1D array to [0, 1]. All-equal input -> zeros (degenerate guard).

    Scale-free: adding a constant or multiplying by a positive constant does not change the
    relative ordering, and after normalization the range is always exactly [0, 1] (or all-zeros).
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    span = hi - lo
    if span <= 0.0:
        # All values equal (or single element): no information to rank on -> flat zeros.
        return np.zeros_like(values)
    return (values - lo) / span


def combine_explore(base: np.ndarray, bonus: np.ndarray, w: float) -> np.ndarray:
    """Common-scale convex blend of an exploit score and an exploration bonus.

    The core fix for the exploration layer: the DLRM exploit score and the LinUCB bonus live on
    completely different scales (a sigmoid in ~[0.9, 1.0] vs. an unbounded ``alpha·√(xᵀA⁻¹x)``),
    so a raw ``base + bonus`` sum does not *blend* -- it ranks by whichever term is numerically
    larger. This helper min-max normalizes BOTH terms across the candidate set to [0, 1] and then
    convex-combines them::

        final = (1 - w) * norm(base) + w * norm(bonus)

    ``w`` (the explore weight) is the single user-facing tradeoff knob:
      - ``w = 0`` -> pure exploit (ranking identical to ``base`` order),
      - ``w = 1`` -> pure explore (ranking identical to ``bonus`` order),
      - in between -> a genuine, scale-free blend.

    The LinUCB ``alpha`` still shapes the *shape* of the bonus (which contexts look uncertain),
    but because the bonus is renormalized here, ``alpha`` no longer competes with the base on an
    accidental scale -- ``w`` is what trades exploit vs. explore.

    Args:
        base: (n,) exploit scores for the candidate set (e.g. served DLRM sigmoid scores).
        bonus: (n,) exploration bonuses for the same candidates (>= 0).
        w: explore weight in [0, 1].

    Returns:
        (n,) blended scores in [0, 1]; rank descending to present.
    """
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"explore weight w must be in [0, 1], got {w}")
    base = np.asarray(base, dtype=np.float64).reshape(-1)
    bonus = np.asarray(bonus, dtype=np.float64).reshape(-1)
    if base.shape != bonus.shape:
        raise ValueError(f"base shape {base.shape} != bonus shape {bonus.shape}")
    nb = _minmax_normalize(base)
    nx = _minmax_normalize(bonus)
    return (1.0 - w) * nb + w * nx


class LinUCBBandit:
    """Single-model (shared-weight) LinUCB contextual bandit.

    Disjoint LinUCB keeps one weight vector per arm; this is the *shared* variant:
    one linear model over the context vector. Arm identity lives inside the context
    (via the item embedding), so the learned weights generalize across items,
    including cold ones.

    Maintains the ridge-regression sufficient statistics:
        A     = l2 * I + sum_t x_t x_t^T      (dim x dim)
        b     = sum_t reward_t * x_t          (dim,)
        theta = A^{-1} b                        (dim,)

    ``A_inv`` is kept in sync with ``A`` incrementally via the Sherman-Morrison
    rank-1 update, avoiding a full matrix inversion on every step.
    """

    def __init__(self, dim: int, alpha: float = 1.0, l2: float = 1.0):
        """
        Args:
            dim: Context dimensionality (== embedding dim, 64 in this repo).
            alpha: Exploration coefficient (higher => more exploration).
            l2: Ridge regularization strength; A is initialized to l2 * I.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if l2 <= 0:
            raise ValueError(f"l2 must be positive (A must be invertible), got {l2}")

        self.dim = int(dim)
        self.alpha = float(alpha)
        self.l2 = float(l2)

        self.A = l2 * np.eye(dim, dtype=np.float64)
        self.A_inv = (1.0 / l2) * np.eye(dim, dtype=np.float64)
        self.b = np.zeros(dim, dtype=np.float64)
        self.theta = self.A_inv @ self.b

    # --- scoring -----------------------------------------------------------

    def exploit_mean(self, contexts: np.ndarray) -> np.ndarray:
        """Exploitation term: theta . x for each row. Returns shape (n,)."""
        x = _as_2d(contexts)
        return x @ self.theta

    def exploration_bonus(self, contexts: np.ndarray) -> np.ndarray:
        """Exploration bonus: alpha * sqrt(x^T A_inv x) per row. Always >= 0.

        Returns shape (n,). The quadratic form is clamped at 0 before the sqrt to
        guard against tiny negative values from floating-point error.
        """
        x = _as_2d(contexts)
        # Row-wise quadratic form x_i^T A_inv x_i without forming the full matrix.
        quad = np.einsum("ij,jk,ik->i", x, self.A_inv, x)
        quad = np.maximum(quad, 0.0)
        return self.alpha * np.sqrt(quad)

    def score(self, contexts: np.ndarray, return_components: bool = False):
        """UCB score = exploit_mean + exploration_bonus.

        Args:
            contexts: (dim,) or (n, dim).
            return_components: if True, return (score, mean, bonus); else just score.

        Returns:
            score of shape (n,), or a (score, mean, bonus) tuple.
        """
        mean = self.exploit_mean(contexts)
        bonus = self.exploration_bonus(contexts)
        total = mean + bonus
        if return_components:
            return total, mean, bonus
        return total

    def rank(self, contexts: np.ndarray) -> np.ndarray:
        """Return row indices sorted by UCB score descending. Shape (n,)."""
        scores = self.score(contexts)
        return np.argsort(scores)[::-1]

    # --- learning ----------------------------------------------------------

    def update(self, context: np.ndarray, reward: float) -> None:
        """Rank-1 update from a single (context, reward) observation.

        Updates A += x x^T, b += reward * x, and refreshes A_inv via
        Sherman-Morrison plus theta = A_inv @ b.
        """
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(f"context dim {x.shape[0]} != bandit dim {self.dim}")
        reward = float(reward)

        # A <- A + x x^T
        self.A += np.outer(x, x)

        # Sherman-Morrison: (A + x x^T)^{-1} = A_inv - (A_inv x x^T A_inv) / (1 + x^T A_inv x)
        Ainv_x = self.A_inv @ x
        denom = 1.0 + float(x @ Ainv_x)
        self.A_inv -= np.outer(Ainv_x, Ainv_x) / denom

        self.b += reward * x
        self.theta = self.A_inv @ self.b

    def update_batch(self, contexts: np.ndarray, rewards: np.ndarray) -> None:
        """Apply a sequence of rank-1 updates. contexts (n, dim), rewards (n,)."""
        x = _as_2d(contexts)
        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        if x.shape[0] != rewards.shape[0]:
            raise ValueError(f"n_contexts {x.shape[0]} != n_rewards {rewards.shape[0]}")
        for i in range(x.shape[0]):
            self.update(x[i], rewards[i])


class EpsilonGreedyBandit:
    """Epsilon-greedy exploration baseline (the control arm).

    With probability (1 - epsilon) it exploits (ranks by ``base_scores`` desc);
    with probability epsilon it injects randomness. Deterministic under ``seed``:
    two instances with the same seed produce identical ranks for identical inputs.
    """

    def __init__(self, epsilon: float, seed: int = 42):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.epsilon = float(epsilon)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

    def rank(self, base_scores: np.ndarray) -> np.ndarray:
        """Return item indices ordered for presentation.

        epsilon == 0 => pure exploit: argsort(base_scores)[::-1].
        epsilon == 1 => fully random permutation.
        In between, each position independently exploits (prob 1-eps) the best
        remaining item or (prob eps) picks a uniformly random remaining item.
        """
        scores = np.asarray(base_scores, dtype=np.float64).reshape(-1)
        n = scores.shape[0]
        if n == 0:
            return np.array([], dtype=np.intp)

        exploit_order = np.argsort(scores)[::-1]

        if self.epsilon == 0.0:
            return exploit_order.astype(np.intp)

        if self.epsilon == 1.0:
            return self.rng.permutation(n).astype(np.intp)

        # Sequential selection: pop from the exploit order or from remaining at random.
        remaining = list(exploit_order)
        result: list[int] = []
        while remaining:
            if self.rng.random() < self.epsilon:
                pick_idx = int(self.rng.integers(len(remaining)))
            else:
                pick_idx = 0  # best remaining by base_scores
            result.append(int(remaining.pop(pick_idx)))
        return np.array(result, dtype=np.intp)


# --- persistence -----------------------------------------------------------


@dataclass
class BanditState:
    """Serializable snapshot of a LinUCBBandit (mirrors serving_context.pkl)."""

    dim: int
    alpha: float
    l2: float
    A: np.ndarray
    A_inv: np.ndarray
    b: np.ndarray
    theta: np.ndarray


def _to_state(bandit: LinUCBBandit) -> BanditState:
    return BanditState(
        dim=bandit.dim,
        alpha=bandit.alpha,
        l2=bandit.l2,
        A=bandit.A.copy(),
        A_inv=bandit.A_inv.copy(),
        b=bandit.b.copy(),
        theta=bandit.theta.copy(),
    )


def _from_state(state: BanditState) -> LinUCBBandit:
    bandit = LinUCBBandit(dim=state.dim, alpha=state.alpha, l2=state.l2)
    bandit.A = np.asarray(state.A, dtype=np.float64).copy()
    bandit.A_inv = np.asarray(state.A_inv, dtype=np.float64).copy()
    bandit.b = np.asarray(state.b, dtype=np.float64).copy()
    bandit.theta = np.asarray(state.theta, dtype=np.float64).copy()
    return bandit


def save_state(bandit: LinUCBBandit, path: str) -> None:
    """Atomically pickle a LinUCBBandit's state to ``path`` (mirrors serving_context.pkl).

    Writes to ``path + ".tmp"`` and then ``os.replace(tmp, path)``. ``os.replace`` is atomic on
    POSIX, so a concurrent reader (or a crash mid-write) never observes a truncated/corrupt pickle:
    it sees either the old file or the fully-written new one. The caller is responsible for
    serializing concurrent *writers* (the /feedback handler holds a lock); this only guards
    reader/crash consistency.
    """
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(_to_state(bandit), f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    logger.info("Saved bandit state to %s", path)


def load_state(path: str) -> LinUCBBandit:
    """Load a LinUCBBandit previously written by ``save_state``."""
    with open(path, "rb") as f:
        state = pickle.load(f)
    return _from_state(state)
