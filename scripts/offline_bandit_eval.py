"""Off-policy (offline) evaluation of the bandit exploration layer on MovieLens logged data.

This is the "proof it works" for the LinUCB exploration layer in ``models/bandit.py``. It
estimates, from *logged* MovieLens interactions alone (no online A/B), what per-interaction
reward a new policy (pure DLRM, the explore-weighted LinUCB bandit, or Most-Popular) would have
earned, using three standard off-policy estimators:

  1. Replay / Rejection sampling (Li et al., WSDM 2011, "Unbiased Offline Evaluation of
     Contextual-bandit-based News Article Recommendation Algorithms"). Stream the logged events
     in timestamp order; at each event the target policy ranks the available items; we KEEP the
     event only if the policy's top-1 choice equals the logged item, and average the logged
     reward over kept events. Under a *uniformly random* logging policy this is an unbiased
     estimate of the target policy's per-interaction reward.

  2. IPS (Inverse Propensity Scoring). Weight each event by 1 / pi0(a_log) and count it only
     when the new action matches the logged one:  IPS = mean( 1[a_new == a_log] / pi0(a_log) * r ).
     Unbiased when pi0 is the *true* logging propensity and has full support.

  3. SNIPS (Self-Normalised IPS, Swaminathan & Joachims 2015). Same importance weights, but
     normalised by the sum of weights instead of N. Trades a small bias for much lower variance.

------------------------------------------------------------------------------------------------
WHAT POLICY IS ACTUALLY EVALUATED (this matches what /recommend_v3 serves)
------------------------------------------------------------------------------------------------
The exploit signal here is the REAL served DLRM sigmoid score: for every candidate we build the
exact same 8-dim continuous feature vector ``recommend_v3``/``_build_user_item_features`` build
from ``serving_context`` and run the FULL model forward (continuous MLP + user/item embeddings +
interaction term + sigmoid). It is NOT ``ctx.sum()`` (a raw embedding dot-product), which was the
previous, mislabelled proxy.

The bandit policy ranks candidates by the SAME common-scale blend the API uses:

    combine_explore(exploit=dlrm_sigmoid, bonus=LinUCB.exploration_bonus, w)

i.e. both terms min-max-normalized to [0, 1] across the candidate set, then convex-combined as
``(1 - w) * norm(dlrm) + w * norm(bonus)``. ``w = 0`` reproduces pure DLRM exactly. The bandit
still learns online (``LinUCB.update`` on every retained event) so the bonus reflects accumulated
uncertainty, but the exploit term is the served DLRM score, not ``theta·x``.

We sweep ``w in {0.0, 0.1, 0.2, 0.3, 0.5, 1.0}`` and record SNIPS reward, distinct-item coverage,
and mean popularity rank per w, and save an **explore-weight frontier** plot
(``experiments/results/bandit_explore_frontier.png``): x = SNIPS reward, y = distinct-item
coverage, one point per w for the bandit, plus DLRM and Popularity as reference stars.

------------------------------------------------------------------------------------------------
ASSUMPTIONS AND THEIR VIOLATIONS (read this before trusting the numbers)
------------------------------------------------------------------------------------------------
MovieLens is an *observational, self-selected* dataset: users chose which movies to rate. It was
NOT collected under a known randomised logging policy. The replay estimator's unbiasedness
theorem assumes uniform-random logging; the IPS/SNIPS unbiasedness assumes we know the true
propensity pi0(a | context). We do NOT. We therefore use a *documented stand-in* for pi0: a
softmax over training-set item popularity (popular items were more likely to be logged),
RENORMALIZED WITHIN each event's candidate pool (not a global softmax over all 1682 items -- a
global normalization makes the absolute IPS scale meaningless because the pool the policy actually
chooses from is tiny). Consequences:

  * The absolute IPS reward estimates are still biased (selection bias is not removed). **SNIPS is
    the estimator to trust for cross-policy comparison** -- it is invariant to a constant scaling
    of the weights and has far lower variance. Treat absolute IPS as uninterpretable here.
  * Popularity's higher *replay* reward is a self-selection / replay artifact: Most-Popular
    retains a tiny, favorably-biased subset of events (the ones where the single most-popular
    candidate happened to be the logged, liked item), so its retained mean is high on a handful of
    events. It is NOT evidence that popularity beats DLRM; the honest cross-policy signal is SNIPS
    plus coverage, reported on the frontier.
  * Replay additionally suffers from low effective sample size: only events where the policy's top
    choice matches the log are retained. We report retained counts / ESS so this is visible.

The estimator FUNCTIONS below (``replay_estimate``, ``ips_estimate``, ``snips_estimate``,
``popularity_propensities``) are pure and import-safe: they have NO dependency on the trained
artifacts and are unit-tested against synthetic data with known analytic values in
``tests/test_offline_eval.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocessing import LIKE_THRESHOLD, load_movielens_data, prepare_splits  # noqa: E402
from models.bandit import LinUCBBandit, build_interaction_context, combine_explore  # noqa: E402

logger = logging.getLogger("offline_bandit_eval")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(REPO_ROOT, "trained_model_movielens.pth")
SERVING_CONTEXT_PATH = os.path.join(REPO_ROOT, "serving_context.pkl")
DATA_PATH = os.path.join(REPO_ROOT, "data", "ml-100k", "u.data")
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")

EMBEDDING_DIM = 64  # == DLRM embedding_dim
N_CONTINUOUS = 8  # == model.num_features; matches _build_user_item_features in api/app.py

# Explore-weight sweep. w=0.0 == pure DLRM; w=1.0 == pure exploration.
EXPLORE_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]


# ================================================================================================
# Pure off-policy estimators (import-safe, unit-tested on synthetic data)
# ================================================================================================


def popularity_propensities(item_counts: np.ndarray, temperature: float = 1.0, floor: float = 1e-6) -> np.ndarray:
    """Documented stand-in logging propensity pi0 over items: softmax over popularity.

    pi0(item) proportional to exp(count / (temperature * max_count)). More-logged (popular) items
    get higher logging probability, mimicking MovieLens self-selection. A ``floor`` is added so no
    item has zero support (IPS requires full support / non-zero propensities).

    Args:
        item_counts: (n_items,) training interaction counts per item index.
        temperature: softmax temperature; higher => flatter (closer to uniform).
        floor: minimum probability mass added to every item before renormalising.

    Returns:
        (n_items,) probability vector summing to 1.
    """
    counts = np.asarray(item_counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError(f"item_counts must be 1D non-empty, got shape {counts.shape}")
    scale = counts.max() if counts.max() > 0 else 1.0
    logits = counts / (temperature * scale)
    logits -= logits.max()  # numerical stability
    probs = np.exp(logits)
    probs = probs + floor
    probs /= probs.sum()
    return probs


def replay_estimate(policy_actions: np.ndarray, logged_actions: np.ndarray, rewards: np.ndarray) -> dict:
    """Li et al. (2011) replay / rejection-sampling estimator.

    Retain events where the policy's top-1 action equals the logged action; the estimate is the
    mean logged reward over retained events. Unbiased under uniform-random logging.

    Args:
        policy_actions: (T,) top-1 item the target policy would show at each event.
        logged_actions: (T,) item actually logged at each event.
        rewards: (T,) reward actually observed at each event (0/1).

    Returns:
        dict with ``value`` (estimated per-interaction reward, nan if nothing retained),
        ``retained`` (count), ``total`` (T), and ``match_rate``.
    """
    policy_actions = np.asarray(policy_actions)
    logged_actions = np.asarray(logged_actions)
    rewards = np.asarray(rewards, dtype=np.float64)
    total = len(logged_actions)
    matches = policy_actions == logged_actions
    retained = int(matches.sum())
    value = float(rewards[matches].mean()) if retained > 0 else float("nan")
    return {"value": value, "retained": retained, "total": total, "match_rate": retained / total if total else 0.0}


def ips_estimate(
    policy_actions: np.ndarray, logged_actions: np.ndarray, rewards: np.ndarray, propensities: np.ndarray
) -> dict:
    """Inverse Propensity Scoring estimator (deterministic target policy).

    IPS = (1/T) * sum_t 1[a_new_t == a_log_t] / pi0(a_log_t) * r_t.

    NOTE (this repo): pi0 is renormalized WITHIN each candidate pool, so absolute IPS is not on an
    interpretable scale; use SNIPS for cross-policy comparison. See the module docstring.

    Args:
        policy_actions: (T,) top-1 item the target policy picks per event.
        logged_actions: (T,) logged item per event.
        rewards: (T,) observed reward per event.
        propensities: (T,) logging propensity pi0(a_log_t) of the LOGGED action at each event.

    Returns:
        dict with ``value`` (IPS estimate), ``ess`` (effective sample size of the weights),
        ``retained`` (matched events), ``total``.
    """
    policy_actions = np.asarray(policy_actions)
    logged_actions = np.asarray(logged_actions)
    rewards = np.asarray(rewards, dtype=np.float64)
    propensities = np.asarray(propensities, dtype=np.float64)
    total = len(logged_actions)
    if total == 0:
        return {"value": float("nan"), "ess": 0.0, "retained": 0, "total": 0}

    matches = (policy_actions == logged_actions).astype(np.float64)
    weights = matches / propensities  # zero for non-matched events
    value = float((weights * rewards).mean())
    ess = _effective_sample_size(weights)
    return {"value": value, "ess": ess, "retained": int(matches.sum()), "total": total}


def snips_estimate(
    policy_actions: np.ndarray, logged_actions: np.ndarray, rewards: np.ndarray, propensities: np.ndarray
) -> dict:
    """Self-Normalised IPS: same weights as IPS but divided by the sum of weights.

    SNIPS = sum_t w_t r_t / sum_t w_t,  with w_t = 1[a_new == a_log] / pi0(a_log).

    Lower variance than IPS (invariant to a constant scaling of the weights) at the cost of a
    small finite-sample bias. This is the estimator to trust for cross-policy comparison here.
    """
    policy_actions = np.asarray(policy_actions)
    logged_actions = np.asarray(logged_actions)
    rewards = np.asarray(rewards, dtype=np.float64)
    propensities = np.asarray(propensities, dtype=np.float64)
    total = len(logged_actions)
    if total == 0:
        return {"value": float("nan"), "ess": 0.0, "retained": 0, "total": 0}

    matches = (policy_actions == logged_actions).astype(np.float64)
    weights = matches / propensities
    wsum = weights.sum()
    value = float((weights * rewards).sum() / wsum) if wsum > 0 else float("nan")
    ess = _effective_sample_size(weights)
    return {"value": value, "ess": ess, "retained": int(matches.sum()), "total": total}


def _effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size: (sum w)^2 / sum w^2. Measures variance inflation from weights."""
    weights = np.asarray(weights, dtype=np.float64)
    denom = float((weights ** 2).sum())
    if denom == 0:
        return 0.0
    return float(weights.sum() ** 2 / denom)


# ================================================================================================
# Served DLRM scorer (the REAL sigmoid, not a dot-product proxy)
# ================================================================================================


class DLRMScorer:
    """Runs the full served DLRM forward to produce the sigmoid score per candidate.

    Replicates ``_build_user_item_features`` from ``api/app.py``: an 8-dim continuous feature
    vector (4 user feats, 3 item feats, 1 interaction feat) plus categorical ``[user_idx,
    item_idx]``, then a full model forward + sigmoid -- exactly the exploit signal /recommend_v3
    serves. Candidate scoring is vectorized (one forward per event over its candidate pool).
    """

    def __init__(self, model, user_features: dict, item_features: dict):
        import torch

        self._torch = torch
        self.model = model
        self.user_features = user_features
        self.item_features = item_features

    def _build_cont(self, user_idx: int, candidate_items: np.ndarray) -> "np.ndarray":
        n = len(candidate_items)
        cont = np.zeros((n, N_CONTINUOUS), dtype=np.float32)
        user_feats = np.asarray(self.user_features[user_idx], dtype=np.float32)
        cont[:, 0:4] = user_feats[0:4]
        item_mat = np.zeros((n, 3), dtype=np.float32)
        for i, item_idx in enumerate(candidate_items):
            item_mat[i] = self.item_features.get(int(item_idx), [0.6, 0.0, 0.0])
        cont[:, 4:7] = item_mat
        cont[:, 7] = (user_feats[0] - item_mat[:, 0] + 1.0) / 2.0
        return cont

    def scores(self, user_idx: int, candidate_items: np.ndarray) -> np.ndarray:
        torch = self._torch
        cont = self._build_cont(user_idx, candidate_items)
        cont_t = torch.tensor(cont, dtype=torch.float32)
        cat_t = torch.tensor([[int(user_idx), int(i)] for i in candidate_items], dtype=torch.int64)
        with torch.no_grad():
            out = self.model(cont_t, cat_t).squeeze()
        if out.dim() == 0:
            out = out.unsqueeze(0)
        return out.detach().numpy().astype(np.float64)


# ================================================================================================
# Policies (over the replay stream)
# ================================================================================================


class Policy:
    """A policy maps (user_idx, candidate item indices) -> a top-1 item to show."""

    name = "policy"

    def top1(self, user_idx: int, candidate_items: np.ndarray) -> int:
        raise NotImplementedError

    def observe(self, user_idx: int, item_idx: int, reward: float) -> None:
        """Optional online learning hook; default no-op."""


class DLRMPolicy(Policy):
    """Pure exploitation: rank candidates by the REAL served DLRM sigmoid score."""

    name = "dlrm"

    def __init__(self, scorer: DLRMScorer):
        self.scorer = scorer

    def top1(self, user_idx: int, candidate_items: np.ndarray) -> int:
        s = self.scorer.scores(user_idx, candidate_items)
        return int(candidate_items[int(np.argmax(s))])


class ExploreWeightedBanditPolicy(Policy):
    """Explore-weighted LinUCB: ranks by combine_explore(dlrm_sigmoid, bonus, w) -- the SAME blend
    /recommend_v3 serves. Learns online via LinUCB.update on every retained event."""

    def __init__(self, scorer: DLRMScorer, user_emb: np.ndarray, item_emb: np.ndarray, w: float,
                 alpha: float = 1.0, l2: float = 1.0):
        self.scorer = scorer
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.w = float(w)
        self.bandit = LinUCBBandit(dim=user_emb.shape[1], alpha=alpha, l2=l2)
        self.name = f"bandit_w{w:g}"

    def top1(self, user_idx: int, candidate_items: np.ndarray) -> int:
        dlrm = self.scorer.scores(user_idx, candidate_items)
        contexts = build_interaction_context(self.user_emb[user_idx], self.item_emb[candidate_items])
        bonus = self.bandit.exploration_bonus(contexts)
        combined = combine_explore(dlrm, bonus, self.w)
        return int(candidate_items[int(np.argmax(combined))])

    def observe(self, user_idx: int, item_idx: int, reward: float) -> None:
        ctx = build_interaction_context(self.user_emb[user_idx], self.item_emb[item_idx])
        self.bandit.update(ctx, reward)


class PopularityPolicy(Policy):
    """Most-Popular: always shows the most-logged candidate (ties broken by lower item index)."""

    name = "popularity"

    def __init__(self, item_counts: np.ndarray):
        self.item_counts = item_counts

    def top1(self, user_idx: int, candidate_items: np.ndarray) -> int:
        counts = self.item_counts[candidate_items]
        return int(candidate_items[int(np.argmax(counts))])


# ================================================================================================
# Replay driver
# ================================================================================================


@dataclass
class ReplayOutcome:
    policy_name: str
    replay: dict
    ips: dict
    snips: dict
    cumulative_reward: list[float]  # cumulative retained reward after each retained event
    retained_indices: list[int]  # stream positions retained by replay
    distinct_items: int
    mean_pop_rank: float  # mean popularity percentile of shown items (0=rare, 1=most popular)
    explore_weight: float | None = None
    extra: dict = field(default_factory=dict)


def _mean_pop_rank(shown_items: list[int], item_counts: np.ndarray) -> float:
    """Mean popularity percentile of the shown items (higher => more popular / less novel)."""
    if not shown_items:
        return float("nan")
    order = np.argsort(np.argsort(item_counts))  # rank of each item by count, 0..n-1
    n = len(item_counts)
    return float(np.mean([order[i] / (n - 1) if n > 1 else 0.0 for i in shown_items]))


def replay_policy(
    policy: Policy,
    users: np.ndarray,
    items: np.ndarray,
    rewards: np.ndarray,
    candidates_per_event: list[np.ndarray],
    logged_propensities: np.ndarray,
    item_counts: np.ndarray,
    explore_weight: float | None = None,
) -> ReplayOutcome:
    """Stream events in order; the policy proposes a top-1 per event; retain on match.

    The bandit policy learns online from every retained (matched) event via ``observe``.
    """
    policy_actions = np.empty(len(items), dtype=np.int64)
    cumulative: list[float] = []
    retained_indices: list[int] = []
    shown_items: list[int] = []
    running = 0.0

    for t in range(len(items)):
        u, a_log, r = int(users[t]), int(items[t]), float(rewards[t])
        cand = candidates_per_event[t]
        a_new = policy.top1(u, cand)
        policy_actions[t] = a_new
        shown_items.append(a_new)
        if a_new == a_log:
            running += r
            cumulative.append(running)
            retained_indices.append(t)
            policy.observe(u, a_log, r)  # online learning on the retained event

    replay = replay_estimate(policy_actions, items, rewards)
    ips = ips_estimate(policy_actions, items, rewards, logged_propensities)
    snips = snips_estimate(policy_actions, items, rewards, logged_propensities)

    return ReplayOutcome(
        policy_name=policy.name,
        replay=replay,
        ips=ips,
        snips=snips,
        cumulative_reward=cumulative,
        retained_indices=retained_indices,
        distinct_items=int(len(set(shown_items))),
        mean_pop_rank=_mean_pop_rank(shown_items, item_counts),
        explore_weight=explore_weight,
    )


# ================================================================================================
# Artifact loading + end-to-end run
# ================================================================================================


def _load_artifacts():
    """Load the trained DLRM model + embeddings + serving context, or None if unavailable."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SERVING_CONTEXT_PATH)):
        return None
    import torch

    from models.dlrm import DLRMModel

    with open(SERVING_CONTEXT_PATH, "rb") as f:
        ctx = pickle.load(f)

    n_users = len(ctx["user2idx"])
    n_items = len(ctx["item2idx"])
    model = DLRMModel(
        num_features=N_CONTINUOUS, cardinalities=[n_users, n_items], embedding_dim=EMBEDDING_DIM,
        mlp_layers=[128, 64, 32],
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    user_emb = model.embeddings[0].weight.detach().numpy().astype(np.float64)
    item_emb = model.embeddings[1].weight.detach().numpy().astype(np.float64)
    return ctx, model, user_emb, item_emb


def _build_stream(ctx):
    """Build the timestamp-ordered TEST replay stream + candidate sets + popularity + propensities.

    Candidate set per event = the items that user interacted with in the test split (the standard
    replay candidate pool for logged data, since those are the only items with a known reward).

    Propensities are the popularity-softmax RENORMALIZED WITHIN each event's candidate pool: the
    logged action's global pi0 divided by the sum of pi0 over that event's candidates. This keeps
    IPS/SNIPS weights on the scale of the pool the policy actually chooses from, rather than a
    global softmax over all 1682 items (which would make absolute IPS meaningless).
    """
    raw = load_movielens_data(DATA_PATH)
    (_tc, _tca, _tt, _sc, _sca, _st, user2idx, item2idx, test_raw) = prepare_splits(raw)

    # Training popularity counts, indexed by item index (0..n_items-1), as in api/app.py.
    sorted_raw = raw[np.argsort(raw[:, 3])]
    train_raw = sorted_raw[: int(len(sorted_raw) * 0.8)]
    n_items = len(item2idx)
    item_counts = np.zeros(n_items, dtype=np.float64)
    for row in train_raw:
        iid = int(row[1])
        if iid in item2idx:
            item_counts[item2idx[iid]] += 1

    # Per-user candidate pools from the test split (items with a known logged reward).
    user_pool: dict[int, list[int]] = {}
    for row in test_raw:
        uidx = user2idx[int(row[0])]
        iidx = item2idx[int(row[1])]
        user_pool.setdefault(uidx, []).append(iidx)

    pi0_global = popularity_propensities(item_counts)

    users, items, rewards, candidates, logged_propensities = [], [], [], [], []
    for row in test_raw:  # test_raw is already timestamp-sorted (prepare_splits sorts then slices)
        uidx = user2idx[int(row[0])]
        iidx = item2idx[int(row[1])]
        cand = np.array(sorted(set(user_pool[uidx])), dtype=np.int64)
        users.append(uidx)
        items.append(iidx)
        rewards.append(1.0 if int(row[2]) >= LIKE_THRESHOLD else 0.0)
        candidates.append(cand)
        # Renormalize pi0 within this event's candidate pool.
        pool_mass = float(pi0_global[cand].sum())
        logged_propensities.append(float(pi0_global[iidx]) / pool_mass if pool_mass > 0 else 1.0)

    users = np.array(users, dtype=np.int64)
    items = np.array(items, dtype=np.int64)
    rewards = np.array(rewards, dtype=np.float64)
    logged_propensities = np.array(logged_propensities, dtype=np.float64)
    return users, items, rewards, candidates, item_counts, logged_propensities


def _save_explore_frontier(outcomes: list[ReplayOutcome], path: str) -> None:
    """Plot the explore-weight coverage/reward frontier and save to ``path``.

    x = SNIPS reward, y = distinct-item coverage. One point per explore weight w for the bandit
    (annotated with w), plus DLRM and Popularity as reference stars. matplotlib is an optional
    (viz-only) dependency; if it is not installed we skip the plot and keep the JSON metrics.
    """
    try:
        import matplotlib
    except ModuleNotFoundError:
        logger.warning("matplotlib not installed; skipping frontier plot (JSON metrics still saved). "
                       "Install with `uv pip install matplotlib` to generate %s", path)
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))

    bandit_pts = [oc for oc in outcomes if oc.explore_weight is not None]
    bandit_pts.sort(key=lambda oc: oc.explore_weight)
    if bandit_pts:
        xs = [oc.snips["value"] for oc in bandit_pts]
        ys = [oc.distinct_items for oc in bandit_pts]
        ax.plot(xs, ys, "-o", color="tab:blue", label="LinUCB bandit (explore-weight sweep)", zorder=3)
        for oc in bandit_pts:
            ax.annotate(f"w={oc.explore_weight:g}", (oc.snips["value"], oc.distinct_items),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)

    refs = {oc.policy_name: oc for oc in outcomes if oc.explore_weight is None}
    if "dlrm" in refs:
        oc = refs["dlrm"]
        ax.scatter([oc.snips["value"]], [oc.distinct_items], marker="*", s=320, color="tab:green",
                   edgecolor="black", zorder=4, label="DLRM (pure exploit)")
    if "popularity" in refs:
        oc = refs["popularity"]
        ax.scatter([oc.snips["value"]], [oc.distinct_items], marker="*", s=320, color="tab:orange",
                   edgecolor="black", zorder=4, label="Most-Popular")

    ax.set_xlabel("SNIPS reward (higher = better; trust this over absolute IPS)")
    ax.set_ylabel("Distinct items shown (catalog coverage)")
    ax.set_title("Explore-weight frontier: catalog coverage vs. off-policy reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_offline_eval(alpha: float = 1.0, l2: float = 1.0) -> dict:
    """Full end-to-end offline eval: sweep the explore weight + reference policies. Returns metrics."""
    loaded = _load_artifacts()
    if loaded is None:
        logger.warning(
            "Trained artifacts not found (%s / %s). Skipping end-to-end eval; estimator functions "
            "remain importable and unit-tested.",
            MODEL_PATH,
            SERVING_CONTEXT_PATH,
        )
        return {}

    ctx, model, user_emb, item_emb = loaded
    users, items, rewards, candidates, item_counts, logged_propensities = _build_stream(ctx)
    logger.info("Replay stream: %d test events, %d items.", len(items), len(item_counts))

    scorer = DLRMScorer(model, ctx["user_features"], ctx.get("item_features", {}))

    outcomes: list[ReplayOutcome] = []

    # Reference: pure DLRM (served sigmoid) and Most-Popular.
    dlrm_oc = replay_policy(DLRMPolicy(scorer), users, items, rewards, candidates,
                            logged_propensities, item_counts)
    outcomes.append(dlrm_oc)

    # Explore-weight sweep. w=0.0 reproduces pure DLRM via combine_explore (sanity check).
    sweep: list[ReplayOutcome] = []
    for w in EXPLORE_WEIGHTS:
        policy = ExploreWeightedBanditPolicy(scorer, user_emb, item_emb, w=w, alpha=alpha, l2=l2)
        oc = replay_policy(policy, users, items, rewards, candidates, logged_propensities,
                           item_counts, explore_weight=w)
        sweep.append(oc)
        outcomes.append(oc)
        logger.info(
            "[w=%.1f] SNIPS=%.4f | replay=%.4f (retained %d/%d) | IPS=%.4f | ESS=%.1f | "
            "distinct=%d | mean_pop_rank=%.3f",
            w, oc.snips["value"], oc.replay["value"], oc.replay["retained"], oc.replay["total"],
            oc.ips["value"], oc.ips["ess"], oc.distinct_items, oc.mean_pop_rank,
        )

    pop_oc = replay_policy(PopularityPolicy(item_counts), users, items, rewards, candidates,
                           logged_propensities, item_counts)
    outcomes.append(pop_oc)

    for oc in (dlrm_oc, pop_oc):
        logger.info(
            "[%s] SNIPS=%.4f | replay=%.4f (retained %d/%d) | IPS=%.4f | ESS=%.1f | "
            "distinct=%d | mean_pop_rank=%.3f",
            oc.policy_name, oc.snips["value"], oc.replay["value"], oc.replay["retained"],
            oc.replay["total"], oc.ips["value"], oc.ips["ess"], oc.distinct_items, oc.mean_pop_rank,
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    frontier_path = os.path.join(RESULTS_DIR, "bandit_explore_frontier.png")
    _save_explore_frontier(outcomes, frontier_path)
    logger.info("Explore-weight frontier saved to %s", frontier_path)

    def _metrics_for(oc: ReplayOutcome) -> dict:
        return {
            "explore_weight": oc.explore_weight,
            "snips_value": oc.snips["value"],
            "replay_value": oc.replay["value"],
            "replay_retained": oc.replay["retained"],
            "ips_value": oc.ips["value"],
            "ips_ess": oc.ips["ess"],
            "distinct_items": oc.distinct_items,
            "mean_pop_rank": oc.mean_pop_rank,
        }

    metrics = {
        "n_events": int(len(items)),
        "n_items": int(len(item_counts)),
        "propensity_model": (
            "softmax over training-set item popularity (documented stand-in), renormalized WITHIN "
            "each candidate pool; absolute IPS uninterpretable, trust SNIPS for cross-policy"
        ),
        "served_policy": "combine_explore(dlrm_sigmoid, LinUCB_bonus, w); exploit = REAL served DLRM sigmoid",
        "explore_weights": EXPLORE_WEIGHTS,
        "reference_policies": {oc.policy_name: _metrics_for(oc) for oc in (dlrm_oc, pop_oc)},
        "explore_weight_sweep": [_metrics_for(oc) for oc in sweep],
        "notes": {
            "popularity_artifact": (
                "Popularity's higher replay reward is a self-selection/replay artifact: it retains "
                "a tiny favorably-biased subset of events, not evidence popularity beats DLRM."
            ),
            "snips_is_the_estimator_to_trust": True,
        },
    }
    json_path = os.path.join(RESULTS_DIR, "bandit_offline_eval.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    logger.info("Metrics saved to %s", json_path)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Off-policy evaluation of the bandit exploration layer.")
    parser.add_argument("--alpha", type=float, default=1.0, help="LinUCB exploration coefficient.")
    parser.add_argument("--l2", type=float, default=1.0, help="LinUCB ridge regularisation.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    metrics = run_offline_eval(alpha=args.alpha, l2=args.l2)
    if not metrics:
        logger.info("Nothing evaluated (missing artifacts). Exiting cleanly.")
        return 0

    print("\n" + "=" * 72)
    print("OFF-POLICY EVALUATION SUMMARY  (trust SNIPS; absolute IPS uninterpretable)")
    print("=" * 72)
    print("  Reference policies:")
    for name, m in metrics["reference_policies"].items():
        print(
            f"    {name:12s} SNIPS={m['snips_value']:.4f}  replay={m['replay_value']:.4f}  "
            f"distinct={m['distinct_items']}  pop_rank={m['mean_pop_rank']:.3f}"
        )
    print("  Explore-weight sweep (bandit):")
    for m in metrics["explore_weight_sweep"]:
        print(
            f"    w={m['explore_weight']:<4g} SNIPS={m['snips_value']:.4f}  replay={m['replay_value']:.4f}  "
            f"distinct={m['distinct_items']}  pop_rank={m['mean_pop_rank']:.3f}"
        )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
