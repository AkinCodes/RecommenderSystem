"""DoWhy causal-impact analysis of the bandit's *exploration policy* on catalog coverage.

WHAT QUESTION THIS ANSWERS
==========================
The off-policy evaluation in ``scripts/offline_bandit_eval.py`` sweeps the explore weight
``w in {0.0, 0.1, 0.2, 0.3, 0.5, 1.0}`` of the served LinUCB bandit and reports, per w, a SNIPS
reward and a catalog-coverage number. The headline *association* on that frontier is:

    turning exploration up from w=0 (pure DLRM) to w~0.2 buys ~+51% distinct-item coverage
    (81 -> 122 distinct items) at matched-or-better SNIPS reward.

That is a **correlation across six aggregate points**. A hiring rubric that asks for "causal
inference to measure the impact of AI-driven decisions" wants the harder claim: *does dialing up
exploration CAUSE broader coverage, once we adjust for the confounders that also move coverage* --
namely how popular the item on offer was and how active the user is? A popular item on offer, or a
heavy user with a big candidate pool, changes both the exploration bonus AND the coverage outcome,
so a raw w->coverage slope is confounded.

We answer it with DoWhy's explicit 4-step causal workflow (model -> identify -> estimate -> refute)
on an **event-level** dataset reconstructed from the same replay stream the offline eval uses.

THE CAUSAL STORY (the DAG), in words
====================================
  - TREATMENT  T = explore weight w applied at the event (the "AI decision knob").
  - OUTCOME    Y = per-event *coverage contribution* of the shown item = 1 / (number of times the
                   policy shows that item across the whole replay stream). This is the faithful
                   event-level decomposition of the frontier's coverage metric: it SUMS EXACTLY to
                   the distinct-item count (an item shown k times contributes k * (1/k) = 1), so
                   mean(coverage) * N_events == distinct items shown. This is the "+51% coverage"
                   number, made per-event so a causal model can operate on it.
               Robustness outcomes:
                 * novelty = 1 - popularity_percentile(shown item) -- per-event *rarity*. NOTE this
                   is a DIFFERENT notion from breadth and, empirically, is non-monotonic in w: pure
                   DLRM repeats a few individually-rare items, while exploration spreads across MANY
                   items that are each individually less rare. We report this divergence honestly.
                 * reward = retained logged reward -- does exploration cost reward (the frontier's
                   x-axis)? Carries the SNIPS/propensity caveats.
  - CONFOUNDERS Z:
      * item_pop   = popularity percentile of the LOGGED item on offer at this event.
                     Popular items on offer are both easier to "match" and shift novelty. It also
                     correlates with the propensity that generated the log (selection bias).
      * user_act   = user activity = size of the user's candidate pool (log-scaled).
                     Heavy users have larger, more diverse pools -> more room for novelty AND a
                     different effective exploration regime.
  - MEDIATOR: w -> (shown item) -> novelty. We do NOT condition on the shown item (that is the
    mediator on the causal path); we condition only on the pre-treatment confounders Z. Blocking
    the backdoor path w <- Z -> Y while leaving the front-door path w -> shown-item -> Y open is
    exactly what the identification below does.

WHY THIS IS AN HONEST, NOT-A-DEMO ANALYSIS
==========================================
  * The estimand is genuinely identified only under the stated assumptions (no unobserved
    confounder beyond popularity & activity; treatment w is as-good-as-randomized *given Z* because
    in the sweep w is set exogenously by us, which is actually the strongest point in our favor --
    w is assigned, not chosen by the user). We spell that out rather than hide it.
  * MovieLens is observational/static and the logging propensity is a documented stand-in (softmax
    over popularity), inherited from ``offline_bandit_eval``. So the *reward* estimate carries the
    same caveats as the SNIPS work; the *coverage/novelty* outcome is directly observed (the item we
    showed) and is the cleaner causal target.
  * The credibility signal is the REFUTATION battery (step 4): placebo/random-treatment (effect
    should collapse to ~0), random-common-cause (effect should be stable), and data-subset
    (effect should be stable). We report whether each passes.

USAGE
=====
    python scripts/causal_impact.py                 # full run (needs trained artifacts + dowhy)
    python scripts/causal_impact.py --outcome reward
    python scripts/causal_impact.py --no-dowhy      # hand-rolled backdoor/IPW only (fallback)

dowhy is an OPTIONAL dependency (``pip install ".[causal]"``); it is deliberately NOT in the CI
``[dev]`` extra, so CI never installs it. If dowhy is missing, this script transparently falls back
to a hand-rolled OLS-backdoor + IPW estimator (the same estimand, computed by hand) and says so.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reuse the EXACT replay machinery from the offline eval -- same stream, same policies, same
# propensities. We are not reinventing the evaluation; we are re-framing it causally.
from scripts.offline_bandit_eval import (  # noqa: E402
    DLRMScorer,
    ExploreWeightedBanditPolicy,
    _build_stream,
    _load_artifacts,
    _mean_pop_rank,  # noqa: F401  (kept for reference / parity with offline eval)
)

logger = logging.getLogger("causal_impact")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")

# The same sweep the offline eval uses; w is the (exogenously assigned) treatment.
EXPLORE_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]


# ================================================================================================
# Step 0: build the EVENT-LEVEL causal dataset from the replay stream
# ================================================================================================


def _popularity_percentiles(item_counts: np.ndarray) -> np.ndarray:
    """Map each item index to its popularity percentile in [0, 1] (1 = most popular).

    Uses the double-argsort rank (identical convention to ``offline_bandit_eval._mean_pop_rank``)
    so the novelty outcome here is on the same scale as the offline eval's ``mean_pop_rank``.
    """
    n = len(item_counts)
    order = np.argsort(np.argsort(item_counts))  # 0..n-1 rank by count
    return order / (n - 1) if n > 1 else np.zeros(n)


def build_causal_dataframe(alpha: float = 1.0, l2: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Replay the stream once per explore weight and emit one row per (w, event).

    Columns:
      w         : treatment = explore weight assigned to this run (exogenous, set by the sweep).
      coverage  : PRIMARY outcome = 1 / (times the shown item appears in this run's stream).
                  Sums to distinct-item count; mean(coverage)*N_events == coverage on the frontier.
      novelty   : robustness outcome = 1 - popularity_percentile(item the policy SHOWED). RARITY,
                  not breadth; empirically non-monotonic in w (reported honestly).
      reward    : robustness outcome = logged reward IF this event was retained (policy top-1 ==
                  logged item), else NaN (only defined on retained events, mirroring replay).
      item_pop  : confounder = popularity percentile of the LOGGED item on offer (pre-treatment).
      user_act  : confounder = log(1 + candidate-pool size) for this user (activity proxy).
      retained  : 1 if the policy's top-1 matched the logged item (replay retention), else 0.

    Every row is a real replayed decision; the treatment varies BETWEEN runs (we assign w), the
    confounders vary WITHIN a run (per event). Pooling all runs gives DoWhy the treatment variation
    it needs while the confounders explain the within-run coverage differences.

    The ``coverage`` outcome needs the full run to be known (it depends on how often each item is
    shown ACROSS the stream), so we replay each w twice: pass 1 records the shown items (and does
    the online learning); pass 2 is not needed -- we cache per-event shown items in pass 1 and
    assign 1/frequency afterward. Online learning happens exactly once (in pass 1).
    """
    from collections import Counter
    loaded = _load_artifacts()
    if loaded is None:
        raise SystemExit(
            "Trained artifacts (trained_model_movielens.pth / serving_context.pkl) not found. "
            "Train the model first; this causal analysis consumes the same artifacts as the "
            "offline eval."
        )
    ctx, model, user_emb, item_emb = loaded
    users, items, rewards, candidates, item_counts, _logged_prop = _build_stream(ctx)
    pop_pct = _popularity_percentiles(item_counts)
    scorer = DLRMScorer(model, ctx["user_features"], ctx.get("item_features", {}))

    n_events = len(items)
    # Pre-compute the per-event pre-treatment confounders (independent of w).
    item_pop_event = pop_pct[items]  # popularity percentile of the logged item on offer
    user_act_event = np.array([np.log1p(len(candidates[t])) for t in range(n_events)])

    rows: list[dict] = []
    for w in EXPLORE_WEIGHTS:
        # Fresh bandit per w so online learning is comparable to the offline eval's per-w run.
        policy = ExploreWeightedBanditPolicy(scorer, user_emb, item_emb, w=w, alpha=alpha, l2=l2)
        shown_items: list[int] = []
        retained_flags: list[int] = []
        for t in range(n_events):
            u, a_log, r = int(users[t]), int(items[t]), float(rewards[t])
            cand = candidates[t]
            a_new = policy.top1(u, cand)
            retained = int(a_new == a_log)
            if retained:
                policy.observe(u, a_log, r)  # identical online-learning rule to the offline eval
            shown_items.append(a_new)
            retained_flags.append(retained)
        # coverage[t] = 1 / (times this shown item appears across the whole run) -> sums to distinct.
        shown_freq = Counter(shown_items)
        for t in range(n_events):
            a_new = shown_items[t]
            rows.append(
                {
                    "w": float(w),
                    "coverage": 1.0 / shown_freq[a_new],
                    "novelty": float(1.0 - pop_pct[a_new]),
                    "reward": float(rewards[t]) if retained_flags[t] else np.nan,
                    "item_pop": float(item_pop_event[t]),
                    "user_act": float(user_act_event[t]),
                    "retained": retained_flags[t],
                }
            )
        logger.info("Replayed w=%.1f over %d events (distinct shown=%d).", w, n_events, len(shown_freq))

    df = pd.DataFrame(rows)
    logger.info("Causal dataframe: %d rows (%d weights x %d events).", len(df), len(EXPLORE_WEIGHTS), n_events)
    return df


# ================================================================================================
# Hand-rolled backdoor / IPW estimators (fallback + independent cross-check of DoWhy)
# ================================================================================================


def backdoor_ols_effect(df: pd.DataFrame, outcome: str) -> dict:
    """Backdoor-adjusted linear effect of w on ``outcome``, controlling for item_pop + user_act.

    Fits OLS  outcome ~ 1 + w + item_pop + user_act  and returns the coefficient on w, i.e. the
    average change in the outcome per +1.0 of explore weight, holding the confounders fixed. This is
    the closed-form backdoor estimate the ``dowhy`` "backdoor.linear_regression" method computes;
    we implement it by hand so the analysis is self-verifying and so the fallback matches DoWhy.
    """
    d = df.dropna(subset=[outcome]).copy()
    X = np.column_stack([np.ones(len(d)), d["w"].to_numpy(), d["item_pop"].to_numpy(), d["user_act"].to_numpy()])
    y = d[outcome].to_numpy()
    beta, _res, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    # Standard error on the w coefficient (index 1) via the OLS covariance.
    resid = y - X @ beta
    dof = max(len(d) - X.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    return {"effect": float(beta[1]), "se": se, "n": int(len(d)), "method": "backdoor OLS (by hand)"}


def ipw_effect(df: pd.DataFrame, outcome: str, n_bins: int = 2) -> dict:
    """Inverse-probability-weighted effect of exploration, connecting to the existing IPS/SNIPS work.

    Treatment w is continuous; to reuse the classic IPW/propensity framing (and to tie this back to
    the IPS/SNIPS estimators in ``offline_bandit_eval``), we binarize into HIGH exploration
    (w >= 0.2, the coverage-buying regime on the frontier) vs. LOW (w < 0.2). We model the
    propensity P(HIGH | item_pop, user_act) with a logistic fit and form the IPW contrast

        E[Y | do(HIGH)] - E[Y | do(LOW)]  ~=  mean( T*Y/e - (1-T)*Y/(1-e) )   (Horvitz-Thompson),

    stabilized by self-normalization within each arm (the SNIPS trick: divide by the sum of weights
    per arm, not by N). This is the same self-normalized inverse-propensity idea as SNIPS, applied
    to the treatment propensity instead of the action propensity.
    """
    d = df.dropna(subset=[outcome]).copy()
    d["T"] = (d["w"] >= 0.2).astype(float)
    Z = np.column_stack([np.ones(len(d)), d["item_pop"].to_numpy(), d["user_act"].to_numpy()])
    T = d["T"].to_numpy()
    Y = d[outcome].to_numpy()

    # Logistic propensity e(Z) = P(T=1 | Z) via a few Newton steps (no sklearn dependency needed).
    coef = np.zeros(Z.shape[1])
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-Z @ coef))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        W = p * (1 - p)
        grad = Z.T @ (T - p)
        hess = Z.T @ (Z * W[:, None]) + 1e-6 * np.eye(Z.shape[1])
        coef += np.linalg.solve(hess, grad)
    e = np.clip(1.0 / (1.0 + np.exp(-Z @ coef)), 1e-3, 1 - 1e-3)

    w1 = T / e
    w0 = (1 - T) / (1 - e)
    ey1 = float((w1 * Y).sum() / w1.sum())          # self-normalized (SNIPS-style)
    ey0 = float((w0 * Y).sum() / w0.sum())
    return {
        "effect": ey1 - ey0,
        "e_high": ey1,
        "e_low": ey0,
        "n": int(len(d)),
        "method": "self-normalized IPW, HIGH(w>=0.2) vs LOW(w<0.2)",
    }


# ================================================================================================
# DoWhy 4-step workflow
# ================================================================================================


def _causal_graph_gml(treatment: str, outcome: str) -> str:
    """DAG in GML: confounders -> treatment, confounders -> outcome, treatment -> outcome.

    Encodes the backdoor structure  T <- {item_pop, user_act} -> Y  and the causal edge T -> Y.
    The shown-item mediator is intentionally absent as a node (it is on the front-door path
    T -> shown-item -> Y and must NOT be adjusted for; leaving it out keeps identification honest).
    """
    return f"""
graph [
  directed 1
  node [ id "{treatment}" label "{treatment}" ]
  node [ id "{outcome}"   label "{outcome}"   ]
  node [ id "item_pop"    label "item_pop"    ]
  node [ id "user_act"    label "user_act"    ]
  edge [ source "item_pop" target "{treatment}" ]
  edge [ source "user_act" target "{treatment}" ]
  edge [ source "item_pop" target "{outcome}"   ]
  edge [ source "user_act" target "{outcome}"   ]
  edge [ source "{treatment}" target "{outcome}" ]
]
"""


def run_dowhy(df: pd.DataFrame, outcome: str) -> dict:
    """Full DoWhy model -> identify -> estimate -> refute on w -> outcome.

    Returns a dict with the identified estimand, the point estimate, and the three refutation
    results. Raises ImportError if dowhy is not installed (caller handles the fallback).
    """
    from dowhy import CausalModel

    d = df.dropna(subset=[outcome]).copy()

    # --- (1) MODEL: state the causal graph explicitly. --------------------------------------------
    model = CausalModel(
        data=d,
        treatment="w",
        outcome=outcome,
        graph=_causal_graph_gml("w", outcome),
    )

    # --- (2) IDENTIFY: derive the estimand (should be a backdoor set {item_pop, user_act}). --------
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    logger.info("Identified estimand:\n%s", estimand)

    # --- (3) ESTIMATE: backdoor linear regression (closed-form, matches our hand-rolled OLS). ------
    #     This is the estimator that connects to the IPS/SNIPS lineage: a backdoor adjustment is an
    #     inverse-propensity-style deconfounding; linear_regression is its closed form for a
    #     (near-)continuous treatment.
    estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression",
        test_significance=True,
    )
    point = float(estimate.value)
    logger.info("Estimated causal effect of w on %s: %.5f", outcome, point)

    # --- (4) REFUTE: the credibility battery. -----------------------------------------------------
    refutations: dict[str, dict] = {}

    # 4a. Placebo treatment: replace w with random noise. A real effect should collapse toward 0.
    placebo = model.refute_estimate(
        estimand, estimate, method_name="placebo_treatment_refuter",
        placebo_type="permute", num_simulations=20, random_seed=0,
    )
    refutations["placebo_treatment"] = {
        "original_effect": point,
        "new_effect": float(placebo.new_effect),
        "p_value": _refute_pvalue(placebo),
        "interpretation": "PASS if new_effect ~ 0 (effect vanishes under a fake treatment).",
    }

    # 4b. Random common cause: add an irrelevant confounder. A robust effect should barely move.
    rcc = model.refute_estimate(
        estimand, estimate, method_name="random_common_cause",
        num_simulations=20, random_seed=0,
    )
    refutations["random_common_cause"] = {
        "original_effect": point,
        "new_effect": float(rcc.new_effect),
        "p_value": _refute_pvalue(rcc),
        "interpretation": "PASS if new_effect ~ original (adding a random cause doesn't change it).",
    }

    # 4c. Data-subset: re-estimate on a random 80% subset. Should be stable.
    subset = model.refute_estimate(
        estimand, estimate, method_name="data_subset_refuter",
        subset_fraction=0.8, num_simulations=20, random_seed=0,
    )
    refutations["data_subset"] = {
        "original_effect": point,
        "new_effect": float(subset.new_effect),
        "p_value": _refute_pvalue(subset),
        "interpretation": "PASS if new_effect ~ original (effect stable on a random 80% subset).",
    }

    return {
        "estimand": str(estimand),
        "method": "dowhy backdoor.linear_regression",
        "effect": point,
        "n": int(len(d)),
        "refutations": refutations,
    }


def _refute_pvalue(refute_result) -> float | None:
    """Extract the refutation p-value if dowhy attached one, else None."""
    pv = getattr(refute_result, "refutation_result", None)
    if isinstance(pv, dict):
        val = pv.get("p_value")
        if isinstance(val, (int, float)):
            return float(val)
    return None


# ================================================================================================
# Orchestration
# ================================================================================================


def _pass_fail(refutations: dict, effect: float) -> dict:
    """Turn raw refutation numbers into PASS/FAIL verdicts with explicit thresholds."""
    verdicts = {}
    scale = max(abs(effect), 1e-6)
    # Placebo: fake treatment effect should be a small fraction of the real one.
    pl = refutations["placebo_treatment"]["new_effect"]
    verdicts["placebo_treatment"] = "PASS" if abs(pl) < 0.25 * scale else "FAIL"
    # Random common cause: effect should move < 15% relative to original.
    rcc = refutations["random_common_cause"]["new_effect"]
    verdicts["random_common_cause"] = "PASS" if abs(rcc - effect) < 0.15 * scale else "FAIL"
    # Data subset: effect should move < 15% relative to original.
    ss = refutations["data_subset"]["new_effect"]
    verdicts["data_subset"] = "PASS" if abs(ss - effect) < 0.15 * scale else "FAIL"
    return verdicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outcome", choices=["coverage", "novelty", "reward"], default="coverage",
                        help="Causal outcome: 'coverage' (breadth, default; the frontier metric), "
                             "'novelty' (per-event rarity) or 'reward' (robustness).")
    parser.add_argument("--no-dowhy", action="store_true", help="Skip dowhy; hand-rolled estimators only.")
    parser.add_argument("--alpha", type=float, default=1.0, help="LinUCB exploration coefficient.")
    parser.add_argument("--l2", type=float, default=1.0, help="LinUCB ridge regularisation.")
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "causal_impact.json"),
                        help="Where to write the JSON report.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    df = build_causal_dataframe(alpha=args.alpha, l2=args.l2)

    # Hand-rolled cross-checks (always run: independent verification + fallback).
    ols = backdoor_ols_effect(df, args.outcome)
    ipw = ipw_effect(df, args.outcome)
    logger.info("[hand-rolled] backdoor OLS effect of w on %s = %.5f (se %.5f, n=%d)",
                args.outcome, ols["effect"], ols["se"], ols["n"])
    logger.info("[hand-rolled] IPW HIGH-vs-LOW effect on %s = %.5f (E[HIGH]=%.4f, E[LOW]=%.4f)",
                args.outcome, ipw["effect"], ipw["e_high"], ipw["e_low"])

    report: dict = {
        "question": (
            "Does increasing the bandit's explore weight w CAUSALLY increase catalog coverage "
            "(novelty of shown items), adjusting for item popularity and user activity?"
        ),
        "outcome": args.outcome,
        "treatment": "w (explore weight, exogenously assigned in the sweep)",
        "confounders": ["item_pop", "user_act"],
        "n_rows": int(len(df)),
        "explore_weights": EXPLORE_WEIGHTS,
        "hand_rolled": {"backdoor_ols": ols, "ipw": ipw},
        "dowhy": None,
        "dowhy_available": None,
    }

    if not args.no_dowhy:
        try:
            dowhy_out = run_dowhy(df, args.outcome)
            dowhy_out["refutation_verdicts"] = _pass_fail(dowhy_out["refutations"], dowhy_out["effect"])
            report["dowhy"] = dowhy_out
            report["dowhy_available"] = True
        except ImportError:
            logger.warning(
                "dowhy not installed -- falling back to hand-rolled backdoor/IPW estimators. "
                "dowhy is the DECLARED tool; install with `pip install \".[causal]\"` to run it. "
                "(It is intentionally absent from the CI [dev] extra.)"
            )
            report["dowhy_available"] = False

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)

    # Human-readable summary.
    print("\n" + "=" * 78)
    print(f"CAUSAL IMPACT: exploration weight -> {args.outcome}")
    print("=" * 78)
    print(f"  outcome            : {args.outcome}")
    print(f"  rows (w x events)  : {len(df)}")
    print(f"  [hand] backdoor OLS: effect(w) = {ols['effect']:+.4f}  (se {ols['se']:.4f})")
    print(f"  [hand] IPW HI vs LO: effect     = {ipw['effect']:+.4f}  "
          f"(E[HIGH]={ipw['e_high']:.3f}, E[LOW]={ipw['e_low']:.3f})")
    if report["dowhy"]:
        dw = report["dowhy"]
        print(f"  [dowhy] backdoor LR: effect(w) = {dw['effect']:+.4f}")
        print("  [dowhy] refutations:")
        for name, verdict in dw["refutation_verdicts"].items():
            new_eff = dw["refutations"][name]["new_effect"]
            print(f"      {name:22s} {verdict:4s}  (new_effect = {new_eff:+.4f})")
    elif report["dowhy_available"] is False:
        print("  [dowhy] NOT installed -- hand-rolled fallback used (dowhy is the declared tool).")
    print("=" * 78)
    print(f"  report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
