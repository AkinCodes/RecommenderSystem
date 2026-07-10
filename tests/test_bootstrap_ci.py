"""Unit tests for the paired bootstrap CI + permutation helper.

``paired_bootstrap_ci`` (scripts/retrain_and_compare.py) is what upgrades the
matched model comparison from a bare point estimate to an honest interval +
significance test, so its behaviour on known inputs must be pinned down.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.retrain_and_compare import paired_bootstrap_ci  # noqa: E402


def test_identical_inputs_ci_contains_zero_and_p_is_one():
    """Same vector twice => zero delta, degenerate CI at 0, p == 1."""
    rng = np.random.default_rng(0)
    x = rng.random(200)
    res = paired_bootstrap_ci(x, x)
    assert res["mean_delta"] == 0.0
    assert res["ci_low"] <= 0.0 <= res["ci_high"]
    assert res["p_value"] == 1.0
    assert res["n_users"] == 200


def test_constant_shift_is_significant_ci_excludes_zero():
    """A clean +0.1 shift on every user must be strongly significant."""
    rng = np.random.default_rng(1)
    x = rng.random(200)
    res = paired_bootstrap_ci(x + 0.1, x)
    assert np.isclose(res["mean_delta"], 0.1)
    # Every paired delta is exactly +0.1, so the CI collapses onto +0.1.
    assert res["ci_low"] > 0.0
    assert res["ci_high"] > 0.0
    assert res["p_value"] < 0.001


def test_noise_only_difference_is_a_tie():
    """No true effect (both drawn i.i.d.) => CI should contain 0, p not tiny."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=500)
    b = rng.normal(size=500)
    res = paired_bootstrap_ci(a, b)
    assert res["ci_low"] <= 0.0 <= res["ci_high"]
    assert res["p_value"] > 0.05


def test_determinism_same_seed_same_result():
    """Same seed => bit-identical CI + p-value (reproducible to the digit)."""
    rng = np.random.default_rng(3)
    a = rng.random(150)
    b = rng.random(150)
    r1 = paired_bootstrap_ci(a, b, seed=42)
    r2 = paired_bootstrap_ci(a, b, seed=42)
    assert r1 == r2


def test_antisymmetry_swapping_arguments_flips_sign():
    """delta(a, b) == -delta(b, a); the CI mirrors and the p-value matches."""
    rng = np.random.default_rng(11)
    a = rng.random(120)
    b = rng.random(120)
    ab = paired_bootstrap_ci(a, b, seed=42)
    ba = paired_bootstrap_ci(b, a, seed=42)
    assert np.isclose(ab["mean_delta"], -ba["mean_delta"])
    assert np.isclose(ab["p_value"], ba["p_value"])


def test_mismatched_lengths_raise():
    with np.testing.assert_raises(ValueError):
        paired_bootstrap_ci(np.arange(5), np.arange(4))


def test_empty_input_raises():
    with np.testing.assert_raises(ValueError):
        paired_bootstrap_ci(np.array([]), np.array([]))
