import numpy as np
import pandas as pd
import pytest
from scipy import stats

from campaign_uplift.learners import LEARNERS
from campaign_uplift.metrics import qini_auc_score, qini_curve, uplift_at_k, uplift_by_bin
from campaign_uplift.policy import best_arm_policy, ipw_value, random_policy


def _synthetic(n=20000, seed=0):
    """Randomized experiment with a known, heterogeneous effect:
    tau(x) = 0.15 if x1 > 0 else 0."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    t = rng.integers(0, 2, n)
    tau = np.where(X["x1"] > 0, 0.15, 0.0)
    p = 0.2 + 0.05 * np.tanh(X["x2"]) + tau * t
    y = (rng.random(n) < p).astype(int)
    return X, t, y, tau


def test_qini_curve_hand_computed():
    y = np.array([1, 0, 1, 0])
    t = np.array([1, 0, 0, 1])
    score = np.array([0.9, 0.8, 0.7, 0.6])
    x, q = qini_curve(y, t, score)
    # k=1: Yt=1, Yc=0 -> 1 ; k=2: Yt=1, Yc=0, Nt=1, Nc=1 -> 1
    # k=3: Yt=1, Yc=1, Nt=1, Nc=2 -> 0.5 ; k=4: Yt=1, Yc=1, Nt=2, Nc=2 -> 0
    np.testing.assert_allclose(q, [0, 1, 1, 0.5, 0])
    np.testing.assert_array_equal(x, [0, 1, 2, 3, 4])


def test_qini_auc_oracle_beats_random():
    X, t, y, tau = _synthetic()
    rng = np.random.default_rng(1)
    # The normalisation by the "perfect" curve (which sorts on observed
    # outcomes) makes even the true tau score far below 1 on noisy data.
    assert qini_auc_score(y, t, tau + rng.normal(0, 1e-6, len(tau))) > 0.06
    assert abs(qini_auc_score(y, t, rng.random(len(y)))) < 0.02


def test_uplift_at_k_and_bins_on_oracle():
    X, t, y, tau = _synthetic()
    assert uplift_at_k(y, t, tau, 0.4) == pytest.approx(0.15, abs=0.03)
    bins = uplift_by_bin(y, t, tau + np.random.default_rng(0).normal(0, 1e-6, len(tau)), bins=2)
    assert bins.loc[1, "observed"] > bins.loc[2, "observed"] + 0.08


@pytest.mark.parametrize("learner_cls", LEARNERS, ids=lambda c: c.name)
def test_learners_recover_heterogeneity(learner_cls):
    X, t, y, tau = _synthetic()
    tr, te = slice(0, 15000), slice(15000, None)
    pred = learner_cls().fit(X.iloc[tr], t[tr], y[tr]).predict(X.iloc[te])
    assert stats.spearmanr(pred, tau[te]).correlation > 0.5
    assert pred[tau[te] > 0].mean() - pred[tau[te] == 0].mean() == pytest.approx(0.15, abs=0.06)


def test_ipw_value_is_unbiased_for_a_known_policy():
    rng = np.random.default_rng(3)
    n = 60000
    arms = ["control", "a", "b"]
    arm = rng.choice(arms, n)
    means = {"control": 0.10, "a": 0.18, "b": 0.14}
    y = (rng.random(n) < pd.Series(arm).map(means)).astype(int)
    prop = {a: 1 / 3 for a in arms}
    assert ipw_value(y, arm, np.full(n, "a"), prop) == pytest.approx(0.18, abs=0.01)
    half = random_policy(n, "b", 0.5, "control", arms)
    assert ipw_value(y, arm, half, prop) == pytest.approx(0.12, abs=0.01)


def test_best_arm_policy():
    tau = pd.DataFrame({"a": [0.10, 0.02, -0.01], "b": [0.05, 0.08, -0.02]})
    np.testing.assert_array_equal(best_arm_policy(tau, "control"), ["a", "b", "a"])
    np.testing.assert_array_equal(best_arm_policy(tau, "control", threshold=0), ["a", "b", "control"])
    np.testing.assert_array_equal(best_arm_policy(tau, "control", share=1 / 3), ["a", "control", "control"])


def test_matches_scikit_uplift_if_installed():
    sk = pytest.importorskip("sklift.metrics")
    X, t, y, tau = _synthetic(n=5000)
    score = np.round(X["x1"].to_numpy() + np.random.default_rng(0).normal(size=len(y)), 1)  # with ties
    assert qini_auc_score(y, t, score) == pytest.approx(sk.qini_auc_score(y, score, t))
    assert uplift_at_k(y, t, score, 0.3) == pytest.approx(sk.uplift_at_k(y, score, t, strategy="overall", k=0.3))
