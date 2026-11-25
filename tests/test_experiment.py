import numpy as np
import pandas as pd
import pytest
from scipy import stats

from campaign_uplift import experiment as ex


def test_srm_detects_imbalance():
    assert ex.srm_test([1000, 1000, 1000])["p_value"] > 0.99
    assert ex.srm_test([1000, 1000, 1150])["p_value"] < 0.01


def test_diff_in_means_matches_welch():
    rng = np.random.default_rng(0)
    yt, yc = rng.normal(1.2, 2, 500), rng.normal(1.0, 1, 400)
    e = ex.diff_in_means(yt, yc)
    assert e.ate == pytest.approx(yt.mean() - yc.mean())
    assert e.p_value == pytest.approx(stats.ttest_ind(yt, yc, equal_var=False).pvalue)
    assert e.ci_low < e.ate < e.ci_high
    assert e.rel_lift == pytest.approx(yt.mean() / yc.mean() - 1)


def test_ci_coverage_by_simulation():
    """The 95 % CI of the difference in proportions should cover the truth ~95 % of the time."""
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(400):
        yt, yc = rng.random(2000) < 0.12, rng.random(2000) < 0.10
        e = ex.diff_in_means(yt, yc)
        hits += e.ci_low <= 0.02 <= e.ci_high
    assert 0.92 <= hits / 400 <= 0.98


def test_regression_adjustment_reduces_variance_when_covariate_is_predictive():
    rng = np.random.default_rng(2)
    n = 4000
    x = rng.normal(size=n)
    t = rng.integers(0, 2, n)
    y = 2 * x + 0.5 * t + rng.normal(size=n)
    adj = ex.regression_adjusted_ate(y, t, pd.DataFrame({"x": x}))
    raw = ex.diff_in_means(y[t == 1], y[t == 0])
    assert adj["ci_low"] < 0.5 < adj["ci_high"]
    assert adj["se"] < raw.se * 0.6


def test_holm_known_values():
    adj = ex.holm([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])


def test_bayesian_binomial():
    b = ex.bayesian_binomial(150, 1000, 100, 1000)
    assert b["prob_better"] > 0.99
    assert b["lift_ci_low"] < 0.5 < b["lift_ci_high"]
    same = ex.bayesian_binomial(100, 1000, 100, 1000)
    assert 0.45 < same["prob_better"] < 0.55


def test_power_formulas_are_consistent():
    n = ex.n_per_arm_proportion(0.10, 0.10)
    mde = ex.mde_proportion(0.10, n)
    assert mde == pytest.approx(0.01, rel=0.05)
