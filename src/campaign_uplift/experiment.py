"""Classical analysis of a randomized experiment (A/B/n test).

- sanity checks: sample ratio mismatch, covariate balance
- average treatment effects with confidence intervals and relative lift
- variance reduction by regression adjustment (Lin 2013, the CUPED idea)
- multiple-testing correction (Holm)
- Bayesian Beta-Binomial reading
- power analysis: minimum detectable effect and required sample size
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


# --------------------------------------------------------------------------- checks
def srm_test(counts: pd.Series, expected_shares=None) -> dict:
    """Chi-square goodness-of-fit test of the arm sizes (sample ratio mismatch)."""
    counts = np.asarray(counts, float)
    shares = np.full(len(counts), 1 / len(counts)) if expected_shares is None else np.asarray(expected_shares)
    chi2, p = stats.chisquare(counts, shares * counts.sum())
    return {"chi2": float(chi2), "p_value": float(p)}


def standardized_mean_differences(X: pd.DataFrame, arm: pd.Series, control: str) -> pd.DataFrame:
    """SMD of every covariate, each arm vs control. |SMD| < 0.1 = balanced."""
    out = {}
    xc = X[arm == control]
    for a in arm.unique():
        if a == control:
            continue
        xt = X[arm == a]
        pooled = np.sqrt((xt.var() + xc.var()) / 2).replace(0, np.nan)
        out[a] = (xt.mean() - xc.mean()) / pooled
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- effects
@dataclass
class Effect:
    metric: str
    arm: str
    control_mean: float
    treated_mean: float
    ate: float
    se: float
    ci_low: float
    ci_high: float
    rel_lift: float
    rel_ci_low: float
    rel_ci_high: float
    p_value: float

    def to_dict(self):
        return asdict(self)


def diff_in_means(y_t, y_c, metric="", arm="", alpha=0.05) -> Effect:
    """Welch difference in means; for 0/1 outcomes it is the unpooled z-test.
    Relative-lift CI by the delta method."""
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    mt, mc = y_t.mean(), y_c.mean()
    vt, vc = y_t.var(ddof=1) / len(y_t), y_c.var(ddof=1) / len(y_c)
    ate, se = mt - mc, np.sqrt(vt + vc)
    z = stats.norm.ppf(1 - alpha / 2)
    p = stats.ttest_ind(y_t, y_c, equal_var=False).pvalue
    rel = mt / mc - 1
    rel_se = np.sqrt(vt / mc**2 + mt**2 * vc / mc**4)
    return Effect(metric, arm, mc, mt, ate, se, ate - z * se, ate + z * se,
                  rel, rel - z * rel_se, rel + z * rel_se, float(p))


def bootstrap_ci(y_t, y_c, n_boot=2000, alpha=0.05, seed=0) -> tuple[float, float]:
    """Percentile bootstrap CI of the difference in means (for heavy tails)."""
    rng = np.random.default_rng(seed)
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        diffs[b] = rng.choice(y_t, len(y_t)).mean() - rng.choice(y_c, len(y_c)).mean()
    return tuple(np.quantile(diffs, [alpha / 2, 1 - alpha / 2]))


def regression_adjusted_ate(y, t, X: pd.DataFrame, alpha=0.05) -> dict:
    """Lin (2013) estimator: OLS of y on T, centred covariates and their
    interactions with T, with HC2 robust errors. Unbiased under
    randomization and never less precise asymptotically than the
    difference in means; this is the regression form of CUPED."""
    Xc = X - X.mean()
    t = np.asarray(t, float)
    design = pd.concat([Xc, Xc.mul(t, axis=0).add_suffix("_x_T")], axis=1)
    design.insert(0, "T", t)
    design = sm.add_constant(design)
    fit = sm.OLS(np.asarray(y, float), design).fit(cov_type="HC2")
    est, se = fit.params["T"], fit.bse["T"]
    z = stats.norm.ppf(1 - alpha / 2)
    return {"ate": float(est), "se": float(se), "ci_low": float(est - z * se),
            "ci_high": float(est + z * se), "p_value": float(fit.pvalues["T"])}


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bayesian_binomial(s_t, n_t, s_c, n_c, draws=200_000, seed=0) -> dict:
    """Beta(1,1) priors. Returns P(treatment > control), the posterior
    relative lift (median, 95 % credible interval) and the expected loss of
    choosing treatment if it is actually worse."""
    rng = np.random.default_rng(seed)
    pt = rng.beta(1 + s_t, 1 + n_t - s_t, draws)
    pc = rng.beta(1 + s_c, 1 + n_c - s_c, draws)
    lift = pt / pc - 1
    return {
        "prob_better": float((pt > pc).mean()),
        "lift_median": float(np.median(lift)),
        "lift_ci_low": float(np.quantile(lift, 0.025)),
        "lift_ci_high": float(np.quantile(lift, 0.975)),
        "expected_loss": float(np.maximum(pc - pt, 0).mean()),
        "samples_lift": lift[:20_000],
    }


# --------------------------------------------------------------------------- power
def mde_proportion(p_base: float, n_per_arm: int, alpha=0.05, power=0.8) -> float:
    """Minimum detectable absolute difference for a two-sided two-sample test."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(2 * p_base * (1 - p_base) / n_per_arm))


def mde_mean(sd: float, n_per_arm: int, alpha=0.05, power=0.8) -> float:
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * sd * np.sqrt(2 / n_per_arm))


def n_per_arm_proportion(p_base: float, rel_lift: float, alpha=0.05, power=0.8) -> int:
    """Sample size per arm to detect a relative lift on a conversion rate."""
    p1 = p_base * (1 + rel_lift)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    var = p_base * (1 - p_base) + p1 * (1 - p1)
    return int(np.ceil(z**2 * var / (p1 - p_base) ** 2))
