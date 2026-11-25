"""Uplift evaluation: Qini / uplift curves, Qini AUC, uplift by bin.

The individual treatment effect is never observed, so uplift models are
judged on groups: customers are ranked by predicted uplift and, within the
top k, the observed response of treated vs control customers is compared.
Definitions follow Radcliffe (2007) and match scikit-uplift (checked in the
tests).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cumulative(y, t, score):
    y, t, score = np.asarray(y, float), np.asarray(t, int), np.asarray(score, float)
    order = np.argsort(score, kind="mergesort")[::-1]
    y, t, score = y[order], t[order], score[order]
    # Evaluate only at the last index of each block of tied scores.
    idx = np.r_[np.where(np.diff(score))[0], len(score) - 1]
    n_t = np.cumsum(t)[idx]
    n_c = (idx + 1) - n_t
    y_t = np.cumsum(y * t)[idx]
    y_c = np.cumsum(y * (1 - t))[idx]
    return idx + 1, n_t, n_c, y_t, y_c


def qini_curve(y, t, score):
    """x = number of customers targeted, q = incremental responses:
    q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k)."""
    k, n_t, n_c, y_t, y_c = _cumulative(y, t, score)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = y_t - np.where(n_c > 0, y_c * n_t / n_c, 0.0)
    return np.r_[0, k], np.r_[0, q]


def uplift_curve(y, t, score):
    """x = customers targeted, u = (response rate T - response rate C) * k."""
    k, n_t, n_c, y_t, y_c = _cumulative(y, t, score)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(n_t > 0, y_t / n_t, 0) - np.where(n_c > 0, y_c / n_c, 0)
    return np.r_[0, k], np.r_[0, rate * k]


def _perfect_qini(y, t):
    y, t = np.asarray(y, float), np.asarray(t, int)
    return qini_curve(y, t, y * t - y * (1 - t))


def qini_auc_score(y, t, score) -> float:
    """Area between the model's Qini curve and the random diagonal,
    normalised by the same area for the perfect ranking (1 = perfect,
    0 = random, < 0 = worse than random)."""
    x, q = qini_curve(y, t, score)
    xp, qp = _perfect_qini(y, t)
    rand = np.trapezoid([0, q[-1]], [0, x[-1]])
    return float((np.trapezoid(q, x) - rand) / (np.trapezoid(qp, xp) - rand))


def uplift_at_k(y, t, score, k=0.3) -> float:
    """Observed uplift (rate T - rate C) in the top k fraction."""
    y, t, score = np.asarray(y, float), np.asarray(t, int), np.asarray(score, float)
    top = np.argsort(score, kind="mergesort")[::-1][: int(np.ceil(k * len(score)))]
    yt, tt = y[top], t[top]
    return float(yt[tt == 1].mean() - yt[tt == 0].mean())


def uplift_by_bin(y, t, score, bins=10) -> pd.DataFrame:
    """Observed vs predicted uplift per bin of predicted uplift (bin 1 = highest)."""
    df = pd.DataFrame({"y": y, "t": t, "score": score})
    df["bin"] = bins - pd.qcut(df["score"].rank(method="first"), bins, labels=False)
    rows = []
    for b, g in df.groupby("bin"):
        yt, yc = g.loc[g.t == 1, "y"], g.loc[g.t == 0, "y"]
        se = np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
        rows.append({"bin": int(b), "n": len(g), "predicted": g["score"].mean(),
                     "observed": yt.mean() - yc.mean(), "se": se})
    return pd.DataFrame(rows).set_index("bin")
