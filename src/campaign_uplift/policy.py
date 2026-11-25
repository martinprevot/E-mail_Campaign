"""Off-policy evaluation of targeting rules on the randomized data.

Because arms were assigned at random with known probabilities e(a), the
value of any targeting policy pi can be estimated without bias by inverse
propensity weighting (Horvitz-Thompson):

    V(pi) = mean_i [ y_i * pi(A_i | x_i) / e(A_i) ]

where pi(a | x) is the probability that the policy sends arm a to a
customer x (0/1 for a deterministic rule). We keep the customers who, by
chance, received what the policy would have sent them, and reweight. This
compares "e-mail everyone", "send nothing", "e-mail a random 30 %" and
model-driven targeting on the same footing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _probs(policy, arms) -> np.ndarray:
    """Policy as an (n, n_arms) matrix of sending probabilities."""
    if isinstance(policy, pd.DataFrame):
        return policy[list(arms)].to_numpy(float)
    policy = np.asarray(policy)
    return (policy[:, None] == np.asarray(arms)[None, :]).astype(float)


def _contributions(y, arm, policy, propensity: dict) -> np.ndarray:
    arms = list(propensity)
    y, arm = np.asarray(y, float), np.asarray(arm)
    P = _probs(policy, arms)
    a_idx = pd.Index(arms).get_indexer(arm)
    e = np.array([propensity[a] for a in arms])[a_idx]
    return y * P[np.arange(len(y)), a_idx] / e


def ipw_value(y, arm, policy, propensity: dict) -> float:
    return float(_contributions(y, arm, policy, propensity).mean())


def ipw_diff_ci(y, arm, policy_a, policy_b, propensity, n_boot=1000, seed=0):
    """V(policy_a) - V(policy_b) with a percentile bootstrap 95 % CI."""
    contrib = _contributions(y, arm, policy_a, propensity) - _contributions(y, arm, policy_b, propensity)
    rng = np.random.default_rng(seed)
    n = len(contrib)
    boots = np.array([contrib[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(contrib.mean()), float(lo), float(hi)


def random_policy(n: int, arm: str, share: float, control: str, arms) -> pd.DataFrame:
    """Stochastic policy: send `arm` with probability `share`, else nothing."""
    P = pd.DataFrame(0.0, index=range(n), columns=list(arms))
    P[arm], P[control] = share, 1 - share
    return P


def best_arm_policy(tau: pd.DataFrame, control: str, share: float = 1.0, threshold: float = -np.inf):
    """For each customer pick the arm with the highest predicted uplift, then
    only e-mail the top `share` of customers (ranked by that uplift) and only
    if the predicted uplift exceeds `threshold`."""
    best = tau.idxmax(axis=1).to_numpy()
    gain = tau.max(axis=1).to_numpy()
    n_send = int(round(share * len(tau)))
    order = np.argsort(-gain, kind="mergesort")
    send = np.zeros(len(tau), bool)
    send[order[:n_send]] = True
    send &= gain > threshold
    return np.where(send, best, control)


def targeting_curve(y, arm, tau: pd.DataFrame, control: str, propensity: dict,
                    blanket_arm: str, shares=np.linspace(0, 1, 21)) -> pd.DataFrame:
    """Total incremental outcome vs share of customers e-mailed, for
    (a) the uplift policy and (b) sending `blanket_arm` to a random subset of
    the same size."""
    n = len(tau)
    base = np.full(n, control)
    v0 = ipw_value(y, arm, base, propensity)
    rows = []
    for s in shares:
        pol = best_arm_policy(tau, control, share=s)
        rnd = random_policy(n, blanket_arm, s, control, propensity)
        rows.append({
            "share": s,
            "uplift_policy": (ipw_value(y, arm, pol, propensity) - v0) * n,
            "random_blanket": (ipw_value(y, arm, rnd, propensity) - v0) * n,
        })
    return pd.DataFrame(rows)
