"""Uplift (CATE) meta-learners for a randomized binary treatment.

Each learner estimates tau(x) = P(Y=1 | X=x, T=1) - P(Y=1 | X=x, T=0).
The experiment is randomized 50/50 within each (arm, control) pair, so the
propensity is a known constant and no propensity model is needed.

- S-learner: one model, treatment as a feature.
- T-learner: one outcome model per group.
- X-learner (Kunzel et al. 2019): imputes individual effects from the
  T-learner, then regresses them on X; better when the effect is simpler
  than the outcome.
- Class transformation (Jaskowski & Jaroszewicz 2012): Z = YT + (1-Y)(1-T);
  with p = 0.5, tau(x) = 2 P(Z=1 | x) - 1. One classifier, directly on uplift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

SEED = 42
# Deliberately shallow and heavily regularised trees: the treatment effect is
# a small signal (a few points of visit rate) buried in the outcome noise, and
# deeper models fit that noise. Settings chosen among 3 candidates by
# cross-validated Qini (see README, "Limites").
BASE = dict(n_estimators=150, learning_rate=0.05, num_leaves=3, min_child_samples=1000,
            subsample=0.8, subsample_freq=1, colsample_bytree=1.0, random_state=SEED, verbose=-1)


def _clf():
    return LGBMClassifier(**BASE)


def _reg():
    return LGBMRegressor(**BASE)


class SLearner:
    name = "S-learner"

    def fit(self, X, t, y):
        self.m = _clf().fit(X.assign(_t=t), y)
        return self

    def predict(self, X):
        p1 = self.m.predict_proba(X.assign(_t=1))[:, 1]
        p0 = self.m.predict_proba(X.assign(_t=0))[:, 1]
        return p1 - p0


class TLearner:
    name = "T-learner"

    def fit(self, X, t, y):
        t = np.asarray(t)
        self.m1 = _clf().fit(X[t == 1], y[t == 1])
        self.m0 = _clf().fit(X[t == 0], y[t == 0])
        return self

    def predict(self, X):
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]


class XLearner:
    name = "X-learner"

    def __init__(self, propensity: float = 0.5):
        self.g = propensity

    def fit(self, X, t, y):
        t, y = np.asarray(t), np.asarray(y, float)
        X1, X0, y1, y0 = X[t == 1], X[t == 0], y[t == 1], y[t == 0]
        m1, m0 = _clf().fit(X1, y1), _clf().fit(X0, y0)
        d1 = y1 - m0.predict_proba(X1)[:, 1]      # imputed effect on treated
        d0 = m1.predict_proba(X0)[:, 1] - y0      # imputed effect on controls
        self.tau1, self.tau0 = _reg().fit(X1, d1), _reg().fit(X0, d0)
        return self

    def predict(self, X):
        return self.g * self.tau0.predict(X) + (1 - self.g) * self.tau1.predict(X)


class ClassTransformation:
    name = "Class transformation"

    def fit(self, X, t, y):
        z = (np.asarray(y) == np.asarray(t)).astype(int)
        self.m = _clf().fit(X, z)
        return self

    def predict(self, X):
        return 2 * self.m.predict_proba(X)[:, 1] - 1


LEARNERS = [SLearner, TLearner, XLearner, ClassTransformation]


def cross_fit(X: pd.DataFrame, t, y, learner_cls, folds) -> np.ndarray:
    """Out-of-fold uplift predictions: every customer is scored by a model
    that never saw them, so curves and policy values are honest."""
    t, y = np.asarray(t), np.asarray(y)
    oof = np.zeros(len(X))
    for train, test in folds:
        model = learner_cls().fit(X.iloc[train], t[train], y[train])
        oof[test] = model.predict(X.iloc[test])
    return oof
