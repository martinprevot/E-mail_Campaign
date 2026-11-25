"""Cross-fitted uplift predictions for every customer and every e-mail arm."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .data import CONTROL

SEED = 42


def cross_fit_all(X: pd.DataFrame, arm: pd.Series, y: pd.Series, learner_cls,
                  treatment_arms, n_splits: int = 5) -> pd.DataFrame:
    """Split the whole population into folds stratified by arm x outcome.
    For each fold and each e-mail arm, the learner is trained on the
    (arm vs control) customers of the other folds and scores *every*
    customer of the held-out fold. Result: an out-of-fold uplift estimate
    for each customer under each e-mail, usable to pick the best one."""
    arm, y = arm.to_numpy(), y.to_numpy()
    strata = pd.Series(arm).astype(str) + "_" + pd.Series(y).astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    tau = pd.DataFrame(0.0, index=X.index, columns=list(treatment_arms))
    for train, test in skf.split(X, strata):
        for a in treatment_arms:
            pair = train[np.isin(arm[train], [a, CONTROL])]
            t = (arm[pair] == a).astype(int)
            model = learner_cls().fit(X.iloc[pair], t, y[pair])
            tau.iloc[test, tau.columns.get_loc(a)] = model.predict(X.iloc[test])
    return tau
