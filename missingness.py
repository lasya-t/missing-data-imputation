"""Inject missing values into a complete dataset under different mechanisms.

MCAR - missing completely at random: every cell has the same chance of being missing.
MAR  - missing at random: whether a value is missing depends on *another* observed column.
MNAR - missing not at random: large values are more likely to be missing (e.g. people
       with high incomes skipping the income question). The hardest case.
"""
from __future__ import annotations

import numpy as np


def inject(X: np.ndarray, rate: float, mechanism: str = "MCAR", cols=None, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = X.astype(float).copy()
    n, d = X.shape
    cols = list(range(d)) if cols is None else list(cols)
    for j in cols:
        if mechanism == "MCAR":
            p = np.full(n, rate)
        elif mechanism == "MAR":
            driver = X[:, (j + 1) % d]  # depends on a neighbouring, still observed column
            p = _scaled_logistic(driver, rate)
        elif mechanism == "MNAR":
            p = _scaled_logistic(X[:, j], rate)  # depends on the value itself
        else:
            raise ValueError(mechanism)
        X[rng.random(n) < p, j] = np.nan
    return X


def _scaled_logistic(v, rate):
    """Probabilities that rise with v and average to `rate`."""
    z = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)
    p = 1 / (1 + np.exp(-2 * z))
    return np.clip(p * rate / p.mean(), 0, 0.95)
