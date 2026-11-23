"""Compare imputation methods on two things that matter:

1. How close the filled-in values are to the truth (RMSE on the hidden cells, in standard deviations)
2. How well a model trained on the imputed data predicts (test ROC-AUC)

Data: breast-cancer features (30 correlated measurements), 40% of columns hit by missingness.
Imputers are fit on the training split only, then applied to the test split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from missingness import inject

IMPUTERS = {
    "mean": lambda: SimpleImputer(strategy="mean"),
    "median": lambda: SimpleImputer(strategy="median"),
    "mean + indicator": lambda: SimpleImputer(strategy="mean", add_indicator=True),
    "KNN (k=5)": lambda: KNNImputer(n_neighbors=5),
    "iterative (MICE-style)": lambda: IterativeImputer(max_iter=15, random_state=0),
}


def run(rate: float, mechanism: str, seed: int = 0) -> list[dict]:
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)                    # work in standard-deviation units
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    cols = np.random.default_rng(seed).choice(X.shape[1], 12, replace=False)
    Mtr = inject(Xtr, rate, mechanism, cols, seed)
    Mte = inject(Xte, rate, mechanism, cols, seed + 1)
    hidden = np.isnan(Mte)

    rows = []
    for name, make in IMPUTERS.items():
        imp = make().fit(Mtr)
        filled_te = imp.transform(Mte)[:, : X.shape[1]]   # drop indicator columns for the RMSE
        rmse = float(np.sqrt(np.mean((filled_te[hidden] - Xte[hidden]) ** 2)))
        model = make_pipeline(make(), LogisticRegression(max_iter=3000)).fit(Mtr, ytr)
        auc = roc_auc_score(yte, model.predict_proba(Mte)[:, 1])
        rows.append({"rate": rate, "mechanism": mechanism, "method": name, "RMSE (sd)": rmse, "AUC": auc})

    # trees handle NaN natively: no imputation at all
    gb = HistGradientBoostingClassifier(random_state=0).fit(Mtr, ytr)
    rows.append({"rate": rate, "mechanism": mechanism, "method": "no imputation (HistGB native NaN)",
                 "RMSE (sd)": np.nan, "AUC": roc_auc_score(yte, gb.predict_proba(Mte)[:, 1])})
    full = LogisticRegression(max_iter=3000).fit(Xtr, ytr)
    rows.append({"rate": rate, "mechanism": mechanism, "method": "(complete data, no missing)",
                 "RMSE (sd)": 0.0, "AUC": roc_auc_score(yte, full.predict_proba(Xte)[:, 1])})
    return rows


if __name__ == "__main__":
    results = []
    for mech in ("MCAR", "MAR", "MNAR"):
        for rate in (0.1, 0.3):
            # average over 5 splits so differences aren't one lucky split
            per_seed = [pd.DataFrame(run(rate, mech, seed)) for seed in range(5)]
            avg = pd.concat(per_seed).groupby(["mechanism", "rate", "method"], sort=False).mean().reset_index()
            results.append(avg)
    df = pd.concat(results)
    pd.set_option("display.width", 140)
    for (mech, rate), g in df.groupby(["mechanism", "rate"], sort=False):
        print(f"\n=== {mech}, {rate:.0%} missing in 12 of 30 columns (mean of 5 splits)")
        print(g[["method", "RMSE (sd)", "AUC"]].round(4).to_string(index=False))
