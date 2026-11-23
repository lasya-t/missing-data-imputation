import numpy as np
import pytest

from benchmark import run
from missingness import inject

X = np.random.default_rng(0).normal(size=(20_000, 3))


@pytest.mark.parametrize("mech", ["MCAR", "MAR", "MNAR"])
def test_rate_is_respected(mech):
    m = inject(X, 0.25, mech, cols=[0], seed=1)
    assert abs(np.isnan(m[:, 0]).mean() - 0.25) < 0.02
    assert not np.isnan(m[:, 1:]).any()


def test_mnar_hides_large_values():
    m = inject(X, 0.3, "MNAR", cols=[0])
    assert X[np.isnan(m[:, 0]), 0].mean() > X[~np.isnan(m[:, 0]), 0].mean() + 0.5


def test_mcar_is_unbiased():
    m = inject(X, 0.3, "MCAR", cols=[0])
    assert abs(X[np.isnan(m[:, 0]), 0].mean() - X[:, 0].mean()) < 0.05


def test_iterative_beats_mean_on_correlated_data():
    rows = {r["method"]: r for r in run(0.2, "MAR", seed=0)}
    assert rows["iterative (MICE-style)"]["RMSE (sd)"] < 0.6 * rows["mean"]["RMSE (sd)"]
