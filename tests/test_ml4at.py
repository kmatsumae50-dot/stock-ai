"""tests/test_ml4at.py

Unit tests for Stefan Jansen "Machine Learning for Algorithmic Trading" (ML4AT) Core Implementations:
1. WorldQuant 101 Formulaic Alphas (Alpha#101, Alpha#12, Alpha#06, Alpha#44, Alpha#54)
2. Information Discreteness (Trend Continuity vs Retail Hype)
3. Cross-Sectional Uniform Rank Normalization
4. Heterogeneous Stacked Ensemble (LightGBM + L2 Regularized Linear Model)
"""

import numpy as np
import pandas as pd
import pytest

from stock_ai.features.ml4at import (
    add_cross_sectional_rank_features,
    add_ml4at_features,
    compute_alpha_06,
    compute_alpha_12,
    compute_alpha_44,
    compute_alpha_54,
    compute_alpha_101,
    compute_information_discreteness,
)
from stock_ai.models.overnight import HeterogeneousEnsembleClassifier


def test_alpha_101_computation():
    close = pd.Series([102.0, 98.0, 100.0])
    open_p = pd.Series([100.0, 100.0, 100.0])
    high = pd.Series([103.0, 101.0, 100.0])
    low = pd.Series([99.0, 97.0, 100.0])

    a101 = compute_alpha_101(close, open_p, high, low)
    # Row 0: (102 - 100) / (103 - 99) = 2 / 4 = 0.5
    assert a101.iloc[0] == pytest.approx(0.5, abs=1e-3)
    # Row 1: (98 - 100) / (101 - 97) = -2 / 4 = -0.5
    assert a101.iloc[1] == pytest.approx(-0.5, abs=1e-3)
    # Row 2: Flat bar (zero division protected)
    assert not np.isnan(a101.iloc[2])


def test_alpha_12_volume_price_divergence():
    # Price falls while volume surges -> strong positive divergence signal
    close = pd.Series([100.0, 95.0])
    volume = pd.Series([1000.0, 3000.0])
    a12 = compute_alpha_12(close, volume)

    # d_vol > 0 (sign = +1), d_close < 0 (-1 * d_close > 0) -> a12 > 0
    assert a12.iloc[1] > 0.0


def test_alpha_06_and_44():
    np.random.seed(42)
    s1 = pd.Series(np.cumsum(np.random.randn(25)) + 100.0)
    vol = pd.Series(np.random.randint(1000, 5000, size=25))

    a06 = compute_alpha_06(s1, vol, window=10)
    a44 = compute_alpha_44(s1, vol, window=10)

    assert len(a06) == 25
    assert not a06.isna().any()
    assert not a44.isna().any()
    assert (a06 >= -1.0).all() and (a06 <= 1.0).all()


def test_alpha_54_dip_buying():
    # Strong lower shadow: open=100, high=101, low=95, close=100.5
    close = pd.Series([100.5])
    open_p = pd.Series([100.0])
    high = pd.Series([101.0])
    low = pd.Series([95.0])

    a54 = compute_alpha_54(close, open_p, high, low)
    assert not np.isnan(a54.iloc[0])
    assert np.isfinite(a54.iloc[0])
    # Mathematical result for this input is approx -0.9075
    assert a54.iloc[0] == pytest.approx(-0.9075, abs=1e-2)


def test_information_discreteness():
    # Continuous uptrend: steady daily gains
    c_steady = pd.Series([100.0 + i * 1.0 for i in range(12)])
    id_steady = compute_information_discreteness(c_steady, window=10)
    # All daily returns positive -> pct_neg = 0, pct_pos = 1 -> sign(+1) * (0 - 1) = -1.0
    assert id_steady.iloc[-1] < 0.0

    # Discrete jump: flat for 9 days, then one big burst, then flat
    c_jump = pd.Series([100.0] * 5 + [120.0] + [119.0, 118.0, 117.0, 116.0, 115.0])
    id_jump = compute_information_discreteness(c_jump, window=10)
    # More negative days after the jump -> ID should be higher (indicating retail hype/fatigue)
    assert id_jump.iloc[-1] > id_steady.iloc[-1]


def test_ml4at_leak_free():
    """Verify strict zero lookahead leakage: appending future rows does not change past alpha values."""
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2024-01-01", periods=n)
    df1 = pd.DataFrame(
        {
            "date": dates,
            "open": np.cumsum(np.random.randn(n)) + 100.0,
            "high": np.cumsum(np.random.randn(n)) + 105.0,
            "low": np.cumsum(np.random.randn(n)) + 95.0,
            "close": np.cumsum(np.random.randn(n)) + 100.0,
            "volume": np.random.randint(1000, 5000, size=n),
        }
    )
    res1 = add_ml4at_features(df1)

    # Append 20 future days
    df2 = pd.concat([df1, df1.tail(20).copy()], ignore_index=True)
    res2 = add_ml4at_features(df2)

    np.testing.assert_allclose(
        res1["ml4at_alpha_101"].values,
        res2["ml4at_alpha_101"].values[:n],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        res1["ml4at_info_discreteness_10d"].values[15:],
        res2["ml4at_info_discreteness_10d"].values[15:n],
        rtol=1e-5,
        atol=1e-5,
    )


def test_cross_sectional_rank_features():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "ticker": ["A", "B", "C", "A", "B"],
            "ml4at_alpha_101": [0.1, 0.5, 0.9, -0.2, 0.8],
        }
    )
    res = add_cross_sectional_rank_features(df, feature_cols=["ml4at_alpha_101"])
    assert "ml4at_alpha_101_cs_rank" in res.columns
    # On 2024-01-01: A has rank lowest, C has rank highest
    ranks_day1 = res[res["date"] == "2024-01-01"]["ml4at_alpha_101_cs_rank"].values
    assert ranks_day1[0] < ranks_day1[1] < ranks_day1[2]


def test_heterogeneous_ensemble_classifier():
    class DummyLGB:
        feature_importances_ = [10, 20]
        def predict_proba(self, X):
            return np.array([[0.2, 0.8], [0.4, 0.6]])

    class DummyLinear:
        def predict_proba(self, X):
            return np.array([[0.4, 0.6], [0.6, 0.4]])

    ensemble = HeterogeneousEnsembleClassifier(
        DummyLGB(), DummyLinear(), w_lgb=0.75, w_lin=0.25
    )
    probs = ensemble.predict_proba(pd.DataFrame([[1, 2], [3, 4]]))

    # Row 0: 0.75 * [0.2, 0.8] + 0.25 * [0.4, 0.6] = [0.15 + 0.10, 0.60 + 0.15] = [0.25, 0.75]
    assert probs[0, 1] == pytest.approx(0.75, abs=1e-4)
    preds = ensemble.predict(pd.DataFrame([[1, 2], [3, 4]]))
    assert preds[0] == 1
