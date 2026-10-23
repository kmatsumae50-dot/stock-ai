"""tests/test_afml.py

Unit tests for AFML (Advances in Financial Machine Learning) Core Algorithms:
1. Fixed-Width Window Fractional Differentiation (FFD)
2. Dynamic Volatility Triple-Barrier Labeling
3. Sample Weighting with Informativeness & Time-Decay
4. MDA Feature Pruning
5. Meta-Labeling & Bet Sizing
"""

import numpy as np
import pandas as pd
import pytest

from stock_ai.features.afml import (
    add_afml_features,
    calculate_bet_size,
    compute_afml_sample_weights,
    compute_dynamic_triple_barrier_events,
    frac_diff_ffd,
    generate_meta_labels,
    get_weights_ffd,
    prune_features_with_mda,
)


def test_ffd_weights():
    d = 0.4
    weights = get_weights_ffd(d=d, thres=1e-3, max_l=50)
    assert len(weights) > 0
    # w_0 is the last element (due to reversal for convolution)
    assert weights[-1] == pytest.approx(1.0, abs=1e-5)
    # Reconstructed forwards weights should decay
    w_forward = weights[::-1]
    assert abs(w_forward[1]) < abs(w_forward[0])
    # Recurrence check: w_1 = -w_0 * d = -0.4
    assert w_forward[1] == pytest.approx(-0.4, abs=1e-5)


def test_ffd_leak_free():
    """Verify strict zero lookahead leakage: appending new future data does NOT alter past FFD values."""
    np.random.seed(42)
    s1 = pd.Series(np.cumsum(np.random.randn(80)) + 100.0)
    ffd_base = frac_diff_ffd(s1, d=0.4, thres=1e-3, max_l=30)

    # Append 20 future observations
    s2 = pd.concat([s1, pd.Series(np.cumsum(np.random.randn(20)) + 100.0)], ignore_index=True)
    ffd_extended = frac_diff_ffd(s2, d=0.4, thres=1e-3, max_l=30)

    # Values for the first 80 bars must match exactly (zero lookahead leakage)
    np.testing.assert_allclose(
        ffd_base.values[:75],
        ffd_extended.values[:75],
        rtol=1e-5,
        atol=1e-5,
    )


def test_add_afml_features():
    dates = pd.date_range("2024-01-01", periods=60)
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "date": dates,
            "close": np.cumsum(np.random.randn(60)) + 100.0,
            "volume": np.random.randint(1000, 5000, size=60),
        }
    )
    result = add_afml_features(df, d=0.4)
    assert "afml_fracdiff_close_d04" in result.columns
    assert "afml_fracdiff_vol_d04" in result.columns
    assert "afml_fracdiff_zscore_20" in result.columns
    assert not result["afml_fracdiff_close_d04"].isna().any()
    assert not result["afml_fracdiff_zscore_20"].isna().any()


def test_dynamic_triple_barrier():
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [103.0, 100.5, 99.5],
            "low": [99.5, 96.0, 99.0],
            "close": [102.0, 97.0, 99.2],
            "prior_vol_20": [0.02, 0.02, 0.02],
        }
    )
    # Upper barrier: 100 * (1 + 1.0 * 0.02) = 102.0
    # Lower barrier: 100 * (1 - 0.8 * 0.02) = 98.4
    res = compute_dynamic_triple_barrier_events(df, k_pt=1.0, k_sl=0.8, cost_bps=20.0)

    assert "tb_upper_barrier" in res.columns
    assert "tb_lower_barrier" in res.columns
    assert "tb_clean_win" in res.columns

    # Bar 0: High=103 >= 102.0 (PT hit), Low=99.5 > 98.4 (SL not hit), Net Ret = +2.0% >= 20bps -> Clean Win = 1
    assert res["tb_hit_upper"].iloc[0]
    assert not res["tb_hit_lower"].iloc[0]
    assert res["tb_clean_win"].iloc[0] == 1

    # Bar 1: Low=96.0 <= 98.4 (SL hit) -> Clean Win = 0
    assert res["tb_hit_lower"].iloc[1]
    assert res["tb_clean_win"].iloc[1] == 0


def test_afml_sample_weights():
    returns = np.array([0.0005, 0.0300, -0.0250, 0.0010])
    dates = pd.date_range("2024-01-01", periods=4)
    w = compute_afml_sample_weights(returns, dates, time_decay=0.50, cost_bps=20.0)

    assert len(w) == 4
    # Weights should be positive and normalize to mean == 1.0
    assert np.all(w > 0)
    assert np.mean(w) == pytest.approx(1.0, abs=1e-4)

    # Large returns (e.g. 0.0300) should have larger weight than tiny returns (0.0005)
    assert w[1] > w[0]


def test_generate_meta_labels_and_bet_size():
    primary_preds = np.array([0.65, 0.60, 0.40, 0.70])
    actual_returns = np.array([0.015, -0.010, 0.020, 0.005])
    lower_hit = np.array([False, True, False, False])

    meta_labels = generate_meta_labels(
        primary_preds,
        actual_returns,
        lower_barrier_hit=lower_hit,
        cost_bps=20.0,
        primary_buy_threshold=0.50,
    )

    # Sample 0: Primary Buy (0.65) + profitable net (0.015) + no SL -> 1
    assert meta_labels[0] == 1
    # Sample 1: Primary Buy (0.60) + SL hit -> 0
    assert meta_labels[1] == 0
    # Sample 2: Primary did NOT buy (0.40) -> 0
    assert meta_labels[2] == 0
    # Sample 3: Primary Buy (0.70) + net profit (+0.005 >= 20bps) -> 1
    assert meta_labels[3] == 1

    # Bet sizing tests
    assert calculate_bet_size(0.50) == 0.0
    assert calculate_bet_size(0.60) == pytest.approx(0.20, abs=1e-4)
    assert calculate_bet_size(0.75) == pytest.approx(0.50, abs=1e-4)
    assert calculate_bet_size(0.95) == pytest.approx(0.90, abs=1e-4)
    assert calculate_bet_size(1.00) == pytest.approx(1.00, abs=1e-4)


def test_mda_feature_pruning():
    import lightgbm as lgb

    np.random.seed(42)
    n = 120
    # Feature 0 is highly informative
    x0 = np.random.randn(n)
    y = (x0 > 0).astype(int)
    # Features 1-19 are pure noise
    X_dict = {"signal": x0}
    for i in range(1, 20):
        X_dict[f"noise_{i}"] = np.random.randn(n)
    X = pd.DataFrame(X_dict)

    model = lgb.LGBMClassifier(n_estimators=30, random_state=42, verbose=-1)
    model.fit(X, y)

    selected, mda_scores = prune_features_with_mda(
        model,
        X,
        y,
        list(X.columns),
        min_importance=0.0,
        n_repeats=2,
    )
    assert "signal" in selected
    assert len(selected) <= len(X.columns)
