from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_ai.features.overnight import (
    OVERNIGHT_CURATED_FEATURES,
    add_overnight_features,
    purged_walk_forward_splits,
    select_overnight_curated_features,
)


def _sample_ohlcv(n: int = 50) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.linspace(100.0, 150.0, n)
    high = close + 2.0
    low = close - 1.5
    open_p = close - 0.5
    vol = np.random.RandomState(42).randint(100_000, 500_000, size=n)

    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * n,
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
            "return_1d": pd.Series(close).pct_change().fillna(0.0),
            "volatility_20": [0.015] * n,
            "rsi_14": [45.0] * n,
            "high_low_range": (high - low) / low,
            "close_position": (close - low) / (high - low),
            "close_vs_ma_5": [0.01] * n,
            "close_vs_ma_25": [0.02] * n,
            "close_vs_ma_75": [0.03] * n,
            "macd": [0.5] * n,
            "macd_signal": [0.4] * n,
            "macd_hist": [0.1] * n,
            "bb_upper": close + 3.0,
            "bb_lower": close - 3.0,
            "bb_middle": close,
            "bb_percent_b": [0.5] * n,
            "sector_mean_return": [0.005] * n,
            "sector_strength": [0.01] * n,
            "sector_volatility": [0.012] * n,
            "sector_trend_score": [1.0] * n,
            "n225_return_1d": [0.004] * n,
            "n225_trend_direction": [1.0] * n,
            "topix_return_1d": [0.003] * n,
            "topix_trend_direction": [1.0] * n,
            "usdjpy_return_1d": [0.001] * n,
            "usdjpy_close_vs_ma_25": [0.01] * n,
            "usdjpy_trend_direction": [1.0] * n,
            "tesla_return_1d": [-0.01] * n,
            "tesla_trend_direction": [-1.0] * n,
            "byd_return_1d": [0.005] * n,
        }
    )


def test_add_overnight_features_creates_required_columns():
    df = _sample_ohlcv(50)
    result = add_overnight_features(df)

    assert "close_location_value" in result.columns
    assert "volume_surge_5d" in result.columns
    assert "volume_surge_20d" in result.columns
    assert "vol_adjusted_momentum_1d" in result.columns
    assert "vol_adjusted_momentum_5d" in result.columns
    assert "rsi_oversold_flag" in result.columns
    assert "rsi_overbought_flag" in result.columns
    assert "sector_relative_return" in result.columns
    assert "sector_relative_strength" in result.columns
    assert "bb_squeeze" in result.columns
    assert "intraday_range_ratio" in result.columns


def test_close_location_value_bounds_and_behavior():
    # Test case 1: Close equals high (CLV must be +1.0)
    high_close_df = pd.DataFrame({
        "open": [100.0], "high": [110.0], "low": [90.0], "close": [110.0], "volume": [1000]
    })
    res_high = add_overnight_features(high_close_df)
    assert pytest.approx(res_high["close_location_value"].iloc[0], rel=1e-3) == 1.0

    # Test case 2: Close equals low (CLV must be -1.0)
    low_close_df = pd.DataFrame({
        "open": [100.0], "high": [110.0], "low": [90.0], "close": [90.0], "volume": [1000]
    })
    res_low = add_overnight_features(low_close_df)
    assert pytest.approx(res_low["close_location_value"].iloc[0], rel=1e-3) == -1.0

    # Test case 3: Close at exact midpoint (CLV must be 0.0)
    mid_close_df = pd.DataFrame({
        "open": [100.0], "high": [110.0], "low": [90.0], "close": [100.0], "volume": [1000]
    })
    res_mid = add_overnight_features(mid_close_df)
    assert pytest.approx(res_mid["close_location_value"].iloc[0], abs=1e-3) == 0.0


def test_select_overnight_curated_features_prunes_noise():
    df = _sample_ohlcv(50)
    df = add_overnight_features(df)
    # Add some noise features
    df["random_noise_feature_1"] = 999.0
    df["dead_lag_feature_2"] = 123.0

    selected = select_overnight_curated_features(df)
    assert "close_location_value" in selected
    assert "volume_surge_5d" in selected
    assert "sector_relative_return" in selected
    assert "random_noise_feature_1" not in selected
    assert "dead_lag_feature_2" not in selected
    assert len(selected) <= len(OVERNIGHT_CURATED_FEATURES)


def test_purged_walk_forward_splits_prevents_leakage():
    df = pd.DataFrame({"x": range(100)})
    splits = purged_walk_forward_splits(df, n_splits=4, embargo_pct=0.02)

    assert len(splits) > 0
    for train_idx, test_idx in splits:
        # Train strictly comes before test
        assert train_idx.max() < test_idx.min()
        # Embargo ensures gap between train_max and test_min
        assert test_idx.min() - train_idx.max() >= 2
