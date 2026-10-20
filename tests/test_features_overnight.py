from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_ai.features.overnight import (
    CROSS_SECTIONAL_FEATURES,
    OVERNIGHT_CURATED_FEATURES,
    add_cross_sectional_overnight_features,
    add_execution_guard_features,
    add_overnight_features,
    check_jumping_catch_risk,
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


def test_add_cross_sectional_overnight_features():
    # Build pooled dataframe with 2 dates and 3 tickers
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    rows = []
    for d in dates:
        # Ticker A: high return, high CLV, high volume surge
        rows.append({
            "date": d,
            "ticker": "A",
            "sector": "Tech",
            "return_1d": 0.03,
            "return_5d": 0.05,
            "close_location_value": 0.9,
            "volume_surge_5d": 2.5,
            "rsi_14": 75.0,
            "next_open_return": 0.02,
        })
        # Ticker B: medium return, medium CLV
        rows.append({
            "date": d,
            "ticker": "B",
            "sector": "Tech",
            "return_1d": 0.01,
            "return_5d": 0.02,
            "close_location_value": 0.1,
            "volume_surge_5d": 1.1,
            "rsi_14": 50.0,
            "next_open_return": 0.005,
        })
        # Ticker C: low return, low CLV
        rows.append({
            "date": d,
            "ticker": "C",
            "sector": "Auto",
            "return_1d": -0.02,
            "return_5d": -0.03,
            "close_location_value": -0.8,
            "volume_surge_5d": 0.7,
            "rsi_14": 30.0,
            "next_open_return": -0.01,
        })

    pooled_df = pd.DataFrame(rows)
    result = add_cross_sectional_overnight_features(pooled_df)

    # 1. Verify required columns
    for col in CROSS_SECTIONAL_FEATURES:
        assert col in result.columns, f"Missing feature: {col}"
    assert "target_cs_outperform" in result.columns

    # 2. Check percentile ranking order on 2024-01-01
    day1 = result[result["date"] == pd.Timestamp("2024-01-01")].set_index("ticker")
    assert day1.loc["A", "cs_rank_intraday_return"] > day1.loc["B", "cs_rank_intraday_return"]
    assert day1.loc["B", "cs_rank_intraday_return"] > day1.loc["C", "cs_rank_intraday_return"]
    assert day1.loc["A", "cs_rank_close_location"] > day1.loc["C", "cs_rank_close_location"]

    # 3. Percentiles are strictly within [0.0, 1.0]
    for col in ["cs_rank_intraday_return", "cs_rank_close_location", "cs_rank_volume_surge", "cs_rank_rsi_14"]:
        assert (result[col] >= 0.0).all()
        assert (result[col] <= 1.0).all()

    # 4. Check Outperform Target: Ticker A (next_open_return=0.02) vs median (0.005) -> 1, C -> 0
    assert day1.loc["A", "target_cs_outperform"] == 1
    assert day1.loc["C", "target_cs_outperform"] == 0

    # 5. Check Sector Relative Return
    assert day1.loc["A", "sector_relative_return_1d"] > 0


def test_check_jumping_catch_risk_triggers():
    # 1. Exhaustion Return Surge: return_1d > +4.5%
    row_surge = pd.Series({
        "return_1d": 0.052,
        "vol_adjusted_momentum_1d": 1.5,
        "rsi_14": 55.0,
        "pullback_from_high": 0.1,
        "intraday_return": 0.03,
    })
    is_risk, reason = check_jumping_catch_risk(row_surge)
    assert is_risk is True
    assert "過熱暴騰警戒" in reason

    # 2. Momentum Burst: vol_adjusted_momentum_1d > 2.2
    row_momentum = pd.Series({
        "return_1d": 0.025,
        "vol_adjusted_momentum_1d": 2.45,
        "rsi_14": 60.0,
        "pullback_from_high": 0.1,
        "intraday_return": 0.015,
    })
    is_risk, reason = check_jumping_catch_risk(row_momentum)
    assert is_risk is True
    assert "過熱モメンタム警戒" in reason

    # 3. RSI Overbought Extension: rsi_14 between 65 and 75 with return_1d > 0.010
    row_rsi = pd.Series({
        "return_1d": 0.018,
        "vol_adjusted_momentum_1d": 1.2,
        "rsi_14": 69.5,
        "pullback_from_high": 0.1,
        "intraday_return": 0.01,
    })
    is_risk, reason = check_jumping_catch_risk(row_rsi)
    assert is_risk is True
    assert "RSI過熱追随警戒" in reason

    # 4. Upper Shadow Stall: pullback_from_high >= 0.60 with intraday_return <= 0 and rsi_14 >= 55
    row_shadow = pd.Series({
        "return_1d": 0.005,
        "vol_adjusted_momentum_1d": 0.8,
        "rsi_14": 60.0,
        "pullback_from_high": 0.72,
        "intraday_return": -0.010,
    })
    is_risk, reason = check_jumping_catch_risk(row_shadow)
    assert is_risk is True
    assert "上ヒゲ失速警戒" in reason

    # 5. Healthy Normal Candidate (Dip bounce / calm advance): should PASS
    row_healthy = pd.Series({
        "return_1d": 0.012,
        "vol_adjusted_momentum_1d": 1.1,
        "rsi_14": 52.0,
        "pullback_from_high": 0.15,
        "intraday_return": 0.008,
    })
    is_risk, reason = check_jumping_catch_risk(row_healthy)
    assert is_risk is False
    assert reason == ""


def test_add_execution_guard_features():
    df = pd.DataFrame({
        "open": [100.0, 102.0],
        "high": [105.0, 110.0],
        "low": [99.0, 101.0],
        "close": [104.0, 102.5],
        "return_1d": [0.01, 0.048],  # Second row triggers surge
        "vol_adjusted_momentum_1d": [1.0, 1.2],
        "rsi_14": [50.0, 58.0],
    })
    res = add_execution_guard_features(df)

    assert "pullback_from_high" in res.columns
    assert "jumping_catch_risk" in res.columns
    assert "jumping_catch_reason" in res.columns
    assert res["jumping_catch_risk"].iloc[0] == False
    assert res["jumping_catch_risk"].iloc[1] == True


