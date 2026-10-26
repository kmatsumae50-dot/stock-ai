"""tests/test_flow_toxicity_tail_risk.py

Unit tests for Quant Series #10:
1. Bulk Volume Classification (BVC) & VPIN Order Flow Toxicity (bounds, conservation of volume, OFI)
2. Kritzman Absorption Ratio (AR) and Systemic Fragility (perfect coupling vs orthogonal dispersion)
3. Asymmetric Short & Exhaustion Sniper Score
4. Strict Zero-Leakage (causality check)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_ai.features.flow_toxicity_tail_risk import (
    compute_absorption_ratio,
    compute_asymmetric_short_exhaustion,
    compute_bvc_buy_sell_volume,
    compute_bvc_vpin_features,
    compute_market_absorption_ratio,
    compute_vpin,
)


def test_bvc_volume_decomposition_and_ofi():
    """Verify BVC volume conservation: V^B + V^S == V and OFI in [-1, 1]."""
    np.random.seed(42)
    n = 100
    prices = 1000.0 + np.cumsum(np.random.randn(n) * 10.0)
    volumes = np.random.randint(1000, 50000, size=n).astype(float)

    close_s = pd.Series(prices)
    vol_s = pd.Series(volumes)

    buy_vol, sell_vol, ofi = compute_bvc_buy_sell_volume(close_s, vol_s, window=20)

    # 1. Total volume conservation
    total_reconstructed = buy_vol + sell_vol
    np.testing.assert_allclose(total_reconstructed.values, vol_s.values, rtol=1e-5)

    # 2. OFI bounds [-1, 1]
    assert (ofi >= -1.0 - 1e-6).all()
    assert (ofi <= 1.0 + 1e-6).all()

    # 3. Non-negativity of buy and sell volumes
    assert (buy_vol >= 0.0).all()
    assert (sell_vol >= 0.0).all()

    # 4. Check upward price spike has buy_vol > sell_vol and ofi > 0
    up_idx = np.where(close_s.diff() > 15.0)[0]
    if len(up_idx) > 0:
        assert (buy_vol.iloc[up_idx] > sell_vol.iloc[up_idx]).all()
        assert (ofi.iloc[up_idx] > 0.0).all()


def test_vpin_bounds_and_toxicity_properties():
    """Verify VPIN toxicity bounds [0, 1] and monotonic response to one-directional flow."""
    n = 50
    # Case A: Perfectly balanced two-way flow -> VPIN near 0
    vol_a = pd.Series([1000.0] * n)
    buy_a = pd.Series([500.0] * n)
    sell_a = pd.Series([500.0] * n)
    vpin_a = compute_vpin(buy_a, sell_a, vol_a, window=10)
    assert (vpin_a.iloc[10:] == 0.0).all()

    # Case B: Completely one-directional informed buying flow -> VPIN == 1.0
    vol_b = pd.Series([1000.0] * n)
    buy_b = pd.Series([1000.0] * n)
    sell_b = pd.Series([0.0] * n)
    vpin_b = compute_vpin(buy_b, sell_b, vol_b, window=10)
    assert (vpin_b.iloc[10:] == 1.0).all()


def test_bvc_vpin_features_scores():
    """Verify full feature generation including Smart Money Accumulation and Informed Distribution."""
    np.random.seed(123)
    n = 60
    # Stock drifting gently up with high buy volume (quiet accumulation)
    returns = np.random.normal(0.001, 0.005, size=n)
    prices = 2000.0 * np.exp(np.cumsum(returns))
    volumes = np.random.uniform(50000, 100000, size=n)

    df = pd.DataFrame({
        "open": prices * 0.998,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": volumes,
    })

    feats = compute_bvc_vpin_features(df)
    assert "vpin_toxicity_10d" in feats.columns
    assert "vpin_signed_ofi_5d" in feats.columns
    assert "vpin_smart_accumulation_score" in feats.columns
    assert "vpin_informed_distribution_score" in feats.columns

    # Verify score bounds
    assert (feats["vpin_smart_accumulation_score"] >= 0.0).all()
    assert (feats["vpin_smart_accumulation_score"] <= 1.0).all()
    assert (feats["vpin_informed_distribution_score"] >= 0.0).all()
    assert (feats["vpin_informed_distribution_score"] <= 1.0).all()


def test_absorption_ratio_synthetic_extreme_cases():
    """Verify Kritzman Absorption Ratio (AR) behavior on synthetic extremes:
    1. Complete coupling (all stocks move in lockstep): AR == 1.0
    2. Complete dispersion (independent uncorrelated assets): AR ~= K / N
    """
    n_days = 100
    n_assets = 20
    k_fraction = 0.10  # K = 2 out of 20

    # 1. Complete coupling: all 20 assets have the exact same returns (rho = 1.0)
    market_factor = np.random.randn(n_days)
    coupled_returns = np.tile(market_factor[:, np.newaxis], (1, n_assets))
    ar_coupled = compute_absorption_ratio(coupled_returns, fraction_eigenvalues=k_fraction, min_k=2)
    # The first eigenvalue absorbs 100% of variance
    assert ar_coupled > 0.999

    # 2. Complete dispersion: independent uncorrelated Gaussian returns
    np.random.seed(99)
    uncoupled_returns = np.random.randn(n_days, n_assets)
    ar_uncoupled = compute_absorption_ratio(uncoupled_returns, fraction_eigenvalues=k_fraction, min_k=2)
    # With K=2 and N=20, theoretical equal variance fraction is 2/20 = 0.10
    # In finite sample it's slightly higher due to sample noise, but << 0.35
    assert ar_uncoupled < 0.35
    assert ar_coupled > ar_uncoupled * 2.5


def test_market_absorption_ratio_time_series_and_fragility():
    """Verify time-series absorption ratio and systemic fragility detection during a simulated shock."""
    np.random.seed(77)
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    n_assets = 15

    # Days 0-70: Low correlation normal market
    normal_rets = np.random.randn(70, n_assets) * 0.01

    # Days 71-99: Sudden macro shock, all stocks crash in lockstep (correlation -> 0.95)
    shock = np.random.randn(30) * 0.03
    shock_rets = np.tile(shock[:, np.newaxis], (1, n_assets)) + np.random.randn(30, n_assets) * 0.002

    all_rets = np.vstack([normal_rets, shock_rets])
    df_ret = pd.DataFrame(all_rets, index=dates, columns=[f"asset_{i}" for i in range(n_assets)])

    res_ar = compute_market_absorption_ratio(df_ret, short_window=15, long_window=50, fraction_eigenvalues=0.15)

    assert "absorption_ratio_short" in res_ar.columns
    assert "delta_absorption_ratio" in res_ar.columns
    assert "systemic_fragility_score" in res_ar.columns

    # During the shock period, short AR should surge compared to pre-shock
    pre_shock_ar = res_ar["absorption_ratio_short"].iloc[60]
    post_shock_ar = res_ar["absorption_ratio_short"].iloc[95]
    assert post_shock_ar > pre_shock_ar

    # Systemic fragility score should be elevated (> 0.70) during the crash phase
    assert res_ar["systemic_fragility_score"].iloc[90:].max() > 0.70


def test_asymmetric_short_exhaustion_score():
    """Verify that upper shadow rejection, high RSI, and VPIN distribution drive the short exhaustion score."""
    # Scenario A: Exhaustion Top (Huge upper shadow, red body, overbought RSI, high VPIN distribution)
    df_exhaustion = pd.DataFrame([{
        "open": 2050.0,
        "high": 2150.0,   # 100 pt upper shadow
        "low": 1990.0,
        "close": 2000.0,  # 50 pt red body (close < open)
        "beta_to_topix": 1.4,
        "rsi_14": 75.0,
        "chan_kalman_zscore": 2.2,
        "vpin_informed_distribution_score": 0.85,
        "leadlag_leader_momentum_z": -1.2,
    }])

    score_ex, shadow_ex = compute_asymmetric_short_exhaustion(df_exhaustion)
    assert score_ex.iloc[0] > 0.65
    assert shadow_ex.iloc[0] > 0.40

    # Scenario B: Clean Bullish Momentum (No upper shadow, green candle, low RSI, negative short score)
    df_bull = pd.DataFrame([{
        "open": 1950.0,
        "high": 2000.0,   # Closes at highs
        "low": 1940.0,
        "close": 2000.0,  # 50 pt green body
        "beta_to_topix": 0.9,
        "rsi_14": 52.0,
        "chan_kalman_zscore": 0.1,
        "vpin_informed_distribution_score": 0.05,
        "leadlag_leader_momentum_z": 1.5,
    }])

    score_bull, shadow_bull = compute_asymmetric_short_exhaustion(df_bull)
    assert score_bull.iloc[0] < 0.25
    assert shadow_bull.iloc[0] == 0.0


def test_zero_future_leakage_causality():
    """Strictly verify that modifying future data at t+k does not alter features at t."""
    np.random.seed(42)
    n = 60
    prices = 1000.0 + np.cumsum(np.random.randn(n) * 5.0)
    vols = np.random.randint(1000, 20000, size=n).astype(float)

    df1 = pd.DataFrame({"close": prices.copy(), "volume": vols.copy(), "high": prices * 1.01, "low": prices * 0.99})
    feats1 = compute_bvc_vpin_features(df1)

    # Now alter the future (days 50 to 59)
    prices2 = prices.copy()
    vols2 = vols.copy()
    prices2[50:] += 500.0  # Massive future spike
    vols2[50:] *= 10.0

    df2 = pd.DataFrame({"close": prices2, "volume": vols2, "high": prices2 * 1.01, "low": prices2 * 0.99})
    feats2 = compute_bvc_vpin_features(df2)

    # Features at indices <= 49 must be BIT-FOR-BIT IDENTICAL
    for col in feats1.columns:
        np.testing.assert_array_almost_equal(
            feats1[col].iloc[:50].values,
            feats2[col].iloc[:50].values,
            decimal=7,
            err_msg=f"Future leakage detected in {col}!"
        )
