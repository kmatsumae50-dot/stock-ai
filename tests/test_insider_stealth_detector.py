"""tests/test_insider_stealth_detector.py

Unit tests for Quant Series #12:
1. Stealth Informed Accumulation & Volatility Compression (Easley PIN & Augustin/Subrahmanyam)
2. Policy, Defense & Strategic Infrastructure Cluster Institutional Spillover
3. Pre-Bad-News Leak & Informed Dumping Detection
4. Strict Zero Future-Data Leakage Verification
"""

import numpy as np
import pandas as pd
import pytest

from stock_ai.features.insider_stealth_detector import (
    STRATEGIC_POLICY_CLUSTERS,
    compute_abnormal_volume_z,
    compute_insider_dumping_risk,
    compute_policy_defense_insider_spillover,
    compute_stealth_absorption_efficiency,
    compute_stealth_insider_features,
    compute_true_range_atr,
    add_insider_stealth_features,
)


def test_abnormal_volume_z():
    """Verify abnormal volume Z-score calculation and bounds."""
    # Constant volume should produce 0.0 Z-score
    v_const = pd.Series([1000.0] * 30)
    z_const = compute_abnormal_volume_z(v_const)
    assert len(z_const) == 30
    assert abs(z_const.iloc[-1]) < 0.1

    # Surging volume should produce positive Z-score > 1.0
    v_surge = pd.Series([1000.0] * 25 + [5000.0] * 5)
    z_surge = compute_abnormal_volume_z(v_surge)
    assert z_surge.iloc[-1] > 1.0
    assert (z_surge >= -3.0).all() and (z_surge <= 4.0).all()


def test_stealth_absorption_efficiency_dynamics():
    """Verify that volume surge with compressed range and positive OFI yields high stealth efficiency."""
    # Scenario A: Stealth Accumulation (High volume, compressed range, positive OFI)
    vol_z_stealth = pd.Series([2.0])
    high_stealth = pd.Series([101.0])
    low_stealth = pd.Series([100.0])  # range = 1.0
    atr_stealth = pd.Series([3.0])    # rel_range = 1/3 = 0.33
    ofi_stealth = pd.Series([0.60])

    eff_stealth = compute_stealth_absorption_efficiency(
        vol_z_stealth, high_stealth, low_stealth, atr_stealth, ofi_stealth
    ).iloc[0]

    # Scenario B: High range breakout (Wide range, ordinary volume, neutral OFI)
    vol_z_break = pd.Series([0.0])
    high_break = pd.Series([110.0])
    low_break = pd.Series([100.0])   # range = 10.0
    atr_break = pd.Series([3.0])     # rel_range = 10/3 = 3.33
    ofi_break = pd.Series([0.10])

    eff_break = compute_stealth_absorption_efficiency(
        vol_z_break, high_break, low_break, atr_break, ofi_break
    ).iloc[0]

    # Stealth accumulation efficiency should be dramatically higher
    assert eff_stealth > eff_break * 3.0
    assert eff_stealth > 1.5


def test_compute_stealth_insider_features_bounds():
    """Verify output columns, shape, and [0, 1] bounds for insider_stealth_accumulation_10d."""
    np.random.seed(42)
    n = 60
    base_price = 1000.0
    returns = np.random.normal(0.0005, 0.015, n)
    prices = base_price * np.cumprod(1.0 + returns)
    volumes = np.random.uniform(50000, 150000, n)
    highs = prices * (1.0 + np.random.uniform(0.002, 0.01, n))
    lows = prices * (1.0 - np.random.uniform(0.002, 0.01, n))

    df = pd.DataFrame({
        "close": prices,
        "high": highs,
        "low": lows,
        "open": prices * 0.999,
        "volume": volumes,
    })

    res = compute_stealth_insider_features(df)
    assert "abnormal_volume_z_20d" in res.columns
    assert "stealth_absorption_efficiency" in res.columns
    assert "insider_stealth_accumulation_10d" in res.columns

    accum = res["insider_stealth_accumulation_10d"]
    assert (accum >= 0.0).all() and (accum <= 1.0).all()
    assert not res.isna().any().any()


def test_policy_defense_insider_spillover():
    """Verify cluster recognition and cross-sectional wave score calculation."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    tickers = ["7011.T", "7013.T", "8035.T", "7203.T", "9984.T"]
    rows = []
    for d in dates:
        for t in tickers:
            # Set 7011 and 7013 (Defense cluster) to have high volume Z and high positive OFI
            is_defense = t in ["7011.T", "7013.T"]
            vz = 2.5 if is_defense else 0.0
            ofi = 0.70 if is_defense else 0.0
            rows.append({
                "date": d,
                "ticker": t,
                "abnormal_volume_z_20d": vz,
                "vpin_signed_ofi_5d": ofi,
                "return_1d": 0.01 if is_defense else 0.0,
            })
    cross_df = pd.DataFrame(rows)

    spill_df = compute_policy_defense_insider_spillover(cross_df)
    assert len(spill_df) == len(cross_df)
    assert "policy_sector_insider_wave_score" in spill_df.columns
    assert "policy_cluster_tag" in spill_df.columns

    # Verify cluster tags
    tag_7011 = spill_df.loc[cross_df["ticker"] == "7011.T", "policy_cluster_tag"].iloc[0]
    tag_7203 = spill_df.loc[cross_df["ticker"] == "7203.T", "policy_cluster_tag"].iloc[0]
    assert tag_7011 == "防衛・重工"
    assert tag_7203 == "一般"

    # Defense tickers should have significantly higher wave score than general tickers
    score_7011 = spill_df.loc[cross_df["ticker"] == "7011.T", "policy_sector_insider_wave_score"].mean()
    score_7203 = spill_df.loc[cross_df["ticker"] == "7203.T", "policy_sector_insider_wave_score"].mean()
    assert score_7011 > score_7203
    assert score_7011 > 0.70


def test_insider_dumping_risk_alert():
    """Verify that severe negative OFI, volume surge, and weak close trigger insider dumping alert."""
    n = 30
    # Create healthy baseline
    close = pd.Series([1000.0] * n)
    high = pd.Series([1005.0] * n)
    low = pd.Series([995.0] * n)
    volume = pd.Series([10000.0] * n)
    signed_ofi = pd.Series([0.10] * n)

    # In the last 3 days, inject severe dumping pattern
    # Volume surges 4x, strong negative OFI, weak close at the low
    for idx in range(n - 3, n):
        close.iloc[idx] = 950.0 - (idx - (n - 3)) * 10
        high.iloc[idx] = 1000.0
        low.iloc[idx] = 940.0
        volume.iloc[idx] = 50000.0
        signed_ofi.iloc[idx] = -0.80

    df = pd.DataFrame({
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "vpin_signed_ofi_5d": signed_ofi,
    })

    res = compute_insider_dumping_risk(df)
    assert (res["insider_dumping_alert_score"] >= 0.0).all() and (res["insider_dumping_alert_score"] <= 1.0).all()

    # Early baseline should not trigger
    assert not res["insider_dumping_alert_flag"].iloc[10]
    # Severe dumping days should trigger alert
    assert res["insider_dumping_alert_flag"].iloc[-1]
    assert res["insider_dumping_alert_score"].iloc[-1] >= 0.70


def test_zero_future_leakage_insider_features():
    """Verify zero lookahead: changing bar t+1 has ZERO impact on bar t features."""
    np.random.seed(123)
    n = 40
    base_price = 2000.0
    returns = np.random.normal(0.001, 0.02, n)
    prices = base_price * np.cumprod(1.0 + returns)
    volumes = np.random.uniform(100000, 300000, n)
    highs = prices * 1.01
    lows = prices * 0.99

    df1 = pd.DataFrame({
        "close": prices,
        "high": highs,
        "low": lows,
        "open": prices,
        "volume": volumes,
    })

    # df2 is identical up to bar 25, but wildly manipulated from bar 26 onward
    df2 = df1.copy()
    df2.loc[26:, "close"] = df2.loc[26:, "close"] * 3.5
    df2.loc[26:, "volume"] = df2.loc[26:, "volume"] * 10.0
    df2.loc[26:, "high"] = df2.loc[26:, "high"] * 4.0

    feat1 = add_insider_stealth_features(df1)
    feat2 = add_insider_stealth_features(df2)

    # Bars 0 to 25 must be numerically identical bit-by-bit
    eval_cols = [
        "abnormal_volume_z_20d",
        "stealth_absorption_efficiency",
        "insider_stealth_accumulation_10d",
        "insider_dumping_alert_score",
    ]
    for col in eval_cols:
        diff = np.abs(feat1.loc[:25, col].to_numpy() - feat2.loc[:25, col].to_numpy())
        max_diff = np.nanmax(diff)
        assert max_diff < 1e-10, f"Future leak detected in {col} at or before bar 25! Max diff: {max_diff}"
