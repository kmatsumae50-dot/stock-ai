"""tests/test_chan_quant.py

Unit tests for Ernest P. Chan quantitative features:
1. Hurst Exponent & Variance Ratio regime classification
2. Ornstein-Uhlenbeck (OU) Half-Life and Z-score
3. Kalman Filter causal smoothing, spread, and innovation z-score
4. Continuous Fractional Half-Kelly capital allocation
5. End-to-end integration via add_chan_quant_features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_ai.features.chan_quant import (
    ALL_CHAN_FEATURES,
    CHAN_BASE_FEATURE_NAMES,
    add_chan_quant_features,
    calculate_half_kelly_fraction,
    compute_hurst_exponent_rolling,
    compute_kalman_filter_features,
    compute_ou_halflife,
    compute_regime_momentum_score,
    compute_variance_ratio,
)


class TestChanQuantHurstAndVarianceRatio:
    """Tests for Chapter 2: Hurst Exponent & Variance Ratio."""

    def test_hurst_trending_vs_mean_reverting(self) -> None:
        # 1. Strongly trending series (geometric trend with low noise) -> H should be > 0.5
        t = np.arange(100, dtype=float)
        trend_prices = pd.Series(100.0 * np.exp(0.015 * t + 0.001 * np.sin(t)))
        hurst_trend = compute_hurst_exponent_rolling(trend_prices, window=30)
        # Check later values where rolling window has full history
        valid_trend = hurst_trend.iloc[40:].mean()
        assert valid_trend > 0.5, f"Expected H > 0.5 for strong trend, got {valid_trend:.3f}"

        # 2. Strongly oscillating / mean-reverting series -> H should be < 0.5
        oscillating = 100.0 + 5.0 * np.sin(np.arange(100) * (np.pi / 2.0))
        osc_prices = pd.Series(oscillating)
        hurst_osc = compute_hurst_exponent_rolling(osc_prices, window=30)
        valid_osc = hurst_osc.iloc[40:].mean()
        assert valid_osc < 0.5, f"Expected H < 0.5 for oscillating series, got {valid_osc:.3f}"

    def test_variance_ratio_trending_vs_reverting(self) -> None:
        t = np.arange(100, dtype=float)
        trend_prices = pd.Series(100.0 * np.exp(0.02 * t))
        vr_trend = compute_variance_ratio(trend_prices, window=30, lag=5)
        # For pure deterministic trend, 5-day variance is 25x 1-day variance, VR = 25 / 5 = 5.0 (capped)
        assert vr_trend.iloc[40:].mean() > 1.2

        # Fast alternating prices: 100, 102, 100, 102 (negative autocorrelation)
        alt = np.tile([100.0, 102.0], 50)
        vr_alt = compute_variance_ratio(pd.Series(alt), window=30, lag=5)
        assert vr_alt.iloc[40:].mean() < 0.8

    def test_regime_momentum_score(self) -> None:
        # High Hurst + Positive 5d return -> Positive score (Trend buy)
        h_high = pd.Series([0.75, 0.80])
        ret_pos = pd.Series([0.03, 0.05])
        score_trend = compute_regime_momentum_score(h_high, ret_pos)
        assert (score_trend > 0.0).all()

        # Low Hurst + Negative 5d return -> Positive score (Oversold Dip-Buy)
        h_low = pd.Series([0.30, 0.25])
        ret_neg = pd.Series([-0.04, -0.02])
        score_dip = compute_regime_momentum_score(h_low, ret_neg)
        assert (score_dip > 0.0).all()

        # Low Hurst + Positive 5d return -> Negative score (Overbought Fade)
        score_fade = compute_regime_momentum_score(h_low, ret_pos)
        assert (score_fade < 0.0).all()


class TestChanQuantOUProcess:
    """Tests for Chapter 3: Ornstein-Uhlenbeck process and Half-life."""

    def test_ou_halflife_synthetic_mean_reverting(self) -> None:
        # Generate synthetic OU series: dx = -theta * x * dt + sigma * dW
        np.random.seed(42)
        n = 200
        theta = 0.2  # Expected theoretical half-life ~ ln(2)/0.2 = 3.47 days
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = x[i - 1] - theta * x[i - 1] + np.random.normal(0, 0.5)

        prices = pd.Series(np.exp(x) * 100.0)
        hl, zscore = compute_ou_halflife(prices, window=30)

        # Average half-life should be short and bounded (around 2 to 10 days)
        valid_hl = hl.iloc[50:].mean()
        assert 1.0 <= valid_hl <= 20.0, f"Expected short half-life, got {valid_hl}"
        assert (zscore >= -3.0).all() and (zscore <= 3.0).all()


class TestChanQuantKalmanFilter:
    """Tests for Chapter 5: 1D Dynamic State-Space Kalman Filter."""

    def test_kalman_filter_causality_and_spread(self) -> None:
        np.random.seed(42)
        n = 100
        true_trend = np.linspace(100.0, 150.0, n)
        noise = np.random.normal(0, 2.0, n)
        noisy_prices = pd.Series(true_trend + noise)

        spread, zscore, slope = compute_kalman_filter_features(noisy_prices, q=1e-4, r=1e-2)

        assert len(spread) == n
        assert len(zscore) == n
        assert len(slope) == n

        # Test zero-leakage / causality:
        # Modifying future values (index 80 onwards) must NOT change earlier estimates (index <= 50)
        modified_prices = noisy_prices.copy()
        modified_prices.iloc[80:] = modified_prices.iloc[80:] * 2.0  # huge future shock

        spread_mod, zscore_mod, slope_mod = compute_kalman_filter_features(modified_prices, q=1e-4, r=1e-2)
        # Prior 50 steps must be identical to within machine precision
        np.testing.assert_allclose(spread.iloc[:50], spread_mod.iloc[:50], atol=1e-12)
        np.testing.assert_allclose(zscore.iloc[:50], zscore_mod.iloc[:50], atol=1e-12)
        np.testing.assert_allclose(slope.iloc[:50], slope_mod.iloc[:50], atol=1e-12)


class TestChanQuantHalfKelly:
    """Tests for Chapter 8: Fractional Half-Kelly Criterion."""

    def test_calculate_half_kelly_fraction(self) -> None:
        # High expected return (100bps = 0.01), low daily vol (1% = 0.01)
        # Full Kelly = 0.01 / (0.01^2) = 100 (leverage capped at 1.0)
        # Half Kelly = 0.5 * 100 = 50 -> capped to 1.0
        f1 = calculate_half_kelly_fraction(0.01, 0.01, max_leverage=1.0)
        assert f1 == 1.0

        # Modest return (20bps = 0.002), volatility 3% (0.03 -> var = 0.0009)
        # Half Kelly = 0.5 * (0.002 / 0.0009) = 1.11 -> capped to 1.0
        f2 = calculate_half_kelly_fraction(0.002, 0.03, max_leverage=1.0)
        assert 0.0 < f2 <= 1.0

        # Small return (5bps = 0.0005), high volatility 4% (0.04 -> var = 0.0016)
        # Half Kelly = 0.5 * (0.0005 / 0.0016) = 0.156
        f3 = calculate_half_kelly_fraction(0.0005, 0.04, max_leverage=1.0)
        assert 0.10 <= f3 <= 0.20

        # Negative expected return -> Zero allocation (strict risk guard)
        f_neg = calculate_half_kelly_fraction(-0.005, 0.02)
        assert f_neg == 0.0

        # Vectorized input test
        rets = pd.Series([0.005, -0.002, 0.001])
        vols = pd.Series([0.02, 0.02, 0.01])
        f_vec = calculate_half_kelly_fraction(rets, vols)
        assert isinstance(f_vec, pd.Series)
        assert f_vec.iloc[0] > 0.0
        assert f_vec.iloc[1] == 0.0
        assert f_vec.iloc[2] > 0.0


class TestChanQuantEndToEnd:
    """Tests for add_chan_quant_features on DataFrame."""

    def test_add_chan_quant_features_multiticker(self) -> None:
        dates = pd.date_range("2026-01-01", periods=40, freq="B")
        df1 = pd.DataFrame({
            "date": dates,
            "ticker": "7011.T",
            "close": 2000.0 * np.exp(np.linspace(0, 0.1, 40)),
            "high": 2010.0,
            "low": 1990.0,
            "volume": 100000.0,
        })
        df2 = pd.DataFrame({
            "date": dates,
            "ticker": "7203.T",
            "close": 2500.0 * np.exp(np.linspace(0, -0.05, 40)),
            "high": 2510.0,
            "low": 2490.0,
            "volume": 200000.0,
        })
        comb = pd.concat([df1, df2], ignore_index=True)

        res = add_chan_quant_features(comb)

        for col in CHAN_BASE_FEATURE_NAMES:
            assert col in res.columns, f"Missing feature {col}"
            assert not res[col].isna().all(), f"All NaN in {col}"

        # Cross-sectional rank columns should also be added
        for col in CHAN_BASE_FEATURE_NAMES:
            assert f"{col}_cs_rank" in res.columns
