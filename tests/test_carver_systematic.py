"""Unit tests for Robert Carver's Systematic Trading quantitative module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_ai.features.carver_systematic import (
    apply_drawdown_throttle,
    calculate_idm,
    compute_carver_position_weights,
    compute_combined_carver_throttle,
    compute_volatility_regime_multiplier,
    scale_and_cap_forecast,
)


class TestCarverForecastScalingAndCapping:
    """Tests for Carver Chapter 10: Forecast Scaling and Capping [-20, +20]."""

    def test_scalar_scaling_and_capping(self) -> None:
        # Standard input with default_abs_norm
        f_norm = scale_and_cap_forecast(0.05, target_abs_forecast=10.0, default_abs_norm=0.05)
        assert pytest.approx(f_norm, abs=1e-5) == 10.0

        # Negative forecast
        f_neg = scale_and_cap_forecast(-0.025, target_abs_forecast=10.0, default_abs_norm=0.05)
        assert pytest.approx(f_neg, abs=1e-5) == -5.0

        # Capping at +20
        f_extreme_pos = scale_and_cap_forecast(0.20, target_abs_forecast=10.0, cap=20.0, default_abs_norm=0.05)
        assert f_extreme_pos == 20.0

        # Capping at -20
        f_extreme_neg = scale_and_cap_forecast(-0.50, target_abs_forecast=10.0, cap=20.0, default_abs_norm=0.05)
        assert f_extreme_neg == -20.0

    def test_array_and_series_scaling(self) -> None:
        raw_arr = np.array([0.02, -0.04, 0.06, 0.15, -0.20])
        # Mean absolute value is (0.02+0.04+0.06+0.15+0.20) / 5 = 0.47 / 5 = 0.094
        scaled = scale_and_cap_forecast(raw_arr, target_abs_forecast=10.0, cap=20.0)

        assert isinstance(scaled, np.ndarray)
        assert len(scaled) == len(raw_arr)
        assert np.all(scaled >= -20.0)
        assert np.all(scaled <= 20.0)
        # 0.15 / 0.094 * 10 = ~15.95 (uncapped)
        assert 14.0 < scaled[3] < 18.0
        # -0.20 / 0.094 * 10 = -21.27 -> capped at -20.0
        assert scaled[4] == -20.0

        # Pandas Series test
        ser = pd.Series([0.01, 0.02, 0.10], index=["A", "B", "C"])
        scaled_ser = scale_and_cap_forecast(ser, target_abs_forecast=10.0, cap=20.0)
        assert isinstance(scaled_ser, pd.Series)
        assert scaled_ser.index.tolist() == ["A", "B", "C"]
        assert np.all(scaled_ser <= 20.0)


class TestCarverInstrumentDiversificationMultiplier:
    """Tests for Carver Chapter 12: Instrument Diversification Multiplier (IDM)."""

    def test_idm_values(self) -> None:
        # N=1 must always give 1.0
        assert calculate_idm(1, avg_correlation=0.40) == 1.0

        # N=2, rho=0.40: 1 / sqrt(0.5 + 0.5*0.4) = 1 / sqrt(0.7) = 1.1952
        idm_2 = calculate_idm(2, avg_correlation=0.40)
        assert pytest.approx(idm_2, abs=1e-3) == 1.195

        # N=5, rho=0.40: 1 / sqrt(0.2 + 0.8*0.4) = 1 / sqrt(0.52) = 1.3867
        idm_5 = calculate_idm(5, avg_correlation=0.40)
        assert pytest.approx(idm_5, abs=1e-3) == 1.387

        # Uncorrelated assets rho=0.0: IDM = sqrt(N)
        idm_uncorr = calculate_idm(4, avg_correlation=0.0)
        assert pytest.approx(idm_uncorr, abs=1e-3) == 2.0

        # Upper limit capping
        idm_capped = calculate_idm(100, avg_correlation=0.0, max_idm=2.2)
        assert idm_capped == 2.2


class TestCarverPositionWeights:
    """Tests for Carver Inverse-Volatility & Risk Parity Sizing."""

    def test_equal_forecast_different_vols(self) -> None:
        # Stock 1 has vol=0.02, Stock 2 has vol=0.04 (double volatility)
        forecasts = [10.0, 10.0]
        vols = [0.02, 0.04]
        weights = compute_carver_position_weights(forecasts, vols, max_weight=0.80)

        # In inverse-volatility, Stock 1 (lower vol) should have higher weight than Stock 2
        assert weights[0] > weights[1]
        # Ratio should be approximately 2 : 1
        assert pytest.approx(weights[0] / weights[1], abs=0.05) == 2.0
        # Both weights should not exceed max_weight
        assert np.all(weights <= 0.80)
        # Total leverage should be <= 1.0
        assert np.sum(weights) <= 1.0001

    def test_higher_forecast_boosts_weight(self) -> None:
        # Same volatility, but Stock 1 has capped forecast 20.0, Stock 2 has 10.0
        forecasts = [20.0, 10.0]
        vols = [0.03, 0.03]
        weights = compute_carver_position_weights(forecasts, vols, max_weight=0.80)

        assert weights[0] > weights[1]
        assert pytest.approx(weights[0] / weights[1], abs=0.05) == 2.0

    def test_strict_max_weight_constraint(self) -> None:
        forecasts = [20.0]
        vols = [0.02]
        weights = compute_carver_position_weights(forecasts, vols, max_weight=0.50)
        assert weights[0] == 0.50


class TestCarverDrawdownThrottle:
    """Tests for Carver Chapter 15 & 16: Dynamic Risk Management & Drawdown Throttle."""

    def test_drawdown_throttle_phases(self) -> None:
        # Phase 1: Drawdown <= 7% -> No throttling (multiplier = 1.0)
        assert apply_drawdown_throttle(0.00, max_dd_limit=0.15, warn_dd=0.07) == 1.0
        assert apply_drawdown_throttle(0.05, max_dd_limit=0.15, warn_dd=0.07) == 1.0
        assert apply_drawdown_throttle(0.07, max_dd_limit=0.15, warn_dd=0.07) == 1.0

        # Phase 2: Drawdown between 7% and 15% -> Linear throttling
        mid_throttle = apply_drawdown_throttle(0.11, max_dd_limit=0.15, warn_dd=0.07, min_multiplier=0.20)
        assert 0.20 < mid_throttle < 1.0
        # At 11% (halfway between 7% and 15%): 1.0 - 0.5 * 0.8 = 0.60
        assert pytest.approx(mid_throttle, abs=0.02) == 0.60

        # Phase 3: Drawdown >= 15% -> Clamped to min_multiplier
        assert apply_drawdown_throttle(0.15, max_dd_limit=0.15, warn_dd=0.07, min_multiplier=0.20) == 0.20
        assert apply_drawdown_throttle(0.25, max_dd_limit=0.15, warn_dd=0.07, min_multiplier=0.20) == 0.20

    def test_volatility_regime_multiplier(self) -> None:
        # Normal vol: ratio = 1.0
        assert compute_volatility_regime_multiplier(0.020, baseline_vol=0.020) == 1.0
        # High vol spike (0.040, double): halves exposure
        assert compute_volatility_regime_multiplier(0.040, baseline_vol=0.020) == 0.50
        # Extreme calm: capped at max_leverage
        assert compute_volatility_regime_multiplier(0.010, baseline_vol=0.020, max_leverage=1.20) == 1.20

    def test_combined_throttle(self) -> None:
        comb = compute_combined_carver_throttle(
            current_drawdown=0.11,
            current_vol=0.025,
            baseline_vol=0.020,
            max_dd_limit=0.15,
            warn_dd=0.07,
        )
        assert 0.15 <= comb <= 1.0
        assert comb < 0.60  # throttled by both DD and above-baseline vol
