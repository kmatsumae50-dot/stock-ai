"""tests/test_intraday.py

Unit and integration tests for Morning 07:30 JST Pre-Market Intraday Model
and Dual-Shift Compound Portfolio Simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_ai.features.intraday import (
    ALL_INTRADAY_FEATURES,
    INTRADAY_BASE_FEATURES,
    INTRADAY_CROSS_SECTIONAL_FEATURES,
    add_cross_sectional_intraday_features,
    add_intraday_premarket_features,
)
from stock_ai.models.intraday import (
    classify_intraday_quantile,
    predict_intraday_for_ticker,
)
from stock_ai.backtest.overnight_portfolio import (
    simulate_intraday_strategy,
    simulate_dual_shift_strategy,
    simulate_overnight_strategy,
)


def test_add_intraday_premarket_features() -> None:
    """Test leak-free feature construction for intraday pre-market model."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 102.0, 101.0, 104.0, 103.0, 105.0, 107.0, 106.0, 108.0, 110.0],
            "high": [103.0, 104.0, 103.0, 106.0, 105.0, 108.0, 109.0, 108.0, 111.0, 112.0],
            "low": [99.0, 100.0, 100.0, 102.0, 101.0, 104.0, 105.0, 104.0, 106.0, 108.0],
            "close": [101.0, 103.0, 102.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0, 111.0],
            "return_1d": [0.01] * 10,
            "rsi_14": [52.0] * 10,
            "tesla_return_1d": [0.02] * 10,
            "usdjpy_return_1d": [0.005] * 10,
        }
    )

    result = add_intraday_premarket_features(df)

    assert "intraday_return" in result.columns
    assert "overnight_gap" in result.columns
    assert "prior_return_1d" in result.columns
    assert "expected_gap" in result.columns
    assert "gap_divergence" in result.columns

    # Check lag-1 guarantee
    assert result["prior_return_1d"].iloc[1] == df["return_1d"].iloc[0]

    # Check expected gap formula: 0.60 * us_shock + 0.40 * fx_shock
    expected = 0.60 * 0.02 + 0.40 * 0.005
    assert np.isclose(result["expected_gap"].iloc[1], expected)


def test_add_cross_sectional_intraday_features() -> None:
    """Test cross-sectional rankings per date without future leaks."""
    dates = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"])
    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T", "8035.T", "7203.T", "8035.T"],
            "sector": ["Auto", "Semi", "Auto", "Semi"],
            "overnight_gap": [-0.01, 0.02, 0.01, -0.02],
            "prior_return_1d": [0.01, -0.01, -0.02, 0.02],
            "prior_rsi_14": [40.0, 60.0, 50.0, 50.0],
            "prior_clv": [0.2, 0.8, -0.1, 0.5],
            "rebound_potential": [0.05, 0.0, 0.0, 0.0],
            "intraday_return": [0.015, -0.005, -0.01, 0.02],
        }
    )

    cs_df = add_cross_sectional_intraday_features(df)

    assert "cs_rank_open_gap" in cs_df.columns
    assert "sector_rel_gap" in cs_df.columns
    assert "target_intraday_cs_outperform" in cs_df.columns

    # On 2024-01-02, 7203.T gap is lower than 8035.T
    d1 = cs_df[cs_df["date"] == "2024-01-02"]
    assert d1.loc[d1["ticker"] == "7203.T", "cs_rank_open_gap"].values[0] < d1.loc[d1["ticker"] == "8035.T", "cs_rank_open_gap"].values[0]


def test_classify_intraday_quantile() -> None:
    """Test quantile classification and conviction thresholding."""
    q1 = classify_intraday_quantile(0.40, pred_ret=-0.005)
    assert q1["quantile"] == "Q1"
    assert q1["direction"] == "down"
    assert not q1["is_high_conviction"]

    q3 = classify_intraday_quantile(0.50, pred_ret=0.001)
    assert q3["quantile"] == "Q3"
    assert q3["direction"] == "neutral"
    assert not q3["is_high_conviction"]

    q5 = classify_intraday_quantile(0.58, pred_ret=0.004)
    assert q5["quantile"] == "Q5"
    assert q5["direction"] == "up"
    assert q5["is_high_conviction"]  # >= 0.50 and >= 30 bps


def test_simulate_intraday_and_dual_shift() -> None:
    """Test simulation math for Intraday and Dual-Shift compounding."""
    dates = pd.date_range("2024-04-01", periods=3, freq="B")
    df = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1], dates[2], dates[2]],
            "ticker": ["7203.T", "8035.T", "7203.T", "8035.T", "7203.T", "8035.T"],
            "next_open_return": [0.01, 0.02, -0.01, 0.00, 0.0, 0.0],
            "intraday_return": [0.015, -0.005, 0.02, 0.01, 0.0, 0.0],
            "pred_intraday_ret": [0.0040, 0.0010, 0.0035, 0.0005, 0.0, 0.0],
            "p_up": [0.55, 0.51, 0.56, 0.50, 0.50, 0.50],
            "quantile": ["Q5", "Q3", "Q5", "Q3", "Q3", "Q3"],
            "auc": [0.60, 0.50, 0.60, 0.50, 0.50, 0.50],
            "win_rate": [0.55, 0.50, 0.55, 0.50, 0.50, 0.50],
            "earnings_risk": [False] * 6,
            "jumping_catch_risk": [False] * 6,
        }
    )

    on_res = simulate_overnight_strategy(df, list(dates), "elite_top2_long", cost=0.0020, initial_capital=1_000_000.0)
    assert "metrics" in on_res
    assert len(on_res["trade_logs"]) == 3

    intra_res = simulate_intraday_strategy(df, list(dates), cost=0.0020, initial_capital=1_000_000.0, hurdle_bps=30.0)
    assert "metrics" in intra_res
    assert len(intra_res["trade_logs"]) == 3

    dual_res = simulate_dual_shift_strategy(on_res, intra_res, cost=0.0020, initial_capital=1_000_000.0)
    assert "metrics" in dual_res
    assert len(dual_res["trade_logs"]) == 3

    # Check compounding: Day 0 net return: on_ret = 0.01 - 0.002 = 0.008, intra_ret = 0.015 - 0.002 = 0.013
    # Compound: (1 + 0.008) * (1 + 0.013) - 1 = 0.021104
    d0 = dual_res["trade_logs"][0]
    assert np.isclose(d0["net_return"], (1 + on_res["trade_logs"][0]["net_return"]) * (1 + intra_res["trade_logs"][0]["net_return"]) - 1.0)


def test_simulate_intraday_waveform() -> None:
    """Test 6-pattern waveform classification, 09:00-15:30 timeline simulation, and actionable zones."""
    from stock_ai.models.intraday import (
        WAVEFORM_PATTERNS,
        simulate_intraday_waveform,
    )

    # 1. Trend Breakout
    wf_breakout = simulate_intraday_waveform(
        cur_p=2000.0,
        open_p=2000.0,
        high_p=2020.0,
        low_p=1995.0,
        prev_c=1980.0,
        prob_up=60.0,
        expected_bps=45.0,
        vol_20=0.02,
        phase="premarket",
        clv=0.8,
        pullback_pct=0.1,
        risk_flags=[],
    )
    assert wf_breakout["waveform_key"] == "trend_breakout"
    assert "独歩高" in wf_breakout["waveform_title"]
    assert len(wf_breakout["timeline_items"]) == 5
    assert wf_breakout["target_final_price"] > 2000.0
    assert wf_breakout["stop_loss_price"] < 2000.0

    # 2. Yoriten Fade
    wf_yoriten = simulate_intraday_waveform(
        cur_p=2000.0,
        open_p=2020.0,
        high_p=2030.0,
        low_p=1990.0,
        prev_c=2000.0,
        prob_up=40.0,
        expected_bps=-20.0,
        vol_20=0.02,
        phase="intraday_live",
        clv=0.25,
        pullback_pct=0.75,
        risk_flags=["⚠️ 寄り天・上ヒゲ失速警戒"],
    )
    assert wf_yoriten["waveform_key"] == "yoriten_fade"
    assert "寄り天" in wf_yoriten["waveform_title"]
    assert "買い厳禁" in wf_yoriten["best_entry_zone"]
    assert len(wf_yoriten["timeline_items"]) == 4  # Live intraday items

    # 3. GD V-shape Rebound
    wf_gd_v = simulate_intraday_waveform(
        cur_p=1990.0,
        open_p=1970.0,
        high_p=1995.0,
        low_p=1965.0,
        prev_c=2000.0,
        prob_up=52.0,
        expected_bps=15.0,
        vol_20=0.02,
        phase="intraday_live",
        clv=0.80,
        pullback_pct=0.15,
        risk_flags=[],
    )
    assert wf_gd_v["waveform_key"] == "gd_v_shape"
    assert "V字反発" in wf_gd_v["waveform_title"]

    # 4. Box Range
    wf_box = simulate_intraday_waveform(
        cur_p=2000.0,
        open_p=2000.0,
        high_p=2005.0,
        low_p=1995.0,
        prev_c=2000.0,
        prob_up=50.0,
        expected_bps=0.0,
        vol_20=0.015,
        phase="premarket",
        clv=0.5,
        pullback_pct=0.5,
        risk_flags=[],
    )
    assert wf_box["waveform_key"] == "box_range"
    assert "ボックス揉み合い" in wf_box["waveform_title"]


def test_intraday_advanced_features() -> None:
    """Test advanced pre-market features: Gap-to-ATR, shadows, and Clean Intraday Win target."""
    dates = pd.date_range("2024-01-01", periods=25, freq="B")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0 + i for i in range(25)],
            "high": [103.0 + i for i in range(25)],
            "low": [98.0 + i for i in range(25)],
            "close": [101.0 + i for i in range(25)],
            "return_1d": [0.01] * 25,
            "volatility_20": [0.02] * 25,
            "rsi_14": [50.0] * 25,
        }
    )

    result = add_intraday_premarket_features(df)

    assert "gap_to_atr_ratio" in result.columns
    assert "rolling_intraday_win_rate_20d" in result.columns
    assert "rolling_gap_fade_corr_20d" in result.columns
    assert "prior_upper_shadow_ratio" in result.columns
    assert "prior_lower_shadow_ratio" in result.columns
    assert "rolling_buying_pressure_5d" in result.columns
    assert "target_clean_intraday_win" in result.columns

    # Verify shadow calculation range [0, 1]
    assert (result["prior_upper_shadow_ratio"] >= 0.0).all()
    assert (result["prior_upper_shadow_ratio"] <= 1.0).all()
    assert (result["prior_lower_shadow_ratio"] >= 0.0).all()
    assert (result["prior_lower_shadow_ratio"] <= 1.0).all()

    # Clean Intraday Win target: binary 0 or 1
    assert set(result["target_clean_intraday_win"].dropna().unique()).issubset({0, 1})


def test_simulate_intraday_waveform_with_microstructure() -> None:
    """Test simulate_intraday_waveform when real-time 5m microstructure is integrated."""
    from stock_ai.models.intraday import simulate_intraday_waveform

    micro_bull = {
        "vwap": 2010.0,
        "latest_price": 2025.0,
        "open_price": 2000.0,
        "day_high": 2030.0,
        "day_low": 1995.0,
        "vwap_deviation_pct": 0.75,
        "is_above_vwap": True,
        "orb_high": 2015.0,
        "orb_low": 1995.0,
        "orb_status": "朝イチ高値ブレイク（順張り加速）",
        "orb_code": "BREAK_HIGH",
        "rvol_intraday": 1.5,
    }

    wf = simulate_intraday_waveform(
        cur_p=2025.0,
        open_p=2000.0,
        high_p=2030.0,
        low_p=1995.0,
        prev_c=2000.0,
        prob_up=62.0,
        expected_bps=40.0,
        vol_20=0.02,
        phase="intraday_live",
        clv=0.85,
        pullback_pct=0.15,
        risk_flags=[],
        micro=micro_bull,
    )

    # Should trigger trend breakout due to ORB break & VWAP support
    assert wf["waveform_key"] == "trend_breakout"
    assert "VWAP" in wf["best_entry_zone"]
    assert "ORB" in wf["best_entry_zone"]
    assert wf["stop_loss_price"] <= 2010.0 * 0.995  # VWAP support guard
    assert any("VWAP" in item[1] for item in wf["timeline_items"] if "実績" in item[0])

