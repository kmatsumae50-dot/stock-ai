"""tests/test_jump_bipower.py

Unit tests for Barndorff-Nielsen & Shephard Realized Bipower Variation / Jump Decomposition
and Optimal Stopping Theory dynamic exit barriers.
"""

import numpy as np
import pandas as pd
import pytest

from stock_ai.features.jump_bipower_optimal_stop import (
    compute_bipower_variation,
    compute_rolling_bipower_jump,
    compute_bipower_jump_decomposition,
    compute_optimal_stopping_barriers,
    add_jump_bipower_features,
)


def test_bipower_variation_continuous_vs_jump():
    """Verify that pure continuous returns yield JumpRatio near 0, while sudden jumps yield high JumpRatio in [0, 1]."""
    np.random.seed(42)
    M = 100
    r_cont = np.random.normal(0, 0.002, M)

    rv_c, bv_c, jp_c, ratio_c = compute_bipower_variation(r_cont)
    assert rv_c > 0.0
    assert bv_c > 0.0
    assert jp_c >= 0.0
    assert 0.0 <= ratio_c <= 1.0
    # Continuous variation should have low jump ratio (typically < 0.15)
    assert ratio_c < 0.20

    # Add single large jump
    r_jump = r_cont.copy()
    r_jump[50] += 0.040
    rv_j, bv_j, jp_j, ratio_j = compute_bipower_variation(r_jump)
    assert rv_j > rv_c
    assert jp_j > jp_c
    assert 0.0 <= ratio_j <= 1.0
    # Jump ratio must surge significantly
    assert ratio_j > 0.50


def test_bipower_variation_edge_cases():
    """Check empty, single element, zero returns, and NaN handling."""
    rv, bv, jp, jr = compute_bipower_variation([])
    assert (rv, bv, jp, jr) == (0.0, 0.0, 0.0, 0.0)

    rv, bv, jp, jr = compute_bipower_variation([0.01])
    assert (rv, bv, jp, jr) == (0.0, 0.0, 0.0, 0.0)

    rv, bv, jp, jr = compute_bipower_variation([0.0, 0.0, 0.0])
    assert (rv, bv, jp, jr) == (0.0, 0.0, 0.0, 0.0)

    rv, bv, jp, jr = compute_bipower_variation([np.nan, 0.005, -0.004, 0.006, np.nan])
    assert rv > 0.0
    assert 0.0 <= jr <= 1.0


def test_rolling_bipower_jump_bounds():
    """Verify that rolling bipower jump outputs are strictly bounded in [0, 1]."""
    rets = pd.Series([0.005, -0.002, 0.003, 0.040, -0.001, 0.002, -0.004])
    rv_s, bv_s, jr_s = compute_rolling_bipower_jump(rets, window=5)

    assert len(rv_s) == len(rets)
    assert len(bv_s) == len(rets)
    assert len(jr_s) == len(rets)

    assert (rv_s >= 0.0).all()
    assert (bv_s >= 0.0).all()
    assert ((jr_s >= 0.0) & (jr_s <= 1.0)).all()
    # At index 3 (the spike), jump ratio should be elevated
    assert jr_s.iloc[3] > 0.40


def test_bipower_jump_decomposition_features():
    """Verify DataFrame feature generation, including fake spike exhaustion flag."""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    np.random.seed(123)
    closes = 1000.0 * np.cumprod(1.0 + np.random.normal(0, 0.01, size=30))
    # Inject a massive fake spike on day 20 with high RSI and high upper shadow
    closes[20] = closes[19] * 1.055

    highs = closes * 1.01
    highs[20] = closes[20] * 1.035  # Large upper shadow on day 20
    lows = closes * 0.99
    opens = closes * 0.995

    df = pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "rsi_14": 75.0,
        "volatility_20": 0.02,
        "vpin_informed_distribution_score": 0.60,
    })

    feats = compute_bipower_jump_decomposition(df)
    jr_5d = feats["jump_ratio_5d"]
    jr_20d = feats["jump_ratio_20d"]
    cont_score = feats["continuous_trend_score"]
    fake_flag = feats["fake_spike_exhaustion_flag"]

    assert len(jr_5d) == 30
    assert ((jr_5d >= 0.0) & (jr_5d <= 1.0)).all()
    assert ((jr_20d >= 0.0) & (jr_20d <= 1.0)).all()
    assert ((cont_score >= 0.0) & (cont_score <= 1.0)).all()

    # Day 20 had massive spike with high jump, high upper shadow, high RSI -> fake spike flag must trigger
    assert fake_flag.iloc[20] == True


def test_optimal_stopping_barriers_dynamic_behavior():
    """Verify that optimal stopping barriers expand in clean trend and tighten in fake spike/high toxicity."""
    # 1. Clean strong trend regime: H > 0.52, continuous trend dominance
    barriers_trend = compute_optimal_stopping_barriers(
        drift=0.015,
        volatility=0.025,
        hurst=0.62,
        jump_ratio=0.10,
        continuous_score=0.85,
        vpin_toxicity=0.20,
        is_fake_spike=False,
    )
    # Take-profit should expand to let profits run (>= 4.0%)
    assert barriers_trend["dynamic_take_profit"] >= 0.038
    assert barriers_trend["dynamic_take_profit_pct"] >= 3.8
    assert barriers_trend["dynamic_stop_loss"] < 0.0
    assert "連続トレンド優位" in barriers_trend["jump_regime_badge"]

    # 2. Fake spike / jump exhaustion regime
    barriers_spike = compute_optimal_stopping_barriers(
        drift=0.030,
        volatility=0.035,
        hurst=0.62,
        jump_ratio=0.65,
        continuous_score=0.20,
        vpin_toxicity=0.65,
        is_fake_spike=True,
    )
    # Take-profit should contract for quick early lock-in (<= 2.5%)
    assert barriers_spike["dynamic_take_profit"] <= 0.025
    assert barriers_spike["dynamic_take_profit_pct"] <= 2.5
    # Stop-loss should tighten to avoid post-spike dump (-1.5% or tighter)
    assert barriers_spike["dynamic_stop_loss"] >= -0.015
    assert "突発ジャンプ過熱" in barriers_spike["jump_regime_badge"]

    # 3. Volatility scaling: higher volatility expands base barriers
    barriers_low_vol = compute_optimal_stopping_barriers(drift=0.005, volatility=0.012)
    barriers_high_vol = compute_optimal_stopping_barriers(drift=0.005, volatility=0.035)
    assert barriers_high_vol["dynamic_take_profit"] > barriers_low_vol["dynamic_take_profit"]


def test_zero_future_leakage():
    """Verify strict causality: values at time t must not change when future data (t+1..T) is modified."""
    np.random.seed(99)
    n = 25
    closes_base = 1000.0 * np.cumprod(1.0 + np.random.normal(0, 0.01, size=n))
    df_base = pd.DataFrame({"close": closes_base})

    feats_base = compute_bipower_jump_decomposition(df_base)
    t = 15
    val_t_jr5_base = feats_base["jump_ratio_5d"].iloc[t]
    val_t_cont_base = feats_base["continuous_trend_score"].iloc[t]

    # Mutate future data after time t (index 16..24)
    closes_mutated = closes_base.copy()
    closes_mutated[16:] = closes_mutated[16:] * 2.5  # extreme future mutation
    df_mutated = pd.DataFrame({"close": closes_mutated})

    feats_mutated = compute_bipower_jump_decomposition(df_mutated)
    val_t_jr5_mut = feats_mutated["jump_ratio_5d"].iloc[t]
    val_t_cont_mut = feats_mutated["continuous_trend_score"].iloc[t]

    assert np.isclose(val_t_jr5_base, val_t_jr5_mut, atol=1e-12)
    assert np.isclose(val_t_cont_base, val_t_cont_mut, atol=1e-12)
