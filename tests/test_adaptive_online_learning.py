"""tests/test_adaptive_online_learning.py

Unit tests for Series #7: Closed-Loop Online Adaptation & Self-Evolution System.
Theoretical Foundations:
1. Nicolò Cesa-Bianchi: Exponentially Weighted Forecaster / Hedge Algorithm.
2. Richard S. Sutton: Temporal-Difference (TD) Prediction Error & Rescorla-Wagner Learning.
3. Contextual Bandit Memory: Failure Archetype Profiler and RBF Kernel Penalty.
4. Temporal Causality: Zero Lookahead / Zero Information Leakage Verification.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from stock_ai.features.adaptive_online_learning import (
    DEFAULT_WAVEFORM_KEYS,
    WEIGHT_CLIP_MAX,
    WEIGHT_CLIP_MIN,
    RollingOnlineAdaptor,
    compute_adaptive_credibility_multiplier,
    compute_failure_similarity_penalty,
    extract_failure_profile_from_trade,
    generate_self_reflection_summary,
    initialize_fresh_state,
    load_adaptation_state,
    save_adaptation_state,
    update_failure_archetypes,
    update_online_adaptation_from_scorecard,
    update_regime_expert_weights,
    update_ticker_td_credibility,
)


# ===========================================================================
# 1. Hedge Algorithm Tests (Cesa-Bianchi Exponential Weights)
# ===========================================================================

def test_hedge_weights_boost_winning_waveform_and_decay_losing():
    """Verify winning waveforms gain weight while losing/stopped-out waveforms lose weight."""
    initial_weights = {wf: 1.0 for wf in DEFAULT_WAVEFORM_KEYS}
    initial_weights["style_momentum"] = 1.0
    initial_weights["style_mean_reversion"] = 1.0

    # Mock trades: trend_breakout wins big (+2.5%), yoriten_fade loses big (-1.8%)
    trades = [
        {
            "ticker": "8035.T",
            "predicted_waveform": "🔥【寄り付き突破 ➔ 終日トレンド独歩高型】",
            "waveform_key": "trend_breakout",
            "result_status": "WIN",
            "net_return_pct": 2.50,
        },
        {
            "ticker": "9984.T",
            "predicted_waveform": "🔥【寄り付き突破 ➔ 終日トレンド独歩高型】",
            "waveform_key": "trend_breakout",
            "result_status": "WIN",
            "net_return_pct": 2.10,
        },
        {
            "ticker": "6758.T",
            "predicted_waveform": "⚠️【寄り天（朝イチ高値）➔ 後場ダラ下げ失速型】",
            "waveform_key": "yoriten_fade",
            "result_status": "LOSS",
            "net_return_pct": -1.80,
        },
    ]

    updated = update_regime_expert_weights(initial_weights, trades, eta=0.20)

    # Check 1: Winning waveform boosted > 1.0
    assert updated["trend_breakout"] > 1.0
    assert updated["trend_breakout"] >= 1.05

    # Check 2: Losing waveform penalized < 1.0
    assert updated["yoriten_fade"] < 1.0
    assert updated["yoriten_fade"] <= 0.95

    # Check 3: Momentum style is boosted since trend_breakout is momentum
    assert updated["style_momentum"] > 1.0

    # Check 4: Weights respect clip boundaries
    for w in updated.values():
        assert WEIGHT_CLIP_MIN <= w <= WEIGHT_CLIP_MAX


def test_hedge_weights_extreme_clipping():
    """Verify Hedge weights strictly respect [0.60, 1.40] under extreme consecutive losses or gains."""
    weights = {wf: 1.0 for wf in DEFAULT_WAVEFORM_KEYS}

    # Simulate 50 consecutive disaster trades
    disaster_trades = [
        {
            "ticker": "6758.T",
            "waveform_key": "yoriten_fade",
            "result_status": "LOSS",
            "net_return_pct": -5.0,
        }
    ]
    for _ in range(50):
        weights = update_regime_expert_weights(weights, disaster_trades, eta=0.30)

    assert weights["yoriten_fade"] == pytest.approx(WEIGHT_CLIP_MIN, abs=1e-4)

    # Simulate 50 consecutive massive win trades
    win_trades = [
        {
            "ticker": "8035.T",
            "waveform_key": "trend_breakout",
            "result_status": "WIN",
            "net_return_pct": +10.0,
        }
    ]
    for _ in range(50):
        weights = update_regime_expert_weights(weights, win_trades, eta=0.30)

    assert weights["trend_breakout"] == pytest.approx(WEIGHT_CLIP_MAX, abs=1e-4)


# ===========================================================================
# 2. Rescorla-Wagner TD Error Tracker Tests (Sutton TD Learning)
# ===========================================================================

def test_ticker_and_sector_td_credibility():
    """Verify TD prediction error delta = R_actual - R_expected updates credibility correctly."""
    ticker_state = {}
    sector_state = {}

    # Trade 1: Ticker A outperforms expectation (Expected +0.20%, Actual +2.20% -> TD error +2.00%)
    trades = [
        {
            "ticker": "8035.T",
            "sector": "電気機器",
            "probability_up": 60.0,
            "expected_return_bps": 20.0,  # 0.20%
            "net_return_pct": 2.20,      # 2.20% -> delta = +0.02
        },
        {
            "ticker": "7203.T",
            "sector": "輸送用機器",
            "probability_up": 60.0,
            "expected_return_bps": 30.0,  # 0.30%
            "net_return_pct": -1.50,     # -1.50% -> delta = -0.018
        },
    ]

    t_state, s_state = update_ticker_td_credibility(
        ticker_state, sector_state, trades, alpha_ticker=0.20, alpha_sector=0.15
    )

    # Ticker A credibility must be positive
    assert t_state["8035.T"]["v"] > 0.0
    assert t_state["8035.T"]["updates_count"] == 1

    # Ticker B credibility must be negative
    assert t_state["7203.T"]["v"] < 0.0

    # Sector Electric Appliances must be positive
    assert s_state["電気機器"]["v"] > 0.0

    # Combined Credibility Multipliers
    mult_a = compute_adaptive_credibility_multiplier("8035.T", "電気機器", t_state, s_state)
    mult_b = compute_adaptive_credibility_multiplier("7203.T", "輸送用機器", t_state, s_state)
    mult_untracked = compute_adaptive_credibility_multiplier("9999.T", "その他", t_state, s_state)

    assert mult_a > 1.0  # Boosted
    assert mult_b < 1.0  # Penalized
    assert mult_untracked == 1.0  # Neutral baseline


# ===========================================================================
# 3. Contextual Bandit Memory: Failure Archetype Penalty Tests
# ===========================================================================

def test_failure_archetype_recording_and_similarity_penalty():
    """Verify stopped out trades are remembered and similar candidate profiles trigger safety brakes."""
    # Step 1: Create a trade that hit STOPPED_OUT
    failed_trade = {
        "ticker": "6758.T",
        "result_code": "STOPPED_OUT",
        "result_status": "LOSS",
        "actual_open": 1000.0,
        "actual_high": 1012.0,
        "actual_low": 980.0,
        "actual_close": 982.0,
        "net_return_pct": -1.80,
        "gap_to_atr": 1.20,
        "chan_hurst": 0.38,
        "micro_upward_elasticity": 0.82,
        "prior_clv": 0.25,
    }

    arch = extract_failure_profile_from_trade(failed_trade)
    assert arch is not None
    assert arch.reason == "STOPPED_OUT"
    assert arch.gap_to_atr == 1.20

    # Step 2: Store in archetypes list
    archetypes = update_failure_archetypes([], [failed_trade])
    assert len(archetypes) == 1

    # Step 3: Candidate with very similar profile (high gap, low hurst, low elasticity, low clv)
    dangerous_candidate = {
        "gap_to_atr": 1.18,
        "chan_hurst_20d": 0.39,
        "micro_upward_elasticity_10d": 0.84,
        "prior_clv": 0.26,
    }
    prob_pen, carver_mult, risk_flag = compute_failure_similarity_penalty(
        dangerous_candidate, archetypes, similarity_threshold=0.72
    )

    # Danger detected:
    assert prob_pen < 0.0  # negative penalty
    assert carver_mult < 1.0  # position dampener
    assert risk_flag is not None
    assert "過去失敗類似パターン検知" in risk_flag

    # Step 4: Healthy candidate with opposite profile (small gap, strong hurst, high elasticity)
    healthy_candidate = {
        "gap_to_atr": 0.30,
        "chan_hurst_20d": 0.65,
        "micro_upward_elasticity_10d": 1.60,
        "prior_clv": 0.85,
    }
    h_prob_pen, h_carver_mult, h_risk_flag = compute_failure_similarity_penalty(
        healthy_candidate, archetypes, similarity_threshold=0.72
    )
    assert h_prob_pen == 0.0
    assert h_carver_mult == 1.0
    assert h_risk_flag is None


# ===========================================================================
# 4. Temporal Causality & Zero Leakage Verification
# ===========================================================================

def test_rolling_adaptor_zero_lookahead_causality():
    """Verify strict causality: Trade at day t CANNOT see day t's settlement outcome."""
    adaptor = RollingOnlineAdaptor()

    # Day 1: Pristine state before trading
    st_day1 = adaptor.get_current_state()
    assert st_day1["total_updates"] == 0
    assert st_day1["regime_expert_weights"]["trend_breakout"] == 1.0
    assert "8035.T" not in st_day1["ticker_td_state"]

    # Execute Day 1 trade decision with st_day1 (at 09:00 JST)
    day1_trade = {
        "ticker": "8035.T",
        "waveform_key": "trend_breakout",
        "result_status": "WIN",
        "net_return_pct": 3.0,
    }

    # At 15:35 JST, day 1 settles and adaptor is updated
    adaptor.update_from_settled_trades("2026-10-24", [day1_trade])

    # Day 2: Now day 1's outcome is reflected for day 2 trading
    st_day2 = adaptor.get_current_state()
    assert st_day2["total_updates"] == 1
    assert st_day2["regime_expert_weights"]["trend_breakout"] > 1.0
    assert "8035.T" in st_day2["ticker_td_state"]
    assert st_day2["ticker_td_state"]["8035.T"]["v"] > 0.0


# ===========================================================================
# 5. Closed-Loop Scorecard Integration & Reflection Summary Test
# ===========================================================================

def test_scorecard_reflection_summary_generation(tmp_path: Path):
    """Verify complete closed-loop update from post-market scorecard data."""
    state_file = tmp_path / "test_adaptation_state.json"

    mock_scorecard = {
        "date": "2026-10-24",
        "total_trades": 2,
        "wins": 1,
        "losses": 1,
        "win_rate_pct": 50.0,
        "total_virtual_pnl_yen": 25000,
        "trades": [
            {
                "ticker": "8035.T",
                "sector": "電気機器",
                "predicted_waveform": "🔥【寄り付き突破 ➔ 終日トレンド独歩高型】",
                "waveform_key": "trend_breakout",
                "result_status": "WIN",
                "result_code": "TARGET_HIT",
                "actual_open": 24000.0,
                "actual_high": 25000.0,
                "actual_low": 23900.0,
                "actual_close": 24800.0,
                "net_return_pct": 3.20,
                "virtual_pnl_100_shares": 50000,
            },
            {
                "ticker": "6758.T",
                "sector": "電気機器",
                "predicted_waveform": "⚠️【寄り天（朝イチ高値）➔ 後場ダラ下げ失速型】",
                "waveform_key": "yoriten_fade",
                "result_status": "LOSS",
                "result_code": "STOPPED_OUT",
                "actual_open": 2800.0,
                "actual_high": 2830.0,
                "actual_low": 2740.0,
                "actual_close": 2750.0,
                "net_return_pct": -1.80,
                "virtual_pnl_100_shares": -25000,
                "gap_to_atr": 1.10,
                "chan_hurst": 0.40,
                "micro_upward_elasticity": 0.90,
                "prior_clv": 0.30,
            },
        ],
    }

    res = update_online_adaptation_from_scorecard(mock_scorecard, state_file=state_file)

    assert "reflection_summary" in res
    assert len(res["reflection_summary"]) == 3
    assert "波形・地合い適応 (Hedge)" in res["reflection_summary"][0]
    assert "銘柄・セクター信用度 (TD学習)" in res["reflection_summary"][1]
    assert "地雷コンテキスト (Bandit Memory)" in res["reflection_summary"][2]

    # Verify file saved
    assert state_file.exists()
    loaded = json.loads(state_file.read_text(encoding="utf-8"))
    assert loaded["total_updates"] == 1
    assert len(loaded["failure_archetypes"]) == 1


def test_four_stage_outcome_and_daytrade_reflection_formatting():
    """Verify 4-stage outcome scoring integration and mobile-first reflection text formatting."""
    from stock_ai.features.adaptive_online_learning import format_daytrade_reflection_overview

    mock_scorecard = {
        "date": "2026-10-26",
        "total_trades": 3,
        "wins": 2,
        "flats": 0,
        "losses": 1,
        "win_rate_pct": 66.7,
        "avg_return_pct": 1.42,
        "total_virtual_pnl_yen": 42500,
        "trades": [
            {
                "ticker": "7011.T",
                "name": "三菱重工業",
                "predicted_waveform": "🚀【GU寄り付き急伸 ➔ 朝押し反発型】",
                "waveform_key": "gu_rebound_high",
                "result_status": "WIN",
                "result_code": "TARGET_HIT",
                "result_badge": "🎯 利確目標到達",
                "net_return_pct": 3.15,
            },
            {
                "ticker": "5803.T",
                "name": "フジクラ",
                "predicted_waveform": "📈【GD寄り付き ➔ 窓埋めV字反発型】",
                "waveform_key": "gd_v_shape",
                "result_status": "WIN",
                "result_code": "WIN_CLOSE",
                "result_badge": "🟢 予想通り上昇",
                "net_return_pct": 1.28,
            },
            {
                "ticker": "8035.T",
                "name": "東京エレクトロン",
                "predicted_waveform": "🚀【寄り付き突破 ➔ 終日トレンド独歩高型】",
                "waveform_key": "trend_breakout",
                "result_status": "LOSS",
                "result_code": "LOSS_CLOSE",
                "result_badge": "🔴 寄り天失速",
                "actual_waveform": "寄り天・前場急伸後失速陰線",
                "net_return_pct": -0.18,
            },
        ],
    }

    overview_text = format_daytrade_reflection_overview(mock_scorecard)

    # 1. Headline section
    assert "📋 **【前日予想の答え合わせ見返し (10/26実績)】**" in overview_text
    assert "推奨トップ3成績: **2勝1敗** (勝率 **66.7%** ｜ 平均リターン **+1.42%**)" in overview_text

    # 2. Top 3 stock rows
    assert "1. `7011` **三菱重工業** (予想: 🚀GU朝押し型) ➔ **🎯 利確目標到達** (+3.15%)" in overview_text
    assert "2. `5803` **フジクラ** (予想: 📈窓埋め反発型) ➔ **🟢 予想通り上昇** (+1.28%)" in overview_text
    assert "3. `8035` **東京エレクトロン** (予想: 🚀ORBブレイク型) ➔ **🔴 寄り天失速** (-0.18%)" in overview_text

    # 3. AI Closed-Loop Learning Feedback
    assert "🧠 **【前日からのAI学習・本日の補正】**" in overview_text
    assert "前日の寄り天失速(TD誤差" in overview_text
    assert "「高Jump Ratio(フェイク急騰)」への警戒ペナルティを強化" in overview_text
    assert "「大口ステルス買い集め×連続トレンド型」の重みを引き上げて" in overview_text
