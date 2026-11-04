"""Unit tests for Series #15: Smart Execution Alpha, Closing Auction Imbalance & Universe Purification."""

import numpy as np
import pandas as pd
import pytest

from stock_ai.features.microstructure_paleologo import compute_closing_auction_distortion
from stock_ai.models.intraday import (
    compute_night_day_regime_allocation,
    compute_smart_execution_tactic,
    compute_ticker_compatibility_rank,
    normalize_daytrade_barriers_and_judgment,
)


def test_closing_auction_distortion_trap_detection():
    """Verify that artificial 15:25-15:30 closing auction imbalance spikes are detected."""
    # Test case 1: Extreme MOC spike at close (high CLV and jump ratio)
    close = pd.Series([1000.0, 1095.0])
    high = pd.Series([1010.0, 1100.0])
    low = pd.Series([990.0, 1000.0])
    jump_ratio = pd.Series([0.10, 0.45])

    res = compute_closing_auction_distortion(close, high, low, jump_ratio_5d=jump_ratio)
    assert "clv" in res
    assert "closing_auction_distortion_score" in res
    assert "is_closing_auction_distortion" in res

    # Second bar: (2*1095 - 1100 - 1000)/(1100-1000) = 90/100 = 0.90 CLV, jump_ratio=0.45
    assert res["clv"].iloc[-1] >= 0.85
    assert bool(res["is_closing_auction_distortion"].iloc[-1]) is True
    assert float(res["closing_auction_distortion_score"].iloc[-1]) > 0.0

    # Test case 2: Normal organic close (moderate CLV)
    close_norm = pd.Series([1000.0, 1050.0])
    high_norm = pd.Series([1010.0, 1100.0])
    low_norm = pd.Series([990.0, 1000.0])
    jump_norm = pd.Series([0.10, 0.15])

    res_norm = compute_closing_auction_distortion(close_norm, high_norm, low_norm, jump_ratio_5d=jump_norm)
    assert bool(res_norm["is_closing_auction_distortion"].iloc[-1]) is False


def test_smart_execution_tactic_breakout_and_pullback():
    """Verify ORB Breakout vs Pullback Limit vs Open Market tactics."""
    open_price = 2000.0

    # Tactic A: Breakout (high predicted return or Gap-ATR sweet spot)
    tactic_breakout = compute_smart_execution_tactic(
        ticker="6857.T",
        open_price=open_price,
        pred_intraday_ret=0.0075,
        gap_to_atr=0.45,
        volatility_20=0.025,
        continuous_trend_score=0.60,
    )
    assert tactic_breakout["tactic_code"] == "ORB_BREAKOUT"
    assert tactic_breakout["trigger_price"] > open_price
    assert "5分足ORB" in tactic_breakout["tactic_badge"]

    # Tactic B: Pullback (Gap down rebound or V-shape)
    tactic_pullback = compute_smart_execution_tactic(
        ticker="7011.T",
        open_price=open_price,
        pred_intraday_ret=0.0040,
        gap_to_atr=-0.80,
        volatility_20=0.020,
        continuous_trend_score=0.58,
    )
    assert tactic_pullback["tactic_code"] == "PULLBACK_LIMIT"
    assert tactic_pullback["trigger_price"] < open_price
    assert "初押し指値" in tactic_pullback["tactic_badge"]

    # Tactic C: Standard Market Open
    tactic_open = compute_smart_execution_tactic(
        ticker="9984.T",
        open_price=open_price,
        pred_intraday_ret=0.0030,
        gap_to_atr=0.05,
        volatility_20=0.020,
        continuous_trend_score=0.45,
    )
    assert tactic_open["tactic_code"] == "OPEN_MARKET"
    assert tactic_open["trigger_price"] == open_price
    assert "寄り付き成行" in tactic_open["tactic_badge"]


def test_ticker_compatibility_ranking_and_purge():
    """Verify S / A / B / D rank assignment and purging of chronic losers."""
    mock_adapt_state = {
        "ticker_td_state": {
            # Elite ticker: 5 trades, 100% win rate, positive return
            "5802.T": {"trades": 5, "wins": 5, "cum_return": 0.148},
            # Chronic loser: 4 trades, 0 wins, negative return
            "5803.T": {"trades": 4, "wins": 0, "cum_return": -0.062},
            # Good ticker: 5 trades, 60% win rate
            "6857.T": {"trades": 5, "wins": 3, "cum_return": 0.018},
            # Neutral / low sample ticker
            "7203.T": {"trades": 1, "wins": 1, "cum_return": 0.010},
        }
    }

    rank_elite = compute_ticker_compatibility_rank("5802.T", adapt_state=mock_adapt_state)
    assert rank_elite["rank"] == "S"
    assert rank_elite["is_boosted"] is True
    assert rank_elite["is_purged"] is False
    assert "ランク: S" in rank_elite["rank_badge"]

    rank_loser = compute_ticker_compatibility_rank("5803.T", adapt_state=mock_adapt_state)
    assert rank_loser["rank"] == "D"
    assert rank_loser["is_boosted"] is False
    assert rank_loser["is_purged"] is True
    assert "ランク: D" in rank_loser["rank_badge"]

    rank_good = compute_ticker_compatibility_rank("6857.T", adapt_state=mock_adapt_state)
    assert rank_good["rank"] == "A"
    assert rank_good["is_purged"] is False

    rank_neutral = compute_ticker_compatibility_rank("7203.T", adapt_state=mock_adapt_state)
    assert rank_neutral["rank"] == "B"
    assert rank_neutral["is_purged"] is False


def test_dual_shift_night_day_regime_allocation():
    """Verify dynamic night-to-day regime allocation weights and badges."""
    # Under low systemic fragility and normal volatility
    alloc_normal = compute_night_day_regime_allocation(
        drawdown_t=0.01,
        systemic_fragility_score=0.25,
        market_volatility_20=0.015,
        has_day_alpha_pick=True,
    )
    assert alloc_normal["regime_mode_code"] == "BALANCED_COMPOUND"
    assert alloc_normal["night_allocation_weight"] >= 0.85
    assert alloc_normal["day_allocation_weight"] >= 1.00

    # Under elevated systemic market fragility
    alloc_fragile = compute_night_day_regime_allocation(
        drawdown_t=0.04,
        systemic_fragility_score=0.75,
        market_volatility_20=0.024,
        has_day_alpha_pick=True,
    )
    assert alloc_fragile["regime_mode_code"] == "DAY_FOCUS"
    # Overnight is throttled down while Day allocation is prioritized
    assert alloc_fragile["night_allocation_weight"] < alloc_fragile["day_allocation_weight"]


def test_normalize_daytrade_barriers_with_smart_exec():
    """Verify that normalize_daytrade_barriers_and_judgment preserves tactical badges."""
    sample_rec = {
        "ticker": "6857.T",
        "open_price": 2500.0,
        "target_1_price": 2525.0,
        "target_final_price": 2575.0,
        "stop_loss_price": 2470.0,
        "volatility_20": 0.022,
        "judgment_code": "BUY",
        "execution_tactic_badge": "🎯 執行戦術: 5分足ORBブレイク確認 (突破目安: 2,507.5円 / 寄り天回避)",
        "compatibility_badge": "🏆 銘柄相性ランク: S (過去勝率 78%)",
    }
    norm = normalize_daytrade_barriers_and_judgment(sample_rec, side="LONG")
    assert norm["target_final_price"] > norm["target_1_price"] > norm["open_price"] > norm["stop_loss_price"]
    assert "execution_tactic_badge" in norm
    assert "compatibility_badge" in norm
