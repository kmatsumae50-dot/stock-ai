"""Unit tests for the 4 Structural Bug Fixes identified in the 3-Day Live Forward Audit (10/29, 10/30, 11/2):
1. Strict barrier & judgment normalization (`top_up`: target_1 > open > stop_loss and judgment_code != "SELL"; `top_down`: target_1 < open < stop_loss).
2. Mandatory `VPIN > 0` gate for Long `unshackled_mode_active` (Full Alpha Mode) and theme unlock threshold.
3. Idempotent guard on `RollingOnlineAdaptor` and `update_online_adaptation_from_scorecard` for duplicate/weekend dates.
4. Gap-to-ATR Yoriten trap detection when `VPIN == 0.0` (especially on Fridays / unbacked gap-ups).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from stock_ai.daytrade.scorecard import (
    evaluate_todays_daytrade_scorecard,
    normalize_daytrade_barriers_and_judgment,
)
from stock_ai.features.adaptive_online_learning import (
    RollingOnlineAdaptor,
    initialize_fresh_state,
    save_adaptation_state,
    update_online_adaptation_from_scorecard,
)
from stock_ai.features.chan_quant import compute_conditional_trend_unshackled_chan
from stock_ai.features.microstructure_paleologo import compute_dynamic_sector_concentration_unlock
from stock_ai.models.intraday import (
    compute_gap_atr_entry_adjustment,
    predict_daytrade_sniper,
)


def test_top_up_and_top_down_barrier_and_judgment_normalization() -> None:
    """Invariant 1: Every stock in top_up strictly satisfies target_1_price > open_price > stop_loss_price and judgment_code != 'SELL'."""
    corrupted_long_candidates: list[dict[str, Any]] = [
        {
            "ticker": "8058.T",
            "name": "三菱商事",
            "open_price": 3628.0,
            "target_1_price": 3599.0,  # Inverted (< open_price)
            "target_final_price": 3541.0,  # Inverted (< open_price)
            "stop_loss_price": 3664.3,  # Inverted (> open_price)
            "judgment_code": "SELL",
            "judgment": "🔴 今日下がる [売り・見送り / 陰線警戒]",
            "probability_up": 44.5,
            "expected_return_bps": -12.0,
            "waveform_key": "gd_downtrend",
            "volatility_20": 0.018,
        },
        {
            "ticker": "4063.T",
            "name": "信越化学工業",
            "open_price": 4769.0,
            "target_1_price": 4730.8,  # Inverted
            "target_final_price": 4654.5,  # Inverted
            "stop_loss_price": 4816.7,  # Inverted
            "judgment_code": "SELL",
            "judgment": "🔴 今日下がる [売り・失速警戒・見送り]",
            "probability_up": 51.2,
            "expected_return_bps": 25.0,
            "waveform_key": "yoriten_fade",
            "volatility_20": 0.020,
        },
    ]

    for raw_rec in corrupted_long_candidates:
        norm = normalize_daytrade_barriers_and_judgment(raw_rec, side="LONG")
        assert norm["judgment_code"] != "SELL"
        assert "下がる" not in norm["judgment"]
        assert norm["target_1_price"] > norm["open_price"] > norm["stop_loss_price"]
        assert norm["target_final_price"] > norm["target_1_price"] > norm["open_price"]
        assert norm["target_1_pct"] > 0.0
        assert norm["target_final_pct"] > norm["target_1_pct"]
        assert norm["stop_loss_pct"] < 0.0
        assert norm["expected_return_bps"] > 0.0
        assert norm["waveform_key"] not in ("gd_downtrend", "yoriten_fade")

    # Verify top_down (SHORT) strictly enforces target_final < target_1 < open < stop_loss
    corrupted_short = {
        "ticker": "6861.T",
        "name": "キーエンス",
        "open_price": 62000.0,
        "target_1_price": 62800.0,  # Wrong direction for short
        "target_final_price": 63500.0,
        "stop_loss_price": 61200.0,
        "judgment_code": "BUY",
        "volatility_20": 0.022,
    }
    norm_short = normalize_daytrade_barriers_and_judgment(corrupted_short, side="SHORT")
    assert norm_short["judgment_code"] == "SELL"
    assert norm_short["target_final_price"] < norm_short["target_1_price"] < norm_short["open_price"] < norm_short["stop_loss_price"]
    assert norm_short["target_1_pct"] < 0.0
    assert norm_short["stop_loss_pct"] > 0.0


def test_vpin_zero_blocks_unshackled_full_alpha_mode() -> None:
    """Invariant 2: When VPIN smart accumulation == 0.0 (and no positive OFI), unshackled_mode_active and is_trend_unshackled must be False."""
    # 1. Chan conditional trend unshackling scalar & series with VPIN == 0.0
    chan_scalar = compute_conditional_trend_unshackled_chan(
        hurst=0.45,
        ou_zscore=1.6,
        kalman_zscore=1.4,
        continuous_trend_score=0.72,
        jump_ratio_5d=0.10,
        vpin_smart_accumulation_score=0.0,
        vpin_signed_ofi_5d=0.0,
    )
    assert chan_scalar["is_trend_unshackled"] is False

    chan_series = compute_conditional_trend_unshackled_chan(
        hurst=pd.Series([0.45, 0.45]),
        continuous_trend_score=pd.Series([0.75, 0.75]),
        jump_ratio_5d=pd.Series([0.12, 0.12]),
        vpin_smart_accumulation_score=pd.Series([0.0, 0.42]),
        vpin_signed_ofi_5d=pd.Series([0.0, 0.18]),
    )
    assert list(chan_series["is_trend_unshackled"]) == [False, True]

    # 2. Neutral policy wave (0.50) must not trigger theme unlock
    theme_neutral = compute_dynamic_sector_concentration_unlock(
        policy_wave_score=0.50,
        leadlag_spillover_score=0.35,
    )
    assert theme_neutral["is_theme_unlocked"] is False

    # 3. End-to-end predict_daytrade_sniper with VPIN == 0.0 must set unshackled_mode_active = False
    dates = pd.date_range("2026-09-01", periods=30, freq="B")
    mock_df = pd.DataFrame({
        "date": dates,
        "open": [1000.0 + i * 2 for i in range(30)],
        "high": [1010.0 + i * 2 for i in range(30)],
        "low": [995.0 + i * 2 for i in range(30)],
        "close": [1005.0 + i * 2 for i in range(30)],
        "volume": [100000] * 30,
        "volatility_20": [0.018] * 30,
        "continuous_trend_score": [0.78] * 30,
        "jump_ratio_5d": [0.10] * 30,
        "vpin_smart_accumulation_score": [0.0] * 30,
        "vpin_signed_ofi_5d": [-0.05] * 30,
        "vpin_toxicity_10d": [0.42] * 30,
        "vpin_informed_distribution_score": [0.20] * 30,
        "insider_stealth_accumulation_10d": [0.25] * 30,
        "policy_sector_insider_wave_score": [0.50] * 30,
        "leadlag_spillover_alpha_1d": [0.20] * 30,
    })
    sniper_res = predict_daytrade_sniper("6861.T", features_df=mock_df, fetch_5m_micro=False)
    assert sniper_res["vpin_smart_accumulation_score"] == 0.0
    assert sniper_res["is_trend_unshackled"] is False
    assert sniper_res["unshackled_mode_active"] is False
    assert "🔓 ブレーキ解除" not in sniper_res["unshackled_badge"]


def test_rolling_online_adaptor_and_scorecard_idempotent_guard(tmp_path: Path) -> None:
    """Invariant 3: Passing the same date (or weekend duplicate) twice to RollingOnlineAdaptor or update_online_adaptation_from_scorecard only updates once."""
    adaptor = RollingOnlineAdaptor()
    sample_trades = [
        {
            "ticker": "7011.T",
            "sector": "機械",
            "probability_up": 62.0,
            "expected_return_bps": 45.0,
            "net_return_pct": 2.45,
            "result_status": "WIN",
            "result_code": "TARGET_HIT",
            "actual_open": 2200.0,
            "actual_high": 2265.0,
            "actual_low": 2195.0,
            "actual_close": 2258.0,
            "predicted_waveform": "🚀【GU朝押し ➔ 後場高値引け型】",
        }
    ]

    first_applied = adaptor.update_from_settled_trades("2026-10-30", sample_trades)
    assert first_applied is True
    assert adaptor.get_current_state()["total_updates"] == 1
    weights_after_first = dict(adaptor.get_current_state()["regime_expert_weights"])
    td_after_first = dict(adaptor.get_current_state()["ticker_td_state"])

    # Second call with same date must be blocked by idempotent guard
    second_applied = adaptor.update_from_settled_trades("2026-10-30", sample_trades)
    assert second_applied is False
    assert adaptor.get_current_state()["total_updates"] == 1
    assert adaptor.get_current_state()["regime_expert_weights"] == weights_after_first
    assert adaptor.get_current_state()["ticker_td_state"] == td_after_first

    # Third call via .update() alias with weekend date mapped to market_date="2026-10-30" must also be blocked
    third_applied = adaptor.update("2026-10-31", sample_trades, market_date="2026-10-30")
    assert third_applied is False
    assert adaptor.get_current_state()["total_updates"] == 1

    # Also verify persistent state file idempotency in update_online_adaptation_from_scorecard
    state_file = tmp_path / "online_adaptation_state.json"
    save_adaptation_state(initialize_fresh_state(), state_file=state_file)
    scorecard_friday = {
        "date": "2026-10-30",
        "market_date": "2026-10-30",
        "wins": 1,
        "losses": 0,
        "win_rate_pct": 100.0,
        "total_virtual_pnl_yen": 5400,
        "trades": sample_trades,
    }
    res1 = update_online_adaptation_from_scorecard(scorecard_friday, state_file=state_file)
    assert res1["skipped_duplicate"] is False
    assert res1["updated_state"]["total_updates"] == 1

    res2 = update_online_adaptation_from_scorecard(scorecard_friday, state_file=state_file)
    assert res2["skipped_duplicate"] is True
    assert res2["updated_state"]["total_updates"] == 1

    # Saturday evaluation pointing to Friday's market_date must also be skipped
    scorecard_saturday = dict(scorecard_friday, date="2026-10-31", market_date="2026-10-30")
    res3 = update_online_adaptation_from_scorecard(scorecard_saturday, state_file=state_file)
    assert res3["skipped_duplicate"] is True
    assert res3["updated_state"]["total_updates"] == 1


def test_gap_atr_vpin_zero_yoriten_guard_on_friday() -> None:
    """Improvement 4: Gap-up stocks with VPIN == 0.0 trigger Yoriten trap filter and never qualify as sweet spot."""
    res = compute_gap_atr_entry_adjustment(
        gap_to_atr=0.35,
        continuous_trend_score=0.65,
        vpin_smart_accumulation_score=0.0,
        policy_wave_score=0.50,
        is_friday_or_weekend_pre=True,
    )
    assert res["is_yoriten_gap_trap"] is True
    assert res["is_sweet_spot_gap"] is False
    assert res["score_multiplier"] == 0.70
    assert res["sizing_multiplier"] == 0.65

