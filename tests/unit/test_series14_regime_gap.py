"""Unit tests for Quant Accuracy Improvement Series #14:
Dynamic Night-Day Regime Allocation & Gap-ATR Entry Optimization.
"""
from __future__ import annotations

from stock_ai.models.intraday import (
    compute_gap_atr_entry_adjustment,
    compute_night_day_regime_allocation,
)


def test_gap_atr_sweet_spot_boost() -> None:
    """Sweet spot (0.15 <= G_ATR <= 1.35 with continuous trend or VPIN accumulation) boosts score, size, and TP barrier."""
    res = compute_gap_atr_entry_adjustment(
        gap_to_atr=0.52,
        continuous_trend_score=0.68,
        vpin_smart_accumulation_score=0.45,
        policy_wave_score=0.60,
    )
    assert res["is_sweet_spot_gap"] is True
    assert res["is_yoriten_gap_trap"] is False
    assert res["is_panic_crash_trap"] is False
    assert res["score_multiplier"] == 1.08
    assert res["sizing_multiplier"] == 1.06
    assert res["tp_barrier_multiplier"] == 1.12
    assert "順張りスイートスポット" in res["gap_atr_badge"]


def test_gap_atr_v_rebound_pullback() -> None:
    """Moderate gap-down (-2.05 <= G_ATR < -0.25) with strong continuous trend is recognized as V-rebound opportunity."""
    res = compute_gap_atr_entry_adjustment(
        gap_to_atr=-0.65,
        continuous_trend_score=0.72,
        vpin_smart_accumulation_score=0.40,
        policy_wave_score=0.58,
    )
    assert res["is_v_rebound_gap"] is True
    assert res["is_sweet_spot_gap"] is False
    assert res["is_panic_crash_trap"] is False
    assert res["score_multiplier"] == 1.05
    assert "強トレンド朝押し・V字窓埋め優位" in res["gap_atr_badge"]


def test_gap_atr_yoriten_excessive_gap_trap() -> None:
    """Excessive gap-up (G_ATR > 1.80 without VPIN smart accumulation) triggers Yoriten fade trap filter."""
    res = compute_gap_atr_entry_adjustment(
        gap_to_atr=2.15,
        continuous_trend_score=0.55,
        vpin_smart_accumulation_score=0.18,
        policy_wave_score=0.45,
    )
    assert res["is_yoriten_gap_trap"] is True
    assert res["is_sweet_spot_gap"] is False
    assert res["score_multiplier"] == 0.70
    assert res["sizing_multiplier"] == 0.65
    assert "過剰GU・寄り天警戒" in res["gap_atr_badge"]


def test_gap_atr_panic_crash_trap() -> None:
    """Market-wide panic gap-down with high systemic fragility and weak trend triggers panic crash filter."""
    res = compute_gap_atr_entry_adjustment(
        gap_to_atr=-1.40,
        continuous_trend_score=0.32,
        vpin_smart_accumulation_score=0.10,
        policy_wave_score=0.40,
        market_gap_atr=-1.95,
        systemic_fragility_score=0.98,
    )
    assert res["is_panic_crash_trap"] is True
    assert res["score_multiplier"] == 0.70
    assert "パニックGD・落ちるナイフ警戒" in res["gap_atr_badge"]


def test_night_day_regime_allocation_balanced_vs_day_focus() -> None:
    """Verify Night-to-Day dynamic regime allocation shifts from Overnight to Intraday under fragility/drawdown."""
    # 1. Normal low-fragility regime with active Intraday Full Alpha pick
    normal = compute_night_day_regime_allocation(
        drawdown_t=0.01,
        systemic_fragility_score=0.22,
        market_volatility_20=0.018,
        has_day_alpha_pick=True,
    )
    assert normal["regime_mode_code"] == "BALANCED_COMPOUND"
    assert normal["night_allocation_weight"] == 1.0
    assert normal["day_allocation_weight"] == 1.12
    assert "昼夜高効率複利モード" in normal["night_day_regime_badge"]

    # 2. High fragility & drawdown regime: Overnight throttles down sharply while Intraday maintains priority
    stressed = compute_night_day_regime_allocation(
        drawdown_t=0.070,
        systemic_fragility_score=0.92,
        market_volatility_20=0.028,
        has_day_alpha_pick=True,
    )
    assert stressed["regime_mode_code"] == "DAY_FOCUS"
    assert stressed["night_allocation_weight"] < 0.60
    assert stressed["day_allocation_weight"] > stressed["night_allocation_weight"]
    assert "日中トレンド集中モード" in stressed["night_day_regime_badge"]


def test_discord_ranking_payload_includes_series14_badges() -> None:
    """Verify Discord ranking payload includes both Night-to-Day Regime Shift badge and Gap-ATR badge."""
    from stock_ai.cloud.discord_bot import build_daytrade_ranking_payloads

    mock_rankings = {
        "date": "2026-10-28",
        "phase": "premarket",
        "phase_title": "☀️ 寄前 09:00〜15:30 当日陽線・上昇予測",
        "systemic_stability_badge": "🛡️ 市場システミック安定度: 正常・分散安定",
        "night_day_regime_badge": "⚖️ 2毛作シフト判定: 昼夜高効率複利モード (Overnight 1.00x × Intraday 1.12x フル稼働)",
        "top_up": [
            {
                "ticker": "7011.T",
                "name": "三菱重工業",
                "probability_up": 62.5,
                "p_calibrated": 64.0,
                "cpcv_grade": "A",
                "expected_return_bps": 85.0,
                "waveform_title": "🔥【寄り付き突破 ➔ 終日トレンド独歩高型】",
                "unshackled_badge": "🔓 ブレーキ解除: 連続トレンド×国策テーマ集中 (Full Alpha Mode)",
                "gap_atr_badge": "📐 寄付きGap/ATR判定: +0.48 ATR [順張りスイートスポット・目標伸長 +12%]",
                "vpin_flow_badge": "🧪 大口フロー: 機関買い吸収",
                "insider_stealth_badge": "🕵️‍♂️ ステルス監視: 大口隠密買い集め検知",
                "best_entry_zone": "2,150円〜2,165円",
                "target_final_price": 2240.0,
                "target_final_pct": 3.8,
            }
        ],
        "top_down": [],
    }
    payloads = build_daytrade_ranking_payloads(rankings_data=mock_rankings)
    assert len(payloads) == 1
    p = payloads[0]
    assert "⚖️ 2毛作シフト判定" in p["content"]
    embed_desc = p["embeds"][0]["description"]
    assert "⚖️ 2毛作シフト判定" in embed_desc
    top_up_val = p["embeds"][0]["fields"][0]["value"]
    assert "📐 寄付きGap/ATR判定" in top_up_val
    assert "🔓 ブレーキ解除" in top_up_val

