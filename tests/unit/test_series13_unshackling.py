from __future__ import annotations

import pandas as pd

from stock_ai.features.chan_quant import compute_conditional_trend_unshackled_chan
from stock_ai.features.microstructure_paleologo import (
    apply_paleologo_cluster_penalty,
    compute_dynamic_sector_concentration_unlock,
)


def test_conditional_trend_unshackling_scalar() -> None:
    # Clean continuous trend should unshackle low raw Hurst (0.42 -> 0.60) and unlock OU/Kalman brake
    res_clean = compute_conditional_trend_unshackled_chan(
        hurst=0.42,
        ou_zscore=2.1,
        kalman_zscore=1.8,
        continuous_trend_score=0.68,
        jump_ratio_5d=0.12,
        vpin_smart_accumulation_score=0.55,
        vpin_signed_ofi_5d=0.25,
    )
    assert res_clean["is_trend_unshackled"] is True
    assert res_clean["is_fake_jump_dump"] is False
    assert res_clean["effective_hurst"] == 0.60
    assert res_clean["effective_ou_zscore"] == 0.0
    assert res_clean["kalman_brake_multiplier"] == 1.0

    # Single-shot fake jump + institutional negative OFI dumping should trigger mean-reversion brake
    res_dump = compute_conditional_trend_unshackled_chan(
        hurst=0.44,
        ou_zscore=2.2,
        kalman_zscore=1.9,
        continuous_trend_score=0.15,
        jump_ratio_5d=0.65,
        vpin_smart_accumulation_score=0.10,
        vpin_signed_ofi_5d=-0.30,
    )
    assert res_dump["is_fake_jump_dump"] is True
    assert res_dump["is_trend_unshackled"] is False
    assert res_dump["effective_hurst"] == 0.44
    assert res_dump["kalman_brake_multiplier"] == 0.75


def test_conditional_trend_unshackling_series() -> None:
    df = pd.DataFrame({
        "hurst": [0.43, 0.41],
        "ou_z": [1.8, 2.0],
        "kz": [1.7, 2.1],
        "cont": [0.65, 0.10],
        "jr": [0.10, 0.70],
        "smart": [0.50, 0.15],
        "ofi": [0.20, -0.25],
    })
    res = compute_conditional_trend_unshackled_chan(
        hurst=df["hurst"],
        ou_zscore=df["ou_z"],
        kalman_zscore=df["kz"],
        continuous_trend_score=df["cont"],
        jump_ratio_5d=df["jr"],
        vpin_smart_accumulation_score=df["smart"],
        vpin_signed_ofi_5d=df["ofi"],
    )
    assert list(res["is_trend_unshackled"]) == [True, False]
    assert list(res["is_fake_jump_dump"]) == [False, True]
    assert res["effective_hurst"].iloc[0] == 0.60
    assert res["effective_hurst"].iloc[1] == 0.41
    assert res["kalman_brake_multiplier"].iloc[0] == 1.0
    assert res["kalman_brake_multiplier"].iloc[1] == 0.75


def test_dynamic_sector_concentration_unlock() -> None:
    # Strong policy wave unlocks max_per_cluster to 2
    unlock_on = compute_dynamic_sector_concentration_unlock(
        policy_wave_score=0.65,
        leadlag_spillover_score=0.10,
        base_gamma_penalty=0.75,
        base_max_per_cluster=1,
    )
    assert unlock_on["is_theme_unlocked"] is True
    assert unlock_on["dynamic_max_per_cluster"] == 2
    assert unlock_on["dynamic_penalty_factor"] >= 0.92

    # Weak theme wave keeps standard max_per_cluster=1
    unlock_off = compute_dynamic_sector_concentration_unlock(
        policy_wave_score=0.45,
        leadlag_spillover_score=0.10,
        base_gamma_penalty=0.75,
        base_max_per_cluster=1,
    )
    assert unlock_off["is_theme_unlocked"] is False
    assert unlock_off["dynamic_max_per_cluster"] == 1
    assert unlock_off["dynamic_penalty_factor"] == 0.75

    # Verify apply_paleologo_cluster_penalty allows top 2 defense stocks without penalty during policy wave
    cands_df = pd.DataFrame({
        "ticker": ["7011.T", "7013.T", "7203.T"],
        "sector": ["機械・防衛", "機械・防衛", "輸送用機器"],
        "composite_score": [100.0, 92.0, 85.0],
        "policy_sector_insider_wave_score": [0.72, 0.72, 0.48],
        "leadlag_spillover_score": [0.80, 0.75, 0.0],
    })
    picks = apply_paleologo_cluster_penalty(
        candidates_df=cands_df,
        score_col="composite_score",
        sector_col="sector",
        ticker_col="ticker",
        penalty_factor=0.75,
        top_k=2,
    )
    assert picks == ["7011.T", "7013.T"]


def test_discord_sniper_unshackled_badge_display() -> None:
    from stock_ai.cloud.discord_bot import build_daytrade_sniper_payloads

    payloads = build_daytrade_sniper_payloads("7011.T", name="三菱重工業")
    assert len(payloads) == 1
    p = payloads[0]
    assert any(b in p["content"] for b in ["🔓 ブレーキ解除:", "⚖️ ブレーキ制御:", "🛑 平均回帰ブレーキ発動:"])
    embed_val = p["embeds"][0]["fields"][0]["value"]
    assert any(b in embed_val for b in ["🔓 ブレーキ解除:", "⚖️ ブレーキ制御:", "🛑 平均回帰ブレーキ発動:"])
    assert "実効H:" in embed_val


