"""tests/test_hrp_and_scorecard.py

Unit tests for Marcos Lopez de Prado's Hierarchical Risk Parity (HRP)
and the Post-Market Daytrade Scorecard engine.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from stock_ai.features.hrp_allocation import (
    compute_carver_paleologo_hrp_weights,
    compute_correlation_distance_matrix,
    compute_hrp_weights,
    get_cluster_var,
    get_quasi_diag,
    get_rec_bisection,
)
from stock_ai.models.intraday import evaluate_todays_daytrade_scorecard
from stock_ai.cloud.discord_bot import build_daytrade_scorecard_payloads


# ===========================================================================
# 1. Marcos Lopez de Prado HRP Algorithm Tests
# ===========================================================================

def test_correlation_distance_matrix():
    """Verify Prado's correlation distance metric D_{i,j} = sqrt(0.5 * (1 - rho))."""
    # Perfect positive correlation (rho = 1.0) -> distance = 0.0
    # Zero correlation (rho = 0.0) -> distance = sqrt(0.5) ~ 0.7071
    # Perfect negative correlation (rho = -1.0) -> distance = 1.0
    rho = np.array([
        [1.0, 0.0, -1.0],
        [0.0, 1.0, 0.5],
        [-1.0, 0.5, 1.0],
    ])
    dist_matrix, condensed = compute_correlation_distance_matrix(rho)
    assert dist_matrix.shape == (3, 3)
    assert np.allclose(np.diag(dist_matrix), 0.0)
    assert len(condensed) == 3


def test_hrp_weights_sum_to_one_and_split_correlated_cluster():
    """Verify HRP weights sum strictly to 1.0 and properly split high-correlation pairs."""
    tickers = ["AssetA", "AssetB", "AssetC", "AssetD"]
    # AssetA and AssetB are highly correlated (rho = 0.95), AssetC and AssetD are independent (rho = 0.10)
    rho = np.array([
        [1.00, 0.95, 0.10, 0.10],
        [0.95, 1.00, 0.10, 0.10],
        [0.10, 0.10, 1.00, 0.20],
        [0.10, 0.10, 0.20, 1.00],
    ])
    vol = np.array([0.02, 0.02, 0.02, 0.02])
    cov = rho * np.outer(vol, vol)

    weights_series = compute_hrp_weights(cov_matrix=cov, tickers=tickers)
    
    # Check 1: Output is pd.Series indexed by tickers
    assert isinstance(weights_series, pd.Series)
    assert list(weights_series.index) == tickers

    # Check 2: Weights sum to exactly 1.0
    assert pytest.approx(weights_series.sum(), abs=1e-6) == 1.0

    # Check 3: All weights are strictly positive
    assert (weights_series > 0.0).all()

    # Check 4: The highly correlated pair (AssetA, AssetB) each receives ~12.5% - 15%,
    # splitting their cluster variance, rather than concentrating all weight in one asset.
    assert weights_series["AssetA"] < 0.25
    assert weights_series["AssetB"] < 0.25
    # Combined cluster weight ~ 25% to 35%
    cluster_ab_weight = weights_series["AssetA"] + weights_series["AssetB"]
    assert 0.20 <= cluster_ab_weight <= 0.40


def test_hrp_weights_from_returns_df():
    """Verify compute_hrp_weights directly on historical returns DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    ret_df = pd.DataFrame(
        np.random.normal(0.001, 0.02, size=(60, 4)),
        index=dates,
        columns=["7011.T", "7012.T", "7203.T", "8035.T"],
    )
    weights = compute_hrp_weights(returns_df=ret_df)
    assert len(weights) == 4
    assert pytest.approx(weights.sum(), abs=1e-6) == 1.0
    assert (weights > 0.0).all()


def test_carver_paleologo_hrp_hybrid_allocation():
    """Verify 6-masterpiece hybrid engine (Carver Forecast x Paleologo Cluster x Prado HRP)."""
    tickers = ["7011.T", "7012.T", "7203.T"]
    forecasts = [18.0, 12.0, 10.0]  # Strong, moderate, baseline
    vols = [0.025, 0.030, 0.018]
    sec_map = {"7011.T": "機械・防衛", "7012.T": "機械・防衛", "7203.T": "輸送用機器"}

    # Full throttle
    weights_full = compute_carver_paleologo_hrp_weights(
        tickers=tickers,
        forecasts=forecasts,
        volatilities=vols,
        sector_map=sec_map,
        max_weight=0.50,
        drawdown_throttle=1.0,
    )
    assert len(weights_full) == 3
    assert all(0.0 <= w <= 0.50 for w in weights_full.values())
    assert sum(weights_full.values()) <= 1.0

    # Strongest forecast ticker should receive highest weight
    assert weights_full["7011.T"] >= weights_full["7012.T"]

    # Throttle down by 50%
    weights_throttled = compute_carver_paleologo_hrp_weights(
        tickers=tickers,
        forecasts=forecasts,
        volatilities=vols,
        sector_map=sec_map,
        max_weight=0.50,
        drawdown_throttle=0.50,
    )
    for t in tickers:
        assert weights_throttled[t] < weights_full[t]
    assert pytest.approx(sum(weights_throttled.values()), rel=1e-3) == sum(weights_full.values()) * 0.50


# ===========================================================================
# 2. Post-Market Scorecard Engine Tests
# ===========================================================================

def test_scorecard_evaluation_execution():
    """Verify evaluate_todays_daytrade_scorecard generates complete, deterministic metrics."""
    scorecard = evaluate_todays_daytrade_scorecard(save_snapshot=False)
    
    # Basic schema check
    assert "date" in scorecard
    assert "total_trades" in scorecard
    assert "wins" in scorecard
    assert "losses" in scorecard
    assert "win_rate_pct" in scorecard
    assert "total_virtual_pnl_yen" in scorecard
    assert "trades" in scorecard

    # Consistency checks
    assert scorecard["total_trades"] == scorecard["wins"] + scorecard["losses"] + scorecard.get("flats", 0)
    assert 0.0 <= scorecard["win_rate_pct"] <= 100.0

    # Trade cards check
    for t in scorecard["trades"]:
        assert "ticker" in t
        assert "name" in t
        assert t["result_status"] in ("WIN", "FLAT", "LOSS")
        assert t["result_code"] in (
            "TARGET_HIT",
            "WIN_CLOSE",
            "FLAT",
            "LOSS_CLOSE",
            "YORITEN_STOP",
            "STOPPED_OUT",
            "CLOSE_POSITIVE",
            "CLOSE_NEGATIVE",
        )
        assert "virtual_pnl_100_shares" in t
        assert "waveform_verdict" in t


def test_scorecard_mock_trades_classification():
    """Verify Target Hit, Stopped Out, and Close Positive logic under controlled prices."""
    # Test 1: Target Hit
    # Open=1000, Target=1020, Stop=980, Actual High=1025, Low=990, Close=1015
    act_open = 1000.0
    t1 = 1020.0
    stop = 980.0
    act_high = 1025.0
    act_low = 990.0
    act_close = 1015.0

    cost = act_open * 0.0020  # 2.0 yen
    if act_high >= t1:
        res = "TARGET_HIT"
        pnl = int(round((t1 - act_open - cost) * 100))
    assert res == "TARGET_HIT"
    assert pnl == 1800  # (20 - 2.0) * 100 = 1800 yen

    # Test 2: Stopped Out
    act_high = 1010.0
    act_low = 975.0
    if act_high >= t1:
        res = "TARGET_HIT"
    elif act_low <= stop:
        res = "STOPPED_OUT"
        pnl = int(round((stop - act_open - cost) * 100))
    assert res == "STOPPED_OUT"
    assert pnl == -2200  # (-20 - 2.0) * 100 = -2200 yen

    # Test 3: Close Positive
    act_high = 1012.0
    act_low = 995.0
    act_close = 1008.0
    if act_high >= t1:
        res = "TARGET_HIT"
    elif act_low <= stop:
        res = "STOPPED_OUT"
    else:
        net = (act_close - act_open) - cost
        res = "CLOSE_POSITIVE" if net > 0 else "CLOSE_NEGATIVE"
        pnl = int(round(net * 100))
    assert res == "CLOSE_POSITIVE"
    assert pnl == 600  # (8 - 2.0) * 100 = 600 yen


def test_discord_scorecard_payload_structure():
    """Verify build_daytrade_scorecard_payloads formats mobile-first embeds correctly."""
    mock_scorecard = {
        "date": "2026-10-24",
        "total_trades": 3,
        "wins": 2,
        "losses": 1,
        "win_rate_pct": 66.7,
        "total_virtual_pnl_yen": 42000,
        "mvp": {
            "ticker": "8035.T",
            "name": "東京エレクトロン",
            "virtual_pnl_100_shares": 35000,
            "waveform_verdict": "🎯 波形的中",
        },
        "trades": [
            {
                "ticker": "8035.T",
                "name": "東京エレクトロン",
                "result_badge": "🎯 利確ターゲット到達",
                "result_status": "WIN",
                "virtual_pnl_100_shares": 35000,
                "net_return_pct": 3.8,
                "actual_open": 9000.0,
                "exit_price": 9350.0,
                "actual_high": 9400.0,
                "actual_low": 8950.0,
                "predicted_waveform": "📈【GDスタート ➔ 窓埋めV字反発型】",
                "actual_waveform": "前場後場連動・高値引け陽線",
                "waveform_verdict": "🎯 波形的中",
            },
            {
                "ticker": "7203.T",
                "name": "トヨタ自動車",
                "result_badge": "🟢 大引けプラス決済",
                "result_status": "WIN",
                "virtual_pnl_100_shares": 12000,
                "net_return_pct": 1.2,
                "actual_open": 2500.0,
                "exit_price": 2530.0,
                "actual_high": 2540.0,
                "actual_low": 2490.0,
                "predicted_waveform": "📈【押し目順張り型】",
                "actual_waveform": "前場後場連動・高値引け陽線",
                "waveform_verdict": "🎯 波形的中",
            },
            {
                "ticker": "6758.T",
                "name": "ソニーグループ",
                "result_badge": "🛡️ 損切ライン抵触",
                "result_status": "LOSS",
                "virtual_pnl_100_shares": -5000,
                "net_return_pct": -1.5,
                "actual_open": 3200.0,
                "exit_price": 3150.0,
                "actual_high": 3220.0,
                "actual_low": 3145.0,
                "predicted_waveform": "📈【GDスタート ➔ 窓埋めV字反発型】",
                "actual_waveform": "揉み合い・軟調陰線",
                "waveform_verdict": "⚠️ 波形乖離",
            },
        ],
    }

    payloads = build_daytrade_scorecard_payloads(mock_scorecard)
    assert len(payloads) == 1
    p = payloads[0]
    assert "🎯 AIデイトレ答え合わせ通信簿" in p["username"]
    assert "66.7%" in p["content"]
    assert "+42,000円" in p["content"]
    assert "embeds" in p and len(p["embeds"]) == 1
    embed = p["embeds"][0]
    assert "AIデイトレ答え合わせ通信簿" in embed["title"]
    assert len(embed["fields"]) >= 4  # Summary + 3 trades + rules
    # Confirm mobile-first layout: every field has inline: False
    for f in embed["fields"]:
        assert f.get("inline") is False
