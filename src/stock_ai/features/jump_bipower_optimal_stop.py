"""src/stock_ai/features/jump_bipower_optimal_stop.py

Barndorff-Nielsen & Shephard Realized Bipower Variation (BV) / Jump Decomposition
and Optimal Stopping Theory (Wald-Bellman Dynamic Programming) Dynamic Exit Barriers.

References:
- Ole E. Barndorff-Nielsen & Neil Shephard (2004, 2006):
  "Power and Bipower Variation with Stochastic Volatility and Jumps",
  Journal of Financial Econometrics.
- Abraham Wald (1947): "Sequential Analysis"
- Richard Bellman (1957): "Dynamic Programming"
- Marcos López de Prado (2018): "Advances in Financial Machine Learning", Chapter 3 (Optimal Stopping).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# BNS normalization constant: pi / 2
BNS_MU1_INV_SQ = np.pi / 2.0  # ~1.5707963267948966


def compute_bipower_variation(returns: np.ndarray | pd.Series | list[float]) -> tuple[float, float, float, float]:
    """Compute Realized Variance (RV), Bipower Variation (BV), Jump Component (JP), and Jump Ratio.

    Args:
        returns: 1D array of asset returns over a defined horizon (e.g., M intraday 5m returns or M daily returns).

    Returns:
        tuple of (RV, BV, JP, JumpRatio)
        - RV: Realized Variance = sum(r_j^2)
        - BV: Bipower Variation = (pi/2) * (M / (M-1)) * sum(|r_j| * |r_{j-1}|)
        - JP: Jump Component = max(0, RV - BV)
        - JumpRatio: JP / (RV + eps) in [0.0, 1.0]
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    M = len(arr)
    if M < 2:
        return 0.0, 0.0, 0.0, 0.0

    rv = float(np.sum(arr**2))
    if rv <= 1e-12:
        return 0.0, 0.0, 0.0, 0.0

    abs_arr = np.abs(arr)
    # BNS asymptotic unbiased scaling factor: (M / (M - 1)) * (pi / 2)
    scale_factor = BNS_MU1_INV_SQ * (float(M) / float(M - 1))
    bv = float(scale_factor * np.sum(abs_arr[1:] * abs_arr[:-1]))

    jp = max(0.0, rv - bv)
    jump_ratio = float(np.clip(jp / (rv + 1e-10), 0.0, 1.0))

    return rv, bv, jp, jump_ratio


def compute_rolling_bipower_jump(
    returns: pd.Series,
    window: int = 5,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute rolling RV, BV, and JumpRatio over a moving window with zero future leakage.

    Args:
        returns: pd.Series of asset returns indexed by date/time.
        window: rolling window length (default: 5 bars).

    Returns:
        (rv_series, bv_series, jump_ratio_series)
    """
    n = len(returns)
    if n < 2:
        empty = pd.Series(0.0, index=returns.index)
        return empty, empty, empty

    values = returns.fillna(0.0).to_numpy(dtype=float)
    rv_arr = np.zeros(n, dtype=float)
    bv_arr = np.zeros(n, dtype=float)
    jr_arr = np.zeros(n, dtype=float)

    for i in range(1, n):
        start_idx = max(0, i - window + 1)
        sub = values[start_idx : i + 1]
        m = len(sub)
        if m < 2:
            continue
        rv_val = np.sum(sub**2)
        abs_sub = np.abs(sub)
        scale = BNS_MU1_INV_SQ * (float(m) / float(m - 1))
        bv_val = scale * np.sum(abs_sub[1:] * abs_sub[:-1])
        jp_val = max(0.0, rv_val - bv_val)
        ratio_val = np.clip(jp_val / (rv_val + 1e-10), 0.0, 1.0)

        rv_arr[i] = rv_val
        bv_arr[i] = bv_val
        jr_arr[i] = ratio_val

    idx = returns.index
    return pd.Series(rv_arr, index=idx), pd.Series(bv_arr, index=idx), pd.Series(jr_arr, index=idx)


def compute_bipower_jump_decomposition(data: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute full Barndorff-Nielsen & Shephard jump features from OHLCV DataFrame.

    Generates:
        - jump_ratio_5d: 5-day rolling jump proportion in [0, 1]
        - jump_ratio_20d: 20-day rolling jump proportion in [0, 1]
        - continuous_trend_score: smooth continuous upward trend conviction in [0, 1]
        - fake_spike_exhaustion_flag: boolean indicator of sudden discontinuous jump exhaustion
    """
    idx = data.index
    if data.empty or len(data) < 2:
        return {
            "jump_ratio_5d": pd.Series(0.0, index=idx),
            "jump_ratio_20d": pd.Series(0.0, index=idx),
            "continuous_trend_score": pd.Series(0.5, index=idx),
            "fake_spike_exhaustion_flag": pd.Series(False, index=idx),
        }

    close = data["close"]
    # Log or simple 1-day returns
    if "return_1d" in data.columns:
        ret_1d = data["return_1d"].fillna(0.0)
    else:
        ret_1d = close.pct_change().fillna(0.0)

    # 1. Rolling 5-day Bipower Jump Ratio
    _, _, jr_5d = compute_rolling_bipower_jump(ret_1d, window=5)

    # 2. Rolling 20-day Bipower Jump Ratio
    _, _, jr_20d = compute_rolling_bipower_jump(ret_1d, window=20)

    # 3. Continuous Trend Score:
    # High when jump ratio is low (variance is purely continuous) and 20d momentum is positive
    ret_20d = ret_1d.rolling(20, min_periods=1).sum()
    vol_20d = ret_1d.rolling(20, min_periods=2).std().fillna(0.015)
    sharpe_proxy = np.tanh(ret_20d / (vol_20d * np.sqrt(20.0) + 1e-6))
    dir_factor = (sharpe_proxy + 1.0) / 2.0  # in [0, 1]

    # Continuous component dominance: (1 - jump_ratio_5d)
    cont_score = (1.0 - jr_5d) * (0.35 + 0.65 * dir_factor)
    cont_score = cont_score.clip(0.0, 1.0)

    # 4. Fake Spike Exhaustion Flag:
    # Triggered when a stock jumps violently without continuous accumulation, hitting overbought/shadow exhaustion
    # Criteria:
    # a. High jump ratio (jr_5d >= 0.45 or jr_20d >= 0.45)
    # b. Large recent price spike (1d return >= 2.5% or 5d return >= 5.0% or overnight gap >= 2.0%)
    # c. Exhaustion evidence: RSI >= 68 or upper shadow ratio >= 0.35 or informed distribution >= 0.45
    ret_5d = ret_1d.rolling(5, min_periods=1).sum()
    spike_ret = (ret_1d >= 0.025) | (ret_5d >= 0.050)
    if "overnight_gap" in data.columns:
        spike_ret = spike_ret | (data["overnight_gap"].fillna(0.0) >= 0.020)

    high_jump = (jr_5d >= 0.45) | (jr_20d >= 0.45)

    rsi_series = data["rsi_14"] if "rsi_14" in data.columns else pd.Series(50.0, index=idx)
    high_rsi = rsi_series.fillna(50.0) >= 68.0

    high_shadow = pd.Series(False, index=idx)
    if "high" in data.columns and "low" in data.columns:
        h = data["high"]
        l = data["low"]
        o = data["open"] if "open" in data.columns else close
        rng = (h - l).replace(0.0, np.nan)
        upper_shadow = (h - np.maximum(o, close)) / (rng + 1e-8)
        high_shadow = upper_shadow.fillna(0.0) >= 0.35
    elif "exhaustion_upper_shadow_ratio" in data.columns:
        high_shadow = data["exhaustion_upper_shadow_ratio"].fillna(0.0) >= 0.35

    high_dist = pd.Series(False, index=idx)
    if "vpin_informed_distribution_score" in data.columns:
        high_dist = data["vpin_informed_distribution_score"].fillna(0.0) >= 0.45

    exhaustion_evidence = high_rsi | high_shadow | high_dist
    fake_spike_flag = spike_ret & high_jump & exhaustion_evidence

    return {
        "jump_ratio_5d": jr_5d,
        "jump_ratio_20d": jr_20d,
        "continuous_trend_score": cont_score,
        "fake_spike_exhaustion_flag": fake_spike_flag,
    }


def compute_optimal_stopping_barriers(
    drift: float,
    volatility: float,
    hurst: float = 0.50,
    jump_ratio: float = 0.20,
    continuous_score: float = 0.50,
    vpin_toxicity: float = 0.30,
    is_fake_spike: bool = False,
    base_tp: float = 0.030,
    base_sl: float = 0.015,
) -> dict[str, Any]:
    """Calculate Wald-Bellman Dynamic Optimal Stopping Boundaries (take-profit and stop-loss).

    Adapts exit thresholds dynamically based on:
    - Persistence (Hurst H > 0.5 expands take-profit barrier; H < 0.45 contracts it)
    - Volatility scaling (normalizes against median daily volatility 2.0%)
    - Discontinuous Jump vs Continuous Trend dominance
    - VPIN order flow toxicity (tightens stop-loss on toxic selling)
    - Fake spike exhaustion (forces early take-profit lock-in before morning reversal)

    Returns:
        dict containing:
        - dynamic_take_profit: float (e.g., +0.042)
        - dynamic_stop_loss: float (e.g., -0.012)
        - dynamic_take_profit_pct: float (e.g., +4.2)
        - dynamic_stop_loss_pct: float (e.g., -1.2)
        - jump_regime_badge: formatted string for Discord and UI
    """
    vol = max(0.005, min(0.080, float(volatility)))
    vol_scale = np.clip(vol / 0.020, 0.70, 1.50)
    h_clamped = np.clip(float(hurst), 0.30, 0.75)
    jr_clamped = np.clip(float(jump_ratio), 0.0, 1.0)
    cont_clamped = np.clip(float(continuous_score), 0.0, 1.0)
    tox_clamped = np.clip(float(vpin_toxicity), 0.0, 1.0)

    if is_fake_spike or jr_clamped >= 0.55:
        # High Jump / Fake Spike Exhaustion Regime:
        # Target must be taken early before the discontinuous spike mean-reverts!
        tp_mult = max(0.50, 0.75 - (jr_clamped - 0.50) * 0.40)
        dynamic_tp = np.clip(base_tp * tp_mult * vol_scale, 0.012, 0.025)

        # Tighten stop-loss to avoid absorbing the post-spike dump
        sl_mult = max(0.70, 0.85 - tox_clamped * 0.30)
        dynamic_sl = -np.clip(base_sl * sl_mult * vol_scale, 0.008, 0.015)

        badge = f"⚠️ 変動構造: 突発ジャンプ過熱 (Jump Ratio: {jr_clamped:.0%} [リバランス警戒]) ➔ 早期利確推奨"

    elif h_clamped >= 0.52 and cont_clamped >= 0.50:
        # Smooth Continuous Trend Dominance:
        # Strong persistence allows expanding take-profit barrier to capture major trend runs!
        trend_expansion = 1.0 + 1.3 * (h_clamped - 0.50) + 0.5 * (cont_clamped - 0.50)
        dynamic_tp = np.clip(base_tp * trend_expansion * vol_scale, 0.035, 0.065)

        # Standard healthy trailing room for stop loss, tightened if toxicity appears
        sl_room = 1.0 + 0.20 * (vol_scale - 1.0) - 0.30 * max(0.0, tox_clamped - 0.40)
        dynamic_sl = -np.clip(base_sl * sl_room, 0.012, 0.025)

        badge = (
            f"⚡ 変動構造: 連続トレンド優位 (Jump Ratio: {jr_clamped:.0%} [クリーン]) ➔ "
            f"動的利確: +{dynamic_tp * 100:.1f}% / 動的損切: {dynamic_sl * 100:.1f}%"
        )

    else:
        # Balanced / Neutral Regime:
        # Modest adjustment by volatility and toxicity
        dynamic_tp = np.clip(base_tp * vol_scale, 0.020, 0.040)
        sl_adj = 1.0 + 0.40 * max(0.0, tox_clamped - 0.35)
        dynamic_sl = -np.clip(base_sl * vol_scale * sl_adj, 0.010, 0.022)

        badge = (
            f"⚡ 変動構造: 通常均衡 (Jump Ratio: {jr_clamped:.0%}) ➔ "
            f"動的利確: +{dynamic_tp * 100:.1f}% / 動的損切: {dynamic_sl * 100:.1f}%"
        )

    return {
        "dynamic_take_profit": float(round(dynamic_tp, 4)),
        "dynamic_stop_loss": float(round(dynamic_sl, 4)),
        "dynamic_take_profit_pct": float(round(dynamic_tp * 100.0, 1)),
        "dynamic_stop_loss_pct": float(round(dynamic_sl * 100.0, 1)),
        "jump_regime_badge": badge,
    }


def add_jump_bipower_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add Barndorff-Nielsen & Shephard Bipower Jump and Optimal Stopping features to DataFrame."""
    if data.empty:
        return data

    res = data.copy()
    if "ticker" in res.columns:
        grouped = res.groupby("ticker", group_keys=False)
        all_jr5 = []
        all_jr20 = []
        all_cont = []
        all_fake = []

        for _, grp in grouped:
            decomp = compute_bipower_jump_decomposition(grp)
            all_jr5.append(decomp["jump_ratio_5d"])
            all_jr20.append(decomp["jump_ratio_20d"])
            all_cont.append(decomp["continuous_trend_score"])
            all_fake.append(decomp["fake_spike_exhaustion_flag"])

        res["jump_ratio_5d"] = pd.concat(all_jr5).sort_index() if all_jr5 else 0.0
        res["jump_ratio_20d"] = pd.concat(all_jr20).sort_index() if all_jr20 else 0.0
        res["continuous_trend_score"] = pd.concat(all_cont).sort_index() if all_cont else 0.5
        res["fake_spike_exhaustion_flag"] = pd.concat(all_fake).sort_index() if all_fake else False
    else:
        decomp = compute_bipower_jump_decomposition(res)
        res["jump_ratio_5d"] = decomp["jump_ratio_5d"]
        res["jump_ratio_20d"] = decomp["jump_ratio_20d"]
        res["continuous_trend_score"] = decomp["continuous_trend_score"]
        res["fake_spike_exhaustion_flag"] = decomp["fake_spike_exhaustion_flag"]

    return res
