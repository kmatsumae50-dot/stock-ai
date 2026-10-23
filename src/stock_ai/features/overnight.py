from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

SEMICON_AI_CABLE_TICKERS = {
    "8035.T", "6857.T", "6723.T", "6146.T", "6920.T", "6526.T",
    "5803.T", "5801.T", "5802.T", "6762.T", "6501.T",
}
BANKING_FINANCE_TICKERS = {
    "8306.T", "8316.T", "8411.T", "8766.T", "8750.T", "8591.T",
}


def is_semicon_cable_ticker(ticker: str | None, sector: str | None = None) -> bool:
    if ticker and ticker in SEMICON_AI_CABLE_TICKERS:
        return True
    if sector and any(s in sector for s in ("Semiconductors", "Non-Ferrous Metals", "半導体", "電線")):
        return True
    return False


def is_banking_finance_ticker(ticker: str | None, sector: str | None = None) -> bool:
    if ticker and ticker in BANKING_FINANCE_TICKERS:
        return True
    if sector and any(s in sector for s in ("Banking", "Insurance", "Finance", "銀行", "保険", "金融")):
        return True
    return False


OVERNIGHT_CURATED_FEATURES = [
    # 1. Japanese Price Action & Closing Demand (大引け引け味)
    "close_location_value",
    "intraday_return",
    "return_1d",
    "high_low_range",
    "close_position",
    # 2. Moving Average & Momentum (ボラティリティ調整後モメンタム)
    "close_vs_ma_5",
    "close_vs_ma_25",
    "close_vs_ma_75",
    "vol_adjusted_momentum_1d",
    "vol_adjusted_momentum_5d",
    # 3. Volume Spikes & Order Flow (出来高急増スパイク)
    "volume_change_1d",
    "volume_surge_5d",
    "volume_surge_20d",
    # 4. Technical Oscillators & Reversals (RSI売られすぎ・MACD)
    "rsi_14",
    "rsi_oversold_flag",
    "rsi_overbought_flag",
    "macd",
    "macd_signal",
    "macd_hist",
    # 5. Volatility & Bands
    "volatility_20",
    "bb_percent_b",
    "bb_squeeze",
    "intraday_range_ratio",
    # 6. Sector & Domestic Context (セクター相対強度・東証指数)
    "sector_relative_return",
    "sector_relative_strength",
    "sector_mean_return",
    "sector_volatility",
    "sector_trend_score",
    "n225_return_1d",
    "n225_trend_direction",
    "topix_return_1d",
    "topix_trend_direction",
    # 7. Clean Shift-1 Overseas Series (1営業日前確定値・リークゼロ)
    "usdjpy_return_1d",
    "usdjpy_close_vs_ma_25",
    "usdjpy_trend_direction",
    "tesla_return_1d",
    "tesla_trend_direction",
    "byd_return_1d",
    # 8. 4大マクロ・外部市場指標 (SOX, NVDA, TNX, VIX: shift-1確定値)
    "sox_return_1d",
    "nvda_return_1d",
    "tnx_return_1d",
    "vix_return_1d",
    # 9. セクター別マクロ感応度・相対特徴量 (Sector-Macro Sensitivity)
    "semicon_sox_beta_return",
    "semicon_nvda_beta_return",
    "macro_sox_relative_return",
    "macro_nvda_relative_return",
    "finance_tnx_sensitivity",
    "macro_tnx_rate_shock",
    "vix_spike_flag",
    "vix_stress_score",
    # 10. AFML 固定窓・分数次差分 (Fractional Differentiation: d=0.4 Memory Features)
    "afml_fracdiff_close_d04",
    "afml_fracdiff_vol_d04",
    "afml_fracdiff_zscore_20",
]


def add_overnight_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add Japanese market micro-structure, demand, AFML memory, and curated overnight features."""
    if data.empty:
        return data

    from stock_ai.features.afml import add_afml_features
    result = add_afml_features(data.copy(), d=0.4)
    grouped = result.groupby("ticker", group_keys=False) if "ticker" in result.columns else None

    close = result["close"]
    high = result["high"] if "high" in result.columns else close
    low = result["low"] if "low" in result.columns else close
    volume = result["volume"] if "volume" in result.columns else pd.Series(1.0, index=result.index)
    vol_20 = result["volatility_20"] if "volatility_20" in result.columns else pd.Series(0.015, index=result.index)

    # 1. Close Location Value (CLV: 大引け引け味の強さ)
    # +1.0 = closed at day's high (intense buying into close), -1.0 = closed at day's low
    hl_range = (high - low).replace(0.0, np.nan)
    clv = (2.0 * close - high - low) / (hl_range + 1e-8)
    result["close_location_value"] = clv.clip(-1.0, 1.0).fillna(0.0)

    # 2. Volume surges (5d and 20d moving averages)
    if grouped is not None:
        vol_ma_5 = grouped["volume"].rolling(5, min_periods=2).mean().reset_index(level=0, drop=True)
        vol_ma_20 = grouped["volume"].rolling(20, min_periods=5).mean().reset_index(level=0, drop=True)
        ret_5 = grouped["close"].pct_change(5).reset_index(level=0, drop=True)
    else:
        vol_ma_5 = volume.rolling(5, min_periods=2).mean()
        vol_ma_20 = volume.rolling(20, min_periods=5).mean()
        ret_5 = close.pct_change(5)

    result["volume_surge_5d"] = (volume / (vol_ma_5 + 1e-8)).clip(0.0, 10.0).fillna(1.0)
    result["volume_surge_20d"] = (volume / (vol_ma_20 + 1e-8)).clip(0.0, 10.0).fillna(1.0)

    # 3. Volatility-adjusted momentum
    ret_1d = result["return_1d"] if "return_1d" in result.columns else close.pct_change()
    result["vol_adjusted_momentum_1d"] = (ret_1d / (vol_20 + 1e-8)).clip(-5.0, 5.0).fillna(0.0)
    result["vol_adjusted_momentum_5d"] = (ret_5 / (vol_20 * np.sqrt(5) + 1e-8)).clip(-5.0, 5.0).fillna(0.0)

    # 4. RSI Overbought/Oversold flags
    if "rsi_14" in result.columns:
        result["rsi_oversold_flag"] = (result["rsi_14"] < 30.0).astype(float)
        result["rsi_overbought_flag"] = (result["rsi_14"] > 70.0).astype(float)
    else:
        result["rsi_oversold_flag"] = 0.0
        result["rsi_overbought_flag"] = 0.0

    # 5. Bollinger Band Squeeze & Intraday Range Ratio
    if "bb_upper" in result.columns and "bb_lower" in result.columns and "bb_middle" in result.columns:
        result["bb_squeeze"] = ((result["bb_upper"] - result["bb_lower"]) / (result["bb_middle"] + 1e-8)).clip(0.0, 2.0).fillna(0.05)
    else:
        result["bb_squeeze"] = 0.05

    if "high_low_range" in result.columns:
        result["intraday_range_ratio"] = (result["high_low_range"] / (vol_20 + 1e-8)).clip(0.0, 10.0).fillna(1.0)
    else:
        result["intraday_range_ratio"] = 1.0

    # 6. Sector relative strength & returns
    if "sector_mean_return" in result.columns:
        result["sector_relative_return"] = ret_1d - result["sector_mean_return"]
    else:
        result["sector_relative_return"] = 0.0

    if "close_vs_ma_25" in result.columns and "sector_strength" in result.columns:
        result["sector_relative_strength"] = result["close_vs_ma_25"] - result["sector_strength"]
    else:
        result["sector_relative_strength"] = 0.0

    # 7. 4大マクロ指標とセクター別感応度特徴量 (SOX, NVDA, TNX, VIX: shift-1確定値)
    t_val = str(result["ticker"].iloc[0]) if "ticker" in result.columns and len(result) > 0 else ""
    s_val = str(result["sector"].iloc[0]) if "sector" in result.columns and len(result) > 0 else ""
    is_semi = is_semicon_cable_ticker(t_val, s_val)
    is_fin = is_banking_finance_ticker(t_val, s_val)

    sox_ret = pd.to_numeric(result["sox_return_1d"], errors="coerce").fillna(0.0) if "sox_return_1d" in result.columns else pd.Series(0.0, index=result.index)
    nvda_ret = pd.to_numeric(result["nvda_return_1d"], errors="coerce").fillna(0.0) if "nvda_return_1d" in result.columns else pd.Series(0.0, index=result.index)
    tnx_ret = pd.to_numeric(result["tnx_return_1d"], errors="coerce").fillna(0.0) if "tnx_return_1d" in result.columns else pd.Series(0.0, index=result.index)
    vix_ret = pd.to_numeric(result["vix_return_1d"], errors="coerce").fillna(0.0) if "vix_return_1d" in result.columns else pd.Series(0.0, index=result.index)
    vix_cl = pd.to_numeric(result["vix_close"], errors="coerce").fillna(20.0) if "vix_close" in result.columns else pd.Series(20.0, index=result.index)
    n225_ret = pd.to_numeric(result["n225_return_1d"], errors="coerce").fillna(0.0) if "n225_return_1d" in result.columns else pd.Series(0.0, index=result.index)

    if "sox_return_1d" not in result.columns:
        result["sox_return_1d"] = sox_ret
    if "nvda_return_1d" not in result.columns:
        result["nvda_return_1d"] = nvda_ret
    if "tnx_return_1d" not in result.columns:
        result["tnx_return_1d"] = tnx_ret
    if "vix_return_1d" not in result.columns:
        result["vix_return_1d"] = vix_ret

    # 半導体セクター感応度（半導体・電線ならSOX/NVDAリターン、他なら0.0）
    result["semicon_sox_beta_return"] = (sox_ret if is_semi else pd.Series(0.0, index=result.index)).clip(-0.15, 0.15)
    result["semicon_nvda_beta_return"] = (nvda_ret if is_semi else pd.Series(0.0, index=result.index)).clip(-0.20, 0.20)
    result["macro_sox_relative_return"] = (sox_ret - n225_ret).clip(-0.10, 0.10)
    result["macro_nvda_relative_return"] = (nvda_ret - n225_ret).clip(-0.15, 0.15)

    # 銀行・金融セクター感応度（金利変動に対する感応度）
    result["finance_tnx_sensitivity"] = (tnx_ret if is_fin else pd.Series(0.0, index=result.index)).clip(-0.15, 0.15)
    result["macro_tnx_rate_shock"] = tnx_ret.clip(-0.15, 0.15)

    # VIXリスクオフ・警戒フラグ（VIX急騰または25以上の高ボラ帯）
    result["vix_spike_flag"] = ((vix_ret > 0.08) | (vix_cl > 25.0)).astype(float)
    result["vix_stress_score"] = (vix_ret / 0.10).clip(-3.0, 3.0)

    return result


CROSS_SECTIONAL_FEATURES = [
    "cs_rank_intraday_return",
    "cs_rank_close_location",
    "cs_rank_volume_surge",
    "cs_rank_rsi_14",
    "cs_zscore_return_1d",
    "cs_zscore_return_5d",
    "sector_relative_return_1d",
    "cs_rank_macro_sensitivity",
]


def add_cross_sectional_overnight_features(pooled_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily cross-sectional (relative ranking & z-score) features across universe.

    Strictly uses information determined at 15:30 JST close for each date (zero lookahead).
    """
    if pooled_df.empty or "date" not in pooled_df.columns:
        return pooled_df

    result = pooled_df.copy()
    result["date"] = pd.to_datetime(result["date"])

    # Ensure prerequisite columns exist
    if "intraday_return" not in result.columns and "open" in result.columns and "close" in result.columns:
        result["intraday_return"] = (result["close"] - result["open"]) / (result["open"] + 1e-8)
    elif "intraday_return" not in result.columns and "return_1d" in result.columns:
        result["intraday_return"] = result["return_1d"]
    elif "intraday_return" not in result.columns:
        result["intraday_return"] = 0.0

    if "close_location_value" not in result.columns and "high" in result.columns and "low" in result.columns and "close" in result.columns:
        hl = (result["high"] - result["low"]).replace(0.0, np.nan)
        result["close_location_value"] = ((2.0 * result["close"] - result["high"] - result["low"]) / (hl + 1e-8)).clip(-1.0, 1.0).fillna(0.0)
    elif "close_location_value" not in result.columns:
        result["close_location_value"] = 0.0

    if "return_1d" not in result.columns and "close" in result.columns:
        if "ticker" in result.columns:
            result["return_1d"] = result.groupby("ticker")["close"].pct_change().fillna(0.0)
        else:
            result["return_1d"] = result["close"].pct_change().fillna(0.0)

    if "return_5d" not in result.columns and "close" in result.columns:
        if "ticker" in result.columns:
            result["return_5d"] = result.groupby("ticker")["close"].pct_change(5).fillna(0.0)
        else:
            result["return_5d"] = result["close"].pct_change(5).fillna(0.0)

    vol_col = "volume_surge_5d" if "volume_surge_5d" in result.columns else ("volume" if "volume" in result.columns else None)
    rsi_col = "rsi_14" if "rsi_14" in result.columns else None

    # Group by date for pure cross-sectional computations on same day
    grouped = result.groupby("date", group_keys=False)

    # 1. Percentile Ranks (0.0 to 1.0)
    result["cs_rank_intraday_return"] = grouped["intraday_return"].rank(pct=True).fillna(0.5)
    result["cs_rank_close_location"] = grouped["close_location_value"].rank(pct=True).fillna(0.5)

    if vol_col is not None:
        result["cs_rank_volume_surge"] = grouped[vol_col].rank(pct=True).fillna(0.5)
    else:
        result["cs_rank_volume_surge"] = 0.5

    if rsi_col is not None:
        result["cs_rank_rsi_14"] = grouped[rsi_col].rank(pct=True).fillna(0.5)
    else:
        result["cs_rank_rsi_14"] = 0.5

    # 2. Z-scores across tickers
    m1 = grouped["return_1d"].transform("mean")
    s1 = grouped["return_1d"].transform("std").replace(0.0, np.nan)
    result["cs_zscore_return_1d"] = ((result["return_1d"] - m1) / (s1 + 1e-8)).clip(-3.0, 3.0).fillna(0.0)

    m5 = grouped["return_5d"].transform("mean")
    s5 = grouped["return_5d"].transform("std").replace(0.0, np.nan)
    result["cs_zscore_return_5d"] = ((result["return_5d"] - m5) / (s5 + 1e-8)).clip(-3.0, 3.0).fillna(0.0)

    # 3. Sector relative return (or market median if sector unavailable)
    if "sector" in result.columns and result["sector"].nunique() > 1:
        sec_m = result.groupby(["date", "sector"])["return_1d"].transform("mean")
        result["sector_relative_return_1d"] = (result["return_1d"] - sec_m).fillna(0.0)
    else:
        market_med = grouped["return_1d"].transform("median")
        result["sector_relative_return_1d"] = (result["return_1d"] - market_med).fillna(0.0)

    # 4. Outperform Target (if next_open_return is present)
    if "next_open_return" in result.columns:
        valid_next = result["next_open_return"].notna()
        daily_med_next = grouped["next_open_return"].transform("median")
        result["target_cs_outperform"] = np.where(
            valid_next,
            (result["next_open_return"] > daily_med_next).astype(float),
            np.nan,
        )

    # 5. Macro Sensitivity Cross-Sectional Rank
    if "semicon_sox_beta_return" in result.columns:
        result["cs_rank_macro_sensitivity"] = grouped["semicon_sox_beta_return"].rank(pct=True).fillna(0.5)
    else:
        result["cs_rank_macro_sensitivity"] = 0.5

    return result


def select_overnight_curated_features(data: pd.DataFrame) -> list[str]:
    """Prune noise and return only the curated, high-importance overnight feature columns."""
    all_curated = OVERNIGHT_CURATED_FEATURES + CROSS_SECTIONAL_FEATURES
    available = [col for col in all_curated if col in data.columns]
    # Fallback to any numeric columns if curated list is not yet populated
    if len(available) < 5:
        from stock_ai.models.baseline import select_feature_columns
        return select_feature_columns(data, target_column="target_next_open_binary")
    return available


def purged_walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate purged, embargoed walk-forward train and test splits to prevent data leakage."""
    n_samples = len(df)
    if n_samples < 50:
        split_point = int(n_samples * 0.8)
        return [(np.arange(0, split_point), np.arange(split_point, n_samples))]

    test_size = n_samples // (n_splits + 1)
    embargo = max(1, int(n_samples * embargo_pct))
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(1, n_splits + 1):
        train_end = test_size * i
        test_start = train_end + embargo
        test_end = min(test_start + test_size, n_samples)
        if test_start >= n_samples or test_end <= test_start:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    return splits


# ---------------------------------------------------------------------------
# Anti-Jumping-Catch Execution Guard Constants & Rules
# ---------------------------------------------------------------------------

OVERNIGHT_EXHAUSTION_RETURN_THRESH = 0.045
OVERNIGHT_MOMENTUM_BURST_THRESH = 2.2
OVERNIGHT_RSI_OVERBOUGHT_MIN = 65.0
OVERNIGHT_RSI_OVERBOUGHT_MAX = 75.0
OVERNIGHT_RSI_OVERBOUGHT_RETURN_THRESH = 0.010
OVERNIGHT_UPPER_SHADOW_PULLBACK_THRESH = 0.60
OVERNIGHT_UPPER_SHADOW_INTRADAY_THRESH = -0.005
OVERNIGHT_UPPER_SHADOW_RSI_MIN = 55.0
OVERNIGHT_PANIC_PLUNGE_THRESH = -0.060


def check_jumping_catch_risk(row: pd.Series | dict[str, Any]) -> tuple[bool, str]:
    """Evaluate whether an overnight long candidate suffers from jumping-catch / exhaustion risk.

    Checks 4 empirical execution trap patterns:
    1. Overheating / Climax Exhaustion: Daily gain > 4.5% or momentum > 2.2x volatility
    2. Overbought Stalemate: RSI 65-75 extending higher (+1.0%)
    3. Upper Shadow Stall: Intraday push collapsed into close with negative intraday return
    4. Panic Plunge / Crashing Knife: Extreme down move < -6.0%

    Returns:
        (is_risky, reason_description)
    """
    ret_1d = float(row.get("return_1d", 0.0) or 0.0)
    mom_1d = float(row.get("vol_adjusted_momentum_1d", 0.0) or 0.0)
    rsi_14 = float(row.get("rsi_14", 50.0) or 50.0)
    intra_ret = float(row.get("intraday_return", 0.0) or 0.0)
    pb = float(row.get("pullback_from_high", 0.0) or 0.0)

    # 1. Overheating Exhaustion Guard
    if ret_1d > OVERNIGHT_EXHAUSTION_RETURN_THRESH:
        return True, f"⚠️ 過熱暴騰警戒（当日騰落率 {ret_1d:+.1%} > +4.5%）"
    if mom_1d > OVERNIGHT_MOMENTUM_BURST_THRESH:
        return True, f"⚠️ 過熱モメンタム警戒（ボラ調整モメンタム {mom_1d:.2f} > 2.2）"

    # 2. Overbought Extension Guard
    if OVERNIGHT_RSI_OVERBOUGHT_MIN <= rsi_14 <= OVERNIGHT_RSI_OVERBOUGHT_MAX and ret_1d > OVERNIGHT_RSI_OVERBOUGHT_RETURN_THRESH:
        return True, f"⚠️ RSI過熱追随警戒（RSI {rsi_14:.1f} 帯での続伸）"

    # 3. Upper Shadow Stall Guard
    if pb >= OVERNIGHT_UPPER_SHADOW_PULLBACK_THRESH and intra_ret < OVERNIGHT_UPPER_SHADOW_INTRADAY_THRESH and rsi_14 >= OVERNIGHT_UPPER_SHADOW_RSI_MIN:
        return True, f"⚠️ 上ヒゲ失速警戒（高値押し戻し率 {pb:.0%}, 陰線引け）"

    # 4. Panic Plunge Guard
    if ret_1d < OVERNIGHT_PANIC_PLUNGE_THRESH:
        return True, f"⚠️ 暴落ナイフ警戒（当日下落率 {ret_1d:+.1%} < -6.0%）"

    return False, ""


def add_execution_guard_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach execution guard features (pullback_from_high, jumping_catch_risk)."""
    if df.empty:
        return df
    result = df.copy()
    if "high" in result.columns and "low" in result.columns and "close" in result.columns:
        hl = (result["high"] - result["low"]).replace(0.0, np.nan)
        result["pullback_from_high"] = ((result["high"] - result["close"]) / (hl + 1e-8)).clip(0.0, 1.0).fillna(0.0)
    elif "pullback_from_high" not in result.columns:
        result["pullback_from_high"] = 0.0

    if "intraday_return" not in result.columns and "open" in result.columns and "close" in result.columns:
        result["intraday_return"] = ((result["close"] - result["open"]) / (result["open"] + 1e-8)).fillna(0.0)
    elif "intraday_return" not in result.columns:
        result["intraday_return"] = 0.0

    risks: list[bool] = []
    reasons: list[str] = []
    for _, row in result.iterrows():
        is_r, rsn = check_jumping_catch_risk(row)
        risks.append(is_r)
        reasons.append(rsn)

    result["jumping_catch_risk"] = risks
    result["jumping_catch_reason"] = reasons
    return result

