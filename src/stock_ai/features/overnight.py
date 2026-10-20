from __future__ import annotations

import numpy as np
import pandas as pd


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
]


def add_overnight_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add Japanese market micro-structure, demand, and curated overnight features."""
    if data.empty:
        return data

    result = data.copy()
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

    return result


CROSS_SECTIONAL_FEATURES = [
    "cs_rank_intraday_return",
    "cs_rank_close_location",
    "cs_rank_volume_surge",
    "cs_rank_rsi_14",
    "cs_zscore_return_1d",
    "cs_zscore_return_5d",
    "sector_relative_return_1d",
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
