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


def select_overnight_curated_features(data: pd.DataFrame) -> list[str]:
    """Prune noise and return only the curated, high-importance overnight feature columns."""
    available = [col for col in OVERNIGHT_CURATED_FEATURES if col in data.columns]
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
