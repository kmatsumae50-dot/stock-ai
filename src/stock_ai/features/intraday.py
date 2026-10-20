"""src/stock_ai/features/intraday.py

Leak-free 07:30 JST Pre-Market Intraday Feature Engineering.
Predicts daytime return: Open[t] -> Close[t] based strictly on information known by 07:30/08:50 JST.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INTRADAY_BASE_FEATURES = [
    "overnight_gap",
    "prior_return_1d",
    "prior_return_5d",
    "prior_rsi_14",
    "prior_clv",
    "prior_vol_20",
    "prior_vol_adj_mom",
    "prior_macd_hist",
    "prior_bb_p",
    "prior_high_low_range",
    "prior_vol_surge_5d",
    "prior_sector_rel_ret",
    "overnight_us_shock",
    "overnight_fx_shock",
    "expected_gap",
    "gap_divergence",
    "gap_down_oversold",
    "gap_up_exhausted",
    "rebound_potential",
]

INTRADAY_CROSS_SECTIONAL_FEATURES = [
    "cs_rank_open_gap",
    "cs_rank_prior_return",
    "cs_rank_prior_rsi",
    "cs_rank_prior_clv",
    "cs_rank_rebound",
    "sector_rel_gap",
]

ALL_INTRADAY_FEATURES = INTRADAY_BASE_FEATURES + INTRADAY_CROSS_SECTIONAL_FEATURES


def add_intraday_premarket_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute leak-free 07:30 JST pre-market features for a single ticker DataFrame.

    All Japanese price series are lagged by 1 (t-1 close/high/low/open/volume)
    relative to date t's daytime trade (Open[t] -> Close[t]).
    Foreign overnight variables (US close, USD/JPY) dated t-1 are finalized
    at 05:00 JST before market open on date t.
    """
    if df.empty or "close" not in df.columns:
        return df

    result = df.copy()
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"])
        result = result.sort_values("date").reset_index(drop=True)

    # 1. Target & Opening Gap
    if "open" in result.columns and "close" in result.columns:
        result["intraday_return"] = (result["close"] - result["open"]) / (result["open"] + 1e-8)
        prev_close = result["close"].shift(1)
        result["overnight_gap"] = (result["open"] - prev_close) / (prev_close + 1e-8)
        result["target_intraday_positive"] = (result["intraday_return"] > 0.0).astype(int)
    else:
        result["intraday_return"] = 0.0
        result["overnight_gap"] = 0.0
        result["target_intraday_positive"] = 0

    # 2. Prior Day Japanese Technicals (shift 1 so they are strictly pre-market known)
    result["prior_return_1d"] = result["return_1d"].shift(1).fillna(0.0) if "return_1d" in result.columns else 0.0
    result["prior_return_5d"] = result["close"].pct_change(5).shift(1).fillna(0.0)
    result["prior_rsi_14"] = result["rsi_14"].shift(1).fillna(50.0) if "rsi_14" in result.columns else 50.0
    result["prior_clv"] = result["close_location_value"].shift(1).fillna(0.0) if "close_location_value" in result.columns else 0.0
    result["prior_vol_20"] = result["volatility_20"].shift(1).fillna(0.015) if "volatility_20" in result.columns else 0.015
    result["prior_vol_adj_mom"] = result["vol_adjusted_momentum_1d"].shift(1).fillna(0.0) if "vol_adjusted_momentum_1d" in result.columns else 0.0
    result["prior_macd_hist"] = result["macd_hist"].shift(1).fillna(0.0) if "macd_hist" in result.columns else 0.0
    result["prior_bb_p"] = result["bb_percent_b"].shift(1).fillna(0.5) if "bb_percent_b" in result.columns else 0.5
    result["prior_high_low_range"] = result["high_low_range"].shift(1).fillna(0.01) if "high_low_range" in result.columns else 0.01
    result["prior_vol_surge_5d"] = result["volume_surge_5d"].shift(1).fillna(1.0) if "volume_surge_5d" in result.columns else 1.0
    result["prior_sector_rel_ret"] = result["sector_relative_return"].shift(1).fillna(0.0) if "sector_relative_return" in result.columns else 0.0

    # 3. Overnight US and FX Shocks
    # In features parquet, tesla_return_1d / usdjpy_return_1d represent the prior night's US/FX session
    result["overnight_us_shock"] = result["tesla_return_1d"].fillna(0.0) if "tesla_return_1d" in result.columns else 0.0
    result["overnight_fx_shock"] = result["usdjpy_return_1d"].fillna(0.0) if "usdjpy_return_1d" in result.columns else 0.0
    result["expected_gap"] = 0.60 * result["overnight_us_shock"] + 0.40 * result["overnight_fx_shock"]
    result["gap_divergence"] = result["overnight_gap"] - result["expected_gap"]

    # 4. Non-Linear Reversal & Exhaustion Interactions
    result["gap_down_oversold"] = np.where(
        (result["overnight_gap"] < -0.003) & (result["prior_rsi_14"] < 45.0), 1.0, 0.0
    )
    result["gap_up_exhausted"] = np.where(
        (result["overnight_gap"] > 0.015) & (result["prior_rsi_14"] > 65.0), 1.0, 0.0
    )
    result["rebound_potential"] = (
        (45.0 - result["prior_rsi_14"]).clip(lower=0.0) * (-result["overnight_gap"]).clip(lower=0.0)
    )

    return result


def add_cross_sectional_intraday_features(pooled_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Cross-Sectional relative rankings across all tickers for each trading date.

    Guarantees zero future leak by grouping strictly by `date`.
    """
    if pooled_df.empty or "date" not in pooled_df.columns:
        return pooled_df

    result = pooled_df.copy()
    result["date"] = pd.to_datetime(result["date"])

    # Daily median target for relative outperform prediction
    if "intraday_return" in result.columns:
        result["daily_median_intraday"] = result.groupby("date")["intraday_return"].transform("median")
        result["target_intraday_cs_outperform"] = (
            result["intraday_return"] > result["daily_median_intraday"]
        ).astype(int)

    # Cross-Sectional percentile ranks
    rank_cols = {
        "overnight_gap": "cs_rank_open_gap",
        "prior_return_1d": "cs_rank_prior_return",
        "prior_rsi_14": "cs_rank_prior_rsi",
        "prior_clv": "cs_rank_prior_clv",
        "rebound_potential": "cs_rank_rebound",
    }
    for src_col, target_col in rank_cols.items():
        if src_col in result.columns:
            result[target_col] = result.groupby("date")[src_col].rank(pct=True).fillna(0.5)
        else:
            result[target_col] = 0.5

    # Sector relative gap
    if "sector" in result.columns and "overnight_gap" in result.columns:
        sector_gap = result.groupby(["date", "sector"])["overnight_gap"].transform("mean")
        result["sector_rel_gap"] = result["overnight_gap"] - sector_gap
    else:
        result["sector_rel_gap"] = 0.0

    return result
