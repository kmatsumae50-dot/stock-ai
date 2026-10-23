"""src/stock_ai/features/intraday.py

Leak-free 07:30 JST Pre-Market Intraday Feature Engineering.
Predicts daytime return: Open[t] -> Close[t] based strictly on information known by 07:30/08:50 JST.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from stock_ai.features.overnight import is_banking_finance_ticker, is_semicon_cable_ticker

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
    # 4大マクロ・外部市場ショック (07:30 JST pre-market確定)
    "overnight_sox_shock",
    "overnight_nvda_shock",
    "overnight_tnx_shock",
    "overnight_vix_shock",
    # セクター属性別マクロ感応度 (Sector-Macro Intraday Sensitivity)
    "semicon_sox_shock",
    "semicon_nvda_shock",
    "finance_tnx_shock",
    "vix_risk_off_shock",
    "semicon_intraday_gap_divergence",
    # 高度デイトレ特徴量 (Gap-to-ATR, 陽線継続クセ, ヒゲ非対称性, 出遅れ初動)
    "gap_to_atr_ratio",
    "rolling_intraday_win_rate_20d",
    "rolling_gap_fade_corr_20d",
    "prior_upper_shadow_ratio",
    "prior_lower_shadow_ratio",
    "rolling_buying_pressure_5d",
    "sector_intraday_laggard_score",
    # AFML 第5章 固定窓・分数次差分 (Pre-Market t-1 Lagged FFD Memory)
    "prior_afml_fracdiff_close",
    "prior_afml_fracdiff_vol",
    "prior_afml_fracdiff_zscore",
    # 11. Stefan Jansen ML4AT WorldQuant Formulaic Alphas (t-1 Lagged, Zero-Leakage)
    "prior_ml4at_alpha_101",
    "prior_ml4at_alpha_101_sma5",
    "prior_ml4at_alpha_12",
    "prior_ml4at_alpha_06",
    "prior_ml4at_alpha_44",
    "prior_ml4at_alpha_54",
    "prior_ml4at_info_discreteness_10d",
]

INTRADAY_CROSS_SECTIONAL_FEATURES = [
    "cs_rank_open_gap",
    "cs_rank_prior_return",
    "cs_rank_prior_rsi",
    "cs_rank_prior_clv",
    "cs_rank_rebound",
    "sector_rel_gap",
    "cs_rank_macro_intraday",
    "cs_rank_gap_to_atr",
    "cs_rank_buying_pressure",
    # ML4AT Cross-Sectional Ranks
    "cs_rank_prior_ml4at_alpha_101",
    "cs_rank_prior_ml4at_alpha_12",
    "cs_rank_prior_ml4at_alpha_54",
    "cs_rank_prior_ml4at_info_discreteness",
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

        # Clean Intraday Win target: MAE >= -1.2% and return >= +20 bps
        if "low" in result.columns:
            mae = (result["low"] - result["open"]) / (result["open"] + 1e-8)
            result["intraday_mae"] = mae
            result["target_clean_intraday_win"] = (
                (mae >= -0.012) & (result["intraday_return"] >= 0.0020)
            ).astype(int)
        else:
            result["intraday_mae"] = 0.0
            result["target_clean_intraday_win"] = (result["intraday_return"] >= 0.0020).astype(int)
    else:
        result["intraday_return"] = 0.0
        result["overnight_gap"] = 0.0
        result["target_intraday_positive"] = 0
        result["intraday_mae"] = 0.0
        result["target_clean_intraday_win"] = 0

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

    # 2b. Advanced Intraday Features (Gap-to-ATR, Rolling Win Rate, Gap-Fade Correlation, Shadows)
    result["gap_to_atr_ratio"] = (result["overnight_gap"] / (result["prior_vol_20"] + 1e-6)).fillna(0.0).clip(-5.0, 5.0)

    # Rolling intraday positive win rate (strictly lag 1)
    past_intra_win = (result["intraday_return"] > 0.0).astype(float).shift(1)
    result["rolling_intraday_win_rate_20d"] = past_intra_win.rolling(20, min_periods=5).mean().fillna(0.5)

    # Rolling correlation between overnight gap and intraday return (fade vs follow propensity)
    past_gap = result["overnight_gap"].shift(1)
    past_ret = result["intraday_return"].shift(1)
    result["rolling_gap_fade_corr_20d"] = past_gap.rolling(20, min_periods=5).corr(past_ret).fillna(0.0).clip(-1.0, 1.0)

    # Candlestick upper and lower shadows asymmetry
    if "high" in result.columns and "low" in result.columns and "open" in result.columns:
        p_hl = (result["high"] - result["low"]).shift(1).replace(0, 1e-6)
        p_open = result["open"].shift(1)
        p_close = result["close"].shift(1)
        p_high = result["high"].shift(1)
        p_low = result["low"].shift(1)

        body_top = np.maximum(p_open, p_close)
        body_bottom = np.minimum(p_open, p_close)

        result["prior_upper_shadow_ratio"] = ((p_high - body_top) / p_hl).fillna(0.0).clip(0.0, 1.0)
        result["prior_lower_shadow_ratio"] = ((body_bottom - p_low) / p_hl).fillna(0.0).clip(0.0, 1.0)
        shadow_diff = result["prior_lower_shadow_ratio"] - result["prior_upper_shadow_ratio"]
        result["rolling_buying_pressure_5d"] = shadow_diff.rolling(5, min_periods=1).mean().fillna(0.0).clip(-1.0, 1.0)
    else:
        result["prior_upper_shadow_ratio"] = 0.0
        result["prior_lower_shadow_ratio"] = 0.0
        result["rolling_buying_pressure_5d"] = 0.0

    # 3. Overnight US, FX and 4 Major Macro Shocks (07:30 JST pre-market finalized)
    t_val = str(result["ticker"].iloc[0]) if "ticker" in result.columns and len(result) > 0 else ""
    s_val = str(result["sector"].iloc[0]) if "sector" in result.columns and len(result) > 0 else ""
    is_semi = is_semicon_cable_ticker(t_val, s_val)
    is_fin = is_banking_finance_ticker(t_val, s_val)

    result["overnight_us_shock"] = pd.to_numeric(result["tesla_return_1d"], errors="coerce").fillna(0.0) if "tesla_return_1d" in result.columns else 0.0
    result["overnight_fx_shock"] = pd.to_numeric(result["usdjpy_return_1d"], errors="coerce").fillna(0.0) if "usdjpy_return_1d" in result.columns else 0.0

    sox_s = pd.to_numeric(result["sox_return_1d"], errors="coerce").fillna(0.0) if "sox_return_1d" in result.columns else pd.Series(0.0, index=result.index)
    nvda_s = pd.to_numeric(result["nvda_return_1d"], errors="coerce").fillna(0.0) if "nvda_return_1d" in result.columns else pd.Series(0.0, index=result.index)
    tnx_s = pd.to_numeric(result["tnx_return_1d"], errors="coerce").fillna(0.0) if "tnx_return_1d" in result.columns else pd.Series(0.0, index=result.index)
    vix_s = pd.to_numeric(result["vix_return_1d"], errors="coerce").fillna(0.0) if "vix_return_1d" in result.columns else pd.Series(0.0, index=result.index)

    result["overnight_sox_shock"] = sox_s
    result["overnight_nvda_shock"] = nvda_s
    result["overnight_tnx_shock"] = tnx_s
    result["overnight_vix_shock"] = vix_s

    result["semicon_sox_shock"] = sox_s if is_semi else pd.Series(0.0, index=result.index)
    result["semicon_nvda_shock"] = nvda_s if is_semi else pd.Series(0.0, index=result.index)
    result["finance_tnx_shock"] = tnx_s if is_fin else pd.Series(0.0, index=result.index)
    result["vix_risk_off_shock"] = (vix_s > 0.08).astype(float)

    if is_semi:
        result["expected_gap"] = 0.50 * sox_s + 0.30 * nvda_s + 0.20 * result["overnight_fx_shock"]
    elif is_fin:
        result["expected_gap"] = 0.40 * tnx_s + 0.60 * result["overnight_fx_shock"]
    else:
        result["expected_gap"] = 0.60 * result["overnight_us_shock"] + 0.40 * result["overnight_fx_shock"]

    result["gap_divergence"] = result["overnight_gap"] - result["expected_gap"]
    result["semicon_intraday_gap_divergence"] = (
        (result["overnight_gap"] - (0.60 * sox_s + 0.40 * nvda_s)) if is_semi else pd.Series(0.0, index=result.index)
    )

    # Initial sector laggard score (ticker vs expected gap)
    result["sector_intraday_laggard_score"] = (
        result["expected_gap"] - result["prior_sector_rel_ret"]
    ).fillna(0.0)

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

    # 5. AFML 第5章 固定窓・分数次差分 (Pre-Market t-1 Lagged FFD Memory)
    from stock_ai.features.afml import frac_diff_ffd
    if "close" in result.columns:
        p_close = pd.to_numeric(result["close"], errors="coerce").shift(1).ffill().bfill()
        log_p_close = np.log(p_close.clip(lower=1e-4))
        result["prior_afml_fracdiff_close"] = frac_diff_ffd(log_p_close, d=0.4).fillna(0.0)

        roll_m = result["prior_afml_fracdiff_close"].rolling(20, min_periods=5).mean()
        roll_s = result["prior_afml_fracdiff_close"].rolling(20, min_periods=5).std().replace(0.0, np.nan)
        result["prior_afml_fracdiff_zscore"] = ((result["prior_afml_fracdiff_close"] - roll_m) / (roll_s + 1e-8)).clip(-3.0, 3.0).fillna(0.0)
    else:
        result["prior_afml_fracdiff_close"] = 0.0
        result["prior_afml_fracdiff_zscore"] = 0.0

    if "volume" in result.columns:
        p_vol = pd.to_numeric(result["volume"], errors="coerce").shift(1).fillna(0.0)
        log_p_vol = np.log1p(p_vol.clip(lower=0.0))
        result["prior_afml_fracdiff_vol"] = frac_diff_ffd(log_p_vol, d=0.4).fillna(0.0)
    else:
        result["prior_afml_fracdiff_vol"] = 0.0

    # 6. Stefan Jansen ML4AT WorldQuant Formulaic Alphas (t-1 Lagged, Zero-Leakage)
    from stock_ai.features.ml4at import (
        compute_alpha_06,
        compute_alpha_12,
        compute_alpha_44,
        compute_alpha_54,
        compute_alpha_101,
        compute_information_discreteness,
    )
    if "close" in result.columns:
        p_c = pd.to_numeric(result["close"], errors="coerce").shift(1).ffill().bfill()
        p_o = pd.to_numeric(result.get("open", p_c), errors="coerce").shift(1).ffill().bfill()
        p_h = pd.to_numeric(result.get("high", p_c), errors="coerce").shift(1).ffill().bfill()
        p_l = pd.to_numeric(result.get("low", p_c), errors="coerce").shift(1).ffill().bfill()
        vol_src = result["volume"] if "volume" in result.columns else pd.Series(1000.0, index=result.index)
        p_v = pd.to_numeric(vol_src, errors="coerce").shift(1).fillna(1000.0)

        a101 = compute_alpha_101(p_c, p_o, p_h, p_l)
        result["prior_ml4at_alpha_101"] = a101
        result["prior_ml4at_alpha_101_sma5"] = a101.rolling(5, min_periods=1).mean().fillna(0.0)
        result["prior_ml4at_alpha_12"] = compute_alpha_12(p_c, p_v)
        result["prior_ml4at_alpha_06"] = compute_alpha_06(p_o, p_v, window=10)
        result["prior_ml4at_alpha_44"] = compute_alpha_44(p_h, p_v, window=10)
        result["prior_ml4at_alpha_54"] = compute_alpha_54(p_c, p_o, p_h, p_l)
        result["prior_ml4at_info_discreteness_10d"] = compute_information_discreteness(p_c, window=10)
    else:
        for f in [
            "prior_ml4at_alpha_101",
            "prior_ml4at_alpha_101_sma5",
            "prior_ml4at_alpha_12",
            "prior_ml4at_alpha_06",
            "prior_ml4at_alpha_44",
            "prior_ml4at_alpha_54",
            "prior_ml4at_info_discreteness_10d",
        ]:
            result[f] = 0.0

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
        "gap_to_atr_ratio": "cs_rank_gap_to_atr",
        "rolling_buying_pressure_5d": "cs_rank_buying_pressure",
    }
    for src_col, target_col in rank_cols.items():
        if src_col in result.columns:
            result[target_col] = result.groupby("date")[src_col].rank(pct=True).fillna(0.5)
        else:
            result[target_col] = 0.5

    # Sector relative gap & sector laggard refinement
    if "sector" in result.columns and "overnight_gap" in result.columns:
        sector_gap = result.groupby(["date", "sector"])["overnight_gap"].transform("mean")
        result["sector_rel_gap"] = result["overnight_gap"] - sector_gap
        # Refine sector_intraday_laggard_score: sector average gap minus ticker overnight gap
        result["sector_intraday_laggard_score"] = (sector_gap - result["overnight_gap"]).fillna(0.0)
    else:
        result["sector_rel_gap"] = 0.0

    # Macro sensitivity cross-sectional rank
    if "semicon_sox_shock" in result.columns:
        result["cs_rank_macro_intraday"] = result.groupby("date")["semicon_sox_shock"].rank(pct=True).fillna(0.5)
    else:
        result["cs_rank_macro_intraday"] = 0.5

    # ML4AT WorldQuant Alphas Daily Cross-Sectional Ranks
    ml4at_ranks = {
        "prior_ml4at_alpha_101": "cs_rank_prior_ml4at_alpha_101",
        "prior_ml4at_alpha_12": "cs_rank_prior_ml4at_alpha_12",
        "prior_ml4at_alpha_54": "cs_rank_prior_ml4at_alpha_54",
        "prior_ml4at_info_discreteness_10d": "cs_rank_prior_ml4at_info_discreteness",
    }
    for src_col, target_col in ml4at_ranks.items():
        if src_col in result.columns:
            result[target_col] = result.groupby("date")[src_col].rank(pct=True).fillna(0.5)
        else:
            result[target_col] = 0.5

    return result
