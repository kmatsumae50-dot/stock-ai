"""src/stock_ai/features/ml4at.py

Stefan Jansen "Machine Learning for Algorithmic Trading" (ML4AT) Core Alphas:
1. WorldQuant 101 Formulaic Alphas (Alpha#101, Alpha#12, Alpha#06, Alpha#44, Alpha#54)
2. Information Discreteness (Da, Engelberg, Gao 2014 & ML4AT Chapter 4)
3. Cross-Sectional Uniform Rank Normalization
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ML4AT_BASE_FEATURE_NAMES = [
    "ml4at_alpha_101",
    "ml4at_alpha_101_sma5",
    "ml4at_alpha_12",
    "ml4at_alpha_06",
    "ml4at_alpha_44",
    "ml4at_alpha_54",
    "ml4at_info_discreteness_10d",
]

ML4AT_CS_RANK_FEATURES = [
    f"{feat}_cs_rank" for feat in ML4AT_BASE_FEATURE_NAMES
]

ALL_ML4AT_FEATURES = ML4AT_BASE_FEATURE_NAMES + ML4AT_CS_RANK_FEATURES


def rolling_correlation(s1: pd.Series, s2: pd.Series, window: int = 10) -> pd.Series:
    """Compute rolling Pearson correlation between two series with fallback."""
    return s1.rolling(window, min_periods=max(3, window // 2)).corr(s2).fillna(0.0).clip(-1.0, 1.0)


def compute_alpha_101(close: pd.Series, open_p: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    """WorldQuant Alpha #101: ((Close - Open) / ((High - Low) + 0.001)).

    Measures candle body strength relative to full range (-1.0 to +1.0).
    """
    body = close - open_p
    hl_range = (high - low).abs() + 1e-5
    alpha = (body / hl_range).clip(-1.0, 1.0)
    return alpha.fillna(0.0)


def compute_alpha_12(close: pd.Series, volume: pd.Series) -> pd.Series:
    """WorldQuant Alpha #12: sign(delta(volume, 1)) * (-1 * delta(close, 1)).

    Identifies volume spikes with falling prices (institutional absorption / panic bottoming).
    Normalized by prior close to ensure cross-price comparability.
    """
    d_vol = volume.diff(1).fillna(0.0)
    d_close = close.diff(1).fillna(0.0)
    ret_1d = d_close / (close.shift(1).abs() + 1e-5)
    sign_vol = np.sign(d_vol)
    # -1 * delta_close: when price drops on huge volume, this produces a strong positive buy signal
    alpha = sign_vol * (-1.0 * ret_1d)
    return alpha.clip(-0.15, 0.15).fillna(0.0)


def compute_alpha_06(open_p: pd.Series, volume: pd.Series, window: int = 10) -> pd.Series:
    """WorldQuant Alpha #06: -1 * correlation(open, volume, 10).

    Captures exhaustion of buying volume at high opens, or silent accumulation at lower opens.
    """
    corr = rolling_correlation(open_p, volume, window=window)
    return (-1.0 * corr).fillna(0.0)


def compute_alpha_44(high: pd.Series, volume: pd.Series, window: int = 10) -> pd.Series:
    """WorldQuant Alpha #44: -1 * correlation(high, volume, 10).

    Negative correlation indicates smart money buying on low-volume pullbacks rather than chasing highs.
    """
    corr = rolling_correlation(high, volume, window=window)
    return (-1.0 * corr).fillna(0.0)


def compute_alpha_54(
    close: pd.Series, open_p: pd.Series, high: pd.Series, low: pd.Series
) -> pd.Series:
    """WorldQuant Alpha #54: -1 * (low - close) * (open**2) / ((low - high - 1e-5) * (close**2)).

    Measures lower shadow bounce efficiency and intra-day reversal power.
    Numerically stabilized and clipped to [-3.0, 3.0].
    """
    num = -1.0 * (low - close) * (open_p ** 2)
    denom = (low - high - 1e-5) * (close ** 2)
    alpha = num / denom.replace(0.0, np.nan)
    return alpha.clip(-3.0, 3.0).fillna(0.0)


def compute_information_discreteness(close: pd.Series, window: int = 10) -> pd.Series:
    """Information Discreteness (Da, Engelberg, Gao 2014 & ML4AT Chapter 4).

    Formula: ID = sign(PR) * (%neg_days - %pos_days)
    where PR is the cumulative return over the window.
    - Continuous Information (ID < 0 during uptrend): Steady stream of small positive returns (institutional accumulation).
    - Discrete Information (ID > 0 during uptrend): 1-day speculative burst followed by stagnation (retail/inago hype).
    """
    ret_1d = close.pct_change().fillna(0.0)
    cum_ret = close.pct_change(window).fillna(0.0)
    sign_cum = np.sign(cum_ret)

    is_neg = (ret_1d < 0.0).astype(float)
    is_pos = (ret_1d > 0.0).astype(float)

    pct_neg = is_neg.rolling(window, min_periods=window // 2).mean().fillna(0.5)
    pct_pos = is_pos.rolling(window, min_periods=window // 2).mean().fillna(0.5)

    id_metric = sign_cum * (pct_neg - pct_pos)
    return id_metric.clip(-1.0, 1.0).fillna(0.0)


def add_ml4at_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach all Stefan Jansen ML4AT core alphas to a DataFrame.

    Requires columns: 'close', 'open', 'high', 'low', 'volume'.
    Strictly uses past / current bar values without lookahead leakage.
    """
    if df.empty:
        return df

    result = df.copy()
    c = pd.to_numeric(result["close"], errors="coerce").ffill().bfill()
    o = pd.to_numeric(result.get("open", c), errors="coerce").ffill().bfill()
    h = pd.to_numeric(result.get("high", c), errors="coerce").ffill().bfill()
    l_p = pd.to_numeric(result.get("low", c), errors="coerce").ffill().bfill()
    v = pd.to_numeric(result["volume"], errors="coerce").fillna(1000.0) if "volume" in result.columns else pd.Series(1000.0, index=result.index)

    # 1. WorldQuant Alpha 101 & 5-day SMA
    a101 = compute_alpha_101(c, o, h, l_p)
    result["ml4at_alpha_101"] = a101
    result["ml4at_alpha_101_sma5"] = a101.rolling(5, min_periods=1).mean().fillna(0.0)

    # 2. WorldQuant Alpha 12 (Volume-Price Divergence)
    result["ml4at_alpha_12"] = compute_alpha_12(c, v)

    # 3. WorldQuant Alpha 06 & 44 (Rolling Volume Correlations)
    result["ml4at_alpha_06"] = compute_alpha_06(o, v, window=10)
    result["ml4at_alpha_44"] = compute_alpha_44(h, v, window=10)

    # 4. WorldQuant Alpha 54 (Asymmetric Lower Shadow Reversal)
    result["ml4at_alpha_54"] = compute_alpha_54(c, o, h, l_p)

    # 5. Information Discreteness (Trend Continuity)
    result["ml4at_info_discreteness_10d"] = compute_information_discreteness(c, window=10)

    return result


def add_cross_sectional_rank_features(
    panel_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Normalize features cross-sectionally per trading day into [0.0, 1.0] uniform percentile ranks.

    Removes market-wide aggregate drift, creating pure idiosyncratic cross-sectional alpha.
    """
    if panel_df.empty or date_col not in panel_df.columns:
        return panel_df

    result = panel_df.copy()
    if feature_cols is None:
        feature_cols = [c for c in ML4AT_BASE_FEATURE_NAMES if c in result.columns]

    for col in feature_cols:
        rank_col = f"{col}_cs_rank"
        # Percentile rank within each date
        result[rank_col] = (
            result.groupby(date_col)[col]
            .rank(pct=True, method="average")
            .fillna(0.5)
        )

    return result
