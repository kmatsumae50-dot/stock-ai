"""src/stock_ai/features/microstructure_paleologo.py

Giuseppe Paleologo "Advanced Portfolio Management" & "The Elements of Quantitative Investing"
+ Albert Kyle / Yakov Amihud / Richard Roll Market Microstructure Theory:

1. [Microstructure] Yakov Amihud (2002) Illiquidity Ratio & Asymmetric Upward Elasticity:
   - ILLIQ_t = |r_t| / Turnover_t (20-day rolling mean & surge)
   - Upward Elasticity = (Mean Pos-day ILLIQ) / (Mean Neg-day ILLIQ) over 10 days
   - Detects "上値真空地帯" (thin upper order book, explosive upside with minimal volume)

2. [Microstructure] Albert Kyle (1985) Price Impact Lambda & Richard Roll (1984) Effective Spread:
   - Kyle's Lambda: Sensitivity of return to signed log-volume (rolling 20-day regression slope)
   - Roll's Spread: 2 * sqrt(max(0, -Cov(r_t, r_{t-1}))) over 20 days

3. [Paleologo Factor Orthogonalization] Idiosyncratic Residual Return & Information Ratio:
   - Pure stock-specific alpha epsilon_{i,t} = r_{i,t} - beta_i * r_{m,t} - sector_excess
   - 5-day cumulative idiosyncratic return & 20-day idiosyncratic Sharpe/IR
   - Market-neutral residual alpha score

4. [Paleologo Dispersion & Cluster Penalty]:
   - Factor Crowding / Concentration Penalty: Penalize 2nd+ candidate from same sector or high correlation (rho > 0.70)
   - Natural diversification across orthogonal economic engines
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MICROSTRUCTURE_BASE_FEATURES = [
    "micro_amihud_20d",
    "micro_upward_elasticity_10d",
    "micro_kyle_lambda_20d",
    "micro_roll_spread_20d",
    "paleo_idio_return_5d",
    "paleo_idio_sharpe_20d",
    "paleo_beta_neutral_score",
]

MICROSTRUCTURE_CS_RANK_FEATURES = [
    f"{feat}_cs_rank" for feat in MICROSTRUCTURE_BASE_FEATURES
]

ALL_MICROSTRUCTURE_PALEOLOGO_FEATURES = (
    MICROSTRUCTURE_BASE_FEATURES + MICROSTRUCTURE_CS_RANK_FEATURES
)

INTRADAY_PRIOR_MICRO_FEATURES = [
    f"prior_{feat}" for feat in MICROSTRUCTURE_BASE_FEATURES
]


def compute_amihud_illiquidity(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """Compute Amihud (2002) Illiquidity Ratio and 5d/20d surge.

    ILLIQ_t = (|r_t| / max(Turnover_t, 1.0)) * 1e9
    """
    ret = close.pct_change().abs().fillna(0.0)
    turnover = (close * volume).clip(lower=1.0)
    illiq = (ret / turnover) * 1e9

    illiq_20 = illiq.rolling(window, min_periods=max(3, window // 4)).mean().fillna(0.0)
    illiq_5 = illiq.rolling(5, min_periods=2).mean().fillna(0.0)
    illiq_surge = (illiq_5 / (illiq_20 + 1e-6)).clip(0.1, 10.0).fillna(1.0)

    return illiq_20, illiq_surge


def compute_upward_elasticity(
    close: pd.Series,
    volume: pd.Series,
    window: int = 10,
) -> pd.Series:
    """Compute Asymmetric Upward Elasticity (上値真空度).

    Ratio of positive-day price impact to negative-day price impact.
    Higher values (> 1.0) indicate that buying produces swift upside (vacuum upper book)
    while selling is absorbed with little downward drift.
    """
    ret = close.pct_change().fillna(0.0)
    turnover = (close * volume).clip(lower=1.0)
    impact = (ret.abs() / turnover) * 1e9

    pos_impact = impact.where(ret > 0, np.nan)
    neg_impact = impact.where(ret < 0, np.nan)

    # Rolling average of positive and negative day impacts
    mean_pos = pos_impact.rolling(window, min_periods=2).mean().ffill().fillna(1.0)
    mean_neg = neg_impact.rolling(window, min_periods=2).mean().ffill().fillna(1.0)

    elasticity = (mean_pos / (mean_neg + 1e-6)).clip(0.1, 10.0).fillna(1.0)
    return elasticity


def compute_kyle_lambda(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Compute Albert Kyle (1985) Price Impact Coefficient Lambda.

    Regression slope of return on signed order flow:
    r_t = lambda * (sign(r_t) * log(1 + volume_t)) + error
    """
    ret = close.pct_change().fillna(0.0)
    signed_flow = np.sign(ret) * np.log1p(volume.clip(lower=0.0))

    # Rolling covariance and variance
    cov = ret.rolling(window, min_periods=max(5, window // 3)).cov(signed_flow)
    var = signed_flow.rolling(window, min_periods=max(5, window // 3)).var()

    kyle_lambda = (cov / (var + 1e-6)).fillna(0.0)
    # Standardize or scale for stable machine learning inputs
    scaled_lambda = kyle_lambda.clip(-5.0, 5.0)
    return scaled_lambda


def compute_roll_effective_spread(
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Compute Richard Roll (1984) Effective Bid-Ask Spread.

    Spread = 2 * sqrt(max(0, -Cov(Delta p_t, Delta p_{t-1})))
    """
    ret = close.pct_change().fillna(0.0)
    lag_ret = ret.shift(1).fillna(0.0)

    cov = ret.rolling(window, min_periods=max(5, window // 3)).cov(lag_ret)
    # Roll spread is non-zero only when autocovariance is negative
    neg_cov = (-cov).clip(lower=0.0)
    roll_spread = 2.0 * np.sqrt(neg_cov)
    return roll_spread.clip(0.0, 0.10).fillna(0.0)


def compute_paleologo_idiosyncratic_features(
    ret_series: pd.Series,
    market_ret_series: pd.Series | None = None,
    sector_ret_series: pd.Series | None = None,
    window: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Giuseppe Paleologo Factor Orthogonalization and Idiosyncratic Alpha:
    1. epsilon_{i,t} = r_{i,t} - beta_i * r_{m,t} - 0.5 * (r_{s,t} - r_{m,t})
    2. paleo_idio_return_5d: 5-day cumulative idiosyncratic return
    3. paleo_idio_sharpe_20d: 20-day idiosyncratic information ratio
    4. paleo_beta_neutral_score: Market-neutral alpha conviction score
    """
    r_i = ret_series.fillna(0.0)

    if market_ret_series is not None:
        r_m = market_ret_series.reindex(r_i.index).fillna(0.0)
    else:
        r_m = pd.Series(0.0, index=r_i.index)

    if sector_ret_series is not None:
        r_s = sector_ret_series.reindex(r_i.index).fillna(r_m)
    else:
        r_s = r_m

    # Rolling market beta
    cov_im = r_i.rolling(window, min_periods=max(5, window // 3)).cov(r_m)
    var_m = r_m.rolling(window, min_periods=max(5, window // 3)).var()
    beta = (cov_im / (var_m + 1e-6)).clip(-0.5, 3.0).fillna(1.0)

    # Pure idiosyncratic residual return
    idio_ret = r_i - (beta * r_m) - 0.5 * (r_s - r_m)

    idio_ret_5d = idio_ret.rolling(5, min_periods=1).sum().clip(-0.25, 0.25).fillna(0.0)

    mean_idio = idio_ret.rolling(window, min_periods=max(5, window // 3)).mean()
    std_idio = idio_ret.rolling(window, min_periods=max(5, window // 3)).std().replace(0.0, np.nan)
    idio_sharpe = ((mean_idio / (std_idio + 1e-6)) * np.sqrt(250.0)).clip(-5.0, 5.0).fillna(0.0)

    # Market & Beta Neutral Score
    beta_neutral_score = (
        0.5 * np.clip(idio_ret_5d * 20.0, -2.5, 2.5) + 0.5 * np.clip(idio_sharpe / 2.0, -2.5, 2.5)
    ).clip(-5.0, 5.0)

    return idio_ret_5d, idio_sharpe, beta_neutral_score


def apply_paleologo_cluster_penalty(
    candidates_df: pd.DataFrame | None = None,
    score_col: str = "p_cs",
    sector_col: str = "sector",
    ticker_col: str = "ticker",
    max_per_cluster: int = 1,
    penalty_factor: float = 0.75,
    top_k: int = 2,
    corr_matrix: pd.DataFrame | None = None,
    corr_threshold: float = 0.70,
    *,
    candidates: Sequence[str] | None = None,
    scores: Sequence[float] | None = None,
    sector_map: Mapping[str, str] | None = None,
    correlation_matrix: pd.DataFrame | None = None,
    correlation_threshold: float = 0.70,
    sector_penalty_decay: float = 0.75,
    correlation_penalty_decay: float = 0.70,
    max_picks: int | None = None,
) -> list[str]:
    """Apply Giuseppe Paleologo Sector & Correlation Cluster Dispersion Penalty.

    Selects top_k assets while penalizing factor crowding and concentration into
    the same sector or highly correlated assets (rho > corr_threshold).

    Supports both DataFrame input and candidate/score sequence keyword arguments.
    """
    if candidates_df is None:
        if not candidates:
            return []
        s_map = sector_map or {}
        scrs = list(scores) if scores is not None else [0.5] * len(candidates)
        cands = pd.DataFrame({
            "ticker": list(candidates),
            "score": scrs,
            "sector": [s_map.get(t, "Unknown") for t in candidates],
        })
        score_col = "score"
        ticker_col = "ticker"
        sector_col = "sector"
        penalty_factor = sector_penalty_decay
        top_k = max_picks if max_picks is not None else (top_k if top_k is not None else len(candidates))
        corr_matrix = correlation_matrix or corr_matrix
        corr_threshold = correlation_threshold or corr_threshold
    else:
        if candidates_df.empty:
            return []
        cands = candidates_df.copy()

    if score_col not in cands.columns:
        score_col = "p_up" if "p_up" in cands.columns else cands.columns[0]

    cands["effective_score"] = pd.to_numeric(cands[score_col], errors="coerce").fillna(0.50)
    selected_tickers: list[str] = []
    selected_sectors: list[str] = []

    while len(selected_tickers) < top_k and not cands.empty:
        # Sort by updated effective score
        cands = cands.sort_values("effective_score", ascending=False)
        best_cand = cands.iloc[0]
        best_ticker = str(best_cand[ticker_col])
        best_sector = str(best_cand.get(sector_col, "Unknown"))

        selected_tickers.append(best_ticker)
        selected_sectors.append(best_sector)
        cands = cands.iloc[1:].copy()

        if cands.empty or len(selected_tickers) >= top_k:
            break

        # Apply Paleologo penalty to remaining candidates in same sector or high correlation
        for idx in cands.index:
            cand_sec = str(cands.loc[idx, sector_col]) if sector_col in cands.columns else ""
            cand_t = str(cands.loc[idx, ticker_col])

            is_crowded = False
            # Check sector concentration
            if cand_sec and selected_sectors.count(cand_sec) >= max_per_cluster:
                is_crowded = True

            # Check correlation clustering
            if not is_crowded and corr_matrix is not None:
                if cand_t in corr_matrix.index:
                    for prev_t in selected_tickers:
                        if prev_t in corr_matrix.columns:
                            rho = float(corr_matrix.loc[cand_t, prev_t])
                            if rho >= corr_threshold:
                                is_crowded = True
                                break

            if is_crowded:
                cands.loc[idx, "effective_score"] *= penalty_factor

    return selected_tickers


def add_microstructure_paleologo_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add Market Microstructure & Paleologo Orthogonalization features to DataFrame.

    Designed for Overnight (Close-to-Open) models with zero lookahead bias at 15:30 JST close.
    """
    if data.empty or "close" not in data.columns:
        return data

    result = data.copy()
    close = pd.to_numeric(result["close"], errors="coerce").ffill().bfill()
    if "volume" in result.columns:
        volume = pd.to_numeric(result["volume"], errors="coerce").fillna(1000.0)
    else:
        volume = pd.Series(1000.0, index=result.index)

    # 1. Amihud Illiquidity
    illiq_20, illiq_surge = compute_amihud_illiquidity(close, volume, window=20)
    result["micro_amihud_20d"] = illiq_20
    result["micro_amihud_surge_5d"] = illiq_surge

    # 2. Upward Elasticity
    result["micro_upward_elasticity_10d"] = compute_upward_elasticity(close, volume, window=10)

    # 3. Kyle's Lambda
    result["micro_kyle_lambda_20d"] = compute_kyle_lambda(close, volume, window=20)

    # 4. Roll Effective Spread
    result["micro_roll_spread_20d"] = compute_roll_effective_spread(close, window=20)

    # 5. Paleologo Idiosyncratic Features
    ret_1d = close.pct_change().fillna(0.0)
    mkt_ret = (
        pd.to_numeric(result["topix_return_1d"], errors="coerce")
        if "topix_return_1d" in result.columns
        else (
            pd.to_numeric(result["n225_return_1d"], errors="coerce")
            if "n225_return_1d" in result.columns
            else None
        )
    )
    sec_ret = (
        pd.to_numeric(result["sector_mean_return"], errors="coerce")
        if "sector_mean_return" in result.columns
        else None
    )

    idio_5d, idio_sharpe, beta_neut = compute_paleologo_idiosyncratic_features(
        ret_series=ret_1d,
        market_ret_series=mkt_ret,
        sector_ret_series=sec_ret,
        window=20,
    )
    result["paleo_idio_return_5d"] = idio_5d
    result["paleo_idio_sharpe_20d"] = idio_sharpe
    result["paleo_beta_neutral_score"] = beta_neut

    return result


def add_premarket_microstructure_paleologo_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add leak-free 07:30 JST Pre-Market Microstructure & Paleologo features.

    Strictly applies shift(1) to all daily price/volume metrics so that only information
    known prior to day t market open is utilized.
    """
    if data.empty or "close" not in data.columns:
        return data

    result = data.copy()
    # Shift(1) guarantees zero leakage of day t daytime price action
    p_close = pd.to_numeric(result["close"], errors="coerce").shift(1).ffill().bfill()
    if "volume" in result.columns:
        p_vol = pd.to_numeric(result["volume"], errors="coerce").shift(1).fillna(1000.0)
    else:
        p_vol = pd.Series(1000.0, index=result.index)

    illiq_20, illiq_surge = compute_amihud_illiquidity(p_close, p_vol, window=20)
    result["prior_micro_amihud_20d"] = illiq_20
    result["prior_micro_amihud_surge_5d"] = illiq_surge
    result["prior_micro_upward_elasticity_10d"] = compute_upward_elasticity(p_close, p_vol, window=10)
    result["prior_micro_kyle_lambda_20d"] = compute_kyle_lambda(p_close, p_vol, window=20)
    result["prior_micro_roll_spread_20d"] = compute_roll_effective_spread(p_close, window=20)

    p_ret1d = p_close.pct_change().fillna(0.0)
    mkt_ret = (
        pd.to_numeric(result["topix_return_1d"], errors="coerce").shift(1)
        if "topix_return_1d" in result.columns
        else None
    )
    sec_ret = (
        pd.to_numeric(result["sector_mean_return"], errors="coerce").shift(1)
        if "sector_mean_return" in result.columns
        else None
    )

    idio_5d, idio_sharpe, beta_neut = compute_paleologo_idiosyncratic_features(
        ret_series=p_ret1d,
        market_ret_series=mkt_ret,
        sector_ret_series=sec_ret,
        window=20,
    )
    result["prior_paleo_idio_return_5d"] = idio_5d
    result["prior_paleo_idio_sharpe_20d"] = idio_sharpe
    result["prior_paleo_beta_neutral_score"] = beta_neut

    return result


def add_cross_sectional_microstructure_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute daily cross-sectional percentile ranks for microstructure and Paleologo features across universe.

    Guarantees zero future leakage by strictly grouping by date.
    """
    if data.empty or "date" not in data.columns:
        return data

    result = data.copy()
    grouped = result.groupby("date")
    for feat in MICROSTRUCTURE_BASE_FEATURES:
        rank_col = f"{feat}_cs_rank"
        if feat in result.columns:
            result[rank_col] = grouped[feat].rank(pct=True).fillna(0.5)
        else:
            result[rank_col] = 0.5

    return result


def add_cross_sectional_premarket_microstructure_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute daily cross-sectional percentile ranks for premarket microstructure features across universe.

    Guarantees zero future leakage by strictly grouping by date on shift(1) lagged metrics.
    """
    if data.empty or "date" not in data.columns:
        return data

    result = data.copy()
    grouped = result.groupby("date")
    for feat in MICROSTRUCTURE_BASE_FEATURES:
        prior_col = f"prior_{feat}"
        cs_col = f"cs_rank_prior_{feat}"
        if prior_col in result.columns:
            result[cs_col] = grouped[prior_col].rank(pct=True).fillna(0.5)
        else:
            result[cs_col] = 0.5

    return result

