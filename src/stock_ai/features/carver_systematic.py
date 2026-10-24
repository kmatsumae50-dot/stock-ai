"""Robert Carver's Systematic Trading Framework: Risk Sizing, Forecast Capping, and Drawdown Control.

References:
    Robert Carver, 'Systematic Trading: A unique new method for designing trading and investing systems'
    - Chapter 10: Forecast Scaling and Capping (Target Absolute Forecast = 10, Hard Cap = [-20, +20])
    - Chapter 11: Combined Forecasts & Position Sizing
    - Chapter 12: Instrument Diversification Multiplier (IDM) & Risk Parity
    - Chapter 15 & 16: Dynamic Volatility Management & Drawdown Throttle
"""

from __future__ import annotations

import logging
from typing import Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "scale_and_cap_forecast",
    "calculate_idm",
    "compute_carver_position_weights",
    "apply_drawdown_throttle",
    "compute_volatility_regime_multiplier",
    "compute_combined_carver_throttle",
    "apply_paleologo_cluster_penalty",
    "compute_carver_paleologo_weights",
]

from stock_ai.features.microstructure_paleologo import apply_paleologo_cluster_penalty



def scale_and_cap_forecast(
    raw_forecast: Union[float, Sequence[float], np.ndarray, pd.Series],
    target_abs_forecast: float = 10.0,
    cap: float = 20.0,
    default_abs_norm: float | None = None,
) -> Union[float, np.ndarray, pd.Series]:
    """Scale a raw forecast to Carver's canonical scale (average absolute value = 10.0) and cap at [-20.0, +20.0].

    Carver Chapter 10:
        "The standard scale has an average absolute value of 10. No individual forecast should exceed
        +20 or fall below -20. This prevents extreme events from causing extreme positions."

    Args:
        raw_forecast: Raw forecast value(s), e.g. centered probability (p - 0.5) or expected return.
        target_abs_forecast: Target average absolute forecast (default 10.0).
        cap: Maximum allowable forecast magnitude (default 20.0).
        default_abs_norm: Optional baseline absolute norm if raw_forecast variance is zero or single scalar.

    Returns:
        Scaled and capped forecast with identical shape/type as input.
    """
    is_scalar = isinstance(raw_forecast, (int, float, np.floating, np.integer))
    arr = np.asarray(raw_forecast, dtype=float)

    if arr.size == 0:
        return arr if not is_scalar else 0.0

    if default_abs_norm is not None and default_abs_norm > 0:
        norm_factor = default_abs_norm
    else:
        abs_vals = np.abs(arr)
        # Exclude exact zeros when computing typical absolute size
        non_zero = abs_vals[abs_vals > 1e-9]
        if len(non_zero) > 0:
            norm_factor = float(np.mean(non_zero))
        else:
            norm_factor = 1.0

    if norm_factor < 1e-9:
        norm_factor = 1.0

    scalar = target_abs_forecast / norm_factor
    scaled = arr * scalar
    capped = np.clip(scaled, -cap, cap)

    if is_scalar:
        return float(capped.item())
    elif isinstance(raw_forecast, pd.Series):
        return pd.Series(capped, index=raw_forecast.index)
    return capped


def calculate_idm(
    num_instruments: int,
    avg_correlation: float = 0.40,
    max_idm: float = 2.5,
) -> float:
    """Calculate Carver's Instrument Diversification Multiplier (IDM).

    Carver Chapter 12:
        IDM = 1.0 / sqrt( (1/N) + ((N - 1)/N) * rho )
        For N=1: IDM = 1.0
        For N=2, rho=0.40: IDM = 1.0 / sqrt(0.5 + 0.2) = 1.195

    Args:
        num_instruments: Number of active uncorrelated / weakly correlated instruments (N >= 1).
        avg_correlation: Estimated average pairwise asset return correlation (typically 0.35 - 0.45 for equities).
        max_idm: Upper limit for IDM to protect against correlation breakdown (Carver recommends <= 2.5).

    Returns:
        Instrument Diversification Multiplier (>= 1.0).
    """
    if num_instruments <= 1:
        return 1.0

    n = float(num_instruments)
    rho = float(np.clip(avg_correlation, 0.0, 0.99))
    denom_sq = (1.0 / n) + ((n - 1.0) / n) * rho
    if denom_sq <= 1e-9:
        return 1.0

    raw_idm = 1.0 / np.sqrt(denom_sq)
    return float(np.clip(raw_idm, 1.0, max_idm))


def compute_carver_position_weights(
    forecasts: Union[Sequence[float], np.ndarray, pd.Series],
    volatilities: Union[Sequence[float], np.ndarray, pd.Series],
    max_weight: float = 0.50,
    idm: float | None = None,
    avg_correlation: float = 0.40,
    max_portfolio_leverage: float = 1.0,
) -> np.ndarray:
    """Compute risk-parity position weights using Carver's Inverse-Volatility & Forecast Scaling.

    Carver Chapter 11 & 12:
        Weight_i is proportional to (Capped_Forecast_i / 10.0) * (1 / Volatility_i) * IDM.
        Unlike Half-Kelly's 1 / sigma^2 denominator (which drastically over-weights low-volatility
        stocks that cannot cover trading friction), Carver's 1 / sigma denominator ensures
        balanced risk contribution across varying volatility regimes.

    Args:
        forecasts: Capped positive forecasts for selected long candidates (e.g. from scale_and_cap_forecast).
        volatilities: Estimated asset volatilities (e.g. 20-day rolling daily volatility, typically 0.015 - 0.05).
        max_weight: Maximum weight for an individual position (default 0.50).
        idm: Optional Instrument Diversification Multiplier. If None, computed automatically via calculate_idm.
        avg_correlation: Average asset correlation used to compute IDM if idm is None.
        max_portfolio_leverage: Maximum total portfolio gross weight (default 1.0 for unleveraged equity).

    Returns:
        Array of optimal position weights in [0.0, max_weight] summing to <= max_portfolio_leverage.
    """
    f_arr = np.asarray(forecasts, dtype=float)
    v_arr = np.asarray(volatilities, dtype=float)

    if f_arr.size == 0 or v_arr.size == 0 or f_arr.size != v_arr.size:
        return np.array([], dtype=float)

    n = f_arr.size
    if idm is None:
        idm = calculate_idm(n, avg_correlation=avg_correlation)

    # Floor volatility to avoid division by zero or extreme weights
    safe_v = np.clip(v_arr, 0.010, 0.10)
    # Forecast multiplier: baseline 10.0 gives 1.0x, cap 20.0 gives 2.0x
    f_mult = np.clip(f_arr / 10.0, 0.0, 2.0)

    # Joint signal strength: (f / 10) * (1 / sigma)
    signals = f_mult / safe_v
    tot_signal = float(np.sum(signals))
    if tot_signal <= 1e-9:
        return np.zeros(n, dtype=float)

    rel_weights = signals / tot_signal

    # Target portfolio allocation scales with average forecast and IDM
    avg_f_mult = float(np.mean(f_mult))
    target_leverage = min(max_portfolio_leverage, avg_f_mult * idm)

    unclipped = rel_weights * target_leverage
    clipped_weights = np.clip(unclipped, 0.0, max_weight)

    # Ensure total does not exceed max_portfolio_leverage
    tot_w = float(np.sum(clipped_weights))
    if tot_w > max_portfolio_leverage and tot_w > 1e-9:
        clipped_weights = clipped_weights * (max_portfolio_leverage / tot_w)

    return clipped_weights


def apply_drawdown_throttle(
    current_drawdown: float,
    max_dd_limit: float = 0.15,
    warn_dd: float = 0.07,
    min_multiplier: float = 0.20,
) -> float:
    """Carver Drawdown Throttle: Dynamically reduce position size as portfolio drawdown approaches the limit.

    Carver Chapter 15 & 16:
        When current drawdown exceeds a warning threshold (e.g. 7%), position sizes are linearly
        reduced towards min_multiplier (e.g. 0.20) as drawdown approaches max_dd_limit (e.g. 15%).
        This ensures the strategy never breaches the maximum allowable drawdown budget.

    Args:
        current_drawdown: Current drawdown fraction from peak equity (e.g. 0.08 for 8% drawdown).
        max_dd_limit: Maximum permissible portfolio drawdown (default 0.15 = 15%).
        warn_dd: Drawdown threshold where throttling begins (default 0.07 = 7%).
        min_multiplier: Minimum exposure multiplier to avoid complete zero shutdown (default 0.20 = 20%).

    Returns:
        Throttle multiplier in [min_multiplier, 1.0].
    """
    dd = float(max(0.0, current_drawdown))

    if dd <= warn_dd:
        return 1.0

    if dd >= max_dd_limit:
        return min_multiplier

    # Linear interpolation from 1.0 down to min_multiplier
    fraction = (dd - warn_dd) / (max_dd_limit - warn_dd)
    multiplier = 1.0 - fraction * (1.0 - min_multiplier)
    return float(np.clip(multiplier, min_multiplier, 1.0))


def compute_volatility_regime_multiplier(
    current_vol: float,
    baseline_vol: float = 0.020,
    min_leverage: float = 0.40,
    max_leverage: float = 1.20,
) -> float:
    """Scale exposure inverse to broader market volatility spikes.

    Carver Chapter 15:
        When broader market volatility doubles, reduce portfolio gearing by 50% to maintain
        constant portfolio daily cash volatility.

    Args:
        current_vol: Current market or portfolio volatility (daily standard deviation).
        baseline_vol: Long-term average daily volatility baseline (default 0.02 = 2.0% daily, ~31.7% annualized).
        min_leverage: Minimum scaling factor during severe volatility spikes.
        max_leverage: Maximum scaling factor during low-volatility regimes.

    Returns:
        Volatility regime multiplier in [min_leverage, max_leverage].
    """
    if current_vol <= 1e-6:
        return 1.0

    ratio = baseline_vol / current_vol
    return float(np.clip(ratio, min_leverage, max_leverage))


def compute_combined_carver_throttle(
    current_drawdown: float,
    current_vol: float | None = None,
    baseline_vol: float = 0.020,
    max_dd_limit: float = 0.15,
    warn_dd: float = 0.07,
) -> float:
    """Compute unified Carver risk throttle combining drawdown throttling and volatility regime scaling.

    Args:
        current_drawdown: Portfolio drawdown from peak equity.
        current_vol: Optional current market daily volatility.
        baseline_vol: Baseline daily volatility.
        max_dd_limit: Maximum drawdown ceiling.
        warn_dd: Warning drawdown threshold.

    Returns:
        Unified multiplier in [0.15, 1.0].
    """
    dd_mult = apply_drawdown_throttle(
        current_drawdown=current_drawdown,
        max_dd_limit=max_dd_limit,
        warn_dd=warn_dd,
        min_multiplier=0.20,
    )

    if current_vol is not None and current_vol > 0:
        vol_mult = compute_volatility_regime_multiplier(
            current_vol=current_vol,
            baseline_vol=baseline_vol,
            min_leverage=0.50,
            max_leverage=1.10,
        )
    else:
        vol_mult = 1.0

    combined = dd_mult * vol_mult
    return float(np.clip(combined, 0.15, 1.0))


def compute_carver_paleologo_weights(
    tickers: Sequence[str],
    forecasts: Sequence[float],
    volatilities: Sequence[float],
    sector_map: dict[str, str] | None = None,
    correlation_matrix: pd.DataFrame | None = None,
    max_weight: float = 0.50,
    correlation_threshold: float = 0.70,
    sector_penalty_decay: float = 0.75,
    correlation_penalty_decay: float = 0.70,
    max_portfolio_leverage: float = 1.0,
    drawdown_throttle: float = 1.0,
) -> dict[str, float]:
    """Combine Paleologo cluster penalty diversification with Carver inverse-volatility weighting.

    Steps:
    1. Apply Paleologo cluster penalty to rank and down-weight crowded/correlated candidates.
    2. Apply Carver inverse-volatility risk parity & forecast scaling.
    3. Apply Carver drawdown throttle and max-weight capping.
    """
    if not tickers:
        return {}

    # Apply Paleologo cluster penalty to crowded sectors / correlated candidates
    seen_sectors: set[str] = set()
    seen_tickers: list[str] = []
    adj_forecasts: list[float] = []
    sec_m = sector_map or {}

    for t, f in zip(tickers, forecasts):
        sec = sec_m.get(t, "")
        is_crowded = False
        if sec and sec in seen_sectors:
            is_crowded = True
        elif correlation_matrix is not None and t in correlation_matrix.index:
            for prev_t in seen_tickers:
                if prev_t in correlation_matrix.columns:
                    if float(correlation_matrix.loc[t, prev_t]) >= correlation_threshold:
                        is_crowded = True
                        break

        if is_crowded:
            adj_forecasts.append(float(f) * sector_penalty_decay)
        else:
            adj_forecasts.append(float(f))
            if sec:
                seen_sectors.add(sec)
            seen_tickers.append(t)

    base_weights = compute_carver_position_weights(
        forecasts=adj_forecasts,
        volatilities=volatilities,
        max_weight=max_weight,
        max_portfolio_leverage=max_portfolio_leverage,
    )

    final_weights = base_weights * float(np.clip(drawdown_throttle, 0.10, 1.0))
    return {t: float(w) for t, w in zip(tickers, final_weights)}

