"""src/stock_ai/features/chan_quant.py

Ernest P. Chan "Algorithmic Trading: Winning Strategies and Their Rationale"
& "Quantitative Trading" Core Quantitative Techniques:

1. [Chapter 2] Hurst Exponent (H) & Variance Ratio (VR) for Regime Classification
   - H > 0.5: Trending / Momentum persistence regime
   - H < 0.5: Mean-reverting / Range-bound regime
   - H = 0.5: Geometric Random Walk
   - Variance Ratio VR(5) = E[r_5^2] / (5 * E[r_1^2])
   - Combined Regime Momentum Score: (H - 0.5) * sign(Return_5d)

2. [Chapter 3] Ornstein-Uhlenbeck (OU) Process & Half-Life of Mean Reversion
   - dx_t = lambda * x_{t-1} + mu + epsilon_t
   - Half-Life = -ln(2) / lambda (days to decay by 50%)
   - OU Spread Z-score for oversold bounce & gap fill detection

3. [Chapter 5] 1D Dynamic State-Space Kalman Filter
   - Q = 1e-4 (state transition noise), R = 1e-2 (measurement noise)
   - Real-time causal filtering (zero forward-looking bias)
   - Kalman fair price spread, standardized innovation z-score, trend slope

4. [Chapter 8] Continuous Fractional Half-Kelly Criterion
   - f* = clip(0.5 * mu_net / (sigma^2 + 1e-6), 0.0, max_leverage)
   - Optimal risk-adjusted capital allocation integrated with AFML & Intraday models
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CHAN_BASE_FEATURE_NAMES = [
    "chan_hurst_20d",
    "chan_variance_ratio_10d",
    "chan_regime_momentum_score",
    "chan_ou_halflife_20d",
    "chan_ou_zscore",
    "chan_kalman_spread",
    "chan_kalman_zscore",
    "chan_kalman_slope",
]

CHAN_CS_RANK_FEATURES = [
    f"{feat}_cs_rank" for feat in CHAN_BASE_FEATURE_NAMES
]

ALL_CHAN_FEATURES = CHAN_BASE_FEATURE_NAMES + CHAN_CS_RANK_FEATURES


def compute_hurst_exponent_rolling(
    close: pd.Series,
    window: int = 20,
    lags: tuple[int, ...] = (2, 3, 4, 5),
) -> pd.Series:
    """Compute rolling Hurst exponent using log-price lag-variance scaling (second moment).

    Relation: E[(y_t - y_{t-tau})^2] ~ tau^(2H)
    Taking logs: ln(M2(tau)) = 2H * ln(tau) + C
    Slope of OLS regression gives 2H, hence H = slope / 2.
    """
    if len(close) < window:
        return pd.Series(0.5, index=close.index, dtype=float)

    log_p = np.log(close.replace(0.0, np.nan).ffill().bfill() + 1e-8)

    # Pre-calculate analytical OLS regression weights for fixed lags
    log_lags = np.log(np.array(lags, dtype=float))
    log_lags_centered = log_lags - np.mean(log_lags)
    denom = np.sum(log_lags_centered ** 2)
    weights = log_lags_centered / denom  # shape (len(lags),)

    # Compute rolling second moment E[(y_t - y_{t-tau})^2] for each lag
    log_m2_list: list[pd.Series] = []
    for lag in lags:
        diff = log_p.diff(lag)
        # Second moment = mean of squared differences
        sq_diff = diff ** 2
        m2_tau = sq_diff.rolling(window, min_periods=max(5, window // 2)).mean()
        # Avoid log(0)
        m2_tau = m2_tau.replace(0.0, np.nan).fillna(1e-8)
        log_m2_list.append(np.log(m2_tau))

    # Analytical slope across lags = sum(weights[i] * log_m2[i])
    slope = pd.Series(0.0, index=close.index, dtype=float)
    for w, lm in zip(weights, log_m2_list):
        slope = slope + w * lm

    # Hurst H = slope / 2
    hurst = (slope / 2.0).clip(0.0, 1.0)
    return hurst.fillna(0.5)


def compute_variance_ratio(
    close: pd.Series,
    window: int = 20,
    lag: int = 5,
) -> pd.Series:
    """Compute Lo & MacKinlay / Ernest Chan Variance Ratio VR(q) = E[r_q^2] / (q * E[r_1^2]).

    - VR > 1.0: Trend persistence (positive autocorrelation)
    - VR < 1.0: Mean reversion (negative autocorrelation)
    - VR ~ 1.0: Geometric Random Walk
    """
    if len(close) < window:
        return pd.Series(1.0, index=close.index, dtype=float)

    log_p = np.log(close.replace(0.0, np.nan).ffill().bfill() + 1e-8)
    r1_sq = log_p.diff(1) ** 2
    rq_sq = log_p.diff(lag) ** 2

    m1 = r1_sq.rolling(window, min_periods=max(5, window // 2)).mean()
    mq = rq_sq.rolling(window, min_periods=max(5, window // 2)).mean()

    denom = float(lag) * m1 + 1e-8
    vr = mq / denom
    return vr.clip(0.0, 5.0).fillna(1.0)


def compute_regime_momentum_score(
    hurst: pd.Series,
    return_5d: pd.Series,
) -> pd.Series:
    """Combine Hurst regime detection with directional momentum.

    Formula: (Hurst - 0.5) * sign(Return_5d)
    - If H > 0.5 (trend regime) & ret > 0 => Positive score (Trend-following Buy)
    - If H < 0.5 (mean-reverting) & ret < 0 => Positive score (Oversold Dip-Buy)
    - If H < 0.5 (mean-reverting) & ret > 0 => Negative score (Overbought Fade/Short)
    """
    centered_h = (hurst - 0.5).clip(-0.5, 0.5)
    direction = np.sign(return_5d.fillna(0.0))
    score = centered_h * direction
    return score.clip(-0.5, 0.5).fillna(0.0)


def compute_ou_halflife(
    close: pd.Series,
    window: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """Fit Ornstein-Uhlenbeck process on rolling detrended spread to extract Half-Life and Z-score.

    Model: dx_t = lambda * x_{t-1} + mu + epsilon_t
    Half-life = -ln(2) / lambda (clipped to [1.0, 100.0] days).
    """
    if len(close) < window:
        zeros = pd.Series(0.0, index=close.index, dtype=float)
        halflife = pd.Series(20.0, index=close.index, dtype=float)
        return halflife, zeros

    log_p = np.log(close.replace(0.0, np.nan).ffill().bfill() + 1e-8)
    # Detrend using rolling 20d mean
    mean_p = log_p.rolling(window, min_periods=5).mean()
    x = log_p - mean_p
    x_lag = x.shift(1).fillna(0.0)
    dx = x.diff(1).fillna(0.0)

    # Rolling linear regression slope lambda = Cov(dx, x_lag) / Var(x_lag)
    cov_dx_x = dx.rolling(window, min_periods=5).cov(x_lag)
    var_x = x_lag.rolling(window, min_periods=5).var()
    lam = cov_dx_x / (var_x + 1e-8)

    # Half-life = -ln(2) / lambda for lambda < 0
    ln2 = np.log(2.0)
    safe_lam = lam.where(lam < -1e-4, -1e-4)
    hl = -ln2 / safe_lam
    hl = hl.where(lam < -1e-4, 100.0).clip(1.0, 100.0).fillna(20.0)

    # Z-score of spread x_t
    std_x = x.rolling(window, min_periods=5).std().replace(0.0, np.nan).fillna(1e-4)
    zscore = (x / std_x).clip(-3.0, 3.0).fillna(0.0)

    return hl, zscore


def compute_kalman_filter_features(
    close: pd.Series,
    q: float = 1e-4,
    r: float = 1e-2,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """1-Dimensional Dynamic Kalman Filter for Log-Price Series (Chan Chapter 5).

    State: theta_t = true smooth price trend
    Observation: y_t = ln(close_t) = theta_t + v_t, v_t ~ N(0, R)
    State transition: theta_t = theta_{t-1} + w_t, w_t ~ N(0, Q)

    Strictly causal (no future leak).
    Returns:
        spread: y_t - theta_t (current price relative to dynamic fair value)
        zscore: innovation error standardized by sqrt(innovation variance)
        slope: 5-day slope of the smooth state estimate theta
    """
    n = len(close)
    if n == 0:
        empty = pd.Series(dtype=float)
        return empty, empty, empty

    y = np.log(close.replace(0.0, np.nan).ffill().bfill().to_numpy(dtype=float) + 1e-8)

    theta_est = np.zeros(n, dtype=float)
    p_est = np.zeros(n, dtype=float)
    innovations = np.zeros(n, dtype=float)
    f_innov_var = np.zeros(n, dtype=float)

    # Initialize at t=0
    theta_prev = y[0]
    p_prev = 1.0

    for t in range(n):
        # 1. Predict
        theta_pred = theta_prev
        p_pred = p_prev + q

        # 2. Innovation
        e = y[t] - theta_pred
        f = p_pred + r

        # 3. Kalman Gain
        k = p_pred / f

        # 4. Update
        theta_curr = theta_pred + k * e
        p_curr = (1.0 - k) * p_pred

        theta_est[t] = theta_curr
        p_est[t] = p_curr
        innovations[t] = e
        f_innov_var[t] = f

        theta_prev = theta_curr
        p_prev = p_curr

    spread = y - theta_est
    zscore = innovations / (np.sqrt(np.maximum(f_innov_var, 1e-8)))

    theta_series = pd.Series(theta_est, index=close.index)
    slope = theta_series.diff(5).fillna(0.0)

    spread_series = pd.Series(spread, index=close.index).clip(-0.25, 0.25).fillna(0.0)
    zscore_series = pd.Series(zscore, index=close.index).clip(-4.0, 4.0).fillna(0.0)
    slope_series = slope.clip(-0.15, 0.15).fillna(0.0)

    return spread_series, zscore_series, slope_series


def calculate_half_kelly_fraction(
    expected_net_return: float | np.ndarray | pd.Series,
    volatility: float | np.ndarray | pd.Series,
    max_leverage: float = 1.0,
) -> float | np.ndarray | pd.Series:
    """Continuous Fractional Half-Kelly Criterion (Chan Chapter 8).

    Full Kelly: f = mu_net / sigma^2
    Half Kelly: f* = 0.5 * mu_net / sigma^2
    Clamped to [0.0, max_leverage].

    When expected return is non-positive, fraction is strictly 0.0 (no entry).
    """
    if isinstance(expected_net_return, (int, float)) and isinstance(volatility, (int, float)):
        if expected_net_return <= 0.0 or volatility <= 0.0:
            return 0.0
        var = volatility ** 2 + 1e-6
        f_star = 0.5 * (expected_net_return / var)
        return float(np.clip(f_star, 0.0, max_leverage))

    ret = np.asarray(expected_net_return, dtype=float)
    vol = np.asarray(volatility, dtype=float)
    var = vol ** 2 + 1e-6

    f_star = 0.5 * (ret / var)
    f_star = np.where(ret > 0.0, f_star, 0.0)
    f_star = np.clip(f_star, 0.0, max_leverage)

    if isinstance(expected_net_return, pd.Series):
        return pd.Series(f_star, index=expected_net_return.index, dtype=float)
    return f_star


def add_chan_quant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all 8 Ernest Chan quantitative features to a multi-ticker or single-ticker DataFrame.

    Handles ticker grouping cleanly to eliminate any cross-sectional leaks or boundary artifacts.
    """
    if df.empty or "close" not in df.columns:
        return df

    out = df.copy()

    def _apply_chan_single_group(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        close = g["close"]
        ret_5d = g["return_5d"] if "return_5d" in g.columns else close.pct_change(5).fillna(0.0)

        # 1. Hurst & Variance Ratio
        hurst = compute_hurst_exponent_rolling(close, window=20)
        vr = compute_variance_ratio(close, window=20, lag=5)
        regime_mom = compute_regime_momentum_score(hurst, ret_5d)

        # 2. OU Process & Half-life
        ou_hl, ou_z = compute_ou_halflife(close, window=20)

        # 3. Kalman Filter
        k_spread, k_z, k_slope = compute_kalman_filter_features(close, q=1e-4, r=1e-2)

        g["chan_hurst_20d"] = hurst
        g["chan_variance_ratio_10d"] = vr
        g["chan_regime_momentum_score"] = regime_mom
        g["chan_ou_halflife_20d"] = ou_hl
        g["chan_ou_zscore"] = ou_z
        g["chan_kalman_spread"] = k_spread
        g["chan_kalman_zscore"] = k_z
        g["chan_kalman_slope"] = k_slope

        return g

    if "ticker" in out.columns and out["ticker"].nunique() > 1:
        group_dfs = []
        for _, grp in out.groupby("ticker", as_index=False):
            group_dfs.append(_apply_chan_single_group(grp))
        out = pd.concat(group_dfs, ignore_index=True)
    else:
        out = _apply_chan_single_group(out)

    # 4. Add Cross-Sectional Ranks if multiple tickers are present per date
    if "date" in out.columns and "ticker" in out.columns and out["ticker"].nunique() > 1:
        for feat in CHAN_BASE_FEATURE_NAMES:
            if feat in out.columns:
                cs_rank_col = f"{feat}_cs_rank"
                out[cs_rank_col] = (
                    out.groupby("date")[feat]
                    .rank(pct=True, method="dense")
                    .fillna(0.5)
                )

    return out
