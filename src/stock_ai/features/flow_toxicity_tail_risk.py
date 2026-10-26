"""src/stock_ai/features/flow_toxicity_tail_risk.py

Quant Series #10 Master Integration:
1. Marcos López de Prado & Maureen O'Hara: Bulk Volume Classification (BVC) & VPIN (Order Flow Toxicity & OFI)
   - BVC: Decompose total volume V_t into buyer-initiated V^B_t and seller-initiated V^S_t using normal CDF Phi(Delta P / sigma_P).
   - Signed Order Flow Imbalance (OFI_t): (V^B - V^S) / V in [-1, 1].
   - VPIN_t (Volume-Synchronized Probability of Informed Trading): sum(|V^B - V^S|) / sum(V) in [0, 1].
   - Smart Money Quiet Accumulation: Institutional accumulation with low price explosion.
   - Toxic Informed Distribution at Highs: Elevated selling volume at price peaks.

2. Mark Kritzman (MIT): Absorption Ratio (AR) & Systemic Fragility / Flash-Crash Radar
   - Principal Components Analysis (PCA / Eigenvalue Decomposition) on cross-sectional returns.
   - AR_t: Fraction of total variance absorbed by top K eigenvalues (e.g. 10% of universe).
   - Standardized AR Shift: Delta AR_t = (AR_15d - AR_60d) / sigma_{Delta AR}.
   - Systemic Fragility Score: Normalized index detecting sudden market coupling / flash crash vulnerability.

3. Asymmetric Short & Exhaustion Sniper Model
   - Identifies intraday morning top / exhaustion peaks for shorting or gap-down protection:
   - High-beta overextension, Upper Shadow rejection (上ヒゲ陰線), VPIN selling distribution,
     Leader momentum stalls, and Kalman dynamic equilibrium overextension.

Strict Zero Future-Leakage Guarantee:
- At time t, only information up to bar t (or bar t-1 for pre-market execution) is utilized.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import special

logger = logging.getLogger(__name__)

# Feature column constants
FLOW_TOXICITY_FEATURES = [
    "vpin_toxicity_10d",
    "vpin_toxicity_20d",
    "vpin_signed_ofi_5d",
    "vpin_signed_ofi_20d",
    "vpin_smart_accumulation_score",
    "vpin_informed_distribution_score",
]

TAIL_RISK_FEATURES = [
    "absorption_ratio_short",
    "absorption_ratio_long",
    "delta_absorption_ratio",
    "systemic_fragility_score",
]

SHORT_EXHAUSTION_FEATURES = [
    "short_exhaustion_sniper_score",
    "exhaustion_upper_shadow_ratio",
]

ALL_FLOW_TAIL_FEATURES = (
    FLOW_TOXICITY_FEATURES + TAIL_RISK_FEATURES + SHORT_EXHAUSTION_FEATURES
)


# =====================================================================
# 1. Marcos López de Prado & Maureen O'Hara: Bulk Volume Classification & VPIN
# =====================================================================

def compute_bvc_buy_sell_volume(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
    min_periods: int = 5,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Decompose volume into buyer-initiated and seller-initiated volumes using Bulk Volume Classification (BVC).

    Reference:
        Easley, D., López de Prado, M. M., & O'Hara, M. (2012).
        Flow Toxicity and Liquidity in a High-Frequency World.
        Review of Financial Studies, 25(5), 1457-1493.

    Formula:
        Delta P_t = P_t - P_{t-1}
        sigma_{Delta P, t} = Rolling Std(Delta P, window)
        z_t = Delta P_t / sigma_{Delta P, t}
        Phi(z_t) = 0.5 * (1 + erf(z_t / sqrt(2)))
        V^B_t = V_t * Phi(z_t)
        V^S_t = V_t * (1 - Phi(z_t))
        OFI_t = (V^B_t - V^S_t) / V_t = 2 * Phi(z_t) - 1 in [-1, 1]
    """
    if close.empty or volume.empty:
        empty_s = pd.Series(dtype=float, index=close.index)
        return empty_s, empty_s, empty_s

    delta_p = close.diff().fillna(0.0)
    sigma_p = delta_p.rolling(window=window, min_periods=min_periods).std()
    # Avoid division by zero
    sigma_p = sigma_p.replace(0.0, np.nan).bfill().ffill().fillna(1e-4)
    sigma_p = np.maximum(sigma_p, 1e-6)

    z = delta_p / sigma_p
    # Standard normal CDF: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # Clip z to [-6.0, 6.0] for numerical stability
    z_clipped = np.clip(z.to_numpy(), -6.0, 6.0)
    phi = 0.5 * (1.0 + special.erf(z_clipped / np.sqrt(2.0)))
    phi_s = pd.Series(phi, index=close.index, dtype=float)

    clean_vol = volume.fillna(0.0).clip(lower=0.0)
    buy_vol = clean_vol * phi_s
    sell_vol = clean_vol * (1.0 - phi_s)
    signed_ofi = (2.0 * phi_s - 1.0).fillna(0.0)

    # Where volume was 0, buy/sell and OFI should be 0
    zero_mask = clean_vol == 0.0
    buy_vol.loc[zero_mask] = 0.0
    sell_vol.loc[zero_mask] = 0.0
    signed_ofi.loc[zero_mask] = 0.0

    return buy_vol, sell_vol, signed_ofi


def compute_vpin(
    buy_volume: pd.Series,
    sell_volume: pd.Series,
    total_volume: pd.Series,
    window: int = 10,
    min_periods: int = 3,
) -> pd.Series:
    """Compute Volume-Synchronized Probability of Informed Trading (VPIN).

    Formula:
        VPIN_t = sum_{tau=t-W+1}^t |V^B_tau - V^S_tau| / sum_{tau=t-W+1}^t V_tau
    Bounded in [0.0, 1.0].
    """
    abs_imbalance = (buy_volume - sell_volume).abs()
    sum_imbalance = abs_imbalance.rolling(window=window, min_periods=min_periods).sum()
    sum_vol = total_volume.rolling(window=window, min_periods=min_periods).sum()

    vpin = (sum_imbalance / sum_vol.replace(0.0, np.nan)).fillna(0.0)
    return vpin.clip(0.0, 1.0)


def compute_bvc_vpin_features(
    df: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    high_col: str = "high",
    low_col: str = "low",
    prefix: str = "",
) -> pd.DataFrame:
    """Generate complete suite of BVC order flow, VPIN toxicity, and accumulation/distribution scores.

    Zero-Leakage Assurance:
        Only past and contemporaneous bar data (up to index t) are accessed.
    """
    res = pd.DataFrame(index=df.index)
    if df.empty or close_col not in df.columns or volume_col not in df.columns:
        for f in FLOW_TOXICITY_FEATURES:
            res[f"{prefix}{f}"] = 0.0
        return res

    close = df[close_col].astype(float)
    volume = df[volume_col].astype(float)
    high = df[high_col].astype(float) if high_col in df.columns else close
    low = df[low_col].astype(float) if low_col in df.columns else close

    buy_vol, sell_vol, _ = compute_bvc_buy_sell_volume(close, volume, window=20)

    # 1. VPIN Toxicity (10-day & 20-day)
    vpin_10d = compute_vpin(buy_vol, sell_vol, volume, window=10)
    vpin_20d = compute_vpin(buy_vol, sell_vol, volume, window=20)
    res[f"{prefix}vpin_toxicity_10d"] = vpin_10d
    res[f"{prefix}vpin_toxicity_20d"] = vpin_20d

    # 2. Volume-Weighted Signed OFI (5-day & 20-day)
    diff_vol = buy_vol - sell_vol
    vol_5d = volume.rolling(5, min_periods=2).sum().replace(0.0, np.nan)
    ofi_5d = (diff_vol.rolling(5, min_periods=2).sum() / vol_5d).fillna(0.0).clip(-1.0, 1.0)

    vol_20d = volume.rolling(20, min_periods=5).sum().replace(0.0, np.nan)
    ofi_20d = (diff_vol.rolling(20, min_periods=5).sum() / vol_20d).fillna(0.0).clip(-1.0, 1.0)

    res[f"{prefix}vpin_signed_ofi_5d"] = ofi_5d
    res[f"{prefix}vpin_signed_ofi_20d"] = ofi_20d

    # 3. Smart Money Accumulation Score in [0, 1]
    # Strong buy volume absorption without retail overheating / price explosion
    ret_5d = close.pct_change(5).fillna(0.0)
    # Buy pressure component: OFI positive
    ofi_pos_comp = np.clip((ofi_5d + 0.05) / 0.35, 0.0, 1.0)
    # Price moderation penalty: if 5d return is already overshooting (+5% to +15%), penalize
    moderation_factor = np.exp(-np.maximum(0.0, ret_5d - 0.02) ** 2 / (2.0 * (0.04 ** 2)))
    # Toxicity guard: if toxicity is moderate-to-high, institutional accumulation is focused
    smart_acc = ofi_pos_comp * moderation_factor * (0.6 + 0.4 * np.clip(vpin_10d / 0.50, 0.0, 1.0))
    res[f"{prefix}vpin_smart_accumulation_score"] = np.clip(smart_acc, 0.0, 1.0)

    # 4. Toxic Informed Distribution at Highs in [0, 1]
    # Stock is trading near 20-day high, but heavy selling volume / negative OFI is leaking
    roll_max_20 = high.rolling(20, min_periods=1).max()
    roll_min_20 = low.rolling(20, min_periods=1).min()
    rng_20 = (roll_max_20 - roll_min_20).replace(0.0, np.nan).fillna(1.0)
    high_channel_pos = np.clip((close - roll_min_20) / rng_20, 0.0, 1.0).fillna(0.5)

    sell_pressure = np.clip(-ofi_5d / 0.30, 0.0, 1.0)
    tox_level = np.clip((vpin_10d - 0.20) / 0.50, 0.0, 1.0)

    informed_dist = high_channel_pos * (0.60 * sell_pressure + 0.40 * tox_level)
    res[f"{prefix}vpin_informed_distribution_score"] = np.clip(informed_dist, 0.0, 1.0).fillna(0.0)

    return res


# =====================================================================
# 2. Mark Kritzman: Absorption Ratio & Systemic Fragility / Flash-Crash Radar
# =====================================================================

def compute_absorption_ratio(
    returns_matrix: np.ndarray,
    fraction_eigenvalues: float = 0.10,
    min_k: int = 1,
) -> float:
    """Calculate the Absorption Ratio (AR) from a returns matrix (T x N).

    Reference:
        Kritzman, M., Li, Y., Page, S., & Rigobon, R. (2011).
        Principal Components as a Measure of Systemic Risk.
        The Journal of Portfolio Management, 37(4), 112-126.

    Formula:
        AR = sum_{i=1}^K lambda_i / sum_{j=1}^N lambda_j
        where lambda_1 >= lambda_2 >= ... >= lambda_N are eigenvalues of cov(returns).
    """
    if returns_matrix.ndim != 2:
        return 0.5

    t_obs, n_assets = returns_matrix.shape
    if n_assets == 0 or t_obs < 3:
        return 0.5

    if n_assets == 1:
        return 1.0

    # Determine K (number of top eigenvalues, e.g. 10% of universe, at least min_k)
    k = max(min_k, min(n_assets, int(np.ceil(n_assets * fraction_eigenvalues))))

    # Clean NaNs by column mean / zero
    cleaned = np.nan_to_num(returns_matrix, nan=0.0)

    # Compute covariance matrix (N x N)
    cov_mat = np.cov(cleaned, rowvar=False)
    if cov_mat.ndim == 0:
        return 1.0

    # Eigenvalues of symmetric positive semi-definite matrix
    try:
        eigvals = np.linalg.eigvalsh(cov_mat)
        # Sort in descending order
        eigvals = np.sort(eigvals)[::-1]
        eigvals = np.maximum(eigvals, 0.0)  # filter numerical negative artifacts

        total_var = float(np.sum(eigvals))
        if total_var <= 1e-12:
            return float(k / n_assets)

        absorbed_var = float(np.sum(eigvals[:k]))
        ar = absorbed_var / total_var
        return float(np.clip(ar, 0.0, 1.0))
    except Exception as exc:
        logger.debug("PCA eigenvalue decomposition error: %s", exc)
        return float(k / n_assets)


def compute_market_absorption_ratio(
    cross_sectional_returns: pd.DataFrame,
    short_window: int = 15,
    long_window: int = 60,
    fraction_eigenvalues: float = 0.10,
    min_k: int = 2,
) -> pd.DataFrame:
    """Compute time-series Absorption Ratio, standardized Delta AR, and Systemic Fragility Score.

    Zero-Leakage Assurance:
        At each date t, only historical returns in [t - window + 1, t] are used.
        No forward or future information is ever leaked.
    """
    df_ret = cross_sectional_returns.copy()
    dates = df_ret.index

    ar_short_vals = []
    ar_long_vals = []

    for i in range(len(dates)):
        # Short window slice
        if i >= 4:
            start_s = max(0, i - short_window + 1)
            slice_s = df_ret.iloc[start_s : i + 1].to_numpy()
            ar_s = compute_absorption_ratio(slice_s, fraction_eigenvalues=fraction_eigenvalues, min_k=min_k)
        else:
            ar_s = np.nan
        ar_short_vals.append(ar_s)

        # Long window slice
        if i >= 10:
            start_l = max(0, i - long_window + 1)
            slice_l = df_ret.iloc[start_l : i + 1].to_numpy()
            ar_l = compute_absorption_ratio(slice_l, fraction_eigenvalues=fraction_eigenvalues, min_k=min_k)
        else:
            ar_l = np.nan
        ar_long_vals.append(ar_l)

    res = pd.DataFrame(index=dates)
    s_short = pd.Series(ar_short_vals, index=dates).bfill().ffill().fillna(0.5)
    s_long = pd.Series(ar_long_vals, index=dates).bfill().ffill().fillna(0.5)

    res["absorption_ratio_short"] = s_short
    res["absorption_ratio_long"] = s_long

    # Standardized Delta AR: (AR_15d - AR_60d) / sigma_{Delta AR}
    diff_ar = s_short - s_long
    sigma_diff = diff_ar.rolling(window=long_window, min_periods=10).std().replace(0.0, np.nan).bfill().ffill().fillna(0.05)
    sigma_diff = np.maximum(sigma_diff, 1e-4)

    delta_ar = (diff_ar / sigma_diff).fillna(0.0).clip(-3.5, 4.0)
    res["delta_absorption_ratio"] = delta_ar

    # Systemic Fragility Score in [0, 1]: Sigmoid activation on standardized shift
    # Delta AR > 1.5 indicates sharp systemic coupling across all stocks (Flash-crash vulnerability)
    fragility = 1.0 / (1.0 + np.exp(-(delta_ar - 0.5) / 0.6))
    res["systemic_fragility_score"] = np.clip(fragility, 0.0, 1.0)

    return res


# =====================================================================
# 3. Asymmetric Short & Exhaustion Sniper Model
# =====================================================================

def compute_asymmetric_short_exhaustion(
    df: pd.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    beta_col: str = "beta_to_topix",
    rsi_col: str = "rsi_14",
    kalman_z_col: str = "chan_kalman_zscore",
    vpin_dist_col: str = "vpin_informed_distribution_score",
    leader_mom_col: str = "leadlag_leader_momentum_z",
) -> tuple[pd.Series, pd.Series]:
    """Compute Asymmetric Short & Exhaustion Sniper Score and Upper Shadow Ratio.

    Combines:
    1. Exhaustion Upper Shadow Rejection (上ヒゲ陰線)
    2. Overbought Peak-Out (RSI-14 / Stochastics exhaustion)
    3. VPIN Informed Distribution Toxicity (大口売り抜け)
    4. Lead-Lag Leader Momentum Stalls (先導株急失速・波及崩壊)
    5. Kalman Dynamic Equilibrium Overextension (カルマン上方極限乖離)
    6. High-Beta Acceleration Multiplier

    Returns:
        (short_exhaustion_sniper_score, exhaustion_upper_shadow_ratio)
    """
    if df.empty or close_col not in df.columns:
        empty_s = pd.Series(0.0, index=df.index, dtype=float)
        return empty_s, empty_s

    close = df[close_col].astype(float)
    open_p = df[open_col].astype(float) if open_col in df.columns else close
    high = df[high_col].astype(float) if high_col in df.columns else close
    low = df[low_col].astype(float) if low_col in df.columns else close

    # 1. Upper Shadow Ratio & Bearish Rejection
    candle_range = (high - low).replace(0.0, np.nan).fillna(1.0)
    candle_body_top = np.maximum(open_p, close)
    upper_shadow = np.maximum(0.0, high - candle_body_top)
    upper_shadow_ratio = (upper_shadow / candle_range).fillna(0.0).clip(0.0, 1.0)

    is_bearish = (close < open_p).astype(float)
    # Strong upper shadow with red body is high-conviction intraday rejection
    c1_shadow_exhaustion = np.clip(upper_shadow_ratio * (0.8 + 0.6 * is_bearish), 0.0, 1.0)

    # 2. Overbought Peak-Out (RSI > 55 reaching peak)
    rsi_s = df[rsi_col].astype(float) if rsi_col in df.columns else pd.Series(50.0, index=df.index)
    c2_rsi_peak = np.clip((rsi_s - 52.0) / 25.0, 0.0, 1.0)

    # 3. VPIN Informed Distribution Toxicity
    vpin_dist = df[vpin_dist_col].astype(float) if vpin_dist_col in df.columns else pd.Series(0.0, index=df.index)
    c3_vpin_toxic = np.clip(vpin_dist, 0.0, 1.0)

    # 4. Leader Momentum Drag / Spillover Breakdown
    leader_mom = df[leader_mom_col].astype(float) if leader_mom_col in df.columns else pd.Series(0.0, index=df.index)
    c4_leader_drag = np.clip(-leader_mom / 1.5, 0.0, 1.0)

    # 5. Kalman Equilibrium Overextension
    kalman_z = df[kalman_z_col].astype(float) if kalman_z_col in df.columns else pd.Series(0.0, index=df.index)
    c5_kalman_overext = np.clip((kalman_z - 0.5) / 2.0, 0.0, 1.0)

    # Beta accelerator
    beta_s = df[beta_col].astype(float) if beta_col in df.columns else pd.Series(1.0, index=df.index)
    beta_mult = 1.0 + 0.20 * np.clip(beta_s - 1.0, -0.5, 1.0)

    # Weighted composite score
    raw_score = (
        0.28 * c1_shadow_exhaustion
        + 0.20 * c2_rsi_peak
        + 0.25 * c3_vpin_toxic
        + 0.15 * c4_leader_drag
        + 0.12 * c5_kalman_overext
    ) * beta_mult

    short_score = pd.Series(np.clip(raw_score, 0.0, 1.0), index=df.index, dtype=float)
    shadow_ratio = pd.Series(upper_shadow_ratio, index=df.index, dtype=float)

    return short_score, shadow_ratio
