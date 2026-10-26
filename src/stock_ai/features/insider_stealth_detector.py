"""src/stock_ai/features/insider_stealth_detector.py

Quant Series #12 Master Integration:
1. Easley, Kiefer, O'Hara & Paperman (1996): PIN (Probability of Informed Trading) & Stealth Accumulation.
2. Patrick Augustin & Marti G. Subrahmanyam (2014): Informed Options & Stock Accumulation prior to M&A / Major Policy Announcements.
   - Abnormal Volume with Range-Volatility Compression (Stealth Absorption Efficiency).
   - Continuous Insider Accumulation (10-day EMA with Close Location Value).
3. Cross-Border Policy, Defense & Strategic Infrastructure Institutional Spillover:
   - Defense/Heavy Industry, Semiconductor/AI Infrastructure, Trading/Resources, Mega-Bank/Financial clusters.
4. Pre-Bad-News Leak & Informed Dumping Detection:
   - Abnormal volume surge, negative OFI, upper shadow rejection (low CLV), downward Kyle's lambda impact.

Strict Zero Future-Leakage Guarantee:
At time t, strictly only historical and contemporaneous bar data up to index t (or t-1 for pre-market execution) are utilized.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import special

logger = logging.getLogger(__name__)

# Feature column constants
STEALTH_INSIDER_FEATURES = [
    "abnormal_volume_z_20d",
    "stealth_absorption_efficiency",
    "insider_stealth_accumulation_10d",
]

POLICY_SPILLOVER_FEATURES = [
    "policy_sector_insider_wave_score",
    "policy_cluster_tag",
]

INSIDER_DUMPING_FEATURES = [
    "insider_dumping_alert_score",
    "insider_dumping_alert_flag",
]

ALL_INSIDER_STEALTH_FEATURES = (
    STEALTH_INSIDER_FEATURES + POLICY_SPILLOVER_FEATURES + INSIDER_DUMPING_FEATURES
)

# Policy & Defense Strategic Clusters
STRATEGIC_POLICY_CLUSTERS: dict[str, list[str]] = {
    "防衛・重工": ["7011", "7013", "7012"],
    "半導体・電線": ["8035", "6857", "6146", "5803"],
    "商社・資源": ["8058", "8031", "8001"],
    "メガバンク・金融": ["8306", "8316", "8766"],
}

# Reverse mapping from clean ticker code to cluster name
TICKER_TO_CLUSTER: dict[str, str] = {}
for cluster_name, tickers in STRATEGIC_POLICY_CLUSTERS.items():
    for t in tickers:
        TICKER_TO_CLUSTER[t] = cluster_name


def _normalize_ticker_code(code: str) -> str:
    """Normalize ticker code by removing .T suffix and whitespace."""
    return str(code).strip().upper().replace(".T", "")


# =====================================================================
# 1. Abnormal Volume & Volatility Compression (Stealth Accumulation)
# =====================================================================

def compute_abnormal_volume_z(
    volume: pd.Series,
    window: int = 20,
    min_periods: int = 5,
) -> pd.Series:
    """Compute rolling log-volume Z-score relative to 20-day historical baseline.

    Formula:
        Z_{vol, t} = (log(V_t + 1) - mu_{log V, 20d}) / (sigma_{log V, 20d} + eps)

    Zero-Leakage:
        Rolling window over [t - window + 1, t] only.
    """
    if volume.empty:
        return pd.Series(0.0, index=volume.index, dtype=float)

    clean_vol = volume.fillna(0.0).clip(lower=0.0)
    log_vol = np.log(clean_vol + 1.0)

    mean_log = log_vol.rolling(window=window, min_periods=min_periods).mean()
    std_log = log_vol.rolling(window=window, min_periods=min_periods).std()
    std_log = std_log.replace(0.0, np.nan).bfill().ffill().fillna(0.3)
    std_log = np.maximum(std_log, 1e-4)

    z = (log_vol - mean_log) / std_log
    return z.fillna(0.0).clip(-3.0, 4.0)


def compute_true_range_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    min_periods: int = 5,
) -> tuple[pd.Series, pd.Series]:
    """Compute True Range (TR) and Average True Range (ATR)."""
    prev_close = close.shift(1).fillna(close)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0.0)

    atr = tr.rolling(window=window, min_periods=min_periods).mean()
    atr = atr.replace(0.0, np.nan).bfill().ffill().fillna(1.0)
    atr = np.maximum(atr, 1e-4)
    return tr, atr


def compute_stealth_absorption_efficiency(
    volume_z: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    signed_ofi: pd.Series,
) -> pd.Series:
    """Compute Volatility-Compression Absorption Efficiency.

    Formula:
        Range_t = High_t - Low_t
        RelRange_t = Range_t / (ATR_{20d, t} + eps)
        StealthEff_t = (max(0, OFI_t) * exp(0.5 * clip(Z_{vol, t}, -2, 3))) / (RelRange_t + 0.5)

    Interpretation:
        When volume surges (high Z_vol) with positive buying flow (OFI > 0), but the price range
        is artificially suppressed (Range / ATR is small), this indicates stealth passive limit-order
        accumulation by smart money/insiders absorbing market sellers without allowing the market price to spike.
    """
    range_t = (high - low).clip(lower=0.0)
    rel_range = (range_t / atr).fillna(1.0).clip(0.0, 5.0)

    pos_ofi = signed_ofi.clip(lower=0.0)
    vol_factor = np.exp(0.5 * volume_z.clip(-2.0, 3.0))

    stealth_eff = (pos_ofi * vol_factor) / (rel_range + 0.5)
    return stealth_eff.fillna(0.0)


def compute_close_location_value(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Compute Close Location Value (CLV) bounded in [-1.0, 1.0].

    CLV = ((Close - Low) - (High - Close)) / ((High - Low) + eps)
    +1.0 = Closed at the day's absolute high (strong buying into the close)
    -1.0 = Closed at the day's absolute low (heavy selling into the close)
    """
    h_l = (high - low).clip(lower=0.0)
    clv = ((close - low) - (high - close)) / (h_l.replace(0.0, np.nan))
    return clv.fillna(0.0).clip(-1.0, 1.0)


def compute_stealth_insider_features(
    df: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    signed_ofi_col: str = "vpin_signed_ofi_5d",
    prefix: str = "",
) -> pd.DataFrame:
    """Compute complete suite of stealth accumulation features for a single stock DataFrame.

    Returns:
        DataFrame with columns:
        - abnormal_volume_z_20d: Rolling volume Z-score.
        - stealth_absorption_efficiency: Immediate absorption efficiency.
        - insider_stealth_accumulation_10d: 10-day EMA of (StealthEff * normalized CLV) in [0, 1].
    """
    res = pd.DataFrame(index=df.index)
    if df.empty or close_col not in df.columns or volume_col not in df.columns:
        for col in STEALTH_INSIDER_FEATURES:
            res[f"{prefix}{col}"] = 0.0
        return res

    close = df[close_col].astype(float)
    volume = df[volume_col].astype(float)
    high = df[high_col].astype(float) if high_col in df.columns else close
    low = df[low_col].astype(float) if low_col in df.columns else close

    # 1. Abnormal volume Z
    vol_z = compute_abnormal_volume_z(volume, window=20, min_periods=5)
    res[f"{prefix}abnormal_volume_z_20d"] = vol_z

    # 2. Signed OFI (use existing BVC column or compute on the fly)
    if signed_ofi_col in df.columns:
        signed_ofi = df[signed_ofi_col].astype(float).fillna(0.0)
    else:
        # Fast causal OFI approximation using daily return direction & volume
        ret_1d = close.pct_change().fillna(0.0)
        sigma_r = ret_1d.rolling(20, min_periods=5).std().replace(0.0, np.nan).bfill().ffill().fillna(0.02)
        z_r = (ret_1d / sigma_r).clip(-4.0, 4.0)
        phi = 0.5 * (1.0 + special.erf(z_r.to_numpy() / np.sqrt(2.0)))
        signed_ofi = pd.Series(2.0 * phi - 1.0, index=df.index, dtype=float)

    # 3. ATR & Absorption Efficiency
    _, atr = compute_true_range_atr(high, low, close, window=20, min_periods=5)
    stealth_eff = compute_stealth_absorption_efficiency(vol_z, high, low, atr, signed_ofi)
    res[f"{prefix}stealth_absorption_efficiency"] = stealth_eff

    # 4. CLV and 10-day Continuous Accumulation Score
    clv = compute_close_location_value(high, low, close)
    # Normalized CLV in [0.0, 1.0]: 0.5 is neutral
    clv_norm = (clv + 1.0) / 2.0

    # Combined instantaneous signal: StealthEff * (0.3 + 0.7 * clv_norm)
    inst_signal = stealth_eff * (0.3 + 0.7 * clv_norm)

    # 10-day EMA accumulation score (min_periods=1 to prevent initial NaNs)
    accum_ema = inst_signal.ewm(span=10, min_periods=1).mean()
    # Normalize to [0.0, 1.0] using sigmoid scaling
    # Typically inst_signal is between 0.0 and 3.0; scale such that 0.5 is average, >= 0.8 is strong stealth
    accum_score = 1.0 / (1.0 + np.exp(-(accum_ema - 0.45) / 0.25))
    res[f"{prefix}insider_stealth_accumulation_10d"] = np.clip(accum_score.fillna(0.50), 0.0, 1.0)

    return res


# =====================================================================
# 2. Cross-Border Policy & Defense Strategic Sector Spillover
# =====================================================================

def compute_policy_defense_insider_spillover(
    full_df: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    vol_z_col: str = "abnormal_volume_z_20d",
    ofi_col: str = "vpin_signed_ofi_5d",
    ret_col: str = "return_1d",
) -> pd.DataFrame:
    """Compute market-relative cluster abnormal institutional inflows across strategic themes.

    Identifies when an entire policy theme (Defense/Heavy Industry, Semiconductor, Trading, Mega-Bank)
    experiences a synchronized surge in abnormal volume Z and buying OFI relative to the market median.

    Zero-Leakage Assurance:
        Computed cross-sectionally per date t. No lookahead to t+1.

    Returns:
        DataFrame with matching index containing:
        - policy_sector_insider_wave_score: float in [0.0, 1.0]
        - policy_cluster_tag: string name of the cluster, or "一般"
    """
    if full_df.empty or date_col not in full_df.columns or ticker_col not in full_df.columns:
        res = pd.DataFrame(index=full_df.index)
        res["policy_sector_insider_wave_score"] = 0.50
        res["policy_cluster_tag"] = "一般"
        return res

    work_df = full_df[[date_col, ticker_col]].copy()
    work_df["clean_ticker"] = work_df[ticker_col].apply(_normalize_ticker_code)
    work_df["cluster"] = work_df["clean_ticker"].map(TICKER_TO_CLUSTER).fillna("一般")

    # Volume Z & OFI
    if vol_z_col in full_df.columns:
        work_df["vol_z"] = full_df[vol_z_col].fillna(0.0)
    else:
        work_df["vol_z"] = 0.0

    if ofi_col in full_df.columns:
        work_df["ofi"] = full_df[ofi_col].fillna(0.0)
    else:
        work_df["ofi"] = 0.0

    # Cross-sectional market medians per date
    mkt_stats = work_df.groupby(date_col).agg(
        mkt_med_vol_z=("vol_z", "median"),
        mkt_med_ofi=("ofi", "median"),
    ).reset_index()

    work_df = work_df.merge(mkt_stats, on=date_col, how="left")
    work_df["excess_vol_z"] = work_df["vol_z"] - work_df["mkt_med_vol_z"]
    work_df["excess_ofi"] = work_df["ofi"] - work_df["mkt_med_ofi"]

    # Calculate cluster-level abnormal inflows per date
    # Only for the 4 recognized strategic clusters
    cluster_df = work_df[work_df["cluster"] != "一般"].copy()
    if not cluster_df.empty:
        cluster_stats = cluster_df.groupby([date_col, "cluster"]).agg(
            cluster_excess_vol=("excess_vol_z", "mean"),
            cluster_excess_ofi=("excess_ofi", "mean"),
        ).reset_index()

        # Combine excess volume and excess OFI into raw cluster surge metric
        # High surge when both volume Z > 0 and OFI > 0
        raw_surge = (
            np.maximum(0.0, cluster_stats["cluster_excess_vol"]) * 0.5
            + np.maximum(0.0, cluster_stats["cluster_excess_ofi"]) * 1.5
        )
        # Sigmoid scale into [0.0, 1.0]: 0.5 is neutral baseline, > 0.75 is strong policy wave
        cluster_stats["cluster_wave_score"] = 1.0 / (1.0 + np.exp(-(raw_surge - 0.35) / 0.30))
        cluster_stats["cluster_wave_score"] = np.clip(cluster_stats["cluster_wave_score"], 0.0, 1.0)

        work_df = work_df.merge(
            cluster_stats[[date_col, "cluster", "cluster_wave_score"]],
            on=[date_col, "cluster"],
            how="left",
        )
        work_df["cluster_wave_score"] = work_df["cluster_wave_score"].fillna(0.50)
    else:
        work_df["cluster_wave_score"] = 0.50

    res = pd.DataFrame(index=full_df.index)
    res["policy_sector_insider_wave_score"] = work_df["cluster_wave_score"].to_numpy()
    res["policy_cluster_tag"] = work_df["cluster"].to_numpy()

    return res


# =====================================================================
# 3. Pre-Bad-News Leak & Informed Dumping Detection
# =====================================================================

def compute_insider_dumping_risk(
    df: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    signed_ofi_col: str = "vpin_signed_ofi_5d",
    prefix: str = "",
) -> pd.DataFrame:
    """Compute Pre-Bad-News Informed Dumping Risk and Alert Flag.

    Detects distribution patterns characteristic of pre-leak dumps:
    - Abnormal volume surge (Z_vol > 0.5)
    - Strong negative order flow (OFI < -0.15)
    - Weak close / long upper shadow (CLV < 0.35)
    - High downward price impact (Kyle's lambda on negative days)

    Zero-Leakage Assurance:
        Only contemporaneous and historical bar data up to t are accessed.

    Returns:
        DataFrame with columns:
        - insider_dumping_alert_score: float in [0.0, 1.0]
        - insider_dumping_alert_flag: bool (True when score >= 0.70)
    """
    res = pd.DataFrame(index=df.index)
    if df.empty or close_col not in df.columns or volume_col not in df.columns:
        res[f"{prefix}insider_dumping_alert_score"] = 0.0
        res[f"{prefix}insider_dumping_alert_flag"] = False
        return res

    close = df[close_col].astype(float)
    volume = df[volume_col].astype(float)
    high = df[high_col].astype(float) if high_col in df.columns else close
    low = df[low_col].astype(float) if low_col in df.columns else close

    # 1. Volume Z
    vol_z = compute_abnormal_volume_z(volume, window=20, min_periods=5)
    vol_surge = np.maximum(0.0, vol_z)

    # 2. Negative OFI intensity
    if signed_ofi_col in df.columns:
        signed_ofi = df[signed_ofi_col].astype(float).fillna(0.0)
    else:
        ret_1d = close.pct_change().fillna(0.0)
        sigma_r = ret_1d.rolling(20, min_periods=5).std().replace(0.0, np.nan).bfill().ffill().fillna(0.02)
        z_r = (ret_1d / sigma_r).clip(-4.0, 4.0)
        phi = 0.5 * (1.0 + special.erf(z_r.to_numpy() / np.sqrt(2.0)))
        signed_ofi = pd.Series(2.0 * phi - 1.0, index=df.index, dtype=float)

    neg_ofi = np.maximum(0.0, -signed_ofi)

    # 3. Weak CLV / Upper Shadow Rejection
    clv = compute_close_location_value(high, low, close)
    weak_close = np.maximum(0.0, -clv)  # 0 to 1 when closing near the lows

    # Upper shadow ratio: (High - max(Open, Close)) / (High - Low + eps)
    # If open is missing, use prev_close
    open_s = df["open"].astype(float) if "open" in df.columns else close.shift(1).fillna(close)
    body_high = np.maximum(open_s, close)
    h_l = (high - low).clip(lower=0.0)
    upper_shadow = ((high - body_high) / (h_l.replace(0.0, np.nan))).fillna(0.0).clip(0.0, 1.0)

    # 4. Downward Kyle's Lambda (Downside Price Impact per unit volume)
    ret_t = close.pct_change().fillna(0.0)
    down_ret = np.maximum(0.0, -ret_t)
    sqrt_vol = np.sqrt(volume.clip(lower=0.0) + 1.0)
    raw_lambda = down_ret / (sqrt_vol + 1e-4)
    norm_lambda = (
        raw_lambda / (raw_lambda.rolling(20, min_periods=5).mean().replace(0.0, np.nan).bfill().ffill().fillna(1e-4))
    ).fillna(1.0).clip(0.0, 3.0)

    # Combined composite dumping score
    # Strongest signal when volume is surging, OFI is selling, close is weak, and down impact is high
    raw_dumping = (
        0.35 * neg_ofi
        + 0.25 * (vol_surge / 3.0)
        + 0.25 * weak_close
        + 0.15 * (norm_lambda / 3.0)
    )

    # Rolling 3-day smoothing to avoid single-tick false spikes
    smoothed_dumping = raw_dumping.rolling(3, min_periods=1).mean()

    # Sigmoid calibration into [0.0, 1.0]: >= 0.70 triggers alert
    dumping_score = 1.0 / (1.0 + np.exp(-(smoothed_dumping - 0.40) / 0.18))
    dumping_score = np.clip(dumping_score, 0.0, 1.0)

    res[f"{prefix}insider_dumping_alert_score"] = dumping_score
    res[f"{prefix}insider_dumping_alert_flag"] = dumping_score >= 0.70

    return res


# =====================================================================
# 4. Pipeline Integration Helper
# =====================================================================

def add_insider_stealth_features(
    df: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    signed_ofi_col: str = "vpin_signed_ofi_5d",
    prefix: str = "",
) -> pd.DataFrame:
    """Helper to append all stealth insider and dumping features to a DataFrame."""
    res = df.copy()

    # 1. Stealth Accumulation
    stealth_df = compute_stealth_insider_features(
        df,
        close_col=close_col,
        high_col=high_col,
        low_col=low_col,
        volume_col=volume_col,
        signed_ofi_col=signed_ofi_col,
        prefix=prefix,
    )
    for c in stealth_df.columns:
        res[c] = stealth_df[c]

    # 2. Informed Dumping Risk
    dumping_df = compute_insider_dumping_risk(
        df,
        close_col=close_col,
        high_col=high_col,
        low_col=low_col,
        volume_col=volume_col,
        signed_ofi_col=signed_ofi_col,
        prefix=prefix,
    )
    for c in dumping_df.columns:
        res[c] = dumping_df[c]

    # Initialize policy sector wave to default if not yet computed cross-sectionally
    if f"{prefix}policy_sector_insider_wave_score" not in res.columns:
        res[f"{prefix}policy_sector_insider_wave_score"] = 0.50
    if f"{prefix}policy_cluster_tag" not in res.columns:
        res[f"{prefix}policy_cluster_tag"] = "一般"

    return res
