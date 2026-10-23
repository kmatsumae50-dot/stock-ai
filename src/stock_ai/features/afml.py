"""src/stock_ai/features/afml.py

Advances in Financial Machine Learning (AFML, Marcos López de Prado) Core Algorithms.
1. Fixed-Width Window Fractional Differentiation (FFD) (Chapter 5)
2. Dynamic Volatility Triple-Barrier Labeling (Chapter 3)
3. Sample Weighting with Absolute Return & Time-Decay (Chapter 4)
4. Out-of-Sample MDA (Mean Decrease Accuracy) Feature Pruning (Chapter 8)
5. Meta-Labeling & Bet Sizing (Chapters 3 & 10)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

AFML_FEATURE_NAMES = [
    "afml_fracdiff_close_d04",
    "afml_fracdiff_vol_d04",
    "afml_fracdiff_zscore_20",
]


# ---------------------------------------------------------------------------
# 1. Fixed-Width Window Fractional Differentiation (FFD) (Chapter 5)
# ---------------------------------------------------------------------------

def get_weights_ffd(d: float, thres: float = 1e-3, max_l: int = 50) -> np.ndarray:
    """Generate memory-preserving weights for Fixed-width window Fractional Differentiation.

    Recurrence relation: w_k = -w_{k-1} * (d - k + 1) / k, w_0 = 1.0.
    Terminates when abs(w_k) < thres or length reaches max_l.
    Returns reversed weights array [w_l, w_{l-1}, ..., w_1, w_0] for direct convolution.
    """
    w = [1.0]
    k = 1
    while k < max_l:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1], dtype=np.float64)


def frac_diff_ffd(
    series: pd.Series,
    d: float = 0.4,
    thres: float = 1e-3,
    max_l: int = 50,
) -> pd.Series:
    """Apply Fixed-width window Fractional Differentiation (FFD) to a series.

    Strict zero-leakage: each point s_t uses only past values s_{t-l+1 ... t}.
    """
    clean_s = pd.to_numeric(series, errors="coerce").dropna()
    if clean_s.empty:
        return pd.Series(0.0, index=series.index)

    weights = get_weights_ffd(d=d, thres=thres, max_l=max_l)
    width = len(weights)
    if len(clean_s) < width:
        # Fallback to simple differencing if history is too short
        diff = clean_s.diff().fillna(0.0)
        return diff.reindex(series.index).fillna(0.0)

    vals = clean_s.values
    conv = np.convolve(vals, weights, mode="valid")
    res = pd.Series(conv, index=clean_s.index[width - 1:])
    return res.reindex(series.index).bfill().fillna(0.0)


def add_afml_features(df: pd.DataFrame, d: float = 0.4) -> pd.DataFrame:
    """Add AFML FFD memory features to a DataFrame.

    Computes:
    - afml_fracdiff_close_d04: log(Close) FFD with d=0.4
    - afml_fracdiff_vol_d04: log(Volume + 1) FFD with d=0.4
    - afml_fracdiff_zscore_20: 20-day rolling z-score of price FFD
    """
    if df.empty:
        return df

    result = df.copy()

    # 1. Price FFD
    close = pd.to_numeric(result["close"], errors="coerce").ffill().bfill()
    log_close = np.log(close.clip(lower=1e-4))
    ffd_close = frac_diff_ffd(log_close, d=d)
    result["afml_fracdiff_close_d04"] = ffd_close.fillna(0.0)

    # 2. Volume FFD
    if "volume" in result.columns:
        vol = pd.to_numeric(result["volume"], errors="coerce").fillna(0.0)
        log_vol = np.log1p(vol.clip(lower=0.0))
        ffd_vol = frac_diff_ffd(log_vol, d=d)
        result["afml_fracdiff_vol_d04"] = ffd_vol.fillna(0.0)
    else:
        result["afml_fracdiff_vol_d04"] = 0.0

    # 3. 20-day rolling Z-score of Price FFD
    roll_mean = result["afml_fracdiff_close_d04"].rolling(20, min_periods=5).mean()
    roll_std = result["afml_fracdiff_close_d04"].rolling(20, min_periods=5).std().replace(0.0, np.nan)
    result["afml_fracdiff_zscore_20"] = ((result["afml_fracdiff_close_d04"] - roll_mean) / (roll_std + 1e-8)).clip(-3.0, 3.0).fillna(0.0)

    return result


# ---------------------------------------------------------------------------
# 2. Dynamic Volatility Triple-Barrier Labeling (Chapter 3)
# ---------------------------------------------------------------------------

def compute_dynamic_triple_barrier_events(
    df: pd.DataFrame,
    k_pt: float = 1.0,
    k_sl: float = 0.8,
    cost_bps: float = 20.0,
) -> pd.DataFrame:
    """Compute Dynamic Volatility Triple-Barrier outcomes for each bar.

    Barriers:
    - Upper Barrier (Profit Take): Open * (1 + k_pt * sigma)
    - Lower Barrier (Stop Loss): Open * (1 - k_sl * sigma)
    - Vertical Barrier: Market Close (15:30)

    Returns DataFrame with:
    - tb_upper_barrier, tb_lower_barrier
    - tb_hit_upper (bool), tb_hit_lower (bool)
    - tb_first_hit (-1: SL hit first, 0: vertical barrier / close, +1: PT hit first)
    - tb_clean_win (1 if profitable after cost without touching SL, else 0)
    """
    result = df.copy()
    c = pd.to_numeric(result.get("close", 1000.0), errors="coerce").fillna(1000.0)
    o = pd.to_numeric(result.get("open", c), errors="coerce").fillna(c)
    h = pd.to_numeric(result.get("high", c), errors="coerce").fillna(c)
    l_p = pd.to_numeric(result.get("low", c), errors="coerce").fillna(c)

    vol_src = result["prior_vol_20"] if "prior_vol_20" in result.columns else (result["volatility_20"] if "volatility_20" in result.columns else 0.018)
    vol = pd.to_numeric(vol_src, errors="coerce")
    if isinstance(vol, (int, float)):
        vol = pd.Series(vol, index=result.index)
    vol = vol.fillna(0.018).clip(lower=0.005)

    # Dynamic horizontal barrier levels relative to Open
    up_dist = k_pt * vol
    dn_dist = k_sl * vol
    result["tb_upper_barrier"] = o * (1.0 + up_dist)
    result["tb_lower_barrier"] = o * (1.0 - dn_dist)

    hit_up = h >= result["tb_upper_barrier"]
    hit_dn = l_p <= result["tb_lower_barrier"]

    result["tb_hit_upper"] = hit_up
    result["tb_hit_lower"] = hit_dn

    # Path evaluation: if both touched, check which one was more extreme or default to SL conservative
    first_hit = pd.Series(0, index=result.index)
    first_hit[hit_up & ~hit_dn] = 1
    first_hit[hit_dn & ~hit_up] = -1
    first_hit[hit_up & hit_dn] = -1  # conservative: assume worst path when both hit in same daily bar
    result["tb_first_hit"] = first_hit

    # Clean Win condition: (Close - Open) / Open >= cost (20bps) AND lower barrier was NOT hit
    ret_intra = (c - o) / o
    cost_dec = cost_bps / 10000.0
    result["tb_clean_win"] = ((ret_intra >= cost_dec) & (~hit_dn)).astype(int)

    return result


# ---------------------------------------------------------------------------
# 3. Sample Weighting with Absolute Return & Time-Decay (Chapter 4)
# ---------------------------------------------------------------------------

def compute_afml_sample_weights(
    returns: pd.Series | np.ndarray,
    dates: pd.Series | pd.DatetimeIndex,
    time_decay: float = 0.50,
    cost_bps: float = 20.0,
) -> np.ndarray:
    """Compute sample weights combining return magnitude (informativeness) and time decay.

    - Returns with magnitude > cost_bps receive higher weights.
    - Recent dates receive higher weight (decaying down to time_decay for oldest dates).
    - Normalized so mean(weights) == 1.0.
    """
    ret_arr = np.asarray(returns, dtype=np.float64)
    abs_ret = np.abs(ret_arr)
    cost_dec = cost_bps / 10000.0

    # 1. Return informativeness weight: linear boost above cost threshold
    w_ret = 1.0 + np.clip((abs_ret - cost_dec) / (cost_dec + 1e-6), 0.0, 5.0)

    # 2. Time decay weight
    d_series = pd.to_datetime(pd.Series(dates))
    min_date = d_series.min()
    max_date = d_series.max()
    span_days = max(1.0, (max_date - min_date).total_seconds() / 86400.0)
    day_diffs = (d_series - min_date).dt.total_seconds() / 86400.0

    # Linearly scale from time_decay (oldest) to 1.0 (newest)
    w_time = time_decay + (1.0 - time_decay) * (day_diffs.values / span_days)

    # 3. Combined & normalized weight
    w_comb = w_ret * w_time
    mean_w = np.mean(w_comb)
    if mean_w > 0:
        w_norm = w_comb / mean_w
    else:
        w_norm = np.ones_like(w_comb)

    return np.asarray(w_norm, dtype=np.float64)


# ---------------------------------------------------------------------------
# 4. Out-of-Sample MDA (Mean Decrease Accuracy) Feature Pruning (Chapter 8)
# ---------------------------------------------------------------------------

def prune_features_with_mda(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    feature_names: list[str],
    min_importance: float = 0.0,
    n_repeats: int = 2,
    random_state: int = 42,
) -> tuple[list[str], dict[str, float]]:
    """Prune noise features by measuring Out-of-Sample Mean Decrease in AUC (MDA).

    Features with MDA <= min_importance (shuffling them does not reduce AUC) are pruned.
    Guarantees at least 15 core features remain even if pruning is aggressive.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_val).astype(int)

    base_probs = model.predict_proba(X_val[feature_names])[:, 1]
    base_auc = float(roc_auc_score(y_true, base_probs))

    mda_scores: dict[str, float] = {}

    for col in feature_names:
        drops: list[float] = []
        for _ in range(n_repeats):
            X_shuffled = X_val[feature_names].copy()
            X_shuffled[col] = rng.permutation(X_shuffled[col].values)
            shuff_probs = model.predict_proba(X_shuffled)[:, 1]
            try:
                shuff_auc = float(roc_auc_score(y_true, shuff_probs))
                drops.append(base_auc - shuff_auc)
            except Exception:
                drops.append(0.0)
        mda_scores[col] = float(np.mean(drops))

    # Keep features with positive contribution, ensuring top 15 features are preserved
    sorted_features = sorted(mda_scores.items(), key=lambda x: x[1], reverse=True)
    selected = [feat for feat, score in sorted_features if score > min_importance]

    # Safety: maintain at least 15 top features
    if len(selected) < 15:
        selected = [feat for feat, _ in sorted_features[:max(15, len(selected))]]

    logger.info("AFML MDA Feature Pruning: kept %d of %d features.", len(selected), len(feature_names))
    return selected, mda_scores


# ---------------------------------------------------------------------------
# 5. Meta-Labeling & Bet Sizing (Chapters 3 & 10)
# ---------------------------------------------------------------------------

def generate_meta_labels(
    primary_preds: pd.Series | np.ndarray,
    actual_returns: pd.Series | np.ndarray,
    lower_barrier_hit: pd.Series | np.ndarray | None = None,
    cost_bps: float = 20.0,
    primary_buy_threshold: float = 0.50,
) -> np.ndarray:
    """Generate Meta-Labels (Y_meta in {0, 1}) for samples where Primary model recommended Buy.

    - Y_meta = 1 (True Positive): Primary model said Buy (p >= threshold),
      actual return >= cost_bps (20bps), and lower barrier was not hit.
    - Y_meta = 0 (False Positive): Primary model said Buy, but trade failed to produce clean net profit.
    """
    p_arr = np.asarray(primary_preds, dtype=np.float64)
    r_arr = np.asarray(actual_returns, dtype=np.float64)
    cost_dec = cost_bps / 10000.0

    if lower_barrier_hit is not None:
        lb_hit = np.asarray(lower_barrier_hit, dtype=bool)
    else:
        lb_hit = np.zeros(len(p_arr), dtype=bool)

    is_primary_buy = p_arr >= primary_buy_threshold
    is_clean_profitable = (r_arr >= cost_dec) & (~lb_hit)

    # For meta classifier, 1 if clean profitable given buy recommendation, 0 if false positive
    meta_labels = np.where(is_primary_buy & is_clean_profitable, 1, 0)
    return meta_labels


def calculate_bet_size(meta_prob: float, base_size: float = 1.0) -> float:
    """Calculate sizing multiplier based on Meta-Labeling probability (AFML Chapter 10).

    Formula: S = max(0.0, 2 * p_meta - 1)
    - p_meta <= 0.50 -> Size = 0.0 (abort / no bet)
    - p_meta = 0.60 -> Size = 0.20
    - p_meta = 0.75 -> Size = 0.50
    - p_meta >= 0.90 -> Size = 0.80 - 1.0 (full conviction)
    """
    raw_size = max(0.0, 2.0 * float(meta_prob) - 1.0)
    return float(np.clip(raw_size * base_size, 0.0, 1.0))
