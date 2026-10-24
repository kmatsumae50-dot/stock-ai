"""src/stock_ai/features/hrp_allocation.py

Marcos López de Prado: Hierarchical Risk Parity (HRP)
From 'Machine Learning for Asset Managers' (2019) & 'Building Diversified Portfolios that Outperform Out-of-Sample' (2016).

HRP addresses the major shortcomings of Markowitz Mean-Variance Optimization:
1. Avoids covariance matrix inversion (no unstable pseudoinverse / condition number issues).
2. Tree clustering (デンドログラム) identifies hierarchical correlation structure.
3. Quasi-diagonalization (Seriation) orders assets so similar assets are adjacent.
4. Recursive Bisection allocates weights top-down using inverse cluster variance.
5. Hybrid integration with Robert Carver's Forecast Scaling and Giuseppe Paleologo's Cluster Dispersion.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


def compute_correlation_distance_matrix(
    corr: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Prado's correlation distance metric and distance between correlation vectors.

    Prado (2016):
        d_{i,j} = sqrt(0.5 * (1 - rho_{i,j}))
        D_{i,j} = sqrt(sum_k (d_{k,i} - d_{k,j})^2)

    Args:
        corr: Correlation matrix (N x N) with values in [-1, 1].

    Returns:
        tuple (dist_matrix, condensed_dist_matrix):
            - dist_matrix: (N x N) distance matrix D.
            - condensed_dist_matrix: 1D condensed array for scipy linkage.
    """
    rho = np.asarray(corr, dtype=float)
    # Clip numerical precision noise
    rho = np.clip(rho, -1.0, 1.0)
    np.fill_diagonal(rho, 1.0)

    # Primary correlation distance
    d = np.sqrt(np.clip(0.5 * (1.0 - rho), 0.0, 1.0))
    np.fill_diagonal(d, 0.0)

    # Distance between correlation distance column vectors (Euclidean distance on D)
    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n)), np.zeros(0)

    # Compute Euclidean distance between column vectors
    d_diff = d[:, :, np.newaxis] - d[:, np.newaxis, :]
    dist_matrix = np.sqrt(np.sum(d_diff**2, axis=0))
    np.fill_diagonal(dist_matrix, 0.0)

    # Condensed distance form for scipy.cluster.hierarchy
    condensed = squareform(dist_matrix, checks=False)
    return dist_matrix, condensed


def get_quasi_diag(link: np.ndarray) -> list[int]:
    """Sort clustered items by distance using recursive dendrogram traversal (Seriation).

    Prado (2016, 2019):
        Reorders the leaves of the hierarchical clustering tree such that the most
        similar items are adjacent to each other, quasi-diagonalizing the covariance matrix.

    Args:
        link: Hierarchical clustering linkage matrix from scipy.cluster.hierarchy.linkage.

    Returns:
        List of integer asset indices in quasi-diagonal order.
    """
    if link.size == 0:
        return [0]

    link_int = link.astype(int)
    sort_ix = pd.Series([link_int[-1, 0], link_int[-1, 1]])
    num_items = int(link_int[-1, 3])  # Total original leaves in tree

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link_int[j, 0]
        df0 = pd.Series(link_int[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return [int(x) for x in sort_ix.tolist()]


def get_cluster_var(cov: np.ndarray, cluster_indices: Sequence[int]) -> float:
    """Calculate the variance of a cluster under inverse-variance internal weighting.

    Prado (2016):
        w_tilde_i = (1 / sigma_i^2) / sum_j (1 / sigma_j^2)
        V_cluster = w_tilde^T * Sigma_cluster * w_tilde

    Args:
        cov: Full covariance matrix (N x N).
        cluster_indices: Indices of assets belonging to the cluster.

    Returns:
        Cluster variance scalar.
    """
    indices = list(cluster_indices)
    if not indices:
        return 1e-4

    sub_cov = cov[np.ix_(indices, indices)]
    diag = np.diag(sub_cov)
    safe_diag = np.where(diag > 1e-8, diag, 1e-4)

    inv_var = 1.0 / safe_diag
    inv_var_sum = float(np.sum(inv_var))
    if inv_var_sum <= 1e-12:
        w = np.ones(len(indices)) / len(indices)
    else:
        w = inv_var / inv_var_sum

    cluster_variance = float(np.dot(np.dot(w, sub_cov), w))
    return max(cluster_variance, 1e-8)


def get_rec_bisection(cov: np.ndarray, sort_ix: Sequence[int]) -> np.ndarray:
    """Recursively bisect ordered assets and allocate weights using inverse cluster variance.

    Prado (2016, 2019):
        1. Start with weight 1.0 for the entire cluster.
        2. Bisect into left and right subclusters C1 and C2.
        3. alpha_1 = 1 - V_1 / (V_1 + V_2) = V_2 / (V_1 + V_2).
        4. w[C1] *= alpha_1, w[C2] *= (1 - alpha_1).
        5. Repeat recursively until subclusters are singletons.

    Args:
        cov: Full covariance matrix (N x N).
        sort_ix: Ordered indices from quasi-diagonalization.

    Returns:
        HRP weights array for the assets in sort_ix order (sums to 1.0).
    """
    ordered = list(sort_ix)
    n = len(ordered)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([1.0], dtype=float)

    # Weights indexed by original asset index
    w_series = pd.Series(1.0, index=ordered, dtype=float)
    clusters: list[list[int]] = [ordered]

    while clusters:
        # Bisect all clusters with length > 1
        new_clusters: list[list[int]] = []
        for c in clusters:
            if len(c) > 1:
                mid = len(c) // 2
                c_left = c[:mid]
                c_right = c[mid:]
                new_clusters.extend([c_left, c_right])

        if not new_clusters:
            break

        # Allocate between pairs of subclusters
        for i in range(0, len(new_clusters), 2):
            c_left = new_clusters[i]
            c_right = new_clusters[i + 1]

            var_left = get_cluster_var(cov, c_left)
            var_right = get_cluster_var(cov, c_right)

            tot_var = var_left + var_right
            if tot_var <= 1e-12:
                alpha = 0.5
            else:
                alpha = 1.0 - (var_left / tot_var)
                alpha = float(np.clip(alpha, 0.01, 0.99))

            w_series.loc[c_left] *= alpha
            w_series.loc[c_right] *= 1.0 - alpha

        clusters = new_clusters

    # Normalize to ensure exactly sum = 1.0
    weights = w_series.loc[ordered].values
    tot_w = float(np.sum(weights))
    if tot_w > 0:
        weights = weights / tot_w
    else:
        weights = np.ones(n, dtype=float) / n

    return weights


def compute_hrp_weights(
    cov_matrix: pd.DataFrame | np.ndarray | None = None,
    returns_df: pd.DataFrame | None = None,
    tickers: Sequence[str] | None = None,
    linkage_method: str = "single",
) -> pd.Series:
    """Compute Marcos López de Prado's Hierarchical Risk Parity (HRP) portfolio weights.

    Accepts either an explicit covariance matrix or a historical returns DataFrame.

    Steps:
        1. Construct covariance & correlation matrices.
        2. Compute tree clustering via correlation distance matrix.
        3. Quasi-diagonalize assets via Seriation traversal.
        4. Apply recursive bisection to compute inverse cluster-variance weights.

    Args:
        cov_matrix: Covariance matrix (pd.DataFrame or np.ndarray).
        returns_df: Historical returns DataFrame (columns = tickers).
        tickers: List of ticker symbols corresponding to matrix rows/columns.
        linkage_method: Hierarchical clustering linkage ('single', 'ward', 'average').

    Returns:
        pd.Series of HRP weights indexed by ticker symbols, summing to 1.0.
    """
    # 1. Determine tickers and covariance matrix
    if returns_df is not None and not returns_df.empty:
        t_list = list(returns_df.columns) if tickers is None else list(tickers)
        ret_sub = returns_df[t_list].dropna(how="all").fillna(0.0)
        cov = ret_sub.cov().values
        t_names = t_list
    elif cov_matrix is not None:
        if isinstance(cov_matrix, pd.DataFrame):
            t_names = list(cov_matrix.columns) if tickers is None else list(tickers)
            cov = cov_matrix.values
        else:
            cov = np.asarray(cov_matrix, dtype=float)
            t_names = list(tickers) if tickers is not None else [f"asset_{i}" for i in range(cov.shape[0])]
    else:
        raise ValueError("Either cov_matrix or returns_df must be provided.")

    n = len(t_names)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=t_names, dtype=float)

    # Special optimized handling for N = 2
    if n == 2:
        var1 = float(max(cov[0, 0], 1e-6))
        var2 = float(max(cov[1, 1], 1e-6))
        w1 = var2 / (var1 + var2)
        w2 = 1.0 - w1
        return pd.Series([w1, w2], index=t_names, dtype=float)

    # 2. Extract correlation matrix
    diag = np.sqrt(np.clip(np.diag(cov), 1e-8, None))
    outer_diag = np.outer(diag, diag)
    corr = np.clip(cov / (outer_diag + 1e-12), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    # 3. Correlation distance & Linkage
    _, condensed_dist = compute_correlation_distance_matrix(corr)
    try:
        link = sch.linkage(condensed_dist, method=linkage_method)
    except Exception as exc:
        logger.debug("HRP linkage fallback to 'average': %s", exc)
        link = sch.linkage(condensed_dist, method="average")

    # 4. Quasi-diagonalization
    sort_ix = get_quasi_diag(link)

    # 5. Recursive bisection
    hrp_weights_ordered = get_rec_bisection(cov, sort_ix)

    # Map back to original ticker order
    weight_map = {t_names[idx]: float(hrp_weights_ordered[i]) for i, idx in enumerate(sort_ix)}
    return pd.Series([weight_map.get(t, 1.0 / n) for t in t_names], index=t_names, dtype=float)


def compute_carver_paleologo_hrp_weights(
    tickers: Sequence[str],
    forecasts: Sequence[float],
    volatilities: Sequence[float],
    sector_map: Mapping[str, str] | None = None,
    correlation_matrix: pd.DataFrame | None = None,
    max_weight: float = 0.50,
    sector_penalty_decay: float = 1.0,
    drawdown_throttle: float = 1.0,
) -> dict[str, float]:
    """Unified Quantitative Allocation Engine (6-Masterpiece Complete Integration).

    Blends:
        1. Marcos López de Prado: Hierarchical Risk Parity (Tree clustering & inverse cluster-variance).
        2. Giuseppe Paleologo: Sector & correlation cluster dispersion penalty.
        3. Robert Carver: Forecast scaling & capping, drawdown throttling, max weight capping.

    Args:
        tickers: Candidate ticker symbols.
        forecasts: Carver scaled & capped forecasts (e.g. baseline 10.0, cap 20.0).
        volatilities: Daily volatilities (sigma) for each candidate.
        sector_map: Optional mapping of ticker -> sector.
        correlation_matrix: Optional pairwise correlation matrix.
        max_weight: Maximum position weight per individual ticker.
        sector_penalty_decay: Paleologo cluster penalty factor for crowded sectors.
        drawdown_throttle: Carver drawdown throttling multiplier in [0.20, 1.0].

    Returns:
        dict of ticker -> final allocated position weight.
    """
    t_list = list(tickers)
    n = len(t_list)
    if n == 0:
        return {}
    if n == 1:
        t = t_list[0]
        f = max(0.0, float(forecasts[0]))
        f_mult = np.clip(f / 10.0, 0.0, 2.0)
        base_w = min(max_weight, f_mult * max_weight)
        final_w = base_w * float(np.clip(drawdown_throttle, 0.10, 1.0))
        return {t: float(final_w)}

    f_arr = np.array(forecasts, dtype=float)
    v_arr = np.array(volatilities, dtype=float)
    safe_v = np.clip(v_arr, 0.010, 0.10)

    # 1. Apply Paleologo cluster penalty to crowded sectors / correlated candidates
    seen_sectors: set[str] = set()
    seen_tickers: list[str] = []
    adj_forecasts: list[float] = []
    sec_m = sector_map or {}

    for t, f in zip(t_list, f_arr):
        sec = sec_m.get(t, "")
        is_crowded = False
        if sec and sec in seen_sectors:
            is_crowded = True
        elif correlation_matrix is not None and t in correlation_matrix.index:
            for prev_t in seen_tickers:
                if prev_t in correlation_matrix.columns:
                    if float(correlation_matrix.loc[t, prev_t]) >= 0.70:
                        is_crowded = True
                        break

        if is_crowded:
            adj_forecasts.append(float(f) * sector_penalty_decay)
        else:
            adj_forecasts.append(float(f))
            if sec:
                seen_sectors.add(sec)
            seen_tickers.append(t)

    adj_f_arr = np.array(adj_forecasts, dtype=float)

    # 2. Construct Covariance Matrix for HRP
    # Sigma_{i,j} = rho_{i,j} * sigma_i * sigma_j
    if correlation_matrix is not None and all(t in correlation_matrix.index for t in t_list):
        rho = correlation_matrix.loc[t_list, t_list].values
    else:
        # Default mild correlation among prime stocks (rho = 0.35 if different, 1.0 on diagonal)
        rho = np.full((n, n), 0.35, dtype=float)
        np.fill_diagonal(rho, 1.0)
        # If same sector, higher base correlation (rho = 0.65)
        for i in range(n):
            for j in range(i + 1, n):
                if sec_m.get(t_list[i]) and sec_m.get(t_list[i]) == sec_m.get(t_list[j]):
                    rho[i, j] = 0.65
                    rho[j, i] = 0.65

    cov = rho * np.outer(safe_v, safe_v)

    # 3. Compute Prado's HRP Base Weights
    hrp_series = compute_hrp_weights(cov_matrix=cov, tickers=t_list)
    hrp_base = hrp_series.values

    # 4. Integrate Carver Forecast Strength Multiplier
    # Baseline forecast 10.0 gives multiplier 1.0x, 15.0 gives 1.5x, 20.0 gives 2.0x
    f_mult = np.clip(adj_f_arr / 10.0, 0.0, 2.0)
    raw_signals = hrp_base * f_mult

    tot_signal = float(np.sum(raw_signals))
    if tot_signal <= 1e-9:
        return {t: 0.0 for t in t_list}

    # Target total exposure scales with average forecast conviction
    avg_f_mult = float(np.mean(f_mult))
    # Target leverage: up to 1.0 (or max_weight * n)
    target_leverage = min(1.0, avg_f_mult * (max_weight * n))

    unclipped = (raw_signals / tot_signal) * target_leverage
    clipped = np.clip(unclipped, 0.0, max_weight)

    # Ensure total does not exceed 1.0
    tot_clipped = float(np.sum(clipped))
    if tot_clipped > 1.0:
        clipped = clipped * (1.0 / tot_clipped)

    # 5. Apply Carver Drawdown Throttle
    final_weights = clipped * float(np.clip(drawdown_throttle, 0.10, 1.0))
    return {t: float(w) for t, w in zip(t_list, final_weights)}
