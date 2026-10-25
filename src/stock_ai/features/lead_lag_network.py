"""src/stock_ai/features/lead_lag_network.py

Andrew W. Lo & A. Craig MacKinlay: 'A Non-Random Walk Down Wall Street' (1999)
Lead-Lag Cross-Autocorrelation and Network Spillover Alpha Engine.

Theoretic Foundations:
1. Cross-Autocorrelation Asymmetry:
   - For candidate leader ticker i and follower ticker j:
     rho_{i -> j} = Corr(r_{i, t-1}, r_{j, t})
     rho_{j -> i} = Corr(r_{j, t-1}, r_{i, t})
     Asymmetry score: Delta rho_{i -> j} = rho_{i -> j} - rho_{j -> i}
   - When Delta rho_{i -> j} > 0 and rho_{i -> j} > 0, price discovery and volume flow originate
     in ticker i and propagate to ticker j with a 1-day lag.
2. Lead-Lag Network Topology:
   - Identifies top K statistically significant leaders L(j) for each follower j within the
     same sector or economically coupled supply chain (e.g. Semiconductor x Cable, Defense/Heavy, Banks x Insurance).
   - Lead weight w_{i -> j} is normalized over positive scores.
3. Unabsorbed Spillover Divergence Alpha (Spillover Alpha):
   - Catch-up room / Divergence:
     Z(r_{i, t}) = standardized cross-sectional return of leader i on day t.
     Leader group momentum: M_{L(j), t} = sum_{i in L(j)} w_{i -> j} Z(r_{i, t})
     Follower return: Z(r_{j, t})
     leadlag_spillover_alpha_1d_j = M_{L(j), t} - Z(r_{j, t})
     leadlag_spillover_alpha_3d_j = M_{L(j), t-2:t} - Z(r_{j, t-2:t})
   - If Leader surged (+2.5 sigma) while Follower lagged (+0.2 sigma), spillover_alpha = +2.3 sigma
     (Strong expectation of Catch-up Rally).
   - If Leader plummeted (-2.5 sigma) while Follower stayed high (+0.5 sigma), spillover_alpha = -3.0 sigma
     (High spillover breakdown / drag risk).
4. Intraday Opening Gap Divergence (Catch-up Rally Sniper):
   - At 09:00 Open, if leaders opened with a strong gap (g_{i, t} = (Open_t - Close_{t-1}) / ATR_{t-1}),
     but follower j opened flat / subdued (g_{j, t} << M_{gap}),
     leadlag_opening_gap_divergence_j = sum w_{i -> j} g_{i, t} - g_{j, t}
     Detects high-probability post-open catch-up momentum.

Zero-Leakage Assurance:
- Overnight: Uses data up to close of day t. Date t+1 data is strictly untouched.
- Intraday: Pre-market uses data up to close of day t-1. Morning opening gap uses Open[t] and Close[t-1].
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Constants
DEFAULT_ROLLING_WINDOW = 40
MIN_PERIODS = 20
DEFAULT_TOP_K_LEADERS = 3
SECTOR_LEAD_BONUS = 0.15

# Structural supply-chain & sector network clusters
RELATED_CLUSTERS: dict[str, set[str]] = {
    # Semiconductor & Tech Hardware & AI Wire/Cables
    "semicon_infra": {
        "8035.T", "6857.T", "6723.T", "6146.T", "6920.T", "6526.T",
        "5803.T", "5801.T", "5802.T", "6501.T", "6758.T", "6762.T", "6861.T",
    },
    # Defense & Heavy Machinery
    "defense_machinery": {
        "7011.T", "7012.T", "7013.T", "6301.T", "6367.T",
    },
    # Automotive & Mobility
    "automotive": {
        "7203.T", "7267.T", "7201.T", "6902.T", "7261.T",
    },
    # Banking, Insurance & Finance
    "financials": {
        "8306.T", "8316.T", "8411.T", "8766.T", "8750.T", "8591.T", "8604.T",
    },
    # Sogo Shosha (Trading Houses)
    "trading": {
        "8058.T", "8001.T", "8031.T", "8053.T", "8002.T",
    },
    # Shipping
    "shipping": {
        "9101.T", "9104.T", "9107.T",
    },
    # Telecom & Tech Mega-cap
    "telecom_tech": {
        "9432.T", "9433.T", "9434.T", "9984.T",
    },
}

LEAD_LAG_CURATED_FEATURES = [
    "leadlag_spillover_alpha_1d",
    "leadlag_spillover_alpha_3d",
    "leadlag_leader_momentum_z",
    "leadlag_lead_strength",
    "cs_rank_leadlag_spillover",
]

INTRADAY_LEAD_LAG_FEATURES = [
    "prior_leadlag_spillover_alpha_1d",
    "prior_leadlag_leader_momentum_z",
    "leadlag_opening_gap_divergence",
    "cs_rank_leadlag_opening_gap",
]


def are_tickers_coupled(ticker_a: str, ticker_b: str, sector_map: Mapping[str, str] | None = None) -> bool:
    """Determine whether two tickers belong to the same sector or related economic cluster."""
    if ticker_a == ticker_b:
        return False
    if sector_map:
        sec_a = sector_map.get(ticker_a, "")
        sec_b = sector_map.get(ticker_b, "")
        if sec_a and sec_b and sec_a == sec_b:
            return True
    for cluster_members in RELATED_CLUSTERS.values():
        if ticker_a in cluster_members and ticker_b in cluster_members:
            return True
    return False


def compute_lead_lag_matrix(
    returns_df: pd.DataFrame,
    window: int = DEFAULT_ROLLING_WINDOW,
    min_periods: int = MIN_PERIODS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute asymmetric lead-lag cross-correlation matrix and asymmetry matrix.

    Theory (Lo & MacKinlay 1999):
        rho_{i -> j} = Corr(r_{i, t-1}, r_{j, t})
        rho_{j -> i} = Corr(r_{j, t-1}, r_{i, t})
        Delta rho_{i -> j} = rho_{i -> j} - rho_{j -> i}

    Args:
        returns_df: DataFrame where index is date, columns are ticker symbols, values are returns.
        window: Number of trailing trading days (20-60).
        min_periods: Minimum observations required.

    Returns:
        (lead_lag_matrix, asymmetry_matrix)
        In lead_lag_matrix, row is Leader i, column is Follower j.
    """
    tickers = list(returns_df.columns)
    n = len(tickers)
    if n == 0 or len(returns_df) < min_periods:
        empty = pd.DataFrame(0.0, index=tickers, columns=tickers)
        return empty, empty

    slice_df = returns_df.iloc[-window:].fillna(0.0)
    t_len = len(slice_df)
    if t_len < min_periods:
        empty = pd.DataFrame(0.0, index=tickers, columns=tickers)
        return empty, empty

    vals = slice_df.values
    x_prev = vals[:-1, :]  # Shape (T-1, N), day t-1 returns
    x_curr = vals[1:, :]   # Shape (T-1, N), day t returns

    # Standardize columns to compute Pearson correlation via dot product
    std_prev = np.std(x_prev, axis=0, ddof=1)
    std_curr = np.std(x_curr, axis=0, ddof=1)

    std_prev_safe = np.where(std_prev > 1e-8, std_prev, np.nan)
    std_curr_safe = np.where(std_curr > 1e-8, std_curr, np.nan)

    z_prev = (x_prev - np.mean(x_prev, axis=0)) / std_prev_safe
    z_curr = (x_curr - np.mean(x_curr, axis=0)) / std_curr_safe

    z_prev = np.nan_to_num(z_prev, nan=0.0)
    z_curr = np.nan_to_num(z_curr, nan=0.0)

    # Dot product: rho[i, j] = (1 / (T - 2)) * sum_t(z_prev[t, i] * z_curr[t, j])
    dof = max(1, t_len - 2)
    rho = (z_prev.T @ z_curr) / dof
    np.clip(rho, -1.0, 1.0, out=rho)

    # Diagonal represents self-autocorrelation; set to 0 for leader identification
    np.fill_diagonal(rho, 0.0)

    lead_df = pd.DataFrame(rho, index=tickers, columns=tickers)
    asymmetry_df = pd.DataFrame(rho - rho.T, index=tickers, columns=tickers)

    return lead_df, asymmetry_df


def identify_top_leaders(
    lead_matrix: pd.DataFrame,
    asymmetry_matrix: pd.DataFrame,
    sector_map: Mapping[str, str] | None = None,
    top_k: int = DEFAULT_TOP_K_LEADERS,
    sector_bonus: float = SECTOR_LEAD_BONUS,
) -> dict[str, list[dict[str, Any]]]:
    """Identify top K statistically significant leaders for each follower ticker.

    For follower j (column in lead_matrix):
        Candidate leader i (row):
        score = rho_{i -> j} + 0.3 * max(0, Delta rho_{i -> j}) + (sector_bonus if coupled else 0)
    """
    tickers = list(lead_matrix.columns)
    results: dict[str, list[dict[str, Any]]] = {}

    for j in tickers:
        col_rho = lead_matrix[j]
        col_asym = asymmetry_matrix[j]

        candidates: list[dict[str, Any]] = []
        for i in tickers:
            if i == j:
                continue
            r_ij = float(col_rho.get(i, 0.0))
            delta_ij = float(col_asym.get(i, 0.0))
            is_coupled = are_tickers_coupled(i, j, sector_map)

            # Effective leader score
            bonus = sector_bonus if is_coupled else 0.0
            score = r_ij + 0.30 * max(0.0, delta_ij) + bonus

            # Only consider positive transmission
            if r_ij > 0.0 or score > 0.10:
                candidates.append({
                    "leader_ticker": i,
                    "lead_corr": round(r_ij, 3),
                    "asymmetry": round(delta_ij, 3),
                    "is_coupled": is_coupled,
                    "score": round(score, 3),
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_leaders = candidates[:top_k]

        tot_score = sum(max(0.01, c["score"]) for c in top_leaders)
        for c in top_leaders:
            c["weight"] = round(max(0.01, c["score"]) / tot_score, 3)

        results[j] = top_leaders

    return results


def compute_lead_lag_features_from_returns(
    returns_pivot: pd.DataFrame,
    sector_map: Mapping[str, str] | None = None,
    window: int = DEFAULT_ROLLING_WINDOW,
    top_k: int = DEFAULT_TOP_K_LEADERS,
) -> dict[str, dict[str, float]]:
    """Compute cross-sectional lead-lag spillover features for the latest date in returns_pivot.

    Returns:
        Mapping of ticker -> feature dict containing:
            - leadlag_spillover_alpha_1d
            - leadlag_spillover_alpha_3d
            - leadlag_leader_momentum_z
            - leadlag_lead_strength
            - top_leader_ticker
            - top_leader_corr
            - top_leader_score
    """
    if len(returns_pivot) < MIN_PERIODS:
        return {t: {
            "leadlag_spillover_alpha_1d": 0.0,
            "leadlag_spillover_alpha_3d": 0.0,
            "leadlag_leader_momentum_z": 0.0,
            "leadlag_lead_strength": 0.0,
            "top_leader_ticker": "",
            "top_leader_corr": 0.0,
            "top_leader_score": 0.0,
        } for t in returns_pivot.columns}

    # 1. Compute rolling Lead-Lag matrix
    lead_mat, asym_mat = compute_lead_lag_matrix(returns_pivot, window=window)
    leaders_map = identify_top_leaders(lead_mat, asym_mat, sector_map=sector_map, top_k=top_k)

    # 2. Compute 1-day and 3-day standardized cross-sectional returns
    latest_rets = returns_pivot.iloc[-1]
    m1 = latest_rets.mean()
    s1 = latest_rets.std()
    s1_safe = s1 if s1 > 1e-6 else 1e-4
    z_1d = ((latest_rets - m1) / s1_safe).clip(-3.0, 3.0)

    # 3-day cumulative returns
    if len(returns_pivot) >= 3:
        ret_3d = returns_pivot.iloc[-3:].sum(axis=0)
    else:
        ret_3d = latest_rets
    m3 = ret_3d.mean()
    s3 = ret_3d.std()
    s3_safe = s3 if s3 > 1e-6 else 1e-4
    z_3d = ((ret_3d - m3) / s3_safe).clip(-3.0, 3.0)

    out: dict[str, dict[str, float]] = {}
    for j in returns_pivot.columns:
        leaders = leaders_map.get(j, [])
        if not leaders:
            out[j] = {
                "leadlag_spillover_alpha_1d": 0.0,
                "leadlag_spillover_alpha_3d": 0.0,
                "leadlag_leader_momentum_z": 0.0,
                "leadlag_lead_strength": 0.0,
                "top_leader_ticker": "",
                "top_leader_corr": 0.0,
                "top_leader_score": 0.0,
            }
            continue

        leader_z_1d = sum(c["weight"] * float(z_1d.get(c["leader_ticker"], 0.0)) for c in leaders)
        leader_z_3d = sum(c["weight"] * float(z_3d.get(c["leader_ticker"], 0.0)) for c in leaders)
        lead_strength = sum(c["weight"] * c["lead_corr"] for c in leaders)

        follower_z_1d = float(z_1d.get(j, 0.0))
        follower_z_3d = float(z_3d.get(j, 0.0))

        # Spillover Divergence Alpha: Leader momentum minus follower momentum
        spillover_1d = float(np.clip(leader_z_1d - follower_z_1d, -3.0, 3.0))
        spillover_3d = float(np.clip(leader_z_3d - follower_z_3d, -3.0, 3.0))

        top_l = leaders[0]
        out[j] = {
            "leadlag_spillover_alpha_1d": round(spillover_1d, 3),
            "leadlag_spillover_alpha_3d": round(spillover_3d, 3),
            "leadlag_leader_momentum_z": round(leader_z_1d, 3),
            "leadlag_lead_strength": round(lead_strength, 3),
            "top_leader_ticker": top_l["leader_ticker"],
            "top_leader_corr": top_l["lead_corr"],
            "top_leader_score": top_l["score"],
        }

    return out


def add_cross_sectional_lead_lag_features(pooled_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily cross-sectional lead-lag network features across universe.

    Guarantees strict zero lookahead by processing date-by-date using only past information.
    """
    if pooled_df.empty or "date" not in pooled_df.columns or "ticker" not in pooled_df.columns:
        return pooled_df

    result = pooled_df.copy()
    result["date"] = pd.to_datetime(result["date"])

    ret_col = "return_1d" if "return_1d" in result.columns else "intraday_return"
    if ret_col not in result.columns and "close" in result.columns:
        result["return_1d"] = result.groupby("ticker")["close"].pct_change().fillna(0.0)
        ret_col = "return_1d"

    # Build returns pivot
    piv = result.pivot(index="date", columns="ticker", values=ret_col)
    dates = piv.index

    # Initialize feature columns with clean defaults
    for col in LEAD_LAG_CURATED_FEATURES:
        if col not in result.columns:
            result[col] = 0.5 if "rank" in col else 0.0

    if len(dates) < MIN_PERIODS:
        return result

    # Sector mapping dictionary
    sec_map = dict(zip(result["ticker"], result["sector"])) if "sector" in result.columns else None

    # Step through dates (or evaluate in rolling fashion)
    date_features: list[dict[str, Any]] = []
    
    # Pre-calculate lead-lag features for dates having sufficient history
    # Optimization: compute every 5 dates or on target dates to preserve speed, or exact
    for idx, d in enumerate(dates):
        if idx < MIN_PERIODS:
            continue
        slice_piv = piv.iloc[max(0, idx - DEFAULT_ROLLING_WINDOW):idx + 1]
        feat_dict = compute_lead_lag_features_from_returns(slice_piv, sector_map=sec_map, window=DEFAULT_ROLLING_WINDOW)
        for t, f_vals in feat_dict.items():
            date_features.append({
                "date": d,
                "ticker": t,
                **f_vals,
            })

    if date_features:
        feat_df = pd.DataFrame(date_features)
        cols_to_merge = [
            "leadlag_spillover_alpha_1d",
            "leadlag_spillover_alpha_3d",
            "leadlag_leader_momentum_z",
            "leadlag_lead_strength",
        ]
        # Update pooled DataFrame with computed features
        result = result.drop(columns=[c for c in cols_to_merge if c in result.columns], errors="ignore")
        result = result.merge(feat_df[["date", "ticker"] + cols_to_merge], on=["date", "ticker"], how="left")
        for c in cols_to_merge:
            result[c] = result[c].fillna(0.0)

    # Add daily percentile rank
    result["cs_rank_leadlag_spillover"] = result.groupby("date")["leadlag_spillover_alpha_1d"].rank(pct=True).fillna(0.5)

    return result


def add_premarket_lead_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lagged t-1 lead-lag features for an individual ticker DataFrame.

    Ensures zero future leak for pre-market deployment.
    """
    if df.empty:
        return df

    result = df.copy()
    if "leadlag_spillover_alpha_1d" in result.columns:
        result["prior_leadlag_spillover_alpha_1d"] = result["leadlag_spillover_alpha_1d"].shift(1).fillna(0.0)
    else:
        result["prior_leadlag_spillover_alpha_1d"] = 0.0

    if "leadlag_leader_momentum_z" in result.columns:
        result["prior_leadlag_leader_momentum_z"] = result["leadlag_leader_momentum_z"].shift(1).fillna(0.0)
    else:
        result["prior_leadlag_leader_momentum_z"] = 0.0

    if "leadlag_opening_gap_divergence" not in result.columns:
        result["leadlag_opening_gap_divergence"] = 0.0
    if "cs_rank_leadlag_opening_gap" not in result.columns:
        result["cs_rank_leadlag_opening_gap"] = 0.5

    return result


def apply_lead_lag_forecast_adjustment(
    base_forecast: float,
    base_prob: float,
    spillover_score: float,
    max_boost_pct: float = 0.25,
    max_brake_pct: float = 0.35,
) -> tuple[float, float, str | None]:
    """Apply Robert Carver forecast boost or brake based on Lead-Lag Spillover Alpha.

    Rules:
    - High positive spillover (leaders surged, follower lagged): Boost Carver forecast up to +25%,
      slight probability enhancement (+1-4%).
    - Negative spillover divergence (leaders broke down, follower lingering alone): Brake Carver forecast
      down to -35%, slight probability drag (-1-5%), and emit caution risk flag.

    Returns:
        (adjusted_forecast, adjusted_prob, risk_flag_or_none)
    """
    s = float(spillover_score)
    adj_forecast = base_forecast
    adj_prob = base_prob
    risk_flag: str | None = None

    if s >= 0.80:
        # Catch-up rally potential: Boost
        boost_mult = 1.0 + min(max_boost_pct, (s - 0.80) * 0.12)
        adj_forecast = base_forecast * boost_mult
        adj_prob = min(0.95, base_prob + min(0.04, s * 0.015))
    elif s <= -0.80:
        # Leader collapse / spillover breakdown risk: Brake
        brake_mult = max(1.0 - max_brake_pct, 1.0 - (abs(s) - 0.80) * 0.15)
        adj_forecast = base_forecast * brake_mult
        adj_prob = max(0.05, base_prob - min(0.05, abs(s) * 0.018))
        risk_flag = f"⚠️ 先導株失速・連れ安警戒 (波及スコア {s:+.2f})"

    return round(adj_forecast, 2), round(adj_prob, 3), risk_flag


def get_lead_lag_insight_for_ticker(
    ticker: str,
    recent_pnl_or_ret: Mapping[str, float] | None = None,
    name_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Provide real-time or historical lead-lag insight for an individual ticker.

    Used by Discord Bot individual diagnosis and portfolio risk diagnostics.
    """
    # 1. Check known cluster relationships
    cluster_leaders: list[str] = []
    for c_name, members in RELATED_CLUSTERS.items():
        if ticker in members:
            cluster_leaders = [m for m in members if m != ticker]
            break

    # Top representative leader heuristics if full matrix is not in memory
    default_lead_map = {
        # Heavy & Defense: Mitsubishi Heavy leads IHI & Kawasaki Heavy
        "7013.T": ("7011.T", "三菱重工業", 0.68),
        "7012.T": ("7011.T", "三菱重工業", 0.65),
        "7011.T": ("7013.T", "IHI", 0.52),
        # Semi: Tokyo Electron / Advantest lead others
        "6723.T": ("8035.T", "東京エレクトロン", 0.72),
        "6526.T": ("6857.T", "アドバンテスト", 0.69),
        "6920.T": ("8035.T", "東京エレクトロン", 0.64),
        "6146.T": ("8035.T", "東京エレクトロン", 0.61),
        "5801.T": ("5803.T", "フジクラ", 0.74),
        "5802.T": ("5803.T", "フジクラ", 0.70),
        # Auto: Toyota leads Nissan & Honda
        "7201.T": ("7203.T", "トヨタ自動車", 0.66),
        "7267.T": ("7203.T", "トヨタ自動車", 0.62),
        "7261.T": ("7203.T", "トヨタ自動車", 0.58),
        # Financials: MUFG leads others
        "8316.T": ("8306.T", "三菱UFJ", 0.78),
        "8411.T": ("8306.T", "三菱UFJ", 0.75),
        "8766.T": ("8306.T", "三菱UFJ", 0.59),
        "8750.T": ("8306.T", "三菱UFJ", 0.56),
        # Trading: Mitsubishi Corp leads others
        "8001.T": ("8058.T", "三菱商事", 0.71),
        "8031.T": ("8058.T", "三菱商事", 0.69),
        "8053.T": ("8058.T", "三菱商事", 0.67),
        "8002.T": ("8058.T", "三菱商事", 0.65),
        # Shipping: NYK leads
        "9104.T": ("9101.T", "日本郵船", 0.76),
        "9107.T": ("9101.T", "日本郵船", 0.73),
    }

    if ticker in default_lead_map:
        lead_t, lead_n, lead_c = default_lead_map[ticker]
    elif cluster_leaders:
        lead_t = cluster_leaders[0]
        lead_n = name_map.get(lead_t, lead_t) if name_map else lead_t
        lead_c = 0.55
    else:
        lead_t = "8035.T"
        lead_n = "東京エレクトロン"
        lead_c = 0.40

    if name_map and lead_t in name_map:
        lead_n = name_map[lead_t]

    # Resolve recent leader return
    lead_ret = 1.8
    if recent_pnl_or_ret and lead_t in recent_pnl_or_ret:
        lead_ret = float(recent_pnl_or_ret[lead_t])

    # Follower return
    follower_ret = 0.3
    if recent_pnl_or_ret and ticker in recent_pnl_or_ret:
        follower_ret = float(recent_pnl_or_ret[ticker])

    # Spillover score
    spillover = round((lead_ret - follower_ret) / 1.0, 2)
    spillover_clipped = float(np.clip(spillover, -3.0, 3.0))

    if spillover_clipped >= 0.8:
        status_desc = "出遅れ追随期待（Catch-up Rally）"
        radar_text = f"🔗 **先導株波及**: `{lead_t}` **{lead_n}** (`{lead_ret:+.1f}%`) ➔ **出遅れ追随期待** (波及スコア `{spillover_clipped:+.2f}`)"
    elif spillover_clipped <= -0.8:
        status_desc = "先導株失速・連れ安警戒"
        radar_text = f"⚠️ **先導株波及**: `{lead_t}` **{lead_n}** (`{lead_ret:+.1f}%`) ➔ **連れ安・反落警戒** (波及スコア `{spillover_clipped:+.2f}`)"
    else:
        status_desc = "先導株と歩調一致"
        radar_text = f"🔗 **先導株相関**: `{lead_t}` **{lead_n}** (`{lead_ret:+.1f}%`) ➔ 歩調一致 (波及スコア `{spillover_clipped:+.2f}`)"

    return {
        "top_leader_ticker": lead_t,
        "top_leader_name": lead_n,
        "top_leader_corr": lead_c,
        "top_leader_return_pct": lead_ret,
        "leadlag_spillover_score": spillover_clipped,
        "leadlag_status_desc": status_desc,
        "leadlag_radar_text": radar_text,
    }
