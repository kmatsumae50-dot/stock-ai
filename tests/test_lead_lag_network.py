"""tests/test_lead_lag_network.py

Comprehensive unit tests for Series #8:
MIT Professors Andrew W. Lo & A. Craig MacKinlay 'A Non-Random Walk Down Wall Street' (1999)
Lead-Lag Cross-Autocorrelation and Network Spillover Alpha Engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_ai.features.lead_lag_network import (
    apply_lead_lag_forecast_adjustment,
    are_tickers_coupled,
    compute_lead_lag_features_from_returns,
    compute_lead_lag_matrix,
    get_lead_lag_insight_for_ticker,
    identify_top_leaders,
)


def test_synthetic_1day_lag_detection():
    """Verify that artificial 1-day lag transmission (A -> B) detects rho_{A->B} > 0.70."""
    np.random.seed(42)
    n_days = 80

    # Ticker A: i.i.d. returns
    r_a = np.random.normal(0.001, 0.02, size=n_days)

    # Ticker B: strongly driven by Ticker A with a 1-day lag
    noise_b = np.random.normal(0.0, 0.005, size=n_days)
    r_b = np.zeros(n_days)
    r_b[1:] = 0.90 * r_a[:-1] + noise_b[1:]

    # Ticker C: independent random noise
    r_c = np.random.normal(0.001, 0.02, size=n_days)

    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    returns_df = pd.DataFrame(
        {"A": r_a, "B": r_b, "C": r_c},
        index=dates,
    )

    lead_mat, asym_mat = compute_lead_lag_matrix(returns_df, window=60, min_periods=20)

    # 1. Lead matrix: row is Leader, column is Follower
    rho_a_to_b = lead_mat.loc["A", "B"]
    rho_b_to_a = lead_mat.loc["B", "A"]
    delta_a_to_b = asym_mat.loc["A", "B"]

    # Verify strong transmission from A to B (> 0.70)
    assert rho_a_to_b > 0.70, f"Expected rho(A->B) > 0.70, got {rho_a_to_b:.3f}"
    # Reverse transmission should be near zero or much smaller
    assert abs(rho_b_to_a) < 0.35, f"Expected small reverse rho(B->A), got {rho_b_to_a:.3f}"
    # Asymmetry Delta rho should be strongly positive (> 0.50)
    assert delta_a_to_b > 0.50, f"Expected Delta rho(A->B) > 0.50, got {delta_a_to_b:.3f}"

    # Diagonal self-autocorrelation must be 0
    assert lead_mat.loc["A", "A"] == 0.0
    assert lead_mat.loc["B", "B"] == 0.0
    assert lead_mat.loc["C", "C"] == 0.0


def test_identify_top_leaders_and_weights():
    """Verify top leader identification, sector coupling bonus, and normalized weights."""
    np.random.seed(42)
    n_days = 60
    r_a = np.random.normal(0.001, 0.02, size=n_days)
    r_b = np.zeros(n_days)
    r_b[1:] = 0.85 * r_a[:-1] + np.random.normal(0.0, 0.005, size=n_days)[1:]
    r_c = np.random.normal(0.001, 0.02, size=n_days)

    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    returns_df = pd.DataFrame({"A": r_a, "B": r_b, "C": r_c}, index=dates)

    lead_mat, asym_mat = compute_lead_lag_matrix(returns_df, window=50)
    sector_map = {"A": "Defense", "B": "Defense", "C": "Retail"}
    leaders_map = identify_top_leaders(lead_mat, asym_mat, sector_map=sector_map, top_k=2)

    # For follower B: A must be top leader
    b_leaders = leaders_map["B"]
    assert len(b_leaders) >= 1
    assert b_leaders[0]["leader_ticker"] == "A"
    assert b_leaders[0]["is_coupled"] is True
    assert b_leaders[0]["score"] > 0.70

    # Weights must sum to 1.0
    tot_w = sum(l["weight"] for l in b_leaders)
    assert pytest.approx(tot_w, abs=1e-2) == 1.0


def test_spillover_divergence_alpha_calculation():
    """Verify unabsorbed catch-up divergence (positive) and leader collapse (negative)."""
    np.random.seed(42)
    n_days = 50
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")

    # Generate historical returns
    rets = pd.DataFrame(
        np.random.normal(0.0, 0.015, size=(n_days, 5)),
        index=dates,
        columns=["7011.T", "7013.T", "7203.T", "8035.T", "6857.T"],
    )

    # Set historical positive lead correlation between 7011 (Leader) and 7013 (Follower)
    for i in range(1, n_days - 1):
        rets.loc[dates[i], "7013.T"] = 0.75 * rets.loc[dates[i - 1], "7011.T"]

    # Scenario 1: On latest date, Leader 7011 surges (+3.5%) while Follower 7013 lags (0.0%)
    rets.iloc[-1] = 0.0
    rets.iloc[-1, rets.columns.get_loc("7011.T")] = 0.035
    rets.iloc[-1, rets.columns.get_loc("7013.T")] = 0.001

    sec_map = {"7011.T": "Defense", "7013.T": "Defense"}
    feats = compute_lead_lag_features_from_returns(rets, sector_map=sec_map, window=40)

    # Follower 7013 should have high positive spillover alpha (Leader surged, follower lagged)
    f_7013 = feats["7013.T"]
    assert f_7013["leadlag_spillover_alpha_1d"] > 0.50, f"Expected positive spillover, got {f_7013}"
    assert f_7013["leadlag_leader_momentum_z"] > 0.50

    # Scenario 2: Leader 7011 collapses (-3.5%) while Follower 7013 stands high (+1.0%)
    rets.iloc[-1, rets.columns.get_loc("7011.T")] = -0.035
    rets.iloc[-1, rets.columns.get_loc("7013.T")] = +0.010

    feats_neg = compute_lead_lag_features_from_returns(rets, sector_map=sec_map, window=40)
    f_7013_neg = feats_neg["7013.T"]
    assert f_7013_neg["leadlag_spillover_alpha_1d"] < -0.50, f"Expected negative spillover, got {f_7013_neg}"


def test_zero_future_leakage():
    """Verify that altering future data at t+1 has strictly zero impact on features at day t."""
    np.random.seed(42)
    n_days = 60
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rets = pd.DataFrame(
        np.random.normal(0.001, 0.02, size=(n_days, 4)),
        index=dates,
        columns=["A", "B", "C", "D"],
    )

    t_idx = 45
    slice_t = rets.iloc[:t_idx + 1]
    feats_orig = compute_lead_lag_features_from_returns(slice_t, window=30)

    # Now simulate future days t+1..t+14 with wild market swings
    rets_future_perturbed = rets.copy()
    rets_future_perturbed.iloc[t_idx + 1:] = np.random.normal(0.10, 0.08, size=rets.iloc[t_idx + 1:].shape)

    # When evaluating at day t, we slice up to t_idx + 1
    slice_t_perturbed = rets_future_perturbed.iloc[:t_idx + 1]
    feats_after = compute_lead_lag_features_from_returns(slice_t_perturbed, window=30)

    for ticker in ["A", "B", "C", "D"]:
        assert feats_orig[ticker]["leadlag_spillover_alpha_1d"] == feats_after[ticker]["leadlag_spillover_alpha_1d"]
        assert feats_orig[ticker]["leadlag_leader_momentum_z"] == feats_after[ticker]["leadlag_leader_momentum_z"]
        assert feats_orig[ticker]["top_leader_score"] == feats_after[ticker]["top_leader_score"]


def test_apply_lead_lag_forecast_adjustment():
    """Verify Robert Carver forecast boost on positive spillover and brake on negative spillover."""
    base_f = 10.0
    base_p = 0.55

    # High positive spillover (+1.5 sigma): Boost
    boost_f, boost_p, risk_flag = apply_lead_lag_forecast_adjustment(base_f, base_p, spillover_score=+1.5)
    assert boost_f > base_f
    assert boost_p > base_p
    assert risk_flag is None

    # High negative spillover (-1.8 sigma): Brake & Caution flag
    brake_f, brake_p, risk_flag = apply_lead_lag_forecast_adjustment(base_f, base_p, spillover_score=-1.8)
    assert brake_f < base_f
    assert brake_p < base_p
    assert risk_flag is not None
    assert "先導株失速" in risk_flag or "波及" in risk_flag


def test_get_lead_lag_insight_for_ticker():
    """Verify lead-lag radar insight generation for individual tickers."""
    # Mitsubishi Heavy -> IHI
    insight_ihi = get_lead_lag_insight_for_ticker("7013.T", recent_pnl_or_ret={"7011.T": 2.8, "7013.T": 0.5})
    assert insight_ihi["top_leader_ticker"] == "7011.T"
    assert "三菱重工業" in insight_ihi["top_leader_name"]
    assert insight_ihi["leadlag_spillover_score"] > 0
    assert "先導株波及" in insight_ihi["leadlag_radar_text"]
    assert "出遅れ追随期待" in insight_ihi["leadlag_status_desc"]

    # Toyota -> Nissan
    insight_nissan = get_lead_lag_insight_for_ticker("7201.T", recent_pnl_or_ret={"7203.T": 1.5, "7201.T": 0.2})
    assert insight_nissan["top_leader_ticker"] == "7203.T"
    assert "トヨタ自動車" in insight_nissan["top_leader_name"]


def test_are_tickers_coupled():
    """Verify sector and cluster coupling heuristics."""
    # Semi x Cables coupling
    assert are_tickers_coupled("8035.T", "6857.T") is True
    assert are_tickers_coupled("8035.T", "5803.T") is True
    # Defense coupling
    assert are_tickers_coupled("7011.T", "7013.T") is True
    # Non-coupled tickers
    assert are_tickers_coupled("7011.T", "9101.T") is False
