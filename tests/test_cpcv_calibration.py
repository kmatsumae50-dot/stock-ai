"""test_cpcv_calibration.py

Unit tests for Marcos Lopez de Prado AFML:
1. Isotonic & Platt probability calibration (monotonicity, Brier score minimization, zero leakage).
2. Purged & Embargoed Cross-Validation (CPCV) folds (purge of overlapping target horizons, embargo barrier).
3. Deflated Sharpe Ratio (DSR) & Probabilistic Sharpe Ratio (PSR) mathematical precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_ai.features.cpcv_calibration import (
    CausalHybridCalibrator,
    calibrate_probabilities_causal,
    calibrate_time_series_walk_forward,
    compute_brier_score,
    compute_brier_skill_score,
    compute_cpcv_path_stability,
    compute_deflated_sharpe_ratio,
    compute_probabilistic_sharpe_ratio,
    generate_cpcv_folds,
)


def test_brier_score_properties():
    """Verify Brier Score calculation and bounds."""
    y_true = np.array([1, 1, 0, 0])
    # Perfect predictions -> 0.0
    assert compute_brier_score(y_true, np.array([1.0, 1.0, 0.0, 0.0])) == 0.0
    # Completely wrong predictions -> 1.0
    assert compute_brier_score(y_true, np.array([0.0, 0.0, 1.0, 1.0])) == 1.0
    # Random coin toss 0.5 -> 0.25
    assert compute_brier_score(y_true, np.array([0.5, 0.5, 0.5, 0.5])) == 0.25


def test_causal_hybrid_calibrator_monotonicity_and_brier_improvement():
    """Verify that calibrated probabilities are strictly monotonic non-decreasing and improve Brier score."""
    np.random.seed(42)
    # Generate uncalibrated probabilities with severe overconfidence (e.g. sigmoid stretched too far)
    n = 200
    true_latent = np.random.normal(0, 1, n)
    # True win probability is moderate (0.35 to 0.65)
    true_prob = 1.0 / (1.0 + np.exp(-0.8 * true_latent))
    y = (np.random.rand(n) < true_prob).astype(int)

    # Model outputs overconfident raw probabilities (peaked near 0 and 1)
    raw_p = 1.0 / (1.0 + np.exp(-2.5 * true_latent))

    raw_brier = compute_brier_score(y, raw_p)

    calibrator = CausalHybridCalibrator(min_samples=30)
    calibrator.fit(raw_p, y)
    assert calibrator.is_fitted

    calibrated_p = calibrator.predict_proba(raw_p)
    cal_brier = compute_brier_score(y, calibrated_p)

    # 1. Brier score must improve (decrease)
    assert cal_brier < raw_brier

    # 2. Monotonicity: sorting inputs must preserve sorted order of outputs
    sorted_raw = np.linspace(0.05, 0.95, 50)
    sorted_cal = calibrator.predict_proba(sorted_raw)
    diffs = np.diff(sorted_cal)
    assert np.all(diffs >= -1e-8), "Calibrated probabilities must be monotonically non-decreasing"

    # 3. Extreme bounds clipping (overconfidence suppression)
    assert np.all(sorted_cal >= 0.02)
    assert np.all(sorted_cal <= 0.98)


def test_time_series_walk_forward_zero_leakage():
    """Verify that walk-forward calibration strictly uses tau < t, with no forward contamination."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    np.random.seed(123)
    raw_probs = np.random.uniform(0.3, 0.7, 100)
    targets = (np.random.rand(100) > 0.45).astype(int)

    df = pd.DataFrame({
        "date": dates,
        "p_up": raw_probs,
        "next_open_binary": targets,
    })

    cal_series = calibrate_time_series_walk_forward(
        df, prob_col="p_up", target_col="next_open_binary", date_col="date", min_history=25
    )

    assert len(cal_series) == 100
    assert not cal_series.isna().any()

    # The first 25 samples should equal raw_p because min_history=25
    np.testing.assert_allclose(cal_series.iloc[:24].values, raw_probs[:24], atol=1e-5)

    # Perturbing the final target value (row 99) MUST NOT change predictions at row 50
    df_perturbed = df.copy()
    df_perturbed.loc[99, "next_open_binary"] = 1 - df_perturbed.loc[99, "next_open_binary"]
    cal_perturbed = calibrate_time_series_walk_forward(
        df_perturbed, prob_col="p_up", target_col="next_open_binary", date_col="date", min_history=25
    )
    assert cal_series.iloc[50] == cal_perturbed.iloc[50], "Future label change leaked into past calibration!"


def test_cpcv_purge_and_embargo_isolation():
    """Verify that CPCV folds strictly purge overlap and enforce embargo without leakage."""
    n_samples = 100
    n_splits = 5
    n_test_splits = 2
    label_lookahead = 2
    embargo_bars = 3

    folds = generate_cpcv_folds(
        n_samples=n_samples,
        n_splits=n_splits,
        n_test_splits=n_test_splits,
        label_lookahead=label_lookahead,
        embargo_bars=embargo_bars,
    )

    # Number of combinations: C(5, 2) = 10
    assert len(folds) == 10

    for fold in folds:
        train_idx = set(fold.train_indices)
        test_idx = set(fold.test_indices)

        # 1. No overlap between train and test
        assert len(train_idx.intersection(test_idx)) == 0

        # 2. Check purge: samples [test_start - label_lookahead, test_start - 1] cannot be in train
        for b in fold.test_blocks:
            b_start = b * (n_samples // n_splits)
            b_end = (b + 1) * (n_samples // n_splits) - 1 if b < n_splits - 1 else n_samples - 1

            for p_bar in range(max(0, b_start - label_lookahead), b_start):
                assert p_bar not in train_idx, f"Purged bar {p_bar} found in train indices!"

            # 3. Check embargo: samples [b_end + 1, b_end + embargo_bars] cannot be in train
            for e_bar in range(b_end + 1, min(n_samples, b_end + 1 + embargo_bars)):
                assert e_bar not in train_idx, f"Embargoed bar {e_bar} found in train indices!"


def test_cpcv_path_stability_metrics():
    """Verify that compute_cpcv_path_stability produces robust metrics and valid grade."""
    np.random.seed(777)
    dates = pd.date_range("2024-01-01", periods=150, freq="B")
    # Consistently positive return when p_up >= 0.50
    signal = np.random.uniform(0.4, 0.7, 150)
    rets = (signal - 0.50) * 0.03 + np.random.normal(0.002, 0.008, 150)
    target = (rets > 0).astype(int)

    df = pd.DataFrame({
        "date": dates,
        "p_up": signal,
        "next_open_binary": target,
        "next_open_return": rets,
    })

    stability = compute_cpcv_path_stability(df, n_splits=5, n_test_splits=2)

    assert 0.0 <= stability.cpcv_stability_score <= 1.0
    assert stability.stability_grade in ("S", "A", "B", "C")
    assert stability.path_count == 10
    assert stability.positive_ic_ratio >= 0.50


def test_probabilistic_and_deflated_sharpe_ratio():
    """Verify PSR and DSR calculation against mathematical properties."""
    np.random.seed(999)
    # A strongly positive Sharpe strategy
    returns_good = np.random.normal(0.0025, 0.008, 500)  # Ann SR ~ 4.9
    psr_good, sr_ann, se, n = compute_probabilistic_sharpe_ratio(returns_good, benchmark_sr=0.0)

    assert sr_ann > 1.5

    assert psr_good > 0.99, "Strong positive return series should yield PSR > 0.99"

    # DSR under multiple testing: test with 100 trials
    dsr_result = compute_deflated_sharpe_ratio(returns_good, num_trials=100, variance_trials=1.0)
    assert 0.0 <= dsr_result.dsr <= 1.0
    assert dsr_result.expected_max_sharpe > 0.0
    # DSR must be less than unpenalized PSR because of the multiple testing penalty
    assert dsr_result.dsr <= dsr_result.psr

    # A random noise strategy with 0 mean
    returns_noise = np.random.normal(0.0, 0.01, 500)
    dsr_noise = compute_deflated_sharpe_ratio(returns_noise, num_trials=50)
    assert dsr_noise.dsr < 0.50
    assert not dsr_noise.is_statistically_significant
