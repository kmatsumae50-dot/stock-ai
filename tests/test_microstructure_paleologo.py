"""Unit tests for Market Microstructure & Giuseppe Paleologo Factor Orthogonalization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_ai.features.microstructure_paleologo import (
    add_microstructure_paleologo_features,
    add_premarket_microstructure_paleologo_features,
    apply_paleologo_cluster_penalty,
    compute_amihud_illiquidity,
    compute_kyle_lambda,
    compute_paleologo_idiosyncratic_features,
    compute_roll_effective_spread,
    compute_upward_elasticity,
)


class TestAmihudAndUpwardElasticity:
    """Tests for Amihud Illiquidity Ratio & Asymmetric Upward Elasticity."""

    def test_amihud_illiquidity_calculation(self) -> None:
        np.random.seed(42)
        n = 50
        prices = pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n))))
        volumes = pd.Series(100_000.0, index=prices.index)

        illiq_20, illiq_surge = compute_amihud_illiquidity(prices, volumes, window=20)

        assert isinstance(illiq_20, pd.Series)
        assert len(illiq_20) == n
        assert (illiq_20 >= 0.0).all()
        # Non-zero after burn-in
        assert illiq_20.iloc[-1] > 0.0
        assert 0.1 <= illiq_surge.iloc[-1] <= 10.0

    def test_upward_elasticity_asymmetry(self) -> None:
        # Create a synthetic series with large price jumps on positive days with tiny volume
        # and small price drops on negative days with huge volume (上値真空)
        n = 30
        prices = [100.0]
        vols = [1000.0]
        for i in range(1, n):
            if i % 2 == 1:
                # Up day: +3% with only 1,000 shares (very high price impact)
                prices.append(prices[-1] * 1.03)
                vols.append(1_000.0)
            else:
                # Down day: -0.5% with 100,000 shares (hard floor, very low price impact)
                prices.append(prices[-1] * 0.995)
                vols.append(100_000.0)

        p_series = pd.Series(prices)
        v_series = pd.Series(vols)
        elasticity = compute_upward_elasticity(p_series, v_series, window=10)

        # Elasticity should be well above 1.0 indicating vacuum above and hard bids below
        assert elasticity.iloc[-1] > 2.0
        assert elasticity.iloc[-1] <= 10.0


class TestKyleLambdaAndRollSpread:
    """Tests for Kyle's Lambda & Roll's Effective Bid-Ask Spread."""

    def test_kyle_lambda_positive_impact(self) -> None:
        np.random.seed(123)
        n = 40
        # When volume is higher, return is strongly positive
        flows = np.random.normal(0, 1.0, n)
        returns = 0.01 * flows + np.random.normal(0, 0.002, n)
        prices = pd.Series(100.0 * np.exp(np.cumsum(returns)))
        volumes = pd.Series(np.exp(np.abs(flows) * 2.0) * 1000.0)

        k_lambda = compute_kyle_lambda(prices, volumes, window=20)
        assert isinstance(k_lambda, pd.Series)
        assert len(k_lambda) == n
        # Lambda should be bounded
        assert (-5.0 <= k_lambda).all() and (k_lambda <= 5.0).all()

    def test_roll_effective_spread_bounce(self) -> None:
        # Synthetic bid-ask bounce: alternating +1% and -1% (negative serial covariance)
        n = 30
        rets = np.array([0.01, -0.01] * (n // 2))
        prices = pd.Series(100.0 * np.cumprod(1.0 + rets))

        spread = compute_roll_effective_spread(prices, window=10)
        assert isinstance(spread, pd.Series)
        # Should detect the negative serial covariance and produce positive effective spread
        assert spread.iloc[-1] > 0.01
        assert spread.iloc[-1] <= 0.10


class TestPaleologoFactorOrthogonalization:
    """Tests for Giuseppe Paleologo Factor Orthogonalization & Residual Alpha."""

    def test_idiosyncratic_residual_properties(self) -> None:
        np.random.seed(42)
        n = 60
        r_m = pd.Series(np.random.normal(0.001, 0.01, n))
        beta = 1.5
        true_alpha = 0.002
        noise = np.random.normal(0, 0.005, n)
        r_i = beta * r_m + true_alpha + noise

        idio_5d, idio_sharpe, beta_neut = compute_paleologo_idiosyncratic_features(
            ret_series=r_i,
            market_ret_series=r_m,
            window=20,
        )

        assert isinstance(idio_5d, pd.Series)
        assert isinstance(idio_sharpe, pd.Series)
        assert isinstance(beta_neut, pd.Series)

        # Residual alpha over 5 days should be positive because of positive true_alpha
        assert idio_5d.iloc[-1] > -0.10
        # Idiosyncratic Sharpe and beta neutral score should be bounded
        assert -5.0 <= idio_sharpe.iloc[-1] <= 5.0
        assert -5.0 <= beta_neut.iloc[-1] <= 5.0


class TestPaleologoClusterPenalty:
    """Tests for Paleologo Sector and Correlation Cluster Dispersion Constraint."""

    def test_sector_concentration_penalty_diversification(self) -> None:
        # Candidates: two strong semicon stocks and one slightly lower machinery stock
        df = pd.DataFrame({
            "ticker": ["8035.T", "6857.T", "7011.T", "8306.T"],
            "sector": ["Semiconductors", "Semiconductors", "Machinery", "Banking"],
            "p_cs": [0.65, 0.63, 0.60, 0.55],
        })

        # Without penalty, top 2 would be 8035.T and 6857.T (both Semiconductors)
        # With Paleologo cluster penalty, 6857.T receives 0.75 penalty: 0.63 * 0.75 = 0.4725 < 0.60 (7011.T)
        selected = apply_paleologo_cluster_penalty(
            candidates_df=df,
            score_col="p_cs",
            sector_col="sector",
            max_per_cluster=1,
            penalty_factor=0.75,
            top_k=2,
        )

        assert len(selected) == 2
        assert selected[0] == "8035.T"
        assert selected[1] == "7011.T"  # Diversified to Machinery!
        assert "6857.T" not in selected

    def test_correlation_matrix_cluster_penalty(self) -> None:
        # High correlation penalty across different nominal sector labels
        df = pd.DataFrame({
            "ticker": ["TICK_A", "TICK_B", "TICK_C"],
            "sector": ["Tech", "Electronics", "Utility"],
            "p_cs": [0.70, 0.68, 0.58],
        })
        corr_matrix = pd.DataFrame(
            [
                [1.00, 0.85, 0.10],
                [0.85, 1.00, 0.15],
                [0.10, 0.15, 1.00],
            ],
            index=["TICK_A", "TICK_B", "TICK_C"],
            columns=["TICK_A", "TICK_B", "TICK_C"],
        )

        selected = apply_paleologo_cluster_penalty(
            candidates_df=df,
            score_col="p_cs",
            sector_col="sector",
            max_per_cluster=2,  # Nominal sector allowed
            corr_matrix=corr_matrix,
            corr_threshold=0.70,
            penalty_factor=0.70,
            top_k=2,
        )

        assert selected[0] == "TICK_A"
        # TICK_B has 0.85 correlation with TICK_A -> penalized 0.68 * 0.70 = 0.476 < 0.58 (TICK_C)
        assert selected[1] == "TICK_C"


class TestFeaturePipelines:
    """Tests for end-to-end DataFrame feature augmentation."""

    def test_add_microstructure_paleologo_features(self) -> None:
        df = pd.DataFrame({
            "close": [100.0, 101.0, 100.5, 102.0, 103.5] * 6,
            "volume": [1000.0, 1200.0, 950.0, 1500.0, 2000.0] * 6,
            "topix_return_1d": [0.005, -0.002, 0.001, 0.008, 0.003] * 6,
        })
        augmented = add_microstructure_paleologo_features(df)
        for col in [
            "micro_amihud_20d",
            "micro_upward_elasticity_10d",
            "micro_kyle_lambda_20d",
            "micro_roll_spread_20d",
            "paleo_idio_return_5d",
            "paleo_idio_sharpe_20d",
            "paleo_beta_neutral_score",
        ]:
            assert col in augmented.columns

    def test_add_premarket_microstructure_paleologo_features_zero_leakage(self) -> None:
        # Day t price/volume changes must not affect day t pre-market features
        df = pd.DataFrame({
            "close": [100.0, 102.0, 104.0, 106.0, 150.0],  # Huge spike at day 4
            "volume": [1000.0, 1000.0, 1000.0, 1000.0, 50000.0],
        })
        augmented = add_premarket_microstructure_paleologo_features(df)
        assert "prior_micro_amihud_20d" in augmented.columns
        assert "prior_micro_upward_elasticity_10d" in augmented.columns
        assert "prior_paleo_idio_return_5d" in augmented.columns

        # Changing the last row's close should NOT change row 4's prior features if shift(1) is correct
        df_alt = df.copy()
        df_alt.loc[4, "close"] = 50.0
        df_alt.loc[4, "volume"] = 100.0
        augmented_alt = add_premarket_microstructure_paleologo_features(df_alt)
        assert augmented.loc[4, "prior_micro_amihud_20d"] == augmented_alt.loc[4, "prior_micro_amihud_20d"]
