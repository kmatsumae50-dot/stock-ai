from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_ai.models.turnaround import (
    TURNAROUND_FEATURE_COLUMNS,
    calculate_bargain_score,
    compute_turnaround_features,
    export_turnaround_reports,
    fetch_cached_fundamentals,
)


def _generate_synthetic_ohlcv(n_days: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    base_price = 1000.0
    returns = np.random.normal(0.0005, 0.02, n_days)
    close_prices = base_price * np.cumprod(1 + returns)

    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    open_prices = (high_prices + low_prices) / 2.0
    volume = np.random.randint(100_000, 1_000_000, n_days)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        }
    )


def test_compute_turnaround_features_columns_and_shapes() -> None:
    df = _generate_synthetic_ohlcv(300)
    feat_df = compute_turnaround_features(df)

    for col in TURNAROUND_FEATURE_COLUMNS:
        assert col in feat_df.columns, f"Missing feature column: {col}"

    assert "target_turnaround_20d" in feat_df.columns
    assert "target_turnaround_60d" in feat_df.columns
    assert "max_fwd_ret_20d" in feat_df.columns
    assert "pos_in_52w_range" in feat_df.columns

    # Check bounds of pos_in_52w_range
    valid_pos = feat_df["pos_in_52w_range"].dropna()
    assert (valid_pos >= 0.0).all()
    assert (valid_pos <= 1.0).all()

    # Targets are binary
    valid_t20 = feat_df["target_turnaround_20d"].dropna()
    assert set(valid_t20.unique()).issubset({0, 1})


def test_calculate_bargain_score_extremes() -> None:
    # Deep value + high bounce probability -> 100 pts
    score_max, disc_max, rec_max = calculate_bargain_score(
        drawdown_252d=-0.45,
        pos_in_52w_range=0.0,
        ma_gap_200d=-0.25,
        p_turnaround=1.0,
    )
    assert score_max == 100.0
    assert disc_max == 50.0
    assert rec_max == 50.0

    # High fly + zero bounce probability -> 0 pts
    score_min, disc_min, rec_min = calculate_bargain_score(
        drawdown_252d=0.05,
        pos_in_52w_range=1.0,
        ma_gap_200d=0.25,
        p_turnaround=0.0,
    )
    assert score_min == 0.0
    assert disc_min == 0.0
    assert rec_min == 0.0

    # Intermediate value
    score_mid, disc_mid, rec_mid = calculate_bargain_score(
        drawdown_252d=-0.20,
        pos_in_52w_range=0.5,
        ma_gap_200d=-0.10,
        p_turnaround=0.5,
    )
    assert 40.0 <= score_mid <= 60.0
    assert 0.0 <= disc_mid <= 50.0
    assert 0.0 <= rec_mid <= 50.0


def test_fetch_cached_fundamentals_fallback(tmp_path: Path) -> None:
    cache_path = tmp_path / "test_fund_cache.json"
    tickers = ["7201.T", "7203.T"]
    funds = fetch_cached_fundamentals(tickers, cache_path=cache_path)

    assert "7201.T" in funds
    assert "7203.T" in funds
    assert "pbr" in funds["7201.T"]
    assert "per" in funds["7201.T"]
    assert "dividend_yield" in funds["7201.T"]


def test_export_turnaround_reports(tmp_path: Path) -> None:
    sample_scan_data = {
        "date": "2026-10-14",
        "rankings": [
            {
                "ticker": "7201.T",
                "name": "日産自動車",
                "sector": "自動車",
                "close": 282.3,
                "drawdown_252d": -0.391,
                "pos_in_52w_range": 0.014,
                "ma_gap_200d": -0.202,
                "p_turnaround": 0.455,
                "discount_factor": 49.3,
                "recovery_factor": 22.8,
                "bargain_score": 72.0,
                "pbr": 0.20,
                "per": 5.4,
                "dividend_yield": 4.8,
                "overnight_p_up": 0.5873,
                "overnight_quantile": "Q4 (強気先行帯)",
                "is_golden_signal": True,
                "status": "💎 超割安・底打ち反転圏 ⭐ 始動",
            }
        ],
        "metrics": {
            "test_auc": 0.5791,
            "market_baseline": {
                "ret_20d_avg": 0.0216,
                "ret_20d_win_rate": 0.574,
                "ret_60d_avg": 0.0684,
                "ret_60d_win_rate": 0.620,
            },
            "q5_band_performance": {
                "ret_20d_avg": 0.0297,
                "ret_20d_win_rate": 0.623,
                "ret_20d_bounce_hit_rate": 0.623,
                "ret_60d_avg": 0.0818,
                "ret_60d_win_rate": 0.628,
            },
        },
    }

    json_path = tmp_path / "results.json"
    md_path = tmp_path / "report.md"

    export_turnaround_reports(sample_scan_data, output_json_path=json_path, output_md_path=md_path)

    assert json_path.exists()
    assert md_path.exists()

    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded_json["date"] == "2026-10-14"
    assert len(loaded_json["rankings"]) == 1

    md_content = md_path.read_text(encoding="utf-8")
    assert "日産自動車" in md_content
    assert "72.0点" in md_content
    assert "0.5791" in md_content


def test_cli_turnaround_parser_registered() -> None:
    import inspect
    from stock_ai.cli import main

    cli_source = inspect.getsource(main)
    assert "scan-turnaround-universe" in cli_source
    assert "--output-json" in cli_source
    assert "--output-md" in cli_source
    assert "--force-train" in cli_source


def test_ui_turnaround_scanner_section_callable() -> None:
    from stock_ai.app.main import _render_turnaround_scanner_section

    assert callable(_render_turnaround_scanner_section)

