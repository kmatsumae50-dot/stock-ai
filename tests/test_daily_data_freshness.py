import pandas as pd

from stock_ai.operations.freshness import (
    expected_latest_trading_date,
    summarize_daily_data_freshness,
    trading_day_lag,
)


def test_expected_latest_trading_date_uses_previous_weekday_for_weekend():
    assert expected_latest_trading_date("2026-06-20") == "2026-06-19"


def test_trading_day_lag_counts_weekdays_only():
    assert trading_day_lag("2026-06-12", "2026-06-15") == 1
    assert trading_day_lag("2026-06-12", "2026-06-19") == 5
    assert trading_day_lag("2026-06-19", "2026-06-19") == 0


def test_summarize_daily_data_freshness_reports_current_data(tmp_path):
    raw_dir = tmp_path / "raw"
    feature_dir = tmp_path / "features"
    raw_dir.mkdir()
    feature_dir.mkdir()
    pd.DataFrame({"date": ["2026-06-18", "2026-06-19"]}).to_parquet(raw_dir / "7203.T.parquet")
    pd.DataFrame({"date": ["2026-06-18", "2026-06-19"]}).to_parquet(feature_dir / "7203.T.parquet")

    summary = summarize_daily_data_freshness(
        ticker="7203.T",
        raw_data_dir=raw_dir,
        feature_data_dir=feature_dir,
        reference_date="2026-06-19",
    )

    assert summary["freshness_status"] == "OK"
    assert summary["stale_at_run_time"] is False
    assert summary["freshness_lag_trading_days"] == 0
    assert summary["latest_feature_date"] == "2026-06-19"


def test_summarize_daily_data_freshness_flags_stale_feature_data(tmp_path):
    raw_dir = tmp_path / "raw"
    feature_dir = tmp_path / "features"
    raw_dir.mkdir()
    feature_dir.mkdir()
    pd.DataFrame({"date": ["2026-06-12"]}).to_parquet(raw_dir / "7203.T.parquet")
    pd.DataFrame({"date": ["2026-06-12"]}).to_parquet(feature_dir / "7203.T.parquet")

    summary = summarize_daily_data_freshness(
        ticker="7203.T",
        raw_data_dir=raw_dir,
        feature_data_dir=feature_dir,
        reference_date="2026-06-15",
    )

    assert summary["freshness_status"] == "WARN"
    assert summary["stale_at_run_time"] is True
    assert summary["freshness_lag_trading_days"] == 1
    assert summary["prediction_feature_date"] == "2026-06-12"


def test_summarize_daily_data_freshness_handles_missing_feature_file(tmp_path):
    raw_dir = tmp_path / "raw"
    feature_dir = tmp_path / "features"
    raw_dir.mkdir()
    feature_dir.mkdir()
    pd.DataFrame({"date": ["2026-06-19"]}).to_parquet(raw_dir / "7203.T.parquet")

    summary = summarize_daily_data_freshness(
        ticker="7203.T",
        raw_data_dir=raw_dir,
        feature_data_dir=feature_dir,
        reference_date="2026-06-19",
    )

    assert summary["freshness_status"] == "MISSING"
    assert summary["stale_at_run_time"] is True
    assert summary["latest_feature_date"] is None

