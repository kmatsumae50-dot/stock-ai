from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


def expected_latest_trading_date(reference_date: str | date | pd.Timestamp | None = None) -> str:
    """Return the latest weekday that daily data is expected to represent."""
    if reference_date is None:
        current = pd.Timestamp.now(tz="Asia/Tokyo").date()
    else:
        current = pd.Timestamp(reference_date).date()
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current.isoformat()


def summarize_daily_data_freshness(
    *,
    ticker: str,
    raw_data_dir: Path,
    feature_data_dir: Path,
    reference_date: str | date | pd.Timestamp | None = None,
) -> dict[str, Any]:
    expected_date = expected_latest_trading_date(reference_date)
    raw_path = Path(raw_data_dir) / f"{ticker}.parquet"
    feature_path = Path(feature_data_dir) / f"{ticker}.parquet"
    latest_raw_date = _latest_date_from_parquet(raw_path)
    latest_feature_date = _latest_date_from_parquet(feature_path)
    raw_lag = trading_day_lag(latest_raw_date, expected_date)
    feature_lag = trading_day_lag(latest_feature_date, expected_date)

    if latest_feature_date is None:
        status = "MISSING"
        stale = True
        message = "Feature data is missing; prediction freshness cannot be confirmed."
    elif feature_lag > 0:
        status = "WARN"
        stale = True
        message = (
            "Feature data is older than the expected latest trading date at run time. "
            "Prediction can continue, but the saved live date may lag the calendar."
        )
    else:
        status = "OK"
        stale = False
        message = "Feature data is current for the expected latest trading date."

    return {
        "ticker": ticker,
        "expected_latest_trading_date": expected_date,
        "latest_raw_date": latest_raw_date,
        "latest_feature_date": latest_feature_date,
        "prediction_feature_date": latest_feature_date,
        "saved_prediction_date_expected": latest_feature_date,
        "raw_lag_trading_days": raw_lag,
        "freshness_lag_trading_days": feature_lag,
        "stale_at_run_time": stale,
        "freshness_status": status,
        "freshness_message": message,
    }


def trading_day_lag(latest_date: str | None, expected_date: str | None) -> int | None:
    if latest_date is None or expected_date is None:
        return None
    latest = pd.to_datetime(latest_date, errors="coerce")
    expected = pd.to_datetime(expected_date, errors="coerce")
    if pd.isna(latest) or pd.isna(expected):
        return None
    latest_date_value = pd.Timestamp(latest).date()
    expected_date_value = pd.Timestamp(expected).date()
    if latest_date_value >= expected_date_value:
        return 0
    count = 0
    current = latest_date_value + timedelta(days=1)
    while current <= expected_date_value:
        if current.weekday() < 5:
            count += 1
        current += timedelta(days=1)
    return count


def _latest_date_from_parquet(path: Path) -> str | None:
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if "date" not in frame.columns:
        return None
    values = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if values.empty:
        return None
    return pd.Timestamp(values.max()).strftime("%Y-%m-%d")
