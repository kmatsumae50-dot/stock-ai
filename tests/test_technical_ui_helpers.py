import pandas as pd

from stock_ai.app.main import (
    _technical_chart_range,
    _technical_feature_snapshot,
    _technical_state_summary,
    _technical_trend_label,
)


def test_technical_trend_and_summary_use_existing_features():
    latest = pd.Series(
        {
            "close": 110.0,
            "ma_5": 108.0,
            "ma_25": 105.0,
            "ma_75": 100.0,
            "rsi_14": 62.5,
            "volatility_20": 0.018,
        }
    )

    assert _technical_trend_label(latest) == "Bullish alignment"
    summary = _technical_state_summary(latest)
    assert "Bullish alignment" in summary
    assert "RSI 62.5" in summary
    assert "1.80%" in summary


def test_technical_chart_range_and_snapshot_limit_rows():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10, freq="B"),
            "close": range(10),
            "return_1d": [0.0] * 10,
            "rsi_14": [50.0] * 10,
            "regime": ["Sideways"] * 10,
        }
    )

    chart = _technical_chart_range(data, 4)
    snapshot = _technical_feature_snapshot(data, rows=3)

    assert len(chart) == 4
    assert len(snapshot) == 3
    assert snapshot.iloc[0]["date"] == "2026-01-14"
    assert {"date", "close", "return_1d", "rsi_14", "regime"}.issubset(snapshot.columns)


def test_technical_helpers_handle_missing_features():
    latest = pd.Series({"close": 100.0})
    empty = pd.DataFrame()

    assert _technical_trend_label(latest) == "Not available"
    assert "RSI is unavailable" in _technical_state_summary(latest)
    assert _technical_chart_range(empty, 20).empty
    assert _technical_feature_snapshot(empty, rows=20).empty
