import pandas as pd

from stock_ai.app.main import (
    _recent_anomaly_event_count,
    _recent_regime_history,
    _regime_history_display_frame,
    _regime_transition_summary,
)


def test_regime_history_and_transition_summary():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2026-06-01", periods=6, freq="B"),
            "regime": [
                "Sideways",
                "Sideways",
                "Bull Trend",
                "Bull Trend",
                "Bull Trend",
                "Bull Trend",
            ],
        }
    )

    history = _recent_regime_history(data)
    display = _regime_history_display_frame(history)
    summary = _regime_transition_summary(history)

    assert len(history) == 6
    assert "Sideways transitioned to Bull Trend" in summary
    assert "4 trading days" in summary
    assert display.iloc[0]["Regime"] == "Bull Trend"
    assert {"Date", "Regime", "Transition"}.issubset(display.columns)


def test_anomaly_recent_count_uses_latest_feature_rows():
    enriched = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=100, freq="B"),
        }
    )
    events = pd.DataFrame(
        {
            "date": [
                enriched.iloc[10]["date"],
                enriched.iloc[50]["date"],
                enriched.iloc[99]["date"],
            ]
        }
    )

    assert _recent_anomaly_event_count(enriched, events, lookback_rows=60) == 2


def test_anomaly_regime_helpers_handle_missing_data():
    empty = pd.DataFrame()

    assert _recent_regime_history(empty).empty
    assert _regime_transition_summary(empty) == "Regime history is not available."
    assert _recent_anomaly_event_count(empty, empty) == 0
