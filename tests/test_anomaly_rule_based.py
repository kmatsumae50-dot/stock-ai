import pandas as pd

from stock_ai.anomaly.rule_based import (
    add_anomaly_scores,
    build_anomaly_comparison_frame,
    build_anomaly_context,
    find_anomaly_events,
    summarize_anomaly_forward_returns,
)


def test_add_anomaly_scores_detects_volume_and_price_shock():
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    returns = [0.001] * 89 + [0.08]
    close = [100.0]
    for value in returns[1:]:
        close.append(close[-1] * (1 + value))
    volume = [100.0] * 89 + [400.0]
    data = pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "volume": volume,
            "return_1d": returns,
            "volume_change_1d": [0.0] * 89 + [3.0],
            "rsi_14": [50.0] * 89 + [82.0],
            "macd_hist": [0.0] * 89 + [1.0],
            "volatility_20": [0.02] * 90,
        }
    )

    scored = add_anomaly_scores(data, lookback=30)
    context = build_anomaly_context(scored)

    assert context is not None
    assert context.score >= 70
    assert "Price Shock" in context.types
    assert "Volume Spike" in context.types
    assert "return_1d" in context.features


def test_anomaly_events_include_forward_returns_and_comparison():
    dates = pd.date_range("2024-01-01", periods=110, freq="D")
    close = [100.0 + index * 0.1 for index in range(110)]
    data = pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "volume": [100.0] * 110,
            "return_1d": [0.001] * 50 + [0.07] + [0.001] * 59,
            "volume_change_1d": [0.0] * 110,
            "rsi_14": [50.0] * 50 + [80.0] + [50.0] * 59,
            "macd_hist": [0.0] * 110,
            "volatility_20": [0.02] * 110,
        }
    )

    scored = add_anomaly_scores(data, lookback=30)
    events = find_anomaly_events(scored, min_score=50)
    summary = summarize_anomaly_forward_returns(events)
    comparison = build_anomaly_comparison_frame(scored, events.iloc[-1]["date"])

    assert not events.empty
    assert "anomaly_return_5d_after" in events.columns
    assert list(summary["horizon"]) == ["1D", "5D", "20D"]
    assert not comparison.empty
    assert {"current", "selected_anomaly"}.issubset(comparison.columns)
