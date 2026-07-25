import pandas as pd

from stock_ai.evaluation.performance import (
    build_evaluation_summaries,
    build_weakness_summary,
    prepare_backtest_evaluation_frame,
    prepare_prediction_history_evaluation_frame,
    summarize_rows,
)


def test_prepare_backtest_evaluation_frame_and_summaries_by_conditions():
    predictions = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="D"),
            "predicted_class": [1, 1, 0, 0, 1, 1, 0, 0],
            "actual_class": [1, 0, 0, 1, 1, 1, 0, 0],
            "probability_up": [0.8, 0.7, 0.2, 0.4, 0.9, 0.6, 0.3, 0.2],
            "probability_down": [0.2, 0.3, 0.8, 0.6, 0.1, 0.4, 0.7, 0.8],
            "confidence": [0.8, 0.7, 0.8, 0.6, 0.9, 0.6, 0.7, 0.8],
            "actual_return": [0.01, -0.02, -0.01, 0.02, 0.03, 0.01, -0.01, -0.02],
            "strategy_return": [0.01, -0.02, 0.0, 0.0, 0.03, 0.01, 0.0, 0.0],
            "trade": [True, True, False, False, True, True, False, False],
            "regime": ["Bull Trend", "High Volatility"] * 4,
            "surge_risk_level": ["High", "Low"] * 4,
            "crash_risk_level": ["Low", "High"] * 4,
            "anomaly_types": ["Volume Spike", "Volatility Spike"] * 4,
            "dominant_narrative": ["EV", "Rates"] * 4,
            "earnings_event_state": ["normal", "post_earnings"] * 4,
            "positive_news_flag": [1, 0] * 4,
            "negative_news_flag": [0, 1] * 4,
        }
    )

    frame = prepare_backtest_evaluation_frame(predictions)
    summaries = build_evaluation_summaries(frame)
    overall = summarize_rows(frame)

    assert overall["count"] == 8
    assert "regime" in summaries
    assert "anomaly_type" in summaries
    assert not summaries["confidence_bucket"].empty
    assert set(summaries["news_sentiment_bucket"]["news_sentiment_bucket"]) == {"Positive", "Negative"}


def test_prediction_history_evaluation_joins_context_and_builds_weakness_summary():
    history = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "ticker": ["7203.T"] * 6,
            "status": ["confirmed"] * 6,
            "predicted_class": [1, 1, 1, 0, 0, 0],
            "actual_class": [1, 0, 0, 0, 1, 0],
            "probability_up": [0.8, 0.7, 0.65, 0.3, 0.4, 0.2],
            "probability_down": [0.2, 0.3, 0.35, 0.7, 0.6, 0.8],
            "confidence": [0.8, 0.7, 0.65, 0.7, 0.6, 0.8],
            "actual_return": [0.01, -0.02, -0.01, -0.01, 0.02, -0.01],
        }
    )
    features = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "ticker": ["7203.T"] * 6,
            "regime": ["High Volatility"] * 3 + ["Bull Trend"] * 3,
            "anomaly_types": ["Volatility Spike"] * 3 + ["None"] * 3,
            "dominant_narrative": ["Rates"] * 3 + ["EV"] * 3,
            "earnings_event_state": ["pre_earnings_week"] * 3 + ["normal"] * 3,
            "negative_news_flag": [1, 1, 1, 0, 0, 0],
            "positive_news_flag": [0, 0, 0, 1, 1, 1],
        }
    )

    frame = prepare_prediction_history_evaluation_frame(history, features)
    summaries = build_evaluation_summaries(frame)
    messages = build_weakness_summary(summaries, summarize_rows(frame), min_count=2)

    assert not frame.empty
    assert "regime" in frame.columns
    assert not summaries["earnings_event_state"].empty
    assert messages
