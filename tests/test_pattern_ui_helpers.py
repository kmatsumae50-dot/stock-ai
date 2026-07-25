import pandas as pd

from stock_ai.app.main import (
    _cross_ticker_scope,
    _cross_ticker_top_summary,
    _lead_lag_top_summary,
    _pattern_latest_feature_date,
    _similar_pattern_top_summary,
)


def test_pattern_overview_helpers_summarize_available_results():
    data = pd.DataFrame({"date": ["2026-06-10", "2026-06-11"]})
    similar = pd.DataFrame(
        {
            "similar_end_date": ["2025-01-15"],
            "similarity": [0.9123],
        }
    )
    cross = pd.DataFrame(
        {
            "ticker_name": ["Honda", "Tesla"],
            "similar_end_date": ["2024-05-01", "2023-04-01"],
            "similarity": [0.88, 0.81],
        }
    )
    lead_lag = pd.DataFrame(
        {
            "name": ["USDJPY"],
            "lag": [3],
            "correlation": [0.42],
            "current_signal": ["bullish"],
        }
    )

    assert _pattern_latest_feature_date(data) == "2026-06-11"
    assert "2025-01-15" in _similar_pattern_top_summary(similar)
    assert _cross_ticker_scope(cross) == "Honda, Tesla"
    assert "Honda" in _cross_ticker_top_summary(cross)
    assert "USDJPY" in _lead_lag_top_summary(lead_lag)
    assert "bullish" in _lead_lag_top_summary(lead_lag)


def test_pattern_overview_helpers_handle_empty_data():
    empty = pd.DataFrame()

    assert _pattern_latest_feature_date(empty) == "-"
    assert "No similar" in _similar_pattern_top_summary(empty)
    assert _cross_ticker_scope(empty) == "No peers available"
    assert "No cross-ticker" in _cross_ticker_top_summary(empty)
    assert "No lead-lag" in _lead_lag_top_summary(empty)
