import pandas as pd

from stock_ai.features.missingness import classify_feature_missingness


def test_rolling_window_initial_na_is_expected_missing():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5),
            "ma_5": [None, None, 10.0, 11.0, 12.0],
        }
    )

    report = classify_feature_missingness(data)
    row = report.rows.set_index("column").loc["ma_5"]

    assert row["classification"] == "expected_missing"
    assert row["reason"] == "rolling_window_initial_na"
    assert report.needs_review_count == 0


def test_latest_primary_feature_missing_needs_review():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3),
            "close": [10.0, 11.0, None],
        }
    )

    report = classify_feature_missingness(data)
    row = report.rows.set_index("column").loc["close"]

    assert row["classification"] == "needs_review"
    assert row["reason"] == "latest_row_missing"
    assert report.needs_review_count == 1


def test_optional_context_missing_is_optional():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3),
            "news_reason": [None, "available", None],
        }
    )

    report = classify_feature_missingness(data)
    row = report.rows.set_index("column").loc["news_reason"]

    assert row["classification"] == "optional_missing"
    assert row["reason"] == "optional_context_column"
    assert report.optional_missing_count == 1


def test_event_distance_without_calendar_is_not_applicable():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3),
            "days_to_earnings": [None, None, None],
        }
    )

    report = classify_feature_missingness(data)
    row = report.rows.set_index("column").loc["days_to_earnings"]

    assert row["classification"] == "not_applicable"
    assert row["reason"] == "event_date_not_available_for_period"
    assert report.not_applicable_count == 1


def test_no_missing_columns_returns_zero_counts():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3),
            "close": [10.0, 11.0, 12.0],
        }
    )

    report = classify_feature_missingness(data)

    assert report.rows.empty
    assert report.columns_with_missing == 0
    assert report.expected_missing_count == 0
    assert report.optional_missing_count == 0
    assert report.needs_review_count == 0
    assert report.latest_row_missing_count == 0
