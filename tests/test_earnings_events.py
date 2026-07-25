from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from stock_ai.events.earnings import (
    EVENT_STATE_DAY,
    EVENT_STATE_POST,
    EVENT_STATE_PRE,
    add_earnings_event_features,
    build_earnings_event_context,
    earnings_event_path_candidates,
    read_earnings_events,
    summarize_earnings_event_files,
    validate_earnings_csv,
)


def _price_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    close = pd.Series([100.0 + index for index in range(20)])
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * 20,
            "open": close * 1.01,
            "high": close * 1.02,
            "low": close * 0.99,
            "close": close,
            "volume": [1_000_000 + index * 50_000 for index in range(20)],
            "return_1d": close.pct_change(),
        }
    )


def test_add_earnings_event_features_from_manual_csv():
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "earnings_7203.T.csv"
        path.write_text(
            "date,event_type,period,note\n2024-01-10,earnings,Q3,\n",
            encoding="utf-8",
        )

        result = add_earnings_event_features(_price_frame(), temp_dir, "7203.T")

    event_row = result[result["date"] == pd.Timestamp("2024-01-10")].iloc[0]
    pre_row = result[result["date"] == pd.Timestamp("2024-01-08")].iloc[0]
    post_row = result[result["date"] == pd.Timestamp("2024-01-12")].iloc[0]

    assert pre_row["days_to_earnings"] == 2
    assert pre_row["earnings_event_state"] == EVENT_STATE_PRE
    assert event_row["earnings_day_flag"] == 1
    assert event_row["earnings_event_state"] == EVENT_STATE_DAY
    assert post_row["days_after_earnings"] == 2
    assert post_row["post_earnings_mode"] == 1
    assert post_row["earnings_event_state"] == EVENT_STATE_POST
    assert "earnings_post_gap" in result.columns
    assert "earnings_after_trend_5" in result.columns


def test_build_earnings_event_context_describes_post_event_mode():
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "earnings_7203.T.csv"
        path.write_text(
            "date,event_type,period,note\n2024-01-10,earnings,Q3,\n",
            encoding="utf-8",
        )
        features = add_earnings_event_features(
            _price_frame().iloc[:12].copy(),
            temp_dir,
            "7203.T",
        )

    context = build_earnings_event_context(features)

    assert context.event_state == EVENT_STATE_POST
    assert context.days_after_earnings == 2
    assert context.post_earnings_mode
    assert "Post-earnings" in context.summary


def test_missing_earnings_csv_keeps_normal_state_features():
    result = add_earnings_event_features(_price_frame(), "missing-events-dir", "7203.T")

    assert (result["earnings_week_flag"] == 0).all()
    assert (result["earnings_event_state"] == "normal").all()


def test_toyota_earnings_csv_is_supported_as_manual_alias():
    with TemporaryDirectory() as temp_dir:
        canonical = Path(temp_dir) / "earnings_7203.T.csv"
        canonical.write_text("date,event_type,period,note\n", encoding="utf-8")
        alias = Path(temp_dir) / "toyota_earnings.csv"
        alias.write_text(
            "date,event_type,period,note\n2024-01-10,earnings,Q3,\n",
            encoding="utf-8",
        )

        result = add_earnings_event_features(_price_frame(), temp_dir, "7203.T")

    event_row = result[result["date"] == pd.Timestamp("2024-01-10")].iloc[0]
    assert event_row["earnings_day_flag"] == 1
    assert event_row["earnings_event_state"] == EVENT_STATE_DAY


def test_earnings_path_candidates_include_toyota_alias():
    candidates = earnings_event_path_candidates("data/events", "7203.T")

    assert Path("data/events/toyota_earnings.csv") in candidates


def test_earnings_csv_summary_marks_header_only_file():
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "earnings_7203.T.csv"
        path.write_text("date,event_type,period,note\n", encoding="utf-8")

        summary = summarize_earnings_event_files(temp_dir, "7203.T", "2024-01-15")

    assert summary["status"] == "header_only"
    assert summary["rows"] == 0
    assert summary["active_path"] is None
    assert summary["files"][0]["status"] == "header_only"


def test_earnings_csv_validation_detects_missing_date_column():
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "earnings_7203.T.csv"
        path.write_text("event_type,period,note\nearnings,Q3,\n", encoding="utf-8")

        result = validate_earnings_csv(path, "7203.T")

    assert result["status"] == "invalid"
    assert result["required_columns_present"] is False
    assert result["missing_required_columns"] == ["date or earnings_date"]


def test_earnings_csv_validation_detects_date_parse_error():
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "earnings_7203.T.csv"
        path.write_text("date,event_type,period,note\nbad-date,earnings,Q3,\n", encoding="utf-8")

        result = validate_earnings_csv(path, "7203.T")

    assert result["status"] == "invalid"
    assert result["invalid_date_count"] == 1


def test_earnings_csv_validation_detects_duplicate_event_dates():
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "earnings_7203.T.csv"
        path.write_text(
            "date,event_type,period,note\n"
            "2024-01-10,earnings,Q3,\n"
            "2024-01-10,earnings,Q3 duplicate,\n",
            encoding="utf-8",
        )

        result = validate_earnings_csv(path, "7203.T", "2024-01-01")

    assert result["status"] == "needs_review"
    assert result["duplicate_event_date_count"] == 1
    assert result["future_event_count"] == 2


def test_earnings_date_alias_matches_build_features_reader():
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "earnings_7203.T.csv"
        path.write_text(
            "earnings_date,event_type,period,note\n2024-01-10,earnings,Q3,\n",
            encoding="utf-8",
        )

        validation = validate_earnings_csv(path, "7203.T", "2024-01-01")
        events = read_earnings_events(temp_dir, "7203.T")

    assert validation["status"] == "populated"
    assert validation["required_columns_present"] is True
    assert validation["next_known_event_date"] == "2024-01-10"
    assert "date" in events.columns
    assert events.iloc[0]["date"] == pd.Timestamp("2024-01-10")
