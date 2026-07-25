import pandas as pd

from stock_ai.app.main import (
    _build_data_overview,
    _data_freshness_label,
    _data_missing_values_frame,
    _known_missed_details_frame,
    _latest_prediction_saved_at_from_rows,
    _next_trading_date_after,
    _prediction_overview_record,
)


def test_build_data_overview_uses_health_and_frame_summaries():
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-11", "2026-06-12"]),
            "close": [2500.0, 2520.0],
        }
    )
    features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-11", "2026-06-12"]),
            "close": [2500.0, 2520.0],
            "return_1d": [0.01, 0.008],
        }
    )
    report = {
        "overall_status": "WARN",
        "warning_summary": {
            "action_required_count": 0,
            "known_ok_count": 1,
        },
        "items": [
            {
                "section": "News",
                "name": "News CSV quality",
                "status": "OK",
                "details": {
                    "rows": 52,
                    "latest_news_date": "2026-06-14",
                    "latest_fetched_at": "2026-06-15 00:13:27",
                },
            },
            {
                "section": "Data",
                "name": "Earnings CSV",
                "status": "OK",
                "details": {},
            },
            {
                "section": "External data",
                "name": "TOPIX parquet",
                "status": "WARN",
                "details": {"data_status": "fallback"},
            },
        ],
    }

    overview = _build_data_overview("7203.T", features, raw, report)

    assert overview["ticker"] == "7203.T"
    assert overview["latest_raw_date"] == "2026-06-12"
    assert overview["latest_feature_date"] == "2026-06-12"
    assert overview["raw_rows"] == 2
    assert overview["feature_rows"] == 2
    assert overview["news_rows"] == 52
    assert overview["earnings_status"] == "OK"
    assert overview["external_status"] == "WARN"
    assert overview["data_freshness"] == "OK"
    assert overview["action_required_count"] == 0
    assert overview["known_ok_count"] == 1


def test_data_freshness_label_distinguishes_stale_and_missing():
    assert _data_freshness_label("2026-06-12", "2026-06-12") == "OK"
    assert _data_freshness_label("2026-06-12", "2026-06-10") == "stale"
    assert _data_freshness_label("2026-06-12", "2026-06-05") == "missing update"
    assert _data_freshness_label(None, "2026-06-12") == "not available"


def test_data_missing_values_frame_reports_counts_and_rates():
    frame = pd.DataFrame(
        {
            "complete": [1.0, 2.0, 3.0],
            "partial": [1.0, None, None],
        }
    )

    result = _data_missing_values_frame(frame)

    assert result.to_dict(orient="records") == [
        {
            "Column": "partial",
            "Missing Count": 2,
            "Missing Rate": "66.67%",
        }
    ]


def test_next_trading_date_after_uses_local_feature_calendar():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-12", "2026-06-15", "2026-06-16"]),
            "close": [2775.5, 2902.5, 2847.5],
        }
    )

    assert _next_trading_date_after(data, "2026-06-12") == pd.Timestamp("2026-06-15")
    assert _next_trading_date_after(data, "2026-06-16") is None
    assert _next_trading_date_after(data, None) is None


def test_prediction_overview_record_labels_feature_and_target_dates():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-12", "2026-06-15"]),
            "close": [2775.5, 2902.5],
        }
    )
    row = pd.Series(
        {
            "date": "2026-06-12",
            "target": "next_close_binary",
            "model_name": "lightgbm_binary",
            "predicted_class": 1,
            "confidence": 0.62,
            "prediction_type": "live",
            "created_at": "2026-06-15 00:14:04",
        }
    )

    record = _prediction_overview_record("Binary", row, data)

    assert record["Feature date"] == "2026-06-12"
    assert record["Target close date"] == "2026-06-15"
    assert record["Source"] == "live"
    assert record["Saved at"] == "2026-06-15 00:14:04"


def test_latest_prediction_saved_at_from_rows_uses_latest_created_at():
    rows = [
        pd.Series({"created_at": "2026-06-15 00:14:04"}),
        pd.Series({"created_at": "2026-06-15 00:14:12"}),
        None,
    ]

    assert _latest_prediction_saved_at_from_rows(rows) == pd.Timestamp("2026-06-15 00:14:12")


def test_known_missed_details_frame_expands_date_rows():
    frame = _known_missed_details_frame(
        [
            {
                "details": {
                    "known_missed_live_dates": [
                        {
                            "missing_date": "2026-06-15",
                            "known_reason_code": "data_not_available_at_run_time",
                            "known_reason": "Data was stale at run time.",
                            "known_action": "do_not_create_live_prediction",
                            "known_allow_backfill": True,
                        },
                        {
                            "missing_date": "2026-06-23",
                            "known_reason_code": "data_not_available_at_run_time",
                            "known_action": "do_not_create_live_prediction",
                            "known_allow_backfill": True,
                        },
                    ]
                }
            }
        ]
    )

    assert len(frame) == 2
    assert list(frame["Feature date"]) == ["2026-06-15", "2026-06-23"]
    assert frame.iloc[0]["Allow backfill"] == "Enabled"
