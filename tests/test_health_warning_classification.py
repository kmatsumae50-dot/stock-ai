from types import SimpleNamespace

from stock_ai.health import checks
from stock_ai.health.checks import classify_health_warnings


def test_topix_fallback_is_known_ok():
    summary = classify_health_warnings(
        [
            {
                "section": "External data",
                "name": "TOPIX parquet",
                "status": "WARN",
                "message": "TOPIX parquet is available via fallback.",
                "details": {"data_status": "fallback"},
            }
        ]
    )

    assert summary["known_ok_count"] == 1
    assert summary["action_required_count"] == 0
    assert summary["known_ok"][0]["name"] == "TOPIX parquet"


def test_unknown_news_sources_are_action_required():
    summary = classify_health_warnings(
        [
            {
                "section": "News",
                "name": "News CSV quality",
                "status": "WARN",
                "message": "News CSV has unknown sources.",
                "details": {"unknown_source_count": 2},
            }
        ]
    )

    assert summary["action_required_count"] == 1
    assert summary["known_ok_count"] == 0
    assert summary["action_required"][0]["name"] == "News CSV quality"


def test_ok_news_report_is_promoted_to_warn_when_unknown_sources_exist(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(checks, "default_news_dir", lambda _: tmp_path)
    monkeypatch.setattr(
        checks,
        "check_news_for_ticker",
        lambda _news_dir, _ticker: {
            "status": "OK",
            "message": "News CSV quality checks passed.",
            "unknown_source_count": 2,
        },
    )
    settings = SimpleNamespace(
        paths=SimpleNamespace(raw_data_dir=tmp_path / "raw"),
    )

    item = checks._check_news_quality(settings, "7203.T")[0]

    assert item.status == "WARN"
    assert "2 unknown news source" in item.message


def test_daily_predict_warning_is_action_required():
    summary = classify_health_warnings(
        [
            {
                "section": "daily_predict",
                "name": "Latest run",
                "status": "WARN",
                "message": "Latest daily_predict run failed.",
                "details": {"latest_error": "daily_predict failed"},
            }
        ]
    )

    assert summary["action_required_count"] == 1
    assert "Daily automation" in summary["action_required"][0]["reason"]


def test_missing_live_dates_after_start_are_action_required():
    summary = classify_health_warnings(
        [
            {
                "section": "SQLite",
                "name": "Missing live dates",
                "status": "WARN",
                "message": "Live predictions are missing after the configured operation start date.",
                "details": {"missing_prediction_count_after_start": 1},
            }
        ]
    )

    assert summary["action_required_count"] == 1
    assert summary["known_ok_count"] == 0


def test_known_missed_live_dates_are_not_action_required():
    summary = classify_health_warnings(
        [
            {
                "section": "SQLite",
                "name": "Missing live dates",
                "status": "WARN",
                "message": "All missing live prediction dates are known missed.",
                "details": {
                    "missing_prediction_count_after_start": 1,
                    "unknown_missing_prediction_count_after_start": 0,
                    "known_missed_live_dates_count": 1,
                    "known_missed_live_dates": [
                        {
                            "missing_date": "2026-06-15",
                            "known_reason_code": "data_not_available_at_run_time",
                            "known_action": "do_not_create_live_prediction",
                            "known_allow_backfill": True,
                        }
                    ],
                },
            }
        ]
    )

    assert summary["action_required_count"] == 0
    assert summary["known_missed_count"] == 1
    assert summary["known_missed_section_count"] == 1
    assert summary["known_missed_date_count"] == 1
    assert summary["known_missed"][0]["details"]["known_missed_live_dates"][0]["known_action"] == (
        "do_not_create_live_prediction"
    )


def test_known_missed_date_count_is_not_section_count():
    summary = classify_health_warnings(
        [
            {
                "section": "SQLite",
                "name": "Missing live dates",
                "status": "WARN",
                "message": "All missing live prediction dates are known missed.",
                "details": {
                    "missing_prediction_count_after_start": 2,
                    "unknown_missing_prediction_count_after_start": 0,
                    "known_missed_live_dates_count": 2,
                    "known_missed_live_dates": [
                        {
                            "missing_date": "2026-06-15",
                            "known_reason_code": "data_not_available_at_run_time",
                            "known_action": "do_not_create_live_prediction",
                            "known_allow_backfill": True,
                        },
                        {
                            "missing_date": "2026-06-23",
                            "known_reason_code": "data_not_available_at_run_time",
                            "known_action": "do_not_create_live_prediction",
                            "known_allow_backfill": True,
                        },
                    ],
                },
            }
        ]
    )

    assert summary["action_required_count"] == 0
    assert summary["known_missed_section_count"] == 1
    assert summary["known_missed_date_count"] == 2
    assert summary["known_missed_count"] == 2


def test_unknown_missing_live_dates_stay_action_required_with_known_missed_present():
    summary = classify_health_warnings(
        [
            {
                "section": "SQLite",
                "name": "Missing live dates",
                "status": "WARN",
                "message": "One unknown date remains missing.",
                "details": {
                    "missing_prediction_count_after_start": 2,
                    "unknown_missing_prediction_count_after_start": 1,
                    "known_missed_live_dates_count": 1,
                },
            }
        ]
    )

    assert summary["action_required_count"] == 1
    assert summary["known_missed_count"] == 0


def test_missing_live_dates_before_start_only_are_known_ok():
    summary = classify_health_warnings(
        [
            {
                "section": "SQLite",
                "name": "Missing live dates",
                "status": "WARN",
                "message": "Only pre-operation feature dates are excluded.",
                "details": {
                    "pre_start_excluded_date_count": 49,
                    "missing_prediction_count_after_start": 0,
                },
            }
        ]
    )

    assert summary["known_ok_count"] == 1
    assert summary["action_required_count"] == 0
