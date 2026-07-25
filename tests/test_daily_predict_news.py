from stock_ai.health.checks import _daily_log_summary


def test_daily_log_summary_tracks_optional_news_warning_without_incomplete_step():
    lines = [
        "[2026-06-03 16:00:00] daily_predict started",
        "[2026-06-03 16:00:00] FetchNews enabled: True",
        "[2026-06-03 16:00:01] START: download 7203.T",
        "[2026-06-03 16:00:02] DONE: download 7203.T",
        "[2026-06-03 16:00:03] NEWS START: fetch-news 7203.T",
        "[2026-06-03 16:00:04] NEWS WARN: fetch-news 7203.T failed with exit code 1",
        "[2026-06-03 16:00:05] START: build-features 7203.T",
        "[2026-06-03 16:00:06] DONE: build-features 7203.T",
        "[2026-06-03 16:00:07] daily_predict finished successfully",
    ]

    summary = _daily_log_summary(lines, log_tail=20)

    assert summary["fetch_news_enabled"] is True
    assert "fetch-news" in str(summary["latest_news_warning"])
    assert summary["incomplete_step"] is None
    assert summary["latest_success"]


def test_daily_log_summary_tracks_fetch_news_disabled():
    lines = [
        "[2026-06-03 16:00:00] daily_predict started",
        "[2026-06-03 16:00:00] FetchNews enabled: False",
        "[2026-06-03 16:00:01] NEWS SKIP: FetchNews disabled for 7203.T",
        "[2026-06-03 16:00:02] daily_predict finished successfully",
    ]

    summary = _daily_log_summary(lines, log_tail=20)

    assert summary["fetch_news_enabled"] is False
    assert summary["latest_news_warning"] is None


def test_daily_log_summary_tracks_data_freshness_details():
    lines = [
        "[2026-06-19 10:25:00] daily_predict started",
        "[2026-06-19 10:25:01] START: data-freshness 7203.T",
        "[2026-06-19 10:25:01] expected latest trading date: 2026-06-19",
        "[2026-06-19 10:25:01] latest raw date: 2026-06-18",
        "[2026-06-19 10:25:01] latest feature date: 2026-06-18",
        "[2026-06-19 10:25:01] prediction feature date: 2026-06-18",
        "[2026-06-19 10:25:01] saved prediction date expected: 2026-06-18",
        "[2026-06-19 10:25:01] raw lag trading days: 1",
        "[2026-06-19 10:25:01] freshness lag trading days: 1",
        "[2026-06-19 10:25:01] stale at run time: true",
        "[2026-06-19 10:25:01] freshness status: WARN",
        "[2026-06-19 10:25:01] freshness message: Feature data is older than expected.",
        "[2026-06-19 10:25:02] DONE: data-freshness 7203.T",
        "[2026-06-19 10:25:03] daily_predict finished successfully",
    ]

    summary = _daily_log_summary(lines, log_tail=20)

    assert summary["data_freshness_status"] == "WARN"
    assert summary["expected_latest_trading_date"] == "2026-06-19"
    assert summary["latest_feature_date_at_run"] == "2026-06-18"
    assert summary["freshness_lag_trading_days"] == 1
    assert summary["stale_at_run_time"] is True
