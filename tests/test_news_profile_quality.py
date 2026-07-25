import pandas as pd

from stock_ai.news.profile_quality import (
    append_fetch_history,
    build_daily_profile_before_after_evaluation,
    build_daily_profile_config_review,
    build_news_quality_history,
    build_newslimit3_daily_summary,
    build_newslimit3_evaluation,
    build_newslimit3_profile_evaluation,
    build_profile_quality_frame,
    build_profile_recommendations,
    read_fetch_history,
    recent_fetch_history,
    save_daily_profile_before_after_report,
    save_daily_profile_config_review_report,
    save_newslimit3_evaluation_report,
    update_latest_fetch_history_quality,
)
from stock_ai.news.query_profiles import NewsQueryProfile


def _profiles() -> list[NewsQueryProfile]:
    return [
        NewsQueryProfile(
            name="toyota_stock",
            query="Toyota stock",
            ticker="7203.T",
            group="core",
            enabled_for_daily=True,
        ),
        NewsQueryProfile(
            name="toyota_recall_jp",
            query="Toyota recall",
            ticker="7203.T",
            group="research",
            language="ja",
            region="JP",
            enabled_for_daily=False,
        ),
        NewsQueryProfile(
            name="empty_daily",
            query="Toyota earnings",
            ticker="7203.T",
            group="core",
            enabled_for_daily=True,
        ),
    ]


def _write_news_csv(news_dir, rows):
    news_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(news_dir / "toyota_news.csv", index=False)


def test_fetch_history_is_appended_and_read(tmp_path):
    news_dir = tmp_path / "news"
    profile = _profiles()[0]

    path = append_fetch_history(
        news_dir,
        profile=profile,
        query=profile.query,
        fetched_count=10,
        added_count=2,
        duplicate_count=8,
    )
    history = read_fetch_history(path)

    assert path.exists()
    assert len(history) == 1
    assert history.loc[0, "profile"] == "toyota_stock"
    assert history.loc[0, "fetched_count"] == 10
    assert history.loc[0, "added_count"] == 2
    assert history.loc[0, "duplicate_count"] == 8
    assert bool(history.loc[0, "enabled_for_daily"]) is True
    assert pd.isna(history.loc[0, "unknown_source_count"])


def test_profile_quality_frame_aggregates_saved_rows_and_duplicate_rate(tmp_path):
    news_dir = tmp_path / "news"
    profiles = _profiles()
    _write_news_csv(
        news_dir,
        [
            {
                "date": "2026-06-01",
                "ticker": "7203.T",
                "title": "Toyota stock gains",
                "source": "Reuters",
                "url": "https://example.com/1",
                "source_credibility_score": "0.90",
                "source_market_impact_weight": "0.90",
                "source_type": "wire",
                "category": "earnings",
                "sentiment_score": "0.50",
                "importance_score": "0.80",
                "summary": "Toyota profit beats expectations",
                "fetched_at": "2026-06-01T09:00:00+00:00",
                "query_profile": "toyota_stock",
            },
            {
                "date": "bad-date",
                "ticker": "7203.T",
                "title": "Toyota recall warning",
                "source": "Yahoo Finance",
                "url": "https://example.com/2",
                "source_credibility_score": "0.70",
                "source_market_impact_weight": "0.65",
                "source_type": "financial_media",
                "category": "recall_safety",
                "sentiment_score": "-0.60",
                "importance_score": "0.75",
                "summary": "Safety issue reported",
                "fetched_at": "2026-06-02T09:00:00+00:00",
                "query_profile": "toyota_stock",
            },
            {
                "date": "2026-06-02",
                "ticker": "7203.T",
                "title": "Recall item",
                "source": "Local News",
                "url": "https://example.com/3",
                "source_credibility_score": "0.45",
                "source_market_impact_weight": "0.40",
                "source_type": "general_news",
                "category": "recall_safety",
                "sentiment_score": "-0.20",
                "importance_score": "0.55",
                "summary": "Toyota recall details",
                "fetched_at": "2026-06-02T10:00:00+00:00",
                "query_profile": "toyota_recall_jp",
            },
        ],
    )
    append_fetch_history(
        news_dir,
        profile=profiles[0],
        query=profiles[0].query,
        fetched_count=10,
        added_count=2,
        duplicate_count=8,
    )

    quality = build_profile_quality_frame(news_dir, "7203.T", profiles=profiles)
    row = quality.set_index("profile").loc["toyota_stock"]

    assert int(row["saved_rows"]) == 2
    assert int(row["history_fetched_count"]) == 10
    assert int(row["history_duplicate_count"]) == 8
    assert row["duplicate_rate"] == 0.8
    assert int(row["unique_title_count"]) == 2
    assert int(row["unique_url_count"]) == 2
    assert "earnings:1" in row["category_distribution"]
    assert "wire:1" in row["source_type_distribution"]
    assert int(row["invalid_date_count"]) == 1


def test_profile_recommendations_flag_review_needed_and_research_only(tmp_path):
    news_dir = tmp_path / "news"
    profiles = _profiles()
    append_fetch_history(
        news_dir,
        profile=profiles[0],
        query=profiles[0].query,
        fetched_count=10,
        added_count=1,
        duplicate_count=9,
    )
    quality = build_profile_quality_frame(news_dir, "7203.T", profiles=profiles)
    recommendations = build_profile_recommendations(quality).set_index("profile")

    assert recommendations.loc["toyota_stock", "recommendation"] == "review_needed"
    assert recommendations.loc["empty_daily", "recommendation"] == "review_needed"
    assert recommendations.loc["toyota_recall_jp", "recommendation"] == "research_only"


def test_profile_quality_handles_old_csv_without_profile_columns(tmp_path):
    news_dir = tmp_path / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "title": "Old CSV row",
                "source": "Reuters",
                "url": "https://example.com/old",
            }
        ],
    ).to_csv(news_dir / "toyota_news.csv", index=False)

    quality = build_profile_quality_frame(news_dir, "7203.T", profiles=_profiles())
    recent = recent_fetch_history(news_dir)

    assert not quality.empty
    assert quality["saved_rows"].sum() == 0
    assert recent.empty


def test_newslimit3_daily_summary_uses_jst_dates(tmp_path):
    news_dir = tmp_path / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "fetched_at": "2026-06-04T23:30:00+00:00",
                "profile": "toyota_stock",
                "query": "Toyota stock",
                "group": "core",
                "enabled_for_daily": True,
                "language": "",
                "region": "",
                "fetched_count": 3,
                "added_count": 1,
                "duplicate_count": 2,
                "warning": "",
                "error": "",
            },
            {
                "fetched_at": "2026-06-05T00:01:00+00:00",
                "profile": "toyota_jp",
                "query": "トヨタ 株価 ニュース",
                "group": "jp",
                "enabled_for_daily": True,
                "language": "ja",
                "region": "JP",
                "fetched_count": 3,
                "added_count": 2,
                "duplicate_count": 1,
                "warning": "",
                "error": "",
            },
        ],
    ).to_csv(news_dir / "fetch_history.csv", index=False)

    daily = build_newslimit3_daily_summary(news_dir, start_date="2026-06-05")

    assert len(daily) == 1
    assert daily.loc[0, "date"] == "2026-06-05"
    assert int(daily.loc[0, "fetched"]) == 6
    assert int(daily.loc[0, "added"]) == 3
    assert int(daily.loc[0, "profiles_count"]) == 2


def test_newslimit3_profile_evaluation_recommends_reduce_for_duplicate_heavy_profile(tmp_path):
    news_dir = tmp_path / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for day in range(5, 8):
        rows.append(
            {
                "fetched_at": f"2026-06-{day:02d}T00:00:00+00:00",
                "profile": "toyota_earnings",
                "query": "Toyota earnings",
                "group": "core",
                "enabled_for_daily": True,
                "language": "",
                "region": "",
                "fetched_count": 3,
                "added_count": 0,
                "duplicate_count": 3,
                "warning": "",
                "error": "",
            }
        )
    pd.DataFrame(rows).to_csv(news_dir / "fetch_history.csv", index=False)

    profiles = build_newslimit3_profile_evaluation(news_dir, start_date="2026-06-05")
    row = profiles.set_index("profile").loc["toyota_earnings"]

    assert int(row["runs"]) == 3
    assert int(row["total_fetched"]) == 9
    assert row["daily_operation_recommendation"] == "daily_reduce_limit"


def test_newslimit3_evaluation_report_is_written_without_changing_configs(tmp_path):
    news_dir = tmp_path / "news"
    report_dir = tmp_path / "reports"
    news_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "fetched_at": "2026-06-05T00:00:00+00:00",
                "profile": "toyota_stock",
                "query": "Toyota stock",
                "group": "core",
                "enabled_for_daily": True,
                "language": "",
                "region": "",
                "fetched_count": 3,
                "added_count": 1,
                "duplicate_count": 2,
                "warning": "",
                "error": "",
            }
        ]
    ).to_csv(news_dir / "fetch_history.csv", index=False)

    evaluation = build_newslimit3_evaluation(news_dir, start_date="2026-06-05")
    path = save_newslimit3_evaluation_report(evaluation, report_dir, generated_date="2026-06-10")

    assert path is not None
    assert path.exists()
    saved = pd.read_csv(path)
    assert "overall_recommendation" in saved.columns
    assert saved.loc[0, "profile"] == "toyota_stock"


def test_daily_profile_config_review_flags_zero_add_earnings_profile(tmp_path):
    news_dir = tmp_path / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for day in range(5, 11):
        rows.append(
            {
                "fetched_at": f"2026-06-{day:02d}T00:00:00+00:00",
                "profile": "toyota_earnings",
                "query": "Toyota earnings",
                "group": "core",
                "enabled_for_daily": True,
                "language": "",
                "region": "",
                "fetched_count": 3,
                "added_count": 0,
                "duplicate_count": 3,
                "warning": "",
                "error": "",
            }
        )
    pd.DataFrame(rows).to_csv(news_dir / "fetch_history.csv", index=False)
    profiles = [
        NewsQueryProfile(
            name="toyota_earnings",
            query="Toyota earnings",
            group="core",
            enabled_for_daily=True,
        )
    ]

    review = build_daily_profile_config_review(news_dir, start_date="2026-06-05", profiles=profiles)
    row = review.set_index("profile").loc["toyota_earnings"]

    assert row["recommendation"] == "disable_candidate"
    assert bool(row["suggested_enabled_for_daily"]) is False
    assert row["config_change"] == "enabled_for_daily true -> false"


def test_daily_profile_config_review_report_is_written(tmp_path):
    review = pd.DataFrame(
        [
            {
                "profile": "toyota_stock",
                "group": "core",
                "observed_enabled_for_daily": True,
                "current_enabled_for_daily": True,
                "current_decision": "daily_keep",
                "total_fetched": 18,
                "total_added": 5,
                "duplicate_rate": 0.72,
                "recommendation": "daily_keep",
                "suggested_enabled_for_daily": True,
                "config_change": "no_change",
                "reason": "keep",
            }
        ]
    )

    path = save_daily_profile_config_review_report(review, tmp_path, generated_date="2026-06-10")

    assert path is not None
    assert path.exists()
    saved = pd.read_csv(path)
    assert saved.loc[0, "profile"] == "toyota_stock"


def test_daily_profile_before_after_evaluation_splits_change_batches(tmp_path):
    news_dir = tmp_path / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    before_profiles = [
        "toyota_stock",
        "toyota_earnings",
        "auto_industry",
        "toyota_jp",
        "toyota_earnings_jp",
    ]
    after_profiles = [
        "toyota_stock",
        "auto_industry",
        "toyota_jp",
        "toyota_earnings_jp",
    ]
    for day in (5, 6, 7):
        for index, profile in enumerate(before_profiles):
            rows.append(
                _fetch_history_test_row(
                    fetched_at=f"2026-06-{day:02d}T00:00:{index:02d}+00:00",
                    profile=profile,
                    added_count=1 if profile == "toyota_stock" else 0,
                )
            )
    for day in (8, 9):
        for index, profile in enumerate(after_profiles):
            rows.append(
                _fetch_history_test_row(
                    fetched_at=f"2026-06-{day:02d}T00:00:{index:02d}+00:00",
                    profile=profile,
                    added_count=1 if profile in {"toyota_stock", "toyota_jp"} else 0,
                )
            )
    pd.DataFrame(rows).to_csv(news_dir / "fetch_history.csv", index=False)

    evaluation = build_daily_profile_before_after_evaluation(
        news_dir,
        profiles=_daily_profile_change_profiles(),
    )
    summary = evaluation["summary"].set_index("period")

    assert int(summary.loc["before", "observation_days"]) == 3
    assert int(summary.loc["after", "observation_days"]) == 2
    assert int(summary.loc["before", "daily_profile_count"]) == 5
    assert int(summary.loc["after", "daily_profile_count"]) == 4
    assert int(summary.loc["before", "theoretical_fetched_per_day"]) == 15
    assert int(summary.loc["after", "theoretical_fetched_per_day"]) == 12
    assert evaluation["overall_recommendation"] == "needs_more_data"


def test_daily_profile_before_after_metrics_are_calculated(tmp_path):
    news_dir = tmp_path / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for day in (5, 6, 7):
        for index, profile in enumerate(
            ["toyota_stock", "toyota_earnings", "auto_industry", "toyota_jp", "toyota_earnings_jp"]
        ):
            rows.append(
                _fetch_history_test_row(
                    fetched_at=f"2026-06-{day:02d}T00:00:{index:02d}+00:00",
                    profile=profile,
                    added_count=0,
                )
            )
    for day in (8, 9, 10):
        for index, profile in enumerate(
            ["toyota_stock", "auto_industry", "toyota_jp", "toyota_earnings_jp"]
        ):
            rows.append(
                _fetch_history_test_row(
                    fetched_at=f"2026-06-{day:02d}T00:00:{index:02d}+00:00",
                    profile=profile,
                    added_count=1 if profile == "toyota_stock" else 0,
                )
            )
    pd.DataFrame(rows).to_csv(news_dir / "fetch_history.csv", index=False)

    evaluation = build_daily_profile_before_after_evaluation(
        news_dir,
        profiles=_daily_profile_change_profiles(),
    )
    summary = evaluation["summary"].set_index("period")

    assert float(summary.loc["before", "duplicate_rate"]) == 1.0
    assert float(summary.loc["after", "duplicate_rate"]) == 33 / 36
    assert float(summary.loc["before", "added_per_day"]) == 0.0
    assert float(summary.loc["after", "added_per_day"]) == 1.0
    assert evaluation["duplicate_rate_change"] == "improved"
    assert evaluation["added_per_day_change"] == "improved"
    assert evaluation["overall_recommendation"] == "keep_4_profiles"


def test_daily_profile_before_after_report_does_not_change_config(tmp_path):
    news_dir = tmp_path / "news"
    report_dir = tmp_path / "reports"
    news_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            _fetch_history_test_row(
                fetched_at="2026-06-05T00:00:00+00:00",
                profile="toyota_earnings",
                added_count=0,
            ),
            _fetch_history_test_row(
                fetched_at="2026-06-06T00:00:00+00:00",
                profile="toyota_stock",
                added_count=1,
            ),
        ]
    ).to_csv(news_dir / "fetch_history.csv", index=False)
    config_path = tmp_path / "news_queries.yaml"
    config_path.write_text("profiles: []\n", encoding="utf-8")
    original = config_path.read_text(encoding="utf-8")

    evaluation = build_daily_profile_before_after_evaluation(
        news_dir,
        profiles=_daily_profile_change_profiles(),
    )
    path = save_daily_profile_before_after_report(
        evaluation,
        report_dir,
        generated_date="2026-06-12",
    )

    assert path is not None
    assert path.exists()
    assert config_path.read_text(encoding="utf-8") == original
    saved = pd.read_csv(path)
    assert {"period_summary", "after_profile"}.issubset(set(saved["row_type"]))


def test_check_news_quality_updates_latest_fetch_batch(tmp_path):
    news_dir = tmp_path / "news"
    profile = _profiles()[0]
    append_fetch_history(
        news_dir,
        profile=profile,
        query=profile.query,
        fetched_count=3,
        added_count=1,
        duplicate_count=2,
    )

    path = update_latest_fetch_history_quality(
        news_dir,
        {
            "status": "WARN",
            "message": "Unknown source detected.",
            "unknown_source_count": 2,
            "unknown_sources": ["Example News", "Local Feed"],
        },
    )
    history = read_fetch_history(news_dir)

    assert path is not None
    assert int(history.loc[0, "unknown_source_count"]) == 2
    assert history.loc[0, "unknown_sources"] == '["Example News", "Local Feed"]'
    assert history.loc[0, "check_news_status"] == "WARN"
    assert history.loc[0, "check_news_message"] == "Unknown source detected."


def test_before_after_unknown_source_zero_and_missing_are_distinct(tmp_path):
    news_dir = tmp_path / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    before = _fetch_history_test_row(
        fetched_at="2026-06-05T00:00:00+00:00",
        profile="toyota_earnings",
        added_count=0,
    )
    after = _fetch_history_test_row(
        fetched_at="2026-06-06T00:00:00+00:00",
        profile="toyota_stock",
        added_count=1,
    )
    after["unknown_source_count"] = 0
    after["unknown_sources"] = "[]"
    after["check_news_status"] = "OK"
    after["check_news_message"] = "News CSV quality checks passed."
    pd.DataFrame([before, after]).to_csv(news_dir / "fetch_history.csv", index=False)

    evaluation = build_daily_profile_before_after_evaluation(
        news_dir,
        profiles=_daily_profile_change_profiles(),
    )
    summary = evaluation["summary"].set_index("period")

    assert pd.isna(summary.loc["before", "unknown_source_count"])
    assert int(summary.loc["before", "unknown_source_observed_batches"]) == 0
    assert int(summary.loc["after", "unknown_source_count"]) == 0
    assert int(summary.loc["after", "unknown_source_occurrences"]) == 0
    assert int(summary.loc["after", "unknown_source_observed_batches"]) == 1
    assert int(summary.loc["after", "fetch_batch_count"]) == 1
    assert int(summary.loc["before", "unknown_source_unrecorded_batches"]) == 1
    assert int(summary.loc["after", "unknown_source_unrecorded_batches"]) == 0
    assert int(summary.loc["before", "check_news_unrecorded_count"]) == 1
    assert int(summary.loc["after", "check_news_ok_count"]) == 1
    assert int(summary.loc["after", "check_news_warn_count"]) == 0
    assert int(summary.loc["after", "check_news_error_count"]) == 0
    assert summary.loc["after", "check_news_status"] == "OK"


def test_before_after_unknown_sources_are_aggregated_by_batch(tmp_path):
    news_dir = tmp_path / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    before = _fetch_history_test_row(
        fetched_at="2026-06-05T00:00:00+00:00",
        profile="toyota_earnings",
        added_count=0,
    )
    after_rows = []
    for index, profile in enumerate(("toyota_stock", "auto_industry")):
        row = _fetch_history_test_row(
            fetched_at=f"2026-06-06T00:00:0{index}+00:00",
            profile=profile,
            added_count=1,
        )
        row["unknown_source_count"] = 1
        row["unknown_sources"] = '["Example News"]'
        row["check_news_status"] = "WARN"
        row["check_news_message"] = "Unknown source detected."
        after_rows.append(row)
    pd.DataFrame([before, *after_rows]).to_csv(news_dir / "fetch_history.csv", index=False)

    evaluation = build_daily_profile_before_after_evaluation(
        news_dir,
        profiles=_daily_profile_change_profiles(),
    )
    after_summary = evaluation["summary"].set_index("period").loc["after"]

    assert int(after_summary["unknown_source_count"]) == 1
    assert int(after_summary["unknown_source_occurrences"]) == 1
    assert int(after_summary["unknown_source_observed_batches"]) == 1
    assert int(after_summary["fetch_batch_count"]) == 1
    assert int(after_summary["check_news_warn_count"]) == 1
    assert after_summary["latest_unknown_sources"] == "Example News"


def test_five_post_change_days_keep_four_profiles_without_changing_config(tmp_path):
    news_dir = tmp_path / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    before_profiles = [
        "toyota_stock",
        "toyota_earnings",
        "auto_industry",
        "toyota_jp",
        "toyota_earnings_jp",
    ]
    after_profiles = [
        "toyota_stock",
        "auto_industry",
        "toyota_jp",
        "toyota_earnings_jp",
    ]
    for day in range(5, 11):
        for index, profile in enumerate(before_profiles):
            added = int(profile in {"toyota_stock", "toyota_jp"})
            if profile == "toyota_earnings" and day in {5, 6}:
                added = 1
            rows.append(
                _fetch_history_test_row(
                    fetched_at=f"2026-06-{day:02d}T00:00:{index:02d}+00:00",
                    profile=profile,
                    added_count=added,
                )
            )
    for day in range(11, 16):
        for index, profile in enumerate(after_profiles):
            added = int(profile == "toyota_stock")
            if profile == "toyota_jp" and day < 15:
                added = 1
            row = _fetch_history_test_row(
                fetched_at=f"2026-06-{day:02d}T00:00:{index:02d}+00:00",
                profile=profile,
                added_count=added,
            )
            row["unknown_source_count"] = 0
            row["unknown_sources"] = "[]"
            row["check_news_status"] = "OK"
            row["check_news_message"] = "News CSV quality checks passed."
            rows.append(row)
    pd.DataFrame(rows).to_csv(news_dir / "fetch_history.csv", index=False)

    evaluation = build_daily_profile_before_after_evaluation(
        news_dir,
        profiles=_daily_profile_change_profiles(),
    )
    summary = evaluation["summary"].set_index("period")
    profiles = evaluation["profile_evaluation"].set_index("profile")

    assert int(summary.loc["after", "observation_days"]) == 5
    assert int(summary.loc["after", "check_news_ok_count"]) == 5
    assert int(summary.loc["after", "unknown_source_count"]) == 0
    assert evaluation["overall_recommendation"] == "keep_4_profiles"
    assert evaluation["unknown_source_change"] == "unavailable"
    assert profiles.loc["auto_industry", "recommendation"] == "research_only_candidate"
    assert int(profiles.loc["toyota_stock", "unknown_source_observed_batches"]) == 5


def test_news_quality_history_summarizes_check_news_status_by_batch(tmp_path):
    news_dir = tmp_path / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, profile in enumerate(("toyota_stock", "auto_industry")):
        row = _fetch_history_test_row(
            fetched_at=f"2026-06-11T00:00:0{index}+00:00",
            profile=profile,
            added_count=1,
        )
        row["unknown_source_count"] = 0
        row["unknown_sources"] = "[]"
        row["check_news_status"] = "OK"
        row["check_news_message"] = "News CSV quality checks passed."
        rows.append(row)
    warning = _fetch_history_test_row(
        fetched_at="2026-06-12T00:00:00+00:00",
        profile="toyota_jp",
        added_count=1,
    )
    warning["unknown_source_count"] = 1
    warning["unknown_sources"] = '["Example News"]'
    warning["check_news_status"] = "WARN"
    warning["check_news_message"] = "Unknown source detected."
    rows.append(warning)
    pd.DataFrame(rows).to_csv(news_dir / "fetch_history.csv", index=False)

    result = build_news_quality_history(news_dir)
    summary = result["summary"]
    batches = result["batches"]

    assert summary["total_batch_count"] == 2
    assert summary["observed_batch_count"] == 2
    assert summary["unrecorded_batch_count"] == 0
    assert summary["ok_count"] == 1
    assert summary["warn_count"] == 1
    assert summary["error_count"] == 0
    assert summary["non_ok_count"] == 1
    assert summary["latest_check_news_status"] == "WARN"
    assert summary["latest_unknown_source_count"] == 1
    assert summary["latest_unknown_sources"] == "Example News"
    assert len(batches) == 2


def test_news_quality_history_distinguishes_zero_from_unrecorded(tmp_path):
    news_dir = tmp_path / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    old_row = _fetch_history_test_row(
        fetched_at="2026-06-10T00:00:00+00:00",
        profile="toyota_stock",
        added_count=1,
    )
    checked_row = _fetch_history_test_row(
        fetched_at="2026-06-11T00:00:00+00:00",
        profile="toyota_stock",
        added_count=1,
    )
    checked_row["unknown_source_count"] = 0
    checked_row["unknown_sources"] = "[]"
    checked_row["check_news_status"] = "OK"
    checked_row["check_news_message"] = "News CSV quality checks passed."
    pd.DataFrame([old_row, checked_row]).to_csv(
        news_dir / "fetch_history.csv",
        index=False,
    )

    result = build_news_quality_history(news_dir)
    summary = result["summary"]
    batches = result["batches"].set_index("fetched_at")

    assert summary["observed_batch_count"] == 1
    assert summary["unrecorded_batch_count"] == 1
    assert summary["latest_unknown_source_count"] == 0
    assert summary["latest_check_news_status"] == "OK"
    old_batch = batches.loc["2026-06-10T00:00:00+00:00"]
    assert pd.isna(old_batch["unknown_source_count"])
    assert old_batch["check_news_status"] == ""


def test_news_quality_history_prioritizes_error_status_within_batch(tmp_path):
    news_dir = tmp_path / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    ok_row = _fetch_history_test_row(
        fetched_at="2026-06-12T00:00:00+00:00",
        profile="toyota_stock",
        added_count=1,
    )
    ok_row["unknown_source_count"] = 0
    ok_row["check_news_status"] = "OK"
    error_row = _fetch_history_test_row(
        fetched_at="2026-06-12T00:00:01+00:00",
        profile="toyota_jp",
        added_count=1,
    )
    error_row["unknown_source_count"] = 0
    error_row["check_news_status"] = "ERROR"
    error_row["check_news_message"] = "News CSV could not be parsed."
    pd.DataFrame([ok_row, error_row]).to_csv(
        news_dir / "fetch_history.csv",
        index=False,
    )

    result = build_news_quality_history(news_dir)

    assert result["summary"]["error_count"] == 1
    assert result["summary"]["non_ok_count"] == 1
    assert result["summary"]["latest_check_news_status"] == "ERROR"
    assert result["summary"]["latest_check_news_message"] == "News CSV could not be parsed."


def _fetch_history_test_row(
    *,
    fetched_at: str,
    profile: str,
    added_count: int,
) -> dict[str, object]:
    return {
        "fetched_at": fetched_at,
        "profile": profile,
        "query": profile,
        "group": "core" if profile in {"toyota_stock", "toyota_earnings"} else "jp",
        "enabled_for_daily": True,
        "language": "",
        "region": "",
        "fetched_count": 3,
        "added_count": added_count,
        "duplicate_count": 3 - added_count,
        "warning": "",
        "error": "",
    }


def _daily_profile_change_profiles() -> list[NewsQueryProfile]:
    return [
        NewsQueryProfile(
            name="toyota_stock",
            query="Toyota stock",
            group="core",
            enabled_for_daily=True,
        ),
        NewsQueryProfile(
            name="toyota_earnings",
            query="Toyota earnings",
            group="core",
            enabled_for_daily=False,
        ),
        NewsQueryProfile(
            name="auto_industry",
            query="Auto industry",
            group="sector",
            enabled_for_daily=True,
        ),
        NewsQueryProfile(
            name="toyota_jp",
            query="Toyota JP",
            group="jp",
            enabled_for_daily=True,
        ),
        NewsQueryProfile(
            name="toyota_earnings_jp",
            query="Toyota earnings JP",
            group="jp",
            enabled_for_daily=True,
        ),
    ]
