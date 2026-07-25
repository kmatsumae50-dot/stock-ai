from pathlib import Path

import pandas as pd
import pytest

from stock_ai.evaluation.news_impact import (
    NewsImpactConfig,
    build_category_source_type_impact,
    build_daily_news_profile_recommendations,
    build_news_impact_messages,
    build_news_impact_frame,
    build_news_impact_summaries,
    build_news_limit_recommendations,
    build_category_price_move_ranking,
    build_profile_news_impact_messages,
    build_profile_news_impact,
    build_daily_news_profile_recommendation_messages,
    category_source_type_examples,
    category_source_type_heatmap_frame,
    daily_news_profile_candidate_names,
    build_news_price_move_chart_frame,
    build_news_price_move_ranking,
    build_source_type_impact_messages,
    category_price_move_examples,
    current_news_impact_context,
    news_examples_for_impact,
    profile_news_impact_examples,
    recommended_daily_news_limit,
    recommended_daily_profile_names,
    summarize_category_impact,
    summarize_query_profile_impact,
    summarize_source_type_impact,
)
from stock_ai.news.profile_quality import append_fetch_history
from stock_ai.news.query_profiles import NewsQueryProfile


def _sample_feature_data() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    rows = []
    for index, date in enumerate(dates):
        negative = 1 if index in {10, 11, 12} else 0
        positive = 1 if index in {20, 21, 22} else 0
        rows.append(
            {
                "date": date,
                "ticker": "7203.T",
                "close": 100 + index,
                "volume": 1000 + index * 10,
                "news_count_1d": negative + positive,
                "news_count_3d": negative + positive,
                "avg_sentiment_1d": -0.4 if negative else 0.4 if positive else 0.0,
                "avg_sentiment_3d": -0.4 if negative else 0.4 if positive else 0.0,
                "max_importance_3d": 0.8 if negative or positive else 0.0,
                "negative_news_flag": negative,
                "positive_news_flag": positive,
                "sentiment_trend_3d": -0.3 if negative else 0.3 if positive else 0.0,
                "sentiment_trend_7d": -0.2 if negative else 0.2 if positive else 0.0,
                "importance_weighted_sentiment": -0.4 if negative else 0.4 if positive else 0.0,
                "negative_news_shock": 1 if index == 10 else 0,
                "positive_news_shock": 1 if index == 20 else 0,
                "sector_news_strength": -0.2 if negative else 0.2 if positive else 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_news_impact_frame_and_summaries():
    data = _sample_feature_data()
    config = NewsImpactConfig(min_history_rows=5)

    frame = build_news_impact_frame(data, config=config)
    summaries = build_news_impact_summaries(frame, config=config)
    messages = build_news_impact_messages(summaries, min_count=1)
    context = current_news_impact_context(frame, summaries)

    assert not frame.empty
    assert {"news_volume_bucket", "news_sentiment_bucket", "news_importance_bucket"}.issubset(
        frame.columns
    )
    assert not summaries["events"].empty
    assert "negative_news_flag" in set(summaries["events"]["event"])
    assert messages
    assert "Current news volume" in context


def test_query_profile_impact_uses_tracked_news_rows(tmp_path):
    data = _sample_feature_data()
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    pd.DataFrame(
        [
            {
                "date": "2026-01-11",
                "ticker": "7203.T",
                "title": "Toyota earnings warning",
                "source": "Google News",
                "url": "https://example.com/news",
                "category": "earnings",
                "sentiment_score": -0.5,
                "importance_score": 0.8,
                "summary": "Toyota earnings warning",
                "fetched_at": "2026-01-11T00:00:00+00:00",
                "query_profile": "toyota_earnings",
            }
        ]
    ).to_csv(news_dir / "toyota_news.csv", index=False)

    profile_summary = summarize_query_profile_impact(
        data,
        news_dir,
        "7203.T",
        config=NewsImpactConfig(min_history_rows=5),
    )

    row = profile_summary[profile_summary["query_profile"] == "toyota_earnings"].iloc[0]
    assert row["count"] == 1
    assert row["tracking_status"] == "tracked"


def test_category_impact_uses_news_category_rows(tmp_path):
    data = _sample_feature_data()
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    pd.DataFrame(
        [
            {
                "date": "2026-01-11",
                "ticker": "7203.T",
                "title": "Toyota recall safety issue",
                "source": "Google News",
                "url": "https://example.com/recall",
                "source_credibility_score": 0.5,
                "source_market_impact_weight": 0.4,
                "source_type": "aggregator",
                "category": "recall_safety",
                "sentiment_score": -0.8,
                "importance_score": 0.9,
                "sentiment_reason": "strong negative: recall",
                "importance_reason": "direct Toyota mention",
                "summary": "Toyota recall safety issue",
                "fetched_at": "2026-01-11T00:00:00+00:00",
                "query_profile": "toyota_stock",
            }
        ]
    ).to_csv(news_dir / "toyota_news.csv", index=False)

    category_summary = summarize_category_impact(
        data,
        news_dir,
        "7203.T",
        config=NewsImpactConfig(min_history_rows=5),
    )

    row = category_summary[category_summary["category"] == "recall_safety"].iloc[0]
    assert row["count"] == 1

    source_summary = summarize_source_type_impact(
        data,
        news_dir,
        "7203.T",
        config=NewsImpactConfig(min_history_rows=5),
    )

    source_row = source_summary[source_summary["source_type"] == "aggregator"].iloc[0]
    assert source_row["count"] == 1
    assert source_row["average_source_weight"] == 0.4

    messages = build_source_type_impact_messages(source_summary, min_count=5)
    assert messages
    assert "aggregator" in messages[0]
    assert "interpretation is limited" in messages[0]

    category_examples = news_examples_for_impact(
        data,
        news_dir,
        "7203.T",
        dimension="category",
        value="recall_safety",
        config=NewsImpactConfig(min_history_rows=5),
    )
    assert category_examples.iloc[0]["title"] == "Toyota recall safety issue"
    assert category_examples.iloc[0]["url"] == "https://example.com/recall"
    assert "return_5d" in category_examples.columns

    sentiment_examples = news_examples_for_impact(
        data,
        news_dir,
        "7203.T",
        dimension="sentiment_bucket",
        value="Negative",
        config=NewsImpactConfig(min_history_rows=5),
    )
    assert not sentiment_examples.empty


def test_news_examples_include_following_returns_for_trading_day(tmp_path):
    data = _business_price_data(periods=25)
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-05",
                "title": "Toyota profit update",
                "url": "https://example.com/trading-day",
                "category": "earnings",
            }
        ],
    )

    examples = news_examples_for_impact(
        data,
        news_dir,
        "7203.T",
        dimension="category",
        value="earnings",
    )

    row = examples.iloc[0]
    assert row["base_trading_date"] == "2026-01-05"
    assert row["return_1d"] == pytest.approx(0.01)
    assert row["return_5d"] == pytest.approx(0.05)
    assert row["return_20d"] == pytest.approx(0.20)
    assert row["max_up_20d"] == pytest.approx(0.20)
    assert row["max_down_20d"] == 0.0


def test_news_examples_use_next_trading_day_for_non_trading_news(tmp_path):
    data = _business_price_data(periods=25)
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-10",
                "title": "Toyota weekend news",
                "url": "https://example.com/weekend",
                "category": "macro",
            }
        ],
    )

    examples = news_examples_for_impact(
        data,
        news_dir,
        "7203.T",
        dimension="category",
        value="macro",
    )

    row = examples.iloc[0]
    assert row["date"] == "2026-01-10"
    assert row["base_trading_date"] == "2026-01-12"
    assert row["return_1d"] == pytest.approx(106 / 105 - 1)


def test_news_examples_do_not_fail_when_future_data_is_short(tmp_path):
    data = _business_price_data(periods=3)
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-05",
                "title": "Toyota short history news",
                "url": "https://example.com/short",
                "category": "other",
            }
        ],
    )

    examples = news_examples_for_impact(
        data,
        news_dir,
        "7203.T",
        dimension="category",
        value="other",
    )

    row = examples.iloc[0]
    assert row["return_1d"] == pytest.approx(0.01)
    assert pd.isna(row["return_5d"])
    assert pd.isna(row["return_20d"])
    assert "not enough future data" in row["future_data_note"]


def test_news_price_move_chart_frame_for_trading_day():
    data = _business_price_data(periods=30)

    chart = build_news_price_move_chart_frame(data, "2026-01-12", pre_days=5, post_days=20)

    assert not chart.empty
    assert chart["day_offset"].min() == -5
    assert chart["day_offset"].max() == 20
    assert chart.loc[chart["day_offset"] == 0, "date"].iloc[0] == pd.Timestamp("2026-01-12")
    assert chart.loc[chart["day_offset"] == 0, "cumulative_return"].iloc[0] == 0.0
    assert chart.loc[chart["day_offset"] == 20, "cumulative_return"].iloc[0] == pytest.approx(20 / 105)


def test_news_price_move_chart_frame_uses_next_trading_day_for_weekend():
    data = _business_price_data(periods=30)

    chart = build_news_price_move_chart_frame(data, "2026-01-10", pre_days=5, post_days=20)

    assert not chart.empty
    base = chart[chart["is_base_trading_date"]].iloc[0]
    assert base["date"] == pd.Timestamp("2026-01-12")
    assert base["day_offset"] == 0


def test_news_price_move_chart_frame_handles_short_future_data():
    data = _business_price_data(periods=8)

    chart = build_news_price_move_chart_frame(data, "2026-01-12", pre_days=5, post_days=20)

    assert not chart.empty
    assert chart["day_offset"].min() == -5
    assert chart["day_offset"].max() == 2


def test_news_price_move_ranking_builds_and_sorts_return_5d(tmp_path):
    data = _ranking_price_data()
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-05",
                "title": "Toyota early positive move",
                "url": "https://example.com/early",
                "category": "earnings",
                "importance_score": 0.7,
            },
            {
                "date": "2026-01-12",
                "title": "Toyota later negative move",
                "url": "https://example.com/later",
                "category": "macro",
                "importance_score": 0.9,
            },
        ],
    )

    ranking = build_news_price_move_ranking(
        data,
        news_dir,
        "7203.T",
        ranking_mode="return_5d_desc",
    )

    assert not ranking.empty
    assert ranking.iloc[0]["rank"] == 1
    assert ranking.iloc[0]["title"] == "Toyota early positive move"
    assert ranking.iloc[0]["return_5d"] == pytest.approx(0.5)


def test_news_price_move_ranking_sorts_max_down_20d(tmp_path):
    data = _ranking_price_data()
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-05",
                "title": "Toyota smaller drawdown",
                "url": "https://example.com/smaller",
                "category": "earnings",
            },
            {
                "date": "2026-01-12",
                "title": "Toyota larger drawdown",
                "url": "https://example.com/larger",
                "category": "macro",
            },
        ],
    )

    ranking = build_news_price_move_ranking(
        data,
        news_dir,
        "7203.T",
        ranking_mode="max_down_20d_asc",
    )

    assert ranking.iloc[0]["title"] == "Toyota larger drawdown"
    assert ranking.iloc[0]["max_down_20d"] == pytest.approx(-0.4)


def test_news_price_move_ranking_handles_short_future_data(tmp_path):
    data = _business_price_data(periods=3)
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-05",
                "title": "Toyota short future",
                "url": "https://example.com/short-future",
                "category": "other",
            }
        ],
    )

    ranking = build_news_price_move_ranking(data, news_dir, "7203.T")

    assert len(ranking) == 1
    assert pd.isna(ranking.iloc[0]["return_5d"])
    assert "not enough future data" in ranking.iloc[0]["future_data_note"]


def test_news_price_move_ranking_handles_missing_title_and_url(tmp_path):
    data = _business_price_data(periods=25)
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-05",
                "category": "other",
            }
        ],
    )

    ranking = build_news_price_move_ranking(data, news_dir, "7203.T")

    assert len(ranking) == 1
    assert "title" in ranking.columns
    assert "url" in ranking.columns


def test_category_price_move_ranking_builds_and_sorts_return_5d(tmp_path):
    data = _ranking_price_data()
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-05",
                "title": "Toyota earnings move",
                "url": "https://example.com/earnings",
                "category": "earnings",
            },
            {
                "date": "2026-01-12",
                "title": "Toyota macro move",
                "url": "https://example.com/macro",
                "category": "macro",
            },
        ],
    )

    ranking = build_category_price_move_ranking(
        data,
        news_dir,
        "7203.T",
        ranking_mode="average_return_5d_desc",
    )

    assert not ranking.empty
    assert ranking.iloc[0]["category"] == "earnings"
    assert ranking.iloc[0]["news_count"] == 1
    assert ranking.iloc[0]["average_return_5d"] == pytest.approx(0.5)


def test_category_price_move_ranking_sorts_max_down_20d(tmp_path):
    data = _ranking_price_data()
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-05",
                "title": "Toyota earnings move",
                "url": "https://example.com/earnings",
                "category": "earnings",
            },
            {
                "date": "2026-01-12",
                "title": "Toyota macro drawdown",
                "url": "https://example.com/macro",
                "category": "macro",
            },
        ],
    )

    ranking = build_category_price_move_ranking(
        data,
        news_dir,
        "7203.T",
        ranking_mode="average_max_down_20d_asc",
    )

    assert ranking.iloc[0]["category"] == "macro"
    assert ranking.iloc[0]["average_max_down_20d"] == pytest.approx(-0.4)


def test_category_price_move_examples_returns_representative_news(tmp_path):
    data = _ranking_price_data()
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-05",
                "title": "Toyota earnings one",
                "url": "https://example.com/one",
                "category": "earnings",
                "importance_score": 0.6,
            },
            {
                "date": "2026-01-06",
                "title": "Toyota earnings two",
                "url": "https://example.com/two",
                "category": "earnings",
                "importance_score": 0.9,
            },
        ],
    )

    examples = category_price_move_examples(data, news_dir, "7203.T", "earnings", limit=1)

    assert len(examples) == 1
    assert examples.iloc[0]["category"] == "earnings"
    assert examples.iloc[0]["title"] in {"Toyota earnings one", "Toyota earnings two"}
    assert "return_5d" in examples.columns


def test_category_price_move_ranking_handles_missing_title_and_url(tmp_path):
    data = _business_price_data(periods=25)
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-05",
                "category": "",
            }
        ],
    )

    ranking = build_category_price_move_ranking(data, news_dir, "7203.T")
    examples = category_price_move_examples(data, news_dir, "7203.T", "other")

    assert ranking.iloc[0]["category"] == "other"
    assert len(examples) == 1
    assert "title" in examples.columns
    assert "url" in examples.columns


def test_category_source_type_impact_builds_and_sorts_return_5d(tmp_path):
    data = _ranking_price_data()
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-05",
                "title": "Toyota earnings wire move",
                "url": "https://example.com/earnings-wire",
                "category": "earnings",
                "source_type": "wire",
            },
            {
                "date": "2026-01-12",
                "title": "Toyota macro media move",
                "url": "https://example.com/macro-media",
                "category": "macro",
                "source_type": "financial_media",
            },
        ],
    )

    ranking = build_category_source_type_impact(
        data,
        news_dir,
        "7203.T",
        ranking_mode="average_return_5d_desc",
    )

    assert not ranking.empty
    assert ranking.iloc[0]["category"] == "earnings"
    assert ranking.iloc[0]["source_type"] == "wire"
    assert ranking.iloc[0]["news_count"] == 1
    assert ranking.iloc[0]["average_return_5d"] == pytest.approx(0.5)


def test_category_source_type_impact_sorts_max_down_20d(tmp_path):
    data = _ranking_price_data()
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-05",
                "title": "Toyota earnings wire",
                "url": "https://example.com/earnings-wire",
                "category": "earnings",
                "source_type": "wire",
            },
            {
                "date": "2026-01-12",
                "title": "Toyota macro media drawdown",
                "url": "https://example.com/macro-media",
                "category": "macro",
                "source_type": "financial_media",
            },
        ],
    )

    ranking = build_category_source_type_impact(
        data,
        news_dir,
        "7203.T",
        ranking_mode="average_max_down_20d_asc",
    )

    assert ranking.iloc[0]["category"] == "macro"
    assert ranking.iloc[0]["source_type"] == "financial_media"
    assert ranking.iloc[0]["average_max_down_20d"] == pytest.approx(-0.4)


def test_category_source_type_heatmap_frame_builds(tmp_path):
    data = _ranking_price_data()
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-05",
                "title": "Toyota earnings wire",
                "category": "earnings",
                "source_type": "wire",
            },
            {
                "date": "2026-01-12",
                "title": "Toyota macro media",
                "category": "macro",
                "source_type": "financial_media",
            },
        ],
    )
    ranking = build_category_source_type_impact(data, news_dir, "7203.T")

    heatmap = category_source_type_heatmap_frame(ranking, metric="average_return_5d")

    assert not heatmap.empty
    assert "wire" in heatmap.columns
    assert "earnings" in heatmap.index


def test_category_source_type_examples_returns_representative_news(tmp_path):
    data = _ranking_price_data()
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-05",
                "title": "Toyota earnings wire one",
                "url": "https://example.com/one",
                "category": "earnings",
                "source_type": "wire",
                "importance_score": 0.5,
            },
            {
                "date": "2026-01-06",
                "title": "Toyota earnings wire two",
                "url": "https://example.com/two",
                "category": "earnings",
                "source_type": "wire",
                "importance_score": 0.9,
            },
        ],
    )

    examples = category_source_type_examples(
        data,
        news_dir,
        "7203.T",
        "earnings",
        "wire",
        limit=1,
    )

    assert len(examples) == 1
    assert examples.iloc[0]["category"] == "earnings"
    assert examples.iloc[0]["source_type"] == "wire"
    assert "return_5d" in examples.columns


def test_category_source_type_impact_handles_missing_title_and_url(tmp_path):
    data = _business_price_data(periods=25)
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-05",
                "category": "",
                "source_type": "",
            }
        ],
    )

    ranking = build_category_source_type_impact(data, news_dir, "7203.T")
    examples = category_source_type_examples(data, news_dir, "7203.T", "other", "unknown")

    assert ranking.iloc[0]["category"] == "other"
    assert ranking.iloc[0]["source_type"] == "unknown"
    assert len(examples) == 1
    assert "title" in examples.columns
    assert "url" in examples.columns


def test_profile_news_impact_builds_and_joins_quality(tmp_path):
    data = _ranking_price_data()
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-05",
                "title": "Toyota stock profile move",
                "url": "https://example.com/profile-stock",
                "query_profile": "toyota_stock",
                "importance_score": 0.7,
            },
            {
                "date": "2026-01-12",
                "title": "Toyota earnings profile move",
                "url": "https://example.com/profile-earnings",
                "query_profile": "toyota_earnings",
                "importance_score": 0.9,
            },
        ],
    )
    append_fetch_history(
        news_dir,
        profile=NewsQueryProfile(
            name="toyota_stock",
            query="Toyota stock",
            ticker="7203.T",
            group="core",
            enabled_for_daily=True,
        ),
        query="Toyota stock",
        fetched_count=10,
        added_count=2,
        duplicate_count=8,
    )

    ranking = build_profile_news_impact(
        data,
        news_dir,
        "7203.T",
        ranking_mode="average_return_5d_desc",
    )
    stock_row = ranking[ranking["query_profile"] == "toyota_stock"].iloc[0]

    assert not ranking.empty
    assert stock_row["news_count"] == 1
    assert stock_row["average_return_5d"] == pytest.approx(0.5)
    assert stock_row["total_fetched_rows"] == 10
    assert stock_row["added_count"] == 2
    assert stock_row["duplicate_rate"] == pytest.approx(0.8)
    assert stock_row["recommendation"] == "review_needed"
    assert "toyota_stock" in stock_row["daily_decision_note"]


def test_profile_news_impact_sorts_max_down_20d(tmp_path):
    data = _ranking_price_data()
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-05",
                "title": "Toyota profile smaller drawdown",
                "url": "https://example.com/profile-small",
                "query_profile": "toyota_stock",
            },
            {
                "date": "2026-01-12",
                "title": "Toyota profile larger drawdown",
                "url": "https://example.com/profile-large",
                "query_profile": "toyota_earnings",
            },
        ],
    )

    ranking = build_profile_news_impact(
        data,
        news_dir,
        "7203.T",
        ranking_mode="average_max_down_20d_asc",
    )

    assert ranking.iloc[0]["query_profile"] == "toyota_earnings"
    assert ranking.iloc[0]["average_max_down_20d"] == pytest.approx(-0.4)


def test_profile_news_impact_keeps_research_only_and_messages(tmp_path):
    data = _business_price_data(periods=25)
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-05",
                "title": "Toyota research profile",
                "url": "https://example.com/research",
                "query_profile": "toyota_recall_jp",
            }
        ],
    )

    ranking = build_profile_news_impact(data, news_dir, "7203.T")
    recall_row = ranking[ranking["query_profile"] == "toyota_recall_jp"].iloc[0]
    messages = build_profile_news_impact_messages(ranking, max_items=3, min_count=3)

    assert recall_row["recommendation"] == "research_only"
    assert "only 1 saved rows" in recall_row["daily_decision_note"]
    assert messages


def test_profile_news_impact_examples_handle_missing_title_and_url(tmp_path):
    data = _business_price_data(periods=25)
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-05",
                "query_profile": "toyota_stock",
                "title": "",
                "url": "",
            }
        ],
    )

    examples = profile_news_impact_examples(data, news_dir, "7203.T", "toyota_stock")

    assert len(examples) == 1
    assert "title" in examples.columns
    assert "url" in examples.columns
    assert "return_5d" in examples.columns


def test_daily_news_profile_recommendations_build_report():
    profile_impact = pd.DataFrame(
        [
            {
                "query_profile": "daily_good",
                "group": "core",
                "enabled_for_daily": True,
                "language": "en",
                "region": "US",
                "news_count": 8,
                "total_saved_rows": 8,
                "latest_fetched_at": "2026-06-01T00:00:00+00:00",
                "added_count": 4,
                "duplicate_count": 1,
                "duplicate_rate": 0.20,
                "quality_status": "OK",
                "recommendation": "daily_keep",
                "average_return_5d": 0.015,
                "average_return_20d": 0.02,
                "average_max_up_20d": 0.04,
                "average_max_down_20d": -0.02,
                "large_up_rate": 0.25,
                "large_down_rate": 0.10,
                "average_crash_risk": 30.0,
                "average_surge_risk": 40.0,
                "average_importance_score": 0.70,
                "average_source_credibility_score": 0.80,
            },
            {
                "query_profile": "daily_dup",
                "group": "core",
                "enabled_for_daily": True,
                "news_count": 4,
                "total_saved_rows": 4,
                "added_count": 0,
                "duplicate_count": 10,
                "duplicate_rate": 0.95,
                "quality_status": "WARN",
                "recommendation": "review_needed",
                "average_return_5d": -0.01,
            },
            {
                "query_profile": "research_event",
                "group": "research",
                "enabled_for_daily": False,
                "news_count": 1,
                "total_saved_rows": 1,
                "added_count": 1,
                "duplicate_count": 0,
                "duplicate_rate": 0.0,
                "quality_status": "OK",
                "recommendation": "research_only",
            },
            {
                "query_profile": "disabled_empty",
                "group": "macro",
                "enabled_for_daily": True,
                "news_count": 0,
                "total_saved_rows": 0,
                "added_count": 0,
                "duplicate_count": 0,
                "duplicate_rate": 0.0,
                "quality_status": "OK",
                "recommendation": "daily_keep",
            },
        ]
    )

    recommendations = build_daily_news_profile_recommendations(profile_impact)
    labels = dict(
        zip(
            recommendations["query_profile"],
            recommendations["daily_operation_recommendation"],
            strict=False,
        )
    )

    assert labels["daily_good"] == "daily_keep"
    assert labels["daily_dup"] == "reduce_limit"
    assert labels["research_event"] == "research_only"
    assert labels["disabled_empty"] == "disable_candidate"
    assert "daily_good" in recommended_daily_profile_names(recommendations)
    assert "NewsLimit" in recommendations[
        recommendations["query_profile"] == "daily_dup"
    ].iloc[0]["config_diff"]


def test_daily_news_profile_recommendations_join_quality_and_impact(tmp_path):
    data = _ranking_price_data()
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    _write_news(
        news_dir,
        [
            {
                "date": "2026-01-05",
                "title": "Toyota profile quality impact",
                "url": "https://example.com/profile-quality",
                "query_profile": "toyota_stock",
                "importance_score": 0.9,
            }
        ],
    )
    append_fetch_history(
        news_dir,
        profile=NewsQueryProfile(
            name="toyota_stock",
            query="Toyota stock",
            ticker="7203.T",
            group="core",
            enabled_for_daily=True,
        ),
        query="Toyota stock",
        fetched_count=10,
        added_count=1,
        duplicate_count=9,
    )

    impact = build_profile_news_impact(data, news_dir, "7203.T")
    recommendations = build_daily_news_profile_recommendations(impact)
    stock_row = recommendations[recommendations["query_profile"] == "toyota_stock"].iloc[0]

    assert stock_row["news_count"] == 1
    assert stock_row["added_count"] == 1
    assert stock_row["duplicate_count"] == 9
    assert stock_row["daily_operation_recommendation"] == "reduce_limit"


def test_daily_news_profile_recommendations_candidate_and_messages():
    profile_impact = pd.DataFrame(
        [
            {
                "query_profile": "macro_candidate",
                "group": "macro",
                "enabled_for_daily": False,
                "news_count": 5,
                "total_saved_rows": 5,
                "added_count": 2,
                "duplicate_count": 1,
                "duplicate_rate": 0.25,
                "quality_status": "OK",
                "recommendation": "research_only",
                "average_return_5d": -0.03,
                "large_down_rate": 0.40,
                "average_importance_score": 0.8,
                "average_source_credibility_score": 0.75,
            }
        ]
    )

    recommendations = build_daily_news_profile_recommendations(profile_impact)
    row = recommendations.iloc[0]
    messages = build_daily_news_profile_recommendation_messages(recommendations, max_items=1)

    assert row["daily_operation_recommendation"] == "daily_candidate"
    assert bool(row["suggested_enabled_for_daily"]) is True
    assert daily_news_profile_candidate_names(recommendations) == ["macro_candidate"]
    assert "macro_candidate" in messages[0]


def test_daily_news_profile_recommendations_no_config_write():
    config_path = Path("configs/news_queries.yaml")
    before = config_path.read_text(encoding="utf-8")

    recommendations = build_daily_news_profile_recommendations(
        pd.DataFrame(
            [
                {
                    "query_profile": "toyota_stock",
                    "group": "core",
                    "enabled_for_daily": True,
                    "news_count": 1,
                    "total_saved_rows": 1,
                    "added_count": 0,
                    "duplicate_count": 5,
                    "duplicate_rate": 1.0,
                    "quality_status": "WARN",
                    "recommendation": "review_needed",
                }
            ]
        )
    )
    after = config_path.read_text(encoding="utf-8")

    assert not recommendations.empty
    assert before == after


def test_news_limit_recommendations_choose_limit_3_for_duplicate_heavy_profile():
    recommendations = pd.DataFrame(
        [
            {
                "query_profile": "daily_dup",
                "enabled_for_daily": True,
                "group": "core",
                "latest_fetched_count": 10,
                "latest_added_count": 0,
                "latest_duplicate_count": 10,
                "duplicate_rate": 0.95,
                "total_saved_rows": 4,
                "quality_status": "WARN",
                "daily_operation_recommendation": "reduce_limit",
                "average_importance_score": 0.6,
            }
        ]
    )

    limit_frame = build_news_limit_recommendations(recommendations)
    row = limit_frame.iloc[0]

    assert row["suggested_daily_news_limit"] == 3
    assert row["suggested_research_news_limit"] == 10
    assert "duplicate-heavy" in row["reason"]


def test_news_limit_recommendations_choose_limit_5_for_profiles_that_add_news():
    recommendations = pd.DataFrame(
        [
            {
                "query_profile": "daily_good",
                "enabled_for_daily": True,
                "group": "core",
                "latest_fetched_count": 10,
                "latest_added_count": 3,
                "latest_duplicate_count": 4,
                "duplicate_rate": 0.40,
                "total_saved_rows": 12,
                "quality_status": "OK",
                "daily_operation_recommendation": "daily_keep",
                "average_importance_score": 0.7,
            }
        ]
    )

    limit_frame = build_news_limit_recommendations(recommendations)
    row = limit_frame.iloc[0]

    assert row["suggested_daily_news_limit"] == 5
    assert recommended_daily_news_limit(limit_frame) == 5
    assert row["limit_5_assessment"] == "standard daily"


def test_news_limit_recommendations_keep_research_only_out_of_daily():
    recommendations = pd.DataFrame(
        [
            {
                "query_profile": "toyota_recall_jp",
                "enabled_for_daily": False,
                "group": "research",
                "latest_fetched_count": 10,
                "latest_added_count": 1,
                "latest_duplicate_count": 9,
                "duplicate_rate": 0.90,
                "total_saved_rows": 1,
                "quality_status": "OK",
                "daily_operation_recommendation": "research_only",
                "average_importance_score": 0.8,
            }
        ]
    )

    limit_frame = build_news_limit_recommendations(recommendations)
    row = limit_frame.iloc[0]

    assert row["suggested_daily_news_limit"] == "daily_off"
    assert row["suggested_research_news_limit"] == 10
    assert row["limit_10_assessment"] == "research/manual"


def test_news_limit_recommendations_do_not_write_config():
    config_path = Path("configs/news_queries.yaml")
    before = config_path.read_text(encoding="utf-8")

    limit_frame = build_news_limit_recommendations(
        pd.DataFrame(
            [
                {
                    "query_profile": "toyota_stock",
                    "enabled_for_daily": True,
                    "group": "core",
                    "latest_fetched_count": 10,
                    "latest_added_count": 0,
                    "latest_duplicate_count": 10,
                    "duplicate_rate": 1.0,
                    "total_saved_rows": 1,
                    "quality_status": "WARN",
                    "daily_operation_recommendation": "reduce_limit",
                }
            ]
        )
    )
    after = config_path.read_text(encoding="utf-8")

    assert not limit_frame.empty
    assert before == after


def _business_price_data(*, periods: int) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=periods)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": "7203.T",
            "close": [100 + index for index in range(periods)],
        }
    )


def _ranking_price_data() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=30)
    closes = [
        100,
        100,
        100,
        100,
        100,
        150,
        140,
        130,
        120,
        110,
        90,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
    ]
    return pd.DataFrame({"date": dates, "ticker": "7203.T", "close": closes})


def _write_news(news_dir, rows: list[dict[str, object]]) -> None:
    defaults = {
        "ticker": "7203.T",
        "source": "Reuters",
        "source_credibility_score": 0.9,
        "source_market_impact_weight": 0.8,
        "source_type": "wire",
        "sentiment_score": 0.2,
        "importance_score": 0.8,
        "sentiment_reason": "test sentiment",
        "importance_reason": "test importance",
        "summary": "Toyota test news",
        "fetched_at": "2026-01-05T00:00:00+00:00",
        "query_profile": "toyota_stock",
    }
    frame = pd.DataFrame([{**defaults, **row} for row in rows])
    frame.to_csv(news_dir / "toyota_news.csv", index=False)
