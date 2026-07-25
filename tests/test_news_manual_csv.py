import pandas as pd

from stock_ai.news.manual_csv import add_news_features, build_news_context, read_news_items


def test_add_news_features_uses_only_past_manual_news(tmp_path):
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "ticker": "7203.T",
                "title": "Positive production update",
                "source": "manual",
                "url": "https://example.com/positive",
                "category": "production",
                "sentiment_score": 0.5,
                "importance_score": 0.6,
                "summary": "Production outlook improved.",
            },
            {
                "date": "2024-01-06",
                "ticker": "toyota",
                "title": "High importance negative item",
                "source": "manual",
                "url": "https://example.com/negative",
                "category": "quality",
                "sentiment_score": -0.8,
                "importance_score": 0.9,
                "summary": "A negative item was registered.",
            },
            {
                "date": "2024-01-09",
                "ticker": "7203.T",
                "title": "Future news",
                "source": "manual",
                "url": "https://example.com/future",
                "category": "future",
                "sentiment_score": 0.9,
                "importance_score": 1.0,
                "summary": "This should not affect earlier rows.",
            },
        ]
    ).to_csv(news_dir / "toyota_news.csv", index=False)
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="D"),
            "ticker": ["7203.T"] * 8,
            "close": [100.0 + index for index in range(8)],
        }
    )

    result = add_news_features(data, news_dir, "7203.T")

    row_before_news = result.loc[result["date"] == pd.Timestamp("2024-01-02")].iloc[0]
    row_first_news = result.loc[result["date"] == pd.Timestamp("2024-01-03")].iloc[0]
    row_negative_news = result.loc[result["date"] == pd.Timestamp("2024-01-06")].iloc[0]
    latest = result.iloc[-1]

    assert row_before_news["news_count_3d"] == 0
    assert row_first_news["news_count_1d"] == 1
    assert row_negative_news["negative_news_flag"] == 1
    assert row_negative_news["max_importance_3d"] == 0.9
    assert latest["news_count_3d"] == 1
    assert latest["avg_sentiment_3d"] == -0.8
    assert latest["positive_news_flag"] == 0
    assert latest["importance_weighted_sentiment"] == -0.8
    assert latest["negative_news_shock"] == 1
    assert latest["sentiment_trend_3d"] < 0


def test_build_news_context_returns_recent_items_and_summary(tmp_path):
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    pd.DataFrame(
        [
            {
                "date": "2024-01-05",
                "ticker": "7203.T",
                "title": "Important positive item",
                "source": "manual",
                "url": "https://example.com/positive",
                "category": "earnings",
                "sentiment_score": 0.7,
                "importance_score": 0.85,
                "summary": "Important positive material.",
            }
        ]
    ).to_csv(news_dir / "toyota_news.csv", index=False)
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-03", periods=3, freq="D"),
            "ticker": ["7203.T"] * 3,
            "close": [100.0, 101.0, 102.0],
        }
    )
    data = add_news_features(data, news_dir, "7203.T")

    context = build_news_context(data, news_dir, "7203.T")
    items = read_news_items(news_dir, "7203.T")

    assert context.news_count_1d == 1
    assert context.news_count_3d == 1
    assert context.positive_news_flag
    assert context.positive_news_shock
    assert context.importance_weighted_sentiment == 0.7
    assert context.impact_summary
    assert "positive" in context.summary.lower()
    assert len(context.recent_items) == 1
    assert len(context.important_items) == 1
    assert context.sentiment_history
    assert items.iloc[0]["title"] == "Important positive item"


def test_missing_news_csv_adds_zero_features(tmp_path):
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "ticker": ["7203.T"] * 3,
            "close": [100.0, 101.0, 102.0],
        }
    )

    result = add_news_features(data, tmp_path / "news", "7203.T")

    assert result["news_count_1d"].sum() == 0
    assert result["news_count_3d"].sum() == 0
    assert result["negative_news_flag"].sum() == 0
    assert result["negative_news_shock"].sum() == 0
    assert result["importance_weighted_sentiment"].sum() == 0
