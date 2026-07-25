import pandas as pd

from stock_ai.news.quality import check_news_csv
from stock_ai.news.refresh_scores import refresh_news_scores_csv


def test_refresh_news_scores_dry_run_does_not_write(tmp_path):
    path = tmp_path / "toyota_news.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "ticker": "7203.T",
                "title": "Toyota recall lawsuit probe",
                "source": "Reuters",
                "url": "https://example.com/recall",
                "category": "",
                "sentiment_score": "",
                "importance_score": "",
                "summary": "Toyota safety issue investigation.",
                "fetched_at": "2026-06-02T00:00:00+00:00",
            }
        ]
    ).to_csv(path, index=False)

    result = refresh_news_scores_csv(path, ticker="7203.T", dry_run=True)
    unchanged = pd.read_csv(path, dtype=str, keep_default_na=False)

    assert result.dry_run is True
    assert result.row_count == 1
    assert result.updated_count == 1
    assert result.category_distribution["recall_safety"] == 1
    assert result.sentiment_distribution["negative"] == 1
    assert "sentiment_reason" not in unchanged.columns


def test_refresh_news_scores_writes_reason_columns_and_check_news_passes(tmp_path):
    path = tmp_path / "toyota_news.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "ticker": "7203.T",
                "title": "Toyota record profit and buyback",
                "source": "Reuters",
                "url": "https://example.com/profit",
                "category": "",
                "sentiment_score": "",
                "importance_score": "",
                "summary": "Toyota beat expectations with strong demand.",
                "fetched_at": "2026-06-02T00:00:00+00:00",
            }
        ]
    ).to_csv(path, index=False)

    result = refresh_news_scores_csv(path, ticker="7203.T")
    refreshed = pd.read_csv(path, dtype=str, keep_default_na=False)
    quality = check_news_csv(path)

    assert result.dry_run is False
    assert result.updated_count == 1
    assert "sentiment_reason" in refreshed.columns
    assert "importance_reason" in refreshed.columns
    assert "source_credibility_score" in refreshed.columns
    assert "source_market_impact_weight" in refreshed.columns
    assert "source_type" in refreshed.columns
    assert refreshed.loc[0, "category"] == "earnings"
    assert refreshed.loc[0, "source_type"] == "wire"
    assert "strong positive" in refreshed.loc[0, "sentiment_reason"]
    assert "direct Toyota mention" in refreshed.loc[0, "importance_reason"]
    assert quality["status"] == "OK"
    assert quality["missing_sentiment_reason_count"] == 0
    assert quality["missing_importance_reason_count"] == 0
    assert quality["missing_source_credibility_count"] == 0
    assert quality["unknown_source_count"] == 0
