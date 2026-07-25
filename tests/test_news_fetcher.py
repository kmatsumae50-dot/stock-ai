import pandas as pd

from stock_ai.news.fetcher import (
    _google_news_rss_url,
    infer_news_category,
    latest_news_status,
    merge_news_items,
    score_news_importance,
    score_news_importance_with_reason,
    score_news_sentiment,
    score_news_sentiment_with_reason,
)
from stock_ai.news.quality import check_news_csv, is_mojibake_suspect


def test_merge_news_items_dedupes_by_url_or_title():
    existing = pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "ticker": "7203.T",
                "title": "Toyota expands hybrid production",
                "source": "manual",
                "url": "https://example.com/hybrid",
                "category": "auto",
                "sentiment_score": 0.2,
                "importance_score": 0.7,
                "summary": "Manual item.",
                "fetched_at": "",
            }
        ]
    )
    fetched = pd.DataFrame(
        [
            {
                "date": "2026-06-02",
                "ticker": "7203.T",
                "title": "Toyota expands hybrid production",
                "source": "Google News",
                "url": "https://example.com/different-url",
                "category": "auto",
                "sentiment_score": 0.2,
                "importance_score": 0.7,
                "summary": "Duplicate by title.",
                "fetched_at": "2026-06-02T00:00:00+00:00",
            },
            {
                "date": "2026-06-02",
                "ticker": "7203.T",
                "title": "Toyota faces recall probe",
                "source": "Google News",
                "url": "https://example.com/recall",
                "category": "auto",
                "sentiment_score": -0.5,
                "importance_score": 0.8,
                "summary": "New item.",
                "fetched_at": "2026-06-02T00:00:00+00:00",
            },
        ]
    )

    combined, added_count, duplicate_count = merge_news_items(existing, fetched)

    assert added_count == 1
    assert duplicate_count == 1
    assert len(combined) == 2
    assert "Toyota faces recall probe" in set(combined["title"])


def test_news_scoring_rules_and_status(tmp_path):
    assert score_news_sentiment("Toyota recall lawsuit probe") < 0
    assert score_news_sentiment("Toyota record profit and strong demand") > 0
    assert score_news_importance("Toyota EV hybrid China update", ticker="7203.T") >= 0.5

    negative = score_news_sentiment_with_reason("Toyota recall lawsuit safety issue")
    positive = score_news_sentiment_with_reason(
        "Toyota record profit buyback and strong demand",
    )
    importance = score_news_importance_with_reason(
        "Toyota earnings guidance China EV update",
        ticker="7203.T",
        title="Toyota earnings guidance China EV update",
        source="Reuters",
    )

    assert negative.score <= -0.9
    assert "strong negative" in negative.reason
    assert positive.score >= 0.9
    assert "strong positive" in positive.reason
    assert importance.score >= 0.8
    assert "direct Toyota mention" in importance.reason
    assert "major source" in importance.reason
    assert infer_news_category("Toyota recall safety investigation") == "recall_safety"
    assert infer_news_category("Toyota earnings guidance profit") == "earnings"
    assert infer_news_category("Toyota EV hybrid demand") == "ev_hybrid"
    assert score_news_sentiment("トヨタ リコール 不具合 調査") < 0
    assert score_news_sentiment("トヨタ 最高益 増益 上方修正") > 0
    assert score_news_importance("トヨタ 決算 中国 EV ハイブリッド", ticker="7203.T") >= 0.5
    assert infer_news_category("トヨタ 決算 業績 見通し") == "earnings"
    assert infer_news_category("トヨタ 円安 関税 金利") == "macro"
    assert infer_news_category("トヨタ リコール 不具合 安全") == "recall_safety"
    assert infer_news_category("トヨタ ハイブリッド 電動車") == "ev_hybrid"

    path = tmp_path / "toyota_news.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "ticker": "7203.T",
                "title": "Toyota update",
                "source": "Google News",
                "url": "https://example.com/item",
                "category": "auto",
                "sentiment_score": 0.1,
                "importance_score": 0.5,
                "summary": "Summary",
                "fetched_at": "2026-06-02T01:02:03+00:00",
            }
        ]
    ).to_csv(path, index=False)

    status = latest_news_status(path)

    assert status["count"] == 1
    assert status["latest_date"] == "2026-06-01 00:00:00"
    assert status["latest_fetched_at"] == "2026-06-02 01:02:03"


def test_google_news_rss_url_supports_japanese_region():
    url = _google_news_rss_url("トヨタ 株価 ニュース", language="ja", region="JP")

    assert "hl=ja" in url
    assert "gl=JP" in url
    assert "ceid=JP%3Aja" in url
    assert "%E3%83%88%E3%83%A8%E3%82%BF" in url


def test_check_news_csv_reports_quality_issues(tmp_path):
    path = tmp_path / "toyota_news.csv"
    pd.DataFrame(
        [
            {
                "date": "bad-date",
                "ticker": "7203.T",
                "title": "Toyota Japan窶冱 largest company",
                "source": "",
                "url": "https://example.com/duplicate",
                "category": "auto",
                "sentiment_score": 2.0,
                "importance_score": 1.2,
                "summary": "Mojibake suspect 窶冱",
                "fetched_at": "",
            },
            {
                "date": "2026-06-02",
                "ticker": "7203.T",
                "title": "Toyota Japan窶冱 largest company",
                "source": "Google News",
                "url": "https://example.com/duplicate",
                "category": "auto",
                "sentiment_score": 0.0,
                "importance_score": 0.5,
                "summary": "Duplicate row.",
                "fetched_at": "2026-06-02T01:02:03+00:00",
            },
        ]
    ).to_csv(path, index=False)

    report = check_news_csv(path)

    assert report["status"] == "FAIL"
    assert report["invalid_date_count"] == 1
    assert report["duplicate_count"] == 2
    assert report["mojibake_suspect_count"] == 2
    assert report["sentiment_range_status"] == "FAIL"
    assert report["importance_range_status"] == "FAIL"
    assert report["empty_source_count"] == 1
    assert report["issue_rows"]
    assert is_mojibake_suspect("Toyota Japan窶冱 largest company")
