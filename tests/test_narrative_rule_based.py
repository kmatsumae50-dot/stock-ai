import pandas as pd

from stock_ai.narrative.rule_based import (
    add_narrative_features,
    build_narrative_context,
    tag_news_narratives,
)


def test_add_narrative_features_detects_dominant_theme(tmp_path):
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    pd.DataFrame(
        [
            {
                "date": "2024-01-05",
                "ticker": "7203.T",
                "title": "Yen weak supports Toyota exporters",
                "source": "manual",
                "url": "https://example.com/yen",
                "category": "macro",
                "sentiment_score": 0.6,
                "importance_score": 0.8,
                "summary": "Weak yen is supportive for exporters.",
            },
            {
                "date": "2024-01-06",
                "ticker": "auto",
                "title": "EV demand slows in China",
                "source": "manual",
                "url": "https://example.com/ev",
                "category": "EV",
                "sentiment_score": -0.7,
                "importance_score": 0.9,
                "summary": "China EV demand slowdown pressures auto makers.",
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

    result = add_narrative_features(data, news_dir, "7203.T")
    context = build_narrative_context(result)

    assert "dominant_narrative" in result.columns
    assert result.iloc[-1]["dominant_narrative"] in {"ev", "china_market"}
    assert result.iloc[-1]["narrative_bearish_pressure"] > 0
    assert context.dominant_label in {"EV", "China Market"}
    assert context.themes
    assert "Dominant narrative" in context.summary


def test_tag_news_narratives_returns_one_row_per_theme():
    news = pd.DataFrame(
        [
            {
                "date": "2024-01-05",
                "ticker": "7203.T",
                "title": "Weak yen and EV battery news",
                "source": "manual",
                "url": "",
                "category": "macro",
                "sentiment_score": 0.2,
                "importance_score": 0.5,
                "summary": "Yen weak, EV battery supply.",
            }
        ]
    )

    tagged = tag_news_narratives(news)

    assert set(tagged["narrative_key"]) >= {"yen_weakness", "ev"}
