from stock_ai.explain.shap_reason import (
    build_shap_reason_text,
    feature_label,
    normalize_shap_contributions,
    split_shap_contributions,
)


def test_split_shap_contributions_returns_upward_and_downward_drivers():
    items = [
        {"feature": "rsi_14", "value": 72.5, "contribution": 0.31},
        {"feature": "volatility_20", "value": 0.04, "contribution": -0.22},
        {"feature": "macd", "value": 12.0, "shap_value": 0.12},
        {"feature": "return_1d", "value": -0.01, "contribution": -0.08},
        {"feature": "ignored", "value": 1.0, "importance": 5.0},
    ]

    normalized = normalize_shap_contributions(items)
    upward, downward = split_shap_contributions(items, top_n=10)
    reason = build_shap_reason_text(items, prediction_label="up")

    assert len(normalized) == 4
    assert [item["feature"] for item in upward] == ["rsi_14", "macd"]
    assert [item["feature"] for item in downward] == ["volatility_20", "return_1d"]
    assert upward[0]["label"] == "RSI"
    assert "上昇方向" in reason
    assert "下落方向" in reason


def test_feature_label_supports_news_features():
    assert feature_label("avg_sentiment_3d") == "News sentiment 3D"
    assert feature_label("max_importance_3d") == "News importance 3D"
    assert feature_label("importance_weighted_sentiment") == "News weighted sentiment"
    assert feature_label("negative_news_shock") == "Negative news shock"
    assert feature_label("dominant_narrative_sentiment") == "Narrative sentiment"
