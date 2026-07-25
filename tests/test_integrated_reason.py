import pandas as pd

from stock_ai.explain.integrated_reason import build_integrated_reasoning


def test_integrated_reasoning_high_when_probability_shap_and_similarity_align():
    shap_items = [
        {"feature": "rsi_14", "label": "RSI", "value": 61.0, "shap_value": 0.30},
        {"feature": "macd", "label": "MACD", "value": 10.0, "shap_value": 0.20},
        {"feature": "volatility_20", "label": "Volatility", "value": 0.02, "shap_value": -0.05},
    ]
    similar_patterns = pd.DataFrame(
        {
            "similar_start_date": [pd.Timestamp("2020-01-01")],
            "similar_end_date": [pd.Timestamp("2020-01-28")],
            "similarity": [0.91],
        }
    )
    similar_summary = pd.DataFrame(
        {
            "horizon": ["1D", "5D", "20D"],
            "average_return": [0.01, 0.03, 0.05],
            "positive_rate": [0.6, 0.8, 0.8],
        }
    )

    result = build_integrated_reasoning(
        prediction_label="up",
        up_probability=0.72,
        down_probability=0.28,
        shap_items=shap_items,
        similar_patterns=similar_patterns,
        similar_summary=similar_summary,
        external_context={
            "summary": (
                "USDJPY shows yen weakening, which can be supportive for exporters. "
                "Auto sector peer signals are broadly strong."
            )
        },
        event_context={
            "summary": "Post-earnings mode is active and the post-event trend is positive."
        },
        news_context={
            "impact_summary": "News impact is positive.",
            "positive_news_shock": 1,
            "negative_news_shock": 0,
            "importance_weighted_sentiment": 0.6,
            "sentiment_trend_3d": 0.4,
            "sector_news_strength": 0.3,
        },
        narrative_context={
            "summary": "Dominant narrative is Yen Weakness with positive sentiment.",
            "dominant_sentiment": 0.5,
            "dominant_importance": 0.8,
            "bullish_pressure": 0.6,
            "bearish_pressure": 0.0,
        },
        regime_state={"name": "Bull Trend"},
        risk_scores={
            "surge": {"score": 85, "level": "High"},
            "crash": {"score": 20, "level": "Low"},
        },
        lead_lag_summary=pd.DataFrame(
            {
                "name": ["Tesla"],
                "lag": [3],
                "correlation": [0.21],
                "direction_agreement": [0.58],
                "current_signal": ["bullish"],
                "score": [22.0],
            }
        ),
        anomaly_context={
            "score": 76,
            "level": "High",
            "types": ["Volume Spike"],
            "summary": "Current Anomaly Score is 76/100 (High). Detected type: Volume Spike.",
        },
    )

    assert result.confidence_level == "High"
    assert result.shap_direction == "up"
    assert result.similarity_direction == "up"
    assert "High" in result.summary
    assert "USDJPY" in result.external_text
    assert "Auto sector" in result.external_text
    assert "Post-earnings" in result.event_text
    assert "positive" in result.news_text
    assert result.news_direction == "up"
    assert result.news_alignment_text
    assert result.narrative_direction == "up"
    assert "Yen Weakness" in result.narrative_text
    assert "Bull Trend" in result.regime_text
    assert "Surge Risk Score" in result.risk_text
    assert result.lead_lag_direction == "up"
    assert "Tesla" in result.lead_lag_text
    assert result.anomaly_direction == "up"
    assert "Anomaly Score" in result.anomaly_text


def test_integrated_reasoning_low_when_similar_patterns_conflict():
    shap_items = [
        {"feature": "rsi_14", "label": "RSI", "value": 61.0, "shap_value": 0.30},
        {"feature": "macd", "label": "MACD", "value": 10.0, "shap_value": 0.20},
    ]
    similar_patterns = pd.DataFrame(
        {
            "similar_start_date": [pd.Timestamp("2020-01-01")],
            "similar_end_date": [pd.Timestamp("2020-01-28")],
            "similarity": [0.91],
        }
    )
    similar_summary = pd.DataFrame(
        {
            "horizon": ["5D"],
            "average_return": [-0.04],
            "positive_rate": [0.2],
        }
    )

    result = build_integrated_reasoning(
        prediction_label="up",
        up_probability=0.61,
        down_probability=0.39,
        shap_items=shap_items,
        similar_patterns=similar_patterns,
        similar_summary=similar_summary,
    )

    assert result.confidence_level == "Low"
    assert result.similarity_direction == "down"
    assert result.conflict_text
