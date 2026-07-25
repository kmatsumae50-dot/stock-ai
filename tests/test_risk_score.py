import pandas as pd

from stock_ai.explain.risk_score import build_surge_crash_scores


def test_build_surge_crash_scores_detects_surge_setup():
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    data = pd.DataFrame(
        {
            "date": dates,
            "volume": [100.0] * 79 + [180.0],
            "volume_change_1d": [0.01] * 79 + [0.55],
            "rsi_14": [50.0] * 75 + [56.0, 58.0, 60.0, 62.0, 66.0],
            "macd": [0.1] * 80,
            "macd_signal": [0.0] * 80,
            "macd_hist": [-0.1] * 77 + [0.02, 0.04, 0.08],
            "volatility_20": [0.02] * 80,
        }
    )
    shap_items = [
        {"feature": "rsi_14", "value": 66.0, "shap_value": 0.30},
        {"feature": "macd", "value": 0.1, "shap_value": 0.20},
        {"feature": "volatility_20", "value": 0.02, "shap_value": -0.02},
    ]
    similar_summary = pd.DataFrame(
        {
            "horizon": ["5D"],
            "average_return": [0.03],
            "positive_rate": [0.8],
        }
    )

    scores = build_surge_crash_scores(data, shap_items, similar_summary)

    assert scores["surge"].level == "High"
    assert scores["surge"].score >= 70
    assert "shap" in scores["surge"].features


def test_build_surge_crash_scores_detects_crash_setup():
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    data = pd.DataFrame(
        {
            "date": dates,
            "volume": [100.0] * 79 + [180.0],
            "volume_change_1d": [0.01] * 79 + [0.55],
            "rsi_14": [55.0] * 75 + [50.0, 47.0, 44.0, 41.0, 36.0],
            "macd": [-0.1] * 80,
            "macd_signal": [0.0] * 80,
            "macd_hist": [0.1] * 77 + [-0.02, -0.04, -0.08],
            "volatility_20": [0.02] * 79 + [0.04],
        }
    )
    shap_items = [
        {"feature": "rsi_14", "value": 36.0, "shap_value": -0.30},
        {"feature": "macd", "value": -0.1, "shap_value": -0.20},
        {"feature": "volume_change_1d", "value": 0.55, "shap_value": 0.02},
    ]
    similar_summary = pd.DataFrame(
        {
            "horizon": ["5D"],
            "average_return": [-0.03],
            "positive_rate": [0.2],
        }
    )

    scores = build_surge_crash_scores(data, shap_items, similar_summary)

    assert scores["crash"].level == "High"
    assert scores["crash"].score >= 70
    assert "volatility_20" in scores["crash"].features


def test_build_surge_crash_scores_uses_news_shock():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    data = pd.DataFrame(
        {
            "date": dates,
            "volume_change_1d": [0.0] * 30,
            "rsi_14": [50.0] * 30,
            "macd_hist": [0.0] * 30,
            "volatility_20": [0.02] * 30,
            "negative_news_shock": [0] * 29 + [1],
            "importance_weighted_sentiment": [0.0] * 29 + [-0.7],
            "sentiment_trend_3d": [0.0] * 29 + [-0.5],
            "sector_news_strength": [0.0] * 29 + [-0.4],
        }
    )
    scores = build_surge_crash_scores(data, shap_items=[], similar_summary=pd.DataFrame())

    assert scores["crash"].score >= 30
    assert "negative_news_shock" in scores["crash"].features
    assert any("negative news" in reason.lower() for reason in scores["crash"].reasons)


def test_build_surge_crash_scores_uses_bearish_narrative():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    data = pd.DataFrame(
        {
            "date": dates,
            "volume_change_1d": [0.0] * 30,
            "rsi_14": [50.0] * 30,
            "macd_hist": [0.0] * 30,
            "volatility_20": [0.02] * 30,
            "narrative_bearish_pressure": [0.0] * 29 + [0.6],
            "dominant_narrative_sentiment": [0.0] * 29 + [-0.5],
            "dominant_narrative_importance": [0.0] * 29 + [0.8],
        }
    )
    scores = build_surge_crash_scores(data, shap_items=[], similar_summary=pd.DataFrame())

    assert scores["crash"].score >= 10
    assert "narrative_bearish_pressure" in scores["crash"].features


def test_build_surge_crash_scores_uses_bearish_lead_lag_signal():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    data = pd.DataFrame(
        {
            "date": dates,
            "volume_change_1d": [0.0] * 30,
            "rsi_14": [50.0] * 30,
            "macd_hist": [0.0] * 30,
            "volatility_20": [0.02] * 30,
        }
    )
    lead_lag_summary = pd.DataFrame(
        {
            "name": ["Tesla"],
            "current_signal": ["bearish"],
            "score": [24.0],
        }
    )

    scores = build_surge_crash_scores(
        data,
        shap_items=[],
        similar_summary=pd.DataFrame(),
        lead_lag_summary=lead_lag_summary,
    )

    assert scores["crash"].score >= 10
    assert "lead_lag" in scores["crash"].features


def test_build_surge_crash_scores_uses_crash_anomaly_context():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    data = pd.DataFrame(
        {
            "date": dates,
            "volume_change_1d": [0.0] * 30,
            "rsi_14": [50.0] * 30,
            "macd_hist": [0.0] * 30,
            "volatility_20": [0.02] * 30,
        }
    )

    scores = build_surge_crash_scores(
        data,
        shap_items=[],
        similar_summary=pd.DataFrame(),
        anomaly_context={
            "score": 78,
            "types": ["Volatility Spike", "Market Divergence"],
        },
    )

    assert scores["crash"].score >= 10
    assert "anomaly" in scores["crash"].features
