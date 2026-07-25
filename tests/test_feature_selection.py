import pandas as pd

from stock_ai.models.baseline import select_feature_columns


def test_open_to_close_feature_selection_excludes_same_day_close_features():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "ticker": ["7203.T"],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [104.0],
            "volume": [1_000_000],
            "ma_25": [101.0],
            "rsi_14": [55.0],
            "usdjpy_return_1d": [0.01],
            "n225_return_1d": [0.02],
            "tesla_return_1d": [-0.01],
            "sector_mean_return": [0.01],
            "days_to_earnings": [3],
            "earnings_week_flag": [1],
            "earnings_pre_volatility": [0.02],
            "news_count_3d": [2],
            "avg_sentiment_3d": [-0.5],
            "max_importance_3d": [0.8],
            "negative_news_flag": [1],
            "sentiment_trend_3d": [-0.4],
            "importance_weighted_sentiment": [-0.6],
            "negative_news_shock": [1],
            "sector_news_strength": [-0.5],
            "dominant_narrative_code": [2],
            "dominant_narrative_sentiment": [-0.6],
            "narrative_bearish_pressure": [0.7],
            "target_next_close_binary": [1],
            "target_open_to_close_class": [3],
        }
    )

    features = select_feature_columns(data, target_column="target_open_to_close_class")

    assert "open" in features
    assert "close" not in features
    assert "high" not in features
    assert "low" not in features
    assert "volume" not in features
    assert "ma_25" not in features
    assert "rsi_14" not in features
    assert "usdjpy_return_1d" not in features
    assert "n225_return_1d" not in features
    assert "tesla_return_1d" not in features
    assert "sector_mean_return" not in features
    assert "days_to_earnings" not in features
    assert "earnings_week_flag" not in features
    assert "earnings_pre_volatility" not in features
    assert "news_count_3d" not in features
    assert "avg_sentiment_3d" not in features
    assert "max_importance_3d" not in features
    assert "negative_news_flag" not in features
    assert "sentiment_trend_3d" not in features
    assert "importance_weighted_sentiment" not in features
    assert "negative_news_shock" not in features
    assert "sector_news_strength" not in features
    assert "dominant_narrative_code" not in features
    assert "dominant_narrative_sentiment" not in features
    assert "narrative_bearish_pressure" not in features


def test_binary_target_is_not_selected_as_feature():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "ticker": ["7203.T"],
            "close": [100.0],
            "target_next_close_binary": [1],
        }
    )

    features = select_feature_columns(data, target_column="target_next_close_binary")

    assert "close" in features
    assert "target_next_close_binary" not in features


def test_future_anomaly_returns_are_not_selected_as_features():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": ["7203.T", "7203.T"],
            "return_1d": [0.01, -0.02],
            "anomaly_return_1d_after": [0.03, -0.01],
            "anomaly_return_5d_after": [0.04, -0.02],
            "target_next_close_binary": [1, 0],
        }
    )

    features = select_feature_columns(data, target_column="target_next_close_binary")

    assert "return_1d" in features
    assert "anomaly_return_1d_after" not in features
    assert "anomaly_return_5d_after" not in features
